#pragma once
// Host-only numerics: deterministic RNG, CPU reference GEMM, the correctness
// criterion, and timing statistics. Deliberately CUDA-free so it can be
// unit-tested on a machine with no GPU (tests/test_host.cpp).

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sgemm {

// ---------------------------------------------------------------------------
// RNG: counter-based (splitmix64 of seed+index), so every element is a pure
// function of (seed, index). Reproducible across runs, machines, and any
// future parallel fill. Values are uniform on [-1, 1) with 24 significant
// bits, i.e. exactly representable in FP32.
//
// Zero-mean inputs matter for the tolerance below: with all-positive inputs
// the partial sums grow linearly in k and FP32 rounding error grows ~K*eps,
// whereas zero-mean partial sums random-walk and the relative error stays O(eps).
// ---------------------------------------------------------------------------
inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

inline void fill_uniform(float* p, size_t n, uint64_t seed) {
  const uint64_t base = splitmix64(seed);
  for (size_t i = 0; i < n; ++i) {
    const uint32_t bits = static_cast<uint32_t>(splitmix64(base + i) >> 40);  // 24 bits
    p[i] = static_cast<float>(bits) * (2.0f / 16777216.0f) - 1.0f;
  }
}

// ---------------------------------------------------------------------------
// CPU reference, accumulated in double. Also returns the per-element error
// *scale*  s_ij = |alpha| * sum_k |a_ik * b_kj| + |beta| * |c_ij|,
// which is the natural denominator for GEMM error: it is what the standard
// forward-error bound |fl(c) - c| <= gamma_K * s_ij is written in terms of.
// Normalising by s_ij (not by |c_ij|) avoids false failures where cancellation
// makes c_ij tiny while the terms that produced it were large.
// O(MNK): only used for small problems; large ones use cuBLAS as reference.
// ---------------------------------------------------------------------------
inline void cpu_reference(int M, int N, int K, float alpha, const float* A,
                          const float* B, float beta, const float* C0,
                          float* ref, float* scale) {
  std::vector<double> acc(N), mag(N);
  for (int i = 0; i < M; ++i) {
    std::fill(acc.begin(), acc.end(), 0.0);
    std::fill(mag.begin(), mag.end(), 0.0);
    for (int k = 0; k < K; ++k) {  // i-k-j order: unit stride over B and acc
      const double a = A[static_cast<size_t>(i) * K + k];
      const float* brow = B + static_cast<size_t>(k) * N;
      for (int j = 0; j < N; ++j) {
        const double t = a * brow[j];
        acc[j] += t;
        mag[j] += std::fabs(t);
      }
    }
    for (int j = 0; j < N; ++j) {
      const size_t idx = static_cast<size_t>(i) * N + j;
      double r = alpha * acc[j];
      double s = std::fabs(alpha) * mag[j];
      if (beta != 0.0f) {  // BLAS: beta == 0 means C0 is never read (may be NaN)
        r += static_cast<double>(beta) * C0[idx];
        s += std::fabs(static_cast<double>(beta) * C0[idx]);
      }
      ref[idx] = static_cast<float>(r);
      scale[idx] = static_cast<float>(s);
    }
  }
}

// ---------------------------------------------------------------------------
// Correctness criterion:   |got - ref| / s_ij  <=  tol(K)
//
//   tol(K) = eps32 * (4*sqrt(K) + 8)
//
// Where the constants come from, for zero-mean uniform inputs:
//  * A sequential FP32 dot product of length K rounds each partial sum s_k.
//    For zero-mean terms |s_k| ~ sqrt(k)*sigma, so the accumulated error has
//    std ~ eps*sigma*sqrt(sum_k k) ~ eps*sigma*K/sqrt(6), while s_ij ~ K*E|ab|.
//    The ratio is ~0.5 eps, independent of K (the worst-case gamma_K ~ K*eps
//    bound is far too pessimistic here). The max over ~10^7 outputs is a few eps.
//  * The reference itself may be cuBLAS-in-FP32, so errors from both sides add.
//  * The alpha scale and beta*C add each contribute <= 1 eps: hence the +8.
//  * 4*sqrt(K) is headroom for summation orders that correlate error
//    (e.g. a long serial chain inside one thread), while staying far below
//    what a real bug produces: dropping one k-term gives normalised error
//    ~|a b| / s_ij ~ 4/K, i.e. ~1e-3 at K=4096 vs tol ~3e-5.
//    tests/test_correctness.cu verifies that sensitivity directly.
// The max normalised error is always reported in units of eps, so the margin
// is visible rather than hidden behind a pass/fail bit.
// ---------------------------------------------------------------------------
inline double tolerance(int K) {
  return static_cast<double>(FLT_EPSILON) * (4.0 * std::sqrt(static_cast<double>(K)) + 8.0);
}

struct CheckResult {
  bool pass = true;
  double max_err = 0.0;      // max normalised error
  double tol = 0.0;
  long long num_bad = 0;
  long long worst_idx = -1;  // index of max error (or first NaN)
  float got_worst = 0.0f, ref_worst = 0.0f;
};

inline CheckResult compare(const float* got, const float* ref, const float* scale,
                           size_t n, int K) {
  CheckResult r;
  r.tol = tolerance(K);
  for (size_t i = 0; i < n; ++i) {
    const double err = std::fabs(static_cast<double>(got[i]) - ref[i]) /
                       (static_cast<double>(scale[i]) + static_cast<double>(FLT_MIN));
    // Written as !(err <= tol) so NaN/Inf in `got` counts as a failure.
    if (!(err <= r.tol)) ++r.num_bad;
    const double e = std::isnan(err) ? INFINITY : err;  // NaN ranks as worst
    if (r.worst_idx < 0 || e > r.max_err) {
      r.max_err = e;
      r.worst_idx = static_cast<long long>(i);
      r.got_worst = got[i];
      r.ref_worst = ref[i];
    }
  }
  r.pass = (r.num_bad == 0);
  return r;
}

// ---------------------------------------------------------------------------
// Timing statistics. GFLOPS is reported from the MEDIAN time: the distribution
// of kernel times has a hard lower bound and a right tail (clock ramp,
// preemption, OS noise), so the median is the robust central estimate; mean and
// stddev are kept to expose that tail rather than hide it.
// ---------------------------------------------------------------------------
struct Stats {
  double mean = 0, median = 0, stddev = 0, min = 0, max = 0;
  int n = 0;
};

inline Stats summarize(std::vector<double> v) {
  Stats s;
  s.n = static_cast<int>(v.size());
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  s.min = v.front();
  s.max = v.back();
  const size_t n = v.size();
  s.median = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / n;
  double ss = 0;
  for (double x : v) ss += (x - s.mean) * (x - s.mean);
  s.stddev = n > 1 ? std::sqrt(ss / (n - 1)) : 0.0;  // sample stddev
  return s;
}

// FLOP count convention: 2*M*N*K (one mul + one add per MAC). The alpha/beta
// epilogue (O(MN)) is excluded, matching how cuBLAS numbers are quoted.
inline double gemm_flops(int M, int N, int K) { return 2.0 * M * N * static_cast<double>(K); }

}  // namespace sgemm
