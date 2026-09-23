// Host-only tests: validate the numerical plumbing without a GPU.
// The key one emulates what our kernels actually do (a serial FP32 FFMA chain
// per output, via std::fmaf) and checks that it PASSES the tolerance, while a
// dropped k-term and a NaN FAIL. That pins the tolerance from both sides using
// real FP32 rounding rather than an argument.

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

#include "sgemm/host_utils.hpp"

using namespace sgemm;

static int g_fail = 0;
#define EXPECT(cond, ...)                         \
  do {                                            \
    if (!(cond)) {                                \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                   \
      std::printf("\n");                          \
      ++g_fail;                                   \
    }                                             \
  } while (0)

// What a GPU thread computes: acc = fma(a, b, acc) in FP32, then the epilogue.
// `order` = 0 sequential, 1 = split into 8 interleaved partial sums (a stand-in
// for the different summation orders later stages use).
static void emulate_fp32(int M, int N, int K, float alpha, const float* A, const float* B,
                         float beta, const float* C0, float* out, int order, int drop_k = -1) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float part[8] = {0};
      for (int k = 0; k < K; ++k) {
        if (k == drop_k) continue;
        float& acc = order ? part[k % 8] : part[0];
        acc = std::fmaf(A[size_t(i) * K + k], B[size_t(k) * N + j], acc);
      }
      float acc = 0;
      for (float p : part) acc += p;
      const size_t idx = size_t(i) * N + j;
      out[idx] = beta == 0.0f ? alpha * acc : alpha * acc + beta * C0[idx];
    }
}

// TF32 input rounding, two ways. RN-away is what cvt.rna.tf32.f32 /
// wmma::__float_to_tf32 does; truncation is the worst case we budget for.
static float tf32_round(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u = (u + 0x1000u) & 0xFFFFE000u;  // add half-ulp of the 13 dropped bits, then clear them
  std::memcpy(&f, &u, 4);
  return f;
}
static float tf32_trunc(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u &= 0xFFFFE000u;
  std::memcpy(&f, &u, 4);
  return f;
}

// A TF32 tensor-core GEMM as far as rounding is concerned: operands converted,
// products accumulated in FP32.
static void emulate_tf32(int M, int N, int K, const float* A, const float* B, float* out,
                         float (*cvt)(float), int drop_k = -1) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0;
      for (int k = 0; k < K; ++k)
        if (k != drop_k) acc = std::fmaf(cvt(A[size_t(i) * K + k]), cvt(B[size_t(k) * N + j]), acc);
      out[size_t(i) * N + j] = acc;
    }
}

int main() {
  // RNG: range and determinism.
  {
    std::vector<float> a(100000), b(100000);
    fill_uniform(a.data(), a.size(), 42);
    fill_uniform(b.data(), b.size(), 42);
    double mean = 0;
    bool in_range = true;
    for (size_t i = 0; i < a.size(); ++i) {
      in_range &= (a[i] >= -1.0f && a[i] < 1.0f);
      mean += a[i];
    }
    mean /= a.size();
    EXPECT(in_range, "fill_uniform out of [-1,1)");
    EXPECT(a == b, "fill_uniform not deterministic");
    EXPECT(std::fabs(mean) < 0.01, "fill_uniform mean %f not ~0", mean);
  }

  // Stats.
  {
    const Stats s = summarize({4, 1, 3, 2});
    EXPECT(s.median == 2.5 && s.min == 1 && s.max == 4 && s.mean == 2.5, "stats even");
    EXPECT(std::fabs(s.stddev - std::sqrt(5.0 / 3.0)) < 1e-12, "stats stddev %f", s.stddev);
    EXPECT(summarize({5, 1, 9}).median == 5, "stats odd median");
  }

  // Tolerance, from both sides, on shapes covering short and long K.
  struct Shp { int M, N, K; };
  const Shp shapes[] = {{1, 1, 1}, {7, 13, 5}, {33, 31, 65}, {16, 16, 1024},
                        {8, 8, 4096}, {4, 4, 16384}};
  const float cases[][2] = {{1.0f, 0.0f}, {1.5f, -0.75f}};
  for (const Shp& s : shapes) {
    std::vector<float> A(size_t(s.M) * s.K), B(size_t(s.K) * s.N), C0(size_t(s.M) * s.N);
    fill_uniform(A.data(), A.size(), 1);
    fill_uniform(B.data(), B.size(), 2);
    fill_uniform(C0.data(), C0.size(), 3);
    for (const auto& c : cases) {
      std::vector<float> ref(C0.size()), scale(C0.size()), got(C0.size());
      cpu_reference(s.M, s.N, s.K, c[0], A.data(), B.data(), c[1], C0.data(), ref.data(),
                    scale.data());
      for (int order = 0; order < 2; ++order) {
        emulate_fp32(s.M, s.N, s.K, c[0], A.data(), B.data(), c[1], C0.data(), got.data(), order);
        const CheckResult r = compare(got.data(), ref.data(), scale.data(), got.size(), s.K);
        EXPECT(r.pass, "correct FP32 result rejected: %dx%dx%d order=%d err=%.2f eps tol=%.2f eps",
               s.M, s.N, s.K, order, r.max_err / FLT_EPSILON, r.tol / FLT_EPSILON);
        std::printf("  fp32 emu %5dx%5dx%5d a=%.2f b=%.2f order=%d  max_err=%6.2f eps  tol=%7.1f eps\n",
                    s.M, s.N, s.K, c[0], c[1], order, r.max_err / FLT_EPSILON, r.tol / FLT_EPSILON);
      }
      if (s.K > 1) {
        emulate_fp32(s.M, s.N, s.K, c[0], A.data(), B.data(), c[1], C0.data(), got.data(), 0,
                     s.K - 1);
        const CheckResult r = compare(got.data(), ref.data(), scale.data(), got.size(), s.K);
        EXPECT(!r.pass, "dropped k-term NOT detected: %dx%dx%d err=%.2f eps", s.M, s.N, s.K,
               r.max_err / FLT_EPSILON);
      }
    }
  }

  // TF32 tolerance, from both sides.
  {
    const Shp tshapes[] = {{8, 8, 64}, {16, 16, 256}, {8, 8, 4096}, {4, 4, 16384}};
    for (const Shp& s : tshapes) {
      std::vector<float> A(size_t(s.M) * s.K), B(size_t(s.K) * s.N), C0(size_t(s.M) * s.N, 0.f);
      fill_uniform(A.data(), A.size(), 21);
      fill_uniform(B.data(), B.size(), 22);
      std::vector<float> ref(C0.size()), scale(C0.size()), got(C0.size());
      cpu_reference(s.M, s.N, s.K, 1.0f, A.data(), B.data(), 0.0f, C0.data(), ref.data(), scale.data());
      for (auto cvt : {&tf32_round, &tf32_trunc}) {
        emulate_tf32(s.M, s.N, s.K, A.data(), B.data(), got.data(), cvt);
        const CheckResult r = compare(got.data(), ref.data(), scale.data(), got.size(), s.K, Precision::TF32);
        EXPECT(r.pass, "correct TF32 result rejected: %dx%dx%d err=%.3g tol=%.3g", s.M, s.N, s.K,
               r.max_err, r.tol);
        std::printf("  tf32 emu %5dx%5dx%5d %-5s  max_err=%.2e  tol=%.2e  (fp32 check would %s)\n",
                    s.M, s.N, s.K, cvt == &tf32_round ? "RN" : "trunc", r.max_err, r.tol,
                    compare(got.data(), ref.data(), scale.data(), got.size(), s.K).pass ? "pass" : "FAIL");
      }
      if (s.K <= 256) {  // dropped-term sensitivity only claimed for small K (see host_utils.hpp)
        emulate_tf32(s.M, s.N, s.K, A.data(), B.data(), got.data(), &tf32_round, s.K - 1);
        const CheckResult r = compare(got.data(), ref.data(), scale.data(), got.size(), s.K, Precision::TF32);
        EXPECT(!r.pass, "TF32: dropped k-term NOT detected at K=%d (err %.3g tol %.3g)", s.K, r.max_err, r.tol);
      }
    }
  }

  // An FP32 kernel that secretly computed in TF32 must FAIL the FP32 check.
  // (Needs enough outputs for the max error to show: 32x32 here.)
  {
    for (int K : {256, 1024, 4096, 8192}) {
      const int M = 32, N = 32;
      std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C0(size_t(M) * N, 0.f);
      fill_uniform(A.data(), A.size(), 31);
      fill_uniform(B.data(), B.size(), 32);
      std::vector<float> ref(C0.size()), scale(C0.size()), got(C0.size());
      cpu_reference(M, N, K, 1.0f, A.data(), B.data(), 0.0f, C0.data(), ref.data(), scale.data());
      emulate_tf32(M, N, K, A.data(), B.data(), got.data(), &tf32_round);
      const CheckResult r = compare(got.data(), ref.data(), scale.data(), got.size(), K);
      EXPECT(!r.pass, "TF32 result PASSED the FP32 check at K=%d (err %.2f eps, tol %.2f eps)", K,
             r.max_err / FLT_EPSILON, r.tol / FLT_EPSILON);
      std::printf("  tf32-vs-fp32-check K=%5d: max_err=%7.1f eps  fp32 tol=%5.1f eps -> %s\n", K,
                  r.max_err / FLT_EPSILON, r.tol / FLT_EPSILON, r.pass ? "ACCEPTED (bad)" : "rejected");
    }
  }

  // NaN must fail even though NaN comparisons are false.
  {
    const float got[2] = {1.0f, NAN}, ref[2] = {1.0f, 2.0f}, scale[2] = {1.0f, 1.0f};
    const CheckResult r = compare(got, ref, scale, 2, 1);
    EXPECT(!r.pass && r.num_bad == 1 && r.worst_idx == 1, "NaN not caught");
  }

  std::printf("%s (%d failures)\n", g_fail ? "HOST TESTS FAILED" : "host tests passed", g_fail);
  return g_fail ? 1 : 0;
}
