// Correctness harness.
//
//   sgemm_test [--kernel all|NAME[,NAME...]] [--quick]
//
// Three layers:
//  1. Checker sensitivity: prove the tolerance is tight enough to catch a
//     realistic bug (one dropped k-term) before trusting any PASS, for both the
//     FP32 and the TF32 tolerance, and prove TF32 cuBLAS output FAILS the FP32
//     check (so an accidental tensor-core path can't pass as FP32).
//  2. Small shapes vs a CPU double-precision reference: odd, prime, degenerate
//     (M/N/K = 1) and non-multiple-of-every-tile sizes. This is where boundary
//     bugs live, and a CPU reference makes a failure unambiguous.
//  3. Large shapes vs cuBLAS (FP32) with |A||B| error scaling.
// Every case runs twice: (alpha=1, beta=0, C pre-filled with NaN) to enforce the
// BLAS "C is not read" rule, and (alpha=1.5, beta=-0.75, random C) to exercise
// the read-modify-write epilogue.

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "sgemm/check.hpp"
#include "sgemm/common.hpp"
#include "sgemm/cublas_ref.hpp"
#include "sgemm/host_utils.hpp"
#include "sgemm/kernels.hpp"

using namespace sgemm;

namespace {

struct Shape { int M, N, K; };

// Small enough for the O(MNK) CPU reference. Chosen to hit: 1-sized dims,
// primes, +-1 around 32/64/128 (every block tile we will use), K < BK, and
// M or N smaller than one warp.
const std::vector<Shape> kSmall = {
    {1, 1, 1},     {1, 1, 7},     {7, 13, 5},    {32, 32, 32},  {31, 33, 17},
    {33, 31, 65},  {64, 64, 64},  {63, 65, 3},   {100, 100, 100}, {127, 129, 65},
    {128, 128, 8}, {128, 256, 64}, {255, 257, 300}, {513, 1, 77},  {1, 513, 77},
    {17, 1000, 3}, {129, 131, 1}, {256, 128, 511}};

const std::vector<Shape> kLarge = {
    {256, 256, 256},  {1024, 1024, 1024}, {1000, 1500, 2003},
    {2048, 512, 4096}, {4097, 4095, 1023}, {4096, 4096, 4096}};

struct Case { float alpha, beta; };
const Case kCases[] = {{1.0f, 0.0f}, {1.5f, -0.75f}};

int g_fail = 0, g_pass = 0;

std::vector<std::string> split(const std::string& s, char d) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, d))
    if (!t.empty()) out.push_back(t);
  return out;
}

void report(const char* who, const Shape& s, const Case& c, const CheckResult& r) {
  const bool ok = r.pass;
  (ok ? g_pass : g_fail)++;
  if (!ok || std::getenv("SGEMM_TEST_VERBOSE")) {
    std::printf("  [%s] %-10s %5dx%5dx%5d a=%g b=%g  max_err=%.1f eps (tol %.1f eps)",
                ok ? "PASS" : "FAIL", who, s.M, s.N, s.K, c.alpha, c.beta,
                r.max_err / FLT_EPSILON, r.tol / FLT_EPSILON);
    if (!ok) {
      const int i = static_cast<int>(r.worst_idx / s.N), j = static_cast<int>(r.worst_idx % s.N);
      std::printf("  bad=%lld worst C[%d,%d] got=%g want=%g", r.num_bad, i, j, r.got_worst,
                  r.ref_worst);
    }
    std::printf("\n");
  }
}

struct Problem {
  Shape s;
  std::vector<float> A, B, C0;
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;

  Problem(Shape sh, uint64_t seed) : s(sh) {
    A.resize(size_t(s.M) * s.K);
    B.resize(size_t(s.K) * s.N);
    C0.resize(size_t(s.M) * s.N);
    fill_uniform(A.data(), A.size(), seed);
    fill_uniform(B.data(), B.size(), seed + 1);
    fill_uniform(C0.data(), C0.size(), seed + 2);
    CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, C0.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
  }
  ~Problem() {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
  }
  Problem(const Problem&) = delete;
  Problem& operator=(const Problem&) = delete;

  void reset_C(float beta) {
    if (beta != 0.0f)
      CUDA_CHECK(cudaMemcpy(dC, C0.data(), C0.size() * sizeof(float), cudaMemcpyHostToDevice));
    else
      CUDA_CHECK(cudaMemset(dC, 0xFF, C0.size() * sizeof(float)));  // NaN
  }
  std::vector<float> run(LaunchFn fn, const Case& c) {
    reset_C(c.beta);
    fn(s.M, s.N, s.K, c.alpha, dA, dB, c.beta, dC, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> out(C0.size());
    CUDA_CHECK(cudaMemcpy(out.data(), dC, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return out;
  }
};

// Layer 1. Build a WRONG result by running cuBLAS with A's last column zeroed
// (i.e. the k = K-1 term dropped) and assert the checker rejects it. If this
// ever passes, the tolerance has become too loose to trust.
void checker_sensitivity(CublasSgemm& blas) {
  std::printf("checker sensitivity (a dropped k-term must FAIL):\n");
  struct Probe { Shape s; Precision p; };
  // TF32's tolerance is ~70x looser, so its dropped-term probe uses small K
  // (host_utils.hpp explains the K limit).
  const Probe probes[] = {{{256, 256, 256}, Precision::FP32},
                          {{1000, 1500, 2003}, Precision::FP32},
                          {{512, 512, 8192}, Precision::FP32},
                          {{256, 256, 256}, Precision::TF32},
                          {{512, 512, 64}, Precision::TF32}};
  for (const Probe& pr : probes) {
    const Shape& s = pr.s;
    Problem p(s, 77);
    const Case c{1.0f, 0.0f};
    const Reference ref = gpu_reference(blas, s.M, s.N, s.K, c.alpha, p.dA, p.dB, c.beta, p.C0);

    std::vector<float> A2 = p.A;
    for (int i = 0; i < s.M; ++i) A2[size_t(i) * s.K + (s.K - 1)] = 0.0f;
    CUDA_CHECK(cudaMemcpy(p.dA, A2.data(), A2.size() * sizeof(float), cudaMemcpyHostToDevice));
    p.reset_C(c.beta);
    blas(s.M, s.N, s.K, c.alpha, p.dA, p.dB, c.beta, p.dC, nullptr);
    std::vector<float> got(size_t(s.M) * s.N);
    CUDA_CHECK(cudaMemcpy(got.data(), p.dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const CheckResult r =
        compare(got.data(), ref.ref.data(), ref.scale.data(), got.size(), s.K, pr.p);
    const bool ok = !r.pass;  // we WANT a failure here
    (ok ? g_pass : g_fail)++;
    std::printf("  [%s] %s %5dx%5dx%5d  max_err=%.1f eps vs tol %.1f eps -> %s\n",
                ok ? "PASS" : "FAIL", precision_name(pr.p), s.M, s.N, s.K,
                r.max_err / FLT_EPSILON, r.tol / FLT_EPSILON,
                ok ? "detected" : "NOT DETECTED (tolerance too loose)");
  }

  // TF32 cuBLAS must pass the TF32 check and FAIL the FP32 check.
  std::printf("tf32 cuBLAS vs FP32 reference (TF32 check must pass, FP32 check must fail):\n");
  CublasSgemm blas_tf32(CublasMath::TF32);
  for (const Shape s : {Shape{512, 512, 512}, Shape{1024, 1024, 4096}}) {
    Problem p(s, 91);
    const Reference ref = gpu_reference(blas, s.M, s.N, s.K, 1.0f, p.dA, p.dB, 0.0f, p.C0);
    p.reset_C(0.0f);
    blas_tf32(s.M, s.N, s.K, 1.0f, p.dA, p.dB, 0.0f, p.dC, nullptr);
    std::vector<float> got(size_t(s.M) * s.N);
    CUDA_CHECK(cudaMemcpy(got.data(), p.dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const CheckResult rt = compare(got.data(), ref.ref.data(), ref.scale.data(), got.size(), s.K, Precision::TF32);
    const CheckResult rf = compare(got.data(), ref.ref.data(), ref.scale.data(), got.size(), s.K);
    const bool ok = rt.pass && !rf.pass;
    (ok ? g_pass : g_fail)++;
    std::printf("  [%s] %5dx%5dx%5d  err=%.1f eps  tf32 tol %.0f -> %s, fp32 tol %.0f -> %s%s\n",
                ok ? "PASS" : "FAIL", s.M, s.N, s.K, rt.max_err / FLT_EPSILON,
                rt.tol / FLT_EPSILON, rt.pass ? "pass" : "FAIL", rf.tol / FLT_EPSILON,
                rf.pass ? "pass" : "fail",
                rf.pass ? "  (cuBLAS TF32 mode may not have used tensor cores here)" : "");
  }
}

// cuBLAS itself vs the CPU reference: validates the row-major wrapper
// (operand swap) and the reference pipeline before any custom kernel is judged.
void cublas_vs_cpu(CublasSgemm& blas) {
  std::printf("cublas row-major wrapper vs CPU reference:\n");
  for (const Shape& s : kSmall) {
    Problem p(s, 5);
    for (const Case& c : kCases) {
      p.reset_C(c.beta);
      blas(s.M, s.N, s.K, c.alpha, p.dA, p.dB, c.beta, p.dC, nullptr);
      std::vector<float> got(p.C0.size()), ref(p.C0.size()), scale(p.C0.size());
      CUDA_CHECK(cudaMemcpy(got.data(), p.dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
      cpu_reference(s.M, s.N, s.K, c.alpha, p.A.data(), p.B.data(), c.beta, p.C0.data(),
                    ref.data(), scale.data());
      report("cublas", s, c, compare(got.data(), ref.data(), scale.data(), got.size(), s.K));
    }
  }
}

void test_kernel(const KernelSpec& k, CublasSgemm& blas, bool quick) {
  std::printf("%s (stage %d, %s):\n", k.name, k.stage, precision_name(k.precision));
  const int fail_before = g_fail;
  for (const Shape& s : kSmall) {
    Problem p(s, 11);
    for (const Case& c : kCases) {
      const std::vector<float> got = p.run(k.launch, c);
      std::vector<float> ref(got.size()), scale(got.size());
      cpu_reference(s.M, s.N, s.K, c.alpha, p.A.data(), p.B.data(), c.beta, p.C0.data(),
                    ref.data(), scale.data());
      report(k.name, s, c, compare(got.data(), ref.data(), scale.data(), got.size(), s.K, k.precision));
    }
  }
  for (const Shape& s : kLarge) {
    if (quick && size_t(s.M) * s.N * s.K > (size_t(1) << 31)) continue;
    Problem p(s, 23);
    for (const Case& c : kCases) {
      const Reference ref = gpu_reference(blas, s.M, s.N, s.K, c.alpha, p.dA, p.dB, c.beta, p.C0);
      const std::vector<float> got = p.run(k.launch, c);
      report(k.name, s, c,
             compare(got.data(), ref.ref.data(), ref.scale.data(), got.size(), s.K, k.precision));
    }
  }
  std::printf("  -> %s\n", g_fail == fail_before ? "all passed" : "FAILURES");
}

}  // namespace

int main(int argc, char** argv) {
  std::string sel = "all";
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--kernel") && i + 1 < argc) sel = argv[++i];
    else if (!std::strcmp(argv[i], "--quick")) quick = true;
    else {
      std::fprintf(stderr, "usage: sgemm_test [--kernel all|NAME[,NAME]] [--quick]\n");
      return 2;
    }
  }

  CublasSgemm blas;
  checker_sensitivity(blas);
  cublas_vs_cpu(blas);

  if (sel == "all") {
    for (const auto& k : all_kernels()) test_kernel(k, blas, quick);
  } else {
    for (const auto& name : split(sel, ',')) {
      const KernelSpec* k = find_kernel(name);
      if (!k) {
        std::fprintf(stderr, "unknown kernel %s\n", name.c_str());
        return 2;
      }
      test_kernel(*k, blas, quick);
    }
  }
  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
