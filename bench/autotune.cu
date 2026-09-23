// Stage 8 — autotuning over a kernel family's template parameters.
//
//   sgemm_autotune [--family blocktiling1d] [--tune 2048,4096] [--eval 3072,4097x4093x4099]
//                  [--warmup 5] [--iters 20] [--csv PATH]
//
// Honesty rule this tool enforces: configurations are RANKED on --tune sizes
// and the winner is REPORTED on disjoint --eval sizes. Picking the best of ~40
// configs on the same sizes you report is selection bias: some of the "win" is
// noise that you selected for. The eval column is the number you're allowed
// to quote.
//
// Some candidates spill registers: 1024-thread configs are capped at 64
// registers/thread by __launch_bounds__, e.g. stage 4 <128,128,*,16>. That's
// a real cost and the tuner measures it rather than filtering it out. Configure
// with -DSGEMM_PTXAS_VERBOSE=ON to see which ones spill before you interpret a
// ranking.
//
// Families are compile-time lists of template instantiations, filtered by a
// constexpr validity predicate so invalid tilings never instantiate. To add
// stage 5/6: write a `register_<family>()` like the one below and add it to
// kFamilies.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "04_blocktiling_1d.cuh"
#include "05_blocktiling_2d.cuh"
#include "06_vectorized.cuh"
#include "07_warptiling.cuh"
#include "09_tf32_wmma.cuh"
#include "sgemm/check.hpp"
#include "sgemm/common.hpp"
#include "sgemm/cublas_ref.hpp"
#include "sgemm/device_info.hpp"
#include "sgemm/host_utils.hpp"
#include "sgemm/kernels.hpp"
#include "sgemm/timing.hpp"

using namespace sgemm;

namespace {

struct Candidate {
  std::string name;  // e.g. "BM64_BN64_BK8_TM8"
  LaunchFn launch;
  Precision precision = Precision::FP32;  // selects the verification tolerance
};

// ---- family: stage 4 (1D block tiling) -------------------------------------
constexpr int k4BM[] = {32, 64, 128};
constexpr int k4BN[] = {32, 64, 128};
constexpr int k4BK[] = {8, 16, 32};
constexpr int k4TM[] = {4, 8, 16};

template <int BM, int BN, int BK, int TM>
constexpr bool valid_04() {
  constexpr int threads = BM * BN / TM;
  return BM % TM == 0 && BN % 32 == 0 && threads % 32 == 0 && threads >= 64 &&
         threads <= 1024 && (BM * BK) % threads == 0 && (BK * BN) % threads == 0 &&
         (BM + BN) * BK * 4 <= 48 * 1024;  // static smem limit
}

template <size_t I>
void add_04(std::vector<Candidate>& v) {
  constexpr int BM = k4BM[I / 27], BN = k4BN[(I / 9) % 3], BK = k4BK[(I / 3) % 3], TM = k4TM[I % 3];
  if constexpr (valid_04<BM, BN, BK, TM>()) {
    v.push_back({"BM" + std::to_string(BM) + "_BN" + std::to_string(BN) + "_BK" +
                     std::to_string(BK) + "_TM" + std::to_string(TM),
                 &launch_04_blocktiling_1d_cfg<BM, BN, BK, TM>});
  }
}

template <size_t... I>
void add_all_04(std::vector<Candidate>& v, std::index_sequence<I...>) {
  (add_04<I>(v), ...);
}

std::vector<Candidate> register_blocktiling1d() {
  std::vector<Candidate> v;
  add_all_04(v, std::make_index_sequence<27 * 3>{});
  return v;
}

// ---- family: stage 5 (2D block tiling) -------------------------------------
constexpr int k5BMN[] = {64, 128};
constexpr int k5BK[] = {8, 16};
constexpr int k5T[] = {4, 8};

template <int BM, int BN, int BK, int TM, int TN>
constexpr bool valid_05() {
  constexpr int threads = BM * BN / (TM * TN);
  return BM % TM == 0 && BN % TN == 0 && threads % 32 == 0 && threads >= 64 &&
         threads <= 1024 && (BM * BK) % threads == 0 && (BK * BN) % threads == 0;
}

template <size_t I>
void add_05(std::vector<Candidate>& v) {
  constexpr int BM = k5BMN[I / 16], BN = k5BMN[(I / 8) % 2], BK = k5BK[(I / 4) % 2],
                TM = k5T[(I / 2) % 2], TN = k5T[I % 2];
  if constexpr (valid_05<BM, BN, BK, TM, TN>()) {
    v.push_back({"BM" + std::to_string(BM) + "_BN" + std::to_string(BN) + "_BK" +
                     std::to_string(BK) + "_TM" + std::to_string(TM) + "_TN" + std::to_string(TN),
                 &launch_05_blocktiling_2d_cfg<BM, BN, BK, TM, TN>});
  }
}

template <size_t... I>
void add_all_05(std::vector<Candidate>& v, std::index_sequence<I...>) {
  (add_05<I>(v), ...);
}

std::vector<Candidate> register_blocktiling2d() {
  std::vector<Candidate> v;
  add_all_05(v, std::make_index_sequence<32>{});
  return v;
}

// ---- family: stage 6 (vectorized; thread tile fixed at 8x8, BN fixed at 128
// ---- because the split-tile conflict argument needs 16 threadCols) ----------
constexpr int k6BM[] = {64, 128, 256};
constexpr int k6BK[] = {8, 16, 32};

template <int BM, int BK>
constexpr bool valid_06() {
  constexpr int threads = BM * 128 / 64;
  return (BM * BK / 4) % threads == 0 && (BK * 128 / 4) % threads == 0 &&
         (BK * (BM + 4) + BK * 128) * 4 <= 48 * 1024;
}

template <size_t I>
void add_06(std::vector<Candidate>& v) {
  constexpr int BM = k6BM[I / 3], BK = k6BK[I % 3];
  if constexpr (valid_06<BM, BK>()) {
    v.push_back({"BM" + std::to_string(BM) + "_BN128_BK" + std::to_string(BK),
                 &launch_06_vectorized_cfg<BM, 128, BK>});
  }
}

template <size_t... I>
void add_all_06(std::vector<Candidate>& v, std::index_sequence<I...>) {
  (add_06<I>(v), ...);
}

std::vector<Candidate> register_vectorized() {
  std::vector<Candidate> v;
  add_all_06(v, std::make_index_sequence<9>{});
  return v;
}

// ---- family: stage 7 (warptiling). Lane layout fixed at 8x4 with 4x4 per lane,
// ---- so WM must be a multiple of 32 and WN of 16. ---------------------------
constexpr int k7BMN[] = {64, 128};
constexpr int k7BK[] = {8, 16};
constexpr int k7WM[] = {32, 64};
constexpr int k7WN[] = {16, 32, 64};

template <int BM, int BN, int BK, int WM, int WN>
constexpr bool valid_07() {
  if (BM % WM != 0 || BN % WN != 0) return false;
  constexpr int threads = (BM / WM) * (BN / WN) * 32;
  constexpr int outputs_per_thread = (WM / 32) * (WN / 16) * 16;
  return threads >= 64 && threads <= 1024 && outputs_per_thread <= 128 &&
         (BM * BK / 4) % threads == 0 && (BK * BN / 4) % threads == 0 &&
         (BK * (BM + 4) + BK * BN) * 4 <= 48 * 1024;
}

template <size_t I>
void add_07(std::vector<Candidate>& v) {
  // I in [0, 48): BM(2) x BN(2) x BK(2) x WM(2) x WN(3), WN fastest.
  constexpr int BM = k7BMN[I / 24], BN = k7BMN[(I / 12) % 2], BK = k7BK[(I / 6) % 2],
                WM = k7WM[(I / 3) % 2], WN = k7WN[I % 3];
  if constexpr (valid_07<BM, BN, BK, WM, WN>()) {
    v.push_back({"BM" + std::to_string(BM) + "_BN" + std::to_string(BN) + "_BK" +
                     std::to_string(BK) + "_WM" + std::to_string(WM) + "_WN" + std::to_string(WN),
                 &launch_07_warptiling_cfg<BM, BN, BK, WM, WN>});
  }
}

template <size_t... I>
void add_all_07(std::vector<Candidate>& v, std::index_sequence<I...>) {
  (add_07<I>(v), ...);
}

std::vector<Candidate> register_warptiling() {
  std::vector<Candidate> v;
  add_all_07(v, std::make_index_sequence<2 * 2 * 2 * 2 * 3>{});
  return v;
}

// ---- family: stage 9 (TF32 wmma). Verified with the TF32 tolerance; rank it
// ---- only against itself, never against the FP32 families. -----------------
constexpr int k9BMN[] = {64, 128};
constexpr int k9BK[] = {16, 32};
constexpr int k9W[] = {32, 64};

template <int BM, int BN, int BK, int WM, int WN>
constexpr bool valid_09() {
  if (BM % WM != 0 || BN % WN != 0) return false;
  constexpr int warps = (BM / WM) * (BN / WN);
  constexpr int threads = warps * 32;
  return threads >= 64 && threads <= 1024 && (BM * BK / 4) % threads == 0 &&
         (BK * BN / 4) % threads == 0 &&
         (BM * (BK + 4) + BK * (BN + 4) + warps * 256) * 4 <= 48 * 1024;
}

template <size_t I>
void add_09(std::vector<Candidate>& v) {
  // I in [0, 32): BM(2) x BN(2) x BK(2) x WM(2) x WN(2), WN fastest.
  constexpr int BM = k9BMN[I / 16], BN = k9BMN[(I / 8) % 2], BK = k9BK[(I / 4) % 2],
                WM = k9W[(I / 2) % 2], WN = k9W[I % 2];
  if constexpr (valid_09<BM, BN, BK, WM, WN>()) {
    v.push_back({"BM" + std::to_string(BM) + "_BN" + std::to_string(BN) + "_BK" +
                     std::to_string(BK) + "_WM" + std::to_string(WM) + "_WN" + std::to_string(WN),
                 &launch_09_tf32_wmma_cfg<BM, BN, BK, WM, WN>, Precision::TF32});
  }
}

template <size_t... I>
void add_all_09(std::vector<Candidate>& v, std::index_sequence<I...>) {
  (add_09<I>(v), ...);
}

std::vector<Candidate> register_tf32_wmma() {
  std::vector<Candidate> v;
  add_all_09(v, std::make_index_sequence<2 * 2 * 2 * 2 * 2>{});
  return v;
}

const std::map<std::string, std::vector<Candidate> (*)()> kFamilies = {
    {"blocktiling1d", &register_blocktiling1d},
    {"blocktiling2d", &register_blocktiling2d},
    {"vectorized", &register_vectorized},
    {"warptiling", &register_warptiling},
    {"tf32_wmma", &register_tf32_wmma},
};

// -----------------------------------------------------------------------------
struct Shape { int M, N, K; };

std::vector<Shape> parse(const std::string& spec) {
  std::vector<Shape> v;
  std::stringstream ss(spec);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    int M = 0, N = 0, K = 0;
    if (std::sscanf(tok.c_str(), "%dx%dx%d", &M, &N, &K) != 3) M = N = K = std::atoi(tok.c_str());
    if (!dims_fit_int32(M, N, K)) {
      std::fprintf(stderr, "bad size %s\n", tok.c_str());
      std::exit(2);
    }
    v.push_back({M, N, K});
  }
  return v;
}

struct Result { double gflops; bool ok; };

// Operands + reference for one shape, built once and shared by all candidates
// (the cuBLAS reference is the expensive part; paid GPU minutes matter).
struct ShapeCtx {
  Shape s;
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  Reference ref;
  ShapeCtx(const Shape& sh, CublasSgemm& blas) : s(sh) {
    const size_t nA = size_t(s.M) * s.K, nB = size_t(s.K) * s.N, nC = size_t(s.M) * s.N;
    std::vector<float> hA(nA), hB(nB), hC0(nC);
    fill_uniform(hA.data(), nA, 1);
    fill_uniform(hB.data(), nB, 2);
    fill_uniform(hC0.data(), nC, 3);
    CUDA_CHECK(cudaMalloc(&dA, nA * 4));
    CUDA_CHECK(cudaMalloc(&dB, nB * 4));
    CUDA_CHECK(cudaMalloc(&dC, nC * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * 4, cudaMemcpyHostToDevice));
    ref = gpu_reference(blas, s.M, s.N, s.K, 1.0f, dA, dB, 0.0f, hC0);
  }
  ~ShapeCtx() {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
  }
  ShapeCtx(const ShapeCtx&) = delete;
  ShapeCtx& operator=(const ShapeCtx&) = delete;
};

// Verify then time one candidate on one shape.
Result measure(const Candidate& c, ShapeCtx& x, int warmup, int iters, cudaStream_t stream,
               void* flush, size_t flush_bytes) {
  const Shape& s = x.s;
  const size_t nC = size_t(s.M) * s.N;
  CUDA_CHECK(cudaMemset(x.dC, 0xFF, nC * 4));  // NaN: beta == 0 must not read C
  c.launch(s.M, s.N, s.K, 1.0f, x.dA, x.dB, 0.0f, x.dC, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<float> got(nC);
  CUDA_CHECK(cudaMemcpy(got.data(), x.dC, nC * 4, cudaMemcpyDeviceToHost));
  const bool ok =
      compare(got.data(), x.ref.ref.data(), x.ref.scale.data(), nC, s.K, c.precision).pass;
  double gf = 0;
  if (ok) {
    const Run run = [&](cudaStream_t st) { c.launch(s.M, s.N, s.K, 1.0f, x.dA, x.dB, 0.0f, x.dC, st); };
    const Stats st = summarize(time_runs(run, warmup, iters, stream, flush, flush_bytes));
    gf = gemm_flops(s.M, s.N, s.K) / (st.median * 1e6);
  }
  return {gf, ok};
}

// Geometric mean: the right average for ratios/throughputs across sizes, since it
// doesn't let the largest size dominate the ranking.
double geomean(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += std::log(std::max(x, 1e-9));
  return std::exp(s / v.size());
}

}  // namespace

int main(int argc, char** argv) {
  std::string family = "blocktiling1d", tune = "2048,4096", eval = "3072,4097x4093x4099", csv;
  int warmup = 5, iters = 20;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--family") family = next();
    else if (a == "--tune") tune = next();
    else if (a == "--eval") eval = next();
    else if (a == "--warmup") warmup = std::atoi(next().c_str());
    else if (a == "--iters") iters = std::atoi(next().c_str());
    else if (a == "--csv") csv = next();
    else { std::fprintf(stderr, "see bench/autotune.cu header for usage\n"); return 2; }
  }
  const auto fam = kFamilies.find(family);
  if (fam == kFamilies.end()) { std::fprintf(stderr, "unknown family %s\n", family.c_str()); return 2; }
  const std::vector<Candidate> cands = fam->second();
  const std::vector<Shape> tune_s = parse(tune), eval_s = parse(eval);

  const DeviceInfo dev = query_device();
  CublasSgemm blas;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  const size_t flush_bytes = 2 * size_t(dev.l2_bytes);
  void* flush;
  CUDA_CHECK(cudaMalloc(&flush, flush_bytes));

  std::printf("# device=%s family=%s candidates=%zu git=%s\n", dev.name.c_str(), family.c_str(),
              cands.size(), SGEMM_GIT_REV);
  std::ofstream out;
  if (!csv.empty()) {
    out.open(csv);
    out << "config,phase,M,N,K,gflops,verified\n";
  }

  // ---- rank on tune sizes ---------------------------------------------------
  std::vector<std::unique_ptr<ShapeCtx>> tune_ctx;
  for (const Shape& s : tune_s) tune_ctx.push_back(std::make_unique<ShapeCtx>(s, blas));
  std::vector<std::pair<double, size_t>> ranked;
  for (size_t ci = 0; ci < cands.size(); ++ci) {
    std::vector<double> gfs;
    bool all_ok = true;
    for (auto& x : tune_ctx) {
      const Shape& s = x->s;
      const Result r = measure(cands[ci], *x, warmup, iters, stream, flush, flush_bytes);
      all_ok &= r.ok;
      gfs.push_back(r.gflops);
      if (out) out << cands[ci].name << ",tune," << s.M << ',' << s.N << ',' << s.K << ',' << r.gflops << ',' << r.ok << '\n';
    }
    const double g = all_ok ? geomean(gfs) : 0.0;
    std::printf("  %-28s tune geomean %8.1f GFLOP/s %s\n", cands[ci].name.c_str(), g,
                all_ok ? "" : "(FAILED verification: excluded)");
    if (all_ok) ranked.push_back({g, ci});
  }
  if (ranked.empty()) { std::fprintf(stderr, "no valid candidate\n"); return 1; }
  std::sort(ranked.rbegin(), ranked.rend());

  // ---- report the top 3 on held-out eval sizes -------------------------------
  tune_ctx.clear();  // free device memory before building the eval set
  std::vector<std::unique_ptr<ShapeCtx>> eval_ctx;
  for (const Shape& s : eval_s) eval_ctx.push_back(std::make_unique<ShapeCtx>(s, blas));
  std::printf("\nheld-out evaluation (the numbers you may quote):\n");
  for (size_t r = 0; r < std::min<size_t>(3, ranked.size()); ++r) {
    const Candidate& c = cands[ranked[r].second];
    std::vector<double> gfs;
    for (auto& x : eval_ctx) {
      const Shape& s = x->s;
      const Result res = measure(c, *x, warmup, iters, stream, flush, flush_bytes);
      gfs.push_back(res.gflops);
      if (out) out << c.name << ",eval," << s.M << ',' << s.N << ',' << s.K << ',' << res.gflops << ',' << res.ok << '\n';
    }
    std::printf("  #%zu %-28s tune %8.1f  eval %8.1f GFLOP/s\n", r + 1, c.name.c_str(),
                ranked[r].first, geomean(gfs));
  }
  CUDA_CHECK(cudaFree(flush));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
