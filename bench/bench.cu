// Benchmark driver.
//
//   sgemm_bench [--kernel all|cublas|cublas_tf32|NAME[,NAME...]] [--sizes PRESET|LIST]
//               [--warmup 10] [--iters 50] [--alpha 1] [--beta 0]
//               [--sm-clock-mhz MHZ] [--csv PATH] [--no-flush-l2]
//               [--no-verify] [--cublas-math default|pedantic] [--profile]
//               [--tf32-flop-per-clk-sm N] [--max-seconds-per-case S] [--list]
//
//   PRESET: quick | square | rect | odd | all.  LIST: 4096,1000x1500x2003,...
//
// Methodology (each choice is here because it changes the numbers):
//  * cuBLAS runs first at every size and is the "% of cuBLAS" denominator,
//    under the exact same timing protocol as our kernels.
//  * Every kernel is VERIFIED against cuBLAS at that size before it is timed.
//    A wrong kernel gets verified=0 in the CSV. Its speed is meaningless.
//  * Each iteration is timed individually with a CUDA event pair, so we get a
//    distribution rather than one averaged number. GFLOPS uses the median.
//  * L2 is flushed (a memset over 2x L2 capacity) BEFORE each timed iteration,
//    outside the event pair. Without it, at sizes where A+B fit in L2 (A100:
//    40 MB, i.e. <= ~2048^2 for A and B together), iteration i+1 finds its
//    operands L2-resident from iteration i and memory-bound kernels look better
//    than they would inside a real workload. --no-flush-l2 turns it off, e.g.
//    to reproduce published numbers that didn't flush.
//  * At small sizes (<= 512) kernel time is ~microseconds: event resolution
//    (~0.5 us) and launch latency are a real fraction of it. Treat those rows
//    as latency measurements, not throughput.
//  * --max-seconds-per-case S caps warmup + timed iterations per (kernel, size)
//    using one probe launch to estimate cost (min 3 timed iterations). Slow
//    early stages at 8192^3 otherwise dominate paid GPU time. The CSV "iters"
//    column records how many were actually timed; a row with few iterations has
//    a less reliable median, and its std shows it.
//  * Precision-matched baselines. FP32 kernels are compared with FP32 cuBLAS
//    (CUDA cores); TF32 kernels (stage 9) with TF32 cuBLAS (tensor cores), and
//    their % of peak uses the TF32 tensor peak. A TF32 kernel is never shown as
//    a % of FP32 cuBLAS. The "precision" CSV column says which applies.
//  * --profile: run each selected kernel ONCE per size (no warmup, no cuBLAS
//    baseline, no verify) so Nsight Compute captures exactly one launch.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "sgemm/check.hpp"
#include "sgemm/common.hpp"
#include "sgemm/cublas_ref.hpp"
#include "sgemm/device_info.hpp"
#include "sgemm/host_utils.hpp"
#include "sgemm/kernels.hpp"
#include "sgemm/timing.hpp"

using namespace sgemm;

namespace {

struct Shape { int M, N, K; };

struct Options {
  std::string kernels = "all";
  std::string sizes = "square";
  int warmup = 10;
  int iters = 50;
  float alpha = 1.0f;
  float beta = 0.0f;
  int sm_clock_mhz = 0;
  std::string csv;
  bool flush_l2 = true;
  bool verify = true;
  bool profile = false;
  CublasMath math = CublasMath::Default;
  int tf32_flop_per_clk_sm = 0;  // 0 = use the device_info.cu table
  double max_seconds_per_case = 0;  // 0 = no cap
  bool list = false;
};

[[noreturn]] void usage(const char* msg) {
  std::fprintf(stderr, "error: %s\n(see header of bench/bench.cu for usage)\n", msg);
  std::exit(2);
}

std::vector<std::string> split(const std::string& s, char d) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, d))
    if (!t.empty()) out.push_back(t);
  return out;
}

std::vector<Shape> parse_sizes(const std::string& spec) {
  // square: the headline sweep. rect: skinny/fat and short/long-K shapes, which
  // stress different things (short K: prologue/epilogue dominate; long K: the
  // main loop). odd: non-multiples of every tile size we'll use, i.e. the
  // boundary-handling cases, benchmarked as well as tested.
  const std::vector<Shape> square = {{256, 256, 256},    {512, 512, 512},
                                     {1024, 1024, 1024}, {2048, 2048, 2048},
                                     {3072, 3072, 3072}, {4096, 4096, 4096},
                                     {6144, 6144, 6144}, {8192, 8192, 8192}};
  const std::vector<Shape> rect = {{8192, 1024, 1024}, {1024, 8192, 1024},
                                   {4096, 4096, 512},  {1024, 1024, 8192},
                                   {4096, 11008, 4096}};
  const std::vector<Shape> odd = {{1000, 1000, 1000}, {2047, 2047, 2047},
                                  {3001, 3001, 3001}, {4097, 4093, 4099}};
  const std::vector<Shape> quick = {{1024, 1024, 1024}, {2048, 2048, 2048},
                                    {4096, 4096, 4096}};
  if (spec == "square") return square;
  if (spec == "rect") return rect;
  if (spec == "odd") return odd;
  if (spec == "quick") return quick;
  if (spec == "all") {
    std::vector<Shape> v = square;
    v.insert(v.end(), rect.begin(), rect.end());
    v.insert(v.end(), odd.begin(), odd.end());
    return v;
  }
  std::vector<Shape> v;
  for (const auto& tok : split(spec, ',')) {
    const auto p = split(tok, 'x');
    if (p.size() == 1) {
      const int n = std::atoi(p[0].c_str());
      v.push_back({n, n, n});
    } else if (p.size() == 3) {
      v.push_back({std::atoi(p[0].c_str()), std::atoi(p[1].c_str()), std::atoi(p[2].c_str())});
    } else {
      usage(("bad size: " + tok).c_str());
    }
    const Shape& s = v.back();
    if (!dims_fit_int32(s.M, s.N, s.K)) usage(("size out of range: " + tok).c_str());
  }
  return v;
}

Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) usage(("missing value for " + a).c_str());
      return argv[++i];
    };
    if (a == "--kernel") o.kernels = next();
    else if (a == "--sizes") o.sizes = next();
    else if (a == "--warmup") o.warmup = std::atoi(next().c_str());
    else if (a == "--iters") o.iters = std::atoi(next().c_str());
    else if (a == "--alpha") o.alpha = std::strtof(next().c_str(), nullptr);
    else if (a == "--beta") o.beta = std::strtof(next().c_str(), nullptr);
    else if (a == "--sm-clock-mhz") o.sm_clock_mhz = std::atoi(next().c_str());
    else if (a == "--csv") o.csv = next();
    else if (a == "--no-flush-l2") o.flush_l2 = false;
    else if (a == "--no-verify") o.verify = false;
    else if (a == "--profile") o.profile = true;
    else if (a == "--tf32-flop-per-clk-sm") o.tf32_flop_per_clk_sm = std::atoi(next().c_str());
    else if (a == "--max-seconds-per-case") o.max_seconds_per_case = std::atof(next().c_str());
    else if (a == "--list") o.list = true;
    else if (a == "--cublas-math") {
      const std::string m = next();
      if (m == "default") o.math = CublasMath::Default;
      else if (m == "pedantic") o.math = CublasMath::Pedantic;
      else usage("--cublas-math must be default|pedantic");
    } else if (a == "-h" || a == "--help") usage("help requested");
    else usage(("unknown arg: " + a).c_str());
  }
  if (o.iters < 1 || o.warmup < 0) usage("bad --iters/--warmup");
  return o;
}

struct Row {
  std::string kernel;
  int stage;
  Precision precision;
  Shape shape;
  Stats st;
  double gflops_median, gflops_best, pct_cublas, pct_peak;
  int verified;         // 1 pass, 0 fail, -1 not checked
  double max_err_eps;   // max normalised error in units of FLT_EPSILON
};

std::string timestamp() {
  char buf[32];
  const std::time_t t = std::time(nullptr);
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
  return buf;
}

}  // namespace

// Warmup/iteration counts for one case, capped by the time budget. One probe
// launch (untimed by the stats, synchronised) estimates the per-launch cost.
std::pair<int, int> budgeted_counts(const Options& opt, const Run& run, cudaStream_t s) {
  if (opt.profile) return {0, 1};
  if (opt.max_seconds_per_case <= 0) return {opt.warmup, opt.iters};
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a, s));
  run(s);
  CUDA_CHECK(cudaEventRecord(b, s));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  const int fit = static_cast<int>(opt.max_seconds_per_case * 1e3 / std::max(ms, 1e-3f));
  const int iters = std::max(3, std::min(opt.iters, fit * 4 / 5));
  const int warmup = std::min(opt.warmup, std::max(1, fit / 5));
  return {warmup, iters};
}

int main(int argc, char** argv) {
  const Options opt = parse_args(argc, argv);
  if (opt.list) {  // machine-readable: name stage precision
    for (const auto& k : all_kernels())
      std::printf("%s %d %s\n", k.name, k.stage, precision_name(k.precision));
    return 0;
  }
  const std::vector<Shape> shapes = parse_sizes(opt.sizes);

  // Resolve kernel selection. "cublas" / "cublas_tf32" can be named explicitly
  // (e.g. for --profile). Otherwise each baseline runs iff some selected kernel
  // has that precision, so every kernel gets its precision-matched denominator.
  bool sel_cublas = false, sel_cublas_tf32 = false;
  std::vector<const KernelSpec*> kernels;
  if (opt.kernels == "all") {
    for (const auto& k : all_kernels()) kernels.push_back(&k);
  } else {
    for (const auto& name : split(opt.kernels, ',')) {
      if (name == "cublas") { sel_cublas = true; continue; }
      if (name == "cublas_tf32") { sel_cublas_tf32 = true; continue; }
      const KernelSpec* k = find_kernel(name);
      if (!k) usage(("unknown kernel: " + name).c_str());
      kernels.push_back(k);
    }
  }
  bool any_fp32 = false, any_tf32 = false;
  for (const KernelSpec* k : kernels) (k->precision == Precision::TF32 ? any_tf32 : any_fp32) = true;
  const bool want_cublas_timed = sel_cublas || (!opt.profile && any_fp32);
  const bool want_cublas_tf32 = sel_cublas_tf32 || (!opt.profile && any_tf32);

  DeviceInfo dev = query_device();
  if (opt.tf32_flop_per_clk_sm > 0) {
    dev.tf32_flop_per_clk_sm = opt.tf32_flop_per_clk_sm;
    dev.tf32_source = "user_override";
  }
  CublasSgemm blas(opt.math);
  CublasSgemm blas_tf32(CublasMath::TF32);
  const int clock_mhz = opt.sm_clock_mhz > 0 ? opt.sm_clock_mhz : dev.max_sm_clock_mhz;
  const double peak = dev.peak_fp32_gflops(clock_mhz);
  const double peak_tf32 = dev.peak_tf32_gflops(clock_mhz);

  // ---- metadata: everything needed to reproduce / judge a number ----------
  std::ostringstream meta;
  meta << "# timestamp=" << timestamp() << "\n"
       << "# git_rev=" << SGEMM_GIT_REV << "\n"
       << "# device=" << dev.name << "\n"
       << "# compute_capability=" << dev.cc_major << "." << dev.cc_minor << "\n"
       << "# sm_count=" << dev.sm_count << "\n"
       << "# fp32_lanes_per_sm=" << dev.fp32_lanes_per_sm << "\n"
       << "# max_sm_clock_mhz=" << dev.max_sm_clock_mhz << "\n"
       << "# peak_clock_mhz=" << clock_mhz << "\n"
       << "# peak_clock_source=" << (opt.sm_clock_mhz > 0 ? "user_locked" : "driver_max_boost") << "\n"
       << "# peak_fp32_gflops=" << peak << "\n"
       << "# tf32_flop_per_clk_sm=" << dev.tf32_flop_per_clk_sm << "\n"
       << "# tf32_peak_source=" << dev.tf32_source << "\n"
       << "# peak_tf32_gflops=" << peak_tf32 << "\n"
       << "# peak_dram_gbs=" << dev.peak_dram_gbs() << "\n"
       << "# l2_bytes=" << dev.l2_bytes << "\n"
       << "# driver_version=" << dev.driver_version << "\n"
       << "# runtime_version=" << dev.runtime_version << "\n"
       << "# cublas_version=" << blas.version() << "\n"
       << "# cublas_math=" << blas.mode_name() << "\n"
       << "# alpha=" << opt.alpha << "\n"
       << "# beta=" << opt.beta << "\n"
       << "# warmup=" << opt.warmup << "\n"
       << "# iters=" << opt.iters << "\n"
       << "# flush_l2=" << (opt.flush_l2 ? 1 : 0) << "\n"
       << "# flops_convention=2MNK\n";
  std::fputs(meta.str().c_str(), stdout);
  if (opt.sm_clock_mhz <= 0 && !opt.profile) {
    std::fprintf(stderr,
                 "[bench] WARNING: %% of peak uses the driver's max boost clock (%d MHz). "
                 "For reportable runs lock clocks (scripts/lock_clocks.sh) and pass "
                 "--sm-clock-mhz.\n", dev.max_sm_clock_mhz);
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  void* flush_buf = nullptr;
  const size_t flush_bytes = 2 * static_cast<size_t>(dev.l2_bytes);
  if (opt.flush_l2 && !opt.profile) CUDA_CHECK(cudaMalloc(&flush_buf, flush_bytes));

  std::vector<Row> rows;
  std::printf("\n%-14s %4s %6s %6s %6s %10s %9s %8s %9s %8s %7s\n", "kernel", "prec", "M", "N",
              "K", "median_ms", "std_ms", "GFLOPS", "%cuBLAS", "%peak", "check");

  for (size_t si = 0; si < shapes.size(); ++si) {
    const Shape s = shapes[si];
    const size_t nA = size_t(s.M) * s.K, nB = size_t(s.K) * s.N, nC = size_t(s.M) * s.N;
    const double flops = gemm_flops(s.M, s.N, s.K);

    std::vector<float> hA(nA), hB(nB), hC0(nC);
    fill_uniform(hA.data(), nA, 1000 + 3 * si);
    fill_uniform(hB.data(), nB, 1001 + 3 * si);
    fill_uniform(hC0.data(), nC, 1002 + 3 * si);

    float *dA, *dB, *dC, *dC0;
    CUDA_CHECK(cudaMalloc(&dA, nA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, nB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC0, nC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC0, hC0.data(), nC * sizeof(float), cudaMemcpyHostToDevice));

    // Reset C to its initial state. For beta == 0, fill with NaN (0xFFFFFFFF)
    // so a kernel that illegally reads C fails verification.
    auto reset_C = [&]() {
      if (opt.beta != 0.0f)
        CUDA_CHECK(cudaMemcpy(dC, dC0, nC * sizeof(float), cudaMemcpyDeviceToDevice));
      else
        CUDA_CHECK(cudaMemset(dC, 0xFF, nC * sizeof(float)));
    };

    Reference ref;
    const bool do_verify = opt.verify && !opt.profile && (!kernels.empty() || want_cublas_tf32);
    if (do_verify) ref = gpu_reference(blas, s.M, s.N, s.K, opt.alpha, dA, dB, opt.beta, hC0);

    // cublas_gf / pk are the PRECISION-MATCHED baseline and peak.
    auto emit = [&](const std::string& name, int stage, Precision prec, const Stats& st,
                    double cublas_gf, int verified, double err_eps) {
      const double pk = prec == Precision::TF32 ? peak_tf32 : peak;
      Row r{name, stage, prec, s, st, flops / (st.median * 1e6), flops / (st.min * 1e6), 0, 0,
            verified, err_eps};
      r.pct_cublas = cublas_gf > 0 ? 100.0 * r.gflops_median / cublas_gf : 0.0;
      r.pct_peak = pk > 0 ? 100.0 * r.gflops_median / pk : 0.0;
      std::printf("%-14s %4s %6d %6d %6d %10.4f %9.4f %8.1f %8.1f%% %7.1f%% %7s\n",
                  name.c_str(), precision_name(prec), s.M, s.N, s.K, st.median, st.stddev,
                  r.gflops_median, r.pct_cublas, r.pct_peak,
                  verified == 1 ? "ok" : verified == 0 ? "FAIL" : "-");
      rows.push_back(r);
    };

    // ---- cuBLAS baseline ---------------------------------------------------
    double cublas_gflops = 0.0;
    if (want_cublas_timed) {
      reset_C();
      const Run run = [&](cudaStream_t st) {
        blas(s.M, s.N, s.K, opt.alpha, dA, dB, opt.beta, dC, st);
      };
      const auto [nw, ni] = budgeted_counts(opt, run, stream);
      const Stats st = summarize(time_runs(run, nw, ni, stream, flush_buf, flush_bytes));
      cublas_gflops = flops / (st.median * 1e6);
      emit("cublas", 0, Precision::FP32, st, cublas_gflops, -1, 0.0);
    }

    // ---- cuBLAS TF32 baseline (only alongside TF32 kernels) -----------------
    // Verified against the FP32 reference with the TF32 tolerance: this also
    // confirms the TF32 baseline itself is numerically what we think it is.
    double cublas_tf32_gflops = 0.0;
    if (want_cublas_tf32) {
      int verified = -1;
      double err_eps = 0.0;
      if (do_verify) {
        reset_C();
        blas_tf32(s.M, s.N, s.K, opt.alpha, dA, dB, opt.beta, dC, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> got(nC);
        CUDA_CHECK(cudaMemcpy(got.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
        const CheckResult cr = compare(got.data(), ref.ref.data(), ref.scale.data(), nC, s.K,
                                       Precision::TF32);
        verified = cr.pass ? 1 : 0;
        err_eps = cr.max_err / FLT_EPSILON;
      }
      reset_C();
      const Run run = [&](cudaStream_t st) {
        blas_tf32(s.M, s.N, s.K, opt.alpha, dA, dB, opt.beta, dC, st);
      };
      const auto [nw, ni] = budgeted_counts(opt, run, stream);
      const Stats st = summarize(time_runs(run, nw, ni, stream, flush_buf, flush_bytes));
      cublas_tf32_gflops = flops / (st.median * 1e6);
      emit("cublas_tf32", 0, Precision::TF32, st, cublas_tf32_gflops, verified, err_eps);
    }

    // ---- our kernels -------------------------------------------------------
    for (const KernelSpec* k : kernels) {
      int verified = -1;
      double err_eps = 0.0;
      if (do_verify) {
        reset_C();
        k->launch(s.M, s.N, s.K, opt.alpha, dA, dB, opt.beta, dC, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> got(nC);
        CUDA_CHECK(cudaMemcpy(got.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
        const CheckResult cr =
            compare(got.data(), ref.ref.data(), ref.scale.data(), nC, s.K, k->precision);
        verified = cr.pass ? 1 : 0;
        err_eps = cr.max_err / FLT_EPSILON;
        if (!cr.pass)
          std::fprintf(stderr, "[bench] %s %dx%dx%d FAILED: %lld bad, worst idx %lld got %g want %g\n",
                       k->name, s.M, s.N, s.K, cr.num_bad, cr.worst_idx, cr.got_worst, cr.ref_worst);
      }
      reset_C();
      const Run run = [&](cudaStream_t st) {
        k->launch(s.M, s.N, s.K, opt.alpha, dA, dB, opt.beta, dC, st);
      };
      const auto [nw, ni] = budgeted_counts(opt, run, stream);
      const Stats st = summarize(time_runs(run, nw, ni, stream, flush_buf, flush_bytes));
      emit(k->name, k->stage, k->precision, st,
           k->precision == Precision::TF32 ? cublas_tf32_gflops : cublas_gflops, verified, err_eps);
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dC0));
  }

  if (flush_buf) CUDA_CHECK(cudaFree(flush_buf));
  CUDA_CHECK(cudaStreamDestroy(stream));

  if (!opt.csv.empty()) {
    std::ofstream f(opt.csv);
    if (!f) usage(("cannot open " + opt.csv).c_str());
    f << meta.str();
    f << "kernel,stage,precision,M,N,K,ms_median,ms_mean,ms_std,ms_min,ms_max,iters,"
         "gflops_median,gflops_best,pct_cublas,pct_peak,verified,max_err_eps\n";
    for (const Row& r : rows) {
      f << r.kernel << ',' << r.stage << ',' << precision_name(r.precision) << ',' << r.shape.M << ',' << r.shape.N << ','
        << r.shape.K << ',' << r.st.median << ',' << r.st.mean << ',' << r.st.stddev << ','
        << r.st.min << ',' << r.st.max << ',' << r.st.n << ',' << r.gflops_median << ','
        << r.gflops_best << ',' << r.pct_cublas << ',' << r.pct_peak << ',' << r.verified
        << ',' << r.max_err_eps << '\n';
    }
    std::printf("\nwrote %s\n", opt.csv.c_str());
  }

  // Non-zero exit if any kernel produced wrong results, so scripts notice.
  for (const Row& r : rows)
    if (r.verified == 0) return 1;
  return 0;
}
