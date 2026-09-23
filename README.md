# Tiled SGEMM in raw CUDA: a measured optimization progression

FP32 GEMM (`C = αAB + βC`, row-major) written from scratch in CUDA C++, one
kernel per optimization stage, plus a TF32 tensor-core stage as a separate arc.
Each stage is benchmarked against cuBLAS under the same protocol, and each
speedup is tied to the hardware bottleneck it removes using Nsight Compute
evidence. The narrative is in **[WRITEUP.md](WRITEUP.md)**.

The goal is not to beat cuBLAS. It's a documented, honest progression. Numbers
are reported as **% of precision-matched cuBLAS** and **% of theoretical peak**
at a stated, locked clock.

## Status

| Stage | Kernel (`--kernel`) | Targets | Status |
|---:|---|---|---|
| 1 | `naive` | baseline: uncoalesced loads, no reuse | ✅ |
| 2 | `coalesced` | global-memory coalescing | ✅ |
| 3 | `smem` | shared-memory tiling (explicit reuse) | ✅ |
| 4 | `blocktiling1d` | 1D register tiling (TM outputs/thread) | ✅ |
| 5 | `blocktiling2d` | 2D register tiling (8×8 outer product) | ✅ |
| 6 | `vectorized` | float4 global access, transposed + padded A, split thread tile | ✅ |
| 7 | `warptiling` | explicit warp tile (8×4 lanes, 2×2 sub-tiles) | ✅ |
| 8 | `sgemm_autotune` | template sweep per family, held-out evaluation | ✅ |
| 9 | `tf32_wmma` | TF32 tensor cores (WMMA 16×16×8), TF32 baseline | ✅ |

✅ = written, compiles for sm_80/sm_86 with no warnings and 0 spills in the
default configs, and passes the CPU emulator suite (18 template
configurations, 468 cases, ASan on). **Not yet run on a GPU. No performance
numbers exist yet.** The first GPU session is one command (below).

## Results

_Filled in from `RUN_SUMMARY.md` after the first locked-clock A100 run._

## Quickstart (cloud GPU)

```bash
SM_CLOCK=1410 scripts/full_run.sh     # everything, into results/run_<stamp>_<gpu>/
```

That runs, in order (cheap gates first, so a bad box costs minutes):

1. Environment check.
2. Build.
3. Correctness.
4. compute-sanitizer.
5. Clock lock.
6. Benchmark sweep with a clock log.
7. Autotuning for 5 families.
8. ncu on every stage and both cuBLAS baselines.

It then plots and writes `RUN_SUMMARY.md` with an environment block, results
tables, a cross-stage attribution table, and tuned-vs-default numbers. It takes
about 30–50 min on an A100. Use `UNLOCKED=1` instead of `SM_CLOCK` on a dev card
without sudo, and `SKIP_SANITIZE/SKIP_AUTOTUNE/SKIP_PROFILE=1` to skip steps.

Individual pieces:

```bash
scripts/env_check.sh                       # counters readable? cuBLAS FP32? peaks?
build/sgemm_test --kernel all              # correctness gate
build/sgemm_bench --kernel warptiling,tf32_wmma --sizes 4096
build/sgemm_autotune --family warptiling
profiling/profile.sh vectorized 4096       # ncu report + compact metrics + summary
```

Without a GPU (e.g. on a Mac): `tests/emu/run_emu_tests.sh` compiles the real
kernel sources as C++ and runs them on a CPU emulator with ASan (needs a C++20
compiler with `<barrier>`).

## Methodology (short version; details in the source headers)

- **Precision-matched baselines.** FP32 stages (1–7) are compared with
  `cublasSgemm` under `CUBLAS_DEFAULT_MATH`, which is FP32 on CUDA cores.
  `env_check.sh` confirms from ncu kernel names that no TF32 kernel is picked.
  The TF32 stage (9) is compared only with cuBLAS in `CUBLAS_TF32_TENSOR_OP_MATH`,
  and its % of peak uses the TF32 tensor peak. The two are never plotted on the
  same axes.
- **Timing.** CUDA events around each iteration, 10 warmup + 50 timed, median
  reported (mean/std/min/max in the CSV). L2 is flushed before every timed
  iteration, outside the timed window. `--max-seconds-per-case` caps very slow
  cases, and the CSV records the actual iteration count.
- **Verification before timing.** Every (kernel, size) is checked before its
  time is recorded. Failures are recorded, excluded from plots, and make the
  bench exit non-zero.
- **Error criterion.** `|got − ref| ≤ tol · (|α|·(|A||B|)ᵢⱼ + |β|·|Cᵢⱼ|)`.
  - FP32: `tol = ε·(16 + √K/2)`.
  - TF32: an extra rigorous input-rounding term of 2⁻⁹.

  The tests pin both from both sides:
  - dropping one k-term must fail;
  - an emulated correct kernel must pass (observed FP32 error ≈ 1–2 ε at any K);
  - a TF32 result must **fail** the FP32 check, so an accidental tensor-core
    path can't pass as FP32. An earlier, looser FP32 tolerance failed this last
    test, which is why it changed.
- **β = 0 is BLAS semantics:** C is not read. Tests pre-fill C with NaN.
- **Reportable runs:** locked clocks, and a clock log next to the CSV. Every CSV
  carries device, clocks, peaks and their source, driver/runtime/cuBLAS
  versions, git revision, and methodology flags as `#` header lines.
- **Profiling:** ncu locks clocks to base by default. Quote ratios
  (sectors/request, wavefronts/inst, pipe %, stall mix) from ncu, and GFLOP/s
  from the bench.

## Layout

```
kernels/         one stage per file: 01_naive.cu … 07_warptiling.cuh, 09_tf32_wmma.cuh
                 (templated stages: .cuh = kernel + explanation, .cu = default config)
include/sgemm/   kernel contract + registry, precision tag, cuBLAS wrapper, checker,
                 timing, device info (FP32 + TF32 peaks)
src/             registry, cuBLAS reference, GPU-side reference/error scale, device info
bench/           sgemm_bench (sweeps → CSV), sgemm_autotune (stage 8)
tests/           sgemm_test (GPU correctness), test_host.cpp (numerics, no GPU),
                 emu/ (CPU emulator: CUDA + WMMA shims, variant configs, ASan)
profiling/       profile.sh (ncu), metrics.txt, summarize.py, reports/
scripts/         full_run, env_check, lock_clocks, run_bench, sanitize, sass_stats,
                 plot.py, collate.py
results/         run directories, CSVs, plots
```

## Tools worth knowing

| Tool | What it answers |
|---|---|
| `scripts/full_run.sh` | Everything needed for the writeup, in one run directory. |
| `scripts/collate.py` | The attribution table: every stage side by side on the ratios that explain its speed. |
| `scripts/sass_stats.sh` | How many LDS / LDS.128 / FFMA / HMMA did ptxas *actually* emit? Source-level counting was wrong three times in this project. |
| `profiling/summarize.py` | Raw ncu counters → sectors/request, wavefronts/inst, FLOP/DRAM byte, top stalls. |
| `sgemm_autotune` | Ranks configs on `--tune` sizes, reports the winners *and the default* on disjoint `--eval` sizes. |
| `tests/emu/run_emu_tests.sh` | Kernel logic without a GPU, including non-default template configs. Mutation-tested: catches an injected OOB read and a barrier after early `return`. |
| `scripts/sanitize.sh` | The GPU equivalent: memcheck, racecheck, synccheck, initcheck. |
