# Tiled SGEMM in raw CUDA: a measured optimization progression

FP32 GEMM (`C = αAB + βC`, row-major) written from scratch in CUDA C++, one
kernel per optimization stage. Each stage is benchmarked against cuBLAS under
the same protocol, and each speedup is tied to the hardware bottleneck it
removes using Nsight Compute evidence. The narrative is in **[WRITEUP.md](WRITEUP.md)**.

The goal is not to beat cuBLAS. It's a documented, honest progression. Numbers
are reported as **% of cuBLAS FP32** and **% of theoretical FP32 peak** at a
stated, locked clock.

## Status

| Stage | Kernel (`--kernel`) | Targets | Status |
|---:|---|---|---|
| 1 | `naive` | baseline: uncoalesced loads, no reuse | ✅ written, logic-tested |
| 2 | `coalesced` | global-memory coalescing | ✅ written, logic-tested |
| 3 | `smem` | shared-memory tiling (explicit reuse) | ✅ written, logic-tested |
| 4 | `blocktiling1d` | 1D register tiling (TM outputs/thread) | ✅ written, logic-tested |
| 5 | `blocktiling2d` | 2D register tiling (TM×TN outer product) | ⬜ spec only, see `kernels/05_blocktiling_2d.cu` |
| 6 | — | vectorized `float4` access + transposed A tile | ⬜ after stage 5 |
| 7 | — | warptiling | ⬜ stretch |
| 8 | `sgemm_autotune` | template-parameter sweep, held-out eval | ✅ infra done (stage 4 family) |
| 9 | — | tensor cores (TF32), a separate baseline | ⬜ optional |

**No GPU numbers yet.** Everything compiles cleanly for `sm_80`/`sm_86` (nvcc
12.0, no warnings, no spills), and every kernel passes the CPU emulator suite
(see below). The first GPU session starts with `scripts/env_check.sh`.

## Results

_Filled in from `results/results_table.md` after the first locked-clock A100 run._

## Quickstart (cloud GPU)

```bash
scripts/env_check.sh                  # counters readable? toolchain? cuBLAS FP32? peak?
sudo scripts/lock_clocks.sh 1410      # A100 example: pick a clock the card can hold
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="80;86" && cmake --build build -j
build/sgemm_test --kernel all         # correctness gate (checker self-test first)
SM_CLOCK=1410 scripts/run_bench.sh all   # sweep + clock log + plots + table
profiling/profile.sh smem 4096        # ncu full report + compact metrics + summary
scripts/sanitize.sh                   # memcheck / racecheck / synccheck / initcheck
```

Without a GPU (e.g. on a Mac): `tests/emu/run_emu_tests.sh` compiles the real
kernel sources as C++ and runs them on a CPU emulator with ASan.

## Methodology (short version; details in the source headers)

- **Baseline:** `cublasSgemm` with `CUBLAS_DEFAULT_MATH`, which for SGEMM is FP32
  on CUDA cores (TF32 is opt-in). `env_check.sh` confirms from ncu kernel names
  that no TF32/tensor-core kernel is selected. Row-major is mapped to cuBLAS by
  computing `Cᵀ = BᵀAᵀ` (operand swap, no data movement).
- **Timing:** CUDA events around each iteration, 10 warmup + 50 timed, median
  reported (mean/std/min/max in the CSV). L2 flushed (2× capacity memset)
  before every timed iteration, outside the timed window.
- **Verification before timing:** every (kernel, size) is checked against a
  reference before its time is recorded. Failures are recorded as `verified=0`,
  excluded from plots, and make the bench exit non-zero.
- **Error criterion:** `|got − ref| ≤ tol(K) · (|α|·(|A||B|)ᵢⱼ + |β|·|Cᵢⱼ|)`,
  with `tol(K) = ε·(4√K + 8)`. The `|A||B|` normalisation is the standard GEMM
  forward-error scale. The tests *prove* the tolerance is tight: dropping a
  single k-term must fail (`checker_sensitivity`), and an emulated FP32 kernel
  must pass (`tests/test_host.cpp`: observed error ≈ 1 ε, independent of K).
- **β = 0 is BLAS semantics:** C is not read. Tests pre-fill C with NaN to
  enforce it.
- **Reportable runs:** locked clocks, and the clock log recorded next to the
  CSV. Every CSV carries device, clocks, driver/runtime/cuBLAS versions,
  git revision, and methodology flags as `#` header lines.
- **Profiling:** ncu locks clocks to base by default, so ncu *durations* aren't
  comparable to bench numbers. Quote ratios (sectors/request, wavefronts/inst,
  pipe %, stall mix) from ncu, and GFLOPS from the bench.

## Layout

```
kernels/         one file per stage (01_naive.cu … ), templated stages as .cuh + .cu
include/sgemm/   kernel contract + registry, cuBLAS wrapper, checker, timing, device info
src/             registry, cuBLAS reference, GPU-side reference/error scale, device info
bench/           sgemm_bench (sweeps → CSV), sgemm_autotune (stage 8)
tests/           sgemm_test (GPU correctness), test_host.cpp (numerics, no GPU),
                 emu/ (CPU emulator for kernel logic, ASan + divergent-barrier check)
profiling/       profile.sh (ncu), metrics.txt, summarize.py, reports/
scripts/         env_check, lock_clocks, run_bench, sanitize, sass_stats, plot.py
results/         CSVs, clock logs, plots, results_table.md
```

## Tools worth knowing

| Tool | What it answers |
|---|---|
| `scripts/sass_stats.sh` | How many LDS / LDS.128 / FFMA did ptxas *actually* emit? Source-level counting is often wrong (see stage 3/4 notes). |
| `profiling/summarize.py` | Turns raw ncu counters into the diagnostic ratios: sectors/request, wavefronts/inst, FLOP/DRAM byte, top stalls. |
| `sgemm_autotune` | Ranks template configs on `--tune` sizes, reports the winners on disjoint `--eval` sizes (no selection bias). |
| `tests/emu/run_emu_tests.sh` | Kernel logic (indexing, bounds, barriers) without a GPU. Mutation-tested: catches an injected OOB read (via ASan) and a barrier after early `return`. |
| `scripts/sanitize.sh` | The GPU equivalent: memcheck, racecheck, synccheck, initcheck. |
