# From 1% to ?% of cuBLAS: building an FP32 GEMM on Ampere, one bottleneck at a time

> **Draft.** Sections marked **TBD(measure)** get filled from the first
> locked-clock A100 run. Each "Prediction" was written *before* measurement.
> Keep them, and state plainly where the measurement disagreed. That
> disagreement is the most credible part of a writeup like this.

## 1. Setup

**Problem.** `C = αAB + βC`, FP32, row-major, packed. FLOPs counted as 2MNK.

**Hardware.** TBD(measure): exact card (A100-SXM4-40GB / 80GB / PCIe), locked
SM clock, driver, CUDA toolkit, cuBLAS version. All of it is in each CSV header.

**Ceilings on A100** (at the 1410 MHz boost clock; recompute at the locked clock):

| | value | derivation |
|---|---:|---|
| FP32 peak | 19.5 TFLOP/s | 108 SMs × 64 FP32 lanes × 2 FLOP/FFMA × 1.41 GHz |
| DRAM bandwidth | ~1.56 TB/s (40 GB) | 5120-bit HBM2 × 1215 MHz × 2 |
| Ridge point | ~12.5 FLOP/B | peak ÷ bandwidth |
| GEMM intensity at n=4096 | ~683 FLOP/B | 2n³ / (3 · 4n²) = n/6, assuming each matrix is touched once |

So large SGEMM is compute-bound by ~50×. Every stage below is about
closing the gap between that and what the kernel *actually* extracts. The
limiting resource shifts from global-memory transactions, to shared-memory
instruction issue, to latency/ILP, and the profiler shows which.

**Methodology.** See README → Methodology. In short: cuBLAS FP32 (no TF32)
under the identical timing protocol; median of 50 event-timed iterations after
10 warmups; L2 flushed before each; every kernel verified before timing
against a `|A||B|`-scaled tolerance that is itself tested for sensitivity.

## 2. Results at a glance

TBD(measure): `results/results_table.md` and `results/plots/stage_progression_4096.png`,
`results/plots/gflops_vs_size.png`.

## 3. Stage by stage

Each stage lists: the change, the bottleneck it targets, the prediction
(written before measurement), and the evidence.

### Stage 1: Naive (`kernels/01_naive.cu`)

**Change.** One thread per output; `threadIdx.x` walks *rows* of C.

**Bottleneck.** A warp is 32 consecutive rows at one column, so every
`A[row*K + k]` load touches 32 different cache lines (32 sectors for 128 useful
bytes; ideal is 4). B is a single broadcast sector.

**Prediction.** Not DRAM-bound: the 32 warps of a block share the same 32 A
rows, so L1 hit rate should be high. Instead the L1TEX/LSU pipe is swamped by
wavefronts. Expect gld sectors/request ≈ 16.5 (the average of 32 and 1),
dominant stalls **LG Throttle** + **Long Scoreboard**, FMA pipe in single
digits, ~1–3% of cuBLAS.

**Evidence.** TBD(measure): table from `profiling/reports/naive_4096.metrics.summary.md`.

### Stage 2: Global memory coalescing (`kernels/02_coalesced.cu`)

**Change.** Swap the thread→(row, col) mapping. Nothing else. `diff` against stage 1.

**Bottleneck removed.** A warp now covers 32 consecutive *columns*: A is a
1-sector broadcast, B and the C store are 4-sector coalesced accesses. Sectors
per warp per k: 33 → 5.

**Prediction.** gld sectors/request ≈ 2.5, "uncoalesced access" warning gone,
LG-throttle share collapses, 5–10× over stage 1.

**Evidence.** TBD(measure).

### Stage 3: Shared-memory tiling (`kernels/03_smem_tiling.cu`)

**Change.** 32×32 tiles of A and B staged cooperatively in shared memory. Two
barriers per tile. Zero-fill at the edges replaces per-element bounds checks in
the inner loop.

**Bottleneck targeted.** Redundant global traffic. Global bytes per FLOP drop 32×
(8 FLOP/B at the block level, up from 0.25).

**What the SASS actually says** (`scripts/sass_stats.sh`, sm_80). The source has
2 LDS per FFMA. ptxas vectorized the As row reads into `LDS.128`, so per 32-k tile
it emitted **32 LDS + 8 LDS.128 : 32 FFMA = 1.25 LDS instructions per FFMA**.
Against the per-SM rates (≈1 warp-wide LDS/clk vs 2 warp-wide FFMA/clk), that's
a ~40% of peak ceiling from shared-memory issue alone.

**Prediction.** Smaller gain over stage 2 than the 32× traffic cut suggests
(1.3–2×), because stage 2 already enjoyed implicit L1/L2 reuse. The
new top stalls should be **MIO Throttle / Short Scoreboard** (smem pipe) and
**Barrier**, with DRAM and L2 throughput far from peak.

**Evidence.** TBD(measure). Check whether the broadcast `LDS.128` costs 1 wavefront
(lds wavefronts/inst).

### Stage 4: 1D register tiling (`kernels/04_blocktiling_1d.cuh`)

**Change.** 64×64 block tile, BK = 8, 512 threads. Each thread accumulates TM = 8
outputs in registers and reuses each loaded B value 8 times.

**Bottleneck targeted.** Shared-memory instruction throughput and per-output
overhead.

**What the SASS says.** 8 LDS + 16 LDS.128 : 64 FFMA = **0.375 LDS per FFMA**
(down from 1.25). The LDS pipe stops being the binding ceiling.

**Prediction.** Roughly 2–3× over stage 3. Stalls shift toward **Long
Scoreboard** (global loads fully exposed: nothing overlaps LDG → STS → barrier)
and **Barrier** (a barrier pair every 8 k-steps). FFMA share of issued
instructions rises sharply.

**Evidence.** TBD(measure).

### Stage 5: 2D register tiling (`kernels/05_blocktiling_2d.cu`, written by Fasih)

TBD. Spec, bars, and pitfalls are in the source file header.

### Stage 6: Vectorized access + transposed A tile

TBD (after stage 5).

### Stage 7: Warptiling (stretch)

TBD.

### Stage 8: Autotuning (`bench/autotune.cu`)

Configurations are ranked on one set of sizes and the winners are reported on a
disjoint set, so the quoted number isn't inflated by selecting the luckiest of
~57 configurations. TBD(measure): stage 4 default vs tuned, on held-out sizes.

## 4. Where predictions were wrong

- *(Pre-measurement, from SASS)* The source-level "loads per FFMA" arithmetic was
  wrong for both stage 3 (2 → actually 1.25) and stage 4 (1.125 → actually
  0.375), because ptxas vectorizes contiguous shared-memory reads. The ratio
  that bounds a kernel is the one in the SASS, not the one in the source.
- TBD(measure): add every case where the profiler disagreed with a prediction.

## 5. Threats to validity

- **Clock behaviour.** Numbers are at a locked clock; the clock log next to each
  CSV shows whether it held. Unlocked, results vary by ~10%.
- **cuBLAS heuristics.** cuBLAS picks different kernels per shape. % of cuBLAS
  is shape-dependent, and non-tile-multiple sizes can penalize cuBLAS less than us.
- **L2 flushing** makes small-size numbers more conservative than
  unflushed published series. Compare like with like.
- **Small sizes** (≤ 512³) are latency measurements (µs-scale kernels vs
  ~0.5 µs event resolution plus launch overhead), not throughput.
- **One card.** Conclusions about *which* bottleneck binds are A100-specific. On
  GA10x/AD10x the FP32:INT ratio and the smem/L1 split differ.

## 6. What I'd do next

TBD: double buffering (overlap global loads with compute), `cp.async`, and the
TF32 tensor-core arc (against a TF32 cuBLAS baseline).
