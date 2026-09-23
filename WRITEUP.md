# From 1% to ?% of cuBLAS: building an FP32 GEMM on Ampere, one bottleneck at a time

> **Draft.** Sections marked **TBD(measure)** get filled from the first
> locked-clock A100 run (`scripts/full_run.sh` → `RUN_SUMMARY.md`). Every
> "Prediction" below was written *before* measurement. Keep them, and state
> plainly where the measurement disagreed. That disagreement is the most
> credible part of a writeup like this.

## 1. Setup

**Problem.** `C = αAB + βC`, FP32 in and out, row-major, packed. FLOPs counted as 2MNK.

**Hardware.** TBD(measure): exact card (A100-SXM4-40GB / 80GB / PCIe), locked
SM clock, driver, CUDA toolkit, cuBLAS version. All of it is in each CSV header
and in `RUN_SUMMARY.md` §1.

**Ceilings on A100** (at the 1410 MHz boost clock; recompute at the locked clock):

| | value | derivation |
|---|---:|---|
| FP32 peak (CUDA cores) | 19.5 TFLOP/s | 108 SMs × 64 FP32 lanes × 2 FLOP/FFMA × 1.41 GHz |
| TF32 peak (tensor cores, dense) | 156 TFLOP/s | 108 SMs × 1024 FLOP/clk/SM × 1.41 GHz |
| DRAM bandwidth | ~1.56 TB/s (40 GB) | 5120-bit HBM2 × 1215 MHz × 2 |
| FP32 ridge point | ~12.5 FLOP/B | FP32 peak ÷ bandwidth |
| GEMM intensity at n=4096 | ~683 FLOP/B | 2n³ / (3 · 4n²) = n/6, each matrix touched once |

So large SGEMM is compute-bound by ~50× even for FP32. Every stage is about
closing the gap between that and what the kernel *actually* extracts. The
limiting resource moves through global-memory transactions, shared-memory
instruction issue, shared-memory bandwidth, and latency/ILP, and the profiler
shows which one binds at each stage.

**Methodology.** See README → Methodology. In short:

- **Baselines:** each kernel against precision-matched cuBLAS, under an
  identical timing protocol.
- **Timing:** median of up to 50 event-timed iterations after 10 warmups, with
  L2 flushed before each.
- **Verification:** every kernel is verified before timing, against a
  `|A||B|`-scaled tolerance that is itself tested from both sides.

## 2. Results at a glance

TBD(measure): `RUN_SUMMARY.md` §2, `plots/stage_progression_4096.png`,
`plots/gflops_vs_size.png`, and the TF32 counterparts.

## 3. Stage by stage

Each stage gives the change, the bottleneck it targets, the prediction (written
before measurement), and the evidence (`RUN_SUMMARY.md` §3 attribution table).

### Stage 1: Naive (`kernels/01_naive.cu`)

**Change.** One thread per output; `threadIdx.x` walks *rows* of C.

**Bottleneck.** A warp is 32 consecutive rows at one column, so every
`A[row*K + k]` load touches 32 different cache lines (32 sectors for 128 useful
bytes; ideal is 4). B is a single broadcast sector.

**Prediction.** Not DRAM-bound: the 32 warps of a block share the same 32 A
rows, so L1 hit rate should be high. Instead the L1TEX/LSU pipe is swamped by
wavefronts. Expect:

- gld sectors/request ≈ 16.5 (the average of 32 and 1);
- dominant stalls **LG Throttle** + **Long Scoreboard**;
- FMA pipe in single digits;
- ~1–3% of cuBLAS.

**Evidence.** TBD(measure).

### Stage 2: Global memory coalescing (`kernels/02_coalesced.cu`)

**Change.** Swap the thread→(row, col) mapping. Nothing else. `diff` against stage 1.

**Bottleneck removed.** A warp now covers 32 consecutive *columns*: A is a
1-sector broadcast, B and the C store are 4-sector coalesced accesses. Sectors
per warp per k: 33 → 5.

**Prediction.** gld sectors/request ≈ 2.5, "uncoalesced access" warning gone,
LG-throttle share collapses, 5–10× over stage 1.

**Evidence.** TBD(measure).

### Stage 3: Shared-memory tiling (`kernels/03_smem_tiling.cu`)

**Change.** 32×32 tiles of A and B staged cooperatively in shared memory, two
barriers per tile, zero-fill at the edges.

**Bottleneck targeted.** Redundant global traffic: 8 FLOP per global byte at the
block level, up from 0.25.

**SASS says.** The source has 2 LDS per FFMA, but ptxas vectorized the A-row
reads into `LDS.128`: **40 LDS instructions : 32 FFMA = 1.25**. Against ~1
warp-wide LDS/clk vs 2 warp-wide FFMA/clk per SM, that's a ~40%-of-peak ceiling.

**Prediction.** A smaller gain over stage 2 than the 32× traffic cut suggests
(1.3–2×), because stage 2 already had implicit L1/L2 reuse. New top stalls:
**MIO Throttle / Short Scoreboard** and **Barrier**.

**Evidence.** TBD(measure).

### Stage 4: 1D register tiling (`kernels/04_blocktiling_1d.cuh`)

**Change.** 64×64 block tile, BK = 8, 512 threads. Each thread accumulates 8
outputs in registers and reuses each loaded B value 8 times.

**SASS says.** **0.375 LDS per FFMA** (8 LDS + 16 LDS.128 : 64 FFMA), down from 1.25.

**Prediction.** 2–3× over stage 3. Stalls shift to **Long Scoreboard**
(LDG → STS → barrier is fully exposed) and **Barrier** (a barrier pair every 8
k-steps).

**Evidence.** TBD(measure).

### Stage 5: 2D register tiling (`kernels/05_blocktiling_2d.cuh`)

**Change.** 128×128 block tile, 256 threads, each owning an 8×8 register tile.
Per k-step: load 8 A and 8 B values, then 64 FFMAs (an outer product).

**SASS says.** 32 LDS.128 per 512 main-loop FFMAs (0.0625 LDS instructions per
FFMA). **133 registers.** That's just over the 128 cutoff, so **one block (8
warps) per SM, 12.5% occupancy.**

**Prediction.** The largest single jump: 1.5–2.5× over stage 4, landing near
60–70% of cuBLAS on published Ampere series. Low occupancy is survivable
because each thread has 64 independent FFMA chains (ILP). Expect lds
wavefronts/inst well above ideal from two known conflicts:

- **A reads:** the two threadRows in a warp read rows 64 words apart, so they
  hit the same 4 banks.
- **B reads:** at a 32-byte stride, 16 threads touch only half the banks.

Experiment: `__launch_bounds__(256, 2)` forces ≤128 registers. Does 2
blocks/SM beat the resulting spills?

**Evidence.** TBD(measure).

### Stage 6: Vectorized access + conflict-free layout (`kernels/06_vectorized.cuh`)

**Change.**

- `float4` global loads and stores.
- A stored transposed in shared memory, padded by 4 floats.
- Each thread's 8×8 tile split into two 4-wide halves BM/2 apart, so every
  warp-wide `LDS.128` reads one contiguous span.
- Scalar fallback when K or N isn't a multiple of 4.

**SASS says.** Same 32 LDS.128 and 576 FFMA as stage 5. The smem *instruction*
count doesn't change. What changes: LDG.E 64 → LDG.E.128 16, STG.E 128 →
STG.E.128 32, and registers 133 → 112, which means 2 blocks/SM.

**Prediction.** lds wavefronts/inst drops toward ideal; sts bank conflicts ≈ 0.
Gain comes from fewer memory instructions, fewer wavefronts, and doubled
occupancy. Plausibly 10–30% over stage 5.

**Evidence.** TBD(measure).

### Stage 7: Warptiling (`kernels/07_warptiling.cuh`)

**Change.** Each warp owns a compact 64×32 region: lanes laid out 8×4, 2×2
sub-tiles of 32×16. Same 64 outputs per thread, just a different assignment.

**Bottleneck targeted.** Shared-memory bandwidth per warp: 4 wavefronts per
k-step instead of stage 6's 6, and 384 vs 576 unique bytes.

**SASS says.** Identical instruction mix to stage 6. The evidence can only come
from ncu wavefront counts. 113 registers, 2 blocks/SM.

**Prediction.** A modest gain (0–15%) at the default config, because smem
bandwidth isn't binding at stage 6. The real payoff is structural: the warp tile
is a tunable dimension for stage 8, and it's exactly the decomposition tensor
cores need (stage 9).

**Evidence.** TBD(measure).

### Stage 8: Autotuning (`bench/autotune.cu`)

**Method.** Five families: stages 4, 5, 6, 7, and 9, with ~6–57 valid configs
each.

1. Rank configs on 2048³ and 4096³.
2. Report the top 3 *and the registered default* on held-out 3072³ and
   4097×4093×4099.

That way "tuning gained X%" is measured on sizes the ranking never saw.

**Prediction.** Single-digit to low-double-digit % over the defaults. The
biggest gains come where the default is clearly suboptimal for the card
(occupancy-limited configs).

**Evidence.** TBD(measure): `RUN_SUMMARY.md` §4.

### Stage 9: TF32 tensor cores (`kernels/09_tf32_wmma.cuh`), a separate arc

**Change.** Stage 7's per-lane 4×4 FFMA tile is replaced by warp-wide WMMA
16×16×8 TF32 MMAs. Each warp does 4×2 fragments, with BK = 32. Operands are
converted explicitly with `__float_to_tf32`, and the epilogue is staged through
a per-warp smem buffer for exact ragged edges and α/β.

**What changes about honesty.**

- **Baseline:** TF32 cuBLAS (tensor-op math), never FP32 cuBLAS.
- **Peak:** the TF32 peak (A100: 156 TFLOP/s).
- **Tolerance:** it must admit TF32's input rounding. A rigorous bound of
  2⁻⁹·(|A||B|)ᵢⱼ covers even truncating conversion. It's ~70× looser than FP32,
  so dropped-term sensitivity is only claimed for K ≲ 2000.

**SASS says.** 128 `HMMA.1684.F32.TF32` per kernel body, fed by 144 scalar LDS.
TF32 WMMA loads aren't vectorized, since `ldmatrix` is 16-bit only. 150
registers means 1 block (8 warps) per SM.

**Prediction.**

- Well above stage 7 in absolute GFLOP/s, and possibly above *FP32* cuBLAS,
  which is not reported as a comparison.
- Only 30–60% of TF32 cuBLAS: there is no `cp.async` and no multi-stage
  pipeline, so global-load latency sits fully exposed in front of MMAs that are
  8× faster than FFMA.
- Expect **Long Scoreboard** / **Barrier** at the top of the stalls, and a
  tensor-pipe % far below cuBLAS's.

**Evidence.** TBD(measure).

## 4. Where predictions were wrong

Found before any GPU run, from SASS and from the test suite:

- **Loads per FFMA** came out wrong from source-level counting three times,
  because ptxas vectorizes contiguous shared-memory reads:
  - stage 3: 2 predicted → 1.25 actual;
  - stage 4: 1.125 predicted → 0.375 actual;
  - stage 5: I claimed the A reads "can't be vectorized", but ptxas vectorized
    them along k.

  The ratio that bounds a kernel is the one in the SASS.
- **Stage 5 registers:** I estimated 100–128 (2 blocks/SM); ptxas used 133,
  which is 1 block/SM.
- **The FP32 correctness tolerance was too loose.** It started at ε·(4√K + 8),
  which grows with K even though the measured FP32 error doesn't. Writing the
  TF32 stage exposed the problem: at K ≥ 4096, a result computed in TF32
  *passed* the FP32 check. So an accidental tensor-core path (e.g. a
  misconfigured cuBLAS) could have been reported as FP32. The tolerance is now
  ε·(16 + √K/2), and a test asserts that TF32 results fail it.

TBD(measure): add every case where the profiler disagreed with a prediction.

## 5. Threats to validity

- **Clock behaviour.** Numbers are at a locked clock, and the clock log shows
  whether it held. Unlocked, results vary by ~10%.
- **cuBLAS heuristics.** cuBLAS picks different kernels per shape. % of cuBLAS
  is shape-dependent, and at non-tile-multiple sizes padding costs differ
  between us and cuBLAS.
- **L2 flushing** makes small-size numbers more conservative than unflushed
  published series. Compare like with like.
- **Small sizes** (≤ 512³) are latency measurements, not throughput.
- **Time-budgeted iterations.** Very slow cases (early stages at 8192³) run
  fewer timed iterations. The CSV's `iters` column shows how many.
- **TF32 peak per SM is SKU-dependent** on consumer Ampere/Ada (GeForce halves
  it). The source of the peak is recorded in the CSV.
- **One card.** Which bottleneck binds is A100-specific. On GA10x/AD10x the
  FP32:INT ratio, smem/L1 split, and tensor rates all differ.

## 6. What I'd do next

- **`cp.async` double buffering:** overlap the next tile's global loads with
  the current tile's math. That's the obvious fix for the stage 7 and 9 stall
  profiles.
- **Raw `mma.sync` with swizzled smem layouts** in place of WMMA. That's the
  CUTLASS recipe.
- **Split-K** for short-and-wide shapes (e.g. 4096×4096×512).
