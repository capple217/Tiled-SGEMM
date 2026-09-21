// Stage 5 — 2D block/register tiling.            // TODO(Fasih): yours to write.
//
// Not compiled yet. When you're ready:
//   1. CMakeLists.txt: uncomment  kernels/05_blocktiling_2d.cu
//   2. include/sgemm/kernels.hpp: uncomment the launch_05_blocktiling_2d declaration
//   3. src/registry.cu: uncomment the "blocktiling2d" entry
//   4. tests/emu/run_emu_tests.sh blocktiling2d      (no GPU needed, ASan on)
//   5. on the GPU: sgemm_test --kernel blocktiling2d, then scripts/sanitize.sh blocktiling2d
//
// ---------------------------------------------------------------------------
// The goal
// ---------------------------------------------------------------------------
// Each thread owns a TM x TN tile of C in registers. Per k-step it loads a
// TM-strip of A and a TN-strip of B from shared memory into registers and does
// their OUTER PRODUCT: TM*TN FFMAs from TM + TN loaded values. Stage 4 got
// TM FFMAs per (1 + TM) values. Work out, before writing code, what the
// LDS-per-FFMA and smem-bytes-per-FLOP ratios become for TM = TN = 8, and what
// the stage-3 roofline (1 LDS/clk vs 2 FFMA/clk per SM) says about the ceiling.
//
// Suggested starting point (not a requirement): BM = BN = 128, BK = 8,
// TM = TN = 8 -> 256 threads/block, 8 KB smem/block, 64 accumulators/thread.
// Template it on <BM, BN, BK, TM, TN> like stage 4 so stage 8 can sweep it.
//
// ---------------------------------------------------------------------------
// The bars (what "done" means for this stage)
// ---------------------------------------------------------------------------
//  [correctness] tests/emu passes (incl. ASan), sgemm_test passes, and
//                scripts/sanitize.sh is clean. Ragged shapes included.
//  [resources]   ptxas -v (cmake -DSGEMM_PTXAS_VERBOSE=ON): 0 bytes spill
//                stores/loads. Registers <= 128/thread, and say in WRITEUP
//                what that means for blocks/SM (65536 regs / (256 threads x regs)).
//  [SASS]        scripts/sass_stats.sh: count LDS / LDS.64 / LDS.128 vs FFMA
//                in your kernel and compare with the ratio you predicted.
//  [profile]     profiling/profile.sh blocktiling2d 4096: identify the NEW
//                top stall reason. Quote gld sectors/request, lds wavefronts/inst
//                and FMA-pipe % next to stage 4's.
//  [speed]       Rough expectation on A100 at 4096^3: ~1.5-2.5x stage 4. Published
//                reference series on similar Ampere parts put this stage near
//                60-70% of cuBLAS. If you're far below that, the profile will say
//                why (my guess at the usual suspects is below).
//  [writeup]     WRITEUP.md stage 5 section: what changed, why it helps, which
//                profiler numbers moved and by how much.
//
// ---------------------------------------------------------------------------
// Pitfalls worth knowing about in advance (no solutions, just the traps)
// ---------------------------------------------------------------------------
//  * Register arrays indexed with anything the compiler can't resolve at compile
//    time get demoted to LOCAL memory (actually DRAM-backed). ptxas -v shows it
//    as "stack frame"/spills. Every loop over TM/TN needs a compile-time trip count
//    and #pragma unroll.
//  * 256 threads now load a 128x8 A tile and an 8x128 B tile: more elements
//    than threads. Your load loop's index->(row, col) mapping decides
//    coalescing on the global side AND bank conflicts on the STS side. Check both.
//  * How you map tid -> (threadRow, threadCol) decides whether the warp's
//    As/Bs reads in the inner loop are broadcasts, conflict-free, or n-way
//    conflicted. Reason it out per warp (32 consecutive tids), then confirm with
//    "lds wavefronts/inst" in the profile.
//  * Epilogue: TM*TN guarded stores. Keep the beta == 0 no-read rule.
//  * Both __syncthreads are still required, and no thread may return early.
//
// ---------------------------------------------------------------------------

#include "sgemm/common.hpp"
#include "sgemm/kernels.hpp"

namespace sgemm {

// template <int BM, int BN, int BK, int TM, int TN>
// __global__ void __launch_bounds__((BM * BN) / (TM * TN))
//     sgemm_05_blocktiling_2d(int M, int N, int K, float alpha, const float* __restrict__ A,
//                             const float* __restrict__ B, float beta, float* __restrict__ C) {
//   // TODO(Fasih)
// }

// void launch_05_blocktiling_2d(int M, int N, int K, float alpha, const float* A,
//                               const float* B, float beta, float* C, cudaStream_t stream) {
//   // TODO(Fasih)
// }

}  // namespace sgemm
