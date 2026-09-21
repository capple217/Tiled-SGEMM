#pragma once
// Stage 4 — 1D block/register tiling. Each thread computes TM outputs (a
// vertical strip of C) instead of one, accumulating in registers.
//
// Why: stage 3 spent 1.25 LDS instructions per FFMA (see 03_smem_tiling.cu).
// Here, per k-step, a thread loads ONE value of B into a register and reuses it
// for TM FFMAs, and the TM values of A it needs are contiguous in smem per row:
//
//     b = Bs[k][col]                         1 LDS
//     for i < TM:  acc[i] += As[r+i][k] * b   TM FFMA
//
// Measured in SASS (scripts/sass_stats.sh, sm_80, default config), per BK=8 tile:
//      8 x LDS      (Bs, one per k)
//     16 x LDS.128  (As: 8 rows x 8 k = 64 floats, fetched 4 at a time)
//     64 x FFMA
// => 24 LDS : 64 FFMA = 0.375 per FFMA, down from 1.25. By the stage-3 roofline
// (1 LDS/clk vs 2 FFMA/clk per SM) the LDS pipe is no longer the binding
// ceiling. That's the point: once smem instructions are cheap, what dominates
// is everything around the FFMAs.
//   * Global-load latency is fully exposed: LDG -> STS -> barrier -> compute,
//     with nothing overlapping the loads. Expect "Long Scoreboard" + "Barrier".
//   * Only 8 k-steps of work between barrier pairs (BK = 8).
//   * Each B value is still used by only TM FFMAs; each As value by only 1
//     thread-column. Stage 5's TM x TN outer product raises reuse on both.
//
// Also changes vs stage 3:
//  * Bigger block tile (64x64 default) with fewer threads (512): each block
//    fetches (BM + BN) * BK floats from global per K-chunk for 2*BM*BN*BK FLOPs,
//    i.e. BM*BN/(2*(BM+BN)) FLOP/B = 16 FLOP/B at 64x64, up from 8.
//  * BK = 8 (small) keeps smem at (BM + BN) * BK * 4 B = 4 KB, and means a
//    __syncthreads pair every 8 k-steps. Barriers become a visible stall reason
//    ("Barrier"). Stage 6+ and double-buffering are where that gets addressed.
//
// Things to verify in the profile / SASS (profiling/profile.sh blocktiling1d):
//  * lds wavefronts/inst ~1 (no bank conflicts; As reads are warp-broadcast
//    because the whole warp shares threadRow when BN >= 32).
//  * FFMA fraction of issued instructions rises sharply vs stage 3.
//  * scripts/sass_stats.sh: the LDS/LDS.128/FFMA counts quoted above. Re-check
//    them whenever you change the tile shape, because vectorisation is ptxas's
//    choice and depends on alignment it can prove.
//  * Registers/thread: TM accumulators + addressing; should stay well under 64.
//
// Template parameters exist for stage 8 (autotuning). Constraints are static_asserted.

#include "sgemm/common.hpp"

namespace sgemm {

template <int BM, int BN, int BK, int TM>
__global__ void __launch_bounds__((BM * BN) / TM)
    sgemm_04_blocktiling_1d(int M, int N, int K, float alpha, const float* __restrict__ A,
                            const float* __restrict__ B, float beta, float* __restrict__ C) {
  constexpr int kThreads = (BM * BN) / TM;
  static_assert(BM % TM == 0, "TM must divide BM");
  static_assert(BN % 32 == 0, "BN >= 32 multiple keeps a warp inside one threadRow (As broadcast)");
  static_assert(kThreads % 32 == 0 && kThreads <= 1024, "bad thread count");
  static_assert((BM * BK) % kThreads == 0 && (BK * BN) % kThreads == 0,
                "each thread must load a whole number of elements per tile");

  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  const int tid = threadIdx.x;
  // Output strip owned by this thread: rows [threadRow*TM, +TM), column threadCol.
  // threadCol varies fastest across tid, so a warp covers 32 consecutive columns:
  // coalesced C stores and conflict-free Bs reads.
  const int threadCol = tid % BN;
  const int threadRow = tid / BN;

  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;

  float acc[TM] = {0.0f};  // fully unrolled index -> lives in registers, not local memory

  for (int k0 = 0; k0 < K; k0 += BK) {
    // ---- gmem -> smem ------------------------------------------------------
    // A tile BM x BK. Linear index i -> (i / BK, i % BK): consecutive threads
    // walk k first, so a warp reads 32/BK rows x BK contiguous floats. With
    // BK = 8 that's 4 fully-used 32B sectors. Sector efficiency is 100%, even
    // though it isn't one 128B line.
#pragma unroll
    for (int j = 0; j < (BM * BK) / kThreads; ++j) {  // compile-time trip count
      const int i = tid + j * kThreads;
      const int r = i / BK, c = i % BK;
      const int gr = blockRow + r, gc = k0 + c;
      As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
    // B tile BK x BN: consecutive threads walk columns, so fully coalesced.
#pragma unroll
    for (int j = 0; j < (BK * BN) / kThreads; ++j) {
      const int i = tid + j * kThreads;
      const int r = i / BN, c = i % BN;
      const int gr = k0 + r, gc = blockCol + c;
      Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
    }
    __syncthreads();

    // ---- compute: the register-reuse loop -----------------------------------
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      const float b = Bs[k][threadCol];  // one LDS, reused TM times from a register
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        acc[i] += As[threadRow * TM + i][k] * b;
      }
    }
    __syncthreads();
  }

  // ---- epilogue ------------------------------------------------------------
  const int col = blockCol + threadCol;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = blockRow + threadRow * TM + i;
    if (row < M && col < N) {
      const int idx = row * N + col;
      C[idx] = (beta == 0.0f) ? alpha * acc[i] : alpha * acc[i] + beta * C[idx];
    }
  }
}

template <int BM, int BN, int BK, int TM>
void launch_04_blocktiling_1d_cfg(int M, int N, int K, float alpha, const float* A,
                                  const float* B, float beta, float* C, cudaStream_t stream) {
  const dim3 block((BM * BN) / TM);
  const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  sgemm_04_blocktiling_1d<BM, BN, BK, TM><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace sgemm
