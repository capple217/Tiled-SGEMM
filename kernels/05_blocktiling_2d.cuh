#pragma once
// Stage 5 — 2D block/register tiling. Each thread owns a TM x TN tile of C.
// Per k-step it loads TM values of A and TN values of B from smem into
// registers and does their outer product: TM*TN FFMAs from TM + TN loads.
//
// Ratios at TM = TN = 8 (source level, before ptxas vectorisation):
//   stage 4: 1 + TM = 9 values per TM = 8 FFMAs     -> 1.125 loads/FFMA
//   stage 5: TM + TN = 16 values per 64 FFMAs        -> 0.25  loads/FFMA
// and with LDS.128 (contiguous regN reads) the instruction count is lower
// still. See scripts/sass_stats.sh for the real numbers.
//
// Block tile 128x128, BK = 8, 256 threads, 8 KB smem.
//   global intensity per block: BM*BN/(2*(BM+BN)) = 32 FLOP/B (stage 4: 16).
//   registers: ptxas -v says 133 on sm_80 (64 accumulators + 16 fragments +
//   addressing). That is just over 128, so 65536 / (256 * 136 allocated) = 1.9,
//   i.e. ONE block/SM = 8 warps = 12.5% occupancy on A100. At <= 128 it would be
//   2 blocks. Experiment: __launch_bounds__(256, 2) forces ptxas under 128;
//   measure whether the extra warps beat whatever spills/rematerialisation that
//   costs. Low occupancy is not automatically bad here: latency is hidden by
//   ILP (64 independent FFMA chains per thread) rather than by warp count.
//   This is the occupancy-vs-ILP trade the brief asks about, made concrete.
//
// SASS (sm_80): 32 LDS.128 per tile. ptxas vectorised BOTH operands: As along k
// (As[row][k..k+3]) and Bs along n. So LDS instructions per FFMA = 32/512.
//
// Known remaining inefficiencies (these are what stage 6 fixes; check them in ncu):
//  * As LDS.128: the warp's two threadRows read rows 64 words apart, which is
//    the same 4 banks: a 2-way conflict.
//  * Bs LDS.128 at Bs[k][threadCol*8 (+4)]: 16 distinct 16B chunks at a 32B
//    stride touch only half the banks, so 4 wavefronts instead of the 2 that
//    256 B needs. Expect lds wavefronts/inst well above ideal.
//  * Global loads are scalar (LDG.E): 4 loads per thread per tile for each operand.

#include "sgemm/common.hpp"

namespace sgemm {

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    sgemm_05_blocktiling_2d(int M, int N, int K, float alpha, const float* __restrict__ A,
                            const float* __restrict__ B, float beta, float* __restrict__ C) {
  constexpr int kThreads = (BM * BN) / (TM * TN);
  constexpr int kThreadsN = BN / TN;  // threads across a row of the block tile
  static_assert(BM % TM == 0 && BN % TN == 0, "thread tile must divide block tile");
  static_assert(kThreads % 32 == 0 && kThreads <= 1024, "bad thread count");
  static_assert((BM * BK) % kThreads == 0 && (BK * BN) % kThreads == 0,
                "each thread must load a whole number of elements per tile");

  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  const int tid = threadIdx.x;
  const int threadRow = tid / kThreadsN;  // which TM-row strip
  const int threadCol = tid % kThreadsN;  // which TN-col strip (fastest across a warp)
  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;

  float acc[TM][TN] = {};  // 64 registers at 8x8, only compile-time indices below
  float regM[TM], regN[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    // gmem -> smem, 4 elements per thread per operand at the default config.
    // A: consecutive tids walk k (BK = 8 wide), so a warp reads 4 rows x 32 B:
    //    4 fully-used sectors per request. B: consecutive tids walk columns,
    //    so fully coalesced 128 B per request.
#pragma unroll
    for (int j = 0; j < (BM * BK) / kThreads; ++j) {
      const int i = tid + j * kThreads;
      const int r = i / BK, c = i % BK;
      const int gr = blockRow + r, gc = k0 + c;
      As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < (BK * BN) / kThreads; ++j) {
      const int i = tid + j * kThreads;
      const int r = i / BN, c = i % BN;
      const int gr = k0 + r, gc = blockCol + c;
      Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      // Load the fragments once per k...
#pragma unroll
      for (int i = 0; i < TM; ++i) regM[i] = As[threadRow * TM + i][k];
#pragma unroll
      for (int j = 0; j < TN; ++j) regN[j] = Bs[k][threadCol * TN + j];
      // ...and reuse each one TN (resp. TM) times from registers.
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = blockRow + threadRow * TM + i;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = blockCol + threadCol * TN + j;
      if (row < M && col < N) {
        const int idx = row * N + col;
        C[idx] = (beta == 0.0f) ? alpha * acc[i][j] : alpha * acc[i][j] + beta * C[idx];
      }
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
void launch_05_blocktiling_2d_cfg(int M, int N, int K, float alpha, const float* A,
                                  const float* B, float beta, float* C, cudaStream_t stream) {
  const dim3 block((BM * BN) / (TM * TN));
  const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  sgemm_05_blocktiling_2d<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace sgemm
