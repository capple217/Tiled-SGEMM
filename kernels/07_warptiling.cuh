#pragma once
// Stage 7 — warptiling: an explicit WARP-level tile between the block tile and
// the thread tile.
//
// Hierarchy (default <BM=128, BN=128, BK=8, WM=64, WN=32>):
//   block  128 x 128  -> 8 warps as a 2 x 4 grid of warp tiles
//   warp    64 x 32   -> iterated as WMITER x WNITER = 2 x 2 sub-tiles of 32 x 16
//   lane     4 x 4    -> per sub-tile; the 32 lanes are laid out 8 (M) x 4 (N)
//   => each thread still owns 64 outputs (2*2*4*4), same register budget as
//      stages 5-6. The only change is WHICH 64 outputs a thread owns.
//
// Why that matters: shared-memory traffic per warp. Per k-step, what a warp
// must read from smem is (rows of A it covers) + (columns of B it covers):
//
//   stage 6  a warp = 2 threadRows x 16 threadCols spread across the WHOLE block
//            width: A 2 LDS.128 x 2 distinct 16B chunks, B 2 LDS.128 x 256 B.
//            Unique bytes 64 + 512 = 576, wavefronts 1+1+2+2 = 6.
//   stage 7  a warp covers a compact 64 x 32 region:
//            A: 2 LDS.128, 8 distinct lanes-rows x 16 B = 128 B contiguous -> 1 wavefront each
//            B: 2 LDS.128, 4 distinct lane-cols x 16 B =  64 B contiguous -> 1 wavefront each
//            Unique bytes 256 + 128 = 384, wavefronts 4.
//   -> 1/3 less shared-memory bandwidth for the same 64 FFMA-instructions per warp.
//
// SASS (sm_80, vector instance): 32 LDS.128, 576 FFMA, 16 LDG.E.128, 32 STG.E.128
// per kernel body, IDENTICAL to stage 6. The instruction mix doesn't change, only
// how many wavefronts each LDS.128 costs, so the evidence for this stage has to
// come from ncu (lds wavefronts/inst, L1/shared throughput %), not SASS counts.
// Registers: 113 (sm_80), 0 spills -> 2 blocks/SM.
//
// Honest expectation: at stage 6, smem is NOT the binding limit (6 wavefronts
// vs 32 FFMA-issue clocks per warp per k-step on A100's 1 wavefront/clk vs
// 2 warp-FFMA/clk). So expect a modest gain (0-15%), mostly visible as lower
// "L1/Shared" throughput % and fewer MIO stalls rather than a big GFLOPS jump.
// The larger value of this stage is structural. It is exactly the decomposition
// tensor cores need (stage 9 swaps the 4x4 per-lane FFMA tile for a warp-wide
// 16x16x8 MMA), and published series gain most from it after autotuning, where
// the warp shape becomes a tunable dimension (WM, WN in bench/autotune.cu).
//
// Unchanged from stage 6: float4 global loads, transposed + padded AsT (the
// padding argument assumes BK = 8; see note in the load loop), scalar fallback
// for K % 4 != 0 or N % 4 != 0.

#include "sgemm/common.hpp"

namespace sgemm {

template <int BM, int BN, int BK, int WM, int WN, bool kVec>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32)
    sgemm_07_warptiling(int M, int N, int K, float alpha, const float* __restrict__ A,
                        const float* __restrict__ B, float beta, float* __restrict__ C) {
  // Fixed intra-warp layout: lanes 8 (M) x 4 (N), each owning 4 x 4 per sub-tile.
  constexpr int kLanesM = 8, kLanesN = 4, TM = 4, TN = 4;
  constexpr int WSUBM = kLanesM * TM;  // 32
  constexpr int WSUBN = kLanesN * TN;  // 16
  constexpr int WMITER = WM / WSUBM, WNITER = WN / WSUBN;
  constexpr int kWarpsN = BN / WN;
  constexpr int kThreads = (BM / WM) * (BN / WN) * 32;
  constexpr int kPadA = 4;
  static_assert(BM % WM == 0 && BN % WN == 0, "warp tile must divide block tile");
  static_assert(WM % WSUBM == 0 && WN % WSUBN == 0, "warp tile must be a multiple of 32 x 16");
  static_assert(BK % 4 == 0, "float4 loads along k");
  static_assert((BM * BK / 4) % kThreads == 0 && (BK * BN / 4) % kThreads == 0,
                "whole number of float4 loads per thread");

  alignas(16) __shared__ float AsT[BK][BM + kPadA];
  alignas(16) __shared__ float Bs[BK][BN];

  const int tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;
  const int warpRow = warp / kWarpsN, warpCol = warp % kWarpsN;
  const int laneRow = lane / kLanesN, laneCol = lane % kLanesN;
  const int blockRow = blockIdx.y * BM, blockCol = blockIdx.x * BN;
  // Offsets of this thread's first element inside the block tile.
  const int mBase = warpRow * WM + laneRow * TM;
  const int nBase = warpCol * WN + laneCol * TN;

  float acc[WMITER * TM][WNITER * TN] = {};
  float regM[WMITER * TM], regN[WNITER * TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    // ---- A -> AsT (transposed). With BK = 8 a warp covers 16 rows x 2 k-groups,
    // and the 4-float pad puts the two k-groups on opposite bank halves.
    // (At BK = 16 a warp spans 4 k-groups and a 2-way STS conflict returns; the
    // autotuner measures whether larger BK is worth it anyway.)
#pragma unroll
    for (int j = 0; j < (BM * BK / 4) / kThreads; ++j) {
      const int i = tid + j * kThreads;
      const int r = i / (BK / 4), c = (i % (BK / 4)) * 4;
      const int gr = blockRow + r, gc = k0 + c;
      float4 v;
      if constexpr (kVec) {
        v = (gr < M && gc < K) ? *reinterpret_cast<const float4*>(&A[gr * K + gc])
                               : make_float4(0.f, 0.f, 0.f, 0.f);
      } else {
        const bool ok = gr < M;
        v.x = (ok && gc + 0 < K) ? A[gr * K + gc + 0] : 0.f;
        v.y = (ok && gc + 1 < K) ? A[gr * K + gc + 1] : 0.f;
        v.z = (ok && gc + 2 < K) ? A[gr * K + gc + 2] : 0.f;
        v.w = (ok && gc + 3 < K) ? A[gr * K + gc + 3] : 0.f;
      }
      AsT[c + 0][r] = v.x;
      AsT[c + 1][r] = v.y;
      AsT[c + 2][r] = v.z;
      AsT[c + 3][r] = v.w;
    }
    // ---- B -> Bs (float4 in, STS.128 out) ----
#pragma unroll
    for (int j = 0; j < (BK * BN / 4) / kThreads; ++j) {
      const int i = tid + j * kThreads;
      const int r = i / (BN / 4), c = (i % (BN / 4)) * 4;
      const int gr = k0 + r, gc = blockCol + c;
      float4 v;
      if constexpr (kVec) {
        v = (gr < K && gc < N) ? *reinterpret_cast<const float4*>(&B[gr * N + gc])
                               : make_float4(0.f, 0.f, 0.f, 0.f);
      } else {
        const bool ok = gr < K;
        v.x = (ok && gc + 0 < N) ? B[gr * N + gc + 0] : 0.f;
        v.y = (ok && gc + 1 < N) ? B[gr * N + gc + 1] : 0.f;
        v.z = (ok && gc + 2 < N) ? B[gr * N + gc + 2] : 0.f;
        v.w = (ok && gc + 3 < N) ? B[gr * N + gc + 3] : 0.f;
      }
      *reinterpret_cast<float4*>(&Bs[r][c]) = v;
    }
    __syncthreads();

    // ---- compute: (WMITER + WNITER) LDS.128 + WMITER*WNITER*16 FFMA per k ----
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int wm = 0; wm < WMITER; ++wm) {
        const float4 a = *reinterpret_cast<const float4*>(&AsT[k][mBase + wm * WSUBM]);
        regM[wm * TM + 0] = a.x; regM[wm * TM + 1] = a.y;
        regM[wm * TM + 2] = a.z; regM[wm * TM + 3] = a.w;
      }
#pragma unroll
      for (int wn = 0; wn < WNITER; ++wn) {
        const float4 b = *reinterpret_cast<const float4*>(&Bs[k][nBase + wn * WSUBN]);
        regN[wn * TN + 0] = b.x; regN[wn * TN + 1] = b.y;
        regN[wn * TN + 2] = b.z; regN[wn * TN + 3] = b.w;
      }
#pragma unroll
      for (int i = 0; i < WMITER * TM; ++i)
#pragma unroll
        for (int jn = 0; jn < WNITER * TN; ++jn) acc[i][jn] += regM[i] * regN[jn];
    }
    __syncthreads();
  }

  // ---- epilogue: acc[wm*4+i][wn*4+j] -> (mBase + wm*32 + i, nBase + wn*16 + j) ----
#pragma unroll
  for (int wm = 0; wm < WMITER; ++wm) {
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int row = blockRow + mBase + wm * WSUBM + i;
      if (row >= M) continue;
#pragma unroll
      for (int wn = 0; wn < WNITER; ++wn) {
        const int col = blockCol + nBase + wn * WSUBN;
        float* c = &C[row * N + col];
        const float* a = &acc[wm * TM + i][wn * TN];
        if constexpr (kVec) {
          if (col < N) {  // N % 4 == 0 -> whole float4 in or out
            float4 out = make_float4(alpha * a[0], alpha * a[1], alpha * a[2], alpha * a[3]);
            if (beta != 0.0f) {
              const float4 old = *reinterpret_cast<const float4*>(c);
              out.x += beta * old.x; out.y += beta * old.y;
              out.z += beta * old.z; out.w += beta * old.w;
            }
            *reinterpret_cast<float4*>(c) = out;
          }
        } else {
#pragma unroll
          for (int q = 0; q < TN; ++q) {
            if (col + q < N) {
              const float v = alpha * a[q];
              c[q] = (beta == 0.0f) ? v : v + beta * c[q];
            }
          }
        }
      }
    }
  }
}

template <int BM, int BN, int BK, int WM, int WN>
void launch_07_warptiling_cfg(int M, int N, int K, float alpha, const float* A, const float* B,
                              float beta, float* C, cudaStream_t stream) {
  const dim3 block((BM / WM) * (BN / WN) * 32);
  const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  const auto aligned16 = [](const void* p) { return (reinterpret_cast<unsigned long long>(p) & 15) == 0; };
  const bool vec = (K % 4 == 0) && (N % 4 == 0) && aligned16(A) && aligned16(B) && aligned16(C);
  if (vec)
    sgemm_07_warptiling<BM, BN, BK, WM, WN, true><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
  else
    sgemm_07_warptiling<BM, BN, BK, WM, WN, false><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace sgemm
