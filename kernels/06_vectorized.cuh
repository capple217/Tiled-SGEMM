#pragma once
// Stage 6 — vectorized (128-bit) global access + a shared-memory layout whose
// inner-loop reads are conflict-free LDS.128s.
//
// Same block tile (128x128x8) and thread tile (8x8) as stage 5. Four changes,
// each aimed at a specific inefficiency stage 5's profile should show:
//
// (1) float4 global loads (LDG.E.128). Stage 5 issued 4 scalar LDGs per operand
//     per thread per tile; now it's 1 LDG.128 each. The same bytes move with 4x
//     fewer memory instructions: less LSU/MIO issue pressure, fewer
//     Long-Scoreboard dependencies to track.
//
// (2) A is stored TRANSPOSED in smem: AsT[k][m], so a thread's 8 A values for
//     one k are contiguous in m. NOTE what this does NOT buy: scripts/sass_stats.sh
//     shows stage 5 already emits 32 LDS.128 per tile, same as this stage.
//     ptxas vectorised stage 5's A reads along k (As[row][k..k+3]) instead.
//     What the transpose buys is the bank pattern: stage 5's LDS.128 of As
//     reads two threadRows' rows 64 words apart, which is the same 4 banks, a
//     2-way conflict. Here the two addresses are adjacent 16B chunks: 1 wavefront.
//     The cost moves to the store side: a float4 of A (4 consecutive k of one
//     row) is scattered into 4 different AsT rows as scalar STS, once per tile.
//
// (3) AsT rows padded by 4 floats (kPadA). Why: in the transposing store a warp
//     writes, for a fixed k, 16 consecutive m values from each of two k-groups
//     (c4 = 0, 1) that are 4 rows apart. Unpadded, 4 rows x 128 words = 512 words
//     = a multiple of 32, so both groups hit the SAME 16 banks: a 2-way conflict.
//     With a row stride of 132, the second group is shifted by 4*132 mod 32 = 16
//     banks, landing on the other 16: conflict-free. 132*4 B = 528 B keeps
//     every row 16B-aligned for the LDS.128 reads.
//
// (4) Split thread tile. Instead of 8 contiguous columns [8*tc, 8*tc+8), thread
//     tc owns two 4-wide groups: [4*tc, 4*tc+4) and [BN/2 + 4*tc, ...+4) (same
//     for rows). Why: with contiguous 8-wide strips, the 16 threadCols of a
//     warp read 16B chunks at a 32B stride, which touches only half the banks,
//     4 times each: a 4-way conflict per LDS.128 (256 B served in 4 wavefronts
//     instead of 2). Split, each LDS.128 across the warp reads one contiguous 256 B
//     span: 2 wavefronts, the minimum. The same split makes each float4 C store
//     of the warp a contiguous 256 B, so coalesced.
//
// SASS vs stage 5 (sm_80, vector instance), per kernel body:
//     LDG.E 64 -> LDG.E.128 16      STG.E 128 -> STG.E.128 32
//     LDS.128 32 -> 32 (unchanged)  FFMA 576 -> 576
// So the smem INSTRUCTION count is unchanged. The expected wins are (a) 4x
// fewer global memory instructions and (b) fewer smem WAVEFRONTS per LDS.128
// (the conflicts removed by points 2 and 4). Expected profile deltas: lds
// wavefronts/inst drops toward the ideal, sts bank conflicts ~0, and the stall
// mix shifts further toward "Long Scoreboard" / "Barrier", i.e. load latency
// that only double buffering removes. Registers: 112 vs stage 5's 133, which
// also moves occupancy from 1 to 2 blocks/SM (see stage 5 header).
//
// Ragged sizes: the vector path needs K % 4 == 0 and N % 4 == 0 (so every
// float4 of A/B/C is either fully in-bounds or fully out) and 16B-aligned
// pointers. Otherwise the launcher uses the kVec = false instance, which has the
// same smem layout and compute loop but scalar global loads/stores.

#include "sgemm/common.hpp"

namespace sgemm {

template <int BM, int BN, int BK, bool kVec>
__global__ void __launch_bounds__((BM * BN) / 64)
    sgemm_06_vectorized(int M, int N, int K, float alpha, const float* __restrict__ A,
                        const float* __restrict__ B, float beta, float* __restrict__ C) {
  constexpr int TM = 8, TN = 8;           // split as 2 x 4 in each dimension
  constexpr int kThreads = (BM * BN) / (TM * TN);
  constexpr int kThreadsN = BN / TN;
  constexpr int kPadA = 4;
  static_assert(BK % 4 == 0 && BM % 8 == 0 && BN % 8 == 0, "float4 tiling");
  static_assert(kThreadsN % 16 == 0,
                "split-tile conflict-freedom assumes a warp row spans >= 16 threadCols");
  static_assert((BM * BK / 4) % kThreads == 0 && (BK * BN / 4) % kThreads == 0,
                "whole number of float4 loads per thread");

  alignas(16) __shared__ float AsT[BK][BM + kPadA];
  alignas(16) __shared__ float Bs[BK][BN];

  const int tid = threadIdx.x;
  const int threadRow = tid / kThreadsN;
  const int threadCol = tid % kThreadsN;
  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;

  float acc[TM][TN] = {};
  float regM[TM], regN[TN];

  for (int k0 = 0; k0 < K; k0 += BK) {
    // ---- A: global (row-major, k contiguous) -> AsT (k-major) -------------------
#pragma unroll
    for (int j = 0; j < (BM * BK / 4) / kThreads; ++j) {
      const int i = tid + j * kThreads;
      const int r = i / (BK / 4), c = (i % (BK / 4)) * 4;
      const int gr = blockRow + r, gc = k0 + c;
      float4 v;
      if constexpr (kVec) {
        // K % 4 == 0 and gc % 4 == 0: the float4 is entirely in or entirely out.
        v = (gr < M && gc < K) ? *reinterpret_cast<const float4*>(&A[gr * K + gc])
                               : make_float4(0.f, 0.f, 0.f, 0.f);
      } else {
        const bool ok = gr < M;
        v.x = (ok && gc + 0 < K) ? A[gr * K + gc + 0] : 0.f;
        v.y = (ok && gc + 1 < K) ? A[gr * K + gc + 1] : 0.f;
        v.z = (ok && gc + 2 < K) ? A[gr * K + gc + 2] : 0.f;
        v.w = (ok && gc + 3 < K) ? A[gr * K + gc + 3] : 0.f;
      }
      AsT[c + 0][r] = v.x;  // transposing scatter: 4 scalar STS, conflict-free
      AsT[c + 1][r] = v.y;  // thanks to kPadA (see header, point 3)
      AsT[c + 2][r] = v.z;
      AsT[c + 3][r] = v.w;
    }
    // ---- B: global -> Bs, float4 in and float4 out (STS.128) --------------------
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

    // ---- compute: 4 x LDS.128 + 64 FFMA per k-step -----------------------------
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      const float4 m0 = *reinterpret_cast<const float4*>(&AsT[k][threadRow * 4]);
      const float4 m1 = *reinterpret_cast<const float4*>(&AsT[k][BM / 2 + threadRow * 4]);
      const float4 n0 = *reinterpret_cast<const float4*>(&Bs[k][threadCol * 4]);
      const float4 n1 = *reinterpret_cast<const float4*>(&Bs[k][BN / 2 + threadCol * 4]);
      regM[0] = m0.x; regM[1] = m0.y; regM[2] = m0.z; regM[3] = m0.w;
      regM[4] = m1.x; regM[5] = m1.y; regM[6] = m1.z; regM[7] = m1.w;
      regN[0] = n0.x; regN[1] = n0.y; regN[2] = n0.z; regN[3] = n0.w;
      regN[4] = n1.x; regN[5] = n1.y; regN[6] = n1.z; regN[7] = n1.w;
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int jn = 0; jn < TN; ++jn) acc[i][jn] += regM[i] * regN[jn];
    }
    __syncthreads();
  }

  // ---- epilogue: acc[i][j] lives at row blockRow + (i/4)*(BM/2) + 4*threadRow + i%4,
  //      col blockCol + (j/4)*(BN/2) + 4*threadCol + j%4 ------------------------------
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = blockRow + (i / 4) * (BM / 2) + threadRow * 4 + (i % 4);
    if (row >= M) continue;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int col = blockCol + h * (BN / 2) + threadCol * 4;
      float* c = &C[row * N + col];
      if constexpr (kVec) {
        if (col < N) {  // N % 4 == 0: the float4 is entirely in or out
          float4 out = make_float4(alpha * acc[i][4 * h + 0], alpha * acc[i][4 * h + 1],
                                   alpha * acc[i][4 * h + 2], alpha * acc[i][4 * h + 3]);
          if (beta != 0.0f) {
            const float4 old = *reinterpret_cast<const float4*>(c);
            out.x += beta * old.x; out.y += beta * old.y;
            out.z += beta * old.z; out.w += beta * old.w;
          }
          *reinterpret_cast<float4*>(c) = out;  // STG.E.128, warp-contiguous 256 B
        }
      } else {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          if (col + q < N) {
            const float v = alpha * acc[i][4 * h + q];
            c[q] = (beta == 0.0f) ? v : v + beta * c[q];
          }
        }
      }
    }
  }
}

inline bool aligned16(const void* p) { return (reinterpret_cast<unsigned long long>(p) & 15) == 0; }

template <int BM, int BN, int BK>
void launch_06_vectorized_cfg(int M, int N, int K, float alpha, const float* A, const float* B,
                              float beta, float* C, cudaStream_t stream) {
  const dim3 block((BM * BN) / 64);
  const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  const bool vec = (K % 4 == 0) && (N % 4 == 0) && aligned16(A) && aligned16(B) && aligned16(C);
  if (vec)
    sgemm_06_vectorized<BM, BN, BK, true><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
  else
    sgemm_06_vectorized<BM, BN, BK, false><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace sgemm
