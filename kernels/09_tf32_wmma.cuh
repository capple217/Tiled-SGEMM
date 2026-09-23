#pragma once
// Stage 9 — TF32 tensor cores via WMMA (sm_80+). A DIFFERENT ARC from stages 1-7:
// it changes the arithmetic, so it has its own honest baseline (cuBLAS with
// CUBLAS_TF32_TENSOR_OP_MATH), its own peak (A100: 156 TFLOPS dense TF32 vs
// 19.5 FP32), and its own, looser tolerance (host_utils.hpp: inputs lose 13
// mantissa bits). bench/bench.cu enforces all three via KernelSpec::precision.
//
// What changes relative to stage 7: the per-lane 4x4 FFMA outer product is
// replaced by a warp-wide MMA instruction. One wmma::mma_sync on a 16x16x8
// TF32 fragment is 2048 FMAs issued as a handful of HMMA instructions, and
// the tensor core retires ~8x the FP32 FFMA rate per SM on A100
// (1024 vs 128 FLOP/clk/SM). Stage 7's warp tile was the setup for this: the
// decomposition is identical (block -> warp tile -> fragments), only the
// innermost unit changed.
//
// Structure (default <BM=128, BN=128, BK=32, WM=64, WN=32>):
//   8 warps; each warp owns a 64x32 region = 4 x 2 accumulator fragments (16x16).
//   Per BK=32 tile: 4 k-steps of 8; per step each warp loads 4 A fragments
//   and 2 B fragments from smem, converts them to TF32, and does 8 mma_syncs.
//
// Design notes:
//  * Explicit __float_to_tf32 on every fragment element. Feeding raw FP32 bits
//    to a TF32 MMA isn't guaranteed to round (it may truncate); the conversion
//    makes the rounding round-to-nearest and the error bound tighter. Cost: 12
//    fragments x 4 elements of conversions per k-step per lane, visible as
//    extra ALU instructions in the profile.
//  * smem rows are padded by 4 floats (ldm = BK+4 for A, BN+4 for B). WMMA
//    loads read 16-row fragments, and without padding consecutive rows alias the
//    same banks. Padding by 4 keeps every fragment pointer 32B-aligned, which
//    load_matrix_sync requires. This is a starting point, not a derived optimum:
//    check "lds wavefronts/inst" in ncu.
//  * Epilogue through smem: fragment element->(row, col) ownership is opaque, so
//    each warp stores a fragment into its own 16x16 staging buffer, then its 32
//    lanes apply alpha/beta with per-element bounds checks. That handles ragged
//    M/N exactly and keeps the beta == 0 no-read rule. It costs 8 extra smem
//    round trips per warp, which is negligible next to the K loop.
//
// SASS (sm_80, vector instance), per kernel body: 128 HMMA.1684.F32.TF32
// (= 4 k-steps x 8 mma_sync x 4 m16n8k4 HMMAs), 144 scalar LDS feeding the
// fragments (WMMA's TF32 loads are not vectorised; ldmatrix only exists for
// 16-bit types), 8 LDG.E.128 per thread per tile. Registers: 150, 0 spills ->
// 65536 / (256 x 152) = ONE block (8 warps) per SM. So latency hiding rests
// almost entirely on ILP within those 8 warps, one more reason the exposed
// global loads will hurt. The autotuner's tf32_wmma family includes smaller
// warp tiles (WM/WN = 32) that trade MMA reuse for occupancy.
//
// Honest expectation: this is the simplest correct tensor-core GEMM. It has no
// cp.async and no multi-stage pipeline, so every k-tile's global loads sit
// fully exposed in front of the MMAs, and tensor cores are fast enough that
// this matters much more than it did for FFMA. Expect a large speedup over
// stage 7 in absolute GFLOPS (it may beat FP32 cuBLAS, which is NOT a fair
// comparison and is never reported as one) but a modest fraction of TF32 cuBLAS,
// plausibly 30-60%. The natural next steps are cp.async double-buffering and
// raw mma.sync with ldmatrix-style layouts, the actual CUTLASS recipe.

#include <mma.h>

#include "sgemm/common.hpp"

namespace sgemm {

template <int BM, int BN, int BK, int WM, int WN, bool kVec>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32)
    sgemm_09_tf32_wmma(int M, int N, int K, float alpha, const float* __restrict__ A,
                       const float* __restrict__ B, float beta, float* __restrict__ C) {
  using namespace nvcuda;
  constexpr int FM = 16, FN = 16, FK = 8;  // the only TF32 WMMA shape
  constexpr int WFM = WM / FM, WFN = WN / FN;
  constexpr int kWarpsN = BN / WN;
  constexpr int kWarps = (BM / WM) * (BN / WN);
  constexpr int kThreads = kWarps * 32;
  constexpr int kLdA = BK + 4, kLdB = BN + 4;  // padded leading dims (floats)
  static_assert(BM % WM == 0 && BN % WN == 0 && WM % FM == 0 && WN % FN == 0, "tiling");
  static_assert(BK % FK == 0 && BK % 4 == 0, "k tiling");
  static_assert((BM * BK / 4) % kThreads == 0 && (BK * BN / 4) % kThreads == 0,
                "whole number of float4 loads per thread");
  static_assert((BM * kLdA + BK * kLdB + kWarps * FM * FN) * 4 <= 48 * 1024, "static smem limit");

  alignas(128) __shared__ float As[BM][kLdA];   // row-major, k contiguous
  alignas(128) __shared__ float Bs[BK][kLdB];   // row-major, n contiguous
  alignas(128) __shared__ float Cs[kWarps][FM * FN];  // per-warp epilogue staging

  const int tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;
  const int warpRow = warp / kWarpsN, warpCol = warp % kWarpsN;
  const int blockRow = blockIdx.y * BM, blockCol = blockIdx.x * BN;

  wmma::fragment<wmma::accumulator, FM, FN, FK, float> acc[WFM][WFN];
#pragma unroll
  for (int i = 0; i < WFM; ++i)
#pragma unroll
    for (int j = 0; j < WFN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  for (int k0 = 0; k0 < K; k0 += BK) {
    // ---- A tile: float4 along k, stored as-is (row-major, padded) ----
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
      *reinterpret_cast<float4*>(&As[r][c]) = v;
    }
    // ---- B tile: float4 along n ----
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

    // ---- tensor-core inner loop ----
#pragma unroll
    for (int kk = 0; kk < BK; kk += FK) {
      wmma::fragment<wmma::matrix_a, FM, FN, FK, wmma::precision::tf32, wmma::row_major> a[WFM];
      wmma::fragment<wmma::matrix_b, FM, FN, FK, wmma::precision::tf32, wmma::row_major> b[WFN];
#pragma unroll
      for (int i = 0; i < WFM; ++i) {
        wmma::load_matrix_sync(a[i], &As[warpRow * WM + i * FM][kk], kLdA);
#pragma unroll
        for (int e = 0; e < a[i].num_elements; ++e) a[i].x[e] = wmma::__float_to_tf32(a[i].x[e]);
      }
#pragma unroll
      for (int j = 0; j < WFN; ++j) {
        wmma::load_matrix_sync(b[j], &Bs[kk][warpCol * WN + j * FN], kLdB);
#pragma unroll
        for (int e = 0; e < b[j].num_elements; ++e) b[j].x[e] = wmma::__float_to_tf32(b[j].x[e]);
      }
#pragma unroll
      for (int i = 0; i < WFM; ++i)
#pragma unroll
        for (int j = 0; j < WFN; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }

  // ---- epilogue: fragment -> per-warp smem -> alpha/beta + bounds -> C ----
  float* stage = Cs[warp];
#pragma unroll
  for (int i = 0; i < WFM; ++i) {
#pragma unroll
    for (int j = 0; j < WFN; ++j) {
      wmma::store_matrix_sync(stage, acc[i][j], FN, wmma::mem_row_major);
      __syncwarp();
      const int row0 = blockRow + warpRow * WM + i * FM;
      const int col0 = blockCol + warpCol * WN + j * FN;
#pragma unroll
      for (int e = lane; e < FM * FN; e += 32) {  // 8 elements per lane
        const int row = row0 + e / FN, col = col0 + e % FN;
        if (row < M && col < N) {
          const float v = alpha * stage[e];
          float* c = &C[row * N + col];
          *c = (beta == 0.0f) ? v : v + beta * *c;
        }
      }
      __syncwarp();  // staging buffer is reused by the next fragment
    }
  }
}

template <int BM, int BN, int BK, int WM, int WN>
void launch_09_tf32_wmma_cfg(int M, int N, int K, float alpha, const float* A, const float* B,
                             float beta, float* C, cudaStream_t stream) {
  const dim3 block((BM / WM) * (BN / WN) * 32);
  const dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
  const auto aligned16 = [](const void* p) { return (reinterpret_cast<unsigned long long>(p) & 15) == 0; };
  const bool vec = (K % 4 == 0) && (N % 4 == 0) && aligned16(A) && aligned16(B);
  if (vec)
    sgemm_09_tf32_wmma<BM, BN, BK, WM, WN, true><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
  else
    sgemm_09_tf32_wmma<BM, BN, BK, WM, WN, false><<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace sgemm
