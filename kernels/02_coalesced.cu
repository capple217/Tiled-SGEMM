// Stage 2 — global-memory coalescing. Diff this file against 01_naive.cu:
// the ONLY change is which thread index walks which dimension of C.
//
//     stage 1: row <- threadIdx.x, col <- threadIdx.y
//     stage 2: row <- threadIdx.y, col <- threadIdx.x      (this file)
//
// Same FLOPs, same instructions, same loop. Only the address pattern that a
// warp presents to the memory system changes, which makes this the cleanest
// possible experiment on coalescing.
//
// A warp is still threadIdx.x = 0..31 at fixed threadIdx.y, but that is now
// ONE row of C, 32 consecutive columns. Per k-iteration:
//
//   LDG A[row*K + k]   all 32 threads read the SAME address -> 1 sector,
//                      broadcast. (Was: 32 sectors.)
//   LDG B[k*N + col]   32 consecutive floats = 128 contiguous bytes -> 4 sectors,
//                      i.e. perfectly coalesced. (Was: 1 sector broadcast.)
//   STG C[row*N + col] 128 contiguous bytes -> 4 sectors. (Was: 32 partial sectors.)
//
// Sectors per warp per k: 1 + 4 = 5, down from 32 + 1 = 33. Every sector
// fetched is now fully used.
//
// Expected profiler deltas vs stage 1 (verify with profiling/profile.sh):
//  * gld sectors/request: ~16.5 -> ~2.5 (average of the 1-sector A broadcast
//    and the 4-sector B request). ncu's "uncoalesced global access" warning
//    should disappear.
//  * L1TEX wavefronts per request drop ~6x, so the LG-throttle stall share falls
//    and FMA-pipe utilisation rises.
//  * Typically a 5-10x speedup. It's the cheapest large win in the project.
//
// Still slow because every FFMA needs 2 global loads (one of them B, which
// comes through L1 at best). The reuse of each B element by the 32 rows of the
// block, and of each A row by 32 columns, happens only if the caches happen to
// hold them. Stage 3 makes that reuse explicit in shared memory.

#include "sgemm/common.hpp"
#include "sgemm/kernels.hpp"

namespace sgemm {
namespace {

constexpr int kBlock = 32;

__global__ void __launch_bounds__(kBlock * kBlock)
    sgemm_02_coalesced(int M, int N, int K, float alpha, const float* __restrict__ A,
                       const float* __restrict__ B, float beta, float* __restrict__ C) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;  // warp-uniform
  const int col = blockIdx.x * blockDim.x + threadIdx.x;  // consecutive across warp

  if (row >= M || col >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
  }

  const int idx = row * N + col;
  C[idx] = (beta == 0.0f) ? alpha * acc : alpha * acc + beta * C[idx];
}

}  // namespace

void launch_02_coalesced(int M, int N, int K, float alpha, const float* A,
                         const float* B, float beta, float* C, cudaStream_t stream) {
  const dim3 block(kBlock, kBlock);
  // grid.x now covers N because threadIdx.x indexes columns.
  const dim3 grid(ceil_div(N, kBlock), ceil_div(M, kBlock));
  sgemm_02_coalesced<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace sgemm
