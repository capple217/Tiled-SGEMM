// Stage 1 — naive SGEMM. One thread per output element, operands straight from
// global memory. This is the floor. The point of it is to have a *measured*,
// profiled baseline whose bottleneck we can name exactly.
//
// Thread -> output mapping (this is what stage 2 changes, and nothing else):
//     row = blockIdx.x * 32 + threadIdx.x      <- threadIdx.x walks ROWS of C
//     col = blockIdx.y * 32 + threadIdx.y
//
// How a warp sees memory. A warp is 32 threads with consecutive linear id
// tid = threadIdx.x + 32*threadIdx.y. With blockDim = (32,32), one warp is
// threadIdx.x = 0..31 at FIXED threadIdx.y, i.e. 32 consecutive ROWS, ONE column.
// On each k-iteration the warp issues:
//
//   LDG A[row*K + k]   32 addresses, stride 4K bytes apart -> 32 distinct 128B
//                      lines -> 32 sectors for 128 useful bytes. Ideal for a
//                      warp-wide 4B load is 4 sectors. 8x sector over-fetch per
//                      request, and 32 separate tag lookups in L1.
//   LDG B[k*N + col]   all 32 threads, SAME address -> 1 sector, broadcast.
//                      Cheap, but only 4 useful bytes per request.
//   FFMA               the only useful instruction: 1 per 2 loads.
//
// Why it's slow (prediction, to be confirmed by ncu):
//  * It's NOT primarily DRAM-bound. All 32 warps of a block read the SAME 32 rows
//    of A (they differ only in col), and each 32B sector of A holds 8 consecutive
//    k's, so L1 hit rate should be high. DRAM traffic is modest.
//  * The binding constraint is the L1TEX/LSU pipeline: every A-load request fans
//    out into 32 sectors/tag lookups, so the load pipe is saturated with
//    wavefronts while the FMA pipe idles. Expect warps stalled on "LG Throttle"
//    (LSU queue full) and "Long Scoreboard" (waiting on L1TEX results).
//  * Arithmetic intensity at the instruction level is 2 FLOP per 8 loaded bytes
//    = 0.25 FLOP/B, versus ~12.5 FLOP/B needed to be compute-bound on A100 DRAM
//    (19.5 TFLOP/s / 1.55 TB/s). Only cache reuse keeps this from being far worse,
//    and stage 3 (shared-memory tiling) is where we take control of that reuse.
//  * The C store has the same 32-sector pattern as A, but it happens once per
//    thread vs K times for the loads, so it's noise here.

#include "sgemm/common.hpp"
#include "sgemm/kernels.hpp"

namespace sgemm {
namespace {

constexpr int kBlock = 32;  // 32x32 = 1024 threads: max block size, 1 row of
                            // threadIdx.x == exactly 1 warp, which makes the
                            // access-pattern argument above exact.

// __launch_bounds__(1024) caps the kernel at 64 registers/thread
// (65536 regs / 1024 threads) so a block can always launch. This kernel needs ~30.
// __restrict__ + const on A/B promise no aliasing, which lets the compiler
// use the read-only (LDG.E.CONSTANT / non-coherent) load path. It's not a big win
// here, but it's free and every later stage relies on it.
__global__ void __launch_bounds__(kBlock * kBlock)
    sgemm_01_naive(int M, int N, int K, float alpha, const float* __restrict__ A,
                   const float* __restrict__ B, float beta, float* __restrict__ C) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;

  // Boundary guard for non-multiple-of-32 M/N. Out-of-range threads exit before
  // touching memory. (There is no __syncthreads in this kernel, so early exit is
  // safe. That stops being true at stage 3.)
  if (row >= M || col >= N) return;

  float acc = 0.0f;  // FP32 accumulator, a serial chain of K dependent FFMAs.
  for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];  // contracted into one FFMA (-fmad=true)
  }

  // Epilogue. `beta == 0` is warp-uniform (a kernel argument), so this branch
  // never diverges. Skipping the C read when beta == 0 is BLAS semantics, not an
  // optimisation: C may be uninitialised/NaN and 0*NaN = NaN.
  const int idx = row * N + col;
  C[idx] = (beta == 0.0f) ? alpha * acc : alpha * acc + beta * C[idx];
}

}  // namespace

void launch_01_naive(int M, int N, int K, float alpha, const float* A,
                     const float* B, float beta, float* C, cudaStream_t stream) {
  const dim3 block(kBlock, kBlock);
  // grid.x covers M because threadIdx.x indexes rows (see header comment).
  const dim3 grid(ceil_div(M, kBlock), ceil_div(N, kBlock));
  sgemm_01_naive<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace sgemm
