// Stage 3 — shared-memory tiling ("tiled GEMM").
//
// Block = 32x32 threads computes a 32x32 tile of C (one output per thread, as
// before). The K dimension is walked in chunks of BK = 32. For each chunk the
// block COOPERATIVELY stages a 32x32 tile of A and a 32x32 tile of B into
// shared memory (one element each per thread), barriers, and then every thread
// runs its 32 FMAs out of shared memory.
//
// What this buys, in numbers (per block, per K-chunk):
//   global loads:  2 * 32*32 floats = 8 KB
//   FLOPs:         2 * 32*32*32     = 65536
//   => 8 FLOP per global byte, vs 0.25 FLOP/B at the instruction level in
//      stages 1-2. Each element of A is fetched from global once per block
//      and reused 32x out of smem (by the 32 columns); likewise B by 32 rows.
//   That is the "reuse factor = tile width" rule: global traffic for the whole
//   GEMM drops from ~2MNK floats to 2MNK/32.
//
// Access patterns (warp = threadIdx.x 0..31 at fixed threadIdx.y = ty):
//   gmem -> smem, A:  A[(tileRow+ty)*K + k0 + tx]   32 consecutive floats: 4 sectors, coalesced
//   gmem -> smem, B:  B[(k0+ty)*N + tileCol + tx]   32 consecutive floats: 4 sectors, coalesced
//   STS  As[ty][tx], Bs[ty][tx]: 32 consecutive words -> 32 distinct banks, conflict-free
//   inner loop LDS As[ty][kk]: same address for the whole warp -> broadcast, 1 wavefront
//   inner loop LDS Bs[kk][tx]: 32 consecutive words -> conflict-free, 1 wavefront
//
// Honest expectation: the speedup over stage 2 is SMALLER than the 32x traffic
// cut suggests, often only 1.3-2x. Stage 2 already got much of this reuse
// implicitly from L1/L2 caching. What changes is that the reuse is now
// guaranteed and cheap (smem has no tag lookup, no miss path), and the new
// bottleneck is exposed: shared-memory INSTRUCTION throughput.
//
// Source-level count: per k, LDS As + LDS Bs + FFMA, i.e. 2 LDS per FFMA.
// What ptxas actually emits (scripts/sass_stats.sh, sm_80), per 32-wide k-tile:
//     32 x LDS      (Bs[kk][tx], one per kk)
//      8 x LDS.128  (As[ty][kk..kk+3]: contiguous in kk, so ptxas vectorised it)
//     32 x FFMA
// => 40 LDS instructions : 32 FFMA = 1.25, not 2. Lesson: count SASS, not source.
// Roofline for that ratio on A100: an SM's shared memory serves 128 B/clk, about
// ONE warp-wide LDS.32 per clock, while its 64 FP32 lanes retire TWO warp-wide
// FFMAs per clock. At 1.25 LDS : 1 FFMA the SM retires at most 0.8 FFMA/clk,
// a ceiling of ~40% of FP32 peak from the LDS pipe alone (optimistically
// assuming each broadcast LDS.128 costs one wavefront; ncu's "lds
// wavefronts/inst" says whether it does). In practice the barrier pair per
// tile and the exposed global-load latency keep it well below that ceiling.
// Expect ncu to show "MIO Throttle" / "Short Scoreboard" (waiting on the smem
// pipe) and "Barrier" stalls, with DRAM and L2 throughput far below peak.
//
// Occupancy: 1024 threads/block and 8 KB smem/block. A100 allows 2048 threads/SM,
// so at most 2 blocks/SM, and only if registers <= 32/thread
// (65536 regs / 2048 threads). launch__occupancy_limit_registers in the profile
// tells you whether ptxas stayed under that.

#include "sgemm/common.hpp"
#include "sgemm/kernels.hpp"

namespace sgemm {
namespace {

constexpr int kTile = 32;  // BM = BN = BK = 32; one thread per (row, col) and per load.

__global__ void __launch_bounds__(kTile * kTile)
    sgemm_03_smem_tiling(int M, int N, int K, float alpha, const float* __restrict__ A,
                         const float* __restrict__ B, float beta, float* __restrict__ C) {
  __shared__ float As[kTile][kTile];
  __shared__ float Bs[kTile][kTile];

  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * kTile + ty;
  const int col = blockIdx.x * kTile + tx;

  // NOTE: no early `return` for out-of-range threads any more. Every thread
  // must reach every __syncthreads(). A thread that exited early would leave
  // the barrier waiting on a thread that never arrives (undefined behaviour;
  // in practice a hang or wrong results). Out-of-range threads still help load
  // tiles, and they just don't store.
  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += kTile) {
    // Cooperative, coalesced loads. Zero-fill outside the matrix: a zero in As
    // or Bs contributes exactly 0 to every dot product, which handles ragged
    // M, N and K with no branches in the inner loop.
    As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + (k0 + tx)] : 0.0f;
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.0f;
    __syncthreads();  // RAW: tiles fully written before anyone reads them

#pragma unroll
    for (int kk = 0; kk < kTile; ++kk) {
      acc += As[ty][kk] * Bs[kk][tx];  // LDS (broadcast) + LDS (conflict-free) + FFMA
    }
    __syncthreads();  // WAR: everyone done reading before the next overwrite
  }

  if (row < M && col < N) {
    const int idx = row * N + col;
    C[idx] = (beta == 0.0f) ? alpha * acc : alpha * acc + beta * C[idx];
  }
}

}  // namespace

void launch_03_smem_tiling(int M, int N, int K, float alpha, const float* A,
                           const float* B, float beta, float* C, cudaStream_t stream) {
  const dim3 block(kTile, kTile);
  const dim3 grid(ceil_div(N, kTile), ceil_div(M, kTile));
  sgemm_03_smem_tiling<<<grid, block, 0, stream>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace sgemm
