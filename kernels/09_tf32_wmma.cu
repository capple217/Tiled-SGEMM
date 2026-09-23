// Stage 9 default configuration (TF32 tensor cores, sm_80+).
// Kernel + explanation in 09_tf32_wmma.cuh.
// Default <BM=128, BN=128, BK=32, WM=64, WN=32>: 8 warps, 4x2 fragments per warp.

#include "09_tf32_wmma.cuh"
#include "sgemm/kernels.hpp"

namespace sgemm {

void launch_09_tf32_wmma(int M, int N, int K, float alpha, const float* A, const float* B,
                         float beta, float* C, cudaStream_t stream) {
  launch_09_tf32_wmma_cfg<128, 128, 32, 64, 32>(M, N, K, alpha, A, B, beta, C, stream);
}

}  // namespace sgemm
