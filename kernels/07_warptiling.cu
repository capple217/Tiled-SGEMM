// Stage 7 default configuration. Kernel + explanation in 07_warptiling.cuh.
// Default <BM=128, BN=128, BK=8, WM=64, WN=32>: 8 warps, 256 threads, 64 outputs/thread.

#include "07_warptiling.cuh"
#include "sgemm/kernels.hpp"

namespace sgemm {

void launch_07_warptiling(int M, int N, int K, float alpha, const float* A, const float* B,
                          float beta, float* C, cudaStream_t stream) {
  launch_07_warptiling_cfg<128, 128, 8, 64, 32>(M, N, K, alpha, A, B, beta, C, stream);
}

}  // namespace sgemm
