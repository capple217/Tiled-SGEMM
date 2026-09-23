// Stage 6 default configuration.
// Kernel + explanation in 06_vectorized.cuh. Default <BM=128, BN=128, BK=8>, 8x8 thread tile.

#include "06_vectorized.cuh"
#include "sgemm/kernels.hpp"

namespace sgemm {

void launch_06_vectorized(int M, int N, int K, float alpha, const float* A, const float* B,
                          float beta, float* C, cudaStream_t stream) {
  launch_06_vectorized_cfg<128, 128, 8>(M, N, K, alpha, A, B, beta, C, stream);
}

}  // namespace sgemm
