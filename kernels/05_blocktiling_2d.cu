// Stage 5 default configuration.
// Kernel + explanation in 05_blocktiling_2d.cuh. Default <128, 128, 8, 8, 8>.

#include "05_blocktiling_2d.cuh"
#include "sgemm/kernels.hpp"

namespace sgemm {

void launch_05_blocktiling_2d(int M, int N, int K, float alpha, const float* A,
                              const float* B, float beta, float* C, cudaStream_t stream) {
  launch_05_blocktiling_2d_cfg<128, 128, 8, 8, 8>(M, N, K, alpha, A, B, beta, C, stream);
}

}  // namespace sgemm
