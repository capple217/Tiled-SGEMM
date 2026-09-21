// Stage 4 default configuration. The kernel template and its explanation live in
// 04_blocktiling_1d.cuh so the autotuner (bench/autotune.cu) can instantiate
// other configurations.
//
// Default <BM=64, BN=64, BK=8, TM=8>: 512 threads, 4 KB smem, 8 accumulators.

#include "04_blocktiling_1d.cuh"
#include "sgemm/kernels.hpp"

namespace sgemm {

void launch_04_blocktiling_1d(int M, int N, int K, float alpha, const float* A,
                              const float* B, float beta, float* C, cudaStream_t stream) {
  launch_04_blocktiling_1d_cfg<64, 64, 8, 8>(M, N, K, alpha, A, B, beta, C, stream);
}

}  // namespace sgemm
