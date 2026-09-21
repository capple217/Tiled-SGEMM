#pragma once
// The contract every stage implements, and the registry the drivers iterate.
//
//   C = alpha * A * B + beta * C
//   A: M x K, B: K x N, C: M x N, all ROW-MAJOR and PACKED (lda=K, ldb=N, ldc=N).
//
// beta == 0 follows BLAS semantics: C is write-only and may hold garbage/NaN on
// entry. A kernel that computes `beta * C` unconditionally propagates NaN
// (0 * NaN = NaN) and also pays for an M*N read cuBLAS skips. The correctness
// harness pre-fills C with NaN for beta == 0 cases to catch exactly this.
//
// Launchers are host functions so each stage owns its grid/block/smem choice.
// They must not synchronize; the harness times them with events on `stream`.

#include <cuda_runtime.h>

#include <string_view>
#include <vector>

namespace sgemm {

using LaunchFn = void (*)(int M, int N, int K, float alpha, const float* A,
                          const float* B, float beta, float* C,
                          cudaStream_t stream);

struct KernelSpec {
  const char* name;      // CLI name, also the CSV "kernel" column
  int stage;             // roadmap stage number
  const char* targets;   // the hardware bottleneck this stage attacks
  LaunchFn launch;
};

const std::vector<KernelSpec>& all_kernels();
const KernelSpec* find_kernel(std::string_view name);

// ---- stage launchers (one translation unit each, in kernels/) --------------
void launch_01_naive(int M, int N, int K, float alpha, const float* A,
                     const float* B, float beta, float* C, cudaStream_t stream);
void launch_02_coalesced(int M, int N, int K, float alpha, const float* A,
                         const float* B, float beta, float* C, cudaStream_t stream);
void launch_03_smem_tiling(int M, int N, int K, float alpha, const float* A,
                           const float* B, float beta, float* C, cudaStream_t stream);
void launch_04_blocktiling_1d(int M, int N, int K, float alpha, const float* A,
                              const float* B, float beta, float* C, cudaStream_t stream);
// SPOILER branch: Claude's reference stages 5 and 6.
void launch_05_blocktiling_2d(int M, int N, int K, float alpha, const float* A,
                              const float* B, float beta, float* C, cudaStream_t stream);
void launch_06_vectorized(int M, int N, int K, float alpha, const float* A,
                          const float* B, float beta, float* C, cudaStream_t stream);

}  // namespace sgemm
