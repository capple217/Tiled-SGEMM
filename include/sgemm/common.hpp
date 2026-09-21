#pragma once
// Error-checking macros and small integer helpers shared by every target.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <limits>

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t err_ = (expr);                                                  \
    if (err_ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                      \
                   cudaGetErrorName(err_), __FILE__, __LINE__,                  \
                   cudaGetErrorString(err_));                                   \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

namespace sgemm {

__host__ __device__ constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Every kernel indexes with 32-bit ints. That is a deliberate performance
// choice: 64-bit address arithmetic costs two IMADs (lo/hi) per offset instead
// of one, and in the inner loop of the early stages that is a real fraction of
// the non-FFMA instruction budget. The price is this guard: the largest linear
// offset we form is max(M*K, K*N, M*N), which must stay below 2^31.
// (8192^2 = 2^26, so the whole sweep has 32x headroom.)
inline bool dims_fit_int32(int M, int N, int K) {
  const long long lim = std::numeric_limits<int>::max();
  return M > 0 && N > 0 && K > 0 &&
         1LL * M * K <= lim && 1LL * K * N <= lim && 1LL * M * N <= lim;
}

}  // namespace sgemm
