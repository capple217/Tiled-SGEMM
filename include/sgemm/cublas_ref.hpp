#pragma once
// cuBLAS SGEMM behind the same row-major contract as our kernels.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUBLAS_CHECK(expr)                                                      \
  do {                                                                          \
    cublasStatus_t st_ = (expr);                                                \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                         \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", static_cast<int>(st_), \
                   __FILE__, __LINE__);                                         \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

namespace sgemm {

// Math-mode policy (the honesty rule from the brief):
//   Default  : CUBLAS_DEFAULT_MATH. For cublasSgemm this is plain FP32 on CUDA
//              cores. TF32 tensor cores are only used if you opt in
//              (CUBLAS_TF32_TENSOR_OP_MATH / *_FAST_TF32 compute types), so this
//              is the apples-to-apples baseline for CUDA-core SGEMM, and it is
//              what cuBLAS "normally" runs, i.e. its best FP32 number.
//   Pedantic : CUBLAS_PEDANTIC_MATH. Prescribed precision for every phase; may
//              exclude some fast algorithms. Useful as a cross-check, but it can
//              under-state cuBLAS and so over-state our % of cuBLAS. Never
//              report headline numbers against Pedantic.
// Verify, don't trust: scripts/env_check.sh runs cuBLAS under ncu and prints the
// kernel name. An FP32 CUDA-core kernel is named like "ampere_sgemm_128x64_nn";
// anything with "tf32"/"s1688"/"16816" means tensor cores are in play.
//   TF32     : CUBLAS_TF32_TENSOR_OP_MATH. The honest baseline for a TF32
//              tensor-core kernel (stage 9) and ONLY for that. Its result is
//              checked with the TF32 tolerance.
enum class CublasMath { Default, Pedantic, TF32 };

class CublasSgemm {
 public:
  explicit CublasSgemm(CublasMath mode = CublasMath::Default);
  ~CublasSgemm();
  CublasSgemm(const CublasSgemm&) = delete;
  CublasSgemm& operator=(const CublasSgemm&) = delete;

  // Row-major C(MxN) = alpha * A(MxK) * B(KxN) + beta * C.
  void operator()(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C, cudaStream_t stream);

  int version() const;
  const char* mode_name() const;

 private:
  cublasHandle_t handle_ = nullptr;
  CublasMath mode_;
};

}  // namespace sgemm
