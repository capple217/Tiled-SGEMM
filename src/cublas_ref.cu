#include "sgemm/cublas_ref.hpp"

#include <cstdlib>
#include <cstring>

namespace sgemm {

CublasSgemm::CublasSgemm(CublasMath mode) : mode_(mode) {
  CUBLAS_CHECK(cublasCreate(&handle_));
  CUBLAS_CHECK(cublasSetMathMode(
      handle_, mode == CublasMath::Pedantic ? CUBLAS_PEDANTIC_MATH : CUBLAS_DEFAULT_MATH));
  // NVIDIA_TF32_OVERRIDE=0 can only *disable* TF32; nothing in the env can force
  // it on for cublasSgemm + DEFAULT_MATH. Still, surface it so a CSV never
  // carries an unexplained environment.
  if (const char* v = std::getenv("NVIDIA_TF32_OVERRIDE")) {
    std::fprintf(stderr, "[cublas] note: NVIDIA_TF32_OVERRIDE=%s\n", v);
  }
}

CublasSgemm::~CublasSgemm() {
  if (handle_) cublasDestroy(handle_);
}

void CublasSgemm::operator()(int M, int N, int K, float alpha, const float* A,
                             const float* B, float beta, float* C,
                             cudaStream_t stream) {
  // cuBLAS is column-major. A row-major X (r x c) has the same bytes as a
  // column-major X^T (c x r). So row-major C = A*B is column-major
  //     C^T (N x M) = B^T (N x K) * A^T (K x M),
  // i.e. call sgemm with the operands SWAPPED and no transposes:
  //     m = N, n = M, k = K,  "A" = B (ld N),  "B" = A (ld K),  C (ld N).
  // No data is moved and no transpose kernel runs, so this costs nothing and
  // cuBLAS sees an ordinary NN problem.
  CUBLAS_CHECK(cublasSetStream(handle_, stream));
  CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                           N, M, K, &alpha,
                           B, N,
                           A, K,
                           &beta, C, N));
}

int CublasSgemm::version() const {
  int v = 0;
  CUBLAS_CHECK(cublasGetVersion(handle_, &v));
  return v;
}

const char* CublasSgemm::mode_name() const {
  return mode_ == CublasMath::Pedantic ? "pedantic" : "default_fp32";
}

}  // namespace sgemm
