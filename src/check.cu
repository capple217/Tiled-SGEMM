#include "sgemm/check.hpp"
#include "sgemm/common.hpp"

#include <cmath>

namespace sgemm {

namespace {
__global__ void abs_copy(const float* __restrict__ in, float* __restrict__ out, size_t n) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<size_t>(gridDim.x) * blockDim.x)
    out[i] = fabsf(in[i]);
}

void launch_abs(const float* in, float* out, size_t n) {
  abs_copy<<<1024, 256>>>(in, out, n);
  CUDA_CHECK(cudaGetLastError());
}
}  // namespace

Reference gpu_reference(CublasSgemm& blas, int M, int N, int K, float alpha,
                        const float* d_A, const float* d_B, float beta,
                        const std::vector<float>& h_C0) {
  const size_t nA = size_t(M) * K, nB = size_t(K) * N, nC = size_t(M) * N;
  Reference r;
  r.ref.resize(nC);
  r.scale.resize(nC);

  float *d_C = nullptr, *d_absA = nullptr, *d_absB = nullptr;
  CUDA_CHECK(cudaMalloc(&d_C, nC * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_absA, nA * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_absB, nB * sizeof(float)));

  // Expected result.
  if (beta != 0.0f) {
    CUDA_CHECK(cudaMemcpy(d_C, h_C0.data(), nC * sizeof(float), cudaMemcpyHostToDevice));
  } else {
    CUDA_CHECK(cudaMemset(d_C, 0, nC * sizeof(float)));  // never read by cuBLAS
  }
  blas(M, N, K, alpha, d_A, d_B, beta, d_C, nullptr);
  CUDA_CHECK(cudaMemcpy(r.ref.data(), d_C, nC * sizeof(float), cudaMemcpyDeviceToHost));

  // Error scale |alpha| * |A||B|  (+ |beta C0| on the host).
  launch_abs(d_A, d_absA, nA);
  launch_abs(d_B, d_absB, nB);
  blas(M, N, K, std::fabs(alpha), d_absA, d_absB, 0.0f, d_C, nullptr);
  CUDA_CHECK(cudaMemcpy(r.scale.data(), d_C, nC * sizeof(float), cudaMemcpyDeviceToHost));
  if (beta != 0.0f) {
    for (size_t i = 0; i < nC; ++i) r.scale[i] += std::fabs(beta * h_C0[i]);
  }

  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(d_absA));
  CUDA_CHECK(cudaFree(d_absB));
  return r;
}

}  // namespace sgemm
