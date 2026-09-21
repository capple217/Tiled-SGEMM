#pragma once
// GPU-side reference for problems too large for the CPU reference.
//
// Reference result:  cuBLAS FP32 (default math) from the same A, B, C0.
// Error scale s_ij:  |alpha| * (|A| |B|)_ij + |beta| * |C0_ij|, computed as a
//                    second cuBLAS GEMM on elementwise-|.| copies of A and B.
// Both are FP32, so the reference carries its own O(eps) error; tolerance()
// in host_utils.hpp budgets for errors on both sides.

#include <vector>

#include "sgemm/cublas_ref.hpp"
#include "sgemm/host_utils.hpp"

namespace sgemm {

struct Reference {
  std::vector<float> ref;    // expected C
  std::vector<float> scale;  // per-element error scale s_ij
};

// d_A, d_B: device operands. h_C0: host initial C (ignored when beta == 0).
Reference gpu_reference(CublasSgemm& blas, int M, int N, int K, float alpha,
                        const float* d_A, const float* d_B, float beta,
                        const std::vector<float>& h_C0);

}  // namespace sgemm
