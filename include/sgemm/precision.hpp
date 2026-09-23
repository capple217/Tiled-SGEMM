#pragma once
// Arithmetic a kernel performs. It decides three things: the correctness
// tolerance (host_utils.hpp), which cuBLAS configuration is the honest baseline
// (FP32 CUDA cores vs TF32 tensor cores), and which theoretical peak is the
// "% of peak" denominator. Comparing a TF32 kernel to FP32 cuBLAS would
// flatter it by the tensor-core factor. The harness never does that.

namespace sgemm {

enum class Precision { FP32, TF32 };

inline const char* precision_name(Precision p) { return p == Precision::TF32 ? "tf32" : "fp32"; }

}  // namespace sgemm
