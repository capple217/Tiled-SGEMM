#pragma once
// CPU stand-in for <mma.h> (WMMA), for the kernel-logic emulator only.
//
// Real WMMA fragments are distributed across the 32 lanes of a warp in an
// unspecified layout. Here every lane holds the WHOLE tile, and every warp-level
// operation is done redundantly by each lane. That preserves the semantics that
// correct kernel code may rely on (load -> elementwise ops on x[] -> mma -> store),
// and nothing that it may not (there's no lane-to-element mapping to depend on).
// store_matrix_sync is performed by lane 0 only, so there's no write race; the
// kernel's __syncwarp() then publishes it to the other lanes, exactly as on a GPU.
//
// TF32 is emulated bit-exactly for the conversion (round-to-nearest, ties away:
// cvt.rna.tf32.f32). The MMA accumulates products in FP32 in k order, which is a
// valid but not necessarily identical order to the hardware's.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "cuda_runtime.h"

namespace nvcuda {
namespace wmma {

struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};
namespace precision {
struct tf32 {};
}  // namespace precision
enum layout_t { mem_row_major, mem_col_major };

template <class Use, int M, int N, int K, class T, class Layout = void>
struct fragment {
  static constexpr int rows = std::is_same_v<Use, matrix_b> ? K : M;
  static constexpr int cols = std::is_same_v<Use, matrix_a> ? K : N;
  static constexpr int num_elements = rows * cols;  // whole tile per lane (emulation)
  float x[num_elements];
};

inline float __float_to_tf32(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7F800000u) != 0x7F800000u) u = (u + 0x1000u) & 0xFFFFE000u;  // skip Inf/NaN
  std::memcpy(&f, &u, 4);
  return f;
}

template <class Use, int M, int N, int K, class T, class L>
inline void fill_fragment(fragment<Use, M, N, K, T, L>& f, float v) {
  for (int i = 0; i < f.num_elements; ++i) f.x[i] = v;
}

// Operand fragments (row_major layout only; that's all the kernels use).
template <class Use, int M, int N, int K, class T>
inline void load_matrix_sync(fragment<Use, M, N, K, T, row_major>& f, const float* p, unsigned ldm) {
  using F = fragment<Use, M, N, K, T, row_major>;
  if (reinterpret_cast<uintptr_t>(p) % 32 != 0) std::abort();  // real WMMA requires 256-bit alignment
  for (int r = 0; r < F::rows; ++r)
    for (int c = 0; c < F::cols; ++c) f.x[r * F::cols + c] = p[r * ldm + c];
}

template <int M, int N, int K, class TA, class TB>
inline void mma_sync(fragment<accumulator, M, N, K, float>& d,
                     const fragment<matrix_a, M, N, K, TA, row_major>& a,
                     const fragment<matrix_b, M, N, K, TB, row_major>& b,
                     const fragment<accumulator, M, N, K, float>& c) {
  float out[M * N];
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float s = c.x[i * N + j];
      for (int k = 0; k < K; ++k) s = std::fmaf(a.x[i * K + k], b.x[k * N + j], s);
      out[i * N + j] = s;
    }
  std::memcpy(d.x, out, sizeof out);  // d may alias c
}

template <int M, int N, int K>
inline void store_matrix_sync(float* p, const fragment<accumulator, M, N, K, float>& f,
                              unsigned ldm, layout_t layout) {
  if (reinterpret_cast<uintptr_t>(p) % 32 != 0) std::abort();
  if (::emu::tl_linear % 32 != 0) return;  // lane 0 writes; __syncwarp publishes
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < N; ++c)
      (layout == mem_row_major ? p[r * ldm + c] : p[c * ldm + r]) = f.x[r * N + c];
}

}  // namespace wmma
}  // namespace nvcuda
