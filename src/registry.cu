#include "sgemm/kernels.hpp"

namespace sgemm {

// Add one line per stage. Order = roadmap order = plot order.
const std::vector<KernelSpec>& all_kernels() {
  static const std::vector<KernelSpec> k = {
      {"naive", 1, "uncoalesced global loads/stores, no data reuse", launch_01_naive},
      {"coalesced", 2, "uncoalesced global loads (fixed by thread->column mapping)", launch_02_coalesced},
      {"smem", 3, "redundant global traffic -> explicit smem reuse; now LDS-bound", launch_03_smem_tiling},
      {"blocktiling1d", 4, "smem instruction throughput -> register reuse of B (TM outputs/thread)", launch_04_blocktiling_1d},
      {"blocktiling2d", 5, "LDS per FFMA -> TMxTN register outer product", launch_05_blocktiling_2d},
      {"vectorized", 6, "memory-instruction count + smem bank conflicts -> float4, AsT, split tile", launch_06_vectorized},
      {"warptiling", 7, "smem bandwidth per warp -> compact warp tile (8x4 lanes, 2x2 sub-tiles)", launch_07_warptiling},
      // Stage 9 changes the arithmetic: TF32 baseline, TF32 peak, TF32 tolerance.
      {"tf32_wmma", 9, "FP32 FFMA throughput ceiling -> TF32 tensor cores (wmma 16x16x8)",
       launch_09_tf32_wmma, Precision::TF32},
  };
  return k;
}

const KernelSpec* find_kernel(std::string_view name) {
  for (const auto& s : all_kernels())
    if (name == s.name) return &s;
  return nullptr;
}

}  // namespace sgemm
