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
  };
  return k;
}

const KernelSpec* find_kernel(std::string_view name) {
  for (const auto& s : all_kernels())
    if (name == s.name) return &s;
  return nullptr;
}

}  // namespace sgemm
