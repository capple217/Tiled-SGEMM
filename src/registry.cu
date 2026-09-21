#include "sgemm/kernels.hpp"

namespace sgemm {

// Add one line per stage. Order = roadmap order = plot order.
const std::vector<KernelSpec>& all_kernels() {
  static const std::vector<KernelSpec> k = {
      {"naive", 1, "uncoalesced global loads/stores, no data reuse", launch_01_naive},
      // {"coalesced", 2, "...", launch_02_coalesced},
  };
  return k;
}

const KernelSpec* find_kernel(std::string_view name) {
  for (const auto& s : all_kernels())
    if (name == s.name) return &s;
  return nullptr;
}

}  // namespace sgemm
