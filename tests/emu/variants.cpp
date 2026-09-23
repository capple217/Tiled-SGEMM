// Non-default template configurations run through the emulator, so logic that
// only matters for other tile shapes (e.g. WMITER = 1, BK = 16, fewer threads
// than tile rows) is checked on the CPU before the autotuner meets it on a GPU.
// Each entry must satisfy the kernel's static_asserts. Add one when you add a
// family to bench/autotune.cu.

#include <vector>

#include "04_blocktiling_1d.emu.hpp"
#include "05_blocktiling_2d.emu.hpp"
#include "06_vectorized.emu.hpp"
#include "07_warptiling.emu.hpp"
#include "sgemm/kernels.hpp"

namespace sgemm {

std::vector<KernelSpec> emu_variants() {
  return {
      {"v04<32,64,16,4>", 4, "", &launch_04_blocktiling_1d_cfg<32, 64, 16, 4>},
      {"v04<128,64,8,16>", 4, "", &launch_04_blocktiling_1d_cfg<128, 64, 8, 16>},
      {"v05<64,64,8,4,4>", 5, "", &launch_05_blocktiling_2d_cfg<64, 64, 8, 4, 4>},
      {"v05<128,64,16,8,4>", 5, "", &launch_05_blocktiling_2d_cfg<128, 64, 16, 8, 4>},
      {"v06<64,128,8>", 6, "", &launch_06_vectorized_cfg<64, 128, 8>},
      {"v06<128,128,16>", 6, "", &launch_06_vectorized_cfg<128, 128, 16>},
      {"v07<64,64,8,32,32>", 7, "", &launch_07_warptiling_cfg<64, 64, 8, 32, 32>},
      {"v07<128,128,16,64,64>", 7, "", &launch_07_warptiling_cfg<128, 128, 16, 64, 64>},
  };
}

}  // namespace sgemm
