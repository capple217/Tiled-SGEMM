#pragma once
// Device facts needed for the "% of theoretical peak" denominators.

#include <string>

namespace sgemm {

struct DeviceInfo {
  std::string name;
  int cc_major = 0, cc_minor = 0;
  int sm_count = 0;
  int fp32_lanes_per_sm = 0;       // 0 = unknown arch
  int max_sm_clock_mhz = 0;        // boost clock reported by the driver
  int mem_clock_mhz = 0;
  int mem_bus_width_bits = 0;
  int l2_bytes = 0;
  int driver_version = 0, runtime_version = 0;

  // Peak FP32 = SMs * FP32 lanes/SM * 2 FLOP/FFMA * clock.
  // Pass the LOCKED clock for reportable runs; the boost clock is what the
  // spec sheet uses (A100: 108 * 64 * 2 * 1.410 GHz = 19.5 TFLOPS).
  double peak_fp32_gflops(int sm_clock_mhz) const;
  // Peak dense TF32 tensor-core throughput at a given clock, or 0 if unknown.
  // Per-SM rate is SKU-dependent (see device_info.cu); tf32_source says which
  // rule was used so the CSV records it. Override with --tf32-flop-per-clk-sm.
  int tf32_flop_per_clk_sm = 0;
  std::string tf32_source;
  double peak_tf32_gflops(int sm_clock_mhz) const;
  // Peak DRAM = 2 (DDR) * mem clock * bus width / 8. (A100-40GB: ~1555 GB/s.)
  double peak_dram_gbs() const;
};

DeviceInfo query_device(int dev = 0);

}  // namespace sgemm
