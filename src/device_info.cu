#include "sgemm/common.hpp"
#include "sgemm/device_info.hpp"

namespace sgemm {

namespace {
// FP32 lanes (a.k.a. "CUDA cores") per SM. This is the one hard-coded table in
// the project because the runtime doesn't expose it.
//   sm_80 (A100): 64 lanes/SM - 4 SMSPs x 16 FP32 lanes.
//   sm_86/89 (GA10x/AD10x): 128 lanes/SM - the second datapath per SMSP can
//     issue FP32 *or* INT32. So the 2x peak on consumer Ampere/Ada is only
//     reachable when there is essentially no integer work in the loop, which
//     is one reason dev-card %-of-peak reads lower than A100's for the same
//     kernel. Keep that in mind before comparing across cards.
int fp32_lanes(int major, int minor) {
  if (major == 7) return 64;                    // Volta / Turing
  if (major == 8 && minor == 0) return 64;      // A100 / A30
  if (major == 8) return 128;                   // GA10x, Orin, AD10x
  if (major == 9) return 128;                   // H100
  if (major >= 10) return 128;                  // Blackwell (verify per SKU)
  return 0;
}
// Dense TF32 tensor-core FLOP per clock per SM, from published dense TF32 peaks
// divided by (SMs x boost clock). Check against your SKU's datasheet.
//   A100 (8.0):  156 TFLOPS / (108 x 1.41 GHz)  = 1024
//   A10/A40/L4 (8.6/8.9 pro): e.g. A10 62.5 / (72 x 1.695) = 512
//   RTX 3090/4090 (8.6/8.9 GeForce): 35.6 / (82 x 1.695)   = 256
//     (GeForce halves tensor throughput with FP32 accumulate)
//   H100 SXM (9.0): 494.7 / (132 x 1.83)            = 2048
int tf32_rate(int major, int minor, const std::string& name, std::string& src) {
  if (major == 8 && minor == 0) { src = "table:sm80"; return 1024; }
  if (major == 8 && (minor == 6 || minor == 9)) {
    const bool geforce = name.find("GeForce") != std::string::npos;
    src = geforce ? "table:sm8x_geforce" : "table:sm8x_pro";
    return geforce ? 256 : 512;
  }
  if (major == 9) { src = "table:sm90"; return 2048; }
  src = "unknown";
  return 0;
}
}  // namespace

double DeviceInfo::peak_tf32_gflops(int sm_clock_mhz) const {
  if (tf32_flop_per_clk_sm == 0 || sm_clock_mhz <= 0) return 0.0;
  return static_cast<double>(sm_count) * tf32_flop_per_clk_sm * sm_clock_mhz / 1e3;
}

double DeviceInfo::peak_fp32_gflops(int sm_clock_mhz) const {
  if (fp32_lanes_per_sm == 0 || sm_clock_mhz <= 0) return 0.0;
  return static_cast<double>(sm_count) * fp32_lanes_per_sm * 2.0 * sm_clock_mhz / 1e3;
}

double DeviceInfo::peak_dram_gbs() const {
  return 2.0 * mem_clock_mhz * 1e6 * (mem_bus_width_bits / 8.0) / 1e9;
}

DeviceInfo query_device(int dev) {
  DeviceInfo d;
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  d.name = p.name;
  d.cc_major = p.major;
  d.cc_minor = p.minor;
  d.sm_count = p.multiProcessorCount;
  d.l2_bytes = p.l2CacheSize;
  d.fp32_lanes_per_sm = fp32_lanes(p.major, p.minor);
  d.tf32_flop_per_clk_sm = tf32_rate(p.major, p.minor, d.name, d.tf32_source);
  // Clock fields on cudaDeviceProp are deprecated (removed in CUDA 13);
  // the attribute API is stable. Values are in kHz.
  int khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev));
  d.max_sm_clock_mhz = khz / 1000;
  CUDA_CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrMemoryClockRate, dev));
  d.mem_clock_mhz = khz / 1000;
  CUDA_CHECK(cudaDeviceGetAttribute(&d.mem_bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
  CUDA_CHECK(cudaDriverGetVersion(&d.driver_version));
  CUDA_CHECK(cudaRuntimeGetVersion(&d.runtime_version));
  return d;
}

}  // namespace sgemm
