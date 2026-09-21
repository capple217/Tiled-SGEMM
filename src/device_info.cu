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
}  // namespace

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
