#pragma once
// CUDA-event timing shared by sgemm_bench and sgemm_autotune.
// See bench/bench.cu header for the methodology (per-iteration events, median,
// L2 flush outside the timed window).

#include <cuda_runtime.h>

#include <functional>
#include <vector>

#include "sgemm/common.hpp"

namespace sgemm {

using Run = std::function<void(cudaStream_t)>;

// Returns per-iteration kernel times in milliseconds.
inline std::vector<double> time_runs(const Run& run, int warmup, int iters, cudaStream_t s,
                                     void* flush_buf, size_t flush_bytes) {
  for (int i = 0; i < warmup; ++i) run(s);
  CUDA_CHECK(cudaStreamSynchronize(s));

  std::vector<cudaEvent_t> start(iters), stop(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventCreate(&start[i]));
    CUDA_CHECK(cudaEventCreate(&stop[i]));
  }
  for (int i = 0; i < iters; ++i) {
    // Outside the timed window: evict A/B/C from L2 by streaming writes over a
    // buffer 2x L2's size. The byte value changes per iteration so nothing can
    // elide it as redundant.
    if (flush_buf) CUDA_CHECK(cudaMemsetAsync(flush_buf, i & 0xff, flush_bytes, s));
    CUDA_CHECK(cudaEventRecord(start[i], s));
    run(s);
    CUDA_CHECK(cudaEventRecord(stop[i], s));
  }
  CUDA_CHECK(cudaStreamSynchronize(s));

  std::vector<double> ms(iters);
  for (int i = 0; i < iters; ++i) {
    float t = 0;
    CUDA_CHECK(cudaEventElapsedTime(&t, start[i], stop[i]));
    ms[i] = t;
    CUDA_CHECK(cudaEventDestroy(start[i]));
    CUDA_CHECK(cudaEventDestroy(stop[i]));
  }
  return ms;
}

}  // namespace sgemm
