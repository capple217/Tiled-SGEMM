#pragma once
// Minimal CPU stand-in for the CUDA runtime, used ONLY by the kernel-logic
// emulator (tests/emu/run_emu_tests.sh). It lets the real kernel sources compile
// with g++ and run one OS thread per CUDA thread, blocks executed one at a time.
//
// What it checks:   indexing, boundary handling, zero-fill, tile loop bounds,
//                   epilogue/beta semantics, barrier placement (a missing
//                   __syncthreads usually shows up as wrong results; a
//                   barrier reached after some thread exited is flagged).
// What it CANNOT check: performance, warp-synchronous behaviour (no warps here),
//                   memory-model subtleties, bank conflicts, alignment faults.
//                   It is a logic test. The GPU harness is the real test.

#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>
#include <algorithm>
#include <vector>

struct dim3 {
  unsigned int x = 1, y = 1, z = 1;
  constexpr dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
};

struct alignas(16) float4 { float x, y, z, w; };
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }

using cudaStream_t = void*;
enum cudaError_t { cudaSuccess = 0 };
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorName(cudaError_t) { return "cudaSuccess"; }
inline const char* cudaGetErrorString(cudaError_t) { return "no error"; }

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static  // one block runs at a time, so a function static is per-block smem
#define __launch_bounds__(...)

namespace emu {
inline thread_local dim3 tl_threadIdx, tl_blockIdx;
inline thread_local unsigned tl_linear = 0;  // linear thread id within the block
// One barrier per warp (32 consecutive linear ids) for __syncwarp.
inline std::vector<std::barrier<>*> g_warp_barriers;
inline dim3 g_blockDim, g_gridDim;
inline std::barrier<>* g_barrier = nullptr;
inline std::atomic<int> g_exited{0};
// Set if any thread reaches __syncthreads after another thread of the same block
// has already exited: that is a barrier in divergent control flow, which is
// undefined behaviour in CUDA even when it happens to work.
inline std::atomic<bool> g_divergent_barrier{false};

// Run `body` as a grid of blocks. Threads of one block run concurrently (so
// barriers are real); blocks run sequentially (so __shared__ statics are safe).
// One pool of blockDim threads is reused for every block: thread creation is
// the dominant cost otherwise, especially under ASan.
inline void launch(dim3 grid, dim3 block, const std::function<void()>& body) {
  g_gridDim = grid;
  g_blockDim = block;
  const unsigned nt = block.x * block.y * block.z;
  const unsigned nblocks = grid.x * grid.y * grid.z;
  std::barrier<> phase(nt);  // separates blocks; never dropped from
  std::vector<std::thread> ts;
  ts.reserve(nt);
  for (unsigned t = 0; t < nt; ++t) {
    ts.emplace_back([&, t] {
      tl_threadIdx = dim3(t % block.x, (t / block.x) % block.y, t / (block.x * block.y));
      tl_linear = t;
      for (unsigned b = 0; b < nblocks; ++b) {
        if (t == 0) {  // fresh per-block barriers, since early exits drop from them
          g_barrier = new std::barrier<>(nt);
          g_exited = 0;
          g_warp_barriers.clear();
          for (unsigned w = 0; w < (nt + 31) / 32; ++w)
            g_warp_barriers.push_back(new std::barrier<>(std::min(32u, nt - 32 * w)));
        }
        phase.arrive_and_wait();  // everyone sees the new barrier
        tl_blockIdx = dim3(b % grid.x, (b / grid.x) % grid.y, b / (grid.x * grid.y));
        body();
        // Let a thread that returned early stop counting toward barriers (so a
        // buggy kernel reports an error instead of deadlocking), and record
        // the exit so __syncthreads can flag the divergence.
        ++g_exited;
        g_barrier->arrive_and_drop();
        g_warp_barriers[t / 32]->arrive_and_drop();
        phase.arrive_and_wait();  // block finished everywhere
        if (t == 0) {
          delete g_barrier;
          for (auto* w : g_warp_barriers) delete w;
          g_warp_barriers.clear();
        }
      }
    });
  }
  for (auto& th : ts) th.join();
}
}  // namespace emu

#define threadIdx (::emu::tl_threadIdx)
#define blockIdx (::emu::tl_blockIdx)
#define blockDim (::emu::g_blockDim)
#define gridDim (::emu::g_gridDim)
inline void __syncwarp(unsigned /*mask*/ = 0xffffffffu) {
  ::emu::g_warp_barriers[::emu::tl_linear / 32]->arrive_and_wait();
}
inline void __syncthreads() {
  if (::emu::g_exited.load() > 0) ::emu::g_divergent_barrier = true;
  ::emu::g_barrier->arrive_and_wait();
}
