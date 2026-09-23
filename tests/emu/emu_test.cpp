// Kernel-logic tests on the CPU emulator (see cuda_runtime.h in this directory).
// Every registered kernel, small ragged shapes, both epilogue cases, vs the
// double-precision CPU reference. No GPU required: runs on a laptop.

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "sgemm/host_utils.hpp"
#include "sgemm/kernels.hpp"

using namespace sgemm;

namespace sgemm {
std::vector<KernelSpec> emu_variants();  // tests/emu/variants.cpp
}

int main(int argc, char** argv) {
  const char* only = argc > 1 ? argv[1] : nullptr;
  struct Shape { int M, N, K; };
  // Ragged around 8/32/64/128 (every tile edge in use), 1-sized dims, K < BK.
  const Shape shapes[] = {{1, 1, 1},     {7, 13, 5},    {31, 33, 17},  {33, 31, 65},
                          {64, 64, 64},  {63, 65, 3},   {127, 129, 65}, {128, 128, 8},
                          {129, 131, 1}, {65, 200, 33}, {1, 300, 20},  {300, 1, 20},
                          {136, 72, 12}};
  const float cases[][2] = {{1.0f, 0.0f}, {1.5f, -0.75f}};
  int fails = 0, runs = 0;

  // Registered defaults first, then non-default template configurations.
  std::vector<KernelSpec> specs = all_kernels();
  for (const KernelSpec& v : emu_variants()) specs.push_back(v);
  for (const KernelSpec& k : specs) {
    // Filter: exact name, or "vNN" prefix match for variants of a stage.
    if (only && std::strcmp(only, k.name) != 0 && std::strncmp(only, k.name, std::strlen(only)) != 0)
      continue;
    int kfails = 0;
    double worst = 0;
    for (const Shape& s : shapes) {
      std::vector<float> A(size_t(s.M) * s.K), B(size_t(s.K) * s.N), C0(size_t(s.M) * s.N);
      fill_uniform(A.data(), A.size(), 11);
      fill_uniform(B.data(), B.size(), 12);
      fill_uniform(C0.data(), C0.size(), 13);
      for (const auto& c : cases) {
        std::vector<float> C(C0.size()), ref(C0.size()), scale(C0.size());
        if (c[1] != 0.0f) C = C0;
        else std::fill(C.begin(), C.end(), NAN);  // beta == 0: C must not be read
        ::emu::g_divergent_barrier = false;
        k.launch(s.M, s.N, s.K, c[0], A.data(), B.data(), c[1], C.data(), nullptr);
        cpu_reference(s.M, s.N, s.K, c[0], A.data(), B.data(), c[1], C0.data(), ref.data(),
                      scale.data());
        const CheckResult r = compare(C.data(), ref.data(), scale.data(), C.size(), s.K, k.precision);
        ++runs;
        worst = std::max(worst, r.max_err / FLT_EPSILON);
        const bool div = ::emu::g_divergent_barrier;
        if (!r.pass || div) {
          ++fails;
          ++kfails;
          std::printf("  FAIL %-24s %4dx%4dx%4d a=%g b=%g: %lld bad (worst idx %lld got %g want %g)%s\n",
                      k.name, s.M, s.N, s.K, c[0], c[1], r.num_bad, r.worst_idx, r.got_worst,
                      r.ref_worst, div ? " [barrier after thread exit]" : "");
        }
      }
    }
    std::printf("%-24s stage %d %s: %s (max err %.2f eps)\n", k.name, k.stage, precision_name(k.precision),
                kfails ? "FAILED" : "ok", worst);
  }
  std::printf("%d runs, %d failures\n", runs, fails);
  return fails ? 1 : 0;
}
