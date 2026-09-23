cmake_minimum_required(VERSION 3.24)

# ---------------------------------------------------------------------------
# Architectures
#   80 = A100 (reportable numbers). 86 = default dev-card guess (A10/A40/3090).
#   Override at configure time, e.g. -DCMAKE_CUDA_ARCHITECTURES="80;89" for an
#   L4/4090, or "native" to build only for the GPU in the box.
#   Why it matters: an sm_86 cubin *runs* on sm_89 (same major, higher minor),
#   but ptxas scheduled it for Ampere's pipeline, so dev numbers on Ada would be
#   quietly pessimistic. Build for the card you measure on.
# ---------------------------------------------------------------------------
#   Must be set BEFORE project(): enabling CUDA otherwise fills in CMake's own
#   default (sm_52), and a later NOT DEFINED check never fires.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86" CACHE STRING "CUDA SM targets")
endif()

project(tiled_sgemm LANGUAGES CXX CUDA)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

option(SGEMM_PTXAS_VERBOSE "Print per-kernel register/smem/spill usage from ptxas" OFF)

# ---------------------------------------------------------------------------
# Flags
#   -O3            host + device optimisation (ptxas is already -O3 by default).
#   -lineinfo      maps SASS back to source lines in Nsight Compute's Source view;
#                  zero runtime cost, and it is how we attribute stalls to lines.
#   NO --use_fast_math. fast-math implies -ftz=true (flush denormals),
#                  -prec-div=false, -prec-sqrt=false. SGEMM has no div/sqrt, so
#                  the only effect here would be FTZ, which changes results for
#                  tiny values and buys ~nothing for an FFMA-bound kernel. Keeping
#                  it off keeps the correctness path identical to cuBLAS's IEEE
#                  semantics. (-fmad=true, the default, stays on: contracting
#                  a*b+c into one FFMA is what cuBLAS does and is *more* accurate.)
# ---------------------------------------------------------------------------
set(SGEMM_CUDA_FLAGS -O3 -lineinfo --expt-relaxed-constexpr)
if(SGEMM_PTXAS_VERBOSE)
  list(APPEND SGEMM_CUDA_FLAGS -Xptxas=-v)
endif()
set(SGEMM_CXX_WARN -Wall -Wextra)

# Record the commit in every CSV so a number can always be traced to code.
execute_process(
  COMMAND git describe --always --dirty
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE SGEMM_GIT_REV OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT SGEMM_GIT_REV)
  set(SGEMM_GIT_REV "unknown")
endif()

function(sgemm_target_setup tgt)
  target_include_directories(${tgt} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_options(${tgt} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${SGEMM_CUDA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 ${SGEMM_CXX_WARN}>)
  target_compile_definitions(${tgt} PRIVATE SGEMM_GIT_REV="${SGEMM_GIT_REV}")
endfunction()

# ---------------------------------------------------------------------------
# Kernel library: one file per stage in kernels/, plus the cuBLAS reference,
# the correctness checker and the registry that the drivers iterate over.
# ---------------------------------------------------------------------------
add_library(sgemm_kernels STATIC
  kernels/01_naive.cu
  kernels/02_coalesced.cu
  kernels/03_smem_tiling.cu
  kernels/04_blocktiling_1d.cu
  kernels/05_blocktiling_2d.cu
  kernels/06_vectorized.cu
  kernels/07_warptiling.cu
  src/registry.cu
  src/cublas_ref.cu
  src/check.cu
  src/device_info.cu)
sgemm_target_setup(sgemm_kernels)
target_include_directories(sgemm_kernels PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/kernels)
target_link_libraries(sgemm_kernels PUBLIC CUDA::cudart CUDA::cublas)

add_executable(sgemm_bench bench/bench.cu)
sgemm_target_setup(sgemm_bench)
target_link_libraries(sgemm_bench PRIVATE sgemm_kernels)

add_executable(sgemm_test tests/test_correctness.cu)
sgemm_target_setup(sgemm_test)
target_link_libraries(sgemm_test PRIVATE sgemm_kernels)

# Host-only unit tests for the checker / CPU reference / stats. Runs without a
# GPU, so the numerical plumbing can be validated anywhere.
add_executable(sgemm_host_test tests/test_host.cpp)
sgemm_target_setup(sgemm_host_test)

enable_testing()
add_test(NAME host_utils COMMAND sgemm_host_test)
add_test(NAME correctness COMMAND sgemm_test --kernel all)

# Stage 8: autotuner over template families (bench/autotune.cu).
add_executable(sgemm_autotune bench/autotune.cu)
sgemm_target_setup(sgemm_autotune)
target_link_libraries(sgemm_autotune PRIVATE sgemm_kernels)
