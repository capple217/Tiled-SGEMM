#!/usr/bin/env bash
# Run every compiled-in kernel on the CPU emulator. Needs only g++ >= 11 (C++20
# std::barrier) and python3. No GPU, no CUDA toolkit. Works on macOS with clang++.
#
#   tests/emu/run_emu_tests.sh [kernel-name | variant prefix, e.g. v07]
#
# Kernel list = the kernels/*.cu lines that are NOT commented out in
# CMakeLists.txt, so the emulator always tests exactly what the GPU build does.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/build-emu
CXX=${CXX:-g++}
mkdir -p "$OUT"

SRCS=$(grep -E '^\s*kernels/[0-9]+_\w+\.cu' "$ROOT/CMakeLists.txt" | awk '{print $1}')
OBJS=()
for f in $SRCS; do
  b=$(basename "$f" .cu)
  python3 "$ROOT/tests/emu/transform.py" "$ROOT/$f" "$OUT/$b.emu.cpp"
  [[ -f $ROOT/kernels/$b.cuh ]] && python3 "$ROOT/tests/emu/transform.py" "$ROOT/kernels/$b.cuh" "$OUT/$b.emu.hpp"
  OBJS+=("$OUT/$b.emu.cpp")
done
cp "$ROOT/src/registry.cu" "$OUT/registry.emu.cpp"

# AddressSanitizer + UBSan by default: a read past the end of A/B is often
# numerically invisible (the garbage gets multiplied by a zero-filled tile of
# the other operand), so only a memory checker catches it. SAN= disables.
SAN=${SAN--fsanitize=address,undefined -fno-omit-frame-pointer}
"$CXX" -std=c++20 -O1 -g -pthread -w $SAN \
  -I"$ROOT/tests/emu" -I"$ROOT/include" -I"$OUT" -I"$ROOT/kernels" \
  "${OBJS[@]}" "$OUT/registry.emu.cpp" "$ROOT/tests/emu/variants.cpp" \
  "$ROOT/tests/emu/emu_test.cpp" -o "$OUT/emu_test"
"$OUT/emu_test" "$@"
