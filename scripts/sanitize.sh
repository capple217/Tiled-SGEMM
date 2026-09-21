#!/usr/bin/env bash
# Run the correctness suite under compute-sanitizer on a GPU. The GPU-side
# counterpart of tests/emu (which catches these on CPU with ASan):
#   memcheck   out-of-bounds / misaligned global + shared accesses. Catches the
#              "numerically invisible" OOB read, where garbage x zero-filled tile = 0.
#   racecheck  shared-memory hazards, e.g. a missing __syncthreads between the
#              compute loop and the next tile's STS (the WAR barrier).
#   synccheck  __syncthreads in divergent code (e.g. an early `return` added
#              to a tiled kernel).
#   initcheck  reads of uninitialised global memory (e.g. reading C when beta == 0).
# Slow (10-100x), so it runs the --quick test set with one kernel at a time.
#
#   scripts/sanitize.sh [kernel=all]
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${BUILD:-$ROOT/build}
K=${1:-all}
STATUS=0
for tool in memcheck racecheck synccheck initcheck; do
  echo "== $tool"
  compute-sanitizer --tool "$tool" --error-exitcode 99 \
    "$BUILD/sgemm_test" --kernel "$K" --quick > "/tmp/sanitize_$tool.log" 2>&1
  rc=$?
  if [[ $rc == 0 ]]; then echo "   clean"; else echo "   ERRORS (rc=$rc), see /tmp/sanitize_$tool.log"; grep -m5 -E "=========.*(Invalid|Race|Barrier|Uninitialized)" "/tmp/sanitize_$tool.log"; STATUS=1; fi
done
exit $STATUS
