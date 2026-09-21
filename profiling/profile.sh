#!/usr/bin/env bash
# Profile one kernel at one size with Nsight Compute.
#
#   profiling/profile.sh <kernel|cublas> [size=4096]
#
# Writes to profiling/reports/:
#   <kernel>_<size>.ncu-rep        full report (--set full, source-correlated).
#                                  Open in the ncu GUI; this is the evidence.
#   <kernel>_<size>.metrics.csv    compact metric set from profiling/metrics.txt,
#                                  the numbers that go into WRITEUP tables.
#
# Env:
#   BUILD=build                    build dir
#   NCU=ncu                        ncu binary
#   NCU_CLOCK_CONTROL=base|none    ncu LOCKS SM CLOCKS TO BASE by default, so its
#                                  durations are NOT comparable with sgemm_bench
#                                  numbers at your locked clock. Ratios, %-of-peak,
#                                  sector counts and stall mixes are what to
#                                  quote from ncu; GFLOPS comes from the bench.
#                                  Use "none" if you already locked clocks and
#                                  want ncu to respect them.
set -euo pipefail

KERNEL=${1:?usage: profile.sh <kernel|cublas> [size]}
SIZE=${2:-4096}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${BUILD:-$ROOT/build}
NCU=${NCU:-ncu}
CLOCK=${NCU_CLOCK_CONTROL:-base}
OUT=$ROOT/profiling/reports
mkdir -p "$OUT"

# Our kernels are named sgemm_NN_*; cuBLAS kernels contain "gemm". Profile every
# matching launch (cuBLAS may split into gemm + reduction kernels).
if [[ $KERNEL == cublas ]]; then FILTER='regex:gemm'; COUNT=4; else FILTER='regex:sgemm_[0-9]'; COUNT=1; fi

BENCH=("$BUILD/sgemm_bench" --profile --kernel "$KERNEL" --sizes "$SIZE")
METRICS=$(grep -v '^\s*#' "$ROOT/profiling/metrics.txt" | grep -v '^\s*$' | paste -sd, -)
TAG="${KERNEL}_${SIZE}"

echo "== full report -> $OUT/$TAG.ncu-rep"
"$NCU" --set full --import-source yes --clock-control "$CLOCK" \
  -k "$FILTER" -c "$COUNT" -f -o "$OUT/$TAG" "${BENCH[@]}" > /dev/null

echo "== compact metrics -> $OUT/$TAG.metrics.csv"
"$NCU" --metrics "$METRICS" --clock-control "$CLOCK" --csv --print-units base \
  -k "$FILTER" -c "$COUNT" "${BENCH[@]}" | grep -E '^"' > "$OUT/$TAG.metrics.csv"

python3 "$ROOT/profiling/summarize.py" "$OUT/$TAG.metrics.csv"
