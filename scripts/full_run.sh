#!/usr/bin/env bash
# The whole GPU pipeline in one command, into one self-contained run directory.
#
#   SM_CLOCK=1410 scripts/full_run.sh            # reportable: clocks locked to 1410
#   UNLOCKED=1 scripts/full_run.sh               # dev card / no sudo: numbers flagged unlocked
#
# Optional env: SIZE=4096 (profiling/attribution size), SKIP_SANITIZE=1,
#   SKIP_AUTOTUNE=1, SKIP_PROFILE=1, CASE_SECONDS=8 (bench time cap per case),
#   PRESET=all (bench size preset), BUILD=build.
#
# Produces results/run_<stamp>_<gpu>/:
#   env_check.log, tests.log, sanitize.log
#   bench.csv, clocks.csv                   the sweep + a clock/power/throttle log
#   autotune_<family>.csv, autotune_<family>.log
#   profiles/<kernel>_<SIZE>.{ncu-rep,metrics.csv,metrics.summary.md}
#   plots/*.png, results_table.md
#   RUN_SUMMARY.md                          environment, results, attribution table, autotune
#
# Rough cost on an A100: 30-50 min, most of it sanitizers and ncu --set full.
# Order is deliberate. Cheap gates run first (counters, correctness), so a broken
# box or a wrong kernel costs minutes, not the whole budget.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${BUILD:-$ROOT/build}
SIZE=${SIZE:-4096}
PRESET=${PRESET:-all}
CASE_SECONDS=${CASE_SECONDS:-8}
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\033[31mFAIL: %s\033[0m\n' "$*"; exit 1; }

if [[ -z ${SM_CLOCK:-} && -z ${UNLOCKED:-} ]]; then
  die "set SM_CLOCK=<MHz> for a reportable run (clocks get locked), or UNLOCKED=1 for a dev run"
fi

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_' | tr -cd '[:alnum:]_-')
RUN="$ROOT/results/run_$(date +%Y%m%d-%H%M%S)_${GPU}"
mkdir -p "$RUN/profiles" "$RUN/plots"
echo "run directory: $RUN"

say "1/8 environment check (counters, toolchain, cuBLAS FP32, peaks)"
BUILD=$BUILD "$ROOT/scripts/env_check.sh" 2>&1 | tee "$RUN/env_check.log"
[[ ${PIPESTATUS[0]} == 0 ]] || die "environment check failed (see env_check.log)"

say "2/8 build"
CC_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
ARCHS=$([[ $CC_ARCH == 80 ]] && echo 80 || echo "80;$CC_ARCH")
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" >/dev/null &&
  cmake --build "$BUILD" -j >/dev/null || die "build failed"

say "3/8 correctness (host numerics, then every kernel incl. ragged + large shapes)"
{ "$BUILD/sgemm_host_test" && "$BUILD/sgemm_test" --kernel all; } 2>&1 | tee "$RUN/tests.log" | tail -5
[[ ${PIPESTATUS[0]} == 0 ]] || die "correctness failed. Nothing after this would be meaningful (see tests.log)"

if [[ -z ${SKIP_SANITIZE:-} ]]; then
  say "4/8 compute-sanitizer (memcheck, racecheck, synccheck, initcheck)"
  BUILD=$BUILD "$ROOT/scripts/sanitize.sh" all 2>&1 | tee "$RUN/sanitize.log"
  [[ ${PIPESTATUS[0]} == 0 ]] || echo "WARNING: sanitizer findings (see sanitize.log). Continuing, but fix before reporting."
fi

LOCKED_HERE=""
CLOCK_ARGS=()
if [[ -n ${SM_CLOCK:-} ]]; then
  say "5/8 lock clocks at $SM_CLOCK MHz"
  if sudo -n true 2>/dev/null; then
    sudo "$ROOT/scripts/lock_clocks.sh" "$SM_CLOCK" && LOCKED_HERE=1
  else
    echo "no passwordless sudo: assuming you already ran  sudo scripts/lock_clocks.sh $SM_CLOCK"
  fi
  CLOCK_ARGS=(--sm-clock-mhz "$SM_CLOCK")
else
  say "5/8 clocks NOT locked (UNLOCKED=1): % of peak uses boost clock; not reportable"
fi
trap '[[ -n $LOCKED_HERE ]] && sudo "$ROOT/scripts/lock_clocks.sh" --reset >/dev/null' EXIT

say "6/8 benchmark sweep ($PRESET), with clock log"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active \
  --format=csv -lms 500 > "$RUN/clocks.csv" &
LOGPID=$!
"$BUILD/sgemm_bench" --kernel all --sizes "$PRESET" "${CLOCK_ARGS[@]}" \
  --max-seconds-per-case "$CASE_SECONDS" --csv "$RUN/bench.csv" | tee "$RUN/bench.log" | tail -25
kill $LOGPID 2>/dev/null

if [[ -z ${SKIP_AUTOTUNE:-} ]]; then
  say "7/8 autotune (rank on 2048,4096; report on held-out 3072 and 4097x4093x4099)"
  for fam in blocktiling1d blocktiling2d vectorized warptiling tf32_wmma; do
    echo "-- $fam"
    "$BUILD/sgemm_autotune" --family "$fam" --csv "$RUN/autotune_$fam.csv" \
      > "$RUN/autotune_$fam.log" 2>&1 && grep -A6 "held-out" "$RUN/autotune_$fam.log"
  done
fi

if [[ -z ${SKIP_PROFILE:-} ]]; then
  say "8/8 profile every stage + both cuBLAS baselines at ${SIZE}^3"
  KERNELS=$("$BUILD/sgemm_bench" --list | awk '{print $1}')
  for k in cublas cublas_tf32 $KERNELS; do
    echo "-- $k"
    OUT="$RUN/profiles" BUILD=$BUILD "$ROOT/profiling/profile.sh" "$k" "$SIZE" > "$RUN/profiles/$k.log" 2>&1 ||
      echo "   profiling $k failed (see profiles/$k.log)"
  done
fi

say "collate"
python3 "$ROOT/scripts/plot.py" "$RUN/bench.csv" --size "$SIZE" --out "$RUN" > /dev/null
python3 "$ROOT/scripts/collate.py" "$RUN" --size "$SIZE" > /dev/null
echo "done. Read $RUN/RUN_SUMMARY.md first."
