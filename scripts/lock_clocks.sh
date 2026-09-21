#!/usr/bin/env bash
# Lock GPU clocks for reproducible benchmarks.
#
#   sudo scripts/lock_clocks.sh <sm_mhz> [mem_mhz]
#   sudo scripts/lock_clocks.sh --reset
#   scripts/lock_clocks.sh --list          # supported clock pairs
#
# Why: with boost enabled, SM clock floats with temperature and power, so the
# same kernel can differ by 10%+ run to run and the % of peak denominator is
# undefined. Locking makes numbers repeatable and lets you state the clock.
#
# Picking the value: lock at a clock the card can SUSTAIN under SGEMM without
# hitting its power cap. On A100 try the max (1410 MHz) and watch the clock log
# that scripts/run_bench.sh records; if it sags or shows a power throttle
# reason, lock lower (e.g. 1275) and use that everywhere. Pass the same value
# to sgemm_bench --sm-clock-mhz.
#
# Memory clock: on HBM parts (A100) -lmc is typically unsupported and the memory
# clock is fixed anyway. Failure there is reported and ignored.
set -euo pipefail
GPU=${GPU:-0}

case "${1:-}" in
  --list)  nvidia-smi -i "$GPU" -q -d SUPPORTED_CLOCKS | head -40; exit 0 ;;
  --reset) nvidia-smi -i "$GPU" -rgc; nvidia-smi -i "$GPU" -rmc || true; exit 0 ;;
  ''|-h|--help) sed -n 2,20p "$0"; exit 1 ;;
esac

SM=$1
nvidia-smi -i "$GPU" -pm 1 >/dev/null
nvidia-smi -i "$GPU" -lgc "$SM,$SM"
if [[ -n ${2:-} ]]; then
  nvidia-smi -i "$GPU" -lmc "$2,$2" || echo "memory clock lock unsupported here (fine on HBM parts)"
fi
nvidia-smi -i "$GPU" --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv
echo "Now pass --sm-clock-mhz $SM to sgemm_bench (scripts/run_bench.sh reads SM_CLOCK=$SM)."
