#!/usr/bin/env bash
# Reportable benchmark run: correctness gate, then the sweep, with a clock /
# power / throttle log recorded alongside so a reviewer can check the clocks
# actually held.
#
#   SM_CLOCK=1410 scripts/run_bench.sh [preset=all] [extra sgemm_bench args...]
#
# Output: results/<date>_<gpu>_<preset>.csv  and  ...clocks.csv
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${BUILD:-$ROOT/build}
PRESET=${1:-all}; shift || true
: "${SM_CLOCK:?set SM_CLOCK to the locked SM clock in MHz (scripts/lock_clocks.sh)}"

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_' | tr -cd '[:alnum:]_-')
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/results/${STAMP}_${GPU}_${PRESET}"
mkdir -p "$ROOT/results"

echo "== correctness gate"
"$BUILD/sgemm_test" --kernel all --quick

echo "== clock log -> $OUT.clocks.csv"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active \
  --format=csv -lms 500 > "$OUT.clocks.csv" &
LOGPID=$!
trap 'kill $LOGPID 2>/dev/null || true' EXIT

echo "== sweep ($PRESET)"
"$BUILD/sgemm_bench" --kernel all --sizes "$PRESET" --sm-clock-mhz "$SM_CLOCK" --csv "$OUT.csv" "$@"

kill $LOGPID 2>/dev/null || true
# Did the SM clock hold? Report min/max observed (MHz) over the run.
awk -F', ' 'NR>1 {gsub(/ MHz/,"",$2); v=$2+0; if(min==""||v<min)min=v; if(v>max)max=v}
            END {printf "observed SM clock during run: min %s / max %s MHz (locked at '"$SM_CLOCK"')\n", min, max}' "$OUT.clocks.csv"
echo "== plots"
python3 "$ROOT/scripts/plot.py" "$OUT.csv" --out "$ROOT/results"
