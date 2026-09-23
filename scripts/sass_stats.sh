#!/usr/bin/env bash
# Count the instructions that matter for SGEMM in each stage's SASS.
#
#   scripts/sass_stats.sh [arch=sm_80] [pattern=sgemm_]
#
# Why: source-level "loads per FFMA" arguments are routinely wrong because
# ptxas vectorises (LDS -> LDS.128), unrolls, and hoists. The ratio that bounds
# the kernel is the one in the SASS. These counts are static (per kernel body,
# unrolled loops counted once per copy), which is the right thing for comparing
# inner-loop mixes. Dynamic counts come from ncu (smsp__inst_executed_*).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ARCH=${1:-sm_80}
PAT=${2:-sgemm_}
LIB=${BUILD:-$ROOT/build}/libsgemm_kernels.a
cuobjdump -sass -arch "$ARCH" "$LIB" | awk -v pat="$PAT" '
  /Function :/ { name = $3; keep = index(name, pat) > 0; next }
  keep {
    for (i = 1; i <= NF; i++) {
      op = $i; sub(/[;].*/, "", op)
      if (op ~ /^(FFMA|HMMA\.[0-9A-Z.]+|LDS|LDS\.[0-9A-Z.]+|STS|STS\.[0-9A-Z.]+|LDG\.[0-9A-Z.]+|STG\.[0-9A-Z.]+|BAR\.SYNC.*)$/) cnt[name, op]++
      if (op ~ /^(FFMA|HMMA\..*|LDS|LDS\..*|STS|STS\..*|LDG\..*|STG\..*|BAR\..*)$/) break
    }
  }
  END { for (k in cnt) { split(k, p, SUBSEP); printf "%-70s %-16s %5d\n", p[1], p[2], cnt[k] } }' | sort
