#!/usr/bin/env bash
# First-10-minutes checklist for a fresh cloud GPU instance. Run this BEFORE
# spending paid minutes on anything else:
#
#   scripts/env_check.sh
#
# Stops at the first hard blocker. Order matters: counters first, because a
# box whose perf counters are blocked is useless for this project no matter how
# fast it is.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${BUILD:-$ROOT/build}
NCU=${NCU:-ncu}
pass() { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; }
die()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; exit 1; }

echo "[0] toolchain present (don't pay GPU minutes to install it)"
command -v nvidia-smi >/dev/null || die "nvidia-smi missing"
command -v nvcc >/dev/null || die "nvcc missing: pick an image with the CUDA toolkit"
command -v "$NCU" >/dev/null || die "ncu missing: pick an image with Nsight Compute"
command -v cmake >/dev/null || die "cmake missing"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/       /'
nvcc --version | tail -1 | sed 's/^/       /'
"$NCU" --version | tail -1 | sed 's/^/       /'
pass "toolchain"

echo "[1] build"
if [[ ! -x $BUILD/sgemm_bench ]]; then
  CC_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
  cmake -S "$ROOT" -B "$BUILD" -DCMAKE_CUDA_ARCHITECTURES="80;$CC_ARCH" >/dev/null || die "cmake configure"
  cmake --build "$BUILD" -j >/dev/null || die "build"
fi
pass "built in $BUILD"

echo "[2] Nsight Compute can read performance counters"
OUT=$("$NCU" --metrics sm__cycles_elapsed.avg -k 'regex:sgemm_01' -c 1 \
        "$BUILD/sgemm_bench" --profile --kernel naive --sizes 256 2>&1)
if grep -q ERR_NVGPUCTRPERM <<<"$OUT"; then
  die "ERR_NVGPUCTRPERM: counters blocked (common on virtualized hosts). Either run with
       sudo / set NVreg_RestrictProfilingToAdminUsers=0, or move to a bare-metal /
       counter-enabled instance. Profiling is the point of this project; don't proceed."
elif grep -q 'sm__cycles_elapsed.avg' <<<"$OUT"; then
  pass "counters readable"
else
  echo "$OUT" | tail -20; die "ncu did not produce metrics (see output above)"
fi

echo "[3] every metric in profiling/metrics.txt exists on this GPU/ncu"
AVAIL=$("$NCU" --query-metrics 2>/dev/null | awk '{print $1}')
MISSING=0
while read -r m; do
  [[ -z $m || $m == \#* ]] && continue
  base=${m%%.*}
  grep -qx "$base" <<<"$AVAIL" || { warn "not found: $m"; MISSING=1; }
done < "$ROOT/profiling/metrics.txt"
[[ $MISSING == 0 ]] && pass "all metrics present" || warn "fix names above in profiling/metrics.txt (ncu --query-metrics | grep ...)"

echo "[4] clocks"
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.limit --format=csv | sed 's/^/       /'
warn "lock clocks before reportable runs: sudo scripts/lock_clocks.sh <sm_mhz>"

echo "[5] cuBLAS is running FP32 CUDA-core SGEMM (not TF32 tensor cores)"
NAMES=$("$NCU" --metrics gpu__time_duration.sum -k regex:gemm -c 4 \
          "$BUILD/sgemm_bench" --profile --kernel cublas --sizes 4096 2>&1 | grep -oE '[a-zA-Z0-9_]*gemm[a-zA-Z0-9_]*' | sort -u)
echo "$NAMES" | sed 's/^/       /'
if grep -qiE 'tf32|1688|16816|xmma.*tf' <<<"$NAMES"; then
  die "cuBLAS picked a TF32/tensor-core kernel: the FP32 comparison would be invalid"
fi
pass "no TF32 kernel names"

echo "[6] cuBLAS baseline + theoretical peak"
"$BUILD/sgemm_bench" --kernel cublas --sizes quick --warmup 5 --iters 20 2>/dev/null \
  | grep -E '^# (device|peak_|max_sm)|^cublas|^kernel'
pass "record the 4096 cuBLAS GFLOPS above; it is the % of cuBLAS denominator"
echo
echo "Environment OK. Next: sudo scripts/lock_clocks.sh <MHz>; scripts/run_bench.sh"
