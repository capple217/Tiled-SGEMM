#!/usr/bin/env python3
"""Assemble a run directory (from scripts/full_run.sh) into RUN_SUMMARY.md.

    python3 scripts/collate.py results/run_<stamp>_<gpu> [--size 4096]

Sections:
  1. Environment: device, clocks (locked value vs what the clock log observed),
     versions, methodology flags.
  2. Results tables: FP32 and TF32 separately (from scripts/plot.py).
  3. Attribution table: every stage side by side on the profiler ratios that
     explain its speed. This is the evidence the WRITEUP quotes, one column per
     stage so each change's effect is a left-to-right read.
  4. Autotuning: per family, the best configs ranked on tune sizes and their
     held-out (eval) GFLOPS vs the default config on the same sizes.
  5. Where the raw evidence lives (.ncu-rep files for the GUI).
"""
import argparse
import csv
import importlib.util
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _import(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plot = _import(ROOT / "scripts" / "plot.py", "plot")
summ = _import(ROOT / "profiling" / "summarize.py", "summarize")

# Rows of the attribution table: (label in summarize.derived, short label).
ATTR_ROWS = [
    ("FMA pipe active %", "FMA pipe %"),
    ("tensor pipe active %", "tensor pipe %"),
    ("issue slots busy %", "issue busy %"),
    ("DRAM throughput %", "DRAM %"),
    ("L1/smem throughput %", "L1/smem %"),
    ("L1 hit %", "L1 hit %"),
    ("gld sectors/request", "gld sectors/req"),
    ("lds wavefronts/inst", "lds wavefronts/inst"),
    ("sts bank conflicts", "sts bank conflicts"),
    ("FFMA / all warp-insts", "FFMA share of insts"),
    ("FLOP / DRAM byte", "FLOP/DRAM byte"),
    ("achieved occupancy %", "occupancy %"),
    ("registers/thread", "regs/thread"),
]


def clock_observed(path):
    """(min, max) SM clock in MHz from the nvidia-smi log, or None."""
    try:
        vals = []
        with open(path) as f:
            for row in csv.reader(f):
                if len(row) > 1:
                    try:  # data rows look like " 1410 MHz"; the header doesn't parse
                        vals.append(float(row[1].replace("MHz", "").strip()))
                    except ValueError:
                        pass
        return (min(vals), max(vals)) if vals else None
    except OSError:
        return None


def section_env(meta, run):
    obs = clock_observed(run / "clocks.csv")
    lines = ["## 1. Environment", "",
             f"- **Device:** {meta.get('device', '?')} (cc {meta.get('compute_capability', '?')}, "
             f"{meta.get('sm_count', '?')} SMs)",
             f"- **SM clock for peaks:** {meta.get('peak_clock_mhz', '?')} MHz "
             f"({meta.get('peak_clock_source', '?')})"]
    if obs:
        held = "held" if obs[0] == obs[1] else "DID NOT HOLD. Treat numbers with care"
        lines.append(f"- **Observed SM clock during the sweep:** {obs[0]:.0f}–{obs[1]:.0f} MHz ({held})")
    lines += [f"- **Peaks:** FP32 {float(meta.get('peak_fp32_gflops', 0)) / 1e3:.1f} TFLOP/s, "
              f"TF32 {float(meta.get('peak_tf32_gflops', 0)) / 1e3:.1f} TFLOP/s "
              f"({meta.get('tf32_peak_source', '?')})",
              f"- **Versions:** driver {meta.get('driver_version', '?')}, runtime "
              f"{meta.get('runtime_version', '?')}, cuBLAS {meta.get('cublas_version', '?')}, "
              f"git {meta.get('git_rev', '?')}",
              f"- **Method:** warmup {meta.get('warmup', '?')}, up to {meta.get('iters', '?')} timed "
              f"iterations (median), L2 flush {meta.get('flush_l2', '?')}, α={meta.get('alpha', '?')}, "
              f"β={meta.get('beta', '?')}, FLOPs = 2MNK", ""]
    return lines


def section_attribution(run, bench_df, n):
    prof = run / "profiles"
    cols = []  # (column label, metrics dict)
    order = [("cublas", 0), ("cublas_tf32", 0)]
    order += sorted({(k, s) for k, s in zip(bench_df.kernel, bench_df.stage)
                     if not k.startswith("cublas")}, key=lambda x: x[1])
    for kernel, stage in order:
        f = prof / f"{kernel}_{n}.metrics.csv"
        if not f.exists():
            continue
        launches = summ.load(f)
        if not launches:
            continue
        # cuBLAS may launch several kernels: take the longest (the GEMM itself).
        m = max(launches.values(), key=lambda d: d.get("gpu__time_duration.sum", 0))
        label = kernel if kernel.startswith("cublas") else f"{stage}. {kernel}"
        cols.append((kernel, label, m))
    if not cols:
        return ["## 3. Attribution table", "", "_No profiler captures found._", ""]
    gf = {k: g for k, g, M, N, K in zip(bench_df.kernel, bench_df.gflops_median, bench_df.M,
                                         bench_df.N, bench_df.K) if M == N == K == n}
    lines = ["## 3. Attribution table", "",
             f"Profiler ratios at {n}³ (ncu, base clocks: compare ratios, not durations). "
             "GFLOP/s is from the benchmark at the locked clock.", "",
             "| metric | " + " | ".join(c[1] for c in cols) + " |",
             "|---|" + "---:|" * len(cols),
             "| **GFLOP/s** | " + " | ".join(f"{gf.get(c[0], float('nan')):,.0f}" for c in cols) + " |"]
    for key, short in ATTR_ROWS:
        vals = []
        for _, _, m in cols:
            v = dict(summ.derived(m)).get(key, float("nan"))
            vals.append("—" if isinstance(v, float) and math.isnan(v) else summ.fmt(v))
        lines.append(f"| {short} | " + " | ".join(vals) + " |")
    lines.append("| top stalls | " + " | ".join(
        "<br>".join(f"{k} {v:.0f}" for k, v in summ.top_stalls(m, 3)) for _, _, m in cols) + " |")
    lines.append("")
    return lines


def section_autotune(run):
    files = sorted(run.glob("autotune_*.csv"))
    if not files:
        return []
    lines = ["## 4. Autotuning (ranked on tune sizes, reported on held-out eval sizes)", "",
             "| family | config | tune geomean | eval geomean | vs default (eval) |",
             "|---|---|---:|---:|---:|"]
    for f in files:
        fam = f.stem.replace("autotune_", "")
        tune, ev, dflt = defaultdict(list), defaultdict(list), None
        with open(f) as fh:
            for r in csv.DictReader(fh):
                g = float(r["gflops"])
                if r["phase"] == "tune":
                    tune[r["config"]].append(g)
                else:
                    ev[r["config"]].append(g)
                    if r["phase"] == "eval_default":
                        dflt = r["config"]

        def gm(v):
            v = [x for x in v if x > 0]
            return math.exp(sum(map(math.log, v)) / len(v)) if v else 0.0
        base = gm(ev[dflt]) if dflt else 0.0
        for cfg in sorted(ev, key=lambda c: -gm(tune[c])):
            e = gm(ev[cfg])
            rel = f"{100 * (e / base - 1):+.1f}%" if base > 0 and cfg != dflt else ("default" if cfg == dflt else "—")
            lines.append(f"| {fam} | {cfg} | {gm(tune[cfg]):,.0f} | {e:,.0f} | {rel} |")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--size", type=int, default=4096)
    a = ap.parse_args()
    bench_csv = a.run / "bench.csv"
    if not bench_csv.exists():
        sys.exit(f"{bench_csv} not found")
    df, meta = plot.load([bench_csv])
    out = [f"# Run summary: {meta.get('device', '?')}, {meta.get('timestamp', '?')}", ""]
    out += section_env(meta, a.run)
    out += ["## 2. Results", "", plot.table(df, meta, a.size), ""]
    out += section_attribution(a.run, df, a.size)
    out += section_autotune(a.run)
    reps = sorted((a.run / "profiles").glob("*.ncu-rep"))
    if reps:
        out += ["## 5. Raw evidence", "", "Open in Nsight Compute (`ncu-ui`):", ""]
        out += [f"- `{p.relative_to(a.run)}`" for p in reps]
        out.append("")
    (a.run / "RUN_SUMMARY.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
