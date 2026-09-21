#!/usr/bin/env python3
"""Plots + README table from sgemm_bench CSVs.

    python3 scripts/plot.py results/<run>.csv [more.csv ...] [--size 4096] [--out results]

Produces
  <out>/plots/gflops_vs_size.png        square sweep, one line per kernel + cuBLAS
  <out>/plots/stage_progression_<N>.png GFLOPS per stage at N^3 with % of cuBLAS
  <out>/results_table.md                markdown table for README/WRITEUP

Rows with verified == 0 are dropped with a warning: a wrong kernel has no speed.
If several CSVs contain the same (kernel, M, N, K), the later file wins, so
pass files oldest -> newest.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed categorical order, keyed by STAGE so a kernel keeps its color across
# every plot and every run (color follows the entity, never its rank).
STAGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CUBLAS_COLOR = "#6b6a63"  # neutral: the reference, not a competitor series
INK, INK_2, GRID = "#1a1a19", "#5c5b55", "#e4e3dc"


def read_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            k, _, v = line[1:].strip().partition("=")
            meta[k] = v
    return meta


def load(paths):
    frames, meta = [], {}
    for p in paths:
        meta = read_meta(p) or meta
        df = pd.read_csv(p, comment="#")
        df["source"] = str(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["kernel", "M", "N", "K"], keep="last")
    bad = df[df.verified == 0]
    if len(bad):
        print(f"WARNING: dropping {len(bad)} rows that FAILED verification:", file=sys.stderr)
        print(bad[["kernel", "M", "N", "K"]].to_string(index=False), file=sys.stderr)
    return df[df.verified != 0].copy(), meta


def color_for(kernel, stage):
    return CUBLAS_COLOR if kernel == "cublas" else STAGE_COLORS[(int(stage) - 1) % len(STAGE_COLORS)]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title_suffix(meta):
    clk = meta.get("peak_clock_mhz", "?")
    src = meta.get("peak_clock_source", "")
    locked = "locked" if src == "user_locked" else "UNLOCKED (boost)"
    return f"{meta.get('device', '?')} @ {clk} MHz {locked}, cuBLAS {meta.get('cublas_math', '?')}"


def plot_vs_size(df, meta, out):
    sq = df[(df.M == df.N) & (df.N == df.K)]
    if sq.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    kernels = sq.sort_values("stage")[["kernel", "stage"]].drop_duplicates()
    for _, r in kernels.iterrows():
        d = sq[sq.kernel == r.kernel].sort_values("M")
        ax.plot(d.M, d.gflops_median, color=color_for(r.kernel, r.stage), linewidth=2,
                marker="o", markersize=5, linestyle="--" if r.kernel == "cublas" else "-",
                label=r.kernel if r.kernel == "cublas" else f"{int(r.stage)}. {r.kernel}")
    peak = float(meta.get("peak_fp32_gflops", 0) or 0)
    if peak > 0:
        ax.axhline(peak, color=INK_2, linewidth=1, linestyle=":")
        ax.annotate(f"FP32 peak {peak / 1000:.1f} TFLOP/s", (sq.M.min(), peak),
                    textcoords="offset points", xytext=(2, 4), color=INK_2, fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(sq.M.unique()))
    ax.set_xticklabels([str(v) for v in sorted(sq.M.unique())])
    ax.set_xlabel("M = N = K", color=INK)
    ax.set_ylabel("GFLOP/s (median)", color=INK)
    ax.set_title(f"SGEMM throughput vs size\n{title_suffix(meta)}", color=INK, fontsize=10, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out / "gflops_vs_size.png", dpi=160)
    plt.close(fig)


def plot_progression(df, meta, out, n):
    d = df[(df.M == n) & (df.N == n) & (df.K == n)].sort_values("stage")
    if d.empty:
        print(f"no rows at {n}^3; skipping progression plot", file=sys.stderr)
        return
    # cuBLAS last so it sits at the bottom as the reference bar.
    d = pd.concat([d[d.kernel != "cublas"], d[d.kernel == "cublas"]])
    labels = [k if k == "cublas" else f"{int(s)}. {k}" for k, s in zip(d.kernel, d.stage)]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.4))
    y = range(len(d))[::-1]
    ax.barh(list(y), d.gflops_median, color=[color_for(k, s) for k, s in zip(d.kernel, d.stage)],
            height=0.6, edgecolor="white", linewidth=2)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, color=INK)
    xmax = d.gflops_median.max()
    for yi, (_, r) in zip(y, d.iterrows()):
        txt = f"{r.gflops_median:,.0f}"
        if r.kernel != "cublas":
            txt += f"  ({r.pct_cublas:.1f}% of cuBLAS)"
        ax.text(r.gflops_median + xmax * 0.01, yi, txt, va="center", fontsize=8, color=INK_2)
    ax.set_xlim(0, xmax * 1.35)
    ax.set_xlabel("GFLOP/s (median)", color=INK)
    ax.set_title(f"Optimization progression at {n}x{n}x{n}\n{title_suffix(meta)}",
                 color=INK, fontsize=10, loc="left")
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / f"stage_progression_{n}.png", dpi=160)
    plt.close(fig)


def table(df, meta, out, n):
    d = df[(df.M == n) & (df.N == n) & (df.K == n)].sort_values("stage")
    lines = [f"Results at {n}x{n}x{n}. {title_suffix(meta)}, "
             f"flush_l2={meta.get('flush_l2', '?')}, iters={meta.get('iters', '?')}, git {meta.get('git_rev', '?')}.",
             "",
             "| Stage | Kernel | GFLOP/s (median) | ± std | % cuBLAS | % FP32 peak | max err (eps) |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for _, r in d.iterrows():
        std_gf = r.gflops_median * (r.ms_std / r.ms_median) if r.ms_median > 0 else 0
        pc = "—" if r.kernel == "cublas" else f"{r.pct_cublas:.1f}%"
        err = "ref" if r.kernel == "cublas" else f"{r.max_err_eps:.1f}"
        lines.append(f"| {int(r.stage)} | {r.kernel} | {r.gflops_median:,.0f} | {std_gf:,.0f} | {pc} | "
                     f"{r.pct_peak:.1f}% | {err} |")
    (out / "results_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    (a.out / "plots").mkdir(parents=True, exist_ok=True)
    df, meta = load(a.csv)
    plot_vs_size(df, meta, a.out / "plots")
    plot_progression(df, meta, a.out / "plots", a.size)
    table(df, meta, a.out, a.size)


if __name__ == "__main__":
    main()
