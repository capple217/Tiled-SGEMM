#!/usr/bin/env python3
"""Plots + README tables from sgemm_bench CSVs.

    python3 scripts/plot.py results/<run>.csv [more.csv ...] [--size 4096] [--out results]

Produces, separately for each arithmetic (fp32 = CUDA cores, tf32 = tensor cores;
they are never drawn on the same axes, because a TF32 kernel over an FP32
baseline is not a fair comparison):
  <out>/plots/gflops_vs_size[_tf32].png        square sweep, one line per kernel + its cuBLAS
  <out>/plots/stage_progression[_tf32]_<N>.png GFLOPS per stage at N^3 with % of cuBLAS
  <out>/results_table.md                       markdown tables for README/WRITEUP

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
SUFFIX = {"fp32": "", "tf32": "_tf32"}


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
        if "precision" not in df.columns:  # CSVs from before stage 9
            df["precision"] = "fp32"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["kernel", "M", "N", "K"], keep="last")
    bad = df[df.verified == 0]
    if len(bad):
        print(f"WARNING: dropping {len(bad)} rows that FAILED verification:", file=sys.stderr)
        print(bad[["kernel", "M", "N", "K"]].to_string(index=False), file=sys.stderr)
    return df[df.verified != 0].copy(), meta


def color_for(kernel, stage):
    if kernel.startswith("cublas"):
        return CUBLAS_COLOR
    return STAGE_COLORS[min(int(stage), len(STAGE_COLORS)) - 1]  # stage 9 -> last slot


def label_for(kernel, stage):
    return kernel if kernel.startswith("cublas") else f"{int(stage)}. {kernel}"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title_suffix(meta, prec):
    clk = meta.get("peak_clock_mhz", "?")
    locked = "locked" if meta.get("peak_clock_source") == "user_locked" else "UNLOCKED (boost)"
    base = "cuBLAS TF32 tensor-op" if prec == "tf32" else f"cuBLAS {meta.get('cublas_math', '?')}"
    return f"{meta.get('device', '?')} @ {clk} MHz {locked}, {base}"


def plot_vs_size(df, meta, out, prec):
    sq = df[(df.M == df.N) & (df.N == df.K) & (df.precision == prec)]
    if sq.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for _, r in sq.sort_values("stage")[["kernel", "stage"]].drop_duplicates().iterrows():
        d = sq[sq.kernel == r.kernel].sort_values("M")
        ax.plot(d.M, d.gflops_median, color=color_for(r.kernel, r.stage), linewidth=2,
                marker="o", markersize=5,
                linestyle="--" if r.kernel.startswith("cublas") else "-",
                label=label_for(r.kernel, r.stage))
    peak = float(meta.get(f"peak_{prec}_gflops", 0) or 0)
    if peak > 0:
        ax.axhline(peak, color=INK_2, linewidth=1, linestyle=":")
        ax.annotate(f"{prec.upper()} peak {peak / 1000:.1f} TFLOP/s", (sq.M.min(), peak),
                    textcoords="offset points", xytext=(2, 4), color=INK_2, fontsize=8)
    ax.set_xscale("log", base=2)
    ticks = sorted(sq.M.unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(v) for v in ticks])
    ax.set_xlabel("M = N = K", color=INK)
    ax.set_ylabel("GFLOP/s (median)", color=INK)
    ax.set_title(f"SGEMM throughput vs size ({prec.upper()})\n{title_suffix(meta, prec)}",
                 color=INK, fontsize=10, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out / f"gflops_vs_size{SUFFIX[prec]}.png", dpi=160)
    plt.close(fig)


def plot_progression(df, meta, out, n, prec):
    d = df[(df.M == n) & (df.N == n) & (df.K == n) & (df.precision == prec)].sort_values("stage")
    if d.empty:
        return
    # cuBLAS last so it sits at the bottom as the reference bar.
    d = pd.concat([d[~d.kernel.str.startswith("cublas")], d[d.kernel.str.startswith("cublas")]])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.4))
    y = list(range(len(d)))[::-1]
    ax.barh(y, d.gflops_median, color=[color_for(k, s) for k, s in zip(d.kernel, d.stage)],
            height=0.6, edgecolor="white", linewidth=2)
    ax.set_yticks(y)
    ax.set_yticklabels([label_for(k, s) for k, s in zip(d.kernel, d.stage)], color=INK)
    xmax = d.gflops_median.max()
    for yi, (_, r) in zip(y, d.iterrows()):
        txt = f"{r.gflops_median:,.0f}"
        if not r.kernel.startswith("cublas"):
            txt += f"  ({r.pct_cublas:.1f}% of cuBLAS)"
        ax.text(r.gflops_median + xmax * 0.01, yi, txt, va="center", fontsize=8, color=INK_2)
    ax.set_xlim(0, xmax * 1.35)
    ax.set_xlabel("GFLOP/s (median)", color=INK)
    ax.set_title(f"Optimization progression at {n}x{n}x{n} ({prec.upper()})\n{title_suffix(meta, prec)}",
                 color=INK, fontsize=10, loc="left")
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / f"stage_progression{SUFFIX[prec]}_{n}.png", dpi=160)
    plt.close(fig)


def table(df, meta, n):
    lines = [f"Results at {n}x{n}x{n}. flush_l2={meta.get('flush_l2', '?')}, "
             f"iters={meta.get('iters', '?')}, git {meta.get('git_rev', '?')}."]
    for prec in ("fp32", "tf32"):
        d = df[(df.M == n) & (df.N == n) & (df.K == n) & (df.precision == prec)].sort_values("stage")
        if d.empty:
            continue
        lines += ["", f"**{prec.upper()}**: {title_suffix(meta, prec)}", "",
                  "| Stage | Kernel | GFLOP/s (median) | ± std | % cuBLAS | % peak | max err (eps) |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
        for _, r in d.iterrows():
            std_gf = r.gflops_median * (r.ms_std / r.ms_median) if r.ms_median > 0 else 0
            is_ref = r.kernel == "cublas"
            pc = "—" if r.kernel.startswith("cublas") else f"{r.pct_cublas:.1f}%"
            err = "ref" if is_ref else f"{r.max_err_eps:.1f}"
            lines.append(f"| {int(r.stage)} | {r.kernel} | {r.gflops_median:,.0f} | {std_gf:,.0f} | "
                         f"{pc} | {r.pct_peak:.1f}% | {err} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    (a.out / "plots").mkdir(parents=True, exist_ok=True)
    df, meta = load(a.csv)
    for prec in ("fp32", "tf32"):
        plot_vs_size(df, meta, a.out / "plots", prec)
        plot_progression(df, meta, a.out / "plots", a.size, prec)
    text = table(df, meta, a.size)
    (a.out / "results_table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
