#!/usr/bin/env python3
"""Turn an `ncu --metrics ... --csv` dump into the derived numbers the writeup uses.

    python3 profiling/summarize.py profiling/reports/naive_4096.metrics.csv [more ...]

Raw counters are hard to read; the diagnosis lives in ratios:
  gld sectors/req   4 = coalesced 4B/thread, 32 = fully scattered, 1 = broadcast,
                    16 = coalesced float4 (a 512B warp request)
  lds wavefronts/inst  1 = conflict-free 32-bit LDS; n = n-way bank conflict
                    (4 = conflict-free LDS.128, since 512B needs 4 wavefronts)
  FLOP/DRAM byte    achieved arithmetic intensity against DRAM
  top stalls        where warps actually wait
Writes <input>.summary.md next to each input.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path


def num(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return float("nan")


def load(path):
    per_kernel = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("ID", "0"), row.get("Kernel Name", "?"))
            per_kernel[key][row["Metric Name"]] = num(row["Metric Value"])
    return per_kernel


def ratio(a, b):
    return a / b if b and b == b and b != 0 else float("nan")


def summarize(name, m):
    def g(k, d=float("nan")):
        return m.get(k, d)
    out = [f"### {name[:90]}", ""]
    rows = [
        ("duration (us, ncu clocks)", g("gpu__time_duration.sum", float("nan")) / 1e3),
        ("SM throughput %", g("sm__throughput.avg.pct_of_peak_sustained_elapsed")),
        ("FMA pipe active %", g("sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active")),
        ("issue slots busy %", g("smsp__issue_active.avg.pct_of_peak_sustained_active")),
        ("DRAM throughput %", g("dram__throughput.avg.pct_of_peak_sustained_elapsed")),
        ("DRAM MB read", g("dram__bytes_read.sum", float("nan")) / 1e6),
        ("L2 hit %", g("lts__t_sector_hit_rate.pct")),
        ("L1 hit %", g("l1tex__t_sector_hit_rate.pct")),
        ("L1 throughput %", g("l1tex__throughput.avg.pct_of_peak_sustained_active")),
        ("gld sectors/request", ratio(g("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"),
                                      g("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"))),
        ("gst sectors/request", ratio(g("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"),
                                      g("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"))),
        ("lds wavefronts/inst", ratio(g("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"),
                                      g("smsp__inst_executed_op_shared_ld.sum"))),
        ("sts wavefronts/inst", ratio(g("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"),
                                      g("smsp__inst_executed_op_shared_st.sum"))),
        ("lds bank conflicts", g("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")),
        ("sts bank conflicts", g("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum")),
        ("FFMA / all warp-insts", ratio(g("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", 0) / 32,
                                        g("smsp__inst_executed.sum"))),
        ("FLOP / DRAM byte", ratio(2 * g("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", 0),
                                   g("dram__bytes_read.sum", 0) + g("dram__bytes_write.sum", 0))),
        ("achieved occupancy %", g("sm__warps_active.avg.pct_of_peak_sustained_active")),
        ("registers/thread", g("launch__registers_per_thread")),
        ("smem/block (B)", g("launch__shared_mem_per_block_static", 0) + g("launch__shared_mem_per_block_dynamic", 0)),
        ("occ. limit: regs / smem / warps (blocks)", "{:.0f} / {:.0f} / {:.0f}".format(
            g("launch__occupancy_limit_registers", float("nan")),
            g("launch__occupancy_limit_shared_mem", float("nan")),
            g("launch__occupancy_limit_warps", float("nan")))),
    ]
    out += ["| metric | value |", "|---|---:|"]
    for k, v in rows:
        out.append(f"| {k} | {v:,.2f} |" if isinstance(v, float) else f"| {k} | {v} |")
    stalls = {k.split("stalled_")[1].split("_per_warp")[0]: v for k, v in m.items()
              if k.startswith("smsp__warp_issue_stalled_") and v == v}
    top = sorted(stalls.items(), key=lambda kv: -kv[1])[:4]
    out += ["", "top stall reasons (% of active warp-cycles): " +
            ", ".join(f"{k} {v:.1f}" for k, v in top), ""]
    return "\n".join(out)


def main():
    for p in map(Path, sys.argv[1:]):
        text = "\n".join(summarize(name, m) for (_, name), m in load(p).items())
        p.with_suffix(".summary.md").write_text(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
