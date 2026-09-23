#!/usr/bin/env python3
"""Turn an `ncu --metrics ... --csv` dump into the derived numbers the writeup uses.

    python3 profiling/summarize.py profiling/reports/naive_4096.metrics.csv [more ...]

Raw counters are hard to read; the diagnosis lives in ratios:
  gld sectors/req   4 = coalesced 4B/thread, 32 = fully scattered, 1 = broadcast,
                    16 = coalesced float4 (a 512B warp request)
  lds wavefronts/inst  1 = conflict-free 32-bit LDS; n = n-way bank conflict
                    (LDS.128 needs >= 1 wavefront per 128 unique bytes)
  FLOP/DRAM byte    achieved arithmetic intensity against DRAM
  top stalls        where warps actually wait
Writes <input>.summary.md next to each input. The functions here are also used
by scripts/collate.py to build the cross-stage attribution table.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

NAN = float("nan")


def num(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return NAN


def load(path):
    """{(launch id, kernel name): {metric: value}}"""
    per_kernel = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("ID", "0"), row.get("Kernel Name", "?"))
            per_kernel[key][row["Metric Name"]] = num(row["Metric Value"])
    return per_kernel


def ratio(a, b):
    return a / b if b and b == b and b != 0 else NAN


def derived(m):
    """Ordered (label, value) pairs of the diagnostic ratios for one kernel launch."""
    def g(k, d=NAN):
        return m.get(k, d)
    ffma = g("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", 0)
    return [
        ("duration (us, ncu clocks)", g("gpu__time_duration.sum") / 1e3),
        ("SM throughput %", g("sm__throughput.avg.pct_of_peak_sustained_elapsed")),
        ("FMA pipe active %", g("sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active")),
        ("tensor pipe active %", g("sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active")),
        ("issue slots busy %", g("smsp__issue_active.avg.pct_of_peak_sustained_active")),
        ("DRAM throughput %", g("dram__throughput.avg.pct_of_peak_sustained_elapsed")),
        ("DRAM MB read", g("dram__bytes_read.sum") / 1e6),
        ("L2 hit %", g("lts__t_sector_hit_rate.pct")),
        ("L1 hit %", g("l1tex__t_sector_hit_rate.pct")),
        ("L1/smem throughput %", g("l1tex__throughput.avg.pct_of_peak_sustained_active")),
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
        ("FFMA / all warp-insts", ratio(ffma / 32, g("smsp__inst_executed.sum"))),
        ("FLOP / DRAM byte", ratio(2 * ffma, g("dram__bytes_read.sum", 0) + g("dram__bytes_write.sum", 0))),
        ("achieved occupancy %", g("sm__warps_active.avg.pct_of_peak_sustained_active")),
        ("registers/thread", g("launch__registers_per_thread")),
        ("smem/block (B)", g("launch__shared_mem_per_block_static", 0) + g("launch__shared_mem_per_block_dynamic", 0)),
        ("occ. limit regs/smem/warps (blocks)", "{:.0f} / {:.0f} / {:.0f}".format(
            g("launch__occupancy_limit_registers"), g("launch__occupancy_limit_shared_mem"),
            g("launch__occupancy_limit_warps"))),
    ]


def top_stalls(m, n=4):
    stalls = {k.split("stalled_")[1].split("_per_warp")[0]: v for k, v in m.items()
              if k.startswith("smsp__warp_issue_stalled_") and v == v}
    return sorted(stalls.items(), key=lambda kv: -kv[1])[:n]


def fmt(v):
    return f"{v:,.2f}" if isinstance(v, float) else str(v)


def summarize(name, m):
    out = [f"### {name[:90]}", "", "| metric | value |", "|---|---:|"]
    out += [f"| {k} | {fmt(v)} |" for k, v in derived(m)]
    out += ["", "top stall reasons (% of active warp-cycles): " +
            ", ".join(f"{k} {v:.1f}" for k, v in top_stalls(m)), ""]
    return "\n".join(out)


def main():
    for p in map(Path, sys.argv[1:]):
        text = "\n".join(summarize(name, m) for (_, name), m in load(p).items())
        p.with_suffix(".summary.md").write_text(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
