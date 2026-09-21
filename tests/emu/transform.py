#!/usr/bin/env python3
"""Rewrite CUDA launch syntax so kernel sources compile as plain C++ for the
emulator:   kern<T...><<<grid, block, smem, stream>>>(args);
       ->   ::emu::launch(grid, block, [&] { kern<T...>(args); });
Usage: transform.py IN OUT"""
import re
import sys

src = open(sys.argv[1]).read()
pat = re.compile(r"([\w:]+(?:<[^<>;]*>)?)\s*<<<(.*?)>>>\s*\((.*?)\);", re.S)


def repl(m):
    cfg = [c.strip() for c in m.group(2).split(",")]
    return f"::emu::launch({cfg[0]}, {cfg[1]}, [&] {{ {m.group(1)}({m.group(3)}); }});"


out = pat.sub(repl, src)
# Kernel templates live in .cuh files included by the .cu; point them at the
# transformed copies too.
out = re.sub(r'#include "(\d\d_[\w]+)\.cuh"', r'#include "\1.emu.hpp"', out)
open(sys.argv[2], "w").write(out)
