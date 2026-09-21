# ⚠️ Spoiler branch: `reference/claude-stages-5-6`

This branch contains Claude's reference implementations of:

- **Stage 5**: 2D register tiling (`kernels/05_blocktiling_2d.cuh`). On `main`,
  this stage is a spec with `TODO(Fasih)`. **Write yours before reading this.**
- **Stage 6**: float4 global access, transposed + padded A tile, split thread
  tile for conflict-free `LDS.128` (`kernels/06_vectorized.cuh`). It's built on
  stage 5, which is why it lives here and not on `main`.
- Autotune families for both (`bench/autotune.cu`).

Suggested use: finish your stage 5 on `main`, then
`git diff main reference/claude-stages-5-6 -- kernels/` and compare
approaches, register counts (`-DSGEMM_PTXAS_VERBOSE=ON`), SASS mix
(`scripts/sass_stats.sh`), and ncu wavefronts/inst. Stage 7 (warptiling)
is deliberately left unwritten for you.

Status: compiled for sm_80/86 (0 spills in the default configs; stage 5 uses
133 regs → 1 block/SM, stage 6 uses 112 → 2 blocks/SM) and logic-tested on the
CPU emulator (ASan, 156 cases, both float4 and scalar-fallback paths). **Not yet
run on a GPU.**
