# Vendored Dependencies

This directory contains source snapshots used for optional benchmark baselines
and compatibility experiments.

## NuMojo

- Path: `vendor/NuMojo`
- Upstream: `https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo`
- Vendored metadata: NuMojo `0.9.0`, package `numojo` `0.9.0`, original
  Modular dependency `0.26.2.*` in `vendor/NuMojo/pixi.toml`
- License: `Apache-2.0 WITH LLVM-exception`
- License text: `vendor/NuMojo/LICENSE`
- Local patch ledger: `vendor/NuMojo/MONPY_PATCHES.md`

The local copy is patched so `benches/bench_numojo_sweep.mojo` can run under
MonPy's Mojo `1.0.0.dev0` development toolchain. It is a benchmark baseline,
not runtime code for the MonPy Python package.
