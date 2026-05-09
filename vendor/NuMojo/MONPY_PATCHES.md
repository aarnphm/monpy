# MonPy NuMojo Patch

This vendored NuMojo tree is kept as an optional benchmark baseline. Upstream
NuMojo `0.9.0` declares Modular `0.26.2.*` in `pixi.toml`; MonPy currently
targets Mojo `1.0.0.dev0`, so the source needs local compatibility edits before
it can compile in the benchmark sweep.

License remains NuMojo's upstream `Apache-2.0 WITH LLVM-exception`. See
`LICENSE` in this directory and the NuMojo README license section.

Patch areas:

- Removed the package-level `std` re-exports that collide with the Mojo stdlib
  module name in current compilers.
- Updated old atomic imports from `std.os.atomic`/`Consistency` to
  `std.atomic`/`Ordering`.
- Added `numojo._compat.vectorize` to preserve NuMojo's older runtime-closure
  vectorize call sites while the current Mojo stdlib expects the closure as a
  compile-time parameter.
- Retargeted NuMojo vectorized routines to the compatibility wrapper and updated
  `unified` closures to current `capturing` syntax.
- Replaced older `HostExecutor` callable plumbing in arithmetic and trigonometry
  hot paths with direct SIMD loops so the benchmark can compile on
  `1.0.0.dev0`.
- Fixed `NDArray` shape and stride ownership so temporary `NDArrayShape` values
  do not leave arrays with dangling metadata after construction.
- Routed `NDArray.__matmul__` through the local `numojo.routines.linalg` import
  and adjusted in-place arithmetic dunders to use the patched arithmetic helpers.
- Replaced deprecated pointer truthiness checks with explicit address checks in
  the vendored memory containers.

The intent is narrow: keep the external NuMojo baseline runnable for
`benches/bench_numojo_sweep.mojo`. These patches should not be treated as an
upstream NuMojo compatibility promise.
