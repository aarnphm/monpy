# MonPy NuMojo Patch

This vendored NuMojo tree is kept as an optional benchmark baseline. Upstream
NuMojo `0.9.0` declares Modular `0.26.2.*` in `pixi.toml`; MonPy currently
targets the Mojo `1.0.0` beta line (`Mojo 1.0.0.dev0` in the local Modular
build), so the source needs local compatibility edits before it can compile
either as a package or inside the benchmark sweep.

License remains NuMojo's upstream `Apache-2.0 WITH LLVM-exception`. See
`LICENSE` in this directory and the NuMojo README license section.

Patch areas:

- Renamed the package-level statistics `std` helper to `stddev` so the NuMojo
  root package no longer collides with the Mojo stdlib module name in current
  compilers. The object methods `NDArray.std()` and `Matrix.std()` keep their
  public spelling.
- Updated old atomic imports from `std.os.atomic`/`Consistency` to
  `std.atomic`/`Ordering`.
- Updated hand-written `__copyinit__` and `__moveinit__` receivers from
  `out self` to `mut self`, which is what the local beta compiler accepts for
  custom copy/move constructors.
- Added `numojo._compat.vectorize` to preserve NuMojo's older runtime-closure
  vectorize call sites while the current Mojo stdlib expects the closure as a
  compile-time parameter.
- Added `numojo._compat.simd_ops` wrappers around SIMD arithmetic, comparisons,
  predicates, and math functions. Current Mojo accepts the direct calls, but
  many NuMojo higher-order call sites no longer accept method references such as
  `SIMD.gt` or `math.exp` as generic callback values.
- Retargeted NuMojo vectorized routines to the compatibility wrapper and updated
  `unified` closures to current `capturing` syntax.
- Updated `HostExecutor` and matrix helper callbacks to use dtype-and-width
  generic kernel parameters. Scalar/array binary paths now splat scalar values to
  the active SIMD width before invoking the kernel.
- Marked generic `apply_along_axis` callback adapters and their callback
  implementations as `capturing raises` so the beta frontend does not treat the
  callback parameter as an unsupported dynamic trait.
- Marked nested logical-operation kernels as `capturing` for the current
  parameterized closure rules.
- Replaced older `HostExecutor` callable plumbing in arithmetic and trigonometry
  hot paths with direct SIMD loops so the benchmark can compile on
  `1.0.0.dev0`.
- Replaced remaining `HostExecutor` method-reference call sites in bitwise,
  comparison, content predicates, extrema, exponent/log, floating, hyperbolic,
  miscellaneous, and rounding routines with the local SIMD compatibility
  wrappers.
- Added floating-point `where` constraints to NuMojo routines that call floating
  stdlib intrinsics, including complex magnitude/power helpers, content
  predicates, hyperbolic helpers, `cbrt`/`sqrt`/`rsqrt`/`scalb`, `copysign`, and
  `hypot`/`hypot_fma`.
- Adjusted local imports that previously referenced the package through
  `numojo.*` from inside NuMojo modules. Current package compilation no longer
  resolves those self-references reliably. This includes `NDArray`,
  `ComplexNDArray`, IO loaders, sorting/indexing/manipulation helpers, and the
  matrix/linalg call paths used by the optional benchmark.
- Simplified the legacy DLPack managed tensor shim enough for beta parsing by
  removing the stored dynamic function pointer field. The DLPack bridge is not
  part of MonPy's benchmark contract; revisit this against the current DLPack
  v1.0 layout before treating it as production interop.
- Fixed `NDArray` shape and stride ownership so temporary `NDArrayShape` values
  do not leave arrays with dangling metadata after construction.
- Routed `NDArray.__matmul__` through the local `numojo.routines.linalg` import
  and adjusted in-place arithmetic dunders to use the patched arithmetic helpers.
- Replaced deprecated pointer truthiness checks with explicit address checks in
  the vendored memory containers.
- Replaced deprecated `len(String)` formatting checks with
  `String.count_codepoints()` in vendored array printers.

The intent is narrow: keep the external NuMojo baseline runnable for
`benches/bench_numojo_sweep.mojo` and keep the vendored package compiling with
`mojo package -I vendor/NuMojo -o /tmp/numojo.mojopkg vendor/NuMojo/numojo`.
These patches should not be treated as an upstream NuMojo compatibility promise.
