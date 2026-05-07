# monpy architecture notes

monpy should be a mojo array library with numpy-shaped python APIs.

## layers

- `src/lib.mojo` is only the cpython extension boundary. it owns
  `PyInit__native`, builds `PythonModuleBuilder("_native")`, registers `Array`,
  and binds python-facing function names into `monpy._native`.
- `src/domain.mojo` owns compact dtype, op, reduction, unary-op, casting, and
  backend codes, plus the supported dtype registry metadata and promotion/cast
  rules.
- `src/storage.mojo` owns the storage record, refcounting, managed allocation,
  and external non-owning allocation records.
- `src/layout.mojo` owns the narrow static layout vocabulary: `Shape[rank]`,
  `Layout[rank]`, `Layout.__call__`, tile helpers, vector-width helpers, and the
  array-to-layout tensor lift used by future static kernels.
- `src/array.mojo` owns the `Array` record, scalar access, metadata methods,
  shape/stride helpers, native cast-copy dispatch for supported dtype pairs, and
  dynamic-rank fallback addressing. dtype metadata and promotion rules delegate
  back to `domain.mojo`.
- `src/create.mojo` owns creation entrypoints and the remaining python-callable
  operation glue that has not yet moved into narrower modules.
- `src/views.mojo`, `src/reductions.mojo`, and `src/matmul.mojo` are the flat
  operation surfaces that `lib.mojo` imports for those python bindings.
- `src/linalg.mojo` contains the linear algebra surface. `lib.mojo` still binds
  the linalg functions through `create.mojo` because top-level `linalg` collides
  with Mojo's standard-library package name in the current mohaus entrypoint.
- `src/elementwise.mojo` owns numeric loops, contiguous fast paths, fused
  elementwise kernels, reductions, matmul dispatch helpers, and LU/LAPACK
  helpers.
- `src/accelerate.mojo` owns Apple Accelerate ffi shims only.
- `python/monpy` is the python API facade. it parses python objects, numpy cpu arrays, dlpack cpu producers, keyword arguments, and numpy-flavored ergonomics, then delegates implemented work into mojo.
- `python/monpy/linalg.py` is the numpy-shaped linear algebra namespace.
- `python/monpy/array_api.py` is the standards-shaped namespace and re-exports
  the same `linalg` module for the currently supported surface.
- `python/monumpy` is a compatibility shim that re-exports `monpy`.

## policy

- implemented array operations must not call numpy internally.
- numpy is allowed as a test oracle and as an explicit cpu interchange boundary
  for numpy array import, `ndarray.__array__`, and dlpack round-trips.
- `copy=False` at an interchange boundary means storage sharing or a loud error;
  `copy=True` means a detached monpy-owned allocation; `copy=None` may copy only
  when dtype conversion or readonly memory makes zero-copy unsafe.
- views retain storage instead of copying raw pointers. external storage is
  non-owning in mojo and is kept alive by python owner slots.
- inserted-axis views use stride-zero native metadata and retain the same
  storage owner.
- unsupported numpy long-tail features should fail loudly with `NotImplementedError`, `BufferError`, or a narrow runtime error.
- cpu-only is the v1 device model.

## performance notes

- generic paths preserve dynamic-rank correctness with shape and stride metadata.
- fast paths should be added only when a dtype/layout/rank predicate makes the cheaper path obvious.
- allocation reuse, `out=`, expression fusion, and wider SIMD coverage are the next major perf levers.
- `sin_add_mul(x, y, scalar)` is the first explicit fused expression kernel.
- the numpy-shaped `sin(x) + y * scalar` pattern lowers through a private python expression object and materializes through the same mojo fused kernel. benchmarks must force materialization so this does not become a fake python-only win.
- matmul uses Apple Accelerate for positive-stride dense macos f32/f64 rank-2
  arrays, including c-contiguous and f-contiguous/transposed views, with scalar
  mojo as the portable fallback.
- `linalg` exposes aliases for matmul and matrix transpose, plus native
  `solve`, `inv`, and `det` implementations. f32/f64 macos inputs use
  Accelerate LAPACK; unsupported accelerated paths fall back to portable
  partial-pivot LU.
- backend markers on native arrays let tests and benchmarks assert that specialized kernels actually ran.

see [apple-backends.md](apple-backends.md) for the apple silicon backend split.
see [ffi-marshaling.md](ffi-marshaling.md) for why the residual `asarray` / `from_dlpack` / `strided_view` / `array_copy` ratios are marshaling tax rather than kernel cost, and the two paths out (cpython buffer protocol or numpy c api).
