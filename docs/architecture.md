# monpy architecture notes

monpy should be a mojo array library with numpy-shaped python APIs.

## layers

- `src/native.mojo` is the python-callable runtime surface. it owns argument conversion and operation-level dispatch.
- `src/native_types.mojo` owns the native array and storage structs, dtype/op constants, allocation, shape/stride metadata, physical offsets, layout predicates, and scalar load/store helpers.
- `src/native_kernels.mojo` owns numeric loops, contiguous fast paths, reductions, and matmul dispatch.
- `src/native_accelerate.mojo` owns Apple Accelerate ffi shims only.
- `src/lib.mojo` is only the cpython extension boundary. it registers `NativeArray` and exported functions into `monpy._native`.
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
