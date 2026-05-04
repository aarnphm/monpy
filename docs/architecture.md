# monpy architecture notes

monpy should be a mojo array library with numpy-shaped python APIs.

## layers

- `src/native.mojo` is the python-callable runtime surface. it owns argument conversion and operation-level dispatch.
- `src/native_types.mojo` owns the one real array struct, dtype/op constants, allocation, shape/stride metadata, physical offsets, and scalar load/store helpers.
- `src/native_kernels.mojo` owns numeric loops, contiguous fast paths, reductions, LayoutTensor smoke paths, and matmul dispatch.
- `src/native_accelerate.mojo` owns Apple Accelerate ffi shims only.
- `src/lib.mojo` is only the cpython extension boundary. it registers `NativeArray` and exported functions into `monpy._native`.
- `python/monpy` is the python API facade. it parses python objects, keyword arguments, and numpy-flavored ergonomics, then delegates implemented work into mojo.
- `python/monpy/array_api.py` is the standards-shaped namespace.
- `python/monumpy` is a compatibility shim that re-exports `monpy`.

## policy

- implemented array operations must not call numpy internally.
- numpy is allowed as a test oracle and as an explicit conversion target for `ndarray.__array__`.
- unsupported numpy long-tail features should fail loudly with `NotImplementedError`, `BufferError`, or a narrow runtime error.
- cpu-only is the v1 device model.

## performance notes

- generic paths preserve dynamic-rank correctness with shape and stride metadata.
- fast paths should be added only when a dtype/layout/rank predicate makes the cheaper path obvious.
- allocation reuse, `out=`, expression fusion, and wider SIMD/LayoutTensor coverage are the next major perf levers.
- `sin_add_mul(x, y, scalar)` is the first explicit fused expression kernel.
- the numpy-shaped `sin(x) + y * scalar` pattern lowers through a private python expression object and materializes through the same mojo fused kernel. benchmarks must force materialization so this does not become a fake python-only win.
- matmul uses Apple Accelerate for contiguous macos f32/f64 rank-2 arrays, with scalar mojo as the portable fallback.
- backend markers on native arrays let tests and benchmarks assert that specialized kernels actually ran.

see [apple-backends.md](apple-backends.md) for the apple silicon backend split.
