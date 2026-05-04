# monpy architecture notes

monpy should be a mojo array library with numpy-shaped python APIs.

## layers

- `src/native.mojo` is the array runtime. it owns allocation, dtype codes, shape and stride metadata, views, generic dynamic-rank kernels, and specialized contiguous kernels.
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
- matmul is currently correctness-first scalar mojo code. serious parity will need tiled/vectorized kernels or a CPU BLAS backend.
