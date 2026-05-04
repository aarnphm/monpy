# monpy roadmap

monpy is aiming at a cpu-only, numpy-shaped v1 before broader device work.

## roadmap-minus-7

roadmap-minus-7 moves array interchange and linear algebra from explicit gaps to
covered v1 behavior:

- numpy array inputs are accepted by `array()` and `asarray()` for supported
  cpu dtypes. `copy=False` shares storage when no dtype conversion is needed,
  `copy=True` materializes native storage, and `copy=None` copies only when
  readonly or dtype conversion makes a mutable zero-copy view unsafe.
- dlpack import and export are supported for the cpu protocol. device reporting
  stays `(1, 0)`, and `from_dlpack(..., device="cpu")` follows the same
  `copy=` contract as numpy array import.
- mixed `int64`/`float32` array promotion now matches numpy instead of carrying
  a strict xfail. python and mojo expose private result-dtype helpers so tests
  can pin both tables to the same rows.
- imported and exported views preserve strides, owner lifetime, and mutation
  behavior across numpy, monpy, and dlpack boundaries.
- rank-2 dense matmul covers contiguous operands and dense-transposed right-hand
  operands through the same cpu fast-path policy.
- `linalg` import smoke coverage exists for `monpy`, `monumpy`, and
  `monpy.array_api`, with local tests for matmul, matrix transpose, `solve`,
  `inv`, `det`, singular matrices, dtype policy, and backend markers.

## design decisions

- `NativeArray` now points at a `NativeStorage` record. storage records carry
  the data pointer, byte length, refcount, and managed/external ownership bit.
  views retain the same storage; only the final managed reference frees bytes.
- external cpu storage is non-owning in mojo. python `ndarray._owner` pins the
  numpy or dlpack producer while mojo holds the external storage descriptor.
- layout predicates live with native shape metadata: c-contiguous,
  f-contiguous, zero-stride, negative-stride, physical offset, and explicit
  `materialize_c_contiguous(src)` for copy-before-kernel call sites.
- the public dtype set remains `bool`, `int64`, `float32`, and `float64`.
  native-endian cpu array-interface inputs outside that set raise explicitly.
- matmul fast paths may use blas only for positive-stride dense rank-2 layouts.
  scalar mojo remains the correctness fallback, and higher-rank matmul stays
  blocked until batch-broadcast semantics are implemented.
- `linalg.solve`, `linalg.inv`, and `linalg.det` use Accelerate LAPACK for
  macos f32/f64 inputs and a partial-pivot LU fallback for portability and
  non-accelerated dtype paths. backend markers remain observable on array
  results.

## still out of scope

- non-cpu devices, including metal-backed arrays.
- broad numpy dtype families such as complex, object, string, structured,
  unsigned, datetime, and narrow integer arrays.
- higher-rank matmul and axis-specific reductions.
- full numpy ufunc objects and their `out=`, `where=`, casting, reduce, and
  accumulate machinery.
