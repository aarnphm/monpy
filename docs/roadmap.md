# monpy roadmap

monpy is aiming at a cpu-only, numpy-shaped v1 before broader device work.

see [numpy-port-gaps.md](numpy-port-gaps.md) for the missing numpy library map:
dtype/scalar machinery, coercion, strided iteration, ufunc dispatch, indexing,
reductions, linalg/tensor operations, random, fft, strings, masked arrays, io,
and compatibility modules.

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

- `Array` now points at a `Storage` record. storage records carry
  the data pointer, byte length, refcount, and managed/external ownership bit.
  views retain the same storage; only the final managed reference frees bytes.
- external cpu storage is non-owning in mojo. python `ndarray._owner` pins the
  numpy or dlpack producer while mojo holds the external storage descriptor.
- layout predicates live with native shape metadata: c-contiguous,
  f-contiguous, zero-stride, negative-stride, physical offset, and explicit
  `materialize_c_contiguous(src)` for copy-before-kernel call sites.
- the public dtype set covers `bool`, signed and unsigned integer families
  through 64-bit, `float16`/`float32`/`float64`, and `complex64`/`complex128`.
  object, string, structured, datetime, and timedelta dtypes still raise
  explicitly.
- matmul fast paths may use blas for dense rank-2 layouts, including complex
  `cgemm`/`zgemm`. scalar mojo remains the correctness fallback, and higher-rank
  matmul stays blocked until batch-broadcast semantics are implemented.
- `linalg.solve`, `linalg.inv`, `linalg.det`, `qr`, `cholesky`, `eig`, `eigh`,
  `svd`, `lstsq`, `pinv`, and rank helpers use Accelerate/LAPACK where
  available, with mojo fallbacks for the smaller v1 surface. backend markers
  remain observable on array results.

## still out of scope

- non-cpu devices, including metal-backed arrays.
- object, string, structured, datetime, and timedelta dtype families.
- higher-rank matmul.
- the deep ufunc tail: `where=` keyword support, `reduceat`, and numpy's full
  floating-point error-state machinery.
