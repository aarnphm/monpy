# apple backend notes

monpy should treat apple silicon as several backend surfaces.

## cpu: accelerate

current monpy policy:

- route positive-stride dense rank-2 `float32 @ float32 -> float32` through
  `cblas_sgemm` when compiling on macos.
- route positive-stride dense rank-2 `float64 @ float64 -> float64` through
  `cblas_dgemm`.
- pass transpose flags and leading dimensions to blas for dense c-contiguous,
  f-contiguous, and transposed views instead of materializing a temporary copy.
- keep scalar mojo matmul as the portable correctness fallback.
- use thresholds once benchmarks show small matrices losing to blas call
  overhead.
- expose `linalg.matmul` and `linalg.matrix_transpose` as namespace aliases over
  the same cpu kernels, so backend behavior does not fork by import path.
- route `linalg.solve`, `linalg.inv`, and `linalg.det` for f32/f64 square
  matrices through Accelerate LAPACK (`sgesv_`/`dgesv_` and
  `sgetrf_`/`dgetrf_`) on macos, with generic partial-pivot LU as the fallback.

## gpu: metal

it needs explicit buffer ownership, command queue lifetime, sync semantics, and copy rules before python can expose `device="metal"`.

likely path:

- start with metal performance shaders matrix multiplication for contiguous
  rank-2 arrays.
- keep cpu arrays and metal arrays as separate native owners.
- require explicit transfer in v1, then consider implicit transfer later.

## neural engine

the neural engine is not a public raw blas target.
