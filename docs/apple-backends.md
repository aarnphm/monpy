# apple backend notes

monpy should treat apple silicon as several backend surfaces, not one magic
accelerator.

## cpu: accelerate

accelerate is the first production target for dense cpu linear algebra. it gives
us blas/lapack on the cpu, with apple selecting the appropriate vector and matrix
instructions at runtime.

current monpy policy:

- route contiguous rank-2 `float32 @ float32 -> float32` through modular's
  `apple_matmul` wrapper when compiling on macos.
- route contiguous rank-2 `float64 @ float64 -> float64` through a local
  `cblas_dgemm` mirror of modular's f32 call shape.
- keep scalar mojo matmul as the portable correctness fallback.
- use thresholds once benchmarks show small matrices losing to blas call
  overhead.

## gpu: metal

metal should become a real device backend, not an accidental cpu array escape
hatch. it needs explicit buffer ownership, command queue lifetime, sync semantics,
and copy rules before python can expose `device="metal"`.

likely path:

- start with metal performance shaders matrix multiplication for contiguous
  rank-2 arrays.
- keep cpu arrays and metal arrays as separate native owners.
- require explicit transfer in v1, then consider implicit transfer later.

## neural engine

the neural engine is not a public raw blas target. public access is core ml
shaped: compile a model or tensor program, select compute units, let core ml pick
the neural engine when eligible.

monpy should not target the neural engine for general ndarray operations in v1.
the sane experiment is a separate core-ml-backed graph path for stable fused
blocks, not ndarray eager dispatch.
