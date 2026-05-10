---
title: monpy research notes
date: 2026-05-07
---

_these are working research notes — math, proofs, and proposals for the kernels and dispatch machinery that landed during the numpy-parity push._

each note is self-contained but the corpus cross-references freely. organized by subsystem rather than by chronology.

## the notes

| file                                                         | scope                                                                                                             |
| ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| [blas-lapack-dispatch.md](blas-lapack-dispatch.md)           | F77 ABI, workspace queries, goto-vanzee microkernel, conjugate-pair eigenvector unpacking, accelerate vs openblas |
| [complex-kernels.md](complex-kernels.md)                     | smith's algorithm overflow proof, branch cuts, euler identities, FMA error analysis, interleaved storage          |
| [cute-layout-algebra.md](cute-layout-algebra.md)             | layout = shape ⊗ stride, composition, coalescing, swizzling, the functor view                                     |
| [dtype-promotion-casting.md](dtype-promotion-casting.md)     | NEP 50 lattice proofs, can_cast as a partial order, complex absorption, NxN structure                             |
| [memory-alignment.md](memory-alignment.md)                   | cache-line alignment, SIMD width, AoS vs SoA, false sharing, allocator proposal for monpy                         |
| [einsum-contraction.md](einsum-contraction.md)               | contraction order NP-hardness, opt_einsum strategies, BLAS-friendly reductions, cost model                        |
| [matrix-power.md](matrix-power.md)                           | NumPy/JAX matrix_power contract, binary-decomposition proof, small-matrix dispatch boundary                       |
| [meshgrid-stride-semantics.md](meshgrid-stride-semantics.md) | NumPy `meshgrid` view contract, sparse/dense stride formulas, `copy=True` materialisation policy                  |
| [simd-vectorisation.md](simd-vectorisation.md)               | width selection, roofline / arithmetic-intensity argument, strided loads, f16 fallbacks, reductions               |
| [recent-field-notes.md](recent-field-notes.md)               | May 8-9 implementation field notes: wrappers, interop, views, threading policy, linalg API frontier               |
| [jax-first-architecture.md](jax-first-architecture.md)       | JAX-shaped primitive spine, NumPy-compatible layout interpreter, Mojo execution planner migration                 |

## conventions

- math typeset in LaTeX inside `$...$` (inline) and `$$...$$` (display)
- file/line citations follow the `path:line` form so the editor can navigate
- proofs are tagged **Lemma**, **Theorem**, or **Proposition** in bold
- references are at the end of each note as a numbered list

## notes

- dtype-promotion:
  - the NEP 50 promotion table is **not** a join-semilattice. Instead, the triple `(int8, uint8, float16)` associates differently depending on parens (jax JEP 9407 documents the same gotcha).
  - monpy resolves the ambiguity by left-folding `result_type(*args)` deterministically; reproduced as Theorem 2.4 in [[dtype-promotion-casting]]
- einsum:
  - **Apple Accelerate exposes no `cblas_*gemm_batch`** as of macOS 15.x
  - monpy's current pair-contract loop probably miscomputes `bij,bjk->bik` if the batch axis isn't materialised explicitly.
    - worth a regression test.
- cute:
  - the bijectivity criterion via prefix-product induction rather than the cleaner tree-depth induction; flagged for a possible 200-word firmer-up.
- recent-field-notes:
  - current local macOS Python-facing frontier is fixed-cost heavy: after the tiny-linalg pass, 145/243 rows are slower than NumPy, 55 are above 1.25x, 17 are above 1.5x, and none are above 2x.
  - pure Mojo kernel rows are mostly healthy: 114 rows, median 0.980x, one row above 1.25x in the full local sweep.
  - next formal notes should cover buffer ingress, view construction economics, threading thresholds, and linalg small-matrix/API fixed costs.
- jax-first:
  - primitive/transform architecture should be the spine; NumPy compatibility is an eager interpreter plus layout/storage contract.
  - every public array operation should bind exactly one primitive before eager execution or tracing.
  - Mojo needs an `ExecutionPlan` / `IterationPlan` substrate so SIMD, striding, broadcasting, and backend selection stop living in one-off dispatcher cascades.
- matrix-power:
  - `n == 3` has the same multiply count under linear and binary algorithms; the win was deleting Python/native crossings and avoiding a BLAS frame for a 2x2.
  - binary decomposition starts paying algorithmically at larger powers: `n=8` is 7 multiplies down to 3, `2.33:1`.
