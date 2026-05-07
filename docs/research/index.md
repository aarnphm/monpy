---
title: monpy research notes
date: 2026-05-07
---

_these are working research notes — math, proofs, and proposals for the kernels and dispatch machinery that landed during the numpy-parity push._

each note is self-contained but the corpus cross-references freely. organized by subsystem rather than by chronology.

## the notes

| file                                                     | scope                                                                                                             |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| [blas-lapack-dispatch.md](blas-lapack-dispatch.md)       | F77 ABI, workspace queries, goto-vanzee microkernel, conjugate-pair eigenvector unpacking, accelerate vs openblas |
| [complex-kernels.md](complex-kernels.md)                 | smith's algorithm overflow proof, branch cuts, euler identities, FMA error analysis, interleaved storage          |
| [cute-layout-algebra.md](cute-layout-algebra.md)         | layout = shape ⊗ stride, composition, coalescing, swizzling, the functor view                                     |
| [dtype-promotion-casting.md](dtype-promotion-casting.md) | NEP 50 lattice proofs, can_cast as a partial order, complex absorption, NxN structure                             |
| [memory-alignment.md](memory-alignment.md)               | cache-line alignment, SIMD width, AoS vs SoA, false sharing, allocator proposal for monpy                         |
| [einsum-contraction.md](einsum-contraction.md)           | contraction order NP-hardness, opt_einsum strategies, BLAS-friendly reductions, cost model                        |
| [simd-vectorisation.md](simd-vectorisation.md)           | width selection, roofline / arithmetic-intensity argument, strided loads, f16 fallbacks, reductions               |

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
