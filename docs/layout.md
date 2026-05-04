# layout proposal

date: 2026-05-04

## notes

`Layout = (shape, stride, offset)` as a function `coord -> physical element offset`.

everything else (composition, divide, complement) is algebra over that function.

- for monpy on cpu, the algebra fires only at comptime to generate tile pointers
- the inner kernels walk raw `ptr + offset` with simd intrinsics.
- layout is tile geometry

see Modular's `linalg/matmul/cpu/default.mojo:32-93`: matmul micro-kernel uses `a_ptr[idx0 * K]` and `b_ptr.load[width=simd_size, alignment=...](idx1*simd_size)`.

## plans

**keep**:
- layout-as-function
- flat `Shape[rank]` (no nested IntTuple)
- `__call__` (crd→offset)
- `tile[*sizes]`
- `vectorize[width]`
- static-vs-dynamic distinction inside one type.

**skip**:
- nested `IntTuple` storage trick
- `composition`
- `complement`
- `logical_divide`
- `logical_product`
- `blocked_product`
- swizzle
- TMA
- tensor_core*
- `RuntimeLayout` as a separate type
- `distribute` (use `sync_parallelize` from std)

## file structure

```
src/
  lib.mojo          cpython extension boundary only
  domain.mojo       dtype/op/reduction/backend code domain
  storage.mojo      storage records, refcounts, managed/external allocation
  layout.mojo       Shape, Layout, LayoutTensor, lift helpers
  array.mojo        Array, scalar access, metadata, dynamic shape helpers
  create.mojo       creation and remaining python-callable glue
  views.mojo        view/slice/broadcast operation surface
  elementwise.mojo  unary, binary, scalar binary, fused loops
  reductions.mojo   reduction operation surface
  matmul.mojo       matmul operation surface
  linalg.mojo       solve/inv/det operation surface
  accelerate.mojo   apple accelerate ffi only
```

## kernel migration

| # | kernel | current home | rationale |
| - | --- | --- | --- |
| 1 | none - land types only | `layout.mojo` | smell-test the abstraction with unit tests |
| 2 | `binary_elementwise[op, dtype, L]` | replaces same-shape contiguous binary loops in `elementwise.mojo` | one static pointer-walk family |
| 3 | `unary_elementwise[op, dtype, L]` | replaces contiguous unary loops in `elementwise.mojo` | same arc |
| 4 | `binary_scalar[op, dtype, L]` | replaces contiguous scalar-binary loops in `elementwise.mojo` | same arc |
| 5 | `reduce[op, dtype, L]` | replaces contiguous reductions in `elementwise.mojo` | preserves scalar-tail |
| 6 | `matmul_microkernel[Mk, Nk, simd]` | replaces the small-matmul inner loop in `matmul.mojo` / `elementwise.mojo` | mirrors the raw-ptr style from modular cpu matmul |

### kernels that never migrate??

- row-broadcast contiguous binary paths - broadcasting needs runtime
  stride-zeroing per axis.
- argmax contiguous fallback - dynamic-rank reduction.
- `lapack_*` / `lu_decompose_partial_pivot` - column-major blas-shaped.
- general binary fallback - runtime polymorphism backstop.

## Q

1. does `comptime for i in range(L.size())` inline cleanly for `L.size() in [16, 64]`?
   - bench: 32-elt static loop vs 32-elt scalar tail; should be within roughly 2%.
2. `InlineArray[Int, rank]` vs `IndexList[rank]` for the backing storage.
   - Modular uses `IndexList` (`runtime_layout.mojo:233-261`); check if reasons generalize
3. does `to_layout_tensor[L]()` compose at the Python boundary without per-call `comptime if` ladder overhead?

## hypothesis

- writing fused-kernel families generically (today `fused_sin_add_mul` is hand-coded; with Layout you write `def fused[ops, L]` once)
- `_DeferredArray` (currently 0.82-1.01× via expression detection) can generate a specialized kernel per expression shape

leverage for the *next* round of perf work

Notes when asking the perf team:
- **do not split `Layout` and `RuntimeLayout`**.
  - We split them because GPU kernels need type-erasure across thousands of compile-time-known shapes.
  - monpy has runtime polymorphism at the Python boundary anyway.
  - one `Layout` struct with comptime fast-paths inside `__call__` is plenty
    - if we want to support something like cupy then we might either have to study typelist-of-CoordLike machinery from `tile_layout.mojo`.
