# `src/cute/` — vendored CuTe-style layout algebra (CPU-only)

date: 2026-05-06
status: live; phase 2 of the numpy-parity roadmap.

## what this is

- a vendored subset of NVIDIA CUTLASS's CuTe layout algebra, living
  at `src/cute/`. roughly 1,100 lines across four files.
- mirrors CUTLASS's `cute/` directory naming so practitioners coming
  from CUTLASS find familiar primitives without hunting.
- **not** an import of Modular's `max/kernels/src/layout/` package.
  that package is ~13.5 K lines of GPU-targeted machinery (TMA,
  tensor cores, async copy, shared-memory swizzles, masked tensors,
  thread layouts) that monpy does not need on day one. vendoring a
  small subset keeps the dependency surface inside the public Mojo
  stdlib while leaving the door open for the GPU primitives to be
  added on top of the same foundation later.

## design intent: cpu first, gpu-portable foundation

- the v1 ships **CPU-only kernels** but the algebra (`IntTuple`,
  `Layout`, `composition`, `coalesce`, `complement`, `select`,
  `logical_divide`) is **direction-agnostic by construction**.
  same primitives, same kernel signatures, target multiple backends.
- this matches how Modular's `max/kernels/` package works: one
  `Layout` algebra at the bottom; CPU iterators, GPU thread layouts,
  TMA descriptors, tensor-core fragments all sit on top as separate
  layers rather than as a parallel hierarchy.
- practical implication for kernel authors:
  - **write kernels parametric on `Layout` / `LayoutTensor`-shaped
    operands**, not on raw `Array` byte offsets. the backend choice
    becomes a parameter swap, not a rewrite.
  - the typed strided kernels that should follow phase 2's iter-swap
    proof of concept are a good test case: they take per-operand
    `Layout` plus a dtype parameter; the CPU body uses SIMD
    intrinsics; a future GPU body uses thread layouts and shared-
    memory tiles, but the operand types are the same.
- things deferred from CUTLASS that come back when we add GPU:
  - `RuntimeLayout` (comptime shape skeleton + dynamic int storage).
    needed once we want shape erasure across thousands of compile-
    time-known GPU kernels.
  - `Swizzle<B, M, S>` for shared-memory bank conflict avoidance.
  - `tiled_mma`, `tiled_copy`, `copy_dram_to_sram`, `cp_async_*`,
    TMA primitives.
  - Address spaces (`AddressSpace.SHARED`/`CONSTANT`/`LOCAL`).
  - Tensor core fragment layouts.
- things designed-out and unlikely to come back:
  - `ComposedLayout` (lazy composition). monpy materializes through
    Python at every public boundary; lazy composition adds
    complexity without the comptime payoff.
  - `*args: Int` variadic factories. mohaus stub generation can't
    emit Python type stubs for `*name`; we use `List[Int]` and the
    rank-2/rank-3 convenience helpers (`flat2`, `flat3`).
- bottom line: this package is the **first backend implementation**
  of a multi-backend layout-algebra plan, not a CPU-final port.
  resist sprinkling CPU-specific assumptions into the kernel
  signatures. when in doubt, look at how CUTLASS structures the
  equivalent kernel and stay close to that shape.

## file map

| file | purpose | size |
|---|---|---|
| `int_tuple.mojo` | recursive `IntTuple` value type and traversal helpers | ~408 lines |
| `layout.mojo` | `Layout` struct, ctors, basic queries (`__call__`, `idx2crd`, `size`, `cosize`, `__getitem__`) | ~190 lines |
| `functional.mojo` | algebra: `coalesce`, `select`, `transpose`, `composition`, `complement`, `logical_divide` | ~290 lines |
| `iter.mojo` | `LayoutIter` and `MultiLayoutIter` — stride-cursor walkers | ~225 lines |
| `__init__.mojo` | re-exports the public surface | ~32 lines |

## naming choices worth recording

- **package is `cute`, not `algorithm` or `layout`.**
  - `algorithm` collides with `std.algorithm` on Mojo's import path.
    the deprecation warning around implicit stdlib imports made the
    collision surface as `unable to locate module 'int_tuple'`
    because Mojo resolved `algorithm` to `std.algorithm` first.
  - `layout` collides with Modular's `max/kernels/src/layout` on the
    `MOHAUS_MOJO` toolchain import path.
  - `cute` is collision-free on the search path and matches the
    CUTLASS provenance.
- **Mojo 1.0 `Copyable` convention.**
  - `IntTuple` declares `Copyable` and provides
    `def __init__(out self, *, copy: Self)`, **not** the older
    `__copyinit__` form.
  - the manual constructor breaks the synthesis cycle between
    `IntTuple: Copyable` and `List[IntTuple]: Copyable`.
  - we do **not** declare `ImplicitlyCopyable` because the compiler
    refuses field-wise synthesis through `List[Self]` recursion.
    consequence: every `_children[i]` read needs an explicit
    `.copy()`.
- **`IntTuple.flat(values: List[Int])`**, not `*values: Int`.
  - mohaus stub generation (`monpy._native.pyi`) refuses `*name`
    Python identifiers when emitting type stubs. variadic factories
    are instead `flat2(a, b)` / `flat3(a, b, c)` plus the list form.

## `IntTuple` — the data model

- recursive, leaf-or-tuple ADT over `Int`.
- representation:
  ```mojo
  struct IntTuple(Copyable, Movable, Defaultable, Equatable, Writable):
      var _is_leaf: Bool
      var _value: Int
      var _children: List[IntTuple]
  ```
- factories:
  - `IntTuple()` — empty rank-0 tuple (rank 0, not a leaf).
  - `IntTuple.leaf(v: Int)` — single-integer leaf.
  - `IntTuple.flat(values: List[Int])` — flat tuple of leaves.
  - `IntTuple.flat2(a, b)` / `IntTuple.flat3(a, b, c)` — variadic
    convenience for the common ranks; all delegate to `flat`.
  - `IntTuple.nested(children: List[IntTuple])` — wrap children;
    rank = `len(children)`.
- query methods:
  - `is_leaf() -> Bool`.
  - `value() raises -> Int` — raises on non-leaf.
  - `rank() -> Int` — `0` for leaves, otherwise `len(_children)`.
  - `__getitem__(i: Int) raises -> IntTuple` — returns a deep copy of
    the i-th child; raises on leaf and out-of-range.
  - `__len__() -> Int` — alias for `rank()`.
  - `__eq__` / `__ne__` — structural equality.
  - `Writable` impl: leaves write as their integer; non-leaves write
    as `(c0, c1, ..., cN)`.
- traversal helpers:
  - `depth(t)` — max nesting depth; `0` for leaf, `1 + max(depth(c))`
    for non-leaf.
  - `product(t)` — multiply all leaves; for a shape this is the
    element count.
  - `sum_of(t)` — sum all leaves.
  - `flatten(t)` — preorder leaves as a flat (rank-len) tuple.
  - `flatten_to_int_list(t)` — same but as `List[Int]`.
  - `unflatten(flat_t, profile)` — re-nest a flat tuple to match
    `profile`'s structure.
- stride helpers:
  - `prefix_product(t)` — running product (column-major-style
    strides). flat-shape `(s0, s1, ...sN)` → `(1, s0, s0*s1, ..., s0*..*s(N-1))`.
    nested shapes recurse with running outer product.
  - `make_row_major_strides(shape)` — numpy/C convention; stride for
    last axis is 1.
  - `_scale_int_tuple(t, factor)` — multiply every leaf by `factor`.
- arithmetic:
  - `inner_product(coord, stride)` — recursive zip-and-sum across
    the IntTuple trees. this is the load-bearing op for
    `Layout.__call__`.
- coordinate ↔ index conversions:
  - `crd2idx(coord, shape, stride)` — handles the case where `coord`
    is a leaf int into a multi-mode shape: unravels via
    `prefix_product(shape)` (column-major default, matching CuTe
    semantics) before `inner_product`.
  - `idx2crd(idx, shape, stride)` — inverse; at each leaf
    `(idx // stride) % shape`. non-bijective layouts: round-trip is
    identity only on the layout's image.
- structural compatibility:
  - `compatible(a, b)` — same nesting and same per-leaf values.
  - `weakly_congruent(a, b)` — same nesting only; values may differ.

## `Layout` — `(shape, stride)` as a coord-to-int function

- struct fields:
  - `var shape: IntTuple`
  - `var stride: IntTuple`
- ctors:
  - `Layout()` — empty rank-0; `size = 1` by convention so it's a
    composition identity.
  - `Layout(shape: IntTuple, stride: IntTuple)`.
  - `make_layout_row_major(shape)` — numpy/C: stride for axis k is
    `product(shape[k+1:])`.
  - `make_layout_col_major(shape)` — Fortran/F: stride for axis 0 is
    1.
  - `make_layout_strided(shape, stride)` — explicit.
  - `make_ordered_layout(shape, order)` — strides assigned by
    ascending priority in `order`. lower priority → smaller stride.
- query methods:
  - `rank()` — top-level mode count (delegates to `shape.rank()`).
  - `size() raises` — `product(shape)`. element count of the layout's
    domain.
  - `cosize() raises` — `max(L) + 1`. for a contiguous bijective
    layout this equals `size`; for layouts with stride-0 modes or
    gaps in the codomain it can be larger or smaller. handles
    leaf-shape (single integer mode) and non-leaf-shape (multi-mode)
    correctly.
  - `__call__(coord: IntTuple) raises -> Int` — apply layout to a
    hierarchical coord; delegates to `crd2idx`.
  - `call_int(idx: Int) raises -> Int` — apply to a 1D linear coord;
    unravels using **row-major** strides (numpy convention; differs
    from CuTe's column-major default for `__call__`).
  - `coord_for(idx: Int) raises -> IntTuple` — inverse of `call_int`.
  - `__getitem__(i: Int) raises -> Layout` — sub-layout at top-level
    mode `i`; deep-copies children.
- the Layout is **linear** (no constant term). the offset rides on
  `Array.offset_elems`. mirrors CuTe's `Layout` / `Tensor` split.
  - reshape, transpose, broadcast, slice all manipulate `Layout`;
    slice additionally accumulates onto `offset_elems`.
- prints as `(shape):(stride)`. e.g. `(4, 6):(6, 1)` for a 4×6
  row-major layout.

## `functional.mojo` — the algebra

### `coalesce(layout)`

- merge adjacent flat modes whose strides chain
  (`d_{i+1} == s_i * d_i`).
- drop modes of size 1.
- preserve modes of stride 0 (broadcasts) — they cannot merge but
  must not be dropped.
- post-condition: `coalesce(L)` represents the same function over
  `[0, size(L))` and has minimal flat rank.
- v1 ships flat coalesce only. nested coalesce that preserves
  hierarchy where possible is a phase-3 want.

### `select(layout, indices: List[Int])` and `transpose(layout, perm)`

- `select` returns a sub-layout consisting of modes at the given
  indices, in order.
- `transpose(L, perm)` is `select(L, perm)`.
- one primitive covers transpose, slice-by-axis, and squeeze.
- `select(L, [k])` is the k-th sub-layout.
- repeated indices in `select` are allowed (caller's responsibility
  to know this is a projection, not a permutation).

### `composition(a, b)`

- `(A ∘ B)(c) = A(B(c))`.
- shape of `C` equals shape of `B`. strides come from how `A`
  interprets `B`'s outputs.
- two layers:
  - **leaf-on-A**: `B`'s shape is a leaf (single integer mode). walk
    `A`'s flat modes; consume `B.stride` worth of stride first
    (skipping or splitting `A`'s modes), then take `B.shape` worth
    of size. uses the same algorithm as
    `cute/algorithm/functional.hpp::composition_impl`.
  - **by-mode B**: `B` is multi-mode. compose each top-level mode of
    `B` with `A` and concatenate.
- raises:
  - `composition: stride does not align with mode size` — `B.stride`
    cannot split `A`'s mode cleanly.
  - `composition: size does not align with mode` — `B.shape` cannot
    split `A`'s mode cleanly.
  - `composition: B's domain exceeds A's size` — `B.shape * B.stride`
    exits `A`'s codomain.
- worked example: `composition((4, 6):(1, 4), 8:1) = (4, 2):(1, 4)`.
  the `B = 8:1` mode reads 8 contiguous elements; `A = (4, 6):(1, 4)`
  is a column-major 4×6. composing yields a 4×2 sub-region.

### `complement(layout, cosize_target)`

- builds the "missing" modes that fill out `[0, cosize_target)`.
- algorithm: sort `A`'s flat modes by stride; emit complementary
  modes for the gaps; finally emit a tail mode if `cosize_target`
  exceeds `A`'s coverage.
- ~50 lines but load-bearing for `logical_divide` on non-bijective
  layouts. without it, `logical_divide` returns garbage for layouts
  whose strides don't tile their domain bijectively.

### `logical_divide(a, tiler)`

- partitions `a`'s top-level modes by `tiler`. per axis `i`:
  - `intra = composition(a[i], tiler[i])` — within-tile layout.
  - `inter = composition(a[i], complement(tiler[i], size(a[i])))`
    — across-tile layout.
  - packed as a nested mode `(intra, inter)`.
- the only tile op needed for v1; tiled reductions (phase 4) and
  tiled matmul (phase 6) both want "give me a layout whose leading
  mode walks within a tile, then through tiles."
- reference: CUTLASS `cute/algorithm/functional.hpp::logical_divide`.

## `iter.mojo` — stride-cursor iterators

### `LayoutIter`

- forward iterator over `[0, size(L))` of a single Layout.
- amortizes the `crd2idx` divmod across the whole walk: each
  `step()` is O(1) (one cursor increment + one stride add for the
  rollover) instead of O(rank) divmod per element.
- ctor: `LayoutIter(layout, item_size, base_offset_bytes=0)`.
- internal state:
  - `flat_shape: List[Int]` — flat-mode shape.
  - `flat_stride: List[Int]` — flat-mode stride.
  - `coords: List[Int]` — current coord per flat mode.
  - `byte_cursor: Int` — current byte offset into the buffer.
  - `visited: Int` / `size: Int` — termination tracking.
- accessors:
  - `has_next() -> Bool`.
  - `offset_bytes() -> Int`.
  - `element_index() -> Int` — `byte_cursor / item_size`. useful for
    kernels indexing `array.data.bitcast[T]()` in element units.
  - `next_inner_loop() -> Int` — count of consecutive elements in
    the innermost mode that share the same outer-coord state. for
    SIMD inner loops.
- mutators:
  - `step()` — advance one element. walks modes from innermost
    (last) to outermost (first); rollover subtracts
    `coord * stride * item_size` and resets coord to zero.
  - `reset()` — back to `base_offset_bytes`.
- convention: innermost mode is the **last** flat mode (matches
  numpy / row-major: axis 0 is outermost, axis -1 is innermost).

### `MultiLayoutIter`

- N broadcasted operands in lockstep.
- callers broadcast each operand to the common output shape
  (injecting stride-0 modes where needed) before constructing the
  iterator. the iterator does not know what broadcasting is — it
  just walks N layouts that happen to have stride-0 modes in some
  places.
- ctor:
  - `output_shape: List[Int]` — shape of the iteration domain
    (after broadcasting).
  - `operand_layouts: List[Layout]` — one per operand; must be
    congruent with `output_shape` (same flat-mode count).
  - `operand_item_sizes: List[Int]`.
  - `operand_base_offsets: List[Int]`.
- accessors:
  - `has_next()`.
  - `offset_bytes(operand: Int)` / `element_index(operand: Int)`.
  - `next_inner_loop()`.
- step semantics: one rollover updates every operand's byte cursor
  using its own stride. stride-0 modes naturally "freeze" their
  operand at the broadcast position.

## `Array ↔ Layout` adapter (in `src/array.mojo`)

- `as_layout(array: Array) raises -> Layout`:
  - flat-rank Layout from `array.shape` and `array.strides`.
  - the Layout is linear; offset stays on `Array.offset_elems`.
- `array_with_layout(source: Array, new_layout: Layout, offset_delta: Int = 0) raises -> Array`:
  - view of `source` whose flat shape/strides come from the
    (possibly nested) `new_layout`. `flatten_to_int_list` collapses
    nesting; `make_view_array` retains storage refcount.
  - new `offset_elems = source.offset_elems + offset_delta`.
- the adapter unblocks future view-op rewrites (reshape, transpose,
  broadcast, slice as Layout transformations) without imposing any
  rewrite right now. the existing implementations in `src/create.mojo`
  continue to work.

## design boundaries — what's in, what's deferred

### in scope (v1)

- recursive IntTuple with leaf/non-leaf discriminant.
- composition (leaf-on-A + by-mode-B), coalesce (flat), complement,
  select / transpose, logical_divide.
- LayoutIter and MultiLayoutIter (forward stride cursors).
- Array ↔ Layout adapter.

### deferred until a real call site demands

- `logical_product`, `blocked_product`, `zipped_divide`,
  `tiled_divide` — none needed for numpy parity v1; some come back
  in phase 4 (blocked reductions) and phase 6 (batched matmul).
- swizzles — XOR-based bank-conflict avoidance. CPU has no shared
  memory banks.
- `ScaledBasis` / `E<I...>` — these encode "stride is the i-th basis
  vector"; only matter for multi-tensor TMA descriptors.
- `ComposedLayout` — the lazy `A ∘ B` representation. CUTLASS uses
  it to delay composition until shapes specialize. monpy normalizes
  eagerly.
- comptime/runtime split (`IntTuple` + `RuntimeTuple` like Modular's
  stdlib). monpy targets CPU and runs Python at the boundary; the
  runtime-polymorphic representation is enough. add the split if
  profiling shows hot kernels are blocked by shape erasure.
- nested-coalesce that preserves hierarchy where possible. v1 ships
  flat coalesce.

### explicit non-goals

- GPU-shaped surface (TMA, tensor cores, async copy, address
  spaces, masked tensors, alignment specialization, thread
  layouts). that's what `max/kernels/src/layout/` is for.

## kernel migration status

- proof-of-concept: `unary_ops` strided fallback in `src/create.mojo`
  walks via `LayoutIter` instead of `physical_offset` per element.
  - 440 → 441 tests pass after the migration.
  - bench movement on `sliced_unary_sin_f32_300x300`: 14.95× → 14.54×.
    minimal because the divmod LayoutIter eliminates is **not** the
    perf bottleneck. the cost is dominated by:
    1. per-element dtype dispatch in `get_physical_as_f64` /
       `set_physical_from_f64`.
    2. Float64 round-trip even for f32 data.
    3. no SIMD vectorization on the strided path.
- conclusion: the Layout primitives are correct, but the right perf
  lever for the strided path is **typed strided kernels** analogous
  to the existing `apply_unary_typed_vec[dtype, width]` for the
  contig path — not just iter swap. the remaining 6 kernel families
  (binary same-shape, scalar, row-broadcast, rank2-tile, reduce,
  matmul-small) are deprioritised for migration on this basis.

## benchmarks

- `benchmarks/bench_strided.py` — strided-fallback baseline. covers
  cases that go through the divmod walker (sliced views, 3-D
  transposed inputs, broadcast). also covers Layout-only view ops
  (flip, rot90, squeeze, moveaxis) and creation helpers (eye,
  meshgrid).
- `benchmarks/bench_array_core.py` — main contiguous bench.
  extended with `views` and `creation` cases for the phase-6 work
  (squeeze, moveaxis, swapaxes, ravel, flatten, concatenate, stack,
  hstack, vstack, eye, identity, tri, logspace, geomspace,
  meshgrid, atleast_2d, indices, plus the flip / rot90 family).

## references

- CUTLASS `media/docs/cute/01_layout.md` and
  `02_layout_algebra.md` — canonical algebra spec.
- `include/cute/layout.hpp` and
  `include/cute/algorithm/functional.hpp` — algorithm skeletons; the
  Mojo port is roughly a transcription of `composition_impl`,
  `coalesce`, `complement`.
- Jay Shah's "CuTe Layout Algebra" series on
  research.colfax-intl.com — practitioner walkthrough with diagrams.
- Modular's `max/kernels/src/layout/` — Mojo-idiomatic reference;
  read but not transcribed. their flat `IntArray` representation
  (positive ints = leaves, negative ints = sub-tuple offsets) avoids
  the `List[Self]: Copyable` recursion problem at the cost of
  significantly more complex traversal code.
