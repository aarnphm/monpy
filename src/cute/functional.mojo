"""CuTe-style layout algebra.

`coalesce`, `select` / `transpose`, `composition`, `complement`, `logical_divide` over `Layout`.
Mirrors CUTLASS's `cute/algorithm/functional.hpp`.

The split of the `Layout` struct (in `layout.mojo`) from these operations follows the CUTLASS convention.
Adding a new layout-shaped operation here doesn't require touching the struct definition.
"""

from std.collections import List

from .int_tuple import IntTuple, flatten_to_int_list
from .layout import Layout


# ============================================================
# coalesce — merge adjacent stride-compatible flat modes
# ============================================================


def coalesce(layout: Layout) raises -> Layout:
    """Merge adjacent flat modes whose strides chain
    (`d_{i+1} == s_i * d_i`). Drop modes of size 1. Preserve modes of
    stride 0 (broadcasts) — they can't merge with anything but they
    shouldn't be dropped either.

    Post-condition: the resulting layout has the same function over
    `[0, size(L))` and minimal flat rank.

    v1 ships flat coalesce only — first flatten, then merge. Nested
    coalesce that preserves hierarchy where possible stays deferred
    until profiling shows it matters.
    """
    var flat_shape = flatten_to_int_list(layout.shape)
    var flat_stride = flatten_to_int_list(layout.stride)
    var n = len(flat_shape)
    if n != len(flat_stride):
        raise Error("coalesce: shape/stride structural mismatch")

    var out_s = List[Int]()
    var out_d = List[Int]()
    for i in range(n):
        var s = flat_shape[i]
        var d = flat_stride[i]
        if s == 1:
            continue
        var k = len(out_s)
        if k > 0 and d != 0 and out_d[k - 1] != 0 and out_d[k - 1] * out_s[k - 1] == d:
            out_s[k - 1] = out_s[k - 1] * s
        else:
            out_s.append(s)
            out_d.append(d)

    if len(out_s) == 0:
        return Layout()

    var shape_children = List[IntTuple]()
    var stride_children = List[IntTuple]()
    for i in range(len(out_s)):
        shape_children.append(IntTuple.leaf(out_s[i]))
        stride_children.append(IntTuple.leaf(out_d[i]))
    return Layout(IntTuple.nested(shape_children^), IntTuple.nested(stride_children^))


# ============================================================
# select / transpose — mode permutation / projection
# ============================================================


def select(layout: Layout, indices: List[Int]) raises -> Layout:
    """Sub-layout consisting of modes at the given indices, in order.
    This is transpose, slice-by-axis, and squeeze in one primitive.

    `select(L, [perm[0], perm[1], ...])` is `transpose(L, perm)`.
    `select(L, [k])` is the k-th sub-layout.
    `select(L, [a, b, c])` with `a, b, c` distinct is a projection.
    """
    var rank = layout.shape.rank()
    var shape_children = List[IntTuple]()
    var stride_children = List[IntTuple]()
    for i in range(len(indices)):
        var idx = indices[i]
        if idx < 0 or idx >= rank:
            raise Error("select: index out of range")
        shape_children.append(layout.shape._children[idx].copy())
        stride_children.append(layout.stride._children[idx].copy())
    return Layout(IntTuple.nested(shape_children^), IntTuple.nested(stride_children^))


def transpose(layout: Layout, perm: List[Int]) raises -> Layout:
    """Permute top-level modes. Equivalent to `select(layout, perm)`."""
    return select(layout, perm)


# ============================================================
# composition — `A ∘ B` = `C` such that `C(c) = A(B(c))`
# ============================================================


def composition(a: Layout, b: Layout) raises -> Layout:
    """Composition `A ∘ B`: layout `C` such that `C(c) = A(B(c))`.

    Shape of `C` equals shape of `B` (the input domain). Strides come
    from how `A` interprets `B`'s outputs.

    Algebra: a `Layout` is the pair `(shape, stride)` interpreted as the
    map `i → Σ_k crd_k(i) · stride[k]`. Composition `C = A ∘ B` is the
    *function composition* of those maps under coordinate decomposition.
    The construction below realizes it as a structural transformation
    rather than a runtime function table.

    Why two cases:
    1. **Leaf-on-A**: when `B` is a single mode `(s, d)`, compose by
       walking `A`'s flat modes, "consuming" stride `d` first (to skip
       past or split the modes that B's stride hops over), then taking
       size `s`. This produces a flattened sub-layout of `A` that sees
       only the elements `B(0), B(1), …, B(s-1)`.
    2. **By-mode B**: when `B` has multiple top-level modes, compose
       each one against `A` independently and concatenate. This works
       because composition distributes over the by-mode decomposition
       of `B`.

    Common failure modes (raises Error):
    - "stride does not align" — `B` reaches into the middle of an `A`
      mode in a way that isn't a divisor.
    - "size does not align" — `B`'s domain exceeds what `A` can express.

    Reference: CUTLASS `cute/01_layout.md` §3 + `02_layout_algebra.md` §1.
    """
    if b.shape.is_leaf():
        var s = b.shape.value()
        var d = b.stride.value()
        return _composition_leaf_on_a(a, s, d)

    if b.shape.rank() == 0:
        return Layout()

    var shape_children = List[IntTuple]()
    var stride_children = List[IntTuple]()
    for i in range(b.shape.rank()):
        var b_sub = b[i]
        var c_sub = composition(a, b_sub)
        shape_children.append(c_sub.shape.copy())
        stride_children.append(c_sub.stride.copy())
    return Layout(IntTuple.nested(shape_children^), IntTuple.nested(stride_children^))


def _composition_leaf_on_a(a: Layout, s_in: Int, d_in: Int) raises -> Layout:
    """Leaf-on-A composition: walk A's flat modes; consume `d_in`
    worth of stride first (skipping or splitting modes as needed),
    then take `s_in` worth of size."""
    var a_shape = flatten_to_int_list(a.shape)
    var a_stride = flatten_to_int_list(a.stride)
    var n = len(a_shape)

    var rest_s = s_in
    var rest_d = d_in
    var out_s = List[Int]()
    var out_d = List[Int]()

    var i = 0
    while i < n and rest_s > 1:
        var a_s = a_shape[i]
        var a_d = a_stride[i]

        if rest_d > 0:
            if rest_d % a_s == 0:
                rest_d = rest_d // a_s
                i += 1
                continue
            if a_s % rest_d != 0:
                raise Error("composition: stride does not align with mode size")
            a_s = a_s // rest_d
            a_d = a_d * rest_d
            rest_d = 0

        if rest_s >= a_s:
            out_s.append(a_s)
            out_d.append(a_d)
            if rest_s % a_s != 0:
                raise Error("composition: size does not align with mode")
            rest_s = rest_s // a_s
        else:
            out_s.append(rest_s)
            out_d.append(a_d)
            rest_s = 1
        i += 1

    if rest_s > 1:
        raise Error("composition: B's domain exceeds A's size")

    if len(out_s) == 0:
        return Layout()

    var shape_children = List[IntTuple]()
    var stride_children = List[IntTuple]()
    for j in range(len(out_s)):
        shape_children.append(IntTuple.leaf(out_s[j]))
        stride_children.append(IntTuple.leaf(out_d[j]))
    return Layout(IntTuple.nested(shape_children^), IntTuple.nested(stride_children^))


# ============================================================
# complement — fills out the missing modes
# ============================================================


def complement(layout: Layout, cosize_target: Int) raises -> Layout:
    """`complement(A, M)` produces a layout `B` such that the disjoint
    union of `A`'s image with `B`'s image equals `{0, 1, …, M−1}` and
    `B`'s strides chain through the gaps that `A` leaves behind.

    Algebra: think of `A` as an injective map
    `i → Σ stride[k]·crd_k(i)` covering some sub-lattice of `[0, M)`.
    Walk `A`'s flat modes in ascending-stride order; for each gap
    between the running coverage `current` and the next stride `d`,
    emit a complementary mode `(d/current, current)` that fills the
    gap. A final tail mode covers `[A's reach, M)` if `M` exceeds
    A's coverage.

    Algorithm:
    1. Filter to modes with size > 1 and stride > 0 — size-1 and
       stride-0 modes contribute nothing to coverage.
    2. Sort surviving modes by stride ascending: gaps must be filled
       in stride order or the resulting layout isn't bijective on
       `[0, M)`.
    3. For each mode in stride order, if its stride exceeds the
       running `current`, emit a fill mode `(stride/current, current)`;
       then advance `current *= mode_size`.
    4. If `cosize_target > current` at the end, emit a final tail
       `(cosize_target/current, current)`.

    Why divisibility matters: a non-divisible gap (e.g. `A` has
    stride 3 but `current = 2`) means `A`'s image isn't aligned to
    any integer sub-lattice — no integer-strided fill completes it.
    That's the "non-divisible gap" raise.

    Used by `logical_divide` to partition non-bijective layouts
    correctly (where the tiler doesn't fill A's image).

    Reference: CUTLASS `cute/01_layout.md` §4 + `02_layout_algebra.md` §2.
    """
    var a_shape = flatten_to_int_list(layout.shape)
    var a_stride = flatten_to_int_list(layout.stride)
    var n = len(a_shape)

    var pairs_size = List[Int]()
    var pairs_stride = List[Int]()
    for i in range(n):
        if a_shape[i] > 1 and a_stride[i] > 0:
            pairs_size.append(a_shape[i])
            pairs_stride.append(a_stride[i])
    var k = len(pairs_size)

    var perm = List[Int]()
    for i in range(k):
        perm.append(i)
    for i in range(1, k):
        var j = i
        while j > 0 and pairs_stride[perm[j - 1]] > pairs_stride[perm[j]]:
            var tmp = perm[j - 1]
            perm[j - 1] = perm[j]
            perm[j] = tmp
            j -= 1

    var out_s = List[Int]()
    var out_d = List[Int]()
    var current = 1
    for kk in range(k):
        var idx_p = perm[kk]
        var s = pairs_size[idx_p]
        var d = pairs_stride[idx_p]
        if d > current:
            if d % current != 0:
                raise Error("complement: non-divisible gap")
            out_s.append(d // current)
            out_d.append(current)
            current = d
        current *= s

    if cosize_target > current:
        if cosize_target % current != 0:
            raise Error("complement: target not divisible")
        out_s.append(cosize_target // current)
        out_d.append(current)

    if len(out_s) == 0:
        return Layout()

    var shape_children = List[IntTuple]()
    var stride_children = List[IntTuple]()
    for j in range(len(out_s)):
        shape_children.append(IntTuple.leaf(out_s[j]))
        stride_children.append(IntTuple.leaf(out_d[j]))
    return Layout(IntTuple.nested(shape_children^), IntTuple.nested(stride_children^))


# ============================================================
# logical_divide — tile decomposition
# ============================================================


def logical_divide(a: Layout, tiler: Layout) raises -> Layout:
    """Partition each top-level mode of `a` by the corresponding mode
    of `tiler`, producing a layout whose leading mode walks *within*
    a tile and trailing mode walks *between* tiles.

    Algebra: `logical_divide(A, T) = D` where, for each axis `i`,
    `D[i] = (intra[i], inter[i])` and:
    - `intra[i] = A[i] ∘ T[i]` — composition of `A[i]` with the tile
      layout `T[i]`. This produces a sub-layout of `A[i]` whose
      domain is `T[i]`'s domain, walking the elements `T[i]` selects.
    - `inter[i] = A[i] ∘ complement(T[i], size(A[i]))` — composition
      of `A[i]` with the *gap-filler* that completes `T[i]` inside
      `A[i]`'s domain. This walks tile origins in row-major order
      over `A[i]`.

    Why this works: composition distributes over the by-mode product.
    Splitting each axis independently into `(intra, inter)` preserves
    the bijection — every element of `A[i]` is reached exactly once
    by the cartesian product of `intra[i]` and `inter[i]` domains.

    Why useful: tiled reductions and tiled matmul both want to write
    `for tile in inter: for elem in intra: …` and trust the strides
    to land in the right place. `logical_divide` produces exactly
    that shape — no manual index math, no aliasing.

    If `tiler` has fewer modes than `a`, leftover `a` axes pass
    through unchanged (no tiling on those axes).

    Reference: CUTLASS `cute/02_layout_algebra.md` §3 +
    `cute/algorithm/functional.hpp::logical_divide`.
    """
    var a_rank = a.shape.rank()
    var t_rank = tiler.shape.rank()
    if t_rank > a_rank:
        raise Error("logical_divide: tiler rank > layout rank")

    var shape_children = List[IntTuple]()
    var stride_children = List[IntTuple]()

    for i in range(a_rank):
        var a_sub = a[i]
        if i >= t_rank:
            shape_children.append(a_sub.shape.copy())
            stride_children.append(a_sub.stride.copy())
            continue

        var t_sub = tiler[i]
        var intra = composition(a_sub, t_sub)
        var a_size = a_sub.size()
        var t_complement = complement(t_sub, a_size)
        var inter = composition(a_sub, t_complement)

        var pair_shape = List[IntTuple]()
        pair_shape.append(intra.shape.copy())
        pair_shape.append(inter.shape.copy())
        var pair_stride = List[IntTuple]()
        pair_stride.append(intra.stride.copy())
        pair_stride.append(inter.stride.copy())

        shape_children.append(IntTuple.nested(pair_shape^))
        stride_children.append(IntTuple.nested(pair_stride^))

    return Layout(IntTuple.nested(shape_children^), IntTuple.nested(stride_children^))
