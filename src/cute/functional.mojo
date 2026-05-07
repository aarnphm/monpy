"""CuTe-style layout algebra.

`coalesce`, `select` / `transpose`, `composition`, `complement`,
`logical_divide` over `Layout`. Mirrors CUTLASS's
`cute/algorithm/functional.hpp`.

The split of the `Layout` struct (in `layout.mojo`) from these
operations follows the CUTLASS convention. Adding a new layout-shaped
operation here doesn't require touching the struct definition.
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
    coalesce that preserves hierarchy where possible is a phase-3 want.
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

    Two layers:
    1. Leaf-on-A case (B is a single mode): walk A's flat modes,
       skip/split them by B's stride, then take B's size.
    2. By-mode B case: compose each top-level mode of B with A and
       concatenate.
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
    """`complement(A, M)` produces a layout that, when concatenated
    with `A`, fills out the codomain `[0, M)` bijectively.

    Algorithm: sort A's flat modes by stride; emit complementary modes
    for the gaps; finally emit a tail mode if `cosize_target` exceeds
    A's coverage.

    Used by `logical_divide` to behave correctly on non-bijective
    layouts.
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
    """Partition `a`'s top-level modes by `tiler`. Per axis `i`:
      - `intra = composition(a[i], tiler[i])` — intra-tile layout.
      - `inter = composition(a[i], complement(tiler[i], size(a[i])))`
        — inter-tile layout (composition of A with the gap-filler that
        completes the tiler within A's domain).
      - packed as a nested mode `(intra, inter)`.

    For numpy parity v1 this is the only tile op needed: tiled
    reductions and tiled matmul both want "give me a layout whose
    leading mode walks within a tile, then through tiles."

    Reference: CUTLASS `cute/algorithm/functional.hpp::logical_divide`.
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
