"""CuTe-style `Layout` — a function from coord to integer offset.

A `Layout` is a (shape, stride) pair of compatible `IntTuple`s. The
shape gives the input domain; the stride says how each leaf of a
coordinate contributes to the output offset.

This file ships the struct, basic queries (`__call__` / `idx2crd` /
`size` / `cosize` / `__getitem__`), and the constructors (`row_major`,
`col_major`, `strided`, `ordered`). The algebra (`coalesce`, `select`,
`transpose`, `composition`, `complement`, `logical_divide`) lives in
`functional.mojo`, mirroring CUTLASS's `cute/layout.hpp` vs
`cute/algorithm/functional.hpp` split.

Reference: NVIDIA CUTLASS `media/docs/cute/01_layout.md` and
`02_layout_algebra.md`. Algorithm skeletons follow
`include/cute/layout.hpp` adapted for monpy's row-major (numpy/C)
convention.
"""

from std.collections import List

from .int_tuple import (
    IntTuple,
    crd2idx,
    flatten_to_int_list,
    idx2crd,
    inner_product,
    make_row_major_strides,
    prefix_product,
    product,
    unflatten,
    weakly_congruent,
)


struct Layout(Copyable, Movable, Defaultable, Equatable, Writable):
    """A `(shape, stride)` pair representing a function `coord -> int`."""

    var shape: IntTuple
    var stride: IntTuple

    def __init__(out self):
        """Empty layout (rank-0). `size` is 1 by convention so it's
        identity for tile composition."""
        self.shape = IntTuple()
        self.stride = IntTuple()

    def __init__(out self, var shape: IntTuple, var stride: IntTuple):
        """Construct from explicit shape and stride. Caller is
        responsible for structural compatibility; we don't validate
        eagerly because composition often produces shapes that differ
        from strides in transient steps."""
        self.shape = shape^
        self.stride = stride^

    def __init__(out self, *, copy: Self):
        self.shape = copy.shape.copy()
        self.stride = copy.stride.copy()

    def __eq__(self, other: Self) -> Bool:
        return self.shape == other.shape and self.stride == other.stride

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.shape, ":", self.stride)

    def rank(self) -> Int:
        """Top-level rank (number of modes)."""
        return self.shape.rank()

    def size(self) raises -> Int:
        """Total element count = `product(shape)`."""
        return product(self.shape)

    def cosize(self) raises -> Int:
        """Maximum value of the layout function over its domain, plus 1.
        For a contiguous bijective layout this equals `size`; for
        layouts with stride-0 modes or gaps in the codomain it can be
        larger or smaller. Handles both leaf-shape (single integer mode)
        and non-leaf-shape (multi-mode) layouts."""
        if self.shape.rank() == 0 and not self.shape.is_leaf():
            return 1
        var shape_leaves = flatten_to_int_list(self.shape)
        var stride_leaves = flatten_to_int_list(self.stride)
        if len(shape_leaves) != len(stride_leaves):
            raise Error("Layout.cosize: shape/stride structural mismatch")
        var max_offset = 0
        for i in range(len(shape_leaves)):
            var s = shape_leaves[i]
            var d = stride_leaves[i]
            if s > 1 and d > 0:
                max_offset += (s - 1) * d
        return max_offset + 1

    def __call__(self, coord: IntTuple) raises -> Int:
        """Apply the layout to a hierarchical coord."""
        return crd2idx(coord, self.shape, self.stride)

    def call_int(self, idx: Int) raises -> Int:
        """Apply the layout to a 1D linear coord. Unravels `idx` against
        `shape` using row-major-style strides — matches monpy / numpy's
        flat-iteration order."""
        if self.shape.rank() == 0:
            return 0
        var default = make_row_major_strides(self.shape)
        var unraveled = idx2crd(idx, self.shape, default)
        return inner_product(unraveled, self.stride)

    def coord_for(self, idx: Int) raises -> IntTuple:
        """Inverse of `call_int` for in-range indices."""
        return idx2crd(idx, self.shape, self.stride)

    def __getitem__(self, i: Int) raises -> Self:
        """Sub-layout at top-level mode `i`. Returns a Layout whose
        shape and stride are the i-th children of `self.shape` and
        `self.stride`."""
        if i < 0 or i >= self.shape.rank():
            raise Error("Layout.__getitem__ index out of range")
        return Self(self.shape._children[i].copy(), self.stride._children[i].copy())


# ============================================================
# constructors
# ============================================================


def make_layout_row_major(shape: IntTuple) raises -> Layout:
    """Numpy / C convention: stride for the last axis is 1, stride for
    the first axis is `product(shape[1:])`."""
    var s = shape.copy()
    var d = make_row_major_strides(shape.copy())
    return Layout(s^, d^)


def make_layout_col_major(shape: IntTuple) raises -> Layout:
    """Fortran / column-major: stride for the first axis is 1.
    `prefix_product(shape)` on a flat shape gives `(1, s0, s0*s1, ...)`."""
    var s = shape.copy()
    var d = prefix_product(shape.copy())
    return Layout(s^, d^)


def make_layout_strided(shape: IntTuple, stride: IntTuple) -> Layout:
    """Construct from explicit shape and stride."""
    return Layout(shape.copy(), stride.copy())


def make_ordered_layout(shape: IntTuple, order: IntTuple) raises -> Layout:
    """Compact (bijective) layout where strides are assigned by
    ascending priority in `order`. Lower priority gets the smaller
    stride. Useful for constructing layouts whose iteration order is a
    permutation of axes without rebuilding strides by hand."""
    if not weakly_congruent(shape, order):
        raise Error("make_ordered_layout: shape and order have different structure")
    var shape_leaves = flatten_to_int_list(shape)
    var order_leaves = flatten_to_int_list(order)
    var n = len(shape_leaves)
    var idx_perm = List[Int]()
    for i in range(n):
        idx_perm.append(i)
    # Insertion sort by order_leaves[idx_perm[i]] ascending
    for i in range(1, n):
        var j = i
        while j > 0 and order_leaves[idx_perm[j - 1]] > order_leaves[idx_perm[j]]:
            var tmp = idx_perm[j - 1]
            idx_perm[j - 1] = idx_perm[j]
            idx_perm[j] = tmp
            j -= 1
    var stride_leaves = List[Int]()
    for _ in range(n):
        stride_leaves.append(0)
    var running = 1
    for k in range(n):
        var axis = idx_perm[k]
        stride_leaves[axis] = running
        running *= shape_leaves[axis]
    var stride_children = List[IntTuple]()
    for i in range(n):
        stride_children.append(IntTuple.leaf(stride_leaves[i]))
    var flat_stride = IntTuple.nested(stride_children^)
    var nested_stride = unflatten(flat_stride, shape)
    return Layout(shape.copy(), nested_stride^)
