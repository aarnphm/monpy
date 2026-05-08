"""Hierarchical integer tuples—the data model under `Layout`.

`IntTuple` is the recursive value type that powers the CuTe-style layout algebra.
Every shape, stride, coordinate, tiler, and "profile" in the layout module is an `IntTuple`.
A leaf carries a single integer; a non-leaf carries a list of child `IntTuple`s.

Representation: a discriminated union (`_is_leaf`, `_value`, `_children`).
We skip the comptime/runtime split that Modular's `int_tuple.mojo` uses (`IntTuple` + `RuntimeTuple`):
- monpy targets CPU and runs Python at the boundary, so the runtime-polymorphic representation is enough.
- Adding the comptime split is a phase-3 want if profiling shows it helps.

Recursive `Copyable` requires a manual `def __init__(out self, *, copy: Self)` because the auto-synthesized constructor cannot resolve `IntTuple: Copyable`
while `_children: List[IntTuple]` is itself awaiting that resolution. The manual definition breaks the cycle.
"""

from std.collections import List


struct IntTuple(Copyable, Defaultable, Equatable, Movable, Writable):
    """A leaf-or-tuple recursive value over `Int`."""

    var _is_leaf: Bool
    var _value: Int
    var _children: List[IntTuple]

    def __init__(out self):
        """Empty rank-0 tuple. Use `leaf`, `flat`, or `nested` for content."""
        self._is_leaf = False
        self._value = 0
        self._children = List[IntTuple]()

    def __init__(out self, *, copy: Self):
        """Deep-copy constructor; required by `Copyable` because the
        compiler can't synthesize copy through `List[IntTuple]`."""
        self._is_leaf = copy._is_leaf
        self._value = copy._value
        self._children = List[IntTuple]()
        for i in range(len(copy._children)):
            self._children.append(copy._children[i].copy())

    @staticmethod
    def leaf(value: Int) -> Self:
        """A single-integer leaf."""
        var t = Self()
        t._is_leaf = True
        t._value = value
        return t^

    @staticmethod
    def flat(vals: List[Int]) -> Self:
        """A flat (rank-N) tuple of leaf integers."""
        var t = Self()
        for i in range(len(vals)):
            t._children.append(IntTuple.leaf(vals[i]))
        return t^

    @staticmethod
    def flat2(a: Int, b: Int) -> Self:
        var l = List[Int]()
        l.append(a)
        l.append(b)
        return Self.flat(l^)

    @staticmethod
    def flat3(a: Int, b: Int, c: Int) -> Self:
        var l = List[Int]()
        l.append(a)
        l.append(b)
        l.append(c)
        return Self.flat(l^)

    @staticmethod
    def nested(var children: List[IntTuple]) -> Self:
        """A non-leaf tuple wrapping the given children."""
        var t = Self()
        t._children = children^
        return t^

    @staticmethod
    def empty_tuple() -> Self:
        """A rank-0 tuple (no children, not a leaf). The identity for some
        layout operations."""
        return Self()

    def is_leaf(self) -> Bool:
        return self._is_leaf

    def value(self) raises -> Int:
        if not self._is_leaf:
            raise Error("IntTuple.value() called on a non-leaf")
        return self._value

    def rank(self) -> Int:
        """Number of top-level modes; 0 for leaves."""
        if self._is_leaf:
            return 0
        return len(self._children)

    def __getitem__(self, i: Int) raises -> Self:
        """Mode `i` of a non-leaf tuple. Returns a deep copy. Raises on
        leaves and out-of-range indices."""
        if self._is_leaf:
            raise Error("IntTuple.__getitem__ on leaf")
        if i < 0 or i >= len(self._children):
            raise Error("IntTuple.__getitem__ index out of range")
        return self._children[i].copy()

    def __len__(self) -> Int:
        """`len(t)` is `rank(t)`. Distinct from `product` (size) and
        `depth` (max nesting)."""
        return self.rank()

    def __eq__(self, other: Self) -> Bool:
        if self._is_leaf != other._is_leaf:
            return False
        if self._is_leaf:
            return self._value == other._value
        if len(self._children) != len(other._children):
            return False
        for i in range(len(self._children)):
            if not (self._children[i] == other._children[i]):
                return False
        return True

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def write_to[W: Writer](self, mut writer: W):
        """Pretty-print: leaves as their integer value, non-leaves as
        comma-separated children inside parentheses."""
        if self._is_leaf:
            writer.write(self._value)
            return
        writer.write("(")
        for i in range(len(self._children)):
            if i > 0:
                writer.write(", ")
            self._children[i].write_to(writer)
        writer.write(")")


# ============================================================
# traversal helpers
# ============================================================


def depth(t: IntTuple) -> Int:
    """Maximum nesting depth. `depth(leaf) == 0`, `depth((a, b))` is
    `1 + max(depth(a), depth(b))`."""
    if t.is_leaf():
        return 0
    var d = 0
    for i in range(t.rank()):
        var ci = depth(t._children[i])
        if ci > d:
            d = ci
    return 1 + d


def product(t: IntTuple) raises -> Int:
    """Product of all leaves. For shapes, this is the total element count
    (`size`)."""
    if t.is_leaf():
        return t.value()
    var p = 1
    for i in range(t.rank()):
        p *= product(t._children[i])
    return p


def sum_of(t: IntTuple) raises -> Int:
    """Sum of all leaves. Useful for cost estimates."""
    if t.is_leaf():
        return t.value()
    var s = 0
    for i in range(t.rank()):
        s += sum_of(t._children[i])
    return s


def flatten(t: IntTuple) -> IntTuple:
    """All leaves in left-to-right preorder, as a flat (rank-len) tuple.
    Idempotent on already-flat inputs."""
    if t.is_leaf():
        var children = List[IntTuple]()
        children.append(t.copy())
        return IntTuple.nested(children^)
    var children = List[IntTuple]()
    for i in range(t.rank()):
        var sub = flatten(t._children[i])
        # sub is rank-len leaves; splat its children in
        for j in range(sub.rank()):
            children.append(sub._children[j].copy())
    return IntTuple.nested(children^)


def flatten_to_int_list(t: IntTuple) raises -> List[Int]:
    """Flat list of leaf values in left-to-right preorder. Convenience
    for callers that don't need the IntTuple wrapper."""
    var out = List[Int]()
    if t.is_leaf():
        out.append(t.value())
        return out^
    for i in range(t.rank()):
        var sub = flatten_to_int_list(t._children[i])
        for j in range(len(sub)):
            out.append(sub[j])
    return out^


def _unflatten_helper(
    flat_values: List[Int],
    mut pos: Int,
    profile: IntTuple,
) raises -> IntTuple:
    """Recursive worker for `unflatten`. Advances `pos` as it consumes
    leaves and returns the reshaped subtree."""
    if profile.is_leaf():
        if pos >= len(flat_values):
            raise Error("unflatten: ran out of flat values")
        var leaf = IntTuple.leaf(flat_values[pos])
        pos += 1
        return leaf^
    var children = List[IntTuple]()
    for i in range(profile.rank()):
        var sub = _unflatten_helper(flat_values, pos, profile._children[i].copy())
        children.append(sub^)
    return IntTuple.nested(children^)


def unflatten(flat_t: IntTuple, profile: IntTuple) raises -> IntTuple:
    """Re-nest a flat tuple to match `profile`'s structure. The leaf
    count must match `len(flatten(profile))`."""
    var values = flatten_to_int_list(flat_t)
    var pos = 0
    var result = _unflatten_helper(values, pos, profile)
    if pos != len(values):
        raise Error("unflatten: profile and flat tuple have different leaf counts")
    return result^


# ============================================================
# stride helpers
# ============================================================


def prefix_product(t: IntTuple) raises -> IntTuple:
    """Running product (column-major-style strides for a shape). For a
    flat shape `(s0, s1, ..., sN)`, returns `(1, s0, s0*s1, ..., s0*..*s(N-1))`.
    Hierarchical inputs recurse: each subtree gets a prefix-product over
    its own leaves, scaled by the running outer product."""
    if t.is_leaf():
        return IntTuple.leaf(1)
    var children = List[IntTuple]()
    var running = 1
    for i in range(t.rank()):
        var child = t._children[i].copy()
        if child.is_leaf():
            children.append(IntTuple.leaf(running))
            running *= child.value()
        else:
            # nested: build sub-prefix scaled by `running`, then advance
            var sub = prefix_product(child.copy())
            children.append(_scale_int_tuple(sub, running))
            running *= product(child)
    return IntTuple.nested(children^)


def _scale_int_tuple(t: IntTuple, factor: Int) raises -> IntTuple:
    """Multiply every leaf by `factor`. Internal helper for
    `prefix_product` over nested shapes."""
    if t.is_leaf():
        return IntTuple.leaf(t.value() * factor)
    var children = List[IntTuple]()
    for i in range(t.rank()):
        children.append(_scale_int_tuple(t._children[i], factor))
    return IntTuple.nested(children^)


def make_row_major_strides(shape: IntTuple) raises -> IntTuple:
    """Numpy / C convention: stride for axis k is `product(shape[k+1:])`.
    The last axis has stride 1; the first axis has the largest stride."""
    if shape.is_leaf():
        return IntTuple.leaf(1)
    var children = List[IntTuple]()
    var n = shape.rank()
    var strides = List[Int]()
    var running = 1
    # Walk from innermost to outermost
    for i in range(n - 1, -1, -1):
        var size_i = product(shape._children[i])
        strides.append(running)
        running *= size_i
    # Reverse into outer-first order
    var rev = List[Int]()
    for i in range(len(strides) - 1, -1, -1):
        rev.append(strides[i])
    # If any child of `shape` is itself nested, the corresponding stride
    # entry must be a sub-tuple of strides. Build per-child.
    for i in range(n):
        var child_shape = shape._children[i].copy()
        var outer_stride = rev[i]
        if child_shape.is_leaf():
            children.append(IntTuple.leaf(outer_stride))
        else:
            var sub_prefix = prefix_product(child_shape)
            children.append(_scale_int_tuple(sub_prefix, outer_stride))
    return IntTuple.nested(children^)


# ============================================================
# inner product (the load-bearing op for `Layout.__call__`)
# ============================================================


def inner_product(coord: IntTuple, stride: IntTuple) raises -> Int:
    """Recursive zip-and-sum across the IntTuple trees. The "function"
    a layout represents is `inner_product(coord, stride)`."""
    if coord.is_leaf() and stride.is_leaf():
        return coord.value() * stride.value()
    if coord.is_leaf() or stride.is_leaf():
        raise Error("inner_product: structural mismatch (leaf vs non-leaf)")
    if coord.rank() != stride.rank():
        raise Error("inner_product: rank mismatch")
    var s = 0
    for i in range(coord.rank()):
        s += inner_product(coord._children[i], stride._children[i])
    return s


# ============================================================
# coordinate <-> index conversions
# ============================================================


def idx2crd(idx: Int, shape: IntTuple, stride: IntTuple) raises -> IntTuple:
    """Inverse of `Layout.__call__` for in-range indices. Walks the
    `shape` tree, projecting `idx` onto each leaf via `(idx // stride) % shape`.
    For non-bijective layouts, the round-trip `crd2idx ∘ idx2crd` is the
    identity only on the layout's image."""
    if shape.is_leaf() and stride.is_leaf():
        var s = shape.value()
        var d = stride.value()
        if d == 0:
            return IntTuple.leaf(0)
        return IntTuple.leaf((idx // d) % s)
    if shape.is_leaf() or stride.is_leaf():
        raise Error("idx2crd: structural mismatch")
    if shape.rank() != stride.rank():
        raise Error("idx2crd: rank mismatch")
    var children = List[IntTuple]()
    for i in range(shape.rank()):
        children.append(idx2crd(idx, shape._children[i], stride._children[i]))
    return IntTuple.nested(children^)


def crd2idx(coord: IntTuple, shape: IntTuple, stride: IntTuple) raises -> Int:
    """Coordinate-to-index. If `coord` is a leaf int, it's first
    unraveled against `shape`'s default (column-major) strides — this
    matches CuTe semantics where a 1D coord into a multi-mode shape
    enumerates elements in column-major order. Monpy's row-major
    user-facing iteration is handled separately by `Layout.call_int`."""
    if coord.is_leaf() and not shape.is_leaf():
        var default = prefix_product(shape)
        var unraveled = idx2crd(coord.value(), shape, default)
        return inner_product(unraveled, stride)
    return inner_product(coord, stride)


# ============================================================
# structural compatibility
# ============================================================


def compatible(a: IntTuple, b: IntTuple) raises -> Bool:
    """`a` is compatible with `b` if they have identical tree structure
    (same nesting, same per-leaf values). Used to validate that a coord
    matches a shape."""
    if a.is_leaf() != b.is_leaf():
        return False
    if a.is_leaf():
        return a.value() == b.value()
    if a.rank() != b.rank():
        return False
    for i in range(a.rank()):
        if not compatible(a._children[i], b._children[i]):
            return False
    return True


def weakly_congruent(a: IntTuple, b: IntTuple) -> Bool:
    """`a` and `b` are weakly congruent if they have identical tree
    structure regardless of leaf values. Used to validate a stride
    against a shape."""
    if a.is_leaf() != b.is_leaf():
        return False
    if a.is_leaf():
        return True
    if a.rank() != b.rank():
        return False
    for i in range(a.rank()):
        if not weakly_congruent(a._children[i], b._children[i]):
            return False
    return True
