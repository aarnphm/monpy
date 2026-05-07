"""Stride-cursor iterators over `Layout` objects.

`LayoutIter` walks `[0, size(L))` of a single layout exposing the
current byte offset at each step. The point of having an iterator is
to amortize the `crd2idx` divmod across the whole walk: each `step()`
is O(1) (one cursor increment + one stride add for the rollover) rather
than O(rank) divmod per element.

`MultiLayoutIter` walks N broadcasted operands in lockstep. Callers
broadcast each operand to the common output shape (injecting stride-0
modes where needed) before constructing the iterator.

Both types use a flattened representation: the layout's flat-mode
shape and stride lists drive the cursors. Hierarchical layouts work
because flatten preserves the leaf walk order.

Convention: the innermost mode is the LAST flat mode (matches
numpy / row-major: axis 0 is outermost, axis -1 is innermost).
"""

from std.collections import List

from .int_tuple import flatten_to_int_list, product
from .layout import Layout


# ============================================================
# LayoutIter — single layout
# ============================================================


struct LayoutIter(Movable):
    """Forward stride-cursor iterator over a `Layout`."""

    var flat_shape: List[Int]
    var flat_stride: List[Int]
    var coords: List[Int]
    var n_modes: Int
    var item_size: Int
    var base_offset_bytes: Int
    var byte_cursor: Int
    var size: Int
    var visited: Int

    def __init__(
        out self,
        layout: Layout,
        item_size: Int,
        base_offset_bytes: Int = 0,
    ) raises:
        self.flat_shape = flatten_to_int_list(layout.shape)
        self.flat_stride = flatten_to_int_list(layout.stride)
        self.n_modes = len(self.flat_shape)
        if self.n_modes != len(self.flat_stride):
            raise Error("LayoutIter: shape/stride structural mismatch")
        self.coords = List[Int]()
        for _ in range(self.n_modes):
            self.coords.append(0)
        self.item_size = item_size
        self.base_offset_bytes = base_offset_bytes
        self.byte_cursor = base_offset_bytes
        self.size = product(layout.shape) if self.n_modes > 0 else 1
        if self.n_modes == 0:
            self.size = 1
        self.visited = 0

    def has_next(self) -> Bool:
        return self.visited < self.size

    def reset(mut self):
        for i in range(self.n_modes):
            self.coords[i] = 0
        self.byte_cursor = self.base_offset_bytes
        self.visited = 0

    def offset_bytes(self) -> Int:
        return self.byte_cursor

    def element_index(self) -> Int:
        """Byte cursor divided by item_size — useful for kernels that
        index `array.data.bitcast[T]()` in element units."""
        if self.item_size == 0:
            return 0
        return self.byte_cursor // self.item_size

    def step(mut self):
        """Advance one element. After calling, `offset_bytes()` is the
        byte offset of the *next* element."""
        self.visited += 1
        if self.visited >= self.size:
            return
        var i = self.n_modes - 1
        while i >= 0:
            self.coords[i] += 1
            self.byte_cursor += self.flat_stride[i] * self.item_size
            if self.coords[i] < self.flat_shape[i]:
                return
            var rollback = self.coords[i] * self.flat_stride[i] * self.item_size
            self.byte_cursor -= rollback
            self.coords[i] = 0
            i -= 1

    def next_inner_loop(self) -> Int:
        """Count of consecutive elements in the innermost mode that
        share the same outer-coord state."""
        if self.n_modes == 0:
            return self.size - self.visited
        var inner_size = self.flat_shape[self.n_modes - 1]
        var elems_remaining_in_inner = inner_size - self.coords[self.n_modes - 1]
        return elems_remaining_in_inner


# ============================================================
# MultiLayoutIter — N broadcasted operands
# ============================================================


struct MultiLayoutIter(Movable):
    """Forward iterator over N layouts that share an output shape.

    Each layout may have stride-0 modes for broadcast dimensions; the
    iterator walks the common shape and updates each operand's byte
    cursor independently.

    `output_shape` is the shape of the iteration domain (after
    broadcasting). Each `operand_layouts[k]` must be congruent with
    `output_shape` (same flat-mode count); zero-stride modes encode
    broadcast.
    """

    var flat_output_shape: List[Int]
    var operand_strides: List[List[Int]]
    var operand_item_sizes: List[Int]
    var operand_base_offsets: List[Int]
    var coords: List[Int]
    var n_modes: Int
    var n_operands: Int
    var byte_cursors: List[Int]
    var size: Int
    var visited: Int

    def __init__(
        out self,
        output_shape: List[Int],
        operand_layouts: List[Layout],
        operand_item_sizes: List[Int],
        operand_base_offsets: List[Int],
    ) raises:
        self.flat_output_shape = output_shape.copy()
        self.n_modes = len(self.flat_output_shape)
        self.n_operands = len(operand_layouts)
        if self.n_operands != len(operand_item_sizes) or self.n_operands != len(operand_base_offsets):
            raise Error("MultiLayoutIter: operand list lengths disagree")

        self.operand_strides = List[List[Int]]()
        for k in range(self.n_operands):
            var s = flatten_to_int_list(operand_layouts[k].stride)
            if len(s) != self.n_modes:
                raise Error("MultiLayoutIter: operand stride rank does not match output shape")
            self.operand_strides.append(s^)
        self.operand_item_sizes = operand_item_sizes.copy()
        self.operand_base_offsets = operand_base_offsets.copy()
        self.coords = List[Int]()
        for _ in range(self.n_modes):
            self.coords.append(0)
        self.byte_cursors = List[Int]()
        for k in range(self.n_operands):
            self.byte_cursors.append(operand_base_offsets[k])

        var total = 1
        for i in range(self.n_modes):
            total *= self.flat_output_shape[i]
        self.size = total
        self.visited = 0

    def has_next(self) -> Bool:
        return self.visited < self.size

    def reset(mut self):
        for i in range(self.n_modes):
            self.coords[i] = 0
        for k in range(self.n_operands):
            self.byte_cursors[k] = self.operand_base_offsets[k]
        self.visited = 0

    def offset_bytes(self, operand: Int) -> Int:
        return self.byte_cursors[operand]

    def element_index(self, operand: Int) -> Int:
        if self.operand_item_sizes[operand] == 0:
            return 0
        return self.byte_cursors[operand] // self.operand_item_sizes[operand]

    def step(mut self):
        self.visited += 1
        if self.visited >= self.size:
            return
        var i = self.n_modes - 1
        while i >= 0:
            self.coords[i] += 1
            for k in range(self.n_operands):
                self.byte_cursors[k] += self.operand_strides[k][i] * self.operand_item_sizes[k]
            if self.coords[i] < self.flat_output_shape[i]:
                return
            for k in range(self.n_operands):
                var rollback = self.coords[i] * self.operand_strides[k][i] * self.operand_item_sizes[k]
                self.byte_cursors[k] -= rollback
            self.coords[i] = 0
            i -= 1

    def next_inner_loop(self) -> Int:
        if self.n_modes == 0:
            return self.size - self.visited
        return self.flat_output_shape[self.n_modes - 1] - self.coords[self.n_modes - 1]
