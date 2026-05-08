"""Array factories + Layout adapters.

Hosts:
  - `make_empty_array` / `make_external_array` / `make_view_array` —
    the three entry points for building Arrays. Empty allocates managed
    storage; external wraps a raw byte pointer; view shares storage with
    a source.
  - `as_layout` — flat-rank Layout from Array shape/strides.
  - `as_broadcast_layout` — Layout virtually broadcast to a target shape
    (size-1 axes get stride 0).
  - `array_with_layout` — produce a view whose shape/strides come from a
    Layout, with optional offset delta.
  - `int_list_from_py` — `PythonObject` → `List[Int]` adapter.

The adapters live alongside the factories because both need the same
`make_view_array` signature; one extra file would just create churn.
"""

from std.collections import List
from std.python import PythonObject

from cute.int_tuple import IntTuple, flatten_to_int_list
from cute.layout import Layout
from domain import BackendKind, dtype_storage_byte_len
from storage import (
    make_external_storage,
    make_managed_storage,
    retain_storage,
)

from .accessors import (
    Array,
    item_size,
    make_c_strides,
    shape_size,
    validate_shape,
)


def make_empty_array(dtype_code: Int, var shape: List[Int]) raises -> Array:
    var size = shape_size(shape)
    var strides = make_c_strides(shape)
    var byte_len = dtype_storage_byte_len(dtype_code, size)
    var storage = make_managed_storage(byte_len)
    # Do not zero-fill here. Creation APIs that promise initialized values write
    # explicitly, and math kernels overwrite every output element before return.
    return Array(
        dtype_code,
        shape^,
        strides^,
        size,
        0,
        storage,
        storage[].data,
        storage[].byte_len,
        BackendKind.GENERIC.value,
    )


def make_external_array(
    dtype_code: Int,
    var shape: List[Int],
    var strides: List[Int],
    offset_elems: Int,
    data: UnsafePointer[UInt8, MutExternalOrigin],
    byte_len: Int,
) raises -> Array:
    if len(shape) != len(strides):
        raise Error("shape and stride rank mismatch")
    var size = shape_size(shape)
    var storage = make_external_storage(data, byte_len)
    return Array(
        dtype_code,
        shape^,
        strides^,
        size,
        offset_elems,
        storage,
        data,
        storage[].byte_len,
        BackendKind.GENERIC.value,
    )


def make_view_array(
    source: Array,
    var shape: List[Int],
    var strides: List[Int],
    size_value: Int,
    offset_elems: Int,
) raises -> Array:
    if len(shape) != len(strides):
        raise Error("shape and stride rank mismatch")
    validate_shape(shape)
    var storage = retain_storage(source.storage)
    return Array(
        source.dtype_code,
        shape^,
        strides^,
        size_value,
        offset_elems,
        storage,
        source.data,
        source.byte_len,
        source.backend_code,
    )


# ============================================================
# Array ↔ Layout adapter
#
# `as_layout` builds a flat-rank Layout from an Array's shape/strides.
# `array_with_layout` produces a view whose shape/strides come from a
# Layout and whose offset accumulates onto the source's offset_elems.
# ============================================================


def as_layout(array: Array) raises -> Layout:
    """Flat-rank Layout matching `array.shape` and `array.strides`. The
    Layout is "linear" (no constant term); the offset rides on the
    Array. Caller is responsible for adding `array.offset_elems` when
    computing absolute byte offsets."""
    var shape_children = List[IntTuple]()
    var stride_children = List[IntTuple]()
    for axis in range(len(array.shape)):
        shape_children.append(IntTuple.leaf(array.shape[axis]))
        stride_children.append(IntTuple.leaf(array.strides[axis]))
    return Layout(IntTuple.nested(shape_children^), IntTuple.nested(stride_children^))


def as_broadcast_layout(array: Array, out_shape: List[Int]) raises -> Layout:
    """Layout of `array` virtually broadcast to `out_shape`. Returned
    layout has the same flat rank as `out_shape`; broadcast modes carry
    stride 0 (either freshly added outer axes, or size-1 axes expanded
    against `out_shape`). Used by walkers that need a single rank-
    aligned cursor per operand against the output shape."""
    var out_ndim = len(out_shape)
    var arr_ndim = len(array.shape)
    if arr_ndim > out_ndim:
        raise Error("as_broadcast_layout: array rank exceeds out_shape rank")
    var shape_children = List[IntTuple]()
    var stride_children = List[IntTuple]()
    for out_axis in range(out_ndim):
        var arr_axis = out_axis - (out_ndim - arr_ndim)
        if arr_axis < 0:
            shape_children.append(IntTuple.leaf(out_shape[out_axis]))
            stride_children.append(IntTuple.leaf(0))
        elif array.shape[arr_axis] == 1 and out_shape[out_axis] != 1:
            shape_children.append(IntTuple.leaf(out_shape[out_axis]))
            stride_children.append(IntTuple.leaf(0))
        else:
            shape_children.append(IntTuple.leaf(array.shape[arr_axis]))
            stride_children.append(IntTuple.leaf(array.strides[arr_axis]))
    return Layout(IntTuple.nested(shape_children^), IntTuple.nested(stride_children^))


def array_with_layout(source: Array, new_layout: Layout, offset_delta: Int = 0) raises -> Array:
    """Build a view of `source` whose shape/strides come from
    `new_layout` and whose `offset_elems` is `source.offset_elems +
    offset_delta`. The Layout is flattened to monpy's flat shape/stride
    list convention before constructing the view; nested-mode layouts
    from `composition` / `logical_divide` get implicitly flattened."""
    var flat_shape = flatten_to_int_list(new_layout.shape)
    var flat_stride = flatten_to_int_list(new_layout.stride)
    if len(flat_shape) != len(flat_stride):
        raise Error("array_with_layout: shape/stride structural mismatch")
    var size_value = 1
    for i in range(len(flat_shape)):
        size_value *= flat_shape[i]
    var new_offset = source.offset_elems + offset_delta
    return make_view_array(source, flat_shape^, flat_stride^, size_value, new_offset)


def int_list_from_py(obj: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(obj)):
        out.append(Int(py=obj[i]))
    return out^
