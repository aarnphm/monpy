from std.collections import List
from std.memory import memcpy
from std.os import abort
from std.python import Python, PythonObject

from domain import (
    ArrayDType,
    BackendKind,
    dtype_item_size,
    dtype_result_for_binary,
    dtype_result_for_linalg,
    dtype_result_for_linalg_binary,
    dtype_result_for_reduction,
    dtype_result_for_unary,
)
from storage import (
    Storage,
    make_external_storage,
    make_managed_storage,
    release_storage,
    retain_storage,
)

from cute.int_tuple import IntTuple, flatten_to_int_list
from cute.layout import Layout


@fieldwise_init
struct Array(Movable, Writable):
    var dtype_code: Int
    var shape: List[Int]
    var strides: List[Int]
    var size_value: Int
    var offset_elems: Int
    var storage: UnsafePointer[Storage, MutExternalOrigin]
    var data: UnsafePointer[UInt8, MutExternalOrigin]
    var byte_len: Int
    var backend_code: Int

    @staticmethod
    def _get_self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("Array method receiver had the wrong type: ", e))

    def __del__(deinit self):
        release_storage(self.storage)

    @staticmethod
    def dtype_code_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].dtype_code)

    @staticmethod
    def ndim_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(len(self_ptr[].shape))

    @staticmethod
    def size_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].size_value)

    @staticmethod
    def shape_at_py(py_self: PythonObject, index_obj: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        var index = Int(py=index_obj)
        return PythonObject(self_ptr[].shape[index])

    @staticmethod
    def stride_at_py(py_self: PythonObject, index_obj: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        var index = Int(py=index_obj)
        return PythonObject(self_ptr[].strides[index])

    @staticmethod
    def item_size_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(item_size(self_ptr[].dtype_code))

    @staticmethod
    def data_address_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        var byte_offset = self_ptr[].offset_elems * item_size(self_ptr[].dtype_code)
        return PythonObject((self_ptr[].data + byte_offset).__int__())

    @staticmethod
    def is_c_contiguous_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(is_c_contiguous(self_ptr[]))

    @staticmethod
    def is_f_contiguous_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(is_f_contiguous(self_ptr[]))

    @staticmethod
    def has_negative_strides_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(has_negative_strides(self_ptr[]))

    @staticmethod
    def has_zero_strides_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(has_zero_strides(self_ptr[]))

    @staticmethod
    def storage_refcount_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].storage[].ref_count)

    @staticmethod
    def used_accelerate_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].backend_code == BackendKind.ACCELERATE.value)

    @staticmethod
    def used_fused_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].backend_code == BackendKind.FUSED.value)

    @staticmethod
    def backend_code_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].backend_code)

    @staticmethod
    def get_scalar_py(py_self: PythonObject, index_obj: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        var index = Int(py=index_obj)
        if self_ptr[].dtype_code == ArrayDType.BOOL.value:
            return PythonObject(get_physical_bool(self_ptr[], physical_offset(self_ptr[], index)))
        if self_ptr[].dtype_code == ArrayDType.INT64.value:
            return PythonObject(get_physical[DType.int64](self_ptr[], physical_offset(self_ptr[], index)))
        if self_ptr[].dtype_code == ArrayDType.INT32.value:
            return PythonObject(Int(get_physical[DType.int32](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.INT16.value:
            return PythonObject(Int(get_physical[DType.int16](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.INT8.value:
            return PythonObject(Int(get_physical[DType.int8](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.UINT64.value:
            return PythonObject(Int(get_physical[DType.uint64](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.UINT32.value:
            return PythonObject(Int(get_physical[DType.uint32](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.UINT16.value:
            return PythonObject(Int(get_physical[DType.uint16](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.UINT8.value:
            return PythonObject(Int(get_physical[DType.uint8](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.FLOAT32.value:
            return PythonObject(get_physical[DType.float32](self_ptr[], physical_offset(self_ptr[], index)))
        if self_ptr[].dtype_code == ArrayDType.FLOAT16.value:
            return PythonObject(Float64(get_physical[DType.float16](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.COMPLEX64.value:
            var phys = physical_offset(self_ptr[], index)
            var real = Float64(get_physical_c64_real(self_ptr[], phys))
            var imag = Float64(get_physical_c64_imag(self_ptr[], phys))
            var builtins = Python.import_module("builtins")
            return builtins.complex(PythonObject(real), PythonObject(imag))
        if self_ptr[].dtype_code == ArrayDType.COMPLEX128.value:
            var phys = physical_offset(self_ptr[], index)
            var real = get_physical_c128_real(self_ptr[], phys)
            var imag = get_physical_c128_imag(self_ptr[], phys)
            var builtins = Python.import_module("builtins")
            return builtins.complex(PythonObject(real), PythonObject(imag))
        return PythonObject(get_physical[DType.float64](self_ptr[], physical_offset(self_ptr[], index)))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Array(dtype_code=")
        writer.write(self.dtype_code)
        writer.write(", shape=")
        writer.write(self.shape)
        writer.write(")")


def item_size(dtype_code: Int) raises -> Int:
    return dtype_item_size(dtype_code)


def validate_shape(shape: List[Int]) raises:
    for i in range(len(shape)):
        if shape[i] < 0:
            raise Error("shape dimensions must be non-negative")


def shape_size(shape: List[Int]) raises -> Int:
    validate_shape(shape)
    var total = 1
    for i in range(len(shape)):
        total *= shape[i]
    return total


def make_c_strides(shape: List[Int]) raises -> List[Int]:
    var strides = List[Int]()
    for _ in range(len(shape)):
        strides.append(0)
    var stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = stride
        stride *= shape[axis]
    return strides^


def make_empty_array(dtype_code: Int, var shape: List[Int]) raises -> Array:
    var size = shape_size(shape)
    var strides = make_c_strides(shape)
    var byte_len = size * item_size(dtype_code)
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


def clone_int_list(values: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(values)):
        out.append(values[i])
    return out^


# ============================================================
# Array ↔ Layout adapter
#
# `as_layout` builds a flat-rank Layout from an Array's shape/strides.
# `array_with_layout` produces a view whose shape/strides come from a
# Layout and whose offset accumulates onto the source's offset_elems.
#
# Both stay here (rather than in a separate adapter file) because they
# need access to make_view_array's signature and Array internals; the
# coupling is tight enough that one extra file would just create churn.
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


def same_shape(lhs: List[Int], rhs: List[Int]) -> Bool:
    if len(lhs) != len(rhs):
        return False
    for i in range(len(lhs)):
        if lhs[i] != rhs[i]:
            return False
    return True


def slice_length(dim: Int, start: Int, stop: Int, step: Int) raises -> Int:
    if step == 0:
        raise Error("slice step cannot be zero")
    if step > 0:
        if start >= stop:
            return 0
        return (stop - start + step - 1) // step
    var negative_step = -step
    if start <= stop:
        return 0
    return (start - stop + negative_step - 1) // negative_step


def is_c_contiguous(array: Array) raises -> Bool:
    var expected = 1
    for axis in range(len(array.shape) - 1, -1, -1):
        if array.shape[axis] == 0:
            return True
        if array.shape[axis] != 1 and array.strides[axis] != expected:
            return False
        expected *= array.shape[axis]
    return True


def is_f_contiguous(array: Array) raises -> Bool:
    var expected = 1
    for axis in range(len(array.shape)):
        if array.shape[axis] == 0:
            return True
        if array.shape[axis] != 1 and array.strides[axis] != expected:
            return False
        expected *= array.shape[axis]
    return True


def has_negative_strides(array: Array) -> Bool:
    for axis in range(len(array.strides)):
        if array.strides[axis] < 0:
            return True
    return False


def has_zero_strides(array: Array) -> Bool:
    for axis in range(len(array.strides)):
        if array.strides[axis] == 0:
            return True
    return False


def is_linearly_addressable(array: Array) raises -> Bool:
    """True if the physical-offset span equals `size_value - 1` — the
    array's elements occupy a contiguous block of `size_value` slots,
    and a flat scan `data[offset_elems .. offset_elems + size_value)`
    visits every element exactly once. Holds for any permutation of
    c/f-contiguous storage (transpose, swapaxes, etc.).
    For commutative reductions (sum/min/max/prod), this lets us skip
    the LayoutIter walk and stride contig — a big win when the logical
    iteration order has a non-unit innermost stride."""
    if array.size_value <= 1:
        return True
    if has_zero_strides(array) or has_negative_strides(array):
        return False
    var max_off = 0
    for axis in range(len(array.shape)):
        var d = array.shape[axis]
        var s = array.strides[axis]
        if d == 0:
            return False
        max_off += (d - 1) * s
    return max_off + 1 == array.size_value


def physical_offset(array: Array, logical: Int) raises -> Int:
    if logical < 0 or logical >= array.size_value:
        raise Error("array index out of bounds")
    var physical = array.offset_elems
    var remainder = logical
    for axis in range(len(array.shape) - 1, -1, -1):
        var dim = array.shape[axis]
        if dim != 0:
            var coord = remainder % dim
            remainder = remainder // dim
            physical += coord * array.strides[axis]
    return physical


# === Parametric leaf accessors ==========================================
#
# `contiguous_ptr[dt]` / `get_physical[dt]` / `set_physical[dt]` are the
# single-source-of-truth typed leaf accessors keyed on Mojo's `DType`.
# Bool stays bespoke (`get_physical_bool`) — single-byte storage with
# nonzero-is-true semantics doesn't round-trip cleanly through
# `Scalar[DType.bool]`. Complex stays bespoke — interleaved (re, im)
# pairs have no `DType` representation.


def contiguous_ptr[dt: DType](array: Array) -> UnsafePointer[Scalar[dt], MutExternalOrigin]:
    return array.data.bitcast[Scalar[dt]]() + array.offset_elems


def get_physical[dt: DType](array: Array, physical: Int) -> Scalar[dt]:
    return array.data.bitcast[Scalar[dt]]()[physical]


def set_physical[dt: DType](mut array: Array, physical: Int, value: Scalar[dt]):
    array.data.bitcast[Scalar[dt]]()[physical] = value


def get_physical_bool(array: Array, physical: Int) raises -> Bool:
    return array.data[physical] != UInt8(0)


# Complex accessors: arrays store interleaved (real, imag) pairs.
# complex64 → 2 × Float32 starting at byte index `physical * 8`.
# complex128 → 2 × Float64 starting at byte index `physical * 16`.
def get_physical_c64_real(array: Array, physical: Int) raises -> Float32:
    return array.data.bitcast[Float32]()[physical * 2]


def get_physical_c64_imag(array: Array, physical: Int) raises -> Float32:
    return array.data.bitcast[Float32]()[physical * 2 + 1]


def get_physical_c128_real(array: Array, physical: Int) raises -> Float64:
    return array.data.bitcast[Float64]()[physical * 2]


def get_physical_c128_imag(array: Array, physical: Int) raises -> Float64:
    return array.data.bitcast[Float64]()[physical * 2 + 1]


def set_physical_c64(mut array: Array, physical: Int, real: Float32, imag: Float32) raises:
    var ptr = array.data.bitcast[Float32]()
    ptr[physical * 2] = real
    ptr[physical * 2 + 1] = imag


def set_physical_c128(mut array: Array, physical: Int, real: Float64, imag: Float64) raises:
    var ptr = array.data.bitcast[Float64]()
    ptr[physical * 2] = real
    ptr[physical * 2 + 1] = imag


# === Runtime dtype-code dispatch =========================================
#
# `dispatch_real_to_f64` / `dispatch_real_write_f64` / `dispatch_int_write_i64`
# are the single-source-of-truth runtime → comptime dtype switches over the
# real-dtype set. Each takes a parametric kernel (a `comptime` def
# parameterized on `DType`) and fans out to the per-dtype branches once.
# Callers that need bool/complex semantics handle them inline above the
# dispatcher call. Mirrors the `BinaryContigKernel` / `dispatch_real_typed_*`
# idiom in `elementwise/__init__.mojo`.


comptime PhysReadAsF64 = def[dt: DType](Array, Int) thin raises -> Float64
"""Reads a physical-offset value of dtype `dt` and returns it as Float64."""


comptime PhysWriteFromF64 = def[dt: DType](mut Array, Int, Float64) thin raises -> None
"""Narrows a Float64 to dtype `dt` and writes it to a physical offset."""


comptime PhysWriteFromI64 = def[dt: DType](mut Array, Int, Int64) thin raises -> None
"""Narrows an Int64 to integer dtype `dt` and writes it to a physical offset."""


def dispatch_real_to_f64[op: PhysReadAsF64](dtype_code: Int, array: Array, physical: Int) raises -> Float64:
    """Maps `dtype_code` to one of the 11 real DTypes and invokes `op[dt]`.
    Caller must handle BOOL / COMPLEX64 / COMPLEX128 before calling."""
    if dtype_code == ArrayDType.FLOAT64.value:
        return op[DType.float64](array, physical)
    if dtype_code == ArrayDType.FLOAT32.value:
        return op[DType.float32](array, physical)
    if dtype_code == ArrayDType.FLOAT16.value:
        return op[DType.float16](array, physical)
    if dtype_code == ArrayDType.INT64.value:
        return op[DType.int64](array, physical)
    if dtype_code == ArrayDType.INT32.value:
        return op[DType.int32](array, physical)
    if dtype_code == ArrayDType.INT16.value:
        return op[DType.int16](array, physical)
    if dtype_code == ArrayDType.INT8.value:
        return op[DType.int8](array, physical)
    if dtype_code == ArrayDType.UINT64.value:
        return op[DType.uint64](array, physical)
    if dtype_code == ArrayDType.UINT32.value:
        return op[DType.uint32](array, physical)
    if dtype_code == ArrayDType.UINT16.value:
        return op[DType.uint16](array, physical)
    if dtype_code == ArrayDType.UINT8.value:
        return op[DType.uint8](array, physical)
    raise Error("dispatch_real_to_f64: dtype not in real set")


def dispatch_real_write_f64[
    op: PhysWriteFromF64,
](dtype_code: Int, mut array: Array, physical: Int, value: Float64) raises -> Bool:
    """Maps `dtype_code` to one of the 11 real DTypes and invokes `op[dt]`.
    Returns True on dispatch; False if the dtype isn't real (caller falls through)."""
    if dtype_code == ArrayDType.FLOAT64.value:
        op[DType.float64](array, physical, value)
        return True
    if dtype_code == ArrayDType.FLOAT32.value:
        op[DType.float32](array, physical, value)
        return True
    if dtype_code == ArrayDType.FLOAT16.value:
        op[DType.float16](array, physical, value)
        return True
    if dtype_code == ArrayDType.INT64.value:
        op[DType.int64](array, physical, value)
        return True
    if dtype_code == ArrayDType.INT32.value:
        op[DType.int32](array, physical, value)
        return True
    if dtype_code == ArrayDType.INT16.value:
        op[DType.int16](array, physical, value)
        return True
    if dtype_code == ArrayDType.INT8.value:
        op[DType.int8](array, physical, value)
        return True
    if dtype_code == ArrayDType.UINT64.value:
        op[DType.uint64](array, physical, value)
        return True
    if dtype_code == ArrayDType.UINT32.value:
        op[DType.uint32](array, physical, value)
        return True
    if dtype_code == ArrayDType.UINT16.value:
        op[DType.uint16](array, physical, value)
        return True
    if dtype_code == ArrayDType.UINT8.value:
        op[DType.uint8](array, physical, value)
        return True
    return False


def dispatch_int_write_i64[
    op: PhysWriteFromI64,
](dtype_code: Int, mut array: Array, physical: Int, value: Int64) raises -> Bool:
    """Maps `dtype_code` to one of the 8 integer DTypes and invokes `op[dt]`.
    Returns True on dispatch; False if the dtype isn't integral (caller falls through)."""
    if dtype_code == ArrayDType.INT64.value:
        op[DType.int64](array, physical, value)
        return True
    if dtype_code == ArrayDType.INT32.value:
        op[DType.int32](array, physical, value)
        return True
    if dtype_code == ArrayDType.INT16.value:
        op[DType.int16](array, physical, value)
        return True
    if dtype_code == ArrayDType.INT8.value:
        op[DType.int8](array, physical, value)
        return True
    if dtype_code == ArrayDType.UINT64.value:
        op[DType.uint64](array, physical, value)
        return True
    if dtype_code == ArrayDType.UINT32.value:
        op[DType.uint32](array, physical, value)
        return True
    if dtype_code == ArrayDType.UINT16.value:
        op[DType.uint16](array, physical, value)
        return True
    if dtype_code == ArrayDType.UINT8.value:
        op[DType.uint8](array, physical, value)
        return True
    return False


def _phys_read_as_f64[dt: DType](array: Array, physical: Int) raises -> Float64:
    comptime if dt.is_floating_point():
        return Float64(get_physical[dt](array, physical))
    return Float64(Int(get_physical[dt](array, physical)))


def _phys_write_from_f64[dt: DType](mut array: Array, physical: Int, value: Float64) raises:
    comptime if dt.is_floating_point():
        set_physical[dt](array, physical, Scalar[dt](value))
    else:
        set_physical[dt](array, physical, Scalar[dt](Int(value)))


def _phys_write_from_i64[dt: DType](mut array: Array, physical: Int, value: Int64) raises:
    set_physical[dt](array, physical, Scalar[dt](Int(value)))


comptime ContigFillFromPy = def[dt: DType](mut Array, PythonObject) thin raises -> None
"""Fills a c-contiguous Array of dtype `dt` with the given PythonObject value."""


comptime PhysWriteFromPy = def[dt: DType](mut Array, Int, PythonObject) thin raises -> None
"""Narrows a PythonObject to dtype `dt` and writes it to a physical offset."""


def dispatch_real_contig_fill_from_py[
    op: ContigFillFromPy,
](dtype_code: Int, mut array: Array, value_obj: PythonObject) raises -> Bool:
    if dtype_code == ArrayDType.FLOAT64.value:
        op[DType.float64](array, value_obj)
        return True
    if dtype_code == ArrayDType.FLOAT32.value:
        op[DType.float32](array, value_obj)
        return True
    if dtype_code == ArrayDType.FLOAT16.value:
        op[DType.float16](array, value_obj)
        return True
    if dtype_code == ArrayDType.INT64.value:
        op[DType.int64](array, value_obj)
        return True
    if dtype_code == ArrayDType.INT32.value:
        op[DType.int32](array, value_obj)
        return True
    if dtype_code == ArrayDType.INT16.value:
        op[DType.int16](array, value_obj)
        return True
    if dtype_code == ArrayDType.INT8.value:
        op[DType.int8](array, value_obj)
        return True
    if dtype_code == ArrayDType.UINT64.value:
        op[DType.uint64](array, value_obj)
        return True
    if dtype_code == ArrayDType.UINT32.value:
        op[DType.uint32](array, value_obj)
        return True
    if dtype_code == ArrayDType.UINT16.value:
        op[DType.uint16](array, value_obj)
        return True
    if dtype_code == ArrayDType.UINT8.value:
        op[DType.uint8](array, value_obj)
        return True
    return False


def dispatch_real_write_from_py[
    op: PhysWriteFromPy,
](dtype_code: Int, mut array: Array, physical: Int, value_obj: PythonObject) raises -> Bool:
    if dtype_code == ArrayDType.FLOAT64.value:
        op[DType.float64](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.FLOAT32.value:
        op[DType.float32](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.FLOAT16.value:
        op[DType.float16](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.INT64.value:
        op[DType.int64](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.INT32.value:
        op[DType.int32](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.INT16.value:
        op[DType.int16](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.INT8.value:
        op[DType.int8](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.UINT64.value:
        op[DType.uint64](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.UINT32.value:
        op[DType.uint32](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.UINT16.value:
        op[DType.uint16](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.UINT8.value:
        op[DType.uint8](array, physical, value_obj)
        return True
    return False


def _scalar_from_py[dt: DType](value_obj: PythonObject) raises -> Scalar[dt]:
    """Narrows a PythonObject to `Scalar[dt]`. Floats go through `Float64(py=...)`,
    integers through `Int(py=...)` — matches the manual conversions in the
    pre-refactor switches."""
    comptime if dt.is_floating_point():
        return Scalar[dt](Float64(py=value_obj))
    return Scalar[dt](Int(py=value_obj))


def _contig_fill_from_py[dt: DType](mut array: Array, value_obj: PythonObject) raises:
    var value = _scalar_from_py[dt](value_obj)
    var ptr = contiguous_ptr[dt](array)
    for i in range(array.size_value):
        ptr[i] = value


def _phys_write_from_py[dt: DType](mut array: Array, physical: Int, value_obj: PythonObject) raises:
    set_physical[dt](array, physical, _scalar_from_py[dt](value_obj))


comptime PairwiseCastKernel = def[src_dt: DType, dst_dt: DType](Array, mut Array) thin raises -> None
"""Reads each element of `src` as Scalar[src_dt], writes to `result` as Scalar[dst_dt]."""


comptime SingleDtypeContigKernel = def[dt: DType](Array, mut Array) thin raises -> None
"""Single-dtype contiguous src→result kernel. Used by bool fast paths where
one side is bool (not parametric) and the other rides DType."""


def _dispatch_dst_real_cast[
    src_dt: DType, op: PairwiseCastKernel,
](dst_code: Int, src: Array, mut result: Array) raises -> Bool:
    """Inner dispatcher: with `src_dt` already fixed, fans out over the 11 real dst dtypes."""
    if dst_code == ArrayDType.FLOAT64.value:
        op[src_dt, DType.float64](src, result)
        return True
    if dst_code == ArrayDType.FLOAT32.value:
        op[src_dt, DType.float32](src, result)
        return True
    if dst_code == ArrayDType.FLOAT16.value:
        op[src_dt, DType.float16](src, result)
        return True
    if dst_code == ArrayDType.INT64.value:
        op[src_dt, DType.int64](src, result)
        return True
    if dst_code == ArrayDType.INT32.value:
        op[src_dt, DType.int32](src, result)
        return True
    if dst_code == ArrayDType.INT16.value:
        op[src_dt, DType.int16](src, result)
        return True
    if dst_code == ArrayDType.INT8.value:
        op[src_dt, DType.int8](src, result)
        return True
    if dst_code == ArrayDType.UINT64.value:
        op[src_dt, DType.uint64](src, result)
        return True
    if dst_code == ArrayDType.UINT32.value:
        op[src_dt, DType.uint32](src, result)
        return True
    if dst_code == ArrayDType.UINT16.value:
        op[src_dt, DType.uint16](src, result)
        return True
    if dst_code == ArrayDType.UINT8.value:
        op[src_dt, DType.uint8](src, result)
        return True
    return False


def dispatch_real_pair_cast[
    op: PairwiseCastKernel,
](src_code: Int, dst_code: Int, src: Array, mut result: Array) raises -> Bool:
    """Real-real pairwise dispatch. 11×11 = 121 monomorphized cast kernels emitted
    by Mojo, fed by 22 source-level branches (11 outer + 11 inner).

    Caller invariant: `src` and `result` are both c-contiguous, and neither dtype
    is BOOL or COMPLEX64/128 (those don't ride `Scalar[dt].cast[dst_dt]()` cleanly)."""
    if src_code == ArrayDType.FLOAT64.value:
        return _dispatch_dst_real_cast[DType.float64, op](dst_code, src, result)
    if src_code == ArrayDType.FLOAT32.value:
        return _dispatch_dst_real_cast[DType.float32, op](dst_code, src, result)
    if src_code == ArrayDType.FLOAT16.value:
        return _dispatch_dst_real_cast[DType.float16, op](dst_code, src, result)
    if src_code == ArrayDType.INT64.value:
        return _dispatch_dst_real_cast[DType.int64, op](dst_code, src, result)
    if src_code == ArrayDType.INT32.value:
        return _dispatch_dst_real_cast[DType.int32, op](dst_code, src, result)
    if src_code == ArrayDType.INT16.value:
        return _dispatch_dst_real_cast[DType.int16, op](dst_code, src, result)
    if src_code == ArrayDType.INT8.value:
        return _dispatch_dst_real_cast[DType.int8, op](dst_code, src, result)
    if src_code == ArrayDType.UINT64.value:
        return _dispatch_dst_real_cast[DType.uint64, op](dst_code, src, result)
    if src_code == ArrayDType.UINT32.value:
        return _dispatch_dst_real_cast[DType.uint32, op](dst_code, src, result)
    if src_code == ArrayDType.UINT16.value:
        return _dispatch_dst_real_cast[DType.uint16, op](dst_code, src, result)
    if src_code == ArrayDType.UINT8.value:
        return _dispatch_dst_real_cast[DType.uint8, op](dst_code, src, result)
    return False


def dispatch_real_typed_contig_pair[
    op: SingleDtypeContigKernel,
](dtype_code: Int, src: Array, mut result: Array) raises -> Bool:
    """Single-DType dispatch over the 11 real dtypes. Used by bool fast paths:
    bool→real (dispatching on dst_dt) and real→bool (dispatching on src_dt)."""
    if dtype_code == ArrayDType.FLOAT64.value:
        op[DType.float64](src, result)
        return True
    if dtype_code == ArrayDType.FLOAT32.value:
        op[DType.float32](src, result)
        return True
    if dtype_code == ArrayDType.FLOAT16.value:
        op[DType.float16](src, result)
        return True
    if dtype_code == ArrayDType.INT64.value:
        op[DType.int64](src, result)
        return True
    if dtype_code == ArrayDType.INT32.value:
        op[DType.int32](src, result)
        return True
    if dtype_code == ArrayDType.INT16.value:
        op[DType.int16](src, result)
        return True
    if dtype_code == ArrayDType.INT8.value:
        op[DType.int8](src, result)
        return True
    if dtype_code == ArrayDType.UINT64.value:
        op[DType.uint64](src, result)
        return True
    if dtype_code == ArrayDType.UINT32.value:
        op[DType.uint32](src, result)
        return True
    if dtype_code == ArrayDType.UINT16.value:
        op[DType.uint16](src, result)
        return True
    if dtype_code == ArrayDType.UINT8.value:
        op[DType.uint8](src, result)
        return True
    return False


def _contig_pair_cast[src_dt: DType, dst_dt: DType](src: Array, mut result: Array) raises:
    var src_ptr = contiguous_ptr[src_dt](src)
    var dst_ptr = contiguous_ptr[dst_dt](result)
    for i in range(src.size_value):
        dst_ptr[i] = src_ptr[i].cast[dst_dt]()


def _bool_src_to_real[dst_dt: DType](src: Array, mut result: Array) raises:
    """Bool source storage is raw UInt8 with nonzero-is-true semantics — bypass
    `Scalar[DType.bool]` and read the byte directly."""
    var src_ptr = src.data + src.offset_elems
    var dst_ptr = contiguous_ptr[dst_dt](result)
    var one = Scalar[dst_dt](1)
    var zero = Scalar[dst_dt](0)
    for i in range(src.size_value):
        dst_ptr[i] = one if src_ptr[i] != UInt8(0) else zero


def _real_to_bool_dst[src_dt: DType](src: Array, mut result: Array) raises:
    """Bool dest storage is raw UInt8; write 1 for nonzero src, 0 otherwise."""
    var src_ptr = contiguous_ptr[src_dt](src)
    var dst_ptr = result.data + result.offset_elems
    var src_zero = Scalar[src_dt](0)
    for i in range(src.size_value):
        dst_ptr[i] = UInt8(1) if src_ptr[i] != src_zero else UInt8(0)


def get_physical_as_f64(array: Array, physical: Int) raises -> Float64:
    var c = array.dtype_code
    if c == ArrayDType.BOOL.value:
        return 1.0 if get_physical_bool(array, physical) else 0.0
    if c == ArrayDType.COMPLEX64.value:
        # Real part only; imag discarded. Matches numpy on real-valued
        # aggregations of complex arrays.
        return Float64(get_physical_c64_real(array, physical))
    if c == ArrayDType.COMPLEX128.value:
        return get_physical_c128_real(array, physical)
    return dispatch_real_to_f64[_phys_read_as_f64](c, array, physical)


def get_logical_as_f64(array: Array, logical: Int) raises -> Float64:
    return get_physical_as_f64(array, physical_offset(array, logical))


def set_physical_from_f64(mut array: Array, physical: Int, value: Float64) raises:
    var c = array.dtype_code
    if c == ArrayDType.BOOL.value:
        array.data[physical] = UInt8(1) if value != 0.0 else UInt8(0)
        return
    if c == ArrayDType.COMPLEX64.value:
        # Real part receives `value`; imag zeroed. Matches numpy's
        # f64-to-complex assignment.
        var ptr = array.data.bitcast[Float32]()
        ptr[physical * 2] = Float32(value)
        ptr[physical * 2 + 1] = 0.0
        return
    if c == ArrayDType.COMPLEX128.value:
        var ptr = array.data.bitcast[Float64]()
        ptr[physical * 2] = value
        ptr[physical * 2 + 1] = 0.0
        return
    if dispatch_real_write_f64[_phys_write_from_f64](c, array, physical, value):
        return
    # Unknown / not-yet-dispatched dtype: write as float64 (matches old fallback).
    set_physical[DType.float64](array, physical, value)


def set_logical_from_f64(mut array: Array, logical: Int, value: Float64) raises:
    set_physical_from_f64(array, physical_offset(array, logical), value)


def set_logical_from_i64(mut array: Array, logical: Int, value: Int64) raises:
    var physical = physical_offset(array, logical)
    if dispatch_int_write_i64[_phys_write_from_i64](array.dtype_code, array, physical, value):
        return
    # Non-int dtype: round-trip through f64 (preserves prior behavior).
    set_physical_from_f64(array, physical, Float64(value))


def set_logical_from_py(mut array: Array, logical: Int, value_obj: PythonObject) raises:
    var c = array.dtype_code
    var physical = physical_offset(array, logical)
    if c == ArrayDType.BOOL.value:
        array.data[physical] = UInt8(1) if Bool(py=value_obj) else UInt8(0)
        return
    if c == ArrayDType.COMPLEX64.value or c == ArrayDType.COMPLEX128.value:
        # Accept python complex or real numbers. Real → imaginary=0.
        var real_val = Float64(py=value_obj.real)
        var imag_val = Float64(py=value_obj.imag)
        if c == ArrayDType.COMPLEX64.value:
            var ptr = array.data.bitcast[Float32]()
            ptr[physical * 2] = Float32(real_val)
            ptr[physical * 2 + 1] = Float32(imag_val)
        else:
            var ptr = array.data.bitcast[Float64]()
            ptr[physical * 2] = real_val
            ptr[physical * 2 + 1] = imag_val
        return
    if dispatch_real_write_from_py[_phys_write_from_py](c, array, physical, value_obj):
        return
    # Unknown dtype: write as float64 (matches old fallback).
    set_physical[DType.float64](array, physical, Float64(py=value_obj))


def scalar_py_as_f64(value_obj: PythonObject, dtype_code: Int) raises -> Float64:
    if dtype_code == ArrayDType.BOOL.value:
        if Bool(py=value_obj):
            return 1.0
        return 0.0
    if (
        dtype_code == ArrayDType.INT64.value
        or dtype_code == ArrayDType.INT32.value
        or dtype_code == ArrayDType.INT16.value
        or dtype_code == ArrayDType.INT8.value
        or dtype_code == ArrayDType.UINT64.value
        or dtype_code == ArrayDType.UINT32.value
        or dtype_code == ArrayDType.UINT16.value
        or dtype_code == ArrayDType.UINT8.value
    ):
        return Float64(Int(py=value_obj))
    return Float64(py=value_obj)


def fill_all_from_py(mut array: Array, value_obj: PythonObject) raises:
    if is_c_contiguous(array):
        var c = array.dtype_code
        if c == ArrayDType.BOOL.value:
            var value = UInt8(1) if Bool(py=value_obj) else UInt8(0)
            var ptr = array.data + array.offset_elems
            for i in range(array.size_value):
                ptr[i] = value
            return
        if c == ArrayDType.COMPLEX64.value:
            var real = Float32(Float64(py=value_obj.real))
            var imag = Float32(Float64(py=value_obj.imag))
            var ptr = array.data.bitcast[Float32]() + array.offset_elems * 2
            for i in range(array.size_value):
                ptr[i * 2] = real
                ptr[i * 2 + 1] = imag
            return
        if c == ArrayDType.COMPLEX128.value:
            var real = Float64(py=value_obj.real)
            var imag = Float64(py=value_obj.imag)
            var ptr = array.data.bitcast[Float64]() + array.offset_elems * 2
            for i in range(array.size_value):
                ptr[i * 2] = real
                ptr[i * 2 + 1] = imag
            return
        if dispatch_real_contig_fill_from_py[_contig_fill_from_py](c, array, value_obj):
            return
    for i in range(array.size_value):
        set_logical_from_py(array, i, value_obj)


def contiguous_as_f64(array: Array, index: Int) raises -> Float64:
    if array.dtype_code == ArrayDType.FLOAT32.value:
        return Float64(contiguous_ptr[DType.float32](array)[index])
    if array.dtype_code == ArrayDType.FLOAT64.value:
        return contiguous_ptr[DType.float64](array)[index]
    return get_physical_as_f64(array, array.offset_elems + index)


def set_contiguous_from_f64(mut array: Array, index: Int, value: Float64) raises:
    if array.dtype_code == ArrayDType.FLOAT32.value:
        contiguous_ptr[DType.float32](array)[index] = Float32(value)
    elif array.dtype_code == ArrayDType.FLOAT64.value:
        contiguous_ptr[DType.float64](array)[index] = value
    else:
        set_physical_from_f64(array, array.offset_elems + index, value)


def _copy_rank2_strided_typed[dtype: DType](src: Array, mut result: Array) raises:
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_row_stride = src.strides[0]
    var src_col_stride = src.strides[1]
    var result_row_stride = result.strides[0]
    var result_col_stride = result.strides[1]
    var src_data = src.data.bitcast[Scalar[dtype]]()
    var result_data = result.data.bitcast[Scalar[dtype]]()
    comptime tile = 8
    var row_block = 0
    while row_block < rows:
        var row_end = row_block + tile
        if row_end > rows:
            row_end = rows
        var col_block = 0
        while col_block < cols:
            var col_end = col_block + tile
            if col_end > cols:
                col_end = cols
            if src_row_stride == 1:
                var col = col_block
                while col < col_end:
                    var row = row_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while row < row_end:
                        result_data[result_index] = src_data[src_index]
                        row += 1
                        src_index += src_row_stride
                        result_index += result_row_stride
                    col += 1
            else:
                var row = row_block
                while row < row_end:
                    var col = col_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while col < col_end:
                        result_data[result_index] = src_data[src_index]
                        col += 1
                        src_index += src_col_stride
                        result_index += result_col_stride
                    row += 1
            col_block += tile
        row_block += tile


def _copy_rank2_strided_bool(src: Array, mut result: Array) raises:
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_row_stride = src.strides[0]
    var src_col_stride = src.strides[1]
    var result_row_stride = result.strides[0]
    var result_col_stride = result.strides[1]
    var src_data = src.data.bitcast[Bool]()
    var result_data = result.data.bitcast[Bool]()
    comptime tile = 8
    var row_block = 0
    while row_block < rows:
        var row_end = row_block + tile
        if row_end > rows:
            row_end = rows
        var col_block = 0
        while col_block < cols:
            var col_end = col_block + tile
            if col_end > cols:
                col_end = cols
            if src_row_stride == 1:
                var col = col_block
                while col < col_end:
                    var row = row_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while row < row_end:
                        result_data[result_index] = src_data[src_index]
                        row += 1
                        src_index += src_row_stride
                        result_index += result_row_stride
                    col += 1
            else:
                var row = row_block
                while row < row_end:
                    var col = col_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while col < col_end:
                        result_data[result_index] = src_data[src_index]
                        col += 1
                        src_index += src_col_stride
                        result_index += result_col_stride
                    row += 1
            col_block += tile
        row_block += tile


def _copy_rank2_strided_complex32(src: Array, mut result: Array) raises:
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_row_stride = src.strides[0]
    var src_col_stride = src.strides[1]
    var result_row_stride = result.strides[0]
    var result_col_stride = result.strides[1]
    var src_data = src.data.bitcast[Float32]()
    var result_data = result.data.bitcast[Float32]()
    comptime tile = 8
    var row_block = 0
    while row_block < rows:
        var row_end = row_block + tile
        if row_end > rows:
            row_end = rows
        var col_block = 0
        while col_block < cols:
            var col_end = col_block + tile
            if col_end > cols:
                col_end = cols
            if src_row_stride == 1:
                var col = col_block
                while col < col_end:
                    var row = row_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while row < row_end:
                        result_data[result_index * 2] = src_data[src_index * 2]
                        result_data[result_index * 2 + 1] = src_data[src_index * 2 + 1]
                        row += 1
                        src_index += src_row_stride
                        result_index += result_row_stride
                    col += 1
            else:
                var row = row_block
                while row < row_end:
                    var col = col_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while col < col_end:
                        result_data[result_index * 2] = src_data[src_index * 2]
                        result_data[result_index * 2 + 1] = src_data[src_index * 2 + 1]
                        col += 1
                        src_index += src_col_stride
                        result_index += result_col_stride
                    row += 1
            col_block += tile
        row_block += tile


def _copy_rank2_strided_complex64(src: Array, mut result: Array) raises:
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_row_stride = src.strides[0]
    var src_col_stride = src.strides[1]
    var result_row_stride = result.strides[0]
    var result_col_stride = result.strides[1]
    var src_data = src.data.bitcast[Float64]()
    var result_data = result.data.bitcast[Float64]()
    comptime tile = 8
    var row_block = 0
    while row_block < rows:
        var row_end = row_block + tile
        if row_end > rows:
            row_end = rows
        var col_block = 0
        while col_block < cols:
            var col_end = col_block + tile
            if col_end > cols:
                col_end = cols
            if src_row_stride == 1:
                var col = col_block
                while col < col_end:
                    var row = row_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while row < row_end:
                        result_data[result_index * 2] = src_data[src_index * 2]
                        result_data[result_index * 2 + 1] = src_data[src_index * 2 + 1]
                        row += 1
                        src_index += src_row_stride
                        result_index += result_row_stride
                    col += 1
            else:
                var row = row_block
                while row < row_end:
                    var col = col_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while col < col_end:
                        result_data[result_index * 2] = src_data[src_index * 2]
                        result_data[result_index * 2 + 1] = src_data[src_index * 2 + 1]
                        col += 1
                        src_index += src_col_stride
                        result_index += result_col_stride
                    row += 1
            col_block += tile
        row_block += tile


def _maybe_copy_rank2_strided(src: Array, mut result: Array) raises -> Bool:
    if len(src.shape) != 2:
        return False
    if src.size_value == 0:
        return True
    if src.dtype_code == ArrayDType.BOOL.value:
        _copy_rank2_strided_bool(src, result)
        return True
    if src.dtype_code == ArrayDType.INT8.value:
        _copy_rank2_strided_typed[DType.int8](src, result)
        return True
    if src.dtype_code == ArrayDType.INT16.value:
        _copy_rank2_strided_typed[DType.int16](src, result)
        return True
    if src.dtype_code == ArrayDType.INT32.value:
        _copy_rank2_strided_typed[DType.int32](src, result)
        return True
    if src.dtype_code == ArrayDType.INT64.value:
        _copy_rank2_strided_typed[DType.int64](src, result)
        return True
    if src.dtype_code == ArrayDType.UINT8.value:
        _copy_rank2_strided_typed[DType.uint8](src, result)
        return True
    if src.dtype_code == ArrayDType.UINT16.value:
        _copy_rank2_strided_typed[DType.uint16](src, result)
        return True
    if src.dtype_code == ArrayDType.UINT32.value:
        _copy_rank2_strided_typed[DType.uint32](src, result)
        return True
    if src.dtype_code == ArrayDType.UINT64.value:
        _copy_rank2_strided_typed[DType.uint64](src, result)
        return True
    if src.dtype_code == ArrayDType.FLOAT16.value:
        _copy_rank2_strided_typed[DType.float16](src, result)
        return True
    if src.dtype_code == ArrayDType.FLOAT32.value:
        _copy_rank2_strided_typed[DType.float32](src, result)
        return True
    if src.dtype_code == ArrayDType.FLOAT64.value:
        _copy_rank2_strided_typed[DType.float64](src, result)
        return True
    if src.dtype_code == ArrayDType.COMPLEX64.value:
        _copy_rank2_strided_complex32(src, result)
        return True
    if src.dtype_code == ArrayDType.COMPLEX128.value:
        _copy_rank2_strided_complex64(src, result)
        return True
    return False


def copy_c_contiguous(src: Array) raises -> Array:
    var shape = clone_int_list(src.shape)
    var result = make_empty_array(src.dtype_code, shape^)
    if is_c_contiguous(src):
        var item_bytes = item_size(src.dtype_code)
        var src_byte_offset = src.offset_elems * item_bytes
        var byte_count = src.size_value * item_bytes
        memcpy(
            dest=result.data,
            src=src.data + src_byte_offset,
            count=byte_count,
        )
        return result^
    if _maybe_copy_rank2_strided(src, result):
        return result^
    for i in range(src.size_value):
        var physical = physical_offset(src, i)
        if src.dtype_code == ArrayDType.BOOL.value:
            if get_physical_bool(src, physical):
                set_logical_from_f64(result, i, 1.0)
            else:
                set_logical_from_f64(result, i, 0.0)
        elif src.dtype_code == ArrayDType.INT64.value:
            set_logical_from_i64(result, i, get_physical[DType.int64](src, physical))
        elif src.dtype_code == ArrayDType.FLOAT32.value:
            set_logical_from_f64(result, i, Float64(get_physical[DType.float32](src, physical)))
        else:
            set_logical_from_f64(result, i, get_physical[DType.float64](src, physical))
    return result^


def _maybe_cast_contiguous_core_dtypes(src: Array, mut result: Array) raises -> Bool:
    """Fast path for c-contiguous → c-contiguous casts.

    Three layers, in order of fall-through:
    1. Bool ↔ real (via `dispatch_real_typed_contig_pair`): UInt8 storage with
       nonzero-is-true semantics is bespoke per-byte; the dispatcher fans out
       on the *real* side. Covers 22 pairs (11 src→bool + bool→11 dst).
    2. Real ↔ real (via `dispatch_real_pair_cast`): 121 pairs through
       `Scalar[src_dt] → Scalar[dst_dt].cast[dst_dt]()`.
    3. Anything involving complex returns False; caller's per-element loop in
       `cast_copy_array` handles complex (interleaved (re,im) doesn't ride DType).
    """
    if not is_c_contiguous(src) or not is_c_contiguous(result):
        return False
    var src_c = src.dtype_code
    var dst_c = result.dtype_code
    if src_c == ArrayDType.BOOL.value:
        return dispatch_real_typed_contig_pair[_bool_src_to_real](dst_c, src, result)
    if dst_c == ArrayDType.BOOL.value:
        return dispatch_real_typed_contig_pair[_real_to_bool_dst](src_c, src, result)
    var src_is_complex = src_c == ArrayDType.COMPLEX64.value or src_c == ArrayDType.COMPLEX128.value
    var dst_is_complex = dst_c == ArrayDType.COMPLEX64.value or dst_c == ArrayDType.COMPLEX128.value
    if src_is_complex or dst_is_complex:
        return False
    return dispatch_real_pair_cast[_contig_pair_cast](src_c, dst_c, src, result)


def cast_copy_array(src: Array, dtype_code: Int) raises -> Array:
    if src.dtype_code == dtype_code:
        return copy_c_contiguous(src)
    var shape = clone_int_list(src.shape)
    var result = make_empty_array(dtype_code, shape^)
    if _maybe_cast_contiguous_core_dtypes(src, result):
        return result^
    var dst_is_complex = dtype_code == ArrayDType.COMPLEX64.value or dtype_code == ArrayDType.COMPLEX128.value
    var src_is_c = is_c_contiguous(src)
    for i in range(src.size_value):
        var physical = src.offset_elems + i
        if not src_is_c:
            physical = physical_offset(src, i)
        if src.dtype_code == ArrayDType.COMPLEX64.value:
            var re = Float64(get_physical_c64_real(src, physical))
            var im = Float64(get_physical_c64_imag(src, physical))
            if dst_is_complex:
                if dtype_code == ArrayDType.COMPLEX64.value:
                    set_physical_c64(result, i, Float32(re), Float32(im))
                else:
                    set_physical_c128(result, i, re, im)
            else:
                set_logical_from_f64(result, i, re)  # numpy drops imag
            continue
        if src.dtype_code == ArrayDType.COMPLEX128.value:
            var re = get_physical_c128_real(src, physical)
            var im = get_physical_c128_imag(src, physical)
            if dst_is_complex:
                if dtype_code == ArrayDType.COMPLEX64.value:
                    set_physical_c64(result, i, Float32(re), Float32(im))
                else:
                    set_physical_c128(result, i, re, im)
            else:
                set_logical_from_f64(result, i, re)
            continue
        # Real → anything (including complex). Read source real value and
        # write through the unified setters (which zero imag for complex).
        if src.dtype_code == ArrayDType.BOOL.value:
            if get_physical_bool(src, physical):
                set_logical_from_i64(result, i, 1)
            else:
                set_logical_from_i64(result, i, 0)
        elif src.dtype_code == ArrayDType.INT64.value:
            set_logical_from_i64(result, i, get_physical[DType.int64](src, physical))
        elif src.dtype_code == ArrayDType.INT32.value:
            set_logical_from_i64(result, i, Int64(Int(get_physical[DType.int32](src, physical))))
        elif src.dtype_code == ArrayDType.INT16.value:
            set_logical_from_i64(result, i, Int64(Int(get_physical[DType.int16](src, physical))))
        elif src.dtype_code == ArrayDType.INT8.value:
            set_logical_from_i64(result, i, Int64(Int(get_physical[DType.int8](src, physical))))
        elif src.dtype_code == ArrayDType.UINT64.value:
            set_logical_from_f64(result, i, Float64(Int(get_physical[DType.uint64](src, physical))))
        elif src.dtype_code == ArrayDType.UINT32.value:
            set_logical_from_i64(result, i, Int64(Int(get_physical[DType.uint32](src, physical))))
        elif src.dtype_code == ArrayDType.UINT16.value:
            set_logical_from_i64(result, i, Int64(Int(get_physical[DType.uint16](src, physical))))
        elif src.dtype_code == ArrayDType.UINT8.value:
            set_logical_from_i64(result, i, Int64(Int(get_physical[DType.uint8](src, physical))))
        elif src.dtype_code == ArrayDType.FLOAT32.value:
            set_logical_from_f64(result, i, Float64(get_physical[DType.float32](src, physical)))
        elif src.dtype_code == ArrayDType.FLOAT16.value:
            set_logical_from_f64(result, i, Float64(get_physical[DType.float16](src, physical)))
        elif src.dtype_code == ArrayDType.FLOAT64.value:
            set_logical_from_f64(result, i, get_physical[DType.float64](src, physical))
        else:
            raise Error("unsupported dtype code")
    return result^


def result_dtype_for_unary(dtype_code: Int) -> Int:
    return dtype_result_for_unary(dtype_code)


def result_dtype_for_unary_preserve(dtype_code: Int) -> Int:
    if dtype_code == ArrayDType.BOOL.value:
        return ArrayDType.INT64.value
    return dtype_code


def result_dtype_for_binary(lhs_dtype: Int, rhs_dtype: Int, op: Int) -> Int:
    return dtype_result_for_binary(lhs_dtype, rhs_dtype, op)


def result_dtype_for_reduction(dtype_code: Int, op: Int) -> Int:
    return dtype_result_for_reduction(dtype_code, op)


def result_dtype_for_linalg(dtype_code: Int) -> Int:
    return dtype_result_for_linalg(dtype_code)


def result_dtype_for_linalg_binary(lhs_dtype: Int, rhs_dtype: Int) -> Int:
    return dtype_result_for_linalg_binary(lhs_dtype, rhs_dtype)


def broadcast_shape(lhs: Array, rhs: Array) raises -> List[Int]:
    var lhs_ndim = len(lhs.shape)
    var rhs_ndim = len(rhs.shape)
    var out_ndim = lhs_ndim
    if rhs_ndim > out_ndim:
        out_ndim = rhs_ndim
    var shape = List[Int]()
    for _ in range(out_ndim):
        shape.append(1)
    for out_axis in range(out_ndim - 1, -1, -1):
        var lhs_axis = out_axis - (out_ndim - lhs_ndim)
        var rhs_axis = out_axis - (out_ndim - rhs_ndim)
        var lhs_dim = 1
        var rhs_dim = 1
        if lhs_axis >= 0:
            lhs_dim = lhs.shape[lhs_axis]
        if rhs_axis >= 0:
            rhs_dim = rhs.shape[rhs_axis]
        if lhs_dim == rhs_dim:
            shape[out_axis] = lhs_dim
        elif lhs_dim == 1:
            shape[out_axis] = rhs_dim
        elif rhs_dim == 1:
            shape[out_axis] = lhs_dim
        else:
            raise Error("operands could not be broadcast together")
    return shape^
