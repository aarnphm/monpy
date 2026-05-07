from std.collections import List
from std.memory import memcpy
from std.os import abort
from std.python import PythonObject

from domain import (
    BACKEND_ACCELERATE,
    BACKEND_FUSED,
    BACKEND_GENERIC,
    DTYPE_BOOL,
    DTYPE_FLOAT16,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT16,
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_INT8,
    DTYPE_UINT16,
    DTYPE_UINT32,
    DTYPE_UINT64,
    DTYPE_UINT8,
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
        return PythonObject(self_ptr[].backend_code == BACKEND_ACCELERATE)

    @staticmethod
    def used_fused_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].backend_code == BACKEND_FUSED)

    @staticmethod
    def backend_code_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].backend_code)

    @staticmethod
    def get_scalar_py(py_self: PythonObject, index_obj: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        var index = Int(py=index_obj)
        if self_ptr[].dtype_code == DTYPE_BOOL:
            return PythonObject(get_physical_bool(self_ptr[], physical_offset(self_ptr[], index)))
        if self_ptr[].dtype_code == DTYPE_INT64:
            return PythonObject(get_physical_i64(self_ptr[], physical_offset(self_ptr[], index)))
        if self_ptr[].dtype_code == DTYPE_INT32:
            return PythonObject(Int(get_physical_i32(self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == DTYPE_INT16:
            return PythonObject(Int(get_physical_i16(self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == DTYPE_INT8:
            return PythonObject(Int(get_physical_i8(self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == DTYPE_UINT64:
            return PythonObject(Int(get_physical_u64(self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == DTYPE_UINT32:
            return PythonObject(Int(get_physical_u32(self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == DTYPE_UINT16:
            return PythonObject(Int(get_physical_u16(self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == DTYPE_UINT8:
            return PythonObject(Int(get_physical_u8(self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == DTYPE_FLOAT32:
            return PythonObject(get_physical_f32(self_ptr[], physical_offset(self_ptr[], index)))
        if self_ptr[].dtype_code == DTYPE_FLOAT16:
            return PythonObject(Float64(get_physical_f16(self_ptr[], physical_offset(self_ptr[], index))))
        return PythonObject(get_physical_f64(self_ptr[], physical_offset(self_ptr[], index)))

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
        BACKEND_GENERIC,
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
        BACKEND_GENERIC,
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
# Array ↔ Layout adapter (phase 2 entry point).
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


def array_with_layout(
    source: Array, new_layout: Layout, offset_delta: Int = 0
) raises -> Array:
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
    return make_view_array(
        source, flat_shape^, flat_stride^, size_value, new_offset
    )


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


def broadcast_physical_offset(array: Array, logical: Int, out_shape: List[Int]) raises -> Int:
    var out_ndim = len(out_shape)
    var array_ndim = len(array.shape)
    var physical = array.offset_elems
    var remainder = logical
    for out_axis in range(out_ndim - 1, -1, -1):
        var dim = out_shape[out_axis]
        var coord = 0
        if dim != 0:
            coord = remainder % dim
            remainder = remainder // dim
        var array_axis = out_axis - (out_ndim - array_ndim)
        if array_axis >= 0:
            if array.shape[array_axis] == 1:
                coord = 0
            physical += coord * array.strides[array_axis]
    return physical


def get_physical_bool(array: Array, physical: Int) raises -> Bool:
    return array.data[physical] != UInt8(0)


def get_physical_i64(array: Array, physical: Int) raises -> Int64:
    return array.data.bitcast[Int64]()[physical]


def get_physical_i32(array: Array, physical: Int) raises -> Int32:
    return array.data.bitcast[Int32]()[physical]


def get_physical_i16(array: Array, physical: Int) raises -> Int16:
    return array.data.bitcast[Int16]()[physical]


def get_physical_i8(array: Array, physical: Int) raises -> Int8:
    return array.data.bitcast[Int8]()[physical]


def get_physical_u64(array: Array, physical: Int) raises -> UInt64:
    return array.data.bitcast[UInt64]()[physical]


def get_physical_u32(array: Array, physical: Int) raises -> UInt32:
    return array.data.bitcast[UInt32]()[physical]


def get_physical_u16(array: Array, physical: Int) raises -> UInt16:
    return array.data.bitcast[UInt16]()[physical]


def get_physical_u8(array: Array, physical: Int) raises -> UInt8:
    return array.data.bitcast[UInt8]()[physical]


def get_physical_f32(array: Array, physical: Int) raises -> Float32:
    return array.data.bitcast[Float32]()[physical]


def get_physical_f64(array: Array, physical: Int) raises -> Float64:
    return array.data.bitcast[Float64]()[physical]


def get_physical_f16(array: Array, physical: Int) raises -> Float16:
    return array.data.bitcast[Float16]()[physical]


def get_physical_as_f64(array: Array, physical: Int) raises -> Float64:
    if array.dtype_code == DTYPE_BOOL:
        if get_physical_bool(array, physical):
            return 1.0
        return 0.0
    if array.dtype_code == DTYPE_INT64:
        return Float64(get_physical_i64(array, physical))
    if array.dtype_code == DTYPE_INT32:
        return Float64(Int(get_physical_i32(array, physical)))
    if array.dtype_code == DTYPE_INT16:
        return Float64(Int(get_physical_i16(array, physical)))
    if array.dtype_code == DTYPE_INT8:
        return Float64(Int(get_physical_i8(array, physical)))
    if array.dtype_code == DTYPE_UINT64:
        return Float64(Int(get_physical_u64(array, physical)))
    if array.dtype_code == DTYPE_UINT32:
        return Float64(Int(get_physical_u32(array, physical)))
    if array.dtype_code == DTYPE_UINT16:
        return Float64(Int(get_physical_u16(array, physical)))
    if array.dtype_code == DTYPE_UINT8:
        return Float64(Int(get_physical_u8(array, physical)))
    if array.dtype_code == DTYPE_FLOAT32:
        return Float64(get_physical_f32(array, physical))
    if array.dtype_code == DTYPE_FLOAT16:
        return Float64(get_physical_f16(array, physical))
    return get_physical_f64(array, physical)


def get_logical_as_f64(array: Array, logical: Int) raises -> Float64:
    return get_physical_as_f64(array, physical_offset(array, logical))


def get_broadcast_as_f64(array: Array, logical: Int, out_shape: List[Int]) raises -> Float64:
    return get_physical_as_f64(array, broadcast_physical_offset(array, logical, out_shape))


def get_broadcast_as_bool(array: Array, logical: Int, out_shape: List[Int]) raises -> Bool:
    var physical = broadcast_physical_offset(array, logical, out_shape)
    if array.dtype_code == DTYPE_BOOL:
        return get_physical_bool(array, physical)
    return get_physical_as_f64(array, physical) != 0.0


def set_physical_from_f64(mut array: Array, physical: Int, value: Float64) raises:
    if array.dtype_code == DTYPE_BOOL:
        if value != 0.0:
            array.data[physical] = UInt8(1)
        else:
            array.data[physical] = UInt8(0)
    elif array.dtype_code == DTYPE_INT64:
        array.data.bitcast[Int64]()[physical] = Int64(value)
    elif array.dtype_code == DTYPE_INT32:
        array.data.bitcast[Int32]()[physical] = Int32(Int(value))
    elif array.dtype_code == DTYPE_INT16:
        array.data.bitcast[Int16]()[physical] = Int16(Int(value))
    elif array.dtype_code == DTYPE_INT8:
        array.data.bitcast[Int8]()[physical] = Int8(Int(value))
    elif array.dtype_code == DTYPE_UINT64:
        array.data.bitcast[UInt64]()[physical] = UInt64(Int(value))
    elif array.dtype_code == DTYPE_UINT32:
        array.data.bitcast[UInt32]()[physical] = UInt32(Int(value))
    elif array.dtype_code == DTYPE_UINT16:
        array.data.bitcast[UInt16]()[physical] = UInt16(Int(value))
    elif array.dtype_code == DTYPE_UINT8:
        array.data.bitcast[UInt8]()[physical] = UInt8(Int(value))
    elif array.dtype_code == DTYPE_FLOAT32:
        array.data.bitcast[Float32]()[physical] = Float32(value)
    elif array.dtype_code == DTYPE_FLOAT16:
        array.data.bitcast[Float16]()[physical] = Float16(value)
    else:
        array.data.bitcast[Float64]()[physical] = value


def set_logical_from_f64(mut array: Array, logical: Int, value: Float64) raises:
    set_physical_from_f64(array, physical_offset(array, logical), value)


def set_logical_from_i64(mut array: Array, logical: Int, value: Int64) raises:
    if array.dtype_code == DTYPE_INT64:
        array.data.bitcast[Int64]()[physical_offset(array, logical)] = value
    elif array.dtype_code == DTYPE_INT32:
        array.data.bitcast[Int32]()[physical_offset(array, logical)] = Int32(Int(value))
    elif array.dtype_code == DTYPE_INT16:
        array.data.bitcast[Int16]()[physical_offset(array, logical)] = Int16(Int(value))
    elif array.dtype_code == DTYPE_INT8:
        array.data.bitcast[Int8]()[physical_offset(array, logical)] = Int8(Int(value))
    elif array.dtype_code == DTYPE_UINT64:
        array.data.bitcast[UInt64]()[physical_offset(array, logical)] = UInt64(Int(value))
    elif array.dtype_code == DTYPE_UINT32:
        array.data.bitcast[UInt32]()[physical_offset(array, logical)] = UInt32(Int(value))
    elif array.dtype_code == DTYPE_UINT16:
        array.data.bitcast[UInt16]()[physical_offset(array, logical)] = UInt16(Int(value))
    elif array.dtype_code == DTYPE_UINT8:
        array.data.bitcast[UInt8]()[physical_offset(array, logical)] = UInt8(Int(value))
    else:
        set_logical_from_f64(array, logical, Float64(value))


def set_logical_from_py(mut array: Array, logical: Int, value_obj: PythonObject) raises:
    var physical = physical_offset(array, logical)
    if array.dtype_code == DTYPE_BOOL:
        if Bool(py=value_obj):
            array.data[physical] = UInt8(1)
        else:
            array.data[physical] = UInt8(0)
    elif array.dtype_code == DTYPE_INT64:
        array.data.bitcast[Int64]()[physical] = Int64(Int(py=value_obj))
    elif array.dtype_code == DTYPE_INT32:
        array.data.bitcast[Int32]()[physical] = Int32(Int(py=value_obj))
    elif array.dtype_code == DTYPE_INT16:
        array.data.bitcast[Int16]()[physical] = Int16(Int(py=value_obj))
    elif array.dtype_code == DTYPE_INT8:
        array.data.bitcast[Int8]()[physical] = Int8(Int(py=value_obj))
    elif array.dtype_code == DTYPE_UINT64:
        array.data.bitcast[UInt64]()[physical] = UInt64(Int(py=value_obj))
    elif array.dtype_code == DTYPE_UINT32:
        array.data.bitcast[UInt32]()[physical] = UInt32(Int(py=value_obj))
    elif array.dtype_code == DTYPE_UINT16:
        array.data.bitcast[UInt16]()[physical] = UInt16(Int(py=value_obj))
    elif array.dtype_code == DTYPE_UINT8:
        array.data.bitcast[UInt8]()[physical] = UInt8(Int(py=value_obj))
    elif array.dtype_code == DTYPE_FLOAT32:
        array.data.bitcast[Float32]()[physical] = Float32(Float64(py=value_obj))
    elif array.dtype_code == DTYPE_FLOAT16:
        array.data.bitcast[Float16]()[physical] = Float16(Float64(py=value_obj))
    else:
        array.data.bitcast[Float64]()[physical] = Float64(py=value_obj)


def scalar_py_as_f64(value_obj: PythonObject, dtype_code: Int) raises -> Float64:
    if dtype_code == DTYPE_BOOL:
        if Bool(py=value_obj):
            return 1.0
        return 0.0
    if (
        dtype_code == DTYPE_INT64
        or dtype_code == DTYPE_INT32
        or dtype_code == DTYPE_INT16
        or dtype_code == DTYPE_INT8
        or dtype_code == DTYPE_UINT64
        or dtype_code == DTYPE_UINT32
        or dtype_code == DTYPE_UINT16
        or dtype_code == DTYPE_UINT8
    ):
        return Float64(Int(py=value_obj))
    return Float64(py=value_obj)


def fill_all_from_py(mut array: Array, value_obj: PythonObject) raises:
    for i in range(array.size_value):
        set_logical_from_py(array, i, value_obj)


def contiguous_f32_ptr(
    array: Array,
) -> UnsafePointer[Float32, MutExternalOrigin]:
    return array.data.bitcast[Float32]() + array.offset_elems


def contiguous_f64_ptr(
    array: Array,
) -> UnsafePointer[Float64, MutExternalOrigin]:
    return array.data.bitcast[Float64]() + array.offset_elems


def contiguous_i32_ptr(
    array: Array,
) -> UnsafePointer[Int32, MutExternalOrigin]:
    return array.data.bitcast[Int32]() + array.offset_elems


def contiguous_i64_ptr(
    array: Array,
) -> UnsafePointer[Int64, MutExternalOrigin]:
    return array.data.bitcast[Int64]() + array.offset_elems


def contiguous_u32_ptr(
    array: Array,
) -> UnsafePointer[UInt32, MutExternalOrigin]:
    return array.data.bitcast[UInt32]() + array.offset_elems


def contiguous_u64_ptr(
    array: Array,
) -> UnsafePointer[UInt64, MutExternalOrigin]:
    return array.data.bitcast[UInt64]() + array.offset_elems


def contiguous_f16_ptr(
    array: Array,
) -> UnsafePointer[Float16, MutExternalOrigin]:
    return array.data.bitcast[Float16]() + array.offset_elems


def contiguous_as_f64(array: Array, index: Int) raises -> Float64:
    if array.dtype_code == DTYPE_FLOAT32:
        return Float64(contiguous_f32_ptr(array)[index])
    if array.dtype_code == DTYPE_FLOAT64:
        return contiguous_f64_ptr(array)[index]
    return get_physical_as_f64(array, array.offset_elems + index)


def set_contiguous_from_f64(mut array: Array, index: Int, value: Float64) raises:
    if array.dtype_code == DTYPE_FLOAT32:
        contiguous_f32_ptr(array)[index] = Float32(value)
    elif array.dtype_code == DTYPE_FLOAT64:
        contiguous_f64_ptr(array)[index] = value
    else:
        set_physical_from_f64(array, array.offset_elems + index, value)


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
    for i in range(src.size_value):
        var physical = physical_offset(src, i)
        if src.dtype_code == DTYPE_BOOL:
            if get_physical_bool(src, physical):
                set_logical_from_f64(result, i, 1.0)
            else:
                set_logical_from_f64(result, i, 0.0)
        elif src.dtype_code == DTYPE_INT64:
            set_logical_from_i64(result, i, get_physical_i64(src, physical))
        elif src.dtype_code == DTYPE_FLOAT32:
            set_logical_from_f64(result, i, Float64(get_physical_f32(src, physical)))
        else:
            set_logical_from_f64(result, i, get_physical_f64(src, physical))
    return result^


def cast_copy_array(src: Array, dtype_code: Int) raises -> Array:
    if src.dtype_code == dtype_code:
        return copy_c_contiguous(src)
    var shape = clone_int_list(src.shape)
    var result = make_empty_array(dtype_code, shape^)
    for i in range(src.size_value):
        var physical = physical_offset(src, i)
        if src.dtype_code == DTYPE_BOOL:
            if get_physical_bool(src, physical):
                set_logical_from_i64(result, i, 1)
            else:
                set_logical_from_i64(result, i, 0)
        elif src.dtype_code == DTYPE_INT64:
            set_logical_from_i64(result, i, get_physical_i64(src, physical))
        elif src.dtype_code == DTYPE_INT32:
            set_logical_from_i64(result, i, Int64(Int(get_physical_i32(src, physical))))
        elif src.dtype_code == DTYPE_INT16:
            set_logical_from_i64(result, i, Int64(Int(get_physical_i16(src, physical))))
        elif src.dtype_code == DTYPE_INT8:
            set_logical_from_i64(result, i, Int64(Int(get_physical_i8(src, physical))))
        elif src.dtype_code == DTYPE_UINT64:
            set_logical_from_f64(result, i, Float64(Int(get_physical_u64(src, physical))))
        elif src.dtype_code == DTYPE_UINT32:
            set_logical_from_i64(result, i, Int64(Int(get_physical_u32(src, physical))))
        elif src.dtype_code == DTYPE_UINT16:
            set_logical_from_i64(result, i, Int64(Int(get_physical_u16(src, physical))))
        elif src.dtype_code == DTYPE_UINT8:
            set_logical_from_i64(result, i, Int64(Int(get_physical_u8(src, physical))))
        elif src.dtype_code == DTYPE_FLOAT32:
            set_logical_from_f64(result, i, Float64(get_physical_f32(src, physical)))
        elif src.dtype_code == DTYPE_FLOAT16:
            set_logical_from_f64(result, i, Float64(get_physical_f16(src, physical)))
        elif src.dtype_code == DTYPE_FLOAT64:
            set_logical_from_f64(result, i, get_physical_f64(src, physical))
        else:
            raise Error("unsupported dtype code")
    return result^


def result_dtype_for_unary(dtype_code: Int) -> Int:
    return dtype_result_for_unary(dtype_code)


def result_dtype_for_unary_preserve(dtype_code: Int) -> Int:
    if dtype_code == DTYPE_BOOL:
        return DTYPE_INT64
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
