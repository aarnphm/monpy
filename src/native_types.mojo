from std.collections import List
from std.memory.unsafe_pointer import alloc
from std.os import abort
from std.python import PythonObject


# Constants are integers on purpose: Python bindings pass op and dtype codes
# directly, and the hot dispatch predicates should stay cheap to read.
comptime DTYPE_BOOL = 0
comptime DTYPE_INT64 = 1
comptime DTYPE_FLOAT32 = 2
comptime DTYPE_FLOAT64 = 3

comptime OP_ADD = 0
comptime OP_SUB = 1
comptime OP_MUL = 2
comptime OP_DIV = 3

comptime UNARY_SIN = 0
comptime UNARY_COS = 1
comptime UNARY_EXP = 2
comptime UNARY_LOG = 3

comptime REDUCE_SUM = 0
comptime REDUCE_MEAN = 1
comptime REDUCE_MIN = 2
comptime REDUCE_MAX = 3
comptime REDUCE_ARGMAX = 4

comptime BACKEND_GENERIC = 0
comptime BACKEND_LAYOUT_TENSOR = 1
comptime BACKEND_ACCELERATE = 2
comptime BACKEND_FUSED = 3


@fieldwise_init
struct NativeArray(Movable, Writable):
    var dtype_code: Int
    var shape: List[Int]
    var strides: List[Int]
    var size_value: Int
    var offset_elems: Int
    var data: UnsafePointer[UInt8, MutExternalOrigin]
    var byte_len: Int
    var owns_data: Bool
    var used_layout_tensor: Bool
    var backend_code: Int

    @staticmethod
    def _get_self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("NativeArray method receiver had the wrong type: ", e))

    def __del__(deinit self):
        if self.owns_data:
            self.data.free()

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
    def shape_at_py(
        py_self: PythonObject, index_obj: PythonObject
    ) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        var index = Int(py=index_obj)
        return PythonObject(self_ptr[].shape[index])

    @staticmethod
    def stride_at_py(
        py_self: PythonObject, index_obj: PythonObject
    ) raises -> PythonObject:
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
        var byte_offset = self_ptr[].offset_elems * item_size(
            self_ptr[].dtype_code
        )
        return PythonObject((self_ptr[].data + byte_offset).__int__())

    @staticmethod
    def is_c_contiguous_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(is_c_contiguous(self_ptr[]))

    @staticmethod
    def used_layout_tensor_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].used_layout_tensor)

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
    def get_scalar_py(
        py_self: PythonObject, index_obj: PythonObject
    ) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        var index = Int(py=index_obj)
        if self_ptr[].dtype_code == DTYPE_BOOL:
            return PythonObject(
                get_physical_bool(
                    self_ptr[], physical_offset(self_ptr[], index)
                )
            )
        if self_ptr[].dtype_code == DTYPE_INT64:
            return PythonObject(
                get_physical_i64(self_ptr[], physical_offset(self_ptr[], index))
            )
        if self_ptr[].dtype_code == DTYPE_FLOAT32:
            return PythonObject(
                get_physical_f32(self_ptr[], physical_offset(self_ptr[], index))
            )
        return PythonObject(
            get_physical_f64(self_ptr[], physical_offset(self_ptr[], index))
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write("NativeArray(dtype_code=")
        writer.write(self.dtype_code)
        writer.write(", shape=")
        writer.write(self.shape)
        writer.write(")")


def item_size(dtype_code: Int) raises -> Int:
    if dtype_code == DTYPE_BOOL:
        return 1
    if dtype_code == DTYPE_INT64:
        return 8
    if dtype_code == DTYPE_FLOAT32:
        return 4
    if dtype_code == DTYPE_FLOAT64:
        return 8
    raise Error("unsupported dtype code")


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


def make_empty_array(
    dtype_code: Int, var shape: List[Int]
) raises -> NativeArray:
    var size = shape_size(shape)
    var strides = make_c_strides(shape)
    var byte_len = size * item_size(dtype_code)
    if byte_len < 1:
        byte_len = 1
    var data = alloc[UInt8](byte_len)
    # Do not zero-fill here. Creation APIs that promise initialized values write
    # explicitly, and math kernels overwrite every output element before return.
    return NativeArray(
        dtype_code,
        shape^,
        strides^,
        size,
        0,
        data,
        byte_len,
        True,
        False,
        BACKEND_GENERIC,
    )


def clone_int_list(values: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(values)):
        out.append(values[i])
    return out^


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


def is_c_contiguous(array: NativeArray) raises -> Bool:
    var expected = 1
    for axis in range(len(array.shape) - 1, -1, -1):
        if array.shape[axis] == 0:
            return True
        if array.shape[axis] != 1 and array.strides[axis] != expected:
            return False
        expected *= array.shape[axis]
    return True


def physical_offset(array: NativeArray, logical: Int) raises -> Int:
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


def broadcast_physical_offset(
    array: NativeArray, logical: Int, out_shape: List[Int]
) raises -> Int:
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


def get_physical_bool(array: NativeArray, physical: Int) raises -> Bool:
    return array.data[physical] != UInt8(0)


def get_physical_i64(array: NativeArray, physical: Int) raises -> Int64:
    return array.data.bitcast[Int64]()[physical]


def get_physical_f32(array: NativeArray, physical: Int) raises -> Float32:
    return array.data.bitcast[Float32]()[physical]


def get_physical_f64(array: NativeArray, physical: Int) raises -> Float64:
    return array.data.bitcast[Float64]()[physical]


def get_physical_as_f64(array: NativeArray, physical: Int) raises -> Float64:
    if array.dtype_code == DTYPE_BOOL:
        if get_physical_bool(array, physical):
            return 1.0
        return 0.0
    if array.dtype_code == DTYPE_INT64:
        return Float64(get_physical_i64(array, physical))
    if array.dtype_code == DTYPE_FLOAT32:
        return Float64(get_physical_f32(array, physical))
    return get_physical_f64(array, physical)


def get_logical_as_f64(array: NativeArray, logical: Int) raises -> Float64:
    return get_physical_as_f64(array, physical_offset(array, logical))


def get_broadcast_as_f64(
    array: NativeArray, logical: Int, out_shape: List[Int]
) raises -> Float64:
    return get_physical_as_f64(
        array, broadcast_physical_offset(array, logical, out_shape)
    )


def get_broadcast_as_bool(
    array: NativeArray, logical: Int, out_shape: List[Int]
) raises -> Bool:
    var physical = broadcast_physical_offset(array, logical, out_shape)
    if array.dtype_code == DTYPE_BOOL:
        return get_physical_bool(array, physical)
    return get_physical_as_f64(array, physical) != 0.0


def set_physical_from_f64(
    mut array: NativeArray, physical: Int, value: Float64
) raises:
    if array.dtype_code == DTYPE_BOOL:
        if value != 0.0:
            array.data[physical] = UInt8(1)
        else:
            array.data[physical] = UInt8(0)
    elif array.dtype_code == DTYPE_INT64:
        array.data.bitcast[Int64]()[physical] = Int64(value)
    elif array.dtype_code == DTYPE_FLOAT32:
        array.data.bitcast[Float32]()[physical] = Float32(value)
    else:
        array.data.bitcast[Float64]()[physical] = value


def set_logical_from_f64(
    mut array: NativeArray, logical: Int, value: Float64
) raises:
    set_physical_from_f64(array, physical_offset(array, logical), value)


def set_logical_from_i64(
    mut array: NativeArray, logical: Int, value: Int64
) raises:
    if array.dtype_code == DTYPE_INT64:
        array.data.bitcast[Int64]()[physical_offset(array, logical)] = value
    else:
        set_logical_from_f64(array, logical, Float64(value))


def set_logical_from_py(
    mut array: NativeArray, logical: Int, value_obj: PythonObject
) raises:
    var physical = physical_offset(array, logical)
    if array.dtype_code == DTYPE_BOOL:
        if Bool(py=value_obj):
            array.data[physical] = UInt8(1)
        else:
            array.data[physical] = UInt8(0)
    elif array.dtype_code == DTYPE_INT64:
        array.data.bitcast[Int64]()[physical] = Int64(Int(py=value_obj))
    elif array.dtype_code == DTYPE_FLOAT32:
        array.data.bitcast[Float32]()[physical] = Float32(Float64(py=value_obj))
    else:
        array.data.bitcast[Float64]()[physical] = Float64(py=value_obj)


def scalar_py_as_f64(
    value_obj: PythonObject, dtype_code: Int
) raises -> Float64:
    if dtype_code == DTYPE_BOOL:
        if Bool(py=value_obj):
            return 1.0
        return 0.0
    if dtype_code == DTYPE_INT64:
        return Float64(Int(py=value_obj))
    return Float64(py=value_obj)


def fill_all_from_py(mut array: NativeArray, value_obj: PythonObject) raises:
    for i in range(array.size_value):
        set_logical_from_py(array, i, value_obj)


def contiguous_f32_ptr(
    array: NativeArray,
) -> UnsafePointer[Float32, MutExternalOrigin]:
    return array.data.bitcast[Float32]() + array.offset_elems


def contiguous_f64_ptr(
    array: NativeArray,
) -> UnsafePointer[Float64, MutExternalOrigin]:
    return array.data.bitcast[Float64]() + array.offset_elems


def contiguous_as_f64(array: NativeArray, index: Int) raises -> Float64:
    if array.dtype_code == DTYPE_FLOAT32:
        return Float64(contiguous_f32_ptr(array)[index])
    if array.dtype_code == DTYPE_FLOAT64:
        return contiguous_f64_ptr(array)[index]
    return get_physical_as_f64(array, array.offset_elems + index)


def set_contiguous_from_f64(
    mut array: NativeArray, index: Int, value: Float64
) raises:
    if array.dtype_code == DTYPE_FLOAT32:
        contiguous_f32_ptr(array)[index] = Float32(value)
    elif array.dtype_code == DTYPE_FLOAT64:
        contiguous_f64_ptr(array)[index] = value
    else:
        set_physical_from_f64(array, array.offset_elems + index, value)


def result_dtype_for_unary(dtype_code: Int) -> Int:
    if dtype_code == DTYPE_FLOAT32:
        return DTYPE_FLOAT32
    return DTYPE_FLOAT64


def result_dtype_for_binary(lhs_dtype: Int, rhs_dtype: Int, op: Int) -> Int:
    if op == OP_DIV:
        if lhs_dtype == DTYPE_FLOAT32 and rhs_dtype == DTYPE_FLOAT32:
            return DTYPE_FLOAT32
        return DTYPE_FLOAT64
    if lhs_dtype == DTYPE_FLOAT64 or rhs_dtype == DTYPE_FLOAT64:
        return DTYPE_FLOAT64
    if lhs_dtype == DTYPE_FLOAT32 or rhs_dtype == DTYPE_FLOAT32:
        return DTYPE_FLOAT32
    if lhs_dtype == DTYPE_INT64 or rhs_dtype == DTYPE_INT64:
        return DTYPE_INT64
    return DTYPE_BOOL


def result_dtype_for_reduction(dtype_code: Int, op: Int) -> Int:
    if op == REDUCE_MEAN:
        if dtype_code == DTYPE_FLOAT32:
            return DTYPE_FLOAT32
        return DTYPE_FLOAT64
    if op == REDUCE_SUM and dtype_code == DTYPE_BOOL:
        return DTYPE_INT64
    return dtype_code


def broadcast_shape(lhs: NativeArray, rhs: NativeArray) raises -> List[Int]:
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
