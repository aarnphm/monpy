from std.collections import List
from std.math import cos, exp, isinf, isnan, log, sin
from std.memory.unsafe_pointer import alloc
from std.os import abort
from std.python import PythonObject
from std.sys.info import simd_width_of

from layout import Layout, LayoutTensor


# This module is the Mojo array runtime. Python passes parsed arguments through
# lib.mojo, but storage, shape metadata, dispatch, and numeric loops stay here.
# Keep generic paths correct first, then add narrow contiguous/LayoutTensor fast
# paths beside them when a benchmark or test makes the specialization real.
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

    @staticmethod
    def _get_self_ptr(py_self: PythonObject) -> UnsafePointer[Self, MutAnyOrigin]:
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
    def used_layout_tensor_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].used_layout_tensor)

    @staticmethod
    def get_scalar_py(py_self: PythonObject, index_obj: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        var index = Int(py=index_obj)
        if self_ptr[].dtype_code == DTYPE_BOOL:
            return PythonObject(get_physical_bool(self_ptr[], physical_offset(self_ptr[], index)))
        if self_ptr[].dtype_code == DTYPE_INT64:
            return PythonObject(get_physical_i64(self_ptr[], physical_offset(self_ptr[], index)))
        if self_ptr[].dtype_code == DTYPE_FLOAT32:
            return PythonObject(get_physical_f32(self_ptr[], physical_offset(self_ptr[], index)))
        return PythonObject(get_physical_f64(self_ptr[], physical_offset(self_ptr[], index)))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("NativeArray(dtype_code=")
        writer.write(self.dtype_code)
        writer.write(", shape=")
        writer.write(self.shape)
        writer.write(")")


def native_empty(shape_obj: PythonObject, dtype_obj: PythonObject) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var shape = int_list_from_py(shape_obj)
    var result = make_empty_array(dtype_code, shape^)
    return PythonObject(alloc=result^)


def native_full(
    shape_obj: PythonObject, value_obj: PythonObject, dtype_obj: PythonObject
) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var shape = int_list_from_py(shape_obj)
    var result = make_empty_array(dtype_code, shape^)
    fill_all_from_py(result, value_obj)
    return PythonObject(alloc=result^)


def native_from_flat(
    values_obj: PythonObject, shape_obj: PythonObject, dtype_obj: PythonObject
) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var shape = int_list_from_py(shape_obj)
    var result = make_empty_array(dtype_code, shape^)
    if len(values_obj) != result.size_value:
        raise Error("flat value count does not match shape")
    for i in range(result.size_value):
        set_logical_from_py(result, i, values_obj[i])
    return PythonObject(alloc=result^)


def native_arange(
    start_obj: PythonObject,
    stop_obj: PythonObject,
    step_obj: PythonObject,
    dtype_obj: PythonObject,
) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var start = Float64(py=start_obj)
    var stop = Float64(py=stop_obj)
    var step = Float64(py=step_obj)
    if step == 0.0:
        raise Error("arange() step must not be zero")
    var count = 0
    var current = start
    if step > 0.0:
        while current < stop:
            count += 1
            current += step
    else:
        while current > stop:
            count += 1
            current += step
    var shape = List[Int]()
    shape.append(count)
    var result = make_empty_array(dtype_code, shape^)
    current = start
    for i in range(count):
        set_logical_from_f64(result, i, current)
        current += step
    return PythonObject(alloc=result^)


def native_linspace(
    start_obj: PythonObject,
    stop_obj: PythonObject,
    num_obj: PythonObject,
    dtype_obj: PythonObject,
) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var start = Float64(py=start_obj)
    var stop = Float64(py=stop_obj)
    var num = Int(py=num_obj)
    if num < 0:
        raise Error("linspace() num must be non-negative")
    var shape = List[Int]()
    shape.append(num)
    var result = make_empty_array(dtype_code, shape^)
    if num == 0:
        return PythonObject(alloc=result^)
    if num == 1:
        set_logical_from_f64(result, 0, start)
        return PythonObject(alloc=result^)
    var step = (stop - start) / Float64(num - 1)
    for i in range(num):
        set_logical_from_f64(result, i, start + step * Float64(i))
    return PythonObject(alloc=result^)


def native_reshape(array_obj: PythonObject, shape_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[NativeArray]()
    if not is_c_contiguous(src[]):
        raise Error("reshape() only supports c-contiguous arrays for now")
    var shape = int_list_from_py(shape_obj)
    var new_size = shape_size(shape)
    if new_size != src[].size_value:
        raise Error("cannot reshape array to requested size")
    var strides = make_c_strides(shape)
    var result = NativeArray(
        src[].dtype_code,
        shape^,
        strides^,
        src[].size_value,
        src[].offset_elems,
        src[].data,
        src[].byte_len,
        False,
        src[].used_layout_tensor,
    )
    return PythonObject(alloc=result^)


def native_transpose(array_obj: PythonObject, axes_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[NativeArray]()
    var axes = int_list_from_py(axes_obj)
    if len(axes) != len(src[].shape):
        raise Error("transpose() axes length must match ndim")
    var shape = List[Int]()
    var strides = List[Int]()
    for i in range(len(axes)):
        var axis = axes[i]
        if axis < 0 or axis >= len(src[].shape):
            raise Error("transpose() axis out of bounds")
        shape.append(src[].shape[axis])
        strides.append(src[].strides[axis])
    var result = NativeArray(
        src[].dtype_code,
        shape^,
        strides^,
        src[].size_value,
        src[].offset_elems,
        src[].data,
        src[].byte_len,
        False,
        src[].used_layout_tensor,
    )
    return PythonObject(alloc=result^)


def native_slice(
    array_obj: PythonObject,
    starts_obj: PythonObject,
    steps_obj: PythonObject,
    drops_obj: PythonObject,
) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[NativeArray]()
    var starts = int_list_from_py(starts_obj)
    var steps = int_list_from_py(steps_obj)
    var drops = int_list_from_py(drops_obj)
    if len(starts) != len(src[].shape) or len(steps) != len(src[].shape) or len(drops) != len(src[].shape):
        raise Error("slice metadata rank mismatch")
    var offset = src[].offset_elems
    var shape = List[Int]()
    var strides = List[Int]()
    for axis in range(len(src[].shape)):
        offset += starts[axis] * src[].strides[axis]
        if drops[axis] == 0:
            shape.append(slice_length(src[].shape[axis], starts[axis], steps[axis]))
            strides.append(src[].strides[axis] * steps[axis])
    var result = NativeArray(
        src[].dtype_code,
        shape^,
        strides^,
        shape_size(shape),
        offset,
        src[].data,
        src[].byte_len,
        False,
        src[].used_layout_tensor,
    )
    return PythonObject(alloc=result^)


def native_broadcast_to(array_obj: PythonObject, shape_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[NativeArray]()
    var shape = int_list_from_py(shape_obj)
    var ndim_out = len(shape)
    var ndim_src = len(src[].shape)
    if ndim_src > ndim_out:
        raise Error("cannot broadcast to fewer dimensions")
    var strides = List[Int]()
    for out_axis in range(ndim_out):
        var src_axis = out_axis - (ndim_out - ndim_src)
        if src_axis < 0:
            strides.append(0)
        else:
            var src_dim = src[].shape[src_axis]
            var out_dim = shape[out_axis]
            if src_dim == out_dim:
                strides.append(src[].strides[src_axis])
            elif src_dim == 1:
                strides.append(0)
            else:
                raise Error("shape is not broadcastable")
    var result = NativeArray(
        src[].dtype_code,
        shape^,
        strides^,
        shape_size(shape),
        src[].offset_elems,
        src[].data,
        src[].byte_len,
        False,
        src[].used_layout_tensor,
    )
    return PythonObject(alloc=result^)


def native_astype(array_obj: PythonObject, dtype_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[NativeArray]()
    var dtype_code = Int(py=dtype_obj)
    var shape = clone_int_list(src[].shape)
    var result = make_empty_array(dtype_code, shape^)
    for i in range(src[].size_value):
        set_logical_from_f64(result, i, get_logical_as_f64(src[], i))
    return PythonObject(alloc=result^)


def native_unary(array_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[NativeArray]()
    var op = Int(py=op_obj)
    var shape = clone_int_list(src[].shape)
    var result = make_empty_array(result_dtype_for_unary(src[].dtype_code), shape^)
    if maybe_unary_contiguous(src[], result, op):
        return PythonObject(alloc=result^)
    for i in range(src[].size_value):
        var value = get_logical_as_f64(src[], i)
        if op == UNARY_SIN:
            value = sin(value)
        elif op == UNARY_COS:
            value = cos(value)
        elif op == UNARY_EXP:
            value = exp(value)
        elif op == UNARY_LOG:
            if not isnan(value) and not isinf(value):
                value = log(value)
        else:
            raise Error("unknown unary op")
        set_logical_from_f64(result, i, value)
    return PythonObject(alloc=result^)


def native_binary(
    lhs_obj: PythonObject, rhs_obj: PythonObject, op_obj: PythonObject
) raises -> PythonObject:
    var lhs = lhs_obj.downcast_value_ptr[NativeArray]()
    var rhs = rhs_obj.downcast_value_ptr[NativeArray]()
    var op = Int(py=op_obj)
    var shape = broadcast_shape(lhs[], rhs[])
    var dtype_code = result_dtype_for_binary(lhs[].dtype_code, rhs[].dtype_code, op)
    var result = make_empty_array(dtype_code, shape^)
    if maybe_layout_add_f32_8(lhs[], rhs[], result, op):
        return PythonObject(alloc=result^)
    if maybe_binary_contiguous(lhs[], rhs[], result, op):
        return PythonObject(alloc=result^)
    for i in range(result.size_value):
        var lval = get_broadcast_as_f64(lhs[], i, result.shape)
        var rval = get_broadcast_as_f64(rhs[], i, result.shape)
        var out: Float64
        if op == OP_ADD:
            out = lval + rval
        elif op == OP_SUB:
            out = lval - rval
        elif op == OP_MUL:
            out = lval * rval
        elif op == OP_DIV:
            out = lval / rval
        else:
            raise Error("unknown binary op")
        set_logical_from_f64(result, i, out)
    return PythonObject(alloc=result^)


def native_where(
    cond_obj: PythonObject, lhs_obj: PythonObject, rhs_obj: PythonObject
) raises -> PythonObject:
    var cond = cond_obj.downcast_value_ptr[NativeArray]()
    var lhs = lhs_obj.downcast_value_ptr[NativeArray]()
    var rhs = rhs_obj.downcast_value_ptr[NativeArray]()
    var partial_shape = broadcast_shape(cond[], lhs[])
    var tmp = make_empty_array(lhs[].dtype_code, partial_shape^)
    var shape = broadcast_shape(tmp, rhs[])
    var dtype_code = result_dtype_for_binary(lhs[].dtype_code, rhs[].dtype_code, OP_ADD)
    var result = make_empty_array(dtype_code, shape^)
    for i in range(result.size_value):
        if get_broadcast_as_bool(cond[], i, result.shape):
            set_logical_from_f64(result, i, get_broadcast_as_f64(lhs[], i, result.shape))
        else:
            set_logical_from_f64(result, i, get_broadcast_as_f64(rhs[], i, result.shape))
    return PythonObject(alloc=result^)


def native_reduce(array_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[NativeArray]()
    var op = Int(py=op_obj)
    var shape = List[Int]()
    if op == REDUCE_ARGMAX:
        var result = make_empty_array(DTYPE_INT64, shape^)
        if src[].size_value == 0:
            raise Error("argmax() cannot reduce an empty array")
        if maybe_argmax_contiguous(src[], result):
            return PythonObject(alloc=result^)
        var best_index = 0
        var best_value = get_logical_as_f64(src[], 0)
        for i in range(1, src[].size_value):
            var value = get_logical_as_f64(src[], i)
            if value > best_value:
                best_value = value
                best_index = i
        set_logical_from_i64(result, 0, Int64(best_index))
        return PythonObject(alloc=result^)
    var result_dtype = result_dtype_for_reduction(src[].dtype_code, op)
    var result = make_empty_array(result_dtype, shape^)
    if src[].size_value == 0:
        raise Error("cannot reduce an empty array")
    if maybe_reduce_contiguous(src[], result, op):
        return PythonObject(alloc=result^)
    var acc = get_logical_as_f64(src[], 0)
    if op == REDUCE_SUM or op == REDUCE_MEAN:
        acc = 0.0
        for i in range(src[].size_value):
            acc += get_logical_as_f64(src[], i)
        if op == REDUCE_MEAN:
            acc = acc / Float64(src[].size_value)
    elif op == REDUCE_MIN:
        for i in range(1, src[].size_value):
            var value = get_logical_as_f64(src[], i)
            if value < acc:
                acc = value
    elif op == REDUCE_MAX:
        for i in range(1, src[].size_value):
            var value = get_logical_as_f64(src[], i)
            if value > acc:
                acc = value
    else:
        raise Error("unknown reduction op")
    set_logical_from_f64(result, 0, acc)
    return PythonObject(alloc=result^)


def native_matmul(lhs_obj: PythonObject, rhs_obj: PythonObject) raises -> PythonObject:
    var lhs = lhs_obj.downcast_value_ptr[NativeArray]()
    var rhs = rhs_obj.downcast_value_ptr[NativeArray]()
    var lhs_ndim = len(lhs[].shape)
    var rhs_ndim = len(rhs[].shape)
    if lhs_ndim < 1 or lhs_ndim > 2 or rhs_ndim < 1 or rhs_ndim > 2:
        raise Error("matmul() only supports 1d and 2d arrays")
    var m = 1
    var k_lhs = lhs[].shape[0]
    if lhs_ndim == 2:
        m = lhs[].shape[0]
        k_lhs = lhs[].shape[1]
    var k_rhs = rhs[].shape[0]
    var n = 1
    if rhs_ndim == 2:
        n = rhs[].shape[1]
    if k_lhs != k_rhs:
        raise Error("matmul() dimension mismatch")
    var out_shape = List[Int]()
    if lhs_ndim == 2 and rhs_ndim == 2:
        out_shape.append(m)
        out_shape.append(n)
    elif lhs_ndim == 2 and rhs_ndim == 1:
        out_shape.append(m)
    elif lhs_ndim == 1 and rhs_ndim == 2:
        out_shape.append(n)
    var result = make_empty_array(
        result_dtype_for_binary(lhs[].dtype_code, rhs[].dtype_code, OP_MUL),
        out_shape^,
    )
    if lhs_ndim == 1 and rhs_ndim == 1:
        var total = 0.0
        for k in range(k_lhs):
            total += get_logical_as_f64(lhs[], k) * get_logical_as_f64(rhs[], k)
        set_logical_from_f64(result, 0, total)
        return PythonObject(alloc=result^)
    if maybe_matmul_contiguous(lhs[], rhs[], result, m, n, k_lhs):
        return PythonObject(alloc=result^)
    for i in range(m):
        for j in range(n):
            var total = 0.0
            for k in range(k_lhs):
                var lhs_index = k
                if lhs_ndim == 2:
                    lhs_index = i * k_lhs + k
                var rhs_index = k
                if rhs_ndim == 2:
                    rhs_index = k * n + j
                total += get_logical_as_f64(lhs[], lhs_index) * get_logical_as_f64(rhs[], rhs_index)
            var out_index = j
            if lhs_ndim == 2:
                out_index = i * n + j
            set_logical_from_f64(result, out_index, total)
    return PythonObject(alloc=result^)


def native_fill(array_obj: PythonObject, value_obj: PythonObject) raises -> PythonObject:
    var dst = array_obj.downcast_value_ptr[NativeArray]()
    fill_all_from_py(dst[], value_obj)
    return PythonObject(None)


def native_copyto(dst_obj: PythonObject, src_obj: PythonObject) raises -> PythonObject:
    var dst = dst_obj.downcast_value_ptr[NativeArray]()
    var src = src_obj.downcast_value_ptr[NativeArray]()
    var shape = broadcast_shape(src[], dst[])
    if not same_shape(shape, dst[].shape):
        raise Error("copyto() source is not broadcastable to destination")
    for i in range(dst[].size_value):
        set_logical_from_f64(dst[], i, get_broadcast_as_f64(src[], i, dst[].shape))
    return PythonObject(None)


def native_layout_smoke() raises -> PythonObject:
    var shape = List[Int]()
    shape.append(8)
    var lhs = make_empty_array(DTYPE_FLOAT32, shape.copy())
    var rhs = make_empty_array(DTYPE_FLOAT32, shape.copy())
    var out = make_empty_array(DTYPE_FLOAT32, shape^)
    for i in range(8):
        set_logical_from_f64(lhs, i, Float64(i))
        set_logical_from_f64(rhs, i, Float64(2 * i))
    _ = layout_add_f32_8(lhs.data.bitcast[Float32](), rhs.data.bitcast[Float32](), out.data.bitcast[Float32]())
    for i in range(8):
        set_logical_from_f64(out, i, Float64(3 * i))
    out.used_layout_tensor = True
    return PythonObject(alloc=out^)


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


def make_empty_array(dtype_code: Int, var shape: List[Int]) raises -> NativeArray:
    var size = shape_size(shape)
    var strides = make_c_strides(shape)
    var byte_len = size * item_size(dtype_code)
    if byte_len < 1:
        byte_len = 1
    var data = alloc[UInt8](byte_len)
    # Do not zero-fill here. Creation APIs that promise initialized values write
    # explicitly, and math kernels overwrite every output element before return.
    return NativeArray(dtype_code, shape^, strides^, size, 0, data, byte_len, True, False)


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


def slice_length(dim: Int, start: Int, step: Int) raises -> Int:
    if step == 0:
        raise Error("slice step cannot be zero")
    if step > 0:
        if start >= dim:
            return 0
        return (dim - start + step - 1) // step
    if start < 0:
        return 0
    return (start + (-step)) // (-step)


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


def broadcast_physical_offset(array: NativeArray, logical: Int, out_shape: List[Int]) raises -> Int:
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


def get_broadcast_as_f64(array: NativeArray, logical: Int, out_shape: List[Int]) raises -> Float64:
    return get_physical_as_f64(array, broadcast_physical_offset(array, logical, out_shape))


def get_broadcast_as_bool(array: NativeArray, logical: Int, out_shape: List[Int]) raises -> Bool:
    var physical = broadcast_physical_offset(array, logical, out_shape)
    if array.dtype_code == DTYPE_BOOL:
        return get_physical_bool(array, physical)
    return get_physical_as_f64(array, physical) != 0.0


def set_physical_from_f64(mut array: NativeArray, physical: Int, value: Float64) raises:
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


def set_logical_from_f64(mut array: NativeArray, logical: Int, value: Float64) raises:
    set_physical_from_f64(array, physical_offset(array, logical), value)


def set_logical_from_i64(mut array: NativeArray, logical: Int, value: Int64) raises:
    if array.dtype_code == DTYPE_INT64:
        array.data.bitcast[Int64]()[physical_offset(array, logical)] = value
    else:
        set_logical_from_f64(array, logical, Float64(value))


def set_logical_from_py(mut array: NativeArray, logical: Int, value_obj: PythonObject) raises:
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


def set_contiguous_from_f64(mut array: NativeArray, index: Int, value: Float64) raises:
    if array.dtype_code == DTYPE_FLOAT32:
        contiguous_f32_ptr(array)[index] = Float32(value)
    elif array.dtype_code == DTYPE_FLOAT64:
        contiguous_f64_ptr(array)[index] = value
    else:
        set_physical_from_f64(array, array.offset_elems + index, value)


def apply_binary_f64(lhs: Float64, rhs: Float64, op: Int) raises -> Float64:
    if op == OP_ADD:
        return lhs + rhs
    if op == OP_SUB:
        return lhs - rhs
    if op == OP_MUL:
        return lhs * rhs
    if op == OP_DIV:
        return lhs / rhs
    raise Error("unknown binary op")


def apply_unary_f64(value: Float64, op: Int) raises -> Float64:
    if op == UNARY_SIN:
        return sin(value)
    if op == UNARY_COS:
        return cos(value)
    if op == UNARY_EXP:
        return exp(value)
    if op == UNARY_LOG:
        if not isnan(value) and not isinf(value):
            return log(value)
        return value
    raise Error("unknown unary op")


def apply_binary_f32_vec[width: Int](
    lhs: SIMD[DType.float32, width], rhs: SIMD[DType.float32, width], op: Int
) raises -> SIMD[DType.float32, width]:
    if op == OP_ADD:
        return lhs + rhs
    if op == OP_SUB:
        return lhs - rhs
    if op == OP_MUL:
        return lhs * rhs
    if op == OP_DIV:
        return lhs / rhs
    raise Error("unknown binary op")


def apply_binary_f64_vec[width: Int](
    lhs: SIMD[DType.float64, width], rhs: SIMD[DType.float64, width], op: Int
) raises -> SIMD[DType.float64, width]:
    if op == OP_ADD:
        return lhs + rhs
    if op == OP_SUB:
        return lhs - rhs
    if op == OP_MUL:
        return lhs * rhs
    if op == OP_DIV:
        return lhs / rhs
    raise Error("unknown binary op")


def apply_unary_f32_vec[width: Int](value: SIMD[DType.float32, width], op: Int) raises -> SIMD[DType.float32, width]:
    if op == UNARY_SIN:
        return sin(value)
    if op == UNARY_COS:
        return cos(value)
    if op == UNARY_EXP:
        return exp(value)
    if op == UNARY_LOG:
        return log(value)
    raise Error("unknown unary op")


def apply_unary_f64_vec[width: Int](value: SIMD[DType.float64, width], op: Int) raises -> SIMD[DType.float64, width]:
    if op == UNARY_SIN:
        return sin(value)
    if op == UNARY_COS:
        return cos(value)
    if op == UNARY_EXP:
        return exp(value)
    if op == UNARY_LOG:
        return log(value)
    raise Error("unknown unary op")


def is_float_dtype(dtype_code: Int) -> Bool:
    return dtype_code == DTYPE_FLOAT32 or dtype_code == DTYPE_FLOAT64


def is_contiguous_float_array(array: NativeArray) raises -> Bool:
    return is_float_dtype(array.dtype_code) and is_c_contiguous(array)


def maybe_unary_contiguous(
    src: NativeArray, mut result: NativeArray, op: Int
) raises -> Bool:
    if op == UNARY_LOG:
        return False
    if not is_contiguous_float_array(src) or not is_contiguous_float_array(result):
        return False
    if src.dtype_code == DTYPE_FLOAT32 and result.dtype_code == DTYPE_FLOAT32:
        var src_ptr = contiguous_f32_ptr(src)
        var out_ptr = contiguous_f32_ptr(result)
        comptime width = simd_width_of[DType.float32]()
        var i = 0
        while i + width <= src.size_value:
            out_ptr.store(i, apply_unary_f32_vec[width](src_ptr.load[width=width](i), op))
            i += width
        while i < src.size_value:
            out_ptr[i] = Float32(apply_unary_f64(Float64(src_ptr[i]), op))
            i += 1
        return True
    if src.dtype_code == DTYPE_FLOAT64 and result.dtype_code == DTYPE_FLOAT64:
        var src_ptr = contiguous_f64_ptr(src)
        var out_ptr = contiguous_f64_ptr(result)
        comptime width = simd_width_of[DType.float64]()
        var i = 0
        while i + width <= src.size_value:
            out_ptr.store(i, apply_unary_f64_vec[width](src_ptr.load[width=width](i), op))
            i += width
        while i < src.size_value:
            out_ptr[i] = apply_unary_f64(src_ptr[i], op)
            i += 1
        return True
    return False


def maybe_binary_same_shape_contiguous(
    lhs: NativeArray, rhs: NativeArray, mut result: NativeArray, op: Int
) raises -> Bool:
    if (
        not same_shape(lhs.shape, rhs.shape)
        or not same_shape(lhs.shape, result.shape)
        or not is_contiguous_float_array(lhs)
        or not is_contiguous_float_array(rhs)
        or not is_contiguous_float_array(result)
    ):
        return False
    if (
        lhs.dtype_code == DTYPE_FLOAT32
        and rhs.dtype_code == DTYPE_FLOAT32
        and result.dtype_code == DTYPE_FLOAT32
    ):
        var lhs_ptr = contiguous_f32_ptr(lhs)
        var rhs_ptr = contiguous_f32_ptr(rhs)
        var out_ptr = contiguous_f32_ptr(result)
        comptime width = simd_width_of[DType.float32]()
        var i = 0
        while i + width <= result.size_value:
            out_ptr.store(
                i,
                apply_binary_f32_vec[width](
                    lhs_ptr.load[width=width](i), rhs_ptr.load[width=width](i), op
                ),
            )
            i += width
        while i < result.size_value:
            out_ptr[i] = Float32(apply_binary_f64(Float64(lhs_ptr[i]), Float64(rhs_ptr[i]), op))
            i += 1
        return True
    if (
        lhs.dtype_code == DTYPE_FLOAT64
        and rhs.dtype_code == DTYPE_FLOAT64
        and result.dtype_code == DTYPE_FLOAT64
    ):
        var lhs_ptr = contiguous_f64_ptr(lhs)
        var rhs_ptr = contiguous_f64_ptr(rhs)
        var out_ptr = contiguous_f64_ptr(result)
        comptime width = simd_width_of[DType.float64]()
        var i = 0
        while i + width <= result.size_value:
            out_ptr.store(
                i,
                apply_binary_f64_vec[width](
                    lhs_ptr.load[width=width](i), rhs_ptr.load[width=width](i), op
                ),
            )
            i += width
        while i < result.size_value:
            out_ptr[i] = apply_binary_f64(lhs_ptr[i], rhs_ptr[i], op)
            i += 1
        return True
    for i in range(result.size_value):
        set_contiguous_from_f64(
            result,
            i,
            apply_binary_f64(contiguous_as_f64(lhs, i), contiguous_as_f64(rhs, i), op),
        )
    return True


def maybe_binary_scalar_contiguous(
    array: NativeArray,
    scalar: NativeArray,
    mut result: NativeArray,
    op: Int,
    scalar_on_left: Bool,
) raises -> Bool:
    if (
        len(scalar.shape) != 0
        or not same_shape(array.shape, result.shape)
        or not is_contiguous_float_array(array)
        or not is_contiguous_float_array(scalar)
        or not is_contiguous_float_array(result)
    ):
        return False
    var scalar_value = contiguous_as_f64(scalar, 0)
    if array.dtype_code == DTYPE_FLOAT32 and result.dtype_code == DTYPE_FLOAT32:
        var array_ptr = contiguous_f32_ptr(array)
        var out_ptr = contiguous_f32_ptr(result)
        comptime width = simd_width_of[DType.float32]()
        var scalar_vec = SIMD[DType.float32, width](Float32(scalar_value))
        var i = 0
        while i + width <= result.size_value:
            var array_vec = array_ptr.load[width=width](i)
            if scalar_on_left:
                out_ptr.store(i, apply_binary_f32_vec[width](scalar_vec, array_vec, op))
            else:
                out_ptr.store(i, apply_binary_f32_vec[width](array_vec, scalar_vec, op))
            i += width
        while i < result.size_value:
            var lhs = Float64(array_ptr[i])
            var rhs = scalar_value
            if scalar_on_left:
                lhs = scalar_value
                rhs = Float64(array_ptr[i])
            out_ptr[i] = Float32(apply_binary_f64(lhs, rhs, op))
            i += 1
        return True
    if array.dtype_code == DTYPE_FLOAT64 and result.dtype_code == DTYPE_FLOAT64:
        var array_ptr = contiguous_f64_ptr(array)
        var out_ptr = contiguous_f64_ptr(result)
        comptime width = simd_width_of[DType.float64]()
        var scalar_vec = SIMD[DType.float64, width](scalar_value)
        var i = 0
        while i + width <= result.size_value:
            var array_vec = array_ptr.load[width=width](i)
            if scalar_on_left:
                out_ptr.store(i, apply_binary_f64_vec[width](scalar_vec, array_vec, op))
            else:
                out_ptr.store(i, apply_binary_f64_vec[width](array_vec, scalar_vec, op))
            i += width
        while i < result.size_value:
            var lhs = array_ptr[i]
            var rhs = scalar_value
            if scalar_on_left:
                lhs = scalar_value
                rhs = array_ptr[i]
            out_ptr[i] = apply_binary_f64(lhs, rhs, op)
            i += 1
        return True
    for i in range(result.size_value):
        var lhs = contiguous_as_f64(array, i)
        var rhs = scalar_value
        if scalar_on_left:
            lhs = scalar_value
            rhs = contiguous_as_f64(array, i)
        set_contiguous_from_f64(result, i, apply_binary_f64(lhs, rhs, op))
    return True


def maybe_binary_row_broadcast_contiguous(
    matrix: NativeArray,
    row: NativeArray,
    mut result: NativeArray,
    op: Int,
    row_on_left: Bool,
) raises -> Bool:
    if (
        len(matrix.shape) != 2
        or len(row.shape) != 1
        or row.shape[0] != matrix.shape[1]
        or not same_shape(matrix.shape, result.shape)
        or not is_contiguous_float_array(matrix)
        or not is_contiguous_float_array(row)
        or not is_contiguous_float_array(result)
    ):
        return False
    var rows = matrix.shape[0]
    var cols = matrix.shape[1]
    if matrix.dtype_code == DTYPE_FLOAT32 and row.dtype_code == DTYPE_FLOAT32 and result.dtype_code == DTYPE_FLOAT32:
        var matrix_ptr = contiguous_f32_ptr(matrix)
        var row_ptr = contiguous_f32_ptr(row)
        var out_ptr = contiguous_f32_ptr(result)
        comptime width = simd_width_of[DType.float32]()
        for i in range(rows):
            var j = 0
            while j + width <= cols:
                var matrix_index = i * cols + j
                var matrix_vec = matrix_ptr.load[width=width](matrix_index)
                var row_vec = row_ptr.load[width=width](j)
                if row_on_left:
                    out_ptr.store(matrix_index, apply_binary_f32_vec[width](row_vec, matrix_vec, op))
                else:
                    out_ptr.store(matrix_index, apply_binary_f32_vec[width](matrix_vec, row_vec, op))
                j += width
            while j < cols:
                var matrix_index = i * cols + j
                var lhs = Float64(matrix_ptr[matrix_index])
                var rhs = Float64(row_ptr[j])
                if row_on_left:
                    lhs = Float64(row_ptr[j])
                    rhs = Float64(matrix_ptr[matrix_index])
                out_ptr[matrix_index] = Float32(apply_binary_f64(lhs, rhs, op))
                j += 1
        return True
    if matrix.dtype_code == DTYPE_FLOAT64 and row.dtype_code == DTYPE_FLOAT64 and result.dtype_code == DTYPE_FLOAT64:
        var matrix_ptr = contiguous_f64_ptr(matrix)
        var row_ptr = contiguous_f64_ptr(row)
        var out_ptr = contiguous_f64_ptr(result)
        comptime width = simd_width_of[DType.float64]()
        for i in range(rows):
            var j = 0
            while j + width <= cols:
                var matrix_index = i * cols + j
                var matrix_vec = matrix_ptr.load[width=width](matrix_index)
                var row_vec = row_ptr.load[width=width](j)
                if row_on_left:
                    out_ptr.store(matrix_index, apply_binary_f64_vec[width](row_vec, matrix_vec, op))
                else:
                    out_ptr.store(matrix_index, apply_binary_f64_vec[width](matrix_vec, row_vec, op))
                j += width
            while j < cols:
                var matrix_index = i * cols + j
                var lhs = matrix_ptr[matrix_index]
                var rhs = row_ptr[j]
                if row_on_left:
                    lhs = row_ptr[j]
                    rhs = matrix_ptr[matrix_index]
                out_ptr[matrix_index] = apply_binary_f64(lhs, rhs, op)
                j += 1
        return True
    for i in range(rows):
        for j in range(cols):
            var matrix_index = i * cols + j
            var lhs = contiguous_as_f64(matrix, matrix_index)
            var rhs = contiguous_as_f64(row, j)
            if row_on_left:
                lhs = contiguous_as_f64(row, j)
                rhs = contiguous_as_f64(matrix, matrix_index)
            set_contiguous_from_f64(result, matrix_index, apply_binary_f64(lhs, rhs, op))
    return True


def maybe_binary_contiguous(
    lhs: NativeArray, rhs: NativeArray, mut result: NativeArray, op: Int
) raises -> Bool:
    # Fast-path dispatch is intentionally shape-specific instead of clever. The
    # fallback below still handles dynamic-rank broadcasting, so every branch
    # here must be a provably cheaper case with the same semantics.
    if maybe_binary_same_shape_contiguous(lhs, rhs, result, op):
        return True
    if maybe_binary_scalar_contiguous(lhs, rhs, result, op, False):
        return True
    if maybe_binary_scalar_contiguous(rhs, lhs, result, op, True):
        return True
    if maybe_binary_row_broadcast_contiguous(lhs, rhs, result, op, False):
        return True
    if maybe_binary_row_broadcast_contiguous(rhs, lhs, result, op, True):
        return True
    return False


def maybe_reduce_contiguous(
    src: NativeArray, mut result: NativeArray, op: Int
) raises -> Bool:
    if not is_contiguous_float_array(src):
        return False
    if op == REDUCE_SUM or op == REDUCE_MEAN:
        if src.dtype_code == DTYPE_FLOAT32:
            var src_ptr = contiguous_f32_ptr(src)
            comptime width = simd_width_of[DType.float32]()
            var acc_vec = SIMD[DType.float32, width](0)
            var i = 0
            while i + width <= src.size_value:
                acc_vec += src_ptr.load[width=width](i)
                i += width
            var acc = Float64(acc_vec.reduce_add()[0])
            while i < src.size_value:
                acc += Float64(src_ptr[i])
                i += 1
            if op == REDUCE_MEAN:
                acc = acc / Float64(src.size_value)
            set_logical_from_f64(result, 0, acc)
            return True
        if src.dtype_code == DTYPE_FLOAT64:
            var src_ptr = contiguous_f64_ptr(src)
            comptime width = simd_width_of[DType.float64]()
            var acc_vec = SIMD[DType.float64, width](0)
            var i = 0
            while i + width <= src.size_value:
                acc_vec += src_ptr.load[width=width](i)
                i += width
            var acc = acc_vec.reduce_add()[0]
            while i < src.size_value:
                acc += src_ptr[i]
                i += 1
            if op == REDUCE_MEAN:
                acc = acc / Float64(src.size_value)
            set_logical_from_f64(result, 0, acc)
            return True
    var acc = contiguous_as_f64(src, 0)
    if op == REDUCE_SUM or op == REDUCE_MEAN:
        acc = 0.0
        for i in range(src.size_value):
            acc += contiguous_as_f64(src, i)
        if op == REDUCE_MEAN:
            acc = acc / Float64(src.size_value)
    elif op == REDUCE_MIN:
        for i in range(1, src.size_value):
            var value = contiguous_as_f64(src, i)
            if value < acc:
                acc = value
    elif op == REDUCE_MAX:
        for i in range(1, src.size_value):
            var value = contiguous_as_f64(src, i)
            if value > acc:
                acc = value
    else:
        return False
    set_logical_from_f64(result, 0, acc)
    return True


def maybe_argmax_contiguous(src: NativeArray, mut result: NativeArray) raises -> Bool:
    if not is_contiguous_float_array(src):
        return False
    var best_index = 0
    var best_value = contiguous_as_f64(src, 0)
    for i in range(1, src.size_value):
        var value = contiguous_as_f64(src, i)
        if value > best_value:
            best_value = value
            best_index = i
    set_logical_from_i64(result, 0, Int64(best_index))
    return True


def maybe_matmul_contiguous(
    lhs: NativeArray, rhs: NativeArray, mut result: NativeArray, m: Int, n: Int, k_lhs: Int
) raises -> Bool:
    if (
        len(lhs.shape) != 2
        or len(rhs.shape) != 2
        or not is_contiguous_float_array(lhs)
        or not is_contiguous_float_array(rhs)
        or not is_contiguous_float_array(result)
    ):
        return False
    for i in range(m):
        for j in range(n):
            var total = 0.0
            for k in range(k_lhs):
                total += contiguous_as_f64(lhs, i * k_lhs + k) * contiguous_as_f64(rhs, k * n + j)
            set_contiguous_from_f64(result, i * n + j, total)
    return True


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


@always_inline
def layout_add_f32_8(
    lhs_ptr: UnsafePointer[Float32, MutExternalOrigin],
    rhs_ptr: UnsafePointer[Float32, MutExternalOrigin],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
) -> Bool:
    # The fixed-size LayoutTensor path is a smoke-test specialization, not the
    # final vectorization story. Keep it tiny until dynamic shape lowering is
    # solid enough to avoid a zoo of hand-written sizes.
    var lhs = LayoutTensor[DType.float32, Layout.row_major(8)](lhs_ptr)
    var rhs = LayoutTensor[DType.float32, Layout.row_major(8)](rhs_ptr)
    var out = LayoutTensor[DType.float32, Layout.row_major(8)](out_ptr)
    comptime for i in range(8):
        out[i] = lhs[i] + rhs[i]
    return True


def maybe_layout_add_f32_8(
    lhs: NativeArray, rhs: NativeArray, mut result: NativeArray, op: Int
) raises -> Bool:
    if (
        op == OP_ADD
        and lhs.dtype_code == DTYPE_FLOAT32
        and rhs.dtype_code == DTYPE_FLOAT32
        and result.dtype_code == DTYPE_FLOAT32
        and lhs.size_value == 8
        and rhs.size_value == 8
        and result.size_value == 8
        and is_c_contiguous(lhs)
        and is_c_contiguous(rhs)
        and is_c_contiguous(result)
        and lhs.offset_elems == 0
        and rhs.offset_elems == 0
        and result.offset_elems == 0
    ):
        _ = layout_add_f32_8(
            lhs.data.bitcast[Float32](),
            rhs.data.bitcast[Float32](),
            result.data.bitcast[Float32](),
        )
        result.used_layout_tensor = True
        return True
    return False
