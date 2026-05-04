from std.collections import List
from std.math import cos, exp, isinf, isnan, log, sin
from std.python import PythonObject

from native_kernels import (
    apply_binary_f64,
    layout_add_f32_8,
    maybe_argmax_contiguous,
    maybe_binary_contiguous,
    maybe_binary_same_shape_contiguous,
    maybe_binary_scalar_value_contiguous,
    is_contiguous_float_array,
    maybe_layout_add_f32_8,
    maybe_matmul_contiguous,
    maybe_reduce_contiguous,
    maybe_sin_add_mul_contiguous,
    maybe_unary_contiguous,
    write_add_f32_1d_into,
)
from native_types import (
    BACKEND_FUSED,
    BACKEND_LAYOUT_TENSOR,
    DTYPE_FLOAT32,
    DTYPE_INT64,
    NativeArray,
    OP_ADD,
    OP_DIV,
    OP_MUL,
    OP_SUB,
    REDUCE_ARGMAX,
    REDUCE_MEAN,
    REDUCE_MAX,
    REDUCE_MIN,
    REDUCE_SUM,
    UNARY_COS,
    UNARY_EXP,
    UNARY_LOG,
    UNARY_SIN,
    broadcast_shape,
    clone_int_list,
    fill_all_from_py,
    get_broadcast_as_bool,
    get_broadcast_as_f64,
    get_logical_as_f64,
    int_list_from_py,
    is_c_contiguous,
    make_c_strides,
    make_empty_array,
    result_dtype_for_binary,
    result_dtype_for_reduction,
    result_dtype_for_unary,
    same_shape,
    scalar_py_as_f64,
    set_logical_from_f64,
    set_logical_from_i64,
    set_logical_from_py,
    shape_size,
    slice_length,
)


# Python-callable entrypoints live here. Storage, shape helpers, backend ffi, and
# tight loops are imported from sibling modules to keep this file readable.
def native_empty(
    shape_obj: PythonObject, dtype_obj: PythonObject
) raises -> PythonObject:
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


def native_reshape(
    array_obj: PythonObject, shape_obj: PythonObject
) raises -> PythonObject:
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
        src[].backend_code,
    )
    return PythonObject(alloc=result^)


def native_transpose(
    array_obj: PythonObject, axes_obj: PythonObject
) raises -> PythonObject:
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
        src[].backend_code,
    )
    return PythonObject(alloc=result^)


def native_slice(
    array_obj: PythonObject,
    starts_obj: PythonObject,
    stops_obj: PythonObject,
    steps_obj: PythonObject,
    drops_obj: PythonObject,
) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[NativeArray]()
    var starts = int_list_from_py(starts_obj)
    var stops = int_list_from_py(stops_obj)
    var steps = int_list_from_py(steps_obj)
    var drops = int_list_from_py(drops_obj)
    if (
        len(starts) != len(src[].shape)
        or len(stops) != len(src[].shape)
        or len(steps) != len(src[].shape)
        or len(drops) != len(src[].shape)
    ):
        raise Error("slice metadata rank mismatch")
    var offset = src[].offset_elems
    var shape = List[Int]()
    var strides = List[Int]()
    for axis in range(len(src[].shape)):
        offset += starts[axis] * src[].strides[axis]
        if drops[axis] == 0:
            shape.append(
                slice_length(
                    src[].shape[axis], starts[axis], stops[axis], steps[axis]
                )
            )
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
        src[].backend_code,
    )
    return PythonObject(alloc=result^)


def native_broadcast_to(
    array_obj: PythonObject, shape_obj: PythonObject
) raises -> PythonObject:
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
        src[].backend_code,
    )
    return PythonObject(alloc=result^)


def native_astype(
    array_obj: PythonObject, dtype_obj: PythonObject
) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[NativeArray]()
    var dtype_code = Int(py=dtype_obj)
    var shape = clone_int_list(src[].shape)
    var result = make_empty_array(dtype_code, shape^)
    for i in range(src[].size_value):
        set_logical_from_f64(result, i, get_logical_as_f64(src[], i))
    return PythonObject(alloc=result^)


def native_unary(
    array_obj: PythonObject, op_obj: PythonObject
) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[NativeArray]()
    var op = Int(py=op_obj)
    var shape = clone_int_list(src[].shape)
    var result = make_empty_array(
        result_dtype_for_unary(src[].dtype_code), shape^
    )
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
    if same_shape(lhs[].shape, rhs[].shape):
        var same_shape_out = clone_int_list(lhs[].shape)
        var same_shape_dtype = result_dtype_for_binary(
            lhs[].dtype_code, rhs[].dtype_code, op
        )
        var same_shape_result = make_empty_array(
            same_shape_dtype, same_shape_out^
        )
        if maybe_layout_add_f32_8(lhs[], rhs[], same_shape_result, op):
            return PythonObject(alloc=same_shape_result^)
        if maybe_binary_same_shape_contiguous(
            lhs[], rhs[], same_shape_result, op
        ):
            return PythonObject(alloc=same_shape_result^)
    var shape = broadcast_shape(lhs[], rhs[])
    var dtype_code = result_dtype_for_binary(
        lhs[].dtype_code, rhs[].dtype_code, op
    )
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


def native_add(
    lhs_obj: PythonObject, rhs_obj: PythonObject
) raises -> PythonObject:
    var lhs = lhs_obj.downcast_value_ptr[NativeArray]()
    var rhs = rhs_obj.downcast_value_ptr[NativeArray]()
    if (
        lhs[].dtype_code == DTYPE_FLOAT32
        and rhs[].dtype_code == DTYPE_FLOAT32
        and len(lhs[].shape) == 1
        and len(rhs[].shape) == 1
        and lhs[].size_value == rhs[].size_value
        and lhs[].size_value != 8
        and (lhs[].shape[0] <= 1 or lhs[].strides[0] == 1)
        and (rhs[].shape[0] <= 1 or rhs[].strides[0] == 1)
    ):
        var fast_shape = clone_int_list(lhs[].shape)
        var fast_result = make_empty_array(DTYPE_FLOAT32, fast_shape^)
        _ = write_add_f32_1d_into(fast_result, lhs[], rhs[])
        return PythonObject(alloc=fast_result^)
    if same_shape(lhs[].shape, rhs[].shape):
        var same_shape_out = clone_int_list(lhs[].shape)
        var same_shape_dtype = result_dtype_for_binary(
            lhs[].dtype_code, rhs[].dtype_code, OP_ADD
        )
        var same_shape_result = make_empty_array(
            same_shape_dtype, same_shape_out^
        )
        if maybe_layout_add_f32_8(lhs[], rhs[], same_shape_result, OP_ADD):
            return PythonObject(alloc=same_shape_result^)
        if maybe_binary_same_shape_contiguous(
            lhs[], rhs[], same_shape_result, OP_ADD
        ):
            return PythonObject(alloc=same_shape_result^)
    var shape = broadcast_shape(lhs[], rhs[])
    var dtype_code = result_dtype_for_binary(
        lhs[].dtype_code, rhs[].dtype_code, OP_ADD
    )
    var result = make_empty_array(dtype_code, shape^)
    if maybe_layout_add_f32_8(lhs[], rhs[], result, OP_ADD):
        return PythonObject(alloc=result^)
    if maybe_binary_contiguous(lhs[], rhs[], result, OP_ADD):
        return PythonObject(alloc=result^)
    for i in range(result.size_value):
        var lval = get_broadcast_as_f64(lhs[], i, result.shape)
        var rval = get_broadcast_as_f64(rhs[], i, result.shape)
        set_logical_from_f64(result, i, lval + rval)
    return PythonObject(alloc=result^)


def native_binary_into(
    dst_obj: PythonObject,
    lhs_obj: PythonObject,
    rhs_obj: PythonObject,
    op_obj: PythonObject,
) raises -> PythonObject:
    var dst = dst_obj.downcast_value_ptr[NativeArray]()
    var lhs = lhs_obj.downcast_value_ptr[NativeArray]()
    var rhs = rhs_obj.downcast_value_ptr[NativeArray]()
    var op = Int(py=op_obj)
    var dtype_code = result_dtype_for_binary(
        lhs[].dtype_code, rhs[].dtype_code, op
    )
    if dst[].dtype_code != dtype_code:
        raise Error("out dtype does not match binary result dtype")
    if same_shape(lhs[].shape, rhs[].shape) and same_shape(
        lhs[].shape, dst[].shape
    ):
        if maybe_layout_add_f32_8(lhs[], rhs[], dst[], op):
            return PythonObject(None)
        if maybe_binary_same_shape_contiguous(lhs[], rhs[], dst[], op):
            return PythonObject(None)
    else:
        var shape = broadcast_shape(lhs[], rhs[])
        if not same_shape(shape, dst[].shape):
            raise Error("out shape does not match binary result shape")
        if maybe_binary_contiguous(lhs[], rhs[], dst[], op):
            return PythonObject(None)
    for i in range(dst[].size_value):
        var lval = get_broadcast_as_f64(lhs[], i, dst[].shape)
        var rval = get_broadcast_as_f64(rhs[], i, dst[].shape)
        set_logical_from_f64(dst[], i, apply_binary_f64(lval, rval, op))
    return PythonObject(None)


def native_add_into(
    dst_obj: PythonObject,
    lhs_obj: PythonObject,
    rhs_obj: PythonObject,
) raises -> PythonObject:
    var dst = dst_obj.downcast_value_ptr[NativeArray]()
    var lhs = lhs_obj.downcast_value_ptr[NativeArray]()
    var rhs = rhs_obj.downcast_value_ptr[NativeArray]()
    if write_add_f32_1d_into(dst[], lhs[], rhs[]):
        return PythonObject(None)
    var dtype_code = result_dtype_for_binary(
        lhs[].dtype_code, rhs[].dtype_code, OP_ADD
    )
    if dst[].dtype_code != dtype_code:
        raise Error("out dtype does not match binary result dtype")
    if same_shape(lhs[].shape, rhs[].shape) and same_shape(
        lhs[].shape, dst[].shape
    ):
        if maybe_layout_add_f32_8(lhs[], rhs[], dst[], OP_ADD):
            return PythonObject(None)
        if maybe_binary_same_shape_contiguous(lhs[], rhs[], dst[], OP_ADD):
            return PythonObject(None)
    else:
        var shape = broadcast_shape(lhs[], rhs[])
        if not same_shape(shape, dst[].shape):
            raise Error("out shape does not match binary result shape")
        if maybe_binary_contiguous(lhs[], rhs[], dst[], OP_ADD):
            return PythonObject(None)
    for i in range(dst[].size_value):
        var lval = get_broadcast_as_f64(lhs[], i, dst[].shape)
        var rval = get_broadcast_as_f64(rhs[], i, dst[].shape)
        set_logical_from_f64(dst[], i, lval + rval)
    return PythonObject(None)


def native_add_f32_into(
    dst_obj: PythonObject,
    lhs_obj: PythonObject,
    rhs_obj: PythonObject,
) raises -> PythonObject:
    var dst = dst_obj.downcast_value_ptr[NativeArray]()
    var lhs = lhs_obj.downcast_value_ptr[NativeArray]()
    var rhs = rhs_obj.downcast_value_ptr[NativeArray]()
    if not write_add_f32_1d_into(dst[], lhs[], rhs[]):
        return PythonObject(False)
    return PythonObject(True)


def native_binary_scalar(
    array_obj: PythonObject,
    scalar_obj: PythonObject,
    scalar_dtype_obj: PythonObject,
    op_obj: PythonObject,
    scalar_on_left_obj: PythonObject,
) raises -> PythonObject:
    var array = array_obj.downcast_value_ptr[NativeArray]()
    var scalar_dtype = Int(py=scalar_dtype_obj)
    var op = Int(py=op_obj)
    var scalar_on_left = Bool(py=scalar_on_left_obj)
    var shape = clone_int_list(array[].shape)
    var dtype_code = result_dtype_for_binary(
        array[].dtype_code, scalar_dtype, op
    )
    var result = make_empty_array(dtype_code, shape^)
    var scalar_value = scalar_py_as_f64(scalar_obj, scalar_dtype)
    if maybe_binary_scalar_value_contiguous(
        array[], scalar_value, result, op, scalar_on_left
    ):
        return PythonObject(alloc=result^)
    for i in range(result.size_value):
        var lhs = get_logical_as_f64(array[], i)
        var rhs = scalar_value
        if scalar_on_left:
            lhs = scalar_value
            rhs = get_logical_as_f64(array[], i)
        set_logical_from_f64(result, i, apply_binary_f64(lhs, rhs, op))
    return PythonObject(alloc=result^)


def native_sin_add_mul(
    lhs_obj: PythonObject,
    rhs_obj: PythonObject,
    scalar_obj: PythonObject,
    scalar_dtype_obj: PythonObject,
) raises -> PythonObject:
    var lhs = lhs_obj.downcast_value_ptr[NativeArray]()
    var rhs = rhs_obj.downcast_value_ptr[NativeArray]()
    var scalar_dtype = Int(py=scalar_dtype_obj)
    var scalar_value = scalar_py_as_f64(scalar_obj, scalar_dtype)
    if (
        same_shape(lhs[].shape, rhs[].shape)
        and lhs[].dtype_code == rhs[].dtype_code
        and lhs[].dtype_code == scalar_dtype
        and is_contiguous_float_array(lhs[])
        and is_contiguous_float_array(rhs[])
    ):
        var fast_shape = clone_int_list(lhs[].shape)
        var fast_result = make_empty_array(lhs[].dtype_code, fast_shape^)
        if maybe_sin_add_mul_contiguous(
            lhs[], rhs[], scalar_value, fast_result
        ):
            return PythonObject(alloc=fast_result^)
    var shape = broadcast_shape(lhs[], rhs[])
    var rhs_mul_dtype = result_dtype_for_binary(
        rhs[].dtype_code, scalar_dtype, OP_MUL
    )
    var dtype_code = result_dtype_for_binary(
        result_dtype_for_unary(lhs[].dtype_code), rhs_mul_dtype, OP_ADD
    )
    var result = make_empty_array(dtype_code, shape^)
    if maybe_sin_add_mul_contiguous(lhs[], rhs[], scalar_value, result):
        return PythonObject(alloc=result^)
    for i in range(result.size_value):
        var out = sin(get_broadcast_as_f64(lhs[], i, result.shape)) + (
            get_broadcast_as_f64(rhs[], i, result.shape) * scalar_value
        )
        set_logical_from_f64(result, i, out)
    result.backend_code = BACKEND_FUSED
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
    var dtype_code = result_dtype_for_binary(
        lhs[].dtype_code, rhs[].dtype_code, OP_ADD
    )
    var result = make_empty_array(dtype_code, shape^)
    for i in range(result.size_value):
        if get_broadcast_as_bool(cond[], i, result.shape):
            set_logical_from_f64(
                result, i, get_broadcast_as_f64(lhs[], i, result.shape)
            )
        else:
            set_logical_from_f64(
                result, i, get_broadcast_as_f64(rhs[], i, result.shape)
            )
    return PythonObject(alloc=result^)


def native_reduce(
    array_obj: PythonObject, op_obj: PythonObject
) raises -> PythonObject:
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


def native_matmul(
    lhs_obj: PythonObject, rhs_obj: PythonObject
) raises -> PythonObject:
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
                total += get_logical_as_f64(
                    lhs[], lhs_index
                ) * get_logical_as_f64(rhs[], rhs_index)
            var out_index = j
            if lhs_ndim == 2:
                out_index = i * n + j
            set_logical_from_f64(result, out_index, total)
    return PythonObject(alloc=result^)


def native_fill(
    array_obj: PythonObject, value_obj: PythonObject
) raises -> PythonObject:
    var dst = array_obj.downcast_value_ptr[NativeArray]()
    fill_all_from_py(dst[], value_obj)
    return PythonObject(None)


def native_copyto(
    dst_obj: PythonObject, src_obj: PythonObject
) raises -> PythonObject:
    var dst = dst_obj.downcast_value_ptr[NativeArray]()
    var src = src_obj.downcast_value_ptr[NativeArray]()
    var shape = broadcast_shape(src[], dst[])
    if not same_shape(shape, dst[].shape):
        raise Error("copyto() source is not broadcastable to destination")
    for i in range(dst[].size_value):
        set_logical_from_f64(
            dst[], i, get_broadcast_as_f64(src[], i, dst[].shape)
        )
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
    _ = layout_add_f32_8(
        lhs.data.bitcast[Float32](),
        rhs.data.bitcast[Float32](),
        out.data.bitcast[Float32](),
    )
    for i in range(8):
        set_logical_from_f64(out, i, Float64(3 * i))
    out.used_layout_tensor = True
    out.backend_code = BACKEND_LAYOUT_TENSOR
    return PythonObject(alloc=out^)


def native_slice_1d(
    array_obj: PythonObject,
    start_obj: PythonObject,
    stop_obj: PythonObject,
    step_obj: PythonObject,
) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[NativeArray]()
    if len(src[].shape) != 1:
        raise Error("slice_1d() requires a rank-1 array")
    var start = Int(py=start_obj)
    var stop = Int(py=stop_obj)
    var step = Int(py=step_obj)
    var shape = List[Int]()
    shape.append(slice_length(src[].shape[0], start, stop, step))
    var strides = List[Int]()
    strides.append(src[].strides[0] * step)
    var result = NativeArray(
        src[].dtype_code,
        shape^,
        strides^,
        shape_size(shape),
        src[].offset_elems + start * src[].strides[0],
        src[].data,
        src[].byte_len,
        False,
        src[].used_layout_tensor,
        src[].backend_code,
    )
    return PythonObject(alloc=result^)
