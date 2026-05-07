from std.collections import List
from std.math import cos, exp, isinf, isnan, log, nan, sin
from std.python import PythonObject

from domain import (
    BACKEND_FUSED,
    DTYPE_BOOL,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT64,
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
    dtype_alignment,
    dtype_can_cast,
    dtype_item_size,
    dtype_kind_code,
    dtype_promote_types,
)
from elementwise import (
    apply_binary_f64,
    maybe_argmax_contiguous,
    maybe_binary_contiguous,
    maybe_binary_same_shape_contiguous,
    maybe_binary_scalar_value_contiguous,
    is_contiguous_float_array,
    lu_det_into,
    lu_inverse_into,
    lu_solve_into,
    maybe_matmul_contiguous,
    maybe_reduce_contiguous,
    maybe_sin_add_mul_contiguous,
    maybe_unary_contiguous,
)
from array import (
    Array,
    broadcast_shape,
    cast_copy_array,
    clone_int_list,
    copy_c_contiguous,
    fill_all_from_py,
    get_broadcast_as_bool,
    get_broadcast_as_f64,
    get_physical_as_f64,
    get_physical_bool,
    get_physical_i64,
    get_logical_as_f64,
    int_list_from_py,
    is_c_contiguous,
    item_size,
    make_c_strides,
    make_empty_array,
    make_external_array,
    make_view_array,
    result_dtype_for_binary,
    result_dtype_for_linalg,
    result_dtype_for_linalg_binary,
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


# Python-callable entrypoints.
# Includes Storage -> Shape -> Backend FFI imports from submodules.
def empty_ops(shape_obj: PythonObject, dtype_obj: PythonObject) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var shape = int_list_from_py(shape_obj)
    var result = make_empty_array(dtype_code, shape^)
    return PythonObject(alloc=result^)


def full_ops(shape_obj: PythonObject, value_obj: PythonObject, dtype_obj: PythonObject) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var shape = int_list_from_py(shape_obj)
    var result = make_empty_array(dtype_code, shape^)
    fill_all_from_py(result, value_obj)
    return PythonObject(alloc=result^)


def from_flat_ops(values_obj: PythonObject, shape_obj: PythonObject, dtype_obj: PythonObject) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var shape = int_list_from_py(shape_obj)
    var result = make_empty_array(dtype_code, shape^)
    if len(values_obj) != result.size_value:
        raise Error("flat value count does not match shape")
    for i in range(result.size_value):
        set_logical_from_py(result, i, values_obj[i])
    return PythonObject(alloc=result^)


def from_external_ops(
    data_address_obj: PythonObject,
    shape_obj: PythonObject,
    strides_obj: PythonObject,
    dtype_obj: PythonObject,
    byte_len_obj: PythonObject,
) raises -> PythonObject:
    var address = Int(py=data_address_obj)
    if address == 0:
        raise Error("array interface data pointer is null")
    var shape = int_list_from_py(shape_obj)
    var strides = int_list_from_py(strides_obj)
    var data = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=address)
    var result = make_external_array(Int(py=dtype_obj), shape^, strides^, 0, data, Int(py=byte_len_obj))
    return PythonObject(alloc=result^)


def copy_from_external_ops(
    data_address_obj: PythonObject,
    shape_obj: PythonObject,
    strides_obj: PythonObject,
    dtype_obj: PythonObject,
    byte_len_obj: PythonObject,
) raises -> PythonObject:
    # One-shot copy: build an external view of `address` with the supplied
    # shape/strides, then materialize a c-contiguous copy. Saves an FFI
    # round-trip vs calling `from_external_ops` then `materialize_c_contiguous_ops`
    # back-to-back. Hits the memcpy fast path when the source is already
    # c-contig, otherwise the elementwise walk in `copy_c_contiguous`.
    var address = Int(py=data_address_obj)
    if address == 0:
        raise Error("array interface data pointer is null")
    var shape = int_list_from_py(shape_obj)
    var strides = int_list_from_py(strides_obj)
    var data = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=address)
    var external = make_external_array(Int(py=dtype_obj), shape^, strides^, 0, data, Int(py=byte_len_obj))
    var result = copy_c_contiguous(external)
    return PythonObject(alloc=result^)


def arange_ops(
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


def linspace_ops(
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


def reshape_ops(array_obj: PythonObject, shape_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    if not is_c_contiguous(src[]):
        raise Error("reshape() only supports c-contiguous arrays for now")
    var shape = int_list_from_py(shape_obj)
    var new_size = shape_size(shape)
    if new_size != src[].size_value:
        raise Error("cannot reshape array to requested size")
    var strides = make_c_strides(shape)
    var result = make_view_array(src[], shape^, strides^, src[].size_value, src[].offset_elems)
    return PythonObject(alloc=result^)


def transpose_ops(array_obj: PythonObject, axes_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
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
    var result = make_view_array(src[], shape^, strides^, src[].size_value, src[].offset_elems)
    return PythonObject(alloc=result^)


def transpose_full_reverse_ops(
    array_obj: PythonObject,
) raises -> PythonObject:
    # Fast path for `.T` on rank>=2: reverse every axis without crossing
    # Python boundaries for an axes tuple. Avoids `int_list_from_py` and
    # the per-axis bounds check; `make_view_array` validates shape/strides
    # match.
    var src = array_obj.downcast_value_ptr[Array]()
    var ndim = len(src[].shape)
    var shape = List[Int]()
    var strides = List[Int]()
    for i in range(ndim - 1, -1, -1):
        shape.append(src[].shape[i])
        strides.append(src[].strides[i])
    var result = make_view_array(src[], shape^, strides^, src[].size_value, src[].offset_elems)
    return PythonObject(alloc=result^)


def slice_ops(
    array_obj: PythonObject,
    starts_obj: PythonObject,
    stops_obj: PythonObject,
    steps_obj: PythonObject,
    drops_obj: PythonObject,
) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
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
            shape.append(slice_length(src[].shape[axis], starts[axis], stops[axis], steps[axis]))
            strides.append(src[].strides[axis] * steps[axis])
    var result = make_view_array(src[], shape^, strides^, shape_size(shape), offset)
    return PythonObject(alloc=result^)


def broadcast_to_ops(array_obj: PythonObject, shape_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
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
    var result = make_view_array(src[], shape^, strides^, shape_size(shape), src[].offset_elems)
    return PythonObject(alloc=result^)


def expand_dims_ops(array_obj: PythonObject, axis_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var axis = Int(py=axis_obj)
    if axis < 0 or axis > len(src[].shape):
        raise Error("axis out of bounds")
    var shape = List[Int]()
    var strides = List[Int]()
    for i in range(axis):
        shape.append(src[].shape[i])
        strides.append(src[].strides[i])
    shape.append(1)
    strides.append(0)
    for i in range(axis, len(src[].shape)):
        shape.append(src[].shape[i])
        strides.append(src[].strides[i])
    var result = make_view_array(src[], shape^, strides^, src[].size_value, src[].offset_elems)
    return PythonObject(alloc=result^)


def astype_ops(array_obj: PythonObject, dtype_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var dtype_code = Int(py=dtype_obj)
    var result = cast_copy_array(src[], dtype_code)
    return PythonObject(alloc=result^)


def materialize_c_contiguous_ops(
    array_obj: PythonObject,
) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var result = copy_c_contiguous(src[])
    return PythonObject(alloc=result^)


def normalize_axis_ops(axis_value: Int, ndim: Int, name: String) raises -> Int:
    var axis = axis_value
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise Error(name, " axis out of bounds")
    return axis


@fieldwise_init
struct DiagonalMetadata(ImplicitlyCopyable, Movable, Writable):
    var length: Int
    var offset: Int
    var stride: Int


def diagonal_metadata_ops(src: Array, offset: Int, axis1: Int, axis2: Int) raises -> DiagonalMetadata:
    if len(src.shape) != 2:
        raise Error("diagonal() and trace() currently require rank-2 arrays")
    if axis1 == axis2:
        raise Error("diagonal axes must be different")
    var rows = src.shape[axis1]
    var cols = src.shape[axis2]
    var row_start = 0
    var col_start = 0
    if offset >= 0:
        col_start = offset
    else:
        row_start = -offset
    var diag_len = 0
    if row_start < rows and col_start < cols:
        var rows_left = rows - row_start
        var cols_left = cols - col_start
        diag_len = rows_left
        if cols_left < diag_len:
            diag_len = cols_left
    var diag_offset = src.offset_elems + row_start * src.strides[axis1] + col_start * src.strides[axis2]
    var diag_stride = src.strides[axis1] + src.strides[axis2]
    return DiagonalMetadata(diag_len, diag_offset, diag_stride)


def diagonal_ops(
    array_obj: PythonObject,
    offset_obj: PythonObject,
    axis1_obj: PythonObject,
    axis2_obj: PythonObject,
) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var axis1 = normalize_axis_ops(Int(py=axis1_obj), len(src[].shape), "axis1")
    var axis2 = normalize_axis_ops(Int(py=axis2_obj), len(src[].shape), "axis2")
    var metadata = diagonal_metadata_ops(src[], Int(py=offset_obj), axis1, axis2)
    var shape = List[Int]()
    shape.append(metadata.length)
    var strides = List[Int]()
    strides.append(metadata.stride)
    var result = make_view_array(src[], shape^, strides^, metadata.length, metadata.offset)
    return PythonObject(alloc=result^)


def trace_ops(
    array_obj: PythonObject,
    offset_obj: PythonObject,
    axis1_obj: PythonObject,
    axis2_obj: PythonObject,
    dtype_obj: PythonObject,
) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var axis1 = normalize_axis_ops(Int(py=axis1_obj), len(src[].shape), "axis1")
    var axis2 = normalize_axis_ops(Int(py=axis2_obj), len(src[].shape), "axis2")
    var metadata = diagonal_metadata_ops(src[], Int(py=offset_obj), axis1, axis2)
    var diag_len = metadata.length
    var diag_offset = metadata.offset
    var diag_stride = metadata.stride
    var shape = List[Int]()
    var dtype_code = Int(py=dtype_obj)
    if dtype_code < 0:
        dtype_code = result_dtype_for_reduction(src[].dtype_code, REDUCE_SUM)
    var result = make_empty_array(dtype_code, shape^)
    if src[].dtype_code == DTYPE_INT64:
        var acc = Int64(0)
        for i in range(diag_len):
            acc += get_physical_i64(src[], diag_offset + i * diag_stride)
        set_logical_from_i64(result, 0, acc)
    elif src[].dtype_code == DTYPE_BOOL:
        var acc = Int64(0)
        for i in range(diag_len):
            if get_physical_bool(src[], diag_offset + i * diag_stride):
                acc += 1
        set_logical_from_i64(result, 0, acc)
    else:
        var acc = 0.0
        for i in range(diag_len):
            acc += get_physical_as_f64(src[], diag_offset + i * diag_stride)
        set_logical_from_f64(result, 0, acc)
    return PythonObject(alloc=result^)


def unary_ops(array_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
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
            if isnan(value):
                pass
            elif isinf(value):
                if value < 0.0:
                    value = nan[DType.float64]()
            else:
                value = log(value)
        else:
            raise Error("unknown unary op")
        set_logical_from_f64(result, i, value)
    return PythonObject(alloc=result^)


def binary_dispatch_ops(lhs: Array, rhs: Array, op: Int) raises -> Array:
    # Internal binary_ops dispatch core. Allocates the result and runs the
    # contiguous / strided fast paths; if all fail, falls back to the
    # broadcast-divmod walker. Used by both the Python `binary_ops` function and
    # the Array.add/sub/mul/div method shortcuts.
    var dtype_code = result_dtype_for_binary(lhs.dtype_code, rhs.dtype_code, op)
    var same = same_shape(lhs.shape, rhs.shape)
    var shape: List[Int]
    if same:
        shape = clone_int_list(lhs.shape)
    else:
        shape = broadcast_shape(lhs, rhs)
    var result = make_empty_array(dtype_code, shape^)
    if same and maybe_binary_same_shape_contiguous(lhs, rhs, result, op):
        return result^
    if maybe_binary_contiguous(lhs, rhs, result, op):
        return result^
    for i in range(result.size_value):
        var lval = get_broadcast_as_f64(lhs, i, result.shape)
        var rval = get_broadcast_as_f64(rhs, i, result.shape)
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
    return result^


def binary_op_method_ops(py_self: PythonObject, other_obj: PythonObject, op: Int) raises -> PythonObject:
    # Method-style entrypoint shared by Array.add/sub/mul/div. The Mojo
    # PythonObject method dispatch is measurably tighter than def_function
    # for arg-marshal-bound calls; per-op trampolines (add_method_py etc.)
    # bake `op` into the dispatch so Python doesn't need to pass it.
    var self_ptr = py_self.downcast_value_ptr[Array]()
    var other_ptr = other_obj.downcast_value_ptr[Array]()
    var result = binary_dispatch_ops(self_ptr[], other_ptr[], op)
    return PythonObject(alloc=result^)


def array_add_method_ops(py_self: PythonObject, other_obj: PythonObject) raises -> PythonObject:
    return binary_op_method_ops(py_self, other_obj, OP_ADD)


def array_sub_method_ops(py_self: PythonObject, other_obj: PythonObject) raises -> PythonObject:
    return binary_op_method_ops(py_self, other_obj, OP_SUB)


def array_mul_method_ops(py_self: PythonObject, other_obj: PythonObject) raises -> PythonObject:
    return binary_op_method_ops(py_self, other_obj, OP_MUL)


def array_div_method_ops(py_self: PythonObject, other_obj: PythonObject) raises -> PythonObject:
    return binary_op_method_ops(py_self, other_obj, OP_DIV)


def array_matmul_method_ops(py_self: PythonObject, other_obj: PythonObject) raises -> PythonObject:
    var self_ptr = py_self.downcast_value_ptr[Array]()
    var other_ptr = other_obj.downcast_value_ptr[Array]()
    var lhs_ndim = len(self_ptr[].shape)
    var rhs_ndim = len(other_ptr[].shape)
    if lhs_ndim < 1 or lhs_ndim > 2 or rhs_ndim < 1 or rhs_ndim > 2:
        raise Error("matmul() only supports 1d and 2d arrays")
    var m = 1
    var k_lhs = self_ptr[].shape[0]
    if lhs_ndim == 2:
        m = self_ptr[].shape[0]
        k_lhs = self_ptr[].shape[1]
    var k_rhs = other_ptr[].shape[0]
    var n = 1
    if rhs_ndim == 2:
        n = other_ptr[].shape[1]
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
        result_dtype_for_binary(self_ptr[].dtype_code, other_ptr[].dtype_code, OP_MUL),
        out_shape^,
    )
    if lhs_ndim == 1 and rhs_ndim == 1:
        var total = 0.0
        for k in range(k_lhs):
            total += get_logical_as_f64(self_ptr[], k) * get_logical_as_f64(other_ptr[], k)
        set_logical_from_f64(result, 0, total)
        return PythonObject(alloc=result^)
    if maybe_matmul_contiguous(self_ptr[], other_ptr[], result, m, n, k_lhs):
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
                total += get_logical_as_f64(self_ptr[], lhs_index) * get_logical_as_f64(other_ptr[], rhs_index)
            var out_index = j
            if lhs_ndim == 2:
                out_index = i * n + j
            set_logical_from_f64(result, out_index, total)
    return PythonObject(alloc=result^)


def binary_ops(lhs_obj: PythonObject, rhs_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
    var op = Int(py=op_obj)
    var result = binary_dispatch_ops(lhs[], rhs[], op)
    return PythonObject(alloc=result^)


def binary_into_ops(
    dst_obj: PythonObject,
    lhs_obj: PythonObject,
    rhs_obj: PythonObject,
    op_obj: PythonObject,
) raises -> PythonObject:
    var dst = dst_obj.downcast_value_ptr[Array]()
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
    var op = Int(py=op_obj)
    var dtype_code = result_dtype_for_binary(lhs[].dtype_code, rhs[].dtype_code, op)
    if dst[].dtype_code != dtype_code:
        raise Error("out dtype does not match binary result dtype")
    if same_shape(lhs[].shape, rhs[].shape) and same_shape(lhs[].shape, dst[].shape):
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


def binary_scalar_ops(
    array_obj: PythonObject,
    scalar_obj: PythonObject,
    scalar_dtype_obj: PythonObject,
    op_obj: PythonObject,
    scalar_on_left_obj: PythonObject,
) raises -> PythonObject:
    var array = array_obj.downcast_value_ptr[Array]()
    var scalar_dtype = Int(py=scalar_dtype_obj)
    var op = Int(py=op_obj)
    var scalar_on_left = Bool(py=scalar_on_left_obj)
    var shape = clone_int_list(array[].shape)
    var dtype_code = result_dtype_for_binary(array[].dtype_code, scalar_dtype, op)
    var result = make_empty_array(dtype_code, shape^)
    var scalar_value = scalar_py_as_f64(scalar_obj, scalar_dtype)
    if maybe_binary_scalar_value_contiguous(array[], scalar_value, result, op, scalar_on_left):
        return PythonObject(alloc=result^)
    for i in range(result.size_value):
        var lhs = get_logical_as_f64(array[], i)
        var rhs = scalar_value
        if scalar_on_left:
            lhs = scalar_value
            rhs = get_logical_as_f64(array[], i)
        set_logical_from_f64(result, i, apply_binary_f64(lhs, rhs, op))
    return PythonObject(alloc=result^)


def result_dtype_for_unary_py_ops(dtype_obj: PythonObject) raises -> PythonObject:
    return PythonObject(result_dtype_for_unary(Int(py=dtype_obj)))


def result_dtype_for_binary_py_ops(
    lhs_dtype_obj: PythonObject,
    rhs_dtype_obj: PythonObject,
    op_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(result_dtype_for_binary(Int(py=lhs_dtype_obj), Int(py=rhs_dtype_obj), Int(py=op_obj)))


def result_dtype_for_reduction_py_ops(dtype_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    return PythonObject(result_dtype_for_reduction(Int(py=dtype_obj), Int(py=op_obj)))


def dtype_item_size_py_ops(dtype_obj: PythonObject) raises -> PythonObject:
    return PythonObject(dtype_item_size(Int(py=dtype_obj)))


def dtype_alignment_py_ops(dtype_obj: PythonObject) raises -> PythonObject:
    return PythonObject(dtype_alignment(Int(py=dtype_obj)))


def dtype_kind_code_py_ops(dtype_obj: PythonObject) raises -> PythonObject:
    return PythonObject(dtype_kind_code(Int(py=dtype_obj)))


def dtype_promote_types_py_ops(lhs_dtype_obj: PythonObject, rhs_dtype_obj: PythonObject) raises -> PythonObject:
    return PythonObject(dtype_promote_types(Int(py=lhs_dtype_obj), Int(py=rhs_dtype_obj)))


def dtype_can_cast_py_ops(
    from_dtype_obj: PythonObject,
    to_dtype_obj: PythonObject,
    casting_obj: PythonObject,
) raises -> PythonObject:
    return PythonObject(dtype_can_cast(Int(py=from_dtype_obj), Int(py=to_dtype_obj), Int(py=casting_obj)))


def sin_add_mul_ops(
    lhs_obj: PythonObject,
    rhs_obj: PythonObject,
    scalar_obj: PythonObject,
    scalar_dtype_obj: PythonObject,
) raises -> PythonObject:
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
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
        if maybe_sin_add_mul_contiguous(lhs[], rhs[], scalar_value, fast_result):
            return PythonObject(alloc=fast_result^)
    var shape = broadcast_shape(lhs[], rhs[])
    var rhs_mul_dtype = result_dtype_for_binary(rhs[].dtype_code, scalar_dtype, OP_MUL)
    var dtype_code = result_dtype_for_binary(result_dtype_for_unary(lhs[].dtype_code), rhs_mul_dtype, OP_ADD)
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


def where_ops(cond_obj: PythonObject, lhs_obj: PythonObject, rhs_obj: PythonObject) raises -> PythonObject:
    var cond = cond_obj.downcast_value_ptr[Array]()
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
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


def reduce_ops(array_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
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


def matmul_ops(lhs_obj: PythonObject, rhs_obj: PythonObject) raises -> PythonObject:
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
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


def solve_ops(a_obj: PythonObject, b_obj: PythonObject) raises -> PythonObject:
    var a = a_obj.downcast_value_ptr[Array]()
    var b = b_obj.downcast_value_ptr[Array]()
    if len(a[].shape) != 2 or a[].shape[0] != a[].shape[1]:
        raise Error("linalg.solve_ops() requires a square rank-2 coefficient matrix")
    var n = a[].shape[0]
    var out_shape = List[Int]()
    if len(b[].shape) == 1:
        if b[].shape[0] != n:
            raise Error("linalg.solve() right-hand side shape mismatch")
        out_shape.append(n)
    elif len(b[].shape) == 2:
        if b[].shape[0] != n:
            raise Error("linalg.solve() right-hand side shape mismatch")
        out_shape.append(n)
        out_shape.append(b[].shape[1])
    else:
        raise Error("linalg.solve() right-hand side must be rank 1 or rank 2")
    var result = make_empty_array(
        result_dtype_for_linalg_binary(a[].dtype_code, b[].dtype_code),
        out_shape^,
    )
    lu_solve_into(a[], b[], result)
    return PythonObject(alloc=result^)


def inv_ops(array_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 2 or src[].shape[0] != src[].shape[1]:
        raise Error("linalg.inv() requires a square rank-2 matrix")
    var shape = List[Int]()
    shape.append(src[].shape[0])
    shape.append(src[].shape[1])
    var result = make_empty_array(result_dtype_for_linalg(src[].dtype_code), shape^)
    lu_inverse_into(src[], result)
    return PythonObject(alloc=result^)


def det_ops(array_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var shape = List[Int]()
    var result = make_empty_array(result_dtype_for_linalg(src[].dtype_code), shape^)
    lu_det_into(src[], result)
    return PythonObject(alloc=result^)


def fill_ops(array_obj: PythonObject, value_obj: PythonObject) raises -> PythonObject:
    var dst = array_obj.downcast_value_ptr[Array]()
    fill_all_from_py(dst[], value_obj)
    return PythonObject(None)


def copyto_ops(dst_obj: PythonObject, src_obj: PythonObject) raises -> PythonObject:
    var dst = dst_obj.downcast_value_ptr[Array]()
    var src = src_obj.downcast_value_ptr[Array]()
    var shape = broadcast_shape(src[], dst[])
    if not same_shape(shape, dst[].shape):
        raise Error("copyto() source is not broadcastable to destination")
    for i in range(dst[].size_value):
        set_logical_from_f64(dst[], i, get_broadcast_as_f64(src[], i, dst[].shape))
    return PythonObject(None)


def slice_1d_ops(
    array_obj: PythonObject,
    start_obj: PythonObject,
    stop_obj: PythonObject,
    step_obj: PythonObject,
) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 1:
        raise Error("slice_1d() requires a rank-1 array")
    var start = Int(py=start_obj)
    var stop = Int(py=stop_obj)
    var step = Int(py=step_obj)
    var shape = List[Int]()
    shape.append(slice_length(src[].shape[0], start, stop, step))
    var strides = List[Int]()
    strides.append(src[].strides[0] * step)
    var result = make_view_array(
        src[],
        shape^,
        strides^,
        shape_size(shape),
        src[].offset_elems + start * src[].strides[0],
    )
    return PythonObject(alloc=result^)
