from std.collections import List
from std.math import cos, exp, isinf, isnan, log, nan, sin
from std.python import PythonObject

from domain import (
    BACKEND_FUSED,
    CMP_EQ,
    CMP_GE,
    CMP_GT,
    CMP_LE,
    CMP_LT,
    CMP_NE,
    DTYPE_BOOL,
    DTYPE_COMPLEX128,
    DTYPE_COMPLEX64,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT64,
    LOGIC_AND,
    LOGIC_OR,
    LOGIC_XOR,
    OP_ADD,
    OP_DIV,
    OP_MUL,
    OP_SUB,
    PRED_ISFINITE,
    PRED_ISINF,
    PRED_ISNAN,
    PRED_SIGNBIT,
    REDUCE_ALL,
    REDUCE_ANY,
    REDUCE_ARGMAX,
    REDUCE_ARGMIN,
    REDUCE_MEAN,
    REDUCE_MAX,
    REDUCE_MIN,
    REDUCE_PROD,
    REDUCE_SUM,
    UNARY_CBRT,
    UNARY_COS,
    UNARY_COSH,
    UNARY_EXP,
    UNARY_EXP2,
    UNARY_EXPM1,
    UNARY_LOG,
    UNARY_LOG10,
    UNARY_LOG1P,
    UNARY_LOG2,
    UNARY_RECIPROCAL,
    UNARY_SIN,
    UNARY_SINH,
    UNARY_SQRT,
    UNARY_TAN,
    UNARY_TANH,
    dtype_alignment,
    dtype_can_cast,
    dtype_item_size,
    dtype_kind_code,
    dtype_promote_types,
)
from elementwise import (
    apply_binary_f64,
    apply_unary_f64,
    lapack_cholesky_f32_into,
    lapack_cholesky_f64_into,
    lapack_eig_f32_real_into,
    lapack_eig_f64_real_into,
    lapack_eigh_f32_into,
    lapack_eigh_f64_into,
    lapack_lstsq_f32_into,
    lapack_lstsq_f64_into,
    lapack_qr_r_only_f32_into,
    lapack_qr_r_only_f64_into,
    lapack_qr_reduced_f32_into,
    lapack_qr_reduced_f64_into,
    lapack_svd_f32_into,
    lapack_svd_f64_into,
    maybe_argmax_contiguous,
    maybe_binary_contiguous,
    maybe_binary_same_shape_contiguous,
    maybe_binary_scalar_value_contiguous,
    maybe_unary_preserve_contiguous,
    is_contiguous_float_array,
    lu_det_into,
    lu_inverse_into,
    lu_solve_into,
    maybe_matmul_contiguous,
    maybe_binary_strided_typed,
    maybe_reduce_contiguous,
    maybe_reduce_strided_typed,
    maybe_sin_add_mul_contiguous,
    maybe_unary_contiguous,
    maybe_unary_rank2_strided,
)
from array import (
    Array,
    array_with_layout,
    as_broadcast_layout,
    as_layout,
    broadcast_shape,
    cast_copy_array,
    clone_int_list,
    contiguous_f32_ptr,
    contiguous_f64_ptr,
    contiguous_i64_ptr,
    copy_c_contiguous,
    fill_all_from_py,
    get_physical_as_f64,
    get_physical_bool,
    get_physical_c128_imag,
    get_physical_c128_real,
    get_physical_c64_imag,
    get_physical_c64_real,
    get_physical_i64,
    get_logical_as_f64,
    int_list_from_py,
    is_c_contiguous,
    item_size,
    make_c_strides,
    make_empty_array,
    make_external_array,
    make_view_array,
    physical_offset,
    result_dtype_for_binary,
    result_dtype_for_linalg,
    result_dtype_for_linalg_binary,
    result_dtype_for_reduction,
    result_dtype_for_unary,
    result_dtype_for_unary_preserve,
    same_shape,
    scalar_py_as_f64,
    set_logical_from_f64,
    set_logical_from_i64,
    set_logical_from_py,
    set_physical_c128,
    set_physical_c64,
    set_physical_from_f64,
    shape_size,
    slice_length,
)
from cute.iter import LayoutIter, MultiLayoutIter
from cute.layout import Layout, make_layout_row_major
from cute.functional import select as cute_select
from cute.int_tuple import IntTuple


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


def full_like_ops(
    prototype_obj: PythonObject, value_obj: PythonObject, dtype_obj: PythonObject
) raises -> PythonObject:
    var src = prototype_obj.downcast_value_ptr[Array]()
    var dtype_code = Int(py=dtype_obj)
    if dtype_code < 0:
        dtype_code = src[].dtype_code
    var shape = clone_int_list(src[].shape)
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


def indices_ops(dimensions_obj: PythonObject, dtype_obj: PythonObject) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    if dtype_code != DTYPE_INT64:
        raise Error("indices() native kernel only supports int64")
    var dims = int_list_from_py(dimensions_obj)
    var rank = len(dims)
    var out_shape = List[Int]()
    out_shape.append(rank)
    var strides = List[Int]()
    for axis in range(rank):
        if dims[axis] < 0:
            raise Error("indices() dimensions must be non-negative")
        out_shape.append(dims[axis])
        strides.append(1)
    var plane_size = 1
    for axis in range(rank - 1, -1, -1):
        strides[axis] = plane_size
        plane_size *= dims[axis]
    var result = make_empty_array(DTYPE_INT64, out_shape^)
    if rank == 0 or plane_size == 0:
        return PythonObject(alloc=result^)
    var out = contiguous_i64_ptr(result)
    for axis in range(rank):
        var stride = strides[axis]
        var dim = dims[axis]
        var base = axis * plane_size
        for logical in range(plane_size):
            out[base + logical] = Int64((logical // stride) % dim)
    return PythonObject(alloc=result^)


def reshape_ops(array_obj: PythonObject, shape_obj: PythonObject) raises -> PythonObject:
    # Layout-algebra view: c-contig source → fresh row-major layout over
    # the new shape (no data movement). Non-contig source → materialize
    # a c-contig copy first, then reshape it. Matches numpy: reshape
    # returns a view when possible, a copy when not.
    var src = array_obj.downcast_value_ptr[Array]()
    var new_shape = int_list_from_py(shape_obj)
    var new_size = shape_size(new_shape)
    if new_size != src[].size_value:
        raise Error("cannot reshape array to requested size")
    if is_c_contiguous(src[]):
        var shape_tuple = IntTuple.flat(clone_int_list(new_shape))
        var new_layout = make_layout_row_major(shape_tuple^)
        var view = array_with_layout(src[], new_layout)
        return PythonObject(alloc=view^)
    var copied = copy_c_contiguous(src[])
    var shape_tuple = IntTuple.flat(clone_int_list(new_shape))
    var copy_layout = make_layout_row_major(shape_tuple^)
    var view = array_with_layout(copied, copy_layout)
    return PythonObject(alloc=view^)


def transpose_ops(array_obj: PythonObject, axes_obj: PythonObject) raises -> PythonObject:
    # Layout-algebra view: `select(L, axes)` permutes the top-level modes.
    # Equivalent to manual stride-list permutation but algebraically sourced.
    var src = array_obj.downcast_value_ptr[Array]()
    var axes = int_list_from_py(axes_obj)
    if len(axes) != len(src[].shape):
        raise Error("transpose() axes length must match ndim")
    for i in range(len(axes)):
        if axes[i] < 0 or axes[i] >= len(src[].shape):
            raise Error("transpose() axis out of bounds")
    var src_layout = as_layout(src[])
    var permuted = cute_select(src_layout, axes)
    var view = array_with_layout(src[], permuted)
    return PythonObject(alloc=view^)


def flip_ops(
    array_obj: PythonObject,
    axes_obj: PythonObject,
) raises -> PythonObject:
    # Flip the iteration order on each axis in `axes` by negating its
    # stride and shifting `offset_elems`. Pure view; no data movement.
    # An empty `axes` list flips every axis (numpy default).
    var src = array_obj.downcast_value_ptr[Array]()
    var ndim = len(src[].shape)
    var shape = clone_int_list(src[].shape)
    var strides = clone_int_list(src[].strides)
    var offset = src[].offset_elems

    var flip_all = len(axes_obj) == 0
    var seen = List[Bool]()
    for _ in range(ndim):
        seen.append(False)

    if flip_all:
        for i in range(ndim):
            seen[i] = True
    else:
        for k in range(len(axes_obj)):
            var ax = Int(py=axes_obj[k])
            if ax < 0:
                ax += ndim
            if ax < 0 or ax >= ndim:
                raise Error("flip() axis out of bounds")
            if seen[ax]:
                raise Error("flip() repeated axis")
            seen[ax] = True

    for i in range(ndim):
        if seen[i]:
            offset += (shape[i] - 1) * strides[i]
            strides[i] = -strides[i]

    var result = make_view_array(src[], shape^, strides^, src[].size_value, offset)
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
    # Layout-algebra view: build the broadcast layout (stride-zero
    # injection on size-1 / new outer dims) and materialize a view.
    var src = array_obj.downcast_value_ptr[Array]()
    var out_shape = int_list_from_py(shape_obj)
    var ndim_out = len(out_shape)
    var ndim_src = len(src[].shape)
    if ndim_src > ndim_out:
        raise Error("cannot broadcast to fewer dimensions")
    # Validate broadcast compatibility before delegating to the layout
    # builder — `as_broadcast_layout` injects stride-0 silently on
    # size-1 mismatches but doesn't catch hard incompatibilities.
    for out_axis in range(ndim_out):
        var src_axis = out_axis - (ndim_out - ndim_src)
        if src_axis >= 0:
            var src_dim = src[].shape[src_axis]
            var out_dim = out_shape[out_axis]
            if src_dim != out_dim and src_dim != 1:
                raise Error("shape is not broadcastable")
    var bcast_layout = as_broadcast_layout(src[], out_shape)
    var view = array_with_layout(src[], bcast_layout)
    return PythonObject(alloc=view^)


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
    # Complex transcendentals: sin/cos/exp/log/sqrt etc. via Euler identities.
    # Output dtype = input dtype (preserved by result_dtype_for_unary).
    if src[].dtype_code == DTYPE_COMPLEX64 or src[].dtype_code == DTYPE_COMPLEX128:
        for i in range(src[].size_value):
            var re = _complex_real(src[], i)
            var im = _complex_imag(src[], i)
            var out_re: Float64
            var out_im: Float64
            (out_re, out_im) = apply_unary_complex_f64(re, im, op)
            _complex_store(result, i, out_re, out_im)
        return PythonObject(alloc=result^)
    if maybe_unary_contiguous(src[], result, op):
        return PythonObject(alloc=result^)
    if maybe_unary_rank2_strided(src[], result, op):
        return PythonObject(alloc=result^)
    # Strided fallback: walk via LayoutIter so the divmod amortizes
    # across the iteration instead of paying physical_offset per element.
    var src_layout = as_layout(src[])
    var dst_layout = as_layout(result)
    var src_item = item_size(src[].dtype_code)
    var dst_item = item_size(result.dtype_code)
    var src_iter = LayoutIter(src_layout, src_item, src[].offset_elems * src_item)
    var dst_iter = LayoutIter(dst_layout, dst_item, result.offset_elems * dst_item)
    while src_iter.has_next():
        var value = get_physical_as_f64(src[], src_iter.element_index())
        var output = apply_unary_f64(value, op)
        set_physical_from_f64(result, dst_iter.element_index(), output)
        src_iter.step()
        dst_iter.step()
    return PythonObject(alloc=result^)


def apply_unary_complex_f64(re: Float64, im: Float64, op: Int) raises -> Tuple[Float64, Float64]:
    """Complex unary transcendentals via Euler identities. Operates on
    (re, im) Float64 pairs and returns the new pair. The python-level
    `unary_ops` walks complex arrays element-by-element through this.
    """
    from std.math import (
        atan2 as _atan2,
        cos as _cos,
        cosh as _cosh,
        exp as _exp,
        log as _log,
        sin as _sin,
        sinh as _sinh,
        sqrt as _sqrt,
    )

    if op == UNARY_EXP:
        # exp(a+bi) = exp(a) * (cos(b) + i sin(b))
        var ea = _exp(re)
        return (ea * _cos(im), ea * _sin(im))
    if op == UNARY_LOG:
        # log(z) = log|z| + i arg(z)
        var modulus = _sqrt(re * re + im * im)
        return (_log(modulus), _atan2(im, re))
    if op == UNARY_SIN:
        # sin(a+bi) = sin(a)cosh(b) + i cos(a)sinh(b)
        return (_sin(re) * _cosh(im), _cos(re) * _sinh(im))
    if op == UNARY_COS:
        # cos(a+bi) = cos(a)cosh(b) - i sin(a)sinh(b)
        return (_cos(re) * _cosh(im), -_sin(re) * _sinh(im))
    if op == UNARY_SINH:
        # sinh(a+bi) = sinh(a)cos(b) + i cosh(a)sin(b)
        return (_sinh(re) * _cos(im), _cosh(re) * _sin(im))
    if op == UNARY_COSH:
        # cosh(a+bi) = cosh(a)cos(b) + i sinh(a)sin(b)
        return (_cosh(re) * _cos(im), _sinh(re) * _sin(im))
    if op == UNARY_TANH:
        # tanh(z) = sinh(z) / cosh(z) — use Euler-form identities and divide.
        var s_re = _sinh(re) * _cos(im)
        var s_im = _cosh(re) * _sin(im)
        var c_re = _cosh(re) * _cos(im)
        var c_im = _sinh(re) * _sin(im)
        var denom = c_re * c_re + c_im * c_im
        return ((s_re * c_re + s_im * c_im) / denom, (s_im * c_re - s_re * c_im) / denom)
    if op == UNARY_TAN:
        # tan(z) = sin(z) / cos(z)
        var s_re = _sin(re) * _cosh(im)
        var s_im = _cos(re) * _sinh(im)
        var c_re = _cos(re) * _cosh(im)
        var c_im = -_sin(re) * _sinh(im)
        var denom = c_re * c_re + c_im * c_im
        return ((s_re * c_re + s_im * c_im) / denom, (s_im * c_re - s_re * c_im) / denom)
    if op == UNARY_SQRT:
        # sqrt(z): principal branch. z = r * exp(i*theta), sqrt(z) = sqrt(r) * exp(i*theta/2).
        var modulus = _sqrt(re * re + im * im)
        var arg = _atan2(im, re)
        var s = _sqrt(modulus)
        return (s * _cos(arg / 2.0), s * _sin(arg / 2.0))
    if op == UNARY_LOG2:
        # log2(z) = log(z) / log(2)
        var modulus = _sqrt(re * re + im * im)
        var ln2 = 0.6931471805599453
        return (_log(modulus) / ln2, _atan2(im, re) / ln2)
    if op == UNARY_LOG10:
        var modulus = _sqrt(re * re + im * im)
        var ln10 = 2.302585092994046
        return (_log(modulus) / ln10, _atan2(im, re) / ln10)
    if op == UNARY_LOG1P:
        # log(1 + z)
        var nre = 1.0 + re
        var modulus = _sqrt(nre * nre + im * im)
        return (_log(modulus), _atan2(im, nre))
    if op == UNARY_EXPM1:
        # exp(z) - 1 — use exp formula then subtract 1 from real part.
        var ea = _exp(re)
        return (ea * _cos(im) - 1.0, ea * _sin(im))
    if op == UNARY_RECIPROCAL:
        # 1 / (a + bi) = (a - bi) / (a² + b²)
        var denom = re * re + im * im
        return (re / denom, -im / denom)
    if op == UNARY_EXP2:
        # 2^z = exp(z * log(2))
        var ln2 = 0.6931471805599453
        var nre = re * ln2
        var nim = im * ln2
        var ea = _exp(nre)
        return (ea * _cos(nim), ea * _sin(nim))
    if op == UNARY_CBRT:
        # cbrt(z) — principal branch.
        var modulus = _sqrt(re * re + im * im)
        var arg = _atan2(im, re)
        var s = modulus.__pow__(1.0 / 3.0)
        return (s * _cos(arg / 3.0), s * _sin(arg / 3.0))
    raise Error("unary op not implemented for complex inputs")


def binary_dispatch_ops(lhs: Array, rhs: Array, op: Int) raises -> Array:
    # Internal binary_ops dispatch core. Allocates the result and runs the
    # contiguous / strided fast paths; if all fail, walks via
    # MultiLayoutIter so the broadcast divmod amortizes across the whole
    # iteration instead of paying per element.
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
    # Typed strided walker: same-dtype broadcasts skip the f64 dispatch
    # cascade entirely. Most ops in practice (arithmetic on matching
    # dtypes) hit this. Mixed-dtype broadcasts fall through.
    if maybe_binary_strided_typed(lhs, rhs, result, op):
        return result^
    # Strided fallback: cursor walk via MultiLayoutIter. Each operand
    # carries stride-zero modes for broadcast dimensions; the output is
    # contiguous (we just allocated it).
    var lhs_layout = as_broadcast_layout(lhs, result.shape)
    var rhs_layout = as_broadcast_layout(rhs, result.shape)
    var out_layout = as_layout(result)
    var item_lhs = item_size(lhs.dtype_code)
    var item_rhs = item_size(rhs.dtype_code)
    var item_out = item_size(result.dtype_code)
    var operand_layouts = List[Layout]()
    operand_layouts.append(lhs_layout^)
    operand_layouts.append(rhs_layout^)
    operand_layouts.append(out_layout^)
    var item_sizes = List[Int]()
    item_sizes.append(item_lhs)
    item_sizes.append(item_rhs)
    item_sizes.append(item_out)
    var base_offsets = List[Int]()
    base_offsets.append(lhs.offset_elems * item_lhs)
    base_offsets.append(rhs.offset_elems * item_rhs)
    base_offsets.append(result.offset_elems * item_out)
    var iter = MultiLayoutIter(result.shape, operand_layouts^, item_sizes^, base_offsets^)
    while iter.has_next():
        var lval = get_physical_as_f64(lhs, iter.element_index(0))
        var rval = get_physical_as_f64(rhs, iter.element_index(1))
        set_physical_from_f64(result, iter.element_index(2), apply_binary_f64(lval, rval, op))
        iter.step()
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
    var lhs_layout = as_broadcast_layout(lhs[], dst[].shape)
    var rhs_layout = as_broadcast_layout(rhs[], dst[].shape)
    var dst_layout = as_layout(dst[])
    var item_lhs = item_size(lhs[].dtype_code)
    var item_rhs = item_size(rhs[].dtype_code)
    var item_dst = item_size(dst[].dtype_code)
    var operand_layouts = List[Layout]()
    operand_layouts.append(lhs_layout^)
    operand_layouts.append(rhs_layout^)
    operand_layouts.append(dst_layout^)
    var item_sizes = List[Int]()
    item_sizes.append(item_lhs)
    item_sizes.append(item_rhs)
    item_sizes.append(item_dst)
    var base_offsets = List[Int]()
    base_offsets.append(lhs[].offset_elems * item_lhs)
    base_offsets.append(rhs[].offset_elems * item_rhs)
    base_offsets.append(dst[].offset_elems * item_dst)
    var iter = MultiLayoutIter(dst[].shape, operand_layouts^, item_sizes^, base_offsets^)
    while iter.has_next():
        var lval = get_physical_as_f64(lhs[], iter.element_index(0))
        var rval = get_physical_as_f64(rhs[], iter.element_index(1))
        set_physical_from_f64(dst[], iter.element_index(2), apply_binary_f64(lval, rval, op))
        iter.step()
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
    var src_layout = as_layout(array[])
    var dst_layout = as_layout(result)
    var src_item = item_size(array[].dtype_code)
    var dst_item = item_size(result.dtype_code)
    var src_iter = LayoutIter(src_layout, src_item, array[].offset_elems * src_item)
    var dst_iter = LayoutIter(dst_layout, dst_item, result.offset_elems * dst_item)
    while src_iter.has_next():
        var lhs = get_physical_as_f64(array[], src_iter.element_index())
        var rhs = scalar_value
        if scalar_on_left:
            lhs = scalar_value
            rhs = get_physical_as_f64(array[], src_iter.element_index())
        set_physical_from_f64(result, dst_iter.element_index(), apply_binary_f64(lhs, rhs, op))
        src_iter.step()
        dst_iter.step()
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
    var lhs_layout = as_broadcast_layout(lhs[], result.shape)
    var rhs_layout = as_broadcast_layout(rhs[], result.shape)
    var out_layout = as_layout(result)
    var item_lhs = item_size(lhs[].dtype_code)
    var item_rhs = item_size(rhs[].dtype_code)
    var item_out = item_size(result.dtype_code)
    var operand_layouts = List[Layout]()
    operand_layouts.append(lhs_layout^)
    operand_layouts.append(rhs_layout^)
    operand_layouts.append(out_layout^)
    var item_sizes = List[Int]()
    item_sizes.append(item_lhs)
    item_sizes.append(item_rhs)
    item_sizes.append(item_out)
    var base_offsets = List[Int]()
    base_offsets.append(lhs[].offset_elems * item_lhs)
    base_offsets.append(rhs[].offset_elems * item_rhs)
    base_offsets.append(result.offset_elems * item_out)
    var iter = MultiLayoutIter(result.shape, operand_layouts^, item_sizes^, base_offsets^)
    while iter.has_next():
        var output = sin(get_physical_as_f64(lhs[], iter.element_index(0))) + (
            get_physical_as_f64(rhs[], iter.element_index(1)) * scalar_value
        )
        set_physical_from_f64(result, iter.element_index(2), output)
        iter.step()
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
    var cond_layout = as_broadcast_layout(cond[], result.shape)
    var lhs_layout = as_broadcast_layout(lhs[], result.shape)
    var rhs_layout = as_broadcast_layout(rhs[], result.shape)
    var out_layout = as_layout(result)
    var item_cond = item_size(cond[].dtype_code)
    var item_lhs = item_size(lhs[].dtype_code)
    var item_rhs = item_size(rhs[].dtype_code)
    var item_out = item_size(result.dtype_code)
    var operand_layouts = List[Layout]()
    operand_layouts.append(cond_layout^)
    operand_layouts.append(lhs_layout^)
    operand_layouts.append(rhs_layout^)
    operand_layouts.append(out_layout^)
    var item_sizes = List[Int]()
    item_sizes.append(item_cond)
    item_sizes.append(item_lhs)
    item_sizes.append(item_rhs)
    item_sizes.append(item_out)
    var base_offsets = List[Int]()
    base_offsets.append(cond[].offset_elems * item_cond)
    base_offsets.append(lhs[].offset_elems * item_lhs)
    base_offsets.append(rhs[].offset_elems * item_rhs)
    base_offsets.append(result.offset_elems * item_out)
    var iter = MultiLayoutIter(result.shape, operand_layouts^, item_sizes^, base_offsets^)
    while iter.has_next():
        var cond_phys = iter.element_index(0)
        var picked: Float64
        if cond[].dtype_code == DTYPE_BOOL:
            picked = 1.0 if get_physical_bool(cond[], cond_phys) else 0.0
        else:
            picked = get_physical_as_f64(cond[], cond_phys)
        if picked != 0.0:
            set_physical_from_f64(result, iter.element_index(3), get_physical_as_f64(lhs[], iter.element_index(1)))
        else:
            set_physical_from_f64(result, iter.element_index(3), get_physical_as_f64(rhs[], iter.element_index(2)))
        iter.step()
    return PythonObject(alloc=result^)


def _reduce_strided_iter(src: Array) raises -> LayoutIter:
    """LayoutIter wrapping `src` for strided whole-array reductions.
    Caller drives `step()` and reads `element_index()`."""
    var src_layout = as_layout(src)
    var src_item = item_size(src.dtype_code)
    return LayoutIter(src_layout, src_item, src.offset_elems * src_item)


def reduce_ops(array_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var op = Int(py=op_obj)
    var shape = List[Int]()
    if op == REDUCE_ARGMAX or op == REDUCE_ARGMIN:
        var result = make_empty_array(DTYPE_INT64, shape^)
        if src[].size_value == 0:
            raise Error("argmax/argmin cannot reduce an empty array")
        if op == REDUCE_ARGMAX and maybe_argmax_contiguous(src[], result):
            return PythonObject(alloc=result^)
        var iter = _reduce_strided_iter(src[])
        var best_index = 0
        var best_value = get_physical_as_f64(src[], iter.element_index())
        iter.step()
        var i = 1
        while iter.has_next():
            var value = get_physical_as_f64(src[], iter.element_index())
            if op == REDUCE_ARGMAX:
                if value > best_value:
                    best_value = value
                    best_index = i
            else:
                if value < best_value:
                    best_value = value
                    best_index = i
            iter.step()
            i += 1
        set_logical_from_i64(result, 0, Int64(best_index))
        return PythonObject(alloc=result^)
    if op == REDUCE_ALL or op == REDUCE_ANY:
        var result = make_empty_array(DTYPE_BOOL, shape^)
        if src[].size_value == 0:
            # numpy: all() of empty → True; any() of empty → False.
            var v: Float64 = 1.0 if op == REDUCE_ALL else 0.0
            set_logical_from_f64(result, 0, v)
            return PythonObject(alloc=result^)
        var iter = _reduce_strided_iter(src[])
        if op == REDUCE_ALL:
            while iter.has_next():
                if get_physical_as_f64(src[], iter.element_index()) == 0.0:
                    set_logical_from_f64(result, 0, 0.0)
                    return PythonObject(alloc=result^)
                iter.step()
            set_logical_from_f64(result, 0, 1.0)
            return PythonObject(alloc=result^)
        # REDUCE_ANY
        while iter.has_next():
            if get_physical_as_f64(src[], iter.element_index()) != 0.0:
                set_logical_from_f64(result, 0, 1.0)
                return PythonObject(alloc=result^)
            iter.step()
        set_logical_from_f64(result, 0, 0.0)
        return PythonObject(alloc=result^)
    var result_dtype = result_dtype_for_reduction(src[].dtype_code, op)
    var result = make_empty_array(result_dtype, shape^)
    if src[].size_value == 0:
        # numpy: sum of empty → 0; prod of empty → 1; min/max raise.
        if op == REDUCE_SUM or op == REDUCE_MEAN:
            set_logical_from_f64(result, 0, 0.0)
            return PythonObject(alloc=result^)
        if op == REDUCE_PROD:
            set_logical_from_f64(result, 0, 1.0)
            return PythonObject(alloc=result^)
        raise Error("cannot reduce an empty array")
    if maybe_reduce_contiguous(src[], result, op):
        return PythonObject(alloc=result^)
    if maybe_reduce_strided_typed(src[], result, op):
        return PythonObject(alloc=result^)
    var iter = _reduce_strided_iter(src[])
    var acc: Float64
    if op == REDUCE_SUM or op == REDUCE_MEAN:
        acc = 0.0
        while iter.has_next():
            acc += get_physical_as_f64(src[], iter.element_index())
            iter.step()
        if op == REDUCE_MEAN:
            acc = acc / Float64(src[].size_value)
    elif op == REDUCE_PROD:
        acc = 1.0
        while iter.has_next():
            acc *= get_physical_as_f64(src[], iter.element_index())
            iter.step()
    elif op == REDUCE_MIN:
        acc = get_physical_as_f64(src[], iter.element_index())
        iter.step()
        while iter.has_next():
            var value = get_physical_as_f64(src[], iter.element_index())
            if value < acc:
                acc = value
            iter.step()
    elif op == REDUCE_MAX:
        acc = get_physical_as_f64(src[], iter.element_index())
        iter.step()
        while iter.has_next():
            var value = get_physical_as_f64(src[], iter.element_index())
            if value > acc:
                acc = value
            iter.step()
    else:
        raise Error("unknown reduction op")
    set_logical_from_f64(result, 0, acc)
    return PythonObject(alloc=result^)


def unary_preserve_ops(array_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    """Preserve-dtype unary ops (negate/abs/square/positive/floor/ceil/ trunc/rint/logical_not).
    Output dtype = input dtype (with bool→int64 promotion).
    Typed-vec contig fast path for f32/f64/i32/i64/u32/u64; other paths fall through to the f64 round-trip.
    """
    var src = array_obj.downcast_value_ptr[Array]()
    var op = Int(py=op_obj)
    var shape = clone_int_list(src[].shape)
    var dtype_code = result_dtype_for_unary_preserve(src[].dtype_code)
    var result = make_empty_array(dtype_code, shape^)
    if maybe_unary_preserve_contiguous(src[], result, op):
        return PythonObject(alloc=result^)
    var src_layout = as_layout(src[])
    var dst_layout = as_layout(result)
    var src_item = item_size(src[].dtype_code)
    var dst_item = item_size(result.dtype_code)
    var src_iter = LayoutIter(src_layout, src_item, src[].offset_elems * src_item)
    var dst_iter = LayoutIter(dst_layout, dst_item, result.offset_elems * dst_item)
    while src_iter.has_next():
        var value = get_physical_as_f64(src[], src_iter.element_index())
        var output = apply_unary_f64(value, op)
        set_physical_from_f64(result, dst_iter.element_index(), output)
        src_iter.step()
        dst_iter.step()
    return PythonObject(alloc=result^)


def compare_ops(
    lhs_obj: PythonObject,
    rhs_obj: PythonObject,
    op_obj: PythonObject,
) raises -> PythonObject:
    """Elementwise comparison; returns a bool array. Operands broadcast.
    Walks via MultiLayoutIter so the broadcast divmod amortizes."""
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
    var op = Int(py=op_obj)
    var same = same_shape(lhs[].shape, rhs[].shape)
    var shape: List[Int]
    if same:
        shape = clone_int_list(lhs[].shape)
    else:
        shape = broadcast_shape(lhs[], rhs[])
    var result = make_empty_array(DTYPE_BOOL, shape^)
    var lhs_layout = as_broadcast_layout(lhs[], result.shape)
    var rhs_layout = as_broadcast_layout(rhs[], result.shape)
    var out_layout = as_layout(result)
    var item_lhs = item_size(lhs[].dtype_code)
    var item_rhs = item_size(rhs[].dtype_code)
    var item_out = item_size(result.dtype_code)
    var operand_layouts = List[Layout]()
    operand_layouts.append(lhs_layout^)
    operand_layouts.append(rhs_layout^)
    operand_layouts.append(out_layout^)
    var item_sizes = List[Int]()
    item_sizes.append(item_lhs)
    item_sizes.append(item_rhs)
    item_sizes.append(item_out)
    var base_offsets = List[Int]()
    base_offsets.append(lhs[].offset_elems * item_lhs)
    base_offsets.append(rhs[].offset_elems * item_rhs)
    base_offsets.append(result.offset_elems * item_out)
    var iter = MultiLayoutIter(result.shape, operand_layouts^, item_sizes^, base_offsets^)
    while iter.has_next():
        var lval = get_physical_as_f64(lhs[], iter.element_index(0))
        var rval = get_physical_as_f64(rhs[], iter.element_index(1))
        var output: Bool
        if op == CMP_EQ:
            output = lval == rval
        elif op == CMP_NE:
            output = lval != rval
        elif op == CMP_LT:
            output = lval < rval
        elif op == CMP_LE:
            output = lval <= rval
        elif op == CMP_GT:
            output = lval > rval
        elif op == CMP_GE:
            output = lval >= rval
        else:
            raise Error("unknown comparison op")
        set_physical_from_f64(result, iter.element_index(2), 1.0 if output else 0.0)
        iter.step()
    return PythonObject(alloc=result^)


def logical_ops(
    lhs_obj: PythonObject,
    rhs_obj: PythonObject,
    op_obj: PythonObject,
) raises -> PythonObject:
    """Elementwise logical_and / or / xor. Operates on truthiness of any
    numeric input; result is bool. Walks via MultiLayoutIter."""
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
    var op = Int(py=op_obj)
    var same = same_shape(lhs[].shape, rhs[].shape)
    var shape: List[Int]
    if same:
        shape = clone_int_list(lhs[].shape)
    else:
        shape = broadcast_shape(lhs[], rhs[])
    var result = make_empty_array(DTYPE_BOOL, shape^)
    var lhs_layout = as_broadcast_layout(lhs[], result.shape)
    var rhs_layout = as_broadcast_layout(rhs[], result.shape)
    var out_layout = as_layout(result)
    var item_lhs = item_size(lhs[].dtype_code)
    var item_rhs = item_size(rhs[].dtype_code)
    var item_out = item_size(result.dtype_code)
    var operand_layouts = List[Layout]()
    operand_layouts.append(lhs_layout^)
    operand_layouts.append(rhs_layout^)
    operand_layouts.append(out_layout^)
    var item_sizes = List[Int]()
    item_sizes.append(item_lhs)
    item_sizes.append(item_rhs)
    item_sizes.append(item_out)
    var base_offsets = List[Int]()
    base_offsets.append(lhs[].offset_elems * item_lhs)
    base_offsets.append(rhs[].offset_elems * item_rhs)
    base_offsets.append(result.offset_elems * item_out)
    var iter = MultiLayoutIter(result.shape, operand_layouts^, item_sizes^, base_offsets^)
    while iter.has_next():
        var l_truthy = get_physical_as_f64(lhs[], iter.element_index(0)) != 0.0
        var r_truthy = get_physical_as_f64(rhs[], iter.element_index(1)) != 0.0
        var output: Bool
        if op == LOGIC_AND:
            output = l_truthy and r_truthy
        elif op == LOGIC_OR:
            output = l_truthy or r_truthy
        elif op == LOGIC_XOR:
            output = l_truthy != r_truthy
        else:
            raise Error("unknown logical op")
        set_physical_from_f64(result, iter.element_index(2), 1.0 if output else 0.0)
        iter.step()
    return PythonObject(alloc=result^)


def predicate_ops(array_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    """Unary predicate (isnan / isinf / isfinite / signbit). Returns
    bool. Walks via LayoutIter so the divmod amortizes across the
    iteration."""
    var src = array_obj.downcast_value_ptr[Array]()
    var op = Int(py=op_obj)
    var shape = clone_int_list(src[].shape)
    var result = make_empty_array(DTYPE_BOOL, shape^)
    var src_layout = as_layout(src[])
    var dst_layout = as_layout(result)
    var src_item = item_size(src[].dtype_code)
    var dst_item = item_size(result.dtype_code)
    var src_iter = LayoutIter(src_layout, src_item, src[].offset_elems * src_item)
    var dst_iter = LayoutIter(dst_layout, dst_item, result.offset_elems * dst_item)
    while src_iter.has_next():
        var value = get_physical_as_f64(src[], src_iter.element_index())
        var output: Bool
        if op == PRED_ISNAN:
            output = isnan(value)
        elif op == PRED_ISINF:
            output = isinf(value)
        elif op == PRED_ISFINITE:
            output = not (isnan(value) or isinf(value))
        elif op == PRED_SIGNBIT:
            # numpy.signbit returns True for -0.0, so use bitcast.
            output = value < 0.0
            if value == 0.0:
                # negative-zero → True via bitcast.
                var bits = SIMD[DType.float64, 1](value).cast[DType.uint64]()[0]
                output = (bits >> 63) != 0
        else:
            raise Error("unknown predicate op")
        set_physical_from_f64(result, dst_iter.element_index(), 1.0 if output else 0.0)
        src_iter.step()
        dst_iter.step()
    return PythonObject(alloc=result^)


def reduce_axis_ops(
    array_obj: PythonObject,
    op_obj: PythonObject,
    axis_obj: PythonObject,
    keepdims_obj: PythonObject,
) raises -> PythonObject:
    """Axis-aware reduction. `axis_obj` is a Python tuple/list of ints;
    `keepdims_obj` is a Python bool. Output shape collapses or keeps the
    reduced axes. Strided per-element f64 round-trip path; SIMD-friendly
    kernels can layer in via LayoutIter once the API stabilises."""
    var src = array_obj.downcast_value_ptr[Array]()
    var op = Int(py=op_obj)
    var keepdims = Bool(py=keepdims_obj)
    var ndim = len(src[].shape)
    # Decode axes.
    var axes = List[Int]()
    for i in range(len(axis_obj)):
        var ax = Int(py=axis_obj[i])
        if ax < 0:
            ax += ndim
        if ax < 0 or ax >= ndim:
            raise Error("reduce axis out of range")
        axes.append(ax)
    # Build result shape.
    var keep_mask = List[Bool]()
    for _ in range(ndim):
        keep_mask.append(True)
    for i in range(len(axes)):
        keep_mask[axes[i]] = False
    var out_shape = List[Int]()
    var keep_axes = List[Int]()
    for d in range(ndim):
        if keep_mask[d]:
            out_shape.append(src[].shape[d])
            keep_axes.append(d)
        elif keepdims:
            out_shape.append(1)
            keep_axes.append(d)
    var result_dtype: Int
    if op == REDUCE_ARGMAX or op == REDUCE_ARGMIN:
        result_dtype = DTYPE_INT64
    elif op == REDUCE_ALL or op == REDUCE_ANY:
        result_dtype = DTYPE_BOOL
    else:
        result_dtype = result_dtype_for_reduction(src[].dtype_code, op)
    var result = make_empty_array(result_dtype, clone_int_list(out_shape))
    # Compute reduce-axis size + strides.
    var reduce_axes = List[Int]()
    for d in range(ndim):
        if not keep_mask[d]:
            reduce_axes.append(d)
    var reduce_size = 1
    for i in range(len(reduce_axes)):
        reduce_size *= src[].shape[reduce_axes[i]]
    if reduce_size == 0:
        # Numpy semantics: sum/prod of empty → identity; min/max raise.
        if op == REDUCE_SUM or op == REDUCE_MEAN:
            for j in range(result.size_value):
                set_logical_from_f64(result, j, 0.0)
            return PythonObject(alloc=result^)
        if op == REDUCE_PROD:
            for j in range(result.size_value):
                set_logical_from_f64(result, j, 1.0)
            return PythonObject(alloc=result^)
        if op == REDUCE_ALL:
            for j in range(result.size_value):
                set_logical_from_f64(result, j, 1.0)
            return PythonObject(alloc=result^)
        if op == REDUCE_ANY:
            for j in range(result.size_value):
                set_logical_from_f64(result, j, 0.0)
            return PythonObject(alloc=result^)
        raise Error("cannot reduce empty axis with this op")
    # Iterate output positions; for each, walk the reduce axes.
    var out_size = result.size_value
    if out_size == 0:
        return PythonObject(alloc=result^)
    # Build coord helper for output index → src physical offset of the
    # first element of the reduced subspace.
    var keep_strides = List[Int]()
    for i in range(len(keep_axes)):
        keep_strides.append(src[].strides[keep_axes[i]])
    var keep_dims = List[Int]()
    for i in range(len(keep_axes)):
        if keepdims:
            keep_dims.append(out_shape[i])
        else:
            keep_dims.append(out_shape[i])
    # If keepdims, the kept axes have size 1 if they were originally
    # reduced, so striding through them must take 0 in that dim.
    var iter_keep_dims = List[Int]()
    var iter_keep_strides = List[Int]()
    for d in range(len(keep_axes)):
        var src_axis = keep_axes[d]
        if keep_mask[src_axis]:
            iter_keep_dims.append(src[].shape[src_axis])
            iter_keep_strides.append(src[].strides[src_axis])
        else:
            iter_keep_dims.append(1)
            iter_keep_strides.append(0)
    # Build per-reduce-axis dim/stride.
    var red_dims = List[Int]()
    var red_strides = List[Int]()
    for i in range(len(reduce_axes)):
        red_dims.append(src[].shape[reduce_axes[i]])
        red_strides.append(src[].strides[reduce_axes[i]])
    # For each output index, decode the kept-axes coordinates, then
    # iterate the reduce subspace.
    var out_strides_logical = List[Int]()
    var stride_logical = 1
    for d in range(len(out_shape) - 1, -1, -1):
        out_strides_logical.append(stride_logical)
        stride_logical *= out_shape[d]
    out_strides_logical.reverse()
    for out_i in range(out_size):
        var src_base = src[].offset_elems
        var rem = out_i
        for d in range(len(out_shape)):
            var dim = out_shape[d]
            var coord = 0
            if dim != 0:
                coord = rem // out_strides_logical[d]
                rem = rem % out_strides_logical[d]
            src_base += coord * iter_keep_strides[d]
        # Walk reduce subspace.
        var first_phys = src_base
        for ai in range(len(reduce_axes)):
            _ = ai
        # Initialise accumulator.
        var acc: Float64
        var best_idx: Int = 0
        if op == REDUCE_SUM or op == REDUCE_MEAN:
            acc = 0.0
        elif op == REDUCE_PROD:
            acc = 1.0
        elif op == REDUCE_ALL:
            acc = 1.0
        elif op == REDUCE_ANY:
            acc = 0.0
        else:
            acc = get_physical_as_f64(src[], src_base)
            best_idx = 0
        # Walk reduce coords.
        var rcoords = List[Int]()
        for _ in range(len(reduce_axes)):
            rcoords.append(0)
        var k = 0
        while True:
            var phys = src_base
            for j in range(len(reduce_axes)):
                phys += rcoords[j] * red_strides[j]
            var value = get_physical_as_f64(src[], phys)
            if op == REDUCE_SUM or op == REDUCE_MEAN:
                acc += value
            elif op == REDUCE_PROD:
                acc *= value
            elif op == REDUCE_MIN:
                if k == 0 or value < acc:
                    acc = value
            elif op == REDUCE_MAX:
                if k == 0 or value > acc:
                    acc = value
            elif op == REDUCE_ALL:
                if value == 0.0:
                    acc = 0.0
                    break
            elif op == REDUCE_ANY:
                if value != 0.0:
                    acc = 1.0
                    break
            elif op == REDUCE_ARGMAX:
                if k == 0 or value > acc:
                    acc = value
                    best_idx = k
            elif op == REDUCE_ARGMIN:
                if k == 0 or value < acc:
                    acc = value
                    best_idx = k
            else:
                raise Error("unknown reduction op")
            k += 1
            # Advance rcoords innermost first.
            var idx = len(reduce_axes) - 1
            var done = False
            while idx >= 0:
                rcoords[idx] += 1
                if rcoords[idx] < red_dims[idx]:
                    break
                rcoords[idx] = 0
                idx -= 1
                if idx < 0:
                    done = True
            if done:
                break
        if op == REDUCE_MEAN:
            acc = acc / Float64(reduce_size)
        if op == REDUCE_ARGMAX or op == REDUCE_ARGMIN:
            set_logical_from_i64(result, out_i, Int64(best_idx))
        else:
            set_logical_from_f64(result, out_i, acc)
        _ = first_phys  # silence unused warning
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
    var dtype_code = result_dtype_for_binary(lhs[].dtype_code, rhs[].dtype_code, OP_MUL)
    var result = make_empty_array(dtype_code, out_shape^)
    var is_complex = dtype_code == DTYPE_COMPLEX64 or dtype_code == DTYPE_COMPLEX128
    if lhs_ndim == 1 and rhs_ndim == 1:
        if is_complex:
            var lhs_re_total = 0.0
            var lhs_im_total = 0.0
            for k in range(k_lhs):
                var l_re = _complex_real(lhs[], k)
                var l_im = _complex_imag(lhs[], k)
                var r_re = _complex_real(rhs[], k)
                var r_im = _complex_imag(rhs[], k)
                lhs_re_total += l_re * r_re - l_im * r_im
                lhs_im_total += l_re * r_im + l_im * r_re
            _complex_store(result, 0, lhs_re_total, lhs_im_total)
        else:
            var total = 0.0
            for k in range(k_lhs):
                total += get_logical_as_f64(lhs[], k) * get_logical_as_f64(rhs[], k)
            set_logical_from_f64(result, 0, total)
        return PythonObject(alloc=result^)
    if maybe_matmul_contiguous(lhs[], rhs[], result, m, n, k_lhs):
        return PythonObject(alloc=result^)
    if is_complex:
        for i in range(m):
            for j in range(n):
                var re_total = 0.0
                var im_total = 0.0
                for k in range(k_lhs):
                    var lhs_index = k
                    if lhs_ndim == 2:
                        lhs_index = i * k_lhs + k
                    var rhs_index = k
                    if rhs_ndim == 2:
                        rhs_index = k * n + j
                    var l_re = _complex_real(lhs[], lhs_index)
                    var l_im = _complex_imag(lhs[], lhs_index)
                    var r_re = _complex_real(rhs[], rhs_index)
                    var r_im = _complex_imag(rhs[], rhs_index)
                    re_total += l_re * r_re - l_im * r_im
                    im_total += l_re * r_im + l_im * r_re
                var out_index = j
                if lhs_ndim == 2:
                    out_index = i * n + j
                _complex_store(result, out_index, re_total, im_total)
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


def _complex_real(arr: Array, logical: Int) raises -> Float64:
    """Helper: read real part of a complex array at logical index."""
    var phys = physical_offset(arr, logical)
    if arr.dtype_code == DTYPE_COMPLEX64:
        return Float64(get_physical_c64_real(arr, phys))
    if arr.dtype_code == DTYPE_COMPLEX128:
        return get_physical_c128_real(arr, phys)
    return get_logical_as_f64(arr, logical)


def _complex_imag(arr: Array, logical: Int) raises -> Float64:
    """Helper: read imag part of a complex array at logical index."""
    var phys = physical_offset(arr, logical)
    if arr.dtype_code == DTYPE_COMPLEX64:
        return Float64(get_physical_c64_imag(arr, phys))
    if arr.dtype_code == DTYPE_COMPLEX128:
        return get_physical_c128_imag(arr, phys)
    return 0.0


def _complex_store(mut arr: Array, logical: Int, real: Float64, imag: Float64) raises:
    """Helper: write real+imag to a complex array at logical index."""
    var phys = physical_offset(arr, logical)
    if arr.dtype_code == DTYPE_COMPLEX64:
        set_physical_c64(arr, phys, Float32(real), Float32(imag))
    elif arr.dtype_code == DTYPE_COMPLEX128:
        set_physical_c128(arr, phys, real, imag)
    else:
        set_logical_from_f64(arr, logical, real)


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


# ============================================================
# Phase-6d LAPACK-backed decomposition entry points.
#
# Each ops returns a Python list (or single Array) so the Python
# wrappers in `monpy.linalg` can pull individual outputs by index. The
# heavy lifting (allocation + LAPACK call + col→row transpose) happens
# entirely in mojo; the python side only pays the dtype-resolve and
# wrap costs.
# ============================================================


def qr_ops(array_obj: PythonObject, mode_obj: PythonObject) raises -> PythonObject:
    from std.python import Python

    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 2:
        raise Error("linalg.qr: input must be rank-2")
    var m = src[].shape[0]
    var n = src[].shape[1]
    var k = m if m < n else n
    var dtype_code = result_dtype_for_linalg(src[].dtype_code)
    var mode = Int(py=mode_obj)  # 0=reduced, 1=complete, 2=r, 3=raw
    if mode == 0:
        # Q is (m, k), R is (k, n)
        var q_shape = List[Int]()
        q_shape.append(m)
        q_shape.append(k)
        var r_shape = List[Int]()
        r_shape.append(k)
        r_shape.append(n)
        var q = make_empty_array(dtype_code, q_shape^)
        var r = make_empty_array(dtype_code, r_shape^)
        if dtype_code == DTYPE_FLOAT32:
            lapack_qr_reduced_f32_into(src[], q, r)
        else:
            lapack_qr_reduced_f64_into(src[], q, r)
        var out = Python.evaluate("[]")
        _ = out.append(PythonObject(alloc=q^))
        _ = out.append(PythonObject(alloc=r^))
        return out
    if mode == 1:
        # mode='complete': Q is (m, m), R is (m, n). Numpy uses sgeqrf
        # then sorgqr to build full Q (m × m) — pad with extra columns
        # via `sorgqr(m, m, k, ...)` once the first k columns are set.
        # For simplicity in v1: error if m > n (would need extra work).
        if m < n:
            raise Error("linalg.qr: mode='complete' for m < n requires extra work — use mode='reduced'")
        var q_shape = List[Int]()
        q_shape.append(m)
        q_shape.append(m)
        var r_shape = List[Int]()
        r_shape.append(m)
        r_shape.append(n)
        var q = make_empty_array(dtype_code, q_shape^)
        var r = make_empty_array(dtype_code, r_shape^)
        if dtype_code == DTYPE_FLOAT32:
            lapack_qr_reduced_f32_into(src[], q, r)
        else:
            lapack_qr_reduced_f64_into(src[], q, r)
        var out = Python.evaluate("[]")
        _ = out.append(PythonObject(alloc=q^))
        _ = out.append(PythonObject(alloc=r^))
        return out
    if mode == 2:
        # mode='r': just R, shape (k, n)
        var r_shape = List[Int]()
        r_shape.append(k)
        r_shape.append(n)
        var r = make_empty_array(dtype_code, r_shape^)
        if dtype_code == DTYPE_FLOAT32:
            lapack_qr_r_only_f32_into(src[], r)
        else:
            lapack_qr_r_only_f64_into(src[], r)
        return PythonObject(alloc=r^)
    raise Error("linalg.qr: unsupported mode")


def cholesky_ops(array_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 2 or src[].shape[0] != src[].shape[1]:
        raise Error("linalg.cholesky: input must be square rank-2")
    var n = src[].shape[0]
    var dtype_code = result_dtype_for_linalg(src[].dtype_code)
    var shape = List[Int]()
    shape.append(n)
    shape.append(n)
    var result = make_empty_array(dtype_code, shape^)
    if dtype_code == DTYPE_FLOAT32:
        lapack_cholesky_f32_into(src[], result)
    else:
        lapack_cholesky_f64_into(src[], result)
    return PythonObject(alloc=result^)


def eigh_ops(array_obj: PythonObject, compute_eigenvectors_obj: PythonObject) raises -> PythonObject:
    from std.python import Python

    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 2 or src[].shape[0] != src[].shape[1]:
        raise Error("linalg.eigh: input must be square rank-2")
    var n = src[].shape[0]
    var compute_v = Bool(py=compute_eigenvectors_obj)
    var dtype_code = result_dtype_for_linalg(src[].dtype_code)
    var w_shape = List[Int]()
    w_shape.append(n)
    var w = make_empty_array(dtype_code, w_shape^)
    var v_shape = List[Int]()
    if compute_v:
        v_shape.append(n)
        v_shape.append(n)
    else:
        # Allocate a 0-element placeholder — caller ignores it for eigvalsh.
        v_shape.append(0)
    var v = make_empty_array(dtype_code, v_shape^)
    if dtype_code == DTYPE_FLOAT32:
        lapack_eigh_f32_into(src[], w, v, compute_v)
    else:
        lapack_eigh_f64_into(src[], w, v, compute_v)
    var out = Python.evaluate("[]")
    _ = out.append(PythonObject(alloc=w^))
    _ = out.append(PythonObject(alloc=v^))
    return out


def eig_ops(array_obj: PythonObject, compute_eigenvectors_obj: PythonObject) raises -> PythonObject:
    from std.python import Python

    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 2 or src[].shape[0] != src[].shape[1]:
        raise Error("linalg.eig: input must be square rank-2")
    var n = src[].shape[0]
    var compute_v = Bool(py=compute_eigenvectors_obj)
    var dtype_code = result_dtype_for_linalg(src[].dtype_code)
    var w_shape = List[Int]()
    w_shape.append(n)
    var wr = make_empty_array(dtype_code, w_shape^)
    var wi_shape = List[Int]()
    wi_shape.append(n)
    var wi = make_empty_array(dtype_code, wi_shape^)
    var v_shape = List[Int]()
    if compute_v:
        v_shape.append(n)
        v_shape.append(n)
    else:
        v_shape.append(0)
    var v = make_empty_array(dtype_code, v_shape^)
    var all_real: Bool
    if dtype_code == DTYPE_FLOAT32:
        all_real = lapack_eig_f32_real_into(src[], wr, wi, v, compute_v)
    else:
        all_real = lapack_eig_f64_real_into(src[], wr, wi, v, compute_v)
    var out = Python.evaluate("[]")
    _ = out.append(PythonObject(alloc=wr^))
    _ = out.append(PythonObject(alloc=wi^))
    _ = out.append(PythonObject(alloc=v^))
    _ = out.append(PythonObject(all_real))
    return out


def svd_ops(
    array_obj: PythonObject,
    full_matrices_obj: PythonObject,
    compute_uv_obj: PythonObject,
) raises -> PythonObject:
    from std.python import Python

    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 2:
        raise Error("linalg.svd: input must be rank-2")
    var m = src[].shape[0]
    var n = src[].shape[1]
    var k = m if m < n else n
    var full_matrices = Bool(py=full_matrices_obj)
    var compute_uv = Bool(py=compute_uv_obj)
    var dtype_code = result_dtype_for_linalg(src[].dtype_code)
    var s_shape = List[Int]()
    s_shape.append(k)
    var s = make_empty_array(dtype_code, s_shape^)
    var u_shape = List[Int]()
    var vt_shape = List[Int]()
    if compute_uv:
        u_shape.append(m)
        if full_matrices:
            u_shape.append(m)
            vt_shape.append(n)
        else:
            u_shape.append(k)
            vt_shape.append(k)
        vt_shape.append(n)
    else:
        u_shape.append(0)
        vt_shape.append(0)
    var u = make_empty_array(dtype_code, u_shape^)
    var vt = make_empty_array(dtype_code, vt_shape^)
    if dtype_code == DTYPE_FLOAT32:
        lapack_svd_f32_into(src[], u, s, vt, full_matrices, compute_uv)
    else:
        lapack_svd_f64_into(src[], u, s, vt, full_matrices, compute_uv)
    var out = Python.evaluate("[]")
    _ = out.append(PythonObject(alloc=u^))
    _ = out.append(PythonObject(alloc=s^))
    _ = out.append(PythonObject(alloc=vt^))
    return out


def concatenate_ops(
    arrays_obj: PythonObject, axis_obj: PythonObject, dtype_code_obj: PythonObject
) raises -> PythonObject:
    # Native concatenation. Replaces the python flat-write path which was
    # ~58000× slower than numpy at N=256.
    from std.memory import memcpy as _memcpy

    var n_arrays = Int(py=len(arrays_obj))
    if n_arrays == 0:
        raise Error("concatenate: need at least one array")
    var axis = Int(py=axis_obj)
    var dtype_code = Int(py=dtype_code_obj)
    # Pull rank from first input.
    var first = arrays_obj[0].downcast_value_ptr[Array]()
    var ndim = len(first[].shape)
    if axis < 0:
        axis = axis + ndim
    if axis < 0 or axis >= ndim:
        raise Error("concatenate: axis out of range")
    # Build out_shape; check shape consistency across arrays.
    var out_shape = List[Int]()
    for d in range(ndim):
        out_shape.append(first[].shape[d])
    out_shape[axis] = 0
    var all_c_contig = True
    for i in range(n_arrays):
        var a = arrays_obj[i].downcast_value_ptr[Array]()
        if len(a[].shape) != ndim:
            raise Error("concatenate: arrays must have same ndim")
        for d in range(ndim):
            if d == axis:
                out_shape[axis] = out_shape[axis] + a[].shape[d]
            elif a[].shape[d] != first[].shape[d]:
                raise Error("concatenate: shape mismatch")
        if a[].dtype_code != dtype_code:
            raise Error("concatenate: dtype mismatch (caller must pre-cast)")
        if not is_c_contiguous(a[]):
            all_c_contig = False
    var out_shape_clone = List[Int]()
    for d in range(ndim):
        out_shape_clone.append(out_shape[d])
    var result = make_empty_array(dtype_code, out_shape_clone^)
    var outer_size = 1
    for d in range(axis):
        outer_size *= out_shape[d]
    var inner_size = 1
    for d in range(axis + 1, ndim):
        inner_size *= out_shape[d]
    var item_bytes = item_size(dtype_code)
    var out_axis_size = out_shape[axis]
    var out_row_size = out_axis_size * inner_size
    var axis_offset = 0
    for i in range(n_arrays):
        var a = arrays_obj[i].downcast_value_ptr[Array]()
        var a_axis = a[].shape[axis]
        var a_slab_size = a_axis * inner_size
        if all_c_contig:
            var src_byte_offset = a[].offset_elems * item_bytes
            for outer in range(outer_size):
                var src_off_bytes = src_byte_offset + outer * a_slab_size * item_bytes
                var dst_off_bytes = (outer * out_row_size + axis_offset * inner_size) * item_bytes
                _memcpy(
                    dest=result.data + dst_off_bytes,
                    src=a[].data + src_off_bytes,
                    count=a_slab_size * item_bytes,
                )
        else:
            for outer in range(outer_size):
                var src_base = outer * a_slab_size
                var dst_base = outer * out_row_size + axis_offset * inner_size
                for k in range(a_slab_size):
                    set_logical_from_f64(result, dst_base + k, get_logical_as_f64(a[], src_base + k))
        axis_offset = axis_offset + a_axis
    return PythonObject(alloc=result^)


def tril_ops(array_obj: PythonObject, k_obj: PythonObject) raises -> PythonObject:
    # Native lower-triangular: copy values where col <= row + k, zero otherwise.
    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) < 2:
        raise Error("tril: requires rank >= 2 input")
    var k = Int(py=k_obj)
    var ndim = len(src[].shape)
    var rows = src[].shape[ndim - 2]
    var cols = src[].shape[ndim - 1]
    var out_shape = List[Int]()
    for d in range(ndim):
        out_shape.append(src[].shape[d])
    var result = make_empty_array(src[].dtype_code, out_shape^)
    var batch = 1
    for d in range(ndim - 2):
        batch *= src[].shape[d]
    if src[].dtype_code == DTYPE_FLOAT32 and is_c_contiguous(src[]):
        var src_ptr = contiguous_f32_ptr(src[])
        var out_ptr = contiguous_f32_ptr(result)
        for b in range(batch):
            for r in range(rows):
                for c in range(cols):
                    var idx = b * rows * cols + r * cols + c
                    if c <= r + k:
                        out_ptr[idx] = src_ptr[idx]
                    else:
                        out_ptr[idx] = Float32(0.0)
        return PythonObject(alloc=result^)
    if src[].dtype_code == DTYPE_FLOAT64 and is_c_contiguous(src[]):
        var src_ptr = contiguous_f64_ptr(src[])
        var out_ptr = contiguous_f64_ptr(result)
        for b in range(batch):
            for r in range(rows):
                for c in range(cols):
                    var idx = b * rows * cols + r * cols + c
                    if c <= r + k:
                        out_ptr[idx] = src_ptr[idx]
                    else:
                        out_ptr[idx] = 0.0
        return PythonObject(alloc=result^)
    for b in range(batch):
        for r in range(rows):
            for c in range(cols):
                var idx = b * rows * cols + r * cols + c
                if c <= r + k:
                    set_logical_from_f64(result, idx, get_logical_as_f64(src[], idx))
                else:
                    set_logical_from_f64(result, idx, 0.0)
    return PythonObject(alloc=result^)


def triu_ops(array_obj: PythonObject, k_obj: PythonObject) raises -> PythonObject:
    # Native upper-triangular: copy values where col >= row + k, zero otherwise.
    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) < 2:
        raise Error("triu: requires rank >= 2 input")
    var k = Int(py=k_obj)
    var ndim = len(src[].shape)
    var rows = src[].shape[ndim - 2]
    var cols = src[].shape[ndim - 1]
    var out_shape = List[Int]()
    for d in range(ndim):
        out_shape.append(src[].shape[d])
    var result = make_empty_array(src[].dtype_code, out_shape^)
    var batch = 1
    for d in range(ndim - 2):
        batch *= src[].shape[d]
    if src[].dtype_code == DTYPE_FLOAT32 and is_c_contiguous(src[]):
        var src_ptr = contiguous_f32_ptr(src[])
        var out_ptr = contiguous_f32_ptr(result)
        for b in range(batch):
            for r in range(rows):
                for c in range(cols):
                    var idx = b * rows * cols + r * cols + c
                    if c >= r + k:
                        out_ptr[idx] = src_ptr[idx]
                    else:
                        out_ptr[idx] = Float32(0.0)
        return PythonObject(alloc=result^)
    if src[].dtype_code == DTYPE_FLOAT64 and is_c_contiguous(src[]):
        var src_ptr = contiguous_f64_ptr(src[])
        var out_ptr = contiguous_f64_ptr(result)
        for b in range(batch):
            for r in range(rows):
                for c in range(cols):
                    var idx = b * rows * cols + r * cols + c
                    if c >= r + k:
                        out_ptr[idx] = src_ptr[idx]
                    else:
                        out_ptr[idx] = 0.0
        return PythonObject(alloc=result^)
    for b in range(batch):
        for r in range(rows):
            for c in range(cols):
                var idx = b * rows * cols + r * cols + c
                if c >= r + k:
                    set_logical_from_f64(result, idx, get_logical_as_f64(src[], idx))
                else:
                    set_logical_from_f64(result, idx, 0.0)
    return PythonObject(alloc=result^)


def eye_ops(
    n_obj: PythonObject,
    m_obj: PythonObject,
    k_obj: PythonObject,
    dtype_code_obj: PythonObject,
) raises -> PythonObject:
    # Native diagonal-fill identity. Replaces python loop that was
    # ~100-250× slower than numpy at N=64.
    var n = Int(py=n_obj)
    var m = Int(py=m_obj)
    var k = Int(py=k_obj)
    var dtype_code = Int(py=dtype_code_obj)
    var shape = List[Int]()
    shape.append(n)
    shape.append(m)
    var result = make_empty_array(dtype_code, shape^)
    if dtype_code == DTYPE_FLOAT32:
        var out_ptr = contiguous_f32_ptr(result)
        for i in range(n * m):
            out_ptr[i] = Float32(0.0)
        var start_i = 0 if k >= 0 else -k
        var max_diag = n if (m - k) > n else (m - k)
        var end_i = max_diag if max_diag > 0 else 0
        if end_i > n:
            end_i = n
        for i in range(start_i, end_i):
            var col = i + k
            if col >= 0 and col < m:
                out_ptr[i * m + col] = Float32(1.0)
        return PythonObject(alloc=result^)
    if dtype_code == DTYPE_FLOAT64:
        var out_ptr = contiguous_f64_ptr(result)
        for i in range(n * m):
            out_ptr[i] = 0.0
        var start_i = 0 if k >= 0 else -k
        var max_diag = n if (m - k) > n else (m - k)
        var end_i = max_diag if max_diag > 0 else 0
        if end_i > n:
            end_i = n
        for i in range(start_i, end_i):
            var col = i + k
            if col >= 0 and col < m:
                out_ptr[i * m + col] = 1.0
        return PythonObject(alloc=result^)
    # Zero out first.
    for i in range(n * m):
        set_logical_from_f64(result, i, 0.0)
    # Walk diagonal.
    var start_i = 0 if k >= 0 else -k
    var max_diag = n if (m - k) > n else (m - k)
    var end_i = max_diag if max_diag > 0 else 0
    if end_i > n:
        end_i = n
    for i in range(start_i, end_i):
        var col = i + k
        if col >= 0 and col < m:
            set_logical_from_f64(result, i * m + col, 1.0)
    return PythonObject(alloc=result^)


def tri_ops(
    n_obj: PythonObject,
    m_obj: PythonObject,
    k_obj: PythonObject,
    dtype_code_obj: PythonObject,
) raises -> PythonObject:
    var n = Int(py=n_obj)
    var m = Int(py=m_obj)
    var k = Int(py=k_obj)
    var dtype_code = Int(py=dtype_code_obj)
    var shape = List[Int]()
    shape.append(n)
    shape.append(m)
    var result = make_empty_array(dtype_code, shape^)
    if dtype_code == DTYPE_FLOAT32:
        var out_ptr = contiguous_f32_ptr(result)
        for r in range(n):
            var row_limit = r + k + 1
            if row_limit < 0:
                row_limit = 0
            if row_limit > m:
                row_limit = m
            for c in range(m):
                if c < row_limit:
                    out_ptr[r * m + c] = Float32(1.0)
                else:
                    out_ptr[r * m + c] = Float32(0.0)
        return PythonObject(alloc=result^)
    if dtype_code == DTYPE_FLOAT64:
        var out_ptr = contiguous_f64_ptr(result)
        for r in range(n):
            var row_limit = r + k + 1
            if row_limit < 0:
                row_limit = 0
            if row_limit > m:
                row_limit = m
            for c in range(m):
                if c < row_limit:
                    out_ptr[r * m + c] = 1.0
                else:
                    out_ptr[r * m + c] = 0.0
        return PythonObject(alloc=result^)
    for r in range(n):
        var row_limit = r + k + 1
        if row_limit < 0:
            row_limit = 0
        if row_limit > m:
            row_limit = m
        for c in range(m):
            var value = 0.0
            if c < row_limit:
                value = 1.0
            set_logical_from_f64(result, r * m + c, value)
    return PythonObject(alloc=result^)


def pad_constant_ops(
    array_obj: PythonObject,
    pad_before_obj: PythonObject,
    pad_after_obj: PythonObject,
    constant_value_obj: PythonObject,
) raises -> PythonObject:
    # Native constant-mode pad. pad_before/pad_after are tuples of length ndim.
    var src = array_obj.downcast_value_ptr[Array]()
    var ndim = len(src[].shape)
    var pad_before = List[Int]()
    var pad_after = List[Int]()
    for d in range(ndim):
        pad_before.append(Int(py=pad_before_obj[d]))
        pad_after.append(Int(py=pad_after_obj[d]))
    var out_shape = List[Int]()
    for d in range(ndim):
        out_shape.append(src[].shape[d] + pad_before[d] + pad_after[d])
    var out_shape_clone = List[Int]()
    for d in range(ndim):
        out_shape_clone.append(out_shape[d])
    var result = make_empty_array(src[].dtype_code, out_shape_clone^)
    var constant_f64 = Float64(py=constant_value_obj)
    # Fill with constant first.
    var out_size = 1
    for d in range(ndim):
        out_size *= out_shape[d]
    for i in range(out_size):
        set_logical_from_f64(result, i, constant_f64)
    # Compute strides for source and result.
    var src_strides = List[Int]()
    var out_strides = List[Int]()
    for _ in range(ndim):
        src_strides.append(0)
        out_strides.append(0)
    var s = 1
    var o = 1
    for d in range(ndim - 1, -1, -1):
        src_strides[d] = s
        s = s * src[].shape[d]
        out_strides[d] = o
        o = o * out_shape[d]
    # Copy source into result with offset.
    var src_size = 1
    for d in range(ndim):
        src_size *= src[].shape[d]
    for i in range(src_size):
        # Decode source coordinate.
        var remainder = i
        var dst_idx = 0
        for d in range(ndim):
            var coord = remainder // src_strides[d]
            remainder = remainder % src_strides[d]
            dst_idx += (coord + pad_before[d]) * out_strides[d]
        set_logical_from_f64(result, dst_idx, get_logical_as_f64(src[], i))
    return PythonObject(alloc=result^)


def lstsq_ops(a_obj: PythonObject, b_obj: PythonObject, rcond_obj: PythonObject) raises -> PythonObject:
    from std.python import Python

    var a = a_obj.downcast_value_ptr[Array]()
    var b = b_obj.downcast_value_ptr[Array]()
    if len(a[].shape) != 2:
        raise Error("linalg.lstsq: a must be rank-2")
    var m = a[].shape[0]
    var n = a[].shape[1]
    var dtype_code = result_dtype_for_linalg_binary(a[].dtype_code, b[].dtype_code)
    var k = m if m < n else n
    var b_is_vec = len(b[].shape) == 1
    var nrhs = 1
    if not b_is_vec:
        if len(b[].shape) != 2:
            raise Error("linalg.lstsq: b must be rank-1 or rank-2")
        nrhs = b[].shape[1]
    if (b_is_vec and b[].shape[0] != m) or (not b_is_vec and b[].shape[0] != m):
        raise Error("linalg.lstsq: shape mismatch on first axis of b")
    var x_shape = List[Int]()
    x_shape.append(n)
    if not b_is_vec:
        x_shape.append(nrhs)
    var x = make_empty_array(dtype_code, x_shape^)
    var s_shape = List[Int]()
    s_shape.append(k)
    var s = make_empty_array(dtype_code, s_shape^)
    var rank_buf = Int(0)
    var rank_ptr = rebind[UnsafePointer[Int, MutExternalOrigin]](UnsafePointer(to=rank_buf))
    if dtype_code == DTYPE_FLOAT32:
        var rcond_f32 = Float32(Float64(py=rcond_obj))
        lapack_lstsq_f32_into(a[], b[], x, s, rcond_f32, rank_ptr)
    else:
        var rcond_f64 = Float64(py=rcond_obj)
        lapack_lstsq_f64_into(a[], b[], x, s, rcond_f64, rank_ptr)
    var out = Python.evaluate("[]")
    _ = out.append(PythonObject(alloc=x^))
    _ = out.append(PythonObject(alloc=s^))
    _ = out.append(PythonObject(rank_buf))
    return out


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
    var src_layout = as_broadcast_layout(src[], dst[].shape)
    var dst_layout = as_layout(dst[])
    var src_item = item_size(src[].dtype_code)
    var dst_item = item_size(dst[].dtype_code)
    var operand_layouts = List[Layout]()
    operand_layouts.append(src_layout^)
    operand_layouts.append(dst_layout^)
    var item_sizes = List[Int]()
    item_sizes.append(src_item)
    item_sizes.append(dst_item)
    var base_offsets = List[Int]()
    base_offsets.append(src[].offset_elems * src_item)
    base_offsets.append(dst[].offset_elems * dst_item)
    var iter = MultiLayoutIter(dst[].shape, operand_layouts^, item_sizes^, base_offsets^)
    while iter.has_next():
        set_physical_from_f64(dst[], iter.element_index(1), get_physical_as_f64(src[], iter.element_index(0)))
        iter.step()
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
