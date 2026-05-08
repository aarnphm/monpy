from std.collections import List
from std.math import cos, exp, isinf, isnan, log, nan, sin
from std.python import PythonObject

from accelerate import libm_pow_f64
from domain import (
    ArrayDType,
    BackendKind,
    BinaryOp,
    CastingRule,
    CompareOp,
    LogicalOp,
    PredicateOp,
    ReduceOp,
    UnaryOp,
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

from ._complex_helpers import _complex_imag, _complex_real, _complex_store
from .creation_ops import (
    arange_ops,
    copy_from_external_ops,
    empty_ops,
    from_external_ops,
    from_flat_ops,
    full_like_ops,
    full_ops,
    indices_ops,
    linspace_ops,
    logspace_ops,
)
from .elementwise_ops import compare_ops, logical_ops, predicate_ops, unary_preserve_ops
from .reduction_ops import reduce_axis_ops, reduce_ops
from .shape_ops import (
    concatenate_ops,
    pad_constant_ops,
    stack_axis0_ops,
    tril_ops,
    triu_ops,
)
from .dtype_ops import (
    astype_ops,
    dtype_alignment_py_ops,
    dtype_can_cast_py_ops,
    dtype_item_size_py_ops,
    dtype_kind_code_py_ops,
    dtype_promote_types_py_ops,
    result_dtype_for_binary_py_ops,
    result_dtype_for_reduction_py_ops,
    result_dtype_for_unary_py_ops,
)
from .linalg_ops import (
    cholesky_ops,
    det_ops,
    eig_ops,
    eigh_ops,
    inv_ops,
    lstsq_ops,
    matmul_ops,
    pinv_ops,
    qr_ops,
    solve_ops,
    svd_ops,
)


# Python-callable entrypoints.
# Includes Storage -> Shape -> Backend FFI imports from submodules.


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


def ravel_ops(array_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var shape = List[Int]()
    shape.append(src[].size_value)
    var strides = List[Int]()
    strides.append(1)
    if is_c_contiguous(src[]):
        var view = make_view_array(src[], shape^, strides^, src[].size_value, src[].offset_elems)
        return PythonObject(alloc=view^)
    var copied = copy_c_contiguous(src[])
    var view = make_view_array(copied, shape^, strides^, copied.size_value, copied.offset_elems)
    return PythonObject(alloc=view^)


def flatten_ops(array_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var copied = copy_c_contiguous(src[])
    var shape = List[Int]()
    shape.append(copied.size_value)
    var strides = List[Int]()
    strides.append(1)
    var view = make_view_array(copied, shape^, strides^, copied.size_value, copied.offset_elems)
    return PythonObject(alloc=view^)


def squeeze_all_ops(array_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var shape = List[Int]()
    var strides = List[Int]()
    for axis in range(len(src[].shape)):
        if src[].shape[axis] != 1:
            shape.append(src[].shape[axis])
            strides.append(src[].strides[axis])
    var result = make_view_array(src[], shape^, strides^, src[].size_value, src[].offset_elems)
    return PythonObject(alloc=result^)


def squeeze_axis_ops(array_obj: PythonObject, axis_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var ndim = len(src[].shape)
    var axis = Int(py=axis_obj)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise Error("squeeze: axis out of range")
    if src[].shape[axis] != 1:
        raise Error("squeeze: cannot select an axis with size != 1")
    var shape = List[Int]()
    var strides = List[Int]()
    for src_axis in range(ndim):
        if src_axis != axis:
            shape.append(src[].shape[src_axis])
            strides.append(src[].strides[src_axis])
    var result = make_view_array(src[], shape^, strides^, src[].size_value, src[].offset_elems)
    return PythonObject(alloc=result^)


def squeeze_axes_ops(array_obj: PythonObject, axes_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var ndim = len(src[].shape)
    var axes = int_list_from_py(axes_obj)
    var drop = List[Bool]()
    for _ in range(ndim):
        drop.append(False)
    for i in range(len(axes)):
        var axis = axes[i]
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise Error("squeeze: axis out of range")
        if src[].shape[axis] != 1:
            raise Error("squeeze: cannot select an axis with size != 1")
        if drop[axis]:
            raise Error("squeeze: repeated axis")
        drop[axis] = True
    var shape = List[Int]()
    var strides = List[Int]()
    for axis in range(ndim):
        if not drop[axis]:
            shape.append(src[].shape[axis])
            strides.append(src[].strides[axis])
    var result = make_view_array(src[], shape^, strides^, src[].size_value, src[].offset_elems)
    return PythonObject(alloc=result^)


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


def swapaxes_ops(array_obj: PythonObject, axis1_obj: PythonObject, axis2_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var ndim = len(src[].shape)
    var axis1 = Int(py=axis1_obj)
    var axis2 = Int(py=axis2_obj)
    if axis1 < 0:
        axis1 += ndim
    if axis2 < 0:
        axis2 += ndim
    if axis1 < 0 or axis1 >= ndim or axis2 < 0 or axis2 >= ndim:
        raise Error("swapaxes: axis out of range")
    var shape = clone_int_list(src[].shape)
    var strides = clone_int_list(src[].strides)
    var tmp_shape = shape[axis1]
    shape[axis1] = shape[axis2]
    shape[axis2] = tmp_shape
    var tmp_stride = strides[axis1]
    strides[axis1] = strides[axis2]
    strides[axis2] = tmp_stride
    var result = make_view_array(src[], shape^, strides^, src[].size_value, src[].offset_elems)
    return PythonObject(alloc=result^)


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
        dtype_code = result_dtype_for_reduction(src[].dtype_code, ReduceOp.SUM.value)
    var result = make_empty_array(dtype_code, shape^)
    if src[].dtype_code == ArrayDType.INT64.value:
        var acc = Int64(0)
        for i in range(diag_len):
            acc += get_physical_i64(src[], diag_offset + i * diag_stride)
        set_logical_from_i64(result, 0, acc)
    elif src[].dtype_code == ArrayDType.BOOL.value:
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
    var op = UnaryOp.from_int(Int(py=op_obj)).value
    var shape = clone_int_list(src[].shape)
    var result = make_empty_array(result_dtype_for_unary(src[].dtype_code), shape^)
    # Complex transcendentals: sin/cos/exp/log/sqrt etc. via Euler identities.
    # Output dtype = input dtype (preserved by result_dtype_for_unary).
    if src[].dtype_code == ArrayDType.COMPLEX64.value or src[].dtype_code == ArrayDType.COMPLEX128.value:
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

    if op == UnaryOp.EXP.value:
        # exp(a+bi) = exp(a) * (cos(b) + i sin(b))
        var ea = _exp(re)
        return (ea * _cos(im), ea * _sin(im))
    if op == UnaryOp.LOG.value:
        # log(z) = log|z| + i arg(z)
        var modulus = _sqrt(re * re + im * im)
        return (_log(modulus), _atan2(im, re))
    if op == UnaryOp.SIN.value:
        # sin(a+bi) = sin(a)cosh(b) + i cos(a)sinh(b)
        return (_sin(re) * _cosh(im), _cos(re) * _sinh(im))
    if op == UnaryOp.COS.value:
        # cos(a+bi) = cos(a)cosh(b) - i sin(a)sinh(b)
        return (_cos(re) * _cosh(im), -_sin(re) * _sinh(im))
    if op == UnaryOp.SINH.value:
        # sinh(a+bi) = sinh(a)cos(b) + i cosh(a)sin(b)
        return (_sinh(re) * _cos(im), _cosh(re) * _sin(im))
    if op == UnaryOp.COSH.value:
        # cosh(a+bi) = cosh(a)cos(b) + i sinh(a)sin(b)
        return (_cosh(re) * _cos(im), _sinh(re) * _sin(im))
    if op == UnaryOp.TANH.value:
        # tanh(z) = sinh(z) / cosh(z) — use Euler-form identities and divide.
        var s_re = _sinh(re) * _cos(im)
        var s_im = _cosh(re) * _sin(im)
        var c_re = _cosh(re) * _cos(im)
        var c_im = _sinh(re) * _sin(im)
        var denom = c_re * c_re + c_im * c_im
        return ((s_re * c_re + s_im * c_im) / denom, (s_im * c_re - s_re * c_im) / denom)
    if op == UnaryOp.TAN.value:
        # tan(z) = sin(z) / cos(z)
        var s_re = _sin(re) * _cosh(im)
        var s_im = _cos(re) * _sinh(im)
        var c_re = _cos(re) * _cosh(im)
        var c_im = -_sin(re) * _sinh(im)
        var denom = c_re * c_re + c_im * c_im
        return ((s_re * c_re + s_im * c_im) / denom, (s_im * c_re - s_re * c_im) / denom)
    if op == UnaryOp.SQRT.value:
        # sqrt(z): principal branch. z = r * exp(i*theta), sqrt(z) = sqrt(r) * exp(i*theta/2).
        var modulus = _sqrt(re * re + im * im)
        var arg = _atan2(im, re)
        var s = _sqrt(modulus)
        return (s * _cos(arg / 2.0), s * _sin(arg / 2.0))
    if op == UnaryOp.LOG2.value:
        # log2(z) = log(z) / log(2)
        var modulus = _sqrt(re * re + im * im)
        var ln2 = 0.6931471805599453
        return (_log(modulus) / ln2, _atan2(im, re) / ln2)
    if op == UnaryOp.LOG10.value:
        var modulus = _sqrt(re * re + im * im)
        var ln10 = 2.302585092994046
        return (_log(modulus) / ln10, _atan2(im, re) / ln10)
    if op == UnaryOp.LOG1P.value:
        # log(1 + z)
        var nre = 1.0 + re
        var modulus = _sqrt(nre * nre + im * im)
        return (_log(modulus), _atan2(im, nre))
    if op == UnaryOp.EXPM1.value:
        # exp(z) - 1 — use exp formula then subtract 1 from real part.
        var ea = _exp(re)
        return (ea * _cos(im) - 1.0, ea * _sin(im))
    if op == UnaryOp.RECIPROCAL.value:
        # 1 / (a + bi) = (a - bi) / (a² + b²)
        var denom = re * re + im * im
        return (re / denom, -im / denom)
    if op == UnaryOp.EXP2.value:
        # 2^z = exp(z * log(2))
        var ln2 = 0.6931471805599453
        var nre = re * ln2
        var nim = im * ln2
        var ea = _exp(nre)
        return (ea * _cos(nim), ea * _sin(nim))
    if op == UnaryOp.CBRT.value:
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
    return binary_op_method_ops(py_self, other_obj, BinaryOp.ADD.value)


def array_sub_method_ops(py_self: PythonObject, other_obj: PythonObject) raises -> PythonObject:
    return binary_op_method_ops(py_self, other_obj, BinaryOp.SUB.value)


def array_mul_method_ops(py_self: PythonObject, other_obj: PythonObject) raises -> PythonObject:
    return binary_op_method_ops(py_self, other_obj, BinaryOp.MUL.value)


def array_div_method_ops(py_self: PythonObject, other_obj: PythonObject) raises -> PythonObject:
    return binary_op_method_ops(py_self, other_obj, BinaryOp.DIV.value)


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
        result_dtype_for_binary(self_ptr[].dtype_code, other_ptr[].dtype_code, BinaryOp.MUL.value),
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
    var op = BinaryOp.from_int(Int(py=op_obj)).value
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
    var op = BinaryOp.from_int(Int(py=op_obj)).value
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
    var op = BinaryOp.from_int(Int(py=op_obj)).value
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
    var rhs_mul_dtype = result_dtype_for_binary(rhs[].dtype_code, scalar_dtype, BinaryOp.MUL.value)
    var dtype_code = result_dtype_for_binary(
        result_dtype_for_unary(lhs[].dtype_code), rhs_mul_dtype, BinaryOp.ADD.value
    )
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
    result.backend_code = BackendKind.FUSED.value
    return PythonObject(alloc=result^)


def where_ops(cond_obj: PythonObject, lhs_obj: PythonObject, rhs_obj: PythonObject) raises -> PythonObject:
    var cond = cond_obj.downcast_value_ptr[Array]()
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
    var partial_shape = broadcast_shape(cond[], lhs[])
    var tmp = make_empty_array(lhs[].dtype_code, partial_shape^)
    var shape = broadcast_shape(tmp, rhs[])
    var dtype_code = result_dtype_for_binary(lhs[].dtype_code, rhs[].dtype_code, BinaryOp.ADD.value)
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
        if cond[].dtype_code == ArrayDType.BOOL.value:
            picked = 1.0 if get_physical_bool(cond[], cond_phys) else 0.0
        else:
            picked = get_physical_as_f64(cond[], cond_phys)
        if picked != 0.0:
            set_physical_from_f64(result, iter.element_index(3), get_physical_as_f64(lhs[], iter.element_index(1)))
        else:
            set_physical_from_f64(result, iter.element_index(3), get_physical_as_f64(rhs[], iter.element_index(2)))
        iter.step()
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
    if dtype_code == ArrayDType.FLOAT32.value:
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
    if dtype_code == ArrayDType.FLOAT64.value:
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
    if dtype_code == ArrayDType.FLOAT32.value:
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
    if dtype_code == ArrayDType.FLOAT64.value:
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
