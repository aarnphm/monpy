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
from .elementwise_ops import (
    apply_unary_complex_f64,
    array_add_method_ops,
    array_div_method_ops,
    array_matmul_method_ops,
    array_mul_method_ops,
    array_sub_method_ops,
    binary_dispatch_ops,
    binary_into_ops,
    binary_op_method_ops,
    binary_ops,
    binary_scalar_ops,
    compare_ops,
    logical_ops,
    predicate_ops,
    sin_add_mul_ops,
    unary_ops,
    unary_preserve_ops,
    where_ops,
)
from .reduction_ops import reduce_axis_ops, reduce_ops
from .shape_ops import (
    DiagonalMetadata,
    broadcast_to_ops,
    concatenate_ops,
    diagonal_metadata_ops,
    diagonal_ops,
    expand_dims_ops,
    flatten_ops,
    flip_ops,
    materialize_c_contiguous_ops,
    normalize_axis_ops,
    pad_constant_ops,
    ravel_ops,
    reshape_ops,
    slice_ops,
    squeeze_all_ops,
    squeeze_axes_ops,
    squeeze_axis_ops,
    stack_axis0_ops,
    swapaxes_ops,
    trace_ops,
    transpose_full_reverse_ops,
    transpose_ops,
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
