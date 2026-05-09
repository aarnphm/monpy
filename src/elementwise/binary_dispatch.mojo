"""Binary maybe_* dispatchers: contig, scalar, row-broadcast, strided, top-level.

These are the entry points called by the upstream `apply_binary_*` ops in
`src/create/ops/elementwise.mojo`. Each `maybe_*` returns True if it took a
fast path, False if the caller should fall back to the f64 round-trip
walker.

`maybe_binary_contiguous` is the top-level fan-out: it tries the
shape-specific fast paths in cost order (same-shape contig → scalar →
row-broadcast → rank-1 vDSP-strided → tile kernels → general strided →
column-broadcast tiles), short-circuiting on the first match.
"""

from std.sys import CompilationTarget

from array import (
    Array,
    contiguous_as_f64,
    contiguous_ptr,
    get_physical_c128_imag,
    get_physical_c128_real,
    get_physical_c64_imag,
    get_physical_c64_real,
    is_c_contiguous,
    same_shape,
    set_contiguous_from_f64,
)
from domain import ArrayDType, BackendKind, BinaryOp

from .accelerate_dispatch import (
    maybe_binary_accelerate,
    maybe_binary_rank1_strided_accelerate,
)
from .apply_scalar import apply_binary_f64
from .kernels.complex import (
    complex_binary_contig_typed,
    complex_scalar_complex_contig_typed,
    complex_scalar_real_contig_typed,
    maybe_complex_binary_contiguous_accelerate,
    maybe_complex_binary_same_shape_strided,
)
from .dispatch_helpers import dispatch_real_typed_simd_binary
from .strided_walkers import strided_binary_walk_typed
from .kernels.tile import (
    maybe_binary_column_broadcast_dispatch,
    maybe_binary_rank2_transposed_tile,
    maybe_binary_rank3_axis0_tile,
    pick_inner_axis_for_strided_binary,
)
from .kernels.typed import (
    binary_column_broadcast_contig_typed,
    binary_row_broadcast_contig_typed,
    binary_same_shape_contig_typed,
    binary_scalar_contig_typed,
)


def maybe_binary_same_shape_contiguous(lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    if (
        not same_shape(lhs.shape, rhs.shape)
        or not same_shape(lhs.shape, result.shape)
        or not is_c_contiguous(lhs)
        or not is_c_contiguous(rhs)
        or not is_c_contiguous(result)
    ):
        return False
    # Complex paths first (storage is interleaved float pairs).
    if (
        lhs.dtype_code == ArrayDType.COMPLEX64.value
        and rhs.dtype_code == ArrayDType.COMPLEX64.value
        and result.dtype_code == ArrayDType.COMPLEX64.value
    ):
        if maybe_complex_binary_contiguous_accelerate(lhs, rhs, result, op):
            return True
        complex_binary_contig_typed[DType.float32](
            contiguous_ptr[DType.float32](lhs),
            contiguous_ptr[DType.float32](rhs),
            contiguous_ptr[DType.float32](result),
            result.size_value,
            op,
        )
        result.backend_code = BackendKind.FUSED.value
        return True
    if (
        lhs.dtype_code == ArrayDType.COMPLEX128.value
        and rhs.dtype_code == ArrayDType.COMPLEX128.value
        and result.dtype_code == ArrayDType.COMPLEX128.value
    ):
        if maybe_complex_binary_contiguous_accelerate(lhs, rhs, result, op):
            return True
        complex_binary_contig_typed[DType.float64](
            contiguous_ptr[DType.float64](lhs),
            contiguous_ptr[DType.float64](rhs),
            contiguous_ptr[DType.float64](result),
            result.size_value,
            op,
        )
        result.backend_code = BackendKind.FUSED.value
        return True
    # Accelerate fast-paths for f32/f64 (macOS only). Try first, fall through to typed kernel.
    comptime if CompilationTarget.is_macos():
        if (
            lhs.dtype_code == ArrayDType.FLOAT32.value
            and rhs.dtype_code == ArrayDType.FLOAT32.value
            and result.dtype_code == ArrayDType.FLOAT32.value
        ):
            if maybe_binary_accelerate[DType.float32](lhs, rhs, result, op):
                return True
        elif (
            lhs.dtype_code == ArrayDType.FLOAT64.value
            and rhs.dtype_code == ArrayDType.FLOAT64.value
            and result.dtype_code == ArrayDType.FLOAT64.value
        ):
            if maybe_binary_accelerate[DType.float64](lhs, rhs, result, op):
                return True
    # 11-way real-dtype dispatch via comptime-fn-parametric helper. Caller invariant:
    # all three arrays share dtype_code (we only enter this path when the dispatcher
    # has already promoted to a single type). f16 KGEN gates atan2/hypot/copysign
    # inside `apply_binary_typed_vec` itself; ARCTAN2/HYPOT/COPYSIGN with f16 inputs
    # are upstream-promoted to f32/f64 via `dtype_result_for_binary` so they never
    # reach this kernel.
    if lhs.dtype_code == rhs.dtype_code and lhs.dtype_code == result.dtype_code:
        if dispatch_real_typed_simd_binary[binary_same_shape_contig_typed](
            result.dtype_code, lhs, rhs, result, result.size_value, op
        ):
            return True
    # Fallback: f64 round-trip for any dtype combo we don't have a typed path for.
    for i in range(result.size_value):
        set_contiguous_from_f64(
            result,
            i,
            apply_binary_f64(contiguous_as_f64(lhs, i), contiguous_as_f64(rhs, i), op),
        )
    return True


def maybe_binary_scalar_contiguous(
    array: Array,
    scalar: Array,
    mut result: Array,
    op: Int,
    scalar_on_left: Bool,
) raises -> Bool:
    if (
        len(scalar.shape) != 0
        or not same_shape(array.shape, result.shape)
        or not is_c_contiguous(array)
        or not is_c_contiguous(scalar)
        or not is_c_contiguous(result)
    ):
        return False
    # Complex paths: array is complex, scalar may be complex or real.
    if array.dtype_code == ArrayDType.COMPLEX64.value and result.dtype_code == ArrayDType.COMPLEX64.value:
        var s_real: Float32
        var s_imag: Float32
        if scalar.dtype_code == ArrayDType.COMPLEX64.value:
            s_real = get_physical_c64_real(scalar, scalar.offset_elems)
            s_imag = get_physical_c64_imag(scalar, scalar.offset_elems)
        elif scalar.dtype_code == ArrayDType.COMPLEX128.value:
            s_real = Float32(get_physical_c128_real(scalar, scalar.offset_elems))
            s_imag = Float32(get_physical_c128_imag(scalar, scalar.offset_elems))
        else:
            s_real = Float32(contiguous_as_f64(scalar, 0))
            s_imag = 0.0
        complex_scalar_complex_contig_typed[DType.float32](
            contiguous_ptr[DType.float32](array),
            s_real,
            s_imag,
            contiguous_ptr[DType.float32](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.COMPLEX128.value and result.dtype_code == ArrayDType.COMPLEX128.value:
        var s_real: Float64
        var s_imag: Float64
        if scalar.dtype_code == ArrayDType.COMPLEX128.value:
            s_real = get_physical_c128_real(scalar, scalar.offset_elems)
            s_imag = get_physical_c128_imag(scalar, scalar.offset_elems)
        elif scalar.dtype_code == ArrayDType.COMPLEX64.value:
            s_real = Float64(get_physical_c64_real(scalar, scalar.offset_elems))
            s_imag = Float64(get_physical_c64_imag(scalar, scalar.offset_elems))
        else:
            s_real = contiguous_as_f64(scalar, 0)
            s_imag = 0.0
        complex_scalar_complex_contig_typed[DType.float64](
            contiguous_ptr[DType.float64](array),
            s_real,
            s_imag,
            contiguous_ptr[DType.float64](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    var scalar_value = contiguous_as_f64(scalar, 0)
    if array.dtype_code == ArrayDType.FLOAT32.value and result.dtype_code == ArrayDType.FLOAT32.value:
        binary_scalar_contig_typed[DType.float32](
            contiguous_ptr[DType.float32](array),
            Float32(scalar_value),
            contiguous_ptr[DType.float32](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.FLOAT64.value and result.dtype_code == ArrayDType.FLOAT64.value:
        binary_scalar_contig_typed[DType.float64](
            contiguous_ptr[DType.float64](array),
            scalar_value,
            contiguous_ptr[DType.float64](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    # Typed int paths: int32/int64/uint32/uint64.
    if array.dtype_code == ArrayDType.INT64.value and result.dtype_code == ArrayDType.INT64.value:
        binary_scalar_contig_typed[DType.int64](
            contiguous_ptr[DType.int64](array),
            Int64(Int(scalar_value)),
            contiguous_ptr[DType.int64](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.INT32.value and result.dtype_code == ArrayDType.INT32.value:
        binary_scalar_contig_typed[DType.int32](
            contiguous_ptr[DType.int32](array),
            Int32(Int(scalar_value)),
            contiguous_ptr[DType.int32](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.UINT64.value and result.dtype_code == ArrayDType.UINT64.value:
        binary_scalar_contig_typed[DType.uint64](
            contiguous_ptr[DType.uint64](array),
            UInt64(Int(scalar_value)),
            contiguous_ptr[DType.uint64](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.UINT32.value and result.dtype_code == ArrayDType.UINT32.value:
        binary_scalar_contig_typed[DType.uint32](
            contiguous_ptr[DType.uint32](array),
            UInt32(Int(scalar_value)),
            contiguous_ptr[DType.uint32](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.INT16.value and result.dtype_code == ArrayDType.INT16.value:
        binary_scalar_contig_typed[DType.int16](
            contiguous_ptr[DType.int16](array),
            Int16(Int(scalar_value)),
            contiguous_ptr[DType.int16](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.INT8.value and result.dtype_code == ArrayDType.INT8.value:
        binary_scalar_contig_typed[DType.int8](
            contiguous_ptr[DType.int8](array),
            Int8(Int(scalar_value)),
            contiguous_ptr[DType.int8](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.UINT16.value and result.dtype_code == ArrayDType.UINT16.value:
        binary_scalar_contig_typed[DType.uint16](
            contiguous_ptr[DType.uint16](array),
            UInt16(Int(scalar_value)),
            contiguous_ptr[DType.uint16](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.UINT8.value and result.dtype_code == ArrayDType.UINT8.value:
        binary_scalar_contig_typed[DType.uint8](
            contiguous_ptr[DType.uint8](array),
            UInt8(Int(scalar_value)),
            contiguous_ptr[DType.uint8](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    for i in range(result.size_value):
        var lhs = contiguous_as_f64(array, i)
        var rhs = scalar_value
        if scalar_on_left:
            lhs = scalar_value
            rhs = contiguous_as_f64(array, i)
        set_contiguous_from_f64(result, i, apply_binary_f64(lhs, rhs, op))
    return True


def maybe_binary_scalar_value_contiguous(
    array: Array,
    scalar_value: Float64,
    mut result: Array,
    op: Int,
    scalar_on_left: Bool,
) raises -> Bool:
    if not same_shape(array.shape, result.shape) or not is_c_contiguous(array) or not is_c_contiguous(result):
        return False
    # Complex × real-scalar paths.
    if array.dtype_code == ArrayDType.COMPLEX64.value and result.dtype_code == ArrayDType.COMPLEX64.value:
        complex_scalar_real_contig_typed[DType.float32](
            contiguous_ptr[DType.float32](array),
            Float32(scalar_value),
            contiguous_ptr[DType.float32](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.COMPLEX128.value and result.dtype_code == ArrayDType.COMPLEX128.value:
        complex_scalar_real_contig_typed[DType.float64](
            contiguous_ptr[DType.float64](array),
            scalar_value,
            contiguous_ptr[DType.float64](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.FLOAT32.value and result.dtype_code == ArrayDType.FLOAT32.value:
        binary_scalar_contig_typed[DType.float32](
            contiguous_ptr[DType.float32](array),
            Float32(scalar_value),
            contiguous_ptr[DType.float32](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.FLOAT64.value and result.dtype_code == ArrayDType.FLOAT64.value:
        binary_scalar_contig_typed[DType.float64](
            contiguous_ptr[DType.float64](array),
            scalar_value,
            contiguous_ptr[DType.float64](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.INT64.value and result.dtype_code == ArrayDType.INT64.value:
        binary_scalar_contig_typed[DType.int64](
            contiguous_ptr[DType.int64](array),
            Int64(Int(scalar_value)),
            contiguous_ptr[DType.int64](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.INT32.value and result.dtype_code == ArrayDType.INT32.value:
        binary_scalar_contig_typed[DType.int32](
            contiguous_ptr[DType.int32](array),
            Int32(Int(scalar_value)),
            contiguous_ptr[DType.int32](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.UINT64.value and result.dtype_code == ArrayDType.UINT64.value:
        binary_scalar_contig_typed[DType.uint64](
            contiguous_ptr[DType.uint64](array),
            UInt64(Int(scalar_value)),
            contiguous_ptr[DType.uint64](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.UINT32.value and result.dtype_code == ArrayDType.UINT32.value:
        binary_scalar_contig_typed[DType.uint32](
            contiguous_ptr[DType.uint32](array),
            UInt32(Int(scalar_value)),
            contiguous_ptr[DType.uint32](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.INT16.value and result.dtype_code == ArrayDType.INT16.value:
        binary_scalar_contig_typed[DType.int16](
            contiguous_ptr[DType.int16](array),
            Int16(Int(scalar_value)),
            contiguous_ptr[DType.int16](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.INT8.value and result.dtype_code == ArrayDType.INT8.value:
        binary_scalar_contig_typed[DType.int8](
            contiguous_ptr[DType.int8](array),
            Int8(Int(scalar_value)),
            contiguous_ptr[DType.int8](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.UINT16.value and result.dtype_code == ArrayDType.UINT16.value:
        binary_scalar_contig_typed[DType.uint16](
            contiguous_ptr[DType.uint16](array),
            UInt16(Int(scalar_value)),
            contiguous_ptr[DType.uint16](result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == ArrayDType.UINT8.value and result.dtype_code == ArrayDType.UINT8.value:
        binary_scalar_contig_typed[DType.uint8](
            contiguous_ptr[DType.uint8](array),
            UInt8(Int(scalar_value)),
            contiguous_ptr[DType.uint8](result),
            result.size_value,
            op,
            scalar_on_left,
        )
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
    matrix: Array,
    row: Array,
    mut result: Array,
    op: Int,
    row_on_left: Bool,
) raises -> Bool:
    if (
        len(matrix.shape) != 2
        or len(row.shape) != 1
        or row.shape[0] != matrix.shape[1]
        or not same_shape(matrix.shape, result.shape)
        or not is_c_contiguous(matrix)
        or not is_c_contiguous(row)
        or not is_c_contiguous(result)
    ):
        return False
    var rows = matrix.shape[0]
    var cols = matrix.shape[1]
    if (
        matrix.dtype_code == ArrayDType.FLOAT32.value
        and row.dtype_code == ArrayDType.FLOAT32.value
        and result.dtype_code == ArrayDType.FLOAT32.value
    ):
        binary_row_broadcast_contig_typed[DType.float32](
            contiguous_ptr[DType.float32](matrix),
            contiguous_ptr[DType.float32](row),
            contiguous_ptr[DType.float32](result),
            rows,
            cols,
            op,
            row_on_left,
        )
        return True
    if (
        matrix.dtype_code == ArrayDType.FLOAT64.value
        and row.dtype_code == ArrayDType.FLOAT64.value
        and result.dtype_code == ArrayDType.FLOAT64.value
    ):
        binary_row_broadcast_contig_typed[DType.float64](
            contiguous_ptr[DType.float64](matrix),
            contiguous_ptr[DType.float64](row),
            contiguous_ptr[DType.float64](result),
            rows,
            cols,
            op,
            row_on_left,
        )
        return True
    # Typed int paths.
    if (
        matrix.dtype_code == ArrayDType.INT64.value
        and row.dtype_code == ArrayDType.INT64.value
        and result.dtype_code == ArrayDType.INT64.value
    ):
        binary_row_broadcast_contig_typed[DType.int64](
            contiguous_ptr[DType.int64](matrix),
            contiguous_ptr[DType.int64](row),
            contiguous_ptr[DType.int64](result),
            rows,
            cols,
            op,
            row_on_left,
        )
        return True
    if (
        matrix.dtype_code == ArrayDType.INT32.value
        and row.dtype_code == ArrayDType.INT32.value
        and result.dtype_code == ArrayDType.INT32.value
    ):
        binary_row_broadcast_contig_typed[DType.int32](
            contiguous_ptr[DType.int32](matrix),
            contiguous_ptr[DType.int32](row),
            contiguous_ptr[DType.int32](result),
            rows,
            cols,
            op,
            row_on_left,
        )
        return True
    if (
        matrix.dtype_code == ArrayDType.UINT64.value
        and row.dtype_code == ArrayDType.UINT64.value
        and result.dtype_code == ArrayDType.UINT64.value
    ):
        binary_row_broadcast_contig_typed[DType.uint64](
            contiguous_ptr[DType.uint64](matrix),
            contiguous_ptr[DType.uint64](row),
            contiguous_ptr[DType.uint64](result),
            rows,
            cols,
            op,
            row_on_left,
        )
        return True
    if (
        matrix.dtype_code == ArrayDType.UINT32.value
        and row.dtype_code == ArrayDType.UINT32.value
        and result.dtype_code == ArrayDType.UINT32.value
    ):
        binary_row_broadcast_contig_typed[DType.uint32](
            contiguous_ptr[DType.uint32](matrix),
            contiguous_ptr[DType.uint32](row),
            contiguous_ptr[DType.uint32](result),
            rows,
            cols,
            op,
            row_on_left,
        )
        return True
    if (
        matrix.dtype_code == ArrayDType.INT16.value
        and row.dtype_code == ArrayDType.INT16.value
        and result.dtype_code == ArrayDType.INT16.value
    ):
        binary_row_broadcast_contig_typed[DType.int16](
            contiguous_ptr[DType.int16](matrix),
            contiguous_ptr[DType.int16](row),
            contiguous_ptr[DType.int16](result),
            rows,
            cols,
            op,
            row_on_left,
        )
        return True
    if (
        matrix.dtype_code == ArrayDType.INT8.value
        and row.dtype_code == ArrayDType.INT8.value
        and result.dtype_code == ArrayDType.INT8.value
    ):
        binary_row_broadcast_contig_typed[DType.int8](
            contiguous_ptr[DType.int8](matrix),
            contiguous_ptr[DType.int8](row),
            contiguous_ptr[DType.int8](result),
            rows,
            cols,
            op,
            row_on_left,
        )
        return True
    if (
        matrix.dtype_code == ArrayDType.UINT16.value
        and row.dtype_code == ArrayDType.UINT16.value
        and result.dtype_code == ArrayDType.UINT16.value
    ):
        binary_row_broadcast_contig_typed[DType.uint16](
            contiguous_ptr[DType.uint16](matrix),
            contiguous_ptr[DType.uint16](row),
            contiguous_ptr[DType.uint16](result),
            rows,
            cols,
            op,
            row_on_left,
        )
        return True
    if (
        matrix.dtype_code == ArrayDType.UINT8.value
        and row.dtype_code == ArrayDType.UINT8.value
        and result.dtype_code == ArrayDType.UINT8.value
    ):
        binary_row_broadcast_contig_typed[DType.uint8](
            contiguous_ptr[DType.uint8](matrix),
            contiguous_ptr[DType.uint8](row),
            contiguous_ptr[DType.uint8](result),
            rows,
            cols,
            op,
            row_on_left,
        )
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


def maybe_binary_column_keepdims_broadcast_contiguous(
    matrix: Array,
    column: Array,
    mut result: Array,
    op: Int,
    column_on_left: Bool,
) raises -> Bool:
    if (
        len(matrix.shape) != 2
        or len(column.shape) != 2
        or column.shape[0] != matrix.shape[0]
        or column.shape[1] != 1
        or not same_shape(matrix.shape, result.shape)
        or not is_c_contiguous(matrix)
        or not is_c_contiguous(column)
        or not is_c_contiguous(result)
    ):
        return False
    var rows = matrix.shape[0]
    var cols = matrix.shape[1]
    if (
        matrix.dtype_code == ArrayDType.FLOAT32.value
        and column.dtype_code == ArrayDType.FLOAT32.value
        and result.dtype_code == ArrayDType.FLOAT32.value
    ):
        binary_column_broadcast_contig_typed[DType.float32](
            contiguous_ptr[DType.float32](matrix),
            contiguous_ptr[DType.float32](column),
            contiguous_ptr[DType.float32](result),
            rows,
            cols,
            op,
            column_on_left,
        )
        return True
    if (
        matrix.dtype_code == ArrayDType.FLOAT64.value
        and column.dtype_code == ArrayDType.FLOAT64.value
        and result.dtype_code == ArrayDType.FLOAT64.value
    ):
        binary_column_broadcast_contig_typed[DType.float64](
            contiguous_ptr[DType.float64](matrix),
            contiguous_ptr[DType.float64](column),
            contiguous_ptr[DType.float64](result),
            rows,
            cols,
            op,
            column_on_left,
        )
        return True
    return False


def maybe_binary_same_shape_strided(lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    # General N-D same-shape strided walker. Walks `inner_axis` with SIMD when
    # possible (full or load-only) and walks the remaining axes with a coord
    # stack that uses incremental offset arithmetic (no divmod per element).
    # Subsumes the previous rank-1 and rank-2 special cases.
    if not same_shape(lhs.shape, rhs.shape) or not same_shape(lhs.shape, result.shape):
        return False
    if lhs.dtype_code != rhs.dtype_code or rhs.dtype_code != result.dtype_code:
        return False
    if (
        lhs.dtype_code != ArrayDType.FLOAT32.value
        and lhs.dtype_code != ArrayDType.FLOAT64.value
        and lhs.dtype_code != ArrayDType.INT64.value
        and lhs.dtype_code != ArrayDType.INT32.value
        and lhs.dtype_code != ArrayDType.UINT64.value
        and lhs.dtype_code != ArrayDType.UINT32.value
    ):
        return False
    var ndim = len(lhs.shape)
    if ndim == 0:
        return False
    var total = lhs.size_value
    if total == 0:
        return True
    var picked = pick_inner_axis_for_strided_binary(lhs, rhs, result)
    var inner_axis = picked.axis
    var inner_kind = picked.kind
    var inner_size = lhs.shape[inner_axis]
    var inner_lhs_stride = lhs.strides[inner_axis]
    var inner_rhs_stride = rhs.strides[inner_axis]
    var inner_result_stride = result.strides[inner_axis]
    # Build the outer axes list (every axis except inner). Pre-compute the
    # carry-back step per outer axis so the inner loop touches only Ints.
    var outer_axes = List[Int]()
    var outer_shape = List[Int]()
    var outer_lhs_stride = List[Int]()
    var outer_rhs_stride = List[Int]()
    var outer_result_stride = List[Int]()
    var outer_lhs_carry = List[Int]()
    var outer_rhs_carry = List[Int]()
    var outer_result_carry = List[Int]()
    for axis in range(ndim):
        if axis == inner_axis:
            continue
        var dim = lhs.shape[axis]
        outer_axes.append(axis)
        outer_shape.append(dim)
        outer_lhs_stride.append(lhs.strides[axis])
        outer_rhs_stride.append(rhs.strides[axis])
        outer_result_stride.append(result.strides[axis])
        var span = 0
        if dim > 1:
            span = dim - 1
        outer_lhs_carry.append(lhs.strides[axis] * span)
        outer_rhs_carry.append(rhs.strides[axis] * span)
        outer_result_carry.append(result.strides[axis] * span)
    if lhs.dtype_code == ArrayDType.FLOAT32.value:
        strided_binary_walk_typed[DType.float32](
            lhs,
            rhs,
            result,
            inner_kind,
            inner_size,
            inner_lhs_stride,
            inner_rhs_stride,
            inner_result_stride,
            outer_shape,
            outer_lhs_stride,
            outer_rhs_stride,
            outer_result_stride,
            outer_lhs_carry,
            outer_rhs_carry,
            outer_result_carry,
            op,
        )
        return True
    if lhs.dtype_code == ArrayDType.FLOAT64.value:
        strided_binary_walk_typed[DType.float64](
            lhs,
            rhs,
            result,
            inner_kind,
            inner_size,
            inner_lhs_stride,
            inner_rhs_stride,
            inner_result_stride,
            outer_shape,
            outer_lhs_stride,
            outer_rhs_stride,
            outer_result_stride,
            outer_lhs_carry,
            outer_rhs_carry,
            outer_result_carry,
            op,
        )
        return True
    if lhs.dtype_code == ArrayDType.INT64.value:
        strided_binary_walk_typed[DType.int64](
            lhs,
            rhs,
            result,
            inner_kind,
            inner_size,
            inner_lhs_stride,
            inner_rhs_stride,
            inner_result_stride,
            outer_shape,
            outer_lhs_stride,
            outer_rhs_stride,
            outer_result_stride,
            outer_lhs_carry,
            outer_rhs_carry,
            outer_result_carry,
            op,
        )
        return True
    if lhs.dtype_code == ArrayDType.INT32.value:
        strided_binary_walk_typed[DType.int32](
            lhs,
            rhs,
            result,
            inner_kind,
            inner_size,
            inner_lhs_stride,
            inner_rhs_stride,
            inner_result_stride,
            outer_shape,
            outer_lhs_stride,
            outer_rhs_stride,
            outer_result_stride,
            outer_lhs_carry,
            outer_rhs_carry,
            outer_result_carry,
            op,
        )
        return True
    if lhs.dtype_code == ArrayDType.UINT64.value:
        strided_binary_walk_typed[DType.uint64](
            lhs,
            rhs,
            result,
            inner_kind,
            inner_size,
            inner_lhs_stride,
            inner_rhs_stride,
            inner_result_stride,
            outer_shape,
            outer_lhs_stride,
            outer_rhs_stride,
            outer_result_stride,
            outer_lhs_carry,
            outer_rhs_carry,
            outer_result_carry,
            op,
        )
        return True
    if lhs.dtype_code == ArrayDType.UINT32.value:
        strided_binary_walk_typed[DType.uint32](
            lhs,
            rhs,
            result,
            inner_kind,
            inner_size,
            inner_lhs_stride,
            inner_rhs_stride,
            inner_result_stride,
            outer_shape,
            outer_lhs_stride,
            outer_rhs_stride,
            outer_result_stride,
            outer_lhs_carry,
            outer_rhs_carry,
            outer_result_carry,
            op,
        )
        return True
    return False


def maybe_binary_contiguous(lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    # Fast-path dispatch is intentionally shape-specific here. We want to be dumb.
    # The fallback below still handles dynamic-rank broadcasting, so every branch
    # here must be a provably cheaper case with the same semantics.
    if maybe_binary_same_shape_contiguous(lhs, rhs, result, op):
        return True
    if maybe_binary_scalar_contiguous(lhs, rhs, result, op, False):
        return True
    if maybe_binary_scalar_contiguous(rhs, lhs, result, op, True):
        return True
    if maybe_complex_binary_same_shape_strided(lhs, rhs, result, op):
        return True
    if maybe_binary_row_broadcast_contiguous(lhs, rhs, result, op, False):
        return True
    if maybe_binary_row_broadcast_contiguous(rhs, lhs, result, op, True):
        return True
    if maybe_binary_column_keepdims_broadcast_contiguous(lhs, rhs, result, op, False):
        return True
    if maybe_binary_column_keepdims_broadcast_contiguous(rhs, lhs, result, op, True):
        return True
    if maybe_binary_rank1_strided_accelerate(lhs, rhs, result, op):
        return True
    if (
        lhs.dtype_code == ArrayDType.FLOAT32.value
        and rhs.dtype_code == ArrayDType.FLOAT32.value
        and result.dtype_code == ArrayDType.FLOAT32.value
    ):
        if maybe_binary_rank2_transposed_tile[DType.float32](lhs, rhs, result, op):
            return True
        if maybe_binary_rank3_axis0_tile[DType.float32](lhs, rhs, result, op):
            return True
    elif (
        lhs.dtype_code == ArrayDType.FLOAT64.value
        and rhs.dtype_code == ArrayDType.FLOAT64.value
        and result.dtype_code == ArrayDType.FLOAT64.value
    ):
        if maybe_binary_rank2_transposed_tile[DType.float64](lhs, rhs, result, op):
            return True
        if maybe_binary_rank3_axis0_tile[DType.float64](lhs, rhs, result, op):
            return True
    elif (
        lhs.dtype_code == ArrayDType.INT64.value
        and rhs.dtype_code == ArrayDType.INT64.value
        and result.dtype_code == ArrayDType.INT64.value
    ):
        if maybe_binary_rank2_transposed_tile[DType.int64](lhs, rhs, result, op):
            return True
        if maybe_binary_rank3_axis0_tile[DType.int64](lhs, rhs, result, op):
            return True
    elif (
        lhs.dtype_code == ArrayDType.INT32.value
        and rhs.dtype_code == ArrayDType.INT32.value
        and result.dtype_code == ArrayDType.INT32.value
    ):
        if maybe_binary_rank2_transposed_tile[DType.int32](lhs, rhs, result, op):
            return True
        if maybe_binary_rank3_axis0_tile[DType.int32](lhs, rhs, result, op):
            return True
    elif (
        lhs.dtype_code == ArrayDType.UINT64.value
        and rhs.dtype_code == ArrayDType.UINT64.value
        and result.dtype_code == ArrayDType.UINT64.value
    ):
        if maybe_binary_rank2_transposed_tile[DType.uint64](lhs, rhs, result, op):
            return True
        if maybe_binary_rank3_axis0_tile[DType.uint64](lhs, rhs, result, op):
            return True
    elif (
        lhs.dtype_code == ArrayDType.UINT32.value
        and rhs.dtype_code == ArrayDType.UINT32.value
        and result.dtype_code == ArrayDType.UINT32.value
    ):
        if maybe_binary_rank2_transposed_tile[DType.uint32](lhs, rhs, result, op):
            return True
        if maybe_binary_rank3_axis0_tile[DType.uint32](lhs, rhs, result, op):
            return True
    if maybe_binary_same_shape_strided(lhs, rhs, result, op):
        return True
    if maybe_binary_column_broadcast_dispatch(lhs, rhs, result, op):
        return True
    return False
