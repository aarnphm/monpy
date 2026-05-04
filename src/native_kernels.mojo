from std.math import cos, exp, isinf, isnan, log, sin
from std.memory.unsafe_pointer import alloc
from std.sys import CompilationTarget, simd_width_of

from native_accelerate import (
    call_vv_f32,
    cblas_dgemm_row_major_ld,
    cblas_sgemm_row_major_ld,
    lapack_dgesv,
    lapack_dgetrf,
    lapack_sgesv,
    lapack_sgetrf,
)
from native_types import (
    BACKEND_ACCELERATE,
    BACKEND_FUSED,
    BACKEND_GENERIC,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    NativeArray,
    OP_ADD,
    OP_DIV,
    OP_MUL,
    OP_SUB,
    REDUCE_MEAN,
    REDUCE_MAX,
    REDUCE_MIN,
    REDUCE_SUM,
    UNARY_COS,
    UNARY_EXP,
    UNARY_LOG,
    UNARY_SIN,
    contiguous_as_f64,
    contiguous_f32_ptr,
    contiguous_f64_ptr,
    get_logical_as_f64,
    has_negative_strides,
    has_zero_strides,
    is_c_contiguous,
    same_shape,
    set_contiguous_from_f64,
    set_logical_from_f64,
    set_logical_from_i64,
)


# Numeric kernels are intentionally plain functions. Add a new predicate plus
# a direct loop here when a benchmark proves the shape/dtype case is worth it.
def write_add_f32_1d_into(
    mut dst: NativeArray,
    lhs: NativeArray,
    rhs: NativeArray,
) raises -> Bool:
    if (
        lhs.dtype_code != DTYPE_FLOAT32
        or rhs.dtype_code != DTYPE_FLOAT32
        or dst.dtype_code != DTYPE_FLOAT32
        or len(lhs.shape) != 1
        or len(rhs.shape) != 1
        or len(dst.shape) != 1
        or lhs.size_value != rhs.size_value
        or lhs.size_value != dst.size_value
        or (lhs.shape[0] > 1 and lhs.strides[0] != 1)
        or (rhs.shape[0] > 1 and rhs.strides[0] != 1)
        or (dst.shape[0] > 1 and dst.strides[0] != 1)
    ):
        return False
    var lhs_ptr = contiguous_f32_ptr(lhs)
    var rhs_ptr = contiguous_f32_ptr(rhs)
    var out_ptr = contiguous_f32_ptr(dst)
    comptime width = simd_width_of[DType.float32]()
    var i = 0
    while i + width <= dst.size_value:
        out_ptr.store(
            i, lhs_ptr.load[width=width](i) + rhs_ptr.load[width=width](i)
        )
        i += width
    while i < dst.size_value:
        out_ptr[i] = lhs_ptr[i] + rhs_ptr[i]
        i += 1
    dst.backend_code = BACKEND_GENERIC
    return True


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


def apply_binary_f32_vec[
    width: Int
](
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


def apply_binary_f64_vec[
    width: Int
](
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


def apply_unary_f32_vec[
    width: Int
](value: SIMD[DType.float32, width], op: Int) raises -> SIMD[
    DType.float32, width
]:
    if op == UNARY_SIN:
        return sin(value)
    if op == UNARY_COS:
        return cos(value)
    if op == UNARY_EXP:
        return exp(value)
    if op == UNARY_LOG:
        return log(value)
    raise Error("unknown unary op")


def apply_unary_f64_vec[
    width: Int
](value: SIMD[DType.float64, width], op: Int) raises -> SIMD[
    DType.float64, width
]:
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


@fieldwise_init
struct Rank2BlasLayout(ImplicitlyCopyable, Movable, Writable):
    var can_use: Bool
    var transpose: Bool
    var leading_dim: Int


def max_int(lhs: Int, rhs: Int) -> Int:
    if lhs > rhs:
        return lhs
    return rhs


def rank2_blas_layout(array: NativeArray) raises -> Rank2BlasLayout:
    if len(array.shape) != 2:
        return Rank2BlasLayout(False, False, 0)
    var rows = array.shape[0]
    var cols = array.shape[1]
    if rows == 0 or cols == 0:
        return Rank2BlasLayout(False, False, 0)
    if has_negative_strides(array) or has_zero_strides(array):
        return Rank2BlasLayout(False, False, 0)
    if cols == 1 or array.strides[1] == 1:
        var lda = array.strides[0]
        if rows == 1:
            lda = max_int(1, cols)
        if lda >= max_int(1, cols):
            return Rank2BlasLayout(True, False, lda)
    if rows == 1 or array.strides[0] == 1:
        var lda = array.strides[1]
        if cols == 1:
            lda = max_int(1, rows)
        if lda >= max_int(1, rows):
            return Rank2BlasLayout(True, True, lda)
    return Rank2BlasLayout(False, False, 0)


def maybe_unary_contiguous(
    src: NativeArray, mut result: NativeArray, op: Int
) raises -> Bool:
    if not is_contiguous_float_array(src) or not is_contiguous_float_array(
        result
    ):
        return False
    if src.dtype_code == DTYPE_FLOAT32 and result.dtype_code == DTYPE_FLOAT32:
        comptime if CompilationTarget.is_macos():
            if maybe_unary_accelerate_f32(src, result, op):
                return True
        var src_ptr = contiguous_f32_ptr(src)
        var out_ptr = contiguous_f32_ptr(result)
        comptime width = simd_width_of[DType.float32]()
        var i = 0
        while i + width <= src.size_value:
            out_ptr.store(
                i, apply_unary_f32_vec[width](src_ptr.load[width=width](i), op)
            )
            i += width
        while i < src.size_value:
            out_ptr[i] = Float32(apply_unary_f64(Float64(src_ptr[i]), op))
            i += 1
        return True
    if op == UNARY_LOG:
        return False
    if src.dtype_code == DTYPE_FLOAT64 and result.dtype_code == DTYPE_FLOAT64:
        var src_ptr = contiguous_f64_ptr(src)
        var out_ptr = contiguous_f64_ptr(result)
        comptime width = simd_width_of[DType.float64]()
        var i = 0
        while i + width <= src.size_value:
            out_ptr.store(
                i, apply_unary_f64_vec[width](src_ptr.load[width=width](i), op)
            )
            i += width
        while i < src.size_value:
            out_ptr[i] = apply_unary_f64(src_ptr[i], op)
            i += 1
        return True
    return False


def maybe_unary_accelerate_f32(
    src: NativeArray, mut result: NativeArray, op: Int
) raises -> Bool:
    var src_ptr = contiguous_f32_ptr(src)
    var out_ptr = contiguous_f32_ptr(result)
    if op == UNARY_SIN:
        call_vv_f32["vvsinf"](out_ptr, src_ptr, src.size_value)
    elif op == UNARY_COS:
        call_vv_f32["vvcosf"](out_ptr, src_ptr, src.size_value)
    elif op == UNARY_EXP:
        call_vv_f32["vvexpf"](out_ptr, src_ptr, src.size_value)
    elif op == UNARY_LOG:
        call_vv_f32["vvlogf"](out_ptr, src_ptr, src.size_value)
    else:
        return False
    result.backend_code = BACKEND_ACCELERATE
    return True


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
                    lhs_ptr.load[width=width](i),
                    rhs_ptr.load[width=width](i),
                    op,
                ),
            )
            i += width
        while i < result.size_value:
            out_ptr[i] = Float32(
                apply_binary_f64(Float64(lhs_ptr[i]), Float64(rhs_ptr[i]), op)
            )
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
                    lhs_ptr.load[width=width](i),
                    rhs_ptr.load[width=width](i),
                    op,
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
            apply_binary_f64(
                contiguous_as_f64(lhs, i), contiguous_as_f64(rhs, i), op
            ),
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
                out_ptr.store(
                    i, apply_binary_f32_vec[width](scalar_vec, array_vec, op)
                )
            else:
                out_ptr.store(
                    i, apply_binary_f32_vec[width](array_vec, scalar_vec, op)
                )
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
                out_ptr.store(
                    i, apply_binary_f64_vec[width](scalar_vec, array_vec, op)
                )
            else:
                out_ptr.store(
                    i, apply_binary_f64_vec[width](array_vec, scalar_vec, op)
                )
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


def maybe_binary_scalar_value_contiguous(
    array: NativeArray,
    scalar_value: Float64,
    mut result: NativeArray,
    op: Int,
    scalar_on_left: Bool,
) raises -> Bool:
    if (
        not same_shape(array.shape, result.shape)
        or not is_contiguous_float_array(array)
        or not is_contiguous_float_array(result)
    ):
        return False
    if array.dtype_code == DTYPE_FLOAT32 and result.dtype_code == DTYPE_FLOAT32:
        var array_ptr = contiguous_f32_ptr(array)
        var out_ptr = contiguous_f32_ptr(result)
        comptime width = simd_width_of[DType.float32]()
        var scalar_vec = SIMD[DType.float32, width](Float32(scalar_value))
        var i = 0
        while i + width <= result.size_value:
            var array_vec = array_ptr.load[width=width](i)
            if scalar_on_left:
                out_ptr.store(
                    i, apply_binary_f32_vec[width](scalar_vec, array_vec, op)
                )
            else:
                out_ptr.store(
                    i, apply_binary_f32_vec[width](array_vec, scalar_vec, op)
                )
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
                out_ptr.store(
                    i, apply_binary_f64_vec[width](scalar_vec, array_vec, op)
                )
            else:
                out_ptr.store(
                    i, apply_binary_f64_vec[width](array_vec, scalar_vec, op)
                )
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
    if (
        matrix.dtype_code == DTYPE_FLOAT32
        and row.dtype_code == DTYPE_FLOAT32
        and result.dtype_code == DTYPE_FLOAT32
    ):
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
                    out_ptr.store(
                        matrix_index,
                        apply_binary_f32_vec[width](row_vec, matrix_vec, op),
                    )
                else:
                    out_ptr.store(
                        matrix_index,
                        apply_binary_f32_vec[width](matrix_vec, row_vec, op),
                    )
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
    if (
        matrix.dtype_code == DTYPE_FLOAT64
        and row.dtype_code == DTYPE_FLOAT64
        and result.dtype_code == DTYPE_FLOAT64
    ):
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
                    out_ptr.store(
                        matrix_index,
                        apply_binary_f64_vec[width](row_vec, matrix_vec, op),
                    )
                else:
                    out_ptr.store(
                        matrix_index,
                        apply_binary_f64_vec[width](matrix_vec, row_vec, op),
                    )
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
            set_contiguous_from_f64(
                result, matrix_index, apply_binary_f64(lhs, rhs, op)
            )
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


def maybe_sin_add_mul_contiguous(
    lhs: NativeArray,
    rhs: NativeArray,
    scalar_value: Float64,
    mut result: NativeArray,
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
        var scalar_vec = SIMD[DType.float32, width](Float32(scalar_value))
        comptime if CompilationTarget.is_macos():
            call_vv_f32["vvsinf"](out_ptr, lhs_ptr, result.size_value)
            var vforce_i = 0
            while vforce_i + width <= result.size_value:
                out_ptr.store(
                    vforce_i,
                    out_ptr.load[width=width](vforce_i)
                    + rhs_ptr.load[width=width](vforce_i) * scalar_vec,
                )
                vforce_i += width
            while vforce_i < result.size_value:
                out_ptr[vforce_i] += rhs_ptr[vforce_i] * Float32(scalar_value)
                vforce_i += 1
            result.backend_code = BACKEND_FUSED
            return True
        var i = 0
        while i + width <= result.size_value:
            out_ptr.store(
                i,
                sin(lhs_ptr.load[width=width](i))
                + rhs_ptr.load[width=width](i) * scalar_vec,
            )
            i += width
        while i < result.size_value:
            out_ptr[i] = Float32(
                sin(Float64(lhs_ptr[i])) + Float64(rhs_ptr[i]) * scalar_value
            )
            i += 1
        result.backend_code = BACKEND_FUSED
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
        var scalar_vec = SIMD[DType.float64, width](scalar_value)
        var i = 0
        while i + width <= result.size_value:
            out_ptr.store(
                i,
                sin(lhs_ptr.load[width=width](i))
                + rhs_ptr.load[width=width](i) * scalar_vec,
            )
            i += width
        while i < result.size_value:
            out_ptr[i] = sin(lhs_ptr[i]) + rhs_ptr[i] * scalar_value
            i += 1
        result.backend_code = BACKEND_FUSED
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


def maybe_argmax_contiguous(
    src: NativeArray, mut result: NativeArray
) raises -> Bool:
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
    lhs: NativeArray,
    rhs: NativeArray,
    mut result: NativeArray,
    m: Int,
    n: Int,
    k_lhs: Int,
) raises -> Bool:
    if (
        len(lhs.shape) != 2
        or len(rhs.shape) != 2
        or not is_contiguous_float_array(result)
    ):
        return False
    var lhs_layout = rank2_blas_layout(lhs)
    var rhs_layout = rank2_blas_layout(rhs)
    if (
        lhs.dtype_code == DTYPE_FLOAT32
        and rhs.dtype_code == DTYPE_FLOAT32
        and result.dtype_code == DTYPE_FLOAT32
    ):
        if (
            is_c_contiguous(lhs)
            and is_c_contiguous(rhs)
            and maybe_matmul_f32_small(lhs, rhs, result, m, n, k_lhs)
        ):
            return True
        comptime if CompilationTarget.is_macos():
            if lhs_layout.can_use and rhs_layout.can_use:
                cblas_sgemm_row_major_ld(
                    m,
                    n,
                    k_lhs,
                    contiguous_f32_ptr(result),
                    contiguous_f32_ptr(lhs),
                    contiguous_f32_ptr(rhs),
                    lhs_layout.transpose,
                    rhs_layout.transpose,
                    lhs_layout.leading_dim,
                    rhs_layout.leading_dim,
                    result.strides[0],
                )
                result.backend_code = BACKEND_ACCELERATE
                return True
    if (
        lhs.dtype_code == DTYPE_FLOAT64
        and rhs.dtype_code == DTYPE_FLOAT64
        and result.dtype_code == DTYPE_FLOAT64
    ):
        comptime if CompilationTarget.is_macos():
            if lhs_layout.can_use and rhs_layout.can_use:
                cblas_dgemm_row_major_ld(
                    m,
                    n,
                    k_lhs,
                    contiguous_f64_ptr(result),
                    contiguous_f64_ptr(lhs),
                    contiguous_f64_ptr(rhs),
                    lhs_layout.transpose,
                    rhs_layout.transpose,
                    lhs_layout.leading_dim,
                    rhs_layout.leading_dim,
                    result.strides[0],
                )
                result.backend_code = BACKEND_ACCELERATE
                return True
    if not is_contiguous_float_array(lhs) or not is_contiguous_float_array(rhs):
        return False
    for i in range(m):
        for j in range(n):
            var total = 0.0
            for k in range(k_lhs):
                total += contiguous_as_f64(
                    lhs, i * k_lhs + k
                ) * contiguous_as_f64(rhs, k * n + j)
            set_contiguous_from_f64(result, i * n + j, total)
    return True


def maybe_matmul_f32_small(
    lhs: NativeArray,
    rhs: NativeArray,
    mut result: NativeArray,
    m: Int,
    n: Int,
    k_lhs: Int,
) raises -> Bool:
    if m > 16 or n > 16 or k_lhs > 16:
        return False
    var lhs_ptr = contiguous_f32_ptr(lhs)
    var rhs_ptr = contiguous_f32_ptr(rhs)
    var out_ptr = contiguous_f32_ptr(result)
    comptime width = simd_width_of[DType.float32]()
    for i in range(m):
        var j = 0
        while j + width <= n:
            var acc = SIMD[DType.float32, width](0)
            for k in range(k_lhs):
                acc += SIMD[DType.float32, width](
                    lhs_ptr[i * k_lhs + k]
                ) * rhs_ptr.load[width=width](k * n + j)
            out_ptr.store(i * n + j, acc)
            j += width
        while j < n:
            var total = Float32(0)
            for k in range(k_lhs):
                total += lhs_ptr[i * k_lhs + k] * rhs_ptr[k * n + j]
            out_ptr[i * n + j] = total
            j += 1
    return True


def abs_f64(value: Float64) -> Float64:
    if value < 0.0:
        return -value
    return value


def maybe_lapack_solve_f32(
    a: NativeArray, b: NativeArray, mut result: NativeArray
) raises -> Bool:
    if (
        a.dtype_code != DTYPE_FLOAT32
        or b.dtype_code != DTYPE_FLOAT32
        or result.dtype_code != DTYPE_FLOAT32
        or len(a.shape) != 2
        or a.shape[0] != a.shape[1]
    ):
        return False
    var n = a.shape[0]
    var rhs_columns = 1
    var vector_result = True
    if len(b.shape) == 2:
        rhs_columns = b.shape[1]
        vector_result = False
    var a_ptr = alloc[Float32](n * n)
    var b_ptr = alloc[Float32](n * rhs_columns)
    var pivots = alloc[Int32](n)
    for row in range(n):
        for col in range(n):
            a_ptr[row + col * n] = Float32(
                get_logical_as_f64(a, row * n + col)
            )
    for row in range(n):
        for col in range(rhs_columns):
            var logical = row
            if not vector_result:
                logical = row * rhs_columns + col
            b_ptr[row + col * n] = Float32(get_logical_as_f64(b, logical))
    var info = 0
    try:
        info = lapack_sgesv(n, rhs_columns, a_ptr, pivots, b_ptr)
    except:
        a_ptr.free()
        b_ptr.free()
        pivots.free()
        return False
    if info != 0:
        a_ptr.free()
        b_ptr.free()
        pivots.free()
        if info > 0:
            raise Error("linalg.solve() singular matrix")
        return False
    for row in range(n):
        for col in range(rhs_columns):
            var out_index = row
            if not vector_result:
                out_index = row * rhs_columns + col
            set_logical_from_f64(result, out_index, Float64(b_ptr[row + col * n]))
    a_ptr.free()
    b_ptr.free()
    pivots.free()
    result.backend_code = BACKEND_ACCELERATE
    return True


def maybe_lapack_solve_f64(
    a: NativeArray, b: NativeArray, mut result: NativeArray
) raises -> Bool:
    if (
        a.dtype_code != DTYPE_FLOAT64
        or b.dtype_code != DTYPE_FLOAT64
        or result.dtype_code != DTYPE_FLOAT64
        or len(a.shape) != 2
        or a.shape[0] != a.shape[1]
    ):
        return False
    var n = a.shape[0]
    var rhs_columns = 1
    var vector_result = True
    if len(b.shape) == 2:
        rhs_columns = b.shape[1]
        vector_result = False
    var a_ptr = alloc[Float64](n * n)
    var b_ptr = alloc[Float64](n * rhs_columns)
    var pivots = alloc[Int32](n)
    for row in range(n):
        for col in range(n):
            a_ptr[row + col * n] = get_logical_as_f64(a, row * n + col)
    for row in range(n):
        for col in range(rhs_columns):
            var logical = row
            if not vector_result:
                logical = row * rhs_columns + col
            b_ptr[row + col * n] = get_logical_as_f64(b, logical)
    var info = 0
    try:
        info = lapack_dgesv(n, rhs_columns, a_ptr, pivots, b_ptr)
    except:
        a_ptr.free()
        b_ptr.free()
        pivots.free()
        return False
    if info != 0:
        a_ptr.free()
        b_ptr.free()
        pivots.free()
        if info > 0:
            raise Error("linalg.solve() singular matrix")
        return False
    for row in range(n):
        for col in range(rhs_columns):
            var out_index = row
            if not vector_result:
                out_index = row * rhs_columns + col
            set_logical_from_f64(result, out_index, b_ptr[row + col * n])
    a_ptr.free()
    b_ptr.free()
    pivots.free()
    result.backend_code = BACKEND_ACCELERATE
    return True


def maybe_lapack_inverse_f32(a: NativeArray, mut result: NativeArray) raises -> Bool:
    if (
        a.dtype_code != DTYPE_FLOAT32
        or result.dtype_code != DTYPE_FLOAT32
        or len(a.shape) != 2
        or a.shape[0] != a.shape[1]
    ):
        return False
    var n = a.shape[0]
    var a_ptr = alloc[Float32](n * n)
    var b_ptr = alloc[Float32](n * n)
    var pivots = alloc[Int32](n)
    for row in range(n):
        for col in range(n):
            a_ptr[row + col * n] = Float32(
                get_logical_as_f64(a, row * n + col)
            )
            if row == col:
                b_ptr[row + col * n] = 1.0
            else:
                b_ptr[row + col * n] = 0.0
    var info = 0
    try:
        info = lapack_sgesv(n, n, a_ptr, pivots, b_ptr)
    except:
        a_ptr.free()
        b_ptr.free()
        pivots.free()
        return False
    if info != 0:
        a_ptr.free()
        b_ptr.free()
        pivots.free()
        if info > 0:
            raise Error("linalg.inv() singular matrix")
        return False
    for row in range(n):
        for col in range(n):
            set_logical_from_f64(
                result, row * n + col, Float64(b_ptr[row + col * n])
            )
    a_ptr.free()
    b_ptr.free()
    pivots.free()
    result.backend_code = BACKEND_ACCELERATE
    return True


def maybe_lapack_inverse_f64(a: NativeArray, mut result: NativeArray) raises -> Bool:
    if (
        a.dtype_code != DTYPE_FLOAT64
        or result.dtype_code != DTYPE_FLOAT64
        or len(a.shape) != 2
        or a.shape[0] != a.shape[1]
    ):
        return False
    var n = a.shape[0]
    var a_ptr = alloc[Float64](n * n)
    var b_ptr = alloc[Float64](n * n)
    var pivots = alloc[Int32](n)
    for row in range(n):
        for col in range(n):
            a_ptr[row + col * n] = get_logical_as_f64(a, row * n + col)
            if row == col:
                b_ptr[row + col * n] = 1.0
            else:
                b_ptr[row + col * n] = 0.0
    var info = 0
    try:
        info = lapack_dgesv(n, n, a_ptr, pivots, b_ptr)
    except:
        a_ptr.free()
        b_ptr.free()
        pivots.free()
        return False
    if info != 0:
        a_ptr.free()
        b_ptr.free()
        pivots.free()
        if info > 0:
            raise Error("linalg.inv() singular matrix")
        return False
    for row in range(n):
        for col in range(n):
            set_logical_from_f64(result, row * n + col, b_ptr[row + col * n])
    a_ptr.free()
    b_ptr.free()
    pivots.free()
    result.backend_code = BACKEND_ACCELERATE
    return True


def lapack_pivot_sign(pivots: UnsafePointer[Int32, MutExternalOrigin], n: Int) -> Float64:
    var sign = 1.0
    for i in range(n):
        if Int(pivots[i]) != i + 1:
            sign = -sign
    return sign


def maybe_lapack_det_f32(a: NativeArray, mut result: NativeArray) raises -> Bool:
    if (
        a.dtype_code != DTYPE_FLOAT32
        or result.dtype_code != DTYPE_FLOAT32
        or len(a.shape) != 2
        or a.shape[0] != a.shape[1]
    ):
        return False
    var n = a.shape[0]
    var a_ptr = alloc[Float32](n * n)
    var pivots = alloc[Int32](n)
    for row in range(n):
        for col in range(n):
            a_ptr[row + col * n] = Float32(
                get_logical_as_f64(a, row * n + col)
            )
    var info = 0
    try:
        info = lapack_sgetrf(n, a_ptr, pivots)
    except:
        a_ptr.free()
        pivots.free()
        return False
    if info < 0:
        a_ptr.free()
        pivots.free()
        return False
    var det = lapack_pivot_sign(pivots, n)
    if info > 0:
        det = 0.0
    else:
        for i in range(n):
            det *= Float64(a_ptr[i + i * n])
    set_logical_from_f64(result, 0, det)
    a_ptr.free()
    pivots.free()
    result.backend_code = BACKEND_ACCELERATE
    return True


def maybe_lapack_det_f64(a: NativeArray, mut result: NativeArray) raises -> Bool:
    if (
        a.dtype_code != DTYPE_FLOAT64
        or result.dtype_code != DTYPE_FLOAT64
        or len(a.shape) != 2
        or a.shape[0] != a.shape[1]
    ):
        return False
    var n = a.shape[0]
    var a_ptr = alloc[Float64](n * n)
    var pivots = alloc[Int32](n)
    for row in range(n):
        for col in range(n):
            a_ptr[row + col * n] = get_logical_as_f64(a, row * n + col)
    var info = 0
    try:
        info = lapack_dgetrf(n, a_ptr, pivots)
    except:
        a_ptr.free()
        pivots.free()
        return False
    if info < 0:
        a_ptr.free()
        pivots.free()
        return False
    var det = lapack_pivot_sign(pivots, n)
    if info > 0:
        det = 0.0
    else:
        for i in range(n):
            det *= a_ptr[i + i * n]
    set_logical_from_f64(result, 0, det)
    a_ptr.free()
    pivots.free()
    result.backend_code = BACKEND_ACCELERATE
    return True


def load_square_matrix_f64(src: NativeArray) raises -> List[Float64]:
    if len(src.shape) != 2 or src.shape[0] != src.shape[1]:
        raise Error("linalg operation requires a square rank-2 matrix")
    var n = src.shape[0]
    var out = List[Float64]()
    for i in range(n * n):
        out.append(get_logical_as_f64(src, i))
    return out^


def make_lu_pivots(n: Int) -> List[Int]:
    var pivots = List[Int]()
    for i in range(n):
        pivots.append(i)
    return pivots^


def swap_lu_rows(mut lu: List[Float64], n: Int, lhs: Int, rhs: Int):
    if lhs == rhs:
        return
    for col in range(n):
        var lhs_index = lhs * n + col
        var rhs_index = rhs * n + col
        var tmp = lu[lhs_index]
        lu[lhs_index] = lu[rhs_index]
        lu[rhs_index] = tmp


def swap_rhs_rows(mut rhs: List[Float64], columns: Int, lhs: Int, rhs_row: Int):
    if lhs == rhs_row:
        return
    for col in range(columns):
        var lhs_index = lhs * columns + col
        var rhs_index = rhs_row * columns + col
        var tmp = rhs[lhs_index]
        rhs[lhs_index] = rhs[rhs_index]
        rhs[rhs_index] = tmp


def lu_decompose_partial_pivot(
    mut lu: List[Float64], mut pivots: List[Int], n: Int
) raises -> Int:
    var sign = 1
    for k in range(n):
        var pivot = k
        var max_abs = abs_f64(lu[k * n + k])
        for row in range(k + 1, n):
            var value_abs = abs_f64(lu[row * n + k])
            if value_abs > max_abs:
                max_abs = value_abs
                pivot = row
        if max_abs == 0.0:
            return 0
        pivots[k] = pivot
        if pivot != k:
            swap_lu_rows(lu, n, k, pivot)
            sign = -sign
        var pivot_value = lu[k * n + k]
        for row in range(k + 1, n):
            var row_base = row * n
            lu[row_base + k] = lu[row_base + k] / pivot_value
            var factor = lu[row_base + k]
            for col in range(k + 1, n):
                lu[row_base + col] -= factor * lu[k * n + col]
    return sign


def solve_lu_factor_into(
    lu: List[Float64],
    pivots: List[Int],
    n: Int,
    mut rhs: List[Float64],
    rhs_columns: Int,
    mut result: NativeArray,
    vector_result: Bool,
) raises:
    for row in range(n):
        swap_rhs_rows(rhs, rhs_columns, row, pivots[row])
    for row in range(n):
        for col in range(rhs_columns):
            var value = rhs[row * rhs_columns + col]
            for k in range(row):
                value -= lu[row * n + k] * rhs[k * rhs_columns + col]
            rhs[row * rhs_columns + col] = value
    for row in range(n - 1, -1, -1):
        for col in range(rhs_columns):
            var value = rhs[row * rhs_columns + col]
            for k in range(row + 1, n):
                value -= lu[row * n + k] * rhs[k * rhs_columns + col]
            rhs[row * rhs_columns + col] = value / lu[row * n + row]
    for row in range(n):
        for col in range(rhs_columns):
            var out_index = row * rhs_columns + col
            if vector_result:
                out_index = row
            set_logical_from_f64(
                result, out_index, rhs[row * rhs_columns + col]
            )


def lu_solve_into(
    a: NativeArray, b: NativeArray, mut result: NativeArray
) raises:
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise Error(
            "linalg.solve() requires a square rank-2 coefficient matrix"
        )
    var n = a.shape[0]
    var rhs_columns = 1
    var vector_result = True
    if len(b.shape) == 1:
        if b.shape[0] != n:
            raise Error("linalg.solve() right-hand side shape mismatch")
    elif len(b.shape) == 2:
        if b.shape[0] != n:
            raise Error("linalg.solve() right-hand side shape mismatch")
        rhs_columns = b.shape[1]
        vector_result = False
    else:
        raise Error("linalg.solve() right-hand side must be rank 1 or rank 2")
    comptime if CompilationTarget.is_macos():
        if maybe_lapack_solve_f32(a, b, result):
            return
        if maybe_lapack_solve_f64(a, b, result):
            return
    var lu = load_square_matrix_f64(a)
    var pivots = make_lu_pivots(n)
    if lu_decompose_partial_pivot(lu, pivots, n) == 0:
        raise Error("linalg.solve() singular matrix")
    var rhs_values = List[Float64]()
    for row in range(n):
        for col in range(rhs_columns):
            var logical = row
            if len(b.shape) == 2:
                logical = row * rhs_columns + col
            rhs_values.append(get_logical_as_f64(b, logical))
    solve_lu_factor_into(
        lu, pivots, n, rhs_values, rhs_columns, result, vector_result
    )


def lu_inverse_into(a: NativeArray, mut result: NativeArray) raises:
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise Error("linalg.inv() requires a square rank-2 matrix")
    comptime if CompilationTarget.is_macos():
        if maybe_lapack_inverse_f32(a, result):
            return
        if maybe_lapack_inverse_f64(a, result):
            return
    var n = a.shape[0]
    var lu = load_square_matrix_f64(a)
    var pivots = make_lu_pivots(n)
    if lu_decompose_partial_pivot(lu, pivots, n) == 0:
        raise Error("linalg.inv() singular matrix")
    var rhs_values = List[Float64]()
    for row in range(n):
        for col in range(n):
            if row == col:
                rhs_values.append(1.0)
            else:
                rhs_values.append(0.0)
    solve_lu_factor_into(lu, pivots, n, rhs_values, n, result, False)


def lu_det(a: NativeArray) raises -> Float64:
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise Error("linalg.det() requires a square rank-2 matrix")
    var n = a.shape[0]
    var lu = load_square_matrix_f64(a)
    var pivots = make_lu_pivots(n)
    var sign = lu_decompose_partial_pivot(lu, pivots, n)
    if sign == 0:
        return 0.0
    var det = Float64(sign)
    for i in range(n):
        det *= lu[i * n + i]
    return det


def lu_det_into(a: NativeArray, mut result: NativeArray) raises:
    comptime if CompilationTarget.is_macos():
        if maybe_lapack_det_f32(a, result):
            return
        if maybe_lapack_det_f64(a, result):
            return
    set_logical_from_f64(result, 0, lu_det(a))
