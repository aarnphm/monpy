from std.math import cos, exp, isinf, isnan, log, sin
from std.sys import CompilationTarget, simd_width_of

from layout import Layout, LayoutTensor
from native_accelerate import (
    call_vv_f32,
    cblas_dgemm_row_major,
    cblas_sgemm_row_major,
)
from native_types import (
    BACKEND_ACCELERATE,
    BACKEND_FUSED,
    BACKEND_GENERIC,
    BACKEND_LAYOUT_TENSOR,
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
    dst.used_layout_tensor = False
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
        if maybe_matmul_f32_small(lhs, rhs, result, m, n, k_lhs):
            return True
        comptime if CompilationTarget.is_macos():
            # Keep the Accelerate fast path local to monpy's row-major storage.
            # The Modular wrapper ultimately reaches this cblas_sgemm shape,
            # but the direct call avoids constructing TileTensor metadata for
            # every tiny Python-facing matmul.
            cblas_sgemm_row_major(
                m,
                n,
                k_lhs,
                contiguous_f32_ptr(result),
                contiguous_f32_ptr(lhs),
                contiguous_f32_ptr(rhs),
            )
            result.backend_code = BACKEND_ACCELERATE
            return True
    if (
        lhs.dtype_code == DTYPE_FLOAT64
        and rhs.dtype_code == DTYPE_FLOAT64
        and result.dtype_code == DTYPE_FLOAT64
    ):
        comptime if CompilationTarget.is_macos():
            # Modular's public wrapper currently covers f32. Mirror the same
            # row-major cblas call shape for f64 so `float64 @ float64` gets the
            # CPU BLAS backend instead of falling back to the scalar triple-loop.
            cblas_dgemm_row_major(
                m,
                n,
                k_lhs,
                contiguous_f64_ptr(result),
                contiguous_f64_ptr(lhs),
                contiguous_f64_ptr(rhs),
            )
            result.backend_code = BACKEND_ACCELERATE
            return True
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
        result.backend_code = BACKEND_LAYOUT_TENSOR
        return True
    return False
