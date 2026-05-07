from std.math import (
    abs as math_abs,
    acos,
    asin,
    atan,
    atan2,
    cbrt,
    ceil as math_ceil,
    copysign,
    cos,
    cosh,
    exp,
    exp2,
    expm1,
    floor as math_floor,
    hypot,
    isinf,
    isnan,
    log,
    log10,
    log1p,
    log2,
    nan,
    round as math_round,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    trunc as math_trunc,
)
from std.memory.unsafe_pointer import alloc
from std.sys import CompilationTarget, simd_width_of

from accelerate import (
    call_vv_f32,
    call_vv_f64,
    cblas_dgemm_row_major_ld,
    cblas_dgemv_row_major_ld,
    cblas_sgemm_row_major_ld,
    cblas_sgemv_row_major_ld,
    lapack_dgesv,
    lapack_dgetrf,
    lapack_sgesv,
    lapack_sgetrf,
)
from domain import (
    BACKEND_ACCELERATE,
    BACKEND_FUSED,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    OP_ADD,
    OP_ARCTAN2,
    OP_COPYSIGN,
    OP_DIV,
    OP_FLOOR_DIV,
    OP_FMAX,
    OP_FMIN,
    OP_HYPOT,
    OP_MAXIMUM,
    OP_MINIMUM,
    OP_MOD,
    OP_MUL,
    OP_POWER,
    OP_SUB,
    REDUCE_ALL,
    REDUCE_ANY,
    REDUCE_ARGMIN,
    REDUCE_MEAN,
    REDUCE_MAX,
    REDUCE_MIN,
    REDUCE_PROD,
    REDUCE_SUM,
    UNARY_ABS,
    UNARY_ARCCOS,
    UNARY_ARCSIN,
    UNARY_ARCTAN,
    UNARY_CBRT,
    UNARY_CEIL,
    UNARY_COS,
    UNARY_COSH,
    UNARY_DEG2RAD,
    UNARY_EXP,
    UNARY_EXP2,
    UNARY_EXPM1,
    UNARY_FLOOR,
    UNARY_LOG,
    UNARY_LOG10,
    UNARY_LOG1P,
    UNARY_LOG2,
    UNARY_LOGICAL_NOT,
    UNARY_NEGATE,
    UNARY_POSITIVE,
    UNARY_RAD2DEG,
    UNARY_RECIPROCAL,
    UNARY_RINT,
    UNARY_SIGN,
    UNARY_SIN,
    UNARY_SINH,
    UNARY_SQRT,
    UNARY_SQUARE,
    UNARY_TAN,
    UNARY_TANH,
    UNARY_TRUNC,
)
from array import (
    Array,
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


def apply_binary_f64(lhs: Float64, rhs: Float64, op: Int) raises -> Float64:
    if op == OP_ADD:
        return lhs + rhs
    if op == OP_SUB:
        return lhs - rhs
    if op == OP_MUL:
        return lhs * rhs
    if op == OP_DIV:
        return lhs / rhs
    if op == OP_FLOOR_DIV:
        # numpy floor_divide on float: floor(a / b); on int it's // .
        # We always operate in f64 here, so floor(quotient).
        return math_floor(lhs / rhs)
    if op == OP_MOD:
        # numpy `mod` (a - floor(a/b)*b); matches python `%` for floats.
        var q = math_floor(lhs / rhs)
        return lhs - q * rhs
    if op == OP_POWER:
        # f64 round-trip: pow via SIMD<f64, 1>.
        var v = SIMD[DType.float64, 1](lhs).__pow__(SIMD[DType.float64, 1](rhs))
        return v[0]
    if op == OP_MAXIMUM:
        # numpy `maximum` propagates NaN: if either is NaN, result is NaN.
        if isnan(lhs) or isnan(rhs):
            return nan[DType.float64]()
        return lhs if lhs > rhs else rhs
    if op == OP_MINIMUM:
        if isnan(lhs) or isnan(rhs):
            return nan[DType.float64]()
        return lhs if lhs < rhs else rhs
    if op == OP_FMAX:
        # NaN-aware: NaN treated as missing.
        if isnan(lhs):
            return rhs
        if isnan(rhs):
            return lhs
        return lhs if lhs > rhs else rhs
    if op == OP_FMIN:
        if isnan(lhs):
            return rhs
        if isnan(rhs):
            return lhs
        return lhs if lhs < rhs else rhs
    if op == OP_ARCTAN2:
        return atan2(SIMD[DType.float64, 1](lhs), SIMD[DType.float64, 1](rhs))[0]
    if op == OP_HYPOT:
        return hypot(SIMD[DType.float64, 1](lhs), SIMD[DType.float64, 1](rhs))[0]
    if op == OP_COPYSIGN:
        return copysign(SIMD[DType.float64, 1](lhs), SIMD[DType.float64, 1](rhs))[0]
    raise Error("unknown binary op")


def apply_unary_f64(value: Float64, op: Int) raises -> Float64:
    if op == UNARY_SIN:
        return sin(value)
    if op == UNARY_COS:
        return cos(value)
    if op == UNARY_EXP:
        return exp(value)
    if op == UNARY_LOG:
        if isnan(value):
            return value
        if isinf(value):
            if value < 0.0:
                return nan[DType.float64]()
            return value
        return log(value)
    if op == UNARY_TAN:
        return tan(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_ARCSIN:
        return asin(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_ARCCOS:
        return acos(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_ARCTAN:
        return atan(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_SINH:
        return sinh(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_COSH:
        return cosh(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_TANH:
        return tanh(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_LOG1P:
        return log1p(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_LOG2:
        return log2(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_LOG10:
        return log10(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_EXP2:
        return exp2(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_EXPM1:
        return expm1(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_SQRT:
        return sqrt(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_CBRT:
        return cbrt(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_DEG2RAD:
        return value * 0.017453292519943295  # pi/180
    if op == UNARY_RAD2DEG:
        return value * 57.29577951308232  # 180/pi
    if op == UNARY_RECIPROCAL:
        return 1.0 / value
    if op == UNARY_NEGATE:
        return -value
    if op == UNARY_POSITIVE:
        return value
    if op == UNARY_ABS:
        return -value if value < 0.0 else value
    if op == UNARY_SQUARE:
        return value * value
    if op == UNARY_SIGN:
        if isnan(value):
            return nan[DType.float64]()
        if value > 0.0:
            return 1.0
        if value < 0.0:
            return -1.0
        return 0.0
    if op == UNARY_FLOOR:
        return math_floor(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_CEIL:
        return math_ceil(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_TRUNC:
        return math_trunc(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_RINT:
        return math_round(SIMD[DType.float64, 1](value))[0]
    if op == UNARY_LOGICAL_NOT:
        return 1.0 if value == 0.0 else 0.0
    raise Error("unknown unary op")


def apply_binary_typed_vec[
    dtype: DType, width: Int
](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width], op: Int
) raises -> SIMD[dtype, width]:
    # Comptime-typed parametric variant of apply_binary_*_vec. Once a kernel
    # has dispatched on dtype at the runtime boundary, all subsequent SIMD
    # work is dtype-monomorphic — this function lets us write each kernel
    # body once instead of duplicating the f32 / f64 paths.
    # Float-only ops (FLOOR_DIV-with-floor, ARCTAN2, HYPOT, COPYSIGN, NaN-
    # propagation in MAXIMUM/MINIMUM/FMIN/FMAX) gate via `comptime if`.
    if op == OP_ADD:
        return lhs + rhs
    if op == OP_SUB:
        return lhs - rhs
    if op == OP_MUL:
        return lhs * rhs
    if op == OP_DIV:
        return lhs / rhs
    if op == OP_FLOOR_DIV:
        comptime if dtype.is_floating_point():
            return math_floor(lhs / rhs)
        else:
            return lhs // rhs
    if op == OP_MOD:
        return lhs % rhs
    if op == OP_POWER:
        return lhs.__pow__(rhs)
    if op == OP_MAXIMUM:
        comptime if dtype.is_floating_point():
            # numpy `maximum` propagates NaN. Mojo's MLIR-typed `pop.cmp` is
            # ordered (returns False when either operand is NaN), so we detect
            # NaN explicitly via std.utils.numerics.isnan.
            var lhs_nan = isnan(lhs)
            var rhs_nan = isnan(rhs)
            var any_nan = lhs_nan | rhs_nan
            var bigger = lhs.gt(rhs).select(lhs, rhs)
            return any_nan.select(SIMD[dtype, width](nan[dtype]()), bigger)
        else:
            return lhs.gt(rhs).select(lhs, rhs)
    if op == OP_MINIMUM:
        comptime if dtype.is_floating_point():
            var lhs_nan = isnan(lhs)
            var rhs_nan = isnan(rhs)
            var any_nan = lhs_nan | rhs_nan
            var smaller = lhs.lt(rhs).select(lhs, rhs)
            return any_nan.select(SIMD[dtype, width](nan[dtype]()), smaller)
        else:
            return lhs.lt(rhs).select(lhs, rhs)
    if op == OP_FMAX:
        comptime if dtype.is_floating_point():
            var lhs_nan = isnan(lhs)
            var rhs_nan = isnan(rhs)
            var bigger = lhs.gt(rhs).select(lhs, rhs)
            var picked = lhs_nan.select(rhs, bigger)
            picked = rhs_nan.select(lhs, picked)
            return picked
        else:
            return lhs.gt(rhs).select(lhs, rhs)
    if op == OP_FMIN:
        comptime if dtype.is_floating_point():
            var lhs_nan = isnan(lhs)
            var rhs_nan = isnan(rhs)
            var smaller = lhs.lt(rhs).select(lhs, rhs)
            var picked = lhs_nan.select(rhs, smaller)
            picked = rhs_nan.select(lhs, picked)
            return picked
        else:
            return lhs.lt(rhs).select(lhs, rhs)
    if op == OP_ARCTAN2:
        comptime if dtype.is_floating_point():
            return atan2(lhs, rhs)
        else:
            raise Error("arctan2 requires floating-point dtype")
    if op == OP_HYPOT:
        comptime if dtype.is_floating_point():
            return hypot(lhs, rhs)
        else:
            raise Error("hypot requires floating-point dtype")
    if op == OP_COPYSIGN:
        comptime if dtype.is_floating_point():
            return copysign(lhs, rhs)
        else:
            raise Error("copysign requires floating-point dtype")
    raise Error("unknown binary op")


def binary_same_shape_contig_typed[
    dtype: DType
](
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    op: Int,
) raises:
    # Single typed kernel for the contig+contig→contig binary case, used by
    # both f32 and f64 callers. Replaces the duplicated f32 / f64 branches in
    # `maybe_binary_same_shape_contiguous`. SIMD width derives from `dtype`
    # at comptime.
    comptime width = simd_width_of[dtype]()
    var i = 0
    while i + width <= size:
        out_ptr.store(
            i,
            apply_binary_typed_vec[dtype, width](
                lhs_ptr.load[width=width](i),
                rhs_ptr.load[width=width](i),
                op,
            ),
        )
        i += width
    while i < size:
        out_ptr[i] = apply_binary_typed_vec[dtype, 1](
            SIMD[dtype, 1](lhs_ptr[i]), SIMD[dtype, 1](rhs_ptr[i]), op
        )[0]
        i += 1


def binary_scalar_contig_typed[
    dtype: DType
](
    array_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    scalar_value: Scalar[dtype],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    op: Int,
    scalar_on_left: Bool,
) raises:
    # Comptime-typed kernel for array⊕scalar (and scalar⊕array). Replaces the
    # f32 / f64 duplicate branches in `maybe_binary_scalar_contiguous`.
    comptime width = simd_width_of[dtype]()
    var scalar_vec = SIMD[dtype, width](scalar_value)
    var i = 0
    while i + width <= size:
        var array_vec = array_ptr.load[width=width](i)
        if scalar_on_left:
            out_ptr.store(
                i,
                apply_binary_typed_vec[dtype, width](
                    scalar_vec, array_vec, op
                ),
            )
        else:
            out_ptr.store(
                i,
                apply_binary_typed_vec[dtype, width](
                    array_vec, scalar_vec, op
                ),
            )
        i += width
    while i < size:
        var lhs_v = SIMD[dtype, 1](array_ptr[i])
        var rhs_v = SIMD[dtype, 1](scalar_value)
        if scalar_on_left:
            out_ptr[i] = apply_binary_typed_vec[dtype, 1](rhs_v, lhs_v, op)[0]
        else:
            out_ptr[i] = apply_binary_typed_vec[dtype, 1](lhs_v, rhs_v, op)[0]
        i += 1


def apply_unary_typed_vec[
    dtype: DType, width: Int
](value: SIMD[dtype, width], op: Int) raises -> SIMD[dtype, width] where dtype.is_floating_point():
    # Comptime-typed parametric variant of apply_unary_*_vec. Constrained
    # to floating-point dtypes since std.math sin/cos/exp/log require it.
    if op == UNARY_SIN:
        return sin(value)
    if op == UNARY_COS:
        return cos(value)
    if op == UNARY_EXP:
        return exp(value)
    if op == UNARY_LOG:
        return log(value)
    if op == UNARY_TAN:
        return tan(value)
    if op == UNARY_ARCSIN:
        return asin(value)
    if op == UNARY_ARCCOS:
        return acos(value)
    if op == UNARY_ARCTAN:
        return atan(value)
    if op == UNARY_SINH:
        return sinh(value)
    if op == UNARY_COSH:
        return cosh(value)
    if op == UNARY_TANH:
        return tanh(value)
    if op == UNARY_LOG1P:
        return log1p(value)
    if op == UNARY_LOG2:
        return log2(value)
    if op == UNARY_LOG10:
        return log10(value)
    if op == UNARY_EXP2:
        return exp2(value)
    if op == UNARY_EXPM1:
        return expm1(value)
    if op == UNARY_SQRT:
        return sqrt(value)
    if op == UNARY_CBRT:
        return cbrt(value)
    if op == UNARY_DEG2RAD:
        return value * SIMD[dtype, width](0.017453292519943295)
    if op == UNARY_RAD2DEG:
        return value * SIMD[dtype, width](57.29577951308232)
    if op == UNARY_RECIPROCAL:
        return SIMD[dtype, width](1.0) / value
    if op == UNARY_NEGATE:
        return -value
    if op == UNARY_POSITIVE:
        return value
    if op == UNARY_ABS:
        var neg = -value
        return value.lt(SIMD[dtype, width](0)).select(neg, value)
    if op == UNARY_SQUARE:
        return value * value
    if op == UNARY_SIGN:
        var pos = value.gt(SIMD[dtype, width](0))
        var neg = value.lt(SIMD[dtype, width](0))
        var nan_mask = isnan(value)
        var s = pos.select(SIMD[dtype, width](1), SIMD[dtype, width](0))
        s = neg.select(SIMD[dtype, width](-1), s)
        return nan_mask.select(SIMD[dtype, width](nan[dtype]()), s)
    if op == UNARY_FLOOR:
        return math_floor(value)
    if op == UNARY_CEIL:
        return math_ceil(value)
    if op == UNARY_TRUNC:
        return math_trunc(value)
    if op == UNARY_RINT:
        return math_round(value)
    if op == UNARY_LOGICAL_NOT:
        return value.eq(SIMD[dtype, width](0)).select(
            SIMD[dtype, width](1), SIMD[dtype, width](0)
        )
    raise Error("unknown unary op")


def strided_binary_walk_typed[
    dtype: DType
](
    lhs: Array,
    rhs: Array,
    mut result: Array,
    inner_kind: Int,
    inner_size: Int,
    inner_lhs_stride: Int,
    inner_rhs_stride: Int,
    inner_result_stride: Int,
    outer_shape: List[Int],
    outer_lhs_stride: List[Int],
    outer_rhs_stride: List[Int],
    outer_result_stride: List[Int],
    outer_lhs_carry: List[Int],
    outer_rhs_carry: List[Int],
    outer_result_carry: List[Int],
    op: Int,
) raises:
    # Comptime-typed body for `maybe_binary_same_shape_strided`. Combines the
    # outer coord-stack walker (axis-iteration with no divmod) with the four
    # inner-loop kinds picked by `pick_inner_axis_for_strided_binary`.
    # Caller must verify dtype_code matches `dtype`.
    var lhs_data = lhs.data.bitcast[Scalar[dtype]]()
    var rhs_data = rhs.data.bitcast[Scalar[dtype]]()
    var result_data = result.data.bitcast[Scalar[dtype]]()
    comptime width = simd_width_of[dtype]()
    var outer_ndim = len(outer_shape)
    var coords = List[Int]()
    for _ in range(outer_ndim):
        coords.append(0)
    var lhs_offset = lhs.offset_elems
    var rhs_offset = rhs.offset_elems
    var result_offset = result.offset_elems
    while True:
        if inner_kind == 1:
            var lp = lhs_data + lhs_offset
            var rp = rhs_data + rhs_offset
            var op_ = result_data + result_offset
            var i = 0
            while i + width <= inner_size:
                op_.store(
                    i,
                    apply_binary_typed_vec[dtype, width](
                        lp.load[width=width](i),
                        rp.load[width=width](i),
                        op,
                    ),
                )
                i += width
            while i < inner_size:
                op_[i] = apply_binary_typed_vec[dtype, 1](
                    SIMD[dtype, 1](lp[i]), SIMD[dtype, 1](rp[i]), op
                )[0]
                i += 1
        elif inner_kind == 2:
            var lp = lhs_data + lhs_offset
            var rp = rhs_data + rhs_offset
            var i = 0
            while i + width <= inner_size:
                var ovec = apply_binary_typed_vec[dtype, width](
                    lp.load[width=width](i),
                    rp.load[width=width](i),
                    op,
                )
                comptime for k in range(width):
                    result_data[
                        result_offset + (i + k) * inner_result_stride
                    ] = ovec[k]
                i += width
            while i < inner_size:
                result_data[
                    result_offset + i * inner_result_stride
                ] = apply_binary_typed_vec[dtype, 1](
                    SIMD[dtype, 1](lp[i]), SIMD[dtype, 1](rp[i]), op
                )[0]
                i += 1
        elif inner_kind == 3:
            # lhs/rhs stride -1, result stride +/-1. Logical position i
            # maps to physical (offset - i). Loading [W] at physical
            # (offset - i - W + 1) gives us elements at logical positions
            # [i+W-1, i+W-2, ..., i] — reverse the SIMD vector to put
            # them back in logical order, then store contiguously (or
            # reversed, matching result's stride sign).
            var i = 0
            while i + width <= inner_size:
                var lvec = (
                    (lhs_data + lhs_offset - i - width + 1)
                    .load[width=width](0)
                    .reversed()
                )
                var rvec = (
                    (rhs_data + rhs_offset - i - width + 1)
                    .load[width=width](0)
                    .reversed()
                )
                var ovec = apply_binary_typed_vec[dtype, width](lvec, rvec, op)
                if inner_result_stride == 1:
                    (result_data + result_offset + i).store(0, ovec)
                else:
                    (result_data + result_offset - i - width + 1).store(
                        0, ovec.reversed()
                    )
                i += width
            while i < inner_size:
                result_data[
                    result_offset + i * inner_result_stride
                ] = apply_binary_typed_vec[dtype, 1](
                    SIMD[dtype, 1](lhs_data[lhs_offset - i]),
                    SIMD[dtype, 1](rhs_data[rhs_offset - i]),
                    op,
                )[0]
                i += 1
        else:
            for i in range(inner_size):
                result_data[
                    result_offset + i * inner_result_stride
                ] = apply_binary_typed_vec[dtype, 1](
                    SIMD[dtype, 1](
                        lhs_data[lhs_offset + i * inner_lhs_stride]
                    ),
                    SIMD[dtype, 1](
                        rhs_data[rhs_offset + i * inner_rhs_stride]
                    ),
                    op,
                )[0]
        if outer_ndim == 0:
            break
        var idx = outer_ndim - 1
        var done = False
        while idx >= 0:
            coords[idx] += 1
            if coords[idx] < outer_shape[idx]:
                lhs_offset += outer_lhs_stride[idx]
                rhs_offset += outer_rhs_stride[idx]
                result_offset += outer_result_stride[idx]
                break
            coords[idx] = 0
            lhs_offset -= outer_lhs_carry[idx]
            rhs_offset -= outer_rhs_carry[idx]
            result_offset -= outer_result_carry[idx]
            idx -= 1
            if idx < 0:
                done = True
        if done:
            break


def binary_row_broadcast_contig_typed[
    dtype: DType
](
    matrix_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    row_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rows: Int,
    cols: Int,
    op: Int,
    row_on_left: Bool,
) raises:
    # Comptime-typed kernel for matrix⊕row (and row⊕matrix). Broadcasts
    # `row` across each row of `matrix`. SIMD width derives from dtype.
    comptime width = simd_width_of[dtype]()
    for i in range(rows):
        var j = 0
        while j + width <= cols:
            var matrix_index = i * cols + j
            var matrix_vec = matrix_ptr.load[width=width](matrix_index)
            var row_vec = row_ptr.load[width=width](j)
            if row_on_left:
                out_ptr.store(
                    matrix_index,
                    apply_binary_typed_vec[dtype, width](
                        row_vec, matrix_vec, op
                    ),
                )
            else:
                out_ptr.store(
                    matrix_index,
                    apply_binary_typed_vec[dtype, width](
                        matrix_vec, row_vec, op
                    ),
                )
            j += width
        while j < cols:
            var matrix_index = i * cols + j
            var lhs_v = SIMD[dtype, 1](matrix_ptr[matrix_index])
            var rhs_v = SIMD[dtype, 1](row_ptr[j])
            if row_on_left:
                out_ptr[matrix_index] = apply_binary_typed_vec[dtype, 1](
                    rhs_v, lhs_v, op
                )[0]
            else:
                out_ptr[matrix_index] = apply_binary_typed_vec[dtype, 1](
                    lhs_v, rhs_v, op
                )[0]
            j += 1


def reduce_sum_typed[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int
) raises -> Float64 where dtype.is_floating_point():
    # Comptime-typed sum kernel using 4 parallel SIMD accumulators to break
    # the FADD latency dependency chain. With one accumulator the loop is
    # bound by FADD latency (~3 cycles on M1); four accumulators let the
    # pipeline issue ~4 FADDs per cycle, which roughly halves time at 1M+
    # elements vs a single-accumulator loop.
    comptime width = simd_width_of[dtype]()
    comptime block = width * 4
    var acc0 = SIMD[dtype, width](0)
    var acc1 = SIMD[dtype, width](0)
    var acc2 = SIMD[dtype, width](0)
    var acc3 = SIMD[dtype, width](0)
    var i = 0
    while i + block <= size:
        acc0 += src_ptr.load[width=width](i)
        acc1 += src_ptr.load[width=width](i + width)
        acc2 += src_ptr.load[width=width](i + 2 * width)
        acc3 += src_ptr.load[width=width](i + 3 * width)
        i += block
    var acc_vec = (acc0 + acc1) + (acc2 + acc3)
    while i + width <= size:
        acc_vec += src_ptr.load[width=width](i)
        i += width
    var acc = Float64(acc_vec.reduce_add()[0])
    while i < size:
        acc += Float64(src_ptr[i])
        i += 1
    return acc


def unary_contig_typed[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    op: Int,
) raises where dtype.is_floating_point():
    # Comptime-typed unary kernel. SIMD width derives from dtype.
    comptime width = simd_width_of[dtype]()
    var i = 0
    while i + width <= size:
        out_ptr.store(
            i,
            apply_unary_typed_vec[dtype, width](
                src_ptr.load[width=width](i), op
            ),
        )
        i += width
    while i < size:
        out_ptr[i] = apply_unary_typed_vec[dtype, 1](
            SIMD[dtype, 1](src_ptr[i]), op
        )[0]
        i += 1


def is_float_dtype(dtype_code: Int) -> Bool:
    return dtype_code == DTYPE_FLOAT32 or dtype_code == DTYPE_FLOAT64


def is_contiguous_float_array(array: Array) raises -> Bool:
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


def rank2_blas_layout(array: Array) raises -> Rank2BlasLayout:
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
    src: Array, mut result: Array, op: Int
) raises -> Bool:
    if not is_contiguous_float_array(src) or not is_contiguous_float_array(
        result
    ):
        return False
    if src.dtype_code == DTYPE_FLOAT32 and result.dtype_code == DTYPE_FLOAT32:
        comptime if CompilationTarget.is_macos():
            if maybe_unary_accelerate_f32(src, result, op):
                return True
        unary_contig_typed[DType.float32](
            contiguous_f32_ptr(src),
            contiguous_f32_ptr(result),
            src.size_value,
            op,
        )
        return True
    if op == UNARY_LOG:
        return False
    if src.dtype_code == DTYPE_FLOAT64 and result.dtype_code == DTYPE_FLOAT64:
        comptime if CompilationTarget.is_macos():
            if maybe_unary_accelerate_f64(src, result, op):
                return True
        unary_contig_typed[DType.float64](
            contiguous_f64_ptr(src),
            contiguous_f64_ptr(result),
            src.size_value,
            op,
        )
        return True
    return False


def maybe_unary_accelerate_f32(
    src: Array, mut result: Array, op: Int
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


def maybe_unary_accelerate_f64(
    src: Array, mut result: Array, op: Int
) raises -> Bool:
    var src_ptr = contiguous_f64_ptr(src)
    var out_ptr = contiguous_f64_ptr(result)
    if op == UNARY_SIN:
        call_vv_f64["vvsin"](out_ptr, src_ptr, src.size_value)
    elif op == UNARY_COS:
        call_vv_f64["vvcos"](out_ptr, src_ptr, src.size_value)
    elif op == UNARY_EXP:
        call_vv_f64["vvexp"](out_ptr, src_ptr, src.size_value)
    else:
        return False
    result.backend_code = BACKEND_ACCELERATE
    return True


def maybe_binary_same_shape_contiguous(
    lhs: Array, rhs: Array, mut result: Array, op: Int
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
        binary_same_shape_contig_typed[DType.float32](
            contiguous_f32_ptr(lhs),
            contiguous_f32_ptr(rhs),
            contiguous_f32_ptr(result),
            result.size_value,
            op,
        )
        return True
    if (
        lhs.dtype_code == DTYPE_FLOAT64
        and rhs.dtype_code == DTYPE_FLOAT64
        and result.dtype_code == DTYPE_FLOAT64
    ):
        binary_same_shape_contig_typed[DType.float64](
            contiguous_f64_ptr(lhs),
            contiguous_f64_ptr(rhs),
            contiguous_f64_ptr(result),
            result.size_value,
            op,
        )
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
    array: Array,
    scalar: Array,
    mut result: Array,
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
        binary_scalar_contig_typed[DType.float32](
            contiguous_f32_ptr(array),
            Float32(scalar_value),
            contiguous_f32_ptr(result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == DTYPE_FLOAT64 and result.dtype_code == DTYPE_FLOAT64:
        binary_scalar_contig_typed[DType.float64](
            contiguous_f64_ptr(array),
            scalar_value,
            contiguous_f64_ptr(result),
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
    if (
        not same_shape(array.shape, result.shape)
        or not is_contiguous_float_array(array)
        or not is_contiguous_float_array(result)
    ):
        return False
    if array.dtype_code == DTYPE_FLOAT32 and result.dtype_code == DTYPE_FLOAT32:
        binary_scalar_contig_typed[DType.float32](
            contiguous_f32_ptr(array),
            Float32(scalar_value),
            contiguous_f32_ptr(result),
            result.size_value,
            op,
            scalar_on_left,
        )
        return True
    if array.dtype_code == DTYPE_FLOAT64 and result.dtype_code == DTYPE_FLOAT64:
        binary_scalar_contig_typed[DType.float64](
            contiguous_f64_ptr(array),
            scalar_value,
            contiguous_f64_ptr(result),
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
        binary_row_broadcast_contig_typed[DType.float32](
            contiguous_f32_ptr(matrix),
            contiguous_f32_ptr(row),
            contiguous_f32_ptr(result),
            rows,
            cols,
            op,
            row_on_left,
        )
        return True
    if (
        matrix.dtype_code == DTYPE_FLOAT64
        and row.dtype_code == DTYPE_FLOAT64
        and result.dtype_code == DTYPE_FLOAT64
    ):
        binary_row_broadcast_contig_typed[DType.float64](
            contiguous_f64_ptr(matrix),
            contiguous_f64_ptr(row),
            contiguous_f64_ptr(result),
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
            set_contiguous_from_f64(
                result, matrix_index, apply_binary_f64(lhs, rhs, op)
            )
    return True


@fieldwise_init
struct StridedInnerChoice(ImplicitlyCopyable, Movable):
    var axis: Int
    var kind: Int  # 0=scalar, 1=full-SIMD, 2=SIMD-load+scatter


def pick_inner_axis_for_strided_binary(
    lhs: Array, rhs: Array, result: Array
) raises -> StridedInnerChoice:
    # Pick the inner axis for a same-shape strided binary walk.
    #   kind == 1: all three operands have stride +1 on `axis` -> full SIMD
    #   kind == 2: lhs and rhs have stride +1 -> SIMD load + scatter store
    #   kind == 3: lhs and rhs have stride -1, result has stride +1 (or -1) ->
    #              SIMD reversed-load + contiguous store. covers `[::-1]+[::-1]`.
    #   kind == 0: scalar walk (no |stride|==1 axis on inputs)
    # Prefers higher-`kind` choices, then the rightmost (innermost) axis among
    # equally-good candidates so iteration order remains C-natural.
    var ndim = len(lhs.shape)
    var inner_axis = ndim - 1
    var inner_kind = 0
    for axis in range(ndim - 1, -1, -1):
        var ls = lhs.strides[axis]
        var rs = rhs.strides[axis]
        var os = result.strides[axis]
        if ls == 1 and rs == 1:
            if os == 1:
                return StridedInnerChoice(axis, 1)
            if inner_kind < 2:
                inner_axis = axis
                inner_kind = 2
        elif ls == -1 and rs == -1 and (os == 1 or os == -1):
            # Reversed inputs with contig (or reversed-contig) output. Covers
            # the rank-1 `arr[::-1] + brr[::-1]` family without falling into
            # the scalar path.
            if inner_kind < 3:
                inner_axis = axis
                inner_kind = 3
    return StridedInnerChoice(inner_axis, inner_kind)


def maybe_binary_rank2_transposed_tile[
    dtype: DType
](lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool where dtype.is_floating_point():
    # Tile-transpose fast path for rank-2 binary ops where both inputs have
    # stride-1 axis 0 (the F-contig pattern that `arr.T` produces from a
    # c-contig `arr`) and the result is c-contig. Without this, the generic
    # strided walker picks kind=2 (SIMD-load + W-element scatter store)
    # which loses 4-8× to scattered stores stalling the write buffer.
    #
    # Strategy: process the array as 4-by-4 tiles. For each tile we do four
    # contiguous SIMD loads from each input along axis 0 (stride-1), apply
    # the binary op, transpose the resulting 4 SIMD vectors in registers,
    # and emit four contiguous SIMD stores to result. Both load and store
    # streams are contiguous; only the in-register shuffle costs.
    #
    # Caller must verify lhs/rhs/result dtype_code matches `dtype` before
    # calling. tile=4 works for any float dtype: f32 fits the natural NEON
    # width directly; f64 uses two NEON registers per "vector", trading
    # some throughput for the same scatter-avoiding pattern.
    if len(lhs.shape) != 2:
        return False
    if not same_shape(lhs.shape, rhs.shape) or not same_shape(
        lhs.shape, result.shape
    ):
        return False
    if lhs.strides[0] != 1 or rhs.strides[0] != 1:
        return False
    if not is_c_contiguous(result):
        return False
    var rows = lhs.shape[0]
    var cols = lhs.shape[1]
    if rows < 4 or cols < 4:
        return False
    var lhs_col_stride = lhs.strides[1]
    var rhs_col_stride = rhs.strides[1]
    var result_row_stride = result.strides[0]
    var lhs_data = lhs.data.bitcast[Scalar[dtype]]() + lhs.offset_elems
    var rhs_data = rhs.data.bitcast[Scalar[dtype]]() + rhs.offset_elems
    var result_data = result.data.bitcast[Scalar[dtype]]() + result.offset_elems
    comptime tile = 4
    var main_rows = rows - (rows % tile)
    var main_cols = cols - (cols % tile)
    var i = 0
    while i < main_rows:
        var j = 0
        while j < main_cols:
            var l0 = (lhs_data + i + j * lhs_col_stride).load[width=tile](0)
            var l1 = (lhs_data + i + (j + 1) * lhs_col_stride).load[width=tile](
                0
            )
            var l2 = (lhs_data + i + (j + 2) * lhs_col_stride).load[width=tile](
                0
            )
            var l3 = (lhs_data + i + (j + 3) * lhs_col_stride).load[width=tile](
                0
            )
            var r0 = (rhs_data + i + j * rhs_col_stride).load[width=tile](0)
            var r1 = (rhs_data + i + (j + 1) * rhs_col_stride).load[width=tile](
                0
            )
            var r2 = (rhs_data + i + (j + 2) * rhs_col_stride).load[width=tile](
                0
            )
            var r3 = (rhs_data + i + (j + 3) * rhs_col_stride).load[width=tile](
                0
            )
            var s0 = apply_binary_typed_vec[dtype, tile](l0, r0, op)
            var s1 = apply_binary_typed_vec[dtype, tile](l1, r1, op)
            var s2 = apply_binary_typed_vec[dtype, tile](l2, r2, op)
            var s3 = apply_binary_typed_vec[dtype, tile](l3, r3, op)
            # 4x4 in-register transpose. The two-step zip pattern:
            # step 1 zips paired SIMD vectors, step 2 zips the 64-bit lanes
            # of the result. Mojo's `shuffle[*mask]` lowers to NEON's
            # `vzip1`/`vzip2` (and the AVX equivalent on x86) directly.
            var u0 = s0.shuffle[0, 4, 1, 5](s1)
            var u1 = s0.shuffle[2, 6, 3, 7](s1)
            var u2 = s2.shuffle[0, 4, 1, 5](s3)
            var u3 = s2.shuffle[2, 6, 3, 7](s3)
            var t0 = u0.shuffle[0, 1, 4, 5](u2)
            var t1 = u0.shuffle[2, 3, 6, 7](u2)
            var t2 = u1.shuffle[0, 1, 4, 5](u3)
            var t3 = u1.shuffle[2, 3, 6, 7](u3)
            (result_data + i * result_row_stride + j).store(0, t0)
            (result_data + (i + 1) * result_row_stride + j).store(0, t1)
            (result_data + (i + 2) * result_row_stride + j).store(0, t2)
            (result_data + (i + 3) * result_row_stride + j).store(0, t3)
            j += tile
        # column tail: scalar walk for residual cols within the main rows
        while j < cols:
            comptime for k in range(tile):
                var lv = lhs_data[i + k + j * lhs_col_stride]
                var rv = rhs_data[i + k + j * rhs_col_stride]
                result_data[(i + k) * result_row_stride + j] = (
                    apply_binary_typed_vec[dtype, 1](
                        SIMD[dtype, 1](lv), SIMD[dtype, 1](rv), op
                    )[0]
                )
            j += 1
        i += tile
    # row tail: scalar walk for residual rows
    while i < rows:
        var j = 0
        while j < cols:
            var lv = lhs_data[i + j * lhs_col_stride]
            var rv = rhs_data[i + j * rhs_col_stride]
            result_data[i * result_row_stride + j] = apply_binary_typed_vec[
                dtype, 1
            ](SIMD[dtype, 1](lv), SIMD[dtype, 1](rv), op)[0]
            j += 1
        i += 1
    return True


def maybe_binary_same_shape_strided(
    lhs: Array, rhs: Array, mut result: Array, op: Int
) raises -> Bool:
    # General N-D same-shape strided walker. Walks `inner_axis` with SIMD when
    # possible (full or load-only) and walks the remaining axes with a coord
    # stack that uses incremental offset arithmetic (no divmod per element).
    # Subsumes the previous rank-1 and rank-2 special cases.
    if not same_shape(lhs.shape, rhs.shape) or not same_shape(
        lhs.shape, result.shape
    ):
        return False
    if lhs.dtype_code != rhs.dtype_code or rhs.dtype_code != result.dtype_code:
        return False
    if lhs.dtype_code != DTYPE_FLOAT32 and lhs.dtype_code != DTYPE_FLOAT64:
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
    if lhs.dtype_code == DTYPE_FLOAT32:
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
    if lhs.dtype_code == DTYPE_FLOAT64:
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
    return False


def maybe_binary_contiguous(
    lhs: Array, rhs: Array, mut result: Array, op: Int
) raises -> Bool:
    # Fast-path dispatch is intentionally shape-specific here. We want to be dumb.
    # The fallback below still handles dynamic-rank broadcasting, so every branch
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
    if (
        lhs.dtype_code == DTYPE_FLOAT32
        and rhs.dtype_code == DTYPE_FLOAT32
        and result.dtype_code == DTYPE_FLOAT32
    ):
        if maybe_binary_rank2_transposed_tile[DType.float32](
            lhs, rhs, result, op
        ):
            return True
    elif (
        lhs.dtype_code == DTYPE_FLOAT64
        and rhs.dtype_code == DTYPE_FLOAT64
        and result.dtype_code == DTYPE_FLOAT64
    ):
        if maybe_binary_rank2_transposed_tile[DType.float64](
            lhs, rhs, result, op
        ):
            return True
    if maybe_binary_same_shape_strided(lhs, rhs, result, op):
        return True
    return False


def maybe_sin_add_mul_contiguous(
    lhs: Array,
    rhs: Array,
    scalar_value: Float64,
    mut result: Array,
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
    src: Array, mut result: Array, op: Int
) raises -> Bool:
    if not is_contiguous_float_array(src):
        return False
    if op == REDUCE_SUM or op == REDUCE_MEAN:
        var acc: Float64
        if src.dtype_code == DTYPE_FLOAT32:
            acc = reduce_sum_typed[DType.float32](
                contiguous_f32_ptr(src), src.size_value
            )
        elif src.dtype_code == DTYPE_FLOAT64:
            acc = reduce_sum_typed[DType.float64](
                contiguous_f64_ptr(src), src.size_value
            )
        else:
            return False
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


def maybe_argmax_contiguous(src: Array, mut result: Array) raises -> Bool:
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
    lhs: Array,
    rhs: Array,
    mut result: Array,
    m: Int,
    n: Int,
    k_lhs: Int,
) raises -> Bool:
    comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
        if maybe_matmul_vector_accelerate(lhs, rhs, result, m, n, k_lhs):
            return True
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
        comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
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
        comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
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


def maybe_matmul_vector_accelerate(
    lhs: Array,
    rhs: Array,
    mut result: Array,
    m: Int,
    n: Int,
    k_lhs: Int,
) raises -> Bool:
    var lhs_ndim = len(lhs.shape)
    var rhs_ndim = len(rhs.shape)
    if (
        lhs_ndim == 2
        and rhs_ndim == 1
        and is_contiguous_float_array(rhs)
        and is_contiguous_float_array(result)
    ):
        var lhs_layout = rank2_blas_layout(lhs)
        if not lhs_layout.can_use:
            return False
        var rows = m
        var cols = k_lhs
        if lhs_layout.transpose:
            rows = k_lhs
            cols = m
        if (
            lhs.dtype_code == DTYPE_FLOAT32
            and rhs.dtype_code == DTYPE_FLOAT32
            and result.dtype_code == DTYPE_FLOAT32
        ):
            cblas_sgemv_row_major_ld(
                rows,
                cols,
                contiguous_f32_ptr(result),
                contiguous_f32_ptr(lhs),
                contiguous_f32_ptr(rhs),
                lhs_layout.transpose,
                lhs_layout.leading_dim,
            )
            result.backend_code = BACKEND_ACCELERATE
            return True
        if (
            lhs.dtype_code == DTYPE_FLOAT64
            and rhs.dtype_code == DTYPE_FLOAT64
            and result.dtype_code == DTYPE_FLOAT64
        ):
            cblas_dgemv_row_major_ld(
                rows,
                cols,
                contiguous_f64_ptr(result),
                contiguous_f64_ptr(lhs),
                contiguous_f64_ptr(rhs),
                lhs_layout.transpose,
                lhs_layout.leading_dim,
            )
            result.backend_code = BACKEND_ACCELERATE
            return True
    if (
        lhs_ndim == 1
        and rhs_ndim == 2
        and is_contiguous_float_array(lhs)
        and is_contiguous_float_array(result)
    ):
        var rhs_layout = rank2_blas_layout(rhs)
        if not rhs_layout.can_use:
            return False
        var rows = k_lhs
        var cols = n
        var transpose_rhs = True
        if rhs_layout.transpose:
            rows = n
            cols = k_lhs
            transpose_rhs = False
        if (
            lhs.dtype_code == DTYPE_FLOAT32
            and rhs.dtype_code == DTYPE_FLOAT32
            and result.dtype_code == DTYPE_FLOAT32
        ):
            cblas_sgemv_row_major_ld(
                rows,
                cols,
                contiguous_f32_ptr(result),
                contiguous_f32_ptr(rhs),
                contiguous_f32_ptr(lhs),
                transpose_rhs,
                rhs_layout.leading_dim,
            )
            result.backend_code = BACKEND_ACCELERATE
            return True
        if (
            lhs.dtype_code == DTYPE_FLOAT64
            and rhs.dtype_code == DTYPE_FLOAT64
            and result.dtype_code == DTYPE_FLOAT64
        ):
            cblas_dgemv_row_major_ld(
                rows,
                cols,
                contiguous_f64_ptr(result),
                contiguous_f64_ptr(rhs),
                contiguous_f64_ptr(lhs),
                transpose_rhs,
                rhs_layout.leading_dim,
            )
            result.backend_code = BACKEND_ACCELERATE
            return True
    return False


def matmul_small_typed[
    dtype: DType
](
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    m: Int,
    n: Int,
    k_lhs: Int,
) raises where dtype.is_floating_point():
    # Comptime-typed small-N matmul. Splat each lhs scalar to a SIMD vector
    # then fma against a contiguous rhs row chunk; this beats cblas_sgemm /
    # cblas_dgemm dispatch overhead at N≤16 by skipping the BLAS frame.
    comptime width = simd_width_of[dtype]()
    for i in range(m):
        var j = 0
        while j + width <= n:
            var acc = SIMD[dtype, width](0)
            for k in range(k_lhs):
                acc += SIMD[dtype, width](
                    lhs_ptr[i * k_lhs + k]
                ) * rhs_ptr.load[width=width](k * n + j)
            out_ptr.store(i * n + j, acc)
            j += width
        while j < n:
            var total = Scalar[dtype](0)
            for k in range(k_lhs):
                total += lhs_ptr[i * k_lhs + k] * rhs_ptr[k * n + j]
            out_ptr[i * n + j] = total
            j += 1


def maybe_matmul_f32_small(
    lhs: Array,
    rhs: Array,
    mut result: Array,
    m: Int,
    n: Int,
    k_lhs: Int,
) raises -> Bool:
    # Thin dispatcher that delegates to the typed kernel. Caller has already
    # verified dtype f32; the typed instantiation is fully specialized at
    # compile time. Kept under the f32 name so existing callers don't need
    # to change; an `_f64` sibling can be added when matmul small-N for f64
    # becomes a hot path.
    if m > 16 or n > 16 or k_lhs > 16:
        return False
    matmul_small_typed[DType.float32](
        contiguous_f32_ptr(lhs),
        contiguous_f32_ptr(rhs),
        contiguous_f32_ptr(result),
        m,
        n,
        k_lhs,
    )
    return True


def abs_f64(value: Float64) -> Float64:
    if value < 0.0:
        return -value
    return value


def transpose_to_col_major_f32(
    src: Array,
    dst: UnsafePointer[Float32, MutExternalOrigin],
    n: Int,
) raises:
    # Copy `src` (rank-2 n×n) into a column-major buffer pointed to by `dst`.
    # Fast path for c-contiguous source skips the per-element `physical_offset`
    # divmod that `get_logical_as_f64` pays. At n=128 this is ~50 µs vs ~5 µs.
    if is_c_contiguous(src):
        var src_ptr = contiguous_f32_ptr(src)
        for row in range(n):
            for col in range(n):
                dst[row + col * n] = src_ptr[row * n + col]
        return
    for row in range(n):
        for col in range(n):
            dst[row + col * n] = Float32(get_logical_as_f64(src, row * n + col))


def transpose_to_col_major_f64(
    src: Array,
    dst: UnsafePointer[Float64, MutExternalOrigin],
    n: Int,
) raises:
    if is_c_contiguous(src):
        var src_ptr = contiguous_f64_ptr(src)
        for row in range(n):
            for col in range(n):
                dst[row + col * n] = src_ptr[row * n + col]
        return
    for row in range(n):
        for col in range(n):
            dst[row + col * n] = get_logical_as_f64(src, row * n + col)


def copy_rhs_to_col_major_f32(
    b: Array,
    dst: UnsafePointer[Float32, MutExternalOrigin],
    n: Int,
    rhs_columns: Int,
    vector_result: Bool,
) raises:
    if is_c_contiguous(b) and b.dtype_code == DTYPE_FLOAT32:
        var src_ptr = contiguous_f32_ptr(b)
        if vector_result:
            for row in range(n):
                dst[row] = src_ptr[row]
            return
        for row in range(n):
            for col in range(rhs_columns):
                dst[row + col * n] = src_ptr[row * rhs_columns + col]
        return
    for row in range(n):
        for col in range(rhs_columns):
            var logical = row
            if not vector_result:
                logical = row * rhs_columns + col
            dst[row + col * n] = Float32(get_logical_as_f64(b, logical))


def copy_rhs_to_col_major_f64(
    b: Array,
    dst: UnsafePointer[Float64, MutExternalOrigin],
    n: Int,
    rhs_columns: Int,
    vector_result: Bool,
) raises:
    if is_c_contiguous(b) and b.dtype_code == DTYPE_FLOAT64:
        var src_ptr = contiguous_f64_ptr(b)
        if vector_result:
            for row in range(n):
                dst[row] = src_ptr[row]
            return
        for row in range(n):
            for col in range(rhs_columns):
                dst[row + col * n] = src_ptr[row * rhs_columns + col]
        return
    for row in range(n):
        for col in range(rhs_columns):
            var logical = row
            if not vector_result:
                logical = row * rhs_columns + col
            dst[row + col * n] = get_logical_as_f64(b, logical)


def write_solve_result_f32(
    src: UnsafePointer[Float32, MutExternalOrigin],
    mut result: Array,
    n: Int,
    rhs_columns: Int,
    vector_result: Bool,
) raises:
    if is_c_contiguous(result) and result.dtype_code == DTYPE_FLOAT32:
        var dst = contiguous_f32_ptr(result)
        if vector_result:
            for row in range(n):
                dst[row] = src[row]
            return
        for row in range(n):
            for col in range(rhs_columns):
                dst[row * rhs_columns + col] = src[row + col * n]
        return
    for row in range(n):
        for col in range(rhs_columns):
            var out_index = row
            if not vector_result:
                out_index = row * rhs_columns + col
            set_logical_from_f64(result, out_index, Float64(src[row + col * n]))


def write_solve_result_f64(
    src: UnsafePointer[Float64, MutExternalOrigin],
    mut result: Array,
    n: Int,
    rhs_columns: Int,
    vector_result: Bool,
) raises:
    if is_c_contiguous(result) and result.dtype_code == DTYPE_FLOAT64:
        var dst = contiguous_f64_ptr(result)
        if vector_result:
            for row in range(n):
                dst[row] = src[row]
            return
        for row in range(n):
            for col in range(rhs_columns):
                dst[row * rhs_columns + col] = src[row + col * n]
        return
    for row in range(n):
        for col in range(rhs_columns):
            var out_index = row
            if not vector_result:
                out_index = row * rhs_columns + col
            set_logical_from_f64(result, out_index, src[row + col * n])


def maybe_lapack_solve_f32(
    a: Array, b: Array, mut result: Array
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
    transpose_to_col_major_f32(a, a_ptr, n)
    copy_rhs_to_col_major_f32(b, b_ptr, n, rhs_columns, vector_result)
    var info: Int
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
    write_solve_result_f32(b_ptr, result, n, rhs_columns, vector_result)
    a_ptr.free()
    b_ptr.free()
    pivots.free()
    result.backend_code = BACKEND_ACCELERATE
    return True


def maybe_lapack_solve_f64(
    a: Array, b: Array, mut result: Array
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
    transpose_to_col_major_f64(a, a_ptr, n)
    copy_rhs_to_col_major_f64(b, b_ptr, n, rhs_columns, vector_result)
    var info: Int
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
    write_solve_result_f64(b_ptr, result, n, rhs_columns, vector_result)
    a_ptr.free()
    b_ptr.free()
    pivots.free()
    result.backend_code = BACKEND_ACCELERATE
    return True


def maybe_lapack_inverse_f32(a: Array, mut result: Array) raises -> Bool:
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
    transpose_to_col_major_f32(a, a_ptr, n)
    for row in range(n):
        for col in range(n):
            if row == col:
                b_ptr[row + col * n] = 1.0
            else:
                b_ptr[row + col * n] = 0.0
    var info: Int
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
    write_solve_result_f32(b_ptr, result, n, n, False)
    a_ptr.free()
    b_ptr.free()
    pivots.free()
    result.backend_code = BACKEND_ACCELERATE
    return True


def maybe_lapack_inverse_f64(a: Array, mut result: Array) raises -> Bool:
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
    transpose_to_col_major_f64(a, a_ptr, n)
    for row in range(n):
        for col in range(n):
            if row == col:
                b_ptr[row + col * n] = 1.0
            else:
                b_ptr[row + col * n] = 0.0
    var info: Int
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
    write_solve_result_f64(b_ptr, result, n, n, False)
    a_ptr.free()
    b_ptr.free()
    pivots.free()
    result.backend_code = BACKEND_ACCELERATE
    return True


def lapack_pivot_sign(
    pivots: UnsafePointer[Int32, MutExternalOrigin], n: Int
) -> Float64:
    var sign = 1.0
    for i in range(n):
        if Int(pivots[i]) != i + 1:
            sign = -sign
    return sign


def maybe_lapack_det_f32(a: Array, mut result: Array) raises -> Bool:
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
    transpose_to_col_major_f32(a, a_ptr, n)
    try:
        var info = lapack_sgetrf(n, a_ptr, pivots)
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
    except:
        a_ptr.free()
        pivots.free()
        return False
    a_ptr.free()
    pivots.free()
    result.backend_code = BACKEND_ACCELERATE
    return True


def maybe_lapack_det_f64(a: Array, mut result: Array) raises -> Bool:
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
    transpose_to_col_major_f64(a, a_ptr, n)
    try:
        var info = lapack_dgetrf(n, a_ptr, pivots)
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
    except:
        a_ptr.free()
        pivots.free()
        return False
    a_ptr.free()
    pivots.free()
    result.backend_code = BACKEND_ACCELERATE
    return True


def load_square_matrix_f64(src: Array) raises -> List[Float64]:
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
    mut result: Array,
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


def lu_solve_into(a: Array, b: Array, mut result: Array) raises:
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
    comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
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


def lu_inverse_into(a: Array, mut result: Array) raises:
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise Error("linalg.inv() requires a square rank-2 matrix")
    comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
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


def lu_det(a: Array) raises -> Float64:
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


def lu_det_into(a: Array, mut result: Array) raises:
    comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
        if maybe_lapack_det_f32(a, result):
            return
        if maybe_lapack_det_f64(a, result):
            return
    set_logical_from_f64(result, 0, lu_det(a))
