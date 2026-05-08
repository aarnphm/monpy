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
    call_vdsp_binary_f32,
    call_vdsp_binary_f64,
    call_vdsp_binary_strided_f32,
    call_vdsp_binary_strided_f64,
    call_vv_f32,
    call_vv_f64,
    cblas_cgemm_row_major,
    cblas_dgemm_row_major_ld,
    cblas_dgemv_row_major_ld,
    cblas_sgemm_row_major_ld,
    cblas_sgemv_row_major_ld,
    cblas_zgemm_row_major,
    lapack_dgeev,
    lapack_dgelsd,
    lapack_dgeqrf,
    lapack_dgesdd,
    lapack_dgesv,
    lapack_dgetrf,
    lapack_dorgqr,
    lapack_dpotrf,
    lapack_dsyev,
    lapack_sgeev,
    lapack_sgelsd,
    lapack_sgeqrf,
    lapack_sgesdd,
    lapack_sgesv,
    lapack_sgetrf,
    lapack_sorgqr,
    lapack_spotrf,
    lapack_ssyev,
)
from domain import (
    ArrayDType,
    BackendKind,
    BinaryOp,
    ReduceOp,
    UnaryOp,
)
from array import (
    Array,
    as_broadcast_layout,
    as_layout,
    contiguous_as_f64,
    contiguous_ptr,
    get_physical,
    get_physical_c128_imag,
    get_physical_c128_real,
    get_physical_c64_imag,
    get_physical_c64_real,
    get_logical_as_f64,
    has_negative_strides,
    has_zero_strides,
    is_c_contiguous,
    is_linearly_addressable,
    item_size,
    physical_offset,
    same_shape,
    set_contiguous_from_f64,
    set_logical_from_f64,
    set_logical_from_i64,
    set_physical,
)
from cute.iter import LayoutIter, MultiLayoutIter
from cute.layout import Layout

from .apply_scalar import apply_binary_f64, apply_unary_f64
from .linalg_kernels import (
    abs_f64,
    copy_rhs_to_col_major,
    lapack_cholesky_into,
    lapack_eig_real_into,
    lapack_eigh_into,
    lapack_lstsq_into,
    lapack_pivot_sign,
    lapack_qr_r_only_into,
    lapack_qr_reduced_into,
    lapack_svd_into,
    load_square_matrix_f64,
    lu_decompose_partial_pivot,
    lu_det,
    lu_det_into,
    lu_inverse_into,
    lu_solve_into,
    make_lu_pivots,
    maybe_lapack_det,
    maybe_lapack_inverse,
    maybe_lapack_solve,
    solve_lu_factor_into,
    swap_lu_rows,
    swap_rhs_rows,
    transpose_to_col_major,
    transpose_to_col_major_rect,
    write_cholesky_lower,
    write_col_major_to_array,
    write_solve_result,
)
from .accelerate_dispatch import (
    maybe_binary_accelerate,
    maybe_binary_rank1_strided_accelerate,
    maybe_complex_binary_rank1_strided_accelerate,
    maybe_unary_accelerate,
)
from .matmul import (
    matmul_small_typed,
    maybe_matmul_complex_accelerate,
    maybe_matmul_contiguous,
    maybe_matmul_f32_small,
    maybe_matmul_vector_accelerate,
)
from .predicates import (
    Rank2BlasLayout,
    is_contiguous_float_array,
    is_contiguous_typed_simd_array,
    is_float_dtype,
    is_typed_simd_dtype,
    max_int,
    rank2_blas_layout,
)
from .complex_kernels import (
    complex_binary_same_shape_strided_typed,
    complex_scalar_complex_contig_typed,
    complex_scalar_real_contig_typed,
    complex_unary_preserve_contig_typed,
    maybe_complex_binary_same_shape_strided,
)
from .fused_kernels import maybe_sin_add_mul_contiguous
from .reduce_kernels import (
    maybe_argmax_contiguous,
    maybe_reduce_contiguous,
    maybe_reduce_strided_typed,
    reduce_strided_typed,
    reduce_sum_typed,
)


def apply_binary_typed_vec[
    dtype: DType, width: Int
](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width], op: Int) raises -> SIMD[dtype, width]:
    # Comptime-typed parametric SIMD binary kernel — body of every
    # `binary_*_contig_typed` callsite after the runtime dtype dispatch.
    #
    # Edge cases by op family:
    # - FLOOR_DIV: float path is `floor(lhs/rhs)`, integer path is `lhs // rhs`
    #   (Mojo's // already floors toward -inf for ints, matching Python/numpy).
    # - DIV: integer divide-by-zero traps; float divide-by-zero produces IEEE
    #   ±inf (signed) or NaN (0/0) and propagates lane-wise.
    # - MAXIMUM/MINIMUM: NaN-propagating per numpy. Implemented via `select`
    #   so a single NaN lane poisons the whole result.
    # - FMIN/FMAX: NaN-suppressing (per IEEE 754-2008 minNum/maxNum) — when
    #   one side is NaN the other side wins. Float-only.
    # - ARCTAN2/HYPOT/COPYSIGN: float-only; gated `comptime if dtype !=
    #   DType.float16` because Mojo KGEN can't legalize the f16 externs.
    #   Upstream `dtype_result_for_binary` promotes f16 inputs to f32/f64
    #   so this path never sees f16 for these three ops.
    # - POWER: integer base with negative exponent traps; float `**` calls
    #   libm `pow`, with `0**0 = 1` per IEEE 754.
    if op == BinaryOp.ADD.value:
        return lhs + rhs
    if op == BinaryOp.SUB.value:
        return lhs - rhs
    if op == BinaryOp.MUL.value:
        return lhs * rhs
    if op == BinaryOp.DIV.value:
        return lhs / rhs
    if op == BinaryOp.FLOOR_DIV.value:
        comptime if dtype.is_floating_point():
            return math_floor(lhs / rhs)
        else:
            return lhs // rhs
    if op == BinaryOp.MOD.value:
        return lhs % rhs
    if op == BinaryOp.POWER.value:
        return lhs.__pow__(rhs)
    if op == BinaryOp.MAXIMUM.value:
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
    if op == BinaryOp.MINIMUM.value:
        comptime if dtype.is_floating_point():
            var lhs_nan = isnan(lhs)
            var rhs_nan = isnan(rhs)
            var any_nan = lhs_nan | rhs_nan
            var smaller = lhs.lt(rhs).select(lhs, rhs)
            return any_nan.select(SIMD[dtype, width](nan[dtype]()), smaller)
        else:
            return lhs.lt(rhs).select(lhs, rhs)
    if op == BinaryOp.FMAX.value:
        comptime if dtype.is_floating_point():
            var lhs_nan = isnan(lhs)
            var rhs_nan = isnan(rhs)
            var bigger = lhs.gt(rhs).select(lhs, rhs)
            var picked = lhs_nan.select(rhs, bigger)
            picked = rhs_nan.select(lhs, picked)
            return picked
        else:
            return lhs.gt(rhs).select(lhs, rhs)
    if op == BinaryOp.FMIN.value:
        comptime if dtype.is_floating_point():
            var lhs_nan = isnan(lhs)
            var rhs_nan = isnan(rhs)
            var smaller = lhs.lt(rhs).select(lhs, rhs)
            var picked = lhs_nan.select(rhs, smaller)
            picked = rhs_nan.select(lhs, picked)
            return picked
        else:
            return lhs.lt(rhs).select(lhs, rhs)
    if op == BinaryOp.ARCTAN2.value:
        # f16 has no `atan2f16` extern; KGEN refuses to legalize the call.
        # F16 paths route to f64 round-trip via `apply_binary_f64` upstream.
        comptime if (dtype.is_floating_point() and dtype != DType.float16):
            return atan2(lhs, rhs)
        else:
            raise Error("arctan2 requires float32/float64 dtype")
    if op == BinaryOp.HYPOT.value:
        comptime if (dtype.is_floating_point() and dtype != DType.float16):
            return hypot(lhs, rhs)
        else:
            raise Error("hypot requires float32/float64 dtype")
    if op == BinaryOp.COPYSIGN.value:
        comptime if (dtype.is_floating_point() and dtype != DType.float16):
            return copysign(lhs, rhs)
        else:
            raise Error("copysign requires float32/float64 dtype")
    raise Error("unknown binary op")


def apply_binary_typed_vec_static[
    dtype: DType, width: Int, op: Int
](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) raises -> SIMD[dtype, width]:
    comptime if op == BinaryOp.ADD.value:
        return lhs + rhs
    else:
        comptime if op == BinaryOp.SUB.value:
            return lhs - rhs
        else:
            comptime if op == BinaryOp.MUL.value:
                return lhs * rhs
            else:
                comptime if op == BinaryOp.DIV.value:
                    return lhs / rhs
                else:
                    return apply_binary_typed_vec[dtype, width](lhs, rhs, op)


def binary_same_shape_contig_typed_static[
    dtype: DType, op: Int
](
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
) raises:
    comptime width = simd_width_of[dtype]()
    var i = 0
    while i + width <= size:
        out_ptr.store(
            i,
            apply_binary_typed_vec_static[dtype, width, op](
                lhs_ptr.load[width=width](i),
                rhs_ptr.load[width=width](i),
            ),
        )
        i += width
    while i < size:
        out_ptr[i] = apply_binary_typed_vec_static[dtype, 1, op](
            SIMD[dtype, 1](lhs_ptr[i]), SIMD[dtype, 1](rhs_ptr[i])
        )[0]
        i += 1


def try_binary_same_shape_contig_typed_static[
    dtype: DType
](
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    op: Int,
) raises -> Bool:
    if op == BinaryOp.ADD.value:
        binary_same_shape_contig_typed_static[dtype, BinaryOp.ADD.value](lhs_ptr, rhs_ptr, out_ptr, size)
        return True
    if op == BinaryOp.SUB.value:
        binary_same_shape_contig_typed_static[dtype, BinaryOp.SUB.value](lhs_ptr, rhs_ptr, out_ptr, size)
        return True
    if op == BinaryOp.MUL.value:
        binary_same_shape_contig_typed_static[dtype, BinaryOp.MUL.value](lhs_ptr, rhs_ptr, out_ptr, size)
        return True
    if op == BinaryOp.DIV.value:
        binary_same_shape_contig_typed_static[dtype, BinaryOp.DIV.value](lhs_ptr, rhs_ptr, out_ptr, size)
        return True
    return False


def binary_same_shape_contig_typed[
    dtype: DType
](
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    op: Int,
) raises:
    if try_binary_same_shape_contig_typed_static[dtype](lhs_ptr, rhs_ptr, out_ptr, size, op):
        return
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
        out_ptr[i] = apply_binary_typed_vec[dtype, 1](SIMD[dtype, 1](lhs_ptr[i]), SIMD[dtype, 1](rhs_ptr[i]), op)[0]
        i += 1


def binary_scalar_contig_typed_static[
    dtype: DType, op: Int
](
    array_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    scalar_value: Scalar[dtype],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    scalar_on_left: Bool,
) raises:
    comptime width = simd_width_of[dtype]()
    var scalar_vec = SIMD[dtype, width](scalar_value)
    var i = 0
    while i + width <= size:
        var array_vec = array_ptr.load[width=width](i)
        if scalar_on_left:
            out_ptr.store(i, apply_binary_typed_vec_static[dtype, width, op](scalar_vec, array_vec))
        else:
            out_ptr.store(i, apply_binary_typed_vec_static[dtype, width, op](array_vec, scalar_vec))
        i += width
    while i < size:
        var lhs_v = SIMD[dtype, 1](array_ptr[i])
        var rhs_v = SIMD[dtype, 1](scalar_value)
        if scalar_on_left:
            out_ptr[i] = apply_binary_typed_vec_static[dtype, 1, op](rhs_v, lhs_v)[0]
        else:
            out_ptr[i] = apply_binary_typed_vec_static[dtype, 1, op](lhs_v, rhs_v)[0]
        i += 1


def try_binary_scalar_contig_typed_static[
    dtype: DType
](
    array_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    scalar_value: Scalar[dtype],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    op: Int,
    scalar_on_left: Bool,
) raises -> Bool:
    if op == BinaryOp.ADD.value:
        binary_scalar_contig_typed_static[dtype, BinaryOp.ADD.value](
            array_ptr, scalar_value, out_ptr, size, scalar_on_left
        )
        return True
    if op == BinaryOp.SUB.value:
        binary_scalar_contig_typed_static[dtype, BinaryOp.SUB.value](
            array_ptr, scalar_value, out_ptr, size, scalar_on_left
        )
        return True
    if op == BinaryOp.MUL.value:
        binary_scalar_contig_typed_static[dtype, BinaryOp.MUL.value](
            array_ptr, scalar_value, out_ptr, size, scalar_on_left
        )
        return True
    if op == BinaryOp.DIV.value:
        binary_scalar_contig_typed_static[dtype, BinaryOp.DIV.value](
            array_ptr, scalar_value, out_ptr, size, scalar_on_left
        )
        return True
    return False


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
    if try_binary_scalar_contig_typed_static[dtype](array_ptr, scalar_value, out_ptr, size, op, scalar_on_left):
        return
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
                apply_binary_typed_vec[dtype, width](scalar_vec, array_vec, op),
            )
        else:
            out_ptr.store(
                i,
                apply_binary_typed_vec[dtype, width](array_vec, scalar_vec, op),
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
    # Comptime-typed parametric variant of apply_unary_*_vec. Constrained to
    # floating-point dtypes — std.math sin/cos/exp/log/etc require IEEE FP at
    # the stdlib layer; integer dtypes raise upstream before reaching here.
    #
    # Each op below lowers to an LLVM intrinsic on most targets (e.g.
    # `@llvm.sin.v4f32`) and libm-per-lane elsewhere — error bound ~1 ulp on
    # glibc, ~2 ulp on macOS libm. Special values follow IEEE: log(0) = -inf,
    # log(<0) = NaN, sqrt(<0) = NaN, atan2(0, 0) = 0.
    #
    # f16 caveat: Mojo KGEN cannot legalise atan2f16/hypotf16/copysignf16
    # (no f16 libm intrinsics on most targets). The dispatcher upstream gates
    # those three ops via `comptime if dtype != DType.float16` and routes f16
    # through f32 promote-demote. This kernel itself never sees an f16 +
    # gated-op pair. Cross-ref `docs/research/simd-vectorisation.md §5`.
    if op == UnaryOp.SIN.value:
        return sin(value)
    if op == UnaryOp.COS.value:
        return cos(value)
    if op == UnaryOp.EXP.value:
        return exp(value)
    if op == UnaryOp.LOG.value:
        return log(value)
    if op == UnaryOp.TAN.value:
        return tan(value)
    if op == UnaryOp.ARCSIN.value:
        return asin(value)
    if op == UnaryOp.ARCCOS.value:
        return acos(value)
    if op == UnaryOp.ARCTAN.value:
        return atan(value)
    if op == UnaryOp.SINH.value:
        return sinh(value)
    if op == UnaryOp.COSH.value:
        return cosh(value)
    if op == UnaryOp.TANH.value:
        return tanh(value)
    if op == UnaryOp.LOG1P.value:
        return log1p(value)
    if op == UnaryOp.LOG2.value:
        return log2(value)
    if op == UnaryOp.LOG10.value:
        return log10(value)
    if op == UnaryOp.EXP2.value:
        return exp2(value)
    if op == UnaryOp.EXPM1.value:
        return expm1(value)
    if op == UnaryOp.SQRT.value:
        return sqrt(value)
    if op == UnaryOp.CBRT.value:
        return cbrt(value)
    if op == UnaryOp.DEG2RAD.value:
        return value * SIMD[dtype, width](0.017453292519943295)
    if op == UnaryOp.RAD2DEG.value:
        return value * SIMD[dtype, width](57.29577951308232)
    if op == UnaryOp.RECIPROCAL.value:
        return SIMD[dtype, width](1.0) / value
    if op == UnaryOp.NEGATE.value:
        return -value
    if op == UnaryOp.POSITIVE.value:
        return value
    if op == UnaryOp.ABS.value:
        var neg = -value
        return value.lt(SIMD[dtype, width](0)).select(neg, value)
    if op == UnaryOp.SQUARE.value:
        return value * value
    if op == UnaryOp.SIGN.value:
        var pos = value.gt(SIMD[dtype, width](0))
        var neg = value.lt(SIMD[dtype, width](0))
        var nan_mask = isnan(value)
        var s = pos.select(SIMD[dtype, width](1), SIMD[dtype, width](0))
        s = neg.select(SIMD[dtype, width](-1), s)
        return nan_mask.select(SIMD[dtype, width](nan[dtype]()), s)
    if op == UnaryOp.FLOOR.value:
        return math_floor(value)
    if op == UnaryOp.CEIL.value:
        return math_ceil(value)
    if op == UnaryOp.TRUNC.value:
        return math_trunc(value)
    if op == UnaryOp.RINT.value:
        return math_round(value)
    if op == UnaryOp.LOGICAL_NOT.value:
        return value.eq(SIMD[dtype, width](0)).select(SIMD[dtype, width](1), SIMD[dtype, width](0))
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
    # Comptime-typed body for `maybe_binary_same_shape_strided`. Combines an
    # outer coord-stack walker (axis-iteration with no divmod — uses
    # pre-computed carry tables to advance offsets per-axis) with one of four
    # inner-loop kinds picked upstream by `pick_inner_axis_for_strided_binary`:
    #
    #   inner_kind == 1: unit-stride contiguous inner. SIMD-vectorise the inner
    #     loop (load[width] / store[width]) until tail; tail is scalar. Highest
    #     bandwidth — useful bytes per cycle close to peak L1.
    #   inner_kind == 2: unit-stride inner with shared outer carry. Same SIMD
    #     vectorise as kind 1; outer carries are non-trivial enough to warrant
    #     a separate path that recomputes offsets per-row.
    #   inner_kind == 3: strided inner (non-unit). Per-element gather/scatter
    #     through `lp[i*stride]` — effective bandwidth is one element per
    #     cycle, no SIMD.
    #   inner_kind == 4: reversed inner (negative stride). Same as kind 3 but
    #     the index walks backward; no SIMD because vector loads can't reverse.
    #
    # The four-way split exists because SIMD bandwidth is dominated by load/store
    # alignment; picking the wrong inner walk leaves 4-8× perf on the floor.
    # Cross-ref `docs/research/memory-alignment.md §3`.
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
                var lvec = lp.load[width=width](i)
                var rvec = rp.load[width=width](i)
                if op == BinaryOp.ADD.value:
                    op_.store(i, lvec + rvec)
                else:
                    op_.store(i, apply_binary_typed_vec[dtype, width](lvec, rvec, op))
                i += width
            while i < inner_size:
                if op == BinaryOp.ADD.value:
                    op_[i] = lp[i] + rp[i]
                else:
                    op_[i] = apply_binary_typed_vec[dtype, 1](SIMD[dtype, 1](lp[i]), SIMD[dtype, 1](rp[i]), op)[0]
                i += 1
        elif inner_kind == 2:
            var lp = lhs_data + lhs_offset
            var rp = rhs_data + rhs_offset
            var i = 0
            while i + width <= inner_size:
                var lvec = lp.load[width=width](i)
                var rvec = rp.load[width=width](i)
                var ovec = lvec + rvec if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, width](
                    lvec, rvec, op
                )
                comptime for k in range(width):
                    result_data[result_offset + (i + k) * inner_result_stride] = ovec[k]
                i += width
            while i < inner_size:
                if op == BinaryOp.ADD.value:
                    result_data[result_offset + i * inner_result_stride] = lp[i] + rp[i]
                else:
                    result_data[result_offset + i * inner_result_stride] = apply_binary_typed_vec[dtype, 1](
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
                var lvec = (lhs_data + lhs_offset - i - width + 1).load[width=width](0).reversed()
                var rvec = (rhs_data + rhs_offset - i - width + 1).load[width=width](0).reversed()
                var ovec = lvec + rvec if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, width](
                    lvec, rvec, op
                )
                if inner_result_stride == 1:
                    (result_data + result_offset + i).store(0, ovec)
                else:
                    (result_data + result_offset - i - width + 1).store(0, ovec.reversed())
                i += width
            while i < inner_size:
                if op == BinaryOp.ADD.value:
                    result_data[result_offset + i * inner_result_stride] = (
                        lhs_data[lhs_offset - i] + rhs_data[rhs_offset - i]
                    )
                else:
                    result_data[result_offset + i * inner_result_stride] = apply_binary_typed_vec[dtype, 1](
                        SIMD[dtype, 1](lhs_data[lhs_offset - i]),
                        SIMD[dtype, 1](rhs_data[rhs_offset - i]),
                        op,
                    )[0]
                i += 1
        else:
            for i in range(inner_size):
                var lval = lhs_data[lhs_offset + i * inner_lhs_stride]
                var rval = rhs_data[rhs_offset + i * inner_rhs_stride]
                if op == BinaryOp.ADD.value:
                    result_data[result_offset + i * inner_result_stride] = lval + rval
                else:
                    result_data[result_offset + i * inner_result_stride] = apply_binary_typed_vec[dtype, 1](
                        SIMD[dtype, 1](lval),
                        SIMD[dtype, 1](rval),
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
            if op == BinaryOp.ADD.value:
                out_ptr.store(matrix_index, matrix_vec + row_vec)
            elif row_on_left:
                out_ptr.store(
                    matrix_index,
                    apply_binary_typed_vec[dtype, width](row_vec, matrix_vec, op),
                )
            else:
                out_ptr.store(
                    matrix_index,
                    apply_binary_typed_vec[dtype, width](matrix_vec, row_vec, op),
                )
            j += width
        while j < cols:
            var matrix_index = i * cols + j
            var lhs_v = SIMD[dtype, 1](matrix_ptr[matrix_index])
            var rhs_v = SIMD[dtype, 1](row_ptr[j])
            if op == BinaryOp.ADD.value:
                out_ptr[matrix_index] = matrix_ptr[matrix_index] + row_ptr[j]
            elif row_on_left:
                out_ptr[matrix_index] = apply_binary_typed_vec[dtype, 1](rhs_v, lhs_v, op)[0]
            else:
                out_ptr[matrix_index] = apply_binary_typed_vec[dtype, 1](lhs_v, rhs_v, op)[0]
            j += 1






def binary_strided_walk_typed[dtype: DType](lhs: Array, rhs: Array, mut result: Array, op: Int) raises:
    # Strided binary walker for same-dtype broadcasts. Output is c-contig
    # (just allocated), so we walk it as a flat counter and maintain
    # incremental cursors for lhs/rhs in element units. Strides for
    # broadcast dims (size-1 source against >1 target, or freshly added
    # outer axes) are zeroed out — same semantics as
    # `as_broadcast_layout` but inlined to skip per-step `* item_size`
    # multiplications and the List[List[Int]] indexing in
    # MultiLayoutIter. Net: ~30× faster on a 256² .T + 1D add.
    var lhs_ptr = lhs.data.bitcast[Scalar[dtype]]()
    var rhs_ptr = rhs.data.bitcast[Scalar[dtype]]()
    var out_ptr = result.data.bitcast[Scalar[dtype]]()
    var ndim = len(result.shape)
    var n = result.size_value
    var lhs_ndim = len(lhs.shape)
    var rhs_ndim = len(rhs.shape)
    var lhs_strides = List[Int]()
    var rhs_strides = List[Int]()
    for d in range(ndim):
        var l_axis = d - (ndim - lhs_ndim)
        var r_axis = d - (ndim - rhs_ndim)
        if l_axis < 0 or (lhs.shape[l_axis] == 1 and result.shape[d] != 1):
            lhs_strides.append(0)
        else:
            lhs_strides.append(lhs.strides[l_axis])
        if r_axis < 0 or (rhs.shape[r_axis] == 1 and result.shape[d] != 1):
            rhs_strides.append(0)
        else:
            rhs_strides.append(rhs.strides[r_axis])
    if ndim == 0:
        # rank-0 result: single scalar.
        out_ptr[result.offset_elems] = apply_binary_typed_vec[dtype, 1](
            SIMD[dtype, 1](lhs_ptr[lhs.offset_elems]),
            SIMD[dtype, 1](rhs_ptr[rhs.offset_elems]),
            op,
        )[0]
        return
    var coords = List[Int]()
    for _ in range(ndim):
        coords.append(0)
    var lhs_idx = lhs.offset_elems
    var rhs_idx = rhs.offset_elems
    var out_idx = result.offset_elems
    var visited = 0
    while visited < n:
        out_ptr[out_idx] = apply_binary_typed_vec[dtype, 1](
            SIMD[dtype, 1](lhs_ptr[lhs_idx]),
            SIMD[dtype, 1](rhs_ptr[rhs_idx]),
            op,
        )[0]
        visited += 1
        if visited >= n:
            break
        out_idx += 1
        var axis = ndim - 1
        while axis >= 0:
            coords[axis] += 1
            lhs_idx += lhs_strides[axis]
            rhs_idx += rhs_strides[axis]
            if coords[axis] < result.shape[axis]:
                break
            var rollback_lhs = coords[axis] * lhs_strides[axis]
            var rollback_rhs = coords[axis] * rhs_strides[axis]
            lhs_idx -= rollback_lhs
            rhs_idx -= rollback_rhs
            coords[axis] = 0
            axis -= 1


def maybe_binary_strided_typed(lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    # Dispatch the strided binary walker by dtype. Requires lhs, rhs,
    # and result to share a dtype (the common broadcast case for same-
    # type ops). Mixed-dtype broadcasts still take the f64 round-trip
    # walker upstream.
    if lhs.dtype_code != rhs.dtype_code or lhs.dtype_code != result.dtype_code:
        return False
    if lhs.dtype_code == ArrayDType.FLOAT64.value:
        binary_strided_walk_typed[DType.float64](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.FLOAT32.value:
        binary_strided_walk_typed[DType.float32](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.INT64.value:
        binary_strided_walk_typed[DType.int64](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.INT32.value:
        binary_strided_walk_typed[DType.int32](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.INT16.value:
        binary_strided_walk_typed[DType.int16](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.INT8.value:
        binary_strided_walk_typed[DType.int8](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.UINT64.value:
        binary_strided_walk_typed[DType.uint64](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.UINT32.value:
        binary_strided_walk_typed[DType.uint32](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.UINT16.value:
        binary_strided_walk_typed[DType.uint16](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.UINT8.value:
        binary_strided_walk_typed[DType.uint8](lhs, rhs, result, op)
        return True
    return False




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
            apply_unary_typed_vec[dtype, width](src_ptr.load[width=width](i), op),
        )
        i += width
    while i < size:
        out_ptr[i] = apply_unary_typed_vec[dtype, 1](SIMD[dtype, 1](src_ptr[i]), op)[0]
        i += 1


def apply_unary_preserve_typed_vec[
    dtype: DType, width: Int
](value: SIMD[dtype, width], op: Int) raises -> SIMD[dtype, width]:
    # Preserve-dtype unary ops: works on both float and integer dtypes.
    # Excludes the float-only ops (sin/cos/exp/log/etc.) which go through
    # `apply_unary_typed_vec` and require a float result anyway via
    # `dtype_result_for_unary`.
    if op == UnaryOp.NEGATE.value:
        return -value
    if op == UnaryOp.POSITIVE.value:
        return value
    if op == UnaryOp.ABS.value:
        comptime if dtype.is_unsigned():
            return value  # unsigned values are already non-negative
        else:
            var neg = -value
            return value.lt(SIMD[dtype, width](0)).select(neg, value)
    if op == UnaryOp.SQUARE.value:
        return value * value
    if op == UnaryOp.SIGN.value:
        comptime if dtype.is_unsigned():
            # uint sign: 0 → 0, anything else → 1.
            return value.gt(SIMD[dtype, width](0)).select(SIMD[dtype, width](1), SIMD[dtype, width](0))
        else:
            comptime if dtype.is_floating_point():
                var pos = value.gt(SIMD[dtype, width](0))
                var neg = value.lt(SIMD[dtype, width](0))
                var nan_mask = isnan(value)
                var s = pos.select(SIMD[dtype, width](1), SIMD[dtype, width](0))
                s = neg.select(SIMD[dtype, width](-1), s)
                return nan_mask.select(SIMD[dtype, width](nan[dtype]()), s)
            else:
                var pos = value.gt(SIMD[dtype, width](0))
                var neg = value.lt(SIMD[dtype, width](0))
                var s = pos.select(SIMD[dtype, width](1), SIMD[dtype, width](0))
                return neg.select(SIMD[dtype, width](-1), s)
    if op == UnaryOp.FLOOR.value or op == UnaryOp.CEIL.value or op == UnaryOp.TRUNC.value or op == UnaryOp.RINT.value:
        comptime if dtype.is_floating_point():
            if op == UnaryOp.FLOOR.value:
                return math_floor(value)
            if op == UnaryOp.CEIL.value:
                return math_ceil(value)
            if op == UnaryOp.TRUNC.value:
                return math_trunc(value)
            return math_round(value)
        else:
            return value  # integers are already integral
    if op == UnaryOp.LOGICAL_NOT.value:
        return value.eq(SIMD[dtype, width](0)).select(SIMD[dtype, width](1), SIMD[dtype, width](0))
    raise Error("unknown unary preserve op")


def unary_preserve_contig_typed[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    op: Int,
) raises:
    comptime width = simd_width_of[dtype]()
    var i = 0
    while i + width <= size:
        out_ptr.store(
            i,
            apply_unary_preserve_typed_vec[dtype, width](src_ptr.load[width=width](i), op),
        )
        i += width
    while i < size:
        out_ptr[i] = apply_unary_preserve_typed_vec[dtype, 1](SIMD[dtype, 1](src_ptr[i]), op)[0]
        i += 1


def maybe_unary_preserve_contiguous(src: Array, mut result: Array, op: Int) raises -> Bool:
    # Same-dtype c-contig fast path for preserve-dtype unary ops.
    if src.dtype_code != result.dtype_code or not is_c_contiguous(src) or not is_c_contiguous(result):
        return False
    # Complex paths.
    if src.dtype_code == ArrayDType.COMPLEX64.value:
        complex_unary_preserve_contig_typed[DType.float32](
            contiguous_ptr[DType.float32](src),
            contiguous_ptr[DType.float32](result),
            src.size_value,
            op,
        )
        return True
    if src.dtype_code == ArrayDType.COMPLEX128.value:
        complex_unary_preserve_contig_typed[DType.float64](
            contiguous_ptr[DType.float64](src),
            contiguous_ptr[DType.float64](result),
            src.size_value,
            op,
        )
        return True
    # 11-way real-dtype dispatch via unary helper. f16 not in `unary_preserve_contig_typed`
    # support set yet (only the 10 real-vec dtypes), so explicit fallback below.
    if dispatch_real_typed_simd_unary[unary_preserve_contig_typed](src.dtype_code, src, result, src.size_value, op):
        return True
    return False


def maybe_unary_contiguous(src: Array, mut result: Array, op: Int) raises -> Bool:
    if not is_contiguous_float_array(src) or not is_contiguous_float_array(result):
        return False
    if src.dtype_code == ArrayDType.FLOAT32.value and result.dtype_code == ArrayDType.FLOAT32.value:
        comptime if CompilationTarget.is_macos():
            if maybe_unary_accelerate[DType.float32](src, result, op):
                return True
        unary_contig_typed[DType.float32](
            contiguous_ptr[DType.float32](src),
            contiguous_ptr[DType.float32](result),
            src.size_value,
            op,
        )
        return True
    if op == UnaryOp.LOG.value:
        return False
    if src.dtype_code == ArrayDType.FLOAT64.value and result.dtype_code == ArrayDType.FLOAT64.value:
        comptime if CompilationTarget.is_macos():
            if maybe_unary_accelerate[DType.float64](src, result, op):
                return True
        unary_contig_typed[DType.float64](
            contiguous_ptr[DType.float64](src),
            contiguous_ptr[DType.float64](result),
            src.size_value,
            op,
        )
        return True
    return False


def unary_rank2_strided_typed[
    dtype: DType
](src: Array, mut result: Array, op: Int) raises where dtype.is_floating_point():
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_row_stride = src.strides[0]
    var src_col_stride = src.strides[1]
    var result_row_stride = result.strides[0]
    var result_col_stride = result.strides[1]
    var src_data = src.data.bitcast[Scalar[dtype]]()
    var result_data = result.data.bitcast[Scalar[dtype]]()
    comptime tile = 8
    var row_block = 0
    while row_block < rows:
        var row_end = row_block + tile
        if row_end > rows:
            row_end = rows
        var col_block = 0
        while col_block < cols:
            var col_end = col_block + tile
            if col_end > cols:
                col_end = cols
            if src_row_stride == 1:
                var col = col_block
                while col < col_end:
                    var row = row_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while row < row_end:
                        if op == UnaryOp.SIN.value:
                            result_data[result_index] = sin(SIMD[dtype, 1](src_data[src_index]))[0]
                        else:
                            result_data[result_index] = apply_unary_typed_vec[dtype, 1](
                                SIMD[dtype, 1](src_data[src_index]), op
                            )[0]
                        row += 1
                        src_index += src_row_stride
                        result_index += result_row_stride
                    col += 1
            else:
                var row = row_block
                while row < row_end:
                    var col = col_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while col < col_end:
                        if op == UnaryOp.SIN.value:
                            result_data[result_index] = sin(SIMD[dtype, 1](src_data[src_index]))[0]
                        else:
                            result_data[result_index] = apply_unary_typed_vec[dtype, 1](
                                SIMD[dtype, 1](src_data[src_index]), op
                            )[0]
                        col += 1
                        src_index += src_col_stride
                        result_index += result_col_stride
                    row += 1
            col_block += tile
        row_block += tile


def maybe_unary_rank2_strided(src: Array, mut result: Array, op: Int) raises -> Bool:
    if len(src.shape) != 2 or src.dtype_code != result.dtype_code or not is_c_contiguous(result):
        return False
    if src.dtype_code == ArrayDType.FLOAT32.value:
        unary_rank2_strided_typed[DType.float32](src, result, op)
        result.backend_code = BackendKind.FUSED.value
        return True
    if src.dtype_code == ArrayDType.FLOAT64.value:
        unary_rank2_strided_typed[DType.float64](src, result, op)
        result.backend_code = BackendKind.FUSED.value
        return True
    return False



def complex_binary_contig_typed[
    dtype: DType
](
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n_elems: Int,
    op: Int,
) raises where dtype.is_floating_point():
    # Complex arithmetic over interleaved (real, imag) pairs. `dtype` is the
    # underlying float (f32 → complex64, f64 → complex128). `n_elems` counts
    # complex values; storage is 2 × n_elems lanes of `dtype`.
    #
    # ADD/SUB: linear in components. Reuse the real-typed kernel over the
    # 2 × n_elems-wide vector — no special handling needed.
    #
    # MUL: schoolbook FMA — `(a+bi)(c+di) = (ac − bd) + (ad + bc)i`.
    # Two FMAs per real lane (`fma(a, c, -b*d)`, `fma(a, d, b*c)`) give
    # componentwise error ≤ √2·ulp·|result|. Avoids Karatsuba's loss of
    # significance on the difference of products.
    #
    # DIV: Smith 1962 algorithm — for `(a+bi)/(c+di)`, branch on |c| vs |d|
    # to scale by the larger denominator first. Without this, computing
    # `c² + d²` overflows for moderate |c|, |d| even when the quotient is
    # representable. Cross-ref `docs/research/complex-kernels.md §2`.
    if op == BinaryOp.ADD.value or op == BinaryOp.SUB.value:
        # Componentwise on the float pairs: add/sub treats interleaved
        # storage as a 2N float vector. Reuse the existing typed kernel.
        binary_same_shape_contig_typed[dtype](
            lhs_ptr,
            rhs_ptr,
            out_ptr,
            n_elems * 2,
            op,
        )
        return
    if op == BinaryOp.MUL.value:
        for i in range(n_elems):
            var a = lhs_ptr[i * 2]
            var b = lhs_ptr[i * 2 + 1]
            var c = rhs_ptr[i * 2]
            var d = rhs_ptr[i * 2 + 1]
            out_ptr[i * 2] = a * c - b * d
            out_ptr[i * 2 + 1] = a * d + b * c
        return
    if op == BinaryOp.DIV.value:
        # Smith's algorithm: avoids overflow when |c|, |d| are very different.
        for i in range(n_elems):
            var a = lhs_ptr[i * 2]
            var b = lhs_ptr[i * 2 + 1]
            var c = rhs_ptr[i * 2]
            var d = rhs_ptr[i * 2 + 1]
            var abs_c = c if c >= Scalar[dtype](0) else -c
            var abs_d = d if d >= Scalar[dtype](0) else -d
            if abs_c >= abs_d:
                var r = d / c
                var den = c + d * r
                out_ptr[i * 2] = (a + b * r) / den
                out_ptr[i * 2 + 1] = (b - a * r) / den
            else:
                var r = c / d
                var den = c * r + d
                out_ptr[i * 2] = (a * r + b) / den
                out_ptr[i * 2 + 1] = (b * r - a) / den
        return
    raise Error("unsupported op for complex binary kernel")


def maybe_complex_binary_contiguous_accelerate(lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    if op != BinaryOp.ADD.value and op != BinaryOp.SUB.value:
        return False
    comptime if not CompilationTarget.is_macos():
        return False
    if (
        lhs.dtype_code == ArrayDType.COMPLEX64.value
        and rhs.dtype_code == ArrayDType.COMPLEX64.value
        and result.dtype_code == ArrayDType.COMPLEX64.value
    ):
        var lhs_ptr = lhs.data.bitcast[Float32]() + lhs.offset_elems * 2
        var rhs_ptr = rhs.data.bitcast[Float32]() + rhs.offset_elems * 2
        var out_ptr = result.data.bitcast[Float32]() + result.offset_elems * 2
        if op == BinaryOp.ADD.value:
            call_vdsp_binary_f32["vDSP_vadd"](lhs_ptr, rhs_ptr, out_ptr, result.size_value * 2)
        else:
            call_vdsp_binary_f32["vDSP_vsub"](rhs_ptr, lhs_ptr, out_ptr, result.size_value * 2)
        result.backend_code = BackendKind.ACCELERATE.value
        return True
    if (
        lhs.dtype_code == ArrayDType.COMPLEX128.value
        and rhs.dtype_code == ArrayDType.COMPLEX128.value
        and result.dtype_code == ArrayDType.COMPLEX128.value
    ):
        var lhs_ptr = lhs.data.bitcast[Float64]() + lhs.offset_elems * 2
        var rhs_ptr = rhs.data.bitcast[Float64]() + rhs.offset_elems * 2
        var out_ptr = result.data.bitcast[Float64]() + result.offset_elems * 2
        if op == BinaryOp.ADD.value:
            call_vdsp_binary_f64["vDSP_vaddD"](lhs_ptr, rhs_ptr, out_ptr, result.size_value * 2)
        else:
            call_vdsp_binary_f64["vDSP_vsubD"](rhs_ptr, lhs_ptr, out_ptr, result.size_value * 2)
        result.backend_code = BackendKind.ACCELERATE.value
        return True
    return False


comptime BinaryContigKernel = def[dt: DType](
    UnsafePointer[Scalar[dt], MutExternalOrigin],
    UnsafePointer[Scalar[dt], MutExternalOrigin],
    UnsafePointer[Scalar[dt], MutExternalOrigin],
    Int,
    Int,
) thin raises -> None
"""Shape of any same-dtype contiguous binary kernel: three pointers, size, op code.
"""


def dispatch_real_typed_simd_binary[
    kernel: BinaryContigKernel,
](dtype_code: Int, lhs: Array, rhs: Array, mut result: Array, size: Int, op: Int,) raises -> Bool:
    """dtype dispatch from runtime `dtype_code` to comptime-typed kernel.

    Caller invariant: `lhs.dtype_code == rhs.dtype_code == result.dtype_code` and all three arrays are c-contiguous.
    Returns True if a typed path was taken; False if the dtype isn't covered (caller should fall through to the f64 round-trip path).
    """
    if dtype_code == ArrayDType.FLOAT32.value:
        kernel[DType.float32](
            contiguous_ptr[DType.float32](lhs),
            contiguous_ptr[DType.float32](rhs),
            contiguous_ptr[DType.float32](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.FLOAT64.value:
        kernel[DType.float64](
            contiguous_ptr[DType.float64](lhs),
            contiguous_ptr[DType.float64](rhs),
            contiguous_ptr[DType.float64](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.INT64.value:
        kernel[DType.int64](
            contiguous_ptr[DType.int64](lhs),
            contiguous_ptr[DType.int64](rhs),
            contiguous_ptr[DType.int64](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.INT32.value:
        kernel[DType.int32](
            contiguous_ptr[DType.int32](lhs),
            contiguous_ptr[DType.int32](rhs),
            contiguous_ptr[DType.int32](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.UINT64.value:
        kernel[DType.uint64](
            contiguous_ptr[DType.uint64](lhs),
            contiguous_ptr[DType.uint64](rhs),
            contiguous_ptr[DType.uint64](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.UINT32.value:
        kernel[DType.uint32](
            contiguous_ptr[DType.uint32](lhs),
            contiguous_ptr[DType.uint32](rhs),
            contiguous_ptr[DType.uint32](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.INT16.value:
        kernel[DType.int16](
            contiguous_ptr[DType.int16](lhs),
            contiguous_ptr[DType.int16](rhs),
            contiguous_ptr[DType.int16](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.INT8.value:
        kernel[DType.int8](
            contiguous_ptr[DType.int8](lhs),
            contiguous_ptr[DType.int8](rhs),
            contiguous_ptr[DType.int8](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.UINT16.value:
        kernel[DType.uint16](
            contiguous_ptr[DType.uint16](lhs),
            contiguous_ptr[DType.uint16](rhs),
            contiguous_ptr[DType.uint16](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.UINT8.value:
        kernel[DType.uint8](
            contiguous_ptr[DType.uint8](lhs),
            contiguous_ptr[DType.uint8](rhs),
            contiguous_ptr[DType.uint8](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.FLOAT16.value:
        kernel[DType.float16](
            contiguous_ptr[DType.float16](lhs),
            contiguous_ptr[DType.float16](rhs),
            contiguous_ptr[DType.float16](result),
            size,
            op,
        )
        return True
    return False


comptime UnaryContigKernel = def[dt: DType](
    UnsafePointer[Scalar[dt], MutExternalOrigin],
    UnsafePointer[Scalar[dt], MutExternalOrigin],
    Int,
    Int,
) thin raises -> None
"""Shape of any same-dtype contiguous unary kernel: src ptr, out ptr, size, op code.

Used by `maybe_unary_preserve_contiguous` (14-way real dispatch, integer-friendly)
and `maybe_unary_contiguous` (float-only sub-variant, integer dtypes raise).
"""


def dispatch_real_typed_simd_unary[
    kernel: UnaryContigKernel,
](dtype_code: Int, src: Array, mut result: Array, size: Int, op: Int,) raises -> Bool:
    """11-way real-dtype dispatch for unary kernels (src ptr → out ptr, both same dtype).

    Caller invariant: `src.dtype_code == result.dtype_code` and both arrays are
    c-contiguous. Returns True if a typed path was taken; False if the dtype isn't
    covered (caller falls through to f64 round-trip or complex specialization).
    """
    if dtype_code == ArrayDType.FLOAT32.value:
        kernel[DType.float32](contiguous_ptr[DType.float32](src), contiguous_ptr[DType.float32](result), size, op)
        return True
    if dtype_code == ArrayDType.FLOAT64.value:
        kernel[DType.float64](contiguous_ptr[DType.float64](src), contiguous_ptr[DType.float64](result), size, op)
        return True
    if dtype_code == ArrayDType.INT64.value:
        kernel[DType.int64](contiguous_ptr[DType.int64](src), contiguous_ptr[DType.int64](result), size, op)
        return True
    if dtype_code == ArrayDType.INT32.value:
        kernel[DType.int32](contiguous_ptr[DType.int32](src), contiguous_ptr[DType.int32](result), size, op)
        return True
    if dtype_code == ArrayDType.UINT64.value:
        kernel[DType.uint64](contiguous_ptr[DType.uint64](src), contiguous_ptr[DType.uint64](result), size, op)
        return True
    if dtype_code == ArrayDType.UINT32.value:
        kernel[DType.uint32](contiguous_ptr[DType.uint32](src), contiguous_ptr[DType.uint32](result), size, op)
        return True
    if dtype_code == ArrayDType.INT16.value:
        kernel[DType.int16](contiguous_ptr[DType.int16](src), contiguous_ptr[DType.int16](result), size, op)
        return True
    if dtype_code == ArrayDType.INT8.value:
        kernel[DType.int8](contiguous_ptr[DType.int8](src), contiguous_ptr[DType.int8](result), size, op)
        return True
    if dtype_code == ArrayDType.UINT16.value:
        kernel[DType.uint16](contiguous_ptr[DType.uint16](src), contiguous_ptr[DType.uint16](result), size, op)
        return True
    if dtype_code == ArrayDType.UINT8.value:
        kernel[DType.uint8](contiguous_ptr[DType.uint8](src), contiguous_ptr[DType.uint8](result), size, op)
        return True
    if dtype_code == ArrayDType.FLOAT16.value:
        kernel[DType.float16](contiguous_ptr[DType.float16](src), contiguous_ptr[DType.float16](result), size, op)
        return True
    return False


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
    # Phase 5b typed int paths: int32/int64/uint32/uint64.
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
    # Phase 5b typed int paths.
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


@fieldwise_init
struct StridedInnerChoice(ImplicitlyCopyable, Movable):
    var axis: Int
    var kind: Int  # 0=scalar, 1=full-SIMD, 2=SIMD-load+scatter


def pick_inner_axis_for_strided_binary(lhs: Array, rhs: Array, result: Array) raises -> StridedInnerChoice:
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
](lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
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
    if not same_shape(lhs.shape, rhs.shape) or not same_shape(lhs.shape, result.shape):
        return False
    if lhs.strides[0] != 1 or rhs.strides[0] != 1:
        return False
    if not is_c_contiguous(result):
        return False
    var rows = lhs.shape[0]
    var cols = lhs.shape[1]
    var lhs_col_stride = lhs.strides[1]
    var rhs_col_stride = rhs.strides[1]
    var result_row_stride = result.strides[0]
    var lhs_data = lhs.data.bitcast[Scalar[dtype]]() + lhs.offset_elems
    var rhs_data = rhs.data.bitcast[Scalar[dtype]]() + rhs.offset_elems
    var result_data = result.data.bitcast[Scalar[dtype]]() + result.offset_elems
    if lhs_col_stride == rows and rhs_col_stride == rows:
        comptime width = simd_width_of[dtype]()
        var total = rows * cols
        var linear = 0
        while linear + width <= total:
            var lvec = lhs_data.load[width=width](linear)
            var rvec = rhs_data.load[width=width](linear)
            var out = lvec + rvec if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, width](lvec, rvec, op)
            result_data.store(linear, out)
            linear += width
        while linear < total:
            if op == BinaryOp.ADD.value:
                result_data[linear] = lhs_data[linear] + rhs_data[linear]
            else:
                result_data[linear] = apply_binary_typed_vec[dtype, 1](
                    SIMD[dtype, 1](lhs_data[linear]), SIMD[dtype, 1](rhs_data[linear]), op
                )[0]
            linear += 1
        result.strides[0] = 1
        result.strides[1] = rows
        result.backend_code = BackendKind.FUSED.value
        return True
    if rows < 4 or cols < 4:
        return False
    comptime tile = 4
    var main_rows = rows - (rows % tile)
    var main_cols = cols - (cols % tile)
    var i = 0
    while i < main_rows:
        var j = 0
        while j < main_cols:
            var l0 = (lhs_data + i + j * lhs_col_stride).load[width=tile](0)
            var l1 = (lhs_data + i + (j + 1) * lhs_col_stride).load[width=tile](0)
            var l2 = (lhs_data + i + (j + 2) * lhs_col_stride).load[width=tile](0)
            var l3 = (lhs_data + i + (j + 3) * lhs_col_stride).load[width=tile](0)
            var r0 = (rhs_data + i + j * rhs_col_stride).load[width=tile](0)
            var r1 = (rhs_data + i + (j + 1) * rhs_col_stride).load[width=tile](0)
            var r2 = (rhs_data + i + (j + 2) * rhs_col_stride).load[width=tile](0)
            var r3 = (rhs_data + i + (j + 3) * rhs_col_stride).load[width=tile](0)
            var s0 = l0 + r0 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l0, r0, op)
            var s1 = l1 + r1 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l1, r1, op)
            var s2 = l2 + r2 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l2, r2, op)
            var s3 = l3 + r3 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l3, r3, op)
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
                if op == BinaryOp.ADD.value:
                    result_data[(i + k) * result_row_stride + j] = lv + rv
                else:
                    result_data[(i + k) * result_row_stride + j] = apply_binary_typed_vec[dtype, 1](
                        SIMD[dtype, 1](lv), SIMD[dtype, 1](rv), op
                    )[0]
            j += 1
        i += tile
    # row tail: scalar walk for residual rows
    while i < rows:
        var j = 0
        while j < cols:
            var lv = lhs_data[i + j * lhs_col_stride]
            var rv = rhs_data[i + j * rhs_col_stride]
            if op == BinaryOp.ADD.value:
                result_data[i * result_row_stride + j] = lv + rv
            else:
                result_data[i * result_row_stride + j] = apply_binary_typed_vec[dtype, 1](
                    SIMD[dtype, 1](lv), SIMD[dtype, 1](rv), op
                )[0]
            j += 1
        i += 1
    result.backend_code = BackendKind.FUSED.value
    return True


def maybe_binary_rank3_axis0_tile[dtype: DType](lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    # Batched variant of the rank-2 transposed tile. Covers layouts like
    # C-contig rank-3 `.transpose((2, 0, 1))`: each middle-axis slice is a
    # rank-2 transpose with input stride-1 on axis 0 and c-contig output.
    if len(lhs.shape) != 3:
        return False
    if not same_shape(lhs.shape, rhs.shape) or not same_shape(lhs.shape, result.shape):
        return False
    if lhs.strides[0] != 1 or rhs.strides[0] != 1:
        return False
    if not is_c_contiguous(result):
        return False
    var rows = lhs.shape[0]
    var batches = lhs.shape[1]
    var cols = lhs.shape[2]
    var lhs_batch_stride = lhs.strides[1]
    var rhs_batch_stride = rhs.strides[1]
    var lhs_col_stride = lhs.strides[2]
    var rhs_col_stride = rhs.strides[2]
    var result_row_stride = result.strides[0]
    var result_batch_stride = result.strides[1]
    var lhs_data = lhs.data.bitcast[Scalar[dtype]]() + lhs.offset_elems
    var rhs_data = rhs.data.bitcast[Scalar[dtype]]() + rhs.offset_elems
    var result_data = result.data.bitcast[Scalar[dtype]]() + result.offset_elems
    var physical_plane = rows * cols
    if (
        lhs_col_stride == rows
        and rhs_col_stride == rows
        and lhs_batch_stride == physical_plane
        and rhs_batch_stride == physical_plane
    ):
        comptime width = simd_width_of[dtype]()
        var total = rows * batches * cols
        var linear = 0
        while linear + width <= total:
            var lvec = lhs_data.load[width=width](linear)
            var rvec = rhs_data.load[width=width](linear)
            var out = lvec + rvec if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, width](lvec, rvec, op)
            result_data.store(linear, out)
            linear += width
        while linear < total:
            if op == BinaryOp.ADD.value:
                result_data[linear] = lhs_data[linear] + rhs_data[linear]
            else:
                result_data[linear] = apply_binary_typed_vec[dtype, 1](
                    SIMD[dtype, 1](lhs_data[linear]), SIMD[dtype, 1](rhs_data[linear]), op
                )[0]
            linear += 1
        result.strides[0] = 1
        result.strides[1] = physical_plane
        result.strides[2] = rows
        result.backend_code = BackendKind.FUSED.value
        return True
    if rows < 4 or cols < 4:
        return False
    comptime tile = 4
    var main_rows = rows - (rows % tile)
    var main_cols = cols - (cols % tile)
    for batch in range(batches):
        var lhs_batch = lhs_data + batch * lhs_batch_stride
        var rhs_batch = rhs_data + batch * rhs_batch_stride
        var result_batch = result_data + batch * result_batch_stride
        var i = 0
        while i < main_rows:
            var j = 0
            while j < main_cols:
                var l0 = (lhs_batch + i + j * lhs_col_stride).load[width=tile](0)
                var l1 = (lhs_batch + i + (j + 1) * lhs_col_stride).load[width=tile](0)
                var l2 = (lhs_batch + i + (j + 2) * lhs_col_stride).load[width=tile](0)
                var l3 = (lhs_batch + i + (j + 3) * lhs_col_stride).load[width=tile](0)
                var r0 = (rhs_batch + i + j * rhs_col_stride).load[width=tile](0)
                var r1 = (rhs_batch + i + (j + 1) * rhs_col_stride).load[width=tile](0)
                var r2 = (rhs_batch + i + (j + 2) * rhs_col_stride).load[width=tile](0)
                var r3 = (rhs_batch + i + (j + 3) * rhs_col_stride).load[width=tile](0)
                var s0 = l0 + r0 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l0, r0, op)
                var s1 = l1 + r1 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l1, r1, op)
                var s2 = l2 + r2 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l2, r2, op)
                var s3 = l3 + r3 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l3, r3, op)
                var u0 = s0.shuffle[0, 4, 1, 5](s1)
                var u1 = s0.shuffle[2, 6, 3, 7](s1)
                var u2 = s2.shuffle[0, 4, 1, 5](s3)
                var u3 = s2.shuffle[2, 6, 3, 7](s3)
                var t0 = u0.shuffle[0, 1, 4, 5](u2)
                var t1 = u0.shuffle[2, 3, 6, 7](u2)
                var t2 = u1.shuffle[0, 1, 4, 5](u3)
                var t3 = u1.shuffle[2, 3, 6, 7](u3)
                (result_batch + i * result_row_stride + j).store(0, t0)
                (result_batch + (i + 1) * result_row_stride + j).store(0, t1)
                (result_batch + (i + 2) * result_row_stride + j).store(0, t2)
                (result_batch + (i + 3) * result_row_stride + j).store(0, t3)
                j += tile
            while j < cols:
                comptime for k in range(tile):
                    var lv = lhs_batch[i + k + j * lhs_col_stride]
                    var rv = rhs_batch[i + k + j * rhs_col_stride]
                    if op == BinaryOp.ADD.value:
                        result_batch[(i + k) * result_row_stride + j] = lv + rv
                    else:
                        result_batch[(i + k) * result_row_stride + j] = apply_binary_typed_vec[dtype, 1](
                            SIMD[dtype, 1](lv), SIMD[dtype, 1](rv), op
                        )[0]
                j += 1
            i += tile
        while i < rows:
            var j = 0
            while j < cols:
                var lv = lhs_batch[i + j * lhs_col_stride]
                var rv = rhs_batch[i + j * rhs_col_stride]
                if op == BinaryOp.ADD.value:
                    result_batch[i * result_row_stride + j] = lv + rv
                else:
                    result_batch[i * result_row_stride + j] = apply_binary_typed_vec[dtype, 1](
                        SIMD[dtype, 1](lv), SIMD[dtype, 1](rv), op
                    )[0]
                j += 1
            i += 1
    result.backend_code = BackendKind.FUSED.value
    return True


def maybe_binary_rank2_transposed_tile_bcast_1d[
    dtype: DType
](lhs: Array, rhs: Array, mut result: Array, op: Int, rhs_on_left: Bool) raises -> Bool:
    # 4×4 SIMD-tile path for the column-broadcast pattern: lhs is rank-2
    # F-contig (stride[0] == 1, the layout `.T` produces from a c-contig
    # rank-2 source) and rhs is rank-1 broadcasting along axis 0 of lhs
    # (rhs.shape[0] == lhs.shape[1]). Without this the generic walker
    # picks a strided inner axis with no SIMD coverage — the .T + 1D
    # pattern hits ~120× monpy/numpy on the scalar walker.
    #
    # Strategy mirrors `maybe_binary_rank2_transposed_tile`: load 4
    # contiguous SIMD vectors from lhs along the stride-1 axis 0, splat
    # the 4 corresponding rhs scalars, do 4 SIMD ops, transpose 4×4 in
    # registers, emit 4 contiguous SIMD stores into c-contig result.
    if len(lhs.shape) != 2 or len(rhs.shape) != 1:
        return False
    if lhs.strides[0] != 1:
        return False
    if rhs.shape[0] != lhs.shape[1]:
        return False
    if rhs.strides[0] != 1:
        return False
    if not is_c_contiguous(result):
        return False
    if not same_shape(lhs.shape, result.shape):
        return False
    var rows = lhs.shape[0]
    var cols = lhs.shape[1]
    if rows < 4 or cols < 4:
        return False
    var lhs_col_stride = lhs.strides[1]
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
            var l1 = (lhs_data + i + (j + 1) * lhs_col_stride).load[width=tile](0)
            var l2 = (lhs_data + i + (j + 2) * lhs_col_stride).load[width=tile](0)
            var l3 = (lhs_data + i + (j + 3) * lhs_col_stride).load[width=tile](0)
            var r0 = SIMD[dtype, tile](rhs_data[j])
            var r1 = SIMD[dtype, tile](rhs_data[j + 1])
            var r2 = SIMD[dtype, tile](rhs_data[j + 2])
            var r3 = SIMD[dtype, tile](rhs_data[j + 3])
            var s0: SIMD[dtype, tile]
            var s1: SIMD[dtype, tile]
            var s2: SIMD[dtype, tile]
            var s3: SIMD[dtype, tile]
            if rhs_on_left:
                s0 = r0 + l0 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](r0, l0, op)
                s1 = r1 + l1 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](r1, l1, op)
                s2 = r2 + l2 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](r2, l2, op)
                s3 = r3 + l3 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](r3, l3, op)
            else:
                s0 = l0 + r0 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l0, r0, op)
                s1 = l1 + r1 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l1, r1, op)
                s2 = l2 + r2 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l2, r2, op)
                s3 = l3 + r3 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l3, r3, op)
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
        # column tail (residual cols within main rows): scalar walk
        while j < cols:
            comptime for k in range(tile):
                var lv = lhs_data[i + k + j * lhs_col_stride]
                var rv = rhs_data[j]
                var lhs_v = lv
                var rhs_v = rv
                if rhs_on_left:
                    lhs_v = rv
                    rhs_v = lv
                if op == BinaryOp.ADD.value:
                    result_data[(i + k) * result_row_stride + j] = lhs_v + rhs_v
                else:
                    result_data[(i + k) * result_row_stride + j] = apply_binary_typed_vec[dtype, 1](
                        SIMD[dtype, 1](lhs_v), SIMD[dtype, 1](rhs_v), op
                    )[0]
            j += 1
        i += tile
    # row tail (residual rows): scalar walk
    while i < rows:
        var j = 0
        while j < cols:
            var lv = lhs_data[i + j * lhs_col_stride]
            var rv = rhs_data[j]
            var lhs_v = lv
            var rhs_v = rv
            if rhs_on_left:
                lhs_v = rv
                rhs_v = lv
            if op == BinaryOp.ADD.value:
                result_data[i * result_row_stride + j] = lhs_v + rhs_v
            else:
                result_data[i * result_row_stride + j] = apply_binary_typed_vec[dtype, 1](
                    SIMD[dtype, 1](lhs_v), SIMD[dtype, 1](rhs_v), op
                )[0]
            j += 1
        i += 1
    result.backend_code = BackendKind.FUSED.value
    return True


def maybe_binary_column_broadcast_dispatch(lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    # Dispatch the column-broadcast tile kernel by dtype. Tries both
    # operand orderings (matrix on left, then 1D on left).
    if lhs.dtype_code != rhs.dtype_code or lhs.dtype_code != result.dtype_code:
        return False
    if lhs.dtype_code == ArrayDType.FLOAT32.value:
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.float32](lhs, rhs, result, op, False):
            return True
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.float32](rhs, lhs, result, op, True):
            return True
        return False
    if lhs.dtype_code == ArrayDType.FLOAT64.value:
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.float64](lhs, rhs, result, op, False):
            return True
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.float64](rhs, lhs, result, op, True):
            return True
        return False
    if lhs.dtype_code == ArrayDType.INT64.value:
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.int64](lhs, rhs, result, op, False):
            return True
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.int64](rhs, lhs, result, op, True):
            return True
        return False
    if lhs.dtype_code == ArrayDType.INT32.value:
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.int32](lhs, rhs, result, op, False):
            return True
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.int32](rhs, lhs, result, op, True):
            return True
        return False
    if lhs.dtype_code == ArrayDType.UINT64.value:
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.uint64](lhs, rhs, result, op, False):
            return True
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.uint64](rhs, lhs, result, op, True):
            return True
        return False
    if lhs.dtype_code == ArrayDType.UINT32.value:
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.uint32](lhs, rhs, result, op, False):
            return True
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.uint32](rhs, lhs, result, op, True):
            return True
        return False
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




