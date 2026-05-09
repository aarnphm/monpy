"""Typed SIMD kernels: comptime-parametric primitives for binary/unary ops.

Hosts the ground-floor kernels that every higher-level dispatcher (binary,
scalar broadcast, row broadcast, tile, strided walker) ultimately calls
into. All kernels are `[dtype: DType]` parametric and resolve SIMD width
via `simd_width_of[dtype]()`. Two flavours per shape:

  - The runtime-`op` flavour (`apply_binary_typed_vec[dtype, width]`,
    `binary_same_shape_contig_typed[dtype]`, etc.) takes `op: Int` and
    branches per-vector. One specialisation per dtype.
  - The comptime-`op` flavour (`*_static[dtype, width, op]`,
    `try_*_static[dtype]`) inlines the op selection into the kernel body
    so the inner loop has no per-iteration branch. The `try_*` helpers
    cover high-frequency ops and fall back to the runtime path for the rest.

f16 caveat: `atan2`, `hypot`, `copysign` have no `*f16` extern in Mojo
KGEN. The runtime kernel raises for those (gated `comptime if dtype !=
DType.float16`); upstream `dtype_result_for_binary` promotes f16 to f32/f64
so the kernel never receives an f16 + gated-op pair. Cross-ref
`docs/research/simd-vectorisation.md §5`.
"""

from std.algorithm import sync_parallelize
from std.math import (
    acos,
    asin,
    atan,
    atan2,
    cbrt,
    ceil as math_ceil,
    ceildiv,
    copysign,
    cos,
    cosh,
    exp,
    exp2,
    expm1,
    floor as math_floor,
    hypot,
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
from std.sys import simd_width_of, size_of

from array import Array
from domain import BinaryOp, UnaryOp

from elementwise.kernels.parallel import (
    ELEMENTWISE_HEAVY_GRAIN,
    ELEMENTWISE_LIGHT_GRAIN,
    worker_count_for_bytes,
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
        var lhs_vec = lhs_ptr.load[width=width](i)
        var rhs_vec = rhs_ptr.load[width=width](i)
        comptime if op == BinaryOp.ADD.value:
            out_ptr.store(i, lhs_vec + rhs_vec)
        else:
            comptime if op == BinaryOp.SUB.value:
                out_ptr.store(i, lhs_vec - rhs_vec)
            else:
                comptime if op == BinaryOp.MUL.value:
                    out_ptr.store(i, lhs_vec * rhs_vec)
                else:
                    comptime if op == BinaryOp.DIV.value:
                        out_ptr.store(i, lhs_vec / rhs_vec)
                    else:
                        out_ptr.store(i, apply_binary_typed_vec_static[dtype, width, op](lhs_vec, rhs_vec))
        i += width
    while i < size:
        var lhs = lhs_ptr[i]
        var rhs = rhs_ptr[i]
        comptime if op == BinaryOp.ADD.value:
            out_ptr[i] = lhs + rhs
        else:
            comptime if op == BinaryOp.SUB.value:
                out_ptr[i] = lhs - rhs
            else:
                comptime if op == BinaryOp.MUL.value:
                    out_ptr[i] = lhs * rhs
                else:
                    comptime if op == BinaryOp.DIV.value:
                        out_ptr[i] = lhs / rhs
                    else:
                        out_ptr[i] = apply_binary_typed_vec_static[dtype, 1, op](
                            SIMD[dtype, 1](lhs), SIMD[dtype, 1](rhs)
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


def _binary_same_shape_contig_typed_serial[
    dtype: DType
](
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    op: Int,
) raises:
    # Single-thread SIMD body. SIMD width derives from `dtype` at comptime.
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
    # Per-worker work budget: ELEMENTWISE_LIGHT_GRAIN bytes. Below that
    # we run the full reduction on one thread (saves ~1-10us spawn cost).
    # Above, each worker gets a contiguous slice and runs the same
    # SIMD body — embarrassingly parallel since outputs are disjoint.
    var byte_count = size * size_of[Scalar[dtype]]()
    var nworkers = worker_count_for_bytes(size, byte_count, ELEMENTWISE_LIGHT_GRAIN)
    if nworkers <= 1:
        _binary_same_shape_contig_typed_serial[dtype](lhs_ptr, rhs_ptr, out_ptr, size, op)
        return

    var chunk = ceildiv(size, nworkers)

    @parameter
    def chunk_worker(i: Int) raises:
        var start = i * chunk
        var end = start + chunk
        if end > size:
            end = size
        if start >= size:
            return
        _binary_same_shape_contig_typed_serial[dtype](
            lhs_ptr + start, rhs_ptr + start, out_ptr + start, end - start, op
        )

    sync_parallelize[chunk_worker](nworkers)


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
    comptime if op == BinaryOp.ADD.value:
        var i = 0
        while i + width <= size:
            out_ptr.store(i, array_ptr.load[width=width](i) + scalar_vec)
            i += width
        while i < size:
            out_ptr[i] = array_ptr[i] + scalar_value
            i += 1
        return
    comptime if op == BinaryOp.MUL.value:
        var i = 0
        while i + width <= size:
            out_ptr.store(i, array_ptr.load[width=width](i) * scalar_vec)
            i += width
        while i < size:
            out_ptr[i] = array_ptr[i] * scalar_value
            i += 1
        return
    var i = 0
    while i + width <= size:
        var array_vec = array_ptr.load[width=width](i)
        if scalar_on_left:
            out_ptr.store(i, apply_binary_typed_vec_static[dtype, width, op](scalar_vec, array_vec))
        else:
            out_ptr.store(i, apply_binary_typed_vec_static[dtype, width, op](array_vec, scalar_vec))
        i += width
    while i < size:
        var array = array_ptr[i]
        var lhs_v = SIMD[dtype, 1](array)
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
    if not scalar_on_left and op == BinaryOp.POWER.value:
        if scalar_value == Scalar[dtype](2):
            var i = 0
            while i + width <= size:
                var value = array_ptr.load[width=width](i)
                out_ptr.store(i, value * value)
                i += width
            while i < size:
                var value = array_ptr[i]
                out_ptr[i] = value * value
                i += 1
            return
        if scalar_value == Scalar[dtype](3):
            var i = 0
            while i + width <= size:
                var value = array_ptr.load[width=width](i)
                out_ptr.store(i, (value * value) * value)
                i += width
            while i < size:
                var value = array_ptr[i]
                out_ptr[i] = (value * value) * value
                i += 1
            return
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


def apply_unary_typed_vec_static[
    dtype: DType, width: Int, op: Int
](value: SIMD[dtype, width]) raises -> SIMD[dtype, width] where dtype.is_floating_point():
    comptime if op == UnaryOp.SIN.value:
        return sin(value)
    else:
        comptime if op == UnaryOp.COS.value:
            return cos(value)
        else:
            comptime if op == UnaryOp.EXP.value:
                return exp(value)
            else:
                comptime if op == UnaryOp.LOG.value:
                    return log(value)
                else:
                    comptime if op == UnaryOp.TANH.value:
                        return tanh(value)
                    else:
                        comptime if op == UnaryOp.SQRT.value:
                            return sqrt(value)
                        else:
                            comptime if op == UnaryOp.NEGATE.value:
                                return -value
                            else:
                                comptime if op == UnaryOp.POSITIVE.value:
                                    return value
                                else:
                                    comptime if op == UnaryOp.SQUARE.value:
                                        return value * value
                                    else:
                                        return apply_unary_typed_vec[dtype, width](value, op)


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


def binary_column_broadcast_contig_typed[
    dtype: DType
](
    matrix_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    column_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rows: Int,
    cols: Int,
    op: Int,
    column_on_left: Bool,
) raises:
    # Comptime-typed kernel for matrix⊕column and column⊕matrix where
    # `column` has shape (rows, 1). This is the attention/layer-norm
    # broadcast pattern after keepdims row reductions.
    comptime width = simd_width_of[dtype]()
    for i in range(rows):
        var column_vec = SIMD[dtype, width](column_ptr[i])
        var column_scalar = SIMD[dtype, 1](column_ptr[i])
        var j = 0
        while j + width <= cols:
            var matrix_index = i * cols + j
            var matrix_vec = matrix_ptr.load[width=width](matrix_index)
            if op == BinaryOp.ADD.value:
                out_ptr.store(matrix_index, matrix_vec + column_vec)
            elif column_on_left:
                out_ptr.store(
                    matrix_index,
                    apply_binary_typed_vec[dtype, width](column_vec, matrix_vec, op),
                )
            else:
                out_ptr.store(
                    matrix_index,
                    apply_binary_typed_vec[dtype, width](matrix_vec, column_vec, op),
                )
            j += width
        while j < cols:
            var matrix_index = i * cols + j
            var matrix_scalar = SIMD[dtype, 1](matrix_ptr[matrix_index])
            if op == BinaryOp.ADD.value:
                out_ptr[matrix_index] = matrix_ptr[matrix_index] + column_ptr[i]
            elif column_on_left:
                out_ptr[matrix_index] = apply_binary_typed_vec[dtype, 1](column_scalar, matrix_scalar, op)[0]
            else:
                out_ptr[matrix_index] = apply_binary_typed_vec[dtype, 1](matrix_scalar, column_scalar, op)[0]
            j += 1


def _unary_contig_typed_serial[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    op: Int,
) raises where dtype.is_floating_point():
    # Single-thread SIMD body. SIMD width derives from dtype.
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


def unary_contig_typed[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    op: Int,
) raises where dtype.is_floating_point():
    if try_unary_contig_typed_static[dtype](src_ptr, out_ptr, size, op):
        return
    # Per-worker work budget: ELEMENTWISE_HEAVY_GRAIN (256KB). Unary ops
    # like exp/log/sin/cos do enough per-element work to amortize spawn
    # cost earlier than binary add/mul. Cheap unary (neg/abs/square)
    # also fans out at this size — slightly under-utilized but correct;
    # an op-kind table refinement is a future improvement.
    var byte_count = size * size_of[Scalar[dtype]]()
    var nworkers = worker_count_for_bytes(size, byte_count, ELEMENTWISE_HEAVY_GRAIN)
    if nworkers <= 1:
        _unary_contig_typed_serial[dtype](src_ptr, out_ptr, size, op)
        return

    var chunk = ceildiv(size, nworkers)

    @parameter
    def chunk_worker(i: Int) raises:
        var start = i * chunk
        var end = start + chunk
        if end > size:
            end = size
        if start >= size:
            return
        _unary_contig_typed_serial[dtype](src_ptr + start, out_ptr + start, end - start, op)

    sync_parallelize[chunk_worker](nworkers)


def unary_contig_typed_static[
    dtype: DType, op: Int
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
) raises where dtype.is_floating_point():
    comptime width = simd_width_of[dtype]()
    var i = 0
    while i + width <= size:
        out_ptr.store(
            i,
            apply_unary_typed_vec_static[dtype, width, op](src_ptr.load[width=width](i)),
        )
        i += width
    while i < size:
        out_ptr[i] = apply_unary_typed_vec_static[dtype, 1, op](SIMD[dtype, 1](src_ptr[i]))[0]
        i += 1


def try_unary_contig_typed_static[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    op: Int,
) raises -> Bool where dtype.is_floating_point():
    if op == UnaryOp.SIN.value:
        unary_contig_typed_static[dtype, UnaryOp.SIN.value](src_ptr, out_ptr, size)
        return True
    if op == UnaryOp.COS.value:
        unary_contig_typed_static[dtype, UnaryOp.COS.value](src_ptr, out_ptr, size)
        return True
    if op == UnaryOp.EXP.value:
        unary_contig_typed_static[dtype, UnaryOp.EXP.value](src_ptr, out_ptr, size)
        return True
    if op == UnaryOp.LOG.value:
        unary_contig_typed_static[dtype, UnaryOp.LOG.value](src_ptr, out_ptr, size)
        return True
    if op == UnaryOp.TANH.value:
        unary_contig_typed_static[dtype, UnaryOp.TANH.value](src_ptr, out_ptr, size)
        return True
    if op == UnaryOp.SQRT.value:
        unary_contig_typed_static[dtype, UnaryOp.SQRT.value](src_ptr, out_ptr, size)
        return True
    if op == UnaryOp.NEGATE.value:
        unary_contig_typed_static[dtype, UnaryOp.NEGATE.value](src_ptr, out_ptr, size)
        return True
    if op == UnaryOp.POSITIVE.value:
        unary_contig_typed_static[dtype, UnaryOp.POSITIVE.value](src_ptr, out_ptr, size)
        return True
    if op == UnaryOp.SQUARE.value:
        unary_contig_typed_static[dtype, UnaryOp.SQUARE.value](src_ptr, out_ptr, size)
        return True
    return False


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
