"""Scalar f64 op-dispatchers — universal fallback for any binary/unary op
whose operands have been promoted/decompressed into f64.

These are used by:
  - complex kernels (after Smith division)
  - boxed scalar paths
  - reduction gather sites that don't have a typed-SIMD specialisation
  - the LayoutIter strided fallbacks in `create/elementwise_ops`

No `Array` dependency — pure scalar in/out plus the int op-code.
"""

from std.math import (
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

from domain import BinaryOp, UnaryOp


def apply_binary_f64(lhs: Float64, rhs: Float64, op: Int) raises -> Float64:
    # Edge-case map (numpy parity):
    # - MAXIMUM/MINIMUM propagate NaN: NaN ∗ x ⇒ NaN. Distinguishes from
    #   FMAX/FMIN which treat NaN as missing (NaN ∗ x ⇒ x).
    # - FLOOR_DIV is `floor(a/b)` here; integer dtypes use Python `//` upstream.
    # - MOD follows numpy's `a - floor(a/b)·b` (matches Python `%`); LAPACK's
    #   fmod truncates toward zero, which we don't want.
    # - POWER routes through SIMD[f64, 1].__pow__ → libm `pow`.
    # - DIV by zero on float: IEEE returns ±inf or NaN, no raise.
    if op == BinaryOp.ADD.value:
        return lhs + rhs
    if op == BinaryOp.SUB.value:
        return lhs - rhs
    if op == BinaryOp.MUL.value:
        return lhs * rhs
    if op == BinaryOp.DIV.value:
        return lhs / rhs
    if op == BinaryOp.FLOOR_DIV.value:
        return math_floor(lhs / rhs)
    if op == BinaryOp.MOD.value:
        var q = math_floor(lhs / rhs)
        return lhs - q * rhs
    if op == BinaryOp.POWER.value:
        var v = SIMD[DType.float64, 1](lhs).__pow__(SIMD[DType.float64, 1](rhs))
        return v[0]
    if op == BinaryOp.MAXIMUM.value:
        if isnan(lhs) or isnan(rhs):
            return nan[DType.float64]()
        return lhs if lhs > rhs else rhs
    if op == BinaryOp.MINIMUM.value:
        if isnan(lhs) or isnan(rhs):
            return nan[DType.float64]()
        return lhs if lhs < rhs else rhs
    if op == BinaryOp.FMAX.value:
        if isnan(lhs):
            return rhs
        if isnan(rhs):
            return lhs
        return lhs if lhs > rhs else rhs
    if op == BinaryOp.FMIN.value:
        if isnan(lhs):
            return rhs
        if isnan(rhs):
            return lhs
        return lhs if lhs < rhs else rhs
    if op == BinaryOp.ARCTAN2.value:
        return atan2(SIMD[DType.float64, 1](lhs), SIMD[DType.float64, 1](rhs))[0]
    if op == BinaryOp.HYPOT.value:
        return hypot(SIMD[DType.float64, 1](lhs), SIMD[DType.float64, 1](rhs))[0]
    if op == BinaryOp.COPYSIGN.value:
        return copysign(SIMD[DType.float64, 1](lhs), SIMD[DType.float64, 1](rhs))[0]
    raise Error("unknown binary op")


def apply_unary_f64(value: Float64, op: Int) raises -> Float64:
    if op == UnaryOp.SIN.value:
        return sin(value)
    if op == UnaryOp.COS.value:
        return cos(value)
    if op == UnaryOp.EXP.value:
        return exp(value)
    if op == UnaryOp.LOG.value:
        if isnan(value):
            return value
        if isinf(value):
            if value < 0.0:
                return nan[DType.float64]()
            return value
        return log(value)
    if op == UnaryOp.TAN.value:
        return tan(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.ARCSIN.value:
        return asin(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.ARCCOS.value:
        return acos(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.ARCTAN.value:
        return atan(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.SINH.value:
        return sinh(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.COSH.value:
        return cosh(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.TANH.value:
        return tanh(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.LOG1P.value:
        return log1p(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.LOG2.value:
        return log2(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.LOG10.value:
        return log10(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.EXP2.value:
        return exp2(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.EXPM1.value:
        return expm1(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.SQRT.value:
        return sqrt(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.CBRT.value:
        return cbrt(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.DEG2RAD.value:
        return value * 0.017453292519943295  # pi/180
    if op == UnaryOp.RAD2DEG.value:
        return value * 57.29577951308232  # 180/pi
    if op == UnaryOp.RECIPROCAL.value:
        return 1.0 / value
    if op == UnaryOp.NEGATE.value:
        return -value
    if op == UnaryOp.POSITIVE.value:
        return value
    if op == UnaryOp.ABS.value:
        return -value if value < 0.0 else value
    if op == UnaryOp.SQUARE.value:
        return value * value
    if op == UnaryOp.SIGN.value:
        if isnan(value):
            return nan[DType.float64]()
        if value > 0.0:
            return 1.0
        if value < 0.0:
            return -1.0
        return 0.0
    if op == UnaryOp.FLOOR.value:
        return math_floor(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.CEIL.value:
        return math_ceil(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.TRUNC.value:
        return math_trunc(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.RINT.value:
        return math_round(SIMD[DType.float64, 1](value))[0]
    if op == UnaryOp.LOGICAL_NOT.value:
        return 1.0 if value == 0.0 else 0.0
    raise Error("unknown unary op")
