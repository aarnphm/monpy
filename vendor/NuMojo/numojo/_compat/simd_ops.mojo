# ===----------------------------------------------------------------------=== #
# NuMojo: SIMD compatibility helpers
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""SIMD helpers for callable-parameter compatibility.

Current Mojo accepts direct calls to SIMD comparison methods, but method
references such as `SIMD.gt` no longer satisfy NuMojo's generic callback
parameters. These wrappers keep the old higher-order call sites explicit.
"""

import std.math as math
import std.math.math as stdlib_math
from std.math import max as builtin_max
from std.math import min as builtin_min
from std.utils.numerics import nextafter as builtin_nextafter


@always_inline
def simd_invert[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width]:
    return value.__invert__()


@always_inline
def simd_gt[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[DType.bool, width]:
    return lhs.gt(rhs)


@always_inline
def simd_ge[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[DType.bool, width]:
    return lhs.ge(rhs)


@always_inline
def simd_lt[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[DType.bool, width]:
    return lhs.lt(rhs)


@always_inline
def simd_le[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[DType.bool, width]:
    return lhs.le(rhs)


@always_inline
def simd_eq[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[DType.bool, width]:
    return lhs.eq(rhs)


@always_inline
def simd_ne[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[DType.bool, width]:
    return lhs.ne(rhs)


@always_inline
def simd_add[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[dtype, width]:
    return lhs + rhs


@always_inline
def simd_sub[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[dtype, width]:
    return lhs - rhs


@always_inline
def simd_mul[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[dtype, width]:
    return lhs * rhs


@always_inline
def simd_div[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[dtype, width]:
    return lhs / rhs


@always_inline
def simd_floor_div[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[dtype, width]:
    return lhs.__floordiv__(rhs)


@always_inline
def simd_mod[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[dtype, width]:
    return lhs % rhs


@always_inline
def simd_abs[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width]:
    return value.__abs__()


@always_inline
def simd_floor[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width]:
    return value.__floor__()


@always_inline
def simd_ceil[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width]:
    return value.__ceil__()


@always_inline
def simd_trunc[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width]:
    return value.__trunc__()


@always_inline
def simd_round[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width]:
    return value.__round__()


@always_inline
def simd_min[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[dtype, width]:
    return builtin_min(lhs, rhs)


@always_inline
def simd_max[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[dtype, width]:
    return builtin_max(lhs, rhs)


@always_inline
def simd_exp[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return math.exp(value)


@always_inline
def simd_exp2[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return math.exp2(value)


@always_inline
def simd_expm1[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return math.expm1(value)


@always_inline
def simd_log[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return math.log(value)


@always_inline
def simd_log2[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return math.log2(value)


@always_inline
def simd_log10[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return math.log10(value)


@always_inline
def simd_log1p[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return math.log1p(value)


@always_inline
def simd_acosh[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return math.acosh(value)


@always_inline
def simd_asinh[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return math.asinh(value)


@always_inline
def simd_atanh[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return math.atanh(value)


@always_inline
def simd_cosh[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return math.cosh(value)


@always_inline
def simd_sinh[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return math.sinh(value)


@always_inline
def simd_tanh[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return math.tanh(value)


@always_inline
def simd_cbrt[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return stdlib_math.cbrt(value)


@always_inline
def simd_sqrt[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return stdlib_math.sqrt(value)


@always_inline
def simd_rsqrt[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return stdlib_math.sqrt(SIMD.__truediv__(SIMD[dtype, width](1), value))


@always_inline
def simd_scalb[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return stdlib_math.scalb(lhs, rhs)


@always_inline
def simd_copysign[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return math.copysign(lhs, rhs)


@always_inline
def simd_nextafter[dtype: DType, width: Int](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) capturing -> SIMD[dtype, width] where dtype.is_floating_point():
    return builtin_nextafter(lhs, rhs)


@always_inline
def simd_isinf[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[DType.bool, width] where dtype.is_floating_point():
    return math.isinf(value)


@always_inline
def simd_isfinite[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[DType.bool, width] where dtype.is_floating_point():
    return math.isfinite(value)


@always_inline
def simd_isnan[dtype: DType, width: Int](
    value: SIMD[dtype, width],
) capturing -> SIMD[DType.bool, width] where dtype.is_floating_point():
    return math.isnan(value)
