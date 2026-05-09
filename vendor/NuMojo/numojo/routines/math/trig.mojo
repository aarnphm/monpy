# ===----------------------------------------------------------------------=== #
# NuMojo: Trigonometric routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Trigonometric routines for NuMojo (numojo.routines.math.trig).

Implements trigonometric and inverse trigonometric functions for NDArrays and Matrices.
"""

import std.math as math
from std.sys import simd_width_of

from numojo.core.ndarray import NDArray
from numojo.core.matrix import Matrix
from numojo.routines.math.misc import sqrt
from numojo.routines.math.arithmetic import fma

comptime _TRIG_ACOS = 0
comptime _TRIG_ASIN = 1
comptime _TRIG_ATAN = 2
comptime _TRIG_COS = 3
comptime _TRIG_SIN = 4
comptime _TRIG_TAN = 5


def _trig_simd[
    dtype: DType,
    width: Int,
    op: Int,
](value: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime assert dtype.is_floating_point(), "trigonometric ops require floating-point dtype"
    comptime if op == _TRIG_ACOS:
        return math.acos(value)
    elif op == _TRIG_ASIN:
        return math.asin(value)
    elif op == _TRIG_ATAN:
        return math.atan(value)
    elif op == _TRIG_COS:
        return math.cos(value)
    elif op == _TRIG_SIN:
        return math.sin(value)
    elif op == _TRIG_TAN:
        return math.tan(value)
    else:
        comptime assert False, "unknown NuMojo trigonometric op"


def _apply_unary[
    dtype: DType,
    op: Int,
](array: NDArray[dtype]) raises -> NDArray[dtype]:
    if not array.is_c_contiguous():
        return _apply_unary[dtype, op](array.contiguous())

    var result = array.deep_copy()
    result.size = array.size
    comptime width = simd_width_of[dtype]()
    for i in range(0, array.size - (array.size % width), width):
        result._buf.ptr.store(
            i, _trig_simd[dtype, width, op](array._buf.ptr.load[width=width](i))
        )
    for i in range(array.size - (array.size % width), array.size):
        result._buf.ptr[i] = _trig_simd[dtype, 1, op](array._buf.ptr[i])
    return result^


def _apply_unary[
    dtype: DType,
    op: Int,
](matrix: Matrix[dtype]) -> Matrix[dtype]:
    var result = Matrix[dtype](shape=matrix.shape, order=matrix.order())
    comptime width = simd_width_of[dtype]()
    for i in range(0, matrix.size - (matrix.size % width), width):
        result._buf.ptr.store(
            i, _trig_simd[dtype, width, op](matrix._buf.ptr.load[width=width](i))
        )
    for i in range(matrix.size - (matrix.size % width), matrix.size):
        result._buf.ptr[i] = _trig_simd[dtype, 1, op](matrix._buf.ptr[i])
    return result^


def _apply_atan2[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    comptime assert dtype.is_floating_point(), "atan2 requires floating-point dtype"
    if array1.shape != array2.shape:
        raise Error("Shape Mismatch error: shapes must match for this function")

    if not array1.is_c_contiguous():
        return _apply_atan2[dtype](array1.contiguous(), array2)
    if not array2.is_c_contiguous():
        return _apply_atan2[dtype](array1, array2.contiguous())

    var result = array1.deep_copy()
    result.size = array1.size
    comptime width = simd_width_of[dtype]()
    for i in range(0, array1.size - (array1.size % width), width):
        result._buf.ptr.store(
            i,
            math.atan2(
                array1._buf.ptr.load[width=width](i),
                array2._buf.ptr.load[width=width](i),
            ),
        )
    for i in range(array1.size - (array1.size % width), array1.size):
        result._buf.ptr[i] = math.atan2(array1._buf.ptr[i], array2._buf.ptr[i])
    return result^

# ===------------------------------------------------------------------------===#
# Inverse Trig (NDArray)
# ===------------------------------------------------------------------------===#


def acos[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply inverse cosine.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise acos of `array`.
    """
    return _apply_unary[dtype, _TRIG_ACOS](array)


def asin[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply inverse sine.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise asin of `array`.
    """
    return _apply_unary[dtype, _TRIG_ASIN](array)


def atan[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply inverse tangent.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise atan of `array`.
    """
    return _apply_unary[dtype, _TRIG_ATAN](array)


def atan2[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply inverse tangent with two arrays.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        The element-wise atan2 of `array1` and `array2`.

    References:
        https://en.wikipedia.org/wiki/Atan2.
    """
    return _apply_atan2[dtype](array1, array2)


# ===------------------------------------------------------------------------===#
# Inverse Trig (Matrix)
# ===------------------------------------------------------------------------===#


def arccos[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse cosine.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise acos of `A`.
    """
    return _apply_unary[dtype, _TRIG_ACOS](A)


def acos[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse cosine.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise acos of `A`.
    """
    return _apply_unary[dtype, _TRIG_ACOS](A)


def arcsin[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse sine.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise asin of `A`.
    """
    return _apply_unary[dtype, _TRIG_ASIN](A)


def asin[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse sine.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise asin of `A`.
    """
    return _apply_unary[dtype, _TRIG_ASIN](A)


def arctan[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse tangent.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise atan of `A`.
    """
    return _apply_unary[dtype, _TRIG_ATAN](A)


def atan[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply inverse tangent.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise atan of `A`.
    """
    return _apply_unary[dtype, _TRIG_ATAN](A)


# ===------------------------------------------------------------------------===#
# Trig (NDArray)
# ===------------------------------------------------------------------------===#


def cos[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply cosine.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise cos of `array`.
    """
    return _apply_unary[dtype, _TRIG_COS](array)


def sin[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply sine.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise sin of `array`.
    """
    return _apply_unary[dtype, _TRIG_SIN](array)


def tan[dtype: DType](array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply tangent.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.

    Returns:
        The element-wise tan of `array`.
    """
    return _apply_unary[dtype, _TRIG_TAN](array)


# ===------------------------------------------------------------------------===#
# Trig (Matrix)
# ===------------------------------------------------------------------------===#


def cos[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply cosine.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise cos of `A`.
    """
    return _apply_unary[dtype, _TRIG_COS](A)


def sin[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply sine.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise sin of `A`.
    """
    return _apply_unary[dtype, _TRIG_SIN](A)


def tan[dtype: DType](A: Matrix[dtype]) -> Matrix[dtype]:
    """
    Apply tangent.

    Parameters:
        dtype: The element type.

    Args:
        A: A Matrix.

    Returns:
        The element-wise tan of `A`.
    """
    return _apply_unary[dtype, _TRIG_TAN](A)


# ===------------------------------------------------------------------------===#
# Hypotenuse
# ===------------------------------------------------------------------------===#


def hypot[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply hypotenuse calculation to two arrays.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        The element-wise hypotenuse of `array1` and `array2`.
    """
    return hypot_fma[dtype](array1, array2)


def hypot_fma[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Apply hypotenuse calculation using fused multiply-add.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        The element-wise hypotenuse of `array1` and `array2`.
    """
    var array2_squared = fma[dtype](array2, array2, SIMD[dtype, 1](0))
    return sqrt[dtype](fma[dtype](array1, array1, array2_squared))
