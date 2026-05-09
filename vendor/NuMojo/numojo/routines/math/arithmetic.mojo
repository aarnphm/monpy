# ===----------------------------------------------------------------------=== #
# NuMojo: Arithmetic routines
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Arithmetic routines for NuMojo (numojo.routines.math.arithmetic).

Implements addition, subtraction, multiplication, division, floor division, fused multiply-add, and remainder helpers for NDArrays.
"""

from std.utils import Variant
from std.sys import simd_width_of

from numojo.core.ndarray import NDArray

comptime _OP_ADD = 0
comptime _OP_SUB = 1
comptime _OP_MOD = 2
comptime _OP_MUL = 3
comptime _OP_DIV = 4
comptime _OP_FLOORDIV = 5


def _binary_simd[
    dtype: DType,
    width: Int,
    op: Int,
](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    comptime if op == _OP_ADD:
        return lhs + rhs
    elif op == _OP_SUB:
        return lhs - rhs
    elif op == _OP_MOD:
        return lhs % rhs
    elif op == _OP_MUL:
        return lhs * rhs
    elif op == _OP_DIV:
        return lhs / rhs
    elif op == _OP_FLOORDIV:
        return lhs // rhs
    else:
        comptime assert False, "unknown NuMojo arithmetic op"


def _apply_binary[
    dtype: DType,
    op: Int,
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    if array1.shape != array2.shape:
        raise Error("Shape Mismatch error: shapes must match for this function")

    if not array1.is_c_contiguous():
        return _apply_binary[dtype, op](array1.contiguous(), array2)
    if not array2.is_c_contiguous():
        return _apply_binary[dtype, op](array1, array2.contiguous())

    var result = array1.deep_copy()
    result.size = array1.size
    comptime width = simd_width_of[dtype]()
    for i in range(0, array1.size - (array1.size % width), width):
        result._buf.ptr.store(
            i,
            _binary_simd[dtype, width, op](
                array1._buf.ptr.load[width=width](i),
                array2._buf.ptr.load[width=width](i),
            ),
        )
    for i in range(array1.size - (array1.size % width), array1.size):
        result._buf.ptr[i] = _binary_simd[dtype, 1, op](
            array1._buf.ptr[i], array2._buf.ptr[i]
        )
    return result^


def _apply_binary[
    dtype: DType,
    op: Int,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    if not array.is_c_contiguous():
        return _apply_binary[dtype, op](array.contiguous(), scalar)

    var result = array.deep_copy()
    result.size = array.size
    comptime width = simd_width_of[dtype]()
    for i in range(0, array.size - (array.size % width), width):
        result._buf.ptr.store(
            i,
            _binary_simd[dtype, width, op](
                array._buf.ptr.load[width=width](i), SIMD[dtype, width](scalar)
            ),
        )
    for i in range(array.size - (array.size % width), array.size):
        result._buf.ptr[i] = _binary_simd[dtype, 1, op](array._buf.ptr[i], scalar)
    return result^


def _apply_binary[
    dtype: DType,
    op: Int,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    if not array.is_c_contiguous():
        return _apply_binary[dtype, op](scalar, array.contiguous())

    var result = array.deep_copy()
    result.size = array.size
    comptime width = simd_width_of[dtype]()
    for i in range(0, array.size - (array.size % width), width):
        result._buf.ptr.store(
            i,
            _binary_simd[dtype, width, op](
                SIMD[dtype, width](scalar), array._buf.ptr.load[width=width](i)
            ),
        )
    for i in range(array.size - (array.size % width), array.size):
        result._buf.ptr[i] = _binary_simd[dtype, 1, op](scalar, array._buf.ptr[i])
    return result^


def _apply_fma[
    dtype: DType,
](
    array1: NDArray[dtype], array2: NDArray[dtype], array3: NDArray[dtype]
) raises -> NDArray[dtype]:
    if array1.shape != array2.shape or array1.shape != array3.shape:
        raise Error("Shape Mismatch error: shapes must match for this function")

    if not array1.is_c_contiguous():
        return _apply_fma[dtype](array1.contiguous(), array2, array3)
    if not array2.is_c_contiguous():
        return _apply_fma[dtype](array1, array2.contiguous(), array3)
    if not array3.is_c_contiguous():
        return _apply_fma[dtype](array1, array2, array3.contiguous())

    var result = array1.deep_copy()
    result.size = array1.size
    comptime width = simd_width_of[dtype]()
    for i in range(0, array1.size - (array1.size % width), width):
        result._buf.ptr.store(
            i,
            array1._buf.ptr.load[width=width](i)
            * array2._buf.ptr.load[width=width](i)
            + array3._buf.ptr.load[width=width](i),
        )
    for i in range(array1.size - (array1.size % width), array1.size):
        result._buf.ptr[i] = array1._buf.ptr[i] * array2._buf.ptr[i] + array3._buf.ptr[i]
    return result^


def _apply_fma[
    dtype: DType,
](
    array1: NDArray[dtype], array2: NDArray[dtype], scalar: Scalar[dtype]
) raises -> NDArray[dtype]:
    if array1.shape != array2.shape:
        raise Error("Shape Mismatch error: shapes must match for this function")

    if not array1.is_c_contiguous():
        return _apply_fma[dtype](array1.contiguous(), array2, scalar)
    if not array2.is_c_contiguous():
        return _apply_fma[dtype](array1, array2.contiguous(), scalar)

    var result = array1.deep_copy()
    result.size = array1.size
    comptime width = simd_width_of[dtype]()
    for i in range(0, array1.size - (array1.size % width), width):
        result._buf.ptr.store(
            i,
            array1._buf.ptr.load[width=width](i)
            * array2._buf.ptr.load[width=width](i)
            + SIMD[dtype, width](scalar),
        )
    for i in range(array1.size - (array1.size % width), array1.size):
        result._buf.ptr[i] = array1._buf.ptr[i] * array2._buf.ptr[i] + scalar
    return result^

# ===------------------------------------------------------------------------===#
# Addition
# ===------------------------------------------------------------------------===#


def add[
    dtype: DType,
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform addition on two arrays.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        The element-wise sum of `array1` and`array2`.
    """
    return _apply_binary[dtype, _OP_ADD](array1, array2)


def add[
    dtype: DType,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform addition on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
        scalar: A Scalar.

    Returns:
        The element-wise sum of array and scalar.
    """
    return _apply_binary[dtype, _OP_ADD](array, scalar)


def add[
    dtype: DType,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform addition on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        scalar: A Scalar.
        array: A NDArray.

    Returns:
        The element-wise sum of scalar and array.
    """
    return _apply_binary[dtype, _OP_ADD](scalar, array)


def add[
    dtype: DType,
](var *values: Variant[NDArray[dtype], Scalar[dtype]]) raises -> NDArray[dtype]:
    """
    Perform addition on a list of arrays and a scalars.

    Parameters:
        dtype: The element type.

    Args:
        values: A list of arrays or Scalars to be added.

    Raises:
        Error: If there are no arrays in the input values.

    Returns:
        The element-wise sum of `array1` and`array2`.
    """
    raise Error(
        "math:arithmetic:add(*values) is disabled in MonPy's vendored NuMojo "
        "compatibility patch"
    )


# ===------------------------------------------------------------------------===#
# Subtraction
# ===------------------------------------------------------------------------===#


def sub[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on two arrays.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        The element-wise difference of `array1` and`array2`.
    """
    return _apply_binary[dtype, _OP_SUB](array1, array2)


def sub[
    dtype: DType,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
        scalar: A Scalar.

    Returns:
        The element-wise difference of array and scalar.
    """
    return _apply_binary[dtype, _OP_SUB](array, scalar)


def sub[
    dtype: DType,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform subtraction on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        scalar: A Scalar.
        array: A NDArray.

    Returns:
        The element-wise difference of scalar and array.
    """
    return _apply_binary[dtype, _OP_SUB](scalar, array)


# ===------------------------------------------------------------------------===#
# Modulo
# ===------------------------------------------------------------------------===#


def mod[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise modulo of array1 and array2.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Returns:
        A NDArray equal to array1 % array2.
    """
    return _apply_binary[dtype, _OP_MOD](array1, array2)


def mod[
    dtype: DType,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise modulo between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
        scalar: A Scalar.

    Returns:
        A NDArray equal to array % scalar.
    """
    return _apply_binary[dtype, _OP_MOD](array, scalar)


def mod[
    dtype: DType,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise modulo between a scalar and an array.

    Parameters:
        dtype: The element type.

    Args:
        scalar: A Scalar.
        array: A NDArray.

    Returns:
        A NDArray equal to scalar % array.
    """
    return _apply_binary[dtype, _OP_MOD](scalar, array)


# ===------------------------------------------------------------------------===#
# Multiplication
# ===------------------------------------------------------------------------===#


def mul[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise product of array1 and array2.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        A NDArray equal to array1*array2.
    """
    return _apply_binary[dtype, _OP_MUL](array1, array2)


def mul[
    dtype: DType,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform multiplication on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
        scalar: A Scalar.

    Returns:
        The element-wise product of array and scalar.
    """
    return _apply_binary[dtype, _OP_MUL](array, scalar)


def mul[
    dtype: DType,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform multiplication on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        scalar: A Scalar.
        array: A NDArray.

    Returns:
        The element-wise product of scalar and array.
    """
    return _apply_binary[dtype, _OP_MUL](scalar, array)


def mul[
    dtype: DType,
](var *values: Variant[NDArray[dtype], Scalar[dtype]]) raises -> NDArray[dtype]:
    """
    Perform multiplication on a list of arrays an arrays and a scalars.

    Parameters:
        dtype: The element type.

    Args:
        values: A list of arrays or Scalars to be added.

    Raises:
        Error: If there are no arrays in the input values.

    Returns:
        The element-wise product of `array1` and`array2`.
    """
    raise Error(
        "math:arithmetic:mul(*values) is disabled in MonPy's vendored NuMojo "
        "compatibility patch"
    )


# ===------------------------------------------------------------------------===#
# Division
# ===------------------------------------------------------------------------===#


def div[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise quotient of array1 and array2.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        A NDArray equal to array1/array2.
    """
    return _apply_binary[dtype, _OP_DIV](array1, array2)


def div[
    dtype: DType,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform true division on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
        scalar: A Scalar.

    Returns:
        The element-wise quotient of array and scalar.
    """
    return _apply_binary[dtype, _OP_DIV](array, scalar)


def div[
    dtype: DType,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform true division between a scalar and an array.

    Parameters:
        dtype: The element type.

    Args:
        scalar: A Scalar.
        array: A NDArray.

    Returns:
        The element-wise quotient of scalar and array.
    """
    return _apply_binary[dtype, _OP_DIV](scalar, array)


# ===------------------------------------------------------------------------===#
# Floor Division
# ===------------------------------------------------------------------------===#


def floor_div[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise quotient of array1 and array2.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        A NDArray equal to array1/array2.
    """
    return _apply_binary[dtype, _OP_FLOORDIV](array1, array2)


def floor_div[
    dtype: DType,
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Perform true division on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
        scalar: A Scalar.

    Returns:
        The element-wise quotient of array and scalar.
    """
    return _apply_binary[dtype, _OP_FLOORDIV](array, scalar)


def floor_div[
    dtype: DType,
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Perform true division on between an array and a scalar.

    Parameters:
        dtype: The element type.

    Args:
        scalar: A Scalar.
        array: A NDArray.

    Returns:
        The element-wise quotient of scalar and array.
    """
    return _apply_binary[dtype, _OP_FLOORDIV](scalar, array)


# ===------------------------------------------------------------------------===#
# Fused Multiply-Add
# ===------------------------------------------------------------------------===#


def fma[
    dtype: DType
](
    array1: NDArray[dtype], array2: NDArray[dtype], array3: NDArray[dtype]
) raises -> NDArray[dtype]:
    """
    Apply a SIMD level fuse multiply add function of three variables and one return to a NDArray.

    Constraints:
        Both arrays must have the same shape.

    Parameters:
        dtype: The element type.


    Args:
        array1: A NDArray.
        array2: A NDArray.
        array3: A NDArray.

    Returns:
        A a new NDArray that is NDArray with the function func applied.
    """
    # TODO: Support passing through the FastMathFlag parameter
    # For now, FastMathFlag.CONTRACT is was default prior to this error.
    return _apply_fma[dtype](array1, array2, array3)


def fma[
    dtype: DType
](
    array1: NDArray[dtype], array2: NDArray[dtype], simd: SIMD[dtype, 1]
) raises -> NDArray[dtype]:
    """
    Apply a SIMD level fuse multiply add function of three variables and one return to a NDArray.

    Constraints:
        Both arrays must have the same shape

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.
        simd: A SIMD[dtype,1] value to be added.

    Returns:
        A a new NDArray that is NDArray with the function func applied.
    """
    return _apply_fma[dtype](array1, array2, simd)


# ===------------------------------------------------------------------------===#
# Remainder
# ===------------------------------------------------------------------------===#


def remainder[
    dtype: DType
](array1: NDArray[dtype], array2: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise remainders of NDArray.

    Parameters:
        dtype: The element type.

    Args:
        array1: A NDArray.
        array2: A NDArray.

    Constraints:
        Both arrays must have the same shapes.

    Returns:
        A NDArray equal to array1//array2.
    """
    return _apply_binary[dtype, _OP_MOD](array1, array2)


def remainder[
    dtype: DType
](array: NDArray[dtype], scalar: Scalar[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise remainders of NDArray.

    Parameters:
        dtype: The element type.

    Args:
        array: A NDArray.
        scalar: A scalar.

    Returns:
        A NDArray equal to array//scalar.
    """
    return _apply_binary[dtype, _OP_MOD](array, scalar)


def remainder[
    dtype: DType
](scalar: Scalar[dtype], array: NDArray[dtype]) raises -> NDArray[dtype]:
    """
    Element-wise remainders of NDArray.

    Parameters:
        dtype: The element type.

    Args:
        scalar: A scalar.
        array: A NDArray.

    Returns:
        A NDArray equal to scalar//array.
    """
    return _apply_binary[dtype, _OP_MOD](scalar, array)
