"""Result-dtype helpers and shape broadcast.

Hosts:
  - `result_dtype_for_*` — thin wrappers around `domain.dtype_result_for_*`
    for unary, unary-preserve, binary, reduction, linalg-unary,
    linalg-binary. The unary-preserve flavour bool→int64 promotion is the
    only case that does work beyond the wrapper.
  - `broadcast_shape` — numpy-style broadcast shape resolution
    (right-aligned, size-1 expands, raises on size mismatch).
"""

from std.collections import List

from domain import (
    ArrayDType,
    dtype_result_for_binary,
    dtype_result_for_linalg,
    dtype_result_for_linalg_binary,
    dtype_result_for_reduction,
    dtype_result_for_unary,
)

from .accessors import Array


def result_dtype_for_unary(dtype_code: Int) -> Int:
    return dtype_result_for_unary(dtype_code)


def result_dtype_for_unary_preserve(dtype_code: Int) -> Int:
    if dtype_code == ArrayDType.BOOL.value:
        return ArrayDType.INT64.value
    return dtype_code


def result_dtype_for_binary(lhs_dtype: Int, rhs_dtype: Int, op: Int) -> Int:
    return dtype_result_for_binary(lhs_dtype, rhs_dtype, op)


def result_dtype_for_reduction(dtype_code: Int, op: Int) -> Int:
    return dtype_result_for_reduction(dtype_code, op)


def result_dtype_for_linalg(dtype_code: Int) -> Int:
    return dtype_result_for_linalg(dtype_code)


def result_dtype_for_linalg_binary(lhs_dtype: Int, rhs_dtype: Int) -> Int:
    return dtype_result_for_linalg_binary(lhs_dtype, rhs_dtype)


def broadcast_shape(lhs: Array, rhs: Array) raises -> List[Int]:
    var lhs_ndim = len(lhs.shape)
    var rhs_ndim = len(rhs.shape)
    var out_ndim = lhs_ndim
    if rhs_ndim > out_ndim:
        out_ndim = rhs_ndim
    var shape = List[Int]()
    for _ in range(out_ndim):
        shape.append(1)
    for out_axis in range(out_ndim - 1, -1, -1):
        var lhs_axis = out_axis - (out_ndim - lhs_ndim)
        var rhs_axis = out_axis - (out_ndim - rhs_ndim)
        var lhs_dim = 1
        var rhs_dim = 1
        if lhs_axis >= 0:
            lhs_dim = lhs.shape[lhs_axis]
        if rhs_axis >= 0:
            rhs_dim = rhs.shape[rhs_axis]
        if lhs_dim == rhs_dim:
            shape[out_axis] = lhs_dim
        elif lhs_dim == 1:
            shape[out_axis] = rhs_dim
        elif rhs_dim == 1:
            shape[out_axis] = lhs_dim
        else:
            raise Error("operands could not be broadcast together")
    return shape^
