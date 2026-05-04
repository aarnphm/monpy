from __future__ import annotations

import operator
from collections.abc import Callable

import monumpy as np
import numpy
import pytest

from _helpers import MONPY_TO_NUMPY_DTYPE, SUPPORTED_DTYPE_PAIRS, assert_same_values


PROMOTION_MATCH_CASES = [
  (lhs_dtype, lhs_numpy_dtype, rhs_dtype, rhs_numpy_dtype)
  for lhs_dtype, lhs_numpy_dtype in SUPPORTED_DTYPE_PAIRS
  for rhs_dtype, rhs_numpy_dtype in SUPPORTED_DTYPE_PAIRS
  if {lhs_dtype, rhs_dtype} != {np.int64, np.float32}
]


def values_for_dtype(monpy_dtype: np.DType) -> list[bool] | list[int] | list[float]:
  if monpy_dtype is np.bool:
    return [True, False]
  if monpy_dtype is np.int64:
    return [1, 2]
  return [1.0, 2.0]


@pytest.mark.parametrize(("lhs_dtype", "lhs_numpy_dtype", "rhs_dtype", "rhs_numpy_dtype"), PROMOTION_MATCH_CASES)
def test_supported_array_promotions_match_numpy(
  lhs_dtype: np.DType,
  lhs_numpy_dtype: type[numpy.generic],
  rhs_dtype: np.DType,
  rhs_numpy_dtype: type[numpy.generic],
) -> None:
  lhs = np.asarray(values_for_dtype(lhs_dtype), dtype=lhs_dtype)
  rhs = np.asarray(values_for_dtype(rhs_dtype), dtype=rhs_dtype)
  oracle_lhs = numpy.asarray(values_for_dtype(lhs_dtype), dtype=lhs_numpy_dtype)
  oracle_rhs = numpy.asarray(values_for_dtype(rhs_dtype), dtype=rhs_numpy_dtype)
  out = lhs + rhs
  expected = oracle_lhs + oracle_rhs

  assert MONPY_TO_NUMPY_DTYPE[out.dtype] == expected.dtype
  assert_same_values(out, expected)


@pytest.mark.xfail(
  strict=True,
  reason="monpy currently promotes int64/float32 arrays to float32 instead of NumPy float64",
)
@pytest.mark.parametrize(("lhs_dtype", "rhs_dtype"), [(np.int64, np.float32), (np.float32, np.int64)])
def test_int64_float32_array_promotion_tracks_numpy_gap(lhs_dtype: np.DType, rhs_dtype: np.DType) -> None:
  lhs = np.asarray(values_for_dtype(lhs_dtype), dtype=lhs_dtype)
  rhs = np.asarray(values_for_dtype(rhs_dtype), dtype=rhs_dtype)

  assert (lhs + rhs).dtype == np.float64


@pytest.mark.parametrize("scalar", [1, 1.5, True])
@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul])
def test_python_scalars_are_weak_for_float32_arrays(
  scalar: int | float | bool,
  op: Callable[[object, object], object],
) -> None:
  arr = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
  oracle = numpy.asarray([1.0, 2.0, 3.0], dtype=numpy.float32)
  out = op(arr, scalar)
  expected = op(oracle, scalar)

  assert out.dtype == np.float32
  assert_same_values(out, expected)


@pytest.mark.parametrize(
  "dtype",
  ["uint64", "complex128", "object", "str", "int8", "uint8", numpy.complex128, numpy.dtype("uint64")],
)
def test_unsupported_promotion_dtype_families_are_explicit_blockers(dtype: object) -> None:
  with pytest.raises(NotImplementedError, match="unsupported dtype"):
    np.asarray([1], dtype=dtype)


@pytest.mark.parametrize("value", [2**70, -(2**70)])
def test_python_integer_overflow_semantics_are_deferred_for_float32(value: int) -> None:
  arr = np.asarray([1.0], dtype=np.float32)
  out = arr + value

  assert out.dtype == np.float32
