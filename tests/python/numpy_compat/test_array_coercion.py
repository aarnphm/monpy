from __future__ import annotations

import monumpy as np
import numpy
import pytest
from _helpers import SUPPORTED_DTYPE_PAIRS, assert_same_shape_dtype, assert_same_values


@pytest.mark.parametrize(
  ("obj", "expected_shape", "expected_dtype", "expected_values"),
  [
    ([], (0,), np.float64, []),
    (True, (), np.bool, True),
    (7, (), np.int64, 7),
    (3.5, (), np.float64, 3.5),
    ([1, 2, 3], (3,), np.int64, [1, 2, 3]),
    ([[1, 2, 3], [4, 5, 6]], (2, 3), np.int64, [[1, 2, 3], [4, 5, 6]]),
  ],
)
def test_array_like_inputs_infer_shape_dtype_and_values(
  obj: object,
  expected_shape: tuple[int, ...],
  expected_dtype: np.DType,
  expected_values: object,
) -> None:
  arr = np.asarray(obj)

  assert arr.shape == expected_shape
  assert arr.dtype == expected_dtype
  assert arr.tolist() == expected_values


def test_float_values_promote_to_default_real_dtype() -> None:
  arr = np.asarray([1, 2.5, 3])

  assert arr.dtype == np.float64
  assert arr.tolist() == [1.0, 2.5, 3.0]


def test_bool_values_remain_bool_dtype() -> None:
  arr = np.asarray([True, False, True])

  assert arr.dtype == np.bool
  assert arr.tolist() == [True, False, True]


@pytest.mark.parametrize("monpy_dtype, numpy_dtype", SUPPORTED_DTYPE_PAIRS)
def test_explicit_supported_dtype_casts_match_numpy(monpy_dtype: np.DType, numpy_dtype: type[numpy.generic]) -> None:
  values = [False, True, True] if monpy_dtype is np.bool else [1, 2, 3]
  arr = np.asarray(values, dtype=monpy_dtype)
  oracle = numpy.asarray(values, dtype=numpy_dtype)

  assert_same_shape_dtype(arr, oracle)
  assert_same_values(arr, oracle)


def test_scalar_array_is_zero_dimensional() -> None:
  arr = np.asarray(3.5, dtype=np.float32)

  assert arr.shape == ()
  assert arr.ndim == 0
  assert float(arr) == pytest.approx(3.5)


def test_ragged_nested_sequences_raise() -> None:
  with pytest.raises(ValueError, match="ragged"):
    np.asarray([[1], [2, 3]])


def test_asarray_returns_same_object_without_copy_or_cast() -> None:
  arr = np.asarray([1, 2, 3])

  assert np.asarray(arr) is arr
  assert np.asarray(arr, dtype=np.float32) is not arr


def test_array_and_asarray_copy_rules_for_existing_monpy_arrays() -> None:
  arr = np.asarray([1, 2, 3])

  assert np.array(arr, copy=False) is arr
  assert np.asarray(arr, copy=False) is arr
  assert np.array(arr, copy=True) is not arr
  assert np.asarray(arr, copy=True) is not arr


def test_astype_copy_false_keeps_identity_for_same_dtype() -> None:
  arr = np.asarray([1, 2, 3], dtype=np.int64)

  assert arr.astype(np.int64, copy=False) is arr
  assert arr.astype(np.float32, copy=False) is not arr


@pytest.mark.parametrize(
  "dtype",
  [
    "int8",
    "uint64",
    "complex128",
    "object",
    "str",
    numpy.float32,
    numpy.dtype("int64"),
    [("field", numpy.int64)],
  ],
)
def test_unsupported_dtype_requests_are_explicit_blockers(dtype: object) -> None:
  with pytest.raises(NotImplementedError, match="unsupported dtype"):
    np.asarray([1], dtype=dtype)


@pytest.mark.parametrize("obj", [[1 + 2j], ["monpy"], [object()]])
def test_unsupported_array_value_types_are_explicit_blockers(obj: object) -> None:
  with pytest.raises(NotImplementedError, match="unsupported array input type"):
    np.asarray(obj)


def test_numpy_array_input_is_an_explicit_import_gap() -> None:
  with pytest.raises(NotImplementedError, match="unsupported array input type"):
    np.asarray(numpy.asarray([1, 2, 3]))
