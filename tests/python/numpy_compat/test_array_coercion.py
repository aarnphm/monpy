from __future__ import annotations

import monumpy as np
import pytest


def test_nested_sequences_infer_shape_and_dtype() -> None:
  arr = np.asarray([[1, 2, 3], [4, 5, 6]])

  assert arr.shape == (2, 3)
  assert arr.dtype == np.int64
  assert arr.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_float_values_promote_to_default_real_dtype() -> None:
  arr = np.asarray([1, 2.5, 3])

  assert arr.dtype == np.float64
  assert arr.tolist() == [1.0, 2.5, 3.0]


def test_bool_values_remain_bool_dtype() -> None:
  arr = np.asarray([True, False, True])

  assert arr.dtype == np.bool
  assert arr.tolist() == [True, False, True]


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
