from __future__ import annotations

import monumpy as np
import numpy
import pytest
from _helpers import assert_same_shape_dtype, assert_same_values


def test_zero_dimensional_empty_index_returns_scalar() -> None:
  arr = np.asarray(42, dtype=np.int64)

  assert arr[()] == 42


def test_empty_index_for_non_scalar_is_error() -> None:
  arr = np.asarray([1, 2, 3])

  with pytest.raises(IndexError, match="empty index"):
    _ = arr[()]


def test_integer_index_drops_axis() -> None:
  arr = np.asarray([[1, 2, 3], [4, 5, 6]])

  assert arr[1].shape == (3,)
  assert arr[1].tolist() == [4, 5, 6]
  assert arr[1, 2] == 6


def test_negative_indices_and_reversed_slices() -> None:
  arr = np.arange(6, dtype=np.int64).reshape(2, 3)

  assert arr[-1, -1] == 5
  assert arr[:, ::-1].tolist() == [[2, 1, 0], [5, 4, 3]]


def test_slice_stop_is_respected() -> None:
  arr = np.arange(10, dtype=np.int64)

  assert arr[2:7:2].tolist() == [2, 4, 6]
  assert arr[8:2:-2].tolist() == [8, 6, 4]


def test_rank1_full_reverse_slice_matches_numpy_and_shares_storage() -> None:
  arr = np.arange(6, dtype=np.float32)
  oracle = numpy.arange(6, dtype=numpy.float32)

  view = arr[::-1]
  expected = oracle[::-1]

  assert_same_shape_dtype(view, expected)
  assert_same_values(view, expected)
  assert view.strides == expected.strides

  view[1] = -3
  expected[1] = -3

  assert_same_values(arr, oracle)


def test_slice_step_zero_raises() -> None:
  arr = np.arange(10, dtype=np.int64)

  with pytest.raises(ValueError, match="slice step"):
    _ = arr[::0]


def test_ellipsis_expands_to_full_slices() -> None:
  arr = np.arange(6, dtype=np.int64).reshape(2, 3)

  assert arr[..., 1].tolist() == [1, 4]


def test_multiple_ellipsis_and_too_many_indices_raise() -> None:
  arr = np.arange(6, dtype=np.int64).reshape(2, 3)

  with pytest.raises(IndexError, match="single ellipsis"):
    _ = arr[..., ...]
  with pytest.raises(IndexError, match="too many indices"):
    _ = arr[0, 0, 0]


def test_scalar_assignment_mutates_array() -> None:
  arr = np.arange(4, dtype=np.int64)
  arr[1] = 99

  assert arr.tolist() == [0, 99, 2, 3]


def test_slice_assignment_mutates_base_array() -> None:
  arr = np.arange(15, dtype=np.int64).reshape(3, 5)
  arr[1:, ::2] = -99

  assert arr.tolist() == [[0, 1, 2, 3, 4], [-99, 6, -99, 8, -99], [-99, 11, -99, 13, -99]]


def test_slice_assignment_from_array_matches_numpy() -> None:
  arr = np.arange(6, dtype=np.int64)
  oracle = numpy.arange(6, dtype=numpy.int64)

  arr[::2] = np.asarray([-1, -2, -3], dtype=np.int64)
  oracle[::2] = numpy.asarray([-1, -2, -3], dtype=numpy.int64)

  assert_same_values(arr, oracle)


def test_newaxis_indexing_matches_numpy_and_shares_storage() -> None:
  arr = np.arange(6, dtype=np.int64).reshape(2, 3)
  oracle = numpy.arange(6, dtype=numpy.int64).reshape(2, 3)

  full_view = arr[:, None, :]
  full_expected = oracle[:, None, :]

  assert_same_shape_dtype(full_view, full_expected)
  assert_same_values(full_view, full_expected)
  assert full_view.strides == full_expected.strides

  full_view[1, 0, 1] = -5
  full_expected[1, 0, 1] = -5

  assert_same_values(arr, oracle)

  view = arr[:, None, ::-1]
  expected = oracle[:, None, ::-1]

  assert_same_shape_dtype(view, expected)
  assert_same_values(view, expected)
  assert view.strides == expected.strides

  view[1, 0, 1] = -7
  oracle[:, None, ::-1][1, 0, 1] = -7

  assert_same_values(arr, oracle)


def test_expand_dims_matches_numpy_for_supported_axes() -> None:
  arr = np.arange(6, dtype=np.float64).reshape(2, 3).T
  oracle = numpy.arange(6, dtype=numpy.float64).reshape(2, 3).T

  assert_same_shape_dtype(np.expand_dims(arr, axis=1), numpy.expand_dims(oracle, axis=1))
  assert_same_values(np.expand_dims(arr, axis=(0, -1)), numpy.expand_dims(oracle, axis=(0, -1)))


@pytest.mark.parametrize(
  ("index", "expected"),
  [
    (np.asarray([True, False, True], dtype=np.bool), [1, 3]),
    (np.asarray([0, 2], dtype=np.int64), [1, 3]),
    ([0, 2], [1, 3]),
  ],
)
def test_boolean_and_integer_array_indexing_returns_gathered_values(
  index: object, expected: list[int]
) -> None:
  arr = np.asarray([1, 2, 3])
  out = arr[index]
  assert out.tolist() == expected


def test_boolean_indexing_2d_matches_numpy() -> None:
  arr = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
  oracle = numpy.asarray([[1, 2, 3], [4, 5, 6]], dtype=numpy.int64)
  mask = np.asarray([[True, False, True], [False, True, False]], dtype=np.bool)
  oracle_mask = numpy.asarray([[True, False, True], [False, True, False]])
  numpy.testing.assert_array_equal(numpy.asarray(arr[mask]), oracle[oracle_mask])


def test_fancy_index_assignment_writes_back() -> None:
  arr = np.asarray([0, 0, 0, 0, 0], dtype=np.int64)
  arr[np.asarray([0, 2, 4], dtype=np.int64)] = np.asarray([10, 20, 30], dtype=np.int64)
  assert arr.tolist() == [10, 0, 20, 0, 30]


def test_boolean_assignment_writes_back() -> None:
  arr = np.asarray([1, 2, 3, 4, 5], dtype=np.int64)
  arr[np.asarray([True, False, True, False, True], dtype=np.bool)] = 0
  assert arr.tolist() == [0, 2, 0, 4, 0]
