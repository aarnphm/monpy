from __future__ import annotations

import monumpy as np
import numpy
import pytest

from _helpers import assert_same_values


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


def test_newaxis_is_an_explicit_v1_gap() -> None:
  arr = np.asarray([1, 2, 3])

  with pytest.raises(NotImplementedError, match="newaxis"):
    _ = arr[None, :]


@pytest.mark.parametrize(
  "index",
  [
    [True, False, True],
    np.asarray([True, False, True], dtype=np.bool),
    np.asarray([0, 2], dtype=np.int64),
  ],
)
def test_boolean_and_integer_array_indexing_are_explicit_v1_gaps(index: object) -> None:
  arr = np.asarray([1, 2, 3])

  with pytest.raises(NotImplementedError, match="integer and slice indexing"):
    _ = arr[index]
