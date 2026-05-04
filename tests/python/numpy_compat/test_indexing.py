from __future__ import annotations

import monumpy as np
import pytest


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


def test_ellipsis_expands_to_full_slices() -> None:
  arr = np.arange(6, dtype=np.int64).reshape(2, 3)

  assert arr[..., 1].tolist() == [1, 4]


def test_slice_assignment_mutates_base_array() -> None:
  arr = np.arange(15, dtype=np.int64).reshape(3, 5)
  arr[1:, ::2] = -99

  assert arr.tolist() == [[0, 1, 2, 3, 4], [-99, 6, -99, 8, -99], [-99, 11, -99, 13, -99]]


def test_newaxis_is_an_explicit_v1_gap() -> None:
  arr = np.asarray([1, 2, 3])

  with pytest.raises(NotImplementedError, match="newaxis"):
    _ = arr[None, :]
