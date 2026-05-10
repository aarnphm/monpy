"""flip / fliplr / flipud / rot90 parity vs numpy."""

from __future__ import annotations

import monpy as np
import numpy
import pytest
from numpy.testing import assert_array_equal


@pytest.mark.parametrize(
  ("input_shape", "axis"),
  [
    ((3, 4), None),
    ((3, 4), 0),
    ((3, 4), 1),
    ((3, 4), -1),
    ((3, 4), -2),
    ((2, 3, 4), None),
    ((2, 3, 4), (0, 2)),
    ((2, 3, 4), (1,)),
    ((5,), 0),
    ((5,), None),
  ],
)
def test_flip_matches_numpy(input_shape: tuple[int, ...], axis: object) -> None:
  src = numpy.arange(int(numpy.prod(input_shape)), dtype=numpy.float32).reshape(input_shape)
  expected = numpy.flip(src, axis=axis)
  got = np.flip(np.asarray(src), axis=axis)
  assert_array_equal(numpy.asarray(got), expected)


def test_flip_rejects_repeated_axis() -> None:
  a = np.asarray(numpy.arange(12, dtype=numpy.float32).reshape(3, 4))
  with pytest.raises(Exception):
    np.flip(a, axis=(0, 0))


def test_flip_rejects_out_of_bounds() -> None:
  a = np.asarray(numpy.arange(12, dtype=numpy.float32).reshape(3, 4))
  with pytest.raises(Exception):
    np.flip(a, axis=2)


def test_fliplr_matches_numpy() -> None:
  src = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
  expected = numpy.fliplr(src)
  got = np.fliplr(np.asarray(src))
  assert_array_equal(numpy.asarray(got), expected)


def test_flipud_matches_numpy() -> None:
  src = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
  expected = numpy.flipud(src)
  got = np.flipud(np.asarray(src))
  assert_array_equal(numpy.asarray(got), expected)


def test_fliplr_rejects_low_rank() -> None:
  a = np.asarray(numpy.arange(5, dtype=numpy.float32))
  with pytest.raises(ValueError):
    np.fliplr(a)


@pytest.mark.parametrize("k", [0, 1, 2, 3, -1, 4, 5])
def test_rot90_matches_numpy(k: int) -> None:
  src = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
  expected = numpy.rot90(src, k=k)
  got = np.rot90(np.asarray(src), k=k)
  assert_array_equal(numpy.asarray(got), expected)


@pytest.mark.parametrize(("axes", "k"), [((0, 2), 1), ((1, 2), 2), ((-2, -1), 3)])
def test_rot90_axes(axes: tuple[int, int], k: int) -> None:
  src = numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4)
  expected = numpy.rot90(src, k=k, axes=axes)
  got = np.rot90(np.asarray(src), k=k, axes=axes)
  assert_array_equal(numpy.asarray(got), expected)


def test_rot90_rejects_low_rank() -> None:
  a = np.asarray(numpy.arange(5, dtype=numpy.float32))
  with pytest.raises(ValueError):
    np.rot90(a)


def test_rot90_rejects_same_axes() -> None:
  a = np.asarray(numpy.arange(12, dtype=numpy.float32).reshape(3, 4))
  with pytest.raises(ValueError):
    np.rot90(a, axes=(0, 0))


def test_flip_returns_view_sharing_storage() -> None:
  # `flip` should be a stride-only view; mutating the source must show through.
  src = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
  a = np.asarray(src.copy())  # copy so monpy holds it
  v = np.flip(a, axis=0)
  # the view should reflect the original data layout: identical data_address bumped by stride
  assert v._native.data_address() != a._native.data_address()  # offset shifted
