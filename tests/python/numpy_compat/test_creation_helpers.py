"""Phase-6b creation helper coverage. Most are pure python compositions
on top of existing primitives, so the tests just check parity with numpy."""

from __future__ import annotations

import math

import monumpy as np
import numpy
import pytest


def test_eye_default() -> None:
  m = np.eye(3)
  numpy.testing.assert_array_equal(numpy.asarray(m), numpy.eye(3))


@pytest.mark.parametrize(("N", "M", "k"), [(3, 4, 0), (3, 4, 1), (3, 4, -1), (5, 3, 2), (4, 4, -2)])
def test_eye_offsets_match_numpy(N: int, M: int, k: int) -> None:
  m = np.eye(N, M, k)
  numpy.testing.assert_array_equal(numpy.asarray(m), numpy.eye(N, M, k))


def test_identity_matches_numpy() -> None:
  numpy.testing.assert_array_equal(numpy.asarray(np.identity(4)), numpy.identity(4))


@pytest.mark.parametrize(("N", "M", "k"), [(3, 3, 0), (3, 4, 0), (3, 3, 1), (4, 4, -1)])
def test_tri_matches_numpy(N: int, M: int, k: int) -> None:
  numpy.testing.assert_array_equal(numpy.asarray(np.tri(N, M, k)), numpy.tri(N, M, k))


def test_atleast_1d_scalar_promotes_to_rank_1() -> None:
  res = np.atleast_1d(5)
  assert res.shape == (1,)


def test_atleast_2d_promotes_rank_1_to_row() -> None:
  res = np.atleast_2d([1, 2, 3])
  assert res.shape == (1, 3)


def test_atleast_3d_promotes_rank_1_with_trailing_axis() -> None:
  res = np.atleast_3d([1, 2, 3])
  assert res.shape == (1, 3, 1)


def test_atleast_2d_rank_2_passes_through() -> None:
  src = np.asarray([[1, 2], [3, 4]])
  res = np.atleast_2d(src)
  assert res.shape == (2, 2)


def test_atleast_1d_multiple_returns_tuple() -> None:
  a, b = np.atleast_1d(5, [1, 2])
  assert a.shape == (1,)
  assert b.shape == (2,)


@pytest.mark.parametrize("num", [0, 1, 4, 50])
def test_logspace_matches_numpy_finite(num: int) -> None:
  res = numpy.asarray(np.logspace(0.0, 3.0, num))
  expected = numpy.logspace(0.0, 3.0, num)
  numpy.testing.assert_allclose(res, expected, rtol=1e-12)


@pytest.mark.parametrize("num", [0, 1, 4, 16])
def test_geomspace_matches_numpy(num: int) -> None:
  res = numpy.asarray(np.geomspace(1.0, 1000.0, num))
  expected = numpy.geomspace(1.0, 1000.0, num)
  numpy.testing.assert_allclose(res, expected, rtol=1e-10)


def test_meshgrid_xy_indexing_swaps_first_two_axes() -> None:
  x = np.arange(3)
  y = np.arange(4)
  xx, yy = np.meshgrid(x, y)
  assert xx.shape == (4, 3)
  assert yy.shape == (4, 3)
  numpy.testing.assert_array_equal(numpy.asarray(xx), numpy.meshgrid(numpy.arange(3), numpy.arange(4))[0])


def test_meshgrid_ij_indexing_keeps_natural_order() -> None:
  x = np.arange(3)
  y = np.arange(4)
  xx, yy = np.meshgrid(x, y, indexing="ij")
  assert xx.shape == (3, 4)
  assert yy.shape == (3, 4)


def test_indices_matches_numpy() -> None:
  res = numpy.asarray(np.indices((2, 3)))
  expected = numpy.indices((2, 3))
  numpy.testing.assert_array_equal(res, expected)


@pytest.mark.parametrize("axis", [None, 0, (0, 2), -4])
def test_squeeze_matches_numpy(axis: int | tuple[int, ...] | None) -> None:
  source = numpy.arange(20, dtype=numpy.float32).reshape(1, 4, 1, 5)
  arr = np.asarray(source)

  out = np.squeeze(arr, axis=axis)

  numpy.testing.assert_array_equal(numpy.asarray(out), numpy.squeeze(source, axis=axis))
  assert out.shape == numpy.squeeze(source, axis=axis).shape


def test_squeeze_rejects_non_singleton_axis() -> None:
  arr = np.asarray(numpy.zeros((1, 4, 1), dtype=numpy.float32))

  with pytest.raises(ValueError, match="size != 1"):
    np.squeeze(arr, axis=1)


def test_ix_returns_outer_index_arrays() -> None:
  a = np.arange(3)
  b = np.arange(4)
  ax, bx = np.ix_(a, b)
  assert ax.shape == (3, 1)
  assert bx.shape == (1, 4)


def test_join_helpers_match_numpy_for_axis0_fast_paths() -> None:
  a = np.arange(4, dtype=np.float32)
  b = np.arange(4, 8, dtype=np.float32)
  a_np = numpy.arange(4, dtype=numpy.float32)
  b_np = numpy.arange(4, 8, dtype=numpy.float32)

  numpy.testing.assert_array_equal(numpy.asarray(np.concatenate([a, b])), numpy.concatenate([a_np, b_np]))
  numpy.testing.assert_array_equal(numpy.asarray(np.stack([a, b], axis=0)), numpy.stack([a_np, b_np], axis=0))
  numpy.testing.assert_array_equal(numpy.asarray(np.vstack([a, b])), numpy.vstack([a_np, b_np]))


def test_stack_axis0_fast_path_preserves_dtype_override() -> None:
  a = np.arange(3, dtype=np.int64)
  b = np.arange(3, 6, dtype=np.int64)
  expected = numpy.stack([numpy.arange(3), numpy.arange(3, 6)], axis=0).astype(numpy.float32)

  out = np.stack([a, b], axis=0, dtype=np.float32)

  assert out.dtype == np.float32
  numpy.testing.assert_array_equal(numpy.asarray(out), expected)


def test_stack_axis0_fast_path_falls_back_for_promotion() -> None:
  a = np.arange(3, dtype=np.int64)
  b = np.arange(3, 6, dtype=np.float32)
  expected = numpy.stack([numpy.arange(3, dtype=numpy.int64), numpy.arange(3, 6, dtype=numpy.float32)], axis=0)

  out = np.stack([a, b], axis=0)

  assert out.dtype == np.float64
  numpy.testing.assert_array_equal(numpy.asarray(out), expected)


def test_vstack_rank2_matches_numpy() -> None:
  a = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
  b = np.asarray([[5, 6]], dtype=np.float32)
  expected = numpy.vstack([numpy.asarray(a), numpy.asarray(b)])

  out = np.vstack([a, b])

  numpy.testing.assert_array_equal(numpy.asarray(out), expected)
