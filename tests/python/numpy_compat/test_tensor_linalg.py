from __future__ import annotations

import monpy as mp
import numpy
import pytest


# ---------------------------------------------------------------------------
# dot / vdot / inner / outer
# ---------------------------------------------------------------------------


def test_dot_1d_1d() -> None:
  a = mp.asarray([1.0, 2.0, 3.0], dtype=mp.float64)
  b = mp.asarray([4.0, 5.0, 6.0], dtype=mp.float64)
  assert float(mp.dot(a, b)) == 32.0


def test_dot_2d_2d_routes_through_matmul() -> None:
  a = mp.asarray([[1, 2], [3, 4]], dtype=mp.float64)
  b = mp.asarray([[5, 6], [7, 8]], dtype=mp.float64)
  out = mp.dot(a, b)
  numpy.testing.assert_array_equal(numpy.asarray(out), [[19, 22], [43, 50]])


def test_vdot_flattens_first() -> None:
  a = mp.asarray([[1, 2], [3, 4]], dtype=mp.float64)
  b = mp.asarray([[5, 6], [7, 8]], dtype=mp.float64)
  assert float(mp.vdot(a, b)) == 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8


def test_outer_basic() -> None:
  a = mp.asarray([1, 2, 3], dtype=mp.float64)
  b = mp.asarray([4, 5], dtype=mp.float64)
  out = mp.outer(a, b)
  numpy.testing.assert_array_equal(numpy.asarray(out), [[4, 5], [8, 10], [12, 15]])


def test_inner_1d() -> None:
  a = mp.asarray([1.0, 2.0, 3.0], dtype=mp.float64)
  b = mp.asarray([4.0, 5.0, 6.0], dtype=mp.float64)
  assert float(mp.inner(a, b)) == 32.0


# ---------------------------------------------------------------------------
# tensordot
# ---------------------------------------------------------------------------


def test_tensordot_axes_int() -> None:
  a = mp.asarray([[1, 2], [3, 4]], dtype=mp.float64)
  b = mp.asarray([[5, 6], [7, 8]], dtype=mp.float64)
  out = mp.tensordot(a, b, axes=1)
  numpy.testing.assert_array_equal(numpy.asarray(out), [[19, 22], [43, 50]])


def test_tensordot_axes_full() -> None:
  a = mp.asarray([[1, 2], [3, 4]], dtype=mp.float64)
  b = mp.asarray([[5, 6], [7, 8]], dtype=mp.float64)
  out = mp.tensordot(a, b, axes=2)
  # Full contraction: 1*5 + 2*6 + 3*7 + 4*8 = 70.
  assert float(out) == 70.0


# ---------------------------------------------------------------------------
# kron / cross
# ---------------------------------------------------------------------------


def test_kron_2x2_with_2x2() -> None:
  a = mp.asarray([[1, 0], [0, 1]], dtype=mp.float64)
  b = mp.asarray([[1, 2], [3, 4]], dtype=mp.float64)
  out = mp.kron(a, b)
  oracle = numpy.kron(numpy.asarray([[1, 0], [0, 1]]), numpy.asarray([[1, 2], [3, 4]]))
  numpy.testing.assert_array_equal(numpy.asarray(out), oracle)


def test_cross_3vec() -> None:
  a = mp.asarray([1.0, 0.0, 0.0], dtype=mp.float64)
  b = mp.asarray([0.0, 1.0, 0.0], dtype=mp.float64)
  out = mp.cross(a, b)
  numpy.testing.assert_array_equal(numpy.asarray(out), [0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# norm
# ---------------------------------------------------------------------------


def test_norm_default_l2() -> None:
  a = mp.asarray([3.0, 4.0], dtype=mp.float64)
  assert float(mp.linalg.norm(a)) == 5.0


def test_norm_l1_and_inf() -> None:
  import math
  a = mp.asarray([1.0, -2.0, 3.0], dtype=mp.float64)
  assert float(mp.linalg.norm(a, ord=1)) == 6.0
  assert float(mp.linalg.norm(a, ord=math.inf)) == 3.0


# ---------------------------------------------------------------------------
# matrix_power / multi_dot
# ---------------------------------------------------------------------------


def test_matrix_power_zero_returns_identity() -> None:
  a = mp.asarray([[2, 0], [0, 2]], dtype=mp.float64)
  out = mp.linalg.matrix_power(a, 0)
  numpy.testing.assert_array_equal(numpy.asarray(out), [[1, 0], [0, 1]])


def test_matrix_power_pos() -> None:
  a = mp.asarray([[1, 1], [0, 1]], dtype=mp.float64)
  out = mp.linalg.matrix_power(a, 3)
  numpy.testing.assert_array_equal(numpy.asarray(out), [[1, 3], [0, 1]])


def test_multi_dot_chained_matmul() -> None:
  a = mp.asarray([[1, 2]], dtype=mp.float64)
  b = mp.asarray([[3], [4]], dtype=mp.float64)
  c = mp.asarray([[5]], dtype=mp.float64)
  out = mp.linalg.multi_dot([a, b, c])
  numpy.testing.assert_array_equal(numpy.asarray(out), [[55]])
