from __future__ import annotations

import monumpy as np
import numpy
import pytest


def assert_same_values(monpy_value: object, numpy_value: object) -> None:
  numpy.testing.assert_allclose(numpy.asarray(monpy_value), numpy.asarray(numpy_value), rtol=1e-6, atol=1e-6)


def test_broadcasted_binary_ops_match_numpy() -> None:
  lhs = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
  rhs = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
  oracle_lhs = numpy.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  oracle_rhs = numpy.asarray([10.0, 20.0, 30.0])

  assert_same_values(lhs + rhs, oracle_lhs + oracle_rhs)
  assert_same_values(lhs * rhs, oracle_lhs * oracle_rhs)
  assert_same_values(lhs / rhs, oracle_lhs / oracle_rhs)


def test_shape_manipulation_matches_numpy() -> None:
  arr = np.arange(6, dtype=np.int64)
  oracle = numpy.arange(6, dtype=numpy.int64)

  assert arr.reshape(2, 3).tolist() == oracle.reshape(2, 3).tolist()
  assert arr.reshape(2, 3).T.tolist() == oracle.reshape(2, 3).T.tolist()
  assert np.broadcast_to(np.asarray([1, 2, 3]), (2, 3)).tolist() == [[1, 2, 3], [1, 2, 3]]


def test_reductions_match_numpy_for_axis_none() -> None:
  arr = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
  oracle = numpy.asarray([[1, 2, 3], [4, 5, 6]], dtype=numpy.int64)

  assert np.sum(arr) == int(numpy.sum(oracle))
  assert np.min(arr) == int(numpy.min(oracle))
  assert np.max(arr) == int(numpy.max(oracle))
  assert np.argmax(arr) == int(numpy.argmax(oracle))
  assert np.mean(arr) == pytest.approx(float(numpy.mean(oracle)))


def test_axis_reductions_are_explicit_v1_gap() -> None:
  arr = np.asarray([[1, 2], [3, 4]])

  with pytest.raises(NotImplementedError, match="axis"):
    arr.sum(axis=0)


def test_matmul_matches_numpy_for_1d_and_2d() -> None:
  mat = np.asarray([[1, 2], [3, 4]], dtype=np.float64)
  rhs = np.asarray([[5, 6], [7, 8]], dtype=np.float64)
  vec = np.asarray([10, 20], dtype=np.float64)
  oracle_mat = numpy.asarray([[1, 2], [3, 4]], dtype=numpy.float64)
  oracle_rhs = numpy.asarray([[5, 6], [7, 8]], dtype=numpy.float64)
  oracle_vec = numpy.asarray([10, 20], dtype=numpy.float64)

  assert_same_values(mat @ rhs, oracle_mat @ oracle_rhs)
  assert_same_values(mat @ vec, oracle_mat @ oracle_vec)
  assert (vec @ vec) == pytest.approx(float(oracle_vec @ oracle_vec))


def test_where_matches_numpy() -> None:
  cond = np.asarray([True, False, True])
  lhs = np.asarray([1, 2, 3])
  rhs = np.asarray([10, 20, 30])

  assert np.where(cond, lhs, rhs).tolist() == numpy.where([True, False, True], [1, 2, 3], [10, 20, 30]).tolist()
