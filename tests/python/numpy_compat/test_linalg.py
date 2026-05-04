from __future__ import annotations

import sys

import monpy.array_api as xp
import monumpy as np
import numpy
import pytest
from _helpers import assert_same_shape_dtype, assert_same_values


def test_linalg_matmul_matches_numpy_rank2() -> None:
  lhs = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
  rhs = np.asarray([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float64)
  oracle_lhs = numpy.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float64)
  oracle_rhs = numpy.asarray([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=numpy.float64)

  out = np.linalg.matmul(lhs, rhs)

  assert_same_shape_dtype(out, numpy.linalg.matmul(oracle_lhs, oracle_rhs))
  assert_same_values(out, numpy.linalg.matmul(oracle_lhs, oracle_rhs))


def test_array_api_linalg_matmul_matches_top_level_matmul() -> None:
  lhs = np.asarray([[1, 2], [3, 4]], dtype=np.int64)
  rhs = np.asarray([[5, 6], [7, 8]], dtype=np.int64)

  assert xp.linalg.matmul(lhs, rhs).tolist() == np.matmul(lhs, rhs).tolist()


def test_linalg_matrix_transpose_matches_numpy() -> None:
  arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
  oracle = numpy.arange(24, dtype=numpy.float64).reshape(2, 3, 4)

  out = np.linalg.matrix_transpose(arr)

  assert_same_shape_dtype(out, numpy.linalg.matrix_transpose(oracle))
  assert_same_values(out, numpy.linalg.matrix_transpose(oracle))
  assert xp.linalg.matrix_transpose(arr).tolist() == numpy.linalg.matrix_transpose(oracle).tolist()


def test_diagonal_returns_rank2_view_and_trace_matches_numpy() -> None:
  arr = np.arange(9, dtype=np.int64).reshape(3, 3)
  oracle = numpy.arange(9, dtype=numpy.int64).reshape(3, 3)

  diag = np.diagonal(arr, offset=1)
  diag[0] = 77
  oracle[0, 1] = 77
  oracle_diag = numpy.diagonal(oracle, offset=1)

  assert diag.tolist() == oracle_diag.tolist()
  assert arr.tolist() == oracle.tolist()
  assert np.trace(arr) == int(numpy.trace(oracle))


def test_trace_dtype_argument_casts_accumulator() -> None:
  arr = np.asarray([[True, False], [True, True]], dtype=np.bool)

  assert np.trace(arr) == 2
  assert isinstance(np.trace(arr, dtype=np.float32), float)
  assert np.trace(arr, dtype=np.float32) == pytest.approx(2.0)


def test_linalg_solve_inv_and_det_match_numpy() -> None:
  a = np.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
  b = np.asarray([9.0, 8.0], dtype=np.float64)
  oracle_a = numpy.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=numpy.float64)
  oracle_b = numpy.asarray([9.0, 8.0], dtype=numpy.float64)

  solve_out = np.linalg.solve(a, b)
  inv_out = np.linalg.inv(a)
  det_out = np.linalg.det(a)

  assert_same_shape_dtype(solve_out, numpy.linalg.solve(oracle_a, oracle_b))
  assert_same_values(solve_out, numpy.linalg.solve(oracle_a, oracle_b))
  assert_same_shape_dtype(inv_out, numpy.linalg.inv(oracle_a))
  assert_same_values(inv_out, numpy.linalg.inv(oracle_a))
  assert det_out == pytest.approx(float(numpy.linalg.det(oracle_a)))
  if sys.platform == "darwin":
    assert solve_out._native.used_accelerate()
    assert inv_out._native.used_accelerate()
  else:
    assert solve_out._native.backend_code() == 0
    assert inv_out._native.backend_code() == 0


def test_linalg_dtype_policy_preserves_float32_and_casts_integral_inputs() -> None:
  f32_a = np.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=np.float32)
  f32_b = np.asarray([2.0, 8.0], dtype=np.float32)
  int_a = np.asarray([[2, 0], [0, 4]], dtype=np.int64)
  int_b = np.asarray([2, 8], dtype=np.int64)

  assert np.linalg.solve(f32_a, f32_b).dtype == np.float32
  assert np.linalg.inv(f32_a).dtype == np.float32
  assert np.linalg.solve(int_a, int_b).dtype == np.float64
  assert np.linalg.inv(int_a).dtype == np.float64


def _well_conditioned_matrix(size: int, numpy_dtype: type[numpy.generic]) -> numpy.ndarray:
  values = numpy.arange(size * size, dtype=numpy.float64).reshape(size, size) / 11.0
  values += numpy.eye(size, dtype=numpy.float64) * float(size + 3)
  if numpy_dtype is numpy.int64:
    return numpy.rint(values * 3).astype(numpy_dtype)
  return values.astype(numpy_dtype)


@pytest.mark.parametrize(
  ("monpy_dtype", "numpy_dtype"),
  [(np.int64, numpy.int64), (np.float32, numpy.float32), (np.float64, numpy.float64)],
)
@pytest.mark.parametrize("size", [1, 2, 3, 5])
def test_linalg_matrix_sweep_matches_numpy(
  monpy_dtype: np.DType,
  numpy_dtype: type[numpy.generic],
  size: int,
) -> None:
  oracle_a = _well_conditioned_matrix(size, numpy_dtype)
  oracle_rhs_vector = numpy.arange(1, size + 1, dtype=numpy_dtype)
  oracle_rhs_matrix = numpy.arange(1, size * 2 + 1, dtype=numpy_dtype).reshape(size, 2)
  a = np.asarray(oracle_a, dtype=monpy_dtype, copy=False)
  rhs_vector = np.asarray(oracle_rhs_vector, dtype=monpy_dtype, copy=False)
  rhs_matrix = np.asarray(oracle_rhs_matrix, dtype=monpy_dtype, copy=False)
  rtol = 1e-4 if monpy_dtype is np.float32 else 1e-8
  atol = 1e-4 if monpy_dtype is np.float32 else 1e-8

  solve_vector = np.linalg.solve(a, rhs_vector)
  solve_matrix = np.linalg.solve(a, rhs_matrix)
  inv_out = np.linalg.inv(a)
  det_out = np.linalg.det(a)

  assert_same_shape_dtype(solve_vector, numpy.linalg.solve(oracle_a, oracle_rhs_vector))
  assert_same_values(solve_vector, numpy.linalg.solve(oracle_a, oracle_rhs_vector), rtol=rtol, atol=atol)
  assert_same_shape_dtype(solve_matrix, numpy.linalg.solve(oracle_a, oracle_rhs_matrix))
  assert_same_values(solve_matrix, numpy.linalg.solve(oracle_a, oracle_rhs_matrix), rtol=rtol, atol=atol)
  assert_same_shape_dtype(inv_out, numpy.linalg.inv(oracle_a))
  assert_same_values(inv_out, numpy.linalg.inv(oracle_a), rtol=rtol, atol=atol)
  assert det_out == pytest.approx(float(numpy.linalg.det(oracle_a)), rel=rtol, abs=atol)


@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2])
def test_diagonal_and_trace_offset_sweep_matches_numpy(offset: int) -> None:
  arr = np.arange(20, dtype=np.float64).reshape(4, 5)
  oracle = numpy.arange(20, dtype=numpy.float64).reshape(4, 5)

  assert_same_shape_dtype(np.diagonal(arr, offset=offset), numpy.diagonal(oracle, offset=offset))
  assert_same_values(np.diagonal(arr, offset=offset), numpy.diagonal(oracle, offset=offset))
  assert np.trace(arr, offset=offset) == pytest.approx(float(numpy.trace(oracle, offset=offset)))


def test_linalg_singular_and_invalid_square_conditions_raise_linalg_error() -> None:
  singular = np.asarray([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
  nonsquare = np.ones((2, 3), dtype=np.float64)

  with pytest.raises(np.linalg.LinAlgError, match="singular"):
    np.linalg.solve(singular, np.asarray([1.0, 2.0], dtype=np.float64))
  with pytest.raises(np.linalg.LinAlgError, match="singular"):
    np.linalg.inv(singular)
  assert np.linalg.det(singular) == pytest.approx(0.0)
  with pytest.raises(np.linalg.LinAlgError, match="square"):
    np.linalg.solve(nonsquare, np.asarray([1.0, 2.0], dtype=np.float64))
