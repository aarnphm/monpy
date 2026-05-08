from __future__ import annotations

import sys

import monpy.array_api as xp
import monumpy as np
import numpy
import pytest
from _helpers import assert_same_shape_dtype, assert_same_values


def test_creation_functions_match_numpy_shape_dtype_and_values() -> None:
  zeros = np.zeros((2, 3), dtype=np.int64)
  ones = np.ones((2, 3), dtype=np.float32)
  full = np.full((2, 2), 7, dtype=np.int64)
  empty = np.empty((3, 0), dtype=np.float64)

  assert_same_shape_dtype(zeros, numpy.zeros((2, 3), dtype=numpy.int64))
  assert zeros.tolist() == [[0, 0, 0], [0, 0, 0]]
  assert_same_shape_dtype(ones, numpy.ones((2, 3), dtype=numpy.float32))
  assert ones.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
  assert_same_values(full, numpy.full((2, 2), 7, dtype=numpy.int64))
  assert_same_shape_dtype(empty, numpy.empty((3, 0), dtype=numpy.float64))


def test_like_creation_and_copy_helpers_match_numpy() -> None:
  base = np.arange(6, dtype=np.int64).reshape(2, 3).T
  oracle = numpy.arange(6, dtype=numpy.int64).reshape(2, 3).T

  empty = np.empty_like(base, dtype=np.float32, shape=(2, 2))
  assert_same_shape_dtype(empty, numpy.empty_like(oracle, dtype=numpy.float32, shape=(2, 2)))
  assert_same_values(np.zeros_like(base), numpy.zeros_like(oracle))
  assert_same_values(np.ones_like(base, dtype=np.float32), numpy.ones_like(oracle, dtype=numpy.float32))
  assert_same_values(np.full_like(base, 7), numpy.full_like(oracle, 7))

  copied = np.copy(base)
  copied[0, 0] = -99
  expected_copy = numpy.copy(oracle)
  expected_copy[0, 0] = -99
  assert_same_values(copied, expected_copy)
  assert base[0, 0] == int(oracle[0, 0])


def test_ascontiguousarray_materializes_only_when_needed() -> None:
  dense = np.arange(4, dtype=np.float64)
  assert np.ascontiguousarray(dense) is dense

  scalar = np.ascontiguousarray(3.5)
  assert_same_shape_dtype(scalar, numpy.ascontiguousarray(3.5))
  assert_same_values(scalar, numpy.ascontiguousarray(3.5))

  view = np.arange(6, dtype=np.float64).reshape(2, 3).T
  oracle = numpy.arange(6, dtype=numpy.float64).reshape(2, 3).T
  out = np.ascontiguousarray(view)
  expected = numpy.ascontiguousarray(oracle)

  assert out is not view
  assert_same_shape_dtype(out, expected)
  assert_same_values(out, expected)
  assert out.strides == expected.strides


def test_arange_and_linspace_match_numpy() -> None:
  assert_same_shape_dtype(np.arange(5), numpy.arange(5, dtype=numpy.int64))
  assert_same_values(np.arange(1.0, 2.0, 0.25), numpy.arange(1.0, 2.0, 0.25))
  assert_same_values(np.linspace(0.0, 1.0, 5, dtype=np.float32), numpy.linspace(0.0, 1.0, 5, dtype=numpy.float32))


def test_invalid_creation_arguments_are_explicit_blockers() -> None:
  with pytest.raises(ValueError, match="negative dimensions"):
    np.zeros((-1,), dtype=np.float64)
  with pytest.raises(Exception, match="step"):
    np.arange(0, 4, 0)
  with pytest.raises(NotImplementedError, match="cpu"):
    np.asarray([1], device="gpu")


def test_broadcasted_binary_ops_match_numpy() -> None:
  lhs = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
  rhs = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
  oracle_lhs = numpy.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  oracle_rhs = numpy.asarray([10.0, 20.0, 30.0])

  assert_same_values(lhs + rhs, oracle_lhs + oracle_rhs)
  assert_same_values(lhs - rhs, oracle_lhs - oracle_rhs)
  assert_same_values(lhs * rhs, oracle_lhs * oracle_rhs)
  assert_same_values(lhs / rhs, oracle_lhs / oracle_rhs)


def test_rank3_transposed_binary_add_uses_tiled_kernel() -> None:
  values = numpy.arange(8 * 8 * 8, dtype=numpy.float32).reshape((8, 8, 8))
  rhs_values = numpy.flip(values, axis=2).copy()
  lhs = np.asarray(values, dtype=np.float32)
  rhs = np.asarray(rhs_values, dtype=np.float32)

  result = lhs.transpose((2, 0, 1)) + rhs.transpose((2, 0, 1))

  assert_same_values(result, values.transpose((2, 0, 1)) + rhs_values.transpose((2, 0, 1)))
  assert result._native.used_fused()


def test_scalar_binary_ops_match_numpy() -> None:
  arr = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
  oracle = numpy.asarray([[1, 2, 3], [4, 5, 6]], dtype=numpy.int64)

  assert_same_values(arr + 2, oracle + 2)
  assert_same_values(2 + arr, 2 + oracle)
  assert_same_values(arr * 3, oracle * 3)
  assert_same_values(12 - arr, 12 - oracle)


def test_python_scalars_are_weak_for_float32_arrays() -> None:
  arr = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

  assert (arr * 3.0).dtype == np.float32
  assert (3.0 * arr).dtype == np.float32


def test_binary_out_writes_existing_destination() -> None:
  lhs = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
  rhs = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
  out = np.empty((3,), dtype=np.float32)

  result = np.add(lhs, rhs, out=out)

  assert result is out
  assert out.tolist() == [11.0, 22.0, 33.0]


def test_shape_manipulation_matches_numpy() -> None:
  arr = np.arange(6, dtype=np.int64)
  oracle = numpy.arange(6, dtype=numpy.int64)

  assert arr.reshape(2, 3).tolist() == oracle.reshape(2, 3).tolist()
  assert arr.reshape(2, 3).T.tolist() == oracle.reshape(2, 3).T.tolist()
  assert np.swapaxes(arr.reshape(1, 2, 3), 0, -1).tolist() == numpy.swapaxes(oracle.reshape(1, 2, 3), 0, -1).tolist()
  assert xp.matrix_transpose(arr.reshape(2, 3)).tolist() == oracle.reshape(2, 3).T.tolist()
  assert np.broadcast_to(np.asarray([1, 2, 3]), (2, 3)).tolist() == [[1, 2, 3], [1, 2, 3]]


def test_ravel_and_flatten_copy_semantics_match_numpy() -> None:
  base = np.arange(6, dtype=np.float64).reshape(2, 3)
  oracle = numpy.arange(6, dtype=numpy.float64).reshape(2, 3)

  raveled = np.ravel(base)
  raveled_expected = numpy.ravel(oracle)
  raveled[1] = -11.0
  raveled_expected[1] = -11.0

  assert_same_values(raveled, raveled_expected)
  assert_same_values(base, oracle)

  copied = np.flatten(base)
  copied_expected = oracle.flatten()
  copied[2] = -7.0
  copied_expected[2] = -7.0

  assert_same_values(copied, copied_expected)
  assert_same_values(base, oracle)

  transposed = base.T
  transposed_oracle = oracle.T
  transposed_ravel = np.ravel(transposed)
  transposed_expected = numpy.ravel(transposed_oracle)
  transposed_ravel[0] = -99.0
  transposed_expected[0] = -99.0

  assert_same_values(transposed_ravel, transposed_expected)
  assert_same_values(base, oracle)


def test_views_remain_safe_across_core_read_paths() -> None:
  base = np.arange(12, dtype=np.float64).reshape(3, 4)
  oracle = numpy.arange(12, dtype=numpy.float64).reshape(3, 4)
  cases = [
    (base, oracle),
    (base[:, 1:3], oracle[:, 1:3]),
    (base[::-1, ::-1], oracle[::-1, ::-1]),
    (base.T, oracle.T),
    (base[1:, 1:], oracle[1:, 1:]),
    (
      np.broadcast_to(np.asarray([1.0, 2.0, 3.0], dtype=np.float64), (2, 3)),
      numpy.broadcast_to(numpy.asarray([1.0, 2.0, 3.0], dtype=numpy.float64), (2, 3)),
    ),
  ]

  for arr, expected in cases:
    assert_same_values(arr + 2.0, expected + 2.0)
    assert_same_values(np.sin(arr), numpy.sin(expected))
    assert np.sum(arr) == pytest.approx(float(numpy.sum(expected)))


def test_array_namespace_info_reports_v1_surface() -> None:
  info = xp.__array_namespace_info__()

  assert info.default_device() == "cpu"
  assert info.devices() == ["cpu"]
  assert info.default_dtypes() == {"integral": np.int64, "real floating": np.float64, "bool": np.bool}
  assert info.capabilities() == {"boolean indexing": False, "data-dependent shapes": False}
  with pytest.raises(NotImplementedError, match="cpu"):
    info.dtypes(device="gpu")


def test_reductions_match_numpy_for_axis_none() -> None:
  arr = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
  oracle = numpy.asarray([[1, 2, 3], [4, 5, 6]], dtype=numpy.int64)

  assert np.sum(arr) == int(numpy.sum(oracle))
  assert np.min(arr) == int(numpy.min(oracle))
  assert np.max(arr) == int(numpy.max(oracle))
  assert np.argmax(arr) == int(numpy.argmax(oracle))
  assert np.mean(arr) == pytest.approx(float(numpy.mean(oracle)))


def test_axis_reductions_match_numpy() -> None:
  arr = np.asarray([[1, 2], [3, 4]])
  oracle = numpy.asarray([[1, 2], [3, 4]])

  numpy.testing.assert_array_equal(numpy.asarray(arr.sum(axis=0)), oracle.sum(axis=0))
  numpy.testing.assert_array_equal(numpy.asarray(np.sum(arr, axis=0)), oracle.sum(axis=0))
  numpy.testing.assert_array_equal(numpy.asarray(np.sum(arr, axis=1)), oracle.sum(axis=1))
  numpy.testing.assert_array_equal(numpy.asarray(np.sum(arr, keepdims=True)), oracle.sum(keepdims=True))


def test_matmul_matches_numpy_for_1d_and_2d() -> None:
  mat = np.asarray([[1, 2], [3, 4]], dtype=np.float64)
  rhs = np.asarray([[5, 6], [7, 8]], dtype=np.float64)
  vec = np.asarray([10, 20], dtype=np.float64)
  oracle_mat = numpy.asarray([[1, 2], [3, 4]], dtype=numpy.float64)
  oracle_rhs = numpy.asarray([[5, 6], [7, 8]], dtype=numpy.float64)
  oracle_vec = numpy.asarray([10, 20], dtype=numpy.float64)

  assert_same_values(mat @ rhs, oracle_mat @ oracle_rhs)
  assert_same_values(mat @ vec, oracle_mat @ oracle_vec)
  assert float(vec @ vec) == pytest.approx(float(oracle_vec @ oracle_vec))


def test_matmul_dense_transpose_rhs_matches_numpy_and_uses_fast_path_on_macos() -> None:
  lhs = (np.arange(18 * 12, dtype=np.float64) / 10.0).reshape(18, 12)
  rhs = (np.arange(20 * 12, dtype=np.float64) / 7.0).reshape(20, 12)
  oracle_lhs = (numpy.arange(18 * 12, dtype=numpy.float64) / 10.0).reshape(18, 12)
  oracle_rhs = (numpy.arange(20 * 12, dtype=numpy.float64) / 7.0).reshape(20, 12)

  out = lhs @ rhs.T

  assert rhs.T.strides == (8, 96)
  assert_same_shape_dtype(out, oracle_lhs @ oracle_rhs.T)
  assert_same_values(out, oracle_lhs @ oracle_rhs.T, rtol=1e-12, atol=1e-12)
  if sys.platform == "darwin" or sys.platform.startswith("linux"):
    assert out._native.used_accelerate()


@pytest.mark.parametrize(
  ("monpy_dtype", "numpy_dtype", "rtol"),
  [(np.float32, numpy.float32, 1e-4), (np.float64, numpy.float64, 1e-12)],
)
def test_matmul_rank1_shapes_match_numpy_and_use_gemv_on_macos(
  monpy_dtype: np.DType,
  numpy_dtype: type[numpy.generic],
  rtol: float,
) -> None:
  mat_oracle = (numpy.arange(18 * 12, dtype=numpy_dtype).reshape(18, 12) / 10).astype(
    numpy_dtype,
    copy=False,
  )
  rhs_oracle = (numpy.arange(20 * 12, dtype=numpy_dtype).reshape(20, 12) / 7).astype(
    numpy_dtype,
    copy=False,
  )
  vec_oracle = numpy.linspace(0.1, 1.2, 12, dtype=numpy_dtype)
  lhs = np.asarray(mat_oracle, dtype=monpy_dtype, copy=False)
  rhs_t = np.asarray(rhs_oracle.T, dtype=monpy_dtype, copy=False)
  vec = np.asarray(vec_oracle, dtype=monpy_dtype, copy=False)

  matvec = lhs @ vec
  vecmat = vec @ rhs_t

  assert_same_shape_dtype(matvec, mat_oracle @ vec_oracle)
  assert_same_values(matvec, mat_oracle @ vec_oracle, rtol=rtol, atol=rtol)
  assert_same_shape_dtype(vecmat, vec_oracle @ rhs_oracle.T)
  assert_same_values(vecmat, vec_oracle @ rhs_oracle.T, rtol=rtol, atol=rtol)
  if sys.platform == "darwin" or sys.platform.startswith("linux"):
    assert matvec._native.used_accelerate()
    assert vecmat._native.used_accelerate()


@pytest.mark.parametrize(
  ("monpy_dtype", "numpy_dtype"),
  [(np.int64, numpy.int64), (np.float32, numpy.float32), (np.float64, numpy.float64)],
)
@pytest.mark.parametrize(("m", "k", "n"), [(1, 1, 1), (2, 3, 4), (5, 4, 3), (16, 8, 5)])
@pytest.mark.parametrize("layout", ["c", "lhs_f", "rhs_transposed", "offset", "negative"])
def test_matmul_matrix_layout_sweep_matches_numpy(
  monpy_dtype: np.DType,
  numpy_dtype: type[numpy.generic],
  m: int,
  k: int,
  n: int,
  layout: str,
) -> None:
  lhs_oracle = (numpy.arange(m * k, dtype=numpy_dtype).reshape(m, k) + 1).astype(
    numpy_dtype,
    copy=False,
  )
  rhs_oracle = (numpy.arange(k * n, dtype=numpy_dtype).reshape(k, n) - 2).astype(
    numpy_dtype,
    copy=False,
  )
  if layout == "lhs_f":
    lhs_oracle = numpy.asfortranarray(lhs_oracle)
  elif layout == "rhs_transposed":
    rhs_oracle = (
      (numpy.arange(n * k, dtype=numpy_dtype).reshape(n, k) - 2)
      .astype(numpy_dtype, copy=False)
      .T
    )
  elif layout == "offset":
    lhs_source = (numpy.arange((m + 1) * k, dtype=numpy_dtype).reshape(m + 1, k) + 1).astype(
      numpy_dtype,
      copy=False,
    )
    rhs_source = (numpy.arange(k * (n + 1), dtype=numpy_dtype).reshape(k, n + 1) - 2).astype(
      numpy_dtype,
      copy=False,
    )
    lhs_oracle = lhs_source[1:, :]
    rhs_oracle = rhs_source[:, 1:]
  elif layout == "negative":
    lhs_oracle = lhs_oracle[::-1, :]
    rhs_oracle = rhs_oracle[:, ::-1]

  lhs = np.asarray(lhs_oracle, dtype=monpy_dtype, copy=False)
  rhs = np.asarray(rhs_oracle, dtype=monpy_dtype, copy=False)
  out = lhs @ rhs
  expected = lhs_oracle @ rhs_oracle

  assert_same_shape_dtype(out, expected)
  assert_same_values(out, expected, rtol=1e-4, atol=1e-4)


def test_higher_rank_matmul_is_an_explicit_v1_gap() -> None:
  lhs = np.ones((1, 2, 2), dtype=np.float64)
  rhs = np.ones((1, 2, 2), dtype=np.float64)

  with pytest.raises(Exception, match="1d and 2d"):
    np.matmul(lhs, rhs)


def test_fused_sin_add_mul_matches_numpy() -> None:
  lhs = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
  rhs = np.asarray([2.0, 3.0, 4.0], dtype=np.float32)
  oracle_lhs = numpy.asarray([0.1, 0.2, 0.3], dtype=numpy.float32)
  oracle_rhs = numpy.asarray([2.0, 3.0, 4.0], dtype=numpy.float32)

  out = np.sin_add_mul(lhs, rhs, 3.0)

  assert out._native.used_fused()
  assert out.dtype == np.float32
  assert_same_values(out, numpy.sin(oracle_lhs) + oracle_rhs * 3.0)


def test_numpy_shaped_expression_lowers_to_fused_kernel() -> None:
  lhs = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
  rhs = np.asarray([2.0, 3.0, 4.0], dtype=np.float32)
  oracle_lhs = numpy.asarray([0.1, 0.2, 0.3], dtype=numpy.float32)
  oracle_rhs = numpy.asarray([2.0, 3.0, 4.0], dtype=numpy.float32)

  out = np.sin(lhs) + rhs * 3.0

  assert out._native.used_fused()
  assert out.dtype == np.float32
  assert_same_values(out, numpy.sin(oracle_lhs) + oracle_rhs * 3.0)


def test_where_matches_numpy() -> None:
  cond = np.asarray([True, False, True])
  lhs = np.asarray([1, 2, 3])
  rhs = np.asarray([10, 20, 30])

  assert np.where(cond, lhs, rhs).tolist() == numpy.where([True, False, True], [1, 2, 3], [10, 20, 30]).tolist()


def test_where_broadcasts_scalars_and_arrays() -> None:
  cond = np.asarray([[True, False, True], [False, True, False]])
  rhs = np.asarray([10, 20, 30])

  assert_same_values(
    np.where(cond, 1, rhs),
    numpy.where([[True, False, True], [False, True, False]], 1, [10, 20, 30]),
  )


def test_reshape_on_noncontiguous_view_matches_numpy() -> None:
  # Cute composition path: reshape a transposed (non-contig) view.
  # Pre-cleanup, monpy raised "reshape() only supports c-contiguous";
  # post-cleanup, the layout algebra handles it (with a copy fallback
  # if the composition raises on shape misalignment).
  arr = np.arange(24, dtype=np.int64).reshape((2, 3, 4))
  oracle = numpy.arange(24, dtype=numpy.int64).reshape((2, 3, 4))

  monpy_t = arr.transpose((2, 0, 1))
  np_t = oracle.transpose((2, 0, 1))

  assert_same_values(monpy_t.reshape((4, 6)), np_t.reshape((4, 6)))
  assert_same_values(monpy_t.reshape((24,)), np_t.reshape((24,)))

  larger = np.arange(60, dtype=np.int64).reshape((3, 4, 5))
  oracle_larger = numpy.arange(60, dtype=numpy.int64).reshape((3, 4, 5))
  assert_same_values(
    larger.transpose((1, 0, 2)).reshape((12, 5)),
    oracle_larger.transpose((1, 0, 2)).reshape((12, 5)),
  )


def test_block_matches_numpy() -> None:
  assert_same_values(
    np.block([[1, 2], [3, 4]]),
    numpy.block([[1, 2], [3, 4]]),
  )

  A = np.eye(2, dtype=np.float64)
  B = np.zeros((2, 3), dtype=np.float64)
  C = np.zeros((3, 2), dtype=np.float64)
  D = np.eye(3, dtype=np.float64)
  assert_same_values(
    np.block([[A, B], [C, D]]),
    numpy.block([[numpy.eye(2), numpy.zeros((2, 3))], [numpy.zeros((3, 2)), numpy.eye(3)]]),
  )


def test_take_along_axis_matches_numpy() -> None:
  arr = np.asarray([[10, 20, 30], [40, 50, 60]])
  oracle = numpy.array([[10, 20, 30], [40, 50, 60]])
  idx = np.asarray([[0, 2, 1], [2, 1, 0]])
  np_idx = numpy.array([[0, 2, 1], [2, 1, 0]])

  assert_same_values(
    np.take_along_axis(arr, idx, axis=1),
    numpy.take_along_axis(oracle, np_idx, axis=1),
  )

  neg_idx = np.asarray([[-1, 0, 1], [0, -1, 1]])
  np_neg_idx = numpy.array([[-1, 0, 1], [0, -1, 1]])
  assert_same_values(
    np.take_along_axis(arr, neg_idx, axis=1),
    numpy.take_along_axis(oracle, np_neg_idx, axis=1),
  )

  broadcast_idx = np.asarray([[0, 1, 2]])
  np_broadcast_idx = numpy.array([[0, 1, 2]])
  assert_same_values(
    np.take_along_axis(arr, broadcast_idx, axis=1),
    numpy.take_along_axis(oracle, np_broadcast_idx, axis=1),
  )


def test_put_along_axis_matches_numpy() -> None:
  monpy_target = np.zeros((2, 3), dtype=np.int64)
  np_target = numpy.zeros((2, 3), dtype=numpy.int64)
  monpy_idx = np.asarray([[0], [2]])
  np_idx = numpy.array([[0], [2]])
  monpy_vals = np.asarray([[99], [77]])
  np_vals = numpy.array([[99], [77]])

  np.put_along_axis(monpy_target, monpy_idx, monpy_vals, axis=1)
  numpy.put_along_axis(np_target, np_idx, np_vals, axis=1)
  assert_same_values(monpy_target, np_target)

  monpy_target = np.zeros((2, 3), dtype=np.int64)
  np_target = numpy.zeros((2, 3), dtype=numpy.int64)
  monpy_idx = np.asarray([[0, 1, 2], [2, 1, 0]])
  np_idx = numpy.array([[0, 1, 2], [2, 1, 0]])
  np.put_along_axis(monpy_target, monpy_idx, np.asarray([7, 8, 9]), axis=1)
  numpy.put_along_axis(np_target, np_idx, numpy.array([7, 8, 9]), axis=1)
  assert_same_values(monpy_target, np_target)

  monpy_target = np.zeros((2, 3), dtype=np.int64)
  np_target = numpy.zeros((2, 3), dtype=numpy.int64)
  np.put_along_axis(monpy_target, monpy_idx, np.asarray([[7], [8]]), axis=1)
  numpy.put_along_axis(np_target, np_idx, numpy.array([[7], [8]]), axis=1)
  assert_same_values(monpy_target, np_target)

  monpy_target = np.zeros((2, 3), dtype=np.int64)
  np_target = numpy.zeros((2, 3), dtype=numpy.int64)
  monpy_idx = np.asarray([[-1], [0]])
  np_idx = numpy.array([[-1], [0]])
  np.put_along_axis(monpy_target, monpy_idx, np.asarray([[99], [77]]), axis=1)
  numpy.put_along_axis(np_target, np_idx, numpy.array([[99], [77]]), axis=1)
  assert_same_values(monpy_target, np_target)

  monpy_target = np.zeros((2, 3), dtype=np.int64)
  np_target = numpy.zeros((2, 3), dtype=numpy.int64)
  monpy_idx = np.asarray([[0, 1, 2]])
  np_idx = numpy.array([[0, 1, 2]])
  np.put_along_axis(monpy_target, monpy_idx, np.asarray([[4, 5, 6]]), axis=1)
  numpy.put_along_axis(np_target, np_idx, numpy.array([[4, 5, 6]]), axis=1)
  assert_same_values(monpy_target, np_target)


def test_put_matches_numpy() -> None:
  monpy_arr = np.zeros((2, 3), dtype=np.int64)
  np_arr = numpy.zeros((2, 3), dtype=numpy.int64)
  np.put(monpy_arr, [0, 4], [10, 20])
  numpy.put(np_arr, [0, 4], [10, 20])
  assert_same_values(monpy_arr, np_arr)

  monpy_arr = np.zeros((2, 3), dtype=np.int64)
  np_arr = numpy.zeros((2, 3), dtype=numpy.int64)
  np.put(monpy_arr, [-1, 0], [70, 80])
  numpy.put(np_arr, [-1, 0], [70, 80])
  assert_same_values(monpy_arr, np_arr)

  with pytest.raises(IndexError):
    np.put(np.zeros((2, 3), dtype=np.int64), [6], [1])


def test_pad_modes_match_numpy() -> None:
  arr = np.asarray([1, 2, 3, 4])
  np_arr = numpy.array([1, 2, 3, 4])
  for mode in ("edge", "reflect", "symmetric", "wrap"):
    assert_same_values(
      np.pad(arr, 2, mode=mode),
      numpy.pad(np_arr, 2, mode=mode),
    )
    assert_same_values(
      np.pad(arr, 7, mode=mode),
      numpy.pad(np_arr, 7, mode=mode),
    )

  arr2 = np.asarray([[1, 2, 3], [4, 5, 6]])
  np_arr2 = numpy.array([[1, 2, 3], [4, 5, 6]])
  for mode in ("edge", "reflect", "symmetric", "wrap"):
    assert_same_values(
      np.pad(arr2, ((1, 1), (2, 1)), mode=mode),
      numpy.pad(np_arr2, ((1, 1), (2, 1)), mode=mode),
    )
