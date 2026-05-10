from __future__ import annotations

from collections.abc import Callable

import monumpy as np
import numpy
from _helpers import assert_same_result_kind, assert_same_shape_dtype, assert_same_values


def _vec() -> np.ndarray:
  return np.asarray([1.0, 2.0, 3.0], dtype=np.float64)


def _vec2() -> np.ndarray:
  return np.asarray([4.0, 5.0, 6.0], dtype=np.float64)


def _mat() -> np.ndarray:
  return np.asarray([[1.0, 2.0], [3.0, 5.0]], dtype=np.float64)


def _mat2() -> np.ndarray:
  return np.asarray([[7.0, 11.0], [13.0, 17.0]], dtype=np.float64)


def _bools() -> np.ndarray:
  return np.asarray([True, False, True], dtype=np.bool)


def _np_vec() -> numpy.ndarray:
  return numpy.asarray([1.0, 2.0, 3.0], dtype=numpy.float64)


def _np_vec2() -> numpy.ndarray:
  return numpy.asarray([4.0, 5.0, 6.0], dtype=numpy.float64)


def _np_mat() -> numpy.ndarray:
  return numpy.asarray([[1.0, 2.0], [3.0, 5.0]], dtype=numpy.float64)


def _np_mat2() -> numpy.ndarray:
  return numpy.asarray([[7.0, 11.0], [13.0, 17.0]], dtype=numpy.float64)


def _np_bools() -> numpy.ndarray:
  return numpy.asarray([True, False, True], dtype=numpy.bool_)


ScalarCase = tuple[str, Callable[[], object], Callable[[], object], bool]


SCALAR_ARRAY_CASES: tuple[ScalarCase, ...] = (
  ("sum", lambda: np.sum(_vec()), lambda: numpy.sum(_np_vec()), True),
  ("mean", lambda: np.mean(_vec()), lambda: numpy.mean(_np_vec()), True),
  ("prod", lambda: np.prod(_vec()), lambda: numpy.prod(_np_vec()), True),
  ("min", lambda: np.min(_vec()), lambda: numpy.min(_np_vec()), True),
  ("max", lambda: np.max(_vec()), lambda: numpy.max(_np_vec()), True),
  ("all", lambda: np.all(_bools()), lambda: numpy.all(_np_bools()), True),
  ("any", lambda: np.any(_bools()), lambda: numpy.any(_np_bools()), True),
  ("argmax", lambda: np.argmax(_vec()), lambda: numpy.argmax(_np_vec()), True),
  ("argmin", lambda: np.argmin(_vec()), lambda: numpy.argmin(_np_vec()), True),
  ("count_nonzero", lambda: np.count_nonzero(_bools()), lambda: numpy.count_nonzero(_np_bools()), True),
  ("std", lambda: np.std(_vec()), lambda: numpy.std(_np_vec()), True),
  ("var", lambda: np.var(_vec()), lambda: numpy.var(_np_vec()), True),
  ("average", lambda: np.average(_vec()), lambda: numpy.average(_np_vec()), True),
  ("median", lambda: np.median(_vec()), lambda: numpy.median(_np_vec()), True),
  ("quantile", lambda: np.quantile(_vec(), 0.5), lambda: numpy.quantile(_np_vec(), 0.5), True),
  ("percentile", lambda: np.percentile(_vec(), 50), lambda: numpy.percentile(_np_vec(), 50), True),
  ("ptp", lambda: np.ptp(_vec()), lambda: numpy.ptp(_np_vec()), True),
  ("trace", lambda: np.trace(_mat()), lambda: numpy.trace(_np_mat()), True),
  ("searchsorted", lambda: np.searchsorted(_vec(), 2.5), lambda: numpy.searchsorted(_np_vec(), 2.5), True),
  ("dot", lambda: np.dot(_vec(), _vec2()), lambda: numpy.dot(_np_vec(), _np_vec2()), True),
  ("vdot", lambda: np.vdot(_vec(), _vec2()), lambda: numpy.vdot(_np_vec(), _np_vec2()), True),
  ("inner", lambda: np.inner(_vec(), _vec2()), lambda: numpy.inner(_np_vec(), _np_vec2()), True),
  ("matmul_1d", lambda: np.matmul(_vec(), _vec2()), lambda: numpy.matmul(_np_vec(), _np_vec2()), True),
  (
    "tensordot",
    lambda: np.tensordot(_mat(), _mat2(), axes=2),
    lambda: numpy.tensordot(_np_mat(), _np_mat2(), axes=2),
    True,
  ),
  ("einsum_dot", lambda: np.einsum("i,i", _vec(), _vec2()), lambda: numpy.einsum("i,i", _np_vec(), _np_vec2()), True),
  ("einsum_trace", lambda: np.einsum("ii", _mat()), lambda: numpy.einsum("ii", _np_mat()), True),
  ("linalg_det", lambda: np.linalg.det(_mat()), lambda: numpy.linalg.det(_np_mat()), True),
  ("slogdet_sign", lambda: np.linalg.slogdet(_mat())[0], lambda: numpy.linalg.slogdet(_np_mat())[0], True),
  ("slogdet_log", lambda: np.linalg.slogdet(_mat())[1], lambda: numpy.linalg.slogdet(_np_mat())[1], True),
  ("linalg_norm", lambda: np.linalg.norm(_vec()), lambda: numpy.linalg.norm(_np_vec()), True),
  ("linalg_trace", lambda: np.linalg.trace(_mat()), lambda: numpy.linalg.trace(_np_mat()), True),
  ("matrix_rank", lambda: np.linalg.matrix_rank(_mat()), lambda: numpy.linalg.matrix_rank(_np_mat()), True),
  (
    "lstsq_rank",
    lambda: np.linalg.lstsq(np.asarray([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), np.asarray([1.0, 2.0, 3.0]))[2],
    lambda: numpy.linalg.lstsq(
      numpy.asarray([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), numpy.asarray([1.0, 2.0, 3.0]), rcond=None
    )[2],
    False,
  ),
)


def test_scalar_producing_apis_match_numpy_result_kind() -> None:
  for name, monpy_fn, numpy_fn, check_dtype in SCALAR_ARRAY_CASES:
    actual = monpy_fn()
    expected = numpy_fn()

    assert_same_result_kind(actual, expected)
    if check_dtype:
      assert_same_shape_dtype(actual, expected)
    else:
      assert tuple(numpy.asarray(actual).shape) == tuple(numpy.asarray(expected).shape)
    assert_same_values(actual, expected, rtol=1e-10, atol=1e-10)
