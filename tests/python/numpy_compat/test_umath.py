from __future__ import annotations

import monumpy as np
import numpy
import pytest
from _helpers import assert_same_shape_dtype, assert_same_values


@pytest.mark.parametrize(
  ("monpy_dtype", "numpy_dtype", "rtol"),
  [(np.float32, numpy.float32, 1e-6), (np.float64, numpy.float64, 1e-12)],
)
@pytest.mark.parametrize("name", ["sin", "cos", "exp", "log"])
def test_unary_math_matches_numpy_float_dtypes(
  monpy_dtype: np.DType,
  numpy_dtype: type[numpy.floating],
  rtol: float,
  name: str,
) -> None:
  arr = np.asarray([0.25, 1.0, 2.0], dtype=monpy_dtype)
  oracle = numpy.asarray([0.25, 1.0, 2.0], dtype=numpy_dtype)
  out = getattr(np, name)(arr)
  numpy_out = getattr(numpy, name)(oracle)

  assert_same_shape_dtype(out, numpy_out)
  assert_same_values(out, numpy_out, rtol=rtol, atol=rtol)


def test_unary_math_preserves_float32_result_dtype() -> None:
  arr = np.asarray([0.25, 1.0, 2.0], dtype=np.float32)
  out = np.sin(arr)

  assert out.dtype == np.float32
  numpy.testing.assert_allclose(numpy.asarray(out), numpy.sin(numpy.asarray(arr)), rtol=1e-6)


def test_nan_and_inf_follow_ieee_float_behavior() -> None:
  arr = np.asarray([np.nan, np.inf, -1.0, 0.0, 1.0], dtype=np.float64)
  oracle = numpy.asarray([numpy.nan, numpy.inf, -1.0, 0.0, 1.0], dtype=numpy.float64)

  with numpy.errstate(invalid="ignore", divide="ignore"):
    out = np.log(arr)
    expected = numpy.log(oracle)

  assert_same_values(out, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("monpy_dtype", [np.float32, np.float64])
def test_log_negative_infinity_matches_numpy(monpy_dtype: np.DType) -> None:
  out = np.log(np.asarray([-np.inf], dtype=monpy_dtype))

  assert numpy.isnan(numpy.asarray(out)[0])


def test_unary_math_is_a_full_numpy_ufunc() -> None:
  arr = np.asarray([1.0, 2.0], dtype=np.float64)

  assert isinstance(np.sin, np.ufunc)
  assert hasattr(np.sin, "reduce")
  assert hasattr(np.sin, "outer")
  assert np.sin.nin == 1
  assert np.sin.nout == 1

  out_buf = np.empty_like(arr)
  np.sin(arr, out=out_buf)
  numpy.testing.assert_allclose(
    numpy.asarray(out_buf), numpy.sin(numpy.asarray([1.0, 2.0])), rtol=1e-12
  )
