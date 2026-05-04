from __future__ import annotations

import monumpy as np
import numpy


def test_unary_math_matches_numpy_float64() -> None:
  arr = np.asarray([0.25, 1.0, 2.0], dtype=np.float64)
  oracle = numpy.asarray([0.25, 1.0, 2.0], dtype=numpy.float64)

  numpy.testing.assert_allclose(numpy.asarray(np.sin(arr)), numpy.sin(oracle), rtol=1e-12)
  numpy.testing.assert_allclose(numpy.asarray(np.cos(arr)), numpy.cos(oracle), rtol=1e-12)
  numpy.testing.assert_allclose(numpy.asarray(np.exp(arr)), numpy.exp(oracle), rtol=1e-12)
  numpy.testing.assert_allclose(numpy.asarray(np.log(arr)), numpy.log(oracle), rtol=1e-12)


def test_unary_math_preserves_float32_result_dtype() -> None:
  arr = np.asarray([0.25, 1.0, 2.0], dtype=np.float32)
  out = np.sin(arr)

  assert out.dtype == np.float32
  numpy.testing.assert_allclose(numpy.asarray(out), numpy.sin(numpy.asarray(arr)), rtol=1e-6)


def test_nan_and_inf_follow_ieee_float_behavior() -> None:
  arr = np.asarray([np.nan, np.inf, 1.0], dtype=np.float64)
  out = np.log(arr)

  assert numpy.isnan(numpy.asarray(out)[0])
  assert numpy.isinf(numpy.asarray(out)[1])
  assert numpy.asarray(out)[2] == 0.0
