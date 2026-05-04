from __future__ import annotations

from typing import Final

import monumpy as np
import numpy


SUPPORTED_DTYPE_PAIRS: Final = (
  (np.bool, numpy.bool_),
  (np.int64, numpy.int64),
  (np.float32, numpy.float32),
  (np.float64, numpy.float64),
)

MONPY_TO_NUMPY_DTYPE: Final = {
  monpy_dtype: numpy.dtype(numpy_dtype) for monpy_dtype, numpy_dtype in SUPPORTED_DTYPE_PAIRS
}


def numpy_dtype_for(monpy_dtype: np.DType) -> numpy.dtype:
  return MONPY_TO_NUMPY_DTYPE[monpy_dtype]


def assert_same_values(
  monpy_value: object,
  numpy_value: object,
  *,
  rtol: float = 1e-6,
  atol: float = 1e-6,
  equal_nan: bool = True,
) -> None:
  numpy.testing.assert_allclose(
    numpy.asarray(monpy_value),
    numpy.asarray(numpy_value),
    rtol=rtol,
    atol=atol,
    equal_nan=equal_nan,
  )


def assert_same_shape_dtype(monpy_value: object, numpy_value: object) -> None:
  monpy_array = monpy_value if isinstance(monpy_value, np.ndarray) else np.asarray(monpy_value)
  numpy_array = numpy.asarray(numpy_value)

  assert monpy_array.shape == tuple(numpy_array.shape)
  assert numpy_dtype_for(monpy_array.dtype) == numpy_array.dtype
