from __future__ import annotations

import monumpy as np
import numpy
import pytest


def test_array_interface_exports_shape_dtype_data_and_strides() -> None:
  arr = np.arange(6, dtype=np.int64).reshape(2, 3)
  interface = arr.__array_interface__

  assert interface["version"] == 3
  assert interface["shape"] == (2, 3)
  assert interface["typestr"] == "<i8"
  assert interface["strides"] is None
  assert isinstance(interface["data"], tuple)
  assert interface["data"][1] is False


def test_numpy_asarray_consumes_monpy_array_interface() -> None:
  arr = np.arange(6, dtype=np.int64).reshape(2, 3)
  converted = numpy.asarray(arr)

  assert converted.dtype == numpy.int64
  assert converted.shape == (2, 3)
  assert converted.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_strided_view_exports_byte_strides_and_lifetime_owner() -> None:
  arr = np.arange(6, dtype=np.int64).reshape(2, 3)
  view = arr[:, ::-1]
  converted = numpy.asarray(view)

  assert view.strides == (24, -8)
  assert converted.strides == (24, -8)
  assert converted.tolist() == [[2, 1, 0], [5, 4, 3]]
  assert converted.base is not None


def test_dlpack_blocker_is_explicit() -> None:
  arr = np.asarray([1, 2, 3])

  assert arr.__dlpack_device__() == (1, 0)
  with pytest.raises(BufferError, match="dlpack"):
    arr.__dlpack__()
