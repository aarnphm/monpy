from __future__ import annotations

import gc

import monumpy as np
import numpy
import pytest
from _helpers import SUPPORTED_DTYPE_PAIRS, assert_same_shape_dtype, assert_same_values


@pytest.mark.parametrize("monpy_dtype, numpy_dtype", SUPPORTED_DTYPE_PAIRS)
def test_array_interface_exports_supported_dtype_metadata(
  monpy_dtype: np.DType,
  numpy_dtype: type[numpy.generic],
) -> None:
  arr = np.asarray([0, 1, 1], dtype=monpy_dtype)
  interface = arr.__array_interface__

  assert interface["version"] == 3
  assert interface["shape"] == (3,)
  assert interface["typestr"] == numpy.dtype(numpy_dtype).str
  assert interface["strides"] is None
  assert isinstance(interface["data"], tuple)
  assert interface["data"][1] is False


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


def test_numpy_asarray_preserves_values_for_reversed_views() -> None:
  arr = np.arange(8, dtype=np.float32)
  view = arr[::-2]
  oracle = numpy.arange(8, dtype=numpy.float32)[::-2]

  assert_same_shape_dtype(view, oracle)
  assert_same_values(numpy.asarray(view), oracle)


def test_strided_view_exports_byte_strides_and_lifetime_owner() -> None:
  arr = np.arange(6, dtype=np.int64).reshape(2, 3)
  view = arr[:, ::-1]
  converted = numpy.asarray(view)

  assert view.strides == (24, -8)
  assert converted.strides == (24, -8)
  assert converted.tolist() == [[2, 1, 0], [5, 4, 3]]
  assert converted.base is not None


def test_numpy_exported_reversed_view_mutates_monpy_base() -> None:
  arr = np.arange(8, dtype=np.int64)
  view = arr[::-2]
  converted = numpy.asarray(view)

  converted[0] = 99

  assert arr.tolist() == [0, 1, 2, 3, 4, 5, 6, 99]
  assert view.tolist() == [99, 5, 3, 1]


def test_numpy_exported_view_keeps_monpy_owner_alive_after_source_scope() -> None:
  def exported_view() -> numpy.ndarray:
    arr = np.arange(8, dtype=np.int64)
    return numpy.asarray(arr[::-2])

  converted = exported_view()
  gc.collect()
  converted[1] = 77

  assert converted.tolist() == [7, 77, 3, 1]


def test_array_dunder_dtype_and_copy_arguments() -> None:
  arr = np.asarray([1, 2, 3], dtype=np.int64)

  view = arr.__array__(copy=False)
  view[0] = 99
  copied = arr.__array__(copy=True)
  copied[1] = 77
  cast = arr.__array__(dtype=numpy.float32)

  assert arr.tolist() == [99, 2, 3]
  assert copied.tolist() == [99, 77, 3]
  assert arr.tolist() == [99, 2, 3]
  assert cast.dtype == numpy.float32
  assert cast.tolist() == [99.0, 2.0, 3.0]


def test_dlpack_export_round_trips_to_numpy_cpu_protocol() -> None:
  arr = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
  exported = numpy.from_dlpack(arr)

  assert arr.__dlpack_device__() == (1, 0)
  assert exported.dtype == numpy.float64
  assert exported.shape == (2, 3)
  assert exported.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

  exported[0, 1] = 99.0

  assert arr[0, 1] == pytest.approx(99.0)


def test_dlpack_import_from_numpy_cpu_protocol_shares_when_copy_false() -> None:
  source = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
  arr = np.from_dlpack(source, device="cpu", copy=False)

  source[1, 2] = 42.0
  arr[0, 0] = -1.0

  assert_same_shape_dtype(arr, source)
  assert arr[1, 2] == pytest.approx(42.0)
  assert source[0, 0] == pytest.approx(-1.0)


def test_dlpack_import_copy_true_detaches_from_numpy_source() -> None:
  source = numpy.arange(4, dtype=numpy.int64)
  arr = np.from_dlpack(source, copy=True)

  source[0] = 99
  arr[1] = 77

  assert arr.tolist() == [0, 77, 2, 3]
  assert source.tolist() == [99, 1, 2, 3]


def test_dlpack_exported_numpy_view_keeps_monpy_owner_alive_after_source_scope() -> None:
  def exported_from_dlpack() -> numpy.ndarray:
    arr = np.arange(5, dtype=np.float32)
    return numpy.from_dlpack(arr)

  converted = exported_from_dlpack()
  gc.collect()
  converted[2] = 8.5

  assert converted.tolist() == [0.0, 1.0, 8.5, 3.0, 4.0]
