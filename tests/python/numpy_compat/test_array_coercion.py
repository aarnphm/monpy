from __future__ import annotations

import monumpy as np
import numpy
import pytest
from _helpers import SUPPORTED_DTYPE_PAIRS, assert_same_shape_dtype, assert_same_values


@pytest.mark.parametrize(
  ("obj", "expected_shape", "expected_dtype", "expected_values"),
  [
    ([], (0,), np.float64, []),
    (True, (), np.bool, True),
    (7, (), np.int64, 7),
    (3.5, (), np.float64, 3.5),
    ([1, 2, 3], (3,), np.int64, [1, 2, 3]),
    ([[1, 2, 3], [4, 5, 6]], (2, 3), np.int64, [[1, 2, 3], [4, 5, 6]]),
  ],
)
def test_array_like_inputs_infer_shape_dtype_and_values(
  obj: object,
  expected_shape: tuple[int, ...],
  expected_dtype: np.DType,
  expected_values: object,
) -> None:
  arr = np.asarray(obj)

  assert arr.shape == expected_shape
  assert arr.dtype == expected_dtype
  assert arr.tolist() == expected_values


def test_float_values_promote_to_default_real_dtype() -> None:
  arr = np.asarray([1, 2.5, 3])

  assert arr.dtype == np.float64
  assert arr.tolist() == [1.0, 2.5, 3.0]


def test_bool_values_remain_bool_dtype() -> None:
  arr = np.asarray([True, False, True])

  assert arr.dtype == np.bool
  assert arr.tolist() == [True, False, True]


@pytest.mark.parametrize("monpy_dtype, numpy_dtype", SUPPORTED_DTYPE_PAIRS)
def test_explicit_supported_dtype_casts_match_numpy(monpy_dtype: np.DType, numpy_dtype: type[numpy.generic]) -> None:
  values = [False, True, True] if monpy_dtype is np.bool else [1, 2, 3]
  arr = np.asarray(values, dtype=monpy_dtype)
  oracle = numpy.asarray(values, dtype=numpy_dtype)

  assert_same_shape_dtype(arr, oracle)
  assert_same_values(arr, oracle)


@pytest.mark.parametrize("src_monpy_dtype, src_numpy_dtype", SUPPORTED_DTYPE_PAIRS)
@pytest.mark.parametrize("dst_monpy_dtype, dst_numpy_dtype", SUPPORTED_DTYPE_PAIRS)
def test_astype_supported_cast_matrix_matches_numpy(
  src_monpy_dtype: np.DType,
  src_numpy_dtype: type[numpy.generic],
  dst_monpy_dtype: np.DType,
  dst_numpy_dtype: type[numpy.generic],
) -> None:
  values = [False, True, True] if src_monpy_dtype is np.bool else [0, 1, 2]
  arr = np.asarray(values, dtype=src_monpy_dtype)[::-1]
  oracle = numpy.asarray(values, dtype=src_numpy_dtype)[::-1]

  cast = arr.astype(dst_monpy_dtype)
  expected = oracle.astype(dst_numpy_dtype)

  assert_same_shape_dtype(cast, expected)
  assert_same_values(cast, expected)


@pytest.mark.parametrize(
  ("dtype", "expected_dtype"),
  [
    (numpy.bool_, np.bool),
    (numpy.int64, np.int64),
    (numpy.float32, np.float32),
    (numpy.float64, np.float64),
    (numpy.dtype("bool"), np.bool),
    (numpy.dtype("int64"), np.int64),
    (numpy.dtype("float32"), np.float32),
    (numpy.dtype("float64"), np.float64),
  ],
)
def test_numpy_dtype_aliases_resolve_to_supported_monpy_dtypes(dtype: object, expected_dtype: np.DType) -> None:
  arr = np.asarray([0, 1, 1], dtype=dtype)

  assert arr.dtype is expected_dtype


def test_scalar_array_is_zero_dimensional() -> None:
  arr = np.asarray(3.5, dtype=np.float32)

  assert arr.shape == ()
  assert arr.ndim == 0
  assert float(arr) == pytest.approx(3.5)


def test_ragged_nested_sequences_raise() -> None:
  with pytest.raises(ValueError, match="ragged"):
    np.asarray([[1], [2, 3]])


def test_asarray_returns_same_object_without_copy_or_cast() -> None:
  arr = np.asarray([1, 2, 3])

  assert np.asarray(arr) is arr
  assert np.asarray(arr, dtype=np.float32) is not arr


def test_array_and_asarray_copy_rules_for_existing_monpy_arrays() -> None:
  arr = np.asarray([1, 2, 3])

  assert np.array(arr, copy=False) is arr
  assert np.asarray(arr, copy=False) is arr
  assert np.array(arr, copy=True) is not arr
  assert np.asarray(arr, copy=True) is not arr


def test_astype_copy_false_keeps_identity_for_same_dtype() -> None:
  arr = np.asarray([1, 2, 3], dtype=np.int64)

  assert arr.astype(np.int64, copy=False) is arr
  assert arr.astype(np.float32, copy=False) is not arr


@pytest.mark.parametrize("monpy_dtype, numpy_dtype", SUPPORTED_DTYPE_PAIRS)
def test_numpy_array_inputs_import_shape_dtype_and_values(
  monpy_dtype: np.DType,
  numpy_dtype: type[numpy.generic],
) -> None:
  source = numpy.asarray([[0, 1, 1], [2, 3, 5]], dtype=numpy_dtype)
  arr = np.asarray(source, dtype=monpy_dtype)

  assert_same_shape_dtype(arr, source)
  assert_same_values(arr, source)


def test_numpy_array_copy_false_shares_storage() -> None:
  source = numpy.arange(6, dtype=numpy.int64).reshape(2, 3)
  arr = np.asarray(source, copy=False)

  source[0, 1] = 99
  arr[1, 2] = -5

  assert arr[0, 1] == 99
  assert source[1, 2] == -5


def test_numpy_array_copy_true_detaches_storage() -> None:
  source = numpy.arange(4, dtype=numpy.int64)
  arr = np.asarray(source, copy=True)

  source[0] = 99
  arr[1] = 77

  assert arr.tolist() == [0, 77, 2, 3]
  assert source.tolist() == [99, 1, 2, 3]


def test_numpy_array_function_defaults_to_detached_copy() -> None:
  source = numpy.arange(4, dtype=numpy.int64)
  arr = np.array(source)

  source[0] = 99
  arr[1] = 77

  assert arr.tolist() == [0, 77, 2, 3]
  assert source.tolist() == [99, 1, 2, 3]


def test_numpy_array_copy_none_allows_supported_cast() -> None:
  source = numpy.asarray([1, 2, 3], dtype=numpy.int64)
  arr = np.asarray(source, dtype=np.float32, copy=None)

  assert arr.dtype == np.float32
  assert arr.tolist() == [1.0, 2.0, 3.0]


def test_core_accepts_numpy_dtype_aliases_through_lazy_numpy_interop() -> None:
  arr = np.asarray([0, 1, 1], dtype=numpy.dtype("int64"))

  assert arr.dtype is np.int64


def test_numpy_array_copy_false_rejects_required_cast() -> None:
  source = numpy.asarray([1, 2, 3], dtype=numpy.int64)

  with pytest.raises(ValueError, match="copy"):
    np.asarray(source, dtype=np.float32, copy=False)


def test_numpy_array_readonly_copy_false_raises_and_copy_none_detaches() -> None:
  source = numpy.arange(4, dtype=numpy.int64)
  source.flags.writeable = False

  with pytest.raises(ValueError, match="readonly"):
    np.asarray(source, copy=False)

  arr = np.asarray(source, copy=None)
  arr[0] = 99

  assert arr.tolist() == [99, 1, 2, 3]
  assert source.tolist() == [0, 1, 2, 3]


def test_numpy_array_strided_view_copy_false_preserves_view_storage() -> None:
  source = numpy.arange(12, dtype=numpy.int64).reshape(3, 4)
  view = source[::2, ::-1]
  arr = np.asarray(view, copy=False)

  source[0, 3] = 99
  arr[1, 1] = -7

  assert arr.strides == view.strides
  assert arr.tolist() == [[99, 2, 1, 0], [11, -7, 9, 8]]
  assert source[2, 2] == -7


@pytest.mark.parametrize(
  "dtype",
  [
    "object",
    "str",
    "datetime64[ns]",
    "timedelta64[ns]",
    [("field", numpy.int64)],
  ],
)
def test_unsupported_dtype_requests_are_explicit_blockers(dtype: object) -> None:
  with pytest.raises(NotImplementedError, match="unsupported dtype"):
    np.asarray([1], dtype=dtype)


def test_phase5d_complex_dtype_allocation_works() -> None:
  arr = np.asarray([1 + 2j, 3 + 4j], dtype=np.complex128)
  assert arr.dtype == np.complex128
  assert arr.tolist() == [1 + 2j, 3 + 4j]
  doubled = arr + arr
  assert doubled.tolist() == [2 + 4j, 6 + 8j]
  squared = arr * arr
  assert squared.tolist() == [(1 + 2j) ** 2, (3 + 4j) ** 2]


@pytest.mark.parametrize("dtype_name", ["uint64", "uint32", "uint16", "uint8"])
def test_phase5b_unsigned_int_dtype_allocation_works(dtype_name: str) -> None:
  # Phase-5b unsigned ints land allocation + arithmetic via the f64 round-trip;
  # promotion follows numpy 2.x.
  dtype = getattr(np, dtype_name)
  arr = np.asarray([1, 2, 3], dtype=dtype)
  assert arr.dtype == dtype
  assert arr.tolist() == [1, 2, 3]
  doubled = arr + arr
  assert doubled.tolist() == [2, 4, 6]


def test_phase5c_float16_dtype_allocation_works() -> None:
  arr = np.asarray([0.5, 1.0, 2.0], dtype=np.float16)
  assert arr.dtype == np.float16
  assert arr.tolist() == [0.5, 1.0, 2.0]
  result = arr + arr
  assert result.tolist() == [1.0, 2.0, 4.0]
  assert result.dtype == np.float16


@pytest.mark.parametrize("dtype_name", ["int32", "int16", "int8"])
def test_phase5a_int_dtype_allocation_works(dtype_name: str) -> None:
  # Phase-5a int kernels landed: int32/16/8 are fully allocatable and arithmetic preserves dtype.
  dtype = getattr(np, dtype_name)
  arr = np.asarray([1, 2, 3], dtype=dtype)
  assert arr.dtype == dtype
  assert arr.tolist() == [1, 2, 3]
  doubled = arr + arr
  assert doubled.tolist() == [2, 4, 6]


@pytest.mark.parametrize("obj", [["monpy"], [object()]])
def test_unsupported_array_value_types_are_explicit_blockers(obj: object) -> None:
  with pytest.raises(NotImplementedError, match="unsupported array input type"):
    np.asarray(obj)


def test_complex_value_inputs_promote_to_default_complex_dtype() -> None:
  arr = np.asarray([1 + 2j, 3 + 4j])
  assert arr.dtype == np.complex128
  assert arr.tolist() == [1 + 2j, 3 + 4j]
