from __future__ import annotations

import operator
from collections.abc import Callable

import monpy
import monumpy as np
import numpy
import pytest
from _helpers import MONPY_TO_NUMPY_DTYPE, SUPPORTED_DTYPE_PAIRS, assert_same_values
from monpy.numpy import ops as ops_numpy

PROMOTION_MATCH_CASES = [
  (lhs_dtype, lhs_numpy_dtype, rhs_dtype, rhs_numpy_dtype)
  for lhs_dtype, lhs_numpy_dtype in SUPPORTED_DTYPE_PAIRS
  for rhs_dtype, rhs_numpy_dtype in SUPPORTED_DTYPE_PAIRS
]
CASTING_CODES = {
  "no": monpy.CASTING_NO,
  "equiv": monpy.CASTING_EQUIV,
  "safe": monpy.CASTING_SAFE,
  "same_kind": monpy.CASTING_SAME_KIND,
  "unsafe": monpy.CASTING_UNSAFE,
}


def values_for_dtype(monpy_dtype: np.DType) -> list[bool] | list[int] | list[float]:
  if monpy_dtype is np.bool:
    return [True, False]
  if monpy_dtype is np.int64:
    return [1, 2]
  return [1.0, 2.0]


@pytest.mark.parametrize(("lhs_dtype", "lhs_numpy_dtype", "rhs_dtype", "rhs_numpy_dtype"), PROMOTION_MATCH_CASES)
def test_supported_array_promotions_match_numpy(
  lhs_dtype: np.DType,
  lhs_numpy_dtype: type[numpy.generic],
  rhs_dtype: np.DType,
  rhs_numpy_dtype: type[numpy.generic],
) -> None:
  lhs = np.asarray(values_for_dtype(lhs_dtype), dtype=lhs_dtype)
  rhs = np.asarray(values_for_dtype(rhs_dtype), dtype=rhs_dtype)
  oracle_lhs = numpy.asarray(values_for_dtype(lhs_dtype), dtype=lhs_numpy_dtype)
  oracle_rhs = numpy.asarray(values_for_dtype(rhs_dtype), dtype=rhs_numpy_dtype)
  out = lhs + rhs
  expected = oracle_lhs + oracle_rhs

  assert MONPY_TO_NUMPY_DTYPE[out.dtype] == expected.dtype
  assert_same_values(out, expected)


@pytest.mark.parametrize("scalar", [1, 1.5, True])
@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul])
def test_python_scalars_are_weak_for_float32_arrays(
  scalar: int | float | bool,
  op: Callable[[object, object], object],
) -> None:
  arr = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
  oracle = numpy.asarray([1.0, 2.0, 3.0], dtype=numpy.float32)
  out = op(arr, scalar)
  expected = op(oracle, scalar)

  assert out.dtype == np.float32
  assert_same_values(out, expected)


@pytest.mark.parametrize(
  "dtype",
  # uint / f16 / complex have dedicated allocation coverage; see
  # test_array_coercion.py::test_unsigned_int_dtype_allocation_works,
  # ::test_float16_dtype_allocation_works, and complex dtype tests.
  ["object", "str"],
)
def test_unsupported_promotion_dtype_families_are_explicit_blockers(dtype: object) -> None:
  with pytest.raises(NotImplementedError, match="unsupported dtype"):
    np.asarray([1], dtype=dtype)


@pytest.mark.parametrize(
  ("lhs", "rhs", "expected"),
  [
    (np.uint8, np.uint16, np.uint16),
    (np.uint16, np.uint32, np.uint32),
    (np.uint32, np.uint64, np.uint64),
    (np.uint8, np.int8, np.int16),
    (np.uint16, np.int16, np.int32),
    (np.uint32, np.int32, np.int64),
    (np.uint64, np.int64, np.float64),
    (np.uint8, np.float32, np.float32),
    (np.uint32, np.float32, np.float64),
    (np.float16, np.float32, np.float32),
    (np.float16, np.uint8, np.float16),
    (np.float16, np.int16, np.float32),
  ],
)
def test_extended_dtype_promotion_matches_numpy(lhs: object, rhs: object, expected: object) -> None:
  # Mirror numpy's promotion rules at the python boundary; the underlying
  # dispatch falls back to `_native._result_dtype_for_binary` for any pair
  # outside the original 4×4 fast-path table.
  out = np.result_type(lhs, rhs)
  assert out == expected
  oracle = numpy.result_type(numpy.dtype(lhs.name), numpy.dtype(rhs.name))
  assert numpy.dtype(out.name) == oracle


@pytest.mark.parametrize(("monpy_dtype", "numpy_dtype"), SUPPORTED_DTYPE_PAIRS)
def test_supported_dtype_metadata_matches_numpy_and_native_registry(
  monpy_dtype: np.DType,
  numpy_dtype: type[numpy.generic],
) -> None:
  oracle = numpy.dtype(numpy_dtype)

  assert monpy_dtype.kind == oracle.kind
  assert monpy_dtype.itemsize == oracle.itemsize
  assert monpy_dtype.alignment == oracle.alignment
  assert monpy_dtype.byteorder == oracle.byteorder
  assert monpy_dtype.typestr == oracle.str
  assert monpy_dtype.format == oracle.char
  assert monpy._native._dtype_itemsize(monpy_dtype.code) == oracle.itemsize
  assert monpy._native._dtype_alignment(monpy_dtype.code) == oracle.alignment
  assert monpy._native._dtype_kind(monpy_dtype.code) == monpy._DTK[monpy_dtype]


def test_native_domain_codes_feed_python_enums_and_aliases() -> None:
  codes = monpy._native._domain_codes()

  assert codes["dtype"]["FLOAT32"] == monpy.DTYPE_FLOAT32 == monpy.DTypeCode["FLOAT32"].value
  assert codes["casting"]["SAME_KIND"] == monpy.CASTING_SAME_KIND == monpy.CastingRule["SAME_KIND"].value
  assert codes["binary"]["ADD"] == monpy.OP_ADD == monpy.BinaryOp["ADD"].value
  assert codes["unary"]["SIN"] == monpy.UNARY_SIN == monpy.UnaryOp["SIN"].value
  assert codes["compare"]["LE"] == monpy.CMP_LE == monpy.CompareOp["LE"].value
  assert codes["logical"]["XOR"] == monpy.LOGIC_XOR == monpy.LogicalOp["XOR"].value
  assert codes["predicate"]["SIGNBIT"] == monpy.PRED_SIGNBIT == monpy.PredicateOp["SIGNBIT"].value
  assert codes["reduce"]["ARGMIN"] == monpy.REDUCE_ARGMIN == monpy.ReduceOp["ARGMIN"].value
  assert codes["backend"]["FUSED"] == monpy.BackendKind["FUSED"].value


@pytest.mark.parametrize(("monpy_dtype", "numpy_dtype"), SUPPORTED_DTYPE_PAIRS)
def test_numpy_dtype_inputs_resolve_through_lazy_numpy_interop(
  monpy_dtype: np.DType,
  numpy_dtype: type[numpy.generic],
) -> None:
  assert monpy.dtype(numpy_dtype) == monpy_dtype
  assert monpy.dtype(numpy.dtype(numpy_dtype)) == monpy_dtype
  assert monpy_dtype == numpy_dtype
  assert monpy_dtype == numpy.dtype(numpy_dtype)
  assert ops_numpy.resolve_dtype(numpy_dtype) == monpy_dtype
  assert ops_numpy.resolve_dtype(numpy.dtype(numpy_dtype)) == monpy_dtype


@pytest.mark.parametrize(("lhs_dtype", "lhs_numpy_dtype", "rhs_dtype", "rhs_numpy_dtype"), PROMOTION_MATCH_CASES)
def test_promote_types_and_result_type_match_numpy(
  lhs_dtype: np.DType,
  lhs_numpy_dtype: type[numpy.generic],
  rhs_dtype: np.DType,
  rhs_numpy_dtype: type[numpy.generic],
) -> None:
  expected = numpy.promote_types(lhs_numpy_dtype, rhs_numpy_dtype)

  assert (
    np.promote_types(lhs_dtype, rhs_dtype) == monpy._DTC[monpy._native._promote_types(lhs_dtype.code, rhs_dtype.code)]
  )
  assert MONPY_TO_NUMPY_DTYPE[np.promote_types(lhs_dtype, rhs_dtype)] == expected
  assert MONPY_TO_NUMPY_DTYPE[np.result_type(lhs_dtype, rhs_dtype)] == numpy.result_type(
    lhs_numpy_dtype, rhs_numpy_dtype
  )


@pytest.mark.parametrize("scalar", [True, 1, 1.5])
@pytest.mark.parametrize(("monpy_dtype", "numpy_dtype"), SUPPORTED_DTYPE_PAIRS)
def test_result_type_keeps_python_scalars_weak_around_arrays(
  monpy_dtype: np.DType,
  numpy_dtype: type[numpy.generic],
  scalar: bool | int | float,
) -> None:
  arr = np.asarray([1], dtype=monpy_dtype)
  oracle = numpy.asarray([1], dtype=numpy_dtype)

  assert MONPY_TO_NUMPY_DTYPE[np.result_type(arr, scalar)] == numpy.result_type(oracle, scalar)


@pytest.mark.parametrize("casting", list(CASTING_CODES))
@pytest.mark.parametrize(("lhs_dtype", "lhs_numpy_dtype", "rhs_dtype", "rhs_numpy_dtype"), PROMOTION_MATCH_CASES)
def test_can_cast_matches_numpy_for_supported_dtypes(
  lhs_dtype: np.DType,
  lhs_numpy_dtype: type[numpy.generic],
  rhs_dtype: np.DType,
  rhs_numpy_dtype: type[numpy.generic],
  casting: str,
) -> None:
  expected = numpy.can_cast(lhs_numpy_dtype, rhs_numpy_dtype, casting=casting)

  assert np.can_cast(lhs_dtype, rhs_dtype, casting=casting) is bool(expected)
  assert monpy._native._can_cast(lhs_dtype.code, rhs_dtype.code, CASTING_CODES[casting]) is bool(expected)


@pytest.mark.parametrize("scalar", [True, 1, 1.5])
def test_can_cast_rejects_python_scalars_like_numpy(scalar: bool | int | float) -> None:
  with pytest.raises(TypeError, match="does not support Python"):
    np.can_cast(scalar, np.float64)


@pytest.mark.parametrize(("monpy_dtype", "numpy_dtype"), SUPPORTED_DTYPE_PAIRS)
def test_dtype_kind_queries_match_numpy(
  monpy_dtype: np.DType,
  numpy_dtype: type[numpy.generic],
) -> None:
  oracle = numpy.dtype(numpy_dtype)
  for kind in [
    "bool",
    "signed integer",
    "unsigned integer",
    "integral",
    "real floating",
    "complex floating",
    "numeric",
  ]:
    assert np.isdtype(monpy_dtype, kind) is bool(numpy.isdtype(oracle, kind))
  assert np.issubdtype(monpy_dtype, monpy_dtype)
  assert np.issubdtype(monpy_dtype, numpy.integer) is bool(numpy.issubdtype(oracle, numpy.integer))


def test_finfo_and_iinfo_match_numpy_for_supported_numeric_limits() -> None:
  for monpy_dtype, numpy_dtype in [(np.float32, numpy.float32), (np.float64, numpy.float64)]:
    out = np.finfo(monpy_dtype)
    expected = numpy.finfo(numpy_dtype)
    assert out.bits == expected.bits
    assert out.eps == pytest.approx(float(expected.eps))
    assert out.max == pytest.approx(float(expected.max))
    assert out.tiny == pytest.approx(float(expected.tiny))

  out_i = np.iinfo(np.int64)
  expected_i = numpy.iinfo(numpy.int64)
  assert out_i.bits == expected_i.bits
  assert out_i.min == expected_i.min
  assert out_i.max == expected_i.max

  with pytest.raises(ValueError, match="not inexact"):
    np.finfo(np.int64)
  with pytest.raises(ValueError, match="Invalid integer"):
    np.iinfo(np.float64)


@pytest.mark.parametrize(("lhs_dtype", "lhs_numpy_dtype", "rhs_dtype", "rhs_numpy_dtype"), PROMOTION_MATCH_CASES)
@pytest.mark.parametrize("op", [monpy.OP_ADD, monpy.OP_SUB, monpy.OP_MUL, monpy.OP_DIV])
def test_python_and_extension_promotion_tables_agree(
  lhs_dtype: np.DType,
  lhs_numpy_dtype: type[numpy.generic],
  rhs_dtype: np.DType,
  rhs_numpy_dtype: type[numpy.generic],
  op: int,
) -> None:
  del lhs_numpy_dtype, rhs_numpy_dtype

  expected = monpy._result_dtype_for_binary(lhs_dtype, rhs_dtype, op)
  extension_code = monpy._native._result_dtype_for_binary(lhs_dtype.code, rhs_dtype.code, op)

  assert extension_code == expected.code


@pytest.mark.parametrize("value", [2**70, -(2**70)])
def test_python_integer_overflow_semantics_are_deferred_for_float32(value: int) -> None:
  arr = np.asarray([1.0], dtype=np.float32)
  out = arr + value

  assert out.dtype == np.float32
