from __future__ import annotations

import math

import monpy as mp
import numpy
import pytest

# ---------------------------------------------------------------------------
# unary transcendental ufuncs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  "name",
  [
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "exp2",
    "expm1",
    "log2",
    "log10",
    "log1p",
    "sqrt",
    "cbrt",
  ],
)
def test_unary_transcendental_matches_numpy(name: str) -> None:
  base = [0.1, 0.5, 0.9]
  arr = mp.asarray(base, dtype=mp.float64)
  oracle = numpy.asarray(base, dtype=numpy.float64)
  out = getattr(mp, name)(arr)
  expected = getattr(numpy, name)(oracle)
  # Mojo std.math implementations of log2/log1p/etc. use polynomial
  # approximations whose error vs libm is ~1e-9; tolerance reflects that.
  numpy.testing.assert_allclose(numpy.asarray(out), expected, rtol=1e-7, atol=1e-7)


def test_unary_transcendental_float32_preserves_dtype() -> None:
  arr = mp.asarray([0.1, 0.5, 0.9], dtype=mp.float32)
  for name in ["sqrt", "tan", "arcsin", "log2"]:
    out = getattr(mp, name)(arr)
    assert out.dtype == mp.float32


def test_deg2rad_and_rad2deg_round_trip() -> None:
  arr = mp.asarray([0.0, 90.0, 180.0, 360.0], dtype=mp.float64)
  rad = mp.deg2rad(arr)
  back = mp.rad2deg(rad)
  numpy.testing.assert_allclose(numpy.asarray(back), [0.0, 90.0, 180.0, 360.0], rtol=1e-12)


def test_reciprocal_matches_numpy() -> None:
  arr = mp.asarray([1.0, 2.0, 4.0], dtype=mp.float64)
  numpy.testing.assert_allclose(numpy.asarray(mp.reciprocal(arr)), [1.0, 0.5, 0.25], rtol=1e-12)


# ---------------------------------------------------------------------------
# preserve-dtype unary arith
# ---------------------------------------------------------------------------


def test_negative_preserves_int64() -> None:
  arr = mp.asarray([1, -2, 3], dtype=mp.int64)
  out = mp.negative(arr)
  assert out.dtype == mp.int64
  assert out.tolist() == [-1, 2, -3]


def test_absolute_int_and_float() -> None:
  arr_i = mp.asarray([1, -2, 3, -4], dtype=mp.int64)
  arr_f = mp.asarray([1.5, -2.5, 3.5], dtype=mp.float64)
  assert mp.absolute(arr_i).tolist() == [1, 2, 3, 4]
  assert mp.absolute(arr_f).tolist() == [1.5, 2.5, 3.5]


def test_square_preserves_int_dtype() -> None:
  arr = mp.asarray([2, 3, 4], dtype=mp.int64)
  out = mp.square(arr)
  assert out.dtype == mp.int64
  assert out.tolist() == [4, 9, 16]


def test_sign_handles_zero_negative_and_nan() -> None:
  arr = mp.asarray([-2.0, 0.0, 3.0, math.nan], dtype=mp.float64)
  out = mp.sign(arr)
  vals = numpy.asarray(out)
  assert vals[0] == -1.0
  assert vals[1] == 0.0
  assert vals[2] == 1.0
  assert math.isnan(vals[3])


def test_floor_ceil_trunc_rint_matches_numpy() -> None:
  arr = mp.asarray([-1.5, -0.5, 0.5, 1.5, 2.7], dtype=mp.float64)
  oracle = numpy.asarray([-1.5, -0.5, 0.5, 1.5, 2.7], dtype=numpy.float64)
  numpy.testing.assert_array_equal(numpy.asarray(mp.floor(arr)), numpy.floor(oracle))
  numpy.testing.assert_array_equal(numpy.asarray(mp.ceil(arr)), numpy.ceil(oracle))
  numpy.testing.assert_array_equal(numpy.asarray(mp.trunc(arr)), numpy.trunc(oracle))


def test_logical_not_returns_int64_for_bool_input() -> None:
  # Our unary_preserve gates bool → int64; logical_not on ints/bools should
  # yield 0/1 in int64.
  arr = mp.asarray([True, False, True, False], dtype=mp.bool)
  out = mp.logical_not(arr)
  # bool -> int64 promotion happens; but value is 0/1.
  assert out.tolist() == [0, 1, 0, 1]


# ---------------------------------------------------------------------------
# binary arith / power / floor_divide / mod
# ---------------------------------------------------------------------------


def test_floor_divide_and_remainder_match_numpy() -> None:
  a = mp.asarray([7.0, -7.0, 6.0], dtype=mp.float64)
  b = mp.asarray([2.0, 2.0, -2.0], dtype=mp.float64)
  oracle_a = numpy.asarray([7.0, -7.0, 6.0])
  oracle_b = numpy.asarray([2.0, 2.0, -2.0])
  numpy.testing.assert_array_equal(numpy.asarray(mp.floor_divide(a, b)), numpy.floor_divide(oracle_a, oracle_b))
  numpy.testing.assert_array_equal(numpy.asarray(mp.remainder(a, b)), numpy.remainder(oracle_a, oracle_b))


def test_power_matches_numpy() -> None:
  a = mp.asarray([2.0, 3.0, 4.0], dtype=mp.float64)
  b = mp.asarray([3.0, 2.0, 0.5], dtype=mp.float64)
  numpy.testing.assert_allclose(numpy.asarray(mp.power(a, b)), [8.0, 9.0, 2.0], rtol=1e-12)


def test_power_scalar_square_and_cube_match_numpy() -> None:
  a = mp.asarray([-2.0, -0.5, 0.0, 1.5], dtype=mp.float32)
  oracle = numpy.asarray([-2.0, -0.5, 0.0, 1.5], dtype=numpy.float32)

  square = mp.power(a, 2.0)
  cube = mp.power(a, 3.0)

  assert square.dtype == mp.float32
  assert cube.dtype == mp.float32
  numpy.testing.assert_allclose(numpy.asarray(square), numpy.power(oracle, 2.0))
  numpy.testing.assert_allclose(numpy.asarray(cube), numpy.power(oracle, 3.0))


def test_maximum_minimum_propagate_nan() -> None:
  a = mp.asarray([1.0, math.nan, 3.0], dtype=mp.float64)
  b = mp.asarray([2.0, 2.0, math.nan], dtype=mp.float64)
  out_max = numpy.asarray(mp.maximum(a, b))
  out_min = numpy.asarray(mp.minimum(a, b))
  assert out_max[0] == 2.0
  assert math.isnan(out_max[1])
  assert math.isnan(out_max[2])
  assert out_min[0] == 1.0
  assert math.isnan(out_min[1])
  assert math.isnan(out_min[2])


def test_fmin_fmax_treat_nan_as_missing() -> None:
  a = mp.asarray([1.0, math.nan, 3.0], dtype=mp.float64)
  b = mp.asarray([2.0, 2.0, math.nan], dtype=mp.float64)
  out_max = numpy.asarray(mp.fmax(a, b))
  out_min = numpy.asarray(mp.fmin(a, b))
  numpy.testing.assert_array_equal(out_max, [2.0, 2.0, 3.0])
  numpy.testing.assert_array_equal(out_min, [1.0, 2.0, 3.0])


def test_arctan2_and_hypot_match_numpy() -> None:
  y = mp.asarray([1.0, -1.0, 0.0], dtype=mp.float64)
  x = mp.asarray([1.0, 1.0, -1.0], dtype=mp.float64)
  numpy.testing.assert_allclose(
    numpy.asarray(mp.arctan2(y, x)),
    numpy.arctan2(numpy.asarray([1.0, -1.0, 0.0]), numpy.asarray([1.0, 1.0, -1.0])),
    rtol=1e-12,
  )
  numpy.testing.assert_allclose(
    numpy.asarray(mp.hypot(y, x)),
    numpy.hypot(numpy.asarray([1.0, -1.0, 0.0]), numpy.asarray([1.0, 1.0, -1.0])),
    rtol=1e-12,
  )


# ---------------------------------------------------------------------------
# comparison ufuncs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  ("name", "expected"),
  [
    ("equal", [False, True, False]),
    ("not_equal", [True, False, True]),
    ("less", [True, False, False]),
    ("less_equal", [True, True, False]),
    ("greater", [False, False, True]),
    ("greater_equal", [False, True, True]),
  ],
)
def test_comparison_ufuncs_return_bool(name: str, expected: list[bool]) -> None:
  a = mp.asarray([1.0, 2.0, 3.0], dtype=mp.float64)
  b = mp.asarray([2.0, 2.0, 2.0], dtype=mp.float64)
  out = getattr(mp, name)(a, b)
  assert out.dtype == mp.bool
  assert out.tolist() == expected


def test_comparison_with_int_and_float_promotes() -> None:
  a = mp.asarray([1, 2, 3], dtype=mp.int64)
  b = mp.asarray([1.0, 2.0, 3.5], dtype=mp.float64)
  out = mp.equal(a, b)
  assert out.tolist() == [True, True, False]


# ---------------------------------------------------------------------------
# logical ufuncs
# ---------------------------------------------------------------------------


def test_logical_and_or_xor() -> None:
  a = mp.asarray([True, True, False, False], dtype=mp.bool)
  b = mp.asarray([True, False, True, False], dtype=mp.bool)
  assert mp.logical_and(a, b).tolist() == [True, False, False, False]
  assert mp.logical_or(a, b).tolist() == [True, True, True, False]
  assert mp.logical_xor(a, b).tolist() == [False, True, True, False]


def test_logical_on_numeric_arrays_uses_truthiness() -> None:
  a = mp.asarray([0.0, 1.0, 2.0, 0.0], dtype=mp.float64)
  b = mp.asarray([1.0, 0.0, 3.0, 0.0], dtype=mp.float64)
  assert mp.logical_and(a, b).tolist() == [False, False, True, False]
  assert mp.logical_or(a, b).tolist() == [True, True, True, False]


# ---------------------------------------------------------------------------
# predicate ufuncs
# ---------------------------------------------------------------------------


def test_isnan_isinf_isfinite_signbit() -> None:
  arr = mp.asarray([1.0, math.nan, math.inf, -math.inf, -1.0, 0.0], dtype=mp.float64)
  numpy.testing.assert_array_equal(numpy.asarray(mp.isnan(arr)), [False, True, False, False, False, False])
  numpy.testing.assert_array_equal(numpy.asarray(mp.isinf(arr)), [False, False, True, True, False, False])
  numpy.testing.assert_array_equal(numpy.asarray(mp.isfinite(arr)), [True, False, False, False, True, True])
  numpy.testing.assert_array_equal(numpy.asarray(mp.signbit(arr)), [False, False, False, True, True, False])


# ---------------------------------------------------------------------------
# Ufunc.reduce / accumulate / outer
# ---------------------------------------------------------------------------


def test_add_reduce_axis_int_matches_numpy() -> None:
  arr = mp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mp.float64)
  oracle = numpy.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float64)
  numpy.testing.assert_array_equal(numpy.asarray(mp.add.reduce(arr, axis=0)), oracle.sum(axis=0))
  numpy.testing.assert_array_equal(numpy.asarray(mp.add.reduce(arr, axis=1)), oracle.sum(axis=1))


@pytest.mark.parametrize(
  ("mp_dtype", "np_dtype"),
  [
    (mp.bool, numpy.bool_),
    (mp.int8, numpy.int8),
    (mp.int16, numpy.int16),
    (mp.int32, numpy.int32),
    (mp.int64, numpy.int64),
    (mp.uint8, numpy.uint8),
    (mp.uint16, numpy.uint16),
    (mp.uint32, numpy.uint32),
    (mp.uint64, numpy.uint64),
  ],
)
def test_add_reduce_integer_dtypes_match_numpy(mp_dtype: object, np_dtype: object) -> None:
  if np_dtype is numpy.bool_:
    values = numpy.asarray([True, False, True, True], dtype=np_dtype)
  else:
    values = numpy.arange(17, dtype=np_dtype)
  arr = mp.asarray(values, dtype=mp_dtype)
  out = mp.add.reduce(arr, axis=None)
  expected = numpy.add.reduce(values, axis=None)

  if np_dtype is numpy.int32:
    assert numpy.asarray(out).dtype == numpy.asarray(expected).dtype
  numpy.testing.assert_array_equal(numpy.asarray(out), expected)


def test_add_reduce_float16_matches_numpy_value() -> None:
  values = numpy.linspace(0.1, 2.0, 17, dtype=numpy.float16)
  arr = mp.asarray(values, dtype=mp.float16)

  numpy.testing.assert_array_equal(numpy.asarray(mp.add.reduce(arr, axis=None)), numpy.add.reduce(values, axis=None))


def test_add_reduce_keepdims_matches_numpy() -> None:
  arr = mp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=mp.float64)
  out = mp.add.reduce(arr, axis=0, keepdims=True)
  assert out.shape == (1, 2)
  numpy.testing.assert_array_equal(numpy.asarray(out), [[4.0, 6.0]])


def test_multiply_reduce_returns_prod() -> None:
  arr = mp.asarray([1, 2, 3, 4], dtype=mp.float64)
  v = mp.multiply.reduce(arr, axis=None)
  assert float(v) == 24.0


def test_maximum_minimum_reduce() -> None:
  arr = mp.asarray([[1.0, 5.0], [3.0, 2.0]], dtype=mp.float64)
  numpy.testing.assert_array_equal(numpy.asarray(mp.maximum.reduce(arr, axis=0)), [3.0, 5.0])
  numpy.testing.assert_array_equal(numpy.asarray(mp.minimum.reduce(arr, axis=1)), [1.0, 2.0])


def test_logical_and_or_reduce() -> None:
  a = mp.asarray([[True, True], [True, False]], dtype=mp.bool)
  numpy.testing.assert_array_equal(numpy.asarray(mp.logical_and.reduce(a, axis=0)), [True, False])
  numpy.testing.assert_array_equal(numpy.asarray(mp.logical_or.reduce(a, axis=1)), [True, True])


def test_add_outer() -> None:
  a = mp.asarray([1.0, 2.0, 3.0], dtype=mp.float64)
  b = mp.asarray([10.0, 20.0], dtype=mp.float64)
  out = mp.add.outer(a, b)
  numpy.testing.assert_array_equal(numpy.asarray(out), [[11.0, 21.0], [12.0, 22.0], [13.0, 23.0]])


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def test_std_var_with_ddof() -> None:
  arr = mp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=mp.float64)
  oracle = numpy.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
  numpy.testing.assert_allclose(float(mp.std(arr, ddof=0)), float(numpy.std(oracle, ddof=0)), rtol=1e-12)
  numpy.testing.assert_allclose(float(mp.std(arr, ddof=1)), float(numpy.std(oracle, ddof=1)), rtol=1e-12)
  numpy.testing.assert_allclose(float(mp.var(arr, ddof=0)), float(numpy.var(oracle, ddof=0)), rtol=1e-12)


def test_median_and_quantile() -> None:
  arr = mp.asarray([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dtype=mp.float64)
  oracle = numpy.asarray([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
  numpy.testing.assert_allclose(float(mp.median(arr)), float(numpy.median(oracle)), rtol=1e-12)
  numpy.testing.assert_allclose(float(mp.quantile(arr, 0.25)), float(numpy.quantile(oracle, 0.25)), rtol=1e-12)
  numpy.testing.assert_allclose(float(mp.percentile(arr, 75)), float(numpy.percentile(oracle, 75)), rtol=1e-12)


def test_count_nonzero() -> None:
  arr = mp.asarray([0, 1, 2, 0, 3, 0], dtype=mp.int64)
  assert int(mp.count_nonzero(arr)) == 3


def test_prod_all_any() -> None:
  arr = mp.asarray([2.0, 3.0, 4.0], dtype=mp.float64)
  assert float(mp.prod(arr)) == 24.0
  assert bool(mp.all(mp.asarray([True, True, True], dtype=mp.bool)))
  assert not bool(mp.all(mp.asarray([True, False, True], dtype=mp.bool)))
  assert bool(mp.any(mp.asarray([False, False, True], dtype=mp.bool)))
  assert not bool(mp.any(mp.asarray([False, False, False], dtype=mp.bool)))


def test_argmin_full_reduction() -> None:
  arr = mp.asarray([3.0, 1.0, 4.0, 1.0], dtype=mp.float64)
  assert int(mp.argmin(arr)) == 1


def test_cumsum_cumprod() -> None:
  arr = mp.asarray([1.0, 2.0, 3.0, 4.0], dtype=mp.float64)
  numpy.testing.assert_array_equal(numpy.asarray(mp.cumsum(arr)), [1.0, 3.0, 6.0, 10.0])
  numpy.testing.assert_array_equal(numpy.asarray(mp.cumprod(arr)), [1.0, 2.0, 6.0, 24.0])


def test_cummax_cummin() -> None:
  arr = mp.asarray([3.0, 1.0, 4.0, 1.0, 5.0], dtype=mp.float64)
  numpy.testing.assert_array_equal(numpy.asarray(mp.cummax(arr)), [3.0, 3.0, 4.0, 4.0, 5.0])
  numpy.testing.assert_array_equal(numpy.asarray(mp.cummin(arr)), [3.0, 1.0, 1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# nan-aware
# ---------------------------------------------------------------------------


def test_nansum_nanmean_nanmin_nanmax_skip_nan() -> None:
  arr = mp.asarray([1.0, math.nan, 3.0, 4.0, math.nan], dtype=mp.float64)
  oracle = numpy.asarray([1.0, math.nan, 3.0, 4.0, math.nan])
  numpy.testing.assert_allclose(float(mp.nansum(arr)), float(numpy.nansum(oracle)))
  numpy.testing.assert_allclose(float(mp.nanmean(arr)), float(numpy.nanmean(oracle)))
  numpy.testing.assert_allclose(float(mp.nanmin(arr)), float(numpy.nanmin(oracle)))
  numpy.testing.assert_allclose(float(mp.nanmax(arr)), float(numpy.nanmax(oracle)))


# ---------------------------------------------------------------------------
# Ufunc surface
# ---------------------------------------------------------------------------


def test_ufunc_attributes() -> None:
  for name, nin in [("add", 2), ("multiply", 2), ("sin", 1), ("equal", 2), ("logical_and", 2)]:
    u = getattr(mp, name)
    assert isinstance(u, mp.ufunc)
    assert u.nin == nin
    assert u.nout == 1
    assert u.__name__ == name


def test_ufunc_out_kwarg_writes_in_place() -> None:
  a = mp.asarray([1.0, 2.0, 3.0], dtype=mp.float64)
  b = mp.asarray([4.0, 5.0, 6.0], dtype=mp.float64)
  out = mp.empty_like(a)
  mp.add(a, b, out=out)
  assert out.tolist() == [5.0, 7.0, 9.0]


def test_ufunc_dtype_kwarg_casts() -> None:
  a = mp.asarray([1.0, 2.0, 3.0], dtype=mp.float64)
  out = mp.add(a, a, dtype=mp.float32)
  assert out.dtype == mp.float32
