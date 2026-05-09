from __future__ import annotations

import math

import monpy as mp
import numpy
import pytest
from monpy.runtime import ops_numpy


# ---------------------------------------------------------------------------
# Phase-5d complex dtype: registration, allocation, basic arithmetic.
# ---------------------------------------------------------------------------


def test_complex64_registered() -> None:
  assert mp.complex64.itemsize == 8
  assert mp.complex64.kind == "c"
  assert mp.dtype(numpy.complex64) == mp.complex64
  assert ops_numpy.resolve_dtype(numpy.complex64) == mp.complex64


def test_complex128_registered() -> None:
  assert mp.complex128.itemsize == 16
  assert mp.complex128.kind == "c"
  assert mp.dtype(numpy.complex128) == mp.complex128
  assert ops_numpy.resolve_dtype(numpy.complex128) == mp.complex128


def test_complex_inference_from_python_values() -> None:
  arr = mp.asarray([1 + 2j, 3 + 4j])
  assert arr.dtype == mp.complex128
  assert arr.tolist() == [1 + 2j, 3 + 4j]


def test_complex_arithmetic_add_sub_mul_div_match_numpy() -> None:
  a_np = numpy.asarray([1 + 2j, 3 + 4j, 5 + 6j])
  b_np = numpy.asarray([2 + 1j, 1 + 1j, 1 + 0j])
  a_mp = mp.asarray(a_np)
  b_mp = mp.asarray(b_np)
  numpy.testing.assert_allclose(numpy.asarray(a_mp + b_mp), a_np + b_np, rtol=1e-12)
  numpy.testing.assert_allclose(numpy.asarray(a_mp - b_mp), a_np - b_np, rtol=1e-12)
  numpy.testing.assert_allclose(numpy.asarray(a_mp * b_mp), a_np * b_np, rtol=1e-12)
  numpy.testing.assert_allclose(numpy.asarray(a_mp / b_mp), a_np / b_np, rtol=1e-12)


def test_complex_strided_arithmetic_preserves_imaginary_part() -> None:
  a_np = numpy.asarray([1 + 2j, -3 + 5j, 7 - 11j, 13 + 17j], dtype=numpy.complex64)
  b_np = numpy.asarray([2 - 1j, 4 + 3j, -5 + 9j, 6 - 7j], dtype=numpy.complex64)
  a_mp = mp.asarray(a_np, dtype=mp.complex64)
  b_mp = mp.asarray(b_np, dtype=mp.complex64)
  add = a_mp[::-1] + b_mp[::-1]
  assert add._native.used_fused()
  numpy.testing.assert_allclose(numpy.asarray(add), a_np[::-1] + b_np[::-1], rtol=1e-6)
  numpy.testing.assert_allclose(numpy.asarray(a_mp[::-1] - b_mp[::-1]), a_np[::-1] - b_np[::-1], rtol=1e-6)
  numpy.testing.assert_allclose(numpy.asarray(a_mp[::-1] * b_mp[::-1]), a_np[::-1] * b_np[::-1], rtol=1e-6)
  numpy.testing.assert_allclose(numpy.asarray(a_mp[::-1] / b_mp[::-1]), a_np[::-1] / b_np[::-1], rtol=1e-6)


def test_complex_negate_preserves_imaginary_part() -> None:
  a = mp.asarray([1 + 2j, 3 + 4j], dtype=mp.complex128)
  numpy.testing.assert_array_equal(numpy.asarray(-a), [-1 - 2j, -3 - 4j])
  numpy.testing.assert_array_equal(numpy.asarray(mp.multiply(a, -1)), [-1 - 2j, -3 - 4j])
  numpy.testing.assert_array_equal(numpy.asarray(mp.multiply(-1, a)), [-1 - 2j, -3 - 4j])


def test_complex_scalar_mul_with_real_int() -> None:
  a = mp.asarray([1 + 2j, 3 + 4j], dtype=mp.complex128)
  numpy.testing.assert_array_equal(numpy.asarray(a * 2), [2 + 4j, 6 + 8j])


def test_complex_scalar_mul_with_complex_constant() -> None:
  a = mp.asarray([1 + 2j, 3 + 4j], dtype=mp.complex128)
  out = numpy.asarray(a * (2 + 1j))
  numpy.testing.assert_allclose(out, numpy.asarray([1 + 2j, 3 + 4j]) * (2 + 1j), rtol=1e-12)


def test_complex64_contiguous_multiply_uses_fused_kernel() -> None:
  a_np = numpy.asarray([1 + 2j, -3 + 5j, 7 - 11j, 13 + 17j], dtype=numpy.complex64)
  b_np = numpy.asarray([2 - 1j, 4 + 3j, -5 + 9j, 6 - 7j], dtype=numpy.complex64)
  a_mp = mp.asarray(a_np, dtype=mp.complex64, copy=True)
  b_mp = mp.asarray(b_np, dtype=mp.complex64, copy=True)

  out = a_mp * b_mp

  assert out._native.used_fused()
  numpy.testing.assert_allclose(numpy.asarray(out), a_np * b_np, rtol=1e-6)


def test_complex_division_uses_smith_for_stability() -> None:
  # Smith's algorithm avoids overflow when |c|, |d| differ greatly.
  a_np = numpy.asarray([1.0 + 1.0j])
  b_np = numpy.asarray([1e154 + 1e-154j])
  a_mp = mp.asarray(a_np)
  b_mp = mp.asarray(b_np)
  numpy.testing.assert_allclose(numpy.asarray(a_mp / b_mp), a_np / b_np, rtol=1e-10)


def test_conjugate_negates_imag_only() -> None:
  a = mp.asarray([1 + 2j, 3 + 4j, 5 + 6j], dtype=mp.complex128)
  numpy.testing.assert_array_equal(
    numpy.asarray(mp.conjugate(a)), [1 - 2j, 3 - 4j, 5 - 6j]
  )
  # `conj` is an alias.
  numpy.testing.assert_array_equal(
    numpy.asarray(mp.conj(a)), [1 - 2j, 3 - 4j, 5 - 6j]
  )


def test_real_imag_split() -> None:
  a = mp.asarray([1 + 2j, 3 + 4j, 5 + 6j], dtype=mp.complex128)
  numpy.testing.assert_array_equal(numpy.asarray(mp.real(a)), [1.0, 3.0, 5.0])
  numpy.testing.assert_array_equal(numpy.asarray(mp.imag(a)), [2.0, 4.0, 6.0])


def test_angle_returns_atan2_imag_real() -> None:
  a = mp.asarray([1 + 1j, 1 - 1j, -1 + 0j], dtype=mp.complex128)
  out = numpy.asarray(mp.angle(a))
  expected = [math.pi / 4, -math.pi / 4, math.pi]
  numpy.testing.assert_allclose(out, expected, rtol=1e-12)


def test_promotion_complex_absorbs_real_dtypes() -> None:
  assert mp.promote_types(mp.complex64, mp.float32) == mp.complex64
  assert mp.promote_types(mp.complex64, mp.float64) == mp.complex128
  assert mp.promote_types(mp.complex64, mp.int64) == mp.complex128
  assert mp.promote_types(mp.complex64, mp.complex128) == mp.complex128
  assert mp.promote_types(mp.complex128, mp.float64) == mp.complex128


def test_complex_dtype_metadata_matches_numpy() -> None:
  for monpy_dt, numpy_dt in [(mp.complex64, numpy.complex64), (mp.complex128, numpy.complex128)]:
    oracle = numpy.dtype(numpy_dt)
    assert monpy_dt.itemsize == oracle.itemsize
    assert monpy_dt.kind == oracle.kind
    assert monpy_dt.alignment == oracle.alignment


def test_can_cast_real_to_complex_safe() -> None:
  assert mp.can_cast(mp.float32, mp.complex64, casting="safe")
  assert mp.can_cast(mp.float32, mp.complex128, casting="safe")
  assert mp.can_cast(mp.float64, mp.complex128, casting="safe")
  # f64 → c64 not safe (precision loss).
  assert not mp.can_cast(mp.float64, mp.complex64, casting="safe")
  # complex → real never safe.
  assert not mp.can_cast(mp.complex64, mp.float32, casting="safe")
  assert not mp.can_cast(mp.complex128, mp.float64, casting="safe")


def test_complex_array_from_numpy_round_trip() -> None:
  source = numpy.asarray([1 + 2j, 3 + 4j], dtype=numpy.complex128)
  arr = mp.asarray(source)
  assert arr.dtype == mp.complex128
  numpy.testing.assert_array_equal(numpy.asarray(arr), source)


def test_complex_array_from_numpy_copy_false_shares_storage() -> None:
  source = numpy.asarray([1 + 2j, 3 + 4j], dtype=numpy.complex64)
  arr = mp.asarray(source, dtype=mp.complex64, copy=False)

  source[0] = 9 + 10j
  arr[1] = 11 + 12j

  assert arr.dtype == mp.complex64
  numpy.testing.assert_array_equal(numpy.asarray(arr), source)


def test_complex_array_from_numpy_copy_true_detaches_storage() -> None:
  source = numpy.asarray([1 + 2j, 3 + 4j], dtype=numpy.complex128)
  arr = mp.asarray(source, copy=True)

  source[0] = 9 + 10j
  arr[1] = 11 + 12j

  assert arr.dtype == mp.complex128
  numpy.testing.assert_array_equal(numpy.asarray(arr), [1 + 2j, 11 + 12j])
  numpy.testing.assert_array_equal(source, [9 + 10j, 3 + 4j])


def test_complex_array_from_strided_numpy_copy_true_detaches_storage() -> None:
  base = numpy.asarray([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=numpy.complex128)
  source = base[::2]
  arr = mp.asarray(source, dtype=mp.complex128, copy=True)

  base[0] = 9 + 10j
  arr[1] = 11 + 12j

  assert arr.dtype == mp.complex128
  numpy.testing.assert_array_equal(numpy.asarray(arr), [1 + 2j, 11 + 12j])
  numpy.testing.assert_array_equal(source, [9 + 10j, 5 + 6j])


def test_complex_astype_drops_imag_to_real_target() -> None:
  a = mp.asarray([1 + 2j, 3 + 4j], dtype=mp.complex128)
  real = a.astype(mp.float64)
  assert real.dtype == mp.float64
  numpy.testing.assert_array_equal(numpy.asarray(real), [1.0, 3.0])


def test_complex_astype_between_widths_matches_numpy() -> None:
  source = numpy.asarray([1 + 2j, -3 + 5j, 7 - 11j], dtype=numpy.complex64)
  arr = mp.asarray(source, dtype=mp.complex64)
  wide = arr.astype(mp.complex128)
  assert wide.dtype == mp.complex128
  numpy.testing.assert_array_equal(numpy.asarray(wide), source.astype(numpy.complex128))
  narrow = wide.astype(mp.complex64)
  assert narrow.dtype == mp.complex64
  numpy.testing.assert_array_equal(numpy.asarray(narrow), source)


def test_real_input_to_complex_array() -> None:
  a = mp.asarray([1.0, 2.0, 3.0], dtype=mp.complex128)
  numpy.testing.assert_array_equal(numpy.asarray(a), [1 + 0j, 2 + 0j, 3 + 0j])
