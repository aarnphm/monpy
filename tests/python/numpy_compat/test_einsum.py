from __future__ import annotations

import monpy as mp
import numpy
import pytest

from _helpers import assert_same_result_kind


# ---------------------------------------------------------------------------
# einsum: parsing + correctness on common patterns.
# ---------------------------------------------------------------------------


def test_einsum_matmul_2d() -> None:
  a = numpy.asarray([[1.0, 2.0], [3.0, 4.0]])
  b = numpy.asarray([[5.0, 6.0], [7.0, 8.0]])
  out = mp.einsum("ij,jk->ik", mp.asarray(a), mp.asarray(b))
  numpy.testing.assert_array_equal(numpy.asarray(out), numpy.einsum("ij,jk->ik", a, b))


def test_einsum_dot_product() -> None:
  v1 = numpy.asarray([1.0, 2.0, 3.0])
  v2 = numpy.asarray([4.0, 5.0, 6.0])
  out = mp.einsum("i,i", mp.asarray(v1), mp.asarray(v2))
  oracle = numpy.einsum("i,i", v1, v2)
  assert_same_result_kind(out, oracle)
  assert float(out) == float(oracle)


def test_einsum_outer_product() -> None:
  v1 = numpy.asarray([1.0, 2.0, 3.0])
  v2 = numpy.asarray([4.0, 5.0])
  out = mp.einsum("i,j->ij", mp.asarray(v1), mp.asarray(v2))
  numpy.testing.assert_array_equal(numpy.asarray(out), numpy.einsum("i,j->ij", v1, v2))


def test_einsum_trace_via_diagonal() -> None:
  a = numpy.asarray([[1.0, 2.0], [3.0, 4.0]])
  out = mp.einsum("ii", mp.asarray(a))
  oracle = numpy.einsum("ii", a)
  assert_same_result_kind(out, oracle)
  assert float(out) == float(oracle)


def test_einsum_transpose() -> None:
  a = numpy.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  out = mp.einsum("ij->ji", mp.asarray(a))
  numpy.testing.assert_array_equal(numpy.asarray(out), numpy.einsum("ij->ji", a))


def test_einsum_sum_along_axis() -> None:
  a = numpy.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  out = mp.einsum("ij->i", mp.asarray(a))
  numpy.testing.assert_array_equal(numpy.asarray(out), numpy.einsum("ij->i", a))


def test_einsum_inner_product_3d() -> None:
  rng = numpy.random.default_rng(0)
  a = rng.standard_normal((2, 3, 4))
  b = rng.standard_normal((4, 5))
  mp_out = mp.einsum("ijk,kl->ijl", mp.asarray(a), mp.asarray(b))
  np_out = numpy.einsum("ijk,kl->ijl", a, b)
  numpy.testing.assert_allclose(numpy.asarray(mp_out), np_out, rtol=1e-10)


def test_einsum_implicit_output() -> None:
  # numpy auto-derives the output (alphabetical order of free labels) when
  # the '->' is absent.
  v1 = numpy.asarray([1.0, 2.0, 3.0])
  v2 = numpy.asarray([4.0, 5.0])
  mp_out = mp.einsum("i,j", mp.asarray(v1), mp.asarray(v2))
  np_out = numpy.einsum("i,j", v1, v2)
  numpy.testing.assert_array_equal(numpy.asarray(mp_out), np_out)


# ---------------------------------------------------------------------------
# tensorinv / tensorsolve.
# ---------------------------------------------------------------------------


def test_tensorinv_4x4_identity_block() -> None:
  a = numpy.eye(4).reshape(2, 2, 2, 2).astype(numpy.float64)
  inv_mp = mp.linalg.tensorinv(mp.asarray(a.tolist(), dtype=mp.float64))
  inv_np = numpy.linalg.tensorinv(a)
  numpy.testing.assert_allclose(numpy.asarray(inv_mp), inv_np, atol=1e-10)


def test_tensorsolve_basic() -> None:
  # Construct a system where a has shape b.shape + x.shape and prod matches.
  # Pick b.shape = (2, 3) so prod(b)=6, x.shape = (2, 3) so prod(x)=6.
  rng = numpy.random.default_rng(1)
  a = rng.standard_normal((2, 3, 2, 3))
  b = rng.standard_normal((2, 3))
  x_mp = mp.linalg.tensorsolve(
    mp.asarray(a.tolist(), dtype=mp.float64), mp.asarray(b.tolist(), dtype=mp.float64)
  )
  x_np = numpy.linalg.tensorsolve(a, b)
  numpy.testing.assert_allclose(numpy.asarray(x_mp), x_np, rtol=1e-8)


# ---------------------------------------------------------------------------
# Complex matmul (cgemm/zgemm via Accelerate).
# ---------------------------------------------------------------------------


def test_complex128_matmul_via_zgemm() -> None:
  a = numpy.asarray([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
  b = numpy.asarray([[1 + 0j, 2 + 1j], [3 + 0j, 4 + 1j]])
  out = mp.matmul(mp.asarray(a, dtype=mp.complex128), mp.asarray(b, dtype=mp.complex128))
  numpy.testing.assert_allclose(numpy.asarray(out), a @ b, rtol=1e-12)


def test_complex64_matmul_via_cgemm() -> None:
  a = numpy.asarray([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=numpy.complex64)
  b = numpy.asarray([[1 + 0j, 2 + 1j], [3 + 0j, 4 + 1j]], dtype=numpy.complex64)
  out = mp.matmul(mp.asarray(a, dtype=mp.complex64), mp.asarray(b, dtype=mp.complex64))
  numpy.testing.assert_allclose(numpy.asarray(out), a @ b, rtol=1e-5)


def test_complex_dot_product() -> None:
  v1 = mp.asarray([1 + 1j, 2 + 2j], dtype=mp.complex128)
  v2 = mp.asarray([3 - 1j, 4 + 0j], dtype=mp.complex128)
  result = mp.matmul(v1, v2)
  expected = numpy.asarray(v1) @ numpy.asarray(v2)
  assert complex(result) == expected


# ---------------------------------------------------------------------------
# Complex transcendentals.
# ---------------------------------------------------------------------------


def test_complex_exp_matches_numpy() -> None:
  a = mp.asarray([1 + 2j, 0 + 1j, 3 + 0j], dtype=mp.complex128)
  numpy.testing.assert_allclose(
    numpy.asarray(mp.exp(a)), numpy.exp(numpy.asarray(a)), rtol=1e-10
  )


def test_complex_log_matches_numpy() -> None:
  a = mp.asarray([1 + 2j, 0 + 1j, 3 + 0j], dtype=mp.complex128)
  # Log uses sqrt(re²+im²) which is slightly less accurate than numpy's
  # specialized clog implementation — relax to 1e-10.
  numpy.testing.assert_allclose(
    numpy.asarray(mp.log(a)), numpy.log(numpy.asarray(a)), rtol=1e-10
  )


def test_complex_sin_cos_matches_numpy() -> None:
  a = mp.asarray([1 + 2j, 0.5 + 1j], dtype=mp.complex128)
  numpy.testing.assert_allclose(
    numpy.asarray(mp.sin(a)), numpy.sin(numpy.asarray(a)), rtol=1e-10
  )
  numpy.testing.assert_allclose(
    numpy.asarray(mp.cos(a)), numpy.cos(numpy.asarray(a)), rtol=1e-10
  )


def test_complex_sqrt_principal_branch() -> None:
  a = mp.asarray([1 + 2j, 0 + 1j, 4 + 0j], dtype=mp.complex128)
  numpy.testing.assert_allclose(
    numpy.asarray(mp.sqrt(a)), numpy.sqrt(numpy.asarray(a)), rtol=1e-10
  )


def test_complex_reciprocal() -> None:
  a = mp.asarray([1 + 2j, 3 + 4j], dtype=mp.complex128)
  numpy.testing.assert_allclose(
    numpy.asarray(mp.reciprocal(a)), 1 / numpy.asarray(a), rtol=1e-12
  )
