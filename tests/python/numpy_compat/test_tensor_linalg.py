from __future__ import annotations

import monpy as mp
import numpy
import pytest


# ---------------------------------------------------------------------------
# dot / vdot / inner / outer
# ---------------------------------------------------------------------------


def test_dot_1d_1d() -> None:
  a = mp.asarray([1.0, 2.0, 3.0], dtype=mp.float64)
  b = mp.asarray([4.0, 5.0, 6.0], dtype=mp.float64)
  assert float(mp.dot(a, b)) == 32.0


def test_dot_2d_2d_routes_through_matmul() -> None:
  a = mp.asarray([[1, 2], [3, 4]], dtype=mp.float64)
  b = mp.asarray([[5, 6], [7, 8]], dtype=mp.float64)
  out = mp.dot(a, b)
  numpy.testing.assert_array_equal(numpy.asarray(out), [[19, 22], [43, 50]])


def test_vdot_flattens_first() -> None:
  a = mp.asarray([[1, 2], [3, 4]], dtype=mp.float64)
  b = mp.asarray([[5, 6], [7, 8]], dtype=mp.float64)
  assert float(mp.vdot(a, b)) == 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8


def test_outer_basic() -> None:
  a = mp.asarray([1, 2, 3], dtype=mp.float64)
  b = mp.asarray([4, 5], dtype=mp.float64)
  out = mp.outer(a, b)
  numpy.testing.assert_array_equal(numpy.asarray(out), [[4, 5], [8, 10], [12, 15]])


def test_outer_strided_inputs_match_numpy() -> None:
  a_np = numpy.asarray([1.0, 2.0, 3.0, 4.0])[::-1]
  b_np = numpy.asarray([5.0, 6.0, 7.0, 8.0])[::2]
  out = mp.outer(mp.asarray(a_np, dtype=mp.float64), mp.asarray(b_np, dtype=mp.float64))
  numpy.testing.assert_array_equal(numpy.asarray(out), numpy.outer(a_np, b_np))


def test_outer_complex_inputs_match_numpy() -> None:
  a_np = numpy.asarray([1.0 + 2.0j, 3.0 - 1.0j])
  b_np = numpy.asarray([2.0 - 1.0j, -4.0 + 0.5j])
  out = mp.outer(mp.asarray(a_np, dtype=mp.complex128), mp.asarray(b_np, dtype=mp.complex128))
  numpy.testing.assert_allclose(numpy.asarray(out), numpy.outer(a_np, b_np), rtol=1e-12)


def test_inner_1d() -> None:
  a = mp.asarray([1.0, 2.0, 3.0], dtype=mp.float64)
  b = mp.asarray([4.0, 5.0, 6.0], dtype=mp.float64)
  assert float(mp.inner(a, b)) == 32.0


def test_vecdot_axis1_matches_numpy() -> None:
  a_np = numpy.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float64)
  b_np = numpy.asarray([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=numpy.float64)
  out = mp.linalg.vecdot(mp.asarray(a_np, dtype=mp.float64), mp.asarray(b_np, dtype=mp.float64), axis=1)
  numpy.testing.assert_allclose(numpy.asarray(out), numpy.linalg.vecdot(a_np, b_np, axis=1), rtol=1e-12)


# ---------------------------------------------------------------------------
# tensordot
# ---------------------------------------------------------------------------


def test_tensordot_axes_int() -> None:
  a = mp.asarray([[1, 2], [3, 4]], dtype=mp.float64)
  b = mp.asarray([[5, 6], [7, 8]], dtype=mp.float64)
  out = mp.tensordot(a, b, axes=1)
  numpy.testing.assert_array_equal(numpy.asarray(out), [[19, 22], [43, 50]])


def test_tensordot_axes_full() -> None:
  a = mp.asarray([[1, 2], [3, 4]], dtype=mp.float64)
  b = mp.asarray([[5, 6], [7, 8]], dtype=mp.float64)
  out = mp.tensordot(a, b, axes=2)
  # Full contraction: 1*5 + 2*6 + 3*7 + 4*8 = 70.
  assert float(out) == 70.0


# ---------------------------------------------------------------------------
# kron / cross
# ---------------------------------------------------------------------------


def test_kron_2x2_with_2x2() -> None:
  a = mp.asarray([[1, 0], [0, 1]], dtype=mp.float64)
  b = mp.asarray([[1, 2], [3, 4]], dtype=mp.float64)
  out = mp.kron(a, b)
  oracle = numpy.kron(numpy.asarray([[1, 0], [0, 1]]), numpy.asarray([[1, 2], [3, 4]]))
  numpy.testing.assert_array_equal(numpy.asarray(out), oracle)


def test_cross_3vec() -> None:
  a = mp.asarray([1.0, 0.0, 0.0], dtype=mp.float64)
  b = mp.asarray([0.0, 1.0, 0.0], dtype=mp.float64)
  out = mp.cross(a, b)
  numpy.testing.assert_array_equal(numpy.asarray(out), [0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# norm
# ---------------------------------------------------------------------------


def test_norm_default_l2() -> None:
  a = mp.asarray([3.0, 4.0], dtype=mp.float64)
  assert float(mp.linalg.norm(a)) == 5.0


def test_norm_last_axis_matches_numpy() -> None:
  a_np = numpy.asarray([[3.0, 4.0], [5.0, 12.0]], dtype=numpy.float64)
  out = mp.linalg.vector_norm(mp.asarray(a_np, dtype=mp.float64), axis=1)
  numpy.testing.assert_allclose(numpy.asarray(out), numpy.linalg.vector_norm(a_np, axis=1), rtol=1e-12)


def test_matrix_norm_fro_matches_numpy() -> None:
  a_np = numpy.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float64)
  out = mp.linalg.matrix_norm(mp.asarray(a_np, dtype=mp.float64))
  numpy.testing.assert_allclose(float(out), float(numpy.linalg.matrix_norm(a_np)), rtol=1e-12)


def test_norm_l1_and_inf() -> None:
  import math
  a = mp.asarray([1.0, -2.0, 3.0], dtype=mp.float64)
  assert float(mp.linalg.norm(a, ord=1)) == 6.0
  assert float(mp.linalg.norm(a, ord=math.inf)) == 3.0


# ---------------------------------------------------------------------------
# matrix_power / multi_dot
# ---------------------------------------------------------------------------


def test_matrix_power_zero_returns_identity() -> None:
  a = mp.asarray([[2, 0], [0, 2]], dtype=mp.float64)
  out = mp.linalg.matrix_power(a, 0)
  numpy.testing.assert_array_equal(numpy.asarray(out), [[1, 0], [0, 1]])


def test_matrix_power_pos() -> None:
  a = mp.asarray([[1, 1], [0, 1]], dtype=mp.float64)
  out = mp.linalg.matrix_power(a, 3)
  numpy.testing.assert_array_equal(numpy.asarray(out), [[1, 3], [0, 1]])


def test_multi_dot_chained_matmul() -> None:
  a = mp.asarray([[1, 2]], dtype=mp.float64)
  b = mp.asarray([[3], [4]], dtype=mp.float64)
  c = mp.asarray([[5]], dtype=mp.float64)
  out = mp.linalg.multi_dot([a, b, c])
  numpy.testing.assert_array_equal(numpy.asarray(out), [[55]])


# ---------------------------------------------------------------------------
# LAPACK-backed decompositions: qr / cholesky / eigh / eig / svd / lstsq / pinv
# ---------------------------------------------------------------------------


def test_cholesky_2x2() -> None:
  a = mp.asarray([[4.0, 2.0], [2.0, 3.0]], dtype=mp.float64)
  l = mp.linalg.cholesky(a)
  numpy.testing.assert_allclose(numpy.asarray(l), numpy.linalg.cholesky(numpy.asarray(a)), rtol=1e-12)


def test_cholesky_5x5_psd() -> None:
  rng = numpy.random.default_rng(0)
  m = rng.standard_normal((5, 5))
  a = m @ m.T + 5 * numpy.eye(5)
  l_mp = mp.linalg.cholesky(mp.asarray(a, dtype=mp.float64))
  l_np = numpy.linalg.cholesky(a)
  numpy.testing.assert_allclose(numpy.asarray(l_mp), l_np, rtol=1e-12)


def test_cholesky_rejects_non_positive_definite() -> None:
  a = mp.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=mp.float64)
  with pytest.raises(mp.linalg.LinAlgError):
    mp.linalg.cholesky(a)


def test_qr_reduced_orthogonal_q() -> None:
  rng = numpy.random.default_rng(0)
  a = rng.standard_normal((4, 3))
  q, r = mp.linalg.qr(mp.asarray(a, dtype=mp.float64))
  qa = numpy.asarray(q)
  ra = numpy.asarray(r)
  numpy.testing.assert_allclose(qa @ ra, a, rtol=1e-10)
  numpy.testing.assert_allclose(qa.T @ qa, numpy.eye(3), atol=1e-12)


def test_qr_r_only() -> None:
  a = numpy.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  r = mp.linalg.qr(mp.asarray(a, dtype=mp.float64), mode="r")
  q_np, r_np = numpy.linalg.qr(a, mode="reduced")
  numpy.testing.assert_allclose(numpy.abs(numpy.asarray(r)), numpy.abs(r_np), rtol=1e-10)


def test_eigh_symmetric_2x2() -> None:
  a = mp.asarray([[2.0, 1.0], [1.0, 3.0]], dtype=mp.float64)
  w, v = mp.linalg.eigh(a)
  numpy.testing.assert_allclose(
    numpy.sort(numpy.asarray(w)), numpy.linalg.eigvalsh(numpy.asarray(a)), rtol=1e-12
  )
  # A v_i = lambda_i v_i (within sign).
  va = numpy.asarray(v)
  for i in range(2):
    lhs = numpy.asarray(a) @ va[:, i]
    rhs = float(numpy.asarray(w)[i]) * va[:, i]
    numpy.testing.assert_allclose(lhs, rhs, atol=1e-10)


def test_eigvalsh_returns_eigenvalues_only() -> None:
  rng = numpy.random.default_rng(7)
  m = rng.standard_normal((4, 4))
  sym = m + m.T
  w_mp = mp.linalg.eigvalsh(mp.asarray(sym, dtype=mp.float64))
  w_np = numpy.linalg.eigvalsh(sym)
  numpy.testing.assert_allclose(numpy.sort(numpy.asarray(w_mp)), w_np, rtol=1e-10)


def test_eig_real_eigenvalues() -> None:
  a = mp.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=mp.float64)
  w, _ = mp.linalg.eig(a)
  out = numpy.sort(numpy.asarray(w))
  numpy.testing.assert_allclose(out, [2.0, 3.0], rtol=1e-12)


def test_eig_returns_complex_eigenvalues_when_present() -> None:
  # 2D rotation matrix has eigenvalues ±i (imaginary).
  a = mp.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=mp.float64)
  w, _ = mp.linalg.eig(a)
  assert w.dtype == mp.complex128
  numpy.testing.assert_allclose(
    numpy.sort_complex(numpy.asarray(w)),
    numpy.sort_complex(numpy.linalg.eigvals(numpy.asarray(a))),
    rtol=1e-12,
  )


def test_svd_thin_reconstruction() -> None:
  rng = numpy.random.default_rng(11)
  a = rng.standard_normal((5, 3))
  u, s, vt = mp.linalg.svd(mp.asarray(a, dtype=mp.float64), full_matrices=False)
  ua = numpy.asarray(u)
  sa = numpy.asarray(s)
  vta = numpy.asarray(vt)
  recon = ua @ numpy.diag(sa) @ vta
  numpy.testing.assert_allclose(recon, a, rtol=1e-10)


def test_svdvals_only_singular_values() -> None:
  a = numpy.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  s_mp = mp.linalg.svdvals(mp.asarray(a, dtype=mp.float64))
  s_np = numpy.linalg.svd(a, compute_uv=False)
  numpy.testing.assert_allclose(numpy.asarray(s_mp), s_np, rtol=1e-12)


def test_lstsq_overdetermined_matches_numpy() -> None:
  A = numpy.asarray([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
  B = numpy.asarray([6.0, 5.0, 7.0, 10.0])
  x_mp, res_mp, rank_mp, _ = mp.linalg.lstsq(
    mp.asarray(A, dtype=mp.float64), mp.asarray(B, dtype=mp.float64)
  )
  x_np, res_np, rank_np, _ = numpy.linalg.lstsq(A, B, rcond=None)
  numpy.testing.assert_allclose(numpy.asarray(x_mp), x_np, rtol=1e-12)
  assert rank_mp == rank_np
  numpy.testing.assert_allclose(numpy.asarray(res_mp), res_np, rtol=1e-12)


def test_pinv_round_trips_within_pseudoinverse_axiom() -> None:
  rng = numpy.random.default_rng(13)
  a = rng.standard_normal((4, 3))
  p = mp.linalg.pinv(mp.asarray(a, dtype=mp.float64))
  # A · pinv(A) · A == A (Moore-Penrose first axiom).
  out = numpy.asarray(mp.matmul(mp.matmul(mp.asarray(a, dtype=mp.float64), p), mp.asarray(a, dtype=mp.float64)))
  numpy.testing.assert_allclose(out, a, atol=1e-10)


def test_pinv_matches_numpy_with_monpy_default_cutoff() -> None:
  rng = numpy.random.default_rng(14)
  a = rng.standard_normal((5, 3))
  rcond = numpy.finfo(numpy.float64).eps * max(a.shape)

  p = mp.linalg.pinv(mp.asarray(a, dtype=mp.float64))

  numpy.testing.assert_allclose(numpy.asarray(p), numpy.linalg.pinv(a, rcond=rcond), rtol=1e-12, atol=1e-12)


def test_matrix_rank_matches_numpy() -> None:
  a = numpy.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]])
  rk_mp = mp.linalg.matrix_rank(mp.asarray(a, dtype=mp.float64))
  rk_np = numpy.linalg.matrix_rank(a)
  assert rk_mp == rk_np


def test_slogdet_matches_numpy_for_positive_negative_and_singular() -> None:
  cases = [
    numpy.asarray([[2.0, 0.0], [0.0, 3.0]]),
    numpy.asarray([[-2.0, 0.0], [0.0, 3.0]]),
    numpy.asarray([[1.0, 2.0], [2.0, 4.0]]),
  ]
  for a in cases:
    sign_mp, logdet_mp = mp.linalg.slogdet(mp.asarray(a, dtype=mp.float64))
    sign_np, logdet_np = numpy.linalg.slogdet(a)
    assert sign_mp == sign_np
    numpy.testing.assert_allclose(logdet_mp, logdet_np, rtol=1e-10, atol=1e-10)
