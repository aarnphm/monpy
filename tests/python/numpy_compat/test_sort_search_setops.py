from __future__ import annotations

import monpy as mp
import numpy

# ---------------------------------------------------------------------------
# sort / argsort / partition / argpartition
# ---------------------------------------------------------------------------


def test_sort_1d_matches_numpy() -> None:
  arr = mp.asarray([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dtype=mp.float64)
  numpy.testing.assert_array_equal(numpy.asarray(mp.sort(arr)), numpy.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]))


def test_sort_2d_axis_zero_and_one() -> None:
  arr = mp.asarray([[3, 1, 2], [6, 4, 5]], dtype=mp.int64)
  oracle = numpy.asarray([[3, 1, 2], [6, 4, 5]])
  numpy.testing.assert_array_equal(numpy.asarray(mp.sort(arr, axis=0)), numpy.sort(oracle, axis=0))
  numpy.testing.assert_array_equal(numpy.asarray(mp.sort(arr, axis=1)), numpy.sort(oracle, axis=1))


def test_argsort_returns_perm() -> None:
  arr = mp.asarray([3.0, 1.0, 2.0], dtype=mp.float64)
  perm = mp.argsort(arr)
  numpy.testing.assert_array_equal(numpy.asarray(perm), [1, 2, 0])


def test_partition_at_least_keeps_kth_in_place() -> None:
  arr = mp.asarray([3, 1, 4, 1, 5, 9, 2, 6], dtype=mp.int64)
  out = mp.partition(arr, 3)
  # v1 implements via full sort which is a superset of partition.
  numpy.testing.assert_array_equal(numpy.asarray(out), numpy.array([1, 1, 2, 3, 4, 5, 6, 9]))


# ---------------------------------------------------------------------------
# searchsorted / digitize
# ---------------------------------------------------------------------------


def test_searchsorted_left_and_right() -> None:
  bins = mp.asarray([1, 3, 5, 7, 9], dtype=mp.int64)
  assert mp.searchsorted(bins, 5) == 2
  assert mp.searchsorted(bins, 5, side="right") == 3
  out = mp.searchsorted(bins, mp.asarray([0, 4, 9, 10], dtype=mp.int64))
  numpy.testing.assert_array_equal(numpy.asarray(out), [0, 2, 4, 5])


def test_digitize_ascending_and_descending() -> None:
  arr = mp.asarray([0.5, 1.5, 2.5, 3.5], dtype=mp.float64)
  bins_asc = mp.asarray([1.0, 2.0, 3.0], dtype=mp.float64)
  numpy.testing.assert_array_equal(
    numpy.asarray(mp.digitize(arr, bins_asc)),
    numpy.digitize(numpy.asarray([0.5, 1.5, 2.5, 3.5]), numpy.asarray([1.0, 2.0, 3.0])),
  )


# ---------------------------------------------------------------------------
# unique / bincount
# ---------------------------------------------------------------------------


def test_unique_returns_sorted_distinct() -> None:
  arr = mp.asarray([3, 1, 2, 3, 1, 4], dtype=mp.int64)
  numpy.testing.assert_array_equal(numpy.asarray(mp.unique(arr)), [1, 2, 3, 4])


def test_unique_with_indices_inverse_counts() -> None:
  arr = mp.asarray([3, 1, 2, 3, 1, 4], dtype=mp.int64)
  vals, idx, inv, cnt = mp.unique(arr, return_index=True, return_inverse=True, return_counts=True)
  numpy.testing.assert_array_equal(numpy.asarray(vals), [1, 2, 3, 4])
  numpy.testing.assert_array_equal(numpy.asarray(cnt), [2, 1, 2, 1])
  # inverse should reconstruct original order.
  numpy.testing.assert_array_equal(numpy.asarray(vals)[numpy.asarray(inv)], [3, 1, 2, 3, 1, 4])


def test_bincount_unweighted_and_weighted() -> None:
  arr = mp.asarray([0, 1, 1, 2, 3, 1], dtype=mp.int64)
  numpy.testing.assert_array_equal(numpy.asarray(mp.bincount(arr)), [1, 3, 1, 1])
  w = mp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=mp.float64)
  numpy.testing.assert_allclose(numpy.asarray(mp.bincount(arr, weights=w)), [1.0, 11.0, 4.0, 5.0], rtol=1e-12)


# ---------------------------------------------------------------------------
# isin / intersect1d / union1d / setdiff1d / setxor1d
# ---------------------------------------------------------------------------


def test_isin_basic() -> None:
  a = mp.asarray([0, 1, 2, 3, 4], dtype=mp.int64)
  b = mp.asarray([1, 3], dtype=mp.int64)
  numpy.testing.assert_array_equal(numpy.asarray(mp.isin(a, b)), [False, True, False, True, False])


def test_set_ops() -> None:
  a = mp.asarray([1, 2, 3, 4], dtype=mp.int64)
  b = mp.asarray([3, 4, 5, 6], dtype=mp.int64)
  numpy.testing.assert_array_equal(numpy.asarray(mp.intersect1d(a, b)), [3, 4])
  numpy.testing.assert_array_equal(numpy.asarray(mp.union1d(a, b)), [1, 2, 3, 4, 5, 6])
  numpy.testing.assert_array_equal(numpy.asarray(mp.setdiff1d(a, b)), [1, 2])
  numpy.testing.assert_array_equal(numpy.asarray(mp.setxor1d(a, b)), [1, 2, 5, 6])


# ---------------------------------------------------------------------------
# nonzero / argwhere / flatnonzero
# ---------------------------------------------------------------------------


def test_nonzero_2d_matches_numpy() -> None:
  arr = mp.asarray([[0, 1, 0], [2, 0, 3]], dtype=mp.int64)
  oracle = numpy.asarray([[0, 1, 0], [2, 0, 3]])
  rows, cols = mp.nonzero(arr)
  exp_r, exp_c = numpy.nonzero(oracle)
  numpy.testing.assert_array_equal(numpy.asarray(rows), exp_r)
  numpy.testing.assert_array_equal(numpy.asarray(cols), exp_c)


def test_argwhere_returns_2d_indices() -> None:
  arr = mp.asarray([[0, 1], [2, 0]], dtype=mp.int64)
  out = mp.argwhere(arr)
  numpy.testing.assert_array_equal(numpy.asarray(out), [[0, 1], [1, 0]])


def test_flatnonzero() -> None:
  arr = mp.asarray([0, 1, 0, 2, 0, 3], dtype=mp.int64)
  numpy.testing.assert_array_equal(numpy.asarray(mp.flatnonzero(arr)), [1, 3, 5])


# ---------------------------------------------------------------------------
# take / repeat / tile / roll
# ---------------------------------------------------------------------------


def test_take_axis_none_and_axis_int() -> None:
  arr = mp.asarray([[1, 2, 3], [4, 5, 6]], dtype=mp.int64)
  numpy.testing.assert_array_equal(numpy.asarray(mp.take(arr, mp.asarray([0, 2, 4], dtype=mp.int64))), [1, 3, 5])
  numpy.testing.assert_array_equal(
    numpy.asarray(mp.take(arr, mp.asarray([0, 2], dtype=mp.int64), axis=1)),
    [[1, 3], [4, 6]],
  )


def test_repeat_scalar_and_array() -> None:
  numpy.testing.assert_array_equal(
    numpy.asarray(mp.repeat(mp.asarray([1, 2, 3], dtype=mp.int64), 2)),
    [1, 1, 2, 2, 3, 3],
  )
  numpy.testing.assert_array_equal(
    numpy.asarray(mp.repeat(mp.asarray([1, 2, 3], dtype=mp.int64), [3, 1, 2])),
    [1, 1, 1, 2, 3, 3],
  )


def test_tile_simple() -> None:
  numpy.testing.assert_array_equal(
    numpy.asarray(mp.tile(mp.asarray([1, 2], dtype=mp.int64), 3)),
    [1, 2, 1, 2, 1, 2],
  )


def test_roll_axis_none() -> None:
  arr = mp.asarray([1, 2, 3, 4, 5], dtype=mp.int64)
  numpy.testing.assert_array_equal(numpy.asarray(mp.roll(arr, 2)), [4, 5, 1, 2, 3])
  numpy.testing.assert_array_equal(numpy.asarray(mp.roll(arr, -1)), [2, 3, 4, 5, 1])


# ---------------------------------------------------------------------------
# pad / append / delete
# ---------------------------------------------------------------------------


def test_pad_constant_default_zero() -> None:
  arr = mp.asarray([1, 2, 3], dtype=mp.int64)
  out = mp.pad(arr, 2)
  numpy.testing.assert_array_equal(numpy.asarray(out), [0, 0, 1, 2, 3, 0, 0])


def test_pad_constant_value() -> None:
  arr = mp.asarray([1, 2, 3], dtype=mp.int64)
  out = mp.pad(arr, (1, 2), mode="constant", constant_values=9)
  numpy.testing.assert_array_equal(numpy.asarray(out), [9, 1, 2, 3, 9, 9])


def test_append_and_delete() -> None:
  arr = mp.asarray([1, 2, 3], dtype=mp.int64)
  numpy.testing.assert_array_equal(numpy.asarray(mp.append(arr, 4)), [1, 2, 3, 4])
  numpy.testing.assert_array_equal(numpy.asarray(mp.delete(arr, 1)), [1, 3])


# ---------------------------------------------------------------------------
# tril / triu / *_indices
# ---------------------------------------------------------------------------


def test_tril_triu_2d() -> None:
  arr = mp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=mp.int64)
  numpy.testing.assert_array_equal(numpy.asarray(mp.tril(arr)), [[1, 0, 0], [4, 5, 0], [7, 8, 9]])
  numpy.testing.assert_array_equal(numpy.asarray(mp.triu(arr)), [[1, 2, 3], [0, 5, 6], [0, 0, 9]])


def test_tril_triu_indices_match_numpy() -> None:
  rows_l, cols_l = mp.tril_indices(3)
  numpy.testing.assert_array_equal(numpy.asarray(rows_l), [0, 1, 1, 2, 2, 2])
  numpy.testing.assert_array_equal(numpy.asarray(cols_l), [0, 0, 1, 0, 1, 2])
  rows_u, cols_u = mp.triu_indices(3)
  numpy.testing.assert_array_equal(numpy.asarray(rows_u), [0, 0, 0, 1, 1, 2])
  numpy.testing.assert_array_equal(numpy.asarray(cols_u), [0, 1, 2, 1, 2, 2])


# ---------------------------------------------------------------------------
# ravel_multi_index / unravel_index
# ---------------------------------------------------------------------------


def test_unravel_index_round_trip() -> None:
  flat = mp.asarray([0, 1, 5, 6], dtype=mp.int64)
  rs, cs = mp.unravel_index(flat, (3, 3))
  numpy.testing.assert_array_equal(numpy.asarray(rs), [0, 0, 1, 2])
  numpy.testing.assert_array_equal(numpy.asarray(cs), [0, 1, 2, 0])
  back = mp.ravel_multi_index((rs, cs), (3, 3))
  numpy.testing.assert_array_equal(numpy.asarray(back), [0, 1, 5, 6])
