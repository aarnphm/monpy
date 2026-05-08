from __future__ import annotations

import array
import struct

import monpy as mp
import pytest


def test_asarray_from_writable_buffer_shares_storage() -> None:
  source = array.array("i", [1, 2, 3])
  arr = mp.asarray(source, copy=False)

  source[1] = 99
  arr[2] = -7

  assert arr.dtype == mp.int32
  assert arr.tolist() == [1, 99, -7]
  assert source.tolist() == [1, 99, -7]


def test_asarray_from_readonly_buffer_copies_by_default_and_rejects_copy_false() -> None:
  source = bytes([1, 2, 3])

  with pytest.raises(ValueError, match="readonly"):
    mp.asarray(source, copy=False)

  arr = mp.asarray(source)
  arr[0] = 88

  assert arr.dtype == mp.uint8
  assert arr.tolist() == [88, 2, 3]
  assert source == bytes([1, 2, 3])


def test_asarray_buffer_dtype_mismatch_copy_policy() -> None:
  source = array.array("i", [1, 2, 3])

  with pytest.raises(ValueError, match="copy"):
    mp.asarray(source, dtype=mp.float32, copy=False)

  arr = mp.asarray(source, dtype=mp.float32, copy=True)

  source[0] = 99
  assert arr.dtype == mp.float32
  assert arr.tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
  ("typecode", "dtype", "values"),
  [
    ("b", mp.int8, [-1, 2]),
    ("B", mp.uint8, [1, 2]),
    ("h", mp.int16, [-1, 2]),
    ("H", mp.uint16, [1, 2]),
    ("i", mp.int32, [-1, 2]),
    ("I", mp.uint32, [1, 2]),
    ("q", mp.int64, [-1, 2]),
    ("Q", mp.uint64, [1, 2]),
    ("f", mp.float32, [1.5, 2.5]),
    ("d", mp.float64, [1.5, 2.5]),
  ],
)
def test_pep3118_array_typecodes_resolve_to_core_dtypes(typecode: str, dtype: mp.DType, values: list[int] | list[float]) -> None:
  arr = mp.asarray(array.array(typecode, values))

  assert arr.dtype == dtype
  if dtype.kind == "f":
    assert arr.tolist() == pytest.approx(values)
  else:
    assert arr.tolist() == values


def test_frombuffer_offset_count_and_bounds() -> None:
  raw = struct.pack("<iiii", 10, 20, 30, 40)
  arr = mp.frombuffer(raw, dtype=mp.int32, count=2, offset=4)

  assert arr.tolist() == [20, 30]

  with pytest.raises(ValueError, match="multiple"):
    mp.frombuffer(raw[1:], dtype=mp.int32)
  with pytest.raises(ValueError, match="smaller"):
    mp.frombuffer(raw, dtype=mp.int32, count=5)
  with pytest.raises(ValueError, match="offset"):
    mp.frombuffer(raw, dtype=mp.int32, offset=len(raw) + 1)
