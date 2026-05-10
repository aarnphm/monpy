from __future__ import annotations

import monpy as mp
import pytest
from monpy.extend import StorageKind, dtype_spec
from monpy.numpy import dtype_info

LOW_PRECISION_DTYPES = (
  mp.bfloat16,
  mp.float8_e4m3fn,
  mp.float8_e4m3fnuz,
  mp.float8_e5m2,
  mp.float8_e5m2fnuz,
  mp.float8_e8m0fnu,
  mp.float4_e2m1fn,
)


def test_low_precision_dtype_codes_append_after_existing_dtypes() -> None:
  assert mp.bfloat16.code == 14
  assert mp.float8_e4m3fn.code == 15
  assert mp.float8_e4m3fnuz.code == 16
  assert mp.float8_e5m2.code == 17
  assert mp.float8_e5m2fnuz.code == 18
  assert mp.float8_e8m0fnu.code == 19
  assert mp.float4_e2m1fn.code == 20
  assert mp._native._dtype_storage_bits(mp.float4_e2m1fn.code) == 4
  assert mp._native._dtype_storage_nbytes(mp.float4_e2m1fn.code, 5) == 3


def test_low_precision_dtype_metadata_is_storage_aware() -> None:
  assert mp.bfloat16.bits == 16
  assert mp.float8_e4m3fn.bits == 8
  assert mp.float4_e2m1fn.bits == 4
  assert mp.float4_e2m1fn.storage_bits == 4
  assert mp.float4_e2m1fn.storage == "packed_subbyte"
  assert mp.float4_e2m1fn.is_packed
  assert not mp.float8_e4m3fn.is_packed
  assert mp.isdtype(mp.bfloat16, "real floating")
  assert mp.isdtype(mp.float8_e4m3fn, "quantized floating")


def test_low_precision_numpy_dtype_info_marks_unsupported_interchange_formats() -> None:
  info = dtype_info(mp.float4_e2m1fn)

  assert info.itemsize == 1
  assert info.typestr == ""
  assert info.format == ""
  assert not info.buffer_exportable


@pytest.mark.parametrize(
  ("dtype", "values"),
  (
    (mp.bfloat16, [0.0, 1.0, -2.0, 6.0]),
    (mp.float8_e4m3fn, [0.0, 1.0, -2.0, 6.0]),
    (mp.float8_e4m3fnuz, [0.0, 1.0, -2.0, 6.0]),
    (mp.float8_e5m2, [0.0, 1.0, -2.0, 6.0]),
    (mp.float8_e5m2fnuz, [0.0, 1.0, -2.0, 6.0]),
    (mp.float8_e8m0fnu, [1.0, 2.0, 8.0, 16.0]),
    (mp.float4_e2m1fn, [0.0, 1.0, -2.0, 6.0]),
  ),
)
def test_low_precision_scalar_storage_round_trips_exact_representable_values(
  dtype: mp.DType, values: list[float]
) -> None:
  arr = mp.asarray(values, dtype=dtype)
  assert arr.dtype == dtype
  assert arr.tolist() == values


def test_fp4_storage_is_packed_and_preserves_odd_offset_views() -> None:
  arr = mp.asarray([0.5, -0.5, 6.0], dtype=mp.float4_e2m1fn)
  assert arr.nbytes == 2

  view = mp.asarray(arr[1:])
  assert view.tolist() == [-0.5, 6.0]
  compact = mp.ascontiguousarray(view)
  assert compact.dtype == mp.float4_e2m1fn
  assert compact.nbytes == 1
  assert compact.tolist() == [-0.5, 6.0]
  assert compact.astype(mp.float32).tolist() == [-0.5, 6.0]


def test_fp4_frombuffer_uses_two_logical_values_per_byte() -> None:
  raw = bytearray([0x21, 0xF6])
  arr = mp.frombuffer(raw, dtype=mp.float4_e2m1fn)
  assert arr.shape == (4,)
  assert arr.nbytes == 2
  assert arr.tolist() == [0.5, 1.0, 4.0, -6.0]

  raw[0] = 0x43
  assert arr.tolist() == [1.5, 2.0, 4.0, -6.0]


def test_fp4_concatenate_repacks_instead_of_byte_memcpy() -> None:
  lhs = mp.asarray([0.5, 1.0, 1.5], dtype=mp.float4_e2m1fn)
  rhs = mp.asarray([-0.5, -1.0], dtype=mp.float4_e2m1fn)

  out = mp.concatenate([lhs, rhs])

  assert out.dtype == mp.float4_e2m1fn
  assert out.shape == (5,)
  assert out.nbytes == 3
  assert out.tolist() == [0.5, 1.0, 1.5, -0.5, -1.0]


def test_fp4_stack_repacks_instead_of_byte_memcpy() -> None:
  lhs = mp.asarray([0.5, 1.0, 1.5], dtype=mp.float4_e2m1fn)
  rhs = mp.asarray([-0.5, -1.0, -1.5], dtype=mp.float4_e2m1fn)

  out = mp.stack([lhs, rhs])

  assert out.dtype == mp.float4_e2m1fn
  assert out.shape == (2, 3)
  assert out.nbytes == 3
  assert out.tolist() == [[0.5, 1.0, 1.5], [-0.5, -1.0, -1.5]]


def test_fp4_assignment_to_odd_offset_view_repacks() -> None:
  dst = mp.full((5,), 0.0, dtype=mp.float4_e2m1fn)
  src = mp.asarray([0.5, 1.0, 1.5, 2.0], dtype=mp.float4_e2m1fn)

  dst[1:] = src

  assert dst.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]


@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
def test_low_precision_interop_rejects_unsupported_buffer_exports(dtype: mp.DType) -> None:
  arr = mp.asarray([1.0], dtype=dtype)
  with pytest.raises(BufferError):
    _ = arr.__array_interface__
  with pytest.raises(BufferError):
    arr.__dlpack__()


def test_kernel_dtype_specs_use_public_low_precision_storage() -> None:
  bf16 = dtype_spec(mp.bfloat16)
  fp8 = dtype_spec(mp.float8_e4m3fn)
  fp4 = dtype_spec(mp.float4_e2m1fn)

  assert bf16.eager_storage_supported
  assert bf16.bits == 16
  assert bf16.storage_bits == 16
  assert fp8.kind.value == "quant_float"
  assert fp8.storage_bits == 8
  assert fp4.storage is StorageKind.PACKED_SUBBYTE
  assert fp4.bits == 4
  assert fp4.storage_bits == 4


@pytest.mark.parametrize("dtype", mp._DT)
def test_public_dtype_maps_to_one_compiler_dtype_spec(dtype: mp.DType) -> None:
  spec = dtype_spec(dtype)

  assert spec.name == dtype.name
  assert spec.code == dtype.code
  assert spec.bits == dtype.bits
  assert spec.storage_bits == dtype.storage_bits
  assert spec.is_packed == dtype.is_packed


def test_compiler_dtype_specs_match_native_domain_codes() -> None:
  native_codes = mp._native._domain_codes()["dtype"]

  for dtype in mp._DT:
    assert dtype_spec(dtype).code == native_codes[dtype.name.upper()]


def test_float4_compiler_dtype_remains_packed_subbyte() -> None:
  spec = dtype_spec(mp.float4_e2m1fn)

  assert spec.storage is StorageKind.PACKED_SUBBYTE
  assert spec.storage_bits == 4
  assert spec.bits == 4
  with pytest.raises(ValueError, match="integer byte width"):
    _ = spec.byte_width
