"""Dtype metadata for the optional kernel compiler path."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DTypeKind(str, Enum):
  BOOL = "bool"
  SIGNED_INT = "signed_int"
  UNSIGNED_INT = "unsigned_int"
  REAL_FLOAT = "real_float"
  COMPLEX_FLOAT = "complex_float"
  QUANT_FLOAT = "quant_float"


class StorageKind(str, Enum):
  VALUE = "value"
  PACKED_SUBBYTE = "packed_subbyte"
  OPAQUE_BYTES = "opaque_bytes"


@dataclass(frozen=True, slots=True)
class DTypeSpec:
  """Backend-neutral dtype contract used by GraphIR.

  `DTypeSpec` is not the eager `monpy.DType` object. It carries enough storage
  and backend mapping metadata for tracing and lowering without importing MAX.
  """

  name: str
  code: int
  kind: DTypeKind
  storage: StorageKind
  bits: int
  storage_bits: int
  max_name: str | None = None
  safetensors_names: tuple[str, ...] = ()
  host_storage_dtype: str | None = None
  eager_storage_supported: bool = False

  @property
  def byte_width(self) -> int:
    if self.storage_bits % 8 != 0:
      raise ValueError(f"{self.name} is packed and does not have an integer byte width")
    return self.storage_bits // 8

  @property
  def is_packed(self) -> bool:
    return self.storage is StorageKind.PACKED_SUBBYTE


def _kind_from_monpy(kind: str) -> DTypeKind:
  if kind == "b":
    return DTypeKind.BOOL
  if kind == "i":
    return DTypeKind.SIGNED_INT
  if kind == "u":
    return DTypeKind.UNSIGNED_INT
  if kind == "f":
    return DTypeKind.REAL_FLOAT
  if kind == "c":
    return DTypeKind.COMPLEX_FLOAT
  raise TypeError(f"unsupported monpy dtype kind: {kind!r}")


def from_monpy_dtype(dtype: object) -> DTypeSpec:
  """Convert a public eager dtype into a kernel dtype spec."""

  if isinstance(dtype, DTypeSpec):
    return dtype
  name = getattr(dtype, "name", None)
  code = getattr(dtype, "code", None)
  kind = getattr(dtype, "kind", None)
  itemsize = getattr(dtype, "itemsize", None)
  if not isinstance(name, str) or not isinstance(code, int) or not isinstance(kind, str) or not isinstance(itemsize, int):
    raise TypeError(f"expected monpy dtype or DTypeSpec, got {type(dtype).__name__}")
  return DTypeSpec(
    name=name,
    code=code,
    kind=_kind_from_monpy(kind),
    storage=StorageKind.VALUE,
    bits=itemsize * 8,
    storage_bits=itemsize * 8,
    max_name=name,
    safetensors_names=(),
    host_storage_dtype=name,
    eager_storage_supported=True,
  )


_EXTRA_DTYPES: tuple[DTypeSpec, ...] = (
  DTypeSpec("bfloat16", 14, DTypeKind.REAL_FLOAT, StorageKind.VALUE, 16, 16, "bfloat16", ("BF16",), "uint16"),
  DTypeSpec("float8_e4m3fn", 15, DTypeKind.QUANT_FLOAT, StorageKind.VALUE, 8, 8, "float8_e4m3fn", (), "uint8"),
  DTypeSpec("float8_e4m3fnuz", 16, DTypeKind.QUANT_FLOAT, StorageKind.VALUE, 8, 8, "float8_e4m3fnuz", (), "uint8"),
  DTypeSpec("float8_e5m2", 17, DTypeKind.QUANT_FLOAT, StorageKind.VALUE, 8, 8, "float8_e5m2", (), "uint8"),
  DTypeSpec("float8_e5m2fnuz", 18, DTypeKind.QUANT_FLOAT, StorageKind.VALUE, 8, 8, "float8_e5m2fnuz", (), "uint8"),
  DTypeSpec("float8_e8m0fnu", 19, DTypeKind.QUANT_FLOAT, StorageKind.VALUE, 8, 8, "float8_e8m0fnu", (), "uint8"),
  DTypeSpec("float4_e2m1fn", 20, DTypeKind.QUANT_FLOAT, StorageKind.PACKED_SUBBYTE, 4, 4, "float4_e2m1fn", (), "uint8"),
)

EXTRA_DTYPES: dict[str, DTypeSpec] = {spec.name: spec for spec in _EXTRA_DTYPES}


def dtype_spec(dtype: object) -> DTypeSpec:
  if isinstance(dtype, str) and dtype in EXTRA_DTYPES:
    return EXTRA_DTYPES[dtype]
  return from_monpy_dtype(dtype)


def to_max_dtype(spec: DTypeSpec) -> Any:
  """Map to `max.dtype.DType`, importing MAX only at the lowering edge."""

  if spec.max_name is None:
    raise TypeError(f"{spec.name} has no MAX dtype mapping")
  from max.dtype import DType

  try:
    return getattr(DType, spec.max_name)
  except AttributeError as exc:
    raise TypeError(f"installed MAX does not expose dtype {spec.max_name!r}") from exc
