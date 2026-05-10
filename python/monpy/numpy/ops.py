from __future__ import annotations

import sys
import typing
from dataclasses import dataclass

from monpy.utils import LazyLoader

if typing.TYPE_CHECKING:
  import numpy
  from numpy.typing import NDArray

  import monpy as _mp
else:
  _mp = LazyLoader("_mp", globals(), "monpy")
  numpy = LazyLoader("numpy", globals(), "numpy")


@dataclass(frozen=True, slots=True)
class NumpyDTypeInfo:
  itemsize: int
  alignment: int
  byteorder: str
  typestr: str
  format: str
  scalar_type: type[object]
  buffer_exportable: bool


_NUMPY_DTYPE_INFO_BY_CODE: dict[int, NumpyDTypeInfo] | None = None
_DTYPE_BY_TYPESTR: dict[str, _mp.DType] | None = None
_DTYPE_BY_BUFFER_FORMAT: dict[str, _mp.DType] | None = None


def _dtype_info_table() -> dict[int, NumpyDTypeInfo]:
  global _NUMPY_DTYPE_INFO_BY_CODE
  if _NUMPY_DTYPE_INFO_BY_CODE is None:
    _NUMPY_DTYPE_INFO_BY_CODE = {
      _mp.bool.code: NumpyDTypeInfo(1, 1, "|", "|b1", "?", bool, True),
      _mp.int64.code: NumpyDTypeInfo(8, 8, "=", "<i8", "l", int, True),
      _mp.float32.code: NumpyDTypeInfo(4, 4, "=", "<f4", "f", float, True),
      _mp.float64.code: NumpyDTypeInfo(8, 8, "=", "<f8", "d", float, True),
      _mp.int32.code: NumpyDTypeInfo(4, 4, "=", "<i4", "i", int, True),
      _mp.int16.code: NumpyDTypeInfo(2, 2, "=", "<i2", "h", int, True),
      _mp.int8.code: NumpyDTypeInfo(1, 1, "|", "|i1", "b", int, True),
      _mp.uint64.code: NumpyDTypeInfo(8, 8, "=", "<u8", "Q", int, True),
      _mp.uint32.code: NumpyDTypeInfo(4, 4, "=", "<u4", "I", int, True),
      _mp.uint16.code: NumpyDTypeInfo(2, 2, "=", "<u2", "H", int, True),
      _mp.uint8.code: NumpyDTypeInfo(1, 1, "|", "|u1", "B", int, True),
      _mp.float16.code: NumpyDTypeInfo(2, 2, "=", "<f2", "e", float, True),
      _mp.complex64.code: NumpyDTypeInfo(8, 4, "=", "<c8", "F", complex, True),
      _mp.complex128.code: NumpyDTypeInfo(16, 8, "=", "<c16", "D", complex, True),
      _mp.bfloat16.code: NumpyDTypeInfo(2, 2, "=", "", "", float, False),
      _mp.float8_e4m3fn.code: NumpyDTypeInfo(1, 1, "|", "", "", float, False),
      _mp.float8_e4m3fnuz.code: NumpyDTypeInfo(1, 1, "|", "", "", float, False),
      _mp.float8_e5m2.code: NumpyDTypeInfo(1, 1, "|", "", "", float, False),
      _mp.float8_e5m2fnuz.code: NumpyDTypeInfo(1, 1, "|", "", "", float, False),
      _mp.float8_e8m0fnu.code: NumpyDTypeInfo(1, 1, "|", "", "", float, False),
      _mp.float4_e2m1fn.code: NumpyDTypeInfo(1, 1, "|", "", "", float, False),
    }
  return _NUMPY_DTYPE_INFO_BY_CODE


def dtype_info(dtype: object) -> NumpyDTypeInfo:
  resolved = _mp._resolve_dtype(dtype)
  try:
    return _dtype_info_table()[resolved.code]
  except KeyError as exc:
    raise NotImplementedError(f"unsupported dtype metadata: {resolved!r}") from exc


def array_interface_typestr(dtype: object) -> str:
  return dtype_info(dtype).typestr


def buffer_format(dtype: object) -> str:
  return dtype_info(dtype).format


def _typestr_table() -> dict[str, _mp.DType]:
  global _DTYPE_BY_TYPESTR
  if _DTYPE_BY_TYPESTR is None:
    table = {
      info.typestr: dtype
      for dtype in _mp._DT
      for info in (dtype_info(dtype),)
      if info.typestr
    }
    table.update(
      {
        "|b1": _mp.bool,
        "|i1": _mp.int8,
        "|u1": _mp.uint8,
      }
    )
    for suffix, dtype in (
      ("i8", _mp.int64),
      ("i4", _mp.int32),
      ("i2", _mp.int16),
      ("u8", _mp.uint64),
      ("u4", _mp.uint32),
      ("u2", _mp.uint16),
      ("f2", _mp.float16),
      ("f4", _mp.float32),
      ("f8", _mp.float64),
      ("c8", _mp.complex64),
      ("c16", _mp.complex128),
    ):
      for prefix in ("<", "="):
        table[prefix + suffix] = dtype
    _DTYPE_BY_TYPESTR = table
  return _DTYPE_BY_TYPESTR


def dtype_from_typestr(typestr: str) -> _mp.DType:
  try:
    return _typestr_table()[typestr]
  except KeyError as exc:
    raise NotImplementedError(f"unsupported array-interface typestr: {typestr}") from exc


def _buffer_format_table() -> dict[str, _mp.DType]:
  global _DTYPE_BY_BUFFER_FORMAT
  if _DTYPE_BY_BUFFER_FORMAT is None:
    _DTYPE_BY_BUFFER_FORMAT = {
      info.format: dtype
      for dtype in _mp._DT
      for info in (dtype_info(dtype),)
      if info.format
    }
  return _DTYPE_BY_BUFFER_FORMAT


def dtype_from_buffer_format(format: str) -> _mp.DType:
  try:
    return _buffer_format_table()[format]
  except KeyError as exc:
    raise NotImplementedError(f"unsupported buffer dtype format: {format}") from exc


def _is_numpy_module_name(name: object) -> bool:
  return isinstance(name, str) and (name == "numpy" or name.startswith("numpy."))


def is_dtype_input(value: object) -> bool:
  if isinstance(value, type):
    return _is_numpy_module_name(getattr(value, "__module__", None))
  return any(
    _is_numpy_module_name(getattr(base, "__module__", None))
    and getattr(base, "__name__", None) in ("dtype", "generic")
    for base in type(value).__mro__
  )


def is_array_input(value: object) -> bool:
  numpy_module = sys.modules.get("numpy")
  if numpy_module is not None:
    ndarray = getattr(numpy_module, "ndarray", None)
    if ndarray is not None:
      return isinstance(value, ndarray)
  return any(
    _is_numpy_module_name(getattr(base, "__module__", None)) and getattr(base, "__name__", None) == "ndarray"
    for base in type(value).__mro__
  )


def abstract_dtype_set(value: object) -> set[_mp.DType] | None:
  if not isinstance(value, type) or not _is_numpy_module_name(getattr(value, "__module__", None)):
    return None
  name = getattr(value, "__name__", None)
  if not isinstance(name, str):
    return None
  return {
    "bool": {_mp.bool},
    "bool_": {_mp.bool},
    "integer": {_mp.int64, _mp.int32, _mp.int16, _mp.int8, _mp.uint64, _mp.uint32, _mp.uint16, _mp.uint8},
    "signedinteger": {_mp.int64, _mp.int32, _mp.int16, _mp.int8},
    "unsignedinteger": {_mp.uint64, _mp.uint32, _mp.uint16, _mp.uint8},
    "floating": {_mp.float32, _mp.float64, _mp.float16},
    "complexfloating": {_mp.complex64, _mp.complex128},
    "inexact": {_mp.float32, _mp.float64, _mp.float16, _mp.complex64, _mp.complex128},
    "number": {
      _mp.int64,
      _mp.int32,
      _mp.int16,
      _mp.int8,
      _mp.uint64,
      _mp.uint32,
      _mp.uint16,
      _mp.uint8,
      _mp.float32,
      _mp.float64,
      _mp.float16,
      _mp.complex64,
      _mp.complex128,
    },
    "generic": set(_mp._DT),
  }.get(name)


def resolve_dtype(dtype: object) -> _mp.DType:
  if isinstance(dtype, _mp.DType):
    return dtype
  try:
    nd = numpy.dtype(dtype)
  except TypeError:
    module = getattr(dtype, "__module__", None)
    if isinstance(dtype, type) and _is_numpy_module_name(module):
      raise NotImplementedError(f"unsupported dtype: {dtype!r}")
    return _mp._resolve_dtype(dtype)
  if nd.fields is not None or nd.subdtype is not None:
    raise NotImplementedError(f"unsupported dtype: {nd}")
  if not nd.isnative:
    raise NotImplementedError(f"unsupported dtype: {nd}")
  return _mp._dtype_from_typestr(nd.str)


def to_numpy(arr: object, dtype: object = None, copy: bool | None = None) -> NDArray[typing.Any]:
  monpy_arr = _mp.asarray(arr)

  class _Owner:
    def __init__(self, a: _mp.ndarray) -> None:
      self._owner = a
      self.__array_interface__ = a.__array_interface__

  return numpy.asarray(typing.cast(typing.Any, _Owner(monpy_arr)), dtype=typing.cast(typing.Any, dtype), copy=copy)


def _from_numpy_unchecked(
  arr: object, dtype: object = None, copy: bool | None = None, device: object = None
) -> _mp.ndarray:
  if device is not None and device != "cpu":
    raise NotImplementedError("monpy v1 only supports cpu arrays")
  resolved = None if dtype is None else resolve_dtype(dtype)
  if resolved is _mp.complex128 and copy is True:
    try:
      return _mp.ndarray(_mp._native.asarray_complex128_copy_from_buffer(arr))
    except Exception as exc:
      if "buffer format unsupported" in str(exc):
        raise NotImplementedError("unsupported dtype") from exc
      raise
  requested = -1 if resolved is None else resolved.code
  copy_flag = -1 if copy is None else (1 if copy else 0)
  try:
    native = _mp._native.asarray_from_buffer(arr, requested, copy_flag)
  except Exception as exc:
    message = str(exc)
    if copy is False and ("copy" in message or "readonly" in message):
      raise ValueError(message) from exc
    if "buffer format unsupported" in message:
      raise NotImplementedError("unsupported dtype") from exc
    raise
  return _mp.ndarray(native, owner=None if copy_flag == 1 else arr)


def from_numpy(arr: object, dtype: object = None, copy: bool | None = None, device: object = None) -> _mp.ndarray:
  if not is_array_input(arr):
    raise TypeError("from_numpy() expects a numpy.ndarray")
  return _from_numpy_unchecked(arr, dtype=dtype, copy=copy, device=device)


def asarray(obj: object, dtype: object = None, copy: bool | None = None, device: object = None) -> _mp.ndarray:
  if is_array_input(obj):
    return _from_numpy_unchecked(obj, dtype=dtype, copy=copy, device=device)
  resolved = None if dtype is None else resolve_dtype(dtype)
  return _mp.asarray(obj, dtype=resolved, copy=copy, device=device)


__all__ = [
  "NumpyDTypeInfo",
  "abstract_dtype_set",
  "array_interface_typestr",
  "asarray",
  "buffer_format",
  "dtype_from_buffer_format",
  "dtype_from_typestr",
  "dtype_info",
  "from_numpy",
  "is_array_input",
  "is_dtype_input",
  "resolve_dtype",
  "to_numpy",
]
