# fmt: off
# ruff: noqa
from __future__ import annotations

import typing
import sys
from monpy.utils import LazyLoader

if typing.TYPE_CHECKING:
  import numpy, monpy as _mp
  from numpy.typing import NDArray
else:
  _mp = LazyLoader("_mp", globals(), "monpy")
  numpy = LazyLoader("numpy", globals(), "numpy")

def _is_numpy_module_name(name:object)->bool:
  return isinstance(name, str) and (name=="numpy" or name.startswith("numpy."))

def is_dtype_input(value:object)->bool:
  if isinstance(value, type):return _is_numpy_module_name(getattr(value, "__module__", None))
  return any(
    _is_numpy_module_name(getattr(base, "__module__", None)) and getattr(base, "__name__", None) in("dtype", "generic")
    for base in type(value).__mro__
  )

def is_array_input(value:object)->bool:
  numpy_module=sys.modules.get("numpy")
  if numpy_module is not None:
    ndarray=getattr(numpy_module, "ndarray", None)
    if ndarray is not None:return isinstance(value, ndarray)
  return any(
    _is_numpy_module_name(getattr(base, "__module__", None)) and getattr(base, "__name__", None)=="ndarray"
    for base in type(value).__mro__
  )

def abstract_dtype_set(value:object)->set[_mp.DType]|None:
  if not isinstance(value, type) or not _is_numpy_module_name(getattr(value, "__module__", None)):return None
  name=getattr(value, "__name__", None)
  if not isinstance(name, str):return None
  return {
    "bool":{_mp.bool}, "bool_":{_mp.bool},
    "integer":{_mp.int64, _mp.int32, _mp.int16, _mp.int8, _mp.uint64, _mp.uint32, _mp.uint16, _mp.uint8},
    "signedinteger":{_mp.int64, _mp.int32, _mp.int16, _mp.int8},
    "unsignedinteger":{_mp.uint64, _mp.uint32, _mp.uint16, _mp.uint8},
    "floating":{_mp.float32, _mp.float64, _mp.float16},
    "complexfloating":{_mp.complex64, _mp.complex128},
    "inexact":{_mp.float32, _mp.float64, _mp.float16, _mp.complex64, _mp.complex128},
    "number":{_mp.int64, _mp.int32, _mp.int16, _mp.int8, _mp.uint64, _mp.uint32, _mp.uint16, _mp.uint8, _mp.float32, _mp.float64, _mp.float16, _mp.complex64, _mp.complex128},
    "generic":set(_mp._DT),
  }.get(name)

def resolve_dtype(dtype:object)->_mp.DType:
  if isinstance(dtype, _mp.DType):return dtype
  try:nd=numpy.dtype(dtype)
  except TypeError:
    module=getattr(dtype, "__module__", None)
    if isinstance(dtype, type) and _is_numpy_module_name(module):
      raise NotImplementedError(f"unsupported dtype: {dtype!r}")
    return _mp._resolve_dtype(dtype)
  if nd.fields is not None or nd.subdtype is not None:raise NotImplementedError(f"unsupported dtype: {nd}")
  if not nd.isnative:raise NotImplementedError(f"unsupported dtype: {nd}")
  return _mp._dtype_from_typestr(nd.str)

def to_numpy(arr:object, dtype:object=None, copy:bool|None=None)->NDArray[typing.Any]:
  monpy_arr=_mp.asarray(arr)
  class _Owner:
    def __init__(self, a:_mp.ndarray)->None:
      self._owner=a
      self.__array_interface__=a.__array_interface__
  return numpy.asarray(typing.cast(typing.Any, _Owner(monpy_arr)), dtype=typing.cast(typing.Any, dtype), copy=copy)

def _from_numpy_unchecked(arr:object, dtype:object=None, copy:bool|None=None, device:object=None)->_mp.ndarray:
  if device is not None and device!="cpu":raise NotImplementedError("monpy v1 only supports cpu arrays")
  source_dtype=_mp._dtype_from_typestr(arr.dtype.str)
  target=None if dtype is None else resolve_dtype(dtype)
  shape=tuple(int(d) for d in arr.shape)
  item_size=source_dtype.itemsize
  strides=tuple(int(s) for s in arr.strides)
  for stride in strides:
    if stride%item_size!=0:raise NotImplementedError("numpy strides must align to dtype itemsize")
  elem_strides=tuple(stride//item_size for stride in strides)
  data_address=int(arr.ctypes.data)
  byte_len=int(arr.size)*item_size
  readonly=not bool(arr.flags.writeable)
  if target is not None and target!=source_dtype:
    if copy is False:raise ValueError(_mp._CFE)
    return _mp.ndarray(_mp._native.copy_from_external(data_address, shape, elem_strides, source_dtype.code, byte_len)).astype(target)
  if copy is True or readonly:
    if readonly and copy is False:raise ValueError("readonly array requires copy=True")
    return _mp.ndarray(_mp._native.copy_from_external(data_address, shape, elem_strides, source_dtype.code, byte_len))
  return _mp.ndarray(_mp._native.from_external(data_address, shape, elem_strides, source_dtype.code, byte_len), owner=arr)

def from_numpy(arr:object, dtype:object=None, copy:bool|None=None, device:object=None)->_mp.ndarray:
  if not is_array_input(arr):raise TypeError("from_numpy() expects a numpy.ndarray")
  return _from_numpy_unchecked(arr, dtype=dtype, copy=copy, device=device)

def asarray(obj:object, dtype:object=None, copy:bool|None=None, device:object=None)->_mp.ndarray:
  if is_array_input(obj):return _from_numpy_unchecked(obj, dtype=dtype, copy=copy, device=device)
  resolved=None if dtype is None else resolve_dtype(dtype)
  return _mp.asarray(obj, dtype=resolved, copy=copy, device=device)

__all__=["abstract_dtype_set", "asarray", "from_numpy", "is_array_input", "is_dtype_input", "resolve_dtype", "to_numpy"]
