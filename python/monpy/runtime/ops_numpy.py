# fmt: off # ruff: noqa
from __future__ import annotations
import typing, numpy, monpy as _mp

if typing.TYPE_CHECKING:from numpy.typing import NDArray

def resolve_dtype(dtype:object)->_mp.DType:
  if isinstance(dtype,_mp.DType):return dtype
  try:nd=numpy.dtype(dtype)
  except TypeError:
    module=getattr(dtype,"__module__",None)
    if isinstance(dtype,type) and isinstance(module,str) and (module=="numpy" or module.startswith("numpy.")):
      raise NotImplementedError(f"unsupported dtype: {dtype!r}")
    return _mp._resolve_dtype(dtype)
  if nd.fields is not None or nd.subdtype is not None:raise NotImplementedError(f"unsupported dtype: {nd}")
  if not nd.isnative:raise NotImplementedError(f"unsupported dtype: {nd}")
  return _mp._dtype_from_typestr(nd.str)

def to_numpy(arr:object,dtype:object=None,copy:bool|None=None)->NDArray[typing.Any]:
  monpy_arr=_mp.asarray(arr)
  class _Owner:
    def __init__(self,a:_mp.ndarray)->None:self._owner=a;self.__array_interface__=a.__array_interface__
  return typing.cast(typing.Any,numpy.asarray(typing.cast(typing.Any,_Owner(monpy_arr)),dtype=typing.cast(typing.Any,dtype),copy=copy))

def asarray(obj:object,dtype:object=None,copy:bool|None=None,device:object=None)->_mp.ndarray:
  resolved=None if dtype is None else resolve_dtype(dtype)
  return _mp.asarray(obj,dtype=resolved,copy=copy,device=device)

__all__=["asarray","resolve_dtype","to_numpy"]
