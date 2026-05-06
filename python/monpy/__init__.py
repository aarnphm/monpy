from __future__ import annotations

import builtins
import importlib
import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import ModuleType

from . import _native

# This module is intentionally the NumPy-shaped Python facade. Implemented
# array work should delegate into _native, which is backed by the Mojo runtime.

DTYPE_BOOL = 0
DTYPE_INT64 = 1
DTYPE_FLOAT32 = 2
DTYPE_FLOAT64 = 3

OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
OP_DIV = 3

UNARY_SIN = 0
UNARY_COS = 1
UNARY_EXP = 2
UNARY_LOG = 3

REDUCE_SUM = 0
REDUCE_MEAN = 1
REDUCE_MIN = 2
REDUCE_MAX = 3
REDUCE_ARGMAX = 4


@dataclass(frozen=True, slots=True)
class DType:
  name: str
  code: int
  itemsize: int
  typestr: str

  def __repr__(self) -> str:
    return f"monpy.{self.name}"


bool = DType("bool", DTYPE_BOOL, 1, "|b1")
int64 = DType("int64", DTYPE_INT64, 8, "<i8")
float32 = DType("float32", DTYPE_FLOAT32, 4, "<f4")
float64 = DType("float64", DTYPE_FLOAT64, 8, "<f8")

_DTYPES_BY_CODE = {
  DTYPE_BOOL: bool,
  DTYPE_INT64: int64,
  DTYPE_FLOAT32: float32,
  DTYPE_FLOAT64: float64,
}
_DTYPES_BY_NAME = {dt.name: dt for dt in _DTYPES_BY_CODE.values()}
_DTYPES_BY_NUMPY_KIND = {
  ("b", 1): bool,
  ("i", 8): int64,
  ("f", 4): float32,
  ("f", 8): float64,
}
_DTYPES_BY_TYPESTR = {
  "|b1": bool,
  "<i8": int64,
  ">i8": int64,
  "=i8": int64,
  "<f4": float32,
  ">f4": float32,
  "=f4": float32,
  "<f8": float64,
  ">f8": float64,
  "=f8": float64,
}
_NUMPY_NDARRAY_TYPE: object = None

_NUMERIC_PROMOTION_TABLE = {
  (bool, bool): bool,
  (bool, int64): int64,
  (bool, float32): float32,
  (bool, float64): float64,
  (int64, bool): int64,
  (int64, int64): int64,
  (int64, float32): float64,
  (int64, float64): float64,
  (float32, bool): float32,
  (float32, int64): float64,
  (float32, float32): float32,
  (float32, float64): float64,
  (float64, bool): float64,
  (float64, int64): float64,
  (float64, float32): float64,
  (float64, float64): float64,
}
_DIVISION_PROMOTION_TABLE = {
  (bool, bool): float64,
  (bool, int64): float64,
  (bool, float32): float32,
  (bool, float64): float64,
  (int64, bool): float64,
  (int64, int64): float64,
  (int64, float32): float64,
  (int64, float64): float64,
  (float32, bool): float32,
  (float32, int64): float64,
  (float32, float32): float32,
  (float32, float64): float64,
  (float64, bool): float64,
  (float64, int64): float64,
  (float64, float32): float64,
  (float64, float64): float64,
}
_BINARY_RESULT_DTYPES = {
  OP_ADD: _NUMERIC_PROMOTION_TABLE,
  OP_SUB: _NUMERIC_PROMOTION_TABLE,
  OP_MUL: _NUMERIC_PROMOTION_TABLE,
  OP_DIV: _DIVISION_PROMOTION_TABLE,
}
_UNARY_RESULT_DTYPES = {
  bool: float64,
  int64: float64,
  float32: float32,
  float64: float64,
}
_COPY_FALSE_ERROR = "unable to avoid copy while creating a monpy array as requested"

newaxis = None
nan = math.nan
inf = math.inf
pi = math.pi
e = math.e


class ndarray:
  __array_priority__ = 1000
  __slots__ = ("_base", "_dtype", "_native", "_owner", "_shape", "_strides")

  def __init__(
    _self,
    native: _native.Array,
    base: ndarray | None = None,
    *,
    dtype: DType | None = None,
    shape: tuple[int, ...] | None = None,
    strides: tuple[int, ...] | None = None,
    owner: object | None = None,
  ) -> None:
    _self._native = native
    _self._base = base
    _self._dtype = dtype
    _self._owner = owner
    _self._shape = shape
    _self._strides = strides

  @property
  def dtype(self) -> DType:
    if self._dtype is None:
      self._dtype = _DTYPES_BY_CODE[int(self._native.dtype_code())]
    return self._dtype

  @property
  def shape(self) -> tuple[int, ...]:
    if self._shape is None:
      ndim = int(self._native.ndim())
      self._shape = tuple(int(self._native.shape_at(axis)) for axis in range(ndim))
    return self._shape

  @property
  def ndim(self) -> int:
    if self._shape is not None:
      return len(self._shape)
    return int(self._native.ndim())

  @property
  def size(self) -> int:
    if self._shape is not None:
      return math.prod(self._shape)
    return int(self._native.size())

  @property
  def itemsize(self) -> int:
    return self.dtype.itemsize

  @property
  def strides(self) -> tuple[int, ...]:
    if self._strides is None:
      itemsize = self.itemsize
      self._strides = tuple(int(self._native.stride_at(axis)) * itemsize for axis in range(self.ndim))
    return self._strides

  @property
  def device(self) -> str:
    return "cpu"

  @property
  def T(self) -> ndarray:
    cached_shape = self._shape
    if cached_shape is not None:
      ndim = len(cached_shape)
    else:
      ndim = int(self._native.ndim())
    if ndim < 2:
      return self
    return ndarray(_native.transpose_full_reverse(self._native), base=self)

  @property
  def mT(self) -> ndarray:
    if self.ndim < 2:
      raise ValueError("matrix transpose requires at least two dimensions")
    axes = tuple(range(self.ndim - 2)) + (self.ndim - 1, self.ndim - 2)
    return self.transpose(axes)

  @property
  def __array_interface__(self) -> dict[str, object]:
    strides = None if _is_c_contiguous_bytes(self.shape, self.strides, self.itemsize) else self.strides
    return {
      "version": 3,
      "shape": self.shape,
      "typestr": self.dtype.typestr,
      "data": (int(self._native.data_address()), False),
      "strides": strides,
    }

  def __array__(self, dtype: object = None, copy: builtins.bool | None = None) -> object:
    import numpy as numpy_oracle

    class _ArrayInterfaceOwner:
      def __init__(self, owner: ndarray) -> None:
        self._owner = owner
        self.__array_interface__ = owner.__array_interface__

    array = numpy_oracle.asarray(_ArrayInterfaceOwner(self))
    if dtype is not None:
      return array.astype(dtype, copy=copy is not False)
    if copy is True:
      return array.copy()
    return array

  def __array_namespace__(self, *, api_version: str | None = None) -> ModuleType:
    if api_version not in (None, "2024.12", "2025.12"):
      raise ValueError(f"unsupported array api version: {api_version}")
    from . import array_api

    return array_api

  def __dlpack__(
    self,
    *,
    stream: object = None,
    max_version: tuple[int, int] | None = None,
    dl_device: tuple[int, int] | None = None,
    copy: builtins.bool | None = None,
  ) -> object:
    if stream is not None:
      raise BufferError("cpu dlpack export requires stream=None")
    if dl_device not in (None, (1, 0)):
      raise BufferError("monpy only exports cpu dlpack tensors")
    array = self.__array__(copy=False)
    return array.__dlpack__(stream=None, max_version=max_version, dl_device=dl_device, copy=copy)

  def __dlpack_device__(self) -> tuple[int, int]:
    return (1, 0)

  def __len__(self) -> int:
    if self.ndim == 0:
      raise TypeError("len() of unsized object")
    return self.shape[0]

  def __iter__(self) -> Iterable[object]:
    for i in range(len(self)):
      yield self[i]

  def __repr__(self) -> str:
    return f"monpy.asarray({self.tolist()!r}, dtype={self.dtype!r})"

  def __getitem__(self, key: object) -> object:
    if isinstance(key, slice) and self.ndim == 1:
      dim = self.shape[0]
      step = 1 if key.step is None else int(key.step)
      if step == 0:
        raise ValueError("slice step cannot be zero")
      if key.start is None and key.stop is None:
        start = dim - 1 if step < 0 else 0
        stop = -1 if step < 0 else dim
      else:
        start, stop, step = key.indices(dim)
      return ndarray(
        _native.slice_1d(self._native, start, stop, step),
        base=self,
        dtype=self.dtype,
        shape=(len(range(start, stop, step)),),
        strides=(self.strides[0] * step,),
      )
    view = self._view_for_key(key)
    if view.ndim == 0:
      return view._scalar()
    return view

  def __setitem__(self, key: object, value: object) -> None:
    view = self._view_for_key(key)
    if isinstance(value, ndarray):
      _native.copyto(view._native, value._native)
      return
    _native.fill(view._native, value)

  def __bool__(self) -> builtins.bool:
    if self.size != 1:
      raise ValueError("the truth value of an array with more than one element is ambiguous")
    return builtins.bool(self._scalar())

  def __int__(self) -> int:
    if self.size != 1:
      raise TypeError("only size-1 arrays can be converted to python scalars")
    return int(self._scalar())

  def __float__(self) -> float:
    if self.size != 1:
      raise TypeError("only size-1 arrays can be converted to python scalars")
    return float(self._scalar())

  def __add__(self, other: object) -> object:
    if type(other) is ndarray:
      self_dtype = self._dtype
      if self_dtype is not None and self_dtype is other._dtype:
        return ndarray(_native.binary(self._native, other._native, OP_ADD))
    return _binary_from_array(self, other, OP_ADD, scalar_on_left=False)

  def __radd__(self, other: object) -> object:
    return _binary_from_array(self, other, OP_ADD, scalar_on_left=True)

  def __sub__(self, other: object) -> object:
    if type(other) is ndarray:
      self_dtype = self._dtype
      if self_dtype is not None and self_dtype is other._dtype:
        return ndarray(_native.binary(self._native, other._native, OP_SUB))
    return _binary_from_array(self, other, OP_SUB, scalar_on_left=False)

  def __rsub__(self, other: object) -> object:
    return _binary_from_array(self, other, OP_SUB, scalar_on_left=True)

  def __mul__(self, other: object) -> object:
    if type(other) is ndarray:
      self_dtype = self._dtype
      if self_dtype is not None and self_dtype is other._dtype:
        return ndarray(_native.binary(self._native, other._native, OP_MUL))
    return _binary_from_array(self, other, OP_MUL, scalar_on_left=False)

  def __rmul__(self, other: object) -> object:
    return _binary_from_array(self, other, OP_MUL, scalar_on_left=True)

  def __truediv__(self, other: object) -> object:
    return _binary_from_array(self, other, OP_DIV, scalar_on_left=False)

  def __rtruediv__(self, other: object) -> object:
    return _binary_from_array(self, other, OP_DIV, scalar_on_left=True)

  def __matmul__(self, other: object) -> ndarray:
    if isinstance(other, ndarray):
      lhs, rhs = _coerce_binary_operands(self, other, OP_MUL)
      return ndarray(_native.matmul(lhs._native, rhs._native))
    return matmul(self, other)

  def __rmatmul__(self, other: object) -> ndarray:
    if isinstance(other, ndarray):
      lhs, rhs = _coerce_binary_operands(other, self, OP_MUL)
      return ndarray(_native.matmul(lhs._native, rhs._native))
    return matmul(other, self)

  def __neg__(self) -> ndarray:
    return multiply(self, -1)

  def __pos__(self) -> ndarray:
    return self

  def reshape(self, *shape: int | Sequence[int]) -> ndarray:
    normalized = _shape_from_args(shape)
    return ndarray(_native.reshape(self._native, normalized), base=self)

  def transpose(self, axes: Sequence[int] | None = None) -> ndarray:
    if axes is None:
      axes = tuple(range(self.ndim - 1, -1, -1))
    normalized = _normalize_axes(axes, self.ndim)
    return ndarray(_native.transpose(self._native, normalized), base=self)

  def astype(self, dtype: object, *, copy: builtins.bool = True, device: object = None) -> ndarray:
    _check_cpu_device(device)
    target = _resolve_dtype(dtype)
    if target == self.dtype and not copy:
      return self
    return ndarray(_native.astype(self._native, target.code))

  def tolist(self) -> object:
    flat = [self._native.get_scalar(i) for i in range(self.size)]
    return _unflatten(flat, self.shape)

  def sum(self, axis: object = None) -> object:
    return sum(self, axis=axis)

  def mean(self, axis: object = None) -> object:
    return mean(self, axis=axis)

  def min(self, axis: object = None) -> object:
    return min(self, axis=axis)

  def max(self, axis: object = None) -> object:
    return max(self, axis=axis)

  def argmax(self, axis: object = None) -> object:
    return argmax(self, axis=axis)

  def _scalar(self) -> object:
    if self.size != 1:
      raise TypeError("only size-1 arrays can be converted to python scalars")
    return self._native.get_scalar(0)

  def _view_for_key(self, key: object) -> ndarray:
    parts = _expand_key(key, self.ndim)
    starts: list[int] = []
    stops: list[int] = []
    steps: list[int] = []
    drops: list[int] = []
    for axis, part in enumerate(parts):
      dim = self.shape[axis]
      if isinstance(part, slice):
        start, stop, step = part.indices(dim)
        starts.append(start)
        stops.append(stop)
        steps.append(step)
        drops.append(0)
      else:
        index = _normalize_index(part, dim)
        starts.append(index)
        stops.append(index + 1)
        steps.append(1)
        drops.append(1)
    return ndarray(_native.slice(self._native, tuple(starts), tuple(stops), tuple(steps), tuple(drops)), base=self)


class _DeferredArray:
  __array_priority__ = 1001
  __slots__ = ("_cached",)

  def __init__(self) -> None:
    self._cached: ndarray | None = None

  @property
  def _native(self) -> _native.Array:
    return self._materialize()._native

  @property
  def dtype(self) -> DType:
    raise NotImplementedError

  @property
  def shape(self) -> tuple[int, ...]:
    raise NotImplementedError

  @property
  def ndim(self) -> int:
    return len(self.shape)

  @property
  def size(self) -> int:
    return math.prod(self.shape)

  @property
  def itemsize(self) -> int:
    return self.dtype.itemsize

  @property
  def strides(self) -> tuple[int, ...]:
    return self._materialize().strides

  @property
  def device(self) -> str:
    return "cpu"

  @property
  def T(self) -> ndarray:
    return self._materialize().T

  @property
  def mT(self) -> ndarray:
    return self._materialize().mT

  @property
  def __array_interface__(self) -> dict[str, object]:
    return self._materialize().__array_interface__

  def __array__(self, dtype: object = None, copy: builtins.bool | None = None) -> object:
    return self._materialize().__array__(dtype=dtype, copy=copy)

  def __array_namespace__(self, *, api_version: str | None = None) -> ModuleType:
    return self._materialize().__array_namespace__(api_version=api_version)

  def __dlpack__(
    self,
    *,
    stream: object = None,
    max_version: tuple[int, int] | None = None,
    dl_device: tuple[int, int] | None = None,
    copy: builtins.bool | None = None,
  ) -> object:
    return self._materialize().__dlpack__(stream=stream, max_version=max_version, dl_device=dl_device, copy=copy)

  def __dlpack_device__(self) -> tuple[int, int]:
    return self._materialize().__dlpack_device__()

  def __len__(self) -> int:
    return len(self._materialize())

  def __iter__(self) -> Iterable[object]:
    return iter(self._materialize())

  def __repr__(self) -> str:
    return repr(self._materialize())

  def __getitem__(self, key: object) -> object:
    return self._materialize()[key]

  def __setitem__(self, key: object, value: object) -> None:
    self._materialize()[key] = value

  def __bool__(self) -> builtins.bool:
    return builtins.bool(self._materialize())

  def __int__(self) -> int:
    return int(self._materialize())

  def __float__(self) -> float:
    return float(self._materialize())

  def __add__(self, other: object) -> object:
    return _binary(self, other, OP_ADD)

  def __radd__(self, other: object) -> object:
    return _binary(other, self, OP_ADD)

  def __sub__(self, other: object) -> object:
    return _binary(self, other, OP_SUB)

  def __rsub__(self, other: object) -> object:
    return _binary(other, self, OP_SUB)

  def __mul__(self, other: object) -> object:
    return _binary(self, other, OP_MUL)

  def __rmul__(self, other: object) -> object:
    return _binary(other, self, OP_MUL)

  def __truediv__(self, other: object) -> object:
    return _binary(self, other, OP_DIV)

  def __rtruediv__(self, other: object) -> object:
    return _binary(other, self, OP_DIV)

  def __matmul__(self, other: object) -> ndarray:
    return matmul(self, other)

  def __rmatmul__(self, other: object) -> ndarray:
    return matmul(other, self)

  def __neg__(self) -> object:
    return multiply(self, -1)

  def __pos__(self) -> object:
    return self

  def reshape(self, *shape: int | Sequence[int]) -> ndarray:
    return self._materialize().reshape(*shape)

  def transpose(self, axes: Sequence[int] | None = None) -> ndarray:
    return self._materialize().transpose(axes)

  def astype(self, dtype: object, *, copy: builtins.bool = True, device: object = None) -> ndarray:
    return self._materialize().astype(dtype, copy=copy, device=device)

  def tolist(self) -> object:
    return self._materialize().tolist()

  def sum(self, axis: object = None) -> object:
    return sum(self, axis=axis)

  def mean(self, axis: object = None) -> object:
    return mean(self, axis=axis)

  def min(self, axis: object = None) -> object:
    return min(self, axis=axis)

  def max(self, axis: object = None) -> object:
    return max(self, axis=axis)

  def argmax(self, axis: object = None) -> object:
    return argmax(self, axis=axis)

  def _materialize(self) -> ndarray:
    if self._cached is None:
      self._cached = self._compute()
    return self._cached

  def _compute(self) -> ndarray:
    raise NotImplementedError


class _UnaryExpression(_DeferredArray):
  __slots__ = ("_base", "_op")

  def __init__(self, base: ndarray | _DeferredArray, op: int) -> None:
    super().__init__()
    self._base = base
    self._op = op

  @property
  def dtype(self) -> DType:
    return _result_dtype_for_unary(self._base.dtype)

  @property
  def shape(self) -> tuple[int, ...]:
    return self._base.shape

  def _compute(self) -> ndarray:
    base = _materialize_array(self._base)
    return ndarray(_native.unary(base._native, self._op))


class _ScalarBinaryExpression(_DeferredArray):
  __slots__ = ("_array", "_op", "_scalar", "_scalar_dtype", "_scalar_on_left")

  def __init__(
    self,
    array: ndarray | _DeferredArray,
    scalar: object,
    scalar_dtype: DType,
    op: int,
    scalar_on_left: builtins.bool,
  ) -> None:
    super().__init__()
    self._array = array
    self._scalar = scalar
    self._scalar_dtype = scalar_dtype
    self._op = op
    self._scalar_on_left = scalar_on_left

  @property
  def dtype(self) -> DType:
    return _result_dtype_for_binary(self._array.dtype, self._scalar_dtype, self._op)

  @property
  def shape(self) -> tuple[int, ...]:
    return self._array.shape

  def _compute(self) -> ndarray:
    array = _materialize_array(self._array)
    return ndarray(
      _native.binary_scalar(array._native, self._scalar, self._scalar_dtype.code, self._op, self._scalar_on_left)
    )


def dtype(value: object) -> DType:
  return _resolve_dtype(value)


def array(obj: object, dtype: object = None, *, copy: builtins.bool | None = True, device: object = None) -> ndarray:
  return asarray(obj, dtype=dtype, copy=copy, device=device)


def asarray(obj: object, dtype: object = None, *, copy: builtins.bool | None = None, device: object = None) -> ndarray:
  if device is not None and device != "cpu":
    raise NotImplementedError("monpy v1 only supports cpu arrays")
  obj_type = type(obj)
  if obj_type is ndarray:
    if dtype is None:
      if copy is True:
        return obj.astype(obj.dtype, copy=True)
      return obj
    target = _resolve_dtype(dtype)
    if target == obj.dtype and copy is not True:
      return obj
    if copy is False:
      raise ValueError(_COPY_FALSE_ERROR)
    return obj.astype(target, copy=True)
  global _NUMPY_NDARRAY_TYPE
  if _NUMPY_NDARRAY_TYPE is None:
    _NUMPY_NDARRAY_TYPE = _numpy_module().ndarray
  if isinstance(obj, _NUMPY_NDARRAY_TYPE):
    return _array_interface_asarray(obj, dtype=dtype, copy=copy)
  if isinstance(obj, _DeferredArray):
    return asarray(obj._materialize(), dtype=dtype, copy=copy, device=device)
  if _has_array_interface(obj):
    return _array_interface_asarray(obj, dtype=dtype, copy=copy)
  if copy is False:
    raise ValueError(_COPY_FALSE_ERROR)
  shape, flat = _flatten(obj)
  target = _resolve_dtype(dtype) if dtype is not None else _infer_dtype(flat)
  return ndarray(
    _native.from_flat(flat, shape, target.code),
    dtype=target,
    shape=shape,
    strides=_c_strides_bytes(shape, target.itemsize),
  )


def empty(shape: int | Sequence[int], dtype: object = None, *, device: object = None) -> ndarray:
  _check_cpu_device(device)
  target = _resolve_dtype(dtype) if dtype is not None else float64
  normalized = _normalize_shape(shape)
  return ndarray(
    _native.empty(normalized, target.code),
    dtype=target,
    shape=normalized,
    strides=_c_strides_bytes(normalized, target.itemsize),
  )


def zeros(shape: int | Sequence[int], dtype: object = None, *, device: object = None) -> ndarray:
  target = _resolve_dtype(dtype) if dtype is not None else float64
  return full(shape, 0, dtype=target, device=device)


def ones(shape: int | Sequence[int], dtype: object = None, *, device: object = None) -> ndarray:
  target = _resolve_dtype(dtype) if dtype is not None else float64
  return full(shape, 1, dtype=target, device=device)


def full(shape: int | Sequence[int], fill_value: object, *, dtype: object = None, device: object = None) -> ndarray:
  _check_cpu_device(device)
  target = _resolve_dtype(dtype) if dtype is not None else _infer_dtype([fill_value])
  normalized = _normalize_shape(shape)
  return ndarray(
    _native.full(normalized, fill_value, target.code),
    dtype=target,
    shape=normalized,
    strides=_c_strides_bytes(normalized, target.itemsize),
  )


def arange(
  start: int | float,
  stop: int | float | None = None,
  step: int | float = 1,
  *,
  dtype: object = None,
  device: object = None,
) -> ndarray:
  _check_cpu_device(device)
  actual_start = 0 if stop is None else start
  actual_stop = start if stop is None else stop
  if dtype is None:
    target = float64 if any(isinstance(v, builtins.float) for v in (actual_start, actual_stop, step)) else int64
  else:
    target = _resolve_dtype(dtype)
  return ndarray(_native.arange(actual_start, actual_stop, step, target.code), dtype=target)


def linspace(
  start: int | float,
  stop: int | float,
  num: int = 50,
  *,
  dtype: object = None,
  device: object = None,
) -> ndarray:
  _check_cpu_device(device)
  target = _resolve_dtype(dtype) if dtype is not None else float64
  return ndarray(
    _native.linspace(start, stop, num, target.code),
    dtype=target,
    shape=(num,),
    strides=(target.itemsize,),
  )


def reshape(x: object, shape: int | Sequence[int]) -> ndarray:
  return asarray(x).reshape(shape)


def transpose(x: object, axes: Sequence[int] | None = None) -> ndarray:
  return asarray(x).transpose(axes)


def matrix_transpose(x: object) -> ndarray:
  return asarray(x).mT


def broadcast_to(x: object, shape: int | Sequence[int]) -> ndarray:
  arr = asarray(x)
  return ndarray(_native.broadcast_to(arr._native, _normalize_shape(shape)), base=arr)


def add(x1: object, x2: object, *, out: ndarray | None = None) -> object:
  return _binary(x1, x2, OP_ADD, out=out)


def subtract(x1: object, x2: object, *, out: ndarray | None = None) -> object:
  return _binary(x1, x2, OP_SUB, out=out)


def multiply(x1: object, x2: object, *, out: ndarray | None = None) -> object:
  return _binary(x1, x2, OP_MUL, out=out)


def divide(x1: object, x2: object, *, out: ndarray | None = None) -> object:
  return _binary(x1, x2, OP_DIV, out=out)


def sin(x: object) -> object:
  return _unary(x, UNARY_SIN)


def cos(x: object) -> object:
  return _unary(x, UNARY_COS)


def exp(x: object) -> object:
  return _unary(x, UNARY_EXP)


def log(x: object) -> object:
  return _unary(x, UNARY_LOG)


def sin_add_mul(x: object, y: object, scalar: object) -> ndarray:
  if not _is_scalar_value(scalar):
    raise NotImplementedError("sin_add_mul currently requires a Python scalar multiplier")
  lhs = _as_array_value(x)
  rhs = _as_array_value(y)
  scalar_dtype = _infer_scalar_dtype_for_array(rhs, scalar)
  lhs_array = _materialize_array(lhs)
  rhs_array = _materialize_array(rhs)
  return ndarray(_native.sin_add_mul(lhs_array._native, rhs_array._native, scalar, scalar_dtype.code))


def where(condition: object, x1: object, x2: object) -> ndarray:
  cond = asarray(condition, dtype=bool)
  lhs = asarray(x1)
  rhs = asarray(x2)
  lhs, rhs = _coerce_binary_operands(lhs, rhs, OP_ADD)
  return ndarray(_native.where(cond._native, lhs._native, rhs._native))


def sum(x: object, axis: object = None) -> object:
  return _reduce(x, axis, REDUCE_SUM)


def mean(x: object, axis: object = None) -> object:
  return _reduce(x, axis, REDUCE_MEAN)


def min(x: object, axis: object = None) -> object:  # noqa: A001
  return _reduce(x, axis, REDUCE_MIN)


def max(x: object, axis: object = None) -> object:  # noqa: A001
  return _reduce(x, axis, REDUCE_MAX)


def argmax(x: object, axis: object = None) -> object:
  return _reduce(x, axis, REDUCE_ARGMAX)


def matmul(x1: object, x2: object) -> ndarray:
  lhs = _materialize_array(_as_array_value(x1))
  rhs = _materialize_array(_as_array_value(x2))
  lhs, rhs = _coerce_binary_operands(lhs, rhs, OP_MUL)
  return ndarray(_native.matmul(lhs._native, rhs._native))


def diagonal(a: object, offset: int = 0, axis1: int = 0, axis2: int = 1) -> ndarray:
  arr = asarray(a)
  diagonal = getattr(_native, "diagonal", None)
  if diagonal is not None:
    return ndarray(diagonal(arr._native, int(offset), int(axis1), int(axis2)))
  return _diagonal_fallback(arr, int(offset), int(axis1), int(axis2))


def trace(
  a: object,
  offset: int = 0,
  axis1: int = 0,
  axis2: int = 1,
  dtype: object = None,
  out: ndarray | None = None,
) -> object:
  arr = asarray(a)
  trace = getattr(_native, "trace", None)
  if trace is not None:
    dtype_code = -1 if dtype is None else _resolve_dtype(dtype).code
    result = ndarray(trace(arr._native, int(offset), int(axis1), int(axis2), dtype_code))
    value: object = result._scalar() if result.ndim == 0 else result
  else:
    value = _trace_fallback(arr, int(offset), int(axis1), int(axis2), dtype)
  if out is not None:
    out_arr = asarray(out)
    out_arr[...] = value
    return out
  return value


def astype(x: object, dtype: object, /, *, copy: builtins.bool = True, device: object = None) -> ndarray:
  return asarray(x).astype(dtype, copy=copy, device=device)


def from_dlpack(x: object, /, *, device: object = None, copy: builtins.bool | None = None) -> ndarray:
  if device is not None and device != "cpu":
    raise NotImplementedError("monpy v1 only supports cpu arrays")
  global _NUMPY_NDARRAY_TYPE
  if _NUMPY_NDARRAY_TYPE is None:
    _NUMPY_NDARRAY_TYPE = _numpy_module().ndarray
  if isinstance(x, _NUMPY_NDARRAY_TYPE):
    return _array_interface_asarray(x, dtype=None, copy=True if copy is True else False)
  array = _numpy_module().from_dlpack(x, device=device, copy=copy)
  return _array_interface_asarray(array, dtype=None, copy=True if copy is True else False)


def __array_namespace_info__() -> object:
  class Info:
    def default_device(self) -> str:
      return "cpu"

    def devices(self) -> list[str]:
      return ["cpu"]

    def dtypes(self, *, device: object = None, kind: object = None) -> dict[str, DType]:
      _check_cpu_device(device)
      return dict(_DTYPES_BY_NAME)

    def default_dtypes(self, *, device: object = None) -> dict[str, DType]:
      _check_cpu_device(device)
      return {"integral": int64, "real floating": float64, "bool": bool}

    def capabilities(self) -> dict[str, object]:
      return {"boolean indexing": False, "data-dependent shapes": False}

  return Info()


def _binary(x1: object, x2: object, op: int, *, out: ndarray | None = None) -> object:
  if out is not None:
    if type(x1) is ndarray and type(x2) is ndarray:
      lhs_dtype = x1._dtype
      if lhs_dtype is not None and lhs_dtype is x2._dtype:
        _native.binary_into(out._native, x1._native, x2._native, op)
        return out
    lhs = _materialize_array(_as_array_value(x1))
    rhs = _materialize_array(_as_array_value(x2))
    lhs, rhs = _coerce_binary_operands(lhs, rhs, op)
    _native.binary_into(out._native, lhs._native, rhs._native, op)
    return out
  fused = _maybe_fuse_binary(x1, x2, op)
  if fused is not None:
    return fused
  if _is_array_value(x1):
    return _binary_from_array(x1, x2, op, scalar_on_left=False)
  if _is_array_value(x2):
    return _binary_from_array(x2, x1, op, scalar_on_left=True)
  lhs = asarray(x1)
  rhs = asarray(x2)
  lhs, rhs = _coerce_binary_operands(lhs, rhs, op)
  return ndarray(_native.binary(lhs._native, rhs._native, op))


def _binary_from_array(
  array: ndarray | _DeferredArray,
  other: object,
  op: int,
  *,
  scalar_on_left: builtins.bool,
) -> object:
  if isinstance(array, ndarray) and isinstance(other, ndarray):
    lhs = other if scalar_on_left else array
    rhs = array if scalar_on_left else other
    if lhs.dtype is rhs.dtype:
      return ndarray(_native.binary(lhs._native, rhs._native, op))
    lhs, rhs = _coerce_binary_operands(lhs, rhs, op)
    return ndarray(_native.binary(lhs._native, rhs._native, op))
  if _is_array_value(other):
    lhs = _materialize_array(other) if scalar_on_left else _materialize_array(array)
    rhs = _materialize_array(array) if scalar_on_left else _materialize_array(other)  # type: ignore[arg-type]
    lhs, rhs = _coerce_binary_operands(lhs, rhs, op)
    return ndarray(_native.binary(lhs._native, rhs._native, op))
  if _is_scalar_value(other):
    scalar_dtype = _infer_scalar_dtype_for_array(array, other)
    if op == OP_MUL and _can_defer_scalar_binary(array, scalar_dtype):
      return _ScalarBinaryExpression(array, other, scalar_dtype, op, scalar_on_left)
    array_value = _materialize_array(array)
    return ndarray(_native.binary_scalar(array_value._native, other, scalar_dtype.code, op, scalar_on_left))
  other_arr = asarray(other)
  if scalar_on_left:
    lhs, rhs = _coerce_binary_operands(other_arr, _materialize_array(array), op)
    return ndarray(_native.binary(lhs._native, rhs._native, op))
  lhs, rhs = _coerce_binary_operands(_materialize_array(array), other_arr, op)
  return ndarray(_native.binary(lhs._native, rhs._native, op))


def _unary(x: object, op: int) -> object:
  arr = _as_array_value(x)
  if _can_defer_unary(arr, op):
    return _UnaryExpression(arr, op)
  array_value = _materialize_array(arr)
  return ndarray(_native.unary(array_value._native, op))


def _reduce(x: object, axis: object, op: int) -> object:
  if axis is not None:
    raise NotImplementedError("axis-specific reductions are not implemented in monpy v1")
  arr = _materialize_array(_as_array_value(x))
  return _native.reduce(arr._native, op).get_scalar(0)


def _is_array_value(value: object) -> builtins.bool:
  return isinstance(value, (ndarray, _DeferredArray))


def _as_array_value(value: object) -> ndarray | _DeferredArray:
  if isinstance(value, (ndarray, _DeferredArray)):
    return value
  return asarray(value)


def _materialize_array(value: ndarray | _DeferredArray) -> ndarray:
  if isinstance(value, _DeferredArray):
    return value._materialize()
  return value


def _coerce_binary_operands(lhs: ndarray, rhs: ndarray, op: int) -> tuple[ndarray, ndarray]:
  target = _result_dtype_for_binary(lhs.dtype, rhs.dtype, op)
  if lhs.dtype != target:
    lhs = lhs.astype(target)
  if rhs.dtype != target:
    rhs = rhs.astype(target)
  return lhs, rhs


def _can_defer_unary(value: ndarray | _DeferredArray, op: int) -> builtins.bool:
  return op == UNARY_SIN and value.dtype in (float32, float64)


def _can_defer_scalar_binary(value: ndarray | _DeferredArray, scalar_dtype: DType) -> builtins.bool:
  return value.dtype in (float32, float64) and scalar_dtype in (float32, float64)


def _maybe_fuse_binary(x1: object, x2: object, op: int) -> ndarray | None:
  if op != OP_ADD:
    return None
  fused = _match_sin_add_mul(x1, x2)
  if fused is not None:
    return fused
  return _match_sin_add_mul(x2, x1)


def _match_sin_add_mul(x1: object, x2: object) -> ndarray | None:
  if not isinstance(x1, _UnaryExpression) or x1._op != UNARY_SIN:
    return None
  if not isinstance(x2, _ScalarBinaryExpression) or x2._op != OP_MUL:
    return None
  lhs = _materialize_array(x1._base)
  rhs = _materialize_array(x2._array)
  return ndarray(_native.sin_add_mul(lhs._native, rhs._native, x2._scalar, x2._scalar_dtype.code))


def _result_dtype_for_unary(dtype_value: DType) -> DType:
  return _UNARY_RESULT_DTYPES[dtype_value]


def _result_dtype_for_binary(lhs_dtype: DType, rhs_dtype: DType, op: int) -> DType:
  return _BINARY_RESULT_DTYPES[op][(lhs_dtype, rhs_dtype)]


def _diagonal_fallback(arr: ndarray, offset: int, axis1: int, axis2: int) -> ndarray:
  if arr.ndim < 2:
    raise ValueError("diag requires an array of at least two dimensions")
  axis1 = _normalize_axis(axis1, arr.ndim)
  axis2 = _normalize_axis(axis2, arr.ndim)
  if axis1 == axis2:
    raise ValueError("axis1 and axis2 cannot be the same")
  row_start = builtins.max(-offset, 0)
  col_start = builtins.max(offset, 0)
  diag_len = builtins.max(0, builtins.min(arr.shape[axis1] - row_start, arr.shape[axis2] - col_start))
  remaining_axes = tuple(axis for axis in range(arr.ndim) if axis not in (axis1, axis2))
  remaining_shape = tuple(arr.shape[axis] for axis in remaining_axes)
  out_shape = remaining_shape + (diag_len,)
  flat: list[object] = []
  for prefix in _iter_indices(remaining_shape):
    key: list[object] = [0] * arr.ndim
    for axis, index in zip(remaining_axes, prefix, strict=True):
      key[axis] = index
    for diag_index in range(diag_len):
      key[axis1] = row_start + diag_index
      key[axis2] = col_start + diag_index
      flat.append(arr[tuple(key)])
  return ndarray(
    _native.from_flat(flat, out_shape, arr.dtype.code),
    dtype=arr.dtype,
    shape=out_shape,
    strides=_c_strides_bytes(out_shape, arr.itemsize),
  )


def _trace_fallback(arr: ndarray, offset: int, axis1: int, axis2: int, dtype_value: object) -> object:
  diag = diagonal(arr, offset=offset, axis1=axis1, axis2=axis2)
  target = _resolve_dtype(dtype_value) if dtype_value is not None else _trace_result_dtype(diag.dtype)
  if diag.dtype != target:
    diag = diag.astype(target)
  if diag.ndim == 1:
    return sum(diag)
  out_shape = diag.shape[:-1]
  flat: list[object] = []
  for prefix in _iter_indices(out_shape):
    total: object = 0.0 if target in (float32, float64) else 0
    for diag_index in range(diag.shape[-1]):
      total += diag[prefix + (diag_index,)]  # type: ignore[operator]
    flat.append(total)
  return ndarray(
    _native.from_flat(flat, out_shape, target.code),
    dtype=target,
    shape=out_shape,
    strides=_c_strides_bytes(out_shape, target.itemsize),
  )


def _trace_result_dtype(dtype_value: DType) -> DType:
  if dtype_value == bool:
    return int64
  return dtype_value


def _broadcast_shapes(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...]:
  out_len = builtins.max(len(lhs), len(rhs))
  out = [1] * out_len
  for out_axis in range(out_len - 1, -1, -1):
    lhs_axis = out_axis - (out_len - len(lhs))
    rhs_axis = out_axis - (out_len - len(rhs))
    lhs_dim = lhs[lhs_axis] if lhs_axis >= 0 else 1
    rhs_dim = rhs[rhs_axis] if rhs_axis >= 0 else 1
    if lhs_dim == rhs_dim:
      out[out_axis] = lhs_dim
    elif lhs_dim == 1:
      out[out_axis] = rhs_dim
    elif rhs_dim == 1:
      out[out_axis] = lhs_dim
    else:
      raise ValueError("operands could not be broadcast together")
  return tuple(out)


def _is_scalar_value(value: object) -> builtins.bool:
  return isinstance(value, (builtins.bool, builtins.int, builtins.float))


def _infer_scalar_dtype(value: object) -> DType:
  if isinstance(value, builtins.bool):
    return bool
  if isinstance(value, builtins.int):
    return int64
  return float64


def _infer_scalar_dtype_for_array(array: ndarray | _DeferredArray, value: object) -> DType:
  array_dtype = array.dtype
  if array_dtype in (float32, float64) and isinstance(value, (builtins.int, builtins.float)):
    return array_dtype
  if array_dtype == int64 and isinstance(value, (builtins.bool, builtins.int)):
    return int64
  return _infer_scalar_dtype(value)


def _resolve_dtype(value: object) -> DType:
  if value is None:
    return float64
  if isinstance(value, DType):
    return value
  if isinstance(value, str):
    try:
      return _DTYPES_BY_NAME[value]
    except KeyError as exc:
      raise NotImplementedError(f"unsupported dtype: {value}") from exc
  if value is builtins.bool:
    return bool
  if value is builtins.int:
    return int64
  if value is builtins.float:
    return float64
  numpy_dtype = _numpy_dtype_from_object(value)
  if numpy_dtype is not None:
    return _dtype_from_numpy_dtype(numpy_dtype)
  raise NotImplementedError(f"unsupported dtype: {value!r}")


def _numpy_module() -> object:
  try:
    import numpy as numpy_oracle
  except ModuleNotFoundError as exc:
    raise NotImplementedError("numpy is required for array interface and dlpack interop") from exc
  return numpy_oracle


def _numpy_dtype_from_object(value: object) -> object | None:
  try:
    numpy_oracle = _numpy_module()
  except NotImplementedError:
    return None
  if isinstance(value, numpy_oracle.dtype):
    return value
  try:
    if isinstance(value, type) and issubclass(value, numpy_oracle.generic):
      return numpy_oracle.dtype(value)
  except TypeError:
    return None
  return None


def _numpy_dtype_for(dtype_value: DType) -> object:
  numpy_oracle = _numpy_module()
  return numpy_oracle.dtype(dtype_value.typestr)


def _dtype_from_numpy_dtype(value: object) -> DType:
  numpy_oracle = _numpy_module()
  numpy_dtype = numpy_oracle.dtype(value)
  if numpy_dtype.fields is not None or numpy_dtype.subdtype is not None:
    raise NotImplementedError(f"unsupported dtype: {numpy_dtype}")
  if not numpy_dtype.isnative:
    raise NotImplementedError(f"unsupported dtype: {numpy_dtype}")
  try:
    return _DTYPES_BY_NUMPY_KIND[(numpy_dtype.kind, numpy_dtype.itemsize)]
  except KeyError as exc:
    raise NotImplementedError(f"unsupported dtype: {numpy_dtype}") from exc


def _has_array_interface(obj: object) -> builtins.bool:
  try:
    interface = obj.__array_interface__
  except Exception:
    return False
  return isinstance(interface, dict)


def _array_interface_asarray(
  obj: object,
  *,
  dtype: object,
  copy: builtins.bool | None,
) -> ndarray:
  numpy_oracle = _numpy_module()
  global _NUMPY_NDARRAY_TYPE
  if _NUMPY_NDARRAY_TYPE is None:
    _NUMPY_NDARRAY_TYPE = numpy_oracle.ndarray
  if isinstance(obj, _NUMPY_NDARRAY_TYPE):
    array = obj
  else:
    array = numpy_oracle.asarray(obj)
  iface = array.__array_interface__
  source_dtype = _DTYPES_BY_TYPESTR.get(iface["typestr"])
  if source_dtype is None:
    source_dtype = _dtype_from_numpy_dtype(array.dtype)
  target = _resolve_dtype(dtype) if dtype is not None else None
  if target is not None and target != source_dtype:
    if copy is False:
      raise ValueError(_COPY_FALSE_ERROR)
    converted = array.astype(_numpy_dtype_for(target), copy=True)
    return _copy_from_numpy_array(converted)
  data_address, readonly = iface["data"]
  if copy is True:
    return _copy_from_numpy_array(array, dtype_value=source_dtype, iface=iface)
  if readonly:
    if copy is False:
      raise ValueError("readonly array requires copy=True")
    return _copy_from_numpy_array(array, dtype_value=source_dtype, iface=iface)
  return _external_from_numpy_array(array, source_dtype, iface=iface, data_address=data_address)


def _external_from_numpy_array(
  array: object,
  dtype_value: DType,
  *,
  iface: dict[str, object] | None = None,
  data_address: int | None = None,
) -> ndarray:
  if iface is None:
    iface = array.__array_interface__
  shape = iface["shape"]
  raw_strides = iface["strides"]
  itemsize = dtype_value.itemsize
  if raw_strides is None:
    byte_strides = _c_strides_bytes(shape, itemsize)
  else:
    byte_strides = raw_strides
    for stride in byte_strides:
      if stride % itemsize != 0:
        raise NotImplementedError("array interface strides must align to dtype itemsize")
  element_strides = tuple(stride // itemsize for stride in byte_strides)
  if data_address is None:
    data_address = iface["data"][0]
  byte_len = math.prod(shape) * itemsize
  native = _native.from_external(data_address, shape, element_strides, dtype_value.code, byte_len)
  return ndarray(native, dtype=dtype_value, shape=shape, strides=byte_strides, owner=array)


def _copy_from_numpy_array(
  array: object,
  *,
  dtype_value: DType | None = None,
  iface: dict[str, object] | None = None,
) -> ndarray:
  if iface is None:
    iface = array.__array_interface__
  if dtype_value is None:
    dtype_value = _DTYPES_BY_TYPESTR.get(iface["typestr"])
    if dtype_value is None:
      dtype_value = _dtype_from_numpy_dtype(array.dtype)
  shape = iface["shape"]
  raw_strides = iface["strides"]
  itemsize = dtype_value.itemsize
  if raw_strides is None:
    element_strides = _c_strides_elements(shape)
  else:
    for stride in raw_strides:
      if stride % itemsize != 0:
        raise NotImplementedError("array interface strides must align to dtype itemsize")
    element_strides = tuple(stride // itemsize for stride in raw_strides)
  data_address = iface["data"][0]
  byte_len = math.prod(shape) * itemsize
  return ndarray(
    _native.copy_from_external(data_address, shape, element_strides, dtype_value.code, byte_len),
    dtype=dtype_value,
    shape=shape,
    strides=_c_strides_bytes(shape, itemsize),
  )


def _c_strides_elements(shape: tuple[int, ...]) -> tuple[int, ...]:
  strides = [1] * len(shape)
  s = 1
  for axis in range(len(shape) - 1, -1, -1):
    strides[axis] = s
    s *= shape[axis]
  return tuple(strides)


def _infer_dtype(flat: Sequence[object]) -> DType:
  if not flat:
    return float64
  has_float = False
  has_int = False
  has_bool = True
  for value in flat:
    if isinstance(value, builtins.bool):
      continue
    has_bool = False
    if isinstance(value, builtins.int):
      has_int = True
      continue
    if isinstance(value, builtins.float):
      has_float = True
      continue
    raise NotImplementedError(f"unsupported array value type: {type(value).__name__}")
  if has_float:
    return float64
  if has_int or not has_bool:
    return int64
  return bool


def _normalize_shape(shape: int | Sequence[int]) -> tuple[int, ...]:
  if isinstance(shape, builtins.int):
    if shape < 0:
      raise ValueError("negative dimensions are not allowed")
    return (shape,)
  out = tuple(int(dim) for dim in shape)
  if any(dim < 0 for dim in out):
    raise ValueError("negative dimensions are not allowed")
  return out


def _c_strides_bytes(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
  strides = [0] * len(shape)
  stride = itemsize
  for axis in range(len(shape) - 1, -1, -1):
    strides[axis] = stride
    stride *= shape[axis]
  return tuple(strides)


def _is_c_contiguous_bytes(shape: tuple[int, ...], strides: tuple[int, ...], itemsize: int) -> builtins.bool:
  expected = itemsize
  for axis in range(len(shape) - 1, -1, -1):
    if shape[axis] == 0:
      return True
    if shape[axis] != 1 and strides[axis] != expected:
      return False
    expected *= shape[axis]
  return True


def _iter_indices(shape: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
  if not shape:
    yield ()
    return
  yield from itertools.product(*(range(dim) for dim in shape))


def _shape_from_args(shape: Sequence[int | Sequence[int]]) -> tuple[int, ...]:
  if len(shape) == 1 and not isinstance(shape[0], builtins.int):
    return _normalize_shape(shape[0])
  return _normalize_shape(shape)  # type: ignore[arg-type]


def _flatten(obj: object) -> tuple[tuple[int, ...], list[object]]:
  if isinstance(obj, (list, tuple)):
    if not obj:
      return (0,), []
    child_shapes: list[tuple[int, ...]] = []
    flat: list[object] = []
    for item in obj:
      child_shape, child_flat = _flatten(item)
      child_shapes.append(child_shape)
      flat.extend(child_flat)
    first_shape = child_shapes[0]
    if any(shape != first_shape for shape in child_shapes):
      raise ValueError("cannot create monpy array from ragged nested sequences")
    return (len(obj),) + first_shape, flat
  if isinstance(obj, ndarray):
    return obj.shape, [obj._native.get_scalar(i) for i in range(obj.size)]
  if isinstance(obj, (builtins.bool, builtins.int, builtins.float)):
    return (), [obj]
  raise NotImplementedError(f"unsupported array input type: {type(obj).__name__}")


def _unflatten(flat: Sequence[object], shape: tuple[int, ...]) -> object:
  if not shape:
    return flat[0]
  if len(shape) == 1:
    return list(flat[: shape[0]])
  step = math.prod(shape[1:])
  return [_unflatten(flat[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]


def _expand_key(key: object, ndim: int) -> tuple[object, ...]:
  if key == ():
    if ndim != 0:
      raise IndexError("empty index is only valid for zero-dimensional arrays")
    return ()
  parts = key if isinstance(key, tuple) else (key,)
  if any(part is None for part in parts):
    raise NotImplementedError("newaxis indexing is not implemented in monpy v1")
  if parts.count(Ellipsis) > 1:
    raise IndexError("an index can only have a single ellipsis")
  if Ellipsis in parts:
    ellipsis_at = parts.index(Ellipsis)
    missing = ndim - (len(parts) - 1)
    parts = parts[:ellipsis_at] + (slice(None),) * missing + parts[ellipsis_at + 1 :]
  if len(parts) > ndim:
    raise IndexError("too many indices for array")
  return parts + (slice(None),) * (ndim - len(parts))


def _normalize_index(index: object, dim: int) -> int:
  if not isinstance(index, builtins.int):
    raise NotImplementedError("monpy v1 supports only integer and slice indexing")
  if index < 0:
    index += dim
  if index < 0 or index >= dim:
    raise IndexError("index out of bounds")
  return index


def _normalize_axis(axis: int, ndim: int) -> int:
  if axis < 0:
    axis += ndim
  if axis < 0 or axis >= ndim:
    raise ValueError("axis out of bounds")
  return axis


def _normalize_axes(axes: Sequence[int], ndim: int) -> tuple[int, ...]:
  normalized = tuple(axis + ndim if axis < 0 else axis for axis in axes)
  if sorted(normalized) != list(range(ndim)):
    raise ValueError("axes must be a permutation of dimensions")
  return normalized


def _check_cpu_device(device: object) -> None:
  if device not in (None, "cpu"):
    raise NotImplementedError("monpy v1 only supports cpu arrays")


linalg = importlib.import_module(f"{__name__}.linalg")


__all__ = [
  "DType",
  "add",
  "arange",
  "array",
  "asarray",
  "argmax",
  "astype",
  "bool",
  "broadcast_to",
  "cos",
  "diagonal",
  "divide",
  "dtype",
  "e",
  "empty",
  "exp",
  "float32",
  "float64",
  "from_dlpack",
  "full",
  "inf",
  "int64",
  "linalg",
  "linspace",
  "log",
  "matmul",
  "matrix_transpose",
  "max",
  "mean",
  "min",
  "multiply",
  "nan",
  "ndarray",
  "newaxis",
  "ones",
  "pi",
  "reshape",
  "sin",
  "sin_add_mul",
  "subtract",
  "sum",
  "trace",
  "transpose",
  "where",
  "zeros",
]
