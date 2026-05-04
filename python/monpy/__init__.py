from __future__ import annotations

import builtins
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

newaxis = None
nan = math.nan
inf = math.inf
pi = math.pi
e = math.e


class ndarray:
  __array_priority__ = 1000

  def __init__(_self, native: _native.NativeArray, base: ndarray | None = None) -> None:
    _self._native = native
    _self._base = base

  @property
  def dtype(self) -> DType:
    return _DTYPES_BY_CODE[int(self._native.dtype_code())]

  @property
  def shape(self) -> tuple[int, ...]:
    return tuple(int(self._native.shape_at(axis)) for axis in range(self.ndim))

  @property
  def ndim(self) -> int:
    return int(self._native.ndim())

  @property
  def size(self) -> int:
    return int(self._native.size())

  @property
  def itemsize(self) -> int:
    return int(self._native.item_size())

  @property
  def strides(self) -> tuple[int, ...]:
    return tuple(int(self._native.stride_at(axis)) * self.itemsize for axis in range(self.ndim))

  @property
  def device(self) -> str:
    return "cpu"

  @property
  def T(self) -> ndarray:
    return self.transpose()

  @property
  def mT(self) -> ndarray:
    if self.ndim < 2:
      raise ValueError("matrix transpose requires at least two dimensions")
    axes = tuple(range(self.ndim - 2)) + (self.ndim - 1, self.ndim - 2)
    return self.transpose(axes)

  @property
  def __array_interface__(self) -> dict[str, object]:
    strides = None if self._native.is_c_contiguous() else self.strides
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

  def __dlpack__(self, *, stream: object = None, max_version: tuple[int, int] | None = None) -> object:
    raise BufferError(
      "monpy dlpack export is not implemented yet; cpu array memory is exposed via __array_interface__"
    )

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

  def __add__(self, other: object) -> ndarray:
    return add(self, other)

  def __radd__(self, other: object) -> ndarray:
    return add(other, self)

  def __sub__(self, other: object) -> ndarray:
    return subtract(self, other)

  def __rsub__(self, other: object) -> ndarray:
    return subtract(other, self)

  def __mul__(self, other: object) -> ndarray:
    return multiply(self, other)

  def __rmul__(self, other: object) -> ndarray:
    return multiply(other, self)

  def __truediv__(self, other: object) -> ndarray:
    return divide(self, other)

  def __rtruediv__(self, other: object) -> ndarray:
    return divide(other, self)

  def __matmul__(self, other: object) -> ndarray:
    return matmul(self, other)

  def __rmatmul__(self, other: object) -> ndarray:
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
    steps: list[int] = []
    drops: list[int] = []
    for axis, part in enumerate(parts):
      dim = self.shape[axis]
      if isinstance(part, slice):
        start, _stop, step = part.indices(dim)
        starts.append(start)
        steps.append(step)
        drops.append(0)
      else:
        index = _normalize_index(part, dim)
        starts.append(index)
        steps.append(1)
        drops.append(1)
    return ndarray(_native.slice(self._native, tuple(starts), tuple(steps), tuple(drops)), base=self)


def dtype(value: object) -> DType:
  return _resolve_dtype(value)


def array(obj: object, dtype: object = None, *, copy: builtins.bool | None = True, device: object = None) -> ndarray:
  return asarray(obj, dtype=dtype, copy=copy, device=device)


def asarray(obj: object, dtype: object = None, *, copy: builtins.bool | None = None, device: object = None) -> ndarray:
  _check_cpu_device(device)
  if isinstance(obj, ndarray):
    if dtype is None:
      if copy is True:
        return obj.astype(obj.dtype, copy=True)
      return obj
    target = _resolve_dtype(dtype)
    if target == obj.dtype and copy is not True:
      return obj
    return obj.astype(target, copy=True)
  shape, flat = _flatten(obj)
  target = _resolve_dtype(dtype) if dtype is not None else _infer_dtype(flat)
  return ndarray(_native.from_flat(flat, shape, target.code))


def empty(shape: int | Sequence[int], dtype: object = None, *, device: object = None) -> ndarray:
  _check_cpu_device(device)
  target = _resolve_dtype(dtype) if dtype is not None else float64
  return ndarray(_native.empty(_normalize_shape(shape), target.code))


def zeros(shape: int | Sequence[int], dtype: object = None, *, device: object = None) -> ndarray:
  target = _resolve_dtype(dtype) if dtype is not None else float64
  return full(shape, 0, dtype=target, device=device)


def ones(shape: int | Sequence[int], dtype: object = None, *, device: object = None) -> ndarray:
  target = _resolve_dtype(dtype) if dtype is not None else float64
  return full(shape, 1, dtype=target, device=device)


def full(shape: int | Sequence[int], fill_value: object, *, dtype: object = None, device: object = None) -> ndarray:
  _check_cpu_device(device)
  target = _resolve_dtype(dtype) if dtype is not None else _infer_dtype([fill_value])
  return ndarray(_native.full(_normalize_shape(shape), fill_value, target.code))


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
  return ndarray(_native.arange(actual_start, actual_stop, step, target.code))


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
  return ndarray(_native.linspace(start, stop, num, target.code))


def reshape(x: object, shape: int | Sequence[int]) -> ndarray:
  return asarray(x).reshape(shape)


def transpose(x: object, axes: Sequence[int] | None = None) -> ndarray:
  return asarray(x).transpose(axes)


def matrix_transpose(x: object) -> ndarray:
  return asarray(x).mT


def broadcast_to(x: object, shape: int | Sequence[int]) -> ndarray:
  arr = asarray(x)
  return ndarray(_native.broadcast_to(arr._native, _normalize_shape(shape)), base=arr)


def add(x1: object, x2: object) -> ndarray:
  return _binary(x1, x2, OP_ADD)


def subtract(x1: object, x2: object) -> ndarray:
  return _binary(x1, x2, OP_SUB)


def multiply(x1: object, x2: object) -> ndarray:
  return _binary(x1, x2, OP_MUL)


def divide(x1: object, x2: object) -> ndarray:
  return _binary(x1, x2, OP_DIV)


def sin(x: object) -> ndarray:
  return _unary(x, UNARY_SIN)


def cos(x: object) -> ndarray:
  return _unary(x, UNARY_COS)


def exp(x: object) -> ndarray:
  return _unary(x, UNARY_EXP)


def log(x: object) -> ndarray:
  return _unary(x, UNARY_LOG)


def where(condition: object, x1: object, x2: object) -> ndarray:
  cond = asarray(condition, dtype=bool)
  lhs = asarray(x1)
  rhs = asarray(x2)
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
  lhs = asarray(x1)
  rhs = asarray(x2)
  return ndarray(_native.matmul(lhs._native, rhs._native))


def astype(x: object, dtype: object, /, *, copy: builtins.bool = True, device: object = None) -> ndarray:
  return asarray(x).astype(dtype, copy=copy, device=device)


def from_dlpack(x: object, /, *, device: object = None, copy: builtins.bool | None = None) -> ndarray:
  raise BufferError("monpy cannot consume dlpack capsules yet")


def layout_smoke() -> ndarray:
  return ndarray(_native.layout_smoke())


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


def _binary(x1: object, x2: object, op: int) -> ndarray:
  lhs = asarray(x1)
  rhs = asarray(x2)
  return ndarray(_native.binary(lhs._native, rhs._native, op))


def _unary(x: object, op: int) -> ndarray:
  arr = asarray(x)
  return ndarray(_native.unary(arr._native, op))


def _reduce(x: object, axis: object, op: int) -> object:
  if axis is not None:
    raise NotImplementedError("axis-specific reductions are not implemented in monpy v1")
  result = ndarray(_native.reduce(asarray(x)._native, op))
  return result._scalar()


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
  raise NotImplementedError(f"unsupported dtype: {value!r}")


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


def _normalize_axes(axes: Sequence[int], ndim: int) -> tuple[int, ...]:
  normalized = tuple(axis + ndim if axis < 0 else axis for axis in axes)
  if sorted(normalized) != list(range(ndim)):
    raise ValueError("axes must be a permutation of dimensions")
  return normalized


def _check_cpu_device(device: object) -> None:
  if device not in (None, "cpu"):
    raise NotImplementedError("monpy v1 only supports cpu arrays")


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
  "layout_smoke",
  "linspace",
  "log",
  "matmul",
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
  "subtract",
  "sum",
  "transpose",
  "where",
  "zeros",
]
