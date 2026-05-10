"""Staged implementations for top-level monpy operations."""

from __future__ import annotations

from collections.abc import Sequence

from ..core import TensorSpec
from .tensor import Tensor


def is_kernel_tensor(value: object) -> bool:
  return getattr(value, "__monpy_kernel_tensor__", False)


def ufunc(name: str, *args: object) -> Tensor:
  if len(args) == 1:
    return _one_tensor(args)._trace.unary(name, args[0])
  if len(args) == 2:
    return _one_tensor(args)._trace.binary(name, args[0], args[1])
  raise TypeError(f"monpy kernel ufunc {name!r} only supports unary and binary calls")


def matmul(lhs: object, rhs: object) -> Tensor:
  return _one_tensor((lhs, rhs))._trace.matmul(lhs, rhs)


def where(condition: object, x: object, y: object) -> Tensor:
  return _one_tensor((condition, x, y))._trace.where(condition, x, y)


def reduce(
  x: object,
  axis: object,
  reduce_op: int,
  *,
  dtype: object = None,
  keepdims: bool = False,
  result_dtype: object = None,
) -> Tensor:
  return _expect_tensor(x)._trace.reduce(
    x,
    axis,
    reduce_op,
    dtype=dtype,
    keepdims=keepdims,
    result_dtype=result_dtype,
  )


def reshape(x: object, shape: int | Sequence[int]) -> Tensor:
  tensor = _expect_tensor(x)
  target = (shape,) if isinstance(shape, int) else tuple(int(dim) for dim in shape)
  return tensor._trace.reshape(tensor, target)


def transpose(x: object, axes: Sequence[int] | None = None) -> Tensor:
  tensor = _expect_tensor(x)
  if axes is None:
    axes = tuple(range(len(tensor.shape) - 1, -1, -1))
  return tensor._trace.transpose(tensor, tuple(int(axis) for axis in axes))


def broadcast_to(x: object, shape: int | Sequence[int]) -> Tensor:
  tensor = _expect_tensor(x)
  target = (shape,) if isinstance(shape, int) else tuple(int(dim) for dim in shape)
  return tensor._trace.broadcast_to(tensor, target)


def cast(x: object, dtype: object) -> Tensor:
  tensor = _expect_tensor(x)
  return tensor._trace.cast(tensor, dtype)


def custom_call(name: str, args: Sequence[object], out: TensorSpec | None = None) -> Tensor:
  tensor = _one_tensor(tuple(args))
  return tensor._trace.custom_call(name, args, tensor.spec if out is None else out)


def _expect_tensor(value: object) -> Tensor:
  if isinstance(value, Tensor):
    return value
  raise TypeError(f"expected traced monpy.Tensor, got {type(value).__name__}")


def _one_tensor(values: tuple[object, ...]) -> Tensor:
  for value in values:
    if isinstance(value, Tensor):
      return value
  raise TypeError("expected at least one traced monpy.Tensor")
