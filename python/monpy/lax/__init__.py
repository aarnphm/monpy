"""Primitive-level public API for monpy staging and graph work."""

from __future__ import annotations

from monpy._src.api import CompiledFunction, JittedFunction, VmapFunction, jit, vmap
from monpy._src.core import (
  PRIMITIVES,
  DeviceSpec,
  GraphIR,
  Node,
  Primitive,
  SymbolicDim,
  TensorSpec,
  add_p,
  broadcast_to_p,
  cast_p,
  constant_p,
  custom_call_p,
  div_p,
  equal_p,
  get_lowering,
  get_primitive,
  greater_equal_p,
  greater_p,
  input_p,
  less_equal_p,
  less_p,
  matmul_p,
  mul_p,
  not_equal_p,
  reduce_p,
  register_lowering,
  register_primitive,
  reshape_p,
  sub_p,
  transpose_p,
  where_p,
)
from monpy._src.dtypes import DTypeKind, DTypeSpec, StorageKind, dtype_spec
from monpy._src.lax.primitives import broadcast_to, cast, custom_call, matmul, reduce, reshape, transpose, ufunc, where
from monpy._src.lax.tensor import Tensor
from monpy._src.layout import Contiguity, LayoutOrder, LayoutSpec, TileSpec


def add(lhs: object, rhs: object) -> Tensor:
  return ufunc("add", lhs, rhs)


def sub(lhs: object, rhs: object) -> Tensor:
  return ufunc("sub", lhs, rhs)


def subtract(lhs: object, rhs: object) -> Tensor:
  return sub(lhs, rhs)


def mul(lhs: object, rhs: object) -> Tensor:
  return ufunc("mul", lhs, rhs)


def multiply(lhs: object, rhs: object) -> Tensor:
  return mul(lhs, rhs)


def div(lhs: object, rhs: object) -> Tensor:
  return ufunc("div", lhs, rhs)


def divide(lhs: object, rhs: object) -> Tensor:
  return div(lhs, rhs)


def equal(lhs: object, rhs: object) -> Tensor:
  return ufunc("equal", lhs, rhs)


def not_equal(lhs: object, rhs: object) -> Tensor:
  return ufunc("not_equal", lhs, rhs)


def less(lhs: object, rhs: object) -> Tensor:
  return ufunc("less", lhs, rhs)


def less_equal(lhs: object, rhs: object) -> Tensor:
  return ufunc("less_equal", lhs, rhs)


def greater(lhs: object, rhs: object) -> Tensor:
  return ufunc("greater", lhs, rhs)


def greater_equal(lhs: object, rhs: object) -> Tensor:
  return ufunc("greater_equal", lhs, rhs)


__all__ = [
  "CompiledFunction",
  "Contiguity",
  "DTypeKind",
  "DTypeSpec",
  "DeviceSpec",
  "GraphIR",
  "JittedFunction",
  "LayoutOrder",
  "LayoutSpec",
  "Node",
  "PRIMITIVES",
  "Primitive",
  "StorageKind",
  "SymbolicDim",
  "Tensor",
  "TensorSpec",
  "TileSpec",
  "VmapFunction",
  "add",
  "add_p",
  "broadcast_to",
  "broadcast_to_p",
  "cast",
  "cast_p",
  "constant_p",
  "custom_call",
  "custom_call_p",
  "div",
  "div_p",
  "divide",
  "dtype_spec",
  "equal",
  "equal_p",
  "get_lowering",
  "get_primitive",
  "greater",
  "greater_equal",
  "greater_equal_p",
  "greater_p",
  "input_p",
  "jit",
  "less",
  "less_equal",
  "less_equal_p",
  "less_p",
  "matmul",
  "matmul_p",
  "mul",
  "mul_p",
  "multiply",
  "not_equal",
  "not_equal_p",
  "reduce",
  "reduce_p",
  "register_lowering",
  "register_primitive",
  "reshape",
  "reshape_p",
  "sub",
  "sub_p",
  "subtract",
  "transpose",
  "transpose_p",
  "ufunc",
  "vmap",
  "where",
  "where_p",
]
