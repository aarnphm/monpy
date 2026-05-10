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
  get_lowering,
  get_primitive,
  input_p,
  matmul_p,
  mul_p,
  register_lowering,
  register_primitive,
  reshape_p,
  sub_p,
  transpose_p,
)
from monpy._src.dtypes import DTypeKind, DTypeSpec, StorageKind, dtype_spec
from monpy._src.lax.primitives import broadcast_to, cast, custom_call, matmul, reshape, transpose, ufunc
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
  "get_lowering",
  "get_primitive",
  "input_p",
  "jit",
  "matmul",
  "matmul_p",
  "mul",
  "mul_p",
  "multiply",
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
]
