"""Symbolic execution for `monpy.lax.jit`."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, cast

from ..core import (
  GraphIR,
  Node,
  Primitive,
  SymbolicDim,
  TensorSpec,
  ValueRef,
  broadcast_to_p,
  cast_p,
  constant_p,
  custom_call_p,
  get_primitive,
  input_p,
  matmul_p,
  reduce_p,
  reshape_p,
  transpose_p,
  where_p,
)
from ..dtypes import BOOL_DTYPE, from_monpy_dtype
from ..lax.tensor import Tensor
from ..layout import LayoutSpec
from ..tree_util import PyTreeDef, tree_flatten


@dataclass(slots=True)
class TraceContext:
  nodes: list[Node] = field(default_factory=list)
  inputs: list[ValueRef] = field(default_factory=list)

  def input(self, spec: TensorSpec) -> Tensor:
    node_ref = self._append(input_p, (), {}, spec)
    self.inputs.append(node_ref)
    return Tensor(node_ref, spec, self)

  def constant(self, value: object, like: Tensor) -> Tensor:
    if not _is_scalar_constant(value):
      raise NotImplementedError("staged non-scalar constants require explicit TensorSpec inputs or external weights")
    spec = TensorSpec((), like.spec.dtype, like.spec.device)
    return Tensor(self._append(constant_p, (), {"value": repr(value)}, spec), spec, self)

  def binary(self, op_name: str, lhs: object, rhs: object) -> Tensor:
    lhs_t, rhs_t = self._ensure_pair(lhs, rhs)
    primitive = get_primitive(op_name)
    shape = _broadcast_shape(lhs_t.spec.shape, rhs_t.spec.shape)
    layout = LayoutSpec.row_major(tuple(_static_dim(dim) for dim in shape))
    dtype = BOOL_DTYPE if primitive.ufunc_kind == "compare" else lhs_t.spec.dtype
    spec = TensorSpec(shape, dtype, lhs_t.spec.device, layout)
    return Tensor(self._append(primitive, (lhs_t.node, rhs_t.node), {}, spec), spec, self)

  def unary(self, op_name: str, x: object) -> Tensor:
    tensor = self.ensure_tensor(x)
    spec = TensorSpec(tensor.spec.shape, tensor.spec.dtype, tensor.spec.device, tensor.spec.layout)
    return Tensor(self._append(custom_call_p, (tensor.node,), {"name": f"monpy.{op_name}"}, spec), spec, self)

  def matmul(self, lhs: object, rhs: object) -> Tensor:
    lhs_t, rhs_t = self._ensure_pair(lhs, rhs)
    out_shape = _matmul_shape(lhs_t.spec.shape, rhs_t.spec.shape)
    spec = TensorSpec(out_shape, lhs_t.spec.dtype, lhs_t.spec.device)
    return Tensor(self._append(matmul_p, (lhs_t.node, rhs_t.node), {}, spec), spec, self)

  def where(self, condition: object, lhs: object, rhs: object) -> Tensor:
    condition_t = self.ensure_tensor(condition)
    lhs_t, rhs_t = self._ensure_pair(lhs, rhs)
    branch_shape = _broadcast_shape(lhs_t.spec.shape, rhs_t.spec.shape)
    out_shape = _broadcast_shape(condition_t.spec.shape, branch_shape)
    layout = LayoutSpec.row_major(tuple(_static_dim(dim) for dim in out_shape))
    spec = TensorSpec(out_shape, lhs_t.spec.dtype, lhs_t.spec.device, layout)
    return Tensor(self._append(where_p, (condition_t.node, lhs_t.node, rhs_t.node), {}, spec), spec, self)

  def reduce(
    self,
    x: object,
    axis: object,
    reduce_op: int,
    *,
    dtype: object = None,
    keepdims: bool = False,
    result_dtype: object = None,
  ) -> Tensor:
    tensor = self.ensure_tensor(x)
    source = self.cast(tensor, dtype) if dtype is not None else tensor
    axes = _normalize_reduce_axes(axis, len(source.spec.shape))
    out_shape = _reduce_shape(source.spec.shape, axes, keepdims)
    out_dtype = from_monpy_dtype(result_dtype) if result_dtype is not None else source.spec.dtype
    layout = LayoutSpec.row_major(tuple(_static_dim(dim) for dim in out_shape))
    spec = TensorSpec(out_shape, out_dtype, source.spec.device, layout)
    attrs: dict[str, object] = {"axes": axes, "keepdims": keepdims, "reduce_op": reduce_op}
    return Tensor(self._append(reduce_p, (source.node,), attrs, spec), spec, self)

  def reshape(self, tensor: Tensor, shape: tuple[int, ...]) -> Tensor:
    layout = tensor.spec.layout.reshape(shape)
    spec = TensorSpec(tuple(shape), tensor.spec.dtype, tensor.spec.device, layout)
    return Tensor(self._append(reshape_p, (tensor.node,), {"shape": shape}, spec), spec, self)

  def transpose(self, tensor: Tensor, axes: tuple[int, ...]) -> Tensor:
    layout = tensor.spec.layout.permute(axes)
    spec = TensorSpec(tuple(tensor.spec.shape[axis] for axis in axes), tensor.spec.dtype, tensor.spec.device, layout)
    return Tensor(self._append(transpose_p, (tensor.node,), {"axes": axes}, spec), spec, self)

  def broadcast_to(self, tensor: Tensor, shape: tuple[int, ...]) -> Tensor:
    layout = tensor.spec.layout.broadcast_to(shape)
    spec = TensorSpec(tuple(shape), tensor.spec.dtype, tensor.spec.device, layout)
    return Tensor(self._append(broadcast_to_p, (tensor.node,), {"shape": shape}, spec), spec, self)

  def cast(self, tensor: Tensor, dtype: object) -> Tensor:
    spec = TensorSpec(tensor.spec.shape, from_monpy_dtype(dtype), tensor.spec.device, tensor.spec.layout)
    return Tensor(self._append(cast_p, (tensor.node,), {"dtype": spec.dtype.name}, spec), spec, self)

  def custom_call(self, name: str, args: Iterable[object], out: TensorSpec) -> Tensor:
    tensors = tuple(self.ensure_tensor(arg) for arg in args)
    return Tensor(self._append(custom_call_p, tuple(t.node for t in tensors), {"name": name}, out), out, self)

  def ensure_tensor(self, value: object) -> Tensor:
    if isinstance(value, Tensor):
      if value._trace is not self:
        raise ValueError("cannot mix tensors from different traces")
      return value
    if self.inputs:
      return self.constant(value, Tensor(self.inputs[0], self.nodes[self.inputs[0]].spec, self))
    raise TypeError("scalar constants require at least one tensor input to infer dtype/device")

  def graph(self, outputs: object) -> GraphIR:
    graph, _ = self.graph_and_output_tree(outputs)
    return graph

  def graph_and_output_tree(self, outputs: object) -> tuple[GraphIR, PyTreeDef]:
    output_tensors, output_tree = _flatten_output_tensors(outputs)
    output_refs = tuple(t.node for t in output_tensors)
    return GraphIR(inputs=tuple(self.inputs), outputs=output_refs, nodes=tuple(self.nodes)), output_tree

  def _ensure_pair(self, lhs: object, rhs: object) -> tuple[Tensor, Tensor]:
    if isinstance(lhs, Tensor):
      lhs_t = self.ensure_tensor(lhs)
      rhs_t = self.ensure_tensor(rhs)
    elif isinstance(rhs, Tensor):
      rhs_t = self.ensure_tensor(rhs)
      lhs_t = self.ensure_tensor(lhs)
    else:
      raise TypeError("at least one operand must be a traced monpy.Tensor")
    return lhs_t, rhs_t

  def _append(
    self, primitive: Primitive, inputs: tuple[ValueRef, ...], attrs: dict[str, object], spec: TensorSpec
  ) -> ValueRef:
    ref = len(self.nodes)
    self.nodes.append(Node(primitive=primitive, inputs=inputs, attrs=attrs, spec=spec))
    return ref


def _flatten_output_tensors(outputs: object) -> tuple[tuple[Tensor, ...], PyTreeDef]:
  leaves, treedef = tree_flatten(outputs)
  tensors: list[Tensor] = []
  for leaf in leaves:
    if not isinstance(leaf, Tensor):
      raise TypeError("jitted functions must return a pytree of monpy.Tensor leaves")
    tensors.append(leaf)
  return tuple(tensors), treedef


def _is_scalar_constant(value: object) -> bool:
  if isinstance(value, (bool, int, float, complex)):
    return True
  shape = getattr(value, "shape", None)
  if isinstance(shape, tuple):
    return len(shape) == 0
  if isinstance(value, (list, tuple, dict, set, frozenset)):
    return False
  return not hasattr(value, "__iter__")


def _static_dim(dim: object) -> int:
  if isinstance(dim, int):
    return dim
  raise ValueError("layout normalization requires static dimensions in this initial slice")


Shape = tuple[int | SymbolicDim | str, ...]


def _normalize_reduce_axes(axis: object, ndim: int) -> tuple[int, ...]:
  if axis is None:
    return tuple(range(ndim))
  if isinstance(axis, int):
    raw_axes = (axis,)
  elif isinstance(axis, Iterable):
    raw_axes = tuple(int(cast(Any, ax)) for ax in axis)
  else:
    raw_axes = (int(cast(Any, axis)),)
  axes = tuple(_normalize_axis(axis, ndim) for axis in raw_axes)
  if len(set(axes)) != len(axes):
    raise ValueError("reduce axes must be unique")
  return axes


def _normalize_axis(axis: int, ndim: int) -> int:
  result = axis + ndim if axis < 0 else axis
  if result < 0 or result >= ndim:
    raise ValueError(f"reduce axis {axis} is out of bounds for rank {ndim}")
  return result


def _reduce_shape(shape: Shape, axes: tuple[int, ...], keepdims: bool) -> Shape:
  if keepdims:
    return tuple(1 if idx in axes else dim for idx, dim in enumerate(shape))
  return tuple(dim for idx, dim in enumerate(shape) if idx not in axes)


def _broadcast_shape(lhs: Shape, rhs: Shape) -> Shape:
  out: list[int | SymbolicDim | str] = []
  for left, right in zip(reversed(lhs), reversed(rhs), strict=False):
    if left == 1:
      out.append(right)
    elif right == 1 or left == right:
      out.append(left)
    else:
      raise ValueError(f"cannot broadcast {lhs} and {rhs}")
  longer = lhs if len(lhs) > len(rhs) else rhs
  missing = len(longer) - len(out)
  out.extend(reversed(longer[:missing]))
  return tuple(reversed(out))


def _matmul_shape(lhs: Shape, rhs: Shape) -> Shape:
  if len(lhs) < 2 or len(rhs) < 2:
    raise ValueError("monpy kernel matmul requires rank >= 2 in this initial slice")
  if lhs[-1] != rhs[-2]:
    raise ValueError(f"matmul dimension mismatch: {lhs} @ {rhs}")
  batch = _broadcast_shape(lhs[:-2], rhs[:-2])
  return batch + (lhs[-2], rhs[-1])
