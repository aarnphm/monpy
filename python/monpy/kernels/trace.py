"""Symbolic execution for `@monpy.jit`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .dtypes import from_monpy_dtype
from .ir import GraphIR, Node, Op, SymbolicDim, TensorSpec, ValueRef
from .layout import LayoutSpec
from .tensor import Tensor


_OP_BY_NAME: dict[str, Op] = {
  "add": Op.ADD,
  "sub": Op.SUB,
  "subtract": Op.SUB,
  "mul": Op.MUL,
  "multiply": Op.MUL,
  "div": Op.DIV,
  "divide": Op.DIV,
}


@dataclass(slots=True)
class TraceContext:
  nodes: list[Node] = field(default_factory=list)
  inputs: list[ValueRef] = field(default_factory=list)

  def input(self, spec: TensorSpec) -> Tensor:
    node_ref = self._append(Op.INPUT, (), {}, spec)
    self.inputs.append(node_ref)
    return Tensor(node_ref, spec, self)

  def constant(self, value: object, like: Tensor) -> Tensor:
    spec = TensorSpec((), like.spec.dtype, like.spec.device)
    return Tensor(self._append(Op.CONSTANT, (), {"value": repr(value)}, spec), spec, self)

  def binary(self, op_name: str, lhs: object, rhs: object) -> Tensor:
    lhs_t, rhs_t = self._ensure_pair(lhs, rhs)
    op = _OP_BY_NAME[op_name]
    shape = _broadcast_shape(lhs_t.spec.shape, rhs_t.spec.shape)
    layout = LayoutSpec.row_major(tuple(_static_dim(dim) for dim in shape))
    spec = TensorSpec(shape, lhs_t.spec.dtype, lhs_t.spec.device, layout)
    return Tensor(self._append(op, (lhs_t.node, rhs_t.node), {}, spec), spec, self)

  def unary(self, op_name: str, x: object) -> Tensor:
    tensor = self.ensure_tensor(x)
    spec = TensorSpec(tensor.spec.shape, tensor.spec.dtype, tensor.spec.device, tensor.spec.layout)
    return Tensor(self._append(Op.CUSTOM_CALL, (tensor.node,), {"name": f"monpy.{op_name}"}, spec), spec, self)

  def matmul(self, lhs: object, rhs: object) -> Tensor:
    lhs_t, rhs_t = self._ensure_pair(lhs, rhs)
    out_shape = _matmul_shape(lhs_t.spec.shape, rhs_t.spec.shape)
    spec = TensorSpec(out_shape, lhs_t.spec.dtype, lhs_t.spec.device)
    return Tensor(self._append(Op.MATMUL, (lhs_t.node, rhs_t.node), {}, spec), spec, self)

  def reshape(self, tensor: Tensor, shape: tuple[int, ...]) -> Tensor:
    layout = tensor.spec.layout.reshape(shape)
    spec = TensorSpec(tuple(shape), tensor.spec.dtype, tensor.spec.device, layout)
    return Tensor(self._append(Op.RESHAPE, (tensor.node,), {"shape": shape}, spec), spec, self)

  def transpose(self, tensor: Tensor, axes: tuple[int, ...]) -> Tensor:
    layout = tensor.spec.layout.permute(axes)
    spec = TensorSpec(tuple(tensor.spec.shape[axis] for axis in axes), tensor.spec.dtype, tensor.spec.device, layout)
    return Tensor(self._append(Op.TRANSPOSE, (tensor.node,), {"axes": axes}, spec), spec, self)

  def broadcast_to(self, tensor: Tensor, shape: tuple[int, ...]) -> Tensor:
    layout = tensor.spec.layout.broadcast_to(shape)
    spec = TensorSpec(tuple(shape), tensor.spec.dtype, tensor.spec.device, layout)
    return Tensor(self._append(Op.BROADCAST_TO, (tensor.node,), {"shape": shape}, spec), spec, self)

  def cast(self, tensor: Tensor, dtype: object) -> Tensor:
    spec = TensorSpec(tensor.spec.shape, from_monpy_dtype(dtype), tensor.spec.device, tensor.spec.layout)
    return Tensor(self._append(Op.CAST, (tensor.node,), {"dtype": spec.dtype.name}, spec), spec, self)

  def custom_call(self, name: str, args: Iterable[object], out: TensorSpec) -> Tensor:
    tensors = tuple(self.ensure_tensor(arg) for arg in args)
    return Tensor(self._append(Op.CUSTOM_CALL, tuple(t.node for t in tensors), {"name": name}, out), out, self)

  def ensure_tensor(self, value: object) -> Tensor:
    if isinstance(value, Tensor):
      if value._trace is not self:
        raise ValueError("cannot mix tensors from different traces")
      return value
    if self.inputs:
      return self.constant(value, Tensor(self.inputs[0], self.nodes[self.inputs[0]].spec, self))
    raise TypeError("scalar constants require at least one tensor input to infer dtype/device")

  def graph(self, outputs: object) -> GraphIR:
    output_refs = tuple(t.node for t in _flatten_outputs(outputs))
    return GraphIR(inputs=tuple(self.inputs), outputs=output_refs, nodes=tuple(self.nodes))

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

  def _append(self, op: Op, inputs: tuple[ValueRef, ...], attrs: dict[str, object], spec: TensorSpec) -> ValueRef:
    ref = len(self.nodes)
    self.nodes.append(Node(op=op, inputs=inputs, attrs=attrs, spec=spec))
    return ref


def _flatten_outputs(outputs: object) -> tuple[Tensor, ...]:
  if isinstance(outputs, Tensor):
    return (outputs,)
  if isinstance(outputs, tuple):
    out: list[Tensor] = []
    for item in outputs:
      out.extend(_flatten_outputs(item))
    return tuple(out)
  raise TypeError("jitted functions must return a Tensor or tuple of Tensors")


def _static_dim(dim: object) -> int:
  if isinstance(dim, int):
    return dim
  raise ValueError("layout normalization requires static dimensions in this initial slice")


Shape = tuple[int | SymbolicDim | str, ...]


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
