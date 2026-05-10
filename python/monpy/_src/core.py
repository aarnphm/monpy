"""Core compiler IR and primitive registry for monpy tracing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Literal, cast

from .dtypes import DTypeSpec, from_monpy_dtype
from .layout import LayoutSpec

AttrValue = bool | int | float | str | tuple["AttrValue", ...] | None
ValueRef = int
AbstractEvalRule = Callable[..., "TensorSpec"]
DTypeRule = Callable[..., DTypeSpec]
EagerBindRule = Callable[..., object]
BatchingRule = Callable[..., object]
LoweringRule = Callable[..., object]
NativeUfuncKind = Literal["logical", "compare", "predicate", "binary", "unary", "unary_preserve"]


class Op(str, Enum):
  INPUT = "input"
  CONSTANT = "constant"
  EXTERNAL_WEIGHT = "external_weight"
  VIEW = "view"
  RESHAPE = "reshape"
  TRANSPOSE = "transpose"
  BROADCAST_TO = "broadcast_to"
  COPY_CONTIGUOUS = "copy_contiguous"
  CAST = "cast"
  ADD = "add"
  SUB = "sub"
  MUL = "mul"
  DIV = "div"
  EQUAL = "equal"
  NOT_EQUAL = "not_equal"
  LESS = "less"
  LESS_EQUAL = "less_equal"
  GREATER = "greater"
  GREATER_EQUAL = "greater_equal"
  MATMUL = "matmul"
  REDUCE = "reduce"
  WHERE = "where"
  CUSTOM_CALL = "custom_call"


class Primitive:
  """Compiler primitive handle used by GraphIR nodes.

  The handle owns the extension points we need now, but graph serialization only
  depends on ``name``. That keeps structural keys stable across processes and
  reloads instead of smuggling Python object identity into the cache key.
  """

  __slots__ = (
    "name",
    "abstract_eval",
    "dtype_rule",
    "eager_impl",
    "batching_rule",
    "target_lowerings",
    "ufunc_kind",
    "ufunc_op",
    "ufunc_nin",
    "ufunc_nout",
    "ufunc_identity",
    "reduce_op",
  )

  def __init__(
    self,
    name: str,
    *,
    abstract_eval: AbstractEvalRule | None = None,
    dtype_rule: DTypeRule | None = None,
    eager_impl: EagerBindRule | None = None,
    batching_rule: BatchingRule | None = None,
    target_lowerings: Mapping[str, LoweringRule] | None = None,
    ufunc_kind: NativeUfuncKind | None = None,
    ufunc_op: int | None = None,
    ufunc_nin: int | None = None,
    ufunc_nout: int | None = None,
    ufunc_identity: object = None,
    reduce_op: int | None = None,
  ) -> None:
    self.name = name
    self.abstract_eval = abstract_eval
    self.dtype_rule = dtype_rule
    self.eager_impl = eager_impl
    self.batching_rule = batching_rule
    self.target_lowerings = dict(target_lowerings or {})
    self.ufunc_kind = ufunc_kind
    self.ufunc_op = ufunc_op
    self.ufunc_nin = ufunc_nin
    self.ufunc_nout = ufunc_nout
    self.ufunc_identity = ufunc_identity
    self.reduce_op = reduce_op

  def def_ufunc(
    self,
    *,
    kind: NativeUfuncKind,
    op: int,
    nin: int,
    nout: int,
    identity: object = None,
    reduce_op: int | None = None,
  ) -> None:
    self.ufunc_kind = kind
    self.ufunc_op = op
    self.ufunc_nin = nin
    self.ufunc_nout = nout
    self.ufunc_identity = identity
    self.reduce_op = reduce_op

  def def_lowering(self, target: str, rule: LoweringRule) -> None:
    self.target_lowerings[target] = rule

  def lowering(self, target: str) -> LoweringRule:
    try:
      return self.target_lowerings[target]
    except KeyError as exc:
      raise KeyError(f"primitive {self.name!r} has no {target!r} lowering") from exc

  def __repr__(self) -> str:
    return f"Primitive({self.name!r})"

  def __eq__(self, other: object) -> bool:
    return isinstance(other, Primitive) and self.name == other.name

  def __hash__(self) -> int:
    return hash(self.name)


@dataclass(slots=True)
class PrimitiveRegistry:
  _primitives: dict[str, Primitive] = field(default_factory=dict)
  _aliases: dict[str, str] = field(default_factory=dict)

  def register(self, primitive: Primitive) -> Primitive:
    if primitive.name in self._primitives or primitive.name in self._aliases:
      raise ValueError(f"primitive already registered: {primitive.name}")
    self._primitives[primitive.name] = primitive
    return primitive

  def alias(self, alias: str, target: str) -> None:
    if alias in self._primitives or alias in self._aliases:
      raise ValueError(f"primitive already registered: {alias}")
    if target not in self._primitives:
      raise KeyError(f"cannot alias unknown monpy primitive: {target}")
    self._aliases[alias] = target

  def define(
    self,
    name: str,
    *,
    abstract_eval: AbstractEvalRule | None = None,
    dtype_rule: DTypeRule | None = None,
    eager_impl: EagerBindRule | None = None,
    batching_rule: BatchingRule | None = None,
  ) -> Primitive:
    return self.register(
      Primitive(
        name,
        abstract_eval=abstract_eval,
        dtype_rule=dtype_rule,
        eager_impl=eager_impl,
        batching_rule=batching_rule,
      )
    )

  def get(self, name: str) -> Primitive:
    target = self._aliases.get(name, name)
    try:
      return self._primitives[target]
    except KeyError as exc:
      raise KeyError(f"unknown monpy primitive: {name}") from exc

  def __contains__(self, name: object) -> bool:
    return isinstance(name, str) and (name in self._primitives or name in self._aliases)

  def names(self) -> tuple[str, ...]:
    return tuple(sorted(self._primitives))


PRIMITIVES = PrimitiveRegistry()


def register_primitive(primitive: Primitive) -> Primitive:
  return PRIMITIVES.register(primitive)


def define_primitive(
  name: str,
  *,
  abstract_eval: AbstractEvalRule | None = None,
  dtype_rule: DTypeRule | None = None,
  eager_impl: EagerBindRule | None = None,
  batching_rule: BatchingRule | None = None,
) -> Primitive:
  return PRIMITIVES.define(
    name,
    abstract_eval=abstract_eval,
    dtype_rule=dtype_rule,
    eager_impl=eager_impl,
    batching_rule=batching_rule,
  )


def get_primitive(name: str) -> Primitive:
  return PRIMITIVES.get(name)


def register_lowering(primitive: Primitive | str, target: str, rule: LoweringRule) -> None:
  handle = get_primitive(primitive) if isinstance(primitive, str) else primitive
  handle.def_lowering(target, rule)


def get_lowering(primitive: Primitive | str, target: str) -> LoweringRule:
  handle = get_primitive(primitive) if isinstance(primitive, str) else primitive
  return handle.lowering(target)


input_p = define_primitive("input")
constant_p = define_primitive("constant")
external_weight_p = define_primitive("external_weight")
view_p = define_primitive("view")
reshape_p = define_primitive("reshape")
transpose_p = define_primitive("transpose")
broadcast_to_p = define_primitive("broadcast_to")
copy_contiguous_p = define_primitive("copy_contiguous")
cast_p = define_primitive("cast")
add_p = define_primitive("add")
sub_p = define_primitive("sub")
mul_p = define_primitive("mul")
div_p = define_primitive("div")
equal_p = define_primitive("equal")
not_equal_p = define_primitive("not_equal")
less_p = define_primitive("less")
less_equal_p = define_primitive("less_equal")
greater_p = define_primitive("greater")
greater_equal_p = define_primitive("greater_equal")
matmul_p = define_primitive("matmul")
reduce_p = define_primitive("reduce")
where_p = define_primitive("where")
custom_call_p = define_primitive("custom_call")
PRIMITIVES.alias("subtract", "sub")
PRIMITIVES.alias("multiply", "mul")
PRIMITIVES.alias("divide", "div")
PRIMITIVES.alias("true_divide", "div")


@dataclass(frozen=True, slots=True)
class SymbolicDim:
  name: str


@dataclass(frozen=True, slots=True)
class DeviceSpec:
  kind: str = "cpu"
  index: int | None = None

  @classmethod
  def coerce(cls: type[DeviceSpec], device: object = None) -> DeviceSpec:
    if isinstance(device, DeviceSpec):
      return device
    if device is None:
      return cls()
    if isinstance(device, str):
      if ":" in device:
        kind, raw_index = device.split(":", 1)
        return cls(kind, int(raw_index))
      return cls(device)
    raise TypeError(f"unsupported device spec: {device!r}")


@dataclass(frozen=True, slots=True, init=False)
class TensorSpec:
  shape: tuple[int | SymbolicDim | str, ...]
  dtype: DTypeSpec
  device: DeviceSpec
  layout: LayoutSpec

  def __init__(
    self,
    shape: tuple[int | SymbolicDim | str, ...],
    dtype: DTypeSpec | object,
    device: DeviceSpec | str | None = None,
    layout: LayoutSpec | None = None,
  ) -> None:
    normalized_shape = tuple(shape)
    normalized_dtype = from_monpy_dtype(dtype)
    normalized_device = DeviceSpec.coerce(device)
    normalized_layout = (
      layout if layout is not None else _default_layout(tuple(_dim_name(dim) for dim in normalized_shape))
    )
    object.__setattr__(self, "shape", normalized_shape)
    object.__setattr__(self, "dtype", normalized_dtype)
    object.__setattr__(self, "device", normalized_device)
    object.__setattr__(self, "layout", normalized_layout)


@dataclass(frozen=True, slots=True)
class Node:
  primitive: Primitive
  inputs: tuple[ValueRef, ...]
  attrs: Mapping[str, Any]
  spec: TensorSpec
  effects: frozenset[str] = frozenset()

  @property
  def op(self) -> str:
    return self.primitive.name


@dataclass(frozen=True, slots=True)
class GraphIR:
  inputs: tuple[ValueRef, ...]
  outputs: tuple[ValueRef, ...]
  nodes: tuple[Node, ...]
  constants: tuple[object, ...] = ()
  structural_key: bytes = b""

  def __post_init__(self) -> None:
    if not self.structural_key:
      object.__setattr__(self, "structural_key", structural_key(self))


def structural_key(graph: GraphIR) -> bytes:
  payload = _stable({
    "inputs": graph.inputs,
    "outputs": graph.outputs,
    "nodes": graph.nodes,
    "constants": graph.constants,
  })
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
  return hashlib.sha256(encoded).digest()


def _dim_name(dim: int | SymbolicDim | str) -> int | str:
  if isinstance(dim, SymbolicDim):
    return dim.name
  return dim


def _default_layout(shape: tuple[int | str, ...]) -> LayoutSpec:
  try:
    return LayoutSpec.row_major(shape)
  except ValueError:
    return LayoutSpec(shape=shape, strides=tuple(f"stride_{idx}" for idx in range(len(shape))))


def _stable(value: object) -> object:
  if isinstance(value, Primitive):
    return value.name
  if isinstance(value, Enum):
    return value.value
  if isinstance(value, bytes):
    return value.hex()
  if is_dataclass(value):
    return {field.name: _stable(getattr(cast(Any, value), field.name)) for field in fields(cast(Any, value))}
  if isinstance(value, Mapping):
    return {str(k): _stable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
  if isinstance(value, (tuple, list)):
    return [_stable(v) for v in value]
  if isinstance(value, frozenset | set):
    return sorted(_stable(v) for v in value)
  return value
