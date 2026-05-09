"""Small immutable IR for monpy kernel tracing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Mapping, cast

from .dtypes import DTypeSpec, from_monpy_dtype
from .layout import LayoutSpec

AttrValue = bool | int | float | str | tuple["AttrValue", ...] | None
ValueRef = int


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
  MATMUL = "matmul"
  REDUCE = "reduce"
  WHERE = "where"
  CUSTOM_CALL = "custom_call"


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
    normalized_layout = layout if layout is not None else _default_layout(tuple(_dim_name(dim) for dim in normalized_shape))
    object.__setattr__(self, "shape", normalized_shape)
    object.__setattr__(self, "dtype", normalized_dtype)
    object.__setattr__(self, "device", normalized_device)
    object.__setattr__(self, "layout", normalized_layout)


@dataclass(frozen=True, slots=True)
class Node:
  op: Op
  inputs: tuple[ValueRef, ...]
  attrs: Mapping[str, Any]
  spec: TensorSpec
  effects: frozenset[str] = frozenset()


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
  payload = _stable(
    {
      "inputs": graph.inputs,
      "outputs": graph.outputs,
      "nodes": graph.nodes,
      "constants": graph.constants,
    }
  )
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
  if isinstance(value, Enum):
    return value.value
  if isinstance(value, bytes):
    return value.hex()
  if is_dataclass(value):
    return _stable(asdict(cast(Any, value)))
  if isinstance(value, Mapping):
    return {str(k): _stable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
  if isinstance(value, (tuple, list)):
    return [_stable(v) for v in value]
  if isinstance(value, frozenset | set):
    return sorted(_stable(v) for v in value)
  return value
