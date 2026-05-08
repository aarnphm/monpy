"""Registration contracts for optional monpy kernel packages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..dtypes import DTypeSpec
from ..ir import Node

LayoutPredicate = Callable[[object], bool]
LoweringHook = Callable[[object, Node], object]


@dataclass(frozen=True, slots=True)
class KernelDescriptor:
  name: str
  inputs: tuple[object, ...]
  outputs: tuple[object, ...]
  supported_dtypes: frozenset[DTypeSpec]
  supported_layouts: LayoutPredicate
  lowering: LoweringHook
  mojo_sources: tuple[Path, ...] = ()


@dataclass(slots=True)
class KernelRegistry:
  _kernels: dict[str, KernelDescriptor] = field(default_factory=dict)

  def register(self, descriptor: KernelDescriptor) -> None:
    if descriptor.name in self._kernels:
      raise ValueError(f"kernel already registered: {descriptor.name}")
    self._kernels[descriptor.name] = descriptor

  def get(self, name: str) -> KernelDescriptor:
    try:
      return self._kernels[name]
    except KeyError as exc:
      raise KeyError(f"unknown monpy kernel extension: {name}") from exc

  def __contains__(self, name: object) -> bool:
    return isinstance(name, str) and name in self._kernels


REGISTRY = KernelRegistry()
