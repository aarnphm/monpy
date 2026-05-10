"""Lowering registration helpers for extension packages."""

from __future__ import annotations

from collections.abc import Callable

from monpy._src.core import Primitive, get_lowering, register_lowering

MAX_TARGET = "max"
MOJO_TARGET = "mojo"
LOWERING_TARGETS = (MAX_TARGET, MOJO_TARGET)
LoweringTarget = str
LoweringRule = Callable[..., object]

__all__ = [
  "LOWERING_TARGETS",
  "MAX_TARGET",
  "MOJO_TARGET",
  "LoweringRule",
  "LoweringTarget",
  "Primitive",
  "get_lowering",
  "register_lowering",
]
