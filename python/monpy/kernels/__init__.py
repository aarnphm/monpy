"""Optional monpy kernel/compiler surface.

This package is intentionally import-light. It owns symbolic kernel IR and MAX
Graph lowering metadata, but importing it must not import MAX, safetensors, or
NumPy. Backend imports belong in the specific lowering call that needs them.
"""

from __future__ import annotations

from .api import CompiledFunction, JittedFunction, jit
from .dtypes import DTypeKind, DTypeSpec, StorageKind
from .ir import DeviceSpec, GraphIR, Node, Op, SymbolicDim, TensorSpec
from .layout import Contiguity, LayoutOrder, LayoutSpec, TileSpec
from .tensor import Tensor

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
  "Op",
  "StorageKind",
  "SymbolicDim",
  "Tensor",
  "TensorSpec",
  "TileSpec",
  "jit",
]
