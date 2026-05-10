"""MAX Graph lowering decisions for monpy GraphIR.

The first implementation keeps MAX imports at the final lowering edge. Layout
lowering is pure metadata so it can be tested without a MAX runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from ..core import GraphIR, TensorSpec
from ..layout import LayoutSpec


class LayoutAction(str, Enum):
  IDENTITY = "identity"
  NATIVE_RESHAPE = "native_reshape"
  NATIVE_PERMUTE = "native_permute"
  NATIVE_BROADCAST = "native_broadcast"
  CUSTOM_ATTRS = "custom_attrs"
  MATERIALIZE = "materialize"
  ERROR = "error"


@dataclass(frozen=True, slots=True)
class LayoutLoweringDecision:
  action: LayoutAction
  op: str | None = None
  parameters: dict[str, bool | int | str] | None = None
  reason: str = ""


@dataclass(frozen=True, slots=True)
class LoweredGraph:
  graph: GraphIR
  layout_decisions: tuple[LayoutLoweringDecision, ...]
  max_graph: object | None = None


class GraphLowerer:
  def lower(self, graph: GraphIR) -> LoweredGraph:
    decisions: list[LayoutLoweringDecision] = []
    for node in graph.nodes:
      if node.inputs:
        source = graph.nodes[node.inputs[0]].spec.layout
        decisions.append(self.lower_layout(None, source, node.spec.layout))
    return LoweredGraph(graph, tuple(decisions))

  def lower_layout(
    self,
    value: Any,
    source: LayoutSpec,
    target: LayoutSpec,
  ) -> LayoutLoweringDecision:
    del value
    if source == target:
      return LayoutLoweringDecision(LayoutAction.IDENTITY, reason="layout already matches")
    if target.tile is not None or target.offset_elems != 0:
      return LayoutLoweringDecision(
        LayoutAction.CUSTOM_ATTRS,
        op="custom",
        parameters=target.as_parameters("layout"),
        reason="layout must be passed as custom-op attributes or materialized by caller",
      )
    if (
      source.is_row_major_compact()
      and target.is_row_major_compact()
      and source.element_count() == target.element_count()
    ):
      return LayoutLoweringDecision(LayoutAction.NATIVE_RESHAPE, op="reshape", parameters=_shape_params(target))
    perm = target.permutation_from(source)
    if perm is not None:
      return LayoutLoweringDecision(
        LayoutAction.NATIVE_PERMUTE,
        op="permute",
        parameters={f"axis_{idx}": axis for idx, axis in enumerate(perm)} | {"rank": len(perm)},
      )
    if target.is_broadcast_from(source):
      return LayoutLoweringDecision(LayoutAction.NATIVE_BROADCAST, op="broadcast_to", parameters=_shape_params(target))
    if not target.is_row_major_compact():
      return LayoutLoweringDecision(
        LayoutAction.CUSTOM_ATTRS,
        op="custom",
        parameters=target.as_parameters("layout"),
        reason="layout must be passed as custom-op attributes or materialized by caller",
      )
    return LayoutLoweringDecision(LayoutAction.ERROR, reason="unsupported layout transform")

  def tensor_type(self, spec: TensorSpec) -> object:
    """Construct `max.graph.TensorType`, importing MAX only here."""

    from max.graph import DeviceRef as _DeviceRef
    from max.graph import TensorType as _TensorType

    from ..dtypes import to_max_dtype

    TensorType = cast(Any, _TensorType)
    DeviceRef = cast(Any, _DeviceRef)
    shape = tuple(dim.name if hasattr(dim, "name") else dim for dim in spec.shape)
    if spec.device.kind in ("gpu", "cuda"):
      device = DeviceRef.GPU(0 if spec.device.index is None else spec.device.index)
    else:
      device = DeviceRef.CPU()
    return TensorType(to_max_dtype(spec.dtype), shape, device)


def _shape_params(layout: LayoutSpec) -> dict[str, int | str]:
  params: dict[str, int | str] = {"rank": len(layout.shape)}
  for idx, dim in enumerate(layout.shape):
    params[f"dim_{idx}"] = dim if isinstance(dim, int) else str(dim)
  return params
