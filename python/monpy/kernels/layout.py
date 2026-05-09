"""Serializable layout metadata for monpy GraphIR."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

Dim = int | str


class LayoutOrder(str, Enum):
  ROW_MAJOR = "row_major"
  COL_MAJOR = "col_major"
  EXPLICIT = "explicit"


class Contiguity(str, Enum):
  UNKNOWN = "unknown"
  ROW_MAJOR = "row_major"
  NON_CONTIGUOUS = "non_contiguous"
  BROADCAST = "broadcast"


@dataclass(frozen=True, slots=True)
class TileSpec:
  tile_shape: tuple[int, ...]
  logical_order: tuple[int, ...] | None = None
  vector_width: int | None = None
  warp_shape: tuple[int, ...] | None = None
  swizzle: str | None = None

  def as_parameters(self, prefix: str = "tile") -> dict[str, int | str]:
    params: dict[str, int | str] = {f"{prefix}_rank": len(self.tile_shape)}
    for idx, value in enumerate(self.tile_shape):
      params[f"{prefix}_shape_{idx}"] = value
    if self.logical_order is not None:
      params[f"{prefix}_logical_order"] = ",".join(str(v) for v in self.logical_order)
    if self.vector_width is not None:
      params[f"{prefix}_vector_width"] = self.vector_width
    if self.warp_shape is not None:
      params[f"{prefix}_warp_shape"] = ",".join(str(v) for v in self.warp_shape)
    if self.swizzle is not None:
      params[f"{prefix}_swizzle"] = self.swizzle
    return params


def row_major_strides(shape: tuple[Dim, ...]) -> tuple[Dim, ...]:
  strides = maybe_row_major_strides(shape)
  if strides is None:
    raise ValueError("row-major strides require static integer dimensions for now")
  return strides


def maybe_row_major_strides(shape: tuple[Dim, ...]) -> tuple[Dim, ...] | None:
  if any(not isinstance(dim, int) for dim in shape):
    return None
  stride = 1
  out: list[int] = [1] * len(shape)
  for axis in range(len(shape) - 1, -1, -1):
    out[axis] = stride
    stride *= int(shape[axis])
  return tuple(out)


def num_elements(shape: tuple[Dim, ...]) -> int | None:
  if any(not isinstance(dim, int) for dim in shape):
    return None
  return math.prod(int(dim) for dim in shape)


@dataclass(frozen=True, slots=True)
class LayoutSpec:
  shape: tuple[Dim, ...]
  strides: tuple[Dim, ...]
  offset_elems: Dim = 0
  order: LayoutOrder = LayoutOrder.EXPLICIT
  contiguity: Contiguity = Contiguity.UNKNOWN
  alignment_bytes: int | None = None
  tile: TileSpec | None = None

  def __post_init__(self) -> None:
    if len(self.shape) != len(self.strides):
      raise ValueError("layout shape and strides must have the same rank")

  @classmethod
  def row_major(cls: type[LayoutSpec], shape: tuple[Dim, ...], *, alignment_bytes: int | None = None) -> LayoutSpec:
    return cls(
      shape=tuple(shape),
      strides=row_major_strides(tuple(shape)),
      order=LayoutOrder.ROW_MAJOR,
      contiguity=Contiguity.ROW_MAJOR,
      alignment_bytes=alignment_bytes,
    )

  @classmethod
  def explicit(
    cls: type[LayoutSpec],
    shape: tuple[Dim, ...],
    strides: tuple[Dim, ...],
    *,
    offset_elems: Dim = 0,
    tile: TileSpec | None = None,
  ) -> LayoutSpec:
    expected = maybe_row_major_strides(tuple(shape))
    contiguity = Contiguity.ROW_MAJOR if expected is not None and tuple(strides) == expected else Contiguity.UNKNOWN
    return cls(tuple(shape), tuple(strides), offset_elems=offset_elems, contiguity=contiguity, tile=tile)

  def is_row_major_compact(self) -> bool:
    expected = maybe_row_major_strides(self.shape)
    return expected is not None and self.offset_elems == 0 and self.tile is None and self.strides == expected

  def element_count(self) -> int | None:
    return num_elements(self.shape)

  def reshape(self, shape: tuple[Dim, ...]) -> LayoutSpec:
    old_count = self.element_count()
    new_count = num_elements(tuple(shape))
    if old_count is not None and new_count is not None and old_count != new_count:
      raise ValueError(f"cannot reshape {self.shape} to {shape}")
    if self.is_row_major_compact():
      return LayoutSpec.row_major(tuple(shape), alignment_bytes=self.alignment_bytes)
    return LayoutSpec.explicit(tuple(shape), row_major_strides(tuple(shape)), offset_elems=0)

  def permute(self, axes: tuple[int, ...]) -> LayoutSpec:
    if sorted(axes) != list(range(len(self.shape))):
      raise ValueError("axes must be a permutation")
    contiguity = Contiguity.ROW_MAJOR if axes == tuple(range(len(axes))) and self.is_row_major_compact() else Contiguity.NON_CONTIGUOUS
    return LayoutSpec(
      shape=tuple(self.shape[axis] for axis in axes),
      strides=tuple(self.strides[axis] for axis in axes),
      offset_elems=self.offset_elems,
      order=LayoutOrder.EXPLICIT,
      contiguity=contiguity,
      alignment_bytes=self.alignment_bytes,
      tile=self.tile,
    )

  def broadcast_to(self, shape: tuple[Dim, ...]) -> LayoutSpec:
    source_shape = self.shape
    source_strides = self.strides
    if len(shape) < len(source_shape):
      raise ValueError("cannot broadcast to fewer dimensions")
    pad = len(shape) - len(source_shape)
    padded_shape: tuple[Dim, ...] = (1,) * pad + source_shape
    padded_strides: tuple[Dim, ...] = (0,) * pad + source_strides
    out_strides: list[Dim] = []
    for src_dim, dst_dim, stride in zip(padded_shape, shape, padded_strides, strict=True):
      if src_dim == dst_dim:
        out_strides.append(stride)
      elif src_dim == 1:
        out_strides.append(0)
      else:
        raise ValueError(f"cannot broadcast dimension {src_dim!r} to {dst_dim!r}")
    return LayoutSpec(tuple(shape), tuple(out_strides), offset_elems=self.offset_elems, contiguity=Contiguity.BROADCAST)

  def permutation_from(self, source: LayoutSpec) -> tuple[int, ...] | None:
    if len(self.shape) != len(source.shape):
      return None
    remaining = list(range(len(source.shape)))
    axes: list[int] = []
    for dst_shape, dst_stride in zip(self.shape, self.strides, strict=True):
      found: int | None = None
      for axis in remaining:
        if source.shape[axis] == dst_shape and source.strides[axis] == dst_stride:
          found = axis
          break
      if found is None:
        return None
      remaining.remove(found)
      axes.append(found)
    return tuple(axes)

  def is_broadcast_from(self, source: LayoutSpec) -> bool:
    try:
      return source.broadcast_to(self.shape).strides == self.strides
    except ValueError:
      return False

  def with_tile(self, tile: TileSpec) -> LayoutSpec:
    return LayoutSpec(
      self.shape,
      self.strides,
      offset_elems=self.offset_elems,
      order=self.order,
      contiguity=self.contiguity,
      alignment_bytes=self.alignment_bytes,
      tile=tile,
    )

  def as_parameters(self, prefix: str) -> dict[str, int | str]:
    params: dict[str, int | str] = {
      f"{prefix}_rank": len(self.shape),
      f"{prefix}_offset_elems": self.offset_elems if isinstance(self.offset_elems, int) else str(self.offset_elems),
      f"{prefix}_contiguity": self.contiguity.value,
    }
    for idx, dim in enumerate(self.shape):
      params[f"{prefix}_shape_{idx}"] = dim if isinstance(dim, int) else str(dim)
    for idx, stride in enumerate(self.strides):
      params[f"{prefix}_stride_{idx}"] = stride if isinstance(stride, int) else str(stride)
    if self.alignment_bytes is not None:
      params[f"{prefix}_alignment_bytes"] = self.alignment_bytes
    if self.tile is not None:
      params.update(self.tile.as_parameters(f"{prefix}_tile"))
    return params
