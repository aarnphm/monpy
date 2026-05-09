# fmt: off
# ruff: noqa
"""Small JAX-shaped eager transforms for monpy."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from . import asarray, moveaxis, ndarray, stack

Axis = int | None


def _is_axis(value: object) -> bool:
  return value is None or type(value) is builtins.int


def _normalize_axis(axis: int, ndim: int, context: str) -> int:
  result = axis + ndim if axis < 0 else axis
  if result < 0 or result >= ndim:
    raise ValueError(f"{context}: axis {axis} is out of bounds for rank {ndim}")
  return result


def _positional_axes(in_axes: object, nargs: int) -> tuple[Axis, ...]:
  if _is_axis(in_axes):
    return (cast(Axis, in_axes),) * nargs
  if isinstance(in_axes, Sequence) and not isinstance(in_axes, (str, bytes)):
    if len(in_axes) != nargs:
      raise ValueError(f"vmap in_axes has {len(in_axes)} entries for {nargs} positional arguments")
    if any(not _is_axis(axis) for axis in in_axes):
      raise NotImplementedError("vmap only supports flat positional in_axes in the eager path")
    return tuple(cast(Axis, axis) for axis in in_axes)
  raise TypeError("vmap in_axes must be an int, None, or a flat sequence of those values")


def _mapped_size(value: object, axis: int, context: str) -> tuple[int, int]:
  arr = asarray(value)
  if arr.ndim == 0:
    raise ValueError(f"{context}: cannot map over a rank-0 value")
  normalized = _normalize_axis(axis, arr.ndim, context)
  return arr.shape[normalized], normalized


def _axis_size(
  requested: int | None,
  args: tuple[object, ...],
  axes: tuple[Axis, ...],
  kwargs: Mapping[str, object],
) -> tuple[int, tuple[Axis, ...], dict[str, int]]:
  sizes: list[int] = []
  normalized_axes: list[Axis] = []
  kw_axes: dict[str, int] = {}
  for index, (arg, axis) in enumerate(zip(args, axes, strict=True)):
    if axis is None:
      normalized_axes.append(None)
      continue
    size, normalized = _mapped_size(arg, axis, f"vmap argument {index}")
    sizes.append(size)
    normalized_axes.append(normalized)
  for name, value in kwargs.items():
    size, normalized = _mapped_size(value, 0, f"vmap keyword argument {name!r}")
    sizes.append(size)
    kw_axes[name] = normalized
  if requested is not None:
    if requested < 0:
      raise ValueError("vmap axis_size must be non-negative")
    sizes.append(requested)
  if not sizes:
    raise ValueError("vmap needs at least one mapped argument or an explicit axis_size")
  first = sizes[0]
  if any(size != first for size in sizes[1:]):
    raise ValueError(f"vmap mapped axes must all have the same size, got {sizes}")
  return first, tuple(normalized_axes), kw_axes


def _slice_axis(value: object, axis: int, index: int) -> object:
  arr = asarray(value)
  key: list[object] = [slice(None)] * arr.ndim
  key[axis] = index
  return arr[tuple(key)]


def _output_axes(out_axes: object, arity: int, context: str) -> tuple[object, ...]:
  if _is_axis(out_axes):
    return (out_axes,) * arity
  if isinstance(out_axes, Sequence) and not isinstance(out_axes, (str, bytes)):
    if len(out_axes) != arity:
      raise ValueError(f"{context}: out_axes has {len(out_axes)} entries for {arity} outputs")
    return tuple(out_axes)
  raise TypeError(f"{context}: out_axes must be an int, None, or a flat sequence")


def _same_value(left: object, right: object) -> bool:
  if isinstance(left, ndarray):
    return isinstance(right, ndarray) and left.dtype == right.dtype and left.shape == right.shape and left.tolist() == right.tolist()
  return left == right


def _assert_unmapped(outputs: Sequence[object]) -> object:
  first = outputs[0]
  if any(not _same_value(first, output) for output in outputs[1:]):
    raise ValueError("vmap out_axes=None requires an unmapped output")
  return first


def _stack_leaf(outputs: Sequence[object], out_axis: object) -> object:
  if out_axis is None:
    return _assert_unmapped(outputs)
  if type(out_axis) is not builtins.int:
    raise TypeError("vmap out_axes leaves must be int or None")
  result = stack(outputs, axis=0)
  return result if out_axis == 0 else moveaxis(result, 0, out_axis)


def _stack_outputs(outputs: Sequence[object], out_axes: object) -> object:
  if not outputs:
    raise NotImplementedError("vmap over an axis of size 0 is not implemented")
  first = outputs[0]
  if first is None:
    for output in outputs[1:]:
      if output is not None:
        raise ValueError("vmap output structure changed across mapped calls")
    return None
  if isinstance(first, tuple):
    tuple_outputs = [cast(tuple[object, ...], output) for output in outputs]
    if any(len(output) != len(first) for output in tuple_outputs[1:]):
      raise ValueError("vmap tuple output arity changed across mapped calls")
    axes = _output_axes(out_axes, len(first), "vmap tuple output")
    return tuple(_stack_outputs([output[index] for output in tuple_outputs], axes[index]) for index in range(len(first)))
  if isinstance(first, list):
    list_outputs = [cast(list[object], output) for output in outputs]
    if any(len(output) != len(first) for output in list_outputs[1:]):
      raise ValueError("vmap list output arity changed across mapped calls")
    axes = _output_axes(out_axes, len(first), "vmap list output")
    return [_stack_outputs([output[index] for output in list_outputs], axes[index]) for index in range(len(first))]
  if isinstance(first, Mapping):
    keys = tuple(first.keys())
    mapping_outputs = [cast(Mapping[object, object], output) for output in outputs]
    if any(tuple(output.keys()) != keys for output in mapping_outputs[1:]):
      raise ValueError("vmap output mapping keys changed across mapped calls")
    if isinstance(out_axes, Mapping):
      out_axis_mapping = cast(Mapping[object, object], out_axes)
      return {key: _stack_outputs([output[key] for output in mapping_outputs], out_axis_mapping[key]) for key in keys}
    return {key: _stack_outputs([output[key] for output in mapping_outputs], out_axes) for key in keys}
  return _stack_leaf(outputs, out_axes)


@dataclass(frozen=True, slots=True)
class _VmapFunction:
  fn: Callable[..., object]
  in_axes: object = 0
  out_axes: object = 0
  axis_name: object | None = None
  axis_size: int | None = None
  spmd_axis_name: object | None = None
  sum_match: bool = False

  def __call__(self, *args: object, **kwargs: object) -> object:
    axes = _positional_axes(self.in_axes, len(args))
    size, normalized_axes, kw_axes = _axis_size(self.axis_size, args, axes, kwargs)
    outputs: list[object] = []
    for index in range(size):
      mapped_args = tuple(
        arg if axis is None else _slice_axis(arg, axis, index)
        for arg, axis in zip(args, normalized_axes, strict=True)
      )
      mapped_kwargs = {name: _slice_axis(value, kw_axes[name], index) for name, value in kwargs.items()}
      outputs.append(self.fn(*mapped_args, **mapped_kwargs))
    return _stack_outputs(outputs, self.out_axes)


def vmap(
  fun: Callable[..., object],
  in_axes: object = 0,
  out_axes: object = 0,
  axis_name: object | None = None,
  axis_size: int | None = None,
  spmd_axis_name: object | None = None,
  sum_match: bool = False,
) -> _VmapFunction:
  """Map ``fun`` over array axes using monpy's eager ndarray surface.

  This mirrors the JAX call shape for eager correctness tests. The fast path is
  still graph-level batching over monpy kernel primitives.
  """

  if not callable(fun):
    raise TypeError("vmap expects a callable")
  return _VmapFunction(
    fun,
    in_axes=in_axes,
    out_axes=out_axes,
    axis_name=axis_name,
    axis_size=axis_size,
    spmd_axis_name=spmd_axis_name,
    sum_match=sum_match,
  )


__all__ = ["vmap"]
