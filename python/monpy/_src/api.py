"""Staging and eager transform entry points for `monpy.lax`."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from .. import asarray, moveaxis, ndarray, stack
from .core import GraphIR, TensorSpec
from .interpreters.tracing import TraceContext
from .tree_util import PyTreeDef, assert_same_structure, tree_flatten, tree_unflatten

Axis = int | None


class _Indexable(Protocol):
  def __getitem__(self, index: int) -> object: ...


@dataclass(frozen=True, slots=True)
class CompiledFunction:
  fn: Callable[..., object]
  graph: GraphIR
  backend: str


@dataclass(frozen=True, slots=True)
class JittedFunction:
  fn: Callable[..., object]
  backend: Literal["auto", "graph", "native"] = "auto"
  dynamic_dims: Mapping[str, int | str] | None = None
  cache_size: int = 64

  def compile(self, *specs: TensorSpec, weights: object | None = None) -> CompiledFunction:
    if weights is not None:
      raise NotImplementedError("external weight binding belongs to the next monpy.lax slice")
    trace = TraceContext()
    inputs = tuple(trace.input(spec) for spec in specs)
    outputs = self.fn(*inputs)
    return CompiledFunction(self.fn, trace.graph(outputs), self.backend)

  def __call__(self, *args: object, **kwargs: object) -> object:
    if any(getattr(arg, "__monpy_kernel_tensor__", False) for arg in args):
      return self.fn(*args, **kwargs)
    raise TypeError("jitted monpy functions are compile boundaries; call .compile(...) with TensorSpec inputs")


def jit(
  fn: Callable[..., object] | None = None,
  *,
  backend: Literal["auto", "graph", "native"] = "auto",
  dynamic_dims: Mapping[str, int | str] | None = None,
  cache_size: int = 64,
) -> JittedFunction | Callable[[Callable[..., object]], JittedFunction]:
  def wrap(inner: Callable[..., object]) -> JittedFunction:
    return JittedFunction(inner, backend=backend, dynamic_dims=dynamic_dims, cache_size=cache_size)

  if fn is None:
    return wrap
  return wrap(fn)


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
  if getattr(value, "__monpy_random_key_batch__", False):
    ndim = cast(int, getattr(value, "ndim"))
    if ndim == 0:
      raise ValueError(f"{context}: cannot map over a rank-0 value")
    normalized = _normalize_axis(axis, ndim, context)
    shape = cast(tuple[int, ...], getattr(value, "shape"))
    return shape[normalized], normalized
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
  if getattr(value, "__monpy_random_key_batch__", False):
    if axis != 0:
      raise NotImplementedError("vmap over KeyBatch currently supports axis 0 only")
    return cast(_Indexable, value)[index]
  arr = asarray(value)
  key: list[object] = [slice(None)] * arr.ndim
  key[axis] = index
  return arr[tuple(key)]


def _out_axes_leaves(out_axes: object, output_def: PyTreeDef) -> tuple[Axis, ...]:
  if output_def.num_leaves == 0:
    return ()
  if _is_axis(out_axes):
    return (cast(Axis, out_axes),) * output_def.num_leaves
  leaves, axes_def = tree_flatten(out_axes)
  assert_same_structure(output_def, axes_def, "vmap out_axes")
  if any(not _is_axis(axis) for axis in leaves):
    raise TypeError("vmap out_axes leaves must be int or None")
  return tuple(cast(Axis, axis) for axis in leaves)


def _same_value(left: object, right: object) -> bool:
  if isinstance(left, ndarray):
    return (
      isinstance(right, ndarray)
      and left.dtype == right.dtype
      and left.shape == right.shape
      and left.tolist() == right.tolist()
    )
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
  first_leaves, output_def = tree_flatten(outputs[0])
  output_leaves = [first_leaves]
  for output in outputs[1:]:
    leaves, treedef = tree_flatten(output)
    assert_same_structure(output_def, treedef, "vmap output")
    output_leaves.append(leaves)
  axes = _out_axes_leaves(out_axes, output_def)
  if not axes:
    return tree_unflatten(output_def, ())
  stacked = tuple(
    _stack_leaf([leaves[index] for leaves in output_leaves], axes[index]) for index in range(len(first_leaves))
  )
  return tree_unflatten(output_def, stacked)


@dataclass(frozen=True, slots=True)
class VmapFunction:
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
        arg if axis is None else _slice_axis(arg, axis, index) for arg, axis in zip(args, normalized_axes, strict=True)
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
) -> VmapFunction:
  """Map ``fun`` over array axes using monpy's eager ndarray surface."""

  if not callable(fun):
    raise TypeError("vmap expects a callable")
  return VmapFunction(
    fun,
    in_axes=in_axes,
    out_axes=out_axes,
    axis_name=axis_name,
    axis_size=axis_size,
    spmd_axis_name=spmd_axis_name,
    sum_match=sum_match,
  )
