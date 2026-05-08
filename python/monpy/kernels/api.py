"""Public `@monpy.jit` implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .ir import GraphIR, TensorSpec
from .trace import TraceContext


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
      raise NotImplementedError("external weight binding belongs to the next monpy.kernels slice")
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
