"""Symbolic tensor handle used inside `@monpy.lax.jit`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ..core import TensorSpec, ValueRef

if TYPE_CHECKING:
  from ..interpreters.tracing import TraceContext


@dataclass(frozen=True, slots=True)
class Tensor:
  node: ValueRef
  spec: TensorSpec
  _trace: TraceContext

  __monpy_kernel_tensor__: ClassVar[bool] = True

  @property
  def shape(self) -> tuple[object, ...]:
    return self.spec.shape

  @property
  def dtype(self) -> object:
    return self.spec.dtype

  @property
  def device(self) -> object:
    return self.spec.device

  def __add__(self, other: object) -> Tensor:
    return self._trace.binary("add", self, other)

  def __radd__(self, other: object) -> Tensor:
    return self._trace.binary("add", other, self)

  def __sub__(self, other: object) -> Tensor:
    return self._trace.binary("sub", self, other)

  def __rsub__(self, other: object) -> Tensor:
    return self._trace.binary("sub", other, self)

  def __mul__(self, other: object) -> Tensor:
    return self._trace.binary("mul", self, other)

  def __rmul__(self, other: object) -> Tensor:
    return self._trace.binary("mul", other, self)

  def __truediv__(self, other: object) -> Tensor:
    return self._trace.binary("div", self, other)

  def __rtruediv__(self, other: object) -> Tensor:
    return self._trace.binary("div", other, self)

  def __matmul__(self, other: object) -> Tensor:
    return self._trace.matmul(self, other)

  def __rmatmul__(self, other: object) -> Tensor:
    return self._trace.matmul(other, self)

  def reshape(self, shape: int | Sequence[int]) -> Tensor:
    if isinstance(shape, int):
      target = (shape,)
    else:
      target = tuple(shape)
    return self._trace.reshape(self, target)

  def transpose(self, axes: Sequence[int] | None = None) -> Tensor:
    if axes is None:
      axes = tuple(range(len(self.shape) - 1, -1, -1))
    return self._trace.transpose(self, tuple(int(axis) for axis in axes))

  def astype(self, dtype: object) -> Tensor:
    return self._trace.cast(self, dtype)
