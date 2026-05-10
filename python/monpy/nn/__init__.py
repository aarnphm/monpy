from __future__ import annotations

import builtins

from monpy import _native, asarray, ascontiguousarray, ndarray
from monpy import bool as bool_dtype


def layer_norm(x: object, gain: object, bias: object, eps: float = 1e-5) -> ndarray:
  arr = asarray(x)
  g = asarray(gain, dtype=arr.dtype, copy=False)
  b = asarray(bias, dtype=arr.dtype, copy=False)
  return ndarray(_native.layer_norm_last_axis(arr._native, g._native, b._native, builtins.float(eps)))


def softmax(x: object, axis: int = -1) -> ndarray:
  arr = asarray(x)
  if axis < 0:
    axis += arr.ndim
  if axis != arr.ndim - 1:
    raise NotImplementedError("softmax currently requires the last axis")
  if arr.ndim != 2:
    raise NotImplementedError("softmax currently requires a rank-2 input")
  if not arr._native.is_c_contiguous():
    arr = ascontiguousarray(arr)
  return ndarray(_native.softmax_last_axis(arr._native))


def scaled_masked_softmax(
  x: object,
  mask: object,
  scale: float = 1.0,
  fill: float = -1.0e9,
  axis: int = -1,
) -> ndarray:
  arr = asarray(x)
  m = asarray(mask, dtype=bool_dtype, copy=False)
  if axis < 0:
    axis += arr.ndim
  if axis != arr.ndim - 1:
    raise NotImplementedError("scaled_masked_softmax currently requires the last axis")
  if arr.ndim != 2:
    raise NotImplementedError("scaled_masked_softmax currently requires a rank-2 input")
  if not arr._native.is_c_contiguous():
    arr = ascontiguousarray(arr)
  if not m._native.is_c_contiguous():
    m = ascontiguousarray(m)
  return ndarray(
    _native.scaled_masked_softmax_last_axis(arr._native, m._native, builtins.float(scale), builtins.float(fill))
  )


__all__ = ["layer_norm", "scaled_masked_softmax", "softmax"]
