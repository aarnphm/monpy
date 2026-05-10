"""NumPy interop namespace for monpy."""

from __future__ import annotations

from . import ops
from .ops import (
  NumpyDTypeInfo,
  array_interface_typestr,
  buffer_format,
  dtype_from_buffer_format,
  dtype_from_typestr,
  dtype_info,
)

__all__ = [
  "NumpyDTypeInfo",
  "array_interface_typestr",
  "buffer_format",
  "dtype_from_buffer_format",
  "dtype_from_typestr",
  "dtype_info",
  "ops",
]
