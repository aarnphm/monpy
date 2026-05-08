"""Extension registration namespace for standalone monpy kernel packages."""

from __future__ import annotations

from .registry import KernelDescriptor, KernelRegistry

__all__ = ["KernelDescriptor", "KernelRegistry"]
