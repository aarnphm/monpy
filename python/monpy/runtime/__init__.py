from __future__ import annotations
import importlib
from types import ModuleType

__all__ = ["ops_numpy"]


def __getattr__(name:str)->ModuleType:
  if name=="ops_numpy":
    module=importlib.import_module(f"{__name__}.ops_numpy")
    globals()[name]=module
    return module
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
