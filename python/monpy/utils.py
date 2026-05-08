from __future__ import annotations

import importlib
import types
from typing import Any


class LazyLoader(types.ModuleType):
  def __init__(self, local_name: str, parent_globals: dict[str, Any], name: str) -> None:
    super().__init__(name)
    self._local_name = local_name
    self._parent_globals = parent_globals
    self._module_name = name
    self._module: types.ModuleType | None = None

  def _load(self) -> types.ModuleType:
    module = self._module
    if module is None:
      module = importlib.import_module(self._module_name)
      self._parent_globals[self._local_name] = module
      self.__dict__.update(module.__dict__)
      self._module = module
    return module

  def __getattr__(self, name: str) -> Any:
    return getattr(self._load(), name)

  def __dir__(self) -> list[str]:
    return sorted(set(super().__dir__()) | set(dir(self._load())))


__all__ = ["LazyLoader"]
