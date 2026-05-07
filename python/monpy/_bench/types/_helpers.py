from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from monpy._bench.core import BenchCase


def prefix_groups(prefix: str, cases: Sequence[BenchCase]) -> list[BenchCase]:
  return [replace(case, group=f"{prefix}/{case.group}") for case in cases]
