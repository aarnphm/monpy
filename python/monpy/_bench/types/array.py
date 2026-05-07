from __future__ import annotations

from collections.abc import Sequence

from monpy._bench.core import BenchCase
from monpy._bench.core import build_cases as _build_array_core_cases
from monpy._bench.types._helpers import prefix_groups


def build_cases(
  *,
  vector_size: int,
  vector_sizes: Sequence[int],
  matrix_sizes: Sequence[int],
  linalg_sizes: Sequence[int],
) -> list[BenchCase]:
  return prefix_groups(
    "array",
    _build_array_core_cases(
      vector_size=vector_size,
      vector_sizes=vector_sizes,
      matrix_sizes=matrix_sizes,
      linalg_sizes=linalg_sizes,
    ),
  )
