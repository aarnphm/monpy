from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import monumpy as mnp
import numpy as np

from monpy._bench.core import BenchCase
from monpy._bench.types._helpers import prefix_groups


def build_cases(
  *,
  vector_size: int,
  vector_sizes: Sequence[int],
  matrix_sizes: Sequence[int],
  linalg_sizes: Sequence[int],
) -> list[BenchCase]:
  size = max(64, min(max(matrix_sizes), 256))
  vector = max(vector_size, 4096)

  a_np = (np.arange(size * size, dtype=np.float32).reshape(size, size) / 101.0).astype(
    np.float32,
    copy=False,
  )
  b_np = np.flip(a_np, axis=1).copy()
  row_np = np.linspace(0.1, 1.0, size, dtype=np.float32)
  a_mp: Any = mnp.asarray(a_np, dtype=mnp.float32, copy=False)
  b_mp: Any = mnp.asarray(b_np, dtype=mnp.float32, copy=False)
  row_mp: Any = mnp.asarray(row_np, dtype=mnp.float32, copy=False)

  x_np = np.linspace(0.1, 2.0, vector, dtype=np.float32)
  y_np = np.linspace(2.0, 4.0, vector, dtype=np.float32)
  x_mp: Any = mnp.asarray(x_np, dtype=mnp.float32, copy=False)
  y_mp: Any = mnp.asarray(y_np, dtype=mnp.float32, copy=False)

  cube_n = max(8, min(size // 4, 32))
  cube_np = (
    np.arange(cube_n * cube_n * cube_n, dtype=np.float32).reshape(cube_n, cube_n, cube_n) / 37.0
  ).astype(np.float32, copy=False)
  cube_rhs_np = np.flip(cube_np, axis=2).copy()
  cube_mp: Any = mnp.asarray(cube_np, dtype=mnp.float32, copy=False)
  cube_rhs_mp: Any = mnp.asarray(cube_rhs_np, dtype=mnp.float32, copy=False)

  cases = [
    BenchCase("elementwise", "transpose_add_f32", lambda: a_mp.T + b_mp.T, lambda: a_np.T + b_np.T),
    BenchCase("elementwise", "broadcast_row_add_f32", lambda: a_mp + row_mp, lambda: a_np + row_np),
    BenchCase(
      "elementwise",
      "reverse_1d_add_f32",
      lambda: x_mp[::-1] + y_mp[::-1],
      lambda: x_np[::-1] + y_np[::-1],
    ),
    BenchCase(
      "elementwise",
      "sliced_unary_sin_f32",
      lambda: mnp.sin(a_mp[::2, 1::2]),
      lambda: np.sin(a_np[::2, 1::2]),
    ),
    BenchCase(
      "elementwise",
      "rank3_transpose_add_f32",
      lambda: cube_mp.transpose((2, 0, 1)) + cube_rhs_mp.transpose((2, 0, 1)),
      lambda: cube_np.transpose((2, 0, 1)) + cube_rhs_np.transpose((2, 0, 1)),
    ),
    BenchCase(
      "views",
      "flip_axis0_f32",
      lambda: mnp.flip(a_mp, axis=0),
      lambda: np.flip(a_np, axis=0),
    ),
    BenchCase(
      "views",
      "flip_axis1_f32",
      lambda: mnp.flip(a_mp, axis=1),
      lambda: np.flip(a_np, axis=1),
    ),
    BenchCase("views", "flip_all_f32", lambda: mnp.flip(a_mp), lambda: np.flip(a_np)),
    BenchCase("views", "rot90_f32", lambda: mnp.rot90(a_mp), lambda: np.rot90(a_np)),
    BenchCase(
      "copy",
      "ascontiguousarray_transpose_f32",
      lambda: mnp.ascontiguousarray(a_mp.T),
      lambda: np.ascontiguousarray(a_np.T),
    ),
  ]
  return prefix_groups("strides", cases)
