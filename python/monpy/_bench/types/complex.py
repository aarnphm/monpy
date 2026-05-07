from __future__ import annotations

from collections.abc import Sequence

import monumpy as mnp
import numpy as np

from monpy._bench.core import BenchCase
from monpy._bench.types._helpers import prefix_groups


def _complex_vector(size: int, dtype: type[np.complexfloating]) -> np.ndarray:
  real = np.linspace(-2.0, 2.0, size, dtype=np.float64)
  imag = np.linspace(1.0, -1.0, size, dtype=np.float64)
  return (real + 1j * imag).astype(dtype)


def build_cases(
  *,
  vector_size: int,
  vector_sizes: Sequence[int],
  matrix_sizes: Sequence[int],
  linalg_sizes: Sequence[int],
) -> list[BenchCase]:
  size = max(vector_size, 1024)
  z64_np = _complex_vector(size, np.complex64)
  w64_np = np.flip(_complex_vector(size, np.complex64)).copy()
  z128_np = _complex_vector(size, np.complex128)
  w128_np = np.flip(_complex_vector(size, np.complex128)).copy()
  z64_mp = mnp.asarray(z64_np, dtype=mnp.complex64, copy=True)
  w64_mp = mnp.asarray(w64_np, dtype=mnp.complex64, copy=True)
  z128_mp = mnp.asarray(z128_np, dtype=mnp.complex128, copy=True)
  w128_mp = mnp.asarray(w128_np, dtype=mnp.complex128, copy=True)

  mat_n = max(8, min(max(matrix_sizes), 64))
  lhs64_np = _complex_vector(mat_n * mat_n, np.complex64).reshape(mat_n, mat_n)
  rhs64_np = np.flip(lhs64_np, axis=1).copy()
  lhs64_mp = mnp.asarray(lhs64_np, dtype=mnp.complex64, copy=True)
  rhs64_mp = mnp.asarray(rhs64_np, dtype=mnp.complex64, copy=True)

  cases = [
    BenchCase(
      "interop",
      "asarray_complex64",
      lambda src=z64_np: mnp.asarray(src, dtype=mnp.complex64, copy=False),
      lambda src=z64_np: np.asarray(src, dtype=np.complex64),
    ),
    BenchCase(
      "interop",
      "array_copy_complex128",
      lambda src=z128_np: mnp.array(src, dtype=mnp.complex128, copy=True),
      lambda src=z128_np: np.array(src, dtype=np.complex128, copy=True),
    ),
    BenchCase(
      "elementwise",
      "binary_add_complex64",
      lambda: z64_mp + w64_mp,
      lambda: z64_np + w64_np,
    ),
    BenchCase(
      "elementwise",
      "binary_mul_complex64",
      lambda: z64_mp * w64_mp,
      lambda: z64_np * w64_np,
    ),
    BenchCase(
      "elementwise",
      "binary_add_complex128",
      lambda: z128_mp + w128_mp,
      lambda: z128_np + w128_np,
    ),
    BenchCase(
      "views",
      "reversed_add_complex64",
      lambda: z64_mp[::-1] + w64_mp[::-1],
      lambda: z64_np[::-1] + w64_np[::-1],
    ),
    BenchCase(
      "casts",
      "astype_complex64_to_complex128",
      lambda: z64_mp.astype(mnp.complex128),
      lambda: z64_np.astype(np.complex128),
    ),
    BenchCase(
      "matmul",
      f"matmul_{mat_n}_complex64",
      lambda lhs=lhs64_mp, rhs=rhs64_mp: lhs @ rhs,
      lambda lhs=lhs64_np, rhs=rhs64_np: lhs @ rhs,
      rtol=1e-3,
      atol=1e-3,
    ),
  ]
  return prefix_groups("complex", cases)
