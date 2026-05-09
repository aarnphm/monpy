# fmt: off
# ruff: noqa
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
import time
import warnings

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import monpy as _monpy
import monumpy as mnp
import numpy as np
import numpy.testing as npt

from tqdm import tqdm as _tqdm

BenchFn = Callable[[], Any]


@dataclass(frozen=True, slots=True)
class BenchCase:
  group: str
  name: str
  monpy_fn: BenchFn
  numpy_fn: BenchFn
  rtol: float = 1e-4
  atol: float = 1e-4
  check_values: bool = True
  check_dtype: bool = True


@dataclass(frozen=True, slots=True)
class BenchSample:
  round_index: int
  monpy_us: float
  numpy_us: float
  ratio: float


@dataclass(frozen=True, slots=True)
class BenchResult:
  group: str
  name: str
  rounds: int
  monpy_median_us: float
  numpy_median_us: float
  ratio_median: float
  monpy_min_us: float
  monpy_max_us: float
  numpy_min_us: float
  numpy_max_us: float
  ratio_min: float
  ratio_max: float
  samples: tuple[BenchSample, ...]


class _NoProgress:
  def __enter__(self) -> _NoProgress: return self

  def __exit__(self, exc_type: object, exc: object, tb: object) -> None: return None

  def update(self, n: int = 1) -> None: return None

  def set_postfix_str(self, s: str, refresh: bool = True) -> None: return None


def progress_bar(*, total: int, enabled: bool) -> Any:
  if not enabled: return _NoProgress()
  return _tqdm(total=total, desc="bench", dynamic_ncols=True, unit="case", leave=False, file=sys.stderr)


def positive_int(value: str) -> int:
  parsed = int(value)
  if parsed < 1: raise argparse.ArgumentTypeError("must be >= 1")
  return parsed


def parse_sizes(value: str) -> tuple[int, ...]:
  sizes = tuple(positive_int(part.strip()) for part in value.split(",") if part.strip())
  if not sizes: raise argparse.ArgumentTypeError("must include at least one size")
  return sizes


def force_monpy(value: Any) -> Any:
  if not isinstance(value, mnp.ndarray):
    try: _ = getattr(value, "_native")
    except AttributeError: pass
  return value


@contextmanager
def suppress_known_numpy_linalg_warnings() -> Iterator[None]:
  with warnings.catch_warnings():
    warnings.filterwarnings(
      "ignore",
      category=RuntimeWarning,
      message="overflow encountered in cast",
      module=r"numpy\.linalg\._linalg",
    )
    yield


def call_bench_fn(fn: BenchFn) -> object:
  with suppress_known_numpy_linalg_warnings(): return fn()


def time_call(fn: BenchFn, *, loops: int, repeats: int, force: Callable[[object], object] | None = None) -> float:
  samples: list[float] = []
  for _ in range(repeats):
    start = time.perf_counter()
    for _ in range(loops):
      result = call_bench_fn(fn)
      if force is not None:
        result = force(result)
      if result is None:
        raise RuntimeError("benchmark function unexpectedly returned None")
    samples.append((time.perf_counter() - start) / loops)
  return statistics.median(samples)


def verify_same(monpy_value: object, numpy_value: object, *, rtol: float = 1e-5, atol: float = 1e-6) -> None:
  npt.assert_allclose(np.asarray(monpy_value), np.asarray(numpy_value), rtol=rtol, atol=atol)


def verify_shape_dtype(monpy_value: object, numpy_value: object, *, check_dtype: bool = True) -> None:
  monpy_array = np.asarray(monpy_value)
  numpy_array = np.asarray(numpy_value)
  if monpy_array.shape != numpy_array.shape:
    raise AssertionError(f"shape mismatch: {monpy_array.shape!r} != {numpy_array.shape!r}")
  if check_dtype and monpy_array.dtype != numpy_array.dtype:
    raise AssertionError(f"dtype mismatch: {monpy_array.dtype!r} != {numpy_array.dtype!r}")


def ratio_value(monpy_us: float, numpy_us: float) -> float:
  return math.inf if numpy_us == 0.0 else monpy_us / numpy_us


def scaled_matrix(size: int, dtype: type[np.floating]) -> np.ndarray:
  return (np.arange(size * size, dtype=dtype).reshape(size, size) / 1000).astype(dtype, copy=False)


def well_conditioned_matrix(size: int, dtype: type[np.floating]) -> np.ndarray:
  values = np.arange(size * size, dtype=np.float64).reshape(size, size) / 11.0
  values += np.eye(size, dtype=np.float64) * float(size + 3)
  return values.astype(dtype)


def dtype_name(dtype: type[np.floating]) -> str:
  return "f32" if dtype is np.float32 else "f64"


def monpy_dtype(dtype: type[np.floating]) -> object:
  return mnp.float32 if dtype is np.float32 else mnp.float64


def build_cases(
  *,
  vector_size: int,
  vector_sizes: Sequence[int],
  matrix_sizes: Sequence[int],
  linalg_sizes: Sequence[int],
) -> list[BenchCase]:
  x_np = np.linspace(0.1, 2.0, vector_size, dtype=np.float32)
  y_np = np.linspace(2.0, 4.0, vector_size, dtype=np.float32)
  x_mp: Any = mnp.asarray(x_np.tolist(), dtype=mnp.float32)
  y_mp: Any = mnp.asarray(y_np.tolist(), dtype=mnp.float32)
  add_out_np = np.empty_like(x_np)
  add_out_mp = mnp.empty(x_mp.shape, dtype=mnp.float32)
  dtype_specs = (
    ("bool", mnp.bool, np.bool_, (np.arange(vector_size) % 2) == 0),
    ("i64", mnp.int64, np.int64, np.arange(vector_size, dtype=np.int64) - (vector_size // 2)),
    ("f32", mnp.float32, np.float32, x_np),
    ("f64", mnp.float64, np.float64, np.linspace(-2.0, 2.0, vector_size, dtype=np.float64)),
  )
  dtype_arrays = tuple(
    (name, monpy_dt, numpy_dt, mnp.asarray(values, dtype=monpy_dt, copy=False), np.asarray(values, dtype=numpy_dt))
    for name, monpy_dt, numpy_dt, values in dtype_specs
  )

  def extension_binary_out_f32() -> object:
    _monpy._native.binary_into(add_out_mp._native, x_mp._native, y_mp._native, _monpy.OP_ADD)
    return add_out_mp

  small_x = mnp.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=mnp.float32)
  small_y = mnp.asarray([10, 20, 30, 40, 50, 60, 70, 80], dtype=mnp.float32)
  layout_out = small_x + small_y
  verify_same(layout_out, np.asarray([11, 22, 33, 44, 55, 66, 77, 88], dtype=np.float32))

  m_np = np.arange(256, dtype=np.float32).reshape(16, 16)
  n_np = np.arange(256, dtype=np.float32).reshape(16, 16)
  row_np = np.ones(16, dtype=np.float32)
  m_mp: Any = mnp.asarray(m_np.tolist(), dtype=mnp.float32)
  n_mp: Any = mnp.asarray(n_np.tolist(), dtype=mnp.float32)
  row_mp: Any = mnp.asarray(row_np.tolist(), dtype=mnp.float32)

  matmul_probe_size = max(64, max(matrix_sizes))
  matmul_probe_np = scaled_matrix(matmul_probe_size, np.float32)
  matmul_probe_mp = mnp.asarray(matmul_probe_np.tolist(), dtype=mnp.float32)
  matmul_probe: Any = matmul_probe_mp @ matmul_probe_mp
  if not matmul_probe._native.used_accelerate():
    raise AssertionError("expected contiguous float32 matmul to exercise the Apple Accelerate path")
  verify_same(matmul_probe, matmul_probe_np @ matmul_probe_np, rtol=1e-4, atol=1e-4)

  matmul_f64_probe_np = scaled_matrix(matmul_probe_size, np.float64)
  matmul_f64_probe_mp = mnp.asarray(matmul_f64_probe_np.tolist(), dtype=mnp.float64)
  matmul_f64_probe: Any = matmul_f64_probe_mp @ matmul_f64_probe_mp
  if not matmul_f64_probe._native.used_accelerate():
    raise AssertionError("expected contiguous float64 matmul to exercise the Apple Accelerate dgemm path")
  verify_same(matmul_f64_probe, matmul_f64_probe_np @ matmul_f64_probe_np, rtol=1e-9, atol=1e-9)

  fused_probe: Any = mnp.sin_add_mul(x_mp, y_mp, 3.0)
  if not fused_probe._native.used_fused():
    raise AssertionError("expected sin_add_mul to exercise the native fused kernel path")
  verify_same(fused_probe, np.sin(x_np) + y_np * 3.0, rtol=1e-4, atol=1e-4)

  eager_fused_probe: Any = mnp.sin(x_mp) + y_mp * 3.0
  if not eager_fused_probe._native.used_fused():
    raise AssertionError("expected sin(x) + y * scalar to lower into the native fused kernel path")
  verify_same(eager_fused_probe, np.sin(x_np) + y_np * 3.0, rtol=1e-4, atol=1e-4)

  cases = [
    BenchCase(
      "dtype",
      "dtype_itemsize_f32",
      lambda: mnp.float32.itemsize,
      lambda: np.dtype(np.float32).itemsize,
    ),
    BenchCase(
      "dtype",
      "promote_types_i64_f32",
      lambda: mnp.promote_types(mnp.int64, mnp.float32).itemsize,
      lambda: np.promote_types(np.int64, np.float32).itemsize,
    ),
    BenchCase(
      "dtype",
      "result_type_array_scalar_f32",
      lambda: mnp.result_type(x_mp, 1.5).itemsize,
      lambda: np.result_type(x_np, 1.5).itemsize,
    ),
    BenchCase(
      "dtype",
      "can_cast_i64_f64_safe",
      lambda: mnp.can_cast(mnp.int64, mnp.float64, casting="safe"),
      lambda: np.can_cast(np.int64, np.float64, casting="safe"),
    ),
    BenchCase(
      "dtype",
      "finfo_f32_eps",
      lambda: mnp.finfo(mnp.float32).eps,
      lambda: float(np.finfo(np.float32).eps),
    ),
    BenchCase(
      "dtype",
      "iinfo_i64_bits",
      lambda: mnp.iinfo(mnp.int64).bits,
      lambda: np.iinfo(np.int64).bits,
    ),
    BenchCase(
      "interop",
      "asarray_zero_copy_f32",
      lambda: mnp.asarray(x_np, dtype=mnp.float32, copy=False),
      lambda: np.asarray(x_np),
    ),
    BenchCase(
      "interop",
      "array_copy_f32",
      lambda: mnp.array(x_np, dtype=mnp.float32, copy=True),
      lambda: np.array(x_np, dtype=np.float32, copy=True),
    ),
    BenchCase(
      "interop",
      "from_dlpack_f32",
      lambda: mnp.from_dlpack(x_np, copy=False),
      lambda: np.from_dlpack(x_np, copy=False),
    ),
    BenchCase("elementwise", "unary_sin_f32", lambda: mnp.sin(x_mp), lambda: np.sin(x_np)),
    BenchCase("elementwise", "binary_add_f32", lambda: x_mp + y_mp, lambda: x_np + y_np),
    BenchCase(
      "elementwise",
      "binary_add_out_f32",
      lambda: mnp.add(x_mp, y_mp, out=add_out_mp),
      lambda: np.add(x_np, y_np, out=add_out_np),
    ),
    BenchCase(
      "elementwise",
      "binary_add_extension_out_f32",
      extension_binary_out_f32,
      lambda: np.add(x_np, y_np, out=add_out_np),
    ),
    BenchCase("broadcast", "broadcast_add_f32", lambda: m_mp + row_mp, lambda: m_np + row_np),
    BenchCase("views", "strided_view_f32", lambda: x_mp[::-2], lambda: x_np[::-2]),
    BenchCase("views", "reversed_add_f32", lambda: x_mp[::-1] + y_mp[::-1], lambda: x_np[::-1] + y_np[::-1]),
    BenchCase("views", "transpose_add_f32", lambda: m_mp.T + n_mp.T, lambda: m_np.T + n_np.T),
    BenchCase("reductions", "sum_f32", lambda: mnp.sum(x_mp), lambda: np.sum(x_np)),
    BenchCase(
      "expressions", "eager_expression_f32", lambda: mnp.sin(x_mp) + y_mp * 3.0, lambda: np.sin(x_np) + y_np * 3.0
    ),
    BenchCase(
      "expressions",
      "fused_sin_add_mul_f32",
      lambda: mnp.sin_add_mul(x_mp, y_mp, 3.0),
      lambda: np.sin(x_np) + y_np * 3.0,
    ),
  ]

  for name, monpy_dt, numpy_dt, values_mp, values_np in dtype_arrays:
    if name != "f32":
      cases.extend([
        BenchCase(
          "interop",
          f"asarray_zero_copy_{name}",
          lambda src=values_np, dt=monpy_dt: mnp.asarray(src, dtype=dt, copy=False),
          lambda src=values_np: np.asarray(src),
        ),
        BenchCase(
          "interop",
          f"array_copy_{name}",
          lambda src=values_np, dt=monpy_dt: mnp.array(src, dtype=dt, copy=True),
          lambda src=values_np, dt=numpy_dt: np.array(src, dtype=dt, copy=True),
        ),
      ])
    for dst_name, dst_monpy_dt, dst_numpy_dt, _, _ in dtype_arrays:
      cases.append(
        BenchCase(
          "casts",
          f"astype_{name}_to_{dst_name}",
          lambda src=values_mp, dt=dst_monpy_dt: src.astype(dt),
          lambda src=values_np, dt=dst_numpy_dt: src.astype(dt),
        )
      )

  helper_np = np.arange(16 * 16, dtype=np.float32).reshape(16, 16).T
  helper_mp = mnp.asarray(helper_np, dtype=mnp.float32, copy=False)
  cases.extend([
    BenchCase(
      "creation",
      "empty_like_shape_override_f32",
      lambda: mnp.empty_like(helper_mp, dtype=mnp.float64, shape=(8, 8)),
      lambda: np.empty_like(helper_np, dtype=np.float64, shape=(8, 8)),
      check_values=False,
    ),
    BenchCase(
      "creation",
      "zeros_like_transpose_f32",
      lambda: mnp.zeros_like(helper_mp),
      lambda: np.zeros_like(helper_np),
    ),
    BenchCase(
      "creation",
      "ones_like_transpose_f32",
      lambda: mnp.ones_like(helper_mp),
      lambda: np.ones_like(helper_np),
    ),
    BenchCase(
      "creation",
      "full_like_transpose_f32",
      lambda: mnp.full_like(helper_mp, 7.0),
      lambda: np.full_like(helper_np, 7.0),
    ),
    BenchCase(
      "copy",
      "copy_transpose_f32",
      lambda: mnp.copy(helper_mp),
      lambda: np.copy(helper_np),
    ),
    BenchCase(
      "copy",
      "ascontiguousarray_dense_f32",
      lambda: mnp.ascontiguousarray(x_mp),
      lambda: np.ascontiguousarray(x_np),
    ),
    BenchCase(
      "copy",
      "ascontiguousarray_transpose_f32",
      lambda: mnp.ascontiguousarray(helper_mp),
      lambda: np.ascontiguousarray(helper_np),
    ),
    BenchCase(
      "copy",
      "ascontiguousarray_scalar_f64",
      lambda: mnp.ascontiguousarray(3.5),
      lambda: np.ascontiguousarray(3.5),
    ),
    BenchCase(
      "views",
      "newaxis_middle_f32",
      lambda: helper_mp[:, None, :],
      lambda: helper_np[:, None, :],
    ),
    BenchCase(
      "views",
      "expand_dims_tuple_f32",
      lambda: mnp.expand_dims(helper_mp, axis=(0, -1)),
      lambda: np.expand_dims(helper_np, axis=(0, -1)),
    ),
    BenchCase(
      "views",
      "flip_axis0_f32",
      lambda: mnp.flip(helper_mp, axis=0),
      lambda: np.flip(helper_np, axis=0),
    ),
    BenchCase(
      "views",
      "flip_all_f32",
      lambda: mnp.flip(helper_mp),
      lambda: np.flip(helper_np),
    ),
    BenchCase(
      "views",
      "fliplr_f32",
      lambda: mnp.fliplr(helper_mp),
      lambda: np.fliplr(helper_np),
    ),
    BenchCase(
      "views",
      "rot90_k1_f32",
      lambda: mnp.rot90(helper_mp, k=1),
      lambda: np.rot90(helper_np, k=1),
    ),
    BenchCase(
      "views",
      "rot90_k2_f32",
      lambda: mnp.rot90(helper_mp, k=2),
      lambda: np.rot90(helper_np, k=2),
    ),
  ])

  diag_np = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
  diag_mp = mnp.asarray(diag_np.tolist(), dtype=mnp.float64)
  cases.extend([
    BenchCase("views", "diagonal_64_f64", lambda: mnp.diagonal(diag_mp), lambda: np.diagonal(diag_np)),
    BenchCase("reductions", "trace_64_f64", lambda: mnp.trace(diag_mp), lambda: np.trace(diag_np)),
  ])

  # shape manipulation + creation helpers.
  s_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
  s_mp = mnp.asarray(s_np)
  squeeze_np = np.zeros((1, 4, 1, 5), dtype=np.float32)
  squeeze_mp = mnp.asarray(squeeze_np)
  vec_a_np = np.arange(8, dtype=np.float32)
  vec_a_mp = mnp.asarray(vec_a_np)
  vec_b_np = np.arange(8, 16, dtype=np.float32)
  vec_b_mp = mnp.asarray(vec_b_np)
  cases.extend([
    BenchCase("views", "squeeze_axis0_f32", lambda: mnp.squeeze(squeeze_mp, axis=0), lambda: np.squeeze(squeeze_np, axis=0)),
    BenchCase("interop", "asarray_squeeze_axis0_f32", lambda: mnp.squeeze(mnp.asarray(squeeze_np), axis=0), lambda: np.squeeze(np.asarray(squeeze_np), axis=0)),
    BenchCase("views", "moveaxis_f32", lambda: mnp.moveaxis(s_mp, 0, -1), lambda: np.moveaxis(s_np, 0, -1)),
    BenchCase("views", "swapaxes_f32", lambda: mnp.swapaxes(s_mp, 0, 2), lambda: np.swapaxes(s_np, 0, 2)),
    BenchCase("views", "ravel_f32", lambda: mnp.ravel(s_mp), lambda: np.ravel(s_np)),
    BenchCase("views", "flatten_f32", lambda: mnp.flatten(s_mp) if hasattr(mnp, "flatten") else mnp.ravel(s_mp), lambda: s_np.flatten()),
    BenchCase("views", "concatenate_axis0_f32", lambda: mnp.concatenate([vec_a_mp, vec_b_mp]), lambda: np.concatenate([vec_a_np, vec_b_np])),
    BenchCase("views", "stack_axis0_f32", lambda: mnp.stack([vec_a_mp, vec_b_mp]), lambda: np.stack([vec_a_np, vec_b_np])),
    BenchCase("views", "hstack_f32", lambda: mnp.hstack([vec_a_mp, vec_b_mp]), lambda: np.hstack([vec_a_np, vec_b_np])),
    BenchCase("views", "vstack_f32", lambda: mnp.vstack([vec_a_mp, vec_b_mp]), lambda: np.vstack([vec_a_np, vec_b_np])),
    BenchCase("creation", "eye_64_f32", lambda: mnp.eye(64, dtype=mnp.float32), lambda: np.eye(64, dtype=np.float32)),
    BenchCase("creation", "identity_64_f32", lambda: mnp.identity(64, dtype=mnp.float32), lambda: np.identity(64, dtype=np.float32)),
    BenchCase("creation", "tri_64_f32", lambda: mnp.tri(64, dtype=mnp.float32), lambda: np.tri(64, dtype=np.float32)),
    BenchCase("creation", "logspace_50", lambda: mnp.logspace(0.0, 1.0, num=50), lambda: np.logspace(0.0, 1.0, num=50)),
    BenchCase("creation", "geomspace_50", lambda: mnp.geomspace(1.0, 1000.0, num=50), lambda: np.geomspace(1.0, 1000.0, num=50)),
    BenchCase("creation", "meshgrid_xy_f32", lambda: mnp.meshgrid(vec_a_mp, vec_b_mp, indexing="xy"), lambda: np.meshgrid(vec_a_np, vec_b_np, indexing="xy")),
    BenchCase("creation", "atleast_2d_f32", lambda: mnp.atleast_2d(vec_a_mp), lambda: np.atleast_2d(vec_a_np)),
    BenchCase("creation", "indices_4x4", lambda: mnp.indices((4, 4)), lambda: np.indices((4, 4))),
  ])

  for size in matrix_sizes:
    for dtype in (np.float32, np.float64):
      suffix = dtype_name(dtype)
      target = monpy_dtype(dtype)
      lhs_np = scaled_matrix(size, dtype)
      rhs_np = scaled_matrix(size, dtype)
      lhs_mp = mnp.asarray(lhs_np.tolist(), dtype=target)
      rhs_mp = mnp.asarray(rhs_np.tolist(), dtype=target)
      vec_np = np.linspace(0.1, 1.0, size, dtype=dtype)
      vec_mp = mnp.asarray(vec_np.tolist(), dtype=target)
      lhs_f_np = np.asfortranarray(lhs_np)
      lhs_f_mp = mnp.asarray(lhs_f_np, dtype=target, copy=False)
      rhs_t_np = rhs_np.T
      rhs_t_mp = mnp.asarray(rhs_t_np, dtype=target, copy=False)
      tolerance = 1e-4 if dtype is np.float32 else 1e-9

      cases.extend([
        BenchCase(
          "matmul",
          f"matmul_{size}_{suffix}",
          lambda lhs=lhs_mp, rhs=rhs_mp: lhs @ rhs,
          lambda lhs=lhs_np, rhs=rhs_np: lhs @ rhs,
          rtol=tolerance,
          atol=tolerance,
        ),
        BenchCase(
          "matmul",
          f"matvec_{size}_{suffix}",
          lambda lhs=lhs_mp, vec=vec_mp: lhs @ vec,
          lambda lhs=lhs_np, vec=vec_np: lhs @ vec,
          rtol=tolerance,
          atol=tolerance,
        ),
        BenchCase(
          "matmul",
          f"vecmat_{size}_{suffix}",
          lambda vec=vec_mp, rhs=rhs_mp: vec @ rhs,
          lambda vec=vec_np, rhs=rhs_np: vec @ rhs,
          rtol=tolerance,
          atol=tolerance,
        ),
        BenchCase(
          "matmul",
          f"matmul_lhs_f_{size}_{suffix}",
          lambda lhs=lhs_f_mp, rhs=rhs_mp: lhs @ rhs,
          lambda lhs=lhs_f_np, rhs=rhs_np: lhs @ rhs,
          rtol=tolerance,
          atol=tolerance,
        ),
        BenchCase(
          "matmul",
          f"matmul_rhs_t_{size}_{suffix}",
          lambda lhs=lhs_mp, rhs=rhs_t_mp: lhs @ rhs,
          lambda lhs=lhs_np, rhs=rhs_t_np: lhs @ rhs,
          rtol=tolerance,
          atol=tolerance,
        ),
      ])

  # Bandwidth-regime elementwise. Cases past 16K elements should land
  # within ~5% of numpy on a single core: the kernel is memory-bandwidth
  # bound and threading would just contend on the same DRAM channels.
  # Wrapper overhead amortizes away once the work-per-call is large enough
  # to absorb it. Anything below 16K stays wrapper-bound and is covered by
  # the existing 1024-element cases.
  for size in vector_sizes:
    if size == vector_size:
      continue
    bw_x_np = np.linspace(0.1, 2.0, size, dtype=np.float32)
    bw_y_np = np.linspace(2.0, 4.0, size, dtype=np.float32)
    bw_x_mp: Any = mnp.asarray(bw_x_np, dtype=mnp.float32, copy=False)
    bw_y_mp: Any = mnp.asarray(bw_y_np, dtype=mnp.float32, copy=False)
    bw_out_np = np.empty_like(bw_x_np)
    bw_out_mp = mnp.empty(bw_x_mp.shape, dtype=mnp.float32)
    suffix = f"{size}_f32"
    cases.extend([
      BenchCase(
        "bandwidth",
        f"binary_add_{suffix}",
        lambda lhs=bw_x_mp, rhs=bw_y_mp: lhs + rhs,
        lambda lhs=bw_x_np, rhs=bw_y_np: lhs + rhs,
      ),
      BenchCase(
        "bandwidth",
        f"binary_add_out_{suffix}",
        lambda lhs=bw_x_mp, rhs=bw_y_mp, out=bw_out_mp: mnp.add(lhs, rhs, out=out),
        lambda lhs=bw_x_np, rhs=bw_y_np, out=bw_out_np: np.add(lhs, rhs, out=out),
      ),
      BenchCase(
        "bandwidth",
        f"unary_sin_{suffix}",
        lambda src=bw_x_mp: mnp.sin(src),
        lambda src=bw_x_np: np.sin(src),
      ),
      BenchCase(
        "bandwidth",
        f"reduce_sum_{suffix}",
        lambda src=bw_x_mp: mnp.sum(src),
        lambda src=bw_x_np: np.sum(src),
      ),
      BenchCase(
        "bandwidth",
        f"fused_sin_add_mul_{suffix}",
        lambda lhs=bw_x_mp, rhs=bw_y_mp: mnp.sin_add_mul(lhs, rhs, 3.0),
        lambda lhs=bw_x_np, rhs=bw_y_np: np.sin(lhs) + rhs * 3.0,
      ),
      BenchCase(
        "bandwidth",
        f"reversed_add_{suffix}",
        lambda lhs=bw_x_mp, rhs=bw_y_mp: lhs[::-1] + rhs[::-1],
        lambda lhs=bw_x_np, rhs=bw_y_np: lhs[::-1] + rhs[::-1],
      ),
    ])

  for size in linalg_sizes:
    for dtype in (np.float32, np.float64):
      suffix = dtype_name(dtype)
      target = monpy_dtype(dtype)
      lhs_np = well_conditioned_matrix(size, dtype)
      rhs_np = np.arange(1, size + 1, dtype=dtype)
      lhs_mp = mnp.asarray(lhs_np.tolist(), dtype=target)
      rhs_mp = mnp.asarray(rhs_np.tolist(), dtype=target)
      tolerance = 1e-4 if dtype is np.float32 else 1e-8

      cases.extend([
        BenchCase(
          "linalg",
          f"solve_{size}_{suffix}",
          lambda lhs=lhs_mp, rhs=rhs_mp: mnp.linalg.solve(lhs, rhs),
          lambda lhs=lhs_np, rhs=rhs_np: np.linalg.solve(lhs, rhs),
          rtol=tolerance,
          atol=tolerance,
        ),
        BenchCase(
          "linalg",
          f"inv_{size}_{suffix}",
          lambda lhs=lhs_mp: mnp.linalg.inv(lhs),
          lambda lhs=lhs_np: np.linalg.inv(lhs),
          rtol=tolerance,
          atol=tolerance,
        ),
        BenchCase(
          "linalg",
          f"det_{size}_{suffix}",
          lambda lhs=lhs_mp: mnp.linalg.det(lhs),
          lambda lhs=lhs_np: np.linalg.det(lhs),
          rtol=tolerance,
          atol=tolerance,
        ),
      ])

  # LAPACK-backed decompositions. Sizes use a subset of linalg_sizes;
  # SVD/EIG are quadratic, so cap at 64 to keep runtime reasonable.
  decomp_sizes = tuple(s for s in linalg_sizes if s <= 64) or linalg_sizes[:1]
  for size in decomp_sizes:
    for dtype in (np.float32, np.float64):
      suffix = dtype_name(dtype)
      target = monpy_dtype(dtype)
      qr_np = well_conditioned_matrix(size, dtype)
      qr_mp = mnp.asarray(qr_np, dtype=target, copy=True)
      psd_np = qr_np @ qr_np.T + size * np.eye(size, dtype=dtype)
      psd_mp = mnp.asarray(psd_np, dtype=target, copy=True)
      sym_np = (qr_np + qr_np.T) / 2
      sym_mp = mnp.asarray(sym_np, dtype=target, copy=True)
      tolerance = 1e-3 if dtype is np.float32 else 1e-8

      cases.extend([
        BenchCase(
          "decomp",
          f"qr_{size}_{suffix}",
          lambda x=qr_mp: mnp.linalg.qr(x),
          lambda x=qr_np: np.linalg.qr(x),
          check_values=False,
        ),
        BenchCase(
          "decomp",
          f"cholesky_{size}_{suffix}",
          lambda x=psd_mp: mnp.linalg.cholesky(x),
          lambda x=psd_np: np.linalg.cholesky(x),
          rtol=tolerance,
          atol=tolerance,
        ),
        BenchCase(
          "decomp",
          f"eigvalsh_{size}_{suffix}",
          lambda x=sym_mp: mnp.linalg.eigvalsh(x),
          lambda x=sym_np: np.linalg.eigvalsh(x),
          check_values=False,
        ),
        BenchCase(
          "decomp",
          f"svdvals_{size}_{suffix}",
          lambda x=qr_mp: mnp.linalg.svdvals(x),
          lambda x=qr_np: np.linalg.svdvals(x),
          check_values=False,
        ),
        BenchCase(
          "decomp",
          f"pinv_{size}_{suffix}",
          lambda x=qr_mp: mnp.linalg.pinv(x),
          lambda x=qr_np: np.linalg.pinv(x),
          rtol=tolerance,
          atol=tolerance,
        ),
      ])

  # Public linalg API breadth. The sized decomp rows above track scaling for
  # the heavy LAPACK-backed primitives; these fixed f64 rows keep every public
  # linalg entrypoint visible in the ratio table, including wrappers that tend
  # to regress through Python-side shape/rank plumbing.
  la_vec_np = np.linspace(0.25, 2.25, 32, dtype=np.float64)
  la_vec_b_np = np.linspace(2.25, 4.25, 32, dtype=np.float64)
  la_vec_mp = mnp.asarray(la_vec_np, dtype=mnp.float64, copy=True)
  la_vec_b_mp = mnp.asarray(la_vec_b_np, dtype=mnp.float64, copy=True)
  la_vec_2d_mp = mnp.reshape(la_vec_mp, (8, 4))
  la_vec_b_2d_mp = mnp.reshape(la_vec_b_mp, (8, 4))
  la_vec_2d_np = la_vec_np.reshape(8, 4)
  la_vec_b_2d_np = la_vec_b_np.reshape(8, 4)
  la_mat16_np = well_conditioned_matrix(16, np.float64)
  la_mat16_mp = mnp.asarray(la_mat16_np, dtype=mnp.float64, copy=True)
  la_rhs16_np = np.linspace(0.5, 8.5, 16, dtype=np.float64)
  la_rhs16_mp = mnp.asarray(la_rhs16_np, dtype=mnp.float64, copy=True)
  la_rhs16_2_np = np.stack([la_rhs16_np, la_rhs16_np + 1.0], axis=1)
  la_rhs16_2_mp = mnp.asarray(la_rhs16_2_np, dtype=mnp.float64, copy=True)
  la_a_np = np.arange(20, dtype=np.float64).reshape(4, 5) / 7.0
  la_b_np = np.arange(15, dtype=np.float64).reshape(5, 3) / 5.0
  la_a_mp = mnp.asarray(la_a_np, dtype=mnp.float64, copy=True)
  la_b_mp = mnp.asarray(la_b_np, dtype=mnp.float64, copy=True)
  la_small_np = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
  la_small_mp = mnp.asarray(la_small_np, dtype=mnp.float64, copy=True)
  la_rank_np = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]], dtype=np.float64)
  la_rank_mp = mnp.asarray(la_rank_np, dtype=mnp.float64, copy=True)
  la_multi_a_np = np.array([[1.0, 2.0]], dtype=np.float64)
  la_multi_b_np = la_small_np
  la_multi_c_np = np.array([[3.0], [4.0]], dtype=np.float64)
  la_multi_a_mp = mnp.asarray(la_multi_a_np, dtype=mnp.float64, copy=True)
  la_multi_b_mp = mnp.asarray(la_multi_b_np, dtype=mnp.float64, copy=True)
  la_multi_c_mp = mnp.asarray(la_multi_c_np, dtype=mnp.float64, copy=True)
  la_tensor_inv_np = np.eye(4, dtype=np.float64).reshape(2, 2, 2, 2)
  la_tensor_inv_mp = mnp.asarray(la_tensor_inv_np, dtype=mnp.float64, copy=True)
  la_tensor_solve_np = well_conditioned_matrix(6, np.float64).reshape(2, 3, 2, 3)
  la_tensor_solve_mp = mnp.asarray(la_tensor_solve_np, dtype=mnp.float64, copy=True)
  la_tensor_rhs_np = np.linspace(1.0, 6.0, 6, dtype=np.float64).reshape(2, 3)
  la_tensor_rhs_mp = mnp.asarray(la_tensor_rhs_np, dtype=mnp.float64, copy=True)
  la_rect_np = np.linspace(0.2, 6.4, 32, dtype=np.float64).reshape(8, 4)
  la_rect_mp = mnp.asarray(la_rect_np, dtype=mnp.float64, copy=True)
  la_rect_rhs_np = np.linspace(1.0, 8.0, 8, dtype=np.float64)
  la_rect_rhs_mp = mnp.asarray(la_rect_rhs_np, dtype=mnp.float64, copy=True)
  la_rect_rcond = np.finfo(np.float64).eps * max(la_rect_np.shape)
  la_tri_np = np.diag([1.0, 2.0, 3.0, 4.0]) + np.triu(np.ones((4, 4), dtype=np.float64) * 0.05, 1)
  la_tri_mp = mnp.asarray(la_tri_np, dtype=mnp.float64, copy=True)
  la_cross_a_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
  la_cross_b_np = np.array([4.0, 5.0, 6.0], dtype=np.float64)
  la_cross_a_mp = mnp.asarray(la_cross_a_np, dtype=mnp.float64, copy=True)
  la_cross_b_mp = mnp.asarray(la_cross_b_np, dtype=mnp.float64, copy=True)
  np_linalg_vecdot: Any = np.linalg.vecdot
  cases.extend([
    BenchCase("linalg_api", "dot_1d_32_f64", lambda: mnp.linalg.dot(la_vec_mp, la_vec_b_mp), lambda: np.dot(la_vec_np, la_vec_b_np), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "vdot_32_f64", lambda: mnp.linalg.vdot(la_vec_mp, la_vec_b_mp), lambda: np.vdot(la_vec_np, la_vec_b_np), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "inner_32_f64", lambda: mnp.linalg.inner(la_vec_mp, la_vec_b_mp), lambda: np.inner(la_vec_np, la_vec_b_np), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "outer_32_f64", lambda: mnp.linalg.outer(la_vec_mp, la_vec_b_mp), lambda: np.linalg.outer(la_vec_np, la_vec_b_np), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "matmul_16_f64", lambda: mnp.linalg.matmul(la_mat16_mp, la_mat16_mp), lambda: np.linalg.matmul(la_mat16_np, la_mat16_np), rtol=1e-9, atol=1e-9),
    BenchCase("linalg_api", "matvec_16_f64", lambda: mnp.linalg.matvec(la_mat16_mp, la_rhs16_mp), lambda: np.matvec(la_mat16_np, la_rhs16_np), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "vecmat_16_f64", lambda: mnp.linalg.vecmat(la_rhs16_mp, la_mat16_mp), lambda: np.vecmat(la_rhs16_np, la_mat16_np), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "vecdot_axis1_8x4_f64", lambda: mnp.linalg.vecdot(la_vec_2d_mp, la_vec_b_2d_mp, axis=1), lambda: np_linalg_vecdot(la_vec_2d_np, la_vec_b_2d_np, axis=1), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "tensordot_axes1_4x5_5x3_f64", lambda: mnp.linalg.tensordot(la_a_mp, la_b_mp, axes=1), lambda: np.linalg.tensordot(la_a_np, la_b_np, axes=1), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "kron_2x2_f64", lambda: mnp.linalg.kron(la_small_mp, la_small_mp), lambda: np.kron(la_small_np, la_small_np), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "cross_3_f64", lambda: mnp.linalg.cross(la_cross_a_mp, la_cross_b_mp), lambda: np.linalg.cross(la_cross_a_np, la_cross_b_np), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "trace_16_f64", lambda: mnp.linalg.trace(la_mat16_mp), lambda: np.linalg.trace(la_mat16_np), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "matrix_transpose_16_f64", lambda: mnp.linalg.matrix_transpose(la_mat16_mp), lambda: np.linalg.matrix_transpose(la_mat16_np), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "norm_vec2_32_f64", lambda: mnp.linalg.norm(la_vec_mp), lambda: np.linalg.norm(la_vec_np), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "norm_vec1_32_f64", lambda: mnp.linalg.norm(la_vec_mp, ord=1), lambda: np.linalg.norm(la_vec_np, ord=1), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "vector_norm_axis1_8x4_f64", lambda: mnp.linalg.vector_norm(la_vec_2d_mp, axis=1), lambda: np.linalg.vector_norm(la_vec_2d_np, axis=1), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "matrix_norm_fro_16_f64", lambda: mnp.linalg.matrix_norm(la_mat16_mp), lambda: np.linalg.matrix_norm(la_mat16_np), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "matrix_rank_3_f64", lambda: mnp.linalg.matrix_rank(la_rank_mp), lambda: np.linalg.matrix_rank(la_rank_np), check_dtype=False),
    BenchCase("linalg_api", "matrix_power_2_n3_f64", lambda: mnp.linalg.matrix_power(la_small_mp, 3), lambda: np.linalg.matrix_power(la_small_np, 3), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "slogdet_16_f64", lambda: mnp.linalg.slogdet(la_mat16_mp)[1], lambda: np.linalg.slogdet(la_mat16_np)[1], rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "multi_dot_2x2_f64", lambda: mnp.linalg.multi_dot([la_multi_a_mp, la_multi_b_mp, la_multi_c_mp]), lambda: np.linalg.multi_dot([la_multi_a_np, la_multi_b_np, la_multi_c_np]), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "tensorinv_2x2x2x2_f64", lambda: mnp.linalg.tensorinv(la_tensor_inv_mp), lambda: np.linalg.tensorinv(la_tensor_inv_np), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "tensorsolve_2x3x2x3_f64", lambda: mnp.linalg.tensorsolve(la_tensor_solve_mp, la_tensor_rhs_mp), lambda: np.linalg.tensorsolve(la_tensor_solve_np, la_tensor_rhs_np), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "solve_matrix_rhs_16_f64", lambda: mnp.linalg.solve(la_mat16_mp, la_rhs16_2_mp), lambda: np.linalg.solve(la_mat16_np, la_rhs16_2_np), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "qr_r_8x4_f64", lambda: mnp.linalg.qr(la_rect_mp, mode="r"), lambda: np.linalg.qr(la_rect_np, mode="r"), check_values=False),
    BenchCase("linalg_api", "eigh_full_2_f64", lambda: mnp.linalg.eigh(la_small_mp)[0], lambda: np.linalg.eigh(la_small_np)[0], rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "eigvals_4_f64", lambda: mnp.linalg.eigvals(la_tri_mp), lambda: np.linalg.eigvals(la_tri_np), rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "eig_full_4_f64", lambda: mnp.linalg.eig(la_tri_mp)[0], lambda: np.linalg.eig(la_tri_np)[0], rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "svd_full_8x4_f64", lambda: mnp.linalg.svd(la_rect_mp, full_matrices=False)[1], lambda: np.linalg.svd(la_rect_np, full_matrices=False)[1], check_values=False),
    BenchCase("linalg_api", "svdvals_8x4_f64", lambda: mnp.linalg.svdvals(la_rect_mp), lambda: np.linalg.svdvals(la_rect_np), check_values=False),
    BenchCase("linalg_api", "lstsq_8x4_f64", lambda: mnp.linalg.lstsq(la_rect_mp, la_rect_rhs_mp)[0], lambda: np.linalg.lstsq(la_rect_np, la_rect_rhs_np, rcond=None)[0], rtol=1e-10, atol=1e-10),
    BenchCase("linalg_api", "pinv_rect_8x4_f64", lambda: mnp.linalg.pinv(la_rect_mp, rcond=la_rect_rcond), lambda: np.linalg.pinv(la_rect_np, rcond=la_rect_rcond), rtol=1e-10, atol=1e-10),
  ])

  # Unsigned-int and float16 dtype family coverage.
  # These are wrapper-bound for now (dispatch goes through the f64 round-trip
  # path); the bench rows surface that and will track typed-kernel work.
  ext_dtype_specs = (
    ("u8", mnp.uint8, np.uint8, np.arange(vector_size, dtype=np.uint8) % 251),
    ("u16", mnp.uint16, np.uint16, np.arange(vector_size, dtype=np.uint16) % 65500),
    ("u32", mnp.uint32, np.uint32, np.arange(vector_size, dtype=np.uint32)),
    ("u64", mnp.uint64, np.uint64, np.arange(vector_size, dtype=np.uint64)),
    ("f16", mnp.float16, np.float16, np.linspace(0.1, 10.0, vector_size, dtype=np.float16)),
    ("i32", mnp.int32, np.int32, np.arange(vector_size, dtype=np.int32) - (vector_size // 2)),
  )
  for name, monpy_dt, numpy_dt, values in ext_dtype_specs:
    a_mp = mnp.asarray(values, dtype=monpy_dt, copy=True)
    a_np = np.asarray(values, dtype=numpy_dt)
    reduction_matches_numpy_dtype = name == "i32"
    cases.extend([
      BenchCase(
        "ext_dtypes",
        f"binary_add_{name}",
        lambda lhs=a_mp: lhs + lhs,
        lambda lhs=a_np: lhs + lhs,
        check_values=False,
      ),
      BenchCase(
        "ext_dtypes",
        f"reduce_sum_{name}",
        lambda src=a_mp: mnp.sum(src),
        lambda src=a_np: np.sum(src),
        check_values=False,
        check_dtype=reduction_matches_numpy_dtype,
      ),
    ])

  # Native creation kernels (eye/tril/triu/concatenate/pad).
  ec_n = 64
  ec_a_np = np.arange(ec_n * ec_n, dtype=np.float64).reshape(ec_n, ec_n)
  ec_a_mp = mnp.asarray(ec_a_np, dtype=mnp.float64, copy=True)
  ec_concat_inputs_mp = [mnp.zeros((128,), dtype=mnp.float64) for _ in range(8)]
  ec_concat_inputs_np = [np.zeros((128,), dtype=np.float64) for _ in range(8)]
  ec_pad_mp = mnp.asarray(np.arange(64, dtype=np.float64), dtype=mnp.float64)
  ec_pad_np = np.arange(64, dtype=np.float64)
  cases.extend([
    BenchCase(
      "native_kernels",
      "eye_64_native_f64",
      lambda: mnp.eye(64, dtype=mnp.float64),
      lambda: np.eye(64, dtype=np.float64),
    ),
    BenchCase(
      "native_kernels",
      "tril_64_native_f64",
      lambda lhs=ec_a_mp: mnp.tril(lhs),
      lambda lhs=ec_a_np: np.tril(lhs),
    ),
    BenchCase(
      "native_kernels",
      "triu_64_native_f64",
      lambda lhs=ec_a_mp: mnp.triu(lhs),
      lambda lhs=ec_a_np: np.triu(lhs),
    ),
    BenchCase(
      "native_kernels",
      "concatenate_axis0_8x128_f64",
      lambda inp=ec_concat_inputs_mp: mnp.concatenate(inp),
      lambda inp=ec_concat_inputs_np: np.concatenate(inp),
    ),
    BenchCase(
      "native_kernels",
      "pad_constant_5_5_f64",
      lambda inp=ec_pad_mp: mnp.pad(inp, 5),
      lambda inp=ec_pad_np: np.pad(inp, 5),
    ),
  ])

  return cases


def run_case(case: BenchCase, *, loops: int, repeats: int, round_index: int) -> BenchSample:
  monpy_result = call_bench_fn(case.monpy_fn)
  numpy_result = call_bench_fn(case.numpy_fn)
  if case.check_values:
    verify_same(monpy_result, numpy_result, rtol=case.rtol, atol=case.atol)
  else:
    verify_shape_dtype(monpy_result, numpy_result, check_dtype=case.check_dtype)
  monpy_us = time_call(case.monpy_fn, loops=loops, repeats=repeats, force=force_monpy) * 1_000_000
  numpy_us = time_call(case.numpy_fn, loops=loops, repeats=repeats) * 1_000_000
  return BenchSample(
    round_index=round_index, monpy_us=monpy_us, numpy_us=numpy_us, ratio=ratio_value(monpy_us, numpy_us)
  )


def summarize(case: BenchCase, samples: Sequence[BenchSample]) -> BenchResult:
  monpy_values = [sample.monpy_us for sample in samples]
  numpy_values = [sample.numpy_us for sample in samples]
  ratio_values = [sample.ratio for sample in samples]
  return BenchResult(
    group=case.group,
    name=case.name,
    rounds=len(samples),
    monpy_median_us=statistics.median(monpy_values),
    numpy_median_us=statistics.median(numpy_values),
    ratio_median=statistics.median(ratio_values),
    monpy_min_us=min(monpy_values),
    monpy_max_us=max(monpy_values),
    numpy_min_us=min(numpy_values),
    numpy_max_us=max(numpy_values),
    ratio_min=min(ratio_values),
    ratio_max=max(ratio_values),
    samples=tuple(samples),
  )


def run_benchmarks(
  cases: Sequence[BenchCase], *, rounds: int, loops: int, repeats: int, progress: bool
) -> list[BenchResult]:
  samples_by_case: dict[tuple[str, str], list[BenchSample]] = {(case.group, case.name): [] for case in cases}
  total = rounds * len(cases)
  with progress_bar(total=total, enabled=progress) as bar:
    for round_index in range(1, rounds + 1):
      for case in cases:
        key = (case.group, case.name)
        samples_by_case[key].append(run_case(case, loops=loops, repeats=repeats, round_index=round_index))
        try:
          bar.set_postfix_str(f"round={round_index} case={case.name}", refresh=False)
          bar.update(1)
        except AttributeError:
          pass
  return [summarize(case, samples_by_case[(case.group, case.name)]) for case in cases]


def format_us(value: float) -> str:
  return "inf" if math.isinf(value) else f"{value:.3f}"


def format_ratio(value: float) -> str:
  return "inf" if math.isinf(value) else f"{value:.3f}x"


def format_range(min_value: float, max_value: float, *, ratio: bool = False) -> str:
  formatter = format_ratio if ratio else format_us
  return f"{formatter(min_value)}..{formatter(max_value)}"


def sorted_results(results: Sequence[BenchResult], *, sort: str) -> list[BenchResult]:
  if sort == "name":
    return sorted(results, key=lambda result: (result.name, result.group))
  if sort == "monpy":
    return sorted(results, key=lambda result: result.monpy_median_us, reverse=True)
  if sort == "fastest":
    # Smallest monpy/numpy first: cases where monpy beats numpy by the largest
    # margin. The "fastest relative to numpy" reading.
    return sorted(results, key=lambda result: result.ratio_median)
  if sort in ("slowest", "ratio"):
    # Largest monpy/numpy first: regressions on top.
    return sorted(results, key=lambda result: result.ratio_median, reverse=True)
  return list(results)


def table_rows(results: Sequence[BenchResult]) -> list[tuple[str, ...]]:
  rows: list[tuple[str, ...]] = [
    ("group", "case", "monpy us", "numpy us", "monpy/numpy", "monpy range", "numpy range", "ratio range", "rounds")
  ]
  for result in results:
    rows.append((
      result.group,
      result.name,
      format_us(result.monpy_median_us),
      format_us(result.numpy_median_us),
      format_ratio(result.ratio_median),
      format_range(result.monpy_min_us, result.monpy_max_us),
      format_range(result.numpy_min_us, result.numpy_max_us),
      format_range(result.ratio_min, result.ratio_max, ratio=True),
      str(result.rounds),
    ))
  return rows


def render_table(results: Sequence[BenchResult], *, rounds: int, loops: int, repeats: int) -> str:
  rows = table_rows(results)
  widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
  numeric_columns = {2, 3, 4, 5, 6, 7, 8}

  def render_row(row: Sequence[str]) -> str:
    cells = [cell.rjust(widths[index]) if index in numeric_columns else cell.ljust(widths[index]) for index, cell in enumerate(row)]
    return " | ".join(cells)

  separator = "-+-".join("-" * width for width in widths)
  rendered = [
    f"rounds={rounds} repeats={repeats} loops={loops} unit=us/call",
    render_row(rows[0]),
    separator,
  ]
  rendered.extend(render_row(row) for row in rows[1:])
  return "\n".join(rendered)


def render_markdown(results: Sequence[BenchResult], *, rounds: int, loops: int, repeats: int) -> str:
  rows = table_rows(results)
  header = "| " + " | ".join(rows[0]) + " |"
  align = "| " + " | ".join(["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]) + " |"
  body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
  return "\n".join([f"rounds={rounds} repeats={repeats} loops={loops} unit=us/call", "", header, align, *body])


def summary_record(result: BenchResult) -> dict[str, object]:
  return {
    "group": result.group,
    "name": result.name,
    "rounds": result.rounds,
    "monpy_median_us": result.monpy_median_us,
    "numpy_median_us": result.numpy_median_us,
    "ratio_median": result.ratio_median,
    "monpy_min_us": result.monpy_min_us,
    "monpy_max_us": result.monpy_max_us,
    "numpy_min_us": result.numpy_min_us,
    "numpy_max_us": result.numpy_max_us,
    "ratio_min": result.ratio_min,
    "ratio_max": result.ratio_max,
  }


def render_csv(results: Sequence[BenchResult]) -> str:
  output = io.StringIO()
  fields = [
    "group",
    "name",
    "rounds",
    "monpy_median_us",
    "numpy_median_us",
    "ratio_median",
    "monpy_min_us",
    "monpy_max_us",
    "numpy_min_us",
    "numpy_max_us",
    "ratio_min",
    "ratio_max",
  ]
  writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
  writer.writeheader()
  writer.writerows(summary_record(result) for result in results)
  return output.getvalue().rstrip()


def render_json(results: Sequence[BenchResult], *, rounds: int, loops: int, repeats: int) -> str:
  payload = {
    "config": {
      "rounds": rounds,
      "repeats": repeats,
      "loops": loops,
      "unit": "us/call",
    },
    "results": [
      {
        **summary_record(result),
        "samples": [
          {
            "round": sample.round_index,
            "monpy_us": sample.monpy_us,
            "numpy_us": sample.numpy_us,
            "ratio": sample.ratio,
          }
          for sample in result.samples
        ],
      }
      for result in results
    ],
  }
  return json.dumps(payload, indent=2, sort_keys=True)


def render_results(results: Sequence[BenchResult], *, args: argparse.Namespace) -> str:
  ordered = sorted_results(results, sort=args.sort)
  if args.format == "csv":
    return render_csv(ordered)
  if args.format == "json":
    return render_json(ordered, rounds=args.rounds, loops=args.loops, repeats=args.repeats)
  if args.format == "markdown":
    return render_markdown(ordered, rounds=args.rounds, loops=args.loops, repeats=args.repeats)
  return render_table(ordered, rounds=args.rounds, loops=args.loops, repeats=args.repeats)
