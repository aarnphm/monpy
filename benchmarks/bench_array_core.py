from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import warnings

import monpy as _monpy
import monumpy as mnp
import numpy as np
import numpy.testing as npt

try:
  from tqdm import tqdm as _tqdm
except ModuleNotFoundError:
  _tqdm = None

BenchFn = Callable[[], object]


@dataclass(frozen=True, slots=True)
class BenchCase:
  group: str
  name: str
  monpy_fn: BenchFn
  numpy_fn: BenchFn
  rtol: float = 1e-4
  atol: float = 1e-4
  check_values: bool = True


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
  def __enter__(self) -> _NoProgress:
    return self

  def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
    return None

  def update(self, n: int = 1) -> None:
    return None

  def set_postfix_str(self, s: str, refresh: bool = True) -> None:
    return None


def progress_bar(*, total: int, enabled: bool) -> object:
  if not enabled:
    return _NoProgress()
  if _tqdm is None:
    print("progress disabled: install tqdm to render a progress bar", file=sys.stderr)
    return _NoProgress()
  return _tqdm(total=total, desc="bench", dynamic_ncols=True, unit="case", leave=False, file=sys.stderr)


def positive_int(value: str) -> int:
  parsed = int(value)
  if parsed < 1:
    raise argparse.ArgumentTypeError("must be >= 1")
  return parsed


def parse_sizes(value: str) -> tuple[int, ...]:
  sizes = tuple(positive_int(part.strip()) for part in value.split(",") if part.strip())
  if not sizes:
    raise argparse.ArgumentTypeError("must include at least one size")
  return sizes


def force_monpy(value: object) -> object:
  if not isinstance(value, mnp.ndarray):
    try:
      _ = value._native
    except AttributeError:
      pass
  return value


@contextmanager
def suppress_known_numpy_linalg_warnings():
  with warnings.catch_warnings():
    warnings.filterwarnings(
      "ignore",
      category=RuntimeWarning,
      message="overflow encountered in cast",
      module=r"numpy\.linalg\._linalg",
    )
    yield


def call_bench_fn(fn: BenchFn) -> object:
  with suppress_known_numpy_linalg_warnings():
    return fn()


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


def verify_shape_dtype(monpy_value: object, numpy_value: object) -> None:
  monpy_array = np.asarray(monpy_value)
  numpy_array = np.asarray(numpy_value)
  if monpy_array.shape != numpy_array.shape:
    raise AssertionError(f"shape mismatch: {monpy_array.shape!r} != {numpy_array.shape!r}")
  if monpy_array.dtype != numpy_array.dtype:
    raise AssertionError(f"dtype mismatch: {monpy_array.dtype!r} != {numpy_array.dtype!r}")


def ratio_value(monpy_us: float, numpy_us: float) -> float:
  if numpy_us == 0.0:
    return math.inf
  return monpy_us / numpy_us


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
  x_mp = mnp.asarray(x_np.tolist(), dtype=mnp.float32)
  y_mp = mnp.asarray(y_np.tolist(), dtype=mnp.float32)
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
  m_mp = mnp.asarray(m_np.tolist(), dtype=mnp.float32)
  n_mp = mnp.asarray(n_np.tolist(), dtype=mnp.float32)
  row_mp = mnp.asarray(row_np.tolist(), dtype=mnp.float32)

  matmul_probe_size = max(64, max(matrix_sizes))
  matmul_probe_np = scaled_matrix(matmul_probe_size, np.float32)
  matmul_probe_mp = mnp.asarray(matmul_probe_np.tolist(), dtype=mnp.float32)
  matmul_probe = matmul_probe_mp @ matmul_probe_mp
  if not matmul_probe._native.used_accelerate():
    raise AssertionError("expected contiguous float32 matmul to exercise the Apple Accelerate path")
  verify_same(matmul_probe, matmul_probe_np @ matmul_probe_np, rtol=1e-4, atol=1e-4)

  matmul_f64_probe_np = scaled_matrix(matmul_probe_size, np.float64)
  matmul_f64_probe_mp = mnp.asarray(matmul_f64_probe_np.tolist(), dtype=mnp.float64)
  matmul_f64_probe = matmul_f64_probe_mp @ matmul_f64_probe_mp
  if not matmul_f64_probe._native.used_accelerate():
    raise AssertionError("expected contiguous float64 matmul to exercise the Apple Accelerate dgemm path")
  verify_same(matmul_f64_probe, matmul_f64_probe_np @ matmul_f64_probe_np, rtol=1e-9, atol=1e-9)

  fused_probe = mnp.sin_add_mul(x_mp, y_mp, 3.0)
  if not fused_probe._native.used_fused():
    raise AssertionError("expected sin_add_mul to exercise the native fused kernel path")
  verify_same(fused_probe, np.sin(x_np) + y_np * 3.0, rtol=1e-4, atol=1e-4)

  eager_fused_probe = mnp.sin(x_mp) + y_mp * 3.0
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

  # Phase-6a / 6b additions: shape manipulation + creation helpers.
  s_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
  s_mp = mnp.asarray(s_np)
  vec_a_np = np.arange(8, dtype=np.float32)
  vec_a_mp = mnp.asarray(vec_a_np)
  vec_b_np = np.arange(8, 16, dtype=np.float32)
  vec_b_mp = mnp.asarray(vec_b_np)
  cases.extend([
    BenchCase("views", "squeeze_axis0_f32", lambda: mnp.squeeze(mnp.asarray(np.zeros((1, 4, 1, 5), dtype=np.float32)), axis=0), lambda: np.squeeze(np.zeros((1, 4, 1, 5), dtype=np.float32), axis=0)),
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
    bw_x_mp = mnp.asarray(bw_x_np, dtype=mnp.float32, copy=False)
    bw_y_mp = mnp.asarray(bw_y_np, dtype=mnp.float32, copy=False)
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

  return cases


def run_case(case: BenchCase, *, loops: int, repeats: int, round_index: int) -> BenchSample:
  monpy_result = call_bench_fn(case.monpy_fn)
  numpy_result = call_bench_fn(case.numpy_fn)
  if case.check_values:
    verify_same(monpy_result, numpy_result, rtol=case.rtol, atol=case.atol)
  else:
    verify_shape_dtype(monpy_result, numpy_result)
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
  samples_by_case: dict[str, list[BenchSample]] = {case.name: [] for case in cases}
  total = rounds * len(cases)
  with progress_bar(total=total, enabled=progress) as bar:
    for round_index in range(1, rounds + 1):
      for case in cases:
        samples_by_case[case.name].append(run_case(case, loops=loops, repeats=repeats, round_index=round_index))
        try:
          bar.set_postfix_str(f"round={round_index} case={case.name}", refresh=False)
          bar.update(1)
        except AttributeError:
          pass
  return [summarize(case, samples_by_case[case.name]) for case in cases]


def format_us(value: float) -> str:
  if math.isinf(value):
    return "inf"
  return f"{value:.3f}"


def format_ratio(value: float) -> str:
  if math.isinf(value):
    return "inf"
  return f"{value:.3f}x"


def format_range(min_value: float, max_value: float, *, ratio: bool = False) -> str:
  formatter = format_ratio if ratio else format_us
  return f"{formatter(min_value)}..{formatter(max_value)}"


def sorted_results(results: Sequence[BenchResult], *, sort: str) -> list[BenchResult]:
  if sort == "name":
    return sorted(results, key=lambda result: (result.name, result.group))
  if sort == "monpy":
    return sorted(results, key=lambda result: result.monpy_median_us, reverse=True)
  if sort == "ratio":
    return sorted(results, key=lambda result: result.ratio_median, reverse=True)
  return list(results)


def table_rows(results: Sequence[BenchResult]) -> list[tuple[str, ...]]:
  rows = [
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
    cells = []
    for index, cell in enumerate(row):
      cells.append(cell.rjust(widths[index]) if index in numeric_columns else cell.ljust(widths[index]))
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
  for result in results:
    writer.writerow(summary_record(result))
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


def main() -> None:
  parser = argparse.ArgumentParser(description="benchmark monpy array-core paths against numpy")
  parser.add_argument("--loops", type=positive_int, default=200, help="inner calls per timing sample")
  parser.add_argument("--repeats", type=positive_int, default=5, help="timing samples per case per round")
  parser.add_argument("--rounds", type=positive_int, default=3, help="full benchmark passes to aggregate")
  parser.add_argument(
    "--vector-size",
    type=positive_int,
    default=1024,
    help="size for the wrapper-bound elementwise cases (default 1024)",
  )
  parser.add_argument(
    "--vector-sizes",
    type=parse_sizes,
    default=parse_sizes("16384,262144,1048576"),
    help="sizes for the bandwidth-regime cases (default 16K,256K,1M)",
  )
  parser.add_argument(
    "--matrix-sizes",
    type=parse_sizes,
    default=parse_sizes("16,64,128,256"),
    help="square matrix sizes for matmul/matvec/vecmat (default 16,64,128,256)",
  )
  parser.add_argument(
    "--linalg-sizes",
    type=parse_sizes,
    default=parse_sizes("2,4,8,32,128"),
    help="square matrix sizes for solve/inv/det (default 2,4,8,32,128)",
  )
  parser.add_argument("--format", choices=("table", "csv", "json", "markdown"), default="table")
  parser.add_argument("--sort", choices=("input", "name", "monpy", "ratio"), default="input")
  parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
  args = parser.parse_args()

  cases = build_cases(
    vector_size=args.vector_size,
    vector_sizes=args.vector_sizes,
    matrix_sizes=args.matrix_sizes,
    linalg_sizes=args.linalg_sizes,
  )
  results = run_benchmarks(cases, rounds=args.rounds, loops=args.loops, repeats=args.repeats, progress=args.progress)
  print(render_results(results, args=args))


if __name__ == "__main__":
  main()
