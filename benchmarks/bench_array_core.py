from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import monpy as _monpy
import monumpy as mnp
import numpy as np
import numpy.testing as npt

BenchFn = Callable[[], object]


def force_monpy(value: object) -> object:
  if not isinstance(value, mnp.ndarray):
    try:
      _ = value._native
    except AttributeError:
      pass
  return value


def time_call(fn: BenchFn, *, loops: int, repeats: int, force: Callable[[object], object] | None = None) -> float:
  samples: list[float] = []
  for _ in range(repeats):
    start = time.perf_counter()
    for _ in range(loops):
      result = fn()
      if force is not None:
        result = force(result)
      if result is None:
        raise RuntimeError("benchmark function unexpectedly returned None")
    samples.append((time.perf_counter() - start) / loops)
  return statistics.median(samples)


def verify_same(monpy_value: object, numpy_value: object, *, rtol: float = 1e-5, atol: float = 1e-6) -> None:
  npt.assert_allclose(np.asarray(monpy_value), np.asarray(numpy_value), rtol=rtol, atol=atol)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--loops", type=int, default=200)
  parser.add_argument("--repeats", type=int, default=5)
  args = parser.parse_args()

  x_np = np.linspace(0.1, 2.0, 1024, dtype=np.float32)
  y_np = np.linspace(2.0, 4.0, 1024, dtype=np.float32)
  x_mp = mnp.asarray(x_np.tolist(), dtype=mnp.float32)
  y_mp = mnp.asarray(y_np.tolist(), dtype=mnp.float32)
  add_out_np = np.empty_like(x_np)
  add_out_mp = mnp.empty(x_mp.shape, dtype=mnp.float32)

  def native_add_out_f32() -> object:
    _monpy._native.add_into(add_out_mp._native, x_mp._native, y_mp._native)
    return add_out_mp

  small_x = mnp.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=mnp.float32)
  small_y = mnp.asarray([10, 20, 30, 40, 50, 60, 70, 80], dtype=mnp.float32)
  layout_out = small_x + small_y
  if not layout_out._native.used_layout_tensor():
    raise AssertionError("expected fixed-size float32 add to exercise the LayoutTensor path")
  verify_same(layout_out, np.asarray([11, 22, 33, 44, 55, 66, 77, 88], dtype=np.float32))

  m_np = np.arange(256, dtype=np.float32).reshape(16, 16)
  n_np = np.arange(256, dtype=np.float32).reshape(16, 16)
  row_np = np.ones(16, dtype=np.float32)
  m_mp = mnp.asarray(m_np.tolist(), dtype=mnp.float32)
  n_mp = mnp.asarray(n_np.tolist(), dtype=mnp.float32)
  row_mp = mnp.asarray(row_np.tolist(), dtype=mnp.float32)

  m64_np = (np.arange(64 * 64, dtype=np.float32).reshape(64, 64) / 1000).astype(np.float32)
  n64_np = (np.arange(64 * 64, dtype=np.float32).reshape(64, 64) / 1000).astype(np.float32)
  m64_mp = mnp.asarray(m64_np.tolist(), dtype=mnp.float32)
  n64_mp = mnp.asarray(n64_np.tolist(), dtype=mnp.float32)

  m128_np = (np.arange(128 * 128, dtype=np.float32).reshape(128, 128) / 1000).astype(np.float32)
  n128_np = (np.arange(128 * 128, dtype=np.float32).reshape(128, 128) / 1000).astype(np.float32)
  m128_mp = mnp.asarray(m128_np.tolist(), dtype=mnp.float32)
  n128_mp = mnp.asarray(n128_np.tolist(), dtype=mnp.float32)

  m64_f64_np = np.arange(64 * 64, dtype=np.float64).reshape(64, 64) / 1000
  n64_f64_np = np.arange(64 * 64, dtype=np.float64).reshape(64, 64) / 1000
  m64_f64_mp = mnp.asarray(m64_f64_np.tolist(), dtype=mnp.float64)
  n64_f64_mp = mnp.asarray(n64_f64_np.tolist(), dtype=mnp.float64)

  m128_f64_np = np.arange(128 * 128, dtype=np.float64).reshape(128, 128) / 1000
  n128_f64_np = np.arange(128 * 128, dtype=np.float64).reshape(128, 128) / 1000
  m128_f64_mp = mnp.asarray(m128_f64_np.tolist(), dtype=mnp.float64)
  n128_f64_mp = mnp.asarray(n128_f64_np.tolist(), dtype=mnp.float64)

  matmul_probe = m128_mp @ n128_mp
  if not matmul_probe._native.used_accelerate():
    raise AssertionError("expected contiguous float32 matmul to exercise the Apple Accelerate path")
  verify_same(matmul_probe, m128_np @ n128_np, rtol=1e-4, atol=1e-4)

  matmul_f64_probe = m64_f64_mp @ n64_f64_mp
  if not matmul_f64_probe._native.used_accelerate():
    raise AssertionError("expected contiguous float64 matmul to exercise the Apple Accelerate dgemm path")
  verify_same(matmul_f64_probe, m64_f64_np @ n64_f64_np, rtol=1e-9, atol=1e-9)

  fused_probe = mnp.sin_add_mul(x_mp, y_mp, 3.0)
  if not fused_probe._native.used_fused():
    raise AssertionError("expected sin_add_mul to exercise the native fused kernel path")
  verify_same(fused_probe, np.sin(x_np) + y_np * 3.0, rtol=1e-4, atol=1e-4)

  eager_fused_probe = mnp.sin(x_mp) + y_mp * 3.0
  if not eager_fused_probe._native.used_fused():
    raise AssertionError("expected sin(x) + y * scalar to lower into the native fused kernel path")
  verify_same(eager_fused_probe, np.sin(x_np) + y_np * 3.0, rtol=1e-4, atol=1e-4)

  benches: list[tuple[str, BenchFn, BenchFn]] = [
    ("unary_sin_f32", lambda: mnp.sin(x_mp), lambda: np.sin(x_np)),
    ("binary_add_f32", lambda: x_mp + y_mp, lambda: x_np + y_np),
    ("binary_add_out_f32", lambda: mnp.add(x_mp, y_mp, out=add_out_mp), lambda: np.add(x_np, y_np, out=add_out_np)),
    ("binary_add_native_out_f32", native_add_out_f32, lambda: np.add(x_np, y_np, out=add_out_np)),
    ("broadcast_add_f32", lambda: m_mp + row_mp, lambda: m_np + row_np),
    ("sum_f32", lambda: mnp.sum(x_mp), lambda: np.sum(x_np)),
    ("strided_view_f32", lambda: x_mp[::-2], lambda: x_np[::-2]),
    ("matmul_16_f32", lambda: m_mp @ n_mp, lambda: m_np @ n_np),
    ("matmul_64_f32", lambda: m64_mp @ n64_mp, lambda: m64_np @ n64_np),
    ("matmul_128_f32", lambda: m128_mp @ n128_mp, lambda: m128_np @ n128_np),
    ("matmul_64_f64", lambda: m64_f64_mp @ n64_f64_mp, lambda: m64_f64_np @ n64_f64_np),
    ("matmul_128_f64", lambda: m128_f64_mp @ n128_f64_mp, lambda: m128_f64_np @ n128_f64_np),
    ("eager_expression_f32", lambda: mnp.sin(x_mp) + y_mp * 3.0, lambda: np.sin(x_np) + y_np * 3.0),
    ("fused_sin_add_mul_f32", lambda: mnp.sin_add_mul(x_mp, y_mp, 3.0), lambda: np.sin(x_np) + y_np * 3.0),
  ]

  print("name,monpy_us,numpy_us,ratio_monpy_to_numpy")
  for name, monpy_fn, numpy_fn in benches:
    monpy_result = monpy_fn()
    numpy_result = numpy_fn()
    verify_same(monpy_result, numpy_result, rtol=1e-4, atol=1e-4)
    monpy_time = time_call(monpy_fn, loops=args.loops, repeats=args.repeats, force=force_monpy)
    numpy_time = time_call(numpy_fn, loops=args.loops, repeats=args.repeats)
    ratio = monpy_time / numpy_time if numpy_time else float("inf")
    print(f"{name},{monpy_time * 1_000_000:.3f},{numpy_time * 1_000_000:.3f},{ratio:.3f}")


if __name__ == "__main__":
  main()
