from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import monumpy as mnp
import numpy as np
import numpy.testing as npt

BenchFn = Callable[[], object]


def time_call(fn: BenchFn, *, loops: int, repeats: int) -> float:
  samples: list[float] = []
  for _ in range(repeats):
    start = time.perf_counter()
    for _ in range(loops):
      result = fn()
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

  benches: list[tuple[str, BenchFn, BenchFn]] = [
    ("unary_sin_f32", lambda: mnp.sin(x_mp), lambda: np.sin(x_np)),
    ("binary_add_f32", lambda: x_mp + y_mp, lambda: x_np + y_np),
    ("broadcast_add_f32", lambda: m_mp + row_mp, lambda: m_np + row_np),
    ("sum_f32", lambda: mnp.sum(x_mp), lambda: np.sum(x_np)),
    ("strided_view_f32", lambda: x_mp[::-2], lambda: x_np[::-2]),
    ("matmul_16_f32", lambda: m_mp @ n_mp, lambda: m_np @ n_np),
    ("matmul_64_f32", lambda: m64_mp @ n64_mp, lambda: m64_np @ n64_np),
    ("matmul_128_f32", lambda: m128_mp @ n128_mp, lambda: m128_np @ n128_np),
    ("fused_expression_f32", lambda: mnp.sin(x_mp) + y_mp * 3.0, lambda: np.sin(x_np) + y_np * 3.0),
  ]

  print("name,monpy_us,numpy_us,ratio_monpy_to_numpy")
  for name, monpy_fn, numpy_fn in benches:
    monpy_result = monpy_fn()
    numpy_result = numpy_fn()
    verify_same(monpy_result, numpy_result, rtol=1e-4, atol=1e-4)
    monpy_time = time_call(monpy_fn, loops=args.loops, repeats=args.repeats)
    numpy_time = time_call(numpy_fn, loops=args.loops, repeats=args.repeats)
    ratio = monpy_time / numpy_time if numpy_time else float("inf")
    print(f"{name},{monpy_time * 1_000_000:.3f},{numpy_time * 1_000_000:.3f},{ratio:.3f}")


if __name__ == "__main__":
  main()
