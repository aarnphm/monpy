"""Strided-fallback benchmark for monpy.

The main `bench_array_core.py` exercises contiguous fast paths. Phase 2
needs a measurement of strided-walker performance — the kernels that
currently go through `physical_offset(...)` divmod-per-element. This
file captures pre-migration baseline so the LayoutIter migration's
per-PR speedup is observable.

Cases:
  - strided_add_f32 (2D non-contig via transpose)
  - broadcast_add_f32 (rank-2 op rank-1)
  - reverse_stride_add_f32 (negative-stride view via [::-1, ::-1])
  - sliced_unary_sin_f32 (sliced view forces strided fallback)
  - 3d_strided_add_f32 (rank-3 non-contig)

For each case we report monpy / numpy time ratio (lower is better).
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

import monpy as mp


def _time(fn: Callable[[], object], *, loops: int = 100, repeats: int = 5) -> float:
    """Return median microseconds per call."""

    def force(x: object) -> object:
        if hasattr(x, "_native"):
            getattr(x._native, "data_address", lambda: None)()
        return x

    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(loops):
            force(fn())
        elapsed = time.perf_counter() - start
        samples.append(elapsed / loops * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


def case_strided_add_f32(N: int = 256) -> tuple[float, float]:
    np_a = np.random.rand(N, N).astype(np.float32)
    np_b = np.random.rand(N, N).astype(np.float32)
    np_a_t = np_a.T  # non-contig view
    np_b_t = np_b.T

    mp_a = mp.asarray(np_a)
    mp_b = mp.asarray(np_b)
    mp_a_t = mp_a.T
    mp_b_t = mp_b.T

    np_us = _time(lambda: np_a_t + np_b_t)
    mp_us = _time(lambda: mp_a_t + mp_b_t)
    return mp_us, np_us


def case_broadcast_add_f32(N: int = 256) -> tuple[float, float]:
    np_a = np.random.rand(N, N).astype(np.float32)
    np_v = np.random.rand(N).astype(np.float32)

    mp_a = mp.asarray(np_a)
    mp_v = mp.asarray(np_v)

    np_us = _time(lambda: np_a + np_v)
    mp_us = _time(lambda: mp_a + mp_v)
    return mp_us, np_us


def case_reverse_stride_add_f32(N: int = 256) -> tuple[float, float]:
    np_a = np.random.rand(N, N).astype(np.float32)
    np_b = np.random.rand(N, N).astype(np.float32)
    np_ar = np_a[::-1, ::-1]
    np_br = np_b[::-1, ::-1]

    mp_a = mp.asarray(np_a)
    mp_b = mp.asarray(np_b)
    # monpy doesn't yet expose ::-1 cleanly; skip if not supported
    try:
        mp_ar = mp_a[::-1, ::-1]
        mp_br = mp_b[::-1, ::-1]
    except Exception:  # pragma: no cover  # noqa: BLE001
        return float("nan"), float("nan")

    np_us = _time(lambda: np_ar + np_br)
    mp_us = _time(lambda: mp_ar + mp_br)
    return mp_us, np_us


def case_sliced_unary_sin_f32(N: int = 512) -> tuple[float, float]:
    np_a = np.random.rand(N, N).astype(np.float32)
    np_view = np_a[100:400, 100:400]

    mp_a = mp.asarray(np_a)
    mp_view = mp_a[100:400, 100:400]

    np_us = _time(lambda: np.sin(np_view))
    mp_us = _time(lambda: mp.sin(mp_view))
    return mp_us, np_us


def case_3d_strided_add_f32(N: int = 32) -> tuple[float, float]:
    np_a = np.random.rand(N, N, N).astype(np.float32)
    np_b = np.random.rand(N, N, N).astype(np.float32)
    # transpose creates non-contig 3D
    np_at = np_a.transpose(2, 0, 1)
    np_bt = np_b.transpose(2, 0, 1)

    mp_a = mp.asarray(np_a)
    mp_b = mp.asarray(np_b)
    try:
        mp_at = mp_a.transpose((2, 0, 1))
        mp_bt = mp_b.transpose((2, 0, 1))
    except Exception:  # pragma: no cover  # noqa: BLE001
        return float("nan"), float("nan")

    np_us = _time(lambda: np_at + np_bt)
    mp_us = _time(lambda: mp_at + mp_bt)
    return mp_us, np_us


def case_flip_axis0_f32(N: int = 256) -> tuple[float, float]:
    np_a = np.random.rand(N, N).astype(np.float32)
    mp_a = mp.asarray(np_a)
    np_us = _time(lambda: np.flip(np_a, axis=0))
    mp_us = _time(lambda: mp.flip(mp_a, axis=0))
    return mp_us, np_us


def case_flip_axis1_f32(N: int = 256) -> tuple[float, float]:
    np_a = np.random.rand(N, N).astype(np.float32)
    mp_a = mp.asarray(np_a)
    np_us = _time(lambda: np.flip(np_a, axis=1))
    mp_us = _time(lambda: mp.flip(mp_a, axis=1))
    return mp_us, np_us


def case_flip_all_f32(N: int = 256) -> tuple[float, float]:
    np_a = np.random.rand(N, N).astype(np.float32)
    mp_a = mp.asarray(np_a)
    np_us = _time(lambda: np.flip(np_a))
    mp_us = _time(lambda: mp.flip(mp_a))
    return mp_us, np_us


def case_rot90_f32(N: int = 256) -> tuple[float, float]:
    np_a = np.random.rand(N, N).astype(np.float32)
    mp_a = mp.asarray(np_a)
    np_us = _time(lambda: np.rot90(np_a))
    mp_us = _time(lambda: mp.rot90(mp_a))
    return mp_us, np_us


def case_concatenate_f32(N: int = 256) -> tuple[float, float]:
    np_a = np.random.rand(N, N).astype(np.float32)
    np_b = np.random.rand(N, N).astype(np.float32)
    mp_a = mp.asarray(np_a)
    mp_b = mp.asarray(np_b)
    np_us = _time(lambda: np.concatenate([np_a, np_b], axis=0))
    mp_us = _time(lambda: mp.concatenate([mp_a, mp_b], axis=0))
    return mp_us, np_us


def case_stack_f32(N: int = 256) -> tuple[float, float]:
    np_a = np.random.rand(N).astype(np.float32)
    np_b = np.random.rand(N).astype(np.float32)
    mp_a = mp.asarray(np_a)
    mp_b = mp.asarray(np_b)
    np_us = _time(lambda: np.stack([np_a, np_b]))
    mp_us = _time(lambda: mp.stack([mp_a, mp_b]))
    return mp_us, np_us


def case_squeeze_f32() -> tuple[float, float]:
    np_a = np.zeros((1, 64, 1, 64), dtype=np.float32)
    mp_a = mp.asarray(np_a)
    np_us = _time(lambda: np.squeeze(np_a))
    mp_us = _time(lambda: mp.squeeze(mp_a))
    return mp_us, np_us


def case_moveaxis_f32() -> tuple[float, float]:
    np_a = np.random.rand(2, 3, 4, 5).astype(np.float32)
    mp_a = mp.asarray(np_a)
    np_us = _time(lambda: np.moveaxis(np_a, 0, -1))
    mp_us = _time(lambda: mp.moveaxis(mp_a, 0, -1))
    return mp_us, np_us


def case_eye_64_f32() -> tuple[float, float]:
    np_us = _time(lambda: np.eye(64, dtype=np.float32))
    mp_us = _time(lambda: mp.eye(64, dtype=mp.float32))
    return mp_us, np_us


def case_meshgrid_64() -> tuple[float, float]:
    np_x = np.linspace(-1, 1, 64, dtype=np.float32)
    np_y = np.linspace(-1, 1, 64, dtype=np.float32)
    mp_x = mp.asarray(np_x)
    mp_y = mp.asarray(np_y)
    np_us = _time(lambda: np.meshgrid(np_x, np_y))
    mp_us = _time(lambda: mp.meshgrid(mp_x, mp_y))
    return mp_us, np_us


# Phase-6d LAPACK decomp benches. Cover qr / cholesky / eigvalsh / svdvals /
# pinv on a 32×32 conditioned matrix. lstsq runs on a tall-skinny system.
def case_qr_32_f64() -> tuple[float, float]:
    np_a = np.random.rand(32, 32).astype(np.float64) + np.eye(32)
    mp_a = mp.asarray(np_a, dtype=mp.float64)
    np_us = _time(lambda: np.linalg.qr(np_a))
    mp_us = _time(lambda: mp.linalg.qr(mp_a))
    return mp_us, np_us


def case_cholesky_32_f64() -> tuple[float, float]:
    base = np.random.rand(32, 32).astype(np.float64)
    np_a = base @ base.T + 32 * np.eye(32)
    mp_a = mp.asarray(np_a, dtype=mp.float64)
    np_us = _time(lambda: np.linalg.cholesky(np_a))
    mp_us = _time(lambda: mp.linalg.cholesky(mp_a))
    return mp_us, np_us


def case_eigvalsh_32_f64() -> tuple[float, float]:
    base = np.random.rand(32, 32).astype(np.float64)
    np_a = (base + base.T) / 2
    mp_a = mp.asarray(np_a, dtype=mp.float64)
    np_us = _time(lambda: np.linalg.eigvalsh(np_a))
    mp_us = _time(lambda: mp.linalg.eigvalsh(mp_a))
    return mp_us, np_us


def case_svdvals_32_f64() -> tuple[float, float]:
    np_a = np.random.rand(32, 32).astype(np.float64)
    mp_a = mp.asarray(np_a, dtype=mp.float64)
    np_us = _time(lambda: np.linalg.svd(np_a, compute_uv=False))
    mp_us = _time(lambda: mp.linalg.svdvals(mp_a))
    return mp_us, np_us


def case_pinv_32_f64() -> tuple[float, float]:
    np_a = np.random.rand(32, 32).astype(np.float64) + np.eye(32)
    mp_a = mp.asarray(np_a, dtype=mp.float64)
    np_us = _time(lambda: np.linalg.pinv(np_a))
    mp_us = _time(lambda: mp.linalg.pinv(mp_a))
    return mp_us, np_us


def case_lstsq_64x32_f64() -> tuple[float, float]:
    np_a = np.random.rand(64, 32).astype(np.float64)
    np_b = np.random.rand(64).astype(np.float64)
    mp_a = mp.asarray(np_a, dtype=mp.float64)
    mp_b = mp.asarray(np_b, dtype=mp.float64)
    np_us = _time(lambda: np.linalg.lstsq(np_a, np_b, rcond=None))
    mp_us = _time(lambda: mp.linalg.lstsq(mp_a, mp_b))
    return mp_us, np_us


# Phase-5b/5c dtype family benches. Wrapper-bound until typed-vec int /
# half kernels land. These rows track the gap.
def case_uint8_add_1024() -> tuple[float, float]:
    np_a = np.arange(1024, dtype=np.uint8)
    mp_a = mp.asarray(np_a, dtype=mp.uint8)
    np_us = _time(lambda: np_a + np_a)
    mp_us = _time(lambda: mp_a + mp_a)
    return mp_us, np_us


def case_uint32_add_1024() -> tuple[float, float]:
    np_a = np.arange(1024, dtype=np.uint32)
    mp_a = mp.asarray(np_a, dtype=mp.uint32)
    np_us = _time(lambda: np_a + np_a)
    mp_us = _time(lambda: mp_a + mp_a)
    return mp_us, np_us


def case_float16_add_1024() -> tuple[float, float]:
    np_a = np.linspace(0.0, 1.0, 1024, dtype=np.float16)
    mp_a = mp.asarray(np_a, dtype=mp.float16)
    np_us = _time(lambda: np_a + np_a)
    mp_us = _time(lambda: mp_a + mp_a)
    return mp_us, np_us


# Phase-6 native creation kernels. These were ~58000× / ~250× / ~100× slower
# pre-native; rows here track the post-native speedup.
def case_native_concatenate_4x256() -> tuple[float, float]:
    np_xs = [np.zeros(256, dtype=np.float64) for _ in range(4)]
    mp_xs = [mp.zeros((256,), dtype=mp.float64) for _ in range(4)]
    np_us = _time(lambda: np.concatenate(np_xs))
    mp_us = _time(lambda: mp.concatenate(mp_xs))
    return mp_us, np_us


def case_native_tril_64x64() -> tuple[float, float]:
    np_a = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
    mp_a = mp.asarray(np_a, dtype=mp.float64)
    np_us = _time(lambda: np.tril(np_a))
    mp_us = _time(lambda: mp.tril(mp_a))
    return mp_us, np_us


def case_native_triu_64x64() -> tuple[float, float]:
    np_a = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
    mp_a = mp.asarray(np_a, dtype=mp.float64)
    np_us = _time(lambda: np.triu(np_a))
    mp_us = _time(lambda: mp.triu(mp_a))
    return mp_us, np_us


def case_native_pad_constant() -> tuple[float, float]:
    np_a = np.arange(64, dtype=np.float64)
    mp_a = mp.asarray(np_a, dtype=mp.float64)
    np_us = _time(lambda: np.pad(np_a, 5))
    mp_us = _time(lambda: mp.pad(mp_a, 5))
    return mp_us, np_us


CASES: list[tuple[str, Callable[[], tuple[float, float]]]] = [
    ("strided_add_f32_256", case_strided_add_f32),
    ("broadcast_add_f32_256", case_broadcast_add_f32),
    ("reverse_stride_add_f32_256", case_reverse_stride_add_f32),
    ("sliced_unary_sin_f32_300x300", case_sliced_unary_sin_f32),
    ("3d_strided_add_f32_32", case_3d_strided_add_f32),
    ("flip_axis0_f32_256", case_flip_axis0_f32),
    ("flip_axis1_f32_256", case_flip_axis1_f32),
    ("flip_all_f32_256", case_flip_all_f32),
    ("rot90_f32_256", case_rot90_f32),
    ("concatenate_f32_256", case_concatenate_f32),
    ("stack_f32_256", case_stack_f32),
    ("squeeze_f32_64x64", case_squeeze_f32),
    ("moveaxis_f32_2x3x4x5", case_moveaxis_f32),
    ("eye_64_f32", case_eye_64_f32),
    ("meshgrid_64", case_meshgrid_64),
    # Phase-6d LAPACK
    ("qr_32_f64", case_qr_32_f64),
    ("cholesky_32_f64", case_cholesky_32_f64),
    ("eigvalsh_32_f64", case_eigvalsh_32_f64),
    ("svdvals_32_f64", case_svdvals_32_f64),
    ("pinv_32_f64", case_pinv_32_f64),
    ("lstsq_64x32_f64", case_lstsq_64x32_f64),
    # Phase-5b/5c dtype families
    ("uint8_add_1024", case_uint8_add_1024),
    ("uint32_add_1024", case_uint32_add_1024),
    ("float16_add_1024", case_float16_add_1024),
    # Phase-6 native kernels
    ("native_concatenate_4x256", case_native_concatenate_4x256),
    ("native_tril_64x64", case_native_tril_64x64),
    ("native_triu_64x64", case_native_triu_64x64),
    ("native_pad_constant", case_native_pad_constant),
]


def main() -> None:
    print(f"{'case':<35}  {'monpy us':>10}  {'numpy us':>10}  {'ratio':>8}")
    print("-" * 70)
    for name, fn in CASES:
        try:
            mp_us, np_us = fn()
        except Exception as exc:  # pragma: no cover  # noqa: BLE001
            print(f"{name:<35}  ERROR: {exc}")
            continue
        if np_us != np_us:  # NaN
            print(f"{name:<35}  SKIPPED (unsupported view op)")
            continue
        ratio = mp_us / np_us if np_us > 0 else float("inf")
        print(f"{name:<35}  {mp_us:>10.2f}  {np_us:>10.2f}  {ratio:>8.2f}x")


if __name__ == "__main__":
    main()
