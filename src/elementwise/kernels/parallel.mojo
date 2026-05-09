"""Parallel-dispatch infrastructure: gates, worker count, env knob.

Layer 0 of monpy's multi-thread stack. Sits between the dispatch ladders
(`binary_dispatch.mojo`, `unary_dispatch.mojo`, `kernels/reduce.mojo`)
and `std.algorithm.{sync_parallelize, parallelize_over_rows}`. No callers
yet; PRs 2-7 wire this into the production paths.

The single env knob is `MONPY_THREADS=1`, which forces every parallel
gate to fall back to the serial path. Useful for debugging non-
deterministic numerical results, or when monpy is called from a host-
side thread pool that already saturates the CPU.

Grain sizes are byte-count thresholds below which serial wins because
thread setup (~1-10us) outpaces the work. Empirically calibrated from
`benches/bench_reduce.mojo`: `sum_par` reaches parity with `sum8` at
~1MB on M3 Pro, beats it ~2x by 16MB. Element-wise heavy (exp/log/sin)
amortises spawn cost earlier because per-element work is heavier; light
ops (add/mul) need more bytes to break even.

Reference patterns:
  - `_Global` lazy singleton: `src/accelerate.mojo:445-449`
  - `getenv`: `std/os/env.mojo:65`
  - `sync_parallelize`: `std/algorithm/backend/cpu/parallelize.mojo:33`
  - `parallelize_over_rows`: `std/algorithm/backend/cpu/parallelize.mojo:303`
  - `num_performance_cores`: `std/sys/info.mojo:1242`
"""

from std.ffi import _Global
from std.os.env import getenv
from std.sys import num_performance_cores


# Grain-size constants — byte counts above which a parallel kernel is
# worth the spawn cost. Reductions reach parity earliest because per-
# thread Float64 partial accumulators are tiny and the combine is O(1).
# Element-wise light ops (add/mul/cmp) need more bytes because per-
# element work is one SIMD instruction; element-wise heavy ops (exp/log/
# sin) amortize earlier because per-element FLOPs are higher.
comptime REDUCE_GRAIN = 1 << 20  # 1MB
comptime ELEMENTWISE_LIGHT_GRAIN = 1 << 21  # 2MB
comptime ELEMENTWISE_HEAVY_GRAIN = 1 << 18  # 256KB
comptime PER_ROW_MIN = 16  # rows below this gate out, regardless of cols


def _init_thread_serial() -> Int:
    # Storage value: 1 = serial-only (MONPY_THREADS=1 was set), else 0.
    # Read once at process init via _Global.
    var raw = getenv("MONPY_THREADS", "")
    if raw == "1":
        return 1
    return 0


comptime MONPY_THREAD_SERIAL = _Global[
    "MONPY_THREAD_SERIAL",
    _init_thread_serial,
]


def is_serial_only() -> Bool:
    """Returns True when `MONPY_THREADS=1` was set at process start.

    Safe to call from any context; returns False if the env read fails
    rather than propagating an exception into hot paths.
    """
    try:
        return MONPY_THREAD_SERIAL.get_or_create_ptr()[] == 1
    except:
        return False


def should_parallelize_bytes(byte_count: Int, grain: Int) -> Bool:
    """Gate for whole-tensor parallelism by byte count + op-class grain.

    Args:
        byte_count: total bytes in the operand (size * sizeof(dtype)).
        grain: grain-size constant — `REDUCE_GRAIN`, `ELEMENTWISE_LIGHT_GRAIN`,
               or `ELEMENTWISE_HEAVY_GRAIN`.

    Returns: True if the work justifies thread fan-out.
    """
    if is_serial_only():
        return False
    return byte_count >= grain


def should_parallelize_rows(row_count: Int) -> Bool:
    """Gate for per-row parallelism (softmax, layernorm, axis-reduce).

    Below `PER_ROW_MIN` rows the thread setup (~1-10us) dominates
    per-row work; serial wins. Above this, even narrow rows can be
    split across workers profitably because the per-row work itself
    is non-trivial (max scan, exp scan, normalize for softmax; sum +
    sumsq scan for layernorm).
    """
    if is_serial_only():
        return False
    return row_count >= PER_ROW_MIN


def worker_count(work_units: Int) -> Int:
    """Resolve worker count for the given `work_units` count.

    Returns 1 when `MONPY_THREADS=1`. Otherwise caps at the number of
    P-cores (`num_performance_cores()`) — never the logical core count.
    On Apple Silicon the E-cores have ~1/3 the FP throughput of P-cores,
    so adding them slows down a P-core-aligned partition.
    """
    if is_serial_only():
        return 1
    var n = num_performance_cores()
    if work_units < n:
        return work_units
    return n
