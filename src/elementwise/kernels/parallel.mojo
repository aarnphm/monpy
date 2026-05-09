"""Parallel-dispatch infrastructure: gates, worker counts, env knob.

Layer 0 of monpy's multi-thread stack. Sits between the dispatch ladders
(`binary_dispatch.mojo`, `unary_dispatch.mojo`, `kernels/reduce.mojo`)
and `std.algorithm.{sync_parallelize, parallelize_over_rows}`.

`MONPY_THREADS=1` forces every parallel gate to fall back to the serial
path. `MONPY_THREADS=N` caps the worker count at N. Useful for debugging
non-deterministic numerical results, NUMA experiments, or when monpy is
called from a host-side thread pool that already saturates the CPU.

Grain sizes are per-worker work thresholds, not global "parallel yes/no"
thresholds. Empirically calibrated from `benches/bench_reduce.mojo`:
`sum_par` reaches parity with `sum8` at ~1MB on M3 Pro, beats it ~2x by
16MB. On many-core Ubuntu boxes this prevents a 1MB tensor from being
split into dozens of cache-cold scraps. Element-wise heavy (exp/log/sin)
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


# Grain-size constants. These are per-worker byte/row-element budgets,
# not total-tensor gates. Reductions reach parity earliest because per-
# worker Float64 partial accumulators are tiny and the combine is O(1).
# Element-wise light ops (add/mul/cmp) need more bytes because per-
# element work is one SIMD instruction; element-wise heavy ops (exp/log/
# sin) amortize earlier because per-element FLOPs are higher.
comptime REDUCE_GRAIN = 1 << 20  # 1MB
comptime ELEMENTWISE_LIGHT_GRAIN = 1 << 21  # 2MB
comptime ELEMENTWISE_HEAVY_GRAIN = 1 << 18  # 256KB
comptime ROW_HEAVY_GRAIN_ELEMS = 1 << 13  # row elements per worker for softmax/layernorm
comptime PER_ROW_MIN = 16  # row-slice budget once total row work is large enough


def _init_thread_limit() -> Int:
    # Storage value: 0 = auto, N > 0 = user cap. Read once at process init
    # via _Global so hot paths do not touch the environment.
    var raw = getenv("MONPY_THREADS", "")
    if raw == "":
        return 0
    try:
        var value = Int(raw)
        if value > 0:
            return value
    except:
        return 0
    return 0


comptime MONPY_THREAD_LIMIT = _Global[
    "MONPY_THREAD_LIMIT",
    _init_thread_limit,
]


def thread_limit() -> Int:
    """Returns the `MONPY_THREADS` cap, or 0 for automatic worker count."""
    try:
        return MONPY_THREAD_LIMIT.get_or_create_ptr()[]
    except:
        return 0


def is_serial_only() -> Bool:
    """Returns True when `MONPY_THREADS=1` was set at process start.

    Safe to call from any context; returns False if the env read fails
    rather than propagating an exception into hot paths.
    """
    return thread_limit() == 1


def worker_count(work_units: Int) -> Int:
    """Resolve the hardware/env-capped worker count for `work_units`.

    This is a core cap only. Prefer `worker_count_for_bytes` or
    `worker_count_for_rows` for production dispatch so many-core machines do
    not split small tensors into too many tiny chunks.
    """
    if is_serial_only() or work_units <= 1:
        return 1
    var n = num_performance_cores()
    if n < 1:
        n = 1
    var limit = thread_limit()
    if limit > 0 and limit < n:
        n = limit
    if work_units < n:
        n = work_units
    if n < 1:
        return 1
    return n


def worker_count_for_bytes(work_units: Int, byte_count: Int, grain_per_worker: Int) -> Int:
    """Resolve worker count by hardware cap and byte budget per worker.

    `work_units` is usually element count, while `byte_count` is
    `work_units * sizeof(dtype)`. On a 64-core system a 1MB reduction should
    stay serial instead of becoming 64 workers with 16KB slices.
    """
    if byte_count < grain_per_worker:
        return 1
    var n = worker_count(work_units)
    var max_by_bytes = byte_count // grain_per_worker
    if max_by_bytes < 1:
        return 1
    if max_by_bytes < n:
        n = max_by_bytes
    if n < 1:
        return 1
    return n


def worker_count_for_rows(row_count: Int) -> Int:
    """Resolve worker count for independent row-wise kernels.

    This legacy row-only form should be used only when per-row work is known
    to be heavy. For shape-dependent kernels, prefer
    `worker_count_for_row_elements`.
    """
    if row_count < PER_ROW_MIN:
        return 1
    var n = worker_count(row_count)
    var max_by_rows = row_count // PER_ROW_MIN
    if max_by_rows < 1:
        return 1
    if max_by_rows < n:
        n = max_by_rows
    if n < 1:
        return 1
    return n


def worker_count_for_row_elements(row_count: Int, elements_per_row: Int, elements_per_worker: Int) -> Int:
    """Resolve worker count for independent row-wise kernels by total work.

    Row kernels can only split between rows, but row count alone is not enough:
    a 32x32 softmax loses to spawn overhead, while a 32x4096 softmax has real
    work in each row. This policy caps by hardware/env, row slices, and total
    row elements per worker.
    """
    if row_count <= 1 or elements_per_row <= 0:
        return 1
    var total_elements = row_count * elements_per_row
    if total_elements < elements_per_worker:
        return 1
    var n = worker_count(row_count)
    var max_by_rows = row_count // PER_ROW_MIN
    if max_by_rows < 1:
        max_by_rows = 1
    var max_by_elements = total_elements // elements_per_worker
    if max_by_elements < 1:
        return 1
    if max_by_rows < n:
        n = max_by_rows
    if max_by_elements < n:
        n = max_by_elements
    if n < 1:
        return 1
    return n


def should_parallelize_bytes(byte_count: Int, grain: Int) -> Bool:
    """Gate for whole-tensor parallelism by byte count + op-class grain.

    Args:
        byte_count: total bytes in the operand (size * sizeof(dtype)).
        grain: grain-size constant — `REDUCE_GRAIN`, `ELEMENTWISE_LIGHT_GRAIN`,
               or `ELEMENTWISE_HEAVY_GRAIN`.

    Returns: True if the work justifies thread fan-out.
    """
    return worker_count_for_bytes(byte_count, byte_count, grain) > 1


def should_parallelize_rows(row_count: Int) -> Bool:
    """Gate for per-row parallelism (softmax, layernorm, axis-reduce).

    Below `PER_ROW_MIN` rows the thread setup (~1-10us) dominates unless each
    row is wide. Production row kernels should prefer
    `worker_count_for_row_elements` so 32x32 attention does not fan out.
    """
    return worker_count_for_rows(row_count) > 1
