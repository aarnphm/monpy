"""Head-to-head benchmark: monpy parallel layer vs single-thread serial baseline.

Five workloads per dtype:

  ser_add / par_add — `binary_same_shape_contig_typed_static[ADD]`, light op.
  ser_neg / par_neg — `unary_contig_typed_static[NEGATE]`, light op.
  ser_exp / par_exp — `unary_contig_typed_static[EXP]`, heavy op.
  ser_softmax_row / par_softmax_row — softmax over the last axis.
                                       Mirrors `_softmax_last_axis_f32` in
                                       `src/elementwise/kernels/nn.mojo`.
  ser_layernorm_row / par_layernorm_row — layernorm over the last axis.
                                           Mirrors `layer_norm_last_axis_typed`.

Element-wise sweep: 256KB, 1MB, 4MB, 16MB, 64MB.
Row-kernel sweep: 32x32 (below ROW_HEAVY_GRAIN_ELEMS=8K — parallel
should NOT help), 256x512, 1024x512, 1024x4096 (transformer-scale).

Why this exists separately from `bench_reduce.mojo` and `bench_mojo_sweep.mojo`:

  bench_reduce       — reductions only (sum/min/max/prod/mean), 4-way vs
                       8-way vs stdlib head-to-head, plus sum_par.
  bench_mojo_sweep   — TSV runner for `monpy-bench-mojo`; one ratio per row,
                       aggregated row-by-row across the kernel surface.
  bench_parallel     — element-wise ops with parallel-vs-serial pairs at
                       sweep sizes. Answers "what's the smallest N where
                       parallel wins for ADD/NEGATE/EXP?" directly.

Caveat: the production dispatch ladder (`binary_dispatch.mojo`,
`unary_dispatch.mojo`) lands at `*_static` for ADD/SUB/MUL/DIV/SIN/COS/
EXP/LOG/TANH/SQRT/NEGATE/POSITIVE/SQUARE. Those static kernels are
single-thread; the parallel-aware runtime entry only fires for less-common
ops (POWER, EXPM1, ATAN, ABS, FLOOR, RECIPROCAL, ...). Bringing parallel
into the static fast-path is a separate Layer-3 improvement and would
need its own measurement pass.

Run:

    $MODULAR_DERIVED_PATH/build/bin/mojo run -I src benches/bench_parallel.mojo
"""

from std.algorithm import sync_parallelize
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)
from std.math import ceildiv, exp as _exp, sqrt as _sqrt
from std.memory.unsafe_pointer import alloc
from std.sys import num_performance_cores, size_of

from domain import BinaryOp, UnaryOp
from elementwise.kernels.typed import (
    binary_same_shape_contig_typed_static,
    unary_contig_typed_static,
)


# ===-----------------------------------------------------------------------===
# Buffer fill — bounded magnitudes so EXP doesn't overflow at any size.
# ===-----------------------------------------------------------------------===

def fill_buffer[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], n: Int):
    # Domain (0.25, 1.5): exp range (1.28, 4.48), no overflow even at f16.
    # Mod 251 picks a prime to break stride patterns the prefetcher might exploit.
    for i in range(n):
        ptr[i] = Scalar[dtype](Float64(i % 251) * 0.005 + 0.25)


# ===-----------------------------------------------------------------------===
# Parallel wrappers around comptime-static kernels.
# ===-----------------------------------------------------------------------===
# These mirror the chunk-and-fan-out pattern from
# `elementwise/kernels/typed.mojo:282-317` (binary) and
# `elementwise/kernels/typed.mojo:700-733` (unary), but call the
# *static* op-specialized kernel inside each worker so the inner loop
# stays branch-free. Production dispatch picks `_static` over `_typed`
# for the same reason on the serial path.

def par_add[
    dtype: DType
](
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    nworkers: Int,
) raises:
    if nworkers <= 1 or size < nworkers:
        binary_same_shape_contig_typed_static[dtype, BinaryOp.ADD.value](lhs_ptr, rhs_ptr, out_ptr, size)
        return

    var chunk = ceildiv(size, nworkers)

    @parameter
    def chunk_worker(i: Int) raises:
        var start = i * chunk
        var end = start + chunk
        if end > size:
            end = size
        if start >= size:
            return
        binary_same_shape_contig_typed_static[dtype, BinaryOp.ADD.value](
            lhs_ptr + start, rhs_ptr + start, out_ptr + start, end - start
        )

    sync_parallelize[chunk_worker](nworkers)


def par_unary[
    dtype: DType, op: Int
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
    nworkers: Int,
) raises where dtype.is_floating_point():
    if nworkers <= 1 or size < nworkers:
        unary_contig_typed_static[dtype, op](src_ptr, out_ptr, size)
        return

    var chunk = ceildiv(size, nworkers)

    @parameter
    def chunk_worker(i: Int) raises:
        var start = i * chunk
        var end = start + chunk
        if end > size:
            end = size
        if start >= size:
            return
        unary_contig_typed_static[dtype, op](src_ptr + start, out_ptr + start, end - start)

    sync_parallelize[chunk_worker](nworkers)


# ===-----------------------------------------------------------------------===
# Binary ADD — light op, 1 cycle per lane.
# ===-----------------------------------------------------------------------===

@parameter
def bench_ser_add[dtype: DType, n: Int](mut b: Bencher) raises:
    var lhs = alloc[Scalar[dtype]](n)
    var rhs = alloc[Scalar[dtype]](n)
    var out = alloc[Scalar[dtype]](n)
    fill_buffer(lhs, n)
    fill_buffer(rhs, n)

    @always_inline
    @parameter
    def call_fn() raises:
        binary_same_shape_contig_typed_static[dtype, BinaryOp.ADD.value](lhs, rhs, out, n)
        keep(out[0])

    b.iter[call_fn]()
    lhs.free()
    rhs.free()
    out.free()


@parameter
def bench_par_add[dtype: DType, n: Int](mut b: Bencher) raises:
    var lhs = alloc[Scalar[dtype]](n)
    var rhs = alloc[Scalar[dtype]](n)
    var out = alloc[Scalar[dtype]](n)
    fill_buffer(lhs, n)
    fill_buffer(rhs, n)

    @always_inline
    @parameter
    def call_fn() raises:
        par_add[dtype](lhs, rhs, out, n, num_performance_cores())
        keep(out[0])

    b.iter[call_fn]()
    lhs.free()
    rhs.free()
    out.free()


# ===-----------------------------------------------------------------------===
# Unary NEGATE — light op, 1 cycle per lane.
# ===-----------------------------------------------------------------------===

@parameter
def bench_ser_neg[dtype: DType, n: Int](mut b: Bencher) raises where dtype.is_floating_point():
    var src = alloc[Scalar[dtype]](n)
    var out = alloc[Scalar[dtype]](n)
    fill_buffer(src, n)

    @always_inline
    @parameter
    def call_fn() raises:
        unary_contig_typed_static[dtype, UnaryOp.NEGATE.value](src, out, n)
        keep(out[0])

    b.iter[call_fn]()
    src.free()
    out.free()


@parameter
def bench_par_neg[dtype: DType, n: Int](mut b: Bencher) raises where dtype.is_floating_point():
    var src = alloc[Scalar[dtype]](n)
    var out = alloc[Scalar[dtype]](n)
    fill_buffer(src, n)

    @always_inline
    @parameter
    def call_fn() raises:
        par_unary[dtype, UnaryOp.NEGATE.value](src, out, n, num_performance_cores())
        keep(out[0])

    b.iter[call_fn]()
    src.free()
    out.free()


# ===-----------------------------------------------------------------------===
# Unary EXP — heavy op, libm call per lane.
# ===-----------------------------------------------------------------------===

@parameter
def bench_ser_exp[dtype: DType, n: Int](mut b: Bencher) raises where dtype.is_floating_point():
    var src = alloc[Scalar[dtype]](n)
    var out = alloc[Scalar[dtype]](n)
    fill_buffer(src, n)

    @always_inline
    @parameter
    def call_fn() raises:
        unary_contig_typed_static[dtype, UnaryOp.EXP.value](src, out, n)
        keep(out[0])

    b.iter[call_fn]()
    src.free()
    out.free()


@parameter
def bench_par_exp[dtype: DType, n: Int](mut b: Bencher) raises where dtype.is_floating_point():
    var src = alloc[Scalar[dtype]](n)
    var out = alloc[Scalar[dtype]](n)
    fill_buffer(src, n)

    @always_inline
    @parameter
    def call_fn() raises:
        par_unary[dtype, UnaryOp.EXP.value](src, out, n, num_performance_cores())
        keep(out[0])

    b.iter[call_fn]()
    src.free()
    out.free()


# ===-----------------------------------------------------------------------===
# Row kernels — softmax + layernorm last-axis
# ===-----------------------------------------------------------------------===
# Mirror the per-row body from `src/elementwise/kernels/nn.mojo`. The
# parallel wrapper does row-range partitioning identical to the
# production `_softmax_last_axis_f32`; we keep the bench standalone
# so MONPY_THREADS doesn't influence the parallel side.

def _softmax_rows[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    row_start: Int,
    row_end: Int,
    cols: Int,
) raises where dtype.is_floating_point():
    for row in range(row_start, row_end):
        var base = row * cols
        var row_max = Float64(src_ptr[base])
        for col in range(1, cols):
            var v = Float64(src_ptr[base + col])
            if v > row_max:
                row_max = v
        var denom = 0.0
        for col in range(cols):
            var w = _exp(Float64(src_ptr[base + col]) - row_max)
            denom += w
            out_ptr[base + col] = Scalar[dtype](w)
        var inv = 1.0 / denom
        for col in range(cols):
            out_ptr[base + col] = Scalar[dtype](Float64(out_ptr[base + col]) * inv)


def softmax_row_serial[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rows: Int,
    cols: Int,
) raises where dtype.is_floating_point():
    _softmax_rows[dtype](src_ptr, out_ptr, 0, rows, cols)


def softmax_row_parallel[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rows: Int,
    cols: Int,
    nworkers: Int,
) raises where dtype.is_floating_point():
    if nworkers <= 1 or rows < nworkers:
        _softmax_rows[dtype](src_ptr, out_ptr, 0, rows, cols)
        return
    var chunk = ceildiv(rows, nworkers)

    @parameter
    def chunk_worker(i: Int) raises:
        var start = i * chunk
        var end = start + chunk
        if end > rows:
            end = rows
        if start >= rows:
            return
        _softmax_rows[dtype](src_ptr, out_ptr, start, end, cols)

    sync_parallelize[chunk_worker](nworkers)


def _layernorm_rows[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    gain_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    row_start: Int,
    row_end: Int,
    cols: Int,
    eps: Float64,
) raises where dtype.is_floating_point():
    for row in range(row_start, row_end):
        var base = row * cols
        var sum = 0.0
        var sumsq = 0.0
        for col in range(cols):
            var value = Float64(src_ptr[base + col])
            sum += value
            sumsq += value * value
        var mean = sum / Float64(cols)
        var variance = sumsq / Float64(cols) - mean * mean
        if variance < 0.0:
            variance = 0.0
        var inv_std = 1.0 / _sqrt(variance + eps)
        for col in range(cols):
            var v = (Float64(src_ptr[base + col]) - mean) * inv_std
            v = v * Float64(gain_ptr[col]) + Float64(bias_ptr[col])
            out_ptr[base + col] = Scalar[dtype](v)


def layernorm_row_serial[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    gain_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rows: Int,
    cols: Int,
    eps: Float64,
) raises where dtype.is_floating_point():
    _layernorm_rows[dtype](src_ptr, gain_ptr, bias_ptr, out_ptr, 0, rows, cols, eps)


def layernorm_row_parallel[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    gain_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rows: Int,
    cols: Int,
    nworkers: Int,
    eps: Float64,
) raises where dtype.is_floating_point():
    if nworkers <= 1 or rows < nworkers:
        _layernorm_rows[dtype](src_ptr, gain_ptr, bias_ptr, out_ptr, 0, rows, cols, eps)
        return
    var chunk = ceildiv(rows, nworkers)

    @parameter
    def chunk_worker(i: Int) raises:
        var start = i * chunk
        var end = start + chunk
        if end > rows:
            end = rows
        if start >= rows:
            return
        _layernorm_rows[dtype](src_ptr, gain_ptr, bias_ptr, out_ptr, start, end, cols, eps)

    sync_parallelize[chunk_worker](nworkers)


@parameter
def bench_ser_softmax_row[dtype: DType, rows: Int, cols: Int](mut b: Bencher) raises where dtype.is_floating_point():
    var src = alloc[Scalar[dtype]](rows * cols)
    var out = alloc[Scalar[dtype]](rows * cols)
    fill_buffer(src, rows * cols)

    @always_inline
    @parameter
    def call_fn() raises:
        softmax_row_serial[dtype](src, out, rows, cols)
        keep(out[0])

    b.iter[call_fn]()
    src.free()
    out.free()


@parameter
def bench_par_softmax_row[dtype: DType, rows: Int, cols: Int](mut b: Bencher) raises where dtype.is_floating_point():
    var src = alloc[Scalar[dtype]](rows * cols)
    var out = alloc[Scalar[dtype]](rows * cols)
    fill_buffer(src, rows * cols)

    @always_inline
    @parameter
    def call_fn() raises:
        softmax_row_parallel[dtype](src, out, rows, cols, num_performance_cores())
        keep(out[0])

    b.iter[call_fn]()
    src.free()
    out.free()


@parameter
def bench_ser_layernorm_row[dtype: DType, rows: Int, cols: Int](mut b: Bencher) raises where dtype.is_floating_point():
    var src = alloc[Scalar[dtype]](rows * cols)
    var out = alloc[Scalar[dtype]](rows * cols)
    var gain = alloc[Scalar[dtype]](cols)
    var bias = alloc[Scalar[dtype]](cols)
    fill_buffer(src, rows * cols)
    fill_buffer(gain, cols)
    fill_buffer(bias, cols)

    @always_inline
    @parameter
    def call_fn() raises:
        layernorm_row_serial[dtype](src, gain, bias, out, rows, cols, 1e-5)
        keep(out[0])

    b.iter[call_fn]()
    src.free()
    out.free()
    gain.free()
    bias.free()


@parameter
def bench_par_layernorm_row[dtype: DType, rows: Int, cols: Int](mut b: Bencher) raises where dtype.is_floating_point():
    var src = alloc[Scalar[dtype]](rows * cols)
    var out = alloc[Scalar[dtype]](rows * cols)
    var gain = alloc[Scalar[dtype]](cols)
    var bias = alloc[Scalar[dtype]](cols)
    fill_buffer(src, rows * cols)
    fill_buffer(gain, cols)
    fill_buffer(bias, cols)

    @always_inline
    @parameter
    def call_fn() raises:
        layernorm_row_parallel[dtype](src, gain, bias, out, rows, cols, num_performance_cores(), 1e-5)
        keep(out[0])

    b.iter[call_fn]()
    src.free()
    out.free()
    gain.free()
    bias.free()


# ===-----------------------------------------------------------------------===
# Throughput measures
# ===-----------------------------------------------------------------------===

def _bytes_binary[dtype: DType, n: Int]() -> ThroughputMeasure:
    # Binary same-shape: 2 reads + 1 write per element.
    return ThroughputMeasure(BenchMetric.bytes, 3 * n * size_of[Scalar[dtype]]())


def _bytes_unary[dtype: DType, n: Int]() -> ThroughputMeasure:
    # Unary: 1 read + 1 write per element.
    return ThroughputMeasure(BenchMetric.bytes, 2 * n * size_of[Scalar[dtype]]())


def _flops[n: Int]() -> ThroughputMeasure:
    return ThroughputMeasure(BenchMetric.flops, n)


def _bytes_row[dtype: DType, rows: Int, cols: Int]() -> ThroughputMeasure:
    # Softmax: 2 reads + 1 write per element. Layernorm: 3 reads + 1 write
    # per element (gain/bias touched once per column row but counted once).
    return ThroughputMeasure(BenchMetric.bytes, 3 * rows * cols * size_of[Scalar[dtype]]())


def _binary_measures[dtype: DType, n: Int]() -> List[ThroughputMeasure]:
    return [_bytes_binary[dtype, n](), _flops[n]()]


def _unary_measures[dtype: DType, n: Int]() -> List[ThroughputMeasure]:
    return [_bytes_unary[dtype, n](), _flops[n]()]


def _row_measures[dtype: DType, rows: Int, cols: Int]() -> List[ThroughputMeasure]:
    return [_bytes_row[dtype, rows, cols](), _flops[rows * cols]()]


# ===-----------------------------------------------------------------------===
# Main — sweep across ELEMENTWISE_LIGHT (2MB) and ELEMENTWISE_HEAVY (256KB) gates.
# ===-----------------------------------------------------------------------===

def main() raises:
    var m = Bench(BenchConfig(num_repetitions=2))

    # f32 sweep — sizes in element count: 64K (256KB), 256K (1MB), 1M (4MB),
    # 4M (16MB), 16M (64MB). Crosses both gates and L2 boundary at ~3-4MB.

    # Binary ADD
    m.bench_function[bench_ser_add[DType.float32, 65536]](BenchId("ser_add_f32_64k"), _binary_measures[DType.float32, 65536]())
    m.bench_function[bench_par_add[DType.float32, 65536]](BenchId("par_add_f32_64k"), _binary_measures[DType.float32, 65536]())

    m.bench_function[bench_ser_add[DType.float32, 262144]](BenchId("ser_add_f32_256k"), _binary_measures[DType.float32, 262144]())
    m.bench_function[bench_par_add[DType.float32, 262144]](BenchId("par_add_f32_256k"), _binary_measures[DType.float32, 262144]())

    m.bench_function[bench_ser_add[DType.float32, 1048576]](BenchId("ser_add_f32_1m"), _binary_measures[DType.float32, 1048576]())
    m.bench_function[bench_par_add[DType.float32, 1048576]](BenchId("par_add_f32_1m"), _binary_measures[DType.float32, 1048576]())

    m.bench_function[bench_ser_add[DType.float32, 4194304]](BenchId("ser_add_f32_4m"), _binary_measures[DType.float32, 4194304]())
    m.bench_function[bench_par_add[DType.float32, 4194304]](BenchId("par_add_f32_4m"), _binary_measures[DType.float32, 4194304]())

    m.bench_function[bench_ser_add[DType.float32, 16777216]](BenchId("ser_add_f32_16m"), _binary_measures[DType.float32, 16777216]())
    m.bench_function[bench_par_add[DType.float32, 16777216]](BenchId("par_add_f32_16m"), _binary_measures[DType.float32, 16777216]())

    # Unary NEGATE
    m.bench_function[bench_ser_neg[DType.float32, 65536]](BenchId("ser_neg_f32_64k"), _unary_measures[DType.float32, 65536]())
    m.bench_function[bench_par_neg[DType.float32, 65536]](BenchId("par_neg_f32_64k"), _unary_measures[DType.float32, 65536]())

    m.bench_function[bench_ser_neg[DType.float32, 1048576]](BenchId("ser_neg_f32_1m"), _unary_measures[DType.float32, 1048576]())
    m.bench_function[bench_par_neg[DType.float32, 1048576]](BenchId("par_neg_f32_1m"), _unary_measures[DType.float32, 1048576]())

    m.bench_function[bench_ser_neg[DType.float32, 4194304]](BenchId("ser_neg_f32_4m"), _unary_measures[DType.float32, 4194304]())
    m.bench_function[bench_par_neg[DType.float32, 4194304]](BenchId("par_neg_f32_4m"), _unary_measures[DType.float32, 4194304]())

    m.bench_function[bench_ser_neg[DType.float32, 16777216]](BenchId("ser_neg_f32_16m"), _unary_measures[DType.float32, 16777216]())
    m.bench_function[bench_par_neg[DType.float32, 16777216]](BenchId("par_neg_f32_16m"), _unary_measures[DType.float32, 16777216]())

    # Unary EXP
    m.bench_function[bench_ser_exp[DType.float32, 65536]](BenchId("ser_exp_f32_64k"), _unary_measures[DType.float32, 65536]())
    m.bench_function[bench_par_exp[DType.float32, 65536]](BenchId("par_exp_f32_64k"), _unary_measures[DType.float32, 65536]())

    m.bench_function[bench_ser_exp[DType.float32, 262144]](BenchId("ser_exp_f32_256k"), _unary_measures[DType.float32, 262144]())
    m.bench_function[bench_par_exp[DType.float32, 262144]](BenchId("par_exp_f32_256k"), _unary_measures[DType.float32, 262144]())

    m.bench_function[bench_ser_exp[DType.float32, 1048576]](BenchId("ser_exp_f32_1m"), _unary_measures[DType.float32, 1048576]())
    m.bench_function[bench_par_exp[DType.float32, 1048576]](BenchId("par_exp_f32_1m"), _unary_measures[DType.float32, 1048576]())

    m.bench_function[bench_ser_exp[DType.float32, 4194304]](BenchId("ser_exp_f32_4m"), _unary_measures[DType.float32, 4194304]())
    m.bench_function[bench_par_exp[DType.float32, 4194304]](BenchId("par_exp_f32_4m"), _unary_measures[DType.float32, 4194304]())

    # f64 — narrower sweep; bandwidth doubles per lane so DRAM ceiling hits sooner.
    m.bench_function[bench_ser_add[DType.float64, 1048576]](BenchId("ser_add_f64_1m"), _binary_measures[DType.float64, 1048576]())
    m.bench_function[bench_par_add[DType.float64, 1048576]](BenchId("par_add_f64_1m"), _binary_measures[DType.float64, 1048576]())

    m.bench_function[bench_ser_add[DType.float64, 4194304]](BenchId("ser_add_f64_4m"), _binary_measures[DType.float64, 4194304]())
    m.bench_function[bench_par_add[DType.float64, 4194304]](BenchId("par_add_f64_4m"), _binary_measures[DType.float64, 4194304]())

    m.bench_function[bench_ser_neg[DType.float64, 1048576]](BenchId("ser_neg_f64_1m"), _unary_measures[DType.float64, 1048576]())
    m.bench_function[bench_par_neg[DType.float64, 1048576]](BenchId("par_neg_f64_1m"), _unary_measures[DType.float64, 1048576]())

    m.bench_function[bench_ser_exp[DType.float64, 262144]](BenchId("ser_exp_f64_256k"), _unary_measures[DType.float64, 262144]())
    m.bench_function[bench_par_exp[DType.float64, 262144]](BenchId("par_exp_f64_256k"), _unary_measures[DType.float64, 262144]())

    m.bench_function[bench_ser_exp[DType.float64, 1048576]](BenchId("ser_exp_f64_1m"), _unary_measures[DType.float64, 1048576]())
    m.bench_function[bench_par_exp[DType.float64, 1048576]](BenchId("par_exp_f64_1m"), _unary_measures[DType.float64, 1048576]())

    # Row kernels — softmax + layernorm last-axis at production attention
    # shapes. The gate `worker_count_for_row_elements` keeps tiny shapes
    # serial; these rows test (a) that the gate makes the right call at
    # 32x32 and (b) that parallel actually wins at the larger shapes.

    # Tiny: below ROW_HEAVY_GRAIN_ELEMS=8K. Parallel wrapper short-circuits
    # back to serial inside the function — the par_ row should match ser_.
    m.bench_function[bench_ser_softmax_row[DType.float32, 32, 32]](BenchId("ser_softmax_f32_32x32"), _row_measures[DType.float32, 32, 32]())
    m.bench_function[bench_par_softmax_row[DType.float32, 32, 32]](BenchId("par_softmax_f32_32x32"), _row_measures[DType.float32, 32, 32]())

    # Mid: 256 rows x 512 cols = 131K elements. 16x ROW_HEAVY_GRAIN_ELEMS;
    # parallel should fan out to ceil(131K / 8K) = 17 workers (capped by hw).
    m.bench_function[bench_ser_softmax_row[DType.float32, 256, 512]](BenchId("ser_softmax_f32_256x512"), _row_measures[DType.float32, 256, 512]())
    m.bench_function[bench_par_softmax_row[DType.float32, 256, 512]](BenchId("par_softmax_f32_256x512"), _row_measures[DType.float32, 256, 512]())

    # Large: 1024 rows x 512 cols = 524K elements.
    m.bench_function[bench_ser_softmax_row[DType.float32, 1024, 512]](BenchId("ser_softmax_f32_1024x512"), _row_measures[DType.float32, 1024, 512]())
    m.bench_function[bench_par_softmax_row[DType.float32, 1024, 512]](BenchId("par_softmax_f32_1024x512"), _row_measures[DType.float32, 1024, 512]())

    # Vocab projection scale: 1024 rows x 4096 cols = 4M elements.
    m.bench_function[bench_ser_softmax_row[DType.float32, 1024, 4096]](BenchId("ser_softmax_f32_1024x4096"), _row_measures[DType.float32, 1024, 4096]())
    m.bench_function[bench_par_softmax_row[DType.float32, 1024, 4096]](BenchId("par_softmax_f32_1024x4096"), _row_measures[DType.float32, 1024, 4096]())

    # Layernorm — same shape sweep, with extra gain/bias bandwidth.
    m.bench_function[bench_ser_layernorm_row[DType.float32, 32, 32]](BenchId("ser_layernorm_f32_32x32"), _row_measures[DType.float32, 32, 32]())
    m.bench_function[bench_par_layernorm_row[DType.float32, 32, 32]](BenchId("par_layernorm_f32_32x32"), _row_measures[DType.float32, 32, 32]())

    m.bench_function[bench_ser_layernorm_row[DType.float32, 256, 512]](BenchId("ser_layernorm_f32_256x512"), _row_measures[DType.float32, 256, 512]())
    m.bench_function[bench_par_layernorm_row[DType.float32, 256, 512]](BenchId("par_layernorm_f32_256x512"), _row_measures[DType.float32, 256, 512]())

    m.bench_function[bench_ser_layernorm_row[DType.float32, 1024, 512]](BenchId("ser_layernorm_f32_1024x512"), _row_measures[DType.float32, 1024, 512]())
    m.bench_function[bench_par_layernorm_row[DType.float32, 1024, 512]](BenchId("par_layernorm_f32_1024x512"), _row_measures[DType.float32, 1024, 512]())

    m.bench_function[bench_ser_layernorm_row[DType.float32, 1024, 4096]](BenchId("ser_layernorm_f32_1024x4096"), _row_measures[DType.float32, 1024, 4096]())
    m.bench_function[bench_par_layernorm_row[DType.float32, 1024, 4096]](BenchId("par_layernorm_f32_1024x4096"), _row_measures[DType.float32, 1024, 4096]())

    m.dump_report()
