"""Head-to-head benchmark: monpy parallel layer vs single-thread serial baseline.

Three workloads at five sizes per dtype:

  ser_add — `binary_same_shape_contig_typed_static[ADD]`, one thread.
  par_add — same kernel, chunked across `num_performance_cores()` threads
            via `sync_parallelize`. Measures the parallel-add primitive
            without the dispatch ladder's static fast-path interfering.
  ser_neg — single-thread `unary_contig_typed_static[NEGATE]`. Light op
            (1 SIMD instruction per element).
  par_neg — same kernel, parallel chunked. Tests whether 2MB
            `ELEMENTWISE_LIGHT_GRAIN` is the correct gate for cheap unary.
  ser_exp — single-thread `unary_contig_typed_static[EXP]`. Heavy op
            (libm-class call per lane).
  par_exp — parallel chunked. Tests whether 256KB
            `ELEMENTWISE_HEAVY_GRAIN` is the correct gate for transcendentals.

Sweep: 256KB, 1MB, 4MB, 16MB, 64MB. Picked to span both grain gates and
the L1/L2/DRAM transitions.

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
from std.math import ceildiv
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


def _binary_measures[dtype: DType, n: Int]() -> List[ThroughputMeasure]:
    return [_bytes_binary[dtype, n](), _flops[n]()]


def _unary_measures[dtype: DType, n: Int]() -> List[ThroughputMeasure]:
    return [_bytes_unary[dtype, n](), _flops[n]()]


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

    m.dump_report()
