"""Head-to-head benchmark: monpy reduce_sum_typed vs std.algorithm.sum.

monpy's kernel (`reduce_sum_typed` from src/elementwise/reduce_kernels.mojo)
is a 4-way SIMD-accumulator contig sum that breaks the FADD latency dep.
The std side is `std.algorithm.reduction.sum`'s Span overload, which
dispatches through `_reduce_generator` (sync_parallelize + vectorize +
log2_floor) with optional CPU/GPU split.

Sweep covers L1-resident through DRAM-bandwidth-bound regimes.
Self-contained: inlines monpy's kernel so we don't need the monpy
build tree.
"""

from std.algorithm import sum as std_sum
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)
from std.memory import Span
from std.memory.unsafe_pointer import alloc
from std.sys import simd_width_of, size_of


def fill_buffer[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], n: Int):
    # Deterministic, non-trivial bytes — avoid all-zero so the compiler
    # can't fold the reduction. Bounded magnitude so f32 doesn't saturate.
    for i in range(n):
        ptr[i] = Scalar[dtype](Float64(i & 0xff) * 0.0078125 + 1.0)


def reduce_sum_typed_monpy[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int
) raises -> Float64:
    """4-parallel SIMD-accumulator sum — verbatim copy of monpy's kernel."""
    comptime width = simd_width_of[dtype]()
    comptime block = width * 4
    var acc0 = SIMD[dtype, width](0)
    var acc1 = SIMD[dtype, width](0)
    var acc2 = SIMD[dtype, width](0)
    var acc3 = SIMD[dtype, width](0)
    var i = 0
    while i + block <= size:
        acc0 += src_ptr.load[width=width](i)
        acc1 += src_ptr.load[width=width](i + width)
        acc2 += src_ptr.load[width=width](i + 2 * width)
        acc3 += src_ptr.load[width=width](i + 3 * width)
        i += block
    var acc_vec = (acc0 + acc1) + (acc2 + acc3)
    while i + width <= size:
        acc_vec += src_ptr.load[width=width](i)
        i += width
    var acc = Float64(acc_vec.reduce_add()[0])
    while i < size:
        acc += Float64(src_ptr[i])
        i += 1
    return acc


@parameter
def bench_monpy_sum[dtype: DType, n: Int](mut b: Bencher) raises:
    var ptr = alloc[Scalar[dtype]](n)
    fill_buffer(ptr, n)

    @always_inline
    @parameter
    def call_fn() raises:
        var s = reduce_sum_typed_monpy(ptr, n)
        keep(s)

    b.iter[call_fn]()
    ptr.free()


@parameter
def bench_std_sum[dtype: DType, n: Int](mut b: Bencher) raises:
    var ptr = alloc[Scalar[dtype]](n)
    fill_buffer(ptr, n)

    @always_inline
    @parameter
    def call_fn() raises:
        var span = Span[Scalar[dtype], MutExternalOrigin](
            ptr=ptr, length=n
        )
        var s = std_sum(span)
        keep(s)

    b.iter[call_fn]()
    ptr.free()


def _bytes[dtype: DType, n: Int]() -> ThroughputMeasure:
    return ThroughputMeasure(
        BenchMetric.bytes, n * size_of[Scalar[dtype]]()
    )


def _flops[n: Int]() -> ThroughputMeasure:
    return ThroughputMeasure(BenchMetric.flops, n)


def _measures[dtype: DType, n: Int]() -> List[ThroughputMeasure]:
    return [_bytes[dtype, n](), _flops[n]()]


def main() raises:
    var m = Bench(BenchConfig(num_repetitions=2))

    # f32 sweep: 1k (L1) → 64k (~L2) → 1M (L3) → 16M (DRAM) → 128M (BW-bound).
    m.bench_function[bench_monpy_sum[DType.float32, 1024]](
        BenchId("monpy_sum_f32_1k"), _measures[DType.float32, 1024]()
    )
    m.bench_function[bench_std_sum[DType.float32, 1024]](
        BenchId("std_sum_f32_1k"), _measures[DType.float32, 1024]()
    )

    m.bench_function[bench_monpy_sum[DType.float32, 65536]](
        BenchId("monpy_sum_f32_64k"), _measures[DType.float32, 65536]()
    )
    m.bench_function[bench_std_sum[DType.float32, 65536]](
        BenchId("std_sum_f32_64k"), _measures[DType.float32, 65536]()
    )

    m.bench_function[bench_monpy_sum[DType.float32, 1048576]](
        BenchId("monpy_sum_f32_1M"), _measures[DType.float32, 1048576]()
    )
    m.bench_function[bench_std_sum[DType.float32, 1048576]](
        BenchId("std_sum_f32_1M"), _measures[DType.float32, 1048576]()
    )

    m.bench_function[bench_monpy_sum[DType.float32, 16777216]](
        BenchId("monpy_sum_f32_16M"), _measures[DType.float32, 16777216]()
    )
    m.bench_function[bench_std_sum[DType.float32, 16777216]](
        BenchId("std_sum_f32_16M"), _measures[DType.float32, 16777216]()
    )

    m.bench_function[bench_monpy_sum[DType.float32, 134217728]](
        BenchId("monpy_sum_f32_128M"),
        _measures[DType.float32, 134217728](),
    )
    m.bench_function[bench_std_sum[DType.float32, 134217728]](
        BenchId("std_sum_f32_128M"),
        _measures[DType.float32, 134217728](),
    )

    # f64 sweep — fewer points, since same story scaled.
    m.bench_function[bench_monpy_sum[DType.float64, 1024]](
        BenchId("monpy_sum_f64_1k"), _measures[DType.float64, 1024]()
    )
    m.bench_function[bench_std_sum[DType.float64, 1024]](
        BenchId("std_sum_f64_1k"), _measures[DType.float64, 1024]()
    )

    m.bench_function[bench_monpy_sum[DType.float64, 1048576]](
        BenchId("monpy_sum_f64_1M"), _measures[DType.float64, 1048576]()
    )
    m.bench_function[bench_std_sum[DType.float64, 1048576]](
        BenchId("std_sum_f64_1M"), _measures[DType.float64, 1048576]()
    )

    m.bench_function[bench_monpy_sum[DType.float64, 16777216]](
        BenchId("monpy_sum_f64_16M"), _measures[DType.float64, 16777216]()
    )
    m.bench_function[bench_std_sum[DType.float64, 16777216]](
        BenchId("std_sum_f64_16M"), _measures[DType.float64, 16777216]()
    )

    m.dump_report()
