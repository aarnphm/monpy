"""Head-to-head benchmark: monpy reduce_*_typed vs std.algorithm.{sum,mean,min,max,product}.

Per primitive, three variants on identical buffers across L1/L2/L3/DRAM-resident sizes:

  monpy4_<op>_* — verbatim copy of monpy's pre-improvement 4-way SIMD kernel
  monpy8_<op>_* — current monpy kernel (8-way unrolled, after this round of work)
  std____<op>_* — `std.algorithm.reduction.<op>`'s Span overload

The 4-way variant exists as a fixed historical baseline so we can keep
demonstrating the pipeline-saturation gap on Apple Silicon (2-IPC FADD/
FMIN/FMAX/FMUL × ~3 cyc latency → ~6 ops in flight; 4 accumulators
under-saturates, 8 saturates).

Self-contained: inlines monpy's kernels so we don't need the monpy
build tree. Run:

    $MODULAR_DERIVED_PATH/build/bin/mojo benches/bench_reduce.mojo

See docs/benchmarks.md for results and the improvement plan.
"""

from std.algorithm import (
    sum as std_sum,
    mean as std_mean,
    min as std_min,
    max as std_max,
    product as std_product,
    sync_parallelize,
)
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)
from std.math import ceildiv, min as _simd_min, max as _simd_max
from std.memory import Span
from std.memory.unsafe_pointer import alloc
from std.sys import num_performance_cores, simd_width_of, size_of


# ===-----------------------------------------------------------------------===
# Buffer fill — deterministic non-trivial data
# ===-----------------------------------------------------------------------===

def fill_buffer[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], n: Int):
    # Bounded magnitude in (1, 3) so:
    # - product over large N stays finite (won't underflow/overflow f32 at 1M)
    # - sum/mean don't catastrophically cancel
    # - min/max have varied positions
    for i in range(n):
        ptr[i] = Scalar[dtype](Float64(i & 0xff) * 0.0078125 + 1.0)


# ===-----------------------------------------------------------------------===
# 4-way (legacy) kernels — fixed historical baseline
# ===-----------------------------------------------------------------------===

def sum4[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int
) raises -> Float64:
    comptime width = simd_width_of[dtype]()
    comptime block = width * 4
    var a0 = SIMD[dtype, width](0)
    var a1 = SIMD[dtype, width](0)
    var a2 = SIMD[dtype, width](0)
    var a3 = SIMD[dtype, width](0)
    var i = 0
    while i + block <= size:
        a0 += src_ptr.load[width=width](i)
        a1 += src_ptr.load[width=width](i + width)
        a2 += src_ptr.load[width=width](i + 2 * width)
        a3 += src_ptr.load[width=width](i + 3 * width)
        i += block
    var acc_vec = (a0 + a1) + (a2 + a3)
    while i + width <= size:
        acc_vec += src_ptr.load[width=width](i)
        i += width
    var acc = Float64(acc_vec.reduce_add()[0])
    while i < size:
        acc += Float64(src_ptr[i])
        i += 1
    return acc


def min4[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int
) raises -> Scalar[dtype]:
    comptime width = simd_width_of[dtype]()
    comptime block = width * 4
    if size < block:
        var acc = src_ptr[0]
        for j in range(1, size):
            var v = src_ptr[j]
            if v < acc:
                acc = v
        return acc
    var v0 = src_ptr.load[width=width](0)
    var v1 = src_ptr.load[width=width](width)
    var v2 = src_ptr.load[width=width](2 * width)
    var v3 = src_ptr.load[width=width](3 * width)
    var i = block
    while i + block <= size:
        v0 = _simd_min(v0, src_ptr.load[width=width](i))
        v1 = _simd_min(v1, src_ptr.load[width=width](i + width))
        v2 = _simd_min(v2, src_ptr.load[width=width](i + 2 * width))
        v3 = _simd_min(v3, src_ptr.load[width=width](i + 3 * width))
        i += block
    var acc_vec = _simd_min(_simd_min(v0, v1), _simd_min(v2, v3))
    while i + width <= size:
        acc_vec = _simd_min(acc_vec, src_ptr.load[width=width](i))
        i += width
    var acc = acc_vec.reduce_min()
    while i < size:
        var v = src_ptr[i]
        if v < acc:
            acc = v
        i += 1
    return acc


def max4[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int
) raises -> Scalar[dtype]:
    comptime width = simd_width_of[dtype]()
    comptime block = width * 4
    if size < block:
        var acc = src_ptr[0]
        for j in range(1, size):
            var v = src_ptr[j]
            if v > acc:
                acc = v
        return acc
    var v0 = src_ptr.load[width=width](0)
    var v1 = src_ptr.load[width=width](width)
    var v2 = src_ptr.load[width=width](2 * width)
    var v3 = src_ptr.load[width=width](3 * width)
    var i = block
    while i + block <= size:
        v0 = _simd_max(v0, src_ptr.load[width=width](i))
        v1 = _simd_max(v1, src_ptr.load[width=width](i + width))
        v2 = _simd_max(v2, src_ptr.load[width=width](i + 2 * width))
        v3 = _simd_max(v3, src_ptr.load[width=width](i + 3 * width))
        i += block
    var acc_vec = _simd_max(_simd_max(v0, v1), _simd_max(v2, v3))
    while i + width <= size:
        acc_vec = _simd_max(acc_vec, src_ptr.load[width=width](i))
        i += width
    var acc = acc_vec.reduce_max()
    while i < size:
        var v = src_ptr[i]
        if v > acc:
            acc = v
        i += 1
    return acc


def prod4[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int
) raises -> Scalar[dtype]:
    comptime width = simd_width_of[dtype]()
    comptime block = width * 4
    var v0 = SIMD[dtype, width](1)
    var v1 = SIMD[dtype, width](1)
    var v2 = SIMD[dtype, width](1)
    var v3 = SIMD[dtype, width](1)
    var i = 0
    while i + block <= size:
        v0 *= src_ptr.load[width=width](i)
        v1 *= src_ptr.load[width=width](i + width)
        v2 *= src_ptr.load[width=width](i + 2 * width)
        v3 *= src_ptr.load[width=width](i + 3 * width)
        i += block
    var acc_vec = (v0 * v1) * (v2 * v3)
    while i + width <= size:
        acc_vec *= src_ptr.load[width=width](i)
        i += width
    var acc = acc_vec.reduce_mul()
    while i < size:
        acc *= src_ptr[i]
        i += 1
    return acc


# ===-----------------------------------------------------------------------===
# 8-way kernels — current monpy production form
# ===-----------------------------------------------------------------------===

def sum8[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int
) raises -> Float64:
    comptime width = simd_width_of[dtype]()
    comptime block = width * 8
    var a0 = SIMD[dtype, width](0)
    var a1 = SIMD[dtype, width](0)
    var a2 = SIMD[dtype, width](0)
    var a3 = SIMD[dtype, width](0)
    var a4 = SIMD[dtype, width](0)
    var a5 = SIMD[dtype, width](0)
    var a6 = SIMD[dtype, width](0)
    var a7 = SIMD[dtype, width](0)
    var i = 0
    while i + block <= size:
        a0 += src_ptr.load[width=width](i)
        a1 += src_ptr.load[width=width](i + width)
        a2 += src_ptr.load[width=width](i + 2 * width)
        a3 += src_ptr.load[width=width](i + 3 * width)
        a4 += src_ptr.load[width=width](i + 4 * width)
        a5 += src_ptr.load[width=width](i + 5 * width)
        a6 += src_ptr.load[width=width](i + 6 * width)
        a7 += src_ptr.load[width=width](i + 7 * width)
        i += block
    var acc_vec = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))
    while i + width <= size:
        acc_vec += src_ptr.load[width=width](i)
        i += width
    var acc = Float64(acc_vec.reduce_add()[0])
    while i < size:
        acc += Float64(src_ptr[i])
        i += 1
    return acc


def min8[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int
) raises -> Scalar[dtype]:
    comptime width = simd_width_of[dtype]()
    comptime block = width * 8
    if size < block:
        var acc = src_ptr[0]
        for j in range(1, size):
            var v = src_ptr[j]
            if v < acc:
                acc = v
        return acc
    var v0 = src_ptr.load[width=width](0)
    var v1 = src_ptr.load[width=width](width)
    var v2 = src_ptr.load[width=width](2 * width)
    var v3 = src_ptr.load[width=width](3 * width)
    var v4 = src_ptr.load[width=width](4 * width)
    var v5 = src_ptr.load[width=width](5 * width)
    var v6 = src_ptr.load[width=width](6 * width)
    var v7 = src_ptr.load[width=width](7 * width)
    var i = block
    while i + block <= size:
        v0 = _simd_min(v0, src_ptr.load[width=width](i))
        v1 = _simd_min(v1, src_ptr.load[width=width](i + width))
        v2 = _simd_min(v2, src_ptr.load[width=width](i + 2 * width))
        v3 = _simd_min(v3, src_ptr.load[width=width](i + 3 * width))
        v4 = _simd_min(v4, src_ptr.load[width=width](i + 4 * width))
        v5 = _simd_min(v5, src_ptr.load[width=width](i + 5 * width))
        v6 = _simd_min(v6, src_ptr.load[width=width](i + 6 * width))
        v7 = _simd_min(v7, src_ptr.load[width=width](i + 7 * width))
        i += block
    var acc_vec = _simd_min(
        _simd_min(_simd_min(v0, v1), _simd_min(v2, v3)),
        _simd_min(_simd_min(v4, v5), _simd_min(v6, v7)),
    )
    while i + width <= size:
        acc_vec = _simd_min(acc_vec, src_ptr.load[width=width](i))
        i += width
    var acc = acc_vec.reduce_min()
    while i < size:
        var v = src_ptr[i]
        if v < acc:
            acc = v
        i += 1
    return acc


def max8[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int
) raises -> Scalar[dtype]:
    comptime width = simd_width_of[dtype]()
    comptime block = width * 8
    if size < block:
        var acc = src_ptr[0]
        for j in range(1, size):
            var v = src_ptr[j]
            if v > acc:
                acc = v
        return acc
    var v0 = src_ptr.load[width=width](0)
    var v1 = src_ptr.load[width=width](width)
    var v2 = src_ptr.load[width=width](2 * width)
    var v3 = src_ptr.load[width=width](3 * width)
    var v4 = src_ptr.load[width=width](4 * width)
    var v5 = src_ptr.load[width=width](5 * width)
    var v6 = src_ptr.load[width=width](6 * width)
    var v7 = src_ptr.load[width=width](7 * width)
    var i = block
    while i + block <= size:
        v0 = _simd_max(v0, src_ptr.load[width=width](i))
        v1 = _simd_max(v1, src_ptr.load[width=width](i + width))
        v2 = _simd_max(v2, src_ptr.load[width=width](i + 2 * width))
        v3 = _simd_max(v3, src_ptr.load[width=width](i + 3 * width))
        v4 = _simd_max(v4, src_ptr.load[width=width](i + 4 * width))
        v5 = _simd_max(v5, src_ptr.load[width=width](i + 5 * width))
        v6 = _simd_max(v6, src_ptr.load[width=width](i + 6 * width))
        v7 = _simd_max(v7, src_ptr.load[width=width](i + 7 * width))
        i += block
    var acc_vec = _simd_max(
        _simd_max(_simd_max(v0, v1), _simd_max(v2, v3)),
        _simd_max(_simd_max(v4, v5), _simd_max(v6, v7)),
    )
    while i + width <= size:
        acc_vec = _simd_max(acc_vec, src_ptr.load[width=width](i))
        i += width
    var acc = acc_vec.reduce_max()
    while i < size:
        var v = src_ptr[i]
        if v > acc:
            acc = v
        i += 1
    return acc


def prod8[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int
) raises -> Scalar[dtype]:
    comptime width = simd_width_of[dtype]()
    comptime block = width * 8
    var v0 = SIMD[dtype, width](1)
    var v1 = SIMD[dtype, width](1)
    var v2 = SIMD[dtype, width](1)
    var v3 = SIMD[dtype, width](1)
    var v4 = SIMD[dtype, width](1)
    var v5 = SIMD[dtype, width](1)
    var v6 = SIMD[dtype, width](1)
    var v7 = SIMD[dtype, width](1)
    var i = 0
    while i + block <= size:
        v0 *= src_ptr.load[width=width](i)
        v1 *= src_ptr.load[width=width](i + width)
        v2 *= src_ptr.load[width=width](i + 2 * width)
        v3 *= src_ptr.load[width=width](i + 3 * width)
        v4 *= src_ptr.load[width=width](i + 4 * width)
        v5 *= src_ptr.load[width=width](i + 5 * width)
        v6 *= src_ptr.load[width=width](i + 6 * width)
        v7 *= src_ptr.load[width=width](i + 7 * width)
        i += block
    var acc_vec = ((v0 * v1) * (v2 * v3)) * ((v4 * v5) * (v6 * v7))
    while i + width <= size:
        acc_vec *= src_ptr.load[width=width](i)
        i += width
    var acc = acc_vec.reduce_mul()
    while i < size:
        acc *= src_ptr[i]
        i += 1
    return acc


# ===-----------------------------------------------------------------------===
# Multi-thread sum — pushes past single-core DRAM ceiling
# ===-----------------------------------------------------------------------===

def sum_par[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int
) raises -> Float64:
    """Parallel sum: 8-way SIMD per chunk, sync_parallelize across P-cores.

    Per-thread scheme:
      1. Pick `nworkers = num_performance_cores()` (8 on M3/M4 Pro).
      2. Worker i sums its strided slice using sum8.
      3. Partials live in a heap-allocated Float64 array; the master
         walks it for the final ~10-element scalar reduction.

    Below `grain_size` the setup cost (thread spawn, partial array
    alloc/free) outweighs the speedup; the caller short-circuits to
    single-thread sum8 in that case.

    Returns: Float64 sum (matches sum8's promotion semantics).
    """
    comptime grain_size = 1 << 20  # 1M elements gate
    if size < grain_size:
        return sum8(src_ptr, size)

    var nworkers = num_performance_cores()
    var partials = alloc[Float64](nworkers)
    var chunk = ceildiv(size, nworkers)

    @parameter
    def worker(i: Int) raises:
        var start = i * chunk
        var end = start + chunk
        if end > size:
            end = size
        if start >= size:
            partials[i] = 0.0
            return
        partials[i] = sum8(src_ptr + start, end - start)

    sync_parallelize[worker](nworkers)

    var total = Float64(0)
    for i in range(nworkers):
        total += partials[i]
    partials.free()
    return total


@parameter
def bench_sum_par[dtype: DType, n: Int](mut b: Bencher) raises:
    var ptr = alloc[Scalar[dtype]](n)
    fill_buffer(ptr, n)

    @always_inline
    @parameter
    def call_fn() raises:
        var s = sum_par(ptr, n)
        keep(s)

    b.iter[call_fn]()
    ptr.free()


# ===-----------------------------------------------------------------------===
# Bench wrappers — one bench function per (kernel, op, dtype, n)
# ===-----------------------------------------------------------------------===

@parameter
def bench_sum4[dtype: DType, n: Int](mut b: Bencher) raises:
    var ptr = alloc[Scalar[dtype]](n)
    fill_buffer(ptr, n)

    @always_inline
    @parameter
    def call_fn() raises:
        var s = sum4(ptr, n)
        keep(s)

    b.iter[call_fn]()
    ptr.free()


@parameter
def bench_sum8[dtype: DType, n: Int](mut b: Bencher) raises:
    var ptr = alloc[Scalar[dtype]](n)
    fill_buffer(ptr, n)

    @always_inline
    @parameter
    def call_fn() raises:
        var s = sum8(ptr, n)
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
        var span = Span[Scalar[dtype], MutExternalOrigin](ptr=ptr, length=n)
        var s = std_sum(span)
        keep(s)

    b.iter[call_fn]()
    ptr.free()


# mean8 = sum8 / n; we don't need a separate kernel, but we do bench std_mean.
@parameter
def bench_mean8[dtype: DType, n: Int](mut b: Bencher) raises:
    var ptr = alloc[Scalar[dtype]](n)
    fill_buffer(ptr, n)

    @always_inline
    @parameter
    def call_fn() raises:
        var s = sum8(ptr, n) / Float64(n)
        keep(s)

    b.iter[call_fn]()
    ptr.free()


@parameter
def bench_std_mean[dtype: DType, n: Int](mut b: Bencher) raises:
    var ptr = alloc[Scalar[dtype]](n)
    fill_buffer(ptr, n)

    @always_inline
    @parameter
    def call_fn() raises:
        var span = Span[Scalar[dtype], MutExternalOrigin](ptr=ptr, length=n)
        var s = std_mean(span)
        keep(s)

    b.iter[call_fn]()
    ptr.free()


@parameter
def bench_min4[dtype: DType, n: Int](mut b: Bencher) raises:
    var ptr = alloc[Scalar[dtype]](n)
    fill_buffer(ptr, n)

    @always_inline
    @parameter
    def call_fn() raises:
        var s = min4(ptr, n)
        keep(s)

    b.iter[call_fn]()
    ptr.free()


@parameter
def bench_min8[dtype: DType, n: Int](mut b: Bencher) raises:
    var ptr = alloc[Scalar[dtype]](n)
    fill_buffer(ptr, n)

    @always_inline
    @parameter
    def call_fn() raises:
        var s = min8(ptr, n)
        keep(s)

    b.iter[call_fn]()
    ptr.free()


@parameter
def bench_std_min[dtype: DType, n: Int](mut b: Bencher) raises:
    var ptr = alloc[Scalar[dtype]](n)
    fill_buffer(ptr, n)

    @always_inline
    @parameter
    def call_fn() raises:
        var span = Span[Scalar[dtype], MutExternalOrigin](ptr=ptr, length=n)
        var s = std_min(span)
        keep(s)

    b.iter[call_fn]()
    ptr.free()


@parameter
def bench_max4[dtype: DType, n: Int](mut b: Bencher) raises:
    var ptr = alloc[Scalar[dtype]](n)
    fill_buffer(ptr, n)

    @always_inline
    @parameter
    def call_fn() raises:
        var s = max4(ptr, n)
        keep(s)

    b.iter[call_fn]()
    ptr.free()


@parameter
def bench_max8[dtype: DType, n: Int](mut b: Bencher) raises:
    var ptr = alloc[Scalar[dtype]](n)
    fill_buffer(ptr, n)

    @always_inline
    @parameter
    def call_fn() raises:
        var s = max8(ptr, n)
        keep(s)

    b.iter[call_fn]()
    ptr.free()


@parameter
def bench_std_max[dtype: DType, n: Int](mut b: Bencher) raises:
    var ptr = alloc[Scalar[dtype]](n)
    fill_buffer(ptr, n)

    @always_inline
    @parameter
    def call_fn() raises:
        var span = Span[Scalar[dtype], MutExternalOrigin](ptr=ptr, length=n)
        var s = std_max(span)
        keep(s)

    b.iter[call_fn]()
    ptr.free()


@parameter
def bench_prod4[dtype: DType, n: Int](mut b: Bencher) raises:
    var ptr = alloc[Scalar[dtype]](n)
    fill_buffer(ptr, n)

    @always_inline
    @parameter
    def call_fn() raises:
        var s = prod4(ptr, n)
        keep(s)

    b.iter[call_fn]()
    ptr.free()


@parameter
def bench_prod8[dtype: DType, n: Int](mut b: Bencher) raises:
    var ptr = alloc[Scalar[dtype]](n)
    fill_buffer(ptr, n)

    @always_inline
    @parameter
    def call_fn() raises:
        var s = prod8(ptr, n)
        keep(s)

    b.iter[call_fn]()
    ptr.free()


@parameter
def bench_std_prod[dtype: DType, n: Int](mut b: Bencher) raises:
    var ptr = alloc[Scalar[dtype]](n)
    fill_buffer(ptr, n)

    @always_inline
    @parameter
    def call_fn() raises:
        var span = Span[Scalar[dtype], MutExternalOrigin](ptr=ptr, length=n)
        var s = std_product(span)
        keep(s)

    b.iter[call_fn]()
    ptr.free()


# ===-----------------------------------------------------------------------===
# Throughput measures
# ===-----------------------------------------------------------------------===

def _bytes[dtype: DType, n: Int]() -> ThroughputMeasure:
    return ThroughputMeasure(
        BenchMetric.bytes, n * size_of[Scalar[dtype]]()
    )


def _flops[n: Int]() -> ThroughputMeasure:
    return ThroughputMeasure(BenchMetric.flops, n)


def _measures[dtype: DType, n: Int]() -> List[ThroughputMeasure]:
    return [_bytes[dtype, n](), _flops[n]()]


# ===-----------------------------------------------------------------------===
# Main — all 5 reductions × 3 implementations × 5 sizes per dtype
# ===-----------------------------------------------------------------------===

def main() raises:
    var m = Bench(BenchConfig(num_repetitions=2))

    # ---- f32 ----------------------------------------------------------
    # SUM
    m.bench_function[bench_sum4[DType.float32, 1024]](BenchId("sum4_f32_1k"), _measures[DType.float32, 1024]())
    m.bench_function[bench_sum8[DType.float32, 1024]](BenchId("sum8_f32_1k"), _measures[DType.float32, 1024]())
    m.bench_function[bench_std_sum[DType.float32, 1024]](BenchId("sumS_f32_1k"), _measures[DType.float32, 1024]())

    m.bench_function[bench_sum4[DType.float32, 65536]](BenchId("sum4_f32_64k"), _measures[DType.float32, 65536]())
    m.bench_function[bench_sum8[DType.float32, 65536]](BenchId("sum8_f32_64k"), _measures[DType.float32, 65536]())
    m.bench_function[bench_std_sum[DType.float32, 65536]](BenchId("sumS_f32_64k"), _measures[DType.float32, 65536]())

    m.bench_function[bench_sum4[DType.float32, 1048576]](BenchId("sum4_f32_1M"), _measures[DType.float32, 1048576]())
    m.bench_function[bench_sum8[DType.float32, 1048576]](BenchId("sum8_f32_1M"), _measures[DType.float32, 1048576]())
    m.bench_function[bench_std_sum[DType.float32, 1048576]](BenchId("sumS_f32_1M"), _measures[DType.float32, 1048576]())
    m.bench_function[bench_sum_par[DType.float32, 1048576]](BenchId("sumP_f32_1M"), _measures[DType.float32, 1048576]())

    # Multi-thread regime — DRAM-bound at single thread, worth parallelising.
    m.bench_function[bench_sum8[DType.float32, 16777216]](BenchId("sum8_f32_16M"), _measures[DType.float32, 16777216]())
    m.bench_function[bench_std_sum[DType.float32, 16777216]](BenchId("sumS_f32_16M"), _measures[DType.float32, 16777216]())
    m.bench_function[bench_sum_par[DType.float32, 16777216]](BenchId("sumP_f32_16M"), _measures[DType.float32, 16777216]())

    m.bench_function[bench_sum8[DType.float32, 134217728]](BenchId("sum8_f32_128M"), _measures[DType.float32, 134217728]())
    m.bench_function[bench_std_sum[DType.float32, 134217728]](BenchId("sumS_f32_128M"), _measures[DType.float32, 134217728]())
    m.bench_function[bench_sum_par[DType.float32, 134217728]](BenchId("sumP_f32_128M"), _measures[DType.float32, 134217728]())

    # MEAN — fewer points; same kernel as sum + scalar div
    m.bench_function[bench_mean8[DType.float32, 65536]](BenchId("mean8_f32_64k"), _measures[DType.float32, 65536]())
    m.bench_function[bench_std_mean[DType.float32, 65536]](BenchId("meanS_f32_64k"), _measures[DType.float32, 65536]())
    m.bench_function[bench_mean8[DType.float32, 1048576]](BenchId("mean8_f32_1M"), _measures[DType.float32, 1048576]())
    m.bench_function[bench_std_mean[DType.float32, 1048576]](BenchId("meanS_f32_1M"), _measures[DType.float32, 1048576]())

    # MIN
    m.bench_function[bench_min4[DType.float32, 1024]](BenchId("min4_f32_1k"), _measures[DType.float32, 1024]())
    m.bench_function[bench_min8[DType.float32, 1024]](BenchId("min8_f32_1k"), _measures[DType.float32, 1024]())
    m.bench_function[bench_std_min[DType.float32, 1024]](BenchId("minS_f32_1k"), _measures[DType.float32, 1024]())

    m.bench_function[bench_min4[DType.float32, 65536]](BenchId("min4_f32_64k"), _measures[DType.float32, 65536]())
    m.bench_function[bench_min8[DType.float32, 65536]](BenchId("min8_f32_64k"), _measures[DType.float32, 65536]())
    m.bench_function[bench_std_min[DType.float32, 65536]](BenchId("minS_f32_64k"), _measures[DType.float32, 65536]())

    m.bench_function[bench_min4[DType.float32, 1048576]](BenchId("min4_f32_1M"), _measures[DType.float32, 1048576]())
    m.bench_function[bench_min8[DType.float32, 1048576]](BenchId("min8_f32_1M"), _measures[DType.float32, 1048576]())
    m.bench_function[bench_std_min[DType.float32, 1048576]](BenchId("minS_f32_1M"), _measures[DType.float32, 1048576]())

    # MAX
    m.bench_function[bench_max4[DType.float32, 1024]](BenchId("max4_f32_1k"), _measures[DType.float32, 1024]())
    m.bench_function[bench_max8[DType.float32, 1024]](BenchId("max8_f32_1k"), _measures[DType.float32, 1024]())
    m.bench_function[bench_std_max[DType.float32, 1024]](BenchId("maxS_f32_1k"), _measures[DType.float32, 1024]())

    m.bench_function[bench_max4[DType.float32, 65536]](BenchId("max4_f32_64k"), _measures[DType.float32, 65536]())
    m.bench_function[bench_max8[DType.float32, 65536]](BenchId("max8_f32_64k"), _measures[DType.float32, 65536]())
    m.bench_function[bench_std_max[DType.float32, 65536]](BenchId("maxS_f32_64k"), _measures[DType.float32, 65536]())

    m.bench_function[bench_max4[DType.float32, 1048576]](BenchId("max4_f32_1M"), _measures[DType.float32, 1048576]())
    m.bench_function[bench_max8[DType.float32, 1048576]](BenchId("max8_f32_1M"), _measures[DType.float32, 1048576]())
    m.bench_function[bench_std_max[DType.float32, 1048576]](BenchId("maxS_f32_1M"), _measures[DType.float32, 1048576]())

    # PROD — narrower sweep (large N overflows for non-1.0 inputs anyway)
    m.bench_function[bench_prod4[DType.float32, 1024]](BenchId("prod4_f32_1k"), _measures[DType.float32, 1024]())
    m.bench_function[bench_prod8[DType.float32, 1024]](BenchId("prod8_f32_1k"), _measures[DType.float32, 1024]())
    m.bench_function[bench_std_prod[DType.float32, 1024]](BenchId("prodS_f32_1k"), _measures[DType.float32, 1024]())

    m.bench_function[bench_prod4[DType.float32, 65536]](BenchId("prod4_f32_64k"), _measures[DType.float32, 65536]())
    m.bench_function[bench_prod8[DType.float32, 65536]](BenchId("prod8_f32_64k"), _measures[DType.float32, 65536]())
    m.bench_function[bench_std_prod[DType.float32, 65536]](BenchId("prodS_f32_64k"), _measures[DType.float32, 65536]())

    # ---- f64 ----------------------------------------------------------
    m.bench_function[bench_sum4[DType.float64, 1024]](BenchId("sum4_f64_1k"), _measures[DType.float64, 1024]())
    m.bench_function[bench_sum8[DType.float64, 1024]](BenchId("sum8_f64_1k"), _measures[DType.float64, 1024]())
    m.bench_function[bench_std_sum[DType.float64, 1024]](BenchId("sumS_f64_1k"), _measures[DType.float64, 1024]())

    m.bench_function[bench_sum4[DType.float64, 1048576]](BenchId("sum4_f64_1M"), _measures[DType.float64, 1048576]())
    m.bench_function[bench_sum8[DType.float64, 1048576]](BenchId("sum8_f64_1M"), _measures[DType.float64, 1048576]())
    m.bench_function[bench_std_sum[DType.float64, 1048576]](BenchId("sumS_f64_1M"), _measures[DType.float64, 1048576]())
    m.bench_function[bench_sum_par[DType.float64, 1048576]](BenchId("sumP_f64_1M"), _measures[DType.float64, 1048576]())

    m.bench_function[bench_sum8[DType.float64, 16777216]](BenchId("sum8_f64_16M"), _measures[DType.float64, 16777216]())
    m.bench_function[bench_std_sum[DType.float64, 16777216]](BenchId("sumS_f64_16M"), _measures[DType.float64, 16777216]())
    m.bench_function[bench_sum_par[DType.float64, 16777216]](BenchId("sumP_f64_16M"), _measures[DType.float64, 16777216]())

    m.bench_function[bench_min4[DType.float64, 1024]](BenchId("min4_f64_1k"), _measures[DType.float64, 1024]())
    m.bench_function[bench_min8[DType.float64, 1024]](BenchId("min8_f64_1k"), _measures[DType.float64, 1024]())
    m.bench_function[bench_std_min[DType.float64, 1024]](BenchId("minS_f64_1k"), _measures[DType.float64, 1024]())

    m.bench_function[bench_max4[DType.float64, 1024]](BenchId("max4_f64_1k"), _measures[DType.float64, 1024]())
    m.bench_function[bench_max8[DType.float64, 1024]](BenchId("max8_f64_1k"), _measures[DType.float64, 1024]())
    m.bench_function[bench_std_max[DType.float64, 1024]](BenchId("maxS_f64_1k"), _measures[DType.float64, 1024]())

    m.bench_function[bench_prod4[DType.float64, 1024]](BenchId("prod4_f64_1k"), _measures[DType.float64, 1024]())
    m.bench_function[bench_prod8[DType.float64, 1024]](BenchId("prod8_f64_1k"), _measures[DType.float64, 1024]())
    m.bench_function[bench_std_prod[DType.float64, 1024]](BenchId("prodS_f64_1k"), _measures[DType.float64, 1024]())

    m.dump_report()
