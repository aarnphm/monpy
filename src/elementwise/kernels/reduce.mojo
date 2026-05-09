"""Reduction kernels: typed-SIMD sum/min/max/prod + strided walkers + dispatch.

Hosts:
  - `reduce_sum_typed[dtype]` — 8-parallel SIMD accumulator sum.
  - `reduce_min_typed[dtype]` / `reduce_max_typed[dtype]` /
    `reduce_prod_typed[dtype]` — 8-parallel SIMD min/max/product.
    All four float-only because integer accumulators need different
    overflow / promotion semantics (handled by separate scalar paths).
  - `reduce_strided_typed[dtype]` — generic strided walker with both
    linearly-addressable fast path (transpose / swapaxes of c-contig)
    and a `LayoutIter` fallback for genuine non-linear views.
  - `maybe_reduce_strided_typed` — 14-way dtype dispatch into the
    strided typed walker (skips bool/complex; argmax/all/any have
    separate paths).
  - `maybe_reduce_contiguous` — c-contig fast path. Bool / int → i64 / u64
    accumulators; f16 → f64 round-trip; f32 / f64 → `reduce_*_typed`.
  - `maybe_argmax_contiguous` — c-contig argmax over float arrays.

The 8-parallel-accumulator trick is per worker, not per machine:
single-accumulator FADD/FMIN/FMAX/FMUL
chains stall because each op depends on the previous (≈3 cyc latency on
Apple Silicon). Modern cores dispatch 2 FP ops per cycle (M-series, AVX2
Intel/AMD), so the depth needed to saturate is `latency × IPC ≈ 6–8`.
Eight independent accumulators saturate that pipeline; the final
tree-reduction collapses them to a scalar. The 4-way version still works
on 1-IPC machines but leaves ~2× on the table on 2-IPC ones.
Multi-core fanout is a separate policy in `kernels/parallel.mojo`: a
64-core Ubuntu host may use many workers, and each worker still runs this
same 8-accumulator SIMD loop on its chunk.

Validated on M-series in `benches/bench_reduce.mojo`: 4-way → 86 GB/s
f32 1k; 8-way → 133 GB/s f32 1k; std.algorithm.sum → 130 GB/s. The
8-way version matches `std.algorithm.reduction` at parity. Cross-ref
`simd-vectorisation.md §6`.
"""

from std.algorithm import sync_parallelize
from std.math import ceildiv, min as _simd_min, max as _simd_max
from std.memory.unsafe_pointer import alloc
from std.sys import simd_width_of

from array import (
    Array,
    as_layout,
    contiguous_as_f64,
    contiguous_ptr,
    is_c_contiguous,
    is_linearly_addressable,
    item_size,
    set_logical_from_f64,
    set_logical_from_i64,
)
from cute.iter import LayoutIter
from domain import ArrayDType, BackendKind, ReduceOp

from elementwise.kernels.parallel import (
    REDUCE_GRAIN,
    worker_count_for_bytes,
    worker_count_for_rows,
)
from elementwise.predicates import is_contiguous_float_array


comptime REDUCE_SIMD_ACCUMULATORS = 8
# Keep this in sync with the explicit a0..a7 / v0..v7 accumulator lists
# below. This is instruction-level parallelism inside one worker, not the
# number of worker threads.


def reduce_sum_typed[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int
) raises -> Float64 where dtype.is_floating_point():
    # 8-parallel SIMD accumulator sum.
    #
    # Single-accumulator loop:
    #   acc = acc + v[i]   ; next iteration depends on previous → latency-bound.
    #   FADD is ~3 cyc latency on M-series and modern x86. Naively each cycle
    #   is a wasted issue slot because the next FADD waits on the previous.
    #
    # 8-accumulator loop:
    #   a0..a7 are independent. To saturate a 2-IPC FADD pipeline at 3 cyc
    #   latency you need ~6 in flight; 8 leaves a comfortable margin and
    #   keeps loop overhead amortised across 8 SIMD vectors per iter.
    #   Validated against `std.algorithm.reduction.sum`: 8-way matches it
    #   at parity (~133 GB/s f32 1k on M-series). The 4-way version was
    #   86 GB/s — same kernel, just fewer accumulators.
    #
    # Promotes to Float64 accumulator regardless of input width to bound
    # the f32 sum's catastrophic cancellation risk (numpy.sum on f32
    # already promotes internally on linalg paths).
    comptime width = simd_width_of[dtype]()
    comptime block = width * REDUCE_SIMD_ACCUMULATORS
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


def reduce_sum_par_typed[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int, nworkers: Int
) raises -> Float64 where dtype.is_floating_point():
    # Multi-thread sum: chunk across the policy-chosen worker count.
    # Each worker calls `reduce_sum_typed` (the 8-accumulator SIMD kernel)
    # on its slice into a Float64 partial. Master thread combines partials
    # at the end (small, scalar — irrelevant cost).
    #
    # Validated in `benches/bench_reduce.mojo`:
    #   - 16M f32: 61 → 117 GB/s (1.9x, hits M3 Pro DRAM ceiling)
    #   - 1M f64 cache-resident: 82 → 374 GB/s (4.6x, L2 ceiling)
    #
    # The caller is expected to pass `worker_count_for_bytes(...)` so this
    # does not fan a 1MB tensor across every core on many-core systems.
    # See `maybe_reduce_contiguous` for the production gate.
    if nworkers <= 1:
        return reduce_sum_typed[dtype](src_ptr, size)

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
        partials[i] = reduce_sum_typed[dtype](src_ptr + start, end - start)

    sync_parallelize[worker](nworkers)

    var total = Float64(0)
    for i in range(nworkers):
        total += partials[i]
    partials.free()
    return total


def reduce_min_typed[
    dtype: DType
](src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    # 8-parallel SIMD min — same multi-accumulator structure as sum, but
    # the dependency-breaking primitive is FMIN instead of FADD. FMIN
    # latency on M-series is ~3 cyc just like FADD; the 8-way unroll
    # saturates the 2-IPC dispatch the same way.
    #
    # Seeded from the first 8 SIMD vectors (rather than +Inf) to avoid
    # one extra round of FMIN against a constant; correctness holds
    # because we require size >= block before entering the SIMD loop.
    if size == 0:
        raise Error("reduce_min_typed: empty source")
    comptime width = simd_width_of[dtype]()
    comptime block = width * REDUCE_SIMD_ACCUMULATORS
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


def reduce_max_typed[
    dtype: DType
](src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    # 8-parallel SIMD max — symmetric to reduce_min_typed.
    if size == 0:
        raise Error("reduce_max_typed: empty source")
    comptime width = simd_width_of[dtype]()
    comptime block = width * REDUCE_SIMD_ACCUMULATORS
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


def reduce_prod_typed[
    dtype: DType
](src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    # 8-parallel SIMD product — FMUL latency on M-series is ~4 cyc at
    # 2 IPC, so the 8-way structure remains correct (8 ≥ 4 × 2 = 8).
    # Seed accumulators with 1.0 so the partial product semantics hold
    # for any size including ones below the SIMD-block threshold.
    comptime width = simd_width_of[dtype]()
    comptime block = width * REDUCE_SIMD_ACCUMULATORS
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
# Multi-thread variants: chunk across P-cores, combine partials at end.
# Same per-chunk SIMD kernel as the serial path; just stitched.
# ===-----------------------------------------------------------------------===


def reduce_min_par_typed[
    dtype: DType
](src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int, nworkers: Int) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    # Worker scheme mirrors reduce_sum_par_typed. Each chunk runs the
    # serial reduce_min_typed (8-way SIMD) on its slice; partials live
    # in a heap Float64 array (so we don't need a Scalar[dtype] heap
    # allocator); master combines via scalar < walk.
    if size == 0:
        raise Error("reduce_min_par_typed: empty source")
    if nworkers <= 1:
        return reduce_min_typed[dtype](src_ptr, size)

    var partials = alloc[Float64](nworkers)
    var chunk = ceildiv(size, nworkers)

    @parameter
    def worker(i: Int) raises:
        var start = i * chunk
        var end = start + chunk
        if end > size:
            end = size
        if start >= size:
            # Sentinel: master skips entries past live_workers.
            partials[i] = 0.0
            return
        partials[i] = Float64(reduce_min_typed[dtype](src_ptr + start, end - start))

    sync_parallelize[worker](nworkers)

    # Live worker count is ceildiv(size, chunk); partials past this are
    # the sentinel zeros and must not participate in min/max.
    var live_workers = ceildiv(size, chunk)
    var acc = partials[0]
    for i in range(1, live_workers):
        var v = partials[i]
        if v < acc:
            acc = v
    partials.free()
    return Scalar[dtype](acc)


def reduce_max_par_typed[
    dtype: DType
](src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int, nworkers: Int) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    # Symmetric to reduce_min_par_typed.
    if size == 0:
        raise Error("reduce_max_par_typed: empty source")
    if nworkers <= 1:
        return reduce_max_typed[dtype](src_ptr, size)

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
        partials[i] = Float64(reduce_max_typed[dtype](src_ptr + start, end - start))

    sync_parallelize[worker](nworkers)

    var live_workers = ceildiv(size, chunk)
    var acc = partials[0]
    for i in range(1, live_workers):
        var v = partials[i]
        if v > acc:
            acc = v
    partials.free()
    return Scalar[dtype](acc)


def reduce_prod_par_typed[
    dtype: DType
](src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int, nworkers: Int) raises -> Scalar[
    dtype
] where dtype.is_floating_point():
    # Worker partials accumulate in Float64 to bound the f32 catastrophic-
    # cancellation surface, mirroring reduce_sum_par_typed. Empty chunks
    # contribute the identity element (1.0) — safe to multiply across.
    if nworkers <= 1:
        return reduce_prod_typed[dtype](src_ptr, size)

    var partials = alloc[Float64](nworkers)
    var chunk = ceildiv(size, nworkers)

    @parameter
    def worker(i: Int) raises:
        var start = i * chunk
        var end = start + chunk
        if end > size:
            end = size
        if start >= size:
            partials[i] = 1.0
            return
        partials[i] = Float64(reduce_prod_typed[dtype](src_ptr + start, end - start))

    sync_parallelize[worker](nworkers)

    var total = Float64(1)
    for i in range(nworkers):
        total *= partials[i]
    partials.free()
    return Scalar[dtype](total)


def reduce_strided_typed[dtype: DType](src: Array, op: Int) raises -> Float64:
    # Strided reduction via typed-pointer scalar walk. Two paths:
    #   1. linearly-addressable (transpose / swapaxes of c-contig):
    #      flat scan `data[offset .. offset+size)`. Cache-friendly even
    #      when the *logical* iteration order has a non-unit innermost
    #      stride. This is the .T.sum() fast path — important because
    #      reductions are commutative so iteration order is free.
    #   2. genuinely non-linear (sliced views, broadcasts): `LayoutIter`
    #      cursor walk over the logical shape.
    # Both accumulate in `Scalar[dtype]` natively and return Float64
    # for the f64-set boundary.
    var ptr = src.data.bitcast[Scalar[dtype]]()
    if is_linearly_addressable(src):
        var base = src.offset_elems
        var n = src.size_value
        if op == ReduceOp.SUM.value or op == ReduceOp.MEAN.value:
            var out: Float64
            comptime if dtype.is_floating_point():
                # Reuse the SIMD-vectorised contig reduce kernel (4-way
                # parallel accumulators). Same primitive as the c-contig
                # path; just rebased onto `offset_elems`.
                out = reduce_sum_typed[dtype](ptr + base, n)
            else:
                var acc = Scalar[dtype](0)
                for i in range(n):
                    acc += ptr[base + i]
                out = Float64(acc)
            if op == ReduceOp.MEAN.value:
                out = out / Float64(n)
            return out
        if op == ReduceOp.PROD.value:
            var acc: Float64
            comptime if dtype.is_floating_point():
                # SIMD 8-way product — same latency-hide trick as sum.
                acc = Float64(reduce_prod_typed[dtype](ptr + base, n))
            else:
                var i_acc = Scalar[dtype](1)
                for i in range(n):
                    i_acc *= ptr[base + i]
                acc = Float64(i_acc)
            return acc
        if n == 0:
            raise Error("reduce_strided_typed: empty source")
        if op == ReduceOp.MIN.value:
            var acc: Float64
            comptime if dtype.is_floating_point():
                acc = Float64(reduce_min_typed[dtype](ptr + base, n))
            else:
                var i_acc = ptr[base]
                for i in range(1, n):
                    var v = ptr[base + i]
                    if v < i_acc:
                        i_acc = v
                acc = Float64(i_acc)
            return acc
        if op == ReduceOp.MAX.value:
            var acc: Float64
            comptime if dtype.is_floating_point():
                acc = Float64(reduce_max_typed[dtype](ptr + base, n))
            else:
                var i_acc = ptr[base]
                for i in range(1, n):
                    var v = ptr[base + i]
                    if v > i_acc:
                        i_acc = v
                acc = Float64(i_acc)
            return acc
        raise Error("reduce_strided_typed: unsupported op")
    var layout = as_layout(src)
    var item = item_size(src.dtype_code)
    var iter = LayoutIter(layout, item, src.offset_elems * item)
    if op == ReduceOp.SUM.value or op == ReduceOp.MEAN.value:
        var acc = Scalar[dtype](0)
        while iter.has_next():
            acc += ptr[iter.element_index()]
            iter.step()
        var out = Float64(acc)
        if op == ReduceOp.MEAN.value:
            out = out / Float64(src.size_value)
        return out
    if op == ReduceOp.PROD.value:
        var acc = Scalar[dtype](1)
        while iter.has_next():
            acc *= ptr[iter.element_index()]
            iter.step()
        return Float64(acc)
    # MIN / MAX: read the first element as the seed.
    if not iter.has_next():
        raise Error("reduce_strided_typed: empty source")
    var acc = ptr[iter.element_index()]
    iter.step()
    if op == ReduceOp.MIN.value:
        while iter.has_next():
            var v = ptr[iter.element_index()]
            if v < acc:
                acc = v
            iter.step()
        return Float64(acc)
    if op == ReduceOp.MAX.value:
        while iter.has_next():
            var v = ptr[iter.element_index()]
            if v > acc:
                acc = v
            iter.step()
        return Float64(acc)
    raise Error("reduce_strided_typed: unsupported op")


@always_inline
def _reduce_strided_and_write[dt: DType](src: Array, mut result: Array, op: Int) raises:
    # Run the typed strided reducer and write the bimodal result:
    # float dtypes → set_logical_from_f64 (acc is already Float64-cast);
    # integer dtypes → set_logical_from_i64 unless op is MEAN, in which
    # case the fractional answer goes through the f64 path.
    var acc = reduce_strided_typed[dt](src, op)
    comptime if dt.is_floating_point():
        set_logical_from_f64(result, 0, acc)
    else:
        if op == ReduceOp.MEAN.value:
            set_logical_from_f64(result, 0, acc)
        else:
            set_logical_from_i64(result, 0, Int64(acc))


def maybe_reduce_strided_typed(src: Array, mut result: Array, op: Int) raises -> Bool:
    # Dispatch to typed strided reduction by source dtype. Returns True
    # iff the typed kernel handled it. Skips:
    # - contig source (handled upstream by `maybe_reduce_contiguous`),
    # - argmax / argmin (need index tracking — separate path),
    # - all / any (boolean short-circuit — separate path),
    # - bool / complex (no native typed accumulator),
    # - mismatched accumulator semantics (e.g. int sum→i64 promotion).
    if (
        op != ReduceOp.SUM.value
        and op != ReduceOp.MEAN.value
        and op != ReduceOp.PROD.value
        and op != ReduceOp.MIN.value
        and op != ReduceOp.MAX.value
    ):
        return False
    if src.dtype_code == ArrayDType.FLOAT64.value:
        _reduce_strided_and_write[DType.float64](src, result, op)
    elif src.dtype_code == ArrayDType.FLOAT32.value:
        _reduce_strided_and_write[DType.float32](src, result, op)
    elif src.dtype_code == ArrayDType.INT64.value:
        _reduce_strided_and_write[DType.int64](src, result, op)
    elif src.dtype_code == ArrayDType.INT32.value:
        _reduce_strided_and_write[DType.int32](src, result, op)
    elif src.dtype_code == ArrayDType.INT16.value:
        _reduce_strided_and_write[DType.int16](src, result, op)
    elif src.dtype_code == ArrayDType.INT8.value:
        _reduce_strided_and_write[DType.int8](src, result, op)
    elif src.dtype_code == ArrayDType.UINT64.value:
        _reduce_strided_and_write[DType.uint64](src, result, op)
    elif src.dtype_code == ArrayDType.UINT32.value:
        _reduce_strided_and_write[DType.uint32](src, result, op)
    elif src.dtype_code == ArrayDType.UINT16.value:
        _reduce_strided_and_write[DType.uint16](src, result, op)
    elif src.dtype_code == ArrayDType.UINT8.value:
        _reduce_strided_and_write[DType.uint8](src, result, op)
    else:
        return False
    return True


@always_inline
def _reduce_axis_last_contiguous_typed[
    dt: DType
](src: Array, mut result: Array, op: Int) raises where dt.is_floating_point():
    # Per-row reduction over the last (contiguous) axis. Each row reads a
    # disjoint `[base, base+cols)` slice of `src` and writes one scalar to
    # `result` at logical index `row`. set_logical_from_f64 mutates only
    # `result.data[physical_offset(result, row)]` (verified in
    # array/dispatch.mojo:388 and array/accessors.mojo:355), so two
    # workers writing different rows never alias.
    var cols = src.shape[len(src.shape) - 1]
    var rows = src.size_value // cols
    var ptr = contiguous_ptr[dt](src)

    @parameter
    def process_row_range(row_start: Int, row_end: Int) raises:
        for row in range(row_start, row_end):
            var base = row * cols
            if op == ReduceOp.SUM.value or op == ReduceOp.MEAN.value:
                var acc = reduce_sum_typed[dt](ptr + base, cols)
                if op == ReduceOp.MEAN.value:
                    acc = acc / Float64(cols)
                set_logical_from_f64(result, row, acc)
            elif op == ReduceOp.PROD.value:
                set_logical_from_f64(result, row, Float64(reduce_prod_typed[dt](ptr + base, cols)))
            elif op == ReduceOp.MIN.value:
                set_logical_from_f64(result, row, Float64(reduce_min_typed[dt](ptr + base, cols)))
            elif op == ReduceOp.MAX.value:
                set_logical_from_f64(result, row, Float64(reduce_max_typed[dt](ptr + base, cols)))

    var nw = worker_count_for_rows(rows)
    if nw > 1:
        var chunk = ceildiv(rows, nw)

        @parameter
        def chunk_worker(i: Int) raises:
            var start = i * chunk
            var end = start + chunk
            if end > rows:
                end = rows
            if start >= rows:
                return
            process_row_range(start, end)

        sync_parallelize[chunk_worker](nw)
    else:
        process_row_range(0, rows)
    result.backend_code = BackendKind.FUSED.value


def maybe_reduce_axis_last_contiguous(src: Array, mut result: Array, op: Int) raises -> Bool:
    """Fast path for row-wise reductions of C-contiguous arrays.

    Attention-style workloads hit `axis=-1, keepdims=True` repeatedly on
    small float32 matrices. The generic axis reducer allocates coordinate
    lists and round-trips each element through f64 dispatch; for a 32x32
    row reduction that bookkeeping costs more than the arithmetic. This
    path treats the last axis as contiguous rows and writes one scalar per
    row, with the existing SIMD sum primitive for sum/mean.
    """
    if (
        op != ReduceOp.SUM.value
        and op != ReduceOp.MEAN.value
        and op != ReduceOp.PROD.value
        and op != ReduceOp.MIN.value
        and op != ReduceOp.MAX.value
    ):
        return False
    if len(src.shape) == 0:
        return False
    if src.shape[len(src.shape) - 1] == 0:
        return False
    if not is_c_contiguous(src):
        return False
    if src.dtype_code == ArrayDType.FLOAT32.value:
        _reduce_axis_last_contiguous_typed[DType.float32](src, result, op)
        return True
    if src.dtype_code == ArrayDType.FLOAT64.value:
        _reduce_axis_last_contiguous_typed[DType.float64](src, result, op)
        return True
    return False


def maybe_reduce_contiguous(src: Array, mut result: Array, op: Int) raises -> Bool:
    if op == ReduceOp.SUM.value and is_c_contiguous(src):
        if src.dtype_code == ArrayDType.BOOL.value:
            var acc = Int64(0)
            var ptr = contiguous_ptr[DType.uint8](src)
            for i in range(src.size_value):
                acc += Int64(Int(ptr[i]))
            set_logical_from_i64(result, 0, acc)
            result.backend_code = BackendKind.FUSED.value
            return True
        if src.dtype_code == ArrayDType.INT64.value:
            var acc = Int64(0)
            var ptr = contiguous_ptr[DType.int64](src)
            for i in range(src.size_value):
                acc += ptr[i]
            set_logical_from_i64(result, 0, acc)
            result.backend_code = BackendKind.FUSED.value
            return True
        if src.dtype_code == ArrayDType.INT32.value:
            var acc = Int64(0)
            var ptr = contiguous_ptr[DType.int32](src)
            for i in range(src.size_value):
                acc += Int64(Int(ptr[i]))
            set_logical_from_i64(result, 0, acc)
            result.backend_code = BackendKind.FUSED.value
            return True
        if src.dtype_code == ArrayDType.INT16.value:
            var acc = Int64(0)
            var ptr = contiguous_ptr[DType.int16](src)
            for i in range(src.size_value):
                acc += Int64(Int(ptr[i]))
            set_logical_from_i64(result, 0, acc)
            result.backend_code = BackendKind.FUSED.value
            return True
        if src.dtype_code == ArrayDType.INT8.value:
            var acc = Int64(0)
            var ptr = contiguous_ptr[DType.int8](src)
            for i in range(src.size_value):
                acc += Int64(Int(ptr[i]))
            set_logical_from_i64(result, 0, acc)
            result.backend_code = BackendKind.FUSED.value
            return True
        if src.dtype_code == ArrayDType.UINT64.value:
            var acc = UInt64(0)
            var ptr = contiguous_ptr[DType.uint64](src)
            for i in range(src.size_value):
                acc += ptr[i]
            result.data.bitcast[UInt64]()[result.offset_elems] = acc
            result.backend_code = BackendKind.FUSED.value
            return True
        if src.dtype_code == ArrayDType.UINT32.value:
            var acc = UInt64(0)
            var ptr = contiguous_ptr[DType.uint32](src)
            for i in range(src.size_value):
                acc += UInt64(Int(ptr[i]))
            result.data.bitcast[UInt64]()[result.offset_elems] = acc
            result.backend_code = BackendKind.FUSED.value
            return True
        if src.dtype_code == ArrayDType.UINT16.value:
            var acc = UInt64(0)
            var ptr = contiguous_ptr[DType.uint16](src)
            for i in range(src.size_value):
                acc += UInt64(Int(ptr[i]))
            result.data.bitcast[UInt64]()[result.offset_elems] = acc
            result.backend_code = BackendKind.FUSED.value
            return True
        if src.dtype_code == ArrayDType.UINT8.value:
            var acc = UInt64(0)
            var ptr = contiguous_ptr[DType.uint8](src)
            for i in range(src.size_value):
                acc += UInt64(Int(ptr[i]))
            result.data.bitcast[UInt64]()[result.offset_elems] = acc
            result.backend_code = BackendKind.FUSED.value
            return True
    if (
        (op == ReduceOp.SUM.value or op == ReduceOp.MEAN.value)
        and src.dtype_code == ArrayDType.FLOAT16.value
        and is_c_contiguous(src)
    ):
        var acc = 0.0
        var ptr = contiguous_ptr[DType.float16](src)
        for i in range(src.size_value):
            acc += Float64(ptr[i])
        if op == ReduceOp.MEAN.value:
            acc = acc / Float64(src.size_value)
        set_logical_from_f64(result, 0, acc)
        result.backend_code = BackendKind.FUSED.value
        return True
    if not is_contiguous_float_array(src):
        return False
    if op == ReduceOp.SUM.value or op == ReduceOp.MEAN.value:
        var acc: Float64
        if src.dtype_code == ArrayDType.FLOAT32.value:
            var ptr = contiguous_ptr[DType.float32](src)
            var nworkers = worker_count_for_bytes(src.size_value, src.size_value * 4, REDUCE_GRAIN)
            if nworkers > 1:
                acc = reduce_sum_par_typed[DType.float32](ptr, src.size_value, nworkers)
            else:
                acc = reduce_sum_typed[DType.float32](ptr, src.size_value)
        elif src.dtype_code == ArrayDType.FLOAT64.value:
            var ptr = contiguous_ptr[DType.float64](src)
            var nworkers = worker_count_for_bytes(src.size_value, src.size_value * 8, REDUCE_GRAIN)
            if nworkers > 1:
                acc = reduce_sum_par_typed[DType.float64](ptr, src.size_value, nworkers)
            else:
                acc = reduce_sum_typed[DType.float64](ptr, src.size_value)
        else:
            return False
        if op == ReduceOp.MEAN.value:
            acc = acc / Float64(src.size_value)
        set_logical_from_f64(result, 0, acc)
        return True
    if op == ReduceOp.PROD.value:
        var acc: Float64
        if src.dtype_code == ArrayDType.FLOAT32.value:
            var ptr = contiguous_ptr[DType.float32](src)
            var nworkers = worker_count_for_bytes(src.size_value, src.size_value * 4, REDUCE_GRAIN)
            if nworkers > 1:
                acc = Float64(reduce_prod_par_typed[DType.float32](ptr, src.size_value, nworkers))
            else:
                acc = Float64(reduce_prod_typed[DType.float32](ptr, src.size_value))
        elif src.dtype_code == ArrayDType.FLOAT64.value:
            var ptr = contiguous_ptr[DType.float64](src)
            var nworkers = worker_count_for_bytes(src.size_value, src.size_value * 8, REDUCE_GRAIN)
            if nworkers > 1:
                acc = Float64(reduce_prod_par_typed[DType.float64](ptr, src.size_value, nworkers))
            else:
                acc = Float64(reduce_prod_typed[DType.float64](ptr, src.size_value))
        else:
            return False
        set_logical_from_f64(result, 0, acc)
        return True
    if op == ReduceOp.MIN.value or op == ReduceOp.MAX.value:
        if src.size_value == 0:
            return False
        var acc: Float64
        if src.dtype_code == ArrayDType.FLOAT32.value:
            var ptr = contiguous_ptr[DType.float32](src)
            var nworkers = worker_count_for_bytes(src.size_value, src.size_value * 4, REDUCE_GRAIN)
            if op == ReduceOp.MIN.value:
                if nworkers > 1:
                    acc = Float64(reduce_min_par_typed[DType.float32](ptr, src.size_value, nworkers))
                else:
                    acc = Float64(reduce_min_typed[DType.float32](ptr, src.size_value))
            else:
                if nworkers > 1:
                    acc = Float64(reduce_max_par_typed[DType.float32](ptr, src.size_value, nworkers))
                else:
                    acc = Float64(reduce_max_typed[DType.float32](ptr, src.size_value))
        elif src.dtype_code == ArrayDType.FLOAT64.value:
            var ptr = contiguous_ptr[DType.float64](src)
            var nworkers = worker_count_for_bytes(src.size_value, src.size_value * 8, REDUCE_GRAIN)
            if op == ReduceOp.MIN.value:
                if nworkers > 1:
                    acc = Float64(reduce_min_par_typed[DType.float64](ptr, src.size_value, nworkers))
                else:
                    acc = Float64(reduce_min_typed[DType.float64](ptr, src.size_value))
            else:
                if nworkers > 1:
                    acc = Float64(reduce_max_par_typed[DType.float64](ptr, src.size_value, nworkers))
                else:
                    acc = Float64(reduce_max_typed[DType.float64](ptr, src.size_value))
        else:
            return False
        set_logical_from_f64(result, 0, acc)
        return True
    var acc = contiguous_as_f64(src, 0)
    if op == ReduceOp.SUM.value or op == ReduceOp.MEAN.value:
        acc = 0.0
        for i in range(src.size_value):
            acc += contiguous_as_f64(src, i)
        if op == ReduceOp.MEAN.value:
            acc = acc / Float64(src.size_value)
    elif op == ReduceOp.MIN.value:
        for i in range(1, src.size_value):
            var value = contiguous_as_f64(src, i)
            if value < acc:
                acc = value
    elif op == ReduceOp.MAX.value:
        for i in range(1, src.size_value):
            var value = contiguous_as_f64(src, i)
            if value > acc:
                acc = value
    else:
        return False
    set_logical_from_f64(result, 0, acc)
    return True


def maybe_argmax_contiguous(src: Array, mut result: Array) raises -> Bool:
    if not is_contiguous_float_array(src):
        return False
    var best_index = 0
    var best_value = contiguous_as_f64(src, 0)
    for i in range(1, src.size_value):
        var value = contiguous_as_f64(src, i)
        if value > best_value:
            best_value = value
            best_index = i
    set_logical_from_i64(result, 0, Int64(best_index))
    return True
