"""Reduction kernels: typed-SIMD sum + strided walkers + dtype dispatchers.

Hosts:
  - `reduce_sum_typed[dtype]` — 4-parallel SIMD accumulator sum (the
    Apple-Silicon FADD-latency-hide trick). Float-only because integer
    accumulators need different overflow semantics.
  - `reduce_strided_typed[dtype]` — generic strided walker with both
    linearly-addressable fast path (transpose / swapaxes of c-contig)
    and a `LayoutIter` fallback for genuine non-linear views.
  - `maybe_reduce_strided_typed` — 14-way dtype dispatch into the
    strided typed walker (skips bool/complex; argmax/all/any have
    separate paths).
  - `maybe_reduce_contiguous` — c-contig fast path. Bool / int → i64 / u64
    accumulators; f16 → f64 round-trip; f32 / f64 → `reduce_sum_typed`.
  - `maybe_argmax_contiguous` — c-contig argmax over float arrays.

The 4-parallel-accumulator trick: single-accumulator FADD chains stall
on Apple Silicon because each add depends on the previous (3 cyc
latency, 1 cyc throughput → 1 add per 3 cycles). Four independent
accumulators saturate the issue rate (4 adds per 3 cycles ≈ 4× speedup
for memory-resident workloads). Cross-ref `simd-vectorisation.md §6`.
"""

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

from .predicates import is_contiguous_float_array
def reduce_sum_typed[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int
) raises -> Float64 where dtype.is_floating_point():
    # 4-parallel SIMD accumulator sum — breaks the FADD latency dependency.
    #
    # Single-accumulator loop:
    #   acc = acc + v[i]   ; next iteration depends on previous → ~latency-bound
    #   On Apple Silicon FADD = 3 cyc latency, 1 cyc throughput. Each cycle is a
    #   wasted issue slot because the next FADD must wait for the previous
    #   result. Effective throughput: 1 add per 3 cycles.
    #
    # 4-accumulator loop:
    #   a0,a1,a2,a3 are independent → pipeline can issue all 4 in flight,
    #   one per cycle. Effective throughput: 4 adds per 3 cycles ≈ 4× speedup
    #   for memory-resident workloads. Final `reduce_add` collapses 4
    #   accumulators down to a scalar via tree reduction.
    #
    # Promotes to Float64 accumulator regardless of input width to bound
    # the f32 sum's catastrophic cancellation risk (numpy.sum on f32
    # already promotes internally on linalg paths).
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
            var acc = Scalar[dtype](1)
            for i in range(n):
                acc *= ptr[base + i]
            return Float64(acc)
        if n == 0:
            raise Error("reduce_strided_typed: empty source")
        if op == ReduceOp.MIN.value:
            var acc = ptr[base]
            for i in range(1, n):
                var v = ptr[base + i]
                if v < acc:
                    acc = v
            return Float64(acc)
        if op == ReduceOp.MAX.value:
            var acc = ptr[base]
            for i in range(1, n):
                var v = ptr[base + i]
                if v > acc:
                    acc = v
            return Float64(acc)
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
def _reduce_strided_and_write[
    dt: DType
](src: Array, mut result: Array, op: Int) raises:
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
            acc = reduce_sum_typed[DType.float32](contiguous_ptr[DType.float32](src), src.size_value)
        elif src.dtype_code == ArrayDType.FLOAT64.value:
            acc = reduce_sum_typed[DType.float64](contiguous_ptr[DType.float64](src), src.size_value)
        else:
            return False
        if op == ReduceOp.MEAN.value:
            acc = acc / Float64(src.size_value)
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
