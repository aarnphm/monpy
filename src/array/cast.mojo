"""Cast + copy: dtype conversions and c-contig copies.

Hosts:
  - Pairwise cast dispatchers — `_dispatch_dst_real_cast`,
    `dispatch_real_pair_cast` (11×11 = 121 monomorphised real→real
    casts), `dispatch_real_typed_contig_pair` (single-DType dispatch
    used by bool fast paths).
  - Bool ↔ real bridges (`_bool_src_to_real`, `_real_to_bool_dst`) —
    bypass `Scalar[DType.bool]` and read/write the raw byte directly.
  - `_copy_rank2_strided_*` family — 8×8 tile-based rank-2 strided
    copy for typed real, bool, complex64, complex128. Picks an inner
    walk by source row stride to keep at most one of load/store on
    the strided side.
  - `_maybe_copy_rank2_strided` — 14-way dispatcher into the typed copy.
  - `copy_c_contiguous` — produce a fresh c-contig copy of `src`. Three
    paths: `memcpy` for already-c-contig, the rank-2 strided tile, or a
    per-element f64 round-trip.
  - `_maybe_cast_contiguous_core_dtypes` — c-contig fast path for casts.
  - `cast_copy_array` — the public entry: dispatches to the contig fast
    path, otherwise per-element casts via the unified setters.
"""

from std.collections import List
from std.memory import memcpy
from std.sys import simd_width_of

from domain import ArrayDType, dtype_byte_offset, dtype_is_packed_subbyte, dtype_storage_byte_len

from .accessors import (
    Array,
    contiguous_ptr,
    clone_int_list,
    get_physical,
    get_physical_bool,
    get_physical_c128_imag,
    get_physical_c128_real,
    get_physical_c64_imag,
    get_physical_c64_real,
    is_c_contiguous,
    physical_offset,
    set_physical_c128,
    set_physical_c64,
)
from .dispatch import (
    get_physical_as_f64,
    set_logical_from_f64,
    set_logical_from_i64,
)
from .factory import make_empty_array


comptime PairwiseCastKernel = def[src_dt: DType, dst_dt: DType](Array, mut Array) thin raises -> None
"""Reads each element of `src` as Scalar[src_dt], writes to `result` as Scalar[dst_dt]."""


comptime SingleDtypeContigKernel = def[dt: DType](Array, mut Array) thin raises -> None
"""Single-dtype contiguous src→result kernel. Used by bool fast paths where
one side is bool (not parametric) and the other rides DType."""


def _dispatch_dst_real_cast[
    src_dt: DType,
    op: PairwiseCastKernel,
](dst_code: Int, src: Array, mut result: Array) raises -> Bool:
    """Inner dispatcher: with `src_dt` already fixed, fans out over the 11 real dst dtypes."""
    if dst_code == ArrayDType.FLOAT64.value:
        op[src_dt, DType.float64](src, result)
        return True
    if dst_code == ArrayDType.FLOAT32.value:
        op[src_dt, DType.float32](src, result)
        return True
    if dst_code == ArrayDType.FLOAT16.value:
        op[src_dt, DType.float16](src, result)
        return True
    if dst_code == ArrayDType.INT64.value:
        op[src_dt, DType.int64](src, result)
        return True
    if dst_code == ArrayDType.INT32.value:
        op[src_dt, DType.int32](src, result)
        return True
    if dst_code == ArrayDType.INT16.value:
        op[src_dt, DType.int16](src, result)
        return True
    if dst_code == ArrayDType.INT8.value:
        op[src_dt, DType.int8](src, result)
        return True
    if dst_code == ArrayDType.UINT64.value:
        op[src_dt, DType.uint64](src, result)
        return True
    if dst_code == ArrayDType.UINT32.value:
        op[src_dt, DType.uint32](src, result)
        return True
    if dst_code == ArrayDType.UINT16.value:
        op[src_dt, DType.uint16](src, result)
        return True
    if dst_code == ArrayDType.UINT8.value:
        op[src_dt, DType.uint8](src, result)
        return True
    return False


def dispatch_real_pair_cast[
    op: PairwiseCastKernel,
](src_code: Int, dst_code: Int, src: Array, mut result: Array) raises -> Bool:
    """Real-real pairwise dispatch. 11×11 = 121 monomorphized cast kernels emitted
    by Mojo, fed by 22 source-level branches (11 outer + 11 inner).

    Caller invariant: `src` and `result` are both c-contiguous, and neither dtype
    is BOOL or COMPLEX64/128 (those don't ride `Scalar[dt].cast[dst_dt]()` cleanly)."""
    if src_code == ArrayDType.FLOAT64.value:
        return _dispatch_dst_real_cast[DType.float64, op](dst_code, src, result)
    if src_code == ArrayDType.FLOAT32.value:
        return _dispatch_dst_real_cast[DType.float32, op](dst_code, src, result)
    if src_code == ArrayDType.FLOAT16.value:
        return _dispatch_dst_real_cast[DType.float16, op](dst_code, src, result)
    if src_code == ArrayDType.INT64.value:
        return _dispatch_dst_real_cast[DType.int64, op](dst_code, src, result)
    if src_code == ArrayDType.INT32.value:
        return _dispatch_dst_real_cast[DType.int32, op](dst_code, src, result)
    if src_code == ArrayDType.INT16.value:
        return _dispatch_dst_real_cast[DType.int16, op](dst_code, src, result)
    if src_code == ArrayDType.INT8.value:
        return _dispatch_dst_real_cast[DType.int8, op](dst_code, src, result)
    if src_code == ArrayDType.UINT64.value:
        return _dispatch_dst_real_cast[DType.uint64, op](dst_code, src, result)
    if src_code == ArrayDType.UINT32.value:
        return _dispatch_dst_real_cast[DType.uint32, op](dst_code, src, result)
    if src_code == ArrayDType.UINT16.value:
        return _dispatch_dst_real_cast[DType.uint16, op](dst_code, src, result)
    if src_code == ArrayDType.UINT8.value:
        return _dispatch_dst_real_cast[DType.uint8, op](dst_code, src, result)
    return False


def dispatch_real_typed_contig_pair[
    op: SingleDtypeContigKernel,
](dtype_code: Int, src: Array, mut result: Array) raises -> Bool:
    """Single-DType dispatch over the 11 real dtypes. Used by bool fast paths:
    bool→real (dispatching on dst_dt) and real→bool (dispatching on src_dt)."""
    if dtype_code == ArrayDType.FLOAT64.value:
        op[DType.float64](src, result)
        return True
    if dtype_code == ArrayDType.FLOAT32.value:
        op[DType.float32](src, result)
        return True
    if dtype_code == ArrayDType.FLOAT16.value:
        op[DType.float16](src, result)
        return True
    if dtype_code == ArrayDType.INT64.value:
        op[DType.int64](src, result)
        return True
    if dtype_code == ArrayDType.INT32.value:
        op[DType.int32](src, result)
        return True
    if dtype_code == ArrayDType.INT16.value:
        op[DType.int16](src, result)
        return True
    if dtype_code == ArrayDType.INT8.value:
        op[DType.int8](src, result)
        return True
    if dtype_code == ArrayDType.UINT64.value:
        op[DType.uint64](src, result)
        return True
    if dtype_code == ArrayDType.UINT32.value:
        op[DType.uint32](src, result)
        return True
    if dtype_code == ArrayDType.UINT16.value:
        op[DType.uint16](src, result)
        return True
    if dtype_code == ArrayDType.UINT8.value:
        op[DType.uint8](src, result)
        return True
    return False


def _contig_pair_cast[src_dt: DType, dst_dt: DType](src: Array, mut result: Array) raises:
    var src_ptr = contiguous_ptr[src_dt](src)
    var dst_ptr = contiguous_ptr[dst_dt](result)
    for i in range(src.size_value):
        dst_ptr[i] = src_ptr[i].cast[dst_dt]()


def _bool_src_to_real[dst_dt: DType](src: Array, mut result: Array) raises:
    """Bool source storage is raw UInt8 with nonzero-is-true semantics — bypass
    `Scalar[DType.bool]` and read the byte directly."""
    var src_ptr = src.data + src.offset_elems
    var dst_ptr = contiguous_ptr[dst_dt](result)
    var one = Scalar[dst_dt](1)
    var zero = Scalar[dst_dt](0)
    for i in range(src.size_value):
        dst_ptr[i] = one if src_ptr[i] != UInt8(0) else zero


def _real_to_bool_dst[src_dt: DType](src: Array, mut result: Array) raises:
    """Bool dest storage is raw UInt8; write 1 for nonzero src, 0 otherwise."""
    var src_ptr = contiguous_ptr[src_dt](src)
    var dst_ptr = result.data + result.offset_elems
    var src_zero = Scalar[src_dt](0)
    for i in range(src.size_value):
        dst_ptr[i] = UInt8(1) if src_ptr[i] != src_zero else UInt8(0)


def _complex_contig_lane_cast[
    src_dt: DType, dst_dt: DType
](src: Array, mut result: Array) raises where src_dt.is_floating_point() and dst_dt.is_floating_point():
    var src_ptr = src.data.bitcast[Scalar[src_dt]]() + src.offset_elems * 2
    var dst_ptr = result.data.bitcast[Scalar[dst_dt]]() + result.offset_elems * 2
    var lane_count = src.size_value * 2
    comptime width = simd_width_of[src_dt]()
    var i = 0
    while i + width <= lane_count:
        dst_ptr.store(i, src_ptr.load[width=width](i).cast[dst_dt]())
        i += width
    while i < lane_count:
        dst_ptr[i] = src_ptr[i].cast[dst_dt]()
        i += 1


def _copy_rank2_strided_typed[dtype: DType](src: Array, mut result: Array) raises:
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_row_stride = src.strides[0]
    var src_col_stride = src.strides[1]
    var result_row_stride = result.strides[0]
    var result_col_stride = result.strides[1]
    var src_data = src.data.bitcast[Scalar[dtype]]()
    var result_data = result.data.bitcast[Scalar[dtype]]()
    comptime tile = 8
    var row_block = 0
    while row_block < rows:
        var row_end = row_block + tile
        if row_end > rows:
            row_end = rows
        var col_block = 0
        while col_block < cols:
            var col_end = col_block + tile
            if col_end > cols:
                col_end = cols
            if src_row_stride == 1:
                var col = col_block
                while col < col_end:
                    var row = row_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while row < row_end:
                        result_data[result_index] = src_data[src_index]
                        row += 1
                        src_index += src_row_stride
                        result_index += result_row_stride
                    col += 1
            else:
                var row = row_block
                while row < row_end:
                    var col = col_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while col < col_end:
                        result_data[result_index] = src_data[src_index]
                        col += 1
                        src_index += src_col_stride
                        result_index += result_col_stride
                    row += 1
            col_block += tile
        row_block += tile


def _copy_rank2_strided_bool(src: Array, mut result: Array) raises:
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_row_stride = src.strides[0]
    var src_col_stride = src.strides[1]
    var result_row_stride = result.strides[0]
    var result_col_stride = result.strides[1]
    var src_data = src.data.bitcast[Bool]()
    var result_data = result.data.bitcast[Bool]()
    comptime tile = 8
    var row_block = 0
    while row_block < rows:
        var row_end = row_block + tile
        if row_end > rows:
            row_end = rows
        var col_block = 0
        while col_block < cols:
            var col_end = col_block + tile
            if col_end > cols:
                col_end = cols
            if src_row_stride == 1:
                var col = col_block
                while col < col_end:
                    var row = row_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while row < row_end:
                        result_data[result_index] = src_data[src_index]
                        row += 1
                        src_index += src_row_stride
                        result_index += result_row_stride
                    col += 1
            else:
                var row = row_block
                while row < row_end:
                    var col = col_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while col < col_end:
                        result_data[result_index] = src_data[src_index]
                        col += 1
                        src_index += src_col_stride
                        result_index += result_col_stride
                    row += 1
            col_block += tile
        row_block += tile


def _copy_rank2_strided_complex32(src: Array, mut result: Array) raises:
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_row_stride = src.strides[0]
    var src_col_stride = src.strides[1]
    var result_row_stride = result.strides[0]
    var result_col_stride = result.strides[1]
    var src_data = src.data.bitcast[Float32]()
    var result_data = result.data.bitcast[Float32]()
    comptime tile = 8
    var row_block = 0
    while row_block < rows:
        var row_end = row_block + tile
        if row_end > rows:
            row_end = rows
        var col_block = 0
        while col_block < cols:
            var col_end = col_block + tile
            if col_end > cols:
                col_end = cols
            if src_row_stride == 1:
                var col = col_block
                while col < col_end:
                    var row = row_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while row < row_end:
                        result_data[result_index * 2] = src_data[src_index * 2]
                        result_data[result_index * 2 + 1] = src_data[src_index * 2 + 1]
                        row += 1
                        src_index += src_row_stride
                        result_index += result_row_stride
                    col += 1
            else:
                var row = row_block
                while row < row_end:
                    var col = col_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while col < col_end:
                        result_data[result_index * 2] = src_data[src_index * 2]
                        result_data[result_index * 2 + 1] = src_data[src_index * 2 + 1]
                        col += 1
                        src_index += src_col_stride
                        result_index += result_col_stride
                    row += 1
            col_block += tile
        row_block += tile


def _copy_rank2_strided_complex64(src: Array, mut result: Array) raises:
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_row_stride = src.strides[0]
    var src_col_stride = src.strides[1]
    var result_row_stride = result.strides[0]
    var result_col_stride = result.strides[1]
    var src_data = src.data.bitcast[Float64]()
    var result_data = result.data.bitcast[Float64]()
    comptime tile = 8
    var row_block = 0
    while row_block < rows:
        var row_end = row_block + tile
        if row_end > rows:
            row_end = rows
        var col_block = 0
        while col_block < cols:
            var col_end = col_block + tile
            if col_end > cols:
                col_end = cols
            if src_row_stride == 1:
                var col = col_block
                while col < col_end:
                    var row = row_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while row < row_end:
                        result_data[result_index * 2] = src_data[src_index * 2]
                        result_data[result_index * 2 + 1] = src_data[src_index * 2 + 1]
                        row += 1
                        src_index += src_row_stride
                        result_index += result_row_stride
                    col += 1
            else:
                var row = row_block
                while row < row_end:
                    var col = col_block
                    var src_index = src.offset_elems + row * src_row_stride + col * src_col_stride
                    var result_index = result.offset_elems + row * result_row_stride + col * result_col_stride
                    while col < col_end:
                        result_data[result_index * 2] = src_data[src_index * 2]
                        result_data[result_index * 2 + 1] = src_data[src_index * 2 + 1]
                        col += 1
                        src_index += src_col_stride
                        result_index += result_col_stride
                    row += 1
            col_block += tile
        row_block += tile


def _maybe_copy_rank2_strided(src: Array, mut result: Array) raises -> Bool:
    if len(src.shape) != 2:
        return False
    if src.size_value == 0:
        return True
    if src.dtype_code == ArrayDType.BOOL.value:
        _copy_rank2_strided_bool(src, result)
        return True
    if src.dtype_code == ArrayDType.INT8.value:
        _copy_rank2_strided_typed[DType.int8](src, result)
        return True
    if src.dtype_code == ArrayDType.INT16.value:
        _copy_rank2_strided_typed[DType.int16](src, result)
        return True
    if src.dtype_code == ArrayDType.INT32.value:
        _copy_rank2_strided_typed[DType.int32](src, result)
        return True
    if src.dtype_code == ArrayDType.INT64.value:
        _copy_rank2_strided_typed[DType.int64](src, result)
        return True
    if src.dtype_code == ArrayDType.UINT8.value:
        _copy_rank2_strided_typed[DType.uint8](src, result)
        return True
    if src.dtype_code == ArrayDType.UINT16.value:
        _copy_rank2_strided_typed[DType.uint16](src, result)
        return True
    if src.dtype_code == ArrayDType.UINT32.value:
        _copy_rank2_strided_typed[DType.uint32](src, result)
        return True
    if src.dtype_code == ArrayDType.UINT64.value:
        _copy_rank2_strided_typed[DType.uint64](src, result)
        return True
    if src.dtype_code == ArrayDType.FLOAT16.value:
        _copy_rank2_strided_typed[DType.float16](src, result)
        return True
    if src.dtype_code == ArrayDType.FLOAT32.value:
        _copy_rank2_strided_typed[DType.float32](src, result)
        return True
    if src.dtype_code == ArrayDType.FLOAT64.value:
        _copy_rank2_strided_typed[DType.float64](src, result)
        return True
    if src.dtype_code == ArrayDType.COMPLEX64.value:
        _copy_rank2_strided_complex32(src, result)
        return True
    if src.dtype_code == ArrayDType.COMPLEX128.value:
        _copy_rank2_strided_complex64(src, result)
        return True
    return False


def copy_c_contiguous(src: Array) raises -> Array:
    var shape = clone_int_list(src.shape)
    var result = make_empty_array(src.dtype_code, shape^)
    if is_c_contiguous(src):
        if not dtype_is_packed_subbyte(src.dtype_code) or src.offset_elems % 2 == 0:
            var src_byte_offset = dtype_byte_offset(src.dtype_code, src.offset_elems)
            var byte_count = dtype_storage_byte_len(src.dtype_code, src.size_value)
            memcpy(
                dest=result.data,
                src=src.data + src_byte_offset,
                count=byte_count,
            )
            return result^
    if _maybe_copy_rank2_strided(src, result):
        return result^
    for i in range(src.size_value):
        var physical = physical_offset(src, i)
        if src.dtype_code == ArrayDType.BOOL.value:
            if get_physical_bool(src, physical):
                set_logical_from_f64(result, i, 1.0)
            else:
                set_logical_from_f64(result, i, 0.0)
        elif src.dtype_code == ArrayDType.INT64.value:
            set_logical_from_i64(result, i, get_physical[DType.int64](src, physical))
        else:
            set_logical_from_f64(result, i, get_physical_as_f64(src, physical))
    return result^


def _maybe_cast_contiguous_core_dtypes(src: Array, mut result: Array) raises -> Bool:
    """Fast path for c-contiguous → c-contiguous casts.

    Three layers, in order of fall-through:
    1. Bool ↔ real (via `dispatch_real_typed_contig_pair`): UInt8 storage with
       nonzero-is-true semantics is bespoke per-byte; the dispatcher fans out
       on the *real* side. Covers 22 pairs (11 src→bool + bool→11 dst).
    2. Real ↔ real (via `dispatch_real_pair_cast`): 121 pairs through
       `Scalar[src_dt] → Scalar[dst_dt].cast[dst_dt]()`.
    3. Complex64 ↔ complex128 width changes: vector-cast the 2N interleaved
       real lanes directly.
    4. Other complex pairs return False; caller's per-element loop handles
       complex → real drops and real → complex zero-imag writes.
    """
    if not is_c_contiguous(src) or not is_c_contiguous(result):
        return False
    var src_c = src.dtype_code
    var dst_c = result.dtype_code
    if src_c == ArrayDType.BOOL.value:
        return dispatch_real_typed_contig_pair[_bool_src_to_real](dst_c, src, result)
    if dst_c == ArrayDType.BOOL.value:
        return dispatch_real_typed_contig_pair[_real_to_bool_dst](src_c, src, result)
    var src_is_complex = src_c == ArrayDType.COMPLEX64.value or src_c == ArrayDType.COMPLEX128.value
    var dst_is_complex = dst_c == ArrayDType.COMPLEX64.value or dst_c == ArrayDType.COMPLEX128.value
    if src_is_complex or dst_is_complex:
        if src_c == ArrayDType.COMPLEX64.value and dst_c == ArrayDType.COMPLEX128.value:
            _complex_contig_lane_cast[DType.float32, DType.float64](src, result)
            return True
        if src_c == ArrayDType.COMPLEX128.value and dst_c == ArrayDType.COMPLEX64.value:
            _complex_contig_lane_cast[DType.float64, DType.float32](src, result)
            return True
        return False
    return dispatch_real_pair_cast[_contig_pair_cast](src_c, dst_c, src, result)


def cast_copy_array(src: Array, dtype_code: Int) raises -> Array:
    if src.dtype_code == dtype_code:
        return copy_c_contiguous(src)
    var shape = clone_int_list(src.shape)
    var result = make_empty_array(dtype_code, shape^)
    if _maybe_cast_contiguous_core_dtypes(src, result):
        return result^
    var dst_is_complex = dtype_code == ArrayDType.COMPLEX64.value or dtype_code == ArrayDType.COMPLEX128.value
    var src_is_c = is_c_contiguous(src)
    for i in range(src.size_value):
        var physical = src.offset_elems + i
        if not src_is_c:
            physical = physical_offset(src, i)
        if src.dtype_code == ArrayDType.COMPLEX64.value:
            var re = Float64(get_physical_c64_real(src, physical))
            var im = Float64(get_physical_c64_imag(src, physical))
            if dst_is_complex:
                if dtype_code == ArrayDType.COMPLEX64.value:
                    set_physical_c64(result, i, Float32(re), Float32(im))
                else:
                    set_physical_c128(result, i, re, im)
            else:
                set_logical_from_f64(result, i, re)  # numpy drops imag
            continue
        if src.dtype_code == ArrayDType.COMPLEX128.value:
            var re = get_physical_c128_real(src, physical)
            var im = get_physical_c128_imag(src, physical)
            if dst_is_complex:
                if dtype_code == ArrayDType.COMPLEX64.value:
                    set_physical_c64(result, i, Float32(re), Float32(im))
                else:
                    set_physical_c128(result, i, re, im)
            else:
                set_logical_from_f64(result, i, re)
            continue
        # Real → anything (including complex). Read source real value and
        # write through the unified setters (which zero imag for complex).
        if src.dtype_code == ArrayDType.BOOL.value:
            if get_physical_bool(src, physical):
                set_logical_from_i64(result, i, 1)
            else:
                set_logical_from_i64(result, i, 0)
        elif src.dtype_code == ArrayDType.INT64.value:
            set_logical_from_i64(result, i, get_physical[DType.int64](src, physical))
        elif src.dtype_code == ArrayDType.INT32.value:
            set_logical_from_i64(result, i, Int64(Int(get_physical[DType.int32](src, physical))))
        elif src.dtype_code == ArrayDType.INT16.value:
            set_logical_from_i64(result, i, Int64(Int(get_physical[DType.int16](src, physical))))
        elif src.dtype_code == ArrayDType.INT8.value:
            set_logical_from_i64(result, i, Int64(Int(get_physical[DType.int8](src, physical))))
        elif src.dtype_code == ArrayDType.UINT64.value:
            set_logical_from_f64(result, i, Float64(Int(get_physical[DType.uint64](src, physical))))
        elif src.dtype_code == ArrayDType.UINT32.value:
            set_logical_from_i64(result, i, Int64(Int(get_physical[DType.uint32](src, physical))))
        elif src.dtype_code == ArrayDType.UINT16.value:
            set_logical_from_i64(result, i, Int64(Int(get_physical[DType.uint16](src, physical))))
        elif src.dtype_code == ArrayDType.UINT8.value:
            set_logical_from_i64(result, i, Int64(Int(get_physical[DType.uint8](src, physical))))
        else:
            set_logical_from_f64(result, i, get_physical_as_f64(src, physical))
    return result^
