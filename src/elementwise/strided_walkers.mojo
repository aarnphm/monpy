"""Strided walkers for same-shape and broadcast binary ops.

Hosts:
  - `strided_binary_walk_typed[dtype]` — N-D coord-stack walker with four
    inner-loop kinds (full SIMD, SIMD+scatter, reversed-SIMD, scalar);
    used by `maybe_binary_same_shape_strided`.
  - `binary_strided_walk_typed[dtype]` — flat counter walker for broadcast
    cases (size-1 dims zeroed on the corresponding stride).
  - `maybe_binary_strided_typed` — 10-way real-dtype dispatch into the
    flat-counter walker.
"""

from std.sys import simd_width_of

from array import Array
from domain import ArrayDType, BinaryOp

from .typed_kernels import apply_binary_typed_vec


def strided_binary_walk_typed[
    dtype: DType
](
    lhs: Array,
    rhs: Array,
    mut result: Array,
    inner_kind: Int,
    inner_size: Int,
    inner_lhs_stride: Int,
    inner_rhs_stride: Int,
    inner_result_stride: Int,
    outer_shape: List[Int],
    outer_lhs_stride: List[Int],
    outer_rhs_stride: List[Int],
    outer_result_stride: List[Int],
    outer_lhs_carry: List[Int],
    outer_rhs_carry: List[Int],
    outer_result_carry: List[Int],
    op: Int,
) raises:
    # Comptime-typed body for `maybe_binary_same_shape_strided`. Combines an
    # outer coord-stack walker (axis-iteration with no divmod — uses
    # pre-computed carry tables to advance offsets per-axis) with one of four
    # inner-loop kinds picked upstream by `pick_inner_axis_for_strided_binary`:
    #
    #   inner_kind == 1: unit-stride contiguous inner. SIMD-vectorise the inner
    #     loop (load[width] / store[width]) until tail; tail is scalar. Highest
    #     bandwidth — useful bytes per cycle close to peak L1.
    #   inner_kind == 2: unit-stride inner with shared outer carry. Same SIMD
    #     vectorise as kind 1; outer carries are non-trivial enough to warrant
    #     a separate path that recomputes offsets per-row.
    #   inner_kind == 3: strided inner (non-unit). Per-element gather/scatter
    #     through `lp[i*stride]` — effective bandwidth is one element per
    #     cycle, no SIMD.
    #   inner_kind == 4: reversed inner (negative stride). Same as kind 3 but
    #     the index walks backward; no SIMD because vector loads can't reverse.
    #
    # The four-way split exists because SIMD bandwidth is dominated by load/store
    # alignment; picking the wrong inner walk leaves 4-8× perf on the floor.
    # Cross-ref `docs/research/memory-alignment.md §3`.
    # Caller must verify dtype_code matches `dtype`.
    var lhs_data = lhs.data.bitcast[Scalar[dtype]]()
    var rhs_data = rhs.data.bitcast[Scalar[dtype]]()
    var result_data = result.data.bitcast[Scalar[dtype]]()
    comptime width = simd_width_of[dtype]()
    var outer_ndim = len(outer_shape)
    var coords = List[Int]()
    for _ in range(outer_ndim):
        coords.append(0)
    var lhs_offset = lhs.offset_elems
    var rhs_offset = rhs.offset_elems
    var result_offset = result.offset_elems
    while True:
        if inner_kind == 1:
            var lp = lhs_data + lhs_offset
            var rp = rhs_data + rhs_offset
            var op_ = result_data + result_offset
            var i = 0
            while i + width <= inner_size:
                var lvec = lp.load[width=width](i)
                var rvec = rp.load[width=width](i)
                if op == BinaryOp.ADD.value:
                    op_.store(i, lvec + rvec)
                else:
                    op_.store(i, apply_binary_typed_vec[dtype, width](lvec, rvec, op))
                i += width
            while i < inner_size:
                if op == BinaryOp.ADD.value:
                    op_[i] = lp[i] + rp[i]
                else:
                    op_[i] = apply_binary_typed_vec[dtype, 1](SIMD[dtype, 1](lp[i]), SIMD[dtype, 1](rp[i]), op)[0]
                i += 1
        elif inner_kind == 2:
            var lp = lhs_data + lhs_offset
            var rp = rhs_data + rhs_offset
            var i = 0
            while i + width <= inner_size:
                var lvec = lp.load[width=width](i)
                var rvec = rp.load[width=width](i)
                var ovec = lvec + rvec if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, width](
                    lvec, rvec, op
                )
                comptime for k in range(width):
                    result_data[result_offset + (i + k) * inner_result_stride] = ovec[k]
                i += width
            while i < inner_size:
                if op == BinaryOp.ADD.value:
                    result_data[result_offset + i * inner_result_stride] = lp[i] + rp[i]
                else:
                    result_data[result_offset + i * inner_result_stride] = apply_binary_typed_vec[dtype, 1](
                        SIMD[dtype, 1](lp[i]), SIMD[dtype, 1](rp[i]), op
                    )[0]
                i += 1
        elif inner_kind == 3:
            # lhs/rhs stride -1, result stride +/-1. Logical position i
            # maps to physical (offset - i). Loading [W] at physical
            # (offset - i - W + 1) gives us elements at logical positions
            # [i+W-1, i+W-2, ..., i] — reverse the SIMD vector to put
            # them back in logical order, then store contiguously (or
            # reversed, matching result's stride sign).
            var i = 0
            while i + width <= inner_size:
                var lvec = (lhs_data + lhs_offset - i - width + 1).load[width=width](0).reversed()
                var rvec = (rhs_data + rhs_offset - i - width + 1).load[width=width](0).reversed()
                var ovec = lvec + rvec if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, width](
                    lvec, rvec, op
                )
                if inner_result_stride == 1:
                    (result_data + result_offset + i).store(0, ovec)
                else:
                    (result_data + result_offset - i - width + 1).store(0, ovec.reversed())
                i += width
            while i < inner_size:
                if op == BinaryOp.ADD.value:
                    result_data[result_offset + i * inner_result_stride] = (
                        lhs_data[lhs_offset - i] + rhs_data[rhs_offset - i]
                    )
                else:
                    result_data[result_offset + i * inner_result_stride] = apply_binary_typed_vec[dtype, 1](
                        SIMD[dtype, 1](lhs_data[lhs_offset - i]),
                        SIMD[dtype, 1](rhs_data[rhs_offset - i]),
                        op,
                    )[0]
                i += 1
        else:
            for i in range(inner_size):
                var lval = lhs_data[lhs_offset + i * inner_lhs_stride]
                var rval = rhs_data[rhs_offset + i * inner_rhs_stride]
                if op == BinaryOp.ADD.value:
                    result_data[result_offset + i * inner_result_stride] = lval + rval
                else:
                    result_data[result_offset + i * inner_result_stride] = apply_binary_typed_vec[dtype, 1](
                        SIMD[dtype, 1](lval),
                        SIMD[dtype, 1](rval),
                        op,
                    )[0]
        if outer_ndim == 0:
            break
        var idx = outer_ndim - 1
        var done = False
        while idx >= 0:
            coords[idx] += 1
            if coords[idx] < outer_shape[idx]:
                lhs_offset += outer_lhs_stride[idx]
                rhs_offset += outer_rhs_stride[idx]
                result_offset += outer_result_stride[idx]
                break
            coords[idx] = 0
            lhs_offset -= outer_lhs_carry[idx]
            rhs_offset -= outer_rhs_carry[idx]
            result_offset -= outer_result_carry[idx]
            idx -= 1
            if idx < 0:
                done = True
        if done:
            break


def binary_strided_walk_typed[dtype: DType](lhs: Array, rhs: Array, mut result: Array, op: Int) raises:
    # Strided binary walker for same-dtype broadcasts. Output is c-contig
    # (just allocated), so we walk it as a flat counter and maintain
    # incremental cursors for lhs/rhs in element units. Strides for
    # broadcast dims (size-1 source against >1 target, or freshly added
    # outer axes) are zeroed out — same semantics as
    # `as_broadcast_layout` but inlined to skip per-step `* item_size`
    # multiplications and the List[List[Int]] indexing in
    # MultiLayoutIter. Net: ~30× faster on a 256² .T + 1D add.
    var lhs_ptr = lhs.data.bitcast[Scalar[dtype]]()
    var rhs_ptr = rhs.data.bitcast[Scalar[dtype]]()
    var out_ptr = result.data.bitcast[Scalar[dtype]]()
    var ndim = len(result.shape)
    var n = result.size_value
    var lhs_ndim = len(lhs.shape)
    var rhs_ndim = len(rhs.shape)
    var lhs_strides = List[Int]()
    var rhs_strides = List[Int]()
    for d in range(ndim):
        var l_axis = d - (ndim - lhs_ndim)
        var r_axis = d - (ndim - rhs_ndim)
        if l_axis < 0 or (lhs.shape[l_axis] == 1 and result.shape[d] != 1):
            lhs_strides.append(0)
        else:
            lhs_strides.append(lhs.strides[l_axis])
        if r_axis < 0 or (rhs.shape[r_axis] == 1 and result.shape[d] != 1):
            rhs_strides.append(0)
        else:
            rhs_strides.append(rhs.strides[r_axis])
    if ndim == 0:
        # rank-0 result: single scalar.
        out_ptr[result.offset_elems] = apply_binary_typed_vec[dtype, 1](
            SIMD[dtype, 1](lhs_ptr[lhs.offset_elems]),
            SIMD[dtype, 1](rhs_ptr[rhs.offset_elems]),
            op,
        )[0]
        return
    var coords = List[Int]()
    for _ in range(ndim):
        coords.append(0)
    var lhs_idx = lhs.offset_elems
    var rhs_idx = rhs.offset_elems
    var out_idx = result.offset_elems
    var visited = 0
    while visited < n:
        out_ptr[out_idx] = apply_binary_typed_vec[dtype, 1](
            SIMD[dtype, 1](lhs_ptr[lhs_idx]),
            SIMD[dtype, 1](rhs_ptr[rhs_idx]),
            op,
        )[0]
        visited += 1
        if visited >= n:
            break
        out_idx += 1
        var axis = ndim - 1
        while axis >= 0:
            coords[axis] += 1
            lhs_idx += lhs_strides[axis]
            rhs_idx += rhs_strides[axis]
            if coords[axis] < result.shape[axis]:
                break
            var rollback_lhs = coords[axis] * lhs_strides[axis]
            var rollback_rhs = coords[axis] * rhs_strides[axis]
            lhs_idx -= rollback_lhs
            rhs_idx -= rollback_rhs
            coords[axis] = 0
            axis -= 1


def maybe_binary_strided_typed(lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    # Dispatch the strided binary walker by dtype. Requires lhs, rhs,
    # and result to share a dtype (the common broadcast case for same-
    # type ops). Mixed-dtype broadcasts still take the f64 round-trip
    # walker upstream.
    if lhs.dtype_code != rhs.dtype_code or lhs.dtype_code != result.dtype_code:
        return False
    if lhs.dtype_code == ArrayDType.FLOAT64.value:
        binary_strided_walk_typed[DType.float64](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.FLOAT32.value:
        binary_strided_walk_typed[DType.float32](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.INT64.value:
        binary_strided_walk_typed[DType.int64](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.INT32.value:
        binary_strided_walk_typed[DType.int32](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.INT16.value:
        binary_strided_walk_typed[DType.int16](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.INT8.value:
        binary_strided_walk_typed[DType.int8](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.UINT64.value:
        binary_strided_walk_typed[DType.uint64](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.UINT32.value:
        binary_strided_walk_typed[DType.uint32](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.UINT16.value:
        binary_strided_walk_typed[DType.uint16](lhs, rhs, result, op)
        return True
    if lhs.dtype_code == ArrayDType.UINT8.value:
        binary_strided_walk_typed[DType.uint8](lhs, rhs, result, op)
        return True
    return False
