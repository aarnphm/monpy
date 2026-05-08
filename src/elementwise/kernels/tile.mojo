"""4×4 SIMD-tile kernels for transposed-input + column-broadcast patterns.

Hosts:
  - `StridedInnerChoice` — small struct returned by `pick_inner_axis_for_strided_binary`.
  - `pick_inner_axis_for_strided_binary` — chooses the inner axis (and inner-loop
    kind) for a same-shape strided binary walk.
  - `maybe_binary_rank2_transposed_tile[dtype]` — rank-2 F-contig (`.T`) +
    F-contig case: process as 4×4 tiles via in-register transpose to keep
    both load and store streams contiguous.
  - `maybe_binary_rank3_axis0_tile[dtype]` — batched rank-3 variant for
    layouts like `arr.transpose((2, 0, 1))`.
  - `maybe_binary_rank2_transposed_tile_bcast_1d[dtype]` — tile path for
    column-broadcast: `lhs` is rank-2 F-contig and `rhs` is rank-1
    broadcasting along axis 0 of lhs.
  - `maybe_binary_column_broadcast_dispatch` — 6-way real dtype dispatcher
    that tries both operand orderings.

The 4×4 in-register transpose pattern (two-step shuffle, lowering to NEON
`vzip1`/`vzip2` or AVX equivalent) is the load-bearing trick: without it
the generic strided walker picks SIMD-load + W-element scatter store,
which loses 4-8× to scattered stores stalling the write buffer.
"""

from std.sys import simd_width_of

from array import (
    Array,
    is_c_contiguous,
    same_shape,
)
from domain import ArrayDType, BackendKind, BinaryOp

from elementwise.kernels.typed import apply_binary_typed_vec


@fieldwise_init
struct StridedInnerChoice(ImplicitlyCopyable, Movable):
    var axis: Int
    var kind: Int  # 0=scalar, 1=full-SIMD, 2=SIMD-load+scatter


def pick_inner_axis_for_strided_binary(lhs: Array, rhs: Array, result: Array) raises -> StridedInnerChoice:
    # Pick the inner axis for a same-shape strided binary walk.
    #   kind == 1: all three operands have stride +1 on `axis` -> full SIMD
    #   kind == 2: lhs and rhs have stride +1 -> SIMD load + scatter store
    #   kind == 3: lhs and rhs have stride -1, result has stride +1 (or -1) ->
    #              SIMD reversed-load + contiguous store. covers `[::-1]+[::-1]`.
    #   kind == 0: scalar walk (no |stride|==1 axis on inputs)
    # Prefers higher-`kind` choices, then the rightmost (innermost) axis among
    # equally-good candidates so iteration order remains C-natural.
    var ndim = len(lhs.shape)
    var inner_axis = ndim - 1
    var inner_kind = 0
    for axis in range(ndim - 1, -1, -1):
        var ls = lhs.strides[axis]
        var rs = rhs.strides[axis]
        var os = result.strides[axis]
        if ls == 1 and rs == 1:
            if os == 1:
                return StridedInnerChoice(axis, 1)
            if inner_kind < 2:
                inner_axis = axis
                inner_kind = 2
        elif ls == -1 and rs == -1 and (os == 1 or os == -1):
            # Reversed inputs with contig (or reversed-contig) output. Covers
            # the rank-1 `arr[::-1] + brr[::-1]` family without falling into
            # the scalar path.
            if inner_kind < 3:
                inner_axis = axis
                inner_kind = 3
    return StridedInnerChoice(inner_axis, inner_kind)


def maybe_binary_rank2_transposed_tile[
    dtype: DType
](lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    # Tile-transpose fast path for rank-2 binary ops where both inputs have
    # stride-1 axis 0 (the F-contig pattern that `arr.T` produces from a
    # c-contig `arr`) and the result is c-contig. Without this, the generic
    # strided walker picks kind=2 (SIMD-load + W-element scatter store)
    # which loses 4-8× to scattered stores stalling the write buffer.
    #
    # Strategy: process the array as 4-by-4 tiles. For each tile we do four
    # contiguous SIMD loads from each input along axis 0 (stride-1), apply
    # the binary op, transpose the resulting 4 SIMD vectors in registers,
    # and emit four contiguous SIMD stores to result. Both load and store
    # streams are contiguous; only the in-register shuffle costs.
    #
    # Caller must verify lhs/rhs/result dtype_code matches `dtype` before
    # calling. tile=4 works for any float dtype: f32 fits the natural NEON
    # width directly; f64 uses two NEON registers per "vector", trading
    # some throughput for the same scatter-avoiding pattern.
    if len(lhs.shape) != 2:
        return False
    if not same_shape(lhs.shape, rhs.shape) or not same_shape(lhs.shape, result.shape):
        return False
    if lhs.strides[0] != 1 or rhs.strides[0] != 1:
        return False
    if not is_c_contiguous(result):
        return False
    var rows = lhs.shape[0]
    var cols = lhs.shape[1]
    var lhs_col_stride = lhs.strides[1]
    var rhs_col_stride = rhs.strides[1]
    var result_row_stride = result.strides[0]
    var lhs_data = lhs.data.bitcast[Scalar[dtype]]() + lhs.offset_elems
    var rhs_data = rhs.data.bitcast[Scalar[dtype]]() + rhs.offset_elems
    var result_data = result.data.bitcast[Scalar[dtype]]() + result.offset_elems
    if lhs_col_stride == rows and rhs_col_stride == rows:
        comptime width = simd_width_of[dtype]()
        var total = rows * cols
        var linear = 0
        while linear + width <= total:
            var lvec = lhs_data.load[width=width](linear)
            var rvec = rhs_data.load[width=width](linear)
            var out = lvec + rvec if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, width](lvec, rvec, op)
            result_data.store(linear, out)
            linear += width
        while linear < total:
            if op == BinaryOp.ADD.value:
                result_data[linear] = lhs_data[linear] + rhs_data[linear]
            else:
                result_data[linear] = apply_binary_typed_vec[dtype, 1](
                    SIMD[dtype, 1](lhs_data[linear]), SIMD[dtype, 1](rhs_data[linear]), op
                )[0]
            linear += 1
        result.strides[0] = 1
        result.strides[1] = rows
        result.backend_code = BackendKind.FUSED.value
        return True
    if rows < 4 or cols < 4:
        return False
    comptime tile = 4
    var main_rows = rows - (rows % tile)
    var main_cols = cols - (cols % tile)
    var i = 0
    while i < main_rows:
        var j = 0
        while j < main_cols:
            var l0 = (lhs_data + i + j * lhs_col_stride).load[width=tile](0)
            var l1 = (lhs_data + i + (j + 1) * lhs_col_stride).load[width=tile](0)
            var l2 = (lhs_data + i + (j + 2) * lhs_col_stride).load[width=tile](0)
            var l3 = (lhs_data + i + (j + 3) * lhs_col_stride).load[width=tile](0)
            var r0 = (rhs_data + i + j * rhs_col_stride).load[width=tile](0)
            var r1 = (rhs_data + i + (j + 1) * rhs_col_stride).load[width=tile](0)
            var r2 = (rhs_data + i + (j + 2) * rhs_col_stride).load[width=tile](0)
            var r3 = (rhs_data + i + (j + 3) * rhs_col_stride).load[width=tile](0)
            var s0 = l0 + r0 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l0, r0, op)
            var s1 = l1 + r1 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l1, r1, op)
            var s2 = l2 + r2 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l2, r2, op)
            var s3 = l3 + r3 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l3, r3, op)
            # 4x4 in-register transpose. The two-step zip pattern:
            # step 1 zips paired SIMD vectors, step 2 zips the 64-bit lanes
            # of the result. Mojo's `shuffle[*mask]` lowers to NEON's
            # `vzip1`/`vzip2` (and the AVX equivalent on x86) directly.
            var u0 = s0.shuffle[0, 4, 1, 5](s1)
            var u1 = s0.shuffle[2, 6, 3, 7](s1)
            var u2 = s2.shuffle[0, 4, 1, 5](s3)
            var u3 = s2.shuffle[2, 6, 3, 7](s3)
            var t0 = u0.shuffle[0, 1, 4, 5](u2)
            var t1 = u0.shuffle[2, 3, 6, 7](u2)
            var t2 = u1.shuffle[0, 1, 4, 5](u3)
            var t3 = u1.shuffle[2, 3, 6, 7](u3)
            (result_data + i * result_row_stride + j).store(0, t0)
            (result_data + (i + 1) * result_row_stride + j).store(0, t1)
            (result_data + (i + 2) * result_row_stride + j).store(0, t2)
            (result_data + (i + 3) * result_row_stride + j).store(0, t3)
            j += tile
        # column tail: scalar walk for residual cols within the main rows
        while j < cols:
            comptime for k in range(tile):
                var lv = lhs_data[i + k + j * lhs_col_stride]
                var rv = rhs_data[i + k + j * rhs_col_stride]
                if op == BinaryOp.ADD.value:
                    result_data[(i + k) * result_row_stride + j] = lv + rv
                else:
                    result_data[(i + k) * result_row_stride + j] = apply_binary_typed_vec[dtype, 1](
                        SIMD[dtype, 1](lv), SIMD[dtype, 1](rv), op
                    )[0]
            j += 1
        i += tile
    # row tail: scalar walk for residual rows
    while i < rows:
        var j = 0
        while j < cols:
            var lv = lhs_data[i + j * lhs_col_stride]
            var rv = rhs_data[i + j * rhs_col_stride]
            if op == BinaryOp.ADD.value:
                result_data[i * result_row_stride + j] = lv + rv
            else:
                result_data[i * result_row_stride + j] = apply_binary_typed_vec[dtype, 1](
                    SIMD[dtype, 1](lv), SIMD[dtype, 1](rv), op
                )[0]
            j += 1
        i += 1
    result.backend_code = BackendKind.FUSED.value
    return True


def maybe_binary_rank3_axis0_tile[dtype: DType](lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    # Batched variant of the rank-2 transposed tile. Covers layouts like
    # C-contig rank-3 `.transpose((2, 0, 1))`: each middle-axis slice is a
    # rank-2 transpose with input stride-1 on axis 0 and c-contig output.
    if len(lhs.shape) != 3:
        return False
    if not same_shape(lhs.shape, rhs.shape) or not same_shape(lhs.shape, result.shape):
        return False
    if lhs.strides[0] != 1 or rhs.strides[0] != 1:
        return False
    if not is_c_contiguous(result):
        return False
    var rows = lhs.shape[0]
    var batches = lhs.shape[1]
    var cols = lhs.shape[2]
    var lhs_batch_stride = lhs.strides[1]
    var rhs_batch_stride = rhs.strides[1]
    var lhs_col_stride = lhs.strides[2]
    var rhs_col_stride = rhs.strides[2]
    var result_row_stride = result.strides[0]
    var result_batch_stride = result.strides[1]
    var lhs_data = lhs.data.bitcast[Scalar[dtype]]() + lhs.offset_elems
    var rhs_data = rhs.data.bitcast[Scalar[dtype]]() + rhs.offset_elems
    var result_data = result.data.bitcast[Scalar[dtype]]() + result.offset_elems
    var physical_plane = rows * cols
    if (
        lhs_col_stride == rows
        and rhs_col_stride == rows
        and lhs_batch_stride == physical_plane
        and rhs_batch_stride == physical_plane
    ):
        comptime width = simd_width_of[dtype]()
        var total = rows * batches * cols
        var linear = 0
        while linear + width <= total:
            var lvec = lhs_data.load[width=width](linear)
            var rvec = rhs_data.load[width=width](linear)
            var out = lvec + rvec if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, width](lvec, rvec, op)
            result_data.store(linear, out)
            linear += width
        while linear < total:
            if op == BinaryOp.ADD.value:
                result_data[linear] = lhs_data[linear] + rhs_data[linear]
            else:
                result_data[linear] = apply_binary_typed_vec[dtype, 1](
                    SIMD[dtype, 1](lhs_data[linear]), SIMD[dtype, 1](rhs_data[linear]), op
                )[0]
            linear += 1
        result.strides[0] = 1
        result.strides[1] = physical_plane
        result.strides[2] = rows
        result.backend_code = BackendKind.FUSED.value
        return True
    if rows < 4 or cols < 4:
        return False
    comptime tile = 4
    var main_rows = rows - (rows % tile)
    var main_cols = cols - (cols % tile)
    for batch in range(batches):
        var lhs_batch = lhs_data + batch * lhs_batch_stride
        var rhs_batch = rhs_data + batch * rhs_batch_stride
        var result_batch = result_data + batch * result_batch_stride
        var i = 0
        while i < main_rows:
            var j = 0
            while j < main_cols:
                var l0 = (lhs_batch + i + j * lhs_col_stride).load[width=tile](0)
                var l1 = (lhs_batch + i + (j + 1) * lhs_col_stride).load[width=tile](0)
                var l2 = (lhs_batch + i + (j + 2) * lhs_col_stride).load[width=tile](0)
                var l3 = (lhs_batch + i + (j + 3) * lhs_col_stride).load[width=tile](0)
                var r0 = (rhs_batch + i + j * rhs_col_stride).load[width=tile](0)
                var r1 = (rhs_batch + i + (j + 1) * rhs_col_stride).load[width=tile](0)
                var r2 = (rhs_batch + i + (j + 2) * rhs_col_stride).load[width=tile](0)
                var r3 = (rhs_batch + i + (j + 3) * rhs_col_stride).load[width=tile](0)
                var s0 = l0 + r0 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l0, r0, op)
                var s1 = l1 + r1 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l1, r1, op)
                var s2 = l2 + r2 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l2, r2, op)
                var s3 = l3 + r3 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l3, r3, op)
                var u0 = s0.shuffle[0, 4, 1, 5](s1)
                var u1 = s0.shuffle[2, 6, 3, 7](s1)
                var u2 = s2.shuffle[0, 4, 1, 5](s3)
                var u3 = s2.shuffle[2, 6, 3, 7](s3)
                var t0 = u0.shuffle[0, 1, 4, 5](u2)
                var t1 = u0.shuffle[2, 3, 6, 7](u2)
                var t2 = u1.shuffle[0, 1, 4, 5](u3)
                var t3 = u1.shuffle[2, 3, 6, 7](u3)
                (result_batch + i * result_row_stride + j).store(0, t0)
                (result_batch + (i + 1) * result_row_stride + j).store(0, t1)
                (result_batch + (i + 2) * result_row_stride + j).store(0, t2)
                (result_batch + (i + 3) * result_row_stride + j).store(0, t3)
                j += tile
            while j < cols:
                comptime for k in range(tile):
                    var lv = lhs_batch[i + k + j * lhs_col_stride]
                    var rv = rhs_batch[i + k + j * rhs_col_stride]
                    if op == BinaryOp.ADD.value:
                        result_batch[(i + k) * result_row_stride + j] = lv + rv
                    else:
                        result_batch[(i + k) * result_row_stride + j] = apply_binary_typed_vec[dtype, 1](
                            SIMD[dtype, 1](lv), SIMD[dtype, 1](rv), op
                        )[0]
                j += 1
            i += tile
        while i < rows:
            var j = 0
            while j < cols:
                var lv = lhs_batch[i + j * lhs_col_stride]
                var rv = rhs_batch[i + j * rhs_col_stride]
                if op == BinaryOp.ADD.value:
                    result_batch[i * result_row_stride + j] = lv + rv
                else:
                    result_batch[i * result_row_stride + j] = apply_binary_typed_vec[dtype, 1](
                        SIMD[dtype, 1](lv), SIMD[dtype, 1](rv), op
                    )[0]
                j += 1
            i += 1
    result.backend_code = BackendKind.FUSED.value
    return True


def maybe_binary_rank2_transposed_tile_bcast_1d[
    dtype: DType
](lhs: Array, rhs: Array, mut result: Array, op: Int, rhs_on_left: Bool) raises -> Bool:
    # 4×4 SIMD-tile path for the column-broadcast pattern: lhs is rank-2
    # F-contig (stride[0] == 1, the layout `.T` produces from a c-contig
    # rank-2 source) and rhs is rank-1 broadcasting along axis 0 of lhs
    # (rhs.shape[0] == lhs.shape[1]). Without this the generic walker
    # picks a strided inner axis with no SIMD coverage — the .T + 1D
    # pattern hits ~120× monpy/numpy on the scalar walker.
    #
    # Strategy mirrors `maybe_binary_rank2_transposed_tile`: load 4
    # contiguous SIMD vectors from lhs along the stride-1 axis 0, splat
    # the 4 corresponding rhs scalars, do 4 SIMD ops, transpose 4×4 in
    # registers, emit 4 contiguous SIMD stores into c-contig result.
    if len(lhs.shape) != 2 or len(rhs.shape) != 1:
        return False
    if lhs.strides[0] != 1:
        return False
    if rhs.shape[0] != lhs.shape[1]:
        return False
    if rhs.strides[0] != 1:
        return False
    if not is_c_contiguous(result):
        return False
    if not same_shape(lhs.shape, result.shape):
        return False
    var rows = lhs.shape[0]
    var cols = lhs.shape[1]
    if rows < 4 or cols < 4:
        return False
    var lhs_col_stride = lhs.strides[1]
    var result_row_stride = result.strides[0]
    var lhs_data = lhs.data.bitcast[Scalar[dtype]]() + lhs.offset_elems
    var rhs_data = rhs.data.bitcast[Scalar[dtype]]() + rhs.offset_elems
    var result_data = result.data.bitcast[Scalar[dtype]]() + result.offset_elems
    comptime tile = 4
    var main_rows = rows - (rows % tile)
    var main_cols = cols - (cols % tile)
    var i = 0
    while i < main_rows:
        var j = 0
        while j < main_cols:
            var l0 = (lhs_data + i + j * lhs_col_stride).load[width=tile](0)
            var l1 = (lhs_data + i + (j + 1) * lhs_col_stride).load[width=tile](0)
            var l2 = (lhs_data + i + (j + 2) * lhs_col_stride).load[width=tile](0)
            var l3 = (lhs_data + i + (j + 3) * lhs_col_stride).load[width=tile](0)
            var r0 = SIMD[dtype, tile](rhs_data[j])
            var r1 = SIMD[dtype, tile](rhs_data[j + 1])
            var r2 = SIMD[dtype, tile](rhs_data[j + 2])
            var r3 = SIMD[dtype, tile](rhs_data[j + 3])
            var s0: SIMD[dtype, tile]
            var s1: SIMD[dtype, tile]
            var s2: SIMD[dtype, tile]
            var s3: SIMD[dtype, tile]
            if rhs_on_left:
                s0 = r0 + l0 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](r0, l0, op)
                s1 = r1 + l1 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](r1, l1, op)
                s2 = r2 + l2 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](r2, l2, op)
                s3 = r3 + l3 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](r3, l3, op)
            else:
                s0 = l0 + r0 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l0, r0, op)
                s1 = l1 + r1 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l1, r1, op)
                s2 = l2 + r2 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l2, r2, op)
                s3 = l3 + r3 if op == BinaryOp.ADD.value else apply_binary_typed_vec[dtype, tile](l3, r3, op)
            var u0 = s0.shuffle[0, 4, 1, 5](s1)
            var u1 = s0.shuffle[2, 6, 3, 7](s1)
            var u2 = s2.shuffle[0, 4, 1, 5](s3)
            var u3 = s2.shuffle[2, 6, 3, 7](s3)
            var t0 = u0.shuffle[0, 1, 4, 5](u2)
            var t1 = u0.shuffle[2, 3, 6, 7](u2)
            var t2 = u1.shuffle[0, 1, 4, 5](u3)
            var t3 = u1.shuffle[2, 3, 6, 7](u3)
            (result_data + i * result_row_stride + j).store(0, t0)
            (result_data + (i + 1) * result_row_stride + j).store(0, t1)
            (result_data + (i + 2) * result_row_stride + j).store(0, t2)
            (result_data + (i + 3) * result_row_stride + j).store(0, t3)
            j += tile
        # column tail (residual cols within main rows): scalar walk
        while j < cols:
            comptime for k in range(tile):
                var lv = lhs_data[i + k + j * lhs_col_stride]
                var rv = rhs_data[j]
                var lhs_v = lv
                var rhs_v = rv
                if rhs_on_left:
                    lhs_v = rv
                    rhs_v = lv
                if op == BinaryOp.ADD.value:
                    result_data[(i + k) * result_row_stride + j] = lhs_v + rhs_v
                else:
                    result_data[(i + k) * result_row_stride + j] = apply_binary_typed_vec[dtype, 1](
                        SIMD[dtype, 1](lhs_v), SIMD[dtype, 1](rhs_v), op
                    )[0]
            j += 1
        i += tile
    # row tail (residual rows): scalar walk
    while i < rows:
        var j = 0
        while j < cols:
            var lv = lhs_data[i + j * lhs_col_stride]
            var rv = rhs_data[j]
            var lhs_v = lv
            var rhs_v = rv
            if rhs_on_left:
                lhs_v = rv
                rhs_v = lv
            if op == BinaryOp.ADD.value:
                result_data[i * result_row_stride + j] = lhs_v + rhs_v
            else:
                result_data[i * result_row_stride + j] = apply_binary_typed_vec[dtype, 1](
                    SIMD[dtype, 1](lhs_v), SIMD[dtype, 1](rhs_v), op
                )[0]
            j += 1
        i += 1
    result.backend_code = BackendKind.FUSED.value
    return True


def maybe_binary_column_broadcast_dispatch(lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    # Dispatch the column-broadcast tile kernel by dtype. Tries both
    # operand orderings (matrix on left, then 1D on left).
    if lhs.dtype_code != rhs.dtype_code or lhs.dtype_code != result.dtype_code:
        return False
    if lhs.dtype_code == ArrayDType.FLOAT32.value:
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.float32](lhs, rhs, result, op, False):
            return True
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.float32](rhs, lhs, result, op, True):
            return True
        return False
    if lhs.dtype_code == ArrayDType.FLOAT64.value:
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.float64](lhs, rhs, result, op, False):
            return True
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.float64](rhs, lhs, result, op, True):
            return True
        return False
    if lhs.dtype_code == ArrayDType.INT64.value:
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.int64](lhs, rhs, result, op, False):
            return True
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.int64](rhs, lhs, result, op, True):
            return True
        return False
    if lhs.dtype_code == ArrayDType.INT32.value:
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.int32](lhs, rhs, result, op, False):
            return True
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.int32](rhs, lhs, result, op, True):
            return True
        return False
    if lhs.dtype_code == ArrayDType.UINT64.value:
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.uint64](lhs, rhs, result, op, False):
            return True
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.uint64](rhs, lhs, result, op, True):
            return True
        return False
    if lhs.dtype_code == ArrayDType.UINT32.value:
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.uint32](lhs, rhs, result, op, False):
            return True
        if maybe_binary_rank2_transposed_tile_bcast_1d[DType.uint32](rhs, lhs, result, op, True):
            return True
        return False
    return False
