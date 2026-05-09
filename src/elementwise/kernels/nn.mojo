"""Neural-network row kernels used by `create/ops/nn.mojo`.

Each kernel iterates rows independently — softmax does max+exp+normalize
per row, layer-norm does mean+variance+normalize per row. Per-row work
has scalar-only scratch (no buffer allocation), so rows can be sliced
across worker threads without contention.

Parallel path: `worker_count_for_row_elements(rows, cols, ...)` picks a
shape-budgeted worker count, then partitions the row range via
`sync_parallelize`. Each worker handles a contiguous slice of rows — same
per-row body as the serial path, just stitched. The `result.backend_code = ...`
writes stay outside the parallel region so worker bodies remain pure.
"""

from std.algorithm import sync_parallelize
from std.math import ceildiv, exp as _exp, sqrt as _sqrt

from array import Array, contiguous_ptr
from domain import BackendKind

from elementwise.kernels.parallel import ROW_HEAVY_GRAIN_ELEMS, worker_count_for_row_elements


def _softmax_last_axis_f32(src: Array, mut result: Array) raises:
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_ptr = contiguous_ptr[DType.float32](src)
    var out_ptr = contiguous_ptr[DType.float32](result)

    @parameter
    def process_row_range(row_start: Int, row_end: Int):
        for row in range(row_start, row_end):
            var base = row * cols
            var row_max = src_ptr[base]
            for col in range(1, cols):
                var value = src_ptr[base + col]
                if value > row_max:
                    row_max = value
            var denom = Float32(0.0)
            for col in range(cols):
                var weight = _exp(src_ptr[base + col] - row_max)
                denom += weight
                out_ptr[base + col] = weight
            var inv_denom = Float32(1.0) / denom
            for col in range(cols):
                out_ptr[base + col] *= inv_denom

    var nw = worker_count_for_row_elements(rows, cols, ROW_HEAVY_GRAIN_ELEMS)
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


def _scaled_masked_softmax_last_axis_f32(
    src: Array, mask: Array, mut result: Array, scale: Float64, fill: Float64
) raises:
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_ptr = contiguous_ptr[DType.float32](src)
    var mask_ptr = mask.data + mask.offset_elems
    var out_ptr = contiguous_ptr[DType.float32](result)
    var scale32 = Float32(scale)
    var fill32 = Float32(fill)

    @parameter
    def process_row_range(row_start: Int, row_end: Int):
        for row in range(row_start, row_end):
            var base = row * cols
            var first = fill32
            if mask_ptr[base] == UInt8(0):
                first = src_ptr[base] * scale32
            var row_max = first
            for col in range(1, cols):
                var index = base + col
                var value = fill32
                if mask_ptr[index] == UInt8(0):
                    value = src_ptr[index] * scale32
                if value > row_max:
                    row_max = value
            var denom = Float32(0.0)
            for col in range(cols):
                var index = base + col
                var value = fill32
                if mask_ptr[index] == UInt8(0):
                    value = src_ptr[index] * scale32
                var weight = _exp(value - row_max)
                denom += weight
                out_ptr[index] = weight
            var inv_denom = Float32(1.0) / denom
            for col in range(cols):
                var index = base + col
                out_ptr[index] *= inv_denom

    var nw = worker_count_for_row_elements(rows, cols, ROW_HEAVY_GRAIN_ELEMS)
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


def layer_norm_last_axis_typed[
    dt: DType
](src: Array, gain: Array, bias: Array, mut result: Array, eps: Float64) raises where dt.is_floating_point():
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_ptr = contiguous_ptr[dt](src)
    var gain_ptr = contiguous_ptr[dt](gain)
    var bias_ptr = contiguous_ptr[dt](bias)
    var out_ptr = contiguous_ptr[dt](result)

    @parameter
    def process_row_range(row_start: Int, row_end: Int):
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
                var value = (Float64(src_ptr[base + col]) - mean) * inv_std
                value = value * Float64(gain_ptr[col]) + Float64(bias_ptr[col])
                out_ptr[base + col] = Scalar[dt](value)

    var nw = worker_count_for_row_elements(rows, cols, ROW_HEAVY_GRAIN_ELEMS)
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


def softmax_last_axis_typed[dt: DType](src: Array, mut result: Array) raises where dt.is_floating_point():
    comptime if dt == DType.float32:
        _softmax_last_axis_f32(src, result)
        result.backend_code = BackendKind.FUSED.value
        return
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_ptr = contiguous_ptr[dt](src)
    var out_ptr = contiguous_ptr[dt](result)

    @parameter
    def process_row_range(row_start: Int, row_end: Int):
        for row in range(row_start, row_end):
            var base = row * cols
            var row_max = Float64(src_ptr[base])
            for col in range(1, cols):
                var value = Float64(src_ptr[base + col])
                if value > row_max:
                    row_max = value
            var denom = 0.0
            for col in range(cols):
                var weight = _exp(Float64(src_ptr[base + col]) - row_max)
                denom += weight
                out_ptr[base + col] = Scalar[dt](weight)
            var inv_denom = 1.0 / denom
            for col in range(cols):
                out_ptr[base + col] = Scalar[dt](Float64(out_ptr[base + col]) * inv_denom)

    var nw = worker_count_for_row_elements(rows, cols, ROW_HEAVY_GRAIN_ELEMS)
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


def scaled_masked_softmax_last_axis_typed[
    dt: DType
](src: Array, mask: Array, mut result: Array, scale: Float64, fill: Float64,) raises where dt.is_floating_point():
    comptime if dt == DType.float32:
        _scaled_masked_softmax_last_axis_f32(src, mask, result, scale, fill)
        result.backend_code = BackendKind.FUSED.value
        return
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_ptr = contiguous_ptr[dt](src)
    var mask_ptr = mask.data + mask.offset_elems
    var out_ptr = contiguous_ptr[dt](result)

    @parameter
    def process_row_range(row_start: Int, row_end: Int):
        for row in range(row_start, row_end):
            var base = row * cols
            var first = fill if mask_ptr[base] != UInt8(0) else Float64(src_ptr[base]) * scale
            var row_max = first
            for col in range(1, cols):
                var index = base + col
                var value = fill if mask_ptr[index] != UInt8(0) else Float64(src_ptr[index]) * scale
                if value > row_max:
                    row_max = value
            var denom = 0.0
            for col in range(cols):
                var index = base + col
                var value = fill if mask_ptr[index] != UInt8(0) else Float64(src_ptr[index]) * scale
                var weight = _exp(value - row_max)
                denom += weight
                out_ptr[index] = Scalar[dt](weight)
            var inv_denom = 1.0 / denom
            for col in range(cols):
                var index = base + col
                out_ptr[index] = Scalar[dt](Float64(out_ptr[index]) * inv_denom)

    var nw = worker_count_for_row_elements(rows, cols, ROW_HEAVY_GRAIN_ELEMS)
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
