"""Neural-network row kernels used by `create/ops/nn.mojo`."""

from std.math import exp as _exp, sqrt as _sqrt

from array import Array, contiguous_ptr
from domain import BackendKind


def layer_norm_last_axis_typed[
    dt: DType
](src: Array, gain: Array, bias: Array, mut result: Array, eps: Float64) raises where dt.is_floating_point():
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_ptr = contiguous_ptr[dt](src)
    var gain_ptr = contiguous_ptr[dt](gain)
    var bias_ptr = contiguous_ptr[dt](bias)
    var out_ptr = contiguous_ptr[dt](result)
    for row in range(rows):
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
    result.backend_code = BackendKind.FUSED.value


def softmax_last_axis_typed[dt: DType](src: Array, mut result: Array) raises where dt.is_floating_point():
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_ptr = contiguous_ptr[dt](src)
    var out_ptr = contiguous_ptr[dt](result)
    for row in range(rows):
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
    result.backend_code = BackendKind.FUSED.value


def scaled_masked_softmax_last_axis_typed[
    dt: DType
](src: Array, mask: Array, mut result: Array, scale: Float64, fill: Float64,) raises where dt.is_floating_point():
    var rows = src.shape[0]
    var cols = src.shape[1]
    var src_ptr = contiguous_ptr[dt](src)
    var mask_ptr = mask.data + mask.offset_elems
    var out_ptr = contiguous_ptr[dt](result)
    for row in range(rows):
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
    result.backend_code = BackendKind.FUSED.value
