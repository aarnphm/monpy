"""Neural-network elementwise kernels exposed through `monpy.nn`."""

from std.math import exp as _exp, sqrt as _sqrt
from std.python import PythonObject

from array import Array, clone_int_list, contiguous_ptr, is_c_contiguous, make_empty_array, same_shape
from domain import ArrayDType, BackendKind


def _layer_norm_last_axis_typed[
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


def layer_norm_last_axis_ops(
    src_obj: PythonObject,
    gain_obj: PythonObject,
    bias_obj: PythonObject,
    eps_obj: PythonObject,
) raises -> PythonObject:
    var src = src_obj.downcast_value_ptr[Array]()
    var gain = gain_obj.downcast_value_ptr[Array]()
    var bias = bias_obj.downcast_value_ptr[Array]()
    var eps = Float64(py=eps_obj)
    if (
        len(src[].shape) != 2
        or len(gain[].shape) != 1
        or len(bias[].shape) != 1
        or gain[].shape[0] != src[].shape[1]
        or bias[].shape[0] != src[].shape[1]
        or src[].dtype_code != gain[].dtype_code
        or src[].dtype_code != bias[].dtype_code
        or not is_c_contiguous(src[])
        or not is_c_contiguous(gain[])
        or not is_c_contiguous(bias[])
    ):
        raise Error(
            "layer_norm_last_axis requires rank-2 input, rank-1 gain/bias, matching float dtype, and C-contiguous"
            " arrays"
        )
    var result = make_empty_array(src[].dtype_code, clone_int_list(src[].shape))
    if src[].dtype_code == ArrayDType.FLOAT32.value:
        _layer_norm_last_axis_typed[DType.float32](src[], gain[], bias[], result, eps)
        return PythonObject(alloc=result^)
    if src[].dtype_code == ArrayDType.FLOAT64.value:
        _layer_norm_last_axis_typed[DType.float64](src[], gain[], bias[], result, eps)
        return PythonObject(alloc=result^)
    raise Error("layer_norm_last_axis requires float32 or float64 inputs")


def _softmax_last_axis_typed[dt: DType](src: Array, mut result: Array) raises where dt.is_floating_point():
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


def softmax_last_axis_ops(src_obj: PythonObject) raises -> PythonObject:
    var src = src_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 2 or not is_c_contiguous(src[]):
        raise Error("softmax_last_axis requires a rank-2 C-contiguous array")
    var result = make_empty_array(src[].dtype_code, clone_int_list(src[].shape))
    if src[].dtype_code == ArrayDType.FLOAT32.value:
        _softmax_last_axis_typed[DType.float32](src[], result)
        return PythonObject(alloc=result^)
    if src[].dtype_code == ArrayDType.FLOAT64.value:
        _softmax_last_axis_typed[DType.float64](src[], result)
        return PythonObject(alloc=result^)
    raise Error("softmax_last_axis requires float32 or float64 input")


def _scaled_masked_softmax_last_axis_typed[
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


def scaled_masked_softmax_last_axis_ops(
    src_obj: PythonObject,
    mask_obj: PythonObject,
    scale_obj: PythonObject,
    fill_obj: PythonObject,
) raises -> PythonObject:
    var src = src_obj.downcast_value_ptr[Array]()
    var mask = mask_obj.downcast_value_ptr[Array]()
    var scale = Float64(py=scale_obj)
    var fill = Float64(py=fill_obj)
    if (
        len(src[].shape) != 2
        or mask[].dtype_code != ArrayDType.BOOL.value
        or not same_shape(src[].shape, mask[].shape)
        or not is_c_contiguous(src[])
        or not is_c_contiguous(mask[])
    ):
        raise Error(
            "scaled_masked_softmax_last_axis requires rank-2 C-contiguous input and bool mask with matching shape"
        )
    var result = make_empty_array(src[].dtype_code, clone_int_list(src[].shape))
    if src[].dtype_code == ArrayDType.FLOAT32.value:
        _scaled_masked_softmax_last_axis_typed[DType.float32](src[], mask[], result, scale, fill)
        return PythonObject(alloc=result^)
    if src[].dtype_code == ArrayDType.FLOAT64.value:
        _scaled_masked_softmax_last_axis_typed[DType.float64](src[], mask[], result, scale, fill)
        return PythonObject(alloc=result^)
    raise Error("scaled_masked_softmax_last_axis requires float32 or float64 input")
