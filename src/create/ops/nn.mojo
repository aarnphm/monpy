"""Neural-network PythonObject bridge ops."""

from std.python import PythonObject

from array import Array, clone_int_list, is_c_contiguous, make_empty_array, same_shape
from domain import ArrayDType
from elementwise import (
    layer_norm_last_axis_typed,
    scaled_masked_softmax_last_axis_typed,
    softmax_last_axis_typed,
)


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
        layer_norm_last_axis_typed[DType.float32](src[], gain[], bias[], result, eps)
        return PythonObject(alloc=result^)
    if src[].dtype_code == ArrayDType.FLOAT64.value:
        layer_norm_last_axis_typed[DType.float64](src[], gain[], bias[], result, eps)
        return PythonObject(alloc=result^)
    raise Error("layer_norm_last_axis requires float32 or float64 inputs")


def softmax_last_axis_ops(src_obj: PythonObject) raises -> PythonObject:
    var src = src_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 2 or not is_c_contiguous(src[]):
        raise Error("softmax_last_axis requires a rank-2 C-contiguous array")
    var result = make_empty_array(src[].dtype_code, clone_int_list(src[].shape))
    if src[].dtype_code == ArrayDType.FLOAT32.value:
        softmax_last_axis_typed[DType.float32](src[], result)
        return PythonObject(alloc=result^)
    if src[].dtype_code == ArrayDType.FLOAT64.value:
        softmax_last_axis_typed[DType.float64](src[], result)
        return PythonObject(alloc=result^)
    raise Error("softmax_last_axis requires float32 or float64 input")


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
        scaled_masked_softmax_last_axis_typed[DType.float32](src[], mask[], result, scale, fill)
        return PythonObject(alloc=result^)
    if src[].dtype_code == ArrayDType.FLOAT64.value:
        scaled_masked_softmax_last_axis_typed[DType.float64](src[], mask[], result, scale, fill)
        return PythonObject(alloc=result^)
    raise Error("scaled_masked_softmax_last_axis requires float32 or float64 input")
