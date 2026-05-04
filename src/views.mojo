from std.python import PythonObject

from create import astype as create_astype
from create import broadcast_to as create_broadcast_to
from create import copyto as create_copyto
from create import diagonal as create_diagonal
from create import fill as create_fill
from create import materialize_c_contiguous as create_materialize_c_contiguous
from create import reshape as create_reshape
from create import slice as create_slice
from create import slice_1d as create_slice_1d
from create import trace as create_trace
from create import transpose as create_transpose


def reshape(
    array_obj: PythonObject, shape_obj: PythonObject
) raises -> PythonObject:
    return create_reshape(array_obj, shape_obj)


def transpose(
    array_obj: PythonObject, axes_obj: PythonObject
) raises -> PythonObject:
    return create_transpose(array_obj, axes_obj)


def slice(
    array_obj: PythonObject,
    starts_obj: PythonObject,
    stops_obj: PythonObject,
    steps_obj: PythonObject,
    drops_obj: PythonObject,
) raises -> PythonObject:
    return create_slice(
        array_obj, starts_obj, stops_obj, steps_obj, drops_obj
    )


def broadcast_to(
    array_obj: PythonObject, shape_obj: PythonObject
) raises -> PythonObject:
    return create_broadcast_to(array_obj, shape_obj)


def astype(
    array_obj: PythonObject, dtype_obj: PythonObject
) raises -> PythonObject:
    return create_astype(array_obj, dtype_obj)


def materialize_c_contiguous(
    array_obj: PythonObject,
) raises -> PythonObject:
    return create_materialize_c_contiguous(array_obj)


def diagonal(
    array_obj: PythonObject,
    offset_obj: PythonObject,
    axis1_obj: PythonObject,
    axis2_obj: PythonObject,
) raises -> PythonObject:
    return create_diagonal(array_obj, offset_obj, axis1_obj, axis2_obj)


def trace(
    array_obj: PythonObject,
    offset_obj: PythonObject,
    axis1_obj: PythonObject,
    axis2_obj: PythonObject,
    dtype_obj: PythonObject,
) raises -> PythonObject:
    return create_trace(
        array_obj, offset_obj, axis1_obj, axis2_obj, dtype_obj
    )


def fill(
    array_obj: PythonObject, value_obj: PythonObject
) raises -> PythonObject:
    return create_fill(array_obj, value_obj)


def copyto(
    dst_obj: PythonObject, src_obj: PythonObject
) raises -> PythonObject:
    return create_copyto(dst_obj, src_obj)


def slice_1d(
    array_obj: PythonObject,
    start_obj: PythonObject,
    stop_obj: PythonObject,
    step_obj: PythonObject,
) raises -> PythonObject:
    return create_slice_1d(array_obj, start_obj, stop_obj, step_obj)
