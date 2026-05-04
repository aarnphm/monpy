from std.python import PythonObject

from create import reduce as create_reduce
from create import result_dtype_for_reduction_py as create_result_dtype_for_reduction_py


def reduce(
    array_obj: PythonObject, op_obj: PythonObject
) raises -> PythonObject:
    return create_reduce(array_obj, op_obj)


def result_dtype_for_reduction_py(
    dtype_obj: PythonObject, op_obj: PythonObject
) raises -> PythonObject:
    return create_result_dtype_for_reduction_py(dtype_obj, op_obj)
