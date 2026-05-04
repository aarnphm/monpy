from std.python import PythonObject

from create import matmul as create_matmul


def matmul(
    lhs_obj: PythonObject, rhs_obj: PythonObject
) raises -> PythonObject:
    return create_matmul(lhs_obj, rhs_obj)
