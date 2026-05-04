from std.python import PythonObject

from create import det as create_det
from create import inv as create_inv
from create import solve as create_solve


def solve(
    a_obj: PythonObject, b_obj: PythonObject
) raises -> PythonObject:
    return create_solve(a_obj, b_obj)


def inv(array_obj: PythonObject) raises -> PythonObject:
    return create_inv(array_obj)


def det(array_obj: PythonObject) raises -> PythonObject:
    return create_det(array_obj)
