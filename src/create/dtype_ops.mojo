"""Python-bridge thin wrappers for dtype helpers.

These are CPython ABI shims — each one is a one-liner that unboxes a
`PythonObject`, calls into the Mojo-side dtype machinery (`array.result_dtype_for_*`,
`domain.dtype_*`), and reboxes the result. Keeping them in their own module keeps
`create/__init__.mojo` focused on kernel-dispatching `_ops` functions.
"""

from std.python import PythonObject

from array import (
    Array,
    cast_copy_array,
    result_dtype_for_binary,
    result_dtype_for_reduction,
    result_dtype_for_unary,
)
from domain import (
    BinaryOp,
    CastingRule,
    ReduceOp,
    dtype_alignment,
    dtype_can_cast,
    dtype_item_size,
    dtype_kind_code,
    dtype_promote_types,
)


def astype_ops(array_obj: PythonObject, dtype_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var dtype_code = Int(py=dtype_obj)
    var result = cast_copy_array(src[], dtype_code)
    return PythonObject(alloc=result^)


def result_dtype_for_unary_py_ops(dtype_obj: PythonObject) raises -> PythonObject:
    return PythonObject(result_dtype_for_unary(Int(py=dtype_obj)))


def result_dtype_for_binary_py_ops(
    lhs_dtype_obj: PythonObject,
    rhs_dtype_obj: PythonObject,
    op_obj: PythonObject,
) raises -> PythonObject:
    var op = BinaryOp.from_int(Int(py=op_obj)).value
    return PythonObject(result_dtype_for_binary(Int(py=lhs_dtype_obj), Int(py=rhs_dtype_obj), op))


def result_dtype_for_reduction_py_ops(dtype_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    var op = ReduceOp.from_int(Int(py=op_obj)).value
    return PythonObject(result_dtype_for_reduction(Int(py=dtype_obj), op))


def dtype_item_size_py_ops(dtype_obj: PythonObject) raises -> PythonObject:
    return PythonObject(dtype_item_size(Int(py=dtype_obj)))


def dtype_alignment_py_ops(dtype_obj: PythonObject) raises -> PythonObject:
    return PythonObject(dtype_alignment(Int(py=dtype_obj)))


def dtype_kind_code_py_ops(dtype_obj: PythonObject) raises -> PythonObject:
    return PythonObject(dtype_kind_code(Int(py=dtype_obj)))


def dtype_promote_types_py_ops(lhs_dtype_obj: PythonObject, rhs_dtype_obj: PythonObject) raises -> PythonObject:
    return PythonObject(dtype_promote_types(Int(py=lhs_dtype_obj), Int(py=rhs_dtype_obj)))


def dtype_can_cast_py_ops(
    from_dtype_obj: PythonObject,
    to_dtype_obj: PythonObject,
    casting_obj: PythonObject,
) raises -> PythonObject:
    var casting = CastingRule.from_int(Int(py=casting_obj)).value
    return PythonObject(dtype_can_cast(Int(py=from_dtype_obj), Int(py=to_dtype_obj), casting))
