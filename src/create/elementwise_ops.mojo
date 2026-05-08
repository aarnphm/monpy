"""Elementwise `_ops` Python-bridge entry points.

Hosts the elementwise compute surface — unary preserve, comparison, logical,
predicate. Each one walks via `LayoutIter` (single-source) or `MultiLayoutIter`
(broadcast pair) to amortize the divmod across iteration; typed contiguous
fast paths route through `maybe_*_contiguous` upstream in elementwise/.

Bool-output ops (compare/logical/predicate) write to a fresh BOOL array;
unary_preserve resolves output dtype via `result_dtype_for_unary_preserve`
(input dtype with bool→int64 promotion).
"""

from std.math import isinf, isnan
from std.python import PythonObject

from array import (
    Array,
    as_broadcast_layout,
    as_layout,
    broadcast_shape,
    clone_int_list,
    get_physical_as_f64,
    item_size,
    make_empty_array,
    result_dtype_for_unary_preserve,
    same_shape,
    set_physical_from_f64,
)
from cute.iter import LayoutIter, MultiLayoutIter
from cute.layout import Layout
from domain import ArrayDType, CompareOp, LogicalOp, PredicateOp, UnaryOp
from elementwise import apply_unary_f64, maybe_unary_preserve_contiguous


def unary_preserve_ops(array_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    """Preserve-dtype unary ops (negate/abs/square/positive/floor/ceil/ trunc/rint/logical_not).
    Output dtype = input dtype (with bool→int64 promotion).
    Typed-vec contig fast path for f32/f64/i32/i64/u32/u64; other paths fall through to the f64 round-trip.
    """
    var src = array_obj.downcast_value_ptr[Array]()
    var op = UnaryOp.from_int(Int(py=op_obj)).value
    var shape = clone_int_list(src[].shape)
    var dtype_code = result_dtype_for_unary_preserve(src[].dtype_code)
    var result = make_empty_array(dtype_code, shape^)
    if maybe_unary_preserve_contiguous(src[], result, op):
        return PythonObject(alloc=result^)
    var src_layout = as_layout(src[])
    var dst_layout = as_layout(result)
    var src_item = item_size(src[].dtype_code)
    var dst_item = item_size(result.dtype_code)
    var src_iter = LayoutIter(src_layout, src_item, src[].offset_elems * src_item)
    var dst_iter = LayoutIter(dst_layout, dst_item, result.offset_elems * dst_item)
    while src_iter.has_next():
        var value = get_physical_as_f64(src[], src_iter.element_index())
        var output = apply_unary_f64(value, op)
        set_physical_from_f64(result, dst_iter.element_index(), output)
        src_iter.step()
        dst_iter.step()
    return PythonObject(alloc=result^)


def compare_ops(
    lhs_obj: PythonObject,
    rhs_obj: PythonObject,
    op_obj: PythonObject,
) raises -> PythonObject:
    """Elementwise comparison; returns a bool array. Operands broadcast.
    Walks via MultiLayoutIter so the broadcast divmod amortizes."""
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
    var op = CompareOp.from_int(Int(py=op_obj)).value
    var same = same_shape(lhs[].shape, rhs[].shape)
    var shape: List[Int]
    if same:
        shape = clone_int_list(lhs[].shape)
    else:
        shape = broadcast_shape(lhs[], rhs[])
    var result = make_empty_array(ArrayDType.BOOL.value, shape^)
    var lhs_layout = as_broadcast_layout(lhs[], result.shape)
    var rhs_layout = as_broadcast_layout(rhs[], result.shape)
    var out_layout = as_layout(result)
    var item_lhs = item_size(lhs[].dtype_code)
    var item_rhs = item_size(rhs[].dtype_code)
    var item_out = item_size(result.dtype_code)
    var operand_layouts = List[Layout]()
    operand_layouts.append(lhs_layout^)
    operand_layouts.append(rhs_layout^)
    operand_layouts.append(out_layout^)
    var item_sizes = List[Int]()
    item_sizes.append(item_lhs)
    item_sizes.append(item_rhs)
    item_sizes.append(item_out)
    var base_offsets = List[Int]()
    base_offsets.append(lhs[].offset_elems * item_lhs)
    base_offsets.append(rhs[].offset_elems * item_rhs)
    base_offsets.append(result.offset_elems * item_out)
    var iter = MultiLayoutIter(result.shape, operand_layouts^, item_sizes^, base_offsets^)
    while iter.has_next():
        var lval = get_physical_as_f64(lhs[], iter.element_index(0))
        var rval = get_physical_as_f64(rhs[], iter.element_index(1))
        var output: Bool
        if op == CompareOp.EQ.value:
            output = lval == rval
        elif op == CompareOp.NE.value:
            output = lval != rval
        elif op == CompareOp.LT.value:
            output = lval < rval
        elif op == CompareOp.LE.value:
            output = lval <= rval
        elif op == CompareOp.GT.value:
            output = lval > rval
        elif op == CompareOp.GE.value:
            output = lval >= rval
        else:
            raise Error("unknown comparison op")
        set_physical_from_f64(result, iter.element_index(2), 1.0 if output else 0.0)
        iter.step()
    return PythonObject(alloc=result^)


def logical_ops(
    lhs_obj: PythonObject,
    rhs_obj: PythonObject,
    op_obj: PythonObject,
) raises -> PythonObject:
    """Elementwise logical_and / or / xor. Operates on truthiness of any
    numeric input; result is bool. Walks via MultiLayoutIter."""
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
    var op = LogicalOp.from_int(Int(py=op_obj)).value
    var same = same_shape(lhs[].shape, rhs[].shape)
    var shape: List[Int]
    if same:
        shape = clone_int_list(lhs[].shape)
    else:
        shape = broadcast_shape(lhs[], rhs[])
    var result = make_empty_array(ArrayDType.BOOL.value, shape^)
    var lhs_layout = as_broadcast_layout(lhs[], result.shape)
    var rhs_layout = as_broadcast_layout(rhs[], result.shape)
    var out_layout = as_layout(result)
    var item_lhs = item_size(lhs[].dtype_code)
    var item_rhs = item_size(rhs[].dtype_code)
    var item_out = item_size(result.dtype_code)
    var operand_layouts = List[Layout]()
    operand_layouts.append(lhs_layout^)
    operand_layouts.append(rhs_layout^)
    operand_layouts.append(out_layout^)
    var item_sizes = List[Int]()
    item_sizes.append(item_lhs)
    item_sizes.append(item_rhs)
    item_sizes.append(item_out)
    var base_offsets = List[Int]()
    base_offsets.append(lhs[].offset_elems * item_lhs)
    base_offsets.append(rhs[].offset_elems * item_rhs)
    base_offsets.append(result.offset_elems * item_out)
    var iter = MultiLayoutIter(result.shape, operand_layouts^, item_sizes^, base_offsets^)
    while iter.has_next():
        var l_truthy = get_physical_as_f64(lhs[], iter.element_index(0)) != 0.0
        var r_truthy = get_physical_as_f64(rhs[], iter.element_index(1)) != 0.0
        var output: Bool
        if op == LogicalOp.AND.value:
            output = l_truthy and r_truthy
        elif op == LogicalOp.OR.value:
            output = l_truthy or r_truthy
        elif op == LogicalOp.XOR.value:
            output = l_truthy != r_truthy
        else:
            raise Error("unknown logical op")
        set_physical_from_f64(result, iter.element_index(2), 1.0 if output else 0.0)
        iter.step()
    return PythonObject(alloc=result^)


def predicate_ops(array_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    """Unary predicate (isnan / isinf / isfinite / signbit). Returns
    bool. Walks via LayoutIter so the divmod amortizes across the
    iteration."""
    var src = array_obj.downcast_value_ptr[Array]()
    var op = PredicateOp.from_int(Int(py=op_obj)).value
    var shape = clone_int_list(src[].shape)
    var result = make_empty_array(ArrayDType.BOOL.value, shape^)
    var src_layout = as_layout(src[])
    var dst_layout = as_layout(result)
    var src_item = item_size(src[].dtype_code)
    var dst_item = item_size(result.dtype_code)
    var src_iter = LayoutIter(src_layout, src_item, src[].offset_elems * src_item)
    var dst_iter = LayoutIter(dst_layout, dst_item, result.offset_elems * dst_item)
    while src_iter.has_next():
        var value = get_physical_as_f64(src[], src_iter.element_index())
        var output: Bool
        if op == PredicateOp.ISNAN.value:
            output = isnan(value)
        elif op == PredicateOp.ISINF.value:
            output = isinf(value)
        elif op == PredicateOp.ISFINITE.value:
            output = not (isnan(value) or isinf(value))
        elif op == PredicateOp.SIGNBIT.value:
            # numpy.signbit returns True for -0.0, so use bitcast.
            output = value < 0.0
            if value == 0.0:
                # negative-zero → True via bitcast.
                var bits = SIMD[DType.float64, 1](value).cast[DType.uint64]()[0]
                output = (bits >> 63) != 0
        else:
            raise Error("unknown predicate op")
        set_physical_from_f64(result, dst_iter.element_index(), 1.0 if output else 0.0)
        src_iter.step()
        dst_iter.step()
    return PythonObject(alloc=result^)
