"""Reduction PythonObject bridge ops.

Hosts whole-array `reduce_ops` and axis-aware `reduce_axis_ops`. Both go
through three speed tiers: typed contiguous SIMD path
(`maybe_reduce_contiguous`), typed strided path
(`maybe_reduce_strided_typed`), and a generic per-element f64 round-trip
fallback that walks via LayoutIter / coord-iteration.

Empty-array semantics follow numpy: sum/all=identity, prod/any=identity,
min/max raise on whole-array reduce but produce identity when reducing
along a 0-length axis with keepdims.
"""

from std.python import PythonObject

from array import (
    Array,
    as_layout,
    clone_int_list,
    get_physical_as_f64,
    item_size,
    make_empty_array,
    result_dtype_for_reduction,
    set_logical_from_f64,
    set_logical_from_i64,
)
from cute.iter import LayoutIter
from domain import ArrayDType, ReduceOp
from elementwise import (
    maybe_argmax_contiguous,
    maybe_reduce_axis_last_contiguous,
    maybe_reduce_contiguous,
    maybe_reduce_strided_typed,
)


def _reduce_strided_iter(src: Array) raises -> LayoutIter:
    """LayoutIter wrapping `src` for strided whole-array reductions.
    Caller drives `step()` and reads `element_index()`."""
    var src_layout = as_layout(src)
    var src_item = item_size(src.dtype_code)
    return LayoutIter(src_layout, src_item, src.offset_elems * src_item)


def reduce_ops(array_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var op = ReduceOp.from_int(Int(py=op_obj)).value
    var shape = List[Int]()
    if op == ReduceOp.ARGMAX.value or op == ReduceOp.ARGMIN.value:
        var result = make_empty_array(ArrayDType.INT64.value, shape^)
        if src[].size_value == 0:
            raise Error("argmax/argmin cannot reduce an empty array")
        if op == ReduceOp.ARGMAX.value and maybe_argmax_contiguous(src[], result):
            return PythonObject(alloc=result^)
        var iter = _reduce_strided_iter(src[])
        var best_index = 0
        var best_value = get_physical_as_f64(src[], iter.element_index())
        iter.step()
        var i = 1
        while iter.has_next():
            var value = get_physical_as_f64(src[], iter.element_index())
            if op == ReduceOp.ARGMAX.value:
                if value > best_value:
                    best_value = value
                    best_index = i
            else:
                if value < best_value:
                    best_value = value
                    best_index = i
            iter.step()
            i += 1
        set_logical_from_i64(result, 0, Int64(best_index))
        return PythonObject(alloc=result^)
    if op == ReduceOp.ALL.value or op == ReduceOp.ANY.value:
        var result = make_empty_array(ArrayDType.BOOL.value, shape^)
        if src[].size_value == 0:
            # numpy: all() of empty → True; any() of empty → False.
            var v: Float64 = 1.0 if op == ReduceOp.ALL.value else 0.0
            set_logical_from_f64(result, 0, v)
            return PythonObject(alloc=result^)
        var iter = _reduce_strided_iter(src[])
        if op == ReduceOp.ALL.value:
            while iter.has_next():
                if get_physical_as_f64(src[], iter.element_index()) == 0.0:
                    set_logical_from_f64(result, 0, 0.0)
                    return PythonObject(alloc=result^)
                iter.step()
            set_logical_from_f64(result, 0, 1.0)
            return PythonObject(alloc=result^)
        # ReduceOp.ANY.value
        while iter.has_next():
            if get_physical_as_f64(src[], iter.element_index()) != 0.0:
                set_logical_from_f64(result, 0, 1.0)
                return PythonObject(alloc=result^)
            iter.step()
        set_logical_from_f64(result, 0, 0.0)
        return PythonObject(alloc=result^)
    var result_dtype = result_dtype_for_reduction(src[].dtype_code, op)
    var result = make_empty_array(result_dtype, shape^)
    if src[].size_value == 0:
        # numpy: sum of empty → 0; prod of empty → 1; min/max raise.
        if op == ReduceOp.SUM.value or op == ReduceOp.MEAN.value:
            set_logical_from_f64(result, 0, 0.0)
            return PythonObject(alloc=result^)
        if op == ReduceOp.PROD.value:
            set_logical_from_f64(result, 0, 1.0)
            return PythonObject(alloc=result^)
        raise Error("cannot reduce an empty array")
    if maybe_reduce_contiguous(src[], result, op):
        return PythonObject(alloc=result^)
    if maybe_reduce_strided_typed(src[], result, op):
        return PythonObject(alloc=result^)
    var iter = _reduce_strided_iter(src[])
    var acc: Float64
    if op == ReduceOp.SUM.value or op == ReduceOp.MEAN.value:
        acc = 0.0
        while iter.has_next():
            acc += get_physical_as_f64(src[], iter.element_index())
            iter.step()
        if op == ReduceOp.MEAN.value:
            acc = acc / Float64(src[].size_value)
    elif op == ReduceOp.PROD.value:
        acc = 1.0
        while iter.has_next():
            acc *= get_physical_as_f64(src[], iter.element_index())
            iter.step()
    elif op == ReduceOp.MIN.value:
        acc = get_physical_as_f64(src[], iter.element_index())
        iter.step()
        while iter.has_next():
            var value = get_physical_as_f64(src[], iter.element_index())
            if value < acc:
                acc = value
            iter.step()
    elif op == ReduceOp.MAX.value:
        acc = get_physical_as_f64(src[], iter.element_index())
        iter.step()
        while iter.has_next():
            var value = get_physical_as_f64(src[], iter.element_index())
            if value > acc:
                acc = value
            iter.step()
    else:
        raise Error("unknown reduction op")
    set_logical_from_f64(result, 0, acc)
    return PythonObject(alloc=result^)


def reduce_axis_ops(
    array_obj: PythonObject,
    op_obj: PythonObject,
    axis_obj: PythonObject,
    keepdims_obj: PythonObject,
) raises -> PythonObject:
    """Axis-aware reduction. `axis_obj` is a Python tuple/list of ints;
    `keepdims_obj` is a Python bool. Output shape collapses or keeps the
    reduced axes. Strided per-element f64 round-trip path; SIMD-friendly
    kernels can layer in via LayoutIter once the API stabilises."""
    var src = array_obj.downcast_value_ptr[Array]()
    var op = ReduceOp.from_int(Int(py=op_obj)).value
    var keepdims = Bool(py=keepdims_obj)
    var ndim = len(src[].shape)
    # Decode axes.
    var axes = List[Int]()
    for i in range(len(axis_obj)):
        var ax = Int(py=axis_obj[i])
        if ax < 0:
            ax += ndim
        if ax < 0 or ax >= ndim:
            raise Error("reduce axis out of range")
        axes.append(ax)
    # Build result shape.
    var keep_mask = List[Bool]()
    for _ in range(ndim):
        keep_mask.append(True)
    for i in range(len(axes)):
        keep_mask[axes[i]] = False
    var out_shape = List[Int]()
    var keep_axes = List[Int]()
    for d in range(ndim):
        if keep_mask[d]:
            out_shape.append(src[].shape[d])
            keep_axes.append(d)
        elif keepdims:
            out_shape.append(1)
            keep_axes.append(d)
    var result_dtype: Int
    if op == ReduceOp.ARGMAX.value or op == ReduceOp.ARGMIN.value:
        result_dtype = ArrayDType.INT64.value
    elif op == ReduceOp.ALL.value or op == ReduceOp.ANY.value:
        result_dtype = ArrayDType.BOOL.value
    else:
        result_dtype = result_dtype_for_reduction(src[].dtype_code, op)
    var result = make_empty_array(result_dtype, clone_int_list(out_shape))
    # Compute reduce-axis size + strides.
    var reduce_axes = List[Int]()
    for d in range(ndim):
        if not keep_mask[d]:
            reduce_axes.append(d)
    var reduce_size = 1
    for i in range(len(reduce_axes)):
        reduce_size *= src[].shape[reduce_axes[i]]
    if reduce_size == 0:
        # Numpy semantics: sum/prod of empty → identity; min/max raise.
        if op == ReduceOp.SUM.value or op == ReduceOp.MEAN.value:
            for j in range(result.size_value):
                set_logical_from_f64(result, j, 0.0)
            return PythonObject(alloc=result^)
        if op == ReduceOp.PROD.value:
            for j in range(result.size_value):
                set_logical_from_f64(result, j, 1.0)
            return PythonObject(alloc=result^)
        if op == ReduceOp.ALL.value:
            for j in range(result.size_value):
                set_logical_from_f64(result, j, 1.0)
            return PythonObject(alloc=result^)
        if op == ReduceOp.ANY.value:
            for j in range(result.size_value):
                set_logical_from_f64(result, j, 0.0)
            return PythonObject(alloc=result^)
        raise Error("cannot reduce empty axis with this op")
    # Iterate output positions; for each, walk the reduce axes.
    var out_size = result.size_value
    if out_size == 0:
        return PythonObject(alloc=result^)
    if len(reduce_axes) == 1 and reduce_axes[0] == ndim - 1:
        if maybe_reduce_axis_last_contiguous(src[], result, op):
            return PythonObject(alloc=result^)
    # Build coord helper for output index → src physical offset of the
    # first element of the reduced subspace.
    var keep_strides = List[Int]()
    for i in range(len(keep_axes)):
        keep_strides.append(src[].strides[keep_axes[i]])
    var keep_dims = List[Int]()
    for i in range(len(keep_axes)):
        if keepdims:
            keep_dims.append(out_shape[i])
        else:
            keep_dims.append(out_shape[i])
    # If keepdims, the kept axes have size 1 if they were originally
    # reduced, so striding through them must take 0 in that dim.
    var iter_keep_dims = List[Int]()
    var iter_keep_strides = List[Int]()
    for d in range(len(keep_axes)):
        var src_axis = keep_axes[d]
        if keep_mask[src_axis]:
            iter_keep_dims.append(src[].shape[src_axis])
            iter_keep_strides.append(src[].strides[src_axis])
        else:
            iter_keep_dims.append(1)
            iter_keep_strides.append(0)
    # Build per-reduce-axis dim/stride.
    var red_dims = List[Int]()
    var red_strides = List[Int]()
    for i in range(len(reduce_axes)):
        red_dims.append(src[].shape[reduce_axes[i]])
        red_strides.append(src[].strides[reduce_axes[i]])
    # For each output index, decode the kept-axes coordinates, then
    # iterate the reduce subspace.
    var out_strides_logical = List[Int]()
    var stride_logical = 1
    for d in range(len(out_shape) - 1, -1, -1):
        out_strides_logical.append(stride_logical)
        stride_logical *= out_shape[d]
    out_strides_logical.reverse()
    for out_i in range(out_size):
        var src_base = src[].offset_elems
        var rem = out_i
        for d in range(len(out_shape)):
            var dim = out_shape[d]
            var coord = 0
            if dim != 0:
                coord = rem // out_strides_logical[d]
                rem = rem % out_strides_logical[d]
            src_base += coord * iter_keep_strides[d]
        # Walk reduce subspace.
        var first_phys = src_base
        for ai in range(len(reduce_axes)):
            _ = ai
        # Initialise accumulator.
        var acc: Float64
        var best_idx: Int = 0
        if op == ReduceOp.SUM.value or op == ReduceOp.MEAN.value:
            acc = 0.0
        elif op == ReduceOp.PROD.value:
            acc = 1.0
        elif op == ReduceOp.ALL.value:
            acc = 1.0
        elif op == ReduceOp.ANY.value:
            acc = 0.0
        else:
            acc = get_physical_as_f64(src[], src_base)
            best_idx = 0
        # Walk reduce coords.
        var rcoords = List[Int]()
        for _ in range(len(reduce_axes)):
            rcoords.append(0)
        var k = 0
        while True:
            var phys = src_base
            for j in range(len(reduce_axes)):
                phys += rcoords[j] * red_strides[j]
            var value = get_physical_as_f64(src[], phys)
            if op == ReduceOp.SUM.value or op == ReduceOp.MEAN.value:
                acc += value
            elif op == ReduceOp.PROD.value:
                acc *= value
            elif op == ReduceOp.MIN.value:
                if k == 0 or value < acc:
                    acc = value
            elif op == ReduceOp.MAX.value:
                if k == 0 or value > acc:
                    acc = value
            elif op == ReduceOp.ALL.value:
                if value == 0.0:
                    acc = 0.0
                    break
            elif op == ReduceOp.ANY.value:
                if value != 0.0:
                    acc = 1.0
                    break
            elif op == ReduceOp.ARGMAX.value:
                if k == 0 or value > acc:
                    acc = value
                    best_idx = k
            elif op == ReduceOp.ARGMIN.value:
                if k == 0 or value < acc:
                    acc = value
                    best_idx = k
            else:
                raise Error("unknown reduction op")
            k += 1
            # Advance rcoords innermost first.
            var idx = len(reduce_axes) - 1
            var done = False
            while idx >= 0:
                rcoords[idx] += 1
                if rcoords[idx] < red_dims[idx]:
                    break
                rcoords[idx] = 0
                idx -= 1
                if idx < 0:
                    done = True
            if done:
                break
        if op == ReduceOp.MEAN.value:
            acc = acc / Float64(reduce_size)
        if op == ReduceOp.ARGMAX.value or op == ReduceOp.ARGMIN.value:
            set_logical_from_i64(result, out_i, Int64(best_idx))
        else:
            set_logical_from_f64(result, out_i, acc)
        _ = first_phys  # silence unused warning
    return PythonObject(alloc=result^)
