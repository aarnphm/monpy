"""Shape-manipulation PythonObject bridge ops.

Hosts the layout/view manipulators (reshape/ravel/flatten/squeeze/transpose/
swapaxes/flip/slice/broadcast_to/expand_dims/materialize), the rank-2
diagonal/trace pair, and the data-moving combinators (concatenate / stack /
tril / triu / pad_constant). The view ops are pure stride-rewrites: they
allocate a fresh `Array` header that aliases the source storage. The
combinators route through `memcpy` when all inputs are c-contiguous and
same-dtype, falling back to per-element f64 round-trip otherwise.

Why grouped: every op shares the same ABI shape (PythonObject in/out) and
either does layout-algebra view construction or fast-path-or-fallback
copy dispatch.
"""

from std.collections import List
from std.memory import memcpy as _memcpy
from std.python import PythonObject

from array import (
    Array,
    array_with_layout,
    as_broadcast_layout,
    as_layout,
    clone_int_list,
    contiguous_ptr,
    copy_c_contiguous,
    get_logical_as_f64,
    get_physical,
    get_physical_as_f64,
    get_physical_bool,
    int_list_from_py,
    is_c_contiguous,
    item_size,
    make_empty_array,
    make_view_array,
    result_dtype_for_reduction,
    set_logical_from_f64,
    set_logical_from_i64,
    shape_size,
    slice_length,
)
from cute.functional import select as cute_select
from cute.int_tuple import IntTuple
from cute.layout import make_layout_row_major
from domain import ArrayDType, ReduceOp


def reshape_ops(array_obj: PythonObject, shape_obj: PythonObject) raises -> PythonObject:
    # Layout-algebra view: c-contig source → fresh row-major layout over
    # the new shape (no data movement). Non-contig source → materialize
    # a c-contig copy first, then reshape it. Matches numpy: reshape
    # returns a view when possible, a copy when not.
    var src = array_obj.downcast_value_ptr[Array]()
    var new_shape = int_list_from_py(shape_obj)
    var new_size = shape_size(new_shape)
    if new_size != src[].size_value:
        raise Error("cannot reshape array to requested size")
    if is_c_contiguous(src[]):
        var shape_tuple = IntTuple.flat(clone_int_list(new_shape))
        var new_layout = make_layout_row_major(shape_tuple^)
        var view = array_with_layout(src[], new_layout)
        return PythonObject(alloc=view^)
    var copied = copy_c_contiguous(src[])
    var shape_tuple = IntTuple.flat(clone_int_list(new_shape))
    var copy_layout = make_layout_row_major(shape_tuple^)
    var view = array_with_layout(copied, copy_layout)
    return PythonObject(alloc=view^)


def ravel_ops(array_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var shape = List[Int]()
    shape.append(src[].size_value)
    var strides = List[Int]()
    strides.append(1)
    if is_c_contiguous(src[]):
        var view = make_view_array(src[], shape^, strides^, src[].size_value, src[].offset_elems)
        return PythonObject(alloc=view^)
    var copied = copy_c_contiguous(src[])
    var view = make_view_array(copied, shape^, strides^, copied.size_value, copied.offset_elems)
    return PythonObject(alloc=view^)


def flatten_ops(array_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var copied = copy_c_contiguous(src[])
    var shape = List[Int]()
    shape.append(copied.size_value)
    var strides = List[Int]()
    strides.append(1)
    var view = make_view_array(copied, shape^, strides^, copied.size_value, copied.offset_elems)
    return PythonObject(alloc=view^)


def squeeze_all_ops(array_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var shape = List[Int]()
    var strides = List[Int]()
    for axis in range(len(src[].shape)):
        if src[].shape[axis] != 1:
            shape.append(src[].shape[axis])
            strides.append(src[].strides[axis])
    var result = make_view_array(src[], shape^, strides^, src[].size_value, src[].offset_elems)
    return PythonObject(alloc=result^)


def squeeze_axis_ops(array_obj: PythonObject, axis_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var ndim = len(src[].shape)
    var axis = Int(py=axis_obj)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise Error("squeeze: axis out of range")
    if src[].shape[axis] != 1:
        raise Error("squeeze: cannot select an axis with size != 1")
    var shape = List[Int]()
    var strides = List[Int]()
    for src_axis in range(ndim):
        if src_axis != axis:
            shape.append(src[].shape[src_axis])
            strides.append(src[].strides[src_axis])
    var result = make_view_array(src[], shape^, strides^, src[].size_value, src[].offset_elems)
    return PythonObject(alloc=result^)


def squeeze_axes_ops(array_obj: PythonObject, axes_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var ndim = len(src[].shape)
    var axes = int_list_from_py(axes_obj)
    var drop = List[Bool]()
    for _ in range(ndim):
        drop.append(False)
    for i in range(len(axes)):
        var axis = axes[i]
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise Error("squeeze: axis out of range")
        if src[].shape[axis] != 1:
            raise Error("squeeze: cannot select an axis with size != 1")
        if drop[axis]:
            raise Error("squeeze: repeated axis")
        drop[axis] = True
    var shape = List[Int]()
    var strides = List[Int]()
    for axis in range(ndim):
        if not drop[axis]:
            shape.append(src[].shape[axis])
            strides.append(src[].strides[axis])
    var result = make_view_array(src[], shape^, strides^, src[].size_value, src[].offset_elems)
    return PythonObject(alloc=result^)


def transpose_ops(array_obj: PythonObject, axes_obj: PythonObject) raises -> PythonObject:
    # Layout-algebra view: `select(L, axes)` permutes the top-level modes.
    # Equivalent to manual stride-list permutation but algebraically sourced.
    var src = array_obj.downcast_value_ptr[Array]()
    var axes = int_list_from_py(axes_obj)
    if len(axes) != len(src[].shape):
        raise Error("transpose() axes length must match ndim")
    for i in range(len(axes)):
        if axes[i] < 0 or axes[i] >= len(src[].shape):
            raise Error("transpose() axis out of bounds")
    var src_layout = as_layout(src[])
    var permuted = cute_select(src_layout, axes)
    var view = array_with_layout(src[], permuted)
    return PythonObject(alloc=view^)


def swapaxes_ops(array_obj: PythonObject, axis1_obj: PythonObject, axis2_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var ndim = len(src[].shape)
    var axis1 = Int(py=axis1_obj)
    var axis2 = Int(py=axis2_obj)
    if axis1 < 0:
        axis1 += ndim
    if axis2 < 0:
        axis2 += ndim
    if axis1 < 0 or axis1 >= ndim or axis2 < 0 or axis2 >= ndim:
        raise Error("swapaxes: axis out of range")
    var shape = clone_int_list(src[].shape)
    var strides = clone_int_list(src[].strides)
    var tmp_shape = shape[axis1]
    shape[axis1] = shape[axis2]
    shape[axis2] = tmp_shape
    var tmp_stride = strides[axis1]
    strides[axis1] = strides[axis2]
    strides[axis2] = tmp_stride
    var result = make_view_array(src[], shape^, strides^, src[].size_value, src[].offset_elems)
    return PythonObject(alloc=result^)


def flip_ops(
    array_obj: PythonObject,
    axes_obj: PythonObject,
) raises -> PythonObject:
    # Flip the iteration order on each axis in `axes` by negating its
    # stride and shifting `offset_elems`. Pure view; no data movement.
    # An empty `axes` list flips every axis (numpy default).
    var src = array_obj.downcast_value_ptr[Array]()
    var ndim = len(src[].shape)
    var shape = clone_int_list(src[].shape)
    var strides = clone_int_list(src[].strides)
    var offset = src[].offset_elems

    var flip_all = len(axes_obj) == 0
    var seen = List[Bool]()
    for _ in range(ndim):
        seen.append(False)

    if flip_all:
        for i in range(ndim):
            seen[i] = True
    else:
        for k in range(len(axes_obj)):
            var ax = Int(py=axes_obj[k])
            if ax < 0:
                ax += ndim
            if ax < 0 or ax >= ndim:
                raise Error("flip() axis out of bounds")
            if seen[ax]:
                raise Error("flip() repeated axis")
            seen[ax] = True

    for i in range(ndim):
        if seen[i]:
            offset += (shape[i] - 1) * strides[i]
            strides[i] = -strides[i]

    var result = make_view_array(src[], shape^, strides^, src[].size_value, offset)
    return PythonObject(alloc=result^)


def transpose_full_reverse_ops(
    array_obj: PythonObject,
) raises -> PythonObject:
    # Fast path for `.T` on rank>=2: reverse every axis without crossing
    # Python boundaries for an axes tuple. Avoids `int_list_from_py` and
    # the per-axis bounds check; `make_view_array` validates shape/strides
    # match.
    var src = array_obj.downcast_value_ptr[Array]()
    var ndim = len(src[].shape)
    var shape = List[Int]()
    var strides = List[Int]()
    for i in range(ndim - 1, -1, -1):
        shape.append(src[].shape[i])
        strides.append(src[].strides[i])
    var result = make_view_array(src[], shape^, strides^, src[].size_value, src[].offset_elems)
    return PythonObject(alloc=result^)


def slice_ops(
    array_obj: PythonObject,
    starts_obj: PythonObject,
    stops_obj: PythonObject,
    steps_obj: PythonObject,
    drops_obj: PythonObject,
) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var starts = int_list_from_py(starts_obj)
    var stops = int_list_from_py(stops_obj)
    var steps = int_list_from_py(steps_obj)
    var drops = int_list_from_py(drops_obj)
    if (
        len(starts) != len(src[].shape)
        or len(stops) != len(src[].shape)
        or len(steps) != len(src[].shape)
        or len(drops) != len(src[].shape)
    ):
        raise Error("slice metadata rank mismatch")
    var offset = src[].offset_elems
    var shape = List[Int]()
    var strides = List[Int]()
    for axis in range(len(src[].shape)):
        offset += starts[axis] * src[].strides[axis]
        if drops[axis] == 0:
            shape.append(slice_length(src[].shape[axis], starts[axis], stops[axis], steps[axis]))
            strides.append(src[].strides[axis] * steps[axis])
    var result = make_view_array(src[], shape^, strides^, shape_size(shape), offset)
    return PythonObject(alloc=result^)


def broadcast_to_ops(array_obj: PythonObject, shape_obj: PythonObject) raises -> PythonObject:
    # Layout-algebra view: build the broadcast layout (stride-zero
    # injection on size-1 / new outer dims) and materialize a view.
    var src = array_obj.downcast_value_ptr[Array]()
    var out_shape = int_list_from_py(shape_obj)
    var ndim_out = len(out_shape)
    var ndim_src = len(src[].shape)
    if ndim_src > ndim_out:
        raise Error("cannot broadcast to fewer dimensions")
    # Validate broadcast compatibility before delegating to the layout
    # builder — `as_broadcast_layout` injects stride-0 silently on
    # size-1 mismatches but doesn't catch hard incompatibilities.
    for out_axis in range(ndim_out):
        var src_axis = out_axis - (ndim_out - ndim_src)
        if src_axis >= 0:
            var src_dim = src[].shape[src_axis]
            var out_dim = out_shape[out_axis]
            if src_dim != out_dim and src_dim != 1:
                raise Error("shape is not broadcastable")
    var bcast_layout = as_broadcast_layout(src[], out_shape)
    var view = array_with_layout(src[], bcast_layout)
    return PythonObject(alloc=view^)


def expand_dims_ops(array_obj: PythonObject, axis_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var axis = Int(py=axis_obj)
    if axis < 0 or axis > len(src[].shape):
        raise Error("axis out of bounds")
    var shape = List[Int]()
    var strides = List[Int]()
    for i in range(axis):
        shape.append(src[].shape[i])
        strides.append(src[].strides[i])
    shape.append(1)
    strides.append(0)
    for i in range(axis, len(src[].shape)):
        shape.append(src[].shape[i])
        strides.append(src[].strides[i])
    var result = make_view_array(src[], shape^, strides^, src[].size_value, src[].offset_elems)
    return PythonObject(alloc=result^)


def materialize_c_contiguous_ops(
    array_obj: PythonObject,
) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var result = copy_c_contiguous(src[])
    return PythonObject(alloc=result^)


def normalize_axis_ops(axis_value: Int, ndim: Int, name: String) raises -> Int:
    var axis = axis_value
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise Error(name, " axis out of bounds")
    return axis


@fieldwise_init
struct DiagonalMetadata(ImplicitlyCopyable, Movable, Writable):
    var length: Int
    var offset: Int
    var stride: Int


def diagonal_metadata_ops(src: Array, offset: Int, axis1: Int, axis2: Int) raises -> DiagonalMetadata:
    if len(src.shape) != 2:
        raise Error("diagonal() and trace() currently require rank-2 arrays")
    if axis1 == axis2:
        raise Error("diagonal axes must be different")
    var rows = src.shape[axis1]
    var cols = src.shape[axis2]
    var row_start = 0
    var col_start = 0
    if offset >= 0:
        col_start = offset
    else:
        row_start = -offset
    var diag_len = 0
    if row_start < rows and col_start < cols:
        var rows_left = rows - row_start
        var cols_left = cols - col_start
        diag_len = rows_left
        if cols_left < diag_len:
            diag_len = cols_left
    var diag_offset = src.offset_elems + row_start * src.strides[axis1] + col_start * src.strides[axis2]
    var diag_stride = src.strides[axis1] + src.strides[axis2]
    return DiagonalMetadata(diag_len, diag_offset, diag_stride)


def diagonal_ops(
    array_obj: PythonObject,
    offset_obj: PythonObject,
    axis1_obj: PythonObject,
    axis2_obj: PythonObject,
) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var axis1 = normalize_axis_ops(Int(py=axis1_obj), len(src[].shape), "axis1")
    var axis2 = normalize_axis_ops(Int(py=axis2_obj), len(src[].shape), "axis2")
    var metadata = diagonal_metadata_ops(src[], Int(py=offset_obj), axis1, axis2)
    var shape = List[Int]()
    shape.append(metadata.length)
    var strides = List[Int]()
    strides.append(metadata.stride)
    var result = make_view_array(src[], shape^, strides^, metadata.length, metadata.offset)
    return PythonObject(alloc=result^)


def slice_1d_ops(
    array_obj: PythonObject,
    start_obj: PythonObject,
    stop_obj: PythonObject,
    step_obj: PythonObject,
) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 1:
        raise Error("slice_1d() requires a rank-1 array")
    var start = Int(py=start_obj)
    var stop = Int(py=stop_obj)
    var step = Int(py=step_obj)
    var shape = List[Int]()
    shape.append(slice_length(src[].shape[0], start, stop, step))
    var strides = List[Int]()
    strides.append(src[].strides[0] * step)
    var result = make_view_array(
        src[],
        shape^,
        strides^,
        shape_size(shape),
        src[].offset_elems + start * src[].strides[0],
    )
    return PythonObject(alloc=result^)


def reverse_1d_ops(array_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 1:
        raise Error("reverse_1d() requires a rank-1 array")
    var length = src[].shape[0]
    var shape = List[Int]()
    shape.append(length)
    var strides = List[Int]()
    strides.append(-src[].strides[0])
    var offset = src[].offset_elems
    if length > 0:
        offset += (length - 1) * src[].strides[0]
    var result = make_view_array(src[], shape^, strides^, src[].size_value, offset)
    return PythonObject(alloc=result^)


def trace_ops(
    array_obj: PythonObject,
    offset_obj: PythonObject,
    axis1_obj: PythonObject,
    axis2_obj: PythonObject,
    dtype_obj: PythonObject,
) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var axis1 = normalize_axis_ops(Int(py=axis1_obj), len(src[].shape), "axis1")
    var axis2 = normalize_axis_ops(Int(py=axis2_obj), len(src[].shape), "axis2")
    var metadata = diagonal_metadata_ops(src[], Int(py=offset_obj), axis1, axis2)
    var diag_len = metadata.length
    var diag_offset = metadata.offset
    var diag_stride = metadata.stride
    var shape = List[Int]()
    var dtype_code = Int(py=dtype_obj)
    if dtype_code < 0:
        dtype_code = result_dtype_for_reduction(src[].dtype_code, ReduceOp.SUM.value)
    var result = make_empty_array(dtype_code, shape^)
    if src[].dtype_code == ArrayDType.INT64.value:
        var acc = Int64(0)
        for i in range(diag_len):
            acc += get_physical[DType.int64](src[], diag_offset + i * diag_stride)
        set_logical_from_i64(result, 0, acc)
    elif src[].dtype_code == ArrayDType.BOOL.value:
        var acc = Int64(0)
        for i in range(diag_len):
            if get_physical_bool(src[], diag_offset + i * diag_stride):
                acc += 1
        set_logical_from_i64(result, 0, acc)
    else:
        var acc = 0.0
        for i in range(diag_len):
            acc += get_physical_as_f64(src[], diag_offset + i * diag_stride)
        set_logical_from_f64(result, 0, acc)
    return PythonObject(alloc=result^)


def concatenate_ops(
    arrays_obj: PythonObject, axis_obj: PythonObject, dtype_code_obj: PythonObject
) raises -> PythonObject:
    # Native concatenation. Replaces the python flat-write path which was
    # ~58000× slower than numpy at N=256.
    var n_arrays = Int(py=len(arrays_obj))
    if n_arrays == 0:
        raise Error("concatenate: need at least one array")
    var axis = Int(py=axis_obj)
    var dtype_code = Int(py=dtype_code_obj)
    var infer_dtype = dtype_code < 0
    # Pull rank from first input.
    var first = arrays_obj[0].downcast_value_ptr[Array]()
    var ndim = len(first[].shape)
    if axis < 0:
        axis = axis + ndim
    if axis < 0 or axis >= ndim:
        raise Error("concatenate: axis out of range")
    if dtype_code < 0:
        dtype_code = first[].dtype_code
    # Build out_shape; check shape consistency across arrays.
    var out_shape = List[Int]()
    for d in range(ndim):
        out_shape.append(first[].shape[d])
    out_shape[axis] = 0
    var all_c_contig = True
    for i in range(n_arrays):
        var a = arrays_obj[i].downcast_value_ptr[Array]()
        if len(a[].shape) != ndim:
            raise Error("concatenate: arrays must have same ndim")
        for d in range(ndim):
            if d == axis:
                out_shape[axis] = out_shape[axis] + a[].shape[d]
            elif a[].shape[d] != first[].shape[d]:
                raise Error("concatenate: shape mismatch")
        if a[].dtype_code != dtype_code:
            raise Error("concatenate: dtype mismatch (caller must pre-cast)")
        if not is_c_contiguous(a[]):
            all_c_contig = False
    if infer_dtype and not all_c_contig:
        raise Error("concatenate: fast path requires c-contiguous inputs")
    var out_shape_clone = List[Int]()
    for d in range(ndim):
        out_shape_clone.append(out_shape[d])
    var result = make_empty_array(dtype_code, out_shape_clone^)
    var outer_size = 1
    for d in range(axis):
        outer_size *= out_shape[d]
    var inner_size = 1
    for d in range(axis + 1, ndim):
        inner_size *= out_shape[d]
    var item_bytes = item_size(dtype_code)
    var out_axis_size = out_shape[axis]
    var out_row_size = out_axis_size * inner_size
    var axis_offset = 0
    for i in range(n_arrays):
        var a = arrays_obj[i].downcast_value_ptr[Array]()
        var a_axis = a[].shape[axis]
        var a_slab_size = a_axis * inner_size
        if all_c_contig:
            var src_byte_offset = a[].offset_elems * item_bytes
            for outer in range(outer_size):
                var src_off_bytes = src_byte_offset + outer * a_slab_size * item_bytes
                var dst_off_bytes = (outer * out_row_size + axis_offset * inner_size) * item_bytes
                _memcpy(
                    dest=result.data + dst_off_bytes,
                    src=a[].data + src_off_bytes,
                    count=a_slab_size * item_bytes,
                )
        else:
            for outer in range(outer_size):
                var src_base = outer * a_slab_size
                var dst_base = outer * out_row_size + axis_offset * inner_size
                for k in range(a_slab_size):
                    set_logical_from_f64(result, dst_base + k, get_logical_as_f64(a[], src_base + k))
        axis_offset = axis_offset + a_axis
    return PythonObject(alloc=result^)


def stack_axis0_ops(
    arrays_obj: PythonObject, dtype_code_obj: PythonObject, require_rank1_obj: PythonObject
) raises -> PythonObject:
    var n_arrays = Int(py=len(arrays_obj))
    if n_arrays == 0:
        raise Error("stack: need at least one array")
    var dtype_code = Int(py=dtype_code_obj)
    var require_rank1 = Bool(py=require_rank1_obj)
    var first = arrays_obj[0].downcast_value_ptr[Array]()
    var ndim = len(first[].shape)
    if require_rank1 and ndim != 1:
        raise Error("vstack: fast path only handles rank-1 inputs")
    var out_shape = List[Int]()
    out_shape.append(n_arrays)
    for d in range(ndim):
        out_shape.append(first[].shape[d])
    var all_c_contig = True
    for i in range(n_arrays):
        var a = arrays_obj[i].downcast_value_ptr[Array]()
        if len(a[].shape) != ndim:
            raise Error("stack: arrays must have identical rank")
        if a[].dtype_code != dtype_code:
            raise Error("stack: dtype mismatch")
        for d in range(ndim):
            if a[].shape[d] != first[].shape[d]:
                raise Error("stack: arrays must have identical shape")
        if not is_c_contiguous(a[]):
            all_c_contig = False
    var result = make_empty_array(dtype_code, out_shape^)
    var item_bytes = item_size(dtype_code)
    var slab_elems = first[].size_value
    var slab_bytes = slab_elems * item_bytes
    for i in range(n_arrays):
        var a = arrays_obj[i].downcast_value_ptr[Array]()
        if all_c_contig:
            _memcpy(
                dest=result.data + i * slab_bytes,
                src=a[].data + a[].offset_elems * item_bytes,
                count=slab_bytes,
            )
        else:
            var dst_base = i * slab_elems
            for logical in range(slab_elems):
                set_logical_from_f64(result, dst_base + logical, get_logical_as_f64(a[], logical))
    return PythonObject(alloc=result^)


def tril_ops(array_obj: PythonObject, k_obj: PythonObject) raises -> PythonObject:
    # Native lower-triangular: copy values where col <= row + k, zero otherwise.
    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) < 2:
        raise Error("tril: requires rank >= 2 input")
    var k = Int(py=k_obj)
    var ndim = len(src[].shape)
    var rows = src[].shape[ndim - 2]
    var cols = src[].shape[ndim - 1]
    var out_shape = List[Int]()
    for d in range(ndim):
        out_shape.append(src[].shape[d])
    var result = make_empty_array(src[].dtype_code, out_shape^)
    var batch = 1
    for d in range(ndim - 2):
        batch *= src[].shape[d]
    if src[].dtype_code == ArrayDType.FLOAT32.value and is_c_contiguous(src[]):
        var src_ptr = contiguous_ptr[DType.float32](src[])
        var out_ptr = contiguous_ptr[DType.float32](result)
        for b in range(batch):
            for r in range(rows):
                for c in range(cols):
                    var idx = b * rows * cols + r * cols + c
                    if c <= r + k:
                        out_ptr[idx] = src_ptr[idx]
                    else:
                        out_ptr[idx] = Float32(0.0)
        return PythonObject(alloc=result^)
    if src[].dtype_code == ArrayDType.FLOAT64.value and is_c_contiguous(src[]):
        var src_ptr = contiguous_ptr[DType.float64](src[])
        var out_ptr = contiguous_ptr[DType.float64](result)
        for b in range(batch):
            for r in range(rows):
                for c in range(cols):
                    var idx = b * rows * cols + r * cols + c
                    if c <= r + k:
                        out_ptr[idx] = src_ptr[idx]
                    else:
                        out_ptr[idx] = 0.0
        return PythonObject(alloc=result^)
    for b in range(batch):
        for r in range(rows):
            for c in range(cols):
                var idx = b * rows * cols + r * cols + c
                if c <= r + k:
                    set_logical_from_f64(result, idx, get_logical_as_f64(src[], idx))
                else:
                    set_logical_from_f64(result, idx, 0.0)
    return PythonObject(alloc=result^)


def triu_ops(array_obj: PythonObject, k_obj: PythonObject) raises -> PythonObject:
    # Native upper-triangular: copy values where col >= row + k, zero otherwise.
    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) < 2:
        raise Error("triu: requires rank >= 2 input")
    var k = Int(py=k_obj)
    var ndim = len(src[].shape)
    var rows = src[].shape[ndim - 2]
    var cols = src[].shape[ndim - 1]
    var out_shape = List[Int]()
    for d in range(ndim):
        out_shape.append(src[].shape[d])
    var result = make_empty_array(src[].dtype_code, out_shape^)
    var batch = 1
    for d in range(ndim - 2):
        batch *= src[].shape[d]
    if src[].dtype_code == ArrayDType.FLOAT32.value and is_c_contiguous(src[]):
        var src_ptr = contiguous_ptr[DType.float32](src[])
        var out_ptr = contiguous_ptr[DType.float32](result)
        for b in range(batch):
            for r in range(rows):
                for c in range(cols):
                    var idx = b * rows * cols + r * cols + c
                    if c >= r + k:
                        out_ptr[idx] = src_ptr[idx]
                    else:
                        out_ptr[idx] = Float32(0.0)
        return PythonObject(alloc=result^)
    if src[].dtype_code == ArrayDType.FLOAT64.value and is_c_contiguous(src[]):
        var src_ptr = contiguous_ptr[DType.float64](src[])
        var out_ptr = contiguous_ptr[DType.float64](result)
        for b in range(batch):
            for r in range(rows):
                for c in range(cols):
                    var idx = b * rows * cols + r * cols + c
                    if c >= r + k:
                        out_ptr[idx] = src_ptr[idx]
                    else:
                        out_ptr[idx] = 0.0
        return PythonObject(alloc=result^)
    for b in range(batch):
        for r in range(rows):
            for c in range(cols):
                var idx = b * rows * cols + r * cols + c
                if c >= r + k:
                    set_logical_from_f64(result, idx, get_logical_as_f64(src[], idx))
                else:
                    set_logical_from_f64(result, idx, 0.0)
    return PythonObject(alloc=result^)


def pad_constant_ops(
    array_obj: PythonObject,
    pad_before_obj: PythonObject,
    pad_after_obj: PythonObject,
    constant_value_obj: PythonObject,
) raises -> PythonObject:
    # Native constant-mode pad. pad_before/pad_after are tuples of length ndim.
    var src = array_obj.downcast_value_ptr[Array]()
    var ndim = len(src[].shape)
    var pad_before = List[Int]()
    var pad_after = List[Int]()
    for d in range(ndim):
        pad_before.append(Int(py=pad_before_obj[d]))
        pad_after.append(Int(py=pad_after_obj[d]))
    var out_shape = List[Int]()
    for d in range(ndim):
        out_shape.append(src[].shape[d] + pad_before[d] + pad_after[d])
    var out_shape_clone = List[Int]()
    for d in range(ndim):
        out_shape_clone.append(out_shape[d])
    var result = make_empty_array(src[].dtype_code, out_shape_clone^)
    var constant_f64 = Float64(py=constant_value_obj)
    # Fill with constant first.
    var out_size = 1
    for d in range(ndim):
        out_size *= out_shape[d]
    for i in range(out_size):
        set_logical_from_f64(result, i, constant_f64)
    # Compute strides for source and result.
    var src_strides = List[Int]()
    var out_strides = List[Int]()
    for _ in range(ndim):
        src_strides.append(0)
        out_strides.append(0)
    var s = 1
    var o = 1
    for d in range(ndim - 1, -1, -1):
        src_strides[d] = s
        s = s * src[].shape[d]
        out_strides[d] = o
        o = o * out_shape[d]
    # Copy source into result with offset.
    var src_size = 1
    for d in range(ndim):
        src_size *= src[].shape[d]
    for i in range(src_size):
        # Decode source coordinate.
        var remainder = i
        var dst_idx = 0
        for d in range(ndim):
            var coord = remainder // src_strides[d]
            remainder = remainder % src_strides[d]
            dst_idx += (coord + pad_before[d]) * out_strides[d]
        set_logical_from_f64(result, dst_idx, get_logical_as_f64(src[], i))
    return PythonObject(alloc=result^)
