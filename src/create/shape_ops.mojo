"""Shape-manipulation `_ops` Python-bridge entry points.

Hosts native fast paths for concatenate / stack / tril / triu / pad_constant.
Each routes through `memcpy` when all inputs are c-contiguous and same-dtype,
and falls back to per-element f64 round-trip when not. The fallback paths
match numpy semantics but are slower.

Why grouped: every op shares the same ABI shape (PythonObject in/out) and
the same fast-path-or-fallback dispatch pattern.
"""

from std.memory import memcpy as _memcpy
from std.python import PythonObject

from array import (
    Array,
    contiguous_f32_ptr,
    contiguous_f64_ptr,
    get_logical_as_f64,
    is_c_contiguous,
    item_size,
    make_empty_array,
    set_logical_from_f64,
)
from domain import ArrayDType


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
        var src_ptr = contiguous_f32_ptr(src[])
        var out_ptr = contiguous_f32_ptr(result)
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
        var src_ptr = contiguous_f64_ptr(src[])
        var out_ptr = contiguous_f64_ptr(result)
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
        var src_ptr = contiguous_f32_ptr(src[])
        var out_ptr = contiguous_f32_ptr(result)
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
        var src_ptr = contiguous_f64_ptr(src[])
        var out_ptr = contiguous_f64_ptr(result)
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
