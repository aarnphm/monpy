"""Array-creation `_ops` Python-bridge entry points.

Hosts the empty/full/from_*/range-style constructors that allocate a fresh
`Array` and fill it. None of these go through kernel dispatch — they're pure
allocate-then-write paths, with `linspace`/`arange`/`logspace` doing scalar
math per element.

Why grouped: every op shares the same shape — unbox shape/dtype, allocate
via `make_empty_array`, write values, return a `PythonObject(alloc=…)`.
The `from_external_ops` and `copy_from_external_ops` pair handle the
__array_interface__ / DLPack bridge to numpy.
"""

from std.python import PythonObject

from accelerate import libm_pow_f64
from array import (
    Array,
    clone_int_list,
    contiguous_i64_ptr,
    copy_c_contiguous,
    fill_all_from_py,
    int_list_from_py,
    make_empty_array,
    make_external_array,
    set_logical_from_f64,
    set_logical_from_py,
)
from domain import ArrayDType


def empty_ops(shape_obj: PythonObject, dtype_obj: PythonObject) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var shape = int_list_from_py(shape_obj)
    var result = make_empty_array(dtype_code, shape^)
    return PythonObject(alloc=result^)


def full_ops(shape_obj: PythonObject, value_obj: PythonObject, dtype_obj: PythonObject) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var shape = int_list_from_py(shape_obj)
    var result = make_empty_array(dtype_code, shape^)
    fill_all_from_py(result, value_obj)
    return PythonObject(alloc=result^)


def full_like_ops(
    prototype_obj: PythonObject, value_obj: PythonObject, dtype_obj: PythonObject
) raises -> PythonObject:
    var src = prototype_obj.downcast_value_ptr[Array]()
    var dtype_code = Int(py=dtype_obj)
    if dtype_code < 0:
        dtype_code = src[].dtype_code
    var shape = clone_int_list(src[].shape)
    var result = make_empty_array(dtype_code, shape^)
    fill_all_from_py(result, value_obj)
    return PythonObject(alloc=result^)


def from_flat_ops(values_obj: PythonObject, shape_obj: PythonObject, dtype_obj: PythonObject) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var shape = int_list_from_py(shape_obj)
    var result = make_empty_array(dtype_code, shape^)
    if len(values_obj) != result.size_value:
        raise Error("flat value count does not match shape")
    for i in range(result.size_value):
        set_logical_from_py(result, i, values_obj[i])
    return PythonObject(alloc=result^)


def from_external_ops(
    data_address_obj: PythonObject,
    shape_obj: PythonObject,
    strides_obj: PythonObject,
    dtype_obj: PythonObject,
    byte_len_obj: PythonObject,
) raises -> PythonObject:
    var address = Int(py=data_address_obj)
    if address == 0:
        raise Error("array interface data pointer is null")
    var shape = int_list_from_py(shape_obj)
    var strides = int_list_from_py(strides_obj)
    var data = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=address)
    var result = make_external_array(Int(py=dtype_obj), shape^, strides^, 0, data, Int(py=byte_len_obj))
    return PythonObject(alloc=result^)


def copy_from_external_ops(
    data_address_obj: PythonObject,
    shape_obj: PythonObject,
    strides_obj: PythonObject,
    dtype_obj: PythonObject,
    byte_len_obj: PythonObject,
) raises -> PythonObject:
    # One-shot copy: build an external view of `address` with the supplied
    # shape/strides, then materialize a c-contiguous copy. Saves an FFI
    # round-trip vs calling `from_external_ops` then `materialize_c_contiguous_ops`
    # back-to-back. Hits the memcpy fast path when the source is already
    # c-contig, otherwise the elementwise walk in `copy_c_contiguous`.
    var address = Int(py=data_address_obj)
    if address == 0:
        raise Error("array interface data pointer is null")
    var shape = int_list_from_py(shape_obj)
    var strides = int_list_from_py(strides_obj)
    var data = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=address)
    var external = make_external_array(Int(py=dtype_obj), shape^, strides^, 0, data, Int(py=byte_len_obj))
    var result = copy_c_contiguous(external)
    return PythonObject(alloc=result^)


def arange_ops(
    start_obj: PythonObject,
    stop_obj: PythonObject,
    step_obj: PythonObject,
    dtype_obj: PythonObject,
) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var start = Float64(py=start_obj)
    var stop = Float64(py=stop_obj)
    var step = Float64(py=step_obj)
    if step == 0.0:
        raise Error("arange() step must not be zero")
    var count = 0
    var current = start
    if step > 0.0:
        while current < stop:
            count += 1
            current += step
    else:
        while current > stop:
            count += 1
            current += step
    var shape = List[Int]()
    shape.append(count)
    var result = make_empty_array(dtype_code, shape^)
    current = start
    for i in range(count):
        set_logical_from_f64(result, i, current)
        current += step
    return PythonObject(alloc=result^)


def linspace_ops(
    start_obj: PythonObject,
    stop_obj: PythonObject,
    num_obj: PythonObject,
    dtype_obj: PythonObject,
) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var start = Float64(py=start_obj)
    var stop = Float64(py=stop_obj)
    var num = Int(py=num_obj)
    if num < 0:
        raise Error("linspace() num must be non-negative")
    var shape = List[Int]()
    shape.append(num)
    var result = make_empty_array(dtype_code, shape^)
    if num == 0:
        return PythonObject(alloc=result^)
    if num == 1:
        set_logical_from_f64(result, 0, start)
        return PythonObject(alloc=result^)
    var step = (stop - start) / Float64(num - 1)
    for i in range(num):
        set_logical_from_f64(result, i, start + step * Float64(i))
    return PythonObject(alloc=result^)


def logspace_ops(
    start_obj: PythonObject,
    stop_obj: PythonObject,
    num_obj: PythonObject,
    endpoint_obj: PythonObject,
    base_obj: PythonObject,
    dtype_obj: PythonObject,
) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var start = Float64(py=start_obj)
    var stop = Float64(py=stop_obj)
    var num = Int(py=num_obj)
    var endpoint = Bool(py=endpoint_obj)
    var base = Float64(py=base_obj)
    if num < 0:
        raise Error("logspace() num must be non-negative")
    var shape = List[Int]()
    shape.append(num)
    var result = make_empty_array(dtype_code, shape^)
    if num == 0:
        return PythonObject(alloc=result^)
    if num == 1:
        set_logical_from_f64(result, 0, libm_pow_f64(base, start))
        return PythonObject(alloc=result^)
    var denom = num - 1 if endpoint else num
    var step = (stop - start) / Float64(denom)
    for i in range(num):
        var exponent = start + step * Float64(i)
        set_logical_from_f64(result, i, libm_pow_f64(base, exponent))
    return PythonObject(alloc=result^)


def indices_ops(dimensions_obj: PythonObject, dtype_obj: PythonObject) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    if dtype_code != ArrayDType.INT64.value:
        raise Error("indices() native kernel only supports int64")
    var dims = int_list_from_py(dimensions_obj)
    var rank = len(dims)
    var out_shape = List[Int]()
    out_shape.append(rank)
    var strides = List[Int]()
    for axis in range(rank):
        if dims[axis] < 0:
            raise Error("indices() dimensions must be non-negative")
        out_shape.append(dims[axis])
        strides.append(1)
    var plane_size = 1
    for axis in range(rank - 1, -1, -1):
        strides[axis] = plane_size
        plane_size *= dims[axis]
    var result = make_empty_array(ArrayDType.INT64.value, out_shape^)
    if rank == 0 or plane_size == 0:
        return PythonObject(alloc=result^)
    var out = contiguous_i64_ptr(result)
    for axis in range(rank):
        var stride = strides[axis]
        var dim = dims[axis]
        var base = axis * plane_size
        for logical in range(plane_size):
            out[base + logical] = Int64((logical // stride) % dim)
    return PythonObject(alloc=result^)
