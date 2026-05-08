"""Runtime dtype dispatch + f64 / Python-object read-write helpers.

Hosts:
  - `dispatch_real_to_f64` / `dispatch_real_write_f64` /
    `dispatch_int_write_i64` — comptime-fn-parametric dispatchers over
    the 11 real / 8 integer dtypes. Each takes a parametric kernel and
    fans out per-dtype; the compiler emits one specialisation per branch.
  - `dispatch_real_contig_fill_from_py` /
    `dispatch_real_write_from_py` — same pattern over `PythonObject`
    inputs. Used by `fill_all_from_py` and `set_logical_from_py`.
  - `get_physical_as_f64` / `set_physical_from_f64` /
    `set_logical_from_*` — universal f64 / i64 read-write paths that
    everyone else falls back to. Bool / complex are handled inline
    before delegating to the real-dtype dispatcher.
  - `contiguous_as_f64` / `set_contiguous_from_f64` — fast-path read-write
    for c-contig f32/f64 (the hot dtypes), with f64 round-trip fallback.
  - `scalar_py_as_f64` — `PythonObject` → Float64 narrowing that respects
    integer dtypes (Int(py=...) instead of Float64(py=...) to avoid the
    silent fractional truncation).

Mirrors the `BinaryContigKernel` / `dispatch_real_typed_*` idiom in
`src/elementwise/dispatch_helpers.mojo` — the runtime → comptime dtype
switch lives in one place.
"""

from std.python import PythonObject

from domain import ArrayDType

from .accessors import (
    Array,
    contiguous_ptr,
    get_physical,
    get_physical_bool,
    get_physical_c128_real,
    get_physical_c64_real,
    is_c_contiguous,
    physical_offset,
    set_physical,
)


comptime PhysReadAsF64 = def[dt: DType](Array, Int) thin raises -> Float64
"""Reads a physical-offset value of dtype `dt` and returns it as Float64."""


comptime PhysWriteFromF64 = def[dt: DType](mut Array, Int, Float64) thin raises -> None
"""Narrows a Float64 to dtype `dt` and writes it to a physical offset."""


comptime PhysWriteFromI64 = def[dt: DType](mut Array, Int, Int64) thin raises -> None
"""Narrows an Int64 to integer dtype `dt` and writes it to a physical offset."""


def dispatch_real_to_f64[op: PhysReadAsF64](dtype_code: Int, array: Array, physical: Int) raises -> Float64:
    """Maps `dtype_code` to one of the 11 real DTypes and invokes `op[dt]`.
    Caller must handle BOOL / COMPLEX64 / COMPLEX128 before calling."""
    if dtype_code == ArrayDType.FLOAT64.value:
        return op[DType.float64](array, physical)
    if dtype_code == ArrayDType.FLOAT32.value:
        return op[DType.float32](array, physical)
    if dtype_code == ArrayDType.FLOAT16.value:
        return op[DType.float16](array, physical)
    if dtype_code == ArrayDType.INT64.value:
        return op[DType.int64](array, physical)
    if dtype_code == ArrayDType.INT32.value:
        return op[DType.int32](array, physical)
    if dtype_code == ArrayDType.INT16.value:
        return op[DType.int16](array, physical)
    if dtype_code == ArrayDType.INT8.value:
        return op[DType.int8](array, physical)
    if dtype_code == ArrayDType.UINT64.value:
        return op[DType.uint64](array, physical)
    if dtype_code == ArrayDType.UINT32.value:
        return op[DType.uint32](array, physical)
    if dtype_code == ArrayDType.UINT16.value:
        return op[DType.uint16](array, physical)
    if dtype_code == ArrayDType.UINT8.value:
        return op[DType.uint8](array, physical)
    raise Error("dispatch_real_to_f64: dtype not in real set")


def dispatch_real_write_f64[
    op: PhysWriteFromF64,
](dtype_code: Int, mut array: Array, physical: Int, value: Float64) raises -> Bool:
    """Maps `dtype_code` to one of the 11 real DTypes and invokes `op[dt]`.
    Returns True on dispatch; False if the dtype isn't real (caller falls through)."""
    if dtype_code == ArrayDType.FLOAT64.value:
        op[DType.float64](array, physical, value)
        return True
    if dtype_code == ArrayDType.FLOAT32.value:
        op[DType.float32](array, physical, value)
        return True
    if dtype_code == ArrayDType.FLOAT16.value:
        op[DType.float16](array, physical, value)
        return True
    if dtype_code == ArrayDType.INT64.value:
        op[DType.int64](array, physical, value)
        return True
    if dtype_code == ArrayDType.INT32.value:
        op[DType.int32](array, physical, value)
        return True
    if dtype_code == ArrayDType.INT16.value:
        op[DType.int16](array, physical, value)
        return True
    if dtype_code == ArrayDType.INT8.value:
        op[DType.int8](array, physical, value)
        return True
    if dtype_code == ArrayDType.UINT64.value:
        op[DType.uint64](array, physical, value)
        return True
    if dtype_code == ArrayDType.UINT32.value:
        op[DType.uint32](array, physical, value)
        return True
    if dtype_code == ArrayDType.UINT16.value:
        op[DType.uint16](array, physical, value)
        return True
    if dtype_code == ArrayDType.UINT8.value:
        op[DType.uint8](array, physical, value)
        return True
    return False


def dispatch_int_write_i64[
    op: PhysWriteFromI64,
](dtype_code: Int, mut array: Array, physical: Int, value: Int64) raises -> Bool:
    """Maps `dtype_code` to one of the 8 integer DTypes and invokes `op[dt]`.
    Returns True on dispatch; False if the dtype isn't integral (caller falls through)."""
    if dtype_code == ArrayDType.INT64.value:
        op[DType.int64](array, physical, value)
        return True
    if dtype_code == ArrayDType.INT32.value:
        op[DType.int32](array, physical, value)
        return True
    if dtype_code == ArrayDType.INT16.value:
        op[DType.int16](array, physical, value)
        return True
    if dtype_code == ArrayDType.INT8.value:
        op[DType.int8](array, physical, value)
        return True
    if dtype_code == ArrayDType.UINT64.value:
        op[DType.uint64](array, physical, value)
        return True
    if dtype_code == ArrayDType.UINT32.value:
        op[DType.uint32](array, physical, value)
        return True
    if dtype_code == ArrayDType.UINT16.value:
        op[DType.uint16](array, physical, value)
        return True
    if dtype_code == ArrayDType.UINT8.value:
        op[DType.uint8](array, physical, value)
        return True
    return False


def _phys_read_as_f64[dt: DType](array: Array, physical: Int) raises -> Float64:
    comptime if dt.is_floating_point():
        return Float64(get_physical[dt](array, physical))
    return Float64(Int(get_physical[dt](array, physical)))


def _phys_write_from_f64[dt: DType](mut array: Array, physical: Int, value: Float64) raises:
    comptime if dt.is_floating_point():
        set_physical[dt](array, physical, Scalar[dt](value))
    else:
        set_physical[dt](array, physical, Scalar[dt](Int(value)))


def _phys_write_from_i64[dt: DType](mut array: Array, physical: Int, value: Int64) raises:
    set_physical[dt](array, physical, Scalar[dt](Int(value)))


comptime ContigFillFromPy = def[dt: DType](mut Array, PythonObject) thin raises -> None
"""Fills a c-contiguous Array of dtype `dt` with the given PythonObject value."""


comptime PhysWriteFromPy = def[dt: DType](mut Array, Int, PythonObject) thin raises -> None
"""Narrows a PythonObject to dtype `dt` and writes it to a physical offset."""


def dispatch_real_contig_fill_from_py[
    op: ContigFillFromPy,
](dtype_code: Int, mut array: Array, value_obj: PythonObject) raises -> Bool:
    if dtype_code == ArrayDType.FLOAT64.value:
        op[DType.float64](array, value_obj)
        return True
    if dtype_code == ArrayDType.FLOAT32.value:
        op[DType.float32](array, value_obj)
        return True
    if dtype_code == ArrayDType.FLOAT16.value:
        op[DType.float16](array, value_obj)
        return True
    if dtype_code == ArrayDType.INT64.value:
        op[DType.int64](array, value_obj)
        return True
    if dtype_code == ArrayDType.INT32.value:
        op[DType.int32](array, value_obj)
        return True
    if dtype_code == ArrayDType.INT16.value:
        op[DType.int16](array, value_obj)
        return True
    if dtype_code == ArrayDType.INT8.value:
        op[DType.int8](array, value_obj)
        return True
    if dtype_code == ArrayDType.UINT64.value:
        op[DType.uint64](array, value_obj)
        return True
    if dtype_code == ArrayDType.UINT32.value:
        op[DType.uint32](array, value_obj)
        return True
    if dtype_code == ArrayDType.UINT16.value:
        op[DType.uint16](array, value_obj)
        return True
    if dtype_code == ArrayDType.UINT8.value:
        op[DType.uint8](array, value_obj)
        return True
    return False


def dispatch_real_write_from_py[
    op: PhysWriteFromPy,
](dtype_code: Int, mut array: Array, physical: Int, value_obj: PythonObject) raises -> Bool:
    if dtype_code == ArrayDType.FLOAT64.value:
        op[DType.float64](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.FLOAT32.value:
        op[DType.float32](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.FLOAT16.value:
        op[DType.float16](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.INT64.value:
        op[DType.int64](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.INT32.value:
        op[DType.int32](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.INT16.value:
        op[DType.int16](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.INT8.value:
        op[DType.int8](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.UINT64.value:
        op[DType.uint64](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.UINT32.value:
        op[DType.uint32](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.UINT16.value:
        op[DType.uint16](array, physical, value_obj)
        return True
    if dtype_code == ArrayDType.UINT8.value:
        op[DType.uint8](array, physical, value_obj)
        return True
    return False


def _scalar_from_py[dt: DType](value_obj: PythonObject) raises -> Scalar[dt]:
    """Narrows a PythonObject to `Scalar[dt]`. Floats go through `Float64(py=...)`,
    integers through `Int(py=...)` — matches the manual conversions in the
    pre-refactor switches."""
    comptime if dt.is_floating_point():
        return Scalar[dt](Float64(py=value_obj))
    return Scalar[dt](Int(py=value_obj))


def _contig_fill_from_py[dt: DType](mut array: Array, value_obj: PythonObject) raises:
    var value = _scalar_from_py[dt](value_obj)
    var ptr = contiguous_ptr[dt](array)
    for i in range(array.size_value):
        ptr[i] = value


def _phys_write_from_py[dt: DType](mut array: Array, physical: Int, value_obj: PythonObject) raises:
    set_physical[dt](array, physical, _scalar_from_py[dt](value_obj))


def get_physical_as_f64(array: Array, physical: Int) raises -> Float64:
    var c = array.dtype_code
    if c == ArrayDType.BOOL.value:
        return 1.0 if get_physical_bool(array, physical) else 0.0
    if c == ArrayDType.COMPLEX64.value:
        # Real part only; imag discarded. Matches numpy on real-valued
        # aggregations of complex arrays.
        return Float64(get_physical_c64_real(array, physical))
    if c == ArrayDType.COMPLEX128.value:
        return get_physical_c128_real(array, physical)
    return dispatch_real_to_f64[_phys_read_as_f64](c, array, physical)


def get_logical_as_f64(array: Array, logical: Int) raises -> Float64:
    return get_physical_as_f64(array, physical_offset(array, logical))


def set_physical_from_f64(mut array: Array, physical: Int, value: Float64) raises:
    var c = array.dtype_code
    if c == ArrayDType.BOOL.value:
        array.data[physical] = UInt8(1) if value != 0.0 else UInt8(0)
        return
    if c == ArrayDType.COMPLEX64.value:
        # Real part receives `value`; imag zeroed. Matches numpy's
        # f64-to-complex assignment.
        var ptr = array.data.bitcast[Float32]()
        ptr[physical * 2] = Float32(value)
        ptr[physical * 2 + 1] = 0.0
        return
    if c == ArrayDType.COMPLEX128.value:
        var ptr = array.data.bitcast[Float64]()
        ptr[physical * 2] = value
        ptr[physical * 2 + 1] = 0.0
        return
    if dispatch_real_write_f64[_phys_write_from_f64](c, array, physical, value):
        return
    # Unknown / not-yet-dispatched dtype: write as float64 (matches old fallback).
    set_physical[DType.float64](array, physical, value)


def set_logical_from_f64(mut array: Array, logical: Int, value: Float64) raises:
    set_physical_from_f64(array, physical_offset(array, logical), value)


def set_logical_from_i64(mut array: Array, logical: Int, value: Int64) raises:
    var physical = physical_offset(array, logical)
    if dispatch_int_write_i64[_phys_write_from_i64](array.dtype_code, array, physical, value):
        return
    # Non-int dtype: round-trip through f64 (preserves prior behavior).
    set_physical_from_f64(array, physical, Float64(value))


def set_logical_from_py(mut array: Array, logical: Int, value_obj: PythonObject) raises:
    var c = array.dtype_code
    var physical = physical_offset(array, logical)
    if c == ArrayDType.BOOL.value:
        array.data[physical] = UInt8(1) if Bool(py=value_obj) else UInt8(0)
        return
    if c == ArrayDType.COMPLEX64.value or c == ArrayDType.COMPLEX128.value:
        # Accept python complex or real numbers. Real → imaginary=0.
        var real_val = Float64(py=value_obj.real)
        var imag_val = Float64(py=value_obj.imag)
        if c == ArrayDType.COMPLEX64.value:
            var ptr = array.data.bitcast[Float32]()
            ptr[physical * 2] = Float32(real_val)
            ptr[physical * 2 + 1] = Float32(imag_val)
        else:
            var ptr = array.data.bitcast[Float64]()
            ptr[physical * 2] = real_val
            ptr[physical * 2 + 1] = imag_val
        return
    if dispatch_real_write_from_py[_phys_write_from_py](c, array, physical, value_obj):
        return
    # Unknown dtype: write as float64 (matches old fallback).
    set_physical[DType.float64](array, physical, Float64(py=value_obj))


def scalar_py_as_f64(value_obj: PythonObject, dtype_code: Int) raises -> Float64:
    if dtype_code == ArrayDType.BOOL.value:
        if Bool(py=value_obj):
            return 1.0
        return 0.0
    if (
        dtype_code == ArrayDType.INT64.value
        or dtype_code == ArrayDType.INT32.value
        or dtype_code == ArrayDType.INT16.value
        or dtype_code == ArrayDType.INT8.value
        or dtype_code == ArrayDType.UINT64.value
        or dtype_code == ArrayDType.UINT32.value
        or dtype_code == ArrayDType.UINT16.value
        or dtype_code == ArrayDType.UINT8.value
    ):
        return Float64(Int(py=value_obj))
    return Float64(py=value_obj)


def fill_all_from_py(mut array: Array, value_obj: PythonObject) raises:
    if is_c_contiguous(array):
        var c = array.dtype_code
        if c == ArrayDType.BOOL.value:
            var value = UInt8(1) if Bool(py=value_obj) else UInt8(0)
            var ptr = array.data + array.offset_elems
            for i in range(array.size_value):
                ptr[i] = value
            return
        if c == ArrayDType.COMPLEX64.value:
            var real = Float32(Float64(py=value_obj.real))
            var imag = Float32(Float64(py=value_obj.imag))
            var ptr = array.data.bitcast[Float32]() + array.offset_elems * 2
            for i in range(array.size_value):
                ptr[i * 2] = real
                ptr[i * 2 + 1] = imag
            return
        if c == ArrayDType.COMPLEX128.value:
            var real = Float64(py=value_obj.real)
            var imag = Float64(py=value_obj.imag)
            var ptr = array.data.bitcast[Float64]() + array.offset_elems * 2
            for i in range(array.size_value):
                ptr[i * 2] = real
                ptr[i * 2 + 1] = imag
            return
        if dispatch_real_contig_fill_from_py[_contig_fill_from_py](c, array, value_obj):
            return
    for i in range(array.size_value):
        set_logical_from_py(array, i, value_obj)


def contiguous_as_f64(array: Array, index: Int) raises -> Float64:
    if array.dtype_code == ArrayDType.FLOAT32.value:
        return Float64(contiguous_ptr[DType.float32](array)[index])
    if array.dtype_code == ArrayDType.FLOAT64.value:
        return contiguous_ptr[DType.float64](array)[index]
    return get_physical_as_f64(array, array.offset_elems + index)


def set_contiguous_from_f64(mut array: Array, index: Int, value: Float64) raises:
    if array.dtype_code == ArrayDType.FLOAT32.value:
        contiguous_ptr[DType.float32](array)[index] = Float32(value)
    elif array.dtype_code == ArrayDType.FLOAT64.value:
        contiguous_ptr[DType.float64](array)[index] = value
    else:
        set_physical_from_f64(array, array.offset_elems + index, value)
