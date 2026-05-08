"""Array struct + pymethods + small read/write primitives.

The Array data type lives here (not in `__init__.mojo`) so sibling
sub-modules (factory, dispatch, cast, result_dtypes) can import it via
relative `from .accessors import Array` without creating a cycle through
the package's `__init__.mojo`. The pymethods stay attached to Array
because `lib.mojo`'s `def_method("Array", ..., Array.method_py)` requires
them as `@staticmethod` members of the struct.

Hosts:
  - `Array` struct + 16 `*_py` static methods.
  - Tiny shape/contig probes the pymethods call (`item_size`,
    `is_c_contiguous`, `is_f_contiguous`, `has_negative_strides`,
    `has_zero_strides`, `is_linearly_addressable`, `physical_offset`,
    `validate_shape`, `shape_size`, `make_c_strides`, `clone_int_list`,
    `same_shape`, `slice_length`).
  - Parametric leaf accessors (`contiguous_ptr[dt]`, `get_physical[dt]`,
    `set_physical[dt]`, `get_physical_bool`, complex `get_physical_c*` /
    `set_physical_c*`).

Bigger logic — factories, runtime dispatch, casting/copying, dtype
result helpers — lives in sibling modules.
"""

from std.collections import List
from std.os import abort
from std.python import Python, PythonObject
from std.utils.numerics import nan

from domain import (
    ArrayDType,
    BackendKind,
    dtype_byte_offset,
    dtype_item_size,
)
from storage import Storage, release_storage


@fieldwise_init
struct Array(Movable, Writable):
    var dtype_code: Int
    var shape: List[Int]
    var strides: List[Int]
    var size_value: Int
    var offset_elems: Int
    var storage: UnsafePointer[Storage, MutExternalOrigin]
    var data: UnsafePointer[UInt8, MutExternalOrigin]
    var byte_len: Int
    var backend_code: Int

    @staticmethod
    def _get_self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("Array method receiver had the wrong type: ", e))

    def __del__(deinit self):
        release_storage(self.storage)

    @staticmethod
    def dtype_code_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].dtype_code)

    @staticmethod
    def ndim_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(len(self_ptr[].shape))

    @staticmethod
    def size_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].size_value)

    @staticmethod
    def shape_at_py(py_self: PythonObject, index_obj: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        var index = Int(py=index_obj)
        return PythonObject(self_ptr[].shape[index])

    @staticmethod
    def stride_at_py(py_self: PythonObject, index_obj: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        var index = Int(py=index_obj)
        return PythonObject(self_ptr[].strides[index])

    @staticmethod
    def item_size_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(item_size(self_ptr[].dtype_code))

    @staticmethod
    def data_address_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        var byte_offset = dtype_byte_offset(self_ptr[].dtype_code, self_ptr[].offset_elems)
        return PythonObject((self_ptr[].data + byte_offset).__int__())

    @staticmethod
    def is_c_contiguous_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(is_c_contiguous(self_ptr[]))

    @staticmethod
    def is_f_contiguous_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(is_f_contiguous(self_ptr[]))

    @staticmethod
    def has_negative_strides_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(has_negative_strides(self_ptr[]))

    @staticmethod
    def has_zero_strides_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(has_zero_strides(self_ptr[]))

    @staticmethod
    def storage_refcount_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].storage[].ref_count)

    @staticmethod
    def used_accelerate_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].backend_code == BackendKind.ACCELERATE.value)

    @staticmethod
    def used_fused_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].backend_code == BackendKind.FUSED.value)

    @staticmethod
    def backend_code_py(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].backend_code)

    @staticmethod
    def get_scalar_py(py_self: PythonObject, index_obj: PythonObject) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        var index = Int(py=index_obj)
        if self_ptr[].dtype_code == ArrayDType.BOOL.value:
            return PythonObject(get_physical_bool(self_ptr[], physical_offset(self_ptr[], index)))
        if self_ptr[].dtype_code == ArrayDType.INT64.value:
            return PythonObject(get_physical[DType.int64](self_ptr[], physical_offset(self_ptr[], index)))
        if self_ptr[].dtype_code == ArrayDType.INT32.value:
            return PythonObject(Int(get_physical[DType.int32](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.INT16.value:
            return PythonObject(Int(get_physical[DType.int16](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.INT8.value:
            return PythonObject(Int(get_physical[DType.int8](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.UINT64.value:
            return PythonObject(Int(get_physical[DType.uint64](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.UINT32.value:
            return PythonObject(Int(get_physical[DType.uint32](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.UINT16.value:
            return PythonObject(Int(get_physical[DType.uint16](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.UINT8.value:
            return PythonObject(Int(get_physical[DType.uint8](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.FLOAT32.value:
            return PythonObject(get_physical[DType.float32](self_ptr[], physical_offset(self_ptr[], index)))
        if self_ptr[].dtype_code == ArrayDType.FLOAT16.value:
            return PythonObject(Float64(get_physical[DType.float16](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.BFLOAT16.value:
            return PythonObject(Float64(get_physical[DType.bfloat16](self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.FLOAT8_E4M3FN.value:
            return PythonObject(
                Float64(get_physical[DType.float8_e4m3fn](self_ptr[], physical_offset(self_ptr[], index)))
            )
        if self_ptr[].dtype_code == ArrayDType.FLOAT8_E4M3FNUZ.value:
            return PythonObject(
                Float64(get_physical[DType.float8_e4m3fnuz](self_ptr[], physical_offset(self_ptr[], index)))
            )
        if self_ptr[].dtype_code == ArrayDType.FLOAT8_E5M2.value:
            return PythonObject(
                Float64(get_physical[DType.float8_e5m2](self_ptr[], physical_offset(self_ptr[], index)))
            )
        if self_ptr[].dtype_code == ArrayDType.FLOAT8_E5M2FNUZ.value:
            return PythonObject(
                Float64(get_physical[DType.float8_e5m2fnuz](self_ptr[], physical_offset(self_ptr[], index)))
            )
        if self_ptr[].dtype_code == ArrayDType.FLOAT8_E8M0FNU.value:
            return PythonObject(get_physical_e8m0fnu(self_ptr[], physical_offset(self_ptr[], index)))
        if self_ptr[].dtype_code == ArrayDType.FLOAT4_E2M1FN.value:
            return PythonObject(Float64(get_physical_fp4_e2m1fn(self_ptr[], physical_offset(self_ptr[], index))))
        if self_ptr[].dtype_code == ArrayDType.COMPLEX64.value:
            var phys = physical_offset(self_ptr[], index)
            var real = Float64(get_physical_c64_real(self_ptr[], phys))
            var imag = Float64(get_physical_c64_imag(self_ptr[], phys))
            var builtins = Python.import_module("builtins")
            return builtins.complex(PythonObject(real), PythonObject(imag))
        if self_ptr[].dtype_code == ArrayDType.COMPLEX128.value:
            var phys = physical_offset(self_ptr[], index)
            var real = get_physical_c128_real(self_ptr[], phys)
            var imag = get_physical_c128_imag(self_ptr[], phys)
            var builtins = Python.import_module("builtins")
            return builtins.complex(PythonObject(real), PythonObject(imag))
        return PythonObject(get_physical[DType.float64](self_ptr[], physical_offset(self_ptr[], index)))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Array(dtype_code=")
        writer.write(self.dtype_code)
        writer.write(", shape=")
        writer.write(self.shape)
        writer.write(")")


def item_size(dtype_code: Int) raises -> Int:
    return dtype_item_size(dtype_code)


def validate_shape(shape: List[Int]) raises:
    for i in range(len(shape)):
        if shape[i] < 0:
            raise Error("shape dimensions must be non-negative")


def shape_size(shape: List[Int]) raises -> Int:
    validate_shape(shape)
    var total = 1
    for i in range(len(shape)):
        total *= shape[i]
    return total


def make_c_strides(shape: List[Int]) raises -> List[Int]:
    var strides = List[Int]()
    for _ in range(len(shape)):
        strides.append(0)
    var stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = stride
        stride *= shape[axis]
    return strides^


def clone_int_list(values: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(values)):
        out.append(values[i])
    return out^


def same_shape(lhs: List[Int], rhs: List[Int]) -> Bool:
    if len(lhs) != len(rhs):
        return False
    for i in range(len(lhs)):
        if lhs[i] != rhs[i]:
            return False
    return True


def slice_length(dim: Int, start: Int, stop: Int, step: Int) raises -> Int:
    if step == 0:
        raise Error("slice step cannot be zero")
    if step > 0:
        if start >= stop:
            return 0
        return (stop - start + step - 1) // step
    var negative_step = -step
    if start <= stop:
        return 0
    return (start - stop + negative_step - 1) // negative_step


def is_c_contiguous(array: Array) raises -> Bool:
    var expected = 1
    for axis in range(len(array.shape) - 1, -1, -1):
        if array.shape[axis] == 0:
            return True
        if array.shape[axis] != 1 and array.strides[axis] != expected:
            return False
        expected *= array.shape[axis]
    return True


def is_f_contiguous(array: Array) raises -> Bool:
    var expected = 1
    for axis in range(len(array.shape)):
        if array.shape[axis] == 0:
            return True
        if array.shape[axis] != 1 and array.strides[axis] != expected:
            return False
        expected *= array.shape[axis]
    return True


def has_negative_strides(array: Array) -> Bool:
    for axis in range(len(array.strides)):
        if array.strides[axis] < 0:
            return True
    return False


def has_zero_strides(array: Array) -> Bool:
    for axis in range(len(array.strides)):
        if array.strides[axis] == 0:
            return True
    return False


def is_linearly_addressable(array: Array) raises -> Bool:
    """True if the physical-offset span equals `size_value - 1` — the
    array's elements occupy a contiguous block of `size_value` slots,
    and a flat scan `data[offset_elems .. offset_elems + size_value)`
    visits every element exactly once. Holds for any permutation of
    c/f-contiguous storage (transpose, swapaxes, etc.).
    For commutative reductions (sum/min/max/prod), this lets us skip
    the LayoutIter walk and stride contig — a big win when the logical
    iteration order has a non-unit innermost stride."""
    if array.size_value <= 1:
        return True
    if has_zero_strides(array) or has_negative_strides(array):
        return False
    var max_off = 0
    for axis in range(len(array.shape)):
        var d = array.shape[axis]
        var s = array.strides[axis]
        if d == 0:
            return False
        max_off += (d - 1) * s
    return max_off + 1 == array.size_value


def physical_offset(array: Array, logical: Int) raises -> Int:
    if logical < 0 or logical >= array.size_value:
        raise Error("array index out of bounds")
    var physical = array.offset_elems
    var remainder = logical
    for axis in range(len(array.shape) - 1, -1, -1):
        var dim = array.shape[axis]
        if dim != 0:
            var coord = remainder % dim
            remainder = remainder // dim
            physical += coord * array.strides[axis]
    return physical


# === Parametric leaf accessors ==========================================
#
# `contiguous_ptr[dt]` / `get_physical[dt]` / `set_physical[dt]` are the
# single-source-of-truth typed leaf accessors keyed on Mojo's `DType`.
# Bool stays bespoke (`get_physical_bool`) — single-byte storage with
# nonzero-is-true semantics doesn't round-trip cleanly through
# `Scalar[DType.bool]`. Complex stays bespoke — interleaved (re, im)
# pairs have no `DType` representation.


def contiguous_ptr[dt: DType](array: Array) -> UnsafePointer[Scalar[dt], MutExternalOrigin]:
    return array.data.bitcast[Scalar[dt]]() + array.offset_elems


def get_physical[dt: DType](array: Array, physical: Int) -> Scalar[dt]:
    return array.data.bitcast[Scalar[dt]]()[physical]


def set_physical[dt: DType](mut array: Array, physical: Int, value: Scalar[dt]):
    array.data.bitcast[Scalar[dt]]()[physical] = value


def get_physical_bool(array: Array, physical: Int) raises -> Bool:
    return array.data[physical] != UInt8(0)


def _pow2_f64(exp: Int) -> Float64:
    var value = 1.0
    if exp >= 0:
        var i = 0
        while i < exp:
            value = value * 2.0
            i += 1
    else:
        var i = 0
        while i < -exp:
            value = value * 0.5
            i += 1
    return value


def get_physical_e8m0fnu(array: Array, physical: Int) raises -> Float64:
    var bits = Int(array.data[physical])
    if bits == 255:
        return Float64(nan[DType.float64]())
    return _pow2_f64(bits - 127)


def _encode_e8m0fnu(value: Float64) -> UInt8:
    if value != value:
        return UInt8(255)
    if value <= 0.0:
        return UInt8(0)
    var scaled = value
    var exp = 0
    while scaled >= 2.0 and exp < 127:
        scaled = scaled * 0.5
        exp += 1
    while scaled < 1.0 and exp > -127:
        scaled = scaled * 2.0
        exp -= 1
    if scaled >= 1.4142135623730951 and exp < 127:
        exp += 1
    if exp >= 127:
        return UInt8(254)
    if exp <= -127:
        return UInt8(0)
    return UInt8(exp + 127)


def set_physical_e8m0fnu(mut array: Array, physical: Int, value: Float64) raises:
    array.data[physical] = _encode_e8m0fnu(value)


def _fp4_nibble_to_f32(code: Int) -> Float32:
    if code == 0:
        return 0.0
    if code == 1:
        return 0.5
    if code == 2:
        return 1.0
    if code == 3:
        return 1.5
    if code == 4:
        return 2.0
    if code == 5:
        return 3.0
    if code == 6:
        return 4.0
    if code == 7:
        return 6.0
    if code == 8:
        return -0.0
    if code == 9:
        return -0.5
    if code == 10:
        return -1.0
    if code == 11:
        return -1.5
    if code == 12:
        return -2.0
    if code == 13:
        return -3.0
    if code == 14:
        return -4.0
    return -6.0


def get_physical_fp4_e2m1fn(array: Array, physical: Int) raises -> Float32:
    var packed = array.data[physical // 2]
    var nibble = packed & UInt8(0x0F)
    if physical % 2 != 0:
        nibble = (packed >> UInt8(4)) & UInt8(0x0F)
    return _fp4_nibble_to_f32(Int(nibble))


def _encode_fp4_e2m1fn(value: Float64) -> UInt8:
    var negative = value < 0.0
    var x = -value if negative else value
    var code: Int
    if x <= 0.25:
        code = 0
    elif x < 0.75:
        code = 1
    elif x <= 1.25:
        code = 2
    elif x < 1.75:
        code = 3
    elif x <= 2.5:
        code = 4
    elif x < 3.5:
        code = 5
    elif x <= 5.0:
        code = 6
    else:
        code = 7
    if negative:
        code += 8
    return UInt8(code)


def set_physical_fp4_e2m1fn(mut array: Array, physical: Int, value: Float64) raises:
    var byte_index = physical // 2
    var code = _encode_fp4_e2m1fn(value)
    var current = array.data[byte_index]
    if physical % 2 == 0:
        array.data[byte_index] = (current & UInt8(0xF0)) | code
    else:
        array.data[byte_index] = (current & UInt8(0x0F)) | (code << UInt8(4))


# Complex accessors: arrays store interleaved (real, imag) pairs.
# complex64 → 2 × Float32 starting at byte index `physical * 8`.
# complex128 → 2 × Float64 starting at byte index `physical * 16`.
def get_physical_c64_real(array: Array, physical: Int) raises -> Float32:
    return array.data.bitcast[Float32]()[physical * 2]


def get_physical_c64_imag(array: Array, physical: Int) raises -> Float32:
    return array.data.bitcast[Float32]()[physical * 2 + 1]


def get_physical_c128_real(array: Array, physical: Int) raises -> Float64:
    return array.data.bitcast[Float64]()[physical * 2]


def get_physical_c128_imag(array: Array, physical: Int) raises -> Float64:
    return array.data.bitcast[Float64]()[physical * 2 + 1]


def set_physical_c64(mut array: Array, physical: Int, real: Float32, imag: Float32) raises:
    var ptr = array.data.bitcast[Float32]()
    ptr[physical * 2] = real
    ptr[physical * 2 + 1] = imag


def set_physical_c128(mut array: Array, physical: Int, real: Float64, imag: Float64) raises:
    var ptr = array.data.bitcast[Float64]()
    ptr[physical * 2] = real
    ptr[physical * 2 + 1] = imag
