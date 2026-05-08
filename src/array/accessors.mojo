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

from domain import (
    ArrayDType,
    BackendKind,
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
        var byte_offset = self_ptr[].offset_elems * item_size(self_ptr[].dtype_code)
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
