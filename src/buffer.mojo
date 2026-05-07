from std.collections import List
from std.ffi import c_char, c_int, _CPointer
from std.memory.unsafe_pointer import unsafe_cast
from std.python import PythonObject
from std.python._cpython import ExternalFunction, Py_ssize_t, PyObjectPtr
from std.python.python import Python

from array import (
    Array,
    copy_c_contiguous,
    make_external_array,
)
from domain import (
    dtype_code_from_format_char,
)


# CPython Py_buffer struct (Python 3.11+ layout). Fields and sizes match
# the C definition in Include/cpython/object.h. On 64-bit systems the
# struct is 80 bytes; readonly + ndim share an 8-byte slot via natural
# alignment so the explicit field order below matches the C layout.
@fieldwise_init
struct Py_buffer(Defaultable):
    var buf: _CPointer[UInt8, MutAnyOrigin]
    var obj: PyObjectPtr
    var len: Py_ssize_t
    var itemsize: Py_ssize_t
    var readonly: c_int
    var ndim: c_int
    var format: _CPointer[c_char, MutAnyOrigin]
    var shape: _CPointer[Py_ssize_t, MutAnyOrigin]
    var strides: _CPointer[Py_ssize_t, MutAnyOrigin]
    var suboffsets: _CPointer[Py_ssize_t, MutAnyOrigin]
    var internal: _CPointer[UInt8, MutAnyOrigin]

    def __init__(out self):
        self.buf = {}
        self.obj = {}
        self.len = 0
        self.itemsize = 0
        self.readonly = 0
        self.ndim = 0
        self.format = {}
        self.shape = {}
        self.strides = {}
        self.suboffsets = {}
        self.internal = {}


# Buffer protocol flags (Python.h)
comptime PyBUF_SIMPLE: c_int = 0
comptime PyBUF_WRITABLE: c_int = 1
comptime PyBUF_FORMAT: c_int = 4
comptime PyBUF_ND: c_int = 8
comptime PyBUF_STRIDES: c_int = 0x10 | PyBUF_ND
comptime PyBUF_RECORDS_RO: c_int = PyBUF_STRIDES | PyBUF_FORMAT


# int PyObject_GetBuffer(PyObject *exporter, Py_buffer *view, int flags)
comptime PyObject_GetBuffer = ExternalFunction[
    "PyObject_GetBuffer",
    def(
        PyObjectPtr, _CPointer[Py_buffer, MutAnyOrigin], c_int
    ) thin -> c_int,
]

# void PyBuffer_Release(Py_buffer *view)
comptime PyBuffer_Release = ExternalFunction[
    "PyBuffer_Release",
    def(_CPointer[Py_buffer, MutAnyOrigin]) thin -> None,
]


def asarray_from_buffer_ops(
    obj: PythonObject,
    requested_dtype_obj: PythonObject,
    copy_obj: PythonObject,
) raises -> PythonObject:
    # Single-FFI numpy / buffer-protocol bridge. One `PyObject_GetBuffer`
    # call replaces the eight-step `__array_interface__` walk in
    # `asarray_from_numpy_ops`, dropping the per-array marshaling cost
    # from ~700–1000 ns to ~150 ns.
    #
    # `requested_dtype_obj` carries the target dtype code or -1 to mean
    # "use whatever the source is". `copy_obj` is the tri-state copy flag:
    # 0 = never, 1 = always, -1 = numpy's default (copy on readonly only).
    var requested_code = Int(py=requested_dtype_obj)
    var copy_flag = Int(py=copy_obj)
    var view = Py_buffer()
    var view_ptr = UnsafePointer(to=view).as_any_origin()
    ref cpy = Python().cpython()
    var get_buffer_fn = PyObject_GetBuffer.load(cpy.lib.borrow())
    var release_fn = PyBuffer_Release.load(cpy.lib.borrow())
    var rc = get_buffer_fn(obj._obj_ptr, view_ptr, PyBUF_RECORDS_RO)
    if Int(rc) != 0:
        raise Error("PyObject_GetBuffer failed")
    # Read fields off the filled-in view. `_CPointer` is `Optional[UnsafePointer]`,
    # so each pointer field is unwrapped via `.value()`. The data pointer,
    # shape, and strides remain valid as long as the source object lives;
    # for numpy arrays that's guaranteed by the python-side `_owner` slot.
    # Release runs below so numpy's buffer-lock counter is balanced.
    var byte_len = Int(view.len)
    var item_bytes = Int(view.itemsize)
    var readonly = Int(view.readonly) != 0
    var ndim = Int(view.ndim)
    if not view.buf:
        release_fn(view_ptr)
        raise Error("buffer view has null data pointer")
    var data_addr = Int(view.buf.value())
    var format_char_value = 0
    if view.format:
        format_char_value = Int(view.format.value()[0])
    var shape = List[Int]()
    var elem_strides = List[Int]()
    if ndim != 0:
        if not view.shape or not view.strides:
            release_fn(view_ptr)
            raise Error("buffer view missing shape or strides")
        var shape_ptr = view.shape.value()
        var strides_ptr = view.strides.value()
        for i in range(ndim):
            shape.append(Int(shape_ptr[i]))
            var byte_stride = Int(strides_ptr[i])
            if byte_stride % item_bytes != 0:
                release_fn(view_ptr)
                raise Error("buffer strides must align to itemsize")
            elem_strides.append(byte_stride // item_bytes)
    # Resolve dtype before releasing so we know whether to copy. dtype
    # mismatch with `requested_code` is a punt back to python; full
    # promotion is handled there via astype.
    var src_dtype_code = dtype_code_from_format_char(
        format_char_value, item_bytes
    )
    release_fn(view_ptr)
    if requested_code >= 0 and requested_code != src_dtype_code:
        raise Error(
            "asarray_from_buffer: dtype conversion handled in python"
        )
    if copy_flag == 0 and readonly:
        raise Error("readonly array requires copy=True")
    var must_copy = (copy_flag == 1) or readonly
    var data = UnsafePointer[UInt8, MutExternalOrigin](
        unsafe_from_address=data_addr
    )
    if must_copy:
        var view_shape = List[Int]()
        for i in range(len(shape)):
            view_shape.append(shape[i])
        var view_strides = List[Int]()
        for i in range(len(elem_strides)):
            view_strides.append(elem_strides[i])
        var external = make_external_array(
            src_dtype_code,
            view_shape^,
            view_strides^,
            0,
            data,
            byte_len,
        )
        var result = copy_c_contiguous(external)
        return PythonObject(alloc=result^)
    var result = make_external_array(
        src_dtype_code, shape^, elem_strides^, 0, data, byte_len
    )
    return PythonObject(alloc=result^)
