from std.ffi import _get_dylib_function as _ffi_get_dylib_function
from std.ffi import _Global, OwnedDLHandle
from std.sys import CompilationTarget


# Apple backend plumbing stays isolated from ndarray semantics. Kernels import
# these two call helpers when a macOS-only fast path proves itself in benchmarks.
comptime LIB_ACC_PATH = (
    "/System/Library/Frameworks/Accelerate.framework/Accelerate"
)


comptime cblas_dgemm_type = def(
    _CBLASOrder,
    _CBLASTranspose,
    _CBLASTranspose,
    Int32,
    Int32,
    Int32,
    Float64,
    UnsafePointer[Float64, ImmutAnyOrigin],
    Int32,
    UnsafePointer[Float64, ImmutAnyOrigin],
    Int32,
    Float64,
    UnsafePointer[Float64, MutAnyOrigin],
    Int32,
) thin -> None

comptime cblas_sgemm_type = def(
    _CBLASOrder,
    _CBLASTranspose,
    _CBLASTranspose,
    Int32,
    Int32,
    Int32,
    Float32,
    UnsafePointer[Float32, ImmutAnyOrigin],
    Int32,
    UnsafePointer[Float32, ImmutAnyOrigin],
    Int32,
    Float32,
    UnsafePointer[Float32, MutAnyOrigin],
    Int32,
) thin -> None

comptime vv_f32_type = def(
    UnsafePointer[Float32, MutAnyOrigin],
    UnsafePointer[Float32, ImmutAnyOrigin],
    UnsafePointer[Int32, ImmutAnyOrigin],
) thin -> None


@fieldwise_init
struct _CBLASOrder(TrivialRegisterPassable):
    var value: Int32
    comptime ROW_MAJOR = _CBLASOrder(101)


@fieldwise_init
struct _CBLASTranspose(TrivialRegisterPassable):
    var value: Int32
    comptime NO_TRANSPOSE = _CBLASTranspose(111)


comptime MONPY_APPLE_ACCELERATE = _Global[
    "MONPY_APPLE_ACCELERATE",
    init_accelerate_dylib,
    on_error_msg=accelerate_error_msg,
]


def accelerate_error_msg() -> Error:
    return Error("cannot find Apple Accelerate at ", LIB_ACC_PATH)


def init_accelerate_dylib() -> OwnedDLHandle:
    try:
        return OwnedDLHandle(LIB_ACC_PATH)
    except:
        return OwnedDLHandle(unsafe_uninitialized=True)


@always_inline
def get_accelerate_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    comptime assert (
        CompilationTarget.is_macos()
    ), "Apple Accelerate requires macOS"
    return _ffi_get_dylib_function[
        MONPY_APPLE_ACCELERATE(),
        func_name,
        result_type,
    ]()


@always_inline
def get_cblas_f64_function() raises -> cblas_dgemm_type:
    return get_accelerate_function["cblas_dgemm", cblas_dgemm_type]()


@always_inline
def get_cblas_f32_function() raises -> cblas_sgemm_type:
    return get_accelerate_function["cblas_sgemm", cblas_sgemm_type]()


@always_inline
def call_vv_f32[
    func_name: StaticString
](
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    src_ptr: UnsafePointer[Float32, MutExternalOrigin],
    count_value: Int,
) raises:
    var function = get_accelerate_function[func_name, vv_f32_type]()
    var count = Int32(count_value)
    function(
        rebind[UnsafePointer[Float32, MutAnyOrigin]](out_ptr),
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](src_ptr),
        rebind[UnsafePointer[Int32, ImmutAnyOrigin]](UnsafePointer(to=count)),
    )


@always_inline
def cblas_sgemm_row_major(
    m: Int,
    n: Int,
    k: Int,
    c_ptr: UnsafePointer[Float32, MutExternalOrigin],
    a_ptr: UnsafePointer[Float32, MutExternalOrigin],
    b_ptr: UnsafePointer[Float32, MutExternalOrigin],
) raises:
    var sgemm = get_cblas_f32_function()
    sgemm(
        _CBLASOrder.ROW_MAJOR,
        _CBLASTranspose.NO_TRANSPOSE,
        _CBLASTranspose.NO_TRANSPOSE,
        Int32(m),
        Int32(n),
        Int32(k),
        1.0,
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](a_ptr),
        Int32(k),
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](b_ptr),
        Int32(n),
        0.0,
        rebind[UnsafePointer[Float32, MutAnyOrigin]](c_ptr),
        Int32(n),
    )


@always_inline
def cblas_dgemm_row_major(
    m: Int,
    n: Int,
    k: Int,
    c_ptr: UnsafePointer[Float64, MutExternalOrigin],
    a_ptr: UnsafePointer[Float64, MutExternalOrigin],
    b_ptr: UnsafePointer[Float64, MutExternalOrigin],
) raises:
    var dgemm = get_cblas_f64_function()
    dgemm(
        _CBLASOrder.ROW_MAJOR,
        _CBLASTranspose.NO_TRANSPOSE,
        _CBLASTranspose.NO_TRANSPOSE,
        Int32(m),
        Int32(n),
        Int32(k),
        1.0,
        rebind[UnsafePointer[Float64, ImmutAnyOrigin]](a_ptr),
        Int32(k),
        rebind[UnsafePointer[Float64, ImmutAnyOrigin]](b_ptr),
        Int32(n),
        0.0,
        rebind[UnsafePointer[Float64, MutAnyOrigin]](c_ptr),
        Int32(n),
    )
