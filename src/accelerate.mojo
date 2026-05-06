from std.ffi import _get_dylib_function as _ffi_get_dylib_function
from std.ffi import _Global, OwnedDLHandle
from std.sys import CompilationTarget


# BLAS / LAPACK backend plumbing kept isolated from ndarray semantics. The
# resolved dylib differs per OS: on macOS we hit Apple's Accelerate
# framework; on Linux we try OpenBLAS first (most common, fastest), then
# fall back to the netlib reference libraries. cblas_* and *gesv_/*getrf_
# function names are identical across both, so the call helpers below are
# platform-agnostic — only the dylib resolution branches.
comptime LIB_ACC_PATH = (
    "/System/Library/Frameworks/Accelerate.framework/Accelerate"
)

# Linux dylib candidates, tried in order in `init_*_dylib`. OpenBLAS is the
# de-facto standard on Linux distros (Ubuntu, Debian, Fedora, Arch all ship
# it as the default BLAS provider when scipy/numpy are installed). The
# unversioned `.so` links are dev packages; the versioned `.so.0` / `.so.3`
# symlinks ship with the runtime package and are what numpy/scipy load too.
comptime LIB_OPENBLAS_PATH_0 = "libopenblas.so.0"
comptime LIB_OPENBLAS_PATH = "libopenblas.so"
comptime LIB_BLAS_PATH_3 = "libblas.so.3"
comptime LIB_BLAS_PATH = "libblas.so"
comptime LIB_LAPACK_PATH_3 = "liblapack.so.3"
comptime LIB_LAPACK_PATH = "liblapack.so"


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

comptime cblas_dgemv_type = def(
    _CBLASOrder,
    _CBLASTranspose,
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

comptime cblas_sgemv_type = def(
    _CBLASOrder,
    _CBLASTranspose,
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

comptime vv_f64_type = def(
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Float64, ImmutAnyOrigin],
    UnsafePointer[Int32, ImmutAnyOrigin],
) thin -> None

comptime lapack_sgesv_type = def(
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
) thin -> None

comptime lapack_dgesv_type = def(
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
) thin -> None

comptime lapack_sgetrf_type = def(
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
) thin -> None

comptime lapack_dgetrf_type = def(
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
) thin -> None


@fieldwise_init
struct _CBLASOrder(TrivialRegisterPassable):
    var value: Int32
    comptime ROW_MAJOR = _CBLASOrder(101)


@fieldwise_init
struct _CBLASTranspose(TrivialRegisterPassable):
    var value: Int32
    comptime NO_TRANSPOSE = _CBLASTranspose(111)
    comptime TRANSPOSE = _CBLASTranspose(112)


comptime MONPY_BLAS_DYLIB = _Global[
    "MONPY_BLAS_DYLIB",
    init_blas_dylib,
    on_error_msg=blas_error_msg,
]

comptime MONPY_LAPACK_DYLIB = _Global[
    "MONPY_LAPACK_DYLIB",
    init_lapack_dylib,
    on_error_msg=lapack_error_msg,
]


def blas_error_msg() -> Error:
    comptime if CompilationTarget.is_macos():
        return Error("cannot find Apple Accelerate at ", LIB_ACC_PATH)
    elif CompilationTarget.is_linux():
        return Error(
            "cannot find OpenBLAS / libblas on Linux. install with `apt"
            " install libopenblas-dev` or equivalent."
        )
    return Error("BLAS backend not configured for this platform")


def lapack_error_msg() -> Error:
    comptime if CompilationTarget.is_macos():
        return Error("cannot find Apple Accelerate at ", LIB_ACC_PATH)
    elif CompilationTarget.is_linux():
        return Error(
            "cannot find LAPACK on Linux. install with `apt install"
            " liblapack-dev` or use OpenBLAS (which bundles LAPACK)."
        )
    return Error("LAPACK backend not configured for this platform")


def init_blas_dylib() -> OwnedDLHandle:
    # macOS: Accelerate.framework. Linux: try OpenBLAS first (bundles
    # cblas + lapack and is what numpy/scipy use), then fall back to the
    # netlib reference libblas. Each `try` short-circuits on first success.
    comptime if CompilationTarget.is_macos():
        try:
            return OwnedDLHandle(LIB_ACC_PATH)
        except:
            return OwnedDLHandle(unsafe_uninitialized=True)
    elif CompilationTarget.is_linux():
        try:
            return OwnedDLHandle(LIB_OPENBLAS_PATH_0)
        except:
            pass
        try:
            return OwnedDLHandle(LIB_OPENBLAS_PATH)
        except:
            pass
        try:
            return OwnedDLHandle(LIB_BLAS_PATH_3)
        except:
            pass
        try:
            return OwnedDLHandle(LIB_BLAS_PATH)
        except:
            pass
    return OwnedDLHandle(unsafe_uninitialized=True)


def init_lapack_dylib() -> OwnedDLHandle:
    # macOS Accelerate exposes both BLAS and LAPACK from the same dylib.
    # Linux: prefer netlib liblapack if installed, then fall back to
    # OpenBLAS which also ships LAPACK symbols.
    comptime if CompilationTarget.is_macos():
        try:
            return OwnedDLHandle(LIB_ACC_PATH)
        except:
            return OwnedDLHandle(unsafe_uninitialized=True)
    elif CompilationTarget.is_linux():
        try:
            return OwnedDLHandle(LIB_LAPACK_PATH_3)
        except:
            pass
        try:
            return OwnedDLHandle(LIB_LAPACK_PATH)
        except:
            pass
        try:
            return OwnedDLHandle(LIB_OPENBLAS_PATH_0)
        except:
            pass
        try:
            return OwnedDLHandle(LIB_OPENBLAS_PATH)
        except:
            pass
    return OwnedDLHandle(unsafe_uninitialized=True)


@always_inline
def is_blas_available() -> Bool:
    # True when we have a path to call into BLAS / LAPACK from this build.
    # Currently macOS (Accelerate) and Linux (OpenBLAS / netlib) are
    # supported; other platforms fall back to the SIMD / scalar kernels.
    comptime if CompilationTarget.is_macos():
        return True
    elif CompilationTarget.is_linux():
        return True
    return False


@always_inline
def get_blas_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    comptime assert (
        CompilationTarget.is_macos() or CompilationTarget.is_linux()
    ), "BLAS backend requires macOS or Linux"
    return _ffi_get_dylib_function[
        MONPY_BLAS_DYLIB(),
        func_name,
        result_type,
    ]()


@always_inline
def get_lapack_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    comptime assert (
        CompilationTarget.is_macos() or CompilationTarget.is_linux()
    ), "LAPACK backend requires macOS or Linux"
    return _ffi_get_dylib_function[
        MONPY_LAPACK_DYLIB(),
        func_name,
        result_type,
    ]()


# Backward-compat alias: existing call sites use `get_accelerate_function`
# generically for both BLAS and LAPACK names. macOS keeps both in one
# dylib (Accelerate), so this routes to BLAS_DYLIB which is identical
# there. On Linux, callers should prefer the explicit BLAS / LAPACK
# helpers below; this alias preserves source compatibility.
@always_inline
def get_accelerate_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return get_blas_function[func_name, result_type]()


@always_inline
def get_cblas_f64_function() raises -> cblas_dgemm_type:
    return get_accelerate_function["cblas_dgemm", cblas_dgemm_type]()


@always_inline
def get_cblas_f32_function() raises -> cblas_sgemm_type:
    return get_accelerate_function["cblas_sgemm", cblas_sgemm_type]()


@always_inline
def get_cblas_gemv_f64_function() raises -> cblas_dgemv_type:
    return get_accelerate_function["cblas_dgemv", cblas_dgemv_type]()


@always_inline
def get_cblas_gemv_f32_function() raises -> cblas_sgemv_type:
    return get_accelerate_function["cblas_sgemv", cblas_sgemv_type]()


@always_inline
def get_lapack_sgesv_function() raises -> lapack_sgesv_type:
    return get_lapack_function["sgesv_", lapack_sgesv_type]()


@always_inline
def get_lapack_dgesv_function() raises -> lapack_dgesv_type:
    return get_lapack_function["dgesv_", lapack_dgesv_type]()


@always_inline
def get_lapack_sgetrf_function() raises -> lapack_sgetrf_type:
    return get_lapack_function["sgetrf_", lapack_sgetrf_type]()


@always_inline
def get_lapack_dgetrf_function() raises -> lapack_dgetrf_type:
    return get_lapack_function["dgetrf_", lapack_dgetrf_type]()


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
def call_vv_f64[
    func_name: StaticString
](
    out_ptr: UnsafePointer[Float64, MutExternalOrigin],
    src_ptr: UnsafePointer[Float64, MutExternalOrigin],
    count_value: Int,
) raises:
    var function = get_accelerate_function[func_name, vv_f64_type]()
    var count = Int32(count_value)
    function(
        rebind[UnsafePointer[Float64, MutAnyOrigin]](out_ptr),
        rebind[UnsafePointer[Float64, ImmutAnyOrigin]](src_ptr),
        rebind[UnsafePointer[Int32, ImmutAnyOrigin]](UnsafePointer(to=count)),
    )


@always_inline
def lapack_sgesv(
    n_value: Int,
    rhs_columns_value: Int,
    a_ptr: UnsafePointer[Float32, MutExternalOrigin],
    pivot_ptr: UnsafePointer[Int32, MutExternalOrigin],
    b_ptr: UnsafePointer[Float32, MutExternalOrigin],
) raises -> Int:
    var function = get_lapack_sgesv_function()
    var n = Int32(n_value)
    var rhs_columns = Int32(rhs_columns_value)
    var leading_a = Int32(n_value)
    var leading_b = Int32(n_value)
    var info = Int32(0)
    function(
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](
            UnsafePointer(to=rhs_columns)
        ),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=leading_a)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](pivot_ptr),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](b_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=leading_b)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    return Int(info)


@always_inline
def lapack_dgesv(
    n_value: Int,
    rhs_columns_value: Int,
    a_ptr: UnsafePointer[Float64, MutExternalOrigin],
    pivot_ptr: UnsafePointer[Int32, MutExternalOrigin],
    b_ptr: UnsafePointer[Float64, MutExternalOrigin],
) raises -> Int:
    var function = get_lapack_dgesv_function()
    var n = Int32(n_value)
    var rhs_columns = Int32(rhs_columns_value)
    var leading_a = Int32(n_value)
    var leading_b = Int32(n_value)
    var info = Int32(0)
    function(
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](
            UnsafePointer(to=rhs_columns)
        ),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=leading_a)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](pivot_ptr),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](b_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=leading_b)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    return Int(info)


@always_inline
def lapack_sgetrf(
    n_value: Int,
    a_ptr: UnsafePointer[Float32, MutExternalOrigin],
    pivot_ptr: UnsafePointer[Int32, MutExternalOrigin],
) raises -> Int:
    var function = get_lapack_sgetrf_function()
    var rows = Int32(n_value)
    var cols = Int32(n_value)
    var leading = Int32(n_value)
    var info = Int32(0)
    function(
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=rows)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=cols)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=leading)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](pivot_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    return Int(info)


@always_inline
def lapack_dgetrf(
    n_value: Int,
    a_ptr: UnsafePointer[Float64, MutExternalOrigin],
    pivot_ptr: UnsafePointer[Int32, MutExternalOrigin],
) raises -> Int:
    var function = get_lapack_dgetrf_function()
    var rows = Int32(n_value)
    var cols = Int32(n_value)
    var leading = Int32(n_value)
    var info = Int32(0)
    function(
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=rows)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=cols)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=leading)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](pivot_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    return Int(info)


@always_inline
def cblas_transpose_flag(transpose: Bool) -> _CBLASTranspose:
    if transpose:
        return _CBLASTranspose.TRANSPOSE
    return _CBLASTranspose.NO_TRANSPOSE


@always_inline
def cblas_sgemm_row_major(
    m: Int,
    n: Int,
    k: Int,
    c_ptr: UnsafePointer[Float32, MutExternalOrigin],
    a_ptr: UnsafePointer[Float32, MutExternalOrigin],
    b_ptr: UnsafePointer[Float32, MutExternalOrigin],
) raises:
    cblas_sgemm_row_major_ld(
        m, n, k, c_ptr, a_ptr, b_ptr, False, False, k, n, n
    )


@always_inline
def cblas_sgemm_row_major_ld(
    m: Int,
    n: Int,
    k: Int,
    c_ptr: UnsafePointer[Float32, MutExternalOrigin],
    a_ptr: UnsafePointer[Float32, MutExternalOrigin],
    b_ptr: UnsafePointer[Float32, MutExternalOrigin],
    transpose_a: Bool,
    transpose_b: Bool,
    lda: Int,
    ldb: Int,
    ldc: Int,
) raises:
    var sgemm = get_cblas_f32_function()
    sgemm(
        _CBLASOrder.ROW_MAJOR,
        cblas_transpose_flag(transpose_a),
        cblas_transpose_flag(transpose_b),
        Int32(m),
        Int32(n),
        Int32(k),
        1.0,
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](a_ptr),
        Int32(lda),
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](b_ptr),
        Int32(ldb),
        0.0,
        rebind[UnsafePointer[Float32, MutAnyOrigin]](c_ptr),
        Int32(ldc),
    )


@always_inline
def cblas_sgemv_row_major_ld(
    rows: Int,
    cols: Int,
    y_ptr: UnsafePointer[Float32, MutExternalOrigin],
    a_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    transpose_a: Bool,
    lda: Int,
) raises:
    var sgemv = get_cblas_gemv_f32_function()
    sgemv(
        _CBLASOrder.ROW_MAJOR,
        cblas_transpose_flag(transpose_a),
        Int32(rows),
        Int32(cols),
        1.0,
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](a_ptr),
        Int32(lda),
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](x_ptr),
        Int32(1),
        0.0,
        rebind[UnsafePointer[Float32, MutAnyOrigin]](y_ptr),
        Int32(1),
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
    cblas_dgemm_row_major_ld(
        m, n, k, c_ptr, a_ptr, b_ptr, False, False, k, n, n
    )


@always_inline
def cblas_dgemm_row_major_ld(
    m: Int,
    n: Int,
    k: Int,
    c_ptr: UnsafePointer[Float64, MutExternalOrigin],
    a_ptr: UnsafePointer[Float64, MutExternalOrigin],
    b_ptr: UnsafePointer[Float64, MutExternalOrigin],
    transpose_a: Bool,
    transpose_b: Bool,
    lda: Int,
    ldb: Int,
    ldc: Int,
) raises:
    var dgemm = get_cblas_f64_function()
    dgemm(
        _CBLASOrder.ROW_MAJOR,
        cblas_transpose_flag(transpose_a),
        cblas_transpose_flag(transpose_b),
        Int32(m),
        Int32(n),
        Int32(k),
        1.0,
        rebind[UnsafePointer[Float64, ImmutAnyOrigin]](a_ptr),
        Int32(lda),
        rebind[UnsafePointer[Float64, ImmutAnyOrigin]](b_ptr),
        Int32(ldb),
        0.0,
        rebind[UnsafePointer[Float64, MutAnyOrigin]](c_ptr),
        Int32(ldc),
    )


@always_inline
def cblas_dgemv_row_major_ld(
    rows: Int,
    cols: Int,
    y_ptr: UnsafePointer[Float64, MutExternalOrigin],
    a_ptr: UnsafePointer[Float64, MutExternalOrigin],
    x_ptr: UnsafePointer[Float64, MutExternalOrigin],
    transpose_a: Bool,
    lda: Int,
) raises:
    var dgemv = get_cblas_gemv_f64_function()
    dgemv(
        _CBLASOrder.ROW_MAJOR,
        cblas_transpose_flag(transpose_a),
        Int32(rows),
        Int32(cols),
        1.0,
        rebind[UnsafePointer[Float64, ImmutAnyOrigin]](a_ptr),
        Int32(lda),
        rebind[UnsafePointer[Float64, ImmutAnyOrigin]](x_ptr),
        Int32(1),
        0.0,
        rebind[UnsafePointer[Float64, MutAnyOrigin]](y_ptr),
        Int32(1),
    )
