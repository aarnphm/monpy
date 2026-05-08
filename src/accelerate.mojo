from std.ffi import _get_dylib_function as _ffi_get_dylib_function
from std.ffi import _Global, OwnedDLHandle
from std.memory.unsafe_pointer import alloc
from std.sys import CompilationTarget


# BLAS / LAPACK backend plumbing kept isolated from ndarray semantics. The
# resolved dylib differs per OS: on macOS we hit Apple's Accelerate
# framework; on Linux we try OpenBLAS first (most common, fastest), then
# fall back to the netlib reference libraries. cblas_* and *gesv_/*getrf_
# function names are identical across both, so the call helpers below are
# platform-agnostic — only the dylib resolution branches.
comptime LIB_ACC_PATH = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
comptime LIB_SYSTEM_PATH = "/usr/lib/libSystem.B.dylib"

# Linux dylib candidates, tried in order in `init_*_dylib`. OpenBLAS is the
# de-facto standard on Linux distros (Ubuntu, Debian, Fedora, Arch all ship
# it as the default BLAS provider when scipy/numpy are installed). The
# unversioned `.so` links are dev packages; the versioned `.so.0` / `.so.3`
# symlinks ship with the runtime package and are what numpy/scipy load too.
comptime LIB_MATH_PATH_6 = "libm.so.6"
comptime LIB_MATH_PATH = "libm.so"
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

comptime cblas_cgemm_type = def(
    _CBLASOrder,
    _CBLASTranspose,
    _CBLASTranspose,
    Int32,
    Int32,
    Int32,
    UnsafePointer[Float32, ImmutAnyOrigin],  # alpha (re, im) pair
    UnsafePointer[Float32, ImmutAnyOrigin],
    Int32,
    UnsafePointer[Float32, ImmutAnyOrigin],
    Int32,
    UnsafePointer[Float32, ImmutAnyOrigin],  # beta (re, im) pair
    UnsafePointer[Float32, MutAnyOrigin],
    Int32,
) thin -> None

comptime cblas_zgemm_type = def(
    _CBLASOrder,
    _CBLASTranspose,
    _CBLASTranspose,
    Int32,
    Int32,
    Int32,
    UnsafePointer[Float64, ImmutAnyOrigin],
    UnsafePointer[Float64, ImmutAnyOrigin],
    Int32,
    UnsafePointer[Float64, ImmutAnyOrigin],
    Int32,
    UnsafePointer[Float64, ImmutAnyOrigin],
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

comptime libm_pow_f64_type = def(Float64, Float64) thin -> Float64

comptime vdsp_binary_f32_type = def(
    UnsafePointer[Float32, ImmutAnyOrigin],
    Int64,
    UnsafePointer[Float32, ImmutAnyOrigin],
    Int64,
    UnsafePointer[Float32, MutAnyOrigin],
    Int64,
    UInt64,
) thin -> None

comptime vdsp_binary_f64_type = def(
    UnsafePointer[Float64, ImmutAnyOrigin],
    Int64,
    UnsafePointer[Float64, ImmutAnyOrigin],
    Int64,
    UnsafePointer[Float64, MutAnyOrigin],
    Int64,
    UInt64,
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


# Phase-6d LAPACK additions: QR, Cholesky, eigendecompositions, SVD,
# least-squares. F77 ABI is column-major; everything goes by pointer.
# Character params (JOBZ, UPLO etc.) are single ASCII bytes passed via
# UnsafePointer[Int8].

comptime lapack_sgeqrf_type = def(
    UnsafePointer[Int32, MutAnyOrigin],  # M
    UnsafePointer[Int32, MutAnyOrigin],  # N
    UnsafePointer[Float32, MutAnyOrigin],  # A[lda*n]
    UnsafePointer[Int32, MutAnyOrigin],  # LDA
    UnsafePointer[Float32, MutAnyOrigin],  # TAU[min(M,N)]
    UnsafePointer[Float32, MutAnyOrigin],  # WORK[lwork]
    UnsafePointer[Int32, MutAnyOrigin],  # LWORK
    UnsafePointer[Int32, MutAnyOrigin],  # INFO
) thin -> None

comptime lapack_dgeqrf_type = def(
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
) thin -> None

comptime lapack_sorgqr_type = def(
    UnsafePointer[Int32, MutAnyOrigin],  # M
    UnsafePointer[Int32, MutAnyOrigin],  # N
    UnsafePointer[Int32, MutAnyOrigin],  # K
    UnsafePointer[Float32, MutAnyOrigin],  # A
    UnsafePointer[Int32, MutAnyOrigin],  # LDA
    UnsafePointer[Float32, MutAnyOrigin],  # TAU
    UnsafePointer[Float32, MutAnyOrigin],  # WORK
    UnsafePointer[Int32, MutAnyOrigin],  # LWORK
    UnsafePointer[Int32, MutAnyOrigin],  # INFO
) thin -> None

comptime lapack_dorgqr_type = def(
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
) thin -> None

comptime lapack_spotrf_type = def(
    UnsafePointer[Int8, MutAnyOrigin],  # UPLO ('U'/'L')
    UnsafePointer[Int32, MutAnyOrigin],  # N
    UnsafePointer[Float32, MutAnyOrigin],  # A
    UnsafePointer[Int32, MutAnyOrigin],  # LDA
    UnsafePointer[Int32, MutAnyOrigin],  # INFO
) thin -> None

comptime lapack_dpotrf_type = def(
    UnsafePointer[Int8, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
) thin -> None

comptime lapack_ssyev_type = def(
    UnsafePointer[Int8, MutAnyOrigin],  # JOBZ ('N'/'V')
    UnsafePointer[Int8, MutAnyOrigin],  # UPLO
    UnsafePointer[Int32, MutAnyOrigin],  # N
    UnsafePointer[Float32, MutAnyOrigin],  # A
    UnsafePointer[Int32, MutAnyOrigin],  # LDA
    UnsafePointer[Float32, MutAnyOrigin],  # W[N]
    UnsafePointer[Float32, MutAnyOrigin],  # WORK
    UnsafePointer[Int32, MutAnyOrigin],  # LWORK
    UnsafePointer[Int32, MutAnyOrigin],  # INFO
) thin -> None

comptime lapack_dsyev_type = def(
    UnsafePointer[Int8, MutAnyOrigin],
    UnsafePointer[Int8, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
) thin -> None

comptime lapack_sgesdd_type = def(
    UnsafePointer[Int8, MutAnyOrigin],  # JOBZ
    UnsafePointer[Int32, MutAnyOrigin],  # M
    UnsafePointer[Int32, MutAnyOrigin],  # N
    UnsafePointer[Float32, MutAnyOrigin],  # A
    UnsafePointer[Int32, MutAnyOrigin],  # LDA
    UnsafePointer[Float32, MutAnyOrigin],  # S
    UnsafePointer[Float32, MutAnyOrigin],  # U
    UnsafePointer[Int32, MutAnyOrigin],  # LDU
    UnsafePointer[Float32, MutAnyOrigin],  # VT
    UnsafePointer[Int32, MutAnyOrigin],  # LDVT
    UnsafePointer[Float32, MutAnyOrigin],  # WORK
    UnsafePointer[Int32, MutAnyOrigin],  # LWORK
    UnsafePointer[Int32, MutAnyOrigin],  # IWORK[8*min(M,N)]
    UnsafePointer[Int32, MutAnyOrigin],  # INFO
) thin -> None

comptime lapack_dgesdd_type = def(
    UnsafePointer[Int8, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
) thin -> None

comptime lapack_sgelsd_type = def(
    UnsafePointer[Int32, MutAnyOrigin],  # M
    UnsafePointer[Int32, MutAnyOrigin],  # N
    UnsafePointer[Int32, MutAnyOrigin],  # NRHS
    UnsafePointer[Float32, MutAnyOrigin],  # A
    UnsafePointer[Int32, MutAnyOrigin],  # LDA
    UnsafePointer[Float32, MutAnyOrigin],  # B
    UnsafePointer[Int32, MutAnyOrigin],  # LDB
    UnsafePointer[Float32, MutAnyOrigin],  # S[min(M,N)]
    UnsafePointer[Float32, MutAnyOrigin],  # RCOND
    UnsafePointer[Int32, MutAnyOrigin],  # RANK
    UnsafePointer[Float32, MutAnyOrigin],  # WORK
    UnsafePointer[Int32, MutAnyOrigin],  # LWORK
    UnsafePointer[Int32, MutAnyOrigin],  # IWORK
    UnsafePointer[Int32, MutAnyOrigin],  # INFO
) thin -> None

comptime lapack_dgelsd_type = def(
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
) thin -> None

comptime lapack_sgeev_type = def(
    UnsafePointer[Int8, MutAnyOrigin],  # JOBVL
    UnsafePointer[Int8, MutAnyOrigin],  # JOBVR
    UnsafePointer[Int32, MutAnyOrigin],  # N
    UnsafePointer[Float32, MutAnyOrigin],  # A
    UnsafePointer[Int32, MutAnyOrigin],  # LDA
    UnsafePointer[Float32, MutAnyOrigin],  # WR[N]
    UnsafePointer[Float32, MutAnyOrigin],  # WI[N]
    UnsafePointer[Float32, MutAnyOrigin],  # VL
    UnsafePointer[Int32, MutAnyOrigin],  # LDVL
    UnsafePointer[Float32, MutAnyOrigin],  # VR
    UnsafePointer[Int32, MutAnyOrigin],  # LDVR
    UnsafePointer[Float32, MutAnyOrigin],  # WORK
    UnsafePointer[Int32, MutAnyOrigin],  # LWORK
    UnsafePointer[Int32, MutAnyOrigin],  # INFO
) thin -> None

comptime lapack_dgeev_type = def(
    UnsafePointer[Int8, MutAnyOrigin],
    UnsafePointer[Int8, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
    UnsafePointer[Int32, MutAnyOrigin],
    UnsafePointer[Float64, MutAnyOrigin],
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

comptime MONPY_MATH_DYLIB = _Global[
    "MONPY_MATH_DYLIB",
    init_math_dylib,
    on_error_msg=math_error_msg,
]


def blas_error_msg() -> Error:
    comptime if CompilationTarget.is_macos():
        return Error("cannot find Apple Accelerate at ", LIB_ACC_PATH)
    elif CompilationTarget.is_linux():
        return Error(
            "cannot find OpenBLAS / libblas on Linux. install with `apt install libopenblas-dev` or equivalent."
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


def math_error_msg() -> Error:
    comptime if CompilationTarget.is_macos():
        return Error("cannot find libSystem at ", LIB_SYSTEM_PATH)
    elif CompilationTarget.is_linux():
        return Error("cannot find libm on Linux")
    return Error("math dylib not configured for this platform")


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


def init_math_dylib() -> OwnedDLHandle:
    comptime if CompilationTarget.is_macos():
        try:
            return OwnedDLHandle(LIB_SYSTEM_PATH)
        except:
            return OwnedDLHandle(unsafe_uninitialized=True)
    elif CompilationTarget.is_linux():
        try:
            return OwnedDLHandle(LIB_MATH_PATH_6)
        except:
            pass
        try:
            return OwnedDLHandle(LIB_MATH_PATH)
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
def get_blas_function[func_name: StaticString, result_type: TrivialRegisterPassable]() raises -> result_type:
    comptime assert (
        CompilationTarget.is_macos() or CompilationTarget.is_linux()
    ), "BLAS backend requires macOS or Linux"
    return _ffi_get_dylib_function[
        MONPY_BLAS_DYLIB(),
        func_name,
        result_type,
    ]()


@always_inline
def get_lapack_function[func_name: StaticString, result_type: TrivialRegisterPassable]() raises -> result_type:
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
def get_accelerate_function[func_name: StaticString, result_type: TrivialRegisterPassable]() raises -> result_type:
    return get_blas_function[func_name, result_type]()


@always_inline
def libm_pow_f64(base: Float64, exponent: Float64) raises -> Float64:
    var function = _ffi_get_dylib_function[MONPY_MATH_DYLIB(), "pow", libm_pow_f64_type]()
    return function(base, exponent)


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
def call_vv[
    dt: DType,
    func_name_f32: StaticString,
    func_name_f64: StaticString,
](
    out_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    src_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    count_value: Int,
) raises:
    """Parametric libvMath unary dispatcher — picks f32/f64 symbol from `dt`."""
    comptime if dt == DType.float32:
        call_vv_f32[func_name_f32](
            rebind[UnsafePointer[Float32, MutExternalOrigin]](out_ptr),
            rebind[UnsafePointer[Float32, MutExternalOrigin]](src_ptr),
            count_value,
        )
    else:
        call_vv_f64[func_name_f64](
            rebind[UnsafePointer[Float64, MutExternalOrigin]](out_ptr),
            rebind[UnsafePointer[Float64, MutExternalOrigin]](src_ptr),
            count_value,
        )


@always_inline
def call_vdsp_binary[
    dt: DType,
    func_name_f32: StaticString,
    func_name_f64: StaticString,
](
    lhs_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    count_value: Int,
) raises:
    """Parametric vDSP binary dispatcher — picks f32/f64 symbol from `dt`."""
    comptime if dt == DType.float32:
        call_vdsp_binary_f32[func_name_f32](
            rebind[UnsafePointer[Float32, MutExternalOrigin]](lhs_ptr),
            rebind[UnsafePointer[Float32, MutExternalOrigin]](rhs_ptr),
            rebind[UnsafePointer[Float32, MutExternalOrigin]](out_ptr),
            count_value,
        )
    else:
        call_vdsp_binary_f64[func_name_f64](
            rebind[UnsafePointer[Float64, MutExternalOrigin]](lhs_ptr),
            rebind[UnsafePointer[Float64, MutExternalOrigin]](rhs_ptr),
            rebind[UnsafePointer[Float64, MutExternalOrigin]](out_ptr),
            count_value,
        )


@always_inline
def call_vdsp_binary_strided[
    dt: DType,
    func_name_f32: StaticString,
    func_name_f64: StaticString,
](
    lhs_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    lhs_stride: Int,
    rhs_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    rhs_stride: Int,
    out_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    out_stride: Int,
    count_value: Int,
) raises:
    """Parametric strided vDSP binary dispatcher."""
    comptime if dt == DType.float32:
        call_vdsp_binary_strided_f32[func_name_f32](
            rebind[UnsafePointer[Float32, MutExternalOrigin]](lhs_ptr),
            lhs_stride,
            rebind[UnsafePointer[Float32, MutExternalOrigin]](rhs_ptr),
            rhs_stride,
            rebind[UnsafePointer[Float32, MutExternalOrigin]](out_ptr),
            out_stride,
            count_value,
        )
    else:
        call_vdsp_binary_strided_f64[func_name_f64](
            rebind[UnsafePointer[Float64, MutExternalOrigin]](lhs_ptr),
            lhs_stride,
            rebind[UnsafePointer[Float64, MutExternalOrigin]](rhs_ptr),
            rhs_stride,
            rebind[UnsafePointer[Float64, MutExternalOrigin]](out_ptr),
            out_stride,
            count_value,
        )


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
def call_vdsp_binary_f32[
    func_name: StaticString
](
    lhs_ptr: UnsafePointer[Float32, MutExternalOrigin],
    rhs_ptr: UnsafePointer[Float32, MutExternalOrigin],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    count_value: Int,
) raises:
    var function = get_accelerate_function[func_name, vdsp_binary_f32_type]()
    function(
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](lhs_ptr),
        Int64(1),
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](rhs_ptr),
        Int64(1),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](out_ptr),
        Int64(1),
        UInt64(count_value),
    )


@always_inline
def call_vdsp_binary_strided_f32[
    func_name: StaticString
](
    lhs_ptr: UnsafePointer[Float32, MutExternalOrigin],
    lhs_stride: Int,
    rhs_ptr: UnsafePointer[Float32, MutExternalOrigin],
    rhs_stride: Int,
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    out_stride: Int,
    count_value: Int,
) raises:
    var function = get_accelerate_function[func_name, vdsp_binary_f32_type]()
    function(
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](lhs_ptr),
        Int64(lhs_stride),
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](rhs_ptr),
        Int64(rhs_stride),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](out_ptr),
        Int64(out_stride),
        UInt64(count_value),
    )


@always_inline
def call_vdsp_binary_f64[
    func_name: StaticString
](
    lhs_ptr: UnsafePointer[Float64, MutExternalOrigin],
    rhs_ptr: UnsafePointer[Float64, MutExternalOrigin],
    out_ptr: UnsafePointer[Float64, MutExternalOrigin],
    count_value: Int,
) raises:
    var function = get_accelerate_function[func_name, vdsp_binary_f64_type]()
    function(
        rebind[UnsafePointer[Float64, ImmutAnyOrigin]](lhs_ptr),
        Int64(1),
        rebind[UnsafePointer[Float64, ImmutAnyOrigin]](rhs_ptr),
        Int64(1),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](out_ptr),
        Int64(1),
        UInt64(count_value),
    )


@always_inline
def call_vdsp_binary_strided_f64[
    func_name: StaticString
](
    lhs_ptr: UnsafePointer[Float64, MutExternalOrigin],
    lhs_stride: Int,
    rhs_ptr: UnsafePointer[Float64, MutExternalOrigin],
    rhs_stride: Int,
    out_ptr: UnsafePointer[Float64, MutExternalOrigin],
    out_stride: Int,
    count_value: Int,
) raises:
    var function = get_accelerate_function[func_name, vdsp_binary_f64_type]()
    function(
        rebind[UnsafePointer[Float64, ImmutAnyOrigin]](lhs_ptr),
        Int64(lhs_stride),
        rebind[UnsafePointer[Float64, ImmutAnyOrigin]](rhs_ptr),
        Int64(rhs_stride),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](out_ptr),
        Int64(out_stride),
        UInt64(count_value),
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
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=rhs_columns)),
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
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=rhs_columns)),
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
    cblas_sgemm_row_major_ld(m, n, k, c_ptr, a_ptr, b_ptr, False, False, k, n, n)


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
    cblas_dgemm_row_major_ld(m, n, k, c_ptr, a_ptr, b_ptr, False, False, k, n, n)


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


@always_inline
def get_cblas_cgemm_function() raises -> cblas_cgemm_type:
    return get_blas_function["cblas_cgemm", cblas_cgemm_type]()


@always_inline
def get_cblas_zgemm_function() raises -> cblas_zgemm_type:
    return get_blas_function["cblas_zgemm", cblas_zgemm_type]()


@always_inline
def cblas_cgemm_row_major(
    m: Int,
    n: Int,
    k: Int,
    c_ptr: UnsafePointer[Float32, MutExternalOrigin],
    a_ptr: UnsafePointer[Float32, MutExternalOrigin],
    b_ptr: UnsafePointer[Float32, MutExternalOrigin],
) raises:
    # Complex single-precision GEMM. Row-major layout, no transposes.
    # Pointers are interleaved (real, imag) float32 pairs; lda/ldb/ldc are
    # in *complex* element units, not float units.
    var cgemm = get_cblas_cgemm_function()
    var alpha = InlineArray[Float32, 2](fill=0.0)
    alpha[0] = 1.0
    var beta = InlineArray[Float32, 2](fill=0.0)
    cgemm(
        _CBLASOrder.ROW_MAJOR,
        _CBLASTranspose.NO_TRANSPOSE,
        _CBLASTranspose.NO_TRANSPOSE,
        Int32(m),
        Int32(n),
        Int32(k),
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](alpha.unsafe_ptr()),
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](a_ptr),
        Int32(k),
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](b_ptr),
        Int32(n),
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](beta.unsafe_ptr()),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](c_ptr),
        Int32(n),
    )


@always_inline
def cblas_zgemm_row_major(
    m: Int,
    n: Int,
    k: Int,
    c_ptr: UnsafePointer[Float64, MutExternalOrigin],
    a_ptr: UnsafePointer[Float64, MutExternalOrigin],
    b_ptr: UnsafePointer[Float64, MutExternalOrigin],
) raises:
    var zgemm = get_cblas_zgemm_function()
    var alpha = InlineArray[Float64, 2](fill=0.0)
    alpha[0] = 1.0
    var beta = InlineArray[Float64, 2](fill=0.0)
    zgemm(
        _CBLASOrder.ROW_MAJOR,
        _CBLASTranspose.NO_TRANSPOSE,
        _CBLASTranspose.NO_TRANSPOSE,
        Int32(m),
        Int32(n),
        Int32(k),
        rebind[UnsafePointer[Float64, ImmutAnyOrigin]](alpha.unsafe_ptr()),
        rebind[UnsafePointer[Float64, ImmutAnyOrigin]](a_ptr),
        Int32(k),
        rebind[UnsafePointer[Float64, ImmutAnyOrigin]](b_ptr),
        Int32(n),
        rebind[UnsafePointer[Float64, ImmutAnyOrigin]](beta.unsafe_ptr()),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](c_ptr),
        Int32(n),
    )


# ============================================================
# Phase-6d LAPACK call wrappers.
#
# Each wrapper handles the F77 ABI ceremony (every argument by pointer,
# even scalars), runs LAPACK's workspace query (LWORK = -1 returns the
# optimal size in WORK[0]), allocates the workspace, runs the real call,
# frees the workspace, and returns INFO. Callers stay in pointer-land so
# the surrounding kernel code can keep working with col-major scratch.
# ============================================================


@always_inline
def get_lapack_sgeqrf_function() raises -> lapack_sgeqrf_type:
    return get_lapack_function["sgeqrf_", lapack_sgeqrf_type]()


@always_inline
def get_lapack_dgeqrf_function() raises -> lapack_dgeqrf_type:
    return get_lapack_function["dgeqrf_", lapack_dgeqrf_type]()


@always_inline
def get_lapack_sorgqr_function() raises -> lapack_sorgqr_type:
    return get_lapack_function["sorgqr_", lapack_sorgqr_type]()


@always_inline
def get_lapack_dorgqr_function() raises -> lapack_dorgqr_type:
    return get_lapack_function["dorgqr_", lapack_dorgqr_type]()


@always_inline
def get_lapack_spotrf_function() raises -> lapack_spotrf_type:
    return get_lapack_function["spotrf_", lapack_spotrf_type]()


@always_inline
def get_lapack_dpotrf_function() raises -> lapack_dpotrf_type:
    return get_lapack_function["dpotrf_", lapack_dpotrf_type]()


@always_inline
def get_lapack_ssyev_function() raises -> lapack_ssyev_type:
    return get_lapack_function["ssyev_", lapack_ssyev_type]()


@always_inline
def get_lapack_dsyev_function() raises -> lapack_dsyev_type:
    return get_lapack_function["dsyev_", lapack_dsyev_type]()


@always_inline
def get_lapack_sgesdd_function() raises -> lapack_sgesdd_type:
    return get_lapack_function["sgesdd_", lapack_sgesdd_type]()


@always_inline
def get_lapack_dgesdd_function() raises -> lapack_dgesdd_type:
    return get_lapack_function["dgesdd_", lapack_dgesdd_type]()


@always_inline
def get_lapack_sgelsd_function() raises -> lapack_sgelsd_type:
    return get_lapack_function["sgelsd_", lapack_sgelsd_type]()


@always_inline
def get_lapack_dgelsd_function() raises -> lapack_dgelsd_type:
    return get_lapack_function["dgelsd_", lapack_dgelsd_type]()


@always_inline
def get_lapack_sgeev_function() raises -> lapack_sgeev_type:
    return get_lapack_function["sgeev_", lapack_sgeev_type]()


@always_inline
def get_lapack_dgeev_function() raises -> lapack_dgeev_type:
    return get_lapack_function["dgeev_", lapack_dgeev_type]()


@always_inline
def lapack_sgeqrf(
    m_value: Int,
    n_value: Int,
    a_ptr: UnsafePointer[Float32, MutExternalOrigin],
    tau_ptr: UnsafePointer[Float32, MutExternalOrigin],
) raises -> Int:
    var function = get_lapack_sgeqrf_function()
    var m = Int32(m_value)
    var n = Int32(n_value)
    var lda = Int32(m_value)
    var info = Int32(0)
    # Workspace query: LWORK = -1, single-element scratch, optimal size in WORK[0].
    var query_lwork = Int32(-1)
    var query_work = Float32(0)
    function(
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=m)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](tau_ptr),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](UnsafePointer(to=query_work)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=query_lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    if info != 0:
        return Int(info)
    var lwork_int = Int(query_work)
    if lwork_int < 1:
        lwork_int = 1
    var lwork = Int32(lwork_int)
    var work = alloc[Float32](lwork_int)
    function(
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=m)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](tau_ptr),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](work),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    work.free()
    return Int(info)


@always_inline
def lapack_dgeqrf(
    m_value: Int,
    n_value: Int,
    a_ptr: UnsafePointer[Float64, MutExternalOrigin],
    tau_ptr: UnsafePointer[Float64, MutExternalOrigin],
) raises -> Int:
    var function = get_lapack_dgeqrf_function()
    var m = Int32(m_value)
    var n = Int32(n_value)
    var lda = Int32(m_value)
    var info = Int32(0)
    var query_lwork = Int32(-1)
    var query_work = Float64(0)
    function(
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=m)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](tau_ptr),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](UnsafePointer(to=query_work)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=query_lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    if info != 0:
        return Int(info)
    var lwork_int = Int(query_work)
    if lwork_int < 1:
        lwork_int = 1
    var lwork = Int32(lwork_int)
    var work = alloc[Float64](lwork_int)
    function(
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=m)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](tau_ptr),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](work),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    work.free()
    return Int(info)


@always_inline
def lapack_sorgqr(
    m_value: Int,
    n_value: Int,
    k_value: Int,
    a_ptr: UnsafePointer[Float32, MutExternalOrigin],
    tau_ptr: UnsafePointer[Float32, MutExternalOrigin],
) raises -> Int:
    var function = get_lapack_sorgqr_function()
    var m = Int32(m_value)
    var n = Int32(n_value)
    var k = Int32(k_value)
    var lda = Int32(m_value)
    var info = Int32(0)
    var query_lwork = Int32(-1)
    var query_work = Float32(0)
    function(
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=m)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=k)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](tau_ptr),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](UnsafePointer(to=query_work)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=query_lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    if info != 0:
        return Int(info)
    var lwork_int = Int(query_work)
    if lwork_int < 1:
        lwork_int = 1
    var lwork = Int32(lwork_int)
    var work = alloc[Float32](lwork_int)
    function(
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=m)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=k)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](tau_ptr),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](work),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    work.free()
    return Int(info)


@always_inline
def lapack_dorgqr(
    m_value: Int,
    n_value: Int,
    k_value: Int,
    a_ptr: UnsafePointer[Float64, MutExternalOrigin],
    tau_ptr: UnsafePointer[Float64, MutExternalOrigin],
) raises -> Int:
    var function = get_lapack_dorgqr_function()
    var m = Int32(m_value)
    var n = Int32(n_value)
    var k = Int32(k_value)
    var lda = Int32(m_value)
    var info = Int32(0)
    var query_lwork = Int32(-1)
    var query_work = Float64(0)
    function(
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=m)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=k)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](tau_ptr),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](UnsafePointer(to=query_work)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=query_lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    if info != 0:
        return Int(info)
    var lwork_int = Int(query_work)
    if lwork_int < 1:
        lwork_int = 1
    var lwork = Int32(lwork_int)
    var work = alloc[Float64](lwork_int)
    function(
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=m)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=k)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](tau_ptr),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](work),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    work.free()
    return Int(info)


@always_inline
def lapack_spotrf(
    n_value: Int,
    a_ptr: UnsafePointer[Float32, MutExternalOrigin],
    upper: Bool,
) raises -> Int:
    var function = get_lapack_spotrf_function()
    var n = Int32(n_value)
    var lda = Int32(n_value)
    var info = Int32(0)
    var uplo_byte = Int8(85) if upper else Int8(76)  # 'U' = 0x55, 'L' = 0x4C
    function(
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=uplo_byte)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    return Int(info)


@always_inline
def lapack_dpotrf(
    n_value: Int,
    a_ptr: UnsafePointer[Float64, MutExternalOrigin],
    upper: Bool,
) raises -> Int:
    var function = get_lapack_dpotrf_function()
    var n = Int32(n_value)
    var lda = Int32(n_value)
    var info = Int32(0)
    var uplo_byte = Int8(85) if upper else Int8(76)
    function(
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=uplo_byte)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    return Int(info)


@always_inline
def lapack_ssyev(
    n_value: Int,
    a_ptr: UnsafePointer[Float32, MutExternalOrigin],
    w_ptr: UnsafePointer[Float32, MutExternalOrigin],
    compute_eigenvectors: Bool,
    upper: Bool,
) raises -> Int:
    var function = get_lapack_ssyev_function()
    var n = Int32(n_value)
    var lda = Int32(n_value)
    var info = Int32(0)
    var jobz_byte = Int8(86) if compute_eigenvectors else Int8(78)  # 'V' or 'N'
    var uplo_byte = Int8(85) if upper else Int8(76)
    var query_lwork = Int32(-1)
    var query_work = Float32(0)
    function(
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=jobz_byte)),
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=uplo_byte)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](w_ptr),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](UnsafePointer(to=query_work)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=query_lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    if info != 0:
        return Int(info)
    var lwork_int = Int(query_work)
    if lwork_int < 1:
        lwork_int = 1
    var lwork = Int32(lwork_int)
    var work = alloc[Float32](lwork_int)
    function(
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=jobz_byte)),
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=uplo_byte)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](w_ptr),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](work),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    work.free()
    return Int(info)


@always_inline
def lapack_dsyev(
    n_value: Int,
    a_ptr: UnsafePointer[Float64, MutExternalOrigin],
    w_ptr: UnsafePointer[Float64, MutExternalOrigin],
    compute_eigenvectors: Bool,
    upper: Bool,
) raises -> Int:
    var function = get_lapack_dsyev_function()
    var n = Int32(n_value)
    var lda = Int32(n_value)
    var info = Int32(0)
    var jobz_byte = Int8(86) if compute_eigenvectors else Int8(78)
    var uplo_byte = Int8(85) if upper else Int8(76)
    var query_lwork = Int32(-1)
    var query_work = Float64(0)
    function(
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=jobz_byte)),
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=uplo_byte)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](w_ptr),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](UnsafePointer(to=query_work)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=query_lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    if info != 0:
        return Int(info)
    var lwork_int = Int(query_work)
    if lwork_int < 1:
        lwork_int = 1
    var lwork = Int32(lwork_int)
    var work = alloc[Float64](lwork_int)
    function(
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=jobz_byte)),
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=uplo_byte)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](w_ptr),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](work),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    work.free()
    return Int(info)


@always_inline
def lapack_sgesdd(
    m_value: Int,
    n_value: Int,
    a_ptr: UnsafePointer[Float32, MutExternalOrigin],
    s_ptr: UnsafePointer[Float32, MutExternalOrigin],
    u_ptr: UnsafePointer[Float32, MutExternalOrigin],
    vt_ptr: UnsafePointer[Float32, MutExternalOrigin],
    full_matrices: Bool,
    compute_uv: Bool,
) raises -> Int:
    # JOBZ = 'A' (full U/VT), 'S' (thin U/VT), 'N' (singular values only).
    var function = get_lapack_sgesdd_function()
    var m = Int32(m_value)
    var n = Int32(n_value)
    var lda = Int32(m_value)
    var ldu = Int32(m_value)
    var ldvt: Int32
    if not compute_uv:
        ldvt = Int32(1)
    elif full_matrices:
        ldvt = Int32(n_value)
    else:
        var thin = m_value if m_value < n_value else n_value
        ldvt = Int32(thin)
    var info = Int32(0)
    var jobz_byte: Int8
    if not compute_uv:
        jobz_byte = Int8(78)  # 'N'
    elif full_matrices:
        jobz_byte = Int8(65)  # 'A'
    else:
        jobz_byte = Int8(83)  # 'S'
    var min_mn = m_value if m_value < n_value else n_value
    if min_mn < 1:
        min_mn = 1
    var iwork = alloc[Int32](8 * min_mn)
    var query_lwork = Int32(-1)
    var query_work = Float32(0)
    function(
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=jobz_byte)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=m)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](s_ptr),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](u_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldu)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](vt_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldvt)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](UnsafePointer(to=query_work)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=query_lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](iwork),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    if info != 0:
        iwork.free()
        return Int(info)
    var lwork_int = Int(query_work)
    if lwork_int < 1:
        lwork_int = 1
    var lwork = Int32(lwork_int)
    var work = alloc[Float32](lwork_int)
    function(
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=jobz_byte)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=m)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](s_ptr),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](u_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldu)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](vt_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldvt)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](work),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](iwork),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    work.free()
    iwork.free()
    return Int(info)


@always_inline
def lapack_dgesdd(
    m_value: Int,
    n_value: Int,
    a_ptr: UnsafePointer[Float64, MutExternalOrigin],
    s_ptr: UnsafePointer[Float64, MutExternalOrigin],
    u_ptr: UnsafePointer[Float64, MutExternalOrigin],
    vt_ptr: UnsafePointer[Float64, MutExternalOrigin],
    full_matrices: Bool,
    compute_uv: Bool,
) raises -> Int:
    var function = get_lapack_dgesdd_function()
    var m = Int32(m_value)
    var n = Int32(n_value)
    var lda = Int32(m_value)
    var ldu = Int32(m_value)
    var ldvt: Int32
    if not compute_uv:
        ldvt = Int32(1)
    elif full_matrices:
        ldvt = Int32(n_value)
    else:
        var thin = m_value if m_value < n_value else n_value
        ldvt = Int32(thin)
    var info = Int32(0)
    var jobz_byte: Int8
    if not compute_uv:
        jobz_byte = Int8(78)
    elif full_matrices:
        jobz_byte = Int8(65)
    else:
        jobz_byte = Int8(83)
    var min_mn = m_value if m_value < n_value else n_value
    if min_mn < 1:
        min_mn = 1
    var iwork = alloc[Int32](8 * min_mn)
    var query_lwork = Int32(-1)
    var query_work = Float64(0)
    function(
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=jobz_byte)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=m)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](s_ptr),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](u_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldu)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](vt_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldvt)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](UnsafePointer(to=query_work)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=query_lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](iwork),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    if info != 0:
        iwork.free()
        return Int(info)
    var lwork_int = Int(query_work)
    if lwork_int < 1:
        lwork_int = 1
    var lwork = Int32(lwork_int)
    var work = alloc[Float64](lwork_int)
    function(
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=jobz_byte)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=m)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](s_ptr),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](u_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldu)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](vt_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldvt)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](work),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](iwork),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    work.free()
    iwork.free()
    return Int(info)


@always_inline
def lapack_sgelsd(
    m_value: Int,
    n_value: Int,
    nrhs_value: Int,
    a_ptr: UnsafePointer[Float32, MutExternalOrigin],
    b_ptr: UnsafePointer[Float32, MutExternalOrigin],
    s_ptr: UnsafePointer[Float32, MutExternalOrigin],
    rcond_value: Float32,
    rank_out: UnsafePointer[Int, MutExternalOrigin],
) raises -> Int:
    # Returns (info, rank). LDB must be >= max(M, N) since LAPACK overwrites
    # B with the result; caller is responsible for that allocation.
    var function = get_lapack_sgelsd_function()
    var m = Int32(m_value)
    var n = Int32(n_value)
    var nrhs = Int32(nrhs_value)
    var lda = Int32(m_value)
    var ldb_int = m_value if m_value > n_value else n_value
    var ldb = Int32(ldb_int)
    var rank = Int32(0)
    var info = Int32(0)
    var rcond = rcond_value
    var min_mn = m_value if m_value < n_value else n_value
    if min_mn < 1:
        min_mn = 1
    # IWORK conservative size per LAPACK 3.x: 3*min_mn*nlvl + 11*min_mn,
    # nlvl = max(0, log2(min_mn/26) + 1). Allocate generous slab.
    var iwork_size = 3 * min_mn * 32 + 11 * min_mn
    if iwork_size < 1:
        iwork_size = 1
    var iwork = alloc[Int32](iwork_size)
    var query_lwork = Int32(-1)
    var query_work = Float32(0)
    function(
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=m)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=nrhs)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](b_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldb)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](s_ptr),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](UnsafePointer(to=rcond)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=rank)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](UnsafePointer(to=query_work)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=query_lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](iwork),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    if info != 0:
        iwork.free()
        rank_out[] = 0
        return Int(info)
    var lwork_int = Int(query_work)
    if lwork_int < 1:
        lwork_int = 1
    var lwork = Int32(lwork_int)
    var work = alloc[Float32](lwork_int)
    function(
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=m)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=nrhs)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](b_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldb)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](s_ptr),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](UnsafePointer(to=rcond)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=rank)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](work),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](iwork),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    work.free()
    iwork.free()
    rank_out[] = Int(rank)
    return Int(info)


@always_inline
def lapack_dgelsd(
    m_value: Int,
    n_value: Int,
    nrhs_value: Int,
    a_ptr: UnsafePointer[Float64, MutExternalOrigin],
    b_ptr: UnsafePointer[Float64, MutExternalOrigin],
    s_ptr: UnsafePointer[Float64, MutExternalOrigin],
    rcond_value: Float64,
    rank_out: UnsafePointer[Int, MutExternalOrigin],
) raises -> Int:
    var function = get_lapack_dgelsd_function()
    var m = Int32(m_value)
    var n = Int32(n_value)
    var nrhs = Int32(nrhs_value)
    var lda = Int32(m_value)
    var ldb_int = m_value if m_value > n_value else n_value
    var ldb = Int32(ldb_int)
    var rank = Int32(0)
    var info = Int32(0)
    var rcond = rcond_value
    var min_mn = m_value if m_value < n_value else n_value
    if min_mn < 1:
        min_mn = 1
    var iwork_size = 3 * min_mn * 32 + 11 * min_mn
    if iwork_size < 1:
        iwork_size = 1
    var iwork = alloc[Int32](iwork_size)
    var query_lwork = Int32(-1)
    var query_work = Float64(0)
    function(
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=m)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=nrhs)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](b_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldb)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](s_ptr),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](UnsafePointer(to=rcond)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=rank)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](UnsafePointer(to=query_work)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=query_lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](iwork),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    if info != 0:
        iwork.free()
        rank_out[] = 0
        return Int(info)
    var lwork_int = Int(query_work)
    if lwork_int < 1:
        lwork_int = 1
    var lwork = Int32(lwork_int)
    var work = alloc[Float64](lwork_int)
    function(
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=m)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=nrhs)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](b_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldb)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](s_ptr),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](UnsafePointer(to=rcond)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=rank)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](work),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](iwork),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    work.free()
    iwork.free()
    rank_out[] = Int(rank)
    return Int(info)


@always_inline
def lapack_sgeev(
    n_value: Int,
    a_ptr: UnsafePointer[Float32, MutExternalOrigin],
    wr_ptr: UnsafePointer[Float32, MutExternalOrigin],
    wi_ptr: UnsafePointer[Float32, MutExternalOrigin],
    vr_ptr: UnsafePointer[Float32, MutExternalOrigin],
    compute_eigenvectors: Bool,
) raises -> Int:
    # Always JOBVL='N' (we never need left eigenvectors); JOBVR='V' or 'N'.
    var function = get_lapack_sgeev_function()
    var n = Int32(n_value)
    var lda = Int32(n_value)
    var ldvr = Int32(n_value) if compute_eigenvectors else Int32(1)
    var ldvl = Int32(1)
    var info = Int32(0)
    var jobvl_byte = Int8(78)  # 'N'
    var jobvr_byte = Int8(86) if compute_eigenvectors else Int8(78)
    var dummy_vl = Float32(0)
    var query_lwork = Int32(-1)
    var query_work = Float32(0)
    function(
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=jobvl_byte)),
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=jobvr_byte)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](wr_ptr),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](wi_ptr),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](UnsafePointer(to=dummy_vl)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldvl)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](vr_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldvr)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](UnsafePointer(to=query_work)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=query_lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    if info != 0:
        return Int(info)
    var lwork_int = Int(query_work)
    if lwork_int < 1:
        lwork_int = 1
    var lwork = Int32(lwork_int)
    var work = alloc[Float32](lwork_int)
    function(
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=jobvl_byte)),
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=jobvr_byte)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](wr_ptr),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](wi_ptr),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](UnsafePointer(to=dummy_vl)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldvl)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](vr_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldvr)),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](work),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    work.free()
    return Int(info)


@always_inline
def lapack_dgeev(
    n_value: Int,
    a_ptr: UnsafePointer[Float64, MutExternalOrigin],
    wr_ptr: UnsafePointer[Float64, MutExternalOrigin],
    wi_ptr: UnsafePointer[Float64, MutExternalOrigin],
    vr_ptr: UnsafePointer[Float64, MutExternalOrigin],
    compute_eigenvectors: Bool,
) raises -> Int:
    var function = get_lapack_dgeev_function()
    var n = Int32(n_value)
    var lda = Int32(n_value)
    var ldvr = Int32(n_value) if compute_eigenvectors else Int32(1)
    var ldvl = Int32(1)
    var info = Int32(0)
    var jobvl_byte = Int8(78)
    var jobvr_byte = Int8(86) if compute_eigenvectors else Int8(78)
    var dummy_vl = Float64(0)
    var query_lwork = Int32(-1)
    var query_work = Float64(0)
    function(
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=jobvl_byte)),
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=jobvr_byte)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](wr_ptr),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](wi_ptr),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](UnsafePointer(to=dummy_vl)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldvl)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](vr_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldvr)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](UnsafePointer(to=query_work)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=query_lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    if info != 0:
        return Int(info)
    var lwork_int = Int(query_work)
    if lwork_int < 1:
        lwork_int = 1
    var lwork = Int32(lwork_int)
    var work = alloc[Float64](lwork_int)
    function(
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=jobvl_byte)),
        rebind[UnsafePointer[Int8, MutAnyOrigin]](UnsafePointer(to=jobvr_byte)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=n)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](a_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lda)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](wr_ptr),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](wi_ptr),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](UnsafePointer(to=dummy_vl)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldvl)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](vr_ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=ldvr)),
        rebind[UnsafePointer[Float64, MutAnyOrigin]](work),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=lwork)),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](UnsafePointer(to=info)),
    )
    work.free()
    return Int(info)


# ============================================================================
# Parametric LAPACK dispatchers — pick s/d at compile time from `dt`.
# These are thin shims so callers in `linalg_kernels.mojo` can write
# `lapack_gesv[dt](...)` instead of `comptime if dt == DType.float32:
# lapack_sgesv(...) else: lapack_dgesv(...)` with manual pointer rebinds.
# ============================================================================


@always_inline
def lapack_gesv[
    dt: DType
](
    n_value: Int,
    rhs_columns_value: Int,
    a_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    pivot_ptr: UnsafePointer[Int32, MutExternalOrigin],
    b_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
) raises -> Int:
    comptime if dt == DType.float32:
        return lapack_sgesv(
            n_value,
            rhs_columns_value,
            rebind[UnsafePointer[Float32, MutExternalOrigin]](a_ptr),
            pivot_ptr,
            rebind[UnsafePointer[Float32, MutExternalOrigin]](b_ptr),
        )
    else:
        return lapack_dgesv(
            n_value,
            rhs_columns_value,
            rebind[UnsafePointer[Float64, MutExternalOrigin]](a_ptr),
            pivot_ptr,
            rebind[UnsafePointer[Float64, MutExternalOrigin]](b_ptr),
        )


@always_inline
def lapack_getrf[
    dt: DType
](
    n_value: Int,
    a_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    pivot_ptr: UnsafePointer[Int32, MutExternalOrigin],
) raises -> Int:
    comptime if dt == DType.float32:
        return lapack_sgetrf(
            n_value,
            rebind[UnsafePointer[Float32, MutExternalOrigin]](a_ptr),
            pivot_ptr,
        )
    else:
        return lapack_dgetrf(
            n_value,
            rebind[UnsafePointer[Float64, MutExternalOrigin]](a_ptr),
            pivot_ptr,
        )


@always_inline
def lapack_geqrf[
    dt: DType
](
    m_value: Int,
    n_value: Int,
    a_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    tau_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
) raises -> Int:
    comptime if dt == DType.float32:
        return lapack_sgeqrf(
            m_value,
            n_value,
            rebind[UnsafePointer[Float32, MutExternalOrigin]](a_ptr),
            rebind[UnsafePointer[Float32, MutExternalOrigin]](tau_ptr),
        )
    else:
        return lapack_dgeqrf(
            m_value,
            n_value,
            rebind[UnsafePointer[Float64, MutExternalOrigin]](a_ptr),
            rebind[UnsafePointer[Float64, MutExternalOrigin]](tau_ptr),
        )


@always_inline
def lapack_orgqr[
    dt: DType
](
    m_value: Int,
    n_value: Int,
    k_value: Int,
    a_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    tau_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
) raises -> Int:
    comptime if dt == DType.float32:
        return lapack_sorgqr(
            m_value,
            n_value,
            k_value,
            rebind[UnsafePointer[Float32, MutExternalOrigin]](a_ptr),
            rebind[UnsafePointer[Float32, MutExternalOrigin]](tau_ptr),
        )
    else:
        return lapack_dorgqr(
            m_value,
            n_value,
            k_value,
            rebind[UnsafePointer[Float64, MutExternalOrigin]](a_ptr),
            rebind[UnsafePointer[Float64, MutExternalOrigin]](tau_ptr),
        )


@always_inline
def lapack_potrf[
    dt: DType
](
    n_value: Int,
    a_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    upper: Bool,
) raises -> Int:
    comptime if dt == DType.float32:
        return lapack_spotrf(
            n_value,
            rebind[UnsafePointer[Float32, MutExternalOrigin]](a_ptr),
            upper,
        )
    else:
        return lapack_dpotrf(
            n_value,
            rebind[UnsafePointer[Float64, MutExternalOrigin]](a_ptr),
            upper,
        )


@always_inline
def lapack_syev[
    dt: DType
](
    n_value: Int,
    a_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    w_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    compute_eigenvectors: Bool,
    upper: Bool,
) raises -> Int:
    comptime if dt == DType.float32:
        return lapack_ssyev(
            n_value,
            rebind[UnsafePointer[Float32, MutExternalOrigin]](a_ptr),
            rebind[UnsafePointer[Float32, MutExternalOrigin]](w_ptr),
            compute_eigenvectors,
            upper,
        )
    else:
        return lapack_dsyev(
            n_value,
            rebind[UnsafePointer[Float64, MutExternalOrigin]](a_ptr),
            rebind[UnsafePointer[Float64, MutExternalOrigin]](w_ptr),
            compute_eigenvectors,
            upper,
        )


@always_inline
def lapack_geev[
    dt: DType
](
    n_value: Int,
    a_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    wr_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    wi_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    vr_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    compute_eigenvectors: Bool,
) raises -> Int:
    comptime if dt == DType.float32:
        return lapack_sgeev(
            n_value,
            rebind[UnsafePointer[Float32, MutExternalOrigin]](a_ptr),
            rebind[UnsafePointer[Float32, MutExternalOrigin]](wr_ptr),
            rebind[UnsafePointer[Float32, MutExternalOrigin]](wi_ptr),
            rebind[UnsafePointer[Float32, MutExternalOrigin]](vr_ptr),
            compute_eigenvectors,
        )
    else:
        return lapack_dgeev(
            n_value,
            rebind[UnsafePointer[Float64, MutExternalOrigin]](a_ptr),
            rebind[UnsafePointer[Float64, MutExternalOrigin]](wr_ptr),
            rebind[UnsafePointer[Float64, MutExternalOrigin]](wi_ptr),
            rebind[UnsafePointer[Float64, MutExternalOrigin]](vr_ptr),
            compute_eigenvectors,
        )


@always_inline
def lapack_gesdd[
    dt: DType
](
    m_value: Int,
    n_value: Int,
    a_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    s_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    u_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    vt_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    full_matrices: Bool,
    compute_uv: Bool,
) raises -> Int:
    comptime if dt == DType.float32:
        return lapack_sgesdd(
            m_value,
            n_value,
            rebind[UnsafePointer[Float32, MutExternalOrigin]](a_ptr),
            rebind[UnsafePointer[Float32, MutExternalOrigin]](s_ptr),
            rebind[UnsafePointer[Float32, MutExternalOrigin]](u_ptr),
            rebind[UnsafePointer[Float32, MutExternalOrigin]](vt_ptr),
            full_matrices,
            compute_uv,
        )
    else:
        return lapack_dgesdd(
            m_value,
            n_value,
            rebind[UnsafePointer[Float64, MutExternalOrigin]](a_ptr),
            rebind[UnsafePointer[Float64, MutExternalOrigin]](s_ptr),
            rebind[UnsafePointer[Float64, MutExternalOrigin]](u_ptr),
            rebind[UnsafePointer[Float64, MutExternalOrigin]](vt_ptr),
            full_matrices,
            compute_uv,
        )


@always_inline
def lapack_gelsd[
    dt: DType
](
    m_value: Int,
    n_value: Int,
    nrhs_value: Int,
    a_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    b_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    s_ptr: UnsafePointer[Scalar[dt], MutExternalOrigin],
    rcond: Scalar[dt],
    rank_out_ptr: UnsafePointer[Int, MutExternalOrigin],
) raises -> Int:
    comptime if dt == DType.float32:
        return lapack_sgelsd(
            m_value,
            n_value,
            nrhs_value,
            rebind[UnsafePointer[Float32, MutExternalOrigin]](a_ptr),
            rebind[UnsafePointer[Float32, MutExternalOrigin]](b_ptr),
            rebind[UnsafePointer[Float32, MutExternalOrigin]](s_ptr),
            rebind[Float32](rcond),
            rank_out_ptr,
        )
    else:
        return lapack_dgelsd(
            m_value,
            n_value,
            nrhs_value,
            rebind[UnsafePointer[Float64, MutExternalOrigin]](a_ptr),
            rebind[UnsafePointer[Float64, MutExternalOrigin]](b_ptr),
            rebind[UnsafePointer[Float64, MutExternalOrigin]](s_ptr),
            rebind[Float64](rcond),
            rank_out_ptr,
        )
