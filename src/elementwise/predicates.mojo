"""Small dtype/layout predicates and the rank-2 BLAS-layout probe.

`is_float_dtype` / `is_typed_simd_dtype` answer "does this dtype hit a
typed-SIMD fast path?" — used by every `maybe_*` dispatcher to decide
whether to recurse into a typed kernel or fall through.

`Rank2BlasLayout` packages the "can this rank-2 array drive a BLAS
GEMM/GEMV call directly?" question. Returns transpose flag and leading
dimension when the answer is yes; otherwise the caller transposes /
copies before calling the BLAS routine. Apple Accelerate's `cblas_*` and
`lapack_*` all expect column-major; we live in row-major land, so we
either ask for `CblasRowMajor` (BLAS only — LAPACK has no row-major flag)
or pre-transpose into a scratch buffer.
"""

from array import (
    Array,
    has_negative_strides,
    has_zero_strides,
    is_c_contiguous,
)
from domain import ArrayDType


def is_float_dtype(dtype_code: Int) -> Bool:
    return dtype_code == ArrayDType.FLOAT32.value or dtype_code == ArrayDType.FLOAT64.value


def is_typed_simd_dtype(dtype_code: Int) -> Bool:
    """Returns True for dtypes that have a typed-vec SIMD dispatch path
    in `maybe_binary_same_shape_contiguous` and friends. All ints + f16
    join the f32/f64 fast paths."""
    return (
        dtype_code == ArrayDType.FLOAT32.value
        or dtype_code == ArrayDType.FLOAT64.value
        or dtype_code == ArrayDType.FLOAT16.value
        or dtype_code == ArrayDType.INT64.value
        or dtype_code == ArrayDType.INT32.value
        or dtype_code == ArrayDType.INT16.value
        or dtype_code == ArrayDType.INT8.value
        or dtype_code == ArrayDType.UINT64.value
        or dtype_code == ArrayDType.UINT32.value
        or dtype_code == ArrayDType.UINT16.value
        or dtype_code == ArrayDType.UINT8.value
    )


def is_contiguous_float_array(array: Array) raises -> Bool:
    return is_float_dtype(array.dtype_code) and is_c_contiguous(array)


def is_contiguous_typed_simd_array(array: Array) raises -> Bool:
    return is_typed_simd_dtype(array.dtype_code) and is_c_contiguous(array)


@fieldwise_init
struct Rank2BlasLayout(ImplicitlyCopyable, Movable, Writable):
    var can_use: Bool
    var transpose: Bool
    var leading_dim: Int


def max_int(lhs: Int, rhs: Int) -> Int:
    if lhs > rhs:
        return lhs
    return rhs


def rank2_blas_layout(array: Array) raises -> Rank2BlasLayout:
    if len(array.shape) != 2:
        return Rank2BlasLayout(False, False, 0)
    var rows = array.shape[0]
    var cols = array.shape[1]
    if rows == 0 or cols == 0:
        return Rank2BlasLayout(False, False, 0)
    if has_negative_strides(array) or has_zero_strides(array):
        return Rank2BlasLayout(False, False, 0)
    if cols == 1 or array.strides[1] == 1:
        var lda = array.strides[0]
        if rows == 1:
            lda = max_int(1, cols)
        if lda >= max_int(1, cols):
            return Rank2BlasLayout(True, False, lda)
    if rows == 1 or array.strides[0] == 1:
        var lda = array.strides[1]
        if cols == 1:
            lda = max_int(1, rows)
        if lda >= max_int(1, rows):
            return Rank2BlasLayout(True, True, lda)
    return Rank2BlasLayout(False, False, 0)
