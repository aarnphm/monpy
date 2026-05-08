"""Complex-array logical-index helpers shared by `unary_ops` (closure path)
and `matmul_ops` (the schoolbook complex inner loop).

These wrap the `get_physical_c{64,128}_real/imag` / `set_physical_c{64,128}`
accessors with `physical_offset` resolution and a real-dtype fallback so call
sites can pretend the array is "complex enough" without branching at every
step. Real-dtype values pass through with imag=0.
"""

from array import (
    Array,
    get_logical_as_f64,
    get_physical_c128_imag,
    get_physical_c128_real,
    get_physical_c64_imag,
    get_physical_c64_real,
    physical_offset,
    set_logical_from_f64,
    set_physical_c128,
    set_physical_c64,
)
from domain import ArrayDType


def _complex_real(arr: Array, logical: Int) raises -> Float64:
    """Helper: read real part of a complex array at logical index."""
    var phys = physical_offset(arr, logical)
    if arr.dtype_code == ArrayDType.COMPLEX64.value:
        return Float64(get_physical_c64_real(arr, phys))
    if arr.dtype_code == ArrayDType.COMPLEX128.value:
        return get_physical_c128_real(arr, phys)
    return get_logical_as_f64(arr, logical)


def _complex_imag(arr: Array, logical: Int) raises -> Float64:
    """Helper: read imag part of a complex array at logical index."""
    var phys = physical_offset(arr, logical)
    if arr.dtype_code == ArrayDType.COMPLEX64.value:
        return Float64(get_physical_c64_imag(arr, phys))
    if arr.dtype_code == ArrayDType.COMPLEX128.value:
        return get_physical_c128_imag(arr, phys)
    return 0.0


def _complex_store(mut arr: Array, logical: Int, real: Float64, imag: Float64) raises:
    """Helper: write real+imag to a complex array at logical index."""
    var phys = physical_offset(arr, logical)
    if arr.dtype_code == ArrayDType.COMPLEX64.value:
        set_physical_c64(arr, phys, Float32(real), Float32(imag))
    elif arr.dtype_code == ArrayDType.COMPLEX128.value:
        set_physical_c128(arr, phys, real, imag)
    else:
        set_logical_from_f64(arr, logical, real)
