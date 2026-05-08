"""dtype-dispatch helpers for typed contig kernels.

`dispatch_real_typed_simd_binary` and `dispatch_real_typed_simd_unary` take
a comptime-typed kernel and a runtime `dtype_code`, and emit one
specialised tail-call per real dtype (11 for unary, 11 for binary). The
kernel parameter is comptime — the compiler unrolls the if-chain into 11
distinct call sites, no vtable, no runtime branch into the kernel body.

Both helpers assume the caller has already established that all relevant
arrays share `dtype_code` and are c-contiguous; they only do the dtype →
typed-pointer projection and the kernel call. Returns False if the
dtype isn't covered (caller falls through to f64 round-trip).
"""

from array import Array, contiguous_ptr
from domain import ArrayDType


comptime BinaryContigKernel = def[dt: DType](
    UnsafePointer[Scalar[dt], MutExternalOrigin],
    UnsafePointer[Scalar[dt], MutExternalOrigin],
    UnsafePointer[Scalar[dt], MutExternalOrigin],
    Int,
    Int,
) thin raises -> None
"""Shape of any same-dtype contiguous binary kernel: three pointers, size, op code.
"""


def dispatch_real_typed_simd_binary[
    kernel: BinaryContigKernel,
](dtype_code: Int, lhs: Array, rhs: Array, mut result: Array, size: Int, op: Int,) raises -> Bool:
    """dtype dispatch from runtime `dtype_code` to comptime-typed kernel.

    Caller invariant: `lhs.dtype_code == rhs.dtype_code == result.dtype_code` and all three arrays are c-contiguous.
    Returns True if a typed path was taken; False if the dtype isn't covered (caller should fall through to the f64 round-trip path).
    """
    if dtype_code == ArrayDType.FLOAT32.value:
        kernel[DType.float32](
            contiguous_ptr[DType.float32](lhs),
            contiguous_ptr[DType.float32](rhs),
            contiguous_ptr[DType.float32](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.FLOAT64.value:
        kernel[DType.float64](
            contiguous_ptr[DType.float64](lhs),
            contiguous_ptr[DType.float64](rhs),
            contiguous_ptr[DType.float64](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.INT64.value:
        kernel[DType.int64](
            contiguous_ptr[DType.int64](lhs),
            contiguous_ptr[DType.int64](rhs),
            contiguous_ptr[DType.int64](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.INT32.value:
        kernel[DType.int32](
            contiguous_ptr[DType.int32](lhs),
            contiguous_ptr[DType.int32](rhs),
            contiguous_ptr[DType.int32](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.UINT64.value:
        kernel[DType.uint64](
            contiguous_ptr[DType.uint64](lhs),
            contiguous_ptr[DType.uint64](rhs),
            contiguous_ptr[DType.uint64](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.UINT32.value:
        kernel[DType.uint32](
            contiguous_ptr[DType.uint32](lhs),
            contiguous_ptr[DType.uint32](rhs),
            contiguous_ptr[DType.uint32](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.INT16.value:
        kernel[DType.int16](
            contiguous_ptr[DType.int16](lhs),
            contiguous_ptr[DType.int16](rhs),
            contiguous_ptr[DType.int16](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.INT8.value:
        kernel[DType.int8](
            contiguous_ptr[DType.int8](lhs),
            contiguous_ptr[DType.int8](rhs),
            contiguous_ptr[DType.int8](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.UINT16.value:
        kernel[DType.uint16](
            contiguous_ptr[DType.uint16](lhs),
            contiguous_ptr[DType.uint16](rhs),
            contiguous_ptr[DType.uint16](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.UINT8.value:
        kernel[DType.uint8](
            contiguous_ptr[DType.uint8](lhs),
            contiguous_ptr[DType.uint8](rhs),
            contiguous_ptr[DType.uint8](result),
            size,
            op,
        )
        return True
    if dtype_code == ArrayDType.FLOAT16.value:
        kernel[DType.float16](
            contiguous_ptr[DType.float16](lhs),
            contiguous_ptr[DType.float16](rhs),
            contiguous_ptr[DType.float16](result),
            size,
            op,
        )
        return True
    return False


comptime UnaryContigKernel = def[dt: DType](
    UnsafePointer[Scalar[dt], MutExternalOrigin],
    UnsafePointer[Scalar[dt], MutExternalOrigin],
    Int,
    Int,
) thin raises -> None
"""Shape of any same-dtype contiguous unary kernel: src ptr, out ptr, size, op code.

Used by `maybe_unary_preserve_contiguous` (14-way real dispatch, integer-friendly)
and `maybe_unary_contiguous` (float-only sub-variant, integer dtypes raise).
"""


def dispatch_real_typed_simd_unary[
    kernel: UnaryContigKernel,
](dtype_code: Int, src: Array, mut result: Array, size: Int, op: Int,) raises -> Bool:
    """11-way real-dtype dispatch for unary kernels (src ptr → out ptr, both same dtype).

    Caller invariant: `src.dtype_code == result.dtype_code` and both arrays are
    c-contiguous. Returns True if a typed path was taken; False if the dtype isn't
    covered (caller falls through to f64 round-trip or complex specialization).
    """
    if dtype_code == ArrayDType.FLOAT32.value:
        kernel[DType.float32](contiguous_ptr[DType.float32](src), contiguous_ptr[DType.float32](result), size, op)
        return True
    if dtype_code == ArrayDType.FLOAT64.value:
        kernel[DType.float64](contiguous_ptr[DType.float64](src), contiguous_ptr[DType.float64](result), size, op)
        return True
    if dtype_code == ArrayDType.INT64.value:
        kernel[DType.int64](contiguous_ptr[DType.int64](src), contiguous_ptr[DType.int64](result), size, op)
        return True
    if dtype_code == ArrayDType.INT32.value:
        kernel[DType.int32](contiguous_ptr[DType.int32](src), contiguous_ptr[DType.int32](result), size, op)
        return True
    if dtype_code == ArrayDType.UINT64.value:
        kernel[DType.uint64](contiguous_ptr[DType.uint64](src), contiguous_ptr[DType.uint64](result), size, op)
        return True
    if dtype_code == ArrayDType.UINT32.value:
        kernel[DType.uint32](contiguous_ptr[DType.uint32](src), contiguous_ptr[DType.uint32](result), size, op)
        return True
    if dtype_code == ArrayDType.INT16.value:
        kernel[DType.int16](contiguous_ptr[DType.int16](src), contiguous_ptr[DType.int16](result), size, op)
        return True
    if dtype_code == ArrayDType.INT8.value:
        kernel[DType.int8](contiguous_ptr[DType.int8](src), contiguous_ptr[DType.int8](result), size, op)
        return True
    if dtype_code == ArrayDType.UINT16.value:
        kernel[DType.uint16](contiguous_ptr[DType.uint16](src), contiguous_ptr[DType.uint16](result), size, op)
        return True
    if dtype_code == ArrayDType.UINT8.value:
        kernel[DType.uint8](contiguous_ptr[DType.uint8](src), contiguous_ptr[DType.uint8](result), size, op)
        return True
    if dtype_code == ArrayDType.FLOAT16.value:
        kernel[DType.float16](contiguous_ptr[DType.float16](src), contiguous_ptr[DType.float16](result), size, op)
        return True
    return False
