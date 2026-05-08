"""Unary maybe_* dispatchers: contiguous, preserve, rank-2 strided.

These are the entry points called by the upstream `apply_unary_*` ops in
`src/create/ops/elementwise.mojo`. Each `maybe_*` returns True if it took a
fast path, False if the caller should fall back to the f64 round-trip.
"""

from std.sys import CompilationTarget

from array import (
    Array,
    contiguous_ptr,
    is_c_contiguous,
)
from domain import ArrayDType, BackendKind, UnaryOp

from .accelerate_dispatch import maybe_unary_accelerate
from .kernels.complex import complex_unary_preserve_contig_typed
from .dispatch_helpers import dispatch_real_typed_simd_unary
from .predicates import is_contiguous_float_array
from .kernels.typed import (
    unary_contig_typed,
    unary_preserve_contig_typed,
    unary_rank2_strided_typed,
)


def maybe_unary_preserve_contiguous(src: Array, mut result: Array, op: Int) raises -> Bool:
    # Same-dtype c-contig fast path for preserve-dtype unary ops.
    if src.dtype_code != result.dtype_code or not is_c_contiguous(src) or not is_c_contiguous(result):
        return False
    # Complex paths.
    if src.dtype_code == ArrayDType.COMPLEX64.value:
        complex_unary_preserve_contig_typed[DType.float32](
            contiguous_ptr[DType.float32](src),
            contiguous_ptr[DType.float32](result),
            src.size_value,
            op,
        )
        return True
    if src.dtype_code == ArrayDType.COMPLEX128.value:
        complex_unary_preserve_contig_typed[DType.float64](
            contiguous_ptr[DType.float64](src),
            contiguous_ptr[DType.float64](result),
            src.size_value,
            op,
        )
        return True
    # 11-way real-dtype dispatch via unary helper. f16 not in `unary_preserve_contig_typed`
    # support set yet (only the 10 real-vec dtypes), so explicit fallback below.
    if dispatch_real_typed_simd_unary[unary_preserve_contig_typed](src.dtype_code, src, result, src.size_value, op):
        return True
    return False


def maybe_unary_contiguous(src: Array, mut result: Array, op: Int) raises -> Bool:
    if not is_contiguous_float_array(src) or not is_contiguous_float_array(result):
        return False
    if src.dtype_code == ArrayDType.FLOAT32.value and result.dtype_code == ArrayDType.FLOAT32.value:
        comptime if CompilationTarget.is_macos():
            if maybe_unary_accelerate[DType.float32](src, result, op):
                return True
        unary_contig_typed[DType.float32](
            contiguous_ptr[DType.float32](src),
            contiguous_ptr[DType.float32](result),
            src.size_value,
            op,
        )
        return True
    if op == UnaryOp.LOG.value:
        return False
    if src.dtype_code == ArrayDType.FLOAT64.value and result.dtype_code == ArrayDType.FLOAT64.value:
        comptime if CompilationTarget.is_macos():
            if maybe_unary_accelerate[DType.float64](src, result, op):
                return True
        unary_contig_typed[DType.float64](
            contiguous_ptr[DType.float64](src),
            contiguous_ptr[DType.float64](result),
            src.size_value,
            op,
        )
        return True
    return False


def maybe_unary_rank2_strided(src: Array, mut result: Array, op: Int) raises -> Bool:
    if len(src.shape) != 2 or src.dtype_code != result.dtype_code or not is_c_contiguous(result):
        return False
    if src.dtype_code == ArrayDType.FLOAT32.value:
        unary_rank2_strided_typed[DType.float32](src, result, op)
        result.backend_code = BackendKind.FUSED.value
        return True
    if src.dtype_code == ArrayDType.FLOAT64.value:
        unary_rank2_strided_typed[DType.float64](src, result, op)
        result.backend_code = BackendKind.FUSED.value
        return True
    return False
