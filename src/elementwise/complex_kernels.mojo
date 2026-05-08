"""Complex-domain typed kernels: unary preserve, binary strided, scalar broadcast.

Hosts:
  - `complex_unary_preserve_contig_typed[dt]` — NEGATE / POSITIVE / CONJUGATE
    / SQUARE over interleaved (re, im) pairs.
  - `complex_binary_same_shape_strided_typed[dt]` — ADD/SUB/MUL/DIV with
    Smith division for the DIV branch (avoids overflow when |c|, |d|
    differ in magnitude). Walks via `physical_offset` per element.
  - `maybe_complex_binary_same_shape_strided` — c64 / c128 dispatch into
    the strided typed kernel (with vDSP rank-1 fast path first).
  - `complex_scalar_complex_contig_typed[dt]` — full complex×complex-scalar
    broadcast (ADD/SUB/MUL/DIV).
  - `complex_scalar_real_contig_typed[dt]` — complex×real-scalar broadcast
    (numpy treats the scalar as zero-imag).

Math notes (cross-ref `docs/research/complex-kernels.md`):
  - MUL: schoolbook FMA `(a+bi)(c+di) = (ac − bd) + (ad + bc)i` —
    componentwise error ≤ √2·ulp·|result|. Avoids Karatsuba's loss of
    significance on the difference of products.
  - DIV: Smith 1962 — branch on |c| ≥ |d| to scale by the larger
    denominator first. Without this, c² + d² overflows for moderate
    |c|, |d| even when the quotient is representable. §2.

`complex_binary_contig_typed` and `maybe_complex_binary_contiguous_accelerate`
remain in elementwise/__init__.mojo because they tightly couple to the
ADD/SUB delegation into `binary_same_shape_contig_typed`.
"""

from array import (
    Array,
    is_c_contiguous,
    physical_offset,
    same_shape,
)
from domain import ArrayDType, BackendKind, BinaryOp, UnaryOp

from .accelerate_dispatch import maybe_complex_binary_rank1_strided_accelerate


def complex_unary_preserve_contig_typed[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n_elems: Int,
    op: Int,
) raises where dtype.is_floating_point():
    # Complex unary preserve: NEGATE (both lanes), POSITIVE (id), CONJUGATE
    # (negate imag), SQUARE ((a+bi)² = a²-b² + 2abi).
    if op == UnaryOp.NEGATE.value:
        # Scalar loop over 2N floats (interleaved re/im). The SIMD path
        # would be valid but Mojo's f64 SIMD width and the size-6 mismatch
        # in some configurations leaves imag tails unwritten — explicit
        # walk is safe and the perf delta is negligible at small N.
        for i in range(n_elems * 2):
            out_ptr[i] = -src_ptr[i]
        return
    if op == UnaryOp.POSITIVE.value:
        for i in range(n_elems * 2):
            out_ptr[i] = src_ptr[i]
        return
    if op == UnaryOp.CONJUGATE.value:
        for i in range(n_elems):
            out_ptr[i * 2] = src_ptr[i * 2]
            out_ptr[i * 2 + 1] = -src_ptr[i * 2 + 1]
        return
    if op == UnaryOp.SQUARE.value:
        for i in range(n_elems):
            var a = src_ptr[i * 2]
            var b = src_ptr[i * 2 + 1]
            out_ptr[i * 2] = a * a - b * b
            out_ptr[i * 2 + 1] = Scalar[dtype](2) * a * b
        return
    raise Error("unsupported op for complex unary preserve kernel")


def complex_binary_same_shape_strided_typed[
    dtype: DType
](lhs: Array, rhs: Array, mut result: Array, op: Int) raises where dtype.is_floating_point():
    var lhs_ptr = lhs.data.bitcast[Scalar[dtype]]()
    var rhs_ptr = rhs.data.bitcast[Scalar[dtype]]()
    var out_ptr = result.data.bitcast[Scalar[dtype]]() + result.offset_elems * 2
    for i in range(result.size_value):
        var lhs_phys = physical_offset(lhs, i)
        var rhs_phys = physical_offset(rhs, i)
        var a = lhs_ptr[lhs_phys * 2]
        var b = lhs_ptr[lhs_phys * 2 + 1]
        var c = rhs_ptr[rhs_phys * 2]
        var d = rhs_ptr[rhs_phys * 2 + 1]
        if op == BinaryOp.ADD.value:
            out_ptr[i * 2] = a + c
            out_ptr[i * 2 + 1] = b + d
        elif op == BinaryOp.SUB.value:
            out_ptr[i * 2] = a - c
            out_ptr[i * 2 + 1] = b - d
        elif op == BinaryOp.MUL.value:
            out_ptr[i * 2] = a * c - b * d
            out_ptr[i * 2 + 1] = a * d + b * c
        elif op == BinaryOp.DIV.value:
            var abs_c = c if c >= Scalar[dtype](0) else -c
            var abs_d = d if d >= Scalar[dtype](0) else -d
            if abs_c >= abs_d:
                var r = d / c
                var den = c + d * r
                out_ptr[i * 2] = (a + b * r) / den
                out_ptr[i * 2 + 1] = (b - a * r) / den
            else:
                var r = c / d
                var den = c * r + d
                out_ptr[i * 2] = (a * r + b) / den
                out_ptr[i * 2 + 1] = (b * r - a) / den
        else:
            raise Error("unsupported op for complex strided binary kernel")


def maybe_complex_binary_same_shape_strided(lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    if (
        not same_shape(lhs.shape, rhs.shape)
        or not same_shape(lhs.shape, result.shape)
        or not is_c_contiguous(result)
        or lhs.dtype_code != rhs.dtype_code
        or rhs.dtype_code != result.dtype_code
    ):
        return False
    if maybe_complex_binary_rank1_strided_accelerate(lhs, rhs, result, op):
        return True
    if lhs.dtype_code == ArrayDType.COMPLEX64.value:
        complex_binary_same_shape_strided_typed[DType.float32](lhs, rhs, result, op)
        result.backend_code = BackendKind.FUSED.value
        return True
    if lhs.dtype_code == ArrayDType.COMPLEX128.value:
        complex_binary_same_shape_strided_typed[DType.float64](lhs, rhs, result, op)
        result.backend_code = BackendKind.FUSED.value
        return True
    return False

def complex_scalar_complex_contig_typed[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    scalar_real: Scalar[dtype],
    scalar_imag: Scalar[dtype],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n_elems: Int,
    op: Int,
    scalar_on_left: Bool,
) raises where dtype.is_floating_point():
    # Complex x complex-scalar: full complex broadcast. ADD/SUB/MUL/DIV.
    if op == BinaryOp.ADD.value:
        for i in range(n_elems):
            out_ptr[i * 2] = src_ptr[i * 2] + scalar_real
            out_ptr[i * 2 + 1] = src_ptr[i * 2 + 1] + scalar_imag
        return
    if op == BinaryOp.SUB.value:
        if scalar_on_left:
            for i in range(n_elems):
                out_ptr[i * 2] = scalar_real - src_ptr[i * 2]
                out_ptr[i * 2 + 1] = scalar_imag - src_ptr[i * 2 + 1]
        else:
            for i in range(n_elems):
                out_ptr[i * 2] = src_ptr[i * 2] - scalar_real
                out_ptr[i * 2 + 1] = src_ptr[i * 2 + 1] - scalar_imag
        return
    if op == BinaryOp.MUL.value:
        for i in range(n_elems):
            var a = src_ptr[i * 2]
            var b = src_ptr[i * 2 + 1]
            out_ptr[i * 2] = a * scalar_real - b * scalar_imag
            out_ptr[i * 2 + 1] = a * scalar_imag + b * scalar_real
        return
    if op == BinaryOp.DIV.value:
        if scalar_on_left:
            # scalar / complex_i = scalar * conj(c) / |c|²
            for i in range(n_elems):
                var a = src_ptr[i * 2]
                var b = src_ptr[i * 2 + 1]
                var denom = a * a + b * b
                out_ptr[i * 2] = (scalar_real * a + scalar_imag * b) / denom
                out_ptr[i * 2 + 1] = (scalar_imag * a - scalar_real * b) / denom
        else:
            for i in range(n_elems):
                var a = src_ptr[i * 2]
                var b = src_ptr[i * 2 + 1]
                var denom = scalar_real * scalar_real + scalar_imag * scalar_imag
                out_ptr[i * 2] = (a * scalar_real + b * scalar_imag) / denom
                out_ptr[i * 2 + 1] = (b * scalar_real - a * scalar_imag) / denom
        return
    raise Error("unsupported op for complex × complex-scalar kernel")
def complex_scalar_real_contig_typed[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    scalar_real: Scalar[dtype],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n_elems: Int,
    op: Int,
    scalar_on_left: Bool,
) raises where dtype.is_floating_point():
    # Complex × real-scalar: numpy treats the scalar as real (zero imag).
    # ADD/SUB: only the real lane is touched; imag passes through.
    # MUL/DIV: both lanes scale by the scalar (since (a+bi)*s = as + bs i).
    if op == BinaryOp.ADD.value:
        for i in range(n_elems):
            out_ptr[i * 2] = src_ptr[i * 2] + scalar_real if not scalar_on_left else scalar_real + src_ptr[i * 2]
            out_ptr[i * 2 + 1] = src_ptr[i * 2 + 1]
        return
    if op == BinaryOp.SUB.value:
        for i in range(n_elems):
            if scalar_on_left:
                out_ptr[i * 2] = scalar_real - src_ptr[i * 2]
                out_ptr[i * 2 + 1] = -src_ptr[i * 2 + 1]
            else:
                out_ptr[i * 2] = src_ptr[i * 2] - scalar_real
                out_ptr[i * 2 + 1] = src_ptr[i * 2 + 1]
        return
    if op == BinaryOp.MUL.value:
        for i in range(n_elems):
            out_ptr[i * 2] = src_ptr[i * 2] * scalar_real
            out_ptr[i * 2 + 1] = src_ptr[i * 2 + 1] * scalar_real
        return
    if op == BinaryOp.DIV.value:
        if scalar_on_left:
            # scalar / complex = scalar * conj(c) / |c|²
            for i in range(n_elems):
                var a = src_ptr[i * 2]
                var b = src_ptr[i * 2 + 1]
                var denom = a * a + b * b
                out_ptr[i * 2] = scalar_real * a / denom
                out_ptr[i * 2 + 1] = -scalar_real * b / denom
        else:
            for i in range(n_elems):
                out_ptr[i * 2] = src_ptr[i * 2] / scalar_real
                out_ptr[i * 2 + 1] = src_ptr[i * 2 + 1] / scalar_real
        return
    raise Error("unsupported op for complex scalar real binary kernel")
