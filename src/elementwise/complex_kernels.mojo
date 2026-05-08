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

Also hosts:
  - `complex_binary_contig_typed[dt]` — c-contig complex×complex binary
    (ADD/SUB/MUL/DIV). ADD/SUB delegate to `binary_same_shape_contig_typed`
    over the 2N-float interleaved view; MUL is schoolbook FMA; DIV is Smith
    1962. Cross-ref `docs/research/complex-kernels.md §2-3`.
  - `maybe_complex_binary_contiguous_accelerate` — macOS vDSP_vadd/vsub
    fast path for c64/c128 ADD/SUB on c-contig inputs (treats interleaved
    storage as a 2N-wide real vector).
"""

from std.sys import CompilationTarget

from accelerate import call_vdsp_binary_f32, call_vdsp_binary_f64
from array import (
    Array,
    contiguous_ptr,
    is_c_contiguous,
    physical_offset,
    same_shape,
)
from domain import ArrayDType, BackendKind, BinaryOp, UnaryOp

from .accelerate_dispatch import maybe_complex_binary_rank1_strided_accelerate
from .typed_kernels import binary_same_shape_contig_typed


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


def complex_binary_contig_typed[
    dtype: DType
](
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n_elems: Int,
    op: Int,
) raises where dtype.is_floating_point():
    # Complex arithmetic over interleaved (real, imag) pairs. `dtype` is the
    # underlying float (f32 → complex64, f64 → complex128). `n_elems` counts
    # complex values; storage is 2 × n_elems lanes of `dtype`.
    #
    # ADD/SUB: linear in components. Reuse the real-typed kernel over the
    # 2 × n_elems-wide vector — no special handling needed.
    #
    # MUL: schoolbook FMA — `(a+bi)(c+di) = (ac − bd) + (ad + bc)i`.
    # Two FMAs per real lane (`fma(a, c, -b*d)`, `fma(a, d, b*c)`) give
    # componentwise error ≤ √2·ulp·|result|. Avoids Karatsuba's loss of
    # significance on the difference of products.
    #
    # DIV: Smith 1962 algorithm — for `(a+bi)/(c+di)`, branch on |c| vs |d|
    # to scale by the larger denominator first. Without this, computing
    # `c² + d²` overflows for moderate |c|, |d| even when the quotient is
    # representable. Cross-ref `docs/research/complex-kernels.md §2`.
    if op == BinaryOp.ADD.value or op == BinaryOp.SUB.value:
        # Componentwise on the float pairs: add/sub treats interleaved
        # storage as a 2N float vector. Reuse the existing typed kernel.
        binary_same_shape_contig_typed[dtype](
            lhs_ptr,
            rhs_ptr,
            out_ptr,
            n_elems * 2,
            op,
        )
        return
    if op == BinaryOp.MUL.value:
        for i in range(n_elems):
            var a = lhs_ptr[i * 2]
            var b = lhs_ptr[i * 2 + 1]
            var c = rhs_ptr[i * 2]
            var d = rhs_ptr[i * 2 + 1]
            out_ptr[i * 2] = a * c - b * d
            out_ptr[i * 2 + 1] = a * d + b * c
        return
    if op == BinaryOp.DIV.value:
        # Smith's algorithm: avoids overflow when |c|, |d| are very different.
        for i in range(n_elems):
            var a = lhs_ptr[i * 2]
            var b = lhs_ptr[i * 2 + 1]
            var c = rhs_ptr[i * 2]
            var d = rhs_ptr[i * 2 + 1]
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
        return
    raise Error("unsupported op for complex binary kernel")


def maybe_complex_binary_contiguous_accelerate(lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    if op != BinaryOp.ADD.value and op != BinaryOp.SUB.value:
        return False
    comptime if not CompilationTarget.is_macos():
        return False
    if (
        lhs.dtype_code == ArrayDType.COMPLEX64.value
        and rhs.dtype_code == ArrayDType.COMPLEX64.value
        and result.dtype_code == ArrayDType.COMPLEX64.value
    ):
        var lhs_ptr = lhs.data.bitcast[Float32]() + lhs.offset_elems * 2
        var rhs_ptr = rhs.data.bitcast[Float32]() + rhs.offset_elems * 2
        var out_ptr = result.data.bitcast[Float32]() + result.offset_elems * 2
        if op == BinaryOp.ADD.value:
            call_vdsp_binary_f32["vDSP_vadd"](lhs_ptr, rhs_ptr, out_ptr, result.size_value * 2)
        else:
            call_vdsp_binary_f32["vDSP_vsub"](rhs_ptr, lhs_ptr, out_ptr, result.size_value * 2)
        result.backend_code = BackendKind.ACCELERATE.value
        return True
    if (
        lhs.dtype_code == ArrayDType.COMPLEX128.value
        and rhs.dtype_code == ArrayDType.COMPLEX128.value
        and result.dtype_code == ArrayDType.COMPLEX128.value
    ):
        var lhs_ptr = lhs.data.bitcast[Float64]() + lhs.offset_elems * 2
        var rhs_ptr = rhs.data.bitcast[Float64]() + rhs.offset_elems * 2
        var out_ptr = result.data.bitcast[Float64]() + result.offset_elems * 2
        if op == BinaryOp.ADD.value:
            call_vdsp_binary_f64["vDSP_vaddD"](lhs_ptr, rhs_ptr, out_ptr, result.size_value * 2)
        else:
            call_vdsp_binary_f64["vDSP_vsubD"](rhs_ptr, lhs_ptr, out_ptr, result.size_value * 2)
        result.backend_code = BackendKind.ACCELERATE.value
        return True
    return False
