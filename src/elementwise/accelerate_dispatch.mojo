"""Accelerate-framework fast paths for unary/binary contiguous + rank-1 strided.

Each `maybe_*_accelerate_*` shim:
  1. checks dtype/shape/contig invariants
  2. extracts c-contig pointers
  3. calls the matching vDSP / libvMath routine
  4. stamps `result.backend_code = BackendKind.ACCELERATE.value`

Only the four basic arithmetic ops (add/sub/mul/div) and the four unary
transcendentals (sin/cos/exp/log) are dispatched here — anything else
falls through to the typed-SIMD path. macOS-only (CompilationTarget gate);
Linux drops through to the next dispatch tier.
"""

from std.sys import CompilationTarget

from accelerate import (
    call_vdsp_binary,
    call_vdsp_binary_f32,
    call_vdsp_binary_strided,
    call_vdsp_binary_strided_f32,
    call_vv,
    call_vv_f32,
)
from array import (
    Array,
    contiguous_ptr,
    is_c_contiguous,
    same_shape,
)
from domain import ArrayDType, BackendKind, BinaryOp, UnaryOp


def maybe_unary_accelerate[dt: DType](src: Array, mut result: Array, op: Int) raises -> Bool:
    var src_ptr = contiguous_ptr[dt](src)
    var out_ptr = contiguous_ptr[dt](result)
    if op == UnaryOp.SIN.value:
        call_vv[dt, "vvsinf", "vvsin"](out_ptr, src_ptr, src.size_value)
    elif op == UnaryOp.COS.value:
        call_vv[dt, "vvcosf", "vvcos"](out_ptr, src_ptr, src.size_value)
    elif op == UnaryOp.EXP.value:
        call_vv[dt, "vvexpf", "vvexp"](out_ptr, src_ptr, src.size_value)
    elif op == UnaryOp.LOG.value:
        # libvMath has vvlogf for f32 but no f64 counterpart; fall through.
        comptime if dt == DType.float32:
            call_vv_f32["vvlogf"](
                rebind[UnsafePointer[Float32, MutExternalOrigin]](out_ptr),
                rebind[UnsafePointer[Float32, MutExternalOrigin]](src_ptr),
                src.size_value,
            )
        else:
            return False
    else:
        return False
    result.backend_code = BackendKind.ACCELERATE.value
    return True


def maybe_binary_accelerate[dt: DType](lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    var lhs_ptr = contiguous_ptr[dt](lhs)
    var rhs_ptr = contiguous_ptr[dt](rhs)
    var out_ptr = contiguous_ptr[dt](result)
    if op == BinaryOp.ADD.value:
        call_vdsp_binary[dt, "vDSP_vadd", "vDSP_vaddD"](lhs_ptr, rhs_ptr, out_ptr, result.size_value)
    elif op == BinaryOp.SUB.value:
        call_vdsp_binary[dt, "vDSP_vsub", "vDSP_vsubD"](rhs_ptr, lhs_ptr, out_ptr, result.size_value)
    elif op == BinaryOp.MUL.value:
        call_vdsp_binary[dt, "vDSP_vmul", "vDSP_vmulD"](lhs_ptr, rhs_ptr, out_ptr, result.size_value)
    elif op == BinaryOp.DIV.value:
        call_vdsp_binary[dt, "vDSP_vdiv", "vDSP_vdivD"](rhs_ptr, lhs_ptr, out_ptr, result.size_value)
    else:
        return False
    result.backend_code = BackendKind.ACCELERATE.value
    return True


def _try_rank1_strided[dt: DType](lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    # Returns True iff lhs/rhs/result all have dtype `dt` and the op was
    # dispatched to the vDSP fast path.
    var matching_code: Int
    comptime if dt == DType.float32:
        matching_code = ArrayDType.FLOAT32.value
    else:
        matching_code = ArrayDType.FLOAT64.value
    if not (
        lhs.dtype_code == matching_code and rhs.dtype_code == matching_code and result.dtype_code == matching_code
    ):
        return False
    var lhs_ptr = contiguous_ptr[dt](lhs)
    var rhs_ptr = contiguous_ptr[dt](rhs)
    var out_ptr = contiguous_ptr[dt](result)
    if op == BinaryOp.ADD.value:
        call_vdsp_binary_strided[dt, "vDSP_vadd", "vDSP_vaddD"](
            lhs_ptr,
            lhs.strides[0],
            rhs_ptr,
            rhs.strides[0],
            out_ptr,
            1,
            result.size_value,
        )
    elif op == BinaryOp.SUB.value:
        # vDSP_vsub computes B - A; numpy is A - B, so swap operands.
        call_vdsp_binary_strided[dt, "vDSP_vsub", "vDSP_vsubD"](
            rhs_ptr,
            rhs.strides[0],
            lhs_ptr,
            lhs.strides[0],
            out_ptr,
            1,
            result.size_value,
        )
    elif op == BinaryOp.MUL.value:
        call_vdsp_binary_strided[dt, "vDSP_vmul", "vDSP_vmulD"](
            lhs_ptr,
            lhs.strides[0],
            rhs_ptr,
            rhs.strides[0],
            out_ptr,
            1,
            result.size_value,
        )
    elif op == BinaryOp.DIV.value:
        # vDSP_vdiv computes B / A; numpy is A / B, so swap operands.
        call_vdsp_binary_strided[dt, "vDSP_vdiv", "vDSP_vdivD"](
            rhs_ptr,
            rhs.strides[0],
            lhs_ptr,
            lhs.strides[0],
            out_ptr,
            1,
            result.size_value,
        )
    else:
        return False
    result.backend_code = BackendKind.ACCELERATE.value
    return True


def maybe_binary_rank1_strided_accelerate(lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    if (
        len(lhs.shape) != 1
        or not same_shape(lhs.shape, rhs.shape)
        or not same_shape(lhs.shape, result.shape)
        or not is_c_contiguous(result)
    ):
        return False
    comptime if not CompilationTarget.is_macos():
        return False
    if _try_rank1_strided[DType.float32](lhs, rhs, result, op):
        return True
    if _try_rank1_strided[DType.float64](lhs, rhs, result, op):
        return True
    return False


def _try_complex_rank1_strided[real_dt: DType](lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    # Treats interleaved (re, im) pairs as a stride-2 real array, then runs
    # the corresponding real vDSP op twice — once on the re channel, once
    # on the im channel — sharing a stride that's 2× the logical stride.
    var complex_code: Int
    comptime if real_dt == DType.float32:
        complex_code = ArrayDType.COMPLEX64.value
    else:
        complex_code = ArrayDType.COMPLEX128.value
    if not (lhs.dtype_code == complex_code and rhs.dtype_code == complex_code and result.dtype_code == complex_code):
        return False
    var lhs_ptr = lhs.data.bitcast[Scalar[real_dt]]() + lhs.offset_elems * 2
    var rhs_ptr = rhs.data.bitcast[Scalar[real_dt]]() + rhs.offset_elems * 2
    var out_ptr = result.data.bitcast[Scalar[real_dt]]() + result.offset_elems * 2
    var lhs_stride = lhs.strides[0] * 2
    var rhs_stride = rhs.strides[0] * 2
    if op == BinaryOp.ADD.value:
        call_vdsp_binary_strided[real_dt, "vDSP_vadd", "vDSP_vaddD"](
            lhs_ptr, lhs_stride, rhs_ptr, rhs_stride, out_ptr, 2, result.size_value
        )
        call_vdsp_binary_strided[real_dt, "vDSP_vadd", "vDSP_vaddD"](
            lhs_ptr + 1, lhs_stride, rhs_ptr + 1, rhs_stride, out_ptr + 1, 2, result.size_value
        )
    else:
        call_vdsp_binary_strided[real_dt, "vDSP_vsub", "vDSP_vsubD"](
            rhs_ptr, rhs_stride, lhs_ptr, lhs_stride, out_ptr, 2, result.size_value
        )
        call_vdsp_binary_strided[real_dt, "vDSP_vsub", "vDSP_vsubD"](
            rhs_ptr + 1, rhs_stride, lhs_ptr + 1, lhs_stride, out_ptr + 1, 2, result.size_value
        )
    result.backend_code = BackendKind.ACCELERATE.value
    return True


def maybe_complex_binary_rank1_strided_accelerate(lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    if (
        len(lhs.shape) != 1
        or not same_shape(lhs.shape, rhs.shape)
        or not same_shape(lhs.shape, result.shape)
        or not is_c_contiguous(result)
        or (op != BinaryOp.ADD.value and op != BinaryOp.SUB.value)
    ):
        return False
    comptime if not CompilationTarget.is_macos():
        return False
    if _try_complex_rank1_strided[DType.float32](lhs, rhs, result, op):
        return True
    if _try_complex_rank1_strided[DType.float64](lhs, rhs, result, op):
        return True
    return False
