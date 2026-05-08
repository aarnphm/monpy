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
    call_vdsp_binary_f32,
    call_vdsp_binary_f64,
    call_vdsp_binary_strided_f32,
    call_vdsp_binary_strided_f64,
    call_vv_f32,
    call_vv_f64,
)
from array import (
    Array,
    contiguous_ptr,
    is_c_contiguous,
    same_shape,
)
from domain import ArrayDType, BackendKind, BinaryOp, UnaryOp


def maybe_unary_accelerate_f32(src: Array, mut result: Array, op: Int) raises -> Bool:
    var src_ptr = contiguous_ptr[DType.float32](src)
    var out_ptr = contiguous_ptr[DType.float32](result)
    if op == UnaryOp.SIN.value:
        call_vv_f32["vvsinf"](out_ptr, src_ptr, src.size_value)
    elif op == UnaryOp.COS.value:
        call_vv_f32["vvcosf"](out_ptr, src_ptr, src.size_value)
    elif op == UnaryOp.EXP.value:
        call_vv_f32["vvexpf"](out_ptr, src_ptr, src.size_value)
    elif op == UnaryOp.LOG.value:
        call_vv_f32["vvlogf"](out_ptr, src_ptr, src.size_value)
    else:
        return False
    result.backend_code = BackendKind.ACCELERATE.value
    return True


def maybe_unary_accelerate_f64(src: Array, mut result: Array, op: Int) raises -> Bool:
    var src_ptr = contiguous_ptr[DType.float64](src)
    var out_ptr = contiguous_ptr[DType.float64](result)
    if op == UnaryOp.SIN.value:
        call_vv_f64["vvsin"](out_ptr, src_ptr, src.size_value)
    elif op == UnaryOp.COS.value:
        call_vv_f64["vvcos"](out_ptr, src_ptr, src.size_value)
    elif op == UnaryOp.EXP.value:
        call_vv_f64["vvexp"](out_ptr, src_ptr, src.size_value)
    else:
        return False
    result.backend_code = BackendKind.ACCELERATE.value
    return True


def maybe_binary_accelerate_f32(lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    var lhs_ptr = contiguous_ptr[DType.float32](lhs)
    var rhs_ptr = contiguous_ptr[DType.float32](rhs)
    var out_ptr = contiguous_ptr[DType.float32](result)
    if op == BinaryOp.ADD.value:
        call_vdsp_binary_f32["vDSP_vadd"](lhs_ptr, rhs_ptr, out_ptr, result.size_value)
    elif op == BinaryOp.SUB.value:
        call_vdsp_binary_f32["vDSP_vsub"](rhs_ptr, lhs_ptr, out_ptr, result.size_value)
    elif op == BinaryOp.MUL.value:
        call_vdsp_binary_f32["vDSP_vmul"](lhs_ptr, rhs_ptr, out_ptr, result.size_value)
    elif op == BinaryOp.DIV.value:
        call_vdsp_binary_f32["vDSP_vdiv"](rhs_ptr, lhs_ptr, out_ptr, result.size_value)
    else:
        return False
    result.backend_code = BackendKind.ACCELERATE.value
    return True


def maybe_binary_accelerate_f64(lhs: Array, rhs: Array, mut result: Array, op: Int) raises -> Bool:
    var lhs_ptr = contiguous_ptr[DType.float64](lhs)
    var rhs_ptr = contiguous_ptr[DType.float64](rhs)
    var out_ptr = contiguous_ptr[DType.float64](result)
    if op == BinaryOp.ADD.value:
        call_vdsp_binary_f64["vDSP_vaddD"](lhs_ptr, rhs_ptr, out_ptr, result.size_value)
    elif op == BinaryOp.SUB.value:
        call_vdsp_binary_f64["vDSP_vsubD"](rhs_ptr, lhs_ptr, out_ptr, result.size_value)
    elif op == BinaryOp.MUL.value:
        call_vdsp_binary_f64["vDSP_vmulD"](lhs_ptr, rhs_ptr, out_ptr, result.size_value)
    elif op == BinaryOp.DIV.value:
        call_vdsp_binary_f64["vDSP_vdivD"](rhs_ptr, lhs_ptr, out_ptr, result.size_value)
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
    if (
        lhs.dtype_code == ArrayDType.FLOAT32.value
        and rhs.dtype_code == ArrayDType.FLOAT32.value
        and result.dtype_code == ArrayDType.FLOAT32.value
    ):
        var lhs_ptr = contiguous_ptr[DType.float32](lhs)
        var rhs_ptr = contiguous_ptr[DType.float32](rhs)
        var out_ptr = contiguous_ptr[DType.float32](result)
        if op == BinaryOp.ADD.value:
            call_vdsp_binary_strided_f32["vDSP_vadd"](
                lhs_ptr,
                lhs.strides[0],
                rhs_ptr,
                rhs.strides[0],
                out_ptr,
                1,
                result.size_value,
            )
        elif op == BinaryOp.SUB.value:
            call_vdsp_binary_strided_f32["vDSP_vsub"](
                rhs_ptr,
                rhs.strides[0],
                lhs_ptr,
                lhs.strides[0],
                out_ptr,
                1,
                result.size_value,
            )
        elif op == BinaryOp.MUL.value:
            call_vdsp_binary_strided_f32["vDSP_vmul"](
                lhs_ptr,
                lhs.strides[0],
                rhs_ptr,
                rhs.strides[0],
                out_ptr,
                1,
                result.size_value,
            )
        elif op == BinaryOp.DIV.value:
            call_vdsp_binary_strided_f32["vDSP_vdiv"](
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
    if (
        lhs.dtype_code == ArrayDType.FLOAT64.value
        and rhs.dtype_code == ArrayDType.FLOAT64.value
        and result.dtype_code == ArrayDType.FLOAT64.value
    ):
        var lhs_ptr = contiguous_ptr[DType.float64](lhs)
        var rhs_ptr = contiguous_ptr[DType.float64](rhs)
        var out_ptr = contiguous_ptr[DType.float64](result)
        if op == BinaryOp.ADD.value:
            call_vdsp_binary_strided_f64["vDSP_vaddD"](
                lhs_ptr,
                lhs.strides[0],
                rhs_ptr,
                rhs.strides[0],
                out_ptr,
                1,
                result.size_value,
            )
        elif op == BinaryOp.SUB.value:
            call_vdsp_binary_strided_f64["vDSP_vsubD"](
                rhs_ptr,
                rhs.strides[0],
                lhs_ptr,
                lhs.strides[0],
                out_ptr,
                1,
                result.size_value,
            )
        elif op == BinaryOp.MUL.value:
            call_vdsp_binary_strided_f64["vDSP_vmulD"](
                lhs_ptr,
                lhs.strides[0],
                rhs_ptr,
                rhs.strides[0],
                out_ptr,
                1,
                result.size_value,
            )
        elif op == BinaryOp.DIV.value:
            call_vdsp_binary_strided_f64["vDSP_vdivD"](
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
    return False


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
    if (
        lhs.dtype_code == ArrayDType.COMPLEX64.value
        and rhs.dtype_code == ArrayDType.COMPLEX64.value
        and result.dtype_code == ArrayDType.COMPLEX64.value
    ):
        var lhs_ptr = lhs.data.bitcast[Float32]() + lhs.offset_elems * 2
        var rhs_ptr = rhs.data.bitcast[Float32]() + rhs.offset_elems * 2
        var out_ptr = result.data.bitcast[Float32]() + result.offset_elems * 2
        var lhs_stride = lhs.strides[0] * 2
        var rhs_stride = rhs.strides[0] * 2
        if op == BinaryOp.ADD.value:
            call_vdsp_binary_strided_f32["vDSP_vadd"](
                lhs_ptr, lhs_stride, rhs_ptr, rhs_stride, out_ptr, 2, result.size_value
            )
            call_vdsp_binary_strided_f32["vDSP_vadd"](
                lhs_ptr + 1, lhs_stride, rhs_ptr + 1, rhs_stride, out_ptr + 1, 2, result.size_value
            )
        else:
            call_vdsp_binary_strided_f32["vDSP_vsub"](
                rhs_ptr, rhs_stride, lhs_ptr, lhs_stride, out_ptr, 2, result.size_value
            )
            call_vdsp_binary_strided_f32["vDSP_vsub"](
                rhs_ptr + 1, rhs_stride, lhs_ptr + 1, lhs_stride, out_ptr + 1, 2, result.size_value
            )
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
        var lhs_stride = lhs.strides[0] * 2
        var rhs_stride = rhs.strides[0] * 2
        if op == BinaryOp.ADD.value:
            call_vdsp_binary_strided_f64["vDSP_vaddD"](
                lhs_ptr, lhs_stride, rhs_ptr, rhs_stride, out_ptr, 2, result.size_value
            )
            call_vdsp_binary_strided_f64["vDSP_vaddD"](
                lhs_ptr + 1, lhs_stride, rhs_ptr + 1, rhs_stride, out_ptr + 1, 2, result.size_value
            )
        else:
            call_vdsp_binary_strided_f64["vDSP_vsubD"](
                rhs_ptr, rhs_stride, lhs_ptr, lhs_stride, out_ptr, 2, result.size_value
            )
            call_vdsp_binary_strided_f64["vDSP_vsubD"](
                rhs_ptr + 1, rhs_stride, lhs_ptr + 1, lhs_stride, out_ptr + 1, 2, result.size_value
            )
        result.backend_code = BackendKind.ACCELERATE.value
        return True
    return False
