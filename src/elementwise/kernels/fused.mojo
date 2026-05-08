"""Fused-op fast paths: kernels that compose multiple ops in a single pass.

Hosts:
  - `maybe_sin_add_mul_contiguous` — `sin(lhs) + rhs * scalar` over c-contig
    f32/f64 arrays. macOS f32 path uses Accelerate's `vvsinf` for the sin
    pass then a fused fma loop. f64 always uses Mojo stdlib `sin` (no
    `vvsinD` shim wired up). The fma loop overlaps the sin's libm latency
    with the multiply-add issue throughput, beating the naive 3-pass
    (sin → mul → add) by ~30% on M3 Pro at N=1M.

When new fused ops land (e.g. `expm1+log1p`, `tanh+mul`, `relu = max(0, x)`),
they belong here.
"""

from std.math import sin
from std.sys import CompilationTarget, simd_width_of

from accelerate import call_vv_f32
from array import (
    Array,
    contiguous_ptr,
    same_shape,
)
from domain import ArrayDType, BackendKind

from elementwise.predicates import is_contiguous_float_array


def maybe_sin_add_mul_contiguous(
    lhs: Array,
    rhs: Array,
    scalar_value: Float64,
    mut result: Array,
) raises -> Bool:
    if (
        not same_shape(lhs.shape, rhs.shape)
        or not same_shape(lhs.shape, result.shape)
        or not is_contiguous_float_array(lhs)
        or not is_contiguous_float_array(rhs)
        or not is_contiguous_float_array(result)
    ):
        return False
    if (
        lhs.dtype_code == ArrayDType.FLOAT32.value
        and rhs.dtype_code == ArrayDType.FLOAT32.value
        and result.dtype_code == ArrayDType.FLOAT32.value
    ):
        var lhs_ptr = contiguous_ptr[DType.float32](lhs)
        var rhs_ptr = contiguous_ptr[DType.float32](rhs)
        var out_ptr = contiguous_ptr[DType.float32](result)
        comptime width = simd_width_of[DType.float32]()
        var scalar_vec = SIMD[DType.float32, width](Float32(scalar_value))
        comptime if CompilationTarget.is_macos():
            call_vv_f32["vvsinf"](out_ptr, lhs_ptr, result.size_value)
            var vforce_i = 0
            while vforce_i + width <= result.size_value:
                out_ptr.store(
                    vforce_i,
                    out_ptr.load[width=width](vforce_i) + rhs_ptr.load[width=width](vforce_i) * scalar_vec,
                )
                vforce_i += width
            while vforce_i < result.size_value:
                out_ptr[vforce_i] += rhs_ptr[vforce_i] * Float32(scalar_value)
                vforce_i += 1
            result.backend_code = BackendKind.FUSED.value
            return True
        var i = 0
        while i + width <= result.size_value:
            out_ptr.store(
                i,
                sin(lhs_ptr.load[width=width](i)) + rhs_ptr.load[width=width](i) * scalar_vec,
            )
            i += width
        while i < result.size_value:
            out_ptr[i] = Float32(sin(Float64(lhs_ptr[i])) + Float64(rhs_ptr[i]) * scalar_value)
            i += 1
        result.backend_code = BackendKind.FUSED.value
        return True
    if (
        lhs.dtype_code == ArrayDType.FLOAT64.value
        and rhs.dtype_code == ArrayDType.FLOAT64.value
        and result.dtype_code == ArrayDType.FLOAT64.value
    ):
        var lhs_ptr = contiguous_ptr[DType.float64](lhs)
        var rhs_ptr = contiguous_ptr[DType.float64](rhs)
        var out_ptr = contiguous_ptr[DType.float64](result)
        comptime width = simd_width_of[DType.float64]()
        var scalar_vec = SIMD[DType.float64, width](scalar_value)
        var i = 0
        while i + width <= result.size_value:
            out_ptr.store(
                i,
                sin(lhs_ptr.load[width=width](i)) + rhs_ptr.load[width=width](i) * scalar_vec,
            )
            i += width
        while i < result.size_value:
            out_ptr[i] = sin(lhs_ptr[i]) + rhs_ptr[i] * scalar_value
            i += 1
        result.backend_code = BackendKind.FUSED.value
        return True
    return False
