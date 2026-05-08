"""Matmul kernels: BLAS GEMM/GEMV fast paths + small-N typed-SIMD path.

`maybe_matmul_contiguous` is the entry point — picks the best path:
  1. `maybe_matmul_vector_accelerate` for rank-2 × rank-1 (GEMV)
  2. `maybe_matmul_complex_accelerate` for complex64/128 (cgemm/zgemm)
  3. `matmul_small_typed` for f32 N≤16 (skip the BLAS frame overhead)
  4. `cblas_dgemm_row_major_ld` / `cblas_sgemm_row_major_ld` for f32/f64 GEMM
  5. naive triple loop f64 round-trip for everything else

The small-N typed path beats Accelerate at low matrix sizes because BLAS
dispatch overhead dominates the actual FMA work — at N=8, the BLAS frame
is ~70% of the total time on M3 Pro.
"""

from std.sys import CompilationTarget, simd_width_of

from accelerate import (
    cblas_cgemm_row_major,
    cblas_dgemm_row_major_ld,
    cblas_dgemv_row_major_ld,
    cblas_sgemm_row_major_ld,
    cblas_sgemv_row_major_ld,
    cblas_zgemm_row_major,
)
from array import (
    Array,
    contiguous_as_f64,
    contiguous_ptr,
    is_c_contiguous,
    set_contiguous_from_f64,
)
from domain import ArrayDType, BackendKind

from .predicates import is_contiguous_float_array, rank2_blas_layout


def maybe_matmul_contiguous(
    lhs: Array,
    rhs: Array,
    mut result: Array,
    m: Int,
    n: Int,
    k_lhs: Int,
) raises -> Bool:
    comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
        if maybe_matmul_vector_accelerate(lhs, rhs, result, m, n, k_lhs):
            return True
        if maybe_matmul_complex_accelerate(lhs, rhs, result, m, n, k_lhs):
            return True
    if len(lhs.shape) != 2 or len(rhs.shape) != 2 or not is_contiguous_float_array(result):
        return False
    var lhs_layout = rank2_blas_layout(lhs)
    var rhs_layout = rank2_blas_layout(rhs)
    if (
        lhs.dtype_code == ArrayDType.FLOAT32.value
        and rhs.dtype_code == ArrayDType.FLOAT32.value
        and result.dtype_code == ArrayDType.FLOAT32.value
    ):
        if is_c_contiguous(lhs) and is_c_contiguous(rhs) and maybe_matmul_f32_small(lhs, rhs, result, m, n, k_lhs):
            return True
        comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
            if lhs_layout.can_use and rhs_layout.can_use:
                cblas_sgemm_row_major_ld(
                    m,
                    n,
                    k_lhs,
                    contiguous_ptr[DType.float32](result),
                    contiguous_ptr[DType.float32](lhs),
                    contiguous_ptr[DType.float32](rhs),
                    lhs_layout.transpose,
                    rhs_layout.transpose,
                    lhs_layout.leading_dim,
                    rhs_layout.leading_dim,
                    result.strides[0],
                )
                result.backend_code = BackendKind.ACCELERATE.value
                return True
    if (
        lhs.dtype_code == ArrayDType.FLOAT64.value
        and rhs.dtype_code == ArrayDType.FLOAT64.value
        and result.dtype_code == ArrayDType.FLOAT64.value
    ):
        comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
            if lhs_layout.can_use and rhs_layout.can_use:
                cblas_dgemm_row_major_ld(
                    m,
                    n,
                    k_lhs,
                    contiguous_ptr[DType.float64](result),
                    contiguous_ptr[DType.float64](lhs),
                    contiguous_ptr[DType.float64](rhs),
                    lhs_layout.transpose,
                    rhs_layout.transpose,
                    lhs_layout.leading_dim,
                    rhs_layout.leading_dim,
                    result.strides[0],
                )
                result.backend_code = BackendKind.ACCELERATE.value
                return True
    if not is_contiguous_float_array(lhs) or not is_contiguous_float_array(rhs):
        return False
    for i in range(m):
        for j in range(n):
            var total = 0.0
            for k in range(k_lhs):
                total += contiguous_as_f64(lhs, i * k_lhs + k) * contiguous_as_f64(rhs, k * n + j)
            set_contiguous_from_f64(result, i * n + j, total)
    return True


def maybe_matmul_vector_accelerate(
    lhs: Array,
    rhs: Array,
    mut result: Array,
    m: Int,
    n: Int,
    k_lhs: Int,
) raises -> Bool:
    var lhs_ndim = len(lhs.shape)
    var rhs_ndim = len(rhs.shape)
    if lhs_ndim == 2 and rhs_ndim == 1 and is_contiguous_float_array(rhs) and is_contiguous_float_array(result):
        var lhs_layout = rank2_blas_layout(lhs)
        if not lhs_layout.can_use:
            return False
        var rows = m
        var cols = k_lhs
        if lhs_layout.transpose:
            rows = k_lhs
            cols = m
        if (
            lhs.dtype_code == ArrayDType.FLOAT32.value
            and rhs.dtype_code == ArrayDType.FLOAT32.value
            and result.dtype_code == ArrayDType.FLOAT32.value
        ):
            cblas_sgemv_row_major_ld(
                rows,
                cols,
                contiguous_ptr[DType.float32](result),
                contiguous_ptr[DType.float32](lhs),
                contiguous_ptr[DType.float32](rhs),
                lhs_layout.transpose,
                lhs_layout.leading_dim,
            )
            result.backend_code = BackendKind.ACCELERATE.value
            return True
        if (
            lhs.dtype_code == ArrayDType.FLOAT64.value
            and rhs.dtype_code == ArrayDType.FLOAT64.value
            and result.dtype_code == ArrayDType.FLOAT64.value
        ):
            cblas_dgemv_row_major_ld(
                rows,
                cols,
                contiguous_ptr[DType.float64](result),
                contiguous_ptr[DType.float64](lhs),
                contiguous_ptr[DType.float64](rhs),
                lhs_layout.transpose,
                lhs_layout.leading_dim,
            )
            result.backend_code = BackendKind.ACCELERATE.value
            return True
    if lhs_ndim == 1 and rhs_ndim == 2 and is_contiguous_float_array(lhs) and is_contiguous_float_array(result):
        var rhs_layout = rank2_blas_layout(rhs)
        if not rhs_layout.can_use:
            return False
        var rows = k_lhs
        var cols = n
        var transpose_rhs = True
        if rhs_layout.transpose:
            rows = n
            cols = k_lhs
            transpose_rhs = False
        if (
            lhs.dtype_code == ArrayDType.FLOAT32.value
            and rhs.dtype_code == ArrayDType.FLOAT32.value
            and result.dtype_code == ArrayDType.FLOAT32.value
        ):
            cblas_sgemv_row_major_ld(
                rows,
                cols,
                contiguous_ptr[DType.float32](result),
                contiguous_ptr[DType.float32](rhs),
                contiguous_ptr[DType.float32](lhs),
                transpose_rhs,
                rhs_layout.leading_dim,
            )
            result.backend_code = BackendKind.ACCELERATE.value
            return True
        if (
            lhs.dtype_code == ArrayDType.FLOAT64.value
            and rhs.dtype_code == ArrayDType.FLOAT64.value
            and result.dtype_code == ArrayDType.FLOAT64.value
        ):
            cblas_dgemv_row_major_ld(
                rows,
                cols,
                contiguous_ptr[DType.float64](result),
                contiguous_ptr[DType.float64](rhs),
                contiguous_ptr[DType.float64](lhs),
                transpose_rhs,
                rhs_layout.leading_dim,
            )
            result.backend_code = BackendKind.ACCELERATE.value
            return True
    return False


def maybe_matmul_complex_accelerate(
    lhs: Array,
    rhs: Array,
    mut result: Array,
    m: Int,
    n: Int,
    k_lhs: Int,
) raises -> Bool:
    # Complex matmul via cgemm/zgemm. Requires both operands rank-2,
    # c-contiguous, matching complex dtype, and result also complex.
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        return False
    if not is_c_contiguous(lhs) or not is_c_contiguous(rhs) or not is_c_contiguous(result):
        return False
    if (
        lhs.dtype_code == ArrayDType.COMPLEX64.value
        and rhs.dtype_code == ArrayDType.COMPLEX64.value
        and result.dtype_code == ArrayDType.COMPLEX64.value
    ):
        cblas_cgemm_row_major(
            m,
            n,
            k_lhs,
            contiguous_ptr[DType.float32](result),
            contiguous_ptr[DType.float32](lhs),
            contiguous_ptr[DType.float32](rhs),
        )
        result.backend_code = BackendKind.ACCELERATE.value
        return True
    if (
        lhs.dtype_code == ArrayDType.COMPLEX128.value
        and rhs.dtype_code == ArrayDType.COMPLEX128.value
        and result.dtype_code == ArrayDType.COMPLEX128.value
    ):
        cblas_zgemm_row_major(
            m,
            n,
            k_lhs,
            contiguous_ptr[DType.float64](result),
            contiguous_ptr[DType.float64](lhs),
            contiguous_ptr[DType.float64](rhs),
        )
        result.backend_code = BackendKind.ACCELERATE.value
        return True
    return False


def matmul_small_typed[
    dtype: DType
](
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    m: Int,
    n: Int,
    k_lhs: Int,
) raises where dtype.is_floating_point():
    # Comptime-typed small-N matmul. Splat each lhs scalar to a SIMD vector
    # then fma against a contiguous rhs row chunk; this beats cblas_sgemm /
    # cblas_dgemm dispatch overhead at N≤16 by skipping the BLAS frame.
    comptime width = simd_width_of[dtype]()
    for i in range(m):
        var j = 0
        while j + width <= n:
            var acc = SIMD[dtype, width](0)
            for k in range(k_lhs):
                acc += SIMD[dtype, width](lhs_ptr[i * k_lhs + k]) * rhs_ptr.load[width=width](k * n + j)
            out_ptr.store(i * n + j, acc)
            j += width
        while j < n:
            var total = Scalar[dtype](0)
            for k in range(k_lhs):
                total += lhs_ptr[i * k_lhs + k] * rhs_ptr[k * n + j]
            out_ptr[i * n + j] = total
            j += 1


def maybe_matmul_f32_small(
    lhs: Array,
    rhs: Array,
    mut result: Array,
    m: Int,
    n: Int,
    k_lhs: Int,
) raises -> Bool:
    # Thin dispatcher that delegates to the typed kernel. Caller has already
    # verified dtype f32; the typed instantiation is fully specialized at
    # compile time. Kept under the f32 name so existing callers don't need
    # to change; an `_f64` sibling can be added when matmul small-N for f64
    # becomes a hot path.
    if m > 16 or n > 16 or k_lhs > 16:
        return False
    matmul_small_typed[DType.float32](
        contiguous_ptr[DType.float32](lhs),
        contiguous_ptr[DType.float32](rhs),
        contiguous_ptr[DType.float32](result),
        m,
        n,
        k_lhs,
    )
    return True
