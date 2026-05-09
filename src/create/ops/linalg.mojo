"""Linalg PythonObject bridge ops.

Hosts matmul + solve/inv/det (LU-backed) + LAPACK-backed
qr/cholesky/eigh/eig/svd/lstsq + pinv (lstsq-driven). All twelve `_ops`
functions allocate result arrays and forward to typed kernels in
`elementwise/`. The two-letter LAPACK precision split (`*_f32_into` /
`*_f64_into`) is dispatched here based on `result_dtype_for_linalg`.

Why grouped: every op shares the same shape — unbox arrays, allocate
result(s), call into elementwise, return a Python list-or-Array.
"""

from std.math import log as _log, sqrt as _sqrt
from std.python import Python, PythonObject
from std.sys import simd_width_of

from array import (
    Array,
    contiguous_ptr,
    get_logical_as_f64,
    is_c_contiguous,
    make_empty_array,
    result_dtype_for_binary,
    result_dtype_for_linalg,
    result_dtype_for_linalg_binary,
    set_logical_from_f64,
)
from domain import ArrayDType, BinaryOp
from elementwise import (
    lapack_cholesky_into,
    lapack_eig_real_into,
    lapack_eigh_into,
    lapack_lstsq_into,
    lapack_qr_r_only_into,
    lapack_qr_reduced_into,
    lapack_svd_into,
    lu_det_into,
    lu_inverse_into,
    lu_solve_into,
    maybe_matmul_contiguous,
)

from create._complex_helpers import _complex_imag, _complex_real, _complex_store


def outer_contiguous_typed[
    dtype: DType
](
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    lhs_size: Int,
    rhs_size: Int,
) raises where dtype.is_floating_point():
    comptime width = simd_width_of[dtype]()
    for i in range(lhs_size):
        var base = i * rhs_size
        var lhs_vec = SIMD[dtype, width](lhs_ptr[i])
        var j = 0
        while j + width <= rhs_size:
            out_ptr.store(base + j, lhs_vec * rhs_ptr.load[width=width](j))
            j += width
        while j < rhs_size:
            out_ptr[base + j] = lhs_ptr[i] * rhs_ptr[j]
            j += 1


def maybe_outer_contiguous(lhs: Array, rhs: Array, mut result: Array) raises -> Bool:
    if not is_c_contiguous(lhs) or not is_c_contiguous(rhs) or not is_c_contiguous(result):
        return False
    if (
        lhs.dtype_code == ArrayDType.FLOAT32.value
        and rhs.dtype_code == ArrayDType.FLOAT32.value
        and result.dtype_code == ArrayDType.FLOAT32.value
    ):
        outer_contiguous_typed[DType.float32](
            contiguous_ptr[DType.float32](lhs),
            contiguous_ptr[DType.float32](rhs),
            contiguous_ptr[DType.float32](result),
            lhs.size_value,
            rhs.size_value,
        )
        return True
    if (
        lhs.dtype_code == ArrayDType.FLOAT64.value
        and rhs.dtype_code == ArrayDType.FLOAT64.value
        and result.dtype_code == ArrayDType.FLOAT64.value
    ):
        outer_contiguous_typed[DType.float64](
            contiguous_ptr[DType.float64](lhs),
            contiguous_ptr[DType.float64](rhs),
            contiguous_ptr[DType.float64](result),
            lhs.size_value,
            rhs.size_value,
        )
        return True
    return False


def outer_ops(lhs_obj: PythonObject, rhs_obj: PythonObject) raises -> PythonObject:
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
    var out_shape = List[Int]()
    out_shape.append(lhs[].size_value)
    out_shape.append(rhs[].size_value)
    var result = make_empty_array(
        result_dtype_for_binary(lhs[].dtype_code, rhs[].dtype_code, BinaryOp.MUL.value),
        out_shape^,
    )
    if maybe_outer_contiguous(lhs[], rhs[], result):
        return PythonObject(alloc=result^)
    var result_is_complex = (
        result.dtype_code == ArrayDType.COMPLEX64.value or result.dtype_code == ArrayDType.COMPLEX128.value
    )
    for i in range(lhs[].size_value):
        for j in range(rhs[].size_value):
            var out_index = i * rhs[].size_value + j
            if result_is_complex:
                var lhs_re = _complex_real(lhs[], i)
                var lhs_im = _complex_imag(lhs[], i)
                var rhs_re = _complex_real(rhs[], j)
                var rhs_im = _complex_imag(rhs[], j)
                _complex_store(
                    result,
                    out_index,
                    lhs_re * rhs_re - lhs_im * rhs_im,
                    lhs_re * rhs_im + lhs_im * rhs_re,
                )
            else:
                set_logical_from_f64(
                    result,
                    out_index,
                    get_logical_as_f64(lhs[], i) * get_logical_as_f64(rhs[], j),
                )
    return PythonObject(alloc=result^)


def norm2_sumsq_contiguous_typed[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], size: Int
) raises -> Float64 where dtype.is_floating_point():
    comptime width = simd_width_of[dtype]()
    var a0 = SIMD[dtype, width](0)
    var a1 = SIMD[dtype, width](0)
    var a2 = SIMD[dtype, width](0)
    var a3 = SIMD[dtype, width](0)
    var i = 0
    comptime block = width * 4
    while i + block <= size:
        var v0 = src_ptr.load[width=width](i)
        var v1 = src_ptr.load[width=width](i + width)
        var v2 = src_ptr.load[width=width](i + 2 * width)
        var v3 = src_ptr.load[width=width](i + 3 * width)
        a0 += v0 * v0
        a1 += v1 * v1
        a2 += v2 * v2
        a3 += v3 * v3
        i += block
    var acc_vec = (a0 + a1) + (a2 + a3)
    while i + width <= size:
        var v = src_ptr.load[width=width](i)
        acc_vec += v * v
        i += width
    var acc = Float64(acc_vec.reduce_add()[0])
    while i < size:
        var v = Float64(src_ptr[i])
        acc += v * v
        i += 1
    return acc


def norm2_all_ops(array_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var shape = List[Int]()
    var result = make_empty_array(src[].dtype_code, shape^)
    if is_c_contiguous(src[]):
        if src[].dtype_code == ArrayDType.FLOAT32.value:
            set_logical_from_f64(
                result,
                0,
                _sqrt(
                    norm2_sumsq_contiguous_typed[DType.float32](contiguous_ptr[DType.float32](src[]), src[].size_value)
                ),
            )
            return PythonObject(alloc=result^)
        if src[].dtype_code == ArrayDType.FLOAT64.value:
            set_logical_from_f64(
                result,
                0,
                _sqrt(
                    norm2_sumsq_contiguous_typed[DType.float64](contiguous_ptr[DType.float64](src[]), src[].size_value)
                ),
            )
            return PythonObject(alloc=result^)
    var acc = 0.0
    for i in range(src[].size_value):
        var v = get_logical_as_f64(src[], i)
        acc += v * v
    set_logical_from_f64(result, 0, _sqrt(acc))
    return PythonObject(alloc=result^)


def norm2_last_axis_contiguous_typed[
    dtype: DType
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rows: Int,
    cols: Int,
) raises where dtype.is_floating_point():
    for row in range(rows):
        out_ptr[row] = Scalar[dtype](_sqrt(norm2_sumsq_contiguous_typed[dtype](src_ptr + row * cols, cols)))


def norm2_last_axis_ops(array_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 2:
        raise Error("linalg_norm2_last_axis requires rank-2 input")
    var rows = src[].shape[0]
    var cols = src[].shape[1]
    var shape = List[Int]()
    shape.append(rows)
    var result = make_empty_array(src[].dtype_code, shape^)
    if is_c_contiguous(src[]):
        if src[].dtype_code == ArrayDType.FLOAT32.value:
            norm2_last_axis_contiguous_typed[DType.float32](
                contiguous_ptr[DType.float32](src[]),
                contiguous_ptr[DType.float32](result),
                rows,
                cols,
            )
            return PythonObject(alloc=result^)
        if src[].dtype_code == ArrayDType.FLOAT64.value:
            norm2_last_axis_contiguous_typed[DType.float64](
                contiguous_ptr[DType.float64](src[]),
                contiguous_ptr[DType.float64](result),
                rows,
                cols,
            )
            return PythonObject(alloc=result^)
    for row in range(rows):
        var acc = 0.0
        var base = row * cols
        for col in range(cols):
            var v = get_logical_as_f64(src[], base + col)
            acc += v * v
        set_logical_from_f64(result, row, _sqrt(acc))
    return PythonObject(alloc=result^)


def vecdot_last_axis_contiguous_typed[
    dtype: DType
](
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rows: Int,
    cols: Int,
) raises where dtype.is_floating_point():
    comptime width = simd_width_of[dtype]()
    comptime block = width * 4
    for row in range(rows):
        var base = row * cols
        var a0 = SIMD[dtype, width](0)
        var a1 = SIMD[dtype, width](0)
        var a2 = SIMD[dtype, width](0)
        var a3 = SIMD[dtype, width](0)
        var col = 0
        while col + block <= cols:
            a0 += lhs_ptr.load[width=width](base + col) * rhs_ptr.load[width=width](base + col)
            a1 += lhs_ptr.load[width=width](base + col + width) * rhs_ptr.load[width=width](base + col + width)
            a2 += lhs_ptr.load[width=width](base + col + 2 * width) * rhs_ptr.load[width=width](base + col + 2 * width)
            a3 += lhs_ptr.load[width=width](base + col + 3 * width) * rhs_ptr.load[width=width](base + col + 3 * width)
            col += block
        var acc_vec = (a0 + a1) + (a2 + a3)
        while col + width <= cols:
            acc_vec += lhs_ptr.load[width=width](base + col) * rhs_ptr.load[width=width](base + col)
            col += width
        var acc = acc_vec.reduce_add()[0]
        while col < cols:
            acc += lhs_ptr[base + col] * rhs_ptr[base + col]
            col += 1
        out_ptr[row] = acc


def vecdot_last_axis_ops(lhs_obj: PythonObject, rhs_obj: PythonObject) raises -> PythonObject:
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
    if len(lhs[].shape) != 2 or len(rhs[].shape) != 2:
        raise Error("linalg_vecdot_last_axis requires rank-2 inputs")
    if lhs[].shape[0] != rhs[].shape[0] or lhs[].shape[1] != rhs[].shape[1]:
        raise Error("linalg_vecdot_last_axis requires matching shapes")
    var rows = lhs[].shape[0]
    var cols = lhs[].shape[1]
    var shape = List[Int]()
    shape.append(rows)
    var result = make_empty_array(
        result_dtype_for_binary(lhs[].dtype_code, rhs[].dtype_code, BinaryOp.MUL.value),
        shape^,
    )
    if is_c_contiguous(lhs[]) and is_c_contiguous(rhs[]) and is_c_contiguous(result):
        if (
            lhs[].dtype_code == ArrayDType.FLOAT32.value
            and rhs[].dtype_code == ArrayDType.FLOAT32.value
            and result.dtype_code == ArrayDType.FLOAT32.value
        ):
            vecdot_last_axis_contiguous_typed[DType.float32](
                contiguous_ptr[DType.float32](lhs[]),
                contiguous_ptr[DType.float32](rhs[]),
                contiguous_ptr[DType.float32](result),
                rows,
                cols,
            )
            return PythonObject(alloc=result^)
        if (
            lhs[].dtype_code == ArrayDType.FLOAT64.value
            and rhs[].dtype_code == ArrayDType.FLOAT64.value
            and result.dtype_code == ArrayDType.FLOAT64.value
        ):
            vecdot_last_axis_contiguous_typed[DType.float64](
                contiguous_ptr[DType.float64](lhs[]),
                contiguous_ptr[DType.float64](rhs[]),
                contiguous_ptr[DType.float64](result),
                rows,
                cols,
            )
            return PythonObject(alloc=result^)
    for row in range(rows):
        var acc = 0.0
        var base = row * cols
        for col in range(cols):
            acc += get_logical_as_f64(lhs[], base + col) * get_logical_as_f64(rhs[], base + col)
        set_logical_from_f64(result, row, acc)
    return PythonObject(alloc=result^)


def matmul_ops(lhs_obj: PythonObject, rhs_obj: PythonObject) raises -> PythonObject:
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
    var lhs_ndim = len(lhs[].shape)
    var rhs_ndim = len(rhs[].shape)
    if lhs_ndim < 1 or lhs_ndim > 2 or rhs_ndim < 1 or rhs_ndim > 2:
        raise Error("matmul() only supports 1d and 2d arrays")
    var m = 1
    var k_lhs = lhs[].shape[0]
    if lhs_ndim == 2:
        m = lhs[].shape[0]
        k_lhs = lhs[].shape[1]
    var k_rhs = rhs[].shape[0]
    var n = 1
    if rhs_ndim == 2:
        n = rhs[].shape[1]
    if k_lhs != k_rhs:
        raise Error("matmul() dimension mismatch")
    var out_shape = List[Int]()
    if lhs_ndim == 2 and rhs_ndim == 2:
        out_shape.append(m)
        out_shape.append(n)
    elif lhs_ndim == 2 and rhs_ndim == 1:
        out_shape.append(m)
    elif lhs_ndim == 1 and rhs_ndim == 2:
        out_shape.append(n)
    var dtype_code = result_dtype_for_binary(lhs[].dtype_code, rhs[].dtype_code, BinaryOp.MUL.value)
    var result = make_empty_array(dtype_code, out_shape^)
    var is_complex = dtype_code == ArrayDType.COMPLEX64.value or dtype_code == ArrayDType.COMPLEX128.value
    if lhs_ndim == 1 and rhs_ndim == 1:
        if is_complex:
            var lhs_re_total = 0.0
            var lhs_im_total = 0.0
            for k in range(k_lhs):
                var l_re = _complex_real(lhs[], k)
                var l_im = _complex_imag(lhs[], k)
                var r_re = _complex_real(rhs[], k)
                var r_im = _complex_imag(rhs[], k)
                lhs_re_total += l_re * r_re - l_im * r_im
                lhs_im_total += l_re * r_im + l_im * r_re
            _complex_store(result, 0, lhs_re_total, lhs_im_total)
        else:
            var total = 0.0
            for k in range(k_lhs):
                total += get_logical_as_f64(lhs[], k) * get_logical_as_f64(rhs[], k)
            set_logical_from_f64(result, 0, total)
        return PythonObject(alloc=result^)
    if maybe_matmul_contiguous(lhs[], rhs[], result, m, n, k_lhs):
        return PythonObject(alloc=result^)
    if is_complex:
        for i in range(m):
            for j in range(n):
                var re_total = 0.0
                var im_total = 0.0
                for k in range(k_lhs):
                    var lhs_index = k
                    if lhs_ndim == 2:
                        lhs_index = i * k_lhs + k
                    var rhs_index = k
                    if rhs_ndim == 2:
                        rhs_index = k * n + j
                    var l_re = _complex_real(lhs[], lhs_index)
                    var l_im = _complex_imag(lhs[], lhs_index)
                    var r_re = _complex_real(rhs[], rhs_index)
                    var r_im = _complex_imag(rhs[], rhs_index)
                    re_total += l_re * r_re - l_im * r_im
                    im_total += l_re * r_im + l_im * r_re
                var out_index = j
                if lhs_ndim == 2:
                    out_index = i * n + j
                _complex_store(result, out_index, re_total, im_total)
        return PythonObject(alloc=result^)
    for i in range(m):
        for j in range(n):
            var total = 0.0
            for k in range(k_lhs):
                var lhs_index = k
                if lhs_ndim == 2:
                    lhs_index = i * k_lhs + k
                var rhs_index = k
                if rhs_ndim == 2:
                    rhs_index = k * n + j
                total += get_logical_as_f64(lhs[], lhs_index) * get_logical_as_f64(rhs[], rhs_index)
            var out_index = j
            if lhs_ndim == 2:
                out_index = i * n + j
            set_logical_from_f64(result, out_index, total)
    return PythonObject(alloc=result^)


def solve_ops(a_obj: PythonObject, b_obj: PythonObject) raises -> PythonObject:
    var a = a_obj.downcast_value_ptr[Array]()
    var b = b_obj.downcast_value_ptr[Array]()
    if len(a[].shape) != 2 or a[].shape[0] != a[].shape[1]:
        raise Error("linalg.solve_ops() requires a square rank-2 coefficient matrix")
    var n = a[].shape[0]
    var out_shape = List[Int]()
    if len(b[].shape) == 1:
        if b[].shape[0] != n:
            raise Error("linalg.solve() right-hand side shape mismatch")
        out_shape.append(n)
    elif len(b[].shape) == 2:
        if b[].shape[0] != n:
            raise Error("linalg.solve() right-hand side shape mismatch")
        out_shape.append(n)
        out_shape.append(b[].shape[1])
    else:
        raise Error("linalg.solve() right-hand side must be rank 1 or rank 2")
    var result = make_empty_array(
        result_dtype_for_linalg_binary(a[].dtype_code, b[].dtype_code),
        out_shape^,
    )
    lu_solve_into(a[], b[], result)
    return PythonObject(alloc=result^)


def inv_ops(array_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 2 or src[].shape[0] != src[].shape[1]:
        raise Error("linalg.inv() requires a square rank-2 matrix")
    var shape = List[Int]()
    shape.append(src[].shape[0])
    shape.append(src[].shape[1])
    var result = make_empty_array(result_dtype_for_linalg(src[].dtype_code), shape^)
    lu_inverse_into(src[], result)
    return PythonObject(alloc=result^)


def det_ops(array_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var shape = List[Int]()
    var result = make_empty_array(result_dtype_for_linalg(src[].dtype_code), shape^)
    lu_det_into(src[], result)
    return PythonObject(alloc=result^)


def slogdet_ops(array_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    var shape = List[Int]()
    var det_result = make_empty_array(result_dtype_for_linalg(src[].dtype_code), shape^)
    lu_det_into(src[], det_result)
    var det_value = get_logical_as_f64(det_result, 0)
    if det_value == 0.0:
        return Python.list(PythonObject(0.0), PythonObject(_log(0.0)))
    var sign = 1.0
    if det_value < 0.0:
        sign = -1.0
    return Python.list(PythonObject(sign), PythonObject(_log(_abs_f64(det_value))))


def qr_ops(array_obj: PythonObject, mode_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 2:
        raise Error("linalg.qr: input must be rank-2")
    var m = src[].shape[0]
    var n = src[].shape[1]
    var k = m if m < n else n
    var dtype_code = result_dtype_for_linalg(src[].dtype_code)
    var mode = Int(py=mode_obj)  # 0=reduced, 1=complete, 2=r, 3=raw
    if mode == 0:
        # Q is (m, k), R is (k, n)
        var q_shape = List[Int]()
        q_shape.append(m)
        q_shape.append(k)
        var r_shape = List[Int]()
        r_shape.append(k)
        r_shape.append(n)
        var q = make_empty_array(dtype_code, q_shape^)
        var r = make_empty_array(dtype_code, r_shape^)
        if dtype_code == ArrayDType.FLOAT32.value:
            lapack_qr_reduced_into[DType.float32](src[], q, r)
        else:
            lapack_qr_reduced_into[DType.float64](src[], q, r)
        return Python.list(PythonObject(alloc=q^), PythonObject(alloc=r^))
    if mode == 1:
        # mode='complete': Q is (m, m), R is (m, n). Numpy uses sgeqrf
        # then sorgqr to build full Q (m × m) — pad with extra columns
        # via `sorgqr(m, m, k, ...)` once the first k columns are set.
        # For simplicity in v1: error if m > n (would need extra work).
        if m < n:
            raise Error("linalg.qr: mode='complete' for m < n requires extra work — use mode='reduced'")
        var q_shape = List[Int]()
        q_shape.append(m)
        q_shape.append(m)
        var r_shape = List[Int]()
        r_shape.append(m)
        r_shape.append(n)
        var q = make_empty_array(dtype_code, q_shape^)
        var r = make_empty_array(dtype_code, r_shape^)
        if dtype_code == ArrayDType.FLOAT32.value:
            lapack_qr_reduced_into[DType.float32](src[], q, r)
        else:
            lapack_qr_reduced_into[DType.float64](src[], q, r)
        return Python.list(PythonObject(alloc=q^), PythonObject(alloc=r^))
    if mode == 2:
        # mode='r': just R, shape (k, n)
        var r_shape = List[Int]()
        r_shape.append(k)
        r_shape.append(n)
        var r = make_empty_array(dtype_code, r_shape^)
        if dtype_code == ArrayDType.FLOAT32.value:
            lapack_qr_r_only_into[DType.float32](src[], r)
        else:
            lapack_qr_r_only_into[DType.float64](src[], r)
        return PythonObject(alloc=r^)
    raise Error("linalg.qr: unsupported mode")


def cholesky_ops(array_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 2 or src[].shape[0] != src[].shape[1]:
        raise Error("linalg.cholesky: input must be square rank-2")
    var n = src[].shape[0]
    var dtype_code = result_dtype_for_linalg(src[].dtype_code)
    var shape = List[Int]()
    shape.append(n)
    shape.append(n)
    var result = make_empty_array(dtype_code, shape^)
    if dtype_code == ArrayDType.FLOAT32.value:
        lapack_cholesky_into[DType.float32](src[], result)
    else:
        lapack_cholesky_into[DType.float64](src[], result)
    return PythonObject(alloc=result^)


def _abs_f64(value: Float64) -> Float64:
    if value < 0.0:
        return -value
    return value


def _write_eigh2_vector(mut out: Array, column: Int, a00: Float64, a10: Float64, a11: Float64, eig: Float64) raises:
    var x = a10
    var y = eig - a00
    var norm = _sqrt(x * x + y * y)
    if norm == 0.0:
        x = eig - a11
        y = a10
        norm = _sqrt(x * x + y * y)
    if norm == 0.0:
        # Degenerate fallback. The diagonal case handles the real branch; this
        # only protects exact repeated roots from returning NaNs.
        if column == 0:
            set_logical_from_f64(out, 0, 1.0)
            set_logical_from_f64(out, 2, 0.0)
        else:
            set_logical_from_f64(out, 1, 0.0)
            set_logical_from_f64(out, 3, 1.0)
        return
    var inv_norm = 1.0 / norm
    set_logical_from_f64(out, column, x * inv_norm)
    set_logical_from_f64(out, 2 + column, y * inv_norm)


def _eigh2_into(src: Array, mut w: Array, mut v: Array, compute_v: Bool) raises:
    var a00 = get_logical_as_f64(src, 0)
    # Native eigh receives the UPLO-adjusted matrix. Read the lower triangle:
    # [[a00, a10], [a10, a11]].
    var a10 = get_logical_as_f64(src, 2)
    var a11 = get_logical_as_f64(src, 3)
    var half_trace = 0.5 * (a00 + a11)
    var half_diff = 0.5 * (a00 - a11)
    var radius = _sqrt(half_diff * half_diff + a10 * a10)
    var l0 = half_trace - radius
    var l1 = half_trace + radius
    set_logical_from_f64(w, 0, l0)
    set_logical_from_f64(w, 1, l1)
    if not compute_v:
        return
    if _abs_f64(a10) == 0.0:
        if a00 <= a11:
            set_logical_from_f64(v, 0, 1.0)
            set_logical_from_f64(v, 1, 0.0)
            set_logical_from_f64(v, 2, 0.0)
            set_logical_from_f64(v, 3, 1.0)
        else:
            set_logical_from_f64(v, 0, 0.0)
            set_logical_from_f64(v, 1, 1.0)
            set_logical_from_f64(v, 2, 1.0)
            set_logical_from_f64(v, 3, 0.0)
        return
    _write_eigh2_vector(v, 0, a00, a10, a11, l0)
    _write_eigh2_vector(v, 1, a00, a10, a11, l1)


def eigh_ops(array_obj: PythonObject, compute_eigenvectors_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 2 or src[].shape[0] != src[].shape[1]:
        raise Error("linalg.eigh: input must be square rank-2")
    var n = src[].shape[0]
    var compute_v = Bool(py=compute_eigenvectors_obj)
    var dtype_code = result_dtype_for_linalg(src[].dtype_code)
    var w_shape = List[Int]()
    w_shape.append(n)
    var w = make_empty_array(dtype_code, w_shape^)
    var v_shape = List[Int]()
    if compute_v:
        v_shape.append(n)
        v_shape.append(n)
    else:
        # Allocate a 0-element placeholder — caller ignores it for eigvalsh.
        v_shape.append(0)
    var v = make_empty_array(dtype_code, v_shape^)
    if n == 2:
        _eigh2_into(src[], w, v, compute_v)
    elif dtype_code == ArrayDType.FLOAT32.value:
        lapack_eigh_into[DType.float32](src[], w, v, compute_v)
    else:
        lapack_eigh_into[DType.float64](src[], w, v, compute_v)
    return Python.list(PythonObject(alloc=w^), PythonObject(alloc=v^))


def eig_ops(array_obj: PythonObject, compute_eigenvectors_obj: PythonObject) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 2 or src[].shape[0] != src[].shape[1]:
        raise Error("linalg.eig: input must be square rank-2")
    var n = src[].shape[0]
    var compute_v = Bool(py=compute_eigenvectors_obj)
    var dtype_code = result_dtype_for_linalg(src[].dtype_code)
    var w_shape = List[Int]()
    w_shape.append(n)
    var wr = make_empty_array(dtype_code, w_shape^)
    var wi_shape = List[Int]()
    wi_shape.append(n)
    var wi = make_empty_array(dtype_code, wi_shape^)
    var v_shape = List[Int]()
    if compute_v:
        v_shape.append(n)
        v_shape.append(n)
    else:
        v_shape.append(0)
    var v = make_empty_array(dtype_code, v_shape^)
    var all_real: Bool
    if dtype_code == ArrayDType.FLOAT32.value:
        all_real = lapack_eig_real_into[DType.float32](src[], wr, wi, v, compute_v)
    else:
        all_real = lapack_eig_real_into[DType.float64](src[], wr, wi, v, compute_v)
    return Python.list(
        PythonObject(alloc=wr^), PythonObject(alloc=wi^), PythonObject(alloc=v^), PythonObject(all_real)
    )


def svd_ops(
    array_obj: PythonObject,
    full_matrices_obj: PythonObject,
    compute_uv_obj: PythonObject,
) raises -> PythonObject:
    var src = array_obj.downcast_value_ptr[Array]()
    if len(src[].shape) != 2:
        raise Error("linalg.svd: input must be rank-2")
    var m = src[].shape[0]
    var n = src[].shape[1]
    var k = m if m < n else n
    var full_matrices = Bool(py=full_matrices_obj)
    var compute_uv = Bool(py=compute_uv_obj)
    var dtype_code = result_dtype_for_linalg(src[].dtype_code)
    var s_shape = List[Int]()
    s_shape.append(k)
    var s = make_empty_array(dtype_code, s_shape^)
    var u_shape = List[Int]()
    var vt_shape = List[Int]()
    if compute_uv:
        u_shape.append(m)
        if full_matrices:
            u_shape.append(m)
            vt_shape.append(n)
        else:
            u_shape.append(k)
            vt_shape.append(k)
        vt_shape.append(n)
    else:
        u_shape.append(0)
        vt_shape.append(0)
    var u = make_empty_array(dtype_code, u_shape^)
    var vt = make_empty_array(dtype_code, vt_shape^)
    if dtype_code == ArrayDType.FLOAT32.value:
        lapack_svd_into[DType.float32](src[], u, s, vt, full_matrices, compute_uv)
    else:
        lapack_svd_into[DType.float64](src[], u, s, vt, full_matrices, compute_uv)
    return Python.list(PythonObject(alloc=u^), PythonObject(alloc=s^), PythonObject(alloc=vt^))


def lstsq_ops(a_obj: PythonObject, b_obj: PythonObject, rcond_obj: PythonObject) raises -> PythonObject:
    var a = a_obj.downcast_value_ptr[Array]()
    var b = b_obj.downcast_value_ptr[Array]()
    if len(a[].shape) != 2:
        raise Error("linalg.lstsq: a must be rank-2")
    var m = a[].shape[0]
    var n = a[].shape[1]
    var dtype_code = result_dtype_for_linalg_binary(a[].dtype_code, b[].dtype_code)
    var k = m if m < n else n
    var b_is_vec = len(b[].shape) == 1
    var nrhs = 1
    if not b_is_vec:
        if len(b[].shape) != 2:
            raise Error("linalg.lstsq: b must be rank-1 or rank-2")
        nrhs = b[].shape[1]
    if (b_is_vec and b[].shape[0] != m) or (not b_is_vec and b[].shape[0] != m):
        raise Error("linalg.lstsq: shape mismatch on first axis of b")
    var x_shape = List[Int]()
    x_shape.append(n)
    if not b_is_vec:
        x_shape.append(nrhs)
    var x = make_empty_array(dtype_code, x_shape^)
    var s_shape = List[Int]()
    s_shape.append(k)
    var s = make_empty_array(dtype_code, s_shape^)
    var rank_buf = Int(0)
    var rank_ptr = rebind[UnsafePointer[Int, MutExternalOrigin]](UnsafePointer(to=rank_buf))
    if dtype_code == ArrayDType.FLOAT32.value:
        var rcond_f32 = Float32(Float64(py=rcond_obj))
        lapack_lstsq_into[DType.float32](a[], b[], x, s, rcond_f32, rank_ptr)
    else:
        var rcond_f64 = Float64(py=rcond_obj)
        lapack_lstsq_into[DType.float64](a[], b[], x, s, rcond_f64, rank_ptr)
    return Python.list(PythonObject(alloc=x^), PythonObject(alloc=s^), PythonObject(rank_buf))


def pinv_ops(a_obj: PythonObject, rcond_obj: PythonObject) raises -> PythonObject:
    var a = a_obj.downcast_value_ptr[Array]()
    if len(a[].shape) != 2:
        raise Error("linalg.pinv: input must be rank-2")
    var m = a[].shape[0]
    var n = a[].shape[1]
    var dtype_code = result_dtype_for_linalg(a[].dtype_code)
    var x_shape = List[Int]()
    x_shape.append(n)
    x_shape.append(m)
    var x = make_empty_array(dtype_code, x_shape^)
    if m == 0 or n == 0:
        return PythonObject(alloc=x^)
    var rhs_shape = List[Int]()
    rhs_shape.append(m)
    rhs_shape.append(m)
    var rhs = make_empty_array(dtype_code, rhs_shape^)
    for row in range(m):
        for col in range(m):
            set_logical_from_f64(rhs, row * m + col, 1.0 if row == col else 0.0)
    var k = m if m < n else n
    var s_shape = List[Int]()
    s_shape.append(k)
    var s = make_empty_array(dtype_code, s_shape^)
    var rank_buf = Int(0)
    var rank_ptr = rebind[UnsafePointer[Int, MutExternalOrigin]](UnsafePointer(to=rank_buf))
    if dtype_code == ArrayDType.FLOAT32.value:
        var rcond_f32 = Float32(Float64(py=rcond_obj))
        lapack_lstsq_into[DType.float32](a[], rhs, x, s, rcond_f32, rank_ptr)
    else:
        var rcond_f64 = Float64(py=rcond_obj)
        lapack_lstsq_into[DType.float64](a[], rhs, x, s, rcond_f64, rank_ptr)
    return PythonObject(alloc=x^)
