"""LAPACK-backed linalg kernels and pure-Mojo LU fallback.

This module owns:
  - column-major transpose / write helpers (rectangular and square)
  - `maybe_lapack_solve_*` / `maybe_lapack_inverse_*` / `maybe_lapack_det_*`
    (the fast paths for `linalg.solve` / `linalg.inv` / `linalg.det`)
  - the pure-Mojo LU stack: `load_square_matrix_f64`, `make_lu_pivots`,
    `swap_lu_rows`, `swap_rhs_rows`, `lu_decompose_partial_pivot`,
    `solve_lu_factor_into`, `lu_solve_into`, `lu_inverse_into`, `lu_det`,
    `lu_det_into` — used as the fallback when LAPACK isn't available
  - `lapack_qr_reduced_*_into`, `lapack_qr_r_only_*_into`,
    `lapack_cholesky_*_into`, `lapack_eigh_*_into`,
    `lapack_eig_*_real_into`, `lapack_svd_*_into`, `lapack_lstsq_*_into`

Pattern (every LAPACK wrapper):
  1. transpose row-major Array → column-major scratch
  2. call LAPACK
  3. transpose / write the column-major result back into a row-major Array

Bool-output backend tracking: each LAPACK fast path stamps
`result.backend_code = BackendKind.ACCELERATE.value` so the Python side
can report which backend ran the op.

Cross-references: the LU pivoting derivation lives at
`docs/research/blas-lapack-dispatch.md §3`; the eig WR/WI compressed
encoding at §4.2.
"""

from std.memory.unsafe_pointer import alloc
from std.sys import CompilationTarget

from accelerate import (
    lapack_geev,
    lapack_gelsd,
    lapack_geqrf,
    lapack_gesdd,
    lapack_gesv,
    lapack_getrf,
    lapack_orgqr,
    lapack_potrf,
    lapack_syev,
)
from array import (
    Array,
    contiguous_ptr,
    get_logical_as_f64,
    is_c_contiguous,
    set_logical_from_f64,
)
from domain import ArrayDType, BackendKind


def abs_f64(value: Float64) -> Float64:
    if value < 0.0:
        return -value
    return value


def transpose_to_col_major[
    dt: DType
](src: Array, dst: UnsafePointer[Scalar[dt], MutExternalOrigin], n: Int,) raises:
    # Copy `src` (rank-2 n×n) into a column-major buffer pointed to by `dst`.
    # Fast path for c-contiguous source skips the per-element `physical_offset`
    # divmod that `get_logical_as_f64` pays. At n=128 this is ~50 µs vs ~5 µs.
    if is_c_contiguous(src):
        var src_ptr = contiguous_ptr[dt](src)
        for row in range(n):
            for col in range(n):
                dst[row + col * n] = src_ptr[row * n + col]
        return
    for row in range(n):
        for col in range(n):
            var v = get_logical_as_f64(src, row * n + col)
            dst[row + col * n] = v.cast[dt]()


def copy_rhs_to_col_major[
    dt: DType
](b: Array, dst: UnsafePointer[Scalar[dt], MutExternalOrigin], n: Int, rhs_columns: Int, vector_result: Bool,) raises:
    var matching_code: Int
    comptime if dt == DType.float32:
        matching_code = ArrayDType.FLOAT32.value
    else:
        matching_code = ArrayDType.FLOAT64.value
    if is_c_contiguous(b) and b.dtype_code == matching_code:
        var src_ptr = contiguous_ptr[dt](b)
        if vector_result:
            for row in range(n):
                dst[row] = src_ptr[row]
            return
        for row in range(n):
            for col in range(rhs_columns):
                dst[row + col * n] = src_ptr[row * rhs_columns + col]
        return
    for row in range(n):
        for col in range(rhs_columns):
            var logical = row
            if not vector_result:
                logical = row * rhs_columns + col
            var v = get_logical_as_f64(b, logical)
            dst[row + col * n] = v.cast[dt]()


def write_solve_result[
    dt: DType
](
    src: UnsafePointer[Scalar[dt], MutExternalOrigin],
    mut result: Array,
    n: Int,
    rhs_columns: Int,
    vector_result: Bool,
) raises:
    var matching_code: Int
    comptime if dt == DType.float32:
        matching_code = ArrayDType.FLOAT32.value
    else:
        matching_code = ArrayDType.FLOAT64.value
    if is_c_contiguous(result) and result.dtype_code == matching_code:
        var dst = contiguous_ptr[dt](result)
        if vector_result:
            for row in range(n):
                dst[row] = src[row]
            return
        for row in range(n):
            for col in range(rhs_columns):
                dst[row * rhs_columns + col] = src[row + col * n]
        return
    for row in range(n):
        for col in range(rhs_columns):
            var out_index = row
            if not vector_result:
                out_index = row * rhs_columns + col
            set_logical_from_f64(result, out_index, Float64(src[row + col * n]))


def maybe_lapack_solve[dt: DType](a: Array, b: Array, mut result: Array) raises -> Bool:
    var matching_code: Int
    comptime if dt == DType.float32:
        matching_code = ArrayDType.FLOAT32.value
    else:
        matching_code = ArrayDType.FLOAT64.value
    if (
        a.dtype_code != matching_code
        or b.dtype_code != matching_code
        or result.dtype_code != matching_code
        or len(a.shape) != 2
        or a.shape[0] != a.shape[1]
    ):
        return False
    var n = a.shape[0]
    var rhs_columns = 1
    var vector_result = True
    if len(b.shape) == 2:
        rhs_columns = b.shape[1]
        vector_result = False
    var a_ptr = alloc[Scalar[dt]](n * n)
    var b_ptr = alloc[Scalar[dt]](n * rhs_columns)
    var pivots = alloc[Int32](n)
    transpose_to_col_major[dt](a, a_ptr, n)
    copy_rhs_to_col_major[dt](b, b_ptr, n, rhs_columns, vector_result)
    var info: Int
    try:
        info = lapack_gesv[dt](n, rhs_columns, a_ptr, pivots, b_ptr)
    except:
        a_ptr.free()
        b_ptr.free()
        pivots.free()
        return False
    if info != 0:
        a_ptr.free()
        b_ptr.free()
        pivots.free()
        if info > 0:
            raise Error("linalg.solve() singular matrix")
        return False
    write_solve_result[dt](b_ptr, result, n, rhs_columns, vector_result)
    a_ptr.free()
    b_ptr.free()
    pivots.free()
    result.backend_code = BackendKind.ACCELERATE.value
    return True


def maybe_lapack_inverse[dt: DType](a: Array, mut result: Array) raises -> Bool:
    var matching_code: Int
    comptime if dt == DType.float32:
        matching_code = ArrayDType.FLOAT32.value
    else:
        matching_code = ArrayDType.FLOAT64.value
    if (
        a.dtype_code != matching_code
        or result.dtype_code != matching_code
        or len(a.shape) != 2
        or a.shape[0] != a.shape[1]
    ):
        return False
    var n = a.shape[0]
    var a_ptr = alloc[Scalar[dt]](n * n)
    var b_ptr = alloc[Scalar[dt]](n * n)
    var pivots = alloc[Int32](n)
    transpose_to_col_major[dt](a, a_ptr, n)
    for row in range(n):
        for col in range(n):
            if row == col:
                b_ptr[row + col * n] = 1.0
            else:
                b_ptr[row + col * n] = 0.0
    var info: Int
    try:
        info = lapack_gesv[dt](n, n, a_ptr, pivots, b_ptr)
    except:
        a_ptr.free()
        b_ptr.free()
        pivots.free()
        return False
    if info != 0:
        a_ptr.free()
        b_ptr.free()
        pivots.free()
        if info > 0:
            raise Error("linalg.inv() singular matrix")
        return False
    write_solve_result[dt](b_ptr, result, n, n, False)
    a_ptr.free()
    b_ptr.free()
    pivots.free()
    result.backend_code = BackendKind.ACCELERATE.value
    return True


def lapack_pivot_sign(pivots: UnsafePointer[Int32, MutExternalOrigin], n: Int) -> Float64:
    var sign = 1.0
    for i in range(n):
        if Int(pivots[i]) != i + 1:
            sign = -sign
    return sign


def maybe_lapack_det[dt: DType](a: Array, mut result: Array) raises -> Bool:
    var matching_code: Int
    comptime if dt == DType.float32:
        matching_code = ArrayDType.FLOAT32.value
    else:
        matching_code = ArrayDType.FLOAT64.value
    if (
        a.dtype_code != matching_code
        or result.dtype_code != matching_code
        or len(a.shape) != 2
        or a.shape[0] != a.shape[1]
    ):
        return False
    var n = a.shape[0]
    var a_ptr = alloc[Scalar[dt]](n * n)
    var pivots = alloc[Int32](n)
    transpose_to_col_major[dt](a, a_ptr, n)
    try:
        var info = lapack_getrf[dt](n, a_ptr, pivots)
        if info < 0:
            a_ptr.free()
            pivots.free()
            return False
        var det = lapack_pivot_sign(pivots, n)
        if info > 0:
            det = 0.0
        else:
            for i in range(n):
                det *= Float64(a_ptr[i + i * n])
        set_logical_from_f64(result, 0, det)
    except:
        a_ptr.free()
        pivots.free()
        return False
    a_ptr.free()
    pivots.free()
    result.backend_code = BackendKind.ACCELERATE.value
    return True


def load_square_matrix_f64(src: Array) raises -> List[Float64]:
    if len(src.shape) != 2 or src.shape[0] != src.shape[1]:
        raise Error("linalg operation requires a square rank-2 matrix")
    var n = src.shape[0]
    var out = List[Float64]()
    for i in range(n * n):
        out.append(get_logical_as_f64(src, i))
    return out^


def make_lu_pivots(n: Int) -> List[Int]:
    var pivots = List[Int]()
    for i in range(n):
        pivots.append(i)
    return pivots^


def swap_lu_rows(mut lu: List[Float64], n: Int, lhs: Int, rhs: Int):
    if lhs == rhs:
        return
    for col in range(n):
        var lhs_index = lhs * n + col
        var rhs_index = rhs * n + col
        var tmp = lu[lhs_index]
        lu[lhs_index] = lu[rhs_index]
        lu[rhs_index] = tmp


def swap_rhs_rows(mut rhs: List[Float64], columns: Int, lhs: Int, rhs_row: Int):
    if lhs == rhs_row:
        return
    for col in range(columns):
        var lhs_index = lhs * columns + col
        var rhs_index = rhs_row * columns + col
        var tmp = rhs[lhs_index]
        rhs[lhs_index] = rhs[rhs_index]
        rhs[rhs_index] = tmp


def lu_decompose_partial_pivot(mut lu: List[Float64], mut pivots: List[Int], n: Int) raises -> Int:
    # Doolittle-style in-place LU with partial (row) pivoting. Returns the
    # permutation parity (+1 / -1) for det() computation, or 0 on singular.
    #
    # Algorithm: for each column k, find row i ≥ k with max |A[i, k]| and
    # swap rows (k, i) — recording the pivot in pivots[k]. Then eliminate
    # below the pivot:
    #   A[i, k]      = A[i, k] / A[k, k]      (multiplier into L)
    #   A[i, j > k] -= A[i, k] · A[k, j]      (update U)
    # for each i > k. After the column-k pass, A[k, k:] holds U's k-th row
    # and A[k+1:, k] holds L's k-th column below the diagonal. L is unit
    # lower-triangular; the unit diagonal is implicit and not stored.
    #
    # Why pivoting: without it LU fails on `[[0, 1], [1, 0]]` (the leading
    # 1×1 minor is 0, so A[i, k] / A[k, k] divides by zero) and the growth
    # factor is unbounded. Partial pivoting bounds the growth factor by
    # 2^(n-1) (Wilkinson) and is empirically much better.
    #
    # Sign tracking: every row swap flips `sign`. Returned to lu_det to
    # compute det(A) = sign(P) · ∏ A[k, k].
    # Cross-ref `docs/research/blas-lapack-dispatch.md §3`.
    var sign = 1
    for k in range(n):
        var pivot = k
        var max_abs = abs_f64(lu[k * n + k])
        for row in range(k + 1, n):
            var value_abs = abs_f64(lu[row * n + k])
            if value_abs > max_abs:
                max_abs = value_abs
                pivot = row
        if max_abs == 0.0:
            return 0
        pivots[k] = pivot
        if pivot != k:
            swap_lu_rows(lu, n, k, pivot)
            sign = -sign
        var pivot_value = lu[k * n + k]
        for row in range(k + 1, n):
            var row_base = row * n
            lu[row_base + k] = lu[row_base + k] / pivot_value
            var factor = lu[row_base + k]
            for col in range(k + 1, n):
                lu[row_base + col] -= factor * lu[k * n + col]
    return sign


def solve_lu_factor_into(
    lu: List[Float64],
    pivots: List[Int],
    n: Int,
    mut rhs: List[Float64],
    rhs_columns: Int,
    mut result: Array,
    vector_result: Bool,
) raises:
    for row in range(n):
        swap_rhs_rows(rhs, rhs_columns, row, pivots[row])
    for row in range(n):
        for col in range(rhs_columns):
            var value = rhs[row * rhs_columns + col]
            for k in range(row):
                value -= lu[row * n + k] * rhs[k * rhs_columns + col]
            rhs[row * rhs_columns + col] = value
    for row in range(n - 1, -1, -1):
        for col in range(rhs_columns):
            var value = rhs[row * rhs_columns + col]
            for k in range(row + 1, n):
                value -= lu[row * n + k] * rhs[k * rhs_columns + col]
            rhs[row * rhs_columns + col] = value / lu[row * n + row]
    for row in range(n):
        for col in range(rhs_columns):
            var out_index = row * rhs_columns + col
            if vector_result:
                out_index = row
            set_logical_from_f64(result, out_index, rhs[row * rhs_columns + col])


def lu_solve_into(a: Array, b: Array, mut result: Array) raises:
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise Error("linalg.solve() requires a square rank-2 coefficient matrix")
    var n = a.shape[0]
    var rhs_columns = 1
    var vector_result = True
    if len(b.shape) == 1:
        if b.shape[0] != n:
            raise Error("linalg.solve() right-hand side shape mismatch")
    elif len(b.shape) == 2:
        if b.shape[0] != n:
            raise Error("linalg.solve() right-hand side shape mismatch")
        rhs_columns = b.shape[1]
        vector_result = False
    else:
        raise Error("linalg.solve() right-hand side must be rank 1 or rank 2")
    comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
        if maybe_lapack_solve[DType.float32](a, b, result):
            return
        if maybe_lapack_solve[DType.float64](a, b, result):
            return
    var lu = load_square_matrix_f64(a)
    var pivots = make_lu_pivots(n)
    if lu_decompose_partial_pivot(lu, pivots, n) == 0:
        raise Error("linalg.solve() singular matrix")
    var rhs_values = List[Float64]()
    for row in range(n):
        for col in range(rhs_columns):
            var logical = row
            if len(b.shape) == 2:
                logical = row * rhs_columns + col
            rhs_values.append(get_logical_as_f64(b, logical))
    solve_lu_factor_into(lu, pivots, n, rhs_values, rhs_columns, result, vector_result)


def lu_inverse_into(a: Array, mut result: Array) raises:
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise Error("linalg.inv() requires a square rank-2 matrix")
    comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
        if maybe_lapack_inverse[DType.float32](a, result):
            return
        if maybe_lapack_inverse[DType.float64](a, result):
            return
    var n = a.shape[0]
    var lu = load_square_matrix_f64(a)
    var pivots = make_lu_pivots(n)
    if lu_decompose_partial_pivot(lu, pivots, n) == 0:
        raise Error("linalg.inv() singular matrix")
    var rhs_values = List[Float64]()
    for row in range(n):
        for col in range(n):
            if row == col:
                rhs_values.append(1.0)
            else:
                rhs_values.append(0.0)
    solve_lu_factor_into(lu, pivots, n, rhs_values, n, result, False)


def lu_det(a: Array) raises -> Float64:
    # det(A) = sign(P) · ∏ A[k, k] from in-place LU with partial pivoting.
    # The product of the diagonal of U equals det(U); det(L) = 1 (unit
    # diagonal by construction); det(P) = ±1 from the pivot parity.
    # Singular case (sign == 0) → det == 0 directly, no diagonal traversal.
    # Cross-ref `docs/research/blas-lapack-dispatch.md §3.2`.
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise Error("linalg.det() requires a square rank-2 matrix")
    var n = a.shape[0]
    var lu = load_square_matrix_f64(a)
    var pivots = make_lu_pivots(n)
    var sign = lu_decompose_partial_pivot(lu, pivots, n)
    if sign == 0:
        return 0.0
    var det = Float64(sign)
    for i in range(n):
        det *= lu[i * n + i]
    return det


def lu_det_into(a: Array, mut result: Array) raises:
    comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
        if maybe_lapack_det[DType.float32](a, result):
            return
        if maybe_lapack_det[DType.float64](a, result):
            return
    set_logical_from_f64(result, 0, lu_det(a))


# ============================================================
# Phase-6d LAPACK-backed decompositions.
#
# Pattern: load row-major Array → column-major scratch (transpose during
# copy), call LAPACK, transpose result back to row-major Array. Each
# decomposition exposes a `qr_into`, `cholesky_into`, `svd_into`, etc.
# entry that allocates and writes into pre-shaped result Arrays.
# ============================================================


def transpose_to_col_major_rect[
    dt: DType
](src: Array, dst: UnsafePointer[Scalar[dt], MutExternalOrigin], rows: Int, cols: Int,) raises:
    # Rectangular variant of transpose_to_col_major: src is rows × cols.
    var matching_code: Int
    comptime if dt == DType.float32:
        matching_code = ArrayDType.FLOAT32.value
    else:
        matching_code = ArrayDType.FLOAT64.value
    if is_c_contiguous(src) and src.dtype_code == matching_code:
        var src_ptr = contiguous_ptr[dt](src)
        for row in range(rows):
            for col in range(cols):
                dst[row + col * rows] = src_ptr[row * cols + col]
        return
    for row in range(rows):
        for col in range(cols):
            var v = get_logical_as_f64(src, row * cols + col)
            dst[row + col * rows] = v.cast[dt]()


def write_col_major_to_array[
    dt: DType
](src: UnsafePointer[Scalar[dt], MutExternalOrigin], mut result: Array, rows: Int, cols: Int,) raises:
    var matching_code: Int
    comptime if dt == DType.float32:
        matching_code = ArrayDType.FLOAT32.value
    else:
        matching_code = ArrayDType.FLOAT64.value
    if is_c_contiguous(result) and result.dtype_code == matching_code:
        var dst = contiguous_ptr[dt](result)
        for row in range(rows):
            for col in range(cols):
                dst[row * cols + col] = src[row + col * rows]
        return
    for row in range(rows):
        for col in range(cols):
            set_logical_from_f64(result, row * cols + col, Float64(src[row + col * rows]))


# ---------- QR ----------
def lapack_qr_reduced_into[dt: DType](a: Array, mut q_out: Array, mut r_out: Array) raises:
    var m = a.shape[0]
    var n = a.shape[1]
    var k = m if m < n else n
    var qr_buf = alloc[Scalar[dt]](m * n)
    var tau = alloc[Scalar[dt]](k if k > 0 else 1)
    transpose_to_col_major_rect[dt](a, qr_buf, m, n)
    var info = lapack_geqrf[dt](m, n, qr_buf, tau)
    if info != 0:
        qr_buf.free()
        tau.free()
        raise Error("linalg.qr: geqrf failed")
    # Extract R: upper triangular of QR (k × n in col-major, but logically
    # k rows × n cols of R). Zero strictly-below-diagonal entries.
    for row in range(k):
        for col in range(n):
            var val: Scalar[dt]
            if col >= row:
                val = qr_buf[row + col * m]
            else:
                val = 0.0
            set_logical_from_f64(r_out, row * n + col, Float64(val))
    # Build Q via {s,d}orgqr (overwrites qr_buf with M×K Q).
    info = lapack_orgqr[dt](m, k, k, qr_buf, tau)
    if info != 0:
        qr_buf.free()
        tau.free()
        raise Error("linalg.qr: orgqr failed")
    write_col_major_to_array[dt](qr_buf, q_out, m, k)
    qr_buf.free()
    tau.free()
    q_out.backend_code = BackendKind.ACCELERATE.value
    r_out.backend_code = BackendKind.ACCELERATE.value


def lapack_qr_r_only_into[dt: DType](a: Array, mut r_out: Array) raises:
    # mode='r': R only.
    var m = a.shape[0]
    var n = a.shape[1]
    var k = m if m < n else n
    var qr_buf = alloc[Scalar[dt]](m * n)
    var tau = alloc[Scalar[dt]](k if k > 0 else 1)
    transpose_to_col_major_rect[dt](a, qr_buf, m, n)
    var info = lapack_geqrf[dt](m, n, qr_buf, tau)
    if info != 0:
        qr_buf.free()
        tau.free()
        raise Error("linalg.qr: geqrf failed")
    for row in range(k):
        for col in range(n):
            var val: Scalar[dt]
            if col >= row:
                val = qr_buf[row + col * m]
            else:
                val = 0.0
            set_logical_from_f64(r_out, row * n + col, Float64(val))
    qr_buf.free()
    tau.free()
    r_out.backend_code = BackendKind.ACCELERATE.value


# ---------- Cholesky ----------
def write_cholesky_lower[
    dt: DType
](src: UnsafePointer[Scalar[dt], MutExternalOrigin], mut result: Array, n: Int,) raises:
    var dst = contiguous_ptr[dt](result)
    for row in range(n):
        var row_base = row * n
        for col in range(n):
            if col <= row:
                dst[row_base + col] = src[col + row_base]
            else:
                dst[row_base + col] = 0.0


def lapack_cholesky_into[dt: DType](a: Array, mut result: Array) raises:
    var n = a.shape[0]
    var buf = alloc[Scalar[dt]](n * n)
    transpose_to_col_major[dt](a, buf, n)
    # numpy returns L (lower); LAPACK's lower-triangular factor in
    # column-major maps to upper-triangular in row-major. So we ask
    # LAPACK for "upper" (UPLO='U') and the resulting buffer in
    # row-major is the lower-triangular L we want.
    var info = lapack_potrf[dt](n, buf, True)
    if info != 0:
        buf.free()
        if info > 0:
            raise Error("linalg.cholesky: matrix is not positive definite")
        raise Error("linalg.cholesky: potrf failed")
    write_cholesky_lower[dt](buf, result, n)
    buf.free()
    result.backend_code = BackendKind.ACCELERATE.value


# ---------- Symmetric eigendecomposition ----------
def lapack_eigh_into[dt: DType](a: Array, mut w_out: Array, mut v_out: Array, compute_eigenvectors: Bool) raises:
    var n = a.shape[0]
    var buf = alloc[Scalar[dt]](n * n)
    var wbuf = alloc[Scalar[dt]](n if n > 0 else 1)
    transpose_to_col_major[dt](a, buf, n)
    # UPLO='L' so LAPACK reads from the lower triangle of col-major buf,
    # which corresponds to the upper triangle of the row-major source.
    # numpy.linalg.eigh defaults to UPLO='L' (read from lower in row-major
    # ordering, i.e. upper in col-major). Pass `upper=False` here so the
    # F77 'L' takes effect.
    var info = lapack_syev[dt](n, buf, wbuf, compute_eigenvectors, False)
    if info != 0:
        buf.free()
        wbuf.free()
        raise Error("linalg.eigh: syev failed to converge")
    for i in range(n):
        set_logical_from_f64(w_out, i, Float64(wbuf[i]))
    if compute_eigenvectors:
        # buf is col-major N×N; eigenvectors are columns. Output expects
        # rows = original-rows, cols = eigenvectors → write as col-major
        # → row-major, which is exactly write_col_major_to_array.
        write_col_major_to_array[dt](buf, v_out, n, n)
        v_out.backend_code = BackendKind.ACCELERATE.value
    buf.free()
    wbuf.free()
    w_out.backend_code = BackendKind.ACCELERATE.value


# ---------- General eigendecomposition (real-result fast path) ----------
def lapack_eig_real_into[
    dt: DType
](a: Array, mut wr_out: Array, mut wi_out: Array, mut vr_out: Array, compute_eigenvectors: Bool,) raises -> Bool:
    """Returns True if all eigenvalues are real (imaginary parts zero).
    Caller must check before exposing to numpy-parity APIs (we don't have
    complex dtype yet; mixed real+complex eigenvalue spectra raise upstream).
    The real-only path writes wr_out, vr_out (column eigenvectors).

    LAPACK {s,d}geev for a real n×n input — eigenvalues come back in compressed
    WR/WI form because real matrices have real eigenvalues OR conjugate-pair
    complex eigenvalues, never odd-out complex.

    Encoding (per LAPACK docs):
    - real eigenvalue at index i: WR[i] = λ, WI[i] = 0
    - complex conjugate pair at indices (i, i+1): WI[i] > 0, WI[i+1] < 0,
      with WR[i] == WR[i+1] (the shared real part) and WI[i+1] == -WI[i]
      (the imaginary parts negated). Eigenvalues are WR[i] ± WI[i]·j.

    Eigenvector compression mirrors this. For a real eigenvalue, column i
    of VR is the real eigenvector. For a conjugate pair, columns i and i+1
    of VR hold the real and imaginary parts of one complex eigenvector;
    the conjugate-pair partner is implicit (real part copied, imag negated).

    `all_real` short-circuit: if every WI[i] is zero the caller can skip
    the complex unpack and treat WR directly as a real eigenvalue array.
    Cross-ref `docs/research/blas-lapack-dispatch.md §4.2`.
    """
    var n = a.shape[0]
    var buf = alloc[Scalar[dt]](n * n)
    var wr = alloc[Scalar[dt]](n if n > 0 else 1)
    var wi = alloc[Scalar[dt]](n if n > 0 else 1)
    var vr_size = n * n if compute_eigenvectors else 1
    var vr = alloc[Scalar[dt]](vr_size)
    transpose_to_col_major[dt](a, buf, n)
    var info = lapack_geev[dt](n, buf, wr, wi, vr, compute_eigenvectors)
    if info != 0:
        buf.free()
        wr.free()
        wi.free()
        vr.free()
        raise Error("linalg.eig: geev failed to converge")
    for i in range(n):
        set_logical_from_f64(wr_out, i, Float64(wr[i]))
        set_logical_from_f64(wi_out, i, Float64(wi[i]))
    var all_real = True
    for i in range(n):
        if wi[i] != 0.0:
            all_real = False
            break
    if compute_eigenvectors:
        write_col_major_to_array[dt](vr, vr_out, n, n)
        vr_out.backend_code = BackendKind.ACCELERATE.value
    buf.free()
    wr.free()
    wi.free()
    vr.free()
    wr_out.backend_code = BackendKind.ACCELERATE.value
    wi_out.backend_code = BackendKind.ACCELERATE.value
    return all_real


# ---------- SVD ----------
def lapack_svd_into[
    dt: DType
](a: Array, mut u_out: Array, mut s_out: Array, mut vt_out: Array, full_matrices: Bool, compute_uv: Bool,) raises:
    var m = a.shape[0]
    var n = a.shape[1]
    var k = m if m < n else n
    var a_buf = alloc[Scalar[dt]](m * n)
    var s_buf = alloc[Scalar[dt]](k if k > 0 else 1)
    var u_rows = m
    var u_cols = m if full_matrices else k
    var vt_rows = n if full_matrices else k
    var vt_cols = n
    var u_size = u_rows * u_cols if compute_uv else 1
    var vt_size = vt_rows * vt_cols if compute_uv else 1
    var u_buf = alloc[Scalar[dt]](u_size if u_size > 0 else 1)
    var vt_buf = alloc[Scalar[dt]](vt_size if vt_size > 0 else 1)
    transpose_to_col_major_rect[dt](a, a_buf, m, n)
    var info = lapack_gesdd[dt](m, n, a_buf, s_buf, u_buf, vt_buf, full_matrices, compute_uv)
    if info != 0:
        a_buf.free()
        s_buf.free()
        u_buf.free()
        vt_buf.free()
        raise Error("linalg.svd: gesdd failed to converge")
    for i in range(k):
        set_logical_from_f64(s_out, i, Float64(s_buf[i]))
    if compute_uv:
        # U is col-major u_rows × u_cols. Transpose to row-major.
        write_col_major_to_array[dt](u_buf, u_out, u_rows, u_cols)
        # VT is col-major vt_rows × vt_cols. Same.
        write_col_major_to_array[dt](vt_buf, vt_out, vt_rows, vt_cols)
        u_out.backend_code = BackendKind.ACCELERATE.value
        vt_out.backend_code = BackendKind.ACCELERATE.value
    a_buf.free()
    s_buf.free()
    u_buf.free()
    vt_buf.free()
    s_out.backend_code = BackendKind.ACCELERATE.value


# ---------- Least squares ----------
def lapack_lstsq_into[
    dt: DType
](
    a: Array,
    b: Array,
    mut x_out: Array,
    mut s_out: Array,
    rcond: Scalar[dt],
    rank_out_ptr: UnsafePointer[Int, MutExternalOrigin],
) raises:
    """Solves ‖A·x − b‖₂ minimum-norm. Writes solution to x_out (shape
    (N,) for vector b, (N, NRHS) for matrix b), singular values to s_out
    (length min(M, N)), and the effective rank to *rank_out_ptr."""
    var m = a.shape[0]
    var n = a.shape[1]
    var nrhs = 1
    var b_is_vec = True
    if len(b.shape) == 2:
        nrhs = b.shape[1]
        b_is_vec = False
    var k = m if m < n else n
    var ldb = m if m > n else n
    var a_buf = alloc[Scalar[dt]](m * n)
    var b_buf = alloc[Scalar[dt]](ldb * nrhs)
    var s_buf = alloc[Scalar[dt]](k if k > 0 else 1)
    transpose_to_col_major_rect[dt](a, a_buf, m, n)
    # Copy b into the LDB × NRHS column-major scratch (rows beyond M
    # padding stays 0; numpy convention).
    for col in range(nrhs):
        for row in range(ldb):
            if row < m:
                var logical = row if b_is_vec else row * nrhs + col
                var v = get_logical_as_f64(b, logical)
                b_buf[row + col * ldb] = v.cast[dt]()
            else:
                b_buf[row + col * ldb] = 0.0
    var info = lapack_gelsd[dt](m, n, nrhs, a_buf, b_buf, s_buf, rcond, rank_out_ptr)
    if info != 0:
        a_buf.free()
        b_buf.free()
        s_buf.free()
        raise Error("linalg.lstsq: gelsd failed to converge")
    # Write x_out: first N rows of the col-major b_buf hold the solution.
    if b_is_vec:
        for row in range(n):
            set_logical_from_f64(x_out, row, Float64(b_buf[row]))
    else:
        for row in range(n):
            for col in range(nrhs):
                set_logical_from_f64(x_out, row * nrhs + col, Float64(b_buf[row + col * ldb]))
    for i in range(k):
        set_logical_from_f64(s_out, i, Float64(s_buf[i]))
    a_buf.free()
    b_buf.free()
    s_buf.free()
    x_out.backend_code = BackendKind.ACCELERATE.value
    s_out.backend_code = BackendKind.ACCELERATE.value
