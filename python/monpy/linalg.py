from __future__ import annotations

import builtins
import operator
import typing  # used by einsum (zip, etc.)

from . import (
  _native,
  _resolve_dtype,
  asarray,
  ascontiguousarray,
  diagonal,
  float32,
  float64,
  matmul,
  matrix_transpose,
  multiply,
  ndarray,
  ravel,
  reshape,
  result_type,
  sqrt,
  square,
  transpose,
  zeros,
)
from . import (
  sum as _sum,
)

_T = typing.TypeVar("_T")
_Scalar: typing.TypeAlias = builtins.bool | builtins.int | builtins.float | builtins.complex
_QRResult: typing.TypeAlias = ndarray | tuple[ndarray, ndarray]
_SVDResult: typing.TypeAlias = ndarray | tuple[ndarray, ndarray, ndarray]


class LinAlgError(ValueError):
  pass


def _w(fn: typing.Callable[..., _T], *a: object) -> _T:
  try:
    return fn(*a)
  except Exception as exc:
    raise LinAlgError(str(exc)) from exc


def _array(value: object) -> ndarray:
  return value if type(value) is ndarray else asarray(value)


def _matmul(a: ndarray, b: ndarray) -> ndarray:
  return ndarray._wrap(_native.matmul(a._native, b._native))


def _dot_scalar(a: ndarray, b: ndarray) -> _Scalar:
  return typing.cast(_Scalar, _native.linalg_dot_scalar(a._native, b._native))


def _dot_scalar_fast(a: ndarray, b: ndarray) -> object | None:
  return _native.linalg_dot_scalar_float_try(a._native, b._native)


def _has_norm2_kernel(x: ndarray) -> bool:
  return x.dtype in (float32, float64)


def _has_vecdot_kernel(a: ndarray, b: ndarray) -> bool:
  return a.dtype == b.dtype and a.dtype in (float32, float64)


def _norm2_all(x: ndarray) -> ndarray:
  return ndarray._wrap(_w(_native.linalg_norm2_all, x._native))


def _norm1_all(x: ndarray) -> ndarray:
  return ndarray._wrap(_w(_native.linalg_norm1_all, x._native))


def _norm2_last_axis(x: ndarray) -> ndarray:
  return ndarray._wrap(_w(_native.linalg_norm2_last_axis, x._native))


def _vecdot_last_axis(a: ndarray, b: ndarray) -> ndarray:
  return ndarray._wrap(_native.linalg_vecdot_last_axis(a._native, b._native))


def _vecdot_last_axis_fast(a: ndarray, b: ndarray) -> ndarray | None:
  out = _native.linalg_vecdot_last_axis_float_try(a._native, b._native)
  return None if out is None else ndarray._wrap(out)


def _matrix_power_fast(a: ndarray, n: int) -> ndarray | None:
  out = _w(_native.linalg_matrix_power_float_try, a._native, n)
  return None if out is None else ndarray._wrap(out)


def _normalize_axis(axis: int, ndim: int) -> int:
  ax = axis + ndim if axis < 0 else axis
  if ax < 0 or ax >= ndim:
    raise ValueError("axis out of range")
  return ax


_INF = float("inf")
_NEG_INF = float("-inf")


def _prod(xs: typing.Iterable[int]) -> int:
  out = 1
  for x in xs:
    out *= x
  return out


def _axis_tuple(axis: object) -> tuple[int, ...]:
  if isinstance(axis, builtins.int):
    return (axis,)
  if not hasattr(axis, "__iter__"):
    return (builtins.int(typing.cast(typing.Any, axis)),)
  return tuple(builtins.int(typing.cast(typing.Any, a)) for a in typing.cast(typing.Iterable[object], axis))


def _eig_parts(wr: ndarray, wi: ndarray, n: int) -> tuple[list[float], list[float], list[complex]]:
  r = [float(wr._native.get_scalar(i)) for i in range(n)]
  im = [float(wi._native.get_scalar(i)) for i in range(n)]
  return r, im, [complex(r[i], im[i]) for i in range(n)]


# Existing native-backed primitives.
def solve(a: object, b: object) -> ndarray:
  l, r = _array(a), _array(b)
  return ndarray(_w(_native.linalg_solve, l._native, r._native))


def inv(a: object) -> ndarray:
  x = _array(a)
  return ndarray(_w(_native.linalg_inv, x._native))


def det(a: object) -> _Scalar:
  x = _array(a)
  return ndarray(_w(_native.linalg_det, x._native))._scalar()


# Expanded linalg surface on top of matmul + reductions. No LAPACK bindings yet;
# the algorithm-heavy decomps (qr/svd/eig/cholesky/lstsq/pinv) raise
# NotImplementedError until the accelerate layer wires them up.
def dot(a: object, b: object) -> object:
  if type(a) is ndarray and type(b) is ndarray:
    out = _dot_scalar_fast(a, b)
    if out is not None:
      return out
  A = _array(a)
  B = _array(b)
  if A.ndim == 0 or B.ndim == 0:
    return multiply(A, B)
  if A.ndim == 1 and B.ndim == 1:
    if A.shape[0] != B.shape[0]:
      raise ValueError("dot: shape mismatch")
    if _has_vecdot_kernel(A, B):
      return _dot_scalar(A, B)
    return _matmul(A, B)._scalar()
  if A.ndim <= 2 and B.ndim <= 2:
    return matmul(A, B)
  raise NotImplementedError("dot: ndim>2 not implemented")


def vdot(a: object, b: object) -> object:
  if type(a) is ndarray and type(b) is ndarray:
    out = _dot_scalar_fast(a, b)
    if out is not None:
      return out
  A = _array(a)
  B = _array(b)
  if A.ndim == 1 and B.ndim == 1:
    if A.shape != B.shape:
      raise ValueError("vdot: shape mismatch")
    if _has_vecdot_kernel(A, B):
      return _dot_scalar(A, B)
    return _matmul(A, B)._scalar()
  A = ravel(A)
  B = ravel(B)
  if A.shape != B.shape:
    raise ValueError("vdot: shape mismatch")
  if _has_vecdot_kernel(A, B):
    return _dot_scalar(A, B)
  return _matmul(A, B)._scalar()


def inner(a: object, b: object) -> object:
  if type(a) is ndarray and type(b) is ndarray:
    out = _dot_scalar_fast(a, b)
    if out is not None:
      return out
  A = _array(a)
  B = _array(b)
  if A.ndim == 0 or B.ndim == 0:
    return multiply(A, B)
  if A.shape[-1] != B.shape[-1]:
    raise ValueError("inner: trailing axis mismatch")
  if A.ndim == 1 and B.ndim == 1:
    if _has_vecdot_kernel(A, B):
      return _dot_scalar(A, B)
    return _matmul(A, B)._scalar()
  if A.ndim == 1:
    return _sum(multiply(A, B), axis=-1)
  if B.ndim == 1:
    return _sum(multiply(A, B), axis=-1)
  raise NotImplementedError("inner: only 1D x 1D / 1D x N / N x 1D supported in v1")


def outer(a: object, b: object) -> ndarray:
  A = _array(a)
  B = _array(b)
  return ndarray._wrap(_w(_native.linalg_outer, A._native, B._native))


def _kron_native(a: ndarray, b: ndarray) -> ndarray:
  return ndarray._wrap(_w(_native.linalg_kron, a._native, b._native))


def tensordot(a: object, b: object, axes: int | tuple[object, object] = 2) -> object:
  A = _array(a)
  B = _array(b)
  if isinstance(axes, int):
    n = axes
    if n == 1 and A.ndim == 2 and B.ndim == 2:
      return _matmul(A, B)
    if n == 2 and A.ndim == 2 and B.ndim == 2 and A.shape == B.shape:
      return _matmul(ravel(A), ravel(B))
    a_axes = tuple(range(A.ndim - n, A.ndim))
    b_axes = tuple(range(n))
  else:
    a_axes, b_axes = axes
    a_axes = _axis_tuple(a_axes)
    b_axes = _axis_tuple(b_axes)
  for ax_a, ax_b in zip(a_axes, b_axes, strict=True):
    if A.shape[ax_a] != B.shape[ax_b]:
      raise ValueError("tensordot: axis size mismatch")
  a_kept = tuple(d for d in range(A.ndim) if d not in a_axes)
  b_kept = tuple(d for d in range(B.ndim) if d not in b_axes)
  A_perm = transpose(A, tuple(a_kept) + a_axes)
  B_perm = transpose(B, b_axes + b_kept)
  a_kept_size = _prod(A.shape[d] for d in a_kept)
  contract_size = _prod(A.shape[d] for d in a_axes)
  b_kept_size = _prod(B.shape[d] for d in b_kept)
  A_mat = ascontiguousarray(A_perm).reshape((a_kept_size, contract_size))
  B_mat = ascontiguousarray(B_perm).reshape((contract_size, b_kept_size))
  out = _matmul(A_mat, B_mat)
  out_shape = tuple(A.shape[d] for d in a_kept) + tuple(B.shape[d] for d in b_kept)
  return reshape(out, out_shape) if out_shape else out.reshape(())


def kron(a: object, b: object) -> ndarray:
  A = _array(a)
  B = _array(b)
  return _kron_native(A, B)


def cross(a: object, b: object, axisa: int = -1, axisb: int = -1, axisc: int = -1, axis: int | None = None) -> ndarray:
  if axis is not None:
    axisa = axisb = axisc = axis
  A = _array(a)
  B = _array(b)
  if A.shape[axisa] != 3 or B.shape[axisb] != 3:
    raise ValueError("cross: only 3D vectors supported")
  if A.ndim != 1 or B.ndim != 1:
    raise NotImplementedError("cross: ndim>1 not implemented in v1")
  ax, ay, az = A._native.get_scalar(0), A._native.get_scalar(1), A._native.get_scalar(2)
  bx, by, bz = B._native.get_scalar(0), B._native.get_scalar(1), B._native.get_scalar(2)
  cx = ay * bz - az * by
  cy = az * bx - ax * bz
  cz = ax * by - ay * bx
  t = result_type(A, B)
  return ndarray(_native.from_flat([cx, cy, cz], (3,), t.code))


def matvec(a: object, b: object) -> ndarray:
  A = _array(a)
  B = _array(b)
  if A.ndim < 2 or B.ndim < 1:
    raise ValueError("matvec: a must be (...,M,N), b must be (...,N)")
  if A.ndim == 2 and B.ndim == 1:
    return _matmul(A, B)
  return typing.cast(ndarray, matmul(A, reshape(B, B.shape + (1,)))).reshape(A.shape[:-1])


def vecmat(a: object, b: object) -> ndarray:
  A = _array(a)
  B = _array(b)
  if A.ndim == 1 and B.ndim == 2:
    return _matmul(A, B)
  return typing.cast(ndarray, matmul(reshape(A, A.shape[:-1] + (1, A.shape[-1])), B)).reshape(
    A.shape[:-1] + B.shape[-1:]
  )


def vecdot(a: object, b: object, axis: int = -1) -> object:
  if type(a) is ndarray and type(b) is ndarray and (axis == -1 or axis == 1):
    out = _vecdot_last_axis_fast(a, b)
    if out is not None:
      return out
  A = _array(a)
  B = _array(b)
  if A.ndim == 2 and B.ndim == 2 and A.shape == B.shape and _has_vecdot_kernel(A, B):
    ax = _normalize_axis(axis, A.ndim)
    if ax == 1:
      return _vecdot_last_axis(A, B)
  return _sum(multiply(A, B), axis=axis)


def trace(a: object, offset: int = 0, axis1: int = 0, axis2: int = 1, dtype: object = None) -> object:
  A = _array(a)
  if A.ndim == 2:
    dc = -1 if dtype is None else _resolve_dtype(dtype).code
    try:
      out = _native.trace(A._native, builtins.int(offset), builtins.int(axis1), builtins.int(axis2), dc)
    except Exception as exc:
      raise ValueError(str(exc)) from exc
    return out.get_scalar(0)
  d = diagonal(A, offset=offset, axis1=axis1, axis2=axis2)
  return _sum(d, dtype=dtype) if d.ndim == 1 else _sum(d, axis=-1, dtype=dtype)


def norm(x: object, ord: object = None, axis: object = None, keepdims: bool = False) -> object:
  X = _array(x)
  if _has_norm2_kernel(X) and not keepdims and ord == 1:
    if axis is None and X.ndim == 1:
      return _norm1_all(X)
    if isinstance(axis, int):
      ax = _normalize_axis(axis, X.ndim)
      if X.ndim == 1 and ax == 0:
        return _norm1_all(X)
  if _has_norm2_kernel(X) and not keepdims and (ord is None or ord == 2):
    if axis is None and X.ndim in (1, 2):
      return _norm2_all(X)
    if isinstance(axis, int):
      ax = _normalize_axis(axis, X.ndim)
      if X.ndim == 1 and ax == 0:
        return _norm2_all(X)
      if X.ndim == 2 and ax == 1:
        return _norm2_last_axis(X)
  if axis is None and X.ndim == 1:
    axis = 0
  if axis is None and X.ndim == 2 and ord is None:
    return sqrt(_sum(square(X)))
  if isinstance(axis, int):
    if ord is None or ord == 2:
      return sqrt(_sum(square(X), axis=axis, keepdims=keepdims))
    if ord == 1:
      from . import absolute as _abs

      return _sum(_abs(X), axis=axis, keepdims=keepdims)
    if ord == _INF:
      from . import absolute as _abs

      abs_x = typing.cast(ndarray, _abs(X))
      return abs_x.max(axis=axis) if not keepdims else abs_x.max(axis=axis)
    if ord == _NEG_INF:
      from . import absolute as _abs

      abs_x = typing.cast(ndarray, _abs(X))
      return abs_x.min(axis=axis) if not keepdims else abs_x.min(axis=axis)
    if ord == 0:
      from . import not_equal
      from . import sum as ne_sum

      return ne_sum(not_equal(X, 0), axis=axis, keepdims=keepdims)
    return _w_pow(X, typing.cast(builtins.int | builtins.float, ord), axis, keepdims)
  if axis is None and X.ndim == 1:
    return sqrt(_sum(square(X)))
  raise NotImplementedError("norm: matrix norms (axis=tuple) not implemented in v1")


def _w_pow(X: ndarray, p: builtins.int | builtins.float, axis: object, keepdims: bool) -> object:
  from . import absolute as _abs
  from . import power as _pow

  return _pow(_sum(_pow(_abs(X), p), axis=axis, keepdims=keepdims), 1.0 / p)


def vector_norm(x: object, axis: object = None, keepdims: bool = False, ord: object = 2) -> object:
  return norm(x, ord=ord, axis=-1 if axis is None else axis, keepdims=keepdims)


def matrix_norm(x: object, axis: object = (-2, -1), keepdims: bool = False, ord: object = None) -> object:
  if ord is not None:
    raise NotImplementedError("matrix_norm with ord != None not implemented in v1")
  X = _array(x)
  if _has_norm2_kernel(X) and X.ndim == 2 and not keepdims:
    axes = tuple(sorted(_normalize_axis(a, X.ndim) for a in _axis_tuple(axis)))
    if axes == (0, 1):
      return _norm2_all(X)
  return sqrt(_sum(square(X), axis=axis, keepdims=keepdims))


def matrix_rank(M: object, tol: object = None, hermitian: bool = False) -> int:
  del hermitian
  X = _floatify(_array(M))
  if X.ndim < 2:
    return 1 if any(X._native.get_scalar(i) != 0 for i in range(X.size)) else 0
  s = svd(X, compute_uv=False)
  smax = max(float(s._native.get_scalar(i)) for i in range(s.size))
  threshold = smax * max(X.shape[-2:]) * _eps_for(X.dtype) if tol is None else float(typing.cast(typing.Any, tol))
  return sum(1 for i in range(s.size) if float(s._native.get_scalar(i)) > threshold)


def matrix_power(a: object, n: object) -> ndarray:
  A = _array(a)
  native = A._native
  ndim = int(native.ndim())
  if ndim < 2:
    raise LinAlgError("Last 2 dimensions of the array must be square")
  matrix_dim = int(native.shape_at(ndim - 1))
  if int(native.shape_at(ndim - 2)) != matrix_dim:
    raise LinAlgError("Last 2 dimensions of the array must be square")
  try:
    n_int = operator.index(typing.cast(typing.SupportsIndex, n))
  except TypeError as exc:
    raise TypeError("exponent must be an integer") from exc
  fast = _matrix_power_fast(A, n_int)
  if fast is not None:
    return fast
  if n_int == 0:
    eye_shape = A.shape
    out = zeros(eye_shape, dtype=A.dtype)
    for i in range(A.shape[-1]):
      out[..., i, i] = 1
    return out
  if n_int < 0:
    return matrix_power(inv(A), -n_int)
  result = A
  for _ in range(n_int - 1):
    result = typing.cast(ndarray, matmul(result, A))
  return result


def slogdet(a: object) -> tuple[_Scalar, _Scalar]:
  A = _array(a)
  sign, logdet = _w(_native.linalg_slogdet, A._native)
  return typing.cast(_Scalar, sign), typing.cast(_Scalar, logdet)


def multi_dot(arrays: typing.Sequence[object]) -> object:
  arrs = [_array(a) for a in arrays]
  if len(arrs) < 2:
    raise ValueError("multi_dot: need at least two arrays")
  out: object = arrs[0]
  for a in arrs[1:]:
    out = matmul(out, a)
  return out


def tensorinv(a: object, ind: int = 2) -> ndarray:
  A = _array(a)
  return ndarray._wrap(_w(_native.linalg_tensorinv, A._native, ind))


def tensorsolve(a: object, b: object, axes: typing.Sequence[int] | None = None) -> ndarray:
  # numpy.linalg.tensorsolve: like solve but with general-rank tensors.
  # Layout: a.shape == b.shape + x.shape (the leading axes of a match
  # b's full shape; the trailing axes form x's shape). After moving the
  # contracted axes to the end (when `axes` is given), reshape into a
  # square (prod(b.shape), prod(x.shape)) matrix and call solve.
  A = _array(a)
  B = _array(b)
  if axes is not None:
    # Move the listed axes to the trailing end (they form x's shape).
    a_axes = list(range(A.ndim))
    for ax in sorted(axes, reverse=True):
      a_axes.pop(ax)
    a_axes += list(axes)
    A = transpose(A, tuple(a_axes))
  if axes is None:
    return ndarray._wrap(_w(_native.linalg_tensorsolve, A._native, B._native))
  prod_b = _prod(B.shape)
  # Leading axes of A correspond to b's shape; trailing axes are x's shape.
  if A.ndim < B.ndim or A.shape[: B.ndim] != B.shape:
    raise ValueError("tensorsolve: leading axes of a must match b.shape")
  x_shape = A.shape[B.ndim :]
  prod_x = _prod(x_shape)
  if prod_b != prod_x:
    raise ValueError("tensorsolve: square reshape required")
  flat_a = A.reshape((prod_b, prod_x))
  flat_b = ravel(B)
  x = solve(flat_a, flat_b)
  return x.reshape(x_shape)


def einsum(subscripts: str, *operands: object, **kwargs: object) -> object:
  # Einstein summation: parse subscripts, build a contraction plan, walk it.
  # v1: supports the (input1,input2,...->output) form with letter labels.
  # Implementation strategy: reduce the sequence of operands pairwise via
  # `_einsum_pair_contract`, building an output indexing that matches the
  # spec. Equivalent to calling `tensordot` along the matched axes.
  if "out" in kwargs:
    raise NotImplementedError("einsum: out= not supported in v1")
  if "optimize" in kwargs:
    pass  # accepted, ignored (we always do pairwise)
  if "->" in subscripts:
    in_subs, out_sub = subscripts.split("->")
  else:
    in_subs, out_sub = subscripts, None
  in_subs = in_subs.split(",")
  if len(in_subs) != len(operands):
    raise ValueError("einsum: subscript / operand count mismatch")
  arrs = [asarray(op) for op in operands]
  for sub, arr in builtins.zip(in_subs, arrs, strict=True):
    if len(sub) != arr.ndim:
      raise ValueError(f"einsum: subscript '{sub}' does not match operand ndim {arr.ndim}")
  # Auto-derive output: every label that appears exactly once across inputs.
  if out_sub is None:
    counts = {}
    for sub in in_subs:
      for ch in sub:
        counts[ch] = counts.get(ch, 0) + 1
    out_sub = "".join(sorted([ch for ch, c in counts.items() if c == 1]))
  # Contract operands left-to-right.
  cur_sub, cur = in_subs[0], arrs[0]
  # Handle internal traces (repeated labels in a single operand) early.
  cur_sub, cur = _einsum_trace_diag(cur_sub, cur)
  for nxt_sub, nxt in builtins.zip(in_subs[1:], arrs[1:]):
    nxt_sub, nxt = _einsum_trace_diag(nxt_sub, nxt)
    cur_sub, cur = _einsum_pair_contract(cur_sub, cur, nxt_sub, nxt)
  # Final reduction: sum out any labels not in the output, then permute.
  return _einsum_finalise(cur_sub, cur, out_sub)


def _einsum_trace_diag(sub: str, arr: ndarray) -> tuple[str, ndarray]:
  # Collapse repeated labels in a single operand by taking the diagonal.
  while True:
    seen = {}
    repeated = None
    for i, ch in enumerate(sub):
      if ch in seen:
        repeated = (seen[ch], i, ch)
        break
      seen[ch] = i
    if repeated is None:
      return sub, arr
    a, b, ch = repeated
    arr = diagonal(arr, offset=0, axis1=a, axis2=b)  # diagonal moves to the end
    new_sub = [c for i, c in enumerate(sub) if i != a and i != b] + [ch]
    sub = "".join(new_sub)


def _einsum_pair_contract(la: str, a: ndarray, lb: str, b: ndarray) -> tuple[str, ndarray]:
  # Pair contract two arrays based on their label strings.
  # Strategy: matched labels (in both) become contraction axes; unique
  # labels become free axes; the result label string is la_unique + lb_unique.
  shared = [ch for ch in la if ch in lb]
  la_unique = [ch for ch in la if ch not in shared]
  lb_unique = [ch for ch in lb if ch not in shared]
  a_axes = tuple(la.index(ch) for ch in shared)
  b_axes = tuple(lb.index(ch) for ch in shared)
  out = tensordot(a, b, axes=(a_axes, b_axes))
  return "".join(la_unique + lb_unique), typing.cast(ndarray, out)


def _einsum_finalise(sub: str, arr: object, out_sub: str) -> object:
  # Sum out labels not in out_sub, then permute the remaining axes
  # to match out_sub's order.
  to_sum = [i for i, ch in enumerate(sub) if ch not in out_sub]
  for ax in sorted(to_sum, reverse=True):
    arr = _sum(arr, axis=ax)
    sub = sub[:ax] + sub[ax + 1 :]
  if not sub and not out_sub:
    return arr._scalar() if isinstance(arr, ndarray) and arr.ndim == 0 else arr
  perm = [sub.index(ch) for ch in out_sub]
  if perm != list(range(len(perm))):
    arr = transpose(arr, tuple(perm))
  return arr


# Helpers for the LAPACK-backed paths below.
_QR_MODES = {"reduced": 0, "complete": 1, "r": 2}


def _eps_for(dt: object) -> float:
  if dt == float32:
    return 1.1920928955078125e-07
  return 2.220446049250313e-16


def _floatify(X: ndarray) -> ndarray:
  # LAPACK paths only handle float32/float64; promote ints/bools.
  if X.dtype in (float32, float64):
    return X
  return X.astype(float64)


# LAPACK-backed decompositions.
@typing.overload
def qr(a: object, mode: typing.Literal["r"]) -> ndarray: ...
@typing.overload
def qr(a: object, mode: typing.Literal["reduced", "complete"] = "reduced") -> tuple[ndarray, ndarray]: ...
@typing.overload
def qr(a: object, mode: str = "reduced") -> _QRResult: ...
def qr(a: object, mode: str = "reduced") -> _QRResult:
  if mode not in _QR_MODES:
    raise ValueError(f"qr: unknown mode {mode!r}")
  X = _floatify(asarray(a))
  if X.ndim != 2:
    raise ValueError("qr: input must be rank-2")
  out = _w(_native.linalg_qr, X._native, _QR_MODES[mode])
  if mode == "r":
    return ndarray(out)
  q = ndarray(out[0])
  r = ndarray(out[1])
  return q, r


def cholesky(a: object) -> ndarray:
  X = _floatify(asarray(a))
  if X.ndim != 2 or X.shape[0] != X.shape[1]:
    raise ValueError("cholesky: input must be square rank-2")
  return ndarray(_w(_native.linalg_cholesky, X._native))


def eigh(a: object, UPLO: str = "L") -> tuple[ndarray, ndarray]:
  if UPLO not in ("L", "U"):
    raise ValueError(f"eigh: invalid UPLO {UPLO!r}")
  X = _floatify(asarray(a))
  if X.ndim != 2 or X.shape[0] != X.shape[1]:
    raise ValueError("eigh: input must be square rank-2")
  if UPLO == "U":
    X = matrix_transpose(X)
  out = _w(_native.linalg_eigh, X._native, True)
  return ndarray(out[0]), ndarray(out[1])


def eigvalsh(a: object, UPLO: str = "L") -> ndarray:
  if UPLO not in ("L", "U"):
    raise ValueError(f"eigvalsh: invalid UPLO {UPLO!r}")
  X = _floatify(asarray(a))
  if X.ndim != 2 or X.shape[0] != X.shape[1]:
    raise ValueError("eigvalsh: input must be square rank-2")
  if UPLO == "U":
    X = matrix_transpose(X)
  out = _w(_native.linalg_eigh, X._native, False)
  return ndarray(out[0])


def eig(a: object) -> tuple[ndarray, ndarray]:
  from . import complex128

  X = _floatify(asarray(a))
  if X.ndim != 2 or X.shape[0] != X.shape[1]:
    raise ValueError("eig: input must be square rank-2")
  out = _w(_native.linalg_eig, X._native, True)
  wr, wi, vr, all_real = ndarray(out[0]), ndarray(out[1]), ndarray(out[2]), bool(out[3])
  if all_real:
    return wr, vr
  # Build complex eigenvalues + eigenvectors. LAPACK packs conjugate
  # pairs as consecutive entries in WR/WI: real eigvals have wi=0,
  # complex pairs come as (a+bi, a-bi). Eigenvectors mirror: real cols
  # are the eigenvectors as-is; complex pairs use vr[:,j] + i*vr[:,j+1]
  # for the +bi eigenvalue and the conjugate for -bi.
  n = X.shape[0]
  wr_flat, wi_flat, w_complex = _eig_parts(wr, wi, n)
  w_arr = ndarray(_native.from_flat(w_complex, (n,), complex128.code))

  def vr_value(row: int, col: int) -> float:
    return float(vr._native.get_scalar(row * n + col))

  v_complex = []
  j = 0
  while j < n:
    if wi_flat[j] == 0.0:
      for i in range(n):
        v_complex.append(complex(vr_value(i, j), 0.0))
      j += 1
    else:
      # Pair: column j → +i*v[:,j+1], column j+1 → -i*v[:,j+1]
      for i in range(n):
        v_complex.append(complex(vr_value(i, j), vr_value(i, j + 1)))
      for i in range(n):
        v_complex.append(complex(vr_value(i, j), -vr_value(i, j + 1)))
      j += 2
  # v_complex is column-major (column-by-column appended); reshape needs row-major.
  v_row_major = [v_complex[c * n + i] for i in range(n) for c in range(n)]
  v_arr = ndarray(_native.from_flat(v_row_major, (n, n), complex128.code))
  return w_arr, v_arr


def eigvals(a: object) -> ndarray:
  from . import complex128

  X = _floatify(asarray(a))
  if X.ndim != 2 or X.shape[0] != X.shape[1]:
    raise ValueError("eigvals: input must be square rank-2")
  out = _w(_native.linalg_eig, X._native, False)
  wr, wi, _, all_real = ndarray(out[0]), ndarray(out[1]), ndarray(out[2]), bool(out[3])
  if all_real:
    return wr
  n = X.shape[0]
  _, _, w_complex = _eig_parts(wr, wi, n)
  return ndarray(_native.from_flat(w_complex, (n,), complex128.code))


@typing.overload
def svd(
  a: object, full_matrices: bool = True, compute_uv: typing.Literal[True] = True, hermitian: bool = False
) -> tuple[ndarray, ndarray, ndarray]: ...
@typing.overload
def svd(
  a: object, full_matrices: bool = True, compute_uv: typing.Literal[False] = False, hermitian: bool = False
) -> ndarray: ...
@typing.overload
def svd(a: object, full_matrices: bool = True, compute_uv: bool = True, hermitian: bool = False) -> _SVDResult: ...
def svd(a: object, full_matrices: bool = True, compute_uv: bool = True, hermitian: bool = False) -> _SVDResult:
  del hermitian
  X = _floatify(asarray(a))
  if X.ndim != 2:
    raise ValueError("svd: input must be rank-2")
  out = _w(_native.linalg_svd, X._native, bool(full_matrices), bool(compute_uv))
  u, s, vt = ndarray(out[0]), ndarray(out[1]), ndarray(out[2])
  if not compute_uv:
    return s
  return u, s, vt


def svdvals(a: object) -> ndarray:
  return svd(a, compute_uv=False)


def lstsq(a: object, b: object, rcond: object = None) -> tuple[ndarray, ndarray, int, ndarray]:
  A = _floatify(asarray(a))
  B = _floatify(asarray(b))
  if A.ndim != 2:
    raise ValueError("lstsq: a must be rank-2")
  if B.ndim not in (1, 2):
    raise ValueError("lstsq: b must be rank-1 or rank-2")
  if A.shape[0] != B.shape[0]:
    raise ValueError("lstsq: shape mismatch on first axis")
  # numpy 2.x default rcond convention: machine epsilon * max(M, N).
  rcond_value = _eps_for(A.dtype) * max(A.shape) if rcond is None else float(typing.cast(typing.Any, rcond))
  out = _w(_native.linalg_lstsq, A._native, B._native, rcond_value)
  x = ndarray(out[0])
  s = ndarray(out[1])
  rank = int(out[2])
  # numpy returns (x, residuals, rank, s). Residuals are size-(NRHS,) for
  # rank-deficient overdetermined; size-0 otherwise. v1: compute on the
  # python side from x and the original A,B.
  m, n = A.shape
  if rank == n and m > n and B.ndim == 1:
    diff = typing.cast(ndarray, matmul(A, x)) - B
    residuals = typing.cast(ndarray, _sum(square(diff), keepdims=True))
  elif rank == n and m > n and B.ndim == 2:
    diff = typing.cast(ndarray, matmul(A, x)) - B
    residuals = typing.cast(ndarray, _sum(square(diff), axis=0))
  else:
    from . import empty as _empty

    residuals = _empty((0,), dtype=A.dtype)
  return x, residuals, rank, s


def pinv(a: object, rcond: object = None, hermitian: bool = False) -> ndarray:
  del hermitian
  A = _floatify(asarray(a))
  if A.ndim != 2:
    raise ValueError("pinv: input must be rank-2")
  m, n = A.shape
  rcond_value = _eps_for(A.dtype) * max(m, n) if rcond is None else float(typing.cast(typing.Any, rcond))
  return ndarray(_w(_native.linalg_pinv, A._native, rcond_value))


__all__ = [
  "LinAlgError",
  "cholesky",
  "cross",
  "det",
  "dot",
  "eig",
  "eigh",
  "eigvals",
  "eigvalsh",
  "inner",
  "inv",
  "kron",
  "lstsq",
  "matmul",
  "matrix_norm",
  "matrix_power",
  "matrix_rank",
  "matrix_transpose",
  "matvec",
  "multi_dot",
  "norm",
  "outer",
  "pinv",
  "qr",
  "slogdet",
  "solve",
  "svd",
  "svdvals",
  "tensordot",
  "tensorinv",
  "tensorsolve",
  "trace",
  "vdot",
  "vecdot",
  "vecmat",
  "vector_norm",
]
