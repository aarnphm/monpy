from __future__ import annotations

from . import _native, asarray, matmul, matrix_transpose, ndarray


class LinAlgError(ValueError):
  """Raised when a supported dense linear algebra operation cannot complete."""


def solve(a: object, b: object) -> ndarray:
  lhs = asarray(a)
  rhs = asarray(b)
  try:
    return ndarray(_native.linalg_solve(lhs._native, rhs._native))
  except Exception as exc:
    raise LinAlgError(str(exc)) from exc


def inv(a: object) -> ndarray:
  arr = asarray(a)
  try:
    return ndarray(_native.linalg_inv(arr._native))
  except Exception as exc:
    raise LinAlgError(str(exc)) from exc


def det(a: object) -> object:
  arr = asarray(a)
  try:
    result = ndarray(_native.linalg_det(arr._native))
  except Exception as exc:
    raise LinAlgError(str(exc)) from exc
  return result._scalar()


__all__ = [
  "LinAlgError",
  "det",
  "inv",
  "matmul",
  "matrix_transpose",
  "solve",
]
