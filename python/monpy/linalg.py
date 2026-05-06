# fmt: off
# ruff: noqa
from __future__ import annotations
from . import _native,asarray,matmul,matrix_transpose,ndarray

class LinAlgError(ValueError):pass

def _w(fn,*a):
  try:return fn(*a)
  except Exception as exc:raise LinAlgError(str(exc)) from exc

def solve(a,b):l,r=asarray(a),asarray(b);return ndarray(_w(_native.linalg_solve,l._native,r._native))
def inv(a):x=asarray(a);return ndarray(_w(_native.linalg_inv,x._native))
def det(a):x=asarray(a);return ndarray(_w(_native.linalg_det,x._native))._scalar()

__all__=["LinAlgError","det","inv","matmul","matrix_transpose","solve"]
