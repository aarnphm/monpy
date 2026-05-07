# fmt: off
# ruff: noqa
from __future__ import annotations
import math as _math
from . import (
  _native,
  asarray,
  ascontiguousarray,
  cumsum,
  diagonal,
  expand_dims,
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
  squeeze,
  sum as _sum,
  transpose,
  zeros,
)

class LinAlgError(ValueError):pass

def _w(fn,*a):
  try:return fn(*a)
  except Exception as exc:raise LinAlgError(str(exc)) from exc

# Existing native-backed primitives.
def solve(a,b):l,r=asarray(a),asarray(b);return ndarray(_w(_native.linalg_solve,l._native,r._native))
def inv(a):x=asarray(a);return ndarray(_w(_native.linalg_inv,x._native))
def det(a):x=asarray(a);return ndarray(_w(_native.linalg_det,x._native))._scalar()

# Phase 6d additions on top of matmul + reductions. No LAPACK bindings yet;
# the algorithm-heavy decomps (qr/svd/eig/cholesky/lstsq/pinv) raise
# NotImplementedError until the accelerate layer wires them up.
def dot(a,b):
  A=asarray(a);B=asarray(b)
  if A.ndim==0 or B.ndim==0:return multiply(A,B)
  if A.ndim==1 and B.ndim==1:
    if A.shape[0]!=B.shape[0]:raise ValueError("dot: shape mismatch")
    return _sum(multiply(A,B))
  if A.ndim<=2 and B.ndim<=2:return matmul(A,B)
  raise NotImplementedError("dot: ndim>2 not implemented")

def vdot(a,b):
  A=ravel(asarray(a));B=ravel(asarray(b))
  if A.shape!=B.shape:raise ValueError("vdot: shape mismatch")
  return _sum(multiply(A,B))

def inner(a,b):
  A=asarray(a);B=asarray(b)
  if A.ndim==0 or B.ndim==0:return multiply(A,B)
  if A.shape[-1]!=B.shape[-1]:raise ValueError("inner: trailing axis mismatch")
  if A.ndim==1 and B.ndim==1:return _sum(multiply(A,B))
  if A.ndim==1:return _sum(multiply(A,B),axis=-1)
  if B.ndim==1:return _sum(multiply(A,B),axis=-1)
  raise NotImplementedError("inner: only 1D x 1D / 1D x N / N x 1D supported in v1")

def outer(a,b):
  A=ravel(asarray(a));B=ravel(asarray(b))
  return matmul(reshape(A,(A.shape[0],1)),reshape(B,(1,B.shape[0])))

def tensordot(a,b,axes=2):
  A=asarray(a);B=asarray(b)
  if isinstance(axes,int):
    n=axes
    a_axes=tuple(range(A.ndim-n,A.ndim))
    b_axes=tuple(range(n))
  else:
    a_axes,b_axes=axes
    a_axes=tuple(a_axes) if hasattr(a_axes,'__iter__') else (a_axes,)
    b_axes=tuple(b_axes) if hasattr(b_axes,'__iter__') else (b_axes,)
  for ax_a,ax_b in zip(a_axes,b_axes,strict=True):
    if A.shape[ax_a]!=B.shape[ax_b]:raise ValueError("tensordot: axis size mismatch")
  a_kept=tuple(d for d in range(A.ndim) if d not in a_axes)
  b_kept=tuple(d for d in range(B.ndim) if d not in b_axes)
  A_perm=transpose(A,tuple(a_kept)+a_axes)
  B_perm=transpose(B,b_axes+b_kept)
  a_kept_size=1
  for d in a_kept:a_kept_size*=A.shape[d]
  contract_size=1
  for d in a_axes:contract_size*=A.shape[d]
  b_kept_size=1
  for d in b_kept:b_kept_size*=B.shape[d]
  A_mat=ascontiguousarray(A_perm).reshape((a_kept_size,contract_size))
  B_mat=ascontiguousarray(B_perm).reshape((contract_size,b_kept_size))
  out=matmul(A_mat,B_mat)
  out_shape=tuple(A.shape[d] for d in a_kept)+tuple(B.shape[d] for d in b_kept)
  return reshape(out,out_shape) if out_shape else _sum(out)

def kron(a,b):
  A=asarray(a);B=asarray(b)
  while A.ndim<B.ndim:A=expand_dims(A,0)
  while B.ndim<A.ndim:B=expand_dims(B,0)
  out_shape=tuple(s_a*s_b for s_a,s_b in zip(A.shape,B.shape,strict=True))
  total=1
  for d in out_shape:total*=d
  flat_a=[A._native.get_scalar(i) for i in range(A.size)]
  flat_b=[B._native.get_scalar(i) for i in range(B.size)]
  in_a_strides=[1]*A.ndim
  for d in range(A.ndim-2,-1,-1):in_a_strides[d]=in_a_strides[d+1]*A.shape[d+1]
  in_b_strides=[1]*B.ndim
  for d in range(B.ndim-2,-1,-1):in_b_strides[d]=in_b_strides[d+1]*B.shape[d+1]
  out_strides=[1]*A.ndim
  for d in range(A.ndim-2,-1,-1):out_strides[d]=out_strides[d+1]*out_shape[d+1]
  out=[]
  for i in range(total):
    rem=i;a_off=0;b_off=0
    for d in range(A.ndim):
      coord=rem//out_strides[d];rem-=coord*out_strides[d]
      a_off+=(coord//B.shape[d])*in_a_strides[d]
      b_off+=(coord%B.shape[d])*in_b_strides[d]
    out.append(flat_a[a_off]*flat_b[b_off])
  t=result_type(A,B)
  return ndarray(_native.from_flat(out,out_shape,t.code))

def cross(a,b,axisa=-1,axisb=-1,axisc=-1,axis=None):
  if axis is not None:axisa=axisb=axisc=axis
  A=asarray(a);B=asarray(b)
  if A.shape[axisa]!=3 or B.shape[axisb]!=3:raise ValueError("cross: only 3D vectors supported")
  if A.ndim!=1 or B.ndim!=1:raise NotImplementedError("cross: ndim>1 not implemented in v1")
  ax,ay,az=A._native.get_scalar(0),A._native.get_scalar(1),A._native.get_scalar(2)
  bx,by,bz=B._native.get_scalar(0),B._native.get_scalar(1),B._native.get_scalar(2)
  cx=ay*bz-az*by;cy=az*bx-ax*bz;cz=ax*by-ay*bx
  t=result_type(A,B)
  return ndarray(_native.from_flat([cx,cy,cz],(3,),t.code))

def matvec(a,b):
  A=asarray(a);B=asarray(b)
  if A.ndim<2 or B.ndim<1:raise ValueError("matvec: a must be (...,M,N), b must be (...,N)")
  return matmul(A,reshape(B,B.shape+(1,))).reshape(A.shape[:-1])

def vecmat(a,b):
  A=asarray(a);B=asarray(b)
  return matmul(reshape(A,A.shape[:-1]+(1,A.shape[-1])),B).reshape(A.shape[:-1]+B.shape[-1:])

def vecdot(a,b,axis=-1):
  A=asarray(a);B=asarray(b)
  return _sum(multiply(A,B),axis=axis)

def trace(a,offset=0,axis1=0,axis2=1,dtype=None):
  d=diagonal(a,offset=offset,axis1=axis1,axis2=axis2)
  return _sum(d,dtype=dtype) if d.ndim==1 else _sum(d,axis=-1,dtype=dtype)

def norm(x,ord=None,axis=None,keepdims=False):
  X=asarray(x)
  if axis is None and X.ndim==1:axis=0
  if axis is None and X.ndim==2 and ord is None:
    return sqrt(_sum(square(X)))
  if isinstance(axis,int):
    if ord is None or ord==2:return sqrt(_sum(square(X),axis=axis,keepdims=keepdims))
    if ord==1:
      from . import absolute as _abs
      return _sum(_abs(X),axis=axis,keepdims=keepdims)
    if ord==_math.inf:
      from . import absolute as _abs
      return _abs(X).max(axis=axis) if not keepdims else _abs(X).max(axis=axis)
    if ord==-_math.inf:
      from . import absolute as _abs
      return _abs(X).min(axis=axis) if not keepdims else _abs(X).min(axis=axis)
    if ord==0:
      from . import not_equal,sum as ne_sum
      return ne_sum(not_equal(X,0),axis=axis,keepdims=keepdims)
    return _w_pow(X,ord,axis,keepdims)
  if axis is None and X.ndim==1:return sqrt(_sum(square(X)))
  raise NotImplementedError("norm: matrix norms (axis=tuple) not implemented in v1")

def _w_pow(X,p,axis,keepdims):
  from . import absolute as _abs,power as _pow
  return _pow(_sum(_pow(_abs(X),p),axis=axis,keepdims=keepdims),1.0/p)

def vector_norm(x,axis=None,keepdims=False,ord=2):return norm(x,ord=ord,axis=-1 if axis is None else axis,keepdims=keepdims)
def matrix_norm(x,axis=(-2,-1),keepdims=False,ord=None):
  if ord is not None:raise NotImplementedError("matrix_norm with ord != None not implemented in v1")
  X=asarray(x)
  return sqrt(_sum(square(X),axis=axis,keepdims=keepdims))

def matrix_rank(M,tol=None,hermitian=False):
  raise NotImplementedError("matrix_rank requires SVD; deferred to LAPACK binding work")

def matrix_power(a,n):
  A=asarray(a)
  if A.ndim<2 or A.shape[-1]!=A.shape[-2]:raise ValueError("matrix_power: a must be square")
  if n==0:
    eye_shape=A.shape
    out=zeros(eye_shape,dtype=A.dtype)
    for i in range(A.shape[-1]):out[...,i,i]=1
    return out
  if n<0:return matrix_power(inv(A),-n)
  result=A
  for _ in range(n-1):result=matmul(result,A)
  return result

def slogdet(a):
  A=asarray(a)
  d=det(A)
  if d==0:return 0.0,float('-inf')
  return (1.0 if d>0 else -1.0),_math.log(abs(d))

def multi_dot(arrays):
  arrs=[asarray(a) for a in arrays]
  if len(arrs)<2:raise ValueError("multi_dot: need at least two arrays")
  out=arrs[0]
  for a in arrs[1:]:out=matmul(out,a)
  return out

def tensorinv(a,ind=2):raise NotImplementedError("tensorinv not implemented in v1")
def tensorsolve(a,b,axes=None):raise NotImplementedError("tensorsolve not implemented in v1")

# LAPACK-backed decompositions; deferred until accelerate.mojo wires them up.
def qr(a,mode="reduced"):raise NotImplementedError("qr requires LAPACK sgeqrf/sorgqr; deferred")
def cholesky(a):raise NotImplementedError("cholesky requires LAPACK spotrf; deferred")
def eig(a):raise NotImplementedError("eig requires LAPACK sgeev; deferred")
def eigh(a,UPLO='L'):raise NotImplementedError("eigh requires LAPACK ssyev; deferred")
def eigvals(a):raise NotImplementedError("eigvals requires LAPACK sgeev; deferred")
def eigvalsh(a,UPLO='L'):raise NotImplementedError("eigvalsh requires LAPACK ssyev; deferred")
def svd(a,full_matrices=True,compute_uv=True,hermitian=False):raise NotImplementedError("svd requires LAPACK sgesvd; deferred")
def svdvals(a):raise NotImplementedError("svdvals requires LAPACK sgesvd; deferred")
def lstsq(a,b,rcond=None):raise NotImplementedError("lstsq requires LAPACK sgelsd; deferred")
def pinv(a,rcond=None,hermitian=False):raise NotImplementedError("pinv requires SVD; deferred")

__all__=["LinAlgError","cholesky","cross","det","dot","eig","eigh","eigvals","eigvalsh","inner","inv","kron","lstsq","matmul","matrix_norm","matrix_power","matrix_rank","matrix_transpose","matvec","multi_dot","norm","outer","pinv","qr","slogdet","solve","svd","svdvals","tensordot","tensorinv","tensorsolve","trace","vdot","vecdot","vecmat","vector_norm"]
