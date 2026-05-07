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
  X=_floatify(asarray(M))
  if X.ndim<2:return 1 if any(X._native.get_scalar(i)!=0 for i in range(X.size)) else 0
  s=svd(X,compute_uv=False)
  smax=max(float(s._native.get_scalar(i)) for i in range(s.size))
  if tol is None:tol=smax*max(X.shape[-2:])*_eps_for(X.dtype)
  return sum(1 for i in range(s.size) if float(s._native.get_scalar(i))>tol)

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

def tensorinv(a,ind=2):
  # numpy.linalg.tensorinv: reshape A as a (prod(in_shape), prod(out_shape))
  # matrix where in_shape = a.shape[ind:], out_shape = a.shape[:ind].
  # Solve for B such that A @ B = I, then reshape B to (in_shape + out_shape).
  A=asarray(a)
  if A.ndim<ind:raise ValueError("tensorinv: ind must be < a.ndim")
  out_shape=A.shape[:ind];in_shape=A.shape[ind:]
  out_size=1
  for d in out_shape:out_size*=d
  in_size=1
  for d in in_shape:in_size*=d
  if out_size!=in_size:raise ValueError("tensorinv: outer / inner volumes must match")
  flat=A.reshape((out_size,in_size))
  inverse=inv(flat)
  return inverse.reshape(in_shape+out_shape)

def tensorsolve(a,b,axes=None):
  # numpy.linalg.tensorsolve: like solve but with general-rank tensors.
  # Layout: a.shape == b.shape + x.shape (the leading axes of a match
  # b's full shape; the trailing axes form x's shape). After moving the
  # contracted axes to the end (when `axes` is given), reshape into a
  # square (prod(b.shape), prod(x.shape)) matrix and call solve.
  A=asarray(a);B=asarray(b)
  if axes is not None:
    # Move the listed axes to the trailing end (they form x's shape).
    a_axes=list(range(A.ndim))
    for ax in sorted(axes,reverse=True):a_axes.pop(ax)
    a_axes+=list(axes)
    A=transpose(A,tuple(a_axes))
  prod_b=1
  for d in B.shape:prod_b*=d
  # Leading axes of A correspond to b's shape; trailing axes are x's shape.
  if A.ndim<B.ndim or A.shape[:B.ndim]!=B.shape:
    raise ValueError("tensorsolve: leading axes of a must match b.shape")
  x_shape=A.shape[B.ndim:]
  prod_x=1
  for d in x_shape:prod_x*=d
  if prod_b!=prod_x:raise ValueError("tensorsolve: square reshape required")
  flat_a=A.reshape((prod_b,prod_x))
  flat_b=ravel(B)
  x=solve(flat_a,flat_b)
  return x.reshape(x_shape)


def einsum(subscripts,*operands,**kwargs):
  # Einstein summation: parse subscripts, build a contraction plan, walk it.
  # v1: supports the (input1,input2,...->output) form with letter labels.
  # Implementation strategy: reduce the sequence of operands pairwise via
  # `_einsum_pair_contract`, building an output indexing that matches the
  # spec. Equivalent to calling `tensordot` along the matched axes.
  if "out" in kwargs:raise NotImplementedError("einsum: out= not supported in v1")
  if "optimize" in kwargs:pass  # accepted, ignored (we always do pairwise)
  if "->" in subscripts:
    in_subs,out_sub=subscripts.split("->")
  else:
    in_subs,out_sub=subscripts,None
  in_subs=in_subs.split(",")
  if len(in_subs)!=len(operands):raise ValueError("einsum: subscript / operand count mismatch")
  arrs=[asarray(op) for op in operands]
  for sub,arr in builtins.zip(in_subs,arrs,strict=True):
    if len(sub)!=arr.ndim:raise ValueError(f"einsum: subscript '{sub}' does not match operand ndim {arr.ndim}")
  # Auto-derive output: every label that appears exactly once across inputs.
  if out_sub is None:
    counts={}
    for sub in in_subs:
      for ch in sub:counts[ch]=counts.get(ch,0)+1
    out_sub="".join(sorted([ch for ch,c in counts.items() if c==1]))
  # Contract operands left-to-right.
  cur_sub,cur=in_subs[0],arrs[0]
  # Handle internal traces (repeated labels in a single operand) early.
  cur_sub,cur=_einsum_trace_diag(cur_sub,cur)
  for nxt_sub,nxt in builtins.zip(in_subs[1:],arrs[1:]):
    nxt_sub,nxt=_einsum_trace_diag(nxt_sub,nxt)
    cur_sub,cur=_einsum_pair_contract(cur_sub,cur,nxt_sub,nxt)
  # Final reduction: sum out any labels not in the output, then permute.
  return _einsum_finalise(cur_sub,cur,out_sub)


def _einsum_trace_diag(sub,arr):
  # Collapse repeated labels in a single operand by taking the diagonal.
  while True:
    seen={}
    repeated=None
    for i,ch in enumerate(sub):
      if ch in seen:repeated=(seen[ch],i,ch);break
      seen[ch]=i
    if repeated is None:return sub,arr
    a,b,ch=repeated
    arr=diagonal(arr,offset=0,axis1=a,axis2=b)  # diagonal moves to the end
    new_sub=[c for i,c in enumerate(sub) if i!=a and i!=b]+[ch]
    sub="".join(new_sub)


def _einsum_pair_contract(la,a,lb,b):
  # Pair contract two arrays based on their label strings.
  # Strategy: matched labels (in both) become contraction axes; unique
  # labels become free axes; the result label string is la_unique + lb_unique.
  shared=[ch for ch in la if ch in lb]
  la_unique=[ch for ch in la if ch not in shared]
  lb_unique=[ch for ch in lb if ch not in shared]
  a_axes=tuple(la.index(ch) for ch in shared)
  b_axes=tuple(lb.index(ch) for ch in shared)
  out=tensordot(a,b,axes=(a_axes,b_axes))
  return "".join(la_unique+lb_unique),out


def _einsum_finalise(sub,arr,out_sub):
  # Sum out labels not in out_sub, then permute the remaining axes
  # to match out_sub's order.
  to_sum=[i for i,ch in enumerate(sub) if ch not in out_sub]
  for ax in sorted(to_sum,reverse=True):
    arr=_sum(arr,axis=ax)
    sub=sub[:ax]+sub[ax+1:]
  if not sub and not out_sub:return arr
  perm=[sub.index(ch) for ch in out_sub]
  if perm!=list(range(len(perm))):arr=transpose(arr,tuple(perm))
  return arr


import builtins  # used by einsum (zip, etc.)

# Helpers for the LAPACK-backed paths below.
_QR_MODES={"reduced":0,"complete":1,"r":2}
def _eps_for(dt):
  if dt==float32:return 1.1920928955078125e-07
  return 2.220446049250313e-16
def _floatify(X):
  # LAPACK paths only handle float32/float64; promote ints/bools.
  if X.dtype in(float32,float64):return X
  return X.astype(float64)

# LAPACK-backed decompositions.
def qr(a,mode="reduced"):
  if mode not in _QR_MODES:raise ValueError(f"qr: unknown mode {mode!r}")
  X=_floatify(asarray(a))
  if X.ndim!=2:raise ValueError("qr: input must be rank-2")
  out=_w(_native.linalg_qr,X._native,_QR_MODES[mode])
  if mode=="r":return ndarray(out)
  q=ndarray(out[0]);r=ndarray(out[1])
  return q,r

def cholesky(a):
  X=_floatify(asarray(a))
  if X.ndim!=2 or X.shape[0]!=X.shape[1]:raise ValueError("cholesky: input must be square rank-2")
  return ndarray(_w(_native.linalg_cholesky,X._native))

def eigh(a,UPLO='L'):
  if UPLO not in('L','U'):raise ValueError(f"eigh: invalid UPLO {UPLO!r}")
  X=_floatify(asarray(a))
  if X.ndim!=2 or X.shape[0]!=X.shape[1]:raise ValueError("eigh: input must be square rank-2")
  if UPLO=='U':X=matrix_transpose(X)
  out=_w(_native.linalg_eigh,X._native,True)
  return ndarray(out[0]),ndarray(out[1])

def eigvalsh(a,UPLO='L'):
  if UPLO not in('L','U'):raise ValueError(f"eigvalsh: invalid UPLO {UPLO!r}")
  X=_floatify(asarray(a))
  if X.ndim!=2 or X.shape[0]!=X.shape[1]:raise ValueError("eigvalsh: input must be square rank-2")
  if UPLO=='U':X=matrix_transpose(X)
  out=_w(_native.linalg_eigh,X._native,False)
  return ndarray(out[0])

def eig(a):
  from . import complex128
  X=_floatify(asarray(a))
  if X.ndim!=2 or X.shape[0]!=X.shape[1]:raise ValueError("eig: input must be square rank-2")
  out=_w(_native.linalg_eig,X._native,True)
  wr,wi,vr,all_real=ndarray(out[0]),ndarray(out[1]),ndarray(out[2]),bool(out[3])
  if all_real:return wr,vr
  # Build complex eigenvalues + eigenvectors. LAPACK packs conjugate
  # pairs as consecutive entries in WR/WI: real eigvals have wi=0,
  # complex pairs come as (a+bi, a-bi). Eigenvectors mirror: real cols
  # are the eigenvectors as-is; complex pairs use vr[:,j] + i*vr[:,j+1]
  # for the +bi eigenvalue and the conjugate for -bi.
  n=X.shape[0]
  wr_flat=[float(wr._native.get_scalar(i)) for i in range(n)]
  wi_flat=[float(wi._native.get_scalar(i)) for i in range(n)]
  w_complex=[complex(wr_flat[i],wi_flat[i]) for i in range(n)]
  w_arr=ndarray(_native.from_flat(w_complex,(n,),complex128.code))
  vr_dense=numpy_asarray_local(vr)
  v_complex=[]
  j=0
  while j<n:
    if wi_flat[j]==0.0:
      for i in range(n):v_complex.append(complex(vr_dense[i,j],0.0))
      j+=1
    else:
      # Pair: column j → +i*v[:,j+1], column j+1 → -i*v[:,j+1]
      for i in range(n):v_complex.append(complex(vr_dense[i,j],vr_dense[i,j+1]))
      for i in range(n):v_complex.append(complex(vr_dense[i,j],-vr_dense[i,j+1]))
      j+=2
  # v_complex is column-major (column-by-column appended); reshape needs row-major.
  v_row_major=[]
  for i in range(n):
    for c in range(n):
      v_row_major.append(v_complex[c*n+i])
  v_arr=ndarray(_native.from_flat(v_row_major,(n,n),complex128.code))
  return w_arr,v_arr

def numpy_asarray_local(monpy_arr):
  # Lazy local helper to avoid a top-level numpy import at module load.
  import numpy
  return numpy.asarray(monpy_arr)

def eigvals(a):
  from . import complex128
  X=_floatify(asarray(a))
  if X.ndim!=2 or X.shape[0]!=X.shape[1]:raise ValueError("eigvals: input must be square rank-2")
  out=_w(_native.linalg_eig,X._native,False)
  wr,wi,_,all_real=ndarray(out[0]),ndarray(out[1]),ndarray(out[2]),bool(out[3])
  if all_real:return wr
  n=X.shape[0]
  wr_flat=[float(wr._native.get_scalar(i)) for i in range(n)]
  wi_flat=[float(wi._native.get_scalar(i)) for i in range(n)]
  w_complex=[complex(wr_flat[i],wi_flat[i]) for i in range(n)]
  return ndarray(_native.from_flat(w_complex,(n,),complex128.code))

def svd(a,full_matrices=True,compute_uv=True,hermitian=False):
  X=_floatify(asarray(a))
  if X.ndim!=2:raise ValueError("svd: input must be rank-2")
  out=_w(_native.linalg_svd,X._native,bool(full_matrices),bool(compute_uv))
  u,s,vt=ndarray(out[0]),ndarray(out[1]),ndarray(out[2])
  if not compute_uv:return s
  return u,s,vt

def svdvals(a):return svd(a,compute_uv=False)

def lstsq(a,b,rcond=None):
  A=_floatify(asarray(a));B=_floatify(asarray(b))
  if A.ndim!=2:raise ValueError("lstsq: a must be rank-2")
  if B.ndim not in(1,2):raise ValueError("lstsq: b must be rank-1 or rank-2")
  if A.shape[0]!=B.shape[0]:raise ValueError("lstsq: shape mismatch on first axis")
  # numpy 2.x default rcond convention: machine epsilon * max(M, N).
  if rcond is None:
    rcond=_eps_for(A.dtype)*max(A.shape)
  out=_w(_native.linalg_lstsq,A._native,B._native,float(rcond))
  x=ndarray(out[0]);s=ndarray(out[1]);rank=int(out[2])
  # numpy returns (x, residuals, rank, s). Residuals are size-(NRHS,) for
  # rank-deficient overdetermined; size-0 otherwise. v1: compute on the
  # python side from x and the original A,B.
  m,n=A.shape
  if rank==n and m>n and B.ndim==1:
    diff=matmul(A,x)-B
    residuals=_sum(square(diff),keepdims=True)
  elif rank==n and m>n and B.ndim==2:
    diff=matmul(A,x)-B
    residuals=_sum(square(diff),axis=0)
  else:
    from . import empty as _empty
    residuals=_empty((0,),dtype=A.dtype)
  return x,residuals,rank,s

def pinv(a,rcond=None,hermitian=False):
  A=_floatify(asarray(a))
  if A.ndim!=2:raise ValueError("pinv: input must be rank-2")
  m,n=A.shape
  u,s,vt=svd(A,full_matrices=False,compute_uv=True)
  smax=max(float(s._native.get_scalar(i)) for i in range(s.size)) if s.size>0 else 0.0
  if rcond is None:rcond=_eps_for(A.dtype)*max(m,n)
  cutoff=rcond*smax
  # Build s_inv: 1/s_i where s_i > cutoff, else 0.
  from . import zeros as _zeros
  s_inv=_zeros(s.shape,dtype=s.dtype)
  for i in range(s.size):
    val=float(s._native.get_scalar(i))
    if val>cutoff:_native.fill(s_inv[i:i+1]._native,1.0/val)
  # pinv = V * diag(s_inv) * U.T
  V=matrix_transpose(vt)
  Ut=matrix_transpose(u)
  # multiply rows of Ut by s_inv (broadcast along axis 0).
  scaled=multiply(reshape(s_inv,(s_inv.shape[0],1)),Ut)
  return matmul(V,scaled)

__all__=["LinAlgError","cholesky","cross","det","dot","eig","eigh","eigvals","eigvalsh","inner","inv","kron","lstsq","matmul","matrix_norm","matrix_power","matrix_rank","matrix_transpose","matvec","multi_dot","norm","outer","pinv","qr","slogdet","solve","svd","svdvals","tensordot","tensorinv","tensorsolve","trace","vdot","vecdot","vecmat","vector_norm"]
