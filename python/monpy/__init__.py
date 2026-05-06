# fmt: off # ruff: noqa
from __future__ import annotations
import builtins,importlib,itertools,math,typing
from collections.abc import Iterable,Sequence
from dataclasses import dataclass
from types import ModuleType,SimpleNamespace
from . import _native
if typing.TYPE_CHECKING: import numpy as np

# dtype codes mirror the Mojo side; identical layout there.
DTYPE_BOOL,DTYPE_INT64,DTYPE_FLOAT32,DTYPE_FLOAT64=range(4)
OP_ADD,OP_SUB,OP_MUL,OP_DIV=range(4)
UNARY_SIN,UNARY_COS,UNARY_EXP,UNARY_LOG=range(4)
REDUCE_SUM,REDUCE_MEAN,REDUCE_MIN,REDUCE_MAX,REDUCE_ARGMAX=range(5)

@dataclass(frozen=True,slots=True)
class DType:
  name:str;code:int;itemsize:int;typestr:str
  def __repr__(self)->str:return f"monpy.{self.name}"

bool=DType("bool",0,1,"|b1")
int64=DType("int64",1,8,"<i8")
float32=DType("float32",2,4,"<f4")
float64=DType("float64",3,8,"<f8")

_DT:tuple[DType,...]=(bool,int64,float32,float64)                                                                # ordered by .code; idx == code
_DTC:dict[int,DType]={d.code:d for d in _DT}                                                                     # code → DType
_DTN:dict[str,DType]={d.name:d for d in _DT}                                                                     # name → DType
_DTNK:dict[tuple[str,int],DType]={("b",1):bool,("i",8):int64,("f",4):float32,("f",8):float64}                    # (numpy kind, itemsize) → DType
_DTBT:dict[str,DType]={"|b1":bool}|{p+s:d for s,d in[("i8",int64),("f4",float32),("f8",float64)] for p in"<>="}  # numpy typestr (incl. byteorder prefix) → DType
_NPTYPE:np.ndarray|None=None                                                                                     # cached numpy.ndarray type (lazy import)

# 4×4 promotion tables flattened to a 16-char digit string, indexed (lhs.code*4 + rhs.code).
# Each char is the result DType's .code in the same _DT ordering. Numpy convention: int+float32→float64.
_TBL=lambda s:{(_DT[i//4],_DT[i%4]):_DT[int(c)] for i,c in enumerate(s)}                                         # decode digit-string → {(lhs,rhs): result}
_NPT:dict[tuple[DType,DType],DType]=_TBL("0123113323233333")                                                     # numeric (add/sub/mul) result codes
_DPT:dict[tuple[DType,DType],DType]=_TBL("3323333323233333")                                                     # division result codes (always promotes to float)
_BR:dict[int,dict[tuple[DType,DType],DType]]={OP_ADD:_NPT,OP_SUB:_NPT,OP_MUL:_NPT,OP_DIV:_DPT}                   # binary op → promotion table
_UR:dict[DType,DType]={bool:float64,int64:float64,float32:float32,float64:float64}                               # unary (sin/cos/exp/log) result: int kinds → f64
_CFE="unable to avoid copy while creating a monpy array as requested"                                            # error msg for copy=False refusal
_S1E="only size-1 arrays can be converted to python scalars"                                                     # error msg for non-size-1 → python scalar

newaxis=None;nan=math.nan;inf=math.inf;pi=math.pi;e=math.e


class ndarray:
  __array_priority__=1000
  __slots__=("_base","_dtype","_native","_owner","_shape","_strides")
  def __init__(_s,native:_native.Array,base:ndarray|None=None,*,dtype:DType|None=None,shape:tuple[int,...]|None=None,strides:tuple[int,...]|None=None,owner:object|None=None)->None:
    _s._native=native;_s._base=base;_s._dtype=dtype;_s._owner=owner;_s._shape=shape;_s._strides=strides
  @property
  def dtype(self)->DType:
    if self._dtype is None:self._dtype=_DTC[int(self._native.dtype_code())]
    return self._dtype
  @property
  def shape(self)->tuple[int,...]:
    if self._shape is None:n=int(self._native.ndim());self._shape=tuple(int(self._native.shape_at(a)) for a in range(n))
    return self._shape
  @property
  def ndim(self)->int:return len(self._shape) if self._shape is not None else int(self._native.ndim())
  @property
  def size(self)->int:return math.prod(self._shape) if self._shape is not None else int(self._native.size())
  @property
  def itemsize(self)->int:return self.dtype.itemsize
  @property
  def strides(self)->tuple[int,...]:
    if self._strides is None:i=self.itemsize;self._strides=tuple(int(self._native.stride_at(a))*i for a in range(self.ndim))  # native strides are in elements; we expose bytes
    return self._strides
  @property
  def device(self)->str:return"cpu"
  @property
  def T(self)->ndarray:
    n=len(self._shape) if self._shape is not None else int(self._native.ndim())
    return self if n<2 else ndarray(self._native.transpose_full_reverse_method(),base=self)
  @property
  def mT(self)->ndarray:
    if self.ndim<2:raise ValueError("matrix transpose requires at least two dimensions")
    return self.transpose(tuple(range(self.ndim-2))+(self.ndim-1,self.ndim-2))
  @property
  def __array_interface__(self)->dict[str,object]:
    s,st,i=self.shape,self.strides,self.itemsize
    return{"version":3,"shape":s,"typestr":self.dtype.typestr,"data":(int(self._native.data_address()),False),"strides":None if _is_c_contig(s,st,i) else st}
  def __array__(self,dtype:object=None,copy:builtins.bool|None=None)->object:
    class _O:
      def __init__(o,a:ndarray)->None:o._owner=a;o.__array_interface__=a.__array_interface__
    a=_npmod().asarray(_O(self))
    if dtype is not None:return a.astype(dtype,copy=copy is not False)
    return a.copy() if copy is True else a
  def __array_namespace__(self,*,api_version:str|None=None)->ModuleType:
    if api_version not in(None,"2024.12","2025.12"):raise ValueError(f"unsupported array api version: {api_version}")
    from . import array_api;return array_api
  def __dlpack__(self,*,stream:object=None,max_version:tuple[int,int]|None=None,dl_device:tuple[int,int]|None=None,copy:builtins.bool|None=None)->object:
    if stream is not None:raise BufferError("cpu dlpack export requires stream=None")
    if dl_device not in(None,(1,0)):raise BufferError("monpy only exports cpu dlpack tensors")
    return self.__array__(copy=False).__dlpack__(stream=None,max_version=max_version,dl_device=dl_device,copy=copy)
  def __dlpack_device__(self)->tuple[int,int]:return(1,0)
  def __len__(self)->int:
    if self.ndim==0:raise TypeError("len() of unsized object")
    return self.shape[0]
  def __iter__(self)->Iterable[object]:
    for i in range(len(self)):yield self[i]
  def __repr__(self)->str:return f"monpy.asarray({self.tolist()!r}, dtype={self.dtype!r})"
  def __getitem__(self,k:object)->object:
    if isinstance(k,slice) and self.ndim==1:                                                                                  # fast 1-D slice path: skip generic _view_for_key
      d=self.shape[0];step=1 if k.step is None else builtins.int(k.step)
      if step==0:raise ValueError("slice step cannot be zero")
      if k.start is None and k.stop is None:start=d-1 if step<0 else 0;stop=-1 if step<0 else d                                # whole-axis defaults; -1 stop tells native to walk past 0 with negative step
      else:start,stop,step=k.indices(d)
      return ndarray(self._native.slice_1d_method(start,stop,step),base=self,dtype=self.dtype,shape=(len(range(start,stop,step)),),strides=(self.strides[0]*step,))
    v=self._view_for_key(k)
    return v._scalar() if v.ndim==0 else v                                                                                    # scalar collapse for full-integer indexing
  def __setitem__(self,k:object,v:object)->None:
    view=self._view_for_key(k)
    if isinstance(v,ndarray):_native.copyto(view._native,v._native);return
    _native.fill(view._native,v)
  def __bool__(self)->builtins.bool:
    if self.size!=1:raise ValueError("the truth value of an array with more than one element is ambiguous")
    return builtins.bool(self._scalar())
  def __int__(self)->int:
    if self.size!=1:raise TypeError(_S1E)
    return builtins.int(self._scalar())
  def __float__(self)->float:
    if self.size!=1:raise TypeError(_S1E)
    return float(self._scalar())
  # binary fast paths: same-dtype same-shape ndarray×ndarray dispatches directly to the native method,
  # bypassing dtype lookup + coercion. We forward _dtype/_shape so chained ops stay on the fast path.
  def __add__(self,o:object)->object:
    if type(o) is ndarray and self._dtype is not None and self._dtype is o._dtype:
      r=ndarray(self._native.add(o._native));r._dtype=self._dtype;r._shape=self._shape;return r
    return _binary_from_array(self,o,OP_ADD,scalar_on_left=False)
  def __radd__(self,o:object)->object:return _binary_from_array(self,o,OP_ADD,scalar_on_left=True)
  def __sub__(self,o:object)->object:
    if type(o) is ndarray and self._dtype is not None and self._dtype is o._dtype:
      r=ndarray(self._native.sub(o._native));r._dtype=self._dtype;r._shape=self._shape;return r
    return _binary_from_array(self,o,OP_SUB,scalar_on_left=False)
  def __rsub__(self,o:object)->object:return _binary_from_array(self,o,OP_SUB,scalar_on_left=True)
  def __mul__(self,o:object)->object:
    if type(o) is ndarray and self._dtype is not None and self._dtype is o._dtype:
      r=ndarray(self._native.mul(o._native));r._dtype=self._dtype;r._shape=self._shape;return r
    return _binary_from_array(self,o,OP_MUL,scalar_on_left=False)
  def __rmul__(self,o:object)->object:return _binary_from_array(self,o,OP_MUL,scalar_on_left=True)
  def __truediv__(self,o:object)->object:
    if type(o) is ndarray and self._dtype is not None and self._dtype is o._dtype and self._dtype in (float32,float64):     # div fast path only valid for floats — int÷int promotes to f64
      r=ndarray(self._native.div(o._native));r._dtype=self._dtype;r._shape=self._shape;return r
    return _binary_from_array(self,o,OP_DIV,scalar_on_left=False)
  def __rtruediv__(self,o:object)->object:return _binary_from_array(self,o,OP_DIV,scalar_on_left=True)
  def __matmul__(self,o:object)->ndarray:
    if type(o) is ndarray and self._dtype is not None and self._dtype is o._dtype:
      r=ndarray(self._native.matmul(o._native));r._dtype=self._dtype;return r
    if isinstance(o,ndarray):l,r=_coerce(self,o,OP_MUL);return ndarray(_native.matmul(l._native,r._native))
    return matmul(self,o)
  def __rmatmul__(self,o:object)->ndarray:
    if type(o) is ndarray and self._dtype is not None and self._dtype is o._dtype:
      r=ndarray(o._native.matmul(self._native));r._dtype=self._dtype;return r
    if isinstance(o,ndarray):l,r=_coerce(o,self,OP_MUL);return ndarray(_native.matmul(l._native,r._native))
    return matmul(o,self)
  def __neg__(self)->ndarray:return multiply(self,-1)
  def __pos__(self)->ndarray:return self
  def reshape(self,*shape:int|Sequence[int])->ndarray:return ndarray(_native.reshape(self._native,_shape_args(shape)),base=self)
  def transpose(self,axes:Sequence[int]|None=None)->ndarray:
    if axes is None:axes=tuple(range(self.ndim-1,-1,-1))
    return ndarray(_native.transpose(self._native,_norm_axes(axes,self.ndim)),base=self)
  def astype(self,dtype:object,*,copy:builtins.bool=True,device:object=None)->ndarray:
    _check_cpu(device);t=_resolve_dtype(dtype)
    if t==self.dtype and not copy:return self
    return ndarray(_native.astype(self._native,t.code))
  def tolist(self)->object:return _unflat([self._native.get_scalar(i) for i in range(self.size)],self.shape)
  def sum(self,axis:object=None)->object:return sum(self,axis=axis)
  def mean(self,axis:object=None)->object:return mean(self,axis=axis)
  def min(self,axis:object=None)->object:return min(self,axis=axis)
  def max(self,axis:object=None)->object:return max(self,axis=axis)
  def argmax(self,axis:object=None)->object:return argmax(self,axis=axis)
  def _scalar(self)->object:
    if self.size!=1:raise TypeError(_S1E)
    return self._native.get_scalar(0)
  def _view_for_key(self,key:object)->ndarray:
    parts=_expand_key(key,self.ndim);starts:list[int]=[];stops:list[int]=[];steps:list[int]=[];drops:list[int]=[]            # drops[axis]==1 → collapse axis (integer index, not slice)
    for axis,part in enumerate(parts):
      d=self.shape[axis]
      if isinstance(part,slice):a,b,c=part.indices(d);starts.append(a);stops.append(b);steps.append(c);drops.append(0)
      else:i=_norm_idx(part,d);starts.append(i);stops.append(i+1);steps.append(1);drops.append(1)
    return ndarray(_native.slice(self._native,tuple(starts),tuple(stops),tuple(steps),tuple(drops)),base=self)


class _DeferredArray:
  # base for lazy expression nodes; subclass declares dtype/shape and overrides _compute().
  # any op needing raw bytes triggers _materialize(), which caches the realized ndarray.
  __array_priority__=1001
  __slots__=("_cached",)
  def __init__(self)->None:self._cached:ndarray|None=None
  @property
  def _native(self)->_native.Array:return self._materialize()._native
  @property
  def dtype(self)->DType:raise NotImplementedError
  @property
  def shape(self)->tuple[int,...]:raise NotImplementedError
  @property
  def ndim(self)->int:return len(self.shape)
  @property
  def size(self)->int:return math.prod(self.shape)
  @property
  def itemsize(self)->int:return self.dtype.itemsize
  @property
  def strides(self)->tuple[int,...]:return self._materialize().strides
  @property
  def device(self)->str:return"cpu"
  @property
  def T(self)->ndarray:return self._materialize().T
  @property
  def mT(self)->ndarray:return self._materialize().mT
  @property
  def __array_interface__(self)->dict[str,object]:return self._materialize().__array_interface__
  def __array__(self,dtype:object=None,copy:builtins.bool|None=None)->object:return self._materialize().__array__(dtype=dtype,copy=copy)
  def __array_namespace__(self,*,api_version:str|None=None)->ModuleType:return self._materialize().__array_namespace__(api_version=api_version)
  def __dlpack__(self,**k:object)->object:return self._materialize().__dlpack__(**k)
  def __dlpack_device__(self)->tuple[int,int]:return(1,0)
  def __len__(self)->int:return len(self._materialize())
  def __iter__(self)->Iterable[object]:return iter(self._materialize())
  def __repr__(self)->str:return repr(self._materialize())
  def __getitem__(self,k:object)->object:return self._materialize()[k]
  def __setitem__(self,k:object,v:object)->None:self._materialize()[k]=v
  def __bool__(self)->builtins.bool:return builtins.bool(self._materialize())
  def __int__(self)->int:return builtins.int(self._materialize())
  def __float__(self)->float:return builtins.float(self._materialize())
  def __add__(self,o:object)->object:return _binary(self,o,OP_ADD)
  def __radd__(self,o:object)->object:return _binary(o,self,OP_ADD)
  def __sub__(self,o:object)->object:return _binary(self,o,OP_SUB)
  def __rsub__(self,o:object)->object:return _binary(o,self,OP_SUB)
  def __mul__(self,o:object)->object:return _binary(self,o,OP_MUL)
  def __rmul__(self,o:object)->object:return _binary(o,self,OP_MUL)
  def __truediv__(self,o:object)->object:return _binary(self,o,OP_DIV)
  def __rtruediv__(self,o:object)->object:return _binary(o,self,OP_DIV)
  def __matmul__(self,o:object)->ndarray:return matmul(self,o)
  def __rmatmul__(self,o:object)->ndarray:return matmul(o,self)
  def __neg__(self)->object:return multiply(self,-1)
  def __pos__(self)->object:return self
  def reshape(self,*shape:int|Sequence[int])->ndarray:return self._materialize().reshape(*shape)
  def transpose(self,axes:Sequence[int]|None=None)->ndarray:return self._materialize().transpose(axes)
  def astype(self,dtype:object,*,copy:builtins.bool=True,device:object=None)->ndarray:return self._materialize().astype(dtype,copy=copy,device=device)
  def tolist(self)->object:return self._materialize().tolist()
  def sum(self,axis:object=None)->object:return sum(self,axis=axis)
  def mean(self,axis:object=None)->object:return mean(self,axis=axis)
  def min(self,axis:object=None)->object:return min(self,axis=axis)
  def max(self,axis:object=None)->object:return max(self,axis=axis)
  def argmax(self,axis:object=None)->object:return argmax(self,axis=axis)
  def _materialize(self)->ndarray:
    if self._cached is None:self._cached=self._compute()
    return self._cached
  def _compute(self)->ndarray:raise NotImplementedError


class _UnaryExpression(_DeferredArray):
  # deferred sin/cos/exp/log over a single base; held as a node so a downstream `+ scalar*y` can fuse via _match_sam.
  __slots__=("_base","_op")
  def __init__(self,base:ndarray|_DeferredArray,op:int)->None:super().__init__();self._base=base;self._op=op
  @property
  def dtype(self)->DType:return _UR[self._base.dtype]
  @property
  def shape(self)->tuple[int,...]:return self._base.shape
  def _compute(self)->ndarray:b=_mat(self._base);return ndarray(_native.unary(b._native,self._op))


class _ScalarBinaryExpression(_DeferredArray):
  # deferred array⋆scalar (multiplication only, see _can_def_sb); fusion partner for sin_add_mul.
  __slots__=("_array","_op","_scalar","_scalar_dtype","_scalar_on_left")
  def __init__(self,array:ndarray|_DeferredArray,scalar:object,scalar_dtype:DType,op:int,scalar_on_left:builtins.bool)->None:
    super().__init__();self._array=array;self._scalar=scalar;self._scalar_dtype=scalar_dtype;self._op=op;self._scalar_on_left=scalar_on_left
  @property
  def dtype(self)->DType:return _BR[self._op][(self._array.dtype,self._scalar_dtype)]
  @property
  def shape(self)->tuple[int,...]:return self._array.shape
  def _compute(self)->ndarray:a=_mat(self._array);return ndarray(_native.binary_scalar(a._native,self._scalar,self._scalar_dtype.code,self._op,self._scalar_on_left))


def dtype(value:object)->DType:return _resolve_dtype(value)

def array(obj:object,dtype:object=None,*,copy:builtins.bool|None=True,device:object=None)->ndarray:return asarray(obj,dtype=dtype,copy=copy,device=device)

def asarray(obj:object,dtype:object=None,*,copy:builtins.bool|None=None,device:object=None)->ndarray:
  if device is not None and device!="cpu":raise NotImplementedError("monpy v1 only supports cpu arrays")
  t=type(obj)
  if t is ndarray:                                                                                                            # already-ours fast path: skip numpy entirely
    if dtype is None:return obj.astype(obj.dtype,copy=True) if copy is True else obj
    tgt=_resolve_dtype(dtype)
    if tgt==obj.dtype and copy is not True:return obj
    if copy is False:raise ValueError(_CFE)
    return obj.astype(tgt,copy=True)
  global _NPTYPE
  if _NPTYPE is None:_NPTYPE=_npmod().ndarray
  if isinstance(obj,_NPTYPE):                                                                                                 # numpy ndarray: native zero-copy if writeable, else fall back to array-interface path
    tc=-1
    if dtype is not None:tc=_resolve_dtype(dtype).code
    if copy is False and not builtins.bool(obj.flags.writeable):raise ValueError("readonly array requires copy=True")
    cf=-1 if copy is None else (1 if copy else 0)                                                                              # tri-state: -1 default, 0 never, 1 always
    try:native=_native.asarray_from_numpy(obj,tc,cf)
    except Exception:return _ai_asarray(obj,dtype=dtype,copy=copy)
    return ndarray(native,owner=None if cf==1 else obj)
  if isinstance(obj,_DeferredArray):return asarray(obj._materialize(),dtype=dtype,copy=copy,device=device)
  if _has_ai(obj):return _ai_asarray(obj,dtype=dtype,copy=copy)
  if copy is False:raise ValueError(_CFE)
  shape,flat=_flat(obj)                                                                                                      # nested list/tuple → (shape, flat values)
  tgt=_resolve_dtype(dtype) if dtype is not None else _infer_dtype(flat)
  return ndarray(_native.from_flat(flat,shape,tgt.code),dtype=tgt,shape=shape,strides=_csb(shape,tgt.itemsize))

def empty(shape:int|Sequence[int],dtype:object=None,*,device:object=None)->ndarray:
  _check_cpu(device);t=_resolve_dtype(dtype) if dtype is not None else float64;n=_norm_shape(shape)
  return ndarray(_native.empty(n,t.code),dtype=t,shape=n,strides=_csb(n,t.itemsize))

def zeros(shape:int|Sequence[int],dtype:object=None,*,device:object=None)->ndarray:return full(shape,0,dtype=_resolve_dtype(dtype) if dtype is not None else float64,device=device)
def ones(shape:int|Sequence[int],dtype:object=None,*,device:object=None)->ndarray:return full(shape,1,dtype=_resolve_dtype(dtype) if dtype is not None else float64,device=device)

def full(shape:int|Sequence[int],fill_value:object,*,dtype:object=None,device:object=None)->ndarray:
  _check_cpu(device);t=_resolve_dtype(dtype) if dtype is not None else _infer_dtype([fill_value]);n=_norm_shape(shape)
  return ndarray(_native.full(n,fill_value,t.code),dtype=t,shape=n,strides=_csb(n,t.itemsize))

def arange(start:int|float,stop:int|float|None=None,step:int|float=1,*,dtype:object=None,device:object=None)->ndarray:
  _check_cpu(device);a=0 if stop is None else start;b=start if stop is None else stop                                         # 1-arg form: stop=start, start=0 (numpy convention)
  t=_resolve_dtype(dtype) if dtype is not None else (float64 if any(isinstance(v,builtins.float) for v in(a,b,step)) else int64)
  return ndarray(_native.arange(a,b,step,t.code),dtype=t)

def linspace(start:int|float,stop:int|float,num:int=50,*,dtype:object=None,device:object=None)->ndarray:
  _check_cpu(device);t=_resolve_dtype(dtype) if dtype is not None else float64
  return ndarray(_native.linspace(start,stop,num,t.code),dtype=t,shape=(num,),strides=(t.itemsize,))

def reshape(x:object,shape:int|Sequence[int])->ndarray:return asarray(x).reshape(shape)
def transpose(x:object,axes:Sequence[int]|None=None)->ndarray:return asarray(x).transpose(axes)
def matrix_transpose(x:object)->ndarray:return asarray(x).mT
def broadcast_to(x:object,shape:int|Sequence[int])->ndarray:a=asarray(x);return ndarray(_native.broadcast_to(a._native,_norm_shape(shape)),base=a)

def add(x1:object,x2:object,*,out:ndarray|None=None)->object:return _binary(x1,x2,OP_ADD,out=out)
def subtract(x1:object,x2:object,*,out:ndarray|None=None)->object:return _binary(x1,x2,OP_SUB,out=out)
def multiply(x1:object,x2:object,*,out:ndarray|None=None)->object:return _binary(x1,x2,OP_MUL,out=out)
def divide(x1:object,x2:object,*,out:ndarray|None=None)->object:return _binary(x1,x2,OP_DIV,out=out)

def sin(x:object)->object:return _unary(x,UNARY_SIN)
def cos(x:object)->object:return _unary(x,UNARY_COS)
def exp(x:object)->object:return _unary(x,UNARY_EXP)
def log(x:object)->object:return _unary(x,UNARY_LOG)

def sin_add_mul(x:object,y:object,scalar:object)->ndarray:
  # explicit fused kernel: sin(x) + scalar*y. The implicit pattern sin(x)+(scalar*y) is also caught by _fuse.
  if not _is_scalar(scalar):raise NotImplementedError("sin_add_mul currently requires a Python scalar multiplier")
  lhs=_av(x);rhs=_av(y);sd=_isd_arr(rhs,scalar);la=_mat(lhs);ra=_mat(rhs)
  return ndarray(_native.sin_add_mul(la._native,ra._native,scalar,sd.code))

def where(condition:object,x1:object,x2:object)->ndarray:
  c=asarray(condition,dtype=bool);l=asarray(x1);r=asarray(x2);l,r=_coerce(l,r,OP_ADD)                                         # use OP_ADD's promotion table — same numeric rules
  return ndarray(_native.where(c._native,l._native,r._native))

def sum(x:object,axis:object=None)->object:return _reduce(x,axis,REDUCE_SUM)
def mean(x:object,axis:object=None)->object:return _reduce(x,axis,REDUCE_MEAN)
def min(x:object,axis:object=None)->object:return _reduce(x,axis,REDUCE_MIN)  # noqa: A001
def max(x:object,axis:object=None)->object:return _reduce(x,axis,REDUCE_MAX)  # noqa: A001
def argmax(x:object,axis:object=None)->object:return _reduce(x,axis,REDUCE_ARGMAX)

def matmul(x1:object,x2:object)->ndarray:
  if type(x1) is ndarray and type(x2) is ndarray and x1._dtype is not None and x1._dtype is x2._dtype:                        # same-dtype fast path
    return ndarray(_native.matmul(x1._native,x2._native))
  l=_mat(_av(x1));r=_mat(_av(x2));l,r=_coerce(l,r,OP_MUL)
  return ndarray(_native.matmul(l._native,r._native))

def diagonal(a:object,offset:int=0,axis1:int=0,axis2:int=1)->ndarray:
  arr=asarray(a);d=getattr(_native,"diagonal",None)                                                                           # native impl is feature-gated; fall back if absent
  return ndarray(d(arr._native,int(offset),int(axis1),int(axis2))) if d is not None else _diag_fallback(arr,int(offset),int(axis1),int(axis2))

def trace(a:object,offset:int=0,axis1:int=0,axis2:int=1,dtype:object=None,out:ndarray|None=None)->object:
  arr=asarray(a);tr=getattr(_native,"trace",None)
  if tr is not None:
    dc=-1 if dtype is None else _resolve_dtype(dtype).code
    r=ndarray(tr(arr._native,int(offset),int(axis1),int(axis2),dc))
    v=r._scalar() if r.ndim==0 else r
  else:
    v=_trace_fallback(arr,int(offset),int(axis1),int(axis2),dtype)
  if out is not None:o=asarray(out);o[...]=v;return out
  return v

def astype(x:object,dtype:object,/,*,copy:builtins.bool=True,device:object=None)->ndarray:return asarray(x).astype(dtype,copy=copy,device=device)

def from_dlpack(x:object,/,*,device:object=None,copy:builtins.bool|None=None)->ndarray:
  if device is not None and device!="cpu":raise NotImplementedError("monpy v1 only supports cpu arrays")
  global _NPTYPE
  if _NPTYPE is None:_NPTYPE=_npmod().ndarray
  if isinstance(x,_NPTYPE):return asarray(x,copy=copy is True)
  return asarray(_npmod().from_dlpack(x,device=device,copy=copy),copy=copy is True)

def __array_namespace_info__()->object:
  def dts(*,device:object=None,kind:object=None)->dict[str,DType]:_check_cpu(device);return dict(_DTN)
  def ddts(*,device:object=None)->dict[str,DType]:_check_cpu(device);return{"integral":int64,"real floating":float64,"bool":bool}
  return SimpleNamespace(default_device=lambda:"cpu",devices=lambda:["cpu"],dtypes=dts,default_dtypes=ddts,capabilities=lambda:{"boolean indexing":False,"data-dependent shapes":False})


def _binary(x1:object,x2:object,op:int,*,out:ndarray|None=None)->object:
  if out is not None:                                                                                                         # `out=` skips the deferred path; everything materialises into out._native
    if type(x1) is ndarray and type(x2) is ndarray and x1._dtype is not None and x1._dtype is x2._dtype:
      _native.binary_into(out._native,x1._native,x2._native,op);return out
    l=_mat(_av(x1));r=_mat(_av(x2));l,r=_coerce(l,r,op)
    _native.binary_into(out._native,l._native,r._native,op);return out
  f=_fuse(x1,x2,op)                                                                                                           # try sin(x)+scalar*y fusion before falling through
  if f is not None:return f
  if _isarrv(x1):return _binary_from_array(x1,x2,op,scalar_on_left=False)
  if _isarrv(x2):return _binary_from_array(x2,x1,op,scalar_on_left=True)
  l=asarray(x1);r=asarray(x2);l,r=_coerce(l,r,op)
  return ndarray(_native.binary(l._native,r._native,op))

def _binary_from_array(arr:ndarray|_DeferredArray,other:object,op:int,*,scalar_on_left:builtins.bool)->object:
  # called when `arr` is known array-like; `other` may be array, scalar, or convertible.
  if isinstance(arr,ndarray) and isinstance(other,ndarray):
    l,r=(other,arr) if scalar_on_left else (arr,other)
    if l.dtype is r.dtype:return ndarray(_native.binary(l._native,r._native,op))
    l,r=_coerce(l,r,op);return ndarray(_native.binary(l._native,r._native,op))
  if _isarrv(other):
    l=_mat(other) if scalar_on_left else _mat(arr)
    r=_mat(arr) if scalar_on_left else _mat(other)
    l,r=_coerce(l,r,op);return ndarray(_native.binary(l._native,r._native,op))
  if _is_scalar(other):
    sd=_isd_arr(arr,other)
    if op==OP_MUL and _can_def_sb(arr,sd):return _ScalarBinaryExpression(arr,other,sd,op,scalar_on_left)                       # defer (only mul, only floats) so a downstream sin(...)+(scalar*y) can fuse
    av=_mat(arr);return ndarray(_native.binary_scalar(av._native,other,sd.code,op,scalar_on_left))
  oa=asarray(other)
  if scalar_on_left:l,r=_coerce(oa,_mat(arr),op);return ndarray(_native.binary(l._native,r._native,op))
  l,r=_coerce(_mat(arr),oa,op);return ndarray(_native.binary(l._native,r._native,op))

def _unary(x:object,op:int)->object:
  a=_av(x)
  if op==UNARY_SIN and a.dtype in(float32,float64):return _UnaryExpression(a,op)                                              # only defer sin on floats — that's what sin_add_mul fuses
  av=_mat(a);return ndarray(_native.unary(av._native,op))

def _reduce(x:object,axis:object,op:int)->object:
  if axis is not None:raise NotImplementedError("axis-specific reductions are not implemented in monpy v1")
  a=_mat(_av(x));return _native.reduce(a._native,op).get_scalar(0)

def _isarrv(v:object)->builtins.bool:return isinstance(v,(ndarray,_DeferredArray))                                            # is "array value": ours, materialised or not
def _av(v:object)->ndarray|_DeferredArray:return v if isinstance(v,(ndarray,_DeferredArray)) else asarray(v)                  # as-array-value: convert non-arrays via asarray
def _mat(v:ndarray|_DeferredArray)->ndarray:return v._materialize() if isinstance(v,_DeferredArray) else v                    # materialize: collapse a deferred node to a concrete ndarray

def _coerce(l:ndarray,r:ndarray,op:int)->tuple[ndarray,ndarray]:
  # cast both operands to the common result dtype before handing to native binary kernel.
  t=_BR[op][(l.dtype,r.dtype)]
  if l.dtype!=t:l=l.astype(t)
  if r.dtype!=t:r=r.astype(t)
  return l,r

def _result_dtype_for_binary(l:DType,r:DType,op:int)->DType:return _BR[op][(l,r)]                                             # exposed to tests/promotions, mirrors Mojo side
def _result_dtype_for_unary(d:DType)->DType:return _UR[d]

def _can_def_sb(v:ndarray|_DeferredArray,sd:DType)->builtins.bool:return v.dtype in(float32,float64) and sd in(float32,float64)  # defer scalar binary only on float×float — int paths must materialise

def _fuse(x1:object,x2:object,op:int)->ndarray|None:
  # try to recognise sin(x)+scalar*y in either argument order.
  if op!=OP_ADD:return None
  r=_match_sam(x1,x2)
  return r if r is not None else _match_sam(x2,x1)

def _match_sam(x1:object,x2:object)->ndarray|None:
  # sam = sin(...) + scalar*(...). returns None unless both operands are the right deferred kinds.
  if not isinstance(x1,_UnaryExpression) or x1._op!=UNARY_SIN:return None
  if not isinstance(x2,_ScalarBinaryExpression) or x2._op!=OP_MUL:return None
  l=_mat(x1._base);r=_mat(x2._array)
  return ndarray(_native.sin_add_mul(l._native,r._native,x2._scalar,x2._scalar_dtype.code))

def _diag_fallback(arr:ndarray,offset:int,axis1:int,axis2:int)->ndarray:
  # python-side diagonal extraction used when the native diagonal op is unavailable.
  if arr.ndim<2:raise ValueError("diag requires an array of at least two dimensions")
  axis1=_norm_axis(axis1,arr.ndim);axis2=_norm_axis(axis2,arr.ndim)
  if axis1==axis2:raise ValueError("axis1 and axis2 cannot be the same")
  rs=builtins.max(-offset,0);cs=builtins.max(offset,0)                                                                         # row/col start: positive offset slides diagonal up
  dl=builtins.max(0,builtins.min(arr.shape[axis1]-rs,arr.shape[axis2]-cs))                                                     # diagonal length
  ra=tuple(a for a in range(arr.ndim) if a not in(axis1,axis2))                                                                # remaining axes (kept as outer loop)
  rsh=tuple(arr.shape[a] for a in ra);out_shape=rsh+(dl,);flat:list[object]=[]
  for prefix in _iter_idx(rsh):
    key:list[object]=[0]*arr.ndim
    for a,i in zip(ra,prefix,strict=True):key[a]=i
    for di in range(dl):key[axis1]=rs+di;key[axis2]=cs+di;flat.append(arr[tuple(key)])
  return ndarray(_native.from_flat(flat,out_shape,arr.dtype.code),dtype=arr.dtype,shape=out_shape,strides=_csb(out_shape,arr.itemsize))

def _trace_fallback(arr:ndarray,offset:int,axis1:int,axis2:int,dt:object)->object:
  # python-side trace when native is missing — reduces along the diagonal.
  d=diagonal(arr,offset=offset,axis1=axis1,axis2=axis2)
  t=_resolve_dtype(dt) if dt is not None else (int64 if d.dtype==bool else d.dtype)                                            # bool inputs accumulate as int64 to match numpy
  if d.dtype!=t:d=d.astype(t)
  if d.ndim==1:return sum(d)
  out_shape=d.shape[:-1];flat:list[object]=[]
  for prefix in _iter_idx(out_shape):
    total:object=0.0 if t in(float32,float64) else 0
    for di in range(d.shape[-1]):total+=d[prefix+(di,)]                                                                         # type: ignore[operator]
    flat.append(total)
  return ndarray(_native.from_flat(flat,out_shape,t.code),dtype=t,shape=out_shape,strides=_csb(out_shape,t.itemsize))

def _is_scalar(v:object)->builtins.bool:return isinstance(v,(builtins.bool,builtins.int,builtins.float))                      # python scalar (bool/int/float)

def _isd(v:object)->DType:                                                                                                    # infer scalar dtype from a python scalar alone
  if isinstance(v,builtins.bool):return bool
  if isinstance(v,builtins.int):return int64
  return float64

def _isd_arr(arr:ndarray|_DeferredArray,v:object)->DType:
  # like _isd but biased toward the array's float dtype, so `f32_array * 2` stays f32 instead of upcasting via int64.
  ad=arr.dtype
  if ad in(float32,float64) and isinstance(v,(builtins.int,builtins.float)):return ad
  if ad==int64 and isinstance(v,(builtins.bool,builtins.int)):return int64
  return _isd(v)

def _resolve_dtype(v:object)->DType:
  if v is None:return float64
  if isinstance(v,DType):return v
  if isinstance(v,str):
    try:return _DTN[v]
    except KeyError as exc:raise NotImplementedError(f"unsupported dtype: {v}") from exc
  if v is builtins.bool:return bool
  if v is builtins.int:return int64
  if v is builtins.float:return float64
  nd=_np_dtype_obj(v)
  if nd is not None:return _dtype_from_np(nd)
  raise NotImplementedError(f"unsupported dtype: {v!r}")

def _npmod()->ModuleType:                                                                                                     # lazy numpy import; loading monpy without numpy must succeed
  try:import numpy as n
  except ModuleNotFoundError as exc:raise NotImplementedError("numpy is required for array interface and dlpack interop") from exc
  return n

def _np_dtype_obj(v:object)->object|None:
  try:n=_npmod()
  except NotImplementedError:return None
  if isinstance(v,n.dtype):return v
  try:
    if isinstance(v,type) and issubclass(v,n.generic):return n.dtype(v)
  except TypeError:return None
  return None

def _np_dtype_for(d:DType)->object:return _npmod().dtype(d.typestr)

def _dtype_from_np(v:object)->DType:
  n=_npmod();nd=n.dtype(v)
  if nd.fields is not None or nd.subdtype is not None:raise NotImplementedError(f"unsupported dtype: {nd}")                    # structured/sub-dtypes are out of scope
  if not nd.isnative:raise NotImplementedError(f"unsupported dtype: {nd}")
  try:return _DTNK[(nd.kind,nd.itemsize)]
  except KeyError as exc:raise NotImplementedError(f"unsupported dtype: {nd}") from exc

def _has_ai(o:object)->builtins.bool:                                                                                         # has __array_interface__ dict — the universal zero-copy hand-off
  try:i=o.__array_interface__
  except Exception:return False
  return isinstance(i,dict)

def _ai_asarray(obj:object,*,dtype:object,copy:builtins.bool|None)->ndarray:
  # generic array-interface ingest. Handles dtype conversion, readonly, and copy semantics
  # before deciding between zero-copy view (_ext_from_np) and explicit copy (_copy_from_np).
  n=_npmod()
  global _NPTYPE
  if _NPTYPE is None:_NPTYPE=n.ndarray
  a=obj if isinstance(obj,_NPTYPE) else n.asarray(obj)
  iface=a.__array_interface__
  sd=_DTBT.get(iface["typestr"]) or _dtype_from_np(a.dtype)                                                                    # source dtype
  t=_resolve_dtype(dtype) if dtype is not None else None
  if t is not None and t!=sd:
    if copy is False:raise ValueError(_CFE)
    return _copy_from_np(a.astype(_np_dtype_for(t),copy=True))
  da,ro=iface["data"]                                                                                                          # data_address, readonly flag
  if copy is True:return _copy_from_np(a,dtype_value=sd,iface=iface)
  if ro:
    if copy is False:raise ValueError("readonly array requires copy=True")
    return _copy_from_np(a,dtype_value=sd,iface=iface)
  return _ext_from_np(a,sd,iface=iface,data_address=da)

def _ext_from_np(a:object,d:DType,*,iface:dict[str,object]|None=None,data_address:int|None=None)->ndarray:
  # zero-copy borrow: native side wraps the foreign buffer; we keep `a` as owner so it isn't GC'd.
  if iface is None:iface=a.__array_interface__
  sh=iface["shape"];rs=iface["strides"];i=d.itemsize
  if rs is None:bs=_csb(sh,i)
  else:
    bs=rs
    for s in bs:
      if s%i!=0:raise NotImplementedError("array interface strides must align to dtype itemsize")
  es=tuple(s//i for s in bs)                                                                                                   # convert byte strides → element strides for native side
  if data_address is None:data_address=iface["data"][0]
  bl=math.prod(sh)*i                                                                                                           # byte length (used by native bounds check)
  return ndarray(_native.from_external(data_address,sh,es,d.code,bl),dtype=d,shape=sh,strides=bs,owner=a)

def _copy_from_np(a:object,*,dtype_value:DType|None=None,iface:dict[str,object]|None=None)->ndarray:
  # forced-copy ingest from any array-interface source.
  if iface is None:iface=a.__array_interface__
  if dtype_value is None:dtype_value=_DTBT.get(iface["typestr"]) or _dtype_from_np(a.dtype)
  sh=iface["shape"];rs=iface["strides"];i=dtype_value.itemsize
  if rs is None:es=_cse(sh)
  else:
    for s in rs:
      if s%i!=0:raise NotImplementedError("array interface strides must align to dtype itemsize")
    es=tuple(s//i for s in rs)
  da=iface["data"][0];bl=math.prod(sh)*i
  return ndarray(_native.copy_from_external(da,sh,es,dtype_value.code,bl),dtype=dtype_value,shape=sh,strides=_csb(sh,i))

def _cse(sh:tuple[int,...])->tuple[int,...]:                                                                                  # c-contiguous strides in elements
  st=[1]*len(sh);s=1
  for a in range(len(sh)-1,-1,-1):st[a]=s;s*=sh[a]
  return tuple(st)

def _infer_dtype(flat:Sequence[object])->DType:
  # promote up the ladder bool→int→float depending on what we see in the input.
  if not flat:return float64
  hf=False;hi=False;hb=True                                                                                                    # has_float / has_int / has_bool (so far still all-bool)
  for v in flat:
    if isinstance(v,builtins.bool):continue
    hb=False
    if isinstance(v,builtins.int):hi=True;continue
    if isinstance(v,builtins.float):hf=True;continue
    raise NotImplementedError(f"unsupported array value type: {type(v).__name__}")
  if hf:return float64
  if hi or not hb:return int64
  return bool

def _norm_shape(shape:int|Sequence[int])->tuple[int,...]:
  if isinstance(shape,builtins.int):
    if shape<0:raise ValueError("negative dimensions are not allowed")
    return(shape,)
  out=tuple(int(d) for d in shape)
  if any(d<0 for d in out):raise ValueError("negative dimensions are not allowed")
  return out

def _csb(sh:tuple[int,...],i:int)->tuple[int,...]:                                                                            # c-contiguous strides in bytes (itemsize=i)
  st=[0]*len(sh);s=i
  for a in range(len(sh)-1,-1,-1):st[a]=s;s*=sh[a]
  return tuple(st)

def _is_c_contig(sh:tuple[int,...],st:tuple[int,...],i:int)->builtins.bool:
  # walk axes right-to-left checking strides match c-contiguous expectation; size-0 axes short-circuit true.
  e=i
  for a in range(len(sh)-1,-1,-1):
    if sh[a]==0:return True
    if sh[a]!=1 and st[a]!=e:return False
    e*=sh[a]
  return True

def _iter_idx(sh:tuple[int,...])->Iterable[tuple[int,...]]:
  if not sh:yield();return
  yield from itertools.product(*(range(d) for d in sh))

def _shape_args(sh:Sequence[int|Sequence[int]])->tuple[int,...]:
  # accept both `reshape(2,3)` and `reshape((2,3))` calling conventions.
  if len(sh)==1 and not isinstance(sh[0],builtins.int):return _norm_shape(sh[0])
  return _norm_shape(sh)  # type: ignore[arg-type]

def _flat(obj:object)->tuple[tuple[int,...],list[object]]:
  # depth-first flatten of nested list/tuple → (shape, flat values). Rejects ragged sequences.
  if isinstance(obj,(list,tuple)):
    if not obj:return(0,),[]
    cs:list[tuple[int,...]]=[];fl:list[object]=[]
    for it in obj:s,f=_flat(it);cs.append(s);fl.extend(f)
    fs=cs[0]
    if any(s!=fs for s in cs):raise ValueError("cannot create monpy array from ragged nested sequences")
    return(len(obj),)+fs,fl
  if isinstance(obj,ndarray):return obj.shape,[obj._native.get_scalar(i) for i in range(obj.size)]
  if isinstance(obj,(builtins.bool,builtins.int,builtins.float)):return(),[obj]
  raise NotImplementedError(f"unsupported array input type: {type(obj).__name__}")

def _unflat(flat:Sequence[object],sh:tuple[int,...])->object:
  if not sh:return flat[0]
  if len(sh)==1:return list(flat[:sh[0]])
  step=math.prod(sh[1:])
  return[_unflat(flat[i*step:(i+1)*step],sh[1:]) for i in range(sh[0])]

def _expand_key(key:object,ndim:int)->tuple[object,...]:
  # normalise an indexing key to a tuple of length ndim, expanding Ellipsis and rejecting newaxis.
  if key==():
    if ndim!=0:raise IndexError("empty index is only valid for zero-dimensional arrays")
    return()
  parts=key if isinstance(key,tuple) else(key,)
  if any(p is None for p in parts):raise NotImplementedError("newaxis indexing is not implemented in monpy v1")
  if parts.count(Ellipsis)>1:raise IndexError("an index can only have a single ellipsis")
  if Ellipsis in parts:
    ea=parts.index(Ellipsis);missing=ndim-(len(parts)-1)
    parts=parts[:ea]+(slice(None),)*missing+parts[ea+1:]
  if len(parts)>ndim:raise IndexError("too many indices for array")
  return parts+(slice(None),)*(ndim-len(parts))

def _norm_idx(i:object,d:int)->int:
  if not isinstance(i,builtins.int):raise NotImplementedError("monpy v1 supports only integer and slice indexing")
  if i<0:i+=d
  if i<0 or i>=d:raise IndexError("index out of bounds")
  return i

def _norm_axis(a:int,n:int)->int:
  if a<0:a+=n
  if a<0 or a>=n:raise ValueError("axis out of bounds")
  return a

def _norm_axes(axes:Sequence[int],n:int)->tuple[int,...]:
  r=tuple(a+n if a<0 else a for a in axes)
  if sorted(r)!=list(range(n)):raise ValueError("axes must be a permutation of dimensions")
  return r

def _check_cpu(d:object)->None:
  if d not in(None,"cpu"):raise NotImplementedError("monpy v1 only supports cpu arrays")


linalg=importlib.import_module(f"{__name__}.linalg")

__all__=["DType","add","arange","array","asarray","argmax","astype","bool","broadcast_to","cos","diagonal","divide","dtype","e","empty","exp","float32","float64","from_dlpack","full","inf","int64","linalg","linspace","log","matmul","matrix_transpose","max","mean","min","multiply","nan","ndarray","newaxis","ones","pi","reshape","sin","sin_add_mul","subtract","sum","trace","transpose","where","zeros"]
