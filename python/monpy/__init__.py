# fmt: off
# ruff: noqa
from __future__ import annotations
import builtins, importlib, itertools, math, typing
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from types import ModuleType, SimpleNamespace
from . import _native
_Scalar:typing.TypeAlias=builtins.bool|builtins.int|builtins.float
_NativeArray:typing.TypeAlias=_native.Array

class _ArrayInterfaceLike(typing.Protocol):
  @property
  def __array_interface__(self)->dict[str, object]:...

_DOMAIN_CODES:dict[str, dict[str, int]]=_native._domain_codes()
DTypeCode:type[IntEnum]=IntEnum("DTypeCode", _DOMAIN_CODES["dtype"])
DTypeKindCode:type[IntEnum]=IntEnum("DTypeKindCode", _DOMAIN_CODES["dtype_kind"])
CastingRule:type[IntEnum]=IntEnum("CastingRule", _DOMAIN_CODES["casting"])
BinaryOp:type[IntEnum]=IntEnum("BinaryOp", _DOMAIN_CODES["binary"])
UnaryOp:type[IntEnum]=IntEnum("UnaryOp", _DOMAIN_CODES["unary"])
CompareOp:type[IntEnum]=IntEnum("CompareOp", _DOMAIN_CODES["compare"])
LogicalOp:type[IntEnum]=IntEnum("LogicalOp", _DOMAIN_CODES["logical"])
PredicateOp:type[IntEnum]=IntEnum("PredicateOp", _DOMAIN_CODES["predicate"])
ReduceOp:type[IntEnum]=IntEnum("ReduceOp", _DOMAIN_CODES["reduce"])
BackendKind:type[IntEnum]=IntEnum("BackendKind", _DOMAIN_CODES["backend"])
_DTCODES=_DOMAIN_CODES["dtype"]
_DTKCODES=_DOMAIN_CODES["dtype_kind"]
_CASTCODES=_DOMAIN_CODES["casting"]
_BINCODES=_DOMAIN_CODES["binary"]
_UNCODES=_DOMAIN_CODES["unary"]
_CMPCODES=_DOMAIN_CODES["compare"]
_LOGCODES=_DOMAIN_CODES["logical"]
_PREDCODES=_DOMAIN_CODES["predicate"]
_REDCODES=_DOMAIN_CODES["reduce"]
DTYPE_BOOL, DTYPE_INT64, DTYPE_FLOAT32, DTYPE_FLOAT64=_DTCODES["BOOL"], _DTCODES["INT64"], _DTCODES["FLOAT32"], _DTCODES["FLOAT64"]
DTYPE_INT32, DTYPE_INT16, DTYPE_INT8=_DTCODES["INT32"], _DTCODES["INT16"], _DTCODES["INT8"]
DTYPE_UINT64, DTYPE_UINT32, DTYPE_UINT16, DTYPE_UINT8=_DTCODES["UINT64"], _DTCODES["UINT32"], _DTCODES["UINT16"], _DTCODES["UINT8"]
DTYPE_FLOAT16=_DTCODES["FLOAT16"]
DTYPE_COMPLEX64, DTYPE_COMPLEX128=_DTCODES["COMPLEX64"], _DTCODES["COMPLEX128"]
DTYPE_KIND_BOOL, DTYPE_KIND_SIGNED_INT, DTYPE_KIND_REAL_FLOAT, DTYPE_KIND_UNSIGNED_INT, DTYPE_KIND_COMPLEX_FLOAT=_DTKCODES["BOOL"], _DTKCODES["SIGNED_INT"], _DTKCODES["REAL_FLOAT"], _DTKCODES["UNSIGNED_INT"], _DTKCODES["COMPLEX_FLOAT"]
CASTING_NO, CASTING_EQUIV, CASTING_SAFE, CASTING_SAME_KIND, CASTING_UNSAFE=_CASTCODES["NO"], _CASTCODES["EQUIV"], _CASTCODES["SAFE"], _CASTCODES["SAME_KIND"], _CASTCODES["UNSAFE"]
OP_ADD, OP_SUB, OP_MUL, OP_DIV=_BINCODES["ADD"], _BINCODES["SUB"], _BINCODES["MUL"], _BINCODES["DIV"]
OP_FLOOR_DIV, OP_MOD, OP_POWER, OP_MAXIMUM, OP_MINIMUM, OP_FMIN, OP_FMAX, OP_ARCTAN2, OP_HYPOT, OP_COPYSIGN=_BINCODES["FLOOR_DIV"], _BINCODES["MOD"], _BINCODES["POWER"], _BINCODES["MAXIMUM"], _BINCODES["MINIMUM"], _BINCODES["FMIN"], _BINCODES["FMAX"], _BINCODES["ARCTAN2"], _BINCODES["HYPOT"], _BINCODES["COPYSIGN"]
UNARY_SIN, UNARY_COS, UNARY_EXP, UNARY_LOG=_UNCODES["SIN"], _UNCODES["COS"], _UNCODES["EXP"], _UNCODES["LOG"]
UNARY_TAN, UNARY_ARCSIN, UNARY_ARCCOS, UNARY_ARCTAN, UNARY_SINH, UNARY_COSH, UNARY_TANH, UNARY_LOG1P, UNARY_LOG2, UNARY_LOG10, UNARY_EXP2, UNARY_EXPM1, UNARY_SQRT, UNARY_CBRT, UNARY_DEG2RAD, UNARY_RAD2DEG, UNARY_RECIPROCAL=_UNCODES["TAN"], _UNCODES["ARCSIN"], _UNCODES["ARCCOS"], _UNCODES["ARCTAN"], _UNCODES["SINH"], _UNCODES["COSH"], _UNCODES["TANH"], _UNCODES["LOG1P"], _UNCODES["LOG2"], _UNCODES["LOG10"], _UNCODES["EXP2"], _UNCODES["EXPM1"], _UNCODES["SQRT"], _UNCODES["CBRT"], _UNCODES["DEG2RAD"], _UNCODES["RAD2DEG"], _UNCODES["RECIPROCAL"]
UNARY_NEGATE, UNARY_POSITIVE, UNARY_ABS, UNARY_SQUARE, UNARY_SIGN, UNARY_FLOOR, UNARY_CEIL, UNARY_TRUNC, UNARY_RINT, UNARY_LOGICAL_NOT=_UNCODES["NEGATE"], _UNCODES["POSITIVE"], _UNCODES["ABS"], _UNCODES["SQUARE"], _UNCODES["SIGN"], _UNCODES["FLOOR"], _UNCODES["CEIL"], _UNCODES["TRUNC"], _UNCODES["RINT"], _UNCODES["LOGICAL_NOT"]
UNARY_CONJUGATE=_UNCODES["CONJUGATE"]
CMP_EQ, CMP_NE, CMP_LT, CMP_LE, CMP_GT, CMP_GE=_CMPCODES["EQ"], _CMPCODES["NE"], _CMPCODES["LT"], _CMPCODES["LE"], _CMPCODES["GT"], _CMPCODES["GE"]
LOGIC_AND, LOGIC_OR, LOGIC_XOR=_LOGCODES["AND"], _LOGCODES["OR"], _LOGCODES["XOR"]
PRED_ISNAN, PRED_ISINF, PRED_ISFINITE, PRED_SIGNBIT=_PREDCODES["ISNAN"], _PREDCODES["ISINF"], _PREDCODES["ISFINITE"], _PREDCODES["SIGNBIT"]
REDUCE_SUM, REDUCE_MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_ARGMAX, REDUCE_PROD, REDUCE_ALL, REDUCE_ANY, REDUCE_ARGMIN=_REDCODES["SUM"], _REDCODES["MEAN"], _REDCODES["MIN"], _REDCODES["MAX"], _REDCODES["ARGMAX"], _REDCODES["PROD"], _REDCODES["ALL"], _REDCODES["ANY"], _REDCODES["ARGMIN"]

@dataclass(frozen=True, slots=True, eq=False)
class DType:
  name:builtins.str
  code:int
  kind:builtins.str
  itemsize:int
  alignment:int
  byteorder:builtins.str
  typestr:builtins.str
  format:builtins.str
  scalar_type:builtins.type[object]
  def __repr__(self)->builtins.str:return f"monpy.{self.name}"
  def __eq__(self, other:object)->builtins.bool:
    if isinstance(other, DType):return self.code==other.code
    try:return self is _resolve_dtype(other)
    except NotImplementedError:return False
  def __hash__(self)->builtins.int:return hash(self.code)
  @property
  def type(self)->builtins.type[object]:return self.scalar_type
  @property
  def char(self)->builtins.str:return self.format
  @property
  def str(self)->builtins.str:return self.typestr

@dataclass(frozen=True, slots=True)
class _FInfo:
  dtype:DType
  bits:builtins.int
  eps:builtins.float
  epsneg:builtins.float
  max:builtins.float
  min:builtins.float
  smallest_normal:builtins.float
  tiny:builtins.float
  resolution:builtins.float
  precision:builtins.int
  nmant:builtins.int
  iexp:builtins.int
  maxexp:builtins.int
  minexp:builtins.int
  machep:builtins.int
  negep:builtins.int

@dataclass(frozen=True, slots=True)
class _IInfo:
  dtype:DType
  bits:builtins.int
  min:builtins.int
  max:builtins.int

bool=DType("bool", 0, "b", 1, 1, "|", "|b1", "?", builtins.bool)
bool_=bool
int64=DType("int64", 1, "i", 8, 8, "=", "<i8", "l", builtins.int)
int_=int64
intp=int64
float32=DType("float32", 2, "f", 4, 4, "=", "<f4", "f", builtins.float)
float64=DType("float64", 3, "f", 8, 8, "=", "<f8", "d", builtins.float)
float_=float64
# signed ints.
int32=DType("int32", 4, "i", 4, 4, "=", "<i4", "i", builtins.int)
int16=DType("int16", 5, "i", 2, 2, "=", "<i2", "h", builtins.int)
int8=DType("int8", 6, "i", 1, 1, "|", "|i1", "b", builtins.int)
# unsigned ints. Allocate via the f64 round-trip path; arithmetic carries through dispatch with promotion rules (see dtype_result_for_binary).
uint64=DType("uint64", 7, "u", 8, 8, "=", "<u8", "Q", builtins.int)
ulonglong=uint64
uint32=DType("uint32", 8, "u", 4, 4, "=", "<u4", "I", builtins.int)
uintc=uint32
uint16=DType("uint16", 9, "u", 2, 2, "=", "<u2", "H", builtins.int)
ushort=uint16
uint8=DType("uint8", 10, "u", 1, 1, "|", "|u1", "B", builtins.int)
ubyte=uint8
# float16. Storage is 2-byte half; arithmetic delegates through f64.
float16=DType("float16", 11, "f", 2, 2, "=", "<f2", "e", builtins.float)
half=float16
# complex. Interleaved (real, imag) storage matching numpy/PEP-3118.
complex64=DType("complex64", 12, "c", 8, 4, "=", "<c8", "F", builtins.complex)
csingle=complex64
complex128=DType("complex128", 13, "c", 16, 8, "=", "<c16", "D", builtins.complex)
cdouble=complex128
clongdouble=complex128
complex_=complex128

_DT:tuple[DType, ...]=(bool, int64, float32, float64, int32, int16, int8, uint64, uint32, uint16, uint8, float16, complex64, complex128)  # ordered by .code; idx == code
_DTC:dict[int, DType]={d.code:d for d in _DT}                                                                                # code → DType
_DTN:dict[str, DType]={d.name:d for d in _DT}|{"bool_":bool, "int_":int64, "float_":float64, "single":float32, "double":float64, "intp":int64, "half":float16, "ubyte":uint8, "ushort":uint16, "uintc":uint32, "ulonglong":uint64, "csingle":complex64, "cdouble":complex128, "clongdouble":complex128, "complex_":complex128} # name → DType
_DTF:dict[str, DType]={d.format:d for d in _DT}|{"?":bool}
_DTBT:dict[str, DType]={"|b1":bool, "|i1":int8, "|u1":uint8}|{p+s:d for s, d in[("i8", int64), ("i4", int32), ("i2", int16), ("u8", uint64), ("u4", uint32), ("u2", uint16), ("f2", float16), ("f4", float32), ("f8", float64), ("c8", complex64), ("c16", complex128)] for p in"<="}  # native/little-endian array-interface typestr → DType
_DTK:dict[DType, int]={bool:DTYPE_KIND_BOOL, int64:DTYPE_KIND_SIGNED_INT, int32:DTYPE_KIND_SIGNED_INT, int16:DTYPE_KIND_SIGNED_INT, int8:DTYPE_KIND_SIGNED_INT, uint64:DTYPE_KIND_UNSIGNED_INT, uint32:DTYPE_KIND_UNSIGNED_INT, uint16:DTYPE_KIND_UNSIGNED_INT, uint8:DTYPE_KIND_UNSIGNED_INT, float32:DTYPE_KIND_REAL_FLOAT, float64:DTYPE_KIND_REAL_FLOAT, float16:DTYPE_KIND_REAL_FLOAT, complex64:DTYPE_KIND_COMPLEX_FLOAT, complex128:DTYPE_KIND_COMPLEX_FLOAT}
_CASTING_CODES:dict[str, int]={"no":CASTING_NO, "equiv":CASTING_EQUIV, "safe":CASTING_SAFE, "same_kind":CASTING_SAME_KIND, "unsafe":CASTING_UNSAFE}
_ISDTYPE_KINDS:dict[str, set[DType]]={"bool":{bool}, "signed integer":{int64, int32, int16, int8}, "unsigned integer":{uint64, uint32, uint16, uint8}, "integral":{int64, int32, int16, int8, uint64, uint32, uint16, uint8}, "real floating":{float32, float64, float16}, "complex floating":{complex64, complex128}, "numeric":{int64, int32, int16, int8, uint64, uint32, uint16, uint8, float32, float64, float16, complex64, complex128}}
_FINFO:dict[DType, _FInfo]={float32:_FInfo(float32, 32, 1.1920928955078125e-07, 5.960464477539063e-08, 3.4028234663852886e38, -3.4028234663852886e38, 1.1754943508222875e-38, 1.1754943508222875e-38, 1e-06, 6, 23, 8, 128, -126, -23, -24), float64:_FInfo(float64, 64, 2.220446049250313e-16, 1.1102230246251565e-16, 1.7976931348623157e308, -1.7976931348623157e308, 2.2250738585072014e-308, 2.2250738585072014e-308, 1e-15, 15, 52, 11, 1024, -1022, -52, -53), float16:_FInfo(float16, 16, 0.0009765625, 0.00048828125, 65504.0, -65504.0, 6.103515625e-05, 6.103515625e-05, 1e-03, 3, 10, 5, 16, -14, -10, -11)}
_IINFO:dict[DType, _IInfo]={int64:_IInfo(int64, 64, -9223372036854775808, 9223372036854775807), int32:_IInfo(int32, 32, -2147483648, 2147483647), int16:_IInfo(int16, 16, -32768, 32767), int8:_IInfo(int8, 8, -128, 127), uint64:_IInfo(uint64, 64, 0, 18446744073709551615), uint32:_IInfo(uint32, 32, 0, 4294967295), uint16:_IInfo(uint16, 16, 0, 65535), uint8:_IInfo(uint8, 8, 0, 255)}

# 4×4 promotion tables flattened to a 16-char digit string, indexed (lhs.code*4 + rhs.code).
# Each char is the result DType's .code in the same _DT ordering. Numpy convention: int+float32→float64.
_TBL=lambda s:{(_DT[i//4], _DT[i%4]):_DT[int(c)] for i, c in enumerate(s)}                                         # decode digit-string → {(lhs,rhs): result}
_NPT:dict[tuple[DType, DType], DType]=_TBL("0123113323233333")                                                     # numeric (add/sub/mul) result codes
_DPT:dict[tuple[DType, DType], DType]=_TBL("3323333323233333")                                                     # division result codes (always promotes to float)
_BR:dict[int, dict[tuple[DType, DType], DType]]={OP_ADD:_NPT, OP_SUB:_NPT, OP_MUL:_NPT, OP_DIV:_DPT,
                                              OP_FLOOR_DIV:_NPT, OP_MOD:_NPT, OP_POWER:_NPT,
                                              OP_MAXIMUM:_NPT, OP_MINIMUM:_NPT, OP_FMIN:_NPT, OP_FMAX:_NPT,
                                              OP_ARCTAN2:_DPT, OP_HYPOT:_DPT, OP_COPYSIGN:_DPT}                    # binary op → promotion table
_UR:dict[DType, DType]={bool:float64, int64:float64, int32:float64, int16:float64, int8:float64, uint64:float64, uint32:float64, uint16:float64, uint8:float64, float16:float16, float32:float32, float64:float64, complex64:complex64, complex128:complex128}  # unary (sin/cos/exp/log) result: int kinds → f64
_UR_PRESERVE:dict[DType, DType]={bool:int64, int64:int64, int32:int32, int16:int16, int8:int8, uint64:uint64, uint32:uint32, uint16:uint16, uint8:uint8, float16:float16, float32:float32, float64:float64, complex64:complex64, complex128:complex128}  # preserve-kind unary: bool → int64
_CFE="unable to avoid copy while creating a monpy array as requested"                                            # error msg for copy=False refusal
_S1E="only size-1 arrays can be converted to python scalars"                                                     # error msg for non-size-1 → python scalar
newaxis=None
nan=math.nan
inf=math.inf
pi=math.pi
e=math.e                                                       # now nan,inf,pi,e is just a reexport from python's math for now.

def _iter(v:object)->Iterable[typing.Any]:return typing.cast(Iterable[typing.Any], v)
def _py_int(v:object)->int:return builtins.int(typing.cast(typing.Any, v))
def _py_float(v:object)->float:return builtins.float(typing.cast(typing.Any, v))
def _axis_tuple(axis:object)->tuple[int, ...]:
  if isinstance(axis, builtins.int):return(axis,)
  if axis is None:raise TypeError("axis cannot be None")
  return tuple(_py_int(a) for a in _iter(axis))
def _axis_int(axis:object, ndim:int, context:str)->int:
  if not isinstance(axis, builtins.int):raise TypeError(f"{context}: axis must be int")
  ax=axis+ndim if axis<0 else axis
  if ax<0 or ax>=ndim:raise ValueError(f"{context}: axis out of range")
  return ax


class Ufunc:
  # ufunc object. The kernel always lives in mojo (one of _native.unary / unary_preserve / binary / compare / logical / predicate)
  # Ufunc just dispatches and applies dtype promotion + casting
  # rules. `kind` selects the native dispatch:
  #   "unary"          → _native.unary, result dtype via _UR (int → f64)
  #   "unary_preserve" → _native.unary_preserve, result dtype via _UR_PRESERVE (bool → int64)
  #   "binary"         → _native.binary, result dtype via _BR[op]
  #   "compare"        → _native.compare, result is bool
  #   "logical"        → _native.logical, result is bool
  #   "predicate"      → _native.predicate, result is bool
  __slots__=("__name__", "nin", "nout", "nargs", "_kind", "_op", "_identity", "_reduce_op")
  def __init__(self, name:str, nin:int, nout:int, kind:typing.Literal["logical", "compare", "predicate", "binary", "unary", "unary_preserve"], op:int, identity:object=None, reduce_op:int|None=None)->None:
    self.__name__=name
    self.nin=nin
    self.nout=nout
    self.nargs=nin+nout
    self._kind=kind
    self._op=op
    self._identity=identity
    self._reduce_op=reduce_op
  def __repr__(self)->str:return f"<monpy.ufunc '{self.__name__}'>"
  @property
  def signature(self)->object:return None
  @property
  def types(self)->tuple[str, ...]:return ()
  @property
  def identity(self)->object:return self._identity
  def __call__(self, *args:object, out:ndarray|None=None, where:object=True, casting:typing.Literal["no", "equiv", "safe", "same_kind", "unsafe"]="same_kind", dtype:object=None)->ndarray:
    if _has_kernel_arg(args):
      if out is not None or where is not True or dtype is not None:raise NotImplementedError(f"{self.__name__}: staged ufunc out/where/dtype are not implemented")
      from .kernels import dsl as _kernel_dsl
      return typing.cast(ndarray, _kernel_dsl.ufunc(self.__name__, *args))
    if where is not True:raise NotImplementedError(f"{self.__name__}: where= not implemented")
    if casting not in("no", "equiv", "safe", "same_kind", "unsafe"):raise ValueError(f"unknown casting: {casting}")
    if len(args)!=self.nin:raise TypeError(f"{self.__name__}() expected {self.nin} positional args, got {len(args)}")
    if self.nin==1:return self._call_unary(args[0], out=out, dtype=dtype)
    if self.nin==2:return self._call_binary(args[0], args[1], out=out, dtype=dtype)
    raise NotImplementedError(f"{self.__name__}: nin={self.nin} not implemented")
  def _call_unary(self, x:object, *, out:ndarray|None, dtype:object)->ndarray:
    av=_av(x)
    # Defer sin/cos/exp/log on floats so `sin(x) + scalar*y` can fuse into the sin_add_mul kernel via _match_sam.
    # Skip when out=/dtype= force eager materialisation.
    if (self._kind=="unary" and self._op in(UNARY_SIN, UNARY_COS, UNARY_EXP, UNARY_LOG)
        and out is None and dtype is None):
      if isinstance(av, _DeferredArray):                           return typing.cast(ndarray, _UnaryExpression(av, self._op))
      if isinstance(av, ndarray) and av.dtype in(float32, float64): return typing.cast(ndarray, _UnaryExpression(av, self._op))
    a=_mat(av)
    if dtype is not None:
      t=_resolve_dtype(dtype)
      if a.dtype!=t:a=a.astype(t)
    if self._kind=="predicate":res=ndarray(_native.predicate(a._native, self._op))
    elif self._kind=="unary_preserve":
      target=_UR_PRESERVE[a.dtype]
      if a.dtype!=target:a=a.astype(target)
      res=ndarray(_native.unary_preserve(a._native, self._op))
    else:res=ndarray(_native.unary(a._native, self._op))
    if out is not None:
      out[...]=res
      return out
    return res
  def _call_binary(self, x1:object, x2:object, *, out:ndarray|None, dtype:object)->ndarray:
    if self._kind=="logical":
      l=asarray(x1, dtype=bool) if not isinstance(x1, ndarray) or x1.dtype!=bool else x1
      r=asarray(x2, dtype=bool) if not isinstance(x2, ndarray) or x2.dtype!=bool else x2
      res=ndarray(_native.logical(l._native, r._native, self._op))
    elif self._kind=="compare":
      l=_mat(_av(x1))
      r=_mat(_av(x2))
      # Promote to common dtype before comparison (numpy parity).
      try:t=_BR[OP_ADD][(l.dtype, r.dtype)]
      except KeyError:t=l.dtype if l.dtype==r.dtype else result_type(l, r)
      if l.dtype!=t:l=l.astype(t)
      if r.dtype!=t:r=r.astype(t)
      res=ndarray(_native.compare(l._native, r._native, self._op))
    else:
      # binary arith.
      if _isarrv(x1) and _is_scalar(x2):
        l=_mat(x1)
        t=_resolve_dtype(dtype) if dtype is not None else _isd_arr(l, x2)
        if l.dtype!=t:l=l.astype(t)
        res=ndarray(_native.binary_scalar(l._native, x2, t.code, self._op, False))
        if out is not None:
          out[...]=res
          return out
        return res
      if _isarrv(x2) and _is_scalar(x1):
        r=_mat(x2)
        t=_resolve_dtype(dtype) if dtype is not None else _isd_arr(r, x1)
        if r.dtype!=t:r=r.astype(t)
        res=ndarray(_native.binary_scalar(r._native, x1, t.code, self._op, True))
        if out is not None:
          out[...]=res
          return out
        return res
      l=_mat(_av(x1))
      r=_mat(_av(x2))
      if dtype is not None:
        t=_resolve_dtype(dtype)
        if l.dtype!=t:l=l.astype(t)
        if r.dtype!=t:r=r.astype(t)
      else:
        l, r=_coerce(l, r, self._op)
      if out is not None:
        _native.binary_into(out._native, l._native, r._native, self._op)
        return out
      res=ndarray(_native.binary(l._native, r._native, self._op))
    if out is not None:
      out[...]=res
      return out
    return res
  def reduce(self, a:object, axis:object=0, dtype:object=None, out:ndarray|None=None, keepdims:builtins.bool=False, initial:object=None, where:object=True)->object:
    if where is not True:raise NotImplementedError(f"{self.__name__}.reduce: where= not implemented")
    if initial is not None:raise NotImplementedError(f"{self.__name__}.reduce: initial= not implemented")
    arr=_mat(_av(a))
    if dtype is not None:
      t=_resolve_dtype(dtype)
      if arr.dtype!=t:arr=arr.astype(t)
    rop=self._reduce_op
    if rop is None:raise TypeError(f"reduce on {self.__name__} is not supported")
    if axis is None:
      # Full reduction.
      if rop in (REDUCE_ARGMAX, REDUCE_ARGMIN, REDUCE_ALL, REDUCE_ANY):
        res=ndarray(_native.reduce(arr._native, rop))
        v=res._scalar() if not keepdims else res.reshape((1,)*arr.ndim)
      else:
        res=ndarray(_native.reduce(arr._native, rop))
        v=res._scalar() if not keepdims else res.reshape((1,)*arr.ndim)
      if out is not None:
        out[...]=v
        return out
      return v
    # axis=int or tuple.
    axes=_axis_tuple(axis)
    res=ndarray(_native.reduce_axis(arr._native, rop, tuple(builtins.int(a) for a in axes), builtins.bool(keepdims)))
    if out is not None:
      out[...]=res
      return out
    return res
  def accumulate(self, a:object, axis:int=0, dtype:object=None, out:ndarray|None=None)->ndarray:
    arr=_mat(_av(a))
    if dtype is not None:
      t=_resolve_dtype(dtype)
      if arr.dtype!=t:arr=arr.astype(t)
    n=arr.ndim
    if n==0:raise TypeError(f"{self.__name__}.accumulate: input is 0-d")
    ax=axis+n if axis<0 else axis
    if ax<0 or ax>=n:raise ValueError(f"{self.__name__}.accumulate: axis out of range")
    # Move axis to last, reshape to (prefix, axis_size), accumulate per row, reshape back.
    moved=moveaxis(arr, ax, -1)
    cont=ascontiguousarray(moved)
    pref=cont.shape[:-1]
    axis_size=cont.shape[-1]
    flat=cont.reshape((math.prod(pref) if pref else 1, axis_size))
    rows=flat.shape[0]
    out_flat=zeros((rows, axis_size), dtype=arr.dtype if dtype is None else _resolve_dtype(dtype))
    # Use raw scalar API for the inner cumulative.
    for i in range(rows):
      acc=flat[i, 0]
      out_flat[i, 0]=acc
      for j in range(1, axis_size):
        v=flat[i, j]
        acc=self.__call__(acc, v) if isinstance(acc, (builtins.int, builtins.float, builtins.bool)) else self.__call__(asarray(acc), asarray(v))._scalar()
        out_flat[i, j]=acc
    res=out_flat.reshape(cont.shape).transpose(_inverse_moveaxis_perm(ax, -1, n)) if n>1 else out_flat.reshape(cont.shape)
    if n==1:res=out_flat.reshape(cont.shape)
    if out is not None:
      out[...]=res
      return out
    return res
  def outer(self, a:object, b:object, *, out:ndarray|None=None, dtype:object=None)->ndarray:
    A=_mat(_av(a))
    B=_mat(_av(b))
    A_b=A.reshape(A.shape+(1,)*B.ndim)
    B_b=B.reshape((1,)*A.ndim+B.shape)
    return self.__call__(A_b, B_b, out=out, dtype=dtype)
  def at(self, a:ndarray, indices:object, b:object=None)->None:
    # Minimal `at`: indices must be integer array; b broadcasts onto indices.
    arr=asarray(a)
    if self.nin==1:
      vals=arr[indices]
      out=self.__call__(vals)
      arr[indices]=out
    else:
      vals=arr[indices]
      out=self.__call__(vals, b)
      arr[indices]=out
  def reduceat(self, a:object, indices:Sequence[int], axis:int=0, dtype:object=None, out:ndarray|None=None)->ndarray:
    raise NotImplementedError(f"{self.__name__}.reduceat is not implemented in monpy v1")

# Numpy ufunc base type name (for isinstance checks); numpy exposes `numpy.ufunc` as the base class.
ufunc=Ufunc

def _inverse_moveaxis_perm(src:int, dst:int, n:int)->tuple[int, ...]:
  # inverse of moveaxis(arr, src, dst): the permutation that undoes it.
  if dst<0:dst+=n
  forward=list(range(n))
  forward.insert(dst, forward.pop(src))
  inv=[0]*n
  for i, p in enumerate(forward):inv[p]=i
  return tuple(inv)


class ndarray:
  # Slim wrapper around _native.Array.
  # We will recompute properties via Mojo each call (~80 ns for dtype, ~150 ns for shape/strides at rank 2).
  # The hot binary-op paths don't read these properties at all (Mojo handles dtype promotion in `binary_dispatch_ops`)
  __array_priority__=1000
  __slots__=("_base", "_native", "_owner")
  def __init__(self, native:_NativeArray, base:ndarray|None=None, *, owner:object|None=None)->None:
    self._native=native
    self._base=base
    self._owner=owner
  @staticmethod
  def _wrap(native:_NativeArray, base:ndarray|None=None)->ndarray: # Hot-path constructor: skip __init__'s arg parsing for fresh op results.
    r=ndarray.__new__(ndarray)
    r._native=native
    r._base=base
    r._owner=None
    return r
  @property
  def dtype(self)->DType:return _DTC[builtins.int(self._native.dtype_code())]
  @property
  def shape(self)->tuple[int, ...]:
    n=builtins.int(self._native.ndim())
    return tuple(builtins.int(self._native.shape_at(a)) for a in range(n))
  @property
  def ndim(self)->int:return builtins.int(self._native.ndim())
  @property
  def size(self)->int:return builtins.int(self._native.size())
  @property
  def itemsize(self)->int:return self.dtype.itemsize
  @property
  def strides(self)->tuple[int, ...]:
    i=self.itemsize
    n=self.ndim
    return tuple(int(self._native.stride_at(a))*i for a in range(n))
  @property
  def device(self)->str:return"cpu"
  @property
  def T(self)->ndarray:
    n=builtins.int(self._native.ndim())
    return self if n<2 else ndarray._wrap(self._native.transpose_full_reverse_method(), base=self)
  @property
  def mT(self)->ndarray:
    if self.ndim<2:raise ValueError("matrix transpose requires at least two dimensions")
    return self.transpose(tuple(range(self.ndim-2))+(self.ndim-1, self.ndim-2))
  @property
  def __array_interface__(self)->dict[str, object]:
    s, st, i=self.shape, self.strides, self.itemsize
    return{"version":3, "shape":s, "typestr":self.dtype.typestr, "data":(builtins.int(self._native.data_address()), False), "strides":None if _is_c_contig(s, st, i) else st}
  def __array_namespace__(self, *, api_version:str|None=None)->ModuleType:
    if api_version not in(None, "2024.12", "2025.12"):raise ValueError(f"unsupported array api version: {api_version}")
    from . import array_api
    return array_api
  def __dlpack__(self, *, stream:object=None, max_version:tuple[int, int]|None=None, dl_device:tuple[int, int]|None=None, copy:builtins.bool|None=None)->object:
    if stream is not None:raise BufferError("cpu dlpack export requires stream=None")
    if dl_device not in(None, (1, 0)):raise BufferError("monpy only exports cpu dlpack tensors")
    from . import _dlpack
    if copy is True:return _dlpack.export_array(asarray(self, copy=True), max_version, copied=True)
    return _dlpack.export_array(self, max_version, copied=False)
  def __dlpack_device__(self)->tuple[int, int]:return(1, 0)
  def __len__(self)->int:
    if self.ndim==0:raise TypeError("len() of unsized object")
    return self.shape[0]
  def __iter__(self)->Iterable[object]:
    for i in range(len(self)):yield self[i]
  def __repr__(self)->str:return f"monpy.asarray({self.tolist()!r}, dtype={self.dtype!r})"
  def __getitem__(self, k:typing.Any)->object:
    if type(k) is tuple and len(k)==3 and k[1] is None:                                                                        # exact rank-2 `a[:, None, :]` view
      p0, p2=k[0], k[2]
      if (
        isinstance(p0, slice)
        and p0.start is None
        and p0.stop is None
        and p0.step is None
        and isinstance(p2, slice)
        and p2.start is None
        and p2.stop is None
        and p2.step is None
        and builtins.int(self._native.ndim())==2
      ):return ndarray._wrap(_native.expand_dims(self._native, 1), base=self)
    if isinstance(k, slice):
      ndim=builtins.int(self._native.ndim())
      if ndim==1:
        if k.start is None and k.stop is None and k.step==-1:
          return ndarray._wrap(self._native.reverse_1d_method(), base=self)
        d=builtins.int(self._native.shape_at(0))
        step=1 if k.step is None else builtins.int(k.step)
        if step==0:raise ValueError("slice step cannot be zero")
        if k.start is None and k.stop is None:
          start=d-1 if step<0 else 0
          stop=-1 if step<0 else d                                # whole-axis defaults; -1 stop tells native to walk past 0 with negative step
        else:start, stop, step=k.indices(d)
        return ndarray._wrap(self._native.slice_1d_method(start, stop, step), base=self)
    # Boolean / fancy integer indexing.
    if _is_advanced_index(k):return _advanced_getitem(self, k)
    if isinstance(k, tuple) and builtins.any(_is_advanced_index(p) for p in k):
      return _advanced_getitem_tuple(self, k)
    v=self._view_for_key(k)
    return v._scalar() if v.ndim==0 else v                                                                                    # scalar collapse for full-integer indexing
  def __setitem__(self, k:object, v:object)->None:
    if _is_advanced_index(k):return _advanced_setitem(self, k, v)
    if isinstance(k, tuple) and builtins.any(_is_advanced_index(p) for p in k):
      return _advanced_setitem_tuple(self, k, v)
    view=self._view_for_key(k)
    if isinstance(v, ndarray):
      _native.copyto(view._native, v._native)
      return
    _native.fill(view._native, v)
  def __bool__(self)->builtins.bool:
    if self.size!=1:raise ValueError("the truth value of an array with more than one element is ambiguous")
    return builtins.bool(self._scalar())
  def __int__(self)->int:
    if self.size!=1:raise TypeError(_S1E)
    s=self._scalar()
    if isinstance(s, builtins.complex):return builtins.int(s.real)
    return builtins.int(s)
  def __float__(self)->float:
    if self.size!=1:raise TypeError(_S1E)
    s=self._scalar()
    if isinstance(s, builtins.complex):return float(s.real)
    return float(s)
  def __complex__(self)->complex:
    if self.size!=1:raise TypeError(_S1E)
    return builtins.complex(self._scalar())
  # binary fast paths: ndarray×ndarray dispatches straight to the native method.
  # Mojo's `binary_dispatch_ops` (in `create.mojo`) handles dtype promotion
  # internally via `result_dtype_for_binary`, so the python-side dtype check
  # is redundant — dropping it skips ~80 ns of dtype-cache reads per call.
  # `_wrap` skips __init__'s arg parsing (~150 ns).
  def __add__(self, o:object)->object:
    if type(o) is ndarray:return ndarray._wrap(self._native.add(o._native))
    return _binary_from_array(self, o, OP_ADD, scalar_on_left=False)
  def __radd__(self, o:object)->object:return _binary_from_array(self, o, OP_ADD, scalar_on_left=True)
  def __sub__(self, o:object)->object:
    if type(o) is ndarray:return ndarray._wrap(self._native.sub(o._native))
    return _binary_from_array(self, o, OP_SUB, scalar_on_left=False)
  def __rsub__(self, o:object)->object:return _binary_from_array(self, o, OP_SUB, scalar_on_left=True)
  def __mul__(self, o:object)->object:
    if type(o) is ndarray:return ndarray._wrap(self._native.mul(o._native))
    return _binary_from_array(self, o, OP_MUL, scalar_on_left=False)
  def __rmul__(self, o:object)->object:return _binary_from_array(self, o, OP_MUL, scalar_on_left=True)
  def __truediv__(self, o:object)->object:
    if type(o) is ndarray:return ndarray._wrap(self._native.div(o._native))
    return _binary_from_array(self, o, OP_DIV, scalar_on_left=False)
  def __rtruediv__(self, o:object)->object:return _binary_from_array(self, o, OP_DIV, scalar_on_left=True)
  def __matmul__(self, o:object)->ndarray:
    if type(o) is ndarray:return ndarray._wrap(self._native.matmul(o._native))
    if isinstance(o, ndarray):
      l, r=_coerce(self, o, OP_MUL)
      return ndarray._wrap(_native.matmul(l._native, r._native))
    return matmul(self, o)
  def __rmatmul__(self, o:object)->ndarray:
    if type(o) is ndarray:return ndarray._wrap(o._native.matmul(self._native))
    if isinstance(o, ndarray):
      l, r=_coerce(o, self, OP_MUL)
      return ndarray._wrap(_native.matmul(l._native, r._native))
    return matmul(o, self)
  def __neg__(self)->object:return multiply(self, -1)
  def __pos__(self)->ndarray:return self
  def reshape(self, *shape:int|Sequence[int])->ndarray:return ndarray(_native.reshape(self._native, _shape_args(shape)), base=self)
  def transpose(self, axes:Sequence[int]|None=None)->ndarray:
    if axes is None:axes=tuple(range(self.ndim-1, -1, -1))
    return ndarray(_native.transpose(self._native, _norm_axes(axes, self.ndim)), base=self)
  def astype(self, dtype:object, *, copy:builtins.bool=True, device:object=None)->ndarray:
    _check_cpu(device)
    t=_resolve_dtype(dtype)
    if t==self.dtype and not copy:return self
    return ndarray(_native.astype(self._native, t.code))
  def tolist(self)->object:return _unflat([self._native.get_scalar(i) for i in range(self.size)], self.shape)
  def sum(self, axis:object=None)->object:return sum(self, axis=axis)
  def mean(self, axis:object=None)->object:return mean(self, axis=axis)
  def min(self, axis:object=None)->object:return min(self, axis=axis)
  def max(self, axis:object=None)->object:return max(self, axis=axis)
  def argmax(self, axis:object=None)->object:return argmax(self, axis=axis)
  def _scalar(self)->_Scalar:
    if self.size!=1:raise TypeError(_S1E)
    return self._native.get_scalar(0)
  def _view_for_key(self, key:object)->ndarray:
    parts=_expand_key(key, self.ndim)
    shape=self.shape
    if builtins.any(p is None for p in parts):
      axis=0
      result_axis=0
      new_axes:list[int]=[]
      only_full_slices=True
      for part in parts:
        if part is None:
          new_axes.append(result_axis)
          result_axis+=1
          continue
        if not isinstance(part, slice):
          only_full_slices=False
          break
        d=shape[axis]
        a, b, c=part.indices(d)
        if a!=0 or b!=d or c!=1:
          only_full_slices=False
          break
        axis+=1
        result_axis+=1
      if only_full_slices and axis==len(shape):
        out=self
        for ax in new_axes:out=ndarray._wrap(_native.expand_dims(out._native, ax), base=out)
        return out
    starts:list[int]=[]
    stops:list[int]=[]
    steps:list[int]=[]
    drops:list[int]=[]                                      # drops[axis]==1 → collapse axis (integer index, not slice)
    axis=0
    result_axis=0
    new_axes=[]
    for part in parts:
      if part is None:
        new_axes.append(result_axis)
        result_axis+=1
        continue
      d=shape[axis]
      if isinstance(part, slice):
        a, b, c=part.indices(d)
        starts.append(a)
        stops.append(b)
        steps.append(c)
        drops.append(0)
      else:
        i=_norm_idx(part, d)
        starts.append(i)
        stops.append(i+1)
        steps.append(1)
        drops.append(1)
      axis+=1
      if isinstance(part, slice):result_axis+=1
    out=ndarray(_native.slice(self._native, tuple(starts), tuple(stops), tuple(steps), tuple(drops)), base=self)
    for axis in new_axes:out=ndarray(_native.expand_dims(out._native, axis), base=out)
    return out


class _DeferredArray:
  # base for lazy expression nodes; subclass declares dtype/shape and overrides _compute().
  # any op needing raw bytes triggers _materialize(), which caches the realized ndarray.
  __array_priority__=1001
  __slots__=("_cached",)
  def __init__(self)->None:self._cached:ndarray|None=None
  @property
  def _native(self)->_NativeArray:return self._materialize()._native
  @property
  def dtype(self)->DType:raise NotImplementedError
  @property
  def shape(self)->tuple[int, ...]:raise NotImplementedError
  @property
  def ndim(self)->int:return len(self.shape)
  @property
  def size(self)->int:return math.prod(self.shape)
  @property
  def itemsize(self)->int:return self.dtype.itemsize
  @property
  def strides(self)->tuple[int, ...]:return self._materialize().strides
  @property
  def device(self)->str:return"cpu"
  @property
  def T(self)->ndarray:return self._materialize().T
  @property
  def mT(self)->ndarray:return self._materialize().mT
  @property
  def __array_interface__(self)->dict[str, object]:return self._materialize().__array_interface__
  def __array_namespace__(self, *, api_version:str|None=None)->ModuleType:return self._materialize().__array_namespace__(api_version=api_version)
  def __dlpack__(self, *, stream:object=None, max_version:tuple[int, int]|None=None, dl_device:tuple[int, int]|None=None, copy:builtins.bool|None=None)->object:
    return self._materialize().__dlpack__(stream=stream, max_version=max_version, dl_device=dl_device, copy=copy)
  def __dlpack_device__(self)->tuple[int, int]:return(1, 0)
  def __len__(self)->int:return len(self._materialize())
  def __iter__(self)->Iterable[object]:return iter(self._materialize())
  def __repr__(self)->str:return repr(self._materialize())
  def __getitem__(self, k:object)->object:return self._materialize()[k]
  def __setitem__(self, k:object, v:object)->None:self._materialize()[k]=v
  def __bool__(self)->builtins.bool:return builtins.bool(self._materialize())
  def __int__(self)->int:return builtins.int(self._materialize())
  def __float__(self)->float:return builtins.float(self._materialize())
  def __add__(self, o:object)->object:return _binary(self, o, OP_ADD)
  def __radd__(self, o:object)->object:return _binary(o, self, OP_ADD)
  def __sub__(self, o:object)->object:return _binary(self, o, OP_SUB)
  def __rsub__(self, o:object)->object:return _binary(o, self, OP_SUB)
  def __mul__(self, o:object)->object:return _binary(self, o, OP_MUL)
  def __rmul__(self, o:object)->object:return _binary(o, self, OP_MUL)
  def __truediv__(self, o:object)->object:return _binary(self, o, OP_DIV)
  def __rtruediv__(self, o:object)->object:return _binary(o, self, OP_DIV)
  def __matmul__(self, o:object)->ndarray:return matmul(self, o)
  def __rmatmul__(self, o:object)->ndarray:return matmul(o, self)
  def __neg__(self)->object:return multiply(self, -1)
  def __pos__(self)->object:return self
  def reshape(self, *shape:int|Sequence[int])->ndarray:return self._materialize().reshape(*shape)
  def transpose(self, axes:Sequence[int]|None=None)->ndarray:return self._materialize().transpose(axes)
  def astype(self, dtype:object, *, copy:builtins.bool=True, device:object=None)->ndarray:return self._materialize().astype(dtype, copy=copy, device=device)
  def tolist(self)->object:return self._materialize().tolist()
  def sum(self, axis:object=None)->object:return sum(self, axis=axis)
  def mean(self, axis:object=None)->object:return mean(self, axis=axis)
  def min(self, axis:object=None)->object:return min(self, axis=axis)
  def max(self, axis:object=None)->object:return max(self, axis=axis)
  def argmax(self, axis:object=None)->object:return argmax(self, axis=axis)
  def _materialize(self)->ndarray:
    if self._cached is None:self._cached=self._compute()
    return self._cached
  def _compute(self)->ndarray:raise NotImplementedError


class _UnaryExpression(_DeferredArray):
  # deferred sin/cos/exp/log over a single base; held as a node so a downstream `+ scalar*y` can fuse via _match_sam.
  __slots__=("_base", "_op")
  def __init__(self, base:ndarray|_DeferredArray, op:int)->None:
    super().__init__()
    self._base=base
    self._op=op
  @property
  def dtype(self)->DType:return _UR[self._base.dtype]
  @property
  def shape(self)->tuple[int, ...]:return self._base.shape
  def _compute(self)->ndarray:
    b=_mat(self._base)
    return ndarray(_native.unary(b._native, self._op))


class _ScalarBinaryExpression(_DeferredArray):
  # deferred array⋆scalar (multiplication only, see _can_def_sb); fusion partner for sin_add_mul.
  __slots__=("_array", "_op", "_scalar", "_scalar_dtype", "_scalar_on_left")
  def __init__(self, array:ndarray|_DeferredArray, scalar:object, scalar_dtype:DType, op:int, scalar_on_left:builtins.bool)->None:
    super().__init__()
    self._array=array
    self._scalar=scalar
    self._scalar_dtype=scalar_dtype
    self._op=op
    self._scalar_on_left=scalar_on_left
  @property
  def dtype(self)->DType:return _BR[self._op][(self._array.dtype, self._scalar_dtype)]
  @property
  def shape(self)->tuple[int, ...]:return self._array.shape
  def _compute(self)->ndarray:
    a=_mat(self._array)
    return ndarray(_native.binary_scalar(a._native, self._scalar, self._scalar_dtype.code, self._op, self._scalar_on_left))


def dtype(value:object)->DType:return _resolve_dtype(value)

def promote_types(type1:object, type2:object)->DType:
  l=_resolve_dtype(type1)
  r=_resolve_dtype(type2)
  return _DTC[builtins.int(_native._promote_types(l.code, r.code))]

def result_type(*arrays_and_dtypes:object)->DType:
  if not arrays_and_dtypes:raise TypeError("result_type() needs at least one array or dtype")
  strong:list[DType]=[]
  weak:list[object]=[]
  for value in arrays_and_dtypes:
    d, is_strong=_dtype_for_result_type_arg(value)
    if is_strong:strong.append(d)
    else:weak.append(value)
  if strong:
    result=strong[0]
    for d in strong[1:]:result=promote_types(result, d)
    for value in weak:result=promote_types(result, _scalar_dtype_for_array_dtype(result, value))
    return result
  result=_isd(weak[0])
  for value in weak[1:]:result=promote_types(result, _isd(value))
  return result

def can_cast(from_:object, to:object, casting:str="safe")->builtins.bool:
  try:casting_code=_CASTING_CODES[casting]
  except KeyError as exc:raise ValueError(f"casting must be one of {tuple(_CASTING_CODES)}") from exc
  f=_dtype_for_can_cast_arg(from_)
  t=_resolve_dtype(to)
  return builtins.bool(_native._can_cast(f.code, t.code, casting_code))

def issubdtype(arg1:object, arg2:object)->builtins.bool:
  d=_resolve_dtype(arg1)
  abstract=_abstract_dtype_set(arg2, for_isdtype=False)
  if abstract is not None:return d in abstract
  try:return d==_resolve_dtype(arg2)
  except NotImplementedError:return False

def isdtype(dtype:object, kind:object)->builtins.bool:
  d=_resolve_dtype(dtype)
  if isinstance(kind, tuple):return builtins.any(isdtype(d, k) for k in kind)
  abstract=_abstract_dtype_set(kind, for_isdtype=True)
  if abstract is not None:return d in abstract
  if kind in(builtins.bool, builtins.int, builtins.float):raise TypeError(f"kind argument must be comprised of monpy dtypes or strings only, but is a {type(kind)!r}")
  try:return d==_resolve_dtype(kind)
  except NotImplementedError as exc:raise TypeError(f"kind argument must be comprised of monpy dtypes or strings only, but is a {type(kind)!r}") from exc

def finfo(dtype:object)->_FInfo:
  d=_resolve_dtype(dtype)
  try:return _FINFO[d]
  except KeyError as exc:raise ValueError(f"data type {d!r} not inexact") from exc

def iinfo(dtype:object)->_IInfo:
  d=_resolve_dtype(dtype)
  try:return _IINFO[d]
  except KeyError as exc:raise ValueError(f"Invalid integer data type {d.kind!r}.") from exc

def array(obj:object, dtype:object=None, *, copy:builtins.bool|None=True, device:object=None)->ndarray:return asarray(obj, dtype=dtype, copy=copy, device=device)

def asarray(obj:object, dtype:object=None, *, copy:builtins.bool|None=None, device:object=None)->ndarray:
  if device is not None and device!="cpu":raise NotImplementedError("monpy v1 only supports cpu arrays")
  t=type(obj)
  if t is ndarray:                                                                                                            # already-ours fast path: skip numpy entirely
    arr=typing.cast(ndarray, obj)
    if dtype is None:return arr.astype(arr.dtype, copy=True) if copy is True else arr
    tgt=_resolve_dtype(dtype)
    if tgt==arr.dtype and copy is not True:return arr
    if copy is False:raise ValueError(_CFE)
    return arr.astype(tgt, copy=True)
  if isinstance(obj, _DeferredArray):return asarray(obj._materialize(), dtype=dtype, copy=copy, device=device)
  if runtime.ops_numpy.is_array_input(obj):return runtime.ops_numpy._from_numpy_unchecked(obj, dtype=dtype, copy=copy, device=device)
  tc=-1 if dtype is None else _resolve_dtype(dtype).code
  cf=-1 if copy is None else (1 if copy else 0)                                                                                # tri-state: -1 default, 0 never, 1 always
  try:
    native=_native.asarray_from_buffer(obj, tc, cf)
    return ndarray(native, owner=None if cf==1 else obj)
  except Exception as exc:
    if copy is False and ("copy" in str(exc) or "readonly" in str(exc)):
      raise ValueError(str(exc)) from exc
    pass
  if _has_ai(obj):return _ai_asarray(obj, dtype=dtype, copy=copy)
  if copy is False:raise ValueError(_CFE)
  shape, flat=_flat(obj)                                                                                                      # nested list/tuple → (shape, flat values)
  tgt=_resolve_dtype(dtype) if dtype is not None else _infer_dtype(flat)
  return ndarray(_native.from_flat(flat, shape, tgt.code))

def empty(shape:int|Sequence[int], dtype:object=None, *, device:object=None)->ndarray:
  _check_cpu(device)
  t=_resolve_dtype(dtype) if dtype is not None else float64
  n=_norm_shape(shape)
  return ndarray(_native.empty(n, t.code))

def zeros(shape:int|Sequence[int], dtype:object=None, *, device:object=None)->ndarray:return full(shape, 0, dtype=_resolve_dtype(dtype) if dtype is not None else float64, device=device)
def ones(shape:int|Sequence[int], dtype:object=None, *, device:object=None)->ndarray:return full(shape, 1, dtype=_resolve_dtype(dtype) if dtype is not None else float64, device=device)

def full(shape:int|Sequence[int], fill_value:object, *, dtype:object=None, device:object=None)->ndarray:
  _check_cpu(device)
  t=_resolve_dtype(dtype) if dtype is not None else _infer_dtype([fill_value])
  n=_norm_shape(shape)
  return ndarray(_native.full(n, fill_value, t.code))

def empty_like(prototype:object, dtype:object=None, order:str="K", subok:builtins.bool=True, shape:int|Sequence[int]|None=None, *, device:object=None)->ndarray:
  _check_order(order)
  del subok
  arr=asarray(prototype)
  t=_resolve_dtype(dtype) if dtype is not None else arr.dtype
  return empty(arr.shape if shape is None else _norm_shape(shape), dtype=t, device=device)

def _full_like(prototype:object, fill_value:object, dtype:object, shape:int|Sequence[int]|None, device:object)->ndarray:
  _check_cpu(device)
  arr=prototype if type(prototype) is ndarray else asarray(prototype)
  if shape is not None:
    t=_resolve_dtype(dtype) if dtype is not None else arr.dtype
    return full(_norm_shape(shape), fill_value, dtype=t, device=device)
  if dtype is None:return ndarray._wrap(_native.full_like(arr._native, fill_value, -1))
  t=_resolve_dtype(dtype)
  return ndarray._wrap(_native.full_like(arr._native, fill_value, t.code))

def zeros_like(prototype:object, dtype:object=None, order:str="K", subok:builtins.bool=True, shape:int|Sequence[int]|None=None, *, device:object=None)->ndarray:
  _check_order(order)
  del subok
  return _full_like(prototype, 0, dtype, shape, device)

def ones_like(prototype:object, dtype:object=None, order:str="K", subok:builtins.bool=True, shape:int|Sequence[int]|None=None, *, device:object=None)->ndarray:
  _check_order(order)
  del subok
  return _full_like(prototype, 1, dtype, shape, device)

def full_like(prototype:object, fill_value:object, dtype:object=None, order:str="K", subok:builtins.bool=True, shape:int|Sequence[int]|None=None, *, device:object=None)->ndarray:
  _check_order(order)
  del subok
  return _full_like(prototype, fill_value, dtype, shape, device)

def arange(start:int|float, stop:int|float|None=None, step:int|float=1, *, dtype:object=None, device:object=None)->ndarray:
  _check_cpu(device)
  a=0 if stop is None else start
  b=start if stop is None else stop                                         # 1-arg form: stop=start, start=0 (numpy convention)
  t=_resolve_dtype(dtype) if dtype is not None else (float64 if builtins.any(isinstance(v, builtins.float) for v in(a, b, step)) else int64)
  return ndarray(_native.arange(a, b, step, t.code))

def linspace(start:int|float, stop:int|float, num:int=50, *, dtype:object=None, device:object=None)->ndarray:
  _check_cpu(device)
  t=_resolve_dtype(dtype) if dtype is not None else float64
  return ndarray(_native.linspace(start, stop, num, t.code))

# Most are pure python composition on top of existing primitives (`zeros`, `arange`, `linspace`, `reshape`, `broadcast_to`).
# They run at python-call speed which is fine for helpers that aren't in the inner loop.
def eye(N:int, M:int|None=None, k:int=0, dtype:object=None, *, device:object=None)->ndarray:
  _check_cpu(device)
  t=_resolve_dtype(dtype) if dtype is not None else float64
  m=N if M is None else M
  return ndarray(_native.eye(N, m, k, t.code))

def identity(n:int, dtype:object=None, *, device:object=None)->ndarray:return eye(n, n, 0, dtype=dtype, device=device)

def tri(N:int, M:int|None=None, k:int=0, dtype:object=None, *, device:object=None)->ndarray:
  _check_cpu(device)
  t=_resolve_dtype(dtype) if dtype is not None else float64
  m=N if M is None else M
  return ndarray(_native.tri(N, m, k, t.code))

def atleast_1d(*arys:object)->ndarray|tuple[ndarray, ...]:
  def _bump(a:object)->ndarray:
    arr=asarray(a)
    return arr if arr.ndim>=1 else arr.reshape((1,))
  res=tuple(_bump(a) for a in arys)
  return res[0] if len(res)==1 else res

def atleast_2d(*arys:object)->ndarray|tuple[ndarray, ...]:
  def _bump(a:object)->ndarray:
    arr=asarray(a)
    if arr.ndim==0:return arr.reshape((1, 1))
    if arr.ndim==1:return ndarray._wrap(_native.expand_dims(arr._native, 0), base=arr)
    return arr
  res=tuple(_bump(a) for a in arys)
  return res[0] if len(res)==1 else res

def atleast_3d(*arys:object)->ndarray|tuple[ndarray, ...]:
  def _bump(a:object)->ndarray:
    arr=asarray(a)
    if arr.ndim==0:return arr.reshape((1, 1, 1))
    if arr.ndim==1:return arr.reshape((1, arr.shape[0], 1))
    if arr.ndim==2:return arr.reshape((arr.shape[0], arr.shape[1], 1))
    return arr
  res=tuple(_bump(a) for a in arys)
  return res[0] if len(res)==1 else res

def logspace(start:float, stop:float, num:int=50, endpoint:builtins.bool=True, base:float=10.0, *, dtype:object=None, device:object=None)->ndarray:
  # The full numpy `axis` keyword is deferred — monpy v1 always emits 1D.
  _check_cpu(device)
  t=_resolve_dtype(dtype) if dtype is not None else float64
  return ndarray(_native.logspace(start, stop, num, endpoint, base, t.code))

def geomspace(start:float, stop:float, num:int=50, endpoint:builtins.bool=True, *, dtype:object=None, device:object=None)->ndarray:
  if start<=0 or stop<=0:raise ValueError("geomspace requires positive start/stop")
  _check_cpu(device)
  t=_resolve_dtype(dtype) if dtype is not None else float64
  if num==0:return asarray([], dtype=t)
  if num==1:return asarray([float(start)], dtype=t)
  log_start=math.log(start)
  log_stop=math.log(stop)
  if endpoint:logs=[log_start+(log_stop-log_start)*i/(num-1) for i in range(num)]
  else:logs=[log_start+(log_stop-log_start)*i/num for i in range(num)]
  vals=[math.exp(L) for L in logs]
  return asarray(vals, dtype=t)

def meshgrid(*xi:object, indexing:str="xy", sparse:builtins.bool=False, copy:builtins.bool=True)->tuple[ndarray, ...]:
  if indexing not in("xy", "ij"):raise ValueError("indexing must be 'xy' or 'ij'")
  if sparse:raise NotImplementedError("sparse meshgrid not implemented in monpy v1")
  del copy  # all monpy view ops produce the same lifetime semantics; copy=True/False is a no-op for our shape model
  arrs=[asarray(x) for x in xi]
  n=len(arrs)
  if n==0:return ()
  for a in arrs:
    if a.ndim>1:raise ValueError("meshgrid input must be 1D")
  shapes=[a.shape[0] for a in arrs]
  out_shape=list(shapes)
  if indexing=="xy" and n>=2:out_shape[0], out_shape[1]=out_shape[1], out_shape[0]
  results:list[ndarray]=[]
  for i, a in enumerate(arrs):
    target_axis=i
    if indexing=="xy" and n>=2:
      if i==0:target_axis=1
      elif i==1:target_axis=0
    bcast_shape=[1]*n
    bcast_shape[target_axis]=shapes[i]
    bcast=a.reshape(tuple(bcast_shape))
    results.append(broadcast_to(bcast, tuple(out_shape)))
  return tuple(results)

def indices(dimensions:Sequence[int], dtype:object=None, sparse:builtins.bool=False)->ndarray|tuple[ndarray, ...]:
  if sparse:raise NotImplementedError("sparse indices not implemented in monpy v1")
  t=_resolve_dtype(dtype) if dtype is not None else int64
  if t!=int64:raise NotImplementedError("indices dtype must be int64 in monpy v1")
  dims=tuple(dimensions)
  n=len(dims)
  if n==0:return zeros((0,), dtype=t)
  return ndarray(_native.indices(dims, t.code))

def ix_(*args:object)->tuple[ndarray, ...]:
  arrs=[asarray(a) for a in args]
  n=len(arrs)
  results:list[ndarray]=[]
  for i, a in enumerate(arrs):
    if a.ndim!=1:raise ValueError("ix_ inputs must be 1D")
    bcast=[1]*n
    bcast[i]=a.shape[0]
    results.append(a.reshape(tuple(bcast)))
  return tuple(results)

def reshape(x:object, shape:int|Sequence[int])->ndarray:
  if _is_kernel_value(x):
    from .kernels import dsl as _kernel_dsl
    return typing.cast(ndarray, _kernel_dsl.reshape(x, shape))
  return asarray(x).reshape(shape)

def squeeze(x:object, axis:int|Sequence[int]|None=None)->ndarray:
  arr=asarray(x)
  try:
    if axis is None:native=_native.squeeze_all(arr._native)
    elif isinstance(axis, builtins.int):native=_native.squeeze_axis(arr._native, axis)
    else:
      native=_native.squeeze_axes(arr._native, tuple(axis))
  except Exception as exc:raise ValueError(str(exc)) from exc
  return ndarray._wrap(native, base=arr)

def moveaxis(x:object, source:int|Sequence[int], destination:int|Sequence[int])->ndarray:
  arr=asarray(x)
  n=arr.ndim
  src=(source,) if isinstance(source, builtins.int) else tuple(source)
  dst=(destination,) if isinstance(destination, builtins.int) else tuple(destination)
  if len(src)!=len(dst):raise ValueError("moveaxis: source/destination length mismatch")
  src_norm=tuple((s+n if s<0 else s) for s in src)
  dst_norm=tuple((d+n if d<0 else d) for d in dst)
  for axis_set in (src_norm, dst_norm):
    if builtins.any(a<0 or a>=n for a in axis_set):raise ValueError("moveaxis: axis out of range")
    if len(set(axis_set))!=len(axis_set):raise ValueError("moveaxis: repeated axis")
  remaining=[i for i in range(n) if i not in set(src_norm)]
  perm=[-1]*n
  for s, d in builtins.zip(src_norm, dst_norm):perm[d]=s
  next_remain=0
  for i in range(n):
    if perm[i]==-1:
      perm[i]=remaining[next_remain]
      next_remain+=1
  return arr.transpose(tuple(perm))

def swapaxes(x:object, axis1:int, axis2:int)->ndarray:
  arr=asarray(x)
  try:return ndarray._wrap(_native.swapaxes(arr._native, axis1, axis2), base=arr)
  except Exception as exc:raise ValueError(str(exc)) from exc

def ravel(x:object, order:str="C")->ndarray:
  if order not in("C", "K", "A"):raise NotImplementedError("ravel order != C/K/A")
  arr=asarray(x)
  return ndarray._wrap(_native.ravel(arr._native), base=arr)

def flatten(x:object, order:str="C")->ndarray:
  if order not in("C", "K", "A"):raise NotImplementedError("flatten order != C/K/A")
  arr=asarray(x)
  return ndarray._wrap(_native.flatten(arr._native))

# concatenate/stack family.
# collect element values via .tolist() and reconstruct via asarray.
# Correct for any shape but slow per-element on the python side.
# A native concatenate kernel (LayoutIter walks output and copies from each input) is the follow-up.
def _concat_into_flat(arrs:Sequence[ndarray], axis:int)->tuple[list[object], tuple[int, ...]]:
  if not arrs:raise ValueError("concatenate: need at least one array")
  ref=arrs[0]
  n=ref.ndim
  ax=axis+n if axis<0 else axis
  if ax<0 or ax>=n:raise ValueError("concatenate: axis out of range")
  for a in arrs:
    if a.ndim!=n:raise ValueError("concatenate: arrays must have same number of dimensions")
    for d in range(n):
      if d==ax:continue
      if a.shape[d]!=ref.shape[d]:raise ValueError(f"concatenate: shape mismatch on axis {d}")
  out_shape=list(ref.shape)
  out_shape[ax]=builtins.sum(a.shape[ax] for a in arrs)
  # Build output by python-level slicing: walk every output position.
  # Skip-the-coord algorithm for correctness over perf.
  total=1
  for d in out_shape:total*=d
  flat:list[object]=[None]*total
  # Compute output strides for row-major.
  out_strides=[0]*n
  running=1
  for d in range(n-1, -1, -1):
    out_strides[d]=running
    running*=out_shape[d]
  # Walk each input and copy.
  cursor=0
  for a in arrs:
    a_size=a.size
    a_view=ascontiguousarray(a).reshape((a_size,))
    a_shape=a.shape
    a_strides=[0]*n
    rk=1
    for d in range(n-1, -1, -1):
      a_strides[d]=rk
      rk*=a_shape[d]
    for i in range(a_size):
      # decode input multi-index via row-major
      coord=[0]*n
      remain=i
      for d in range(n):
        coord[d]=remain//a_strides[d]
        remain%=a_strides[d]
      # offset on the concatenation axis
      coord[ax]+=cursor
      out_idx=builtins.sum(coord[d]*out_strides[d] for d in range(n))
      flat[out_idx]=a_view[i]
    cursor+=a.shape[ax]
  return flat, tuple(out_shape)

def concatenate(arrays:Sequence[object], axis:int=0, *, out:object=None, dtype:object=None, casting:str="same_kind")->ndarray:
  if out is not None:raise NotImplementedError("concatenate out= not implemented in monpy v1")
  arrs=[asarray(a) for a in arrays]
  if not arrs:raise ValueError("concatenate: need at least one array")
  natives=[a._native for a in arrs]
  if dtype is None:
    try:return ndarray._wrap(_native.concatenate(natives, axis, -1))
    except Exception as exc:
      message=str(exc)
      if "dtype mismatch" not in message and "c-contiguous" not in message:raise
  t=_resolve_dtype(dtype) if dtype is not None else result_type(*arrs)
  # Pre-cast all inputs and pre-materialise into c-contig — native concat
  # walks logical indices, but the f64 round-trip path is faster on
  # contiguous data so we hand it that.
  arrs_cast=[ascontiguousarray(a if a.dtype==t else a.astype(t)) for a in arrs]
  return ndarray(_native.concatenate([a._native for a in arrs_cast], axis, t.code))

def stack(arrays:Sequence[object], axis:int=0, *, out:object=None, dtype:object=None)->ndarray:
  if out is not None:raise NotImplementedError("stack out= not implemented in monpy v1")
  arrs=[asarray(a) for a in arrays]
  if not arrs:raise ValueError("stack: need at least one array")
  if dtype is None:
    ax0=axis+arrs[0].ndim+1 if axis<0 else axis
    if ax0==0:
      try:return ndarray._wrap(_native.stack_axis0([a._native for a in arrs], builtins.int(arrs[0]._native.dtype_code()), False))
      except Exception:pass
  ref=arrs[0]
  ref_shape=ref.shape
  for a in arrs:
    if a.shape!=ref_shape:raise ValueError("stack: arrays must have identical shape")
  n=ref.ndim
  ax=axis+n+1 if axis<0 else axis
  if ax<0 or ax>n:raise ValueError("stack: axis out of range")
  if ax==0 and n>0:
    flat=concatenate(arrs, axis=0, dtype=dtype)
    return flat.reshape((len(arrs),)+ref_shape)
  expanded=[expand_dims(a, ax) for a in arrs]
  return concatenate(expanded, axis=ax, dtype=dtype)

def hstack(arrays:Sequence[object])->ndarray:
  arrs=[asarray(a) for a in arrays]
  if not arrs:raise ValueError("hstack: empty input")
  if arrs[0].ndim<=1:return concatenate(arrs, axis=0)
  return concatenate(arrs, axis=1)

def vstack(arrays:Sequence[object])->ndarray:
  raw=[asarray(a) for a in arrays]
  if raw:
    try:return ndarray._wrap(_native.stack_axis0([a._native for a in raw], builtins.int(raw[0]._native.dtype_code()), True))
    except Exception:pass
  if raw and builtins.all(a.ndim==1 for a in raw):
    row_width=raw[0].shape[0]
    if builtins.all(a.shape[0]==row_width for a in raw):
      flat=concatenate(raw, axis=0)
      return flat.reshape((len(raw), row_width))
  arrs=[atleast_2d(a) for a in raw]
  return concatenate(arrs, axis=0)

def dstack(arrays:Sequence[object])->ndarray:
  arrs=[atleast_3d(a) for a in arrays]
  return concatenate(arrs, axis=2)

def column_stack(arrays:Sequence[object])->ndarray:
  arrs:list[ndarray]=[]
  for a in arrays:
    arr=asarray(a)
    if arr.ndim==1:arr=arr.reshape((arr.shape[0], 1))
    arrs.append(arr)
  return concatenate(arrs, axis=1)

def _split_indices(arr_size:int, indices_or_sections:int|Sequence[int], *, allow_uneven:builtins.bool)->list[tuple[int, int]]:
  if isinstance(indices_or_sections, builtins.int):
    n=indices_or_sections
    if n<=0:raise ValueError("split: number of sections must be > 0")
    if not allow_uneven and arr_size%n!=0:raise ValueError("split: array does not divide evenly")
    base, rem=divmod(arr_size, n)
    chunks:list[tuple[int, int]]=[]
    start=0
    for i in range(n):
      extra=1 if i<rem else 0
      end=start+base+extra
      chunks.append((start, end))
      start=end
    return chunks
  pts=[builtins.int(p) for p in indices_or_sections]
  prev=0
  chunks=[]
  for p in pts:
    chunks.append((prev, p))
    prev=p
  chunks.append((prev, arr_size))
  return chunks

def split(ary:object, indices_or_sections:int|Sequence[int], axis:int=0)->list[ndarray]:
  arr=asarray(ary)
  n=arr.ndim
  ax=axis+n if axis<0 else axis
  if ax<0 or ax>=n:raise ValueError("split: axis out of range")
  ranges=_split_indices(arr.shape[ax], indices_or_sections, allow_uneven=False)
  out:list[ndarray]=[]
  for start, end in ranges:
    key=tuple(slice(start, end) if d==ax else slice(None) for d in range(n))
    out.append(typing.cast(ndarray, arr[key]))
  return out

def array_split(ary:object, indices_or_sections:int|Sequence[int], axis:int=0)->list[ndarray]:
  arr=asarray(ary)
  n=arr.ndim
  ax=axis+n if axis<0 else axis
  if ax<0 or ax>=n:raise ValueError("array_split: axis out of range")
  ranges=_split_indices(arr.shape[ax], indices_or_sections, allow_uneven=True)
  out:list[ndarray]=[]
  for start, end in ranges:
    key=tuple(slice(start, end) if d==ax else slice(None) for d in range(n))
    out.append(typing.cast(ndarray, arr[key]))
  return out

def hsplit(ary:object, indices_or_sections:int|Sequence[int])->list[ndarray]:
  arr=asarray(ary)
  if arr.ndim==1:return split(arr, indices_or_sections, axis=0)
  return split(arr, indices_or_sections, axis=1)

def vsplit(ary:object, indices_or_sections:int|Sequence[int])->list[ndarray]:
  arr=asarray(ary)
  if arr.ndim<2:raise ValueError("vsplit: array must have at least 2 dimensions")
  return split(arr, indices_or_sections, axis=0)

def dsplit(ary:object, indices_or_sections:int|Sequence[int])->list[ndarray]:
  arr=asarray(ary)
  if arr.ndim<3:raise ValueError("dsplit: array must have at least 3 dimensions")
  return split(arr, indices_or_sections, axis=2)
def transpose(x:object, axes:Sequence[int]|None=None)->ndarray:
  if _is_kernel_value(x):
    from .kernels import dsl as _kernel_dsl
    return typing.cast(ndarray, _kernel_dsl.transpose(x, axes))
  return asarray(x).transpose(axes)
def matrix_transpose(x:object)->ndarray:return asarray(x).mT
def broadcast_to(x:object, shape:int|Sequence[int])->ndarray:
  if _is_kernel_value(x):
    from .kernels import dsl as _kernel_dsl
    return typing.cast(ndarray, _kernel_dsl.broadcast_to(x, shape))
  a=asarray(x)
  return ndarray(_native.broadcast_to(a._native, _norm_shape(shape)), base=a)
def flip(m:object, axis:int|Sequence[int]|None=None)->ndarray:
  arr=asarray(m)
  if axis is None:axes:tuple[int, ...]=()                                                                                       # () means flip all axes (mojo flip_ops convention)
  elif isinstance(axis, builtins.int):axes=(axis,)
  else:axes=tuple(axis)
  return ndarray(_native.flip(arr._native, axes), base=arr)

def fliplr(m:object)->ndarray:
  arr=asarray(m)
  if arr.ndim<2:raise ValueError("fliplr() requires array with at least 2 dimensions")
  return flip(arr, axis=1)

def flipud(m:object)->ndarray:
  arr=asarray(m)
  if arr.ndim<1:raise ValueError("flipud() requires array with at least 1 dimension")
  return flip(arr, axis=0)

def rot90(m:object, k:int=1, axes:tuple[int, int]=(0, 1))->ndarray:
  arr=asarray(m)
  if arr.ndim<2:raise ValueError("rot90() requires array with at least 2 dimensions")
  ax0, ax1=axes
  if ax0<0:ax0+=arr.ndim
  if ax1<0:ax1+=arr.ndim
  if ax0==ax1 or ax0<0 or ax1<0 or ax0>=arr.ndim or ax1>=arr.ndim:raise ValueError("rot90() axes must be different and in bounds")
  k=k%4
  if k==0:return arr
  if k==2:return flip(flip(arr, axis=ax0), axis=ax1)
  perm=list(range(arr.ndim))
  perm[ax0], perm[ax1]=perm[ax1], perm[ax0]
  if k==1:return transpose(flip(arr, axis=ax1), tuple(perm))
  return flip(transpose(arr, tuple(perm)), axis=ax1)                                                                              # k==3

def expand_dims(a:object, axis:int|Sequence[int])->ndarray:
  arr=asarray(a)
  axes=_axis_tuple(axis)
  ndim=arr.ndim+len(axes)
  norm:list[int]=[]
  for ax in axes:
    n=ax+ndim if ax<0 else ax
    if n<0 or n>=ndim:raise ValueError("axis out of bounds")
    norm.append(n)
  if len(set(norm))!=len(norm):raise ValueError("repeated axis")
  out=arr
  for ax in sorted(norm):out=ndarray(_native.expand_dims(out._native, ax), base=out)
  return out

# ufunc instances.
# Every public ufunc routes through `Ufunc`, which delegates to the right native dispatch
# (`_native.binary` / `unary` / `unary_preserve` / `compare` / `logical` / `predicate`) and applies dtype promotion + casting + out= + reduce/accumulate/outer/at semantics.
add=Ufunc("add", 2, 1, "binary", OP_ADD, identity=0, reduce_op=REDUCE_SUM)
subtract=Ufunc("subtract", 2, 1, "binary", OP_SUB, reduce_op=None)
multiply=Ufunc("multiply", 2, 1, "binary", OP_MUL, identity=1, reduce_op=REDUCE_PROD)
divide=Ufunc("divide", 2, 1, "binary", OP_DIV, reduce_op=None)
true_divide=divide
floor_divide=Ufunc("floor_divide", 2, 1, "binary", OP_FLOOR_DIV, reduce_op=None)
remainder=Ufunc("remainder", 2, 1, "binary", OP_MOD, reduce_op=None)
mod=remainder
power=Ufunc("power", 2, 1, "binary", OP_POWER, identity=1, reduce_op=None)
maximum=Ufunc("maximum", 2, 1, "binary", OP_MAXIMUM, reduce_op=REDUCE_MAX)
minimum=Ufunc("minimum", 2, 1, "binary", OP_MINIMUM, reduce_op=REDUCE_MIN)
fmax=Ufunc("fmax", 2, 1, "binary", OP_FMAX, reduce_op=None)
fmin=Ufunc("fmin", 2, 1, "binary", OP_FMIN, reduce_op=None)
arctan2=Ufunc("arctan2", 2, 1, "binary", OP_ARCTAN2, reduce_op=None)
hypot=Ufunc("hypot", 2, 1, "binary", OP_HYPOT, reduce_op=None)
copysign=Ufunc("copysign", 2, 1, "binary", OP_COPYSIGN, reduce_op=None)

# Transcendental unary (float-result; int → f64 promotion).
sin=Ufunc("sin", 1, 1, "unary", UNARY_SIN)
cos=Ufunc("cos", 1, 1, "unary", UNARY_COS)
tan=Ufunc("tan", 1, 1, "unary", UNARY_TAN)
arcsin=Ufunc("arcsin", 1, 1, "unary", UNARY_ARCSIN)
arccos=Ufunc("arccos", 1, 1, "unary", UNARY_ARCCOS)
arctan=Ufunc("arctan", 1, 1, "unary", UNARY_ARCTAN)
sinh=Ufunc("sinh", 1, 1, "unary", UNARY_SINH)
cosh=Ufunc("cosh", 1, 1, "unary", UNARY_COSH)
tanh=Ufunc("tanh", 1, 1, "unary", UNARY_TANH)
exp=Ufunc("exp", 1, 1, "unary", UNARY_EXP)
exp2=Ufunc("exp2", 1, 1, "unary", UNARY_EXP2)
expm1=Ufunc("expm1", 1, 1, "unary", UNARY_EXPM1)
log=Ufunc("log", 1, 1, "unary", UNARY_LOG)
log2=Ufunc("log2", 1, 1, "unary", UNARY_LOG2)
log10=Ufunc("log10", 1, 1, "unary", UNARY_LOG10)
log1p=Ufunc("log1p", 1, 1, "unary", UNARY_LOG1P)
sqrt=Ufunc("sqrt", 1, 1, "unary", UNARY_SQRT)
cbrt=Ufunc("cbrt", 1, 1, "unary", UNARY_CBRT)
deg2rad=Ufunc("deg2rad", 1, 1, "unary", UNARY_DEG2RAD)
radians=deg2rad
rad2deg=Ufunc("rad2deg", 1, 1, "unary", UNARY_RAD2DEG)
degrees=rad2deg
reciprocal=Ufunc("reciprocal", 1, 1, "unary", UNARY_RECIPROCAL)

# Preserve-dtype unary arith (int → int, float → float; bool → int64).
negative=Ufunc("negative", 1, 1, "unary_preserve", UNARY_NEGATE)
positive=Ufunc("positive", 1, 1, "unary_preserve", UNARY_POSITIVE)
absolute=Ufunc("absolute", 1, 1, "unary_preserve", UNARY_ABS)
abs=absolute
fabs=absolute
square=Ufunc("square", 1, 1, "unary_preserve", UNARY_SQUARE)
sign=Ufunc("sign", 1, 1, "unary_preserve", UNARY_SIGN)
floor=Ufunc("floor", 1, 1, "unary_preserve", UNARY_FLOOR)
ceil=Ufunc("ceil", 1, 1, "unary_preserve", UNARY_CEIL)
trunc=Ufunc("trunc", 1, 1, "unary_preserve", UNARY_TRUNC)
fix=trunc
rint=Ufunc("rint", 1, 1, "unary_preserve", UNARY_RINT)
logical_not=Ufunc("logical_not", 1, 1, "unary_preserve", UNARY_LOGICAL_NOT)
# complex-only ufunc. Conjugate negates imag for complex; identity for real.
conjugate=Ufunc("conjugate", 1, 1, "unary_preserve", UNARY_CONJUGATE)
conj=conjugate

# Real / imag / angle. These are accessor ops that return real-valued
# arrays (component split)—implemented as python wrappers since their
# output dtype differs from the input (complex → real).
# Writes through `_native.from_flat` so the output is c-contig regardless of input strides.
def real(x:object)->ndarray:
  a=_mat(_av(x))
  if a.dtype not in(complex64, complex128):return a  # real input, identity
  rdt=float32 if a.dtype==complex64 else float64
  flat=[a._native.get_scalar(i).real for i in range(a.size)]
  return ndarray(_native.from_flat(flat, a.shape, rdt.code))

def imag(x:object)->ndarray:
  a=_mat(_av(x))
  if a.dtype not in(complex64, complex128):return zeros_like(a)
  rdt=float32 if a.dtype==complex64 else float64
  flat=[a._native.get_scalar(i).imag for i in range(a.size)]
  return ndarray(_native.from_flat(flat, a.shape, rdt.code))

def angle(z:object, deg:builtins.bool=False)->ndarray:
  a=_mat(_av(z))
  if a.dtype not in(complex64, complex128):
    # real input: angle is 0 for non-negatives, pi for negatives.
    rdt=float32 if a.dtype==float32 else float64
    flat=[(math.pi if a._native.get_scalar(i)<0 else 0.0) for i in range(a.size)]
    if deg:flat=[math.degrees(v) for v in flat]
    return ndarray(_native.from_flat(flat, a.shape, rdt.code))
  rdt=float32 if a.dtype==complex64 else float64
  flat=[]
  for i in range(a.size):
    c=a._native.get_scalar(i)
    val=math.atan2(c.imag, c.real)
    if deg:val=math.degrees(val)
    flat.append(val)
  return ndarray(_native.from_flat(flat, a.shape, rdt.code))

# Comparison ufuncs (return bool).
equal=Ufunc("equal", 2, 1, "compare", CMP_EQ)
not_equal=Ufunc("not_equal", 2, 1, "compare", CMP_NE)
less=Ufunc("less", 2, 1, "compare", CMP_LT)
less_equal=Ufunc("less_equal", 2, 1, "compare", CMP_LE)
greater=Ufunc("greater", 2, 1, "compare", CMP_GT)
greater_equal=Ufunc("greater_equal", 2, 1, "compare", CMP_GE)

# Logical ufuncs (operate on truthiness, return bool).
logical_and=Ufunc("logical_and", 2, 1, "logical", LOGIC_AND, identity=True, reduce_op=REDUCE_ALL)
logical_or=Ufunc("logical_or", 2, 1, "logical", LOGIC_OR, identity=False, reduce_op=REDUCE_ANY)
logical_xor=Ufunc("logical_xor", 2, 1, "logical", LOGIC_XOR, identity=False)

# Float predicate ufuncs (return bool).
isnan=Ufunc("isnan", 1, 1, "predicate", PRED_ISNAN)
isinf=Ufunc("isinf", 1, 1, "predicate", PRED_ISINF)
isfinite=Ufunc("isfinite", 1, 1, "predicate", PRED_ISFINITE)
signbit=Ufunc("signbit", 1, 1, "predicate", PRED_SIGNBIT)

def sin_add_mul(x:object, y:object, scalar:object)->ndarray:
  # explicit fused kernel: sin(x) + scalar*y. The implicit pattern sin(x)+(scalar*y) is also caught by _fuse.
  if not _is_scalar(scalar):raise NotImplementedError("sin_add_mul currently requires a Python scalar multiplier")
  lhs=_av(x)
  rhs=_av(y)
  sd=_isd_arr(rhs, scalar)
  la=_mat(lhs)
  ra=_mat(rhs)
  return ndarray(_native.sin_add_mul(la._native, ra._native, scalar, sd.code))

def where(condition:object, x1:object, x2:object)->ndarray:
  c=asarray(condition, dtype=bool)
  l=asarray(x1)
  r=asarray(x2)
  l, r=_coerce(l, r, OP_ADD)                                         # use OP_ADD's promotion table — same numeric rules
  return ndarray(_native.where(c._native, l._native, r._native))

def sum(x:object, axis:object=None, *, dtype:object=None, out:ndarray|None=None, keepdims:builtins.bool=False)->object:return _reduce_dispatch(x, axis, REDUCE_SUM, dtype=dtype, out=out, keepdims=keepdims)
def mean(x:object, axis:object=None, *, dtype:object=None, out:ndarray|None=None, keepdims:builtins.bool=False)->object:return _reduce_dispatch(x, axis, REDUCE_MEAN, dtype=dtype, out=out, keepdims=keepdims)
def min(x:object, axis:object=None, *, out:ndarray|None=None, keepdims:builtins.bool=False)->object:return _reduce_dispatch(x, axis, REDUCE_MIN, dtype=None, out=out, keepdims=keepdims)
def max(x:object, axis:object=None, *, out:ndarray|None=None, keepdims:builtins.bool=False)->object:return _reduce_dispatch(x, axis, REDUCE_MAX, dtype=None, out=out, keepdims=keepdims)
def prod(x:object, axis:object=None, *, dtype:object=None, out:ndarray|None=None, keepdims:builtins.bool=False)->object:return _reduce_dispatch(x, axis, REDUCE_PROD, dtype=dtype, out=out, keepdims=keepdims)
def all(x:object, axis:object=None, *, out:ndarray|None=None, keepdims:builtins.bool=False)->object:return _reduce_dispatch(x, axis, REDUCE_ALL, dtype=None, out=out, keepdims=keepdims)
def any(x:object, axis:object=None, *, out:ndarray|None=None, keepdims:builtins.bool=False)->object:return _reduce_dispatch(x, axis, REDUCE_ANY, dtype=None, out=out, keepdims=keepdims)
def argmax(x:object, axis:object=None, *, keepdims:builtins.bool=False)->object:return _reduce_dispatch(x, axis, REDUCE_ARGMAX, dtype=None, out=None, keepdims=keepdims)
def argmin(x:object, axis:object=None, *, keepdims:builtins.bool=False)->object:return _reduce_dispatch(x, axis, REDUCE_ARGMIN, dtype=None, out=None, keepdims=keepdims)
def count_nonzero(x:object, axis:object=None, *, keepdims:builtins.bool=False)->object:
  arr=asarray(x)
  mask=not_equal(arr, 0)
  return _reduce_dispatch(mask, axis, REDUCE_SUM, dtype=int64, out=None, keepdims=keepdims)
def ptp(x:object, axis:object=None, *, out:ndarray|None=None, keepdims:builtins.bool=False)->object:
  hi=max(x, axis=axis, keepdims=keepdims)
  lo=min(x, axis=axis, keepdims=keepdims)
  return subtract(hi, lo, out=out)

def _reduce_dispatch(x:object, axis:object, op:int, *, dtype:object, out:ndarray|None, keepdims:builtins.bool)->object:
  arr=_mat(_av(x))
  if dtype is not None:
    t=_resolve_dtype(dtype)
    if arr.dtype!=t:arr=arr.astype(t)
  if axis is None:
    res=ndarray(_native.reduce(arr._native, op))
    if keepdims:
      keep=tuple(1 for _ in range(arr.ndim))
      v=res.reshape(keep) if keep else res
    else:
      v=res._scalar() if res.ndim==0 else res
    if out is not None:
      out[...]=v
      return out
    return v
  axes=_axis_tuple(axis)
  res=ndarray(_native.reduce_axis(arr._native, op, tuple(builtins.int(a) for a in axes), builtins.bool(keepdims)))
  if out is not None:
    out[...]=res
    return out
  return res

def std(x:object, axis:object=None, *, ddof:int=0, dtype:object=None, out:ndarray|None=None, keepdims:builtins.bool=False)->object:
  arr=_mat(_av(x))
  t=_resolve_dtype(dtype) if dtype is not None else (arr.dtype if arr.dtype in(float32, float64) else float64)
  if arr.dtype!=t:arr=arr.astype(t)
  m=mean(arr, axis=axis, keepdims=True)
  diff=subtract(arr, m)
  sq=multiply(diff, diff)
  s=sum(sq, axis=axis, keepdims=keepdims)
  n=arr.size if axis is None else math.prod(arr.shape[a] for a in _axis_tuple(axis))
  div=builtins.max(0, n-ddof)
  if div==0:res=multiply(s, float("nan"))
  else:res=sqrt(divide(s, div))
  if out is not None:
    out[...]=res
    return out
  return res

def var(x:object, axis:object=None, *, ddof:int=0, dtype:object=None, out:ndarray|None=None, keepdims:builtins.bool=False)->object:
  arr=_mat(_av(x))
  t=_resolve_dtype(dtype) if dtype is not None else (arr.dtype if arr.dtype in(float32, float64) else float64)
  if arr.dtype!=t:arr=arr.astype(t)
  m=mean(arr, axis=axis, keepdims=True)
  diff=subtract(arr, m)
  sq=multiply(diff, diff)
  s=sum(sq, axis=axis, keepdims=keepdims)
  n=arr.size if axis is None else math.prod(arr.shape[a] for a in _axis_tuple(axis))
  div=builtins.max(0, n-ddof)
  if div==0:res=multiply(s, float("nan"))
  else:res=divide(s, div)
  if out is not None:
    out[...]=res
    return out
  return res

def average(x:object, axis:object=None, weights:object=None, returned:builtins.bool=False, *, keepdims:builtins.bool=False)->object:
  if weights is None:return mean(x, axis=axis, keepdims=keepdims)
  arr=asarray(x)
  w=asarray(weights)
  if w.shape!=arr.shape and axis is not None and isinstance(axis, builtins.int):
    if w.ndim==1 and w.shape[0]==arr.shape[axis]:
      shape=[1]*arr.ndim
      shape[axis]=w.shape[0]
      w=w.reshape(tuple(shape))
  num=sum(multiply(arr, w), axis=axis, keepdims=keepdims)
  den=sum(w, axis=axis, keepdims=keepdims) if w.shape!=arr.shape and not keepdims else sum(broadcast_to(w, arr.shape), axis=axis, keepdims=keepdims)
  res=divide(num, den)
  if returned:return res, den
  return res

def median(x:object, axis:object=None, *, keepdims:builtins.bool=False)->object:return quantile(x, 0.5, axis=axis, keepdims=keepdims)

def quantile(x:object, q:object, axis:object=None, *, keepdims:builtins.bool=False, method:str="linear")->object:
  if method!="linear":raise NotImplementedError("monpy v1 only supports method='linear'")
  arr=_mat(_av(x))
  if axis is None:
    flat=arr.reshape((arr.size,)) if arr.ndim>0 else arr.reshape((1,))
    vals=sorted(builtins.float(flat._native.get_scalar(i)) for i in range(flat.size))
    return _quantile_from_sorted(vals, q, keepdims_axes=(), orig_shape=arr.shape) if keepdims else _quantile_from_sorted(vals, q, keepdims_axes=None, orig_shape=arr.shape)
  ax=_axis_int(axis, arr.ndim, "quantile")
  moved=ascontiguousarray(moveaxis(arr, ax, -1))
  pref=moved.shape[:-1]
  axis_size=moved.shape[-1]
  flat=moved.reshape((math.prod(pref) if pref else 1, axis_size))
  rows=flat.shape[0]
  qs=q if isinstance(q, (list, tuple)) else [q]
  out_shape=(len(qs),)+pref if isinstance(q, (list, tuple)) else pref
  if keepdims:
    out_shape=(out_shape if isinstance(q, (list, tuple)) else out_shape)+(1,)
  flat_out:list[float]=[]
  for k_q in qs:
    for r in range(rows):
      vals=sorted(builtins.float(flat._native.get_scalar(r*axis_size+i)) for i in range(axis_size))
      flat_out.append(_py_float(_quantile_from_sorted(vals, k_q, keepdims_axes=None, orig_shape=())))
  if not isinstance(q, (list, tuple)):
    if keepdims:return ndarray(_native.from_flat(flat_out, out_shape, float64.code))
    if not pref:return flat_out[0]
    return ndarray(_native.from_flat(flat_out, pref, float64.code))
  if keepdims:return ndarray(_native.from_flat(flat_out, out_shape, float64.code))
  return ndarray(_native.from_flat(flat_out, out_shape, float64.code))

def _quantile_from_sorted(sorted_vals:Sequence[float], q:object, *, keepdims_axes:object, orig_shape:tuple[int, ...])->object:
  qf=_py_float(q)
  n=len(sorted_vals)
  if n==0:raise ValueError("quantile of empty array")
  pos=qf*(n-1)
  lo=builtins.int(pos)
  hi=builtins.min(lo+1, n-1)
  frac=pos-lo
  v=sorted_vals[lo]*(1-frac)+sorted_vals[hi]*frac
  if keepdims_axes is None:return v
  shape=tuple(1 for _ in orig_shape)
  return ndarray(_native.from_flat([v], shape, float64.code)) if shape else v

def percentile(x:object, q:object, axis:object=None, *, keepdims:builtins.bool=False, method:str="linear")->object:
  if isinstance(q, (list, tuple)):qf=[_py_float(v)/100.0 for v in q]
  else:qf=_py_float(q)/100.0
  return quantile(x, qf, axis=axis, keepdims=keepdims, method=method)

def cumsum(x:object, axis:object=None, *, dtype:object=None, out:ndarray|None=None)->object:return _cumulative_dispatch(x, axis, "sum", dtype=dtype, out=out)
def cumprod(x:object, axis:object=None, *, dtype:object=None, out:ndarray|None=None)->object:return _cumulative_dispatch(x, axis, "prod", dtype=dtype, out=out)

def _cumulative_dispatch(x:object, axis:object, kind:str, *, dtype:object, out:ndarray|None)->object:
  arr=_mat(_av(x))
  if dtype is not None:
    t=_resolve_dtype(dtype)
    if arr.dtype!=t:arr=arr.astype(t)
  if axis is None:
    flat=arr.reshape((arr.size,)) if arr.ndim>0 else arr.reshape((1,))
    n=flat.size
    if kind=="sum":vals=[builtins.float(flat._native.get_scalar(0))]
    else:vals=[builtins.float(flat._native.get_scalar(0))]
    for i in range(1, n):
      cur=builtins.float(flat._native.get_scalar(i))
      vals.append(vals[-1]+cur if kind=="sum" else vals[-1]*cur)
    res=ndarray(_native.from_flat(vals, (n,), arr.dtype.code))
  else:
    ax=_axis_int(axis, arr.ndim, "cumulative")
    moved=ascontiguousarray(moveaxis(arr, ax, -1))
    pref=moved.shape[:-1]
    axis_size=moved.shape[-1]
    rows=math.prod(pref) if pref else 1
    flat=moved.reshape((rows, axis_size))
    out_vals:list[object]=[]
    for r in range(rows):
      acc=builtins.float(flat._native.get_scalar(r*axis_size+0))
      out_vals.append(acc)
      for i in range(1, axis_size):
        cur=builtins.float(flat._native.get_scalar(r*axis_size+i))
        acc=acc+cur if kind=="sum" else acc*cur
        out_vals.append(acc)
    res=ndarray(_native.from_flat(out_vals, moved.shape, arr.dtype.code))
    inv=_inverse_moveaxis_perm(ax, -1, arr.ndim)
    res=res.transpose(inv) if arr.ndim>1 else res
  if out is not None:
    out[...]=res
    return out
  return res

def cummax(x:object, axis:object=None, *, out:ndarray|None=None)->object:return _cumulative_minmax(x, axis, "max", out=out)
def cummin(x:object, axis:object=None, *, out:ndarray|None=None)->object:return _cumulative_minmax(x, axis, "min", out=out)

def _cumulative_minmax(x:object, axis:object, kind:str, *, out:ndarray|None)->object:
  arr=_mat(_av(x))
  if axis is None:
    n=arr.size
    flat=arr.reshape((n,)) if arr.ndim>0 else arr.reshape((1,))
    vals=[builtins.float(flat._native.get_scalar(0))]
    for i in range(1, n):
      cur=builtins.float(flat._native.get_scalar(i))
      vals.append(builtins.max(vals[-1], cur) if kind=="max" else builtins.min(vals[-1], cur))
    res=ndarray(_native.from_flat(vals, (n,), arr.dtype.code))
  else:
    ax=_axis_int(axis, arr.ndim, "cumulative")
    moved=ascontiguousarray(moveaxis(arr, ax, -1))
    pref=moved.shape[:-1]
    axis_size=moved.shape[-1]
    rows=math.prod(pref) if pref else 1
    flat=moved.reshape((rows, axis_size))
    out_vals:list[object]=[]
    for r in range(rows):
      acc=builtins.float(flat._native.get_scalar(r*axis_size+0))
      out_vals.append(acc)
      for i in range(1, axis_size):
        cur=builtins.float(flat._native.get_scalar(r*axis_size+i))
        acc=builtins.max(acc, cur) if kind=="max" else builtins.min(acc, cur)
        out_vals.append(acc)
    res=ndarray(_native.from_flat(out_vals, moved.shape, arr.dtype.code))
    inv=_inverse_moveaxis_perm(ax, -1, arr.ndim)
    res=res.transpose(inv) if arr.ndim>1 else res
  if out is not None:
    out[...]=res
    return out
  return res

# NaN-aware reductions; treat NaN as missing.
def nansum(x:object, axis:object=None, *, dtype:object=None, out:ndarray|None=None, keepdims:builtins.bool=False)->object:
  arr=asarray(x)
  mask=isnan(arr) if arr.dtype in(float32, float64) else None
  if mask is None:return sum(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
  zero=zeros_like(arr)
  clean=where(mask, zero, arr)
  return sum(clean, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

def nanmean(x:object, axis:object=None, *, dtype:object=None, out:ndarray|None=None, keepdims:builtins.bool=False)->object:
  arr=asarray(x)
  if arr.dtype not in(float32, float64):return mean(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
  mask=isnan(arr)
  zero=zeros_like(arr)
  clean=where(mask, zero, arr)
  s=sum(clean, axis=axis, dtype=dtype, keepdims=keepdims)
  cnt=sum(logical_not(mask), axis=axis, dtype=int64, keepdims=keepdims)
  res=divide(s, cnt)
  if out is not None:
    out[...]=res
    return out
  return res

def nanmin(x:object, axis:object=None, *, out:ndarray|None=None, keepdims:builtins.bool=False)->object:
  arr=asarray(x)
  if arr.dtype not in(float32, float64):return min(arr, axis=axis, out=out, keepdims=keepdims)
  mask=isnan(arr)
  big=full_like(arr, float("inf"))
  clean=where(mask, big, arr)
  return min(clean, axis=axis, out=out, keepdims=keepdims)

def nanmax(x:object, axis:object=None, *, out:ndarray|None=None, keepdims:builtins.bool=False)->object:
  arr=asarray(x)
  if arr.dtype not in(float32, float64):return max(arr, axis=axis, out=out, keepdims=keepdims)
  mask=isnan(arr)
  small=full_like(arr, float("-inf"))
  clean=where(mask, small, arr)
  return max(clean, axis=axis, out=out, keepdims=keepdims)

def nanprod(x:object, axis:object=None, *, dtype:object=None, out:ndarray|None=None, keepdims:builtins.bool=False)->object:
  arr=asarray(x)
  if arr.dtype not in(float32, float64):return prod(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
  mask=isnan(arr)
  ones_arr=full_like(arr, 1.0)
  clean=where(mask, ones_arr, arr)
  return prod(clean, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

def nanstd(x:object, axis:object=None, *, ddof:int=0, dtype:object=None, out:ndarray|None=None, keepdims:builtins.bool=False)->object:
  arr=asarray(x)
  if arr.dtype not in(float32, float64):return std(arr, axis=axis, ddof=ddof, dtype=dtype, out=out, keepdims=keepdims)
  return sqrt(nanvar(arr, axis=axis, ddof=ddof, dtype=dtype, out=out, keepdims=keepdims))

def nanvar(x:object, axis:object=None, *, ddof:int=0, dtype:object=None, out:ndarray|None=None, keepdims:builtins.bool=False)->object:
  arr=asarray(x)
  if arr.dtype not in(float32, float64):return var(arr, axis=axis, ddof=ddof, dtype=dtype, out=out, keepdims=keepdims)
  m=nanmean(arr, axis=axis, keepdims=True)
  diff=subtract(arr, m)
  sq=multiply(diff, diff)
  mask=isnan(arr)
  zero=zeros_like(arr)
  sq_clean=where(mask, zero, sq)
  s=sum(sq_clean, axis=axis, keepdims=keepdims)
  cnt=sum(logical_not(mask), axis=axis, dtype=int64, keepdims=keepdims)
  div=subtract(cnt, ddof)
  res=divide(s, div)
  if out is not None:
    out[...]=res
    return out
  return res

def nanargmax(x:object, axis:object=None, *, keepdims:builtins.bool=False)->object:
  arr=asarray(x)
  if arr.dtype not in(float32, float64):return argmax(arr, axis=axis, keepdims=keepdims)
  mask=isnan(arr)
  small=full_like(arr, float("-inf"))
  clean=where(mask, small, arr)
  return argmax(clean, axis=axis, keepdims=keepdims)

def nanargmin(x:object, axis:object=None, *, keepdims:builtins.bool=False)->object:
  arr=asarray(x)
  if arr.dtype not in(float32, float64):return argmin(arr, axis=axis, keepdims=keepdims)
  mask=isnan(arr)
  big=full_like(arr, float("inf"))
  clean=where(mask, big, arr)
  return argmin(clean, axis=axis, keepdims=keepdims)

def nanmedian(x:object, axis:object=None, *, keepdims:builtins.bool=False)->object:return nanquantile(x, 0.5, axis=axis, keepdims=keepdims)
def nanquantile(x:object, q:object, axis:object=None, *, keepdims:builtins.bool=False, method:str="linear")->object:
  arr=asarray(x)
  if arr.dtype not in(float32, float64):return quantile(arr, q, axis=axis, keepdims=keepdims, method=method)
  if axis is not None:raise NotImplementedError("nanquantile axis= not implemented in monpy v1")
  flat=[v for v in (builtins.float(arr._native.get_scalar(i)) for i in range(arr.size)) if not math.isnan(v)]
  flat.sort()
  if not flat:raise ValueError("all values nan")
  if isinstance(q, (list, tuple)):
    return ndarray(_native.from_flat([_quantile_from_sorted(flat, qq, keepdims_axes=None, orig_shape=()) for qq in q], (len(q),), float64.code))
  return _quantile_from_sorted(flat, q, keepdims_axes=None, orig_shape=())
def nanpercentile(x:object, q:object, axis:object=None, *, keepdims:builtins.bool=False, method:str="linear")->object:
  if isinstance(q, (list, tuple)):qf=[_py_float(v)/100.0 for v in q]
  else:qf=_py_float(q)/100.0
  return nanquantile(x, qf, axis=axis, keepdims=keepdims, method=method)
def nancumsum(x:object, axis:object=None, *, dtype:object=None, out:ndarray|None=None)->object:
  arr=asarray(x)
  if arr.dtype not in(float32, float64):return cumsum(arr, axis=axis, dtype=dtype, out=out)
  mask=isnan(arr)
  zero=zeros_like(arr)
  clean=where(mask, zero, arr)
  return cumsum(clean, axis=axis, dtype=dtype, out=out)
def nancumprod(x:object, axis:object=None, *, dtype:object=None, out:ndarray|None=None)->object:
  arr=asarray(x)
  if arr.dtype not in(float32, float64):return cumprod(arr, axis=axis, dtype=dtype, out=out)
  mask=isnan(arr)
  one=full_like(arr, 1.0)
  clean=where(mask, one, arr)
  return cumprod(clean, axis=axis, dtype=dtype, out=out)

# ---------------------------------------------------------------------------
# sort / search / set ops. Python-level implementations on top of
# `_native.from_flat` and friends; native kernels can layer in later.
# ---------------------------------------------------------------------------

def _flat_values(arr:ndarray)->list[typing.Any]:
  return [arr._native.get_scalar(i) for i in range(arr.size)]

def sort(a:object, axis:int=-1, kind:str|None=None, order:object=None)->ndarray:
  if order is not None:raise NotImplementedError("sort order= not implemented")
  del kind  # introsort across the board for v1
  arr=_mat(_av(a))
  n=arr.ndim
  if n==0:return arr
  if axis is None:
    flat=_flat_values(ravel(arr))
    flat.sort()
    return ndarray(_native.from_flat(flat, (arr.size,), arr.dtype.code))
  ax=axis+n if axis<0 else axis
  if ax<0 or ax>=n:raise ValueError("sort axis out of range")
  moved=ascontiguousarray(moveaxis(arr, ax, -1))
  pref=moved.shape[:-1]
  axis_size=moved.shape[-1]
  rows=math.prod(pref) if pref else 1
  flat=moved.reshape((rows, axis_size))
  out_vals:list[object]=[]
  for r in range(rows):
    row=[flat._native.get_scalar(r*axis_size+i) for i in range(axis_size)]
    row.sort()
    out_vals.extend(row)
  res=ndarray(_native.from_flat(out_vals, moved.shape, arr.dtype.code))
  return res.transpose(_inverse_moveaxis_perm(ax, -1, n)) if n>1 else res

def argsort(a:object, axis:int=-1, kind:str|None=None, order:object=None)->ndarray:
  if order is not None:raise NotImplementedError("argsort order= not implemented")
  del kind
  arr=_mat(_av(a))
  n=arr.ndim
  if n==0:return ndarray(_native.from_flat([0], (), int64.code))
  if axis is None:
    flat=_flat_values(ravel(arr))
    perm=sorted(range(len(flat)), key=lambda i:flat[i])
    return ndarray(_native.from_flat(perm, (len(perm),), int64.code))
  ax=axis+n if axis<0 else axis
  moved=ascontiguousarray(moveaxis(arr, ax, -1))
  pref=moved.shape[:-1]
  axis_size=moved.shape[-1]
  rows=math.prod(pref) if pref else 1
  flat=moved.reshape((rows, axis_size))
  out_vals:list[int]=[]
  for r in range(rows):
    row=[flat._native.get_scalar(r*axis_size+i) for i in range(axis_size)]
    perm=sorted(range(axis_size), key=lambda i:row[i])
    out_vals.extend(perm)
  res=ndarray(_native.from_flat(out_vals, moved.shape, int64.code))
  return res.transpose(_inverse_moveaxis_perm(ax, -1, n)) if n>1 else res

def partition(a:object, kth:int|Sequence[int], axis:int=-1, kind:str="introselect", order:object=None)->ndarray:
  # numpy partition: after the call, the kth element(s) are in their final
  # sorted position; smaller go before, larger after. v1 uses full sort
  # which is a superset of partition.
  del kind, kth
  if order is not None:raise NotImplementedError("partition order= not implemented")
  return sort(a, axis=axis)

def argpartition(a:object, kth:int|Sequence[int], axis:int=-1, kind:str="introselect", order:object=None)->ndarray:
  del kind, kth
  if order is not None:raise NotImplementedError("argpartition order= not implemented")
  return argsort(a, axis=axis)

def lexsort(keys:Sequence[object], axis:int=-1)->ndarray:
  arrs=[asarray(k) for k in keys]
  if not arrs:raise ValueError("lexsort: need at least one key")
  ref=arrs[0]
  for a in arrs:
    if a.shape!=ref.shape:raise ValueError("lexsort: keys must share shape")
  if ref.ndim==0:return ndarray(_native.from_flat([0], (), int64.code))
  if ref.ndim==1:
    n=ref.shape[0]
    rows=[tuple(a._native.get_scalar(i) for a in reversed(arrs)) for i in range(n)]
    perm=sorted(range(n), key=lambda i:rows[i])
    return ndarray(_native.from_flat(perm, (n,), int64.code))
  raise NotImplementedError("lexsort with ndim>1 not implemented")

def searchsorted(a:object, v:object, side:str="left", sorter:object=None)->object:
  if sorter is not None:raise NotImplementedError("searchsorted sorter= not implemented")
  if side not in("left", "right"):raise ValueError("side must be 'left' or 'right'")
  arr=asarray(a)
  if arr.ndim!=1:raise ValueError("searchsorted: a must be 1D")
  haystack=_flat_values(arr)
  is_scalar=_is_scalar(v)
  needles:list[typing.Any]=[typing.cast(typing.Any, v)] if is_scalar else _flat_values(asarray(v))
  out=[]
  import bisect
  for x in needles:
    if side=="left":out.append(bisect.bisect_left(haystack, x))
    else:out.append(bisect.bisect_right(haystack, x))
  if is_scalar:return out[0]
  shape=asarray(v).shape
  return ndarray(_native.from_flat(out, shape, int64.code))

def digitize(x:object, bins:object, right:builtins.bool=False)->ndarray:
  arr=asarray(x)
  bn=asarray(bins)
  if bn.ndim!=1:raise ValueError("digitize: bins must be 1D")
  bins_list=_flat_values(bn)
  ascending=builtins.all(bins_list[i]<=bins_list[i+1] for i in range(len(bins_list)-1))
  if not ascending:bins_list=list(reversed(bins_list))
  vals=_flat_values(ravel(arr))
  import bisect
  out=[]
  for v in vals:
    if right:idx=bisect.bisect_left(bins_list, v)
    else:idx=bisect.bisect_right(bins_list, v)
    out.append(idx if ascending else len(bins_list)-idx)
  return ndarray(_native.from_flat(out, arr.shape, int64.code))

def unique(ar:object, return_index:builtins.bool=False, return_inverse:builtins.bool=False, return_counts:builtins.bool=False, axis:object=None)->object:
  if axis is not None:raise NotImplementedError("unique axis= not implemented")
  arr=asarray(ar)
  flat=_flat_values(ravel(arr))
  paired=sorted(enumerate(flat), key=lambda p:p[1])
  unique_vals:list[object]=[]
  unique_indices:list[int]=[]
  inverse=[0]*len(flat)
  counts:list[int]=[]
  for orig_idx, val in paired:
    if not unique_vals or unique_vals[-1]!=val:
      unique_vals.append(val)
      unique_indices.append(orig_idx)
      counts.append(1)
    else:counts[-1]+=1
    inverse[orig_idx]=len(unique_vals)-1
  u_arr=ndarray(_native.from_flat(unique_vals, (len(unique_vals),), arr.dtype.code))
  if not(return_index or return_inverse or return_counts):return u_arr
  out=[u_arr]
  if return_index:out.append(ndarray(_native.from_flat(unique_indices, (len(unique_indices),), int64.code)))
  if return_inverse:out.append(ndarray(_native.from_flat(inverse, (len(inverse),), int64.code)))
  if return_counts:out.append(ndarray(_native.from_flat(counts, (len(counts),), int64.code)))
  return tuple(out)

def bincount(x:object, weights:object=None, minlength:int=0)->ndarray:
  arr=asarray(x)
  if arr.ndim!=1:raise ValueError("bincount: input must be 1D")
  vals=[builtins.int(arr._native.get_scalar(i)) for i in range(arr.size)]
  if vals and builtins.min(vals)<0:raise ValueError("bincount: negative values not allowed")
  n=builtins.max(minlength, (builtins.max(vals)+1) if vals else 0)
  if weights is None:
    out=[0]*n
    for v in vals:out[v]+=1
    return ndarray(_native.from_flat(out, (n,), int64.code))
  w=asarray(weights)
  if w.shape!=arr.shape:raise ValueError("bincount: weights shape mismatch")
  out_f=[0.0]*n
  for i, v in enumerate(vals):out_f[v]+=builtins.float(w._native.get_scalar(i))
  return ndarray(_native.from_flat(out_f, (n,), float64.code))

def isin(element:object, test_elements:object, assume_unique:builtins.bool=False, invert:builtins.bool=False)->ndarray:
  arr=asarray(element)
  test=asarray(test_elements)
  test_set=set(_flat_values(ravel(test)))
  vals=_flat_values(ravel(arr))
  if invert:res=[v not in test_set for v in vals]
  else:res=[v in test_set for v in vals]
  return ndarray(_native.from_flat(res, arr.shape, bool.code))

def intersect1d(ar1:object, ar2:object, assume_unique:builtins.bool=False, return_indices:builtins.bool=False)->object:
  s1=set(_flat_values(ravel(asarray(ar1))))
  s2=set(_flat_values(ravel(asarray(ar2))))
  common=sorted(s1&s2)
  arr=ndarray(_native.from_flat(common, (len(common),), asarray(ar1).dtype.code))
  if not return_indices:return arr
  flat1=_flat_values(ravel(asarray(ar1)))
  flat2=_flat_values(ravel(asarray(ar2)))
  i1=[flat1.index(v) for v in common]
  i2=[flat2.index(v) for v in common]
  return arr, ndarray(_native.from_flat(i1, (len(i1),), int64.code)), ndarray(_native.from_flat(i2, (len(i2),), int64.code))

def union1d(ar1:object, ar2:object)->ndarray:
  s=sorted(set(_flat_values(ravel(asarray(ar1))))|set(_flat_values(ravel(asarray(ar2)))))
  return ndarray(_native.from_flat(s, (len(s),), asarray(ar1).dtype.code))

def setdiff1d(ar1:object, ar2:object, assume_unique:builtins.bool=False)->ndarray:
  s1=set(_flat_values(ravel(asarray(ar1))))
  s2=set(_flat_values(ravel(asarray(ar2))))
  out=sorted(s1-s2)
  return ndarray(_native.from_flat(out, (len(out),), asarray(ar1).dtype.code))

def setxor1d(ar1:object, ar2:object, assume_unique:builtins.bool=False)->ndarray:
  s1=set(_flat_values(ravel(asarray(ar1))))
  s2=set(_flat_values(ravel(asarray(ar2))))
  out=sorted(s1^s2)
  return ndarray(_native.from_flat(out, (len(out),), asarray(ar1).dtype.code))

# minimum surface — index-array helpers used by sort/search.
def nonzero(a:object)->tuple[ndarray, ...]:
  arr=asarray(a)
  flat=_flat_values(ravel(arr))
  n=arr.ndim
  if n==0:
    if builtins.bool(flat[0]):return tuple(ndarray(_native.from_flat([0], (1,), int64.code)) for _ in range(builtins.max(1, n)))
    return tuple(ndarray(_native.from_flat([], (0,), int64.code)) for _ in range(builtins.max(1, n)))
  shape=arr.shape
  total=arr.size
  result_axes:list[list[int]]=[[] for _ in range(n)]
  for i in range(total):
    if not builtins.bool(flat[i]):continue
    rem=i
    for d in range(n-1, -1, -1):
      coord=rem%shape[d] if shape[d]>0 else 0
      rem=rem//shape[d] if shape[d]>0 else rem
      result_axes[d].insert(0, coord) if False else None
  # Simpler: compute fresh.
  result_axes=[[] for _ in range(n)]
  strides=[1]*n
  for d in range(n-2, -1, -1):strides[d]=strides[d+1]*shape[d+1]
  for i in range(total):
    if not builtins.bool(flat[i]):continue
    rem=i
    for d in range(n):
      coord=rem//strides[d]
      rem=rem-coord*strides[d]
      result_axes[d].append(coord)
  return tuple(ndarray(_native.from_flat(axis_idx, (len(axis_idx),), int64.code)) for axis_idx in result_axes)

def argwhere(a:object)->ndarray:
  arr=asarray(a)
  parts=nonzero(arr)
  if not parts:return ndarray(_native.from_flat([], (0, arr.ndim), int64.code))
  n=arr.ndim
  count=parts[0].size
  flat:list[int]=[]
  for i in range(count):
    for d in range(n):
      flat.append(builtins.int(parts[d]._native.get_scalar(i)))
  return ndarray(_native.from_flat(flat, (count, n), int64.code))

def flatnonzero(a:object)->ndarray:
  arr=ravel(asarray(a))
  flat=_flat_values(arr)
  out=[i for i, v in enumerate(flat) if builtins.bool(v)]
  return ndarray(_native.from_flat(out, (len(out),), int64.code))

def ravel_multi_index(multi_index:Sequence[object], dims:Sequence[int], mode:str="raise", order:str="C")->ndarray:
  if order!="C":raise NotImplementedError("ravel_multi_index order != 'C'")
  if mode!="raise":raise NotImplementedError("ravel_multi_index mode != 'raise'")
  axes=[asarray(a) for a in multi_index]
  if not axes:raise ValueError("multi_index is empty")
  shape=axes[0].shape
  n=len(dims)
  if n!=len(axes):raise ValueError("multi_index / dims length mismatch")
  flat_each=[_flat_values(ravel(a)) for a in axes]
  cnt=len(flat_each[0])
  strides=[1]*n
  for d in range(n-2, -1, -1):strides[d]=strides[d+1]*dims[d+1]
  out=[]
  for i in range(cnt):
    idx=0
    for d in range(n):
      v=builtins.int(flat_each[d][i])
      if v<0 or v>=dims[d]:raise ValueError("ravel_multi_index: index out of bounds")
      idx+=v*strides[d]
    out.append(idx)
  return ndarray(_native.from_flat(out, shape, int64.code))

def unravel_index(indices:object, shape:Sequence[int], order:str="C")->tuple[ndarray, ...]:
  if order!="C":raise NotImplementedError("unravel_index order != 'C'")
  arr=asarray(indices)
  flat=_flat_values(ravel(arr))
  dims=tuple(dims for dims in shape)
  n=len(dims)
  out_axes=[[] for _ in range(n)]
  strides=[1]*n
  for d in range(n-2, -1, -1):strides[d]=strides[d+1]*dims[d+1]
  for v in flat:
    rem=builtins.int(v)
    for d in range(n):
      coord=rem//strides[d]
      rem=rem-coord*strides[d]
      out_axes[d].append(coord)
  return tuple(ndarray(_native.from_flat(axis_idx, arr.shape, int64.code)) for axis_idx in out_axes)

def _flat_to_multi(idx:int, shape:tuple[int, ...])->tuple[int, ...]:
  if not shape:return ()
  out=[0]*len(shape)
  rem=idx
  for d in range(len(shape)-1, -1, -1):
    s=shape[d]
    out[d]=rem%s if s else 0
    rem=rem//s if s else 0
  return tuple(out)

def _multi_to_flat(coords:tuple[int, ...], shape:tuple[int, ...])->int:
  flat=0
  stride=1
  for d in range(len(shape)-1, -1, -1):
    flat+=coords[d]*stride
    stride*=shape[d]
  return flat

def _normalize_axis_index(value:object, axis_size:int, axis:int)->int:
  raw=_py_int(value)
  idx=raw+axis_size if raw<0 else raw
  if idx<0 or idx>=axis_size:raise IndexError(f"index {raw} is out of bounds for axis {axis} with size {axis_size}")
  return idx

def _broadcast_coord(coords:tuple[int, ...], source_shape:tuple[int, ...], target_shape:tuple[int, ...], context:str)->tuple[int, ...]:
  if not source_shape:return ()
  if len(source_shape)>len(target_shape):raise ValueError(f"{context}: value array could not be broadcast to indexing result")
  offset=len(target_shape)-len(source_shape)
  out=[]
  for i, dim in enumerate(source_shape):
    target_dim=target_shape[offset+i]
    if dim==1:out.append(0)
    elif dim==target_dim:out.append(coords[offset+i])
    else:raise ValueError(f"{context}: value array could not be broadcast to indexing result")
  return tuple(out)

def take(a:object, indices:object, axis:object=None, *, out:ndarray|None=None, mode:str="raise")->ndarray:
  if mode!="raise":raise NotImplementedError("take mode != 'raise'")
  arr=asarray(a)
  idx=asarray(indices)
  if axis is None:
    flat=_flat_values(ravel(arr))
    idx_flat=_flat_values(ravel(idx))
    vals=[flat[builtins.int(i)] for i in idx_flat]
    res=ndarray(_native.from_flat(vals, idx.shape, arr.dtype.code))
  else:
    n=arr.ndim
    ax=_axis_int(axis, n, "take")
    moved=ascontiguousarray(moveaxis(arr, ax, -1))
    pref=moved.shape[:-1]
    axis_size=moved.shape[-1]
    rows=math.prod(pref) if pref else 1
    flat=moved.reshape((rows, axis_size))
    idx_flat=_flat_values(ravel(idx))
    out_vals=[]
    for r in range(rows):
      row=[flat._native.get_scalar(r*axis_size+i) for i in range(axis_size)]
      for i in idx_flat:out_vals.append(row[builtins.int(i)])
    out_shape=pref+idx.shape
    res=ndarray(_native.from_flat(out_vals, out_shape, arr.dtype.code))
    perm=_inverse_moveaxis_perm(ax, -1, len(out_shape)) if len(out_shape)>1 else None
    if perm:res=res.transpose(perm)
  if out is not None:
    out[...]=res
    return out
  return res

def take_along_axis(arr:object, indices:object, axis:object)->ndarray:
  # Gather: output[i_0..i_{k-1}] = arr[i_0..i_{ax-1}, indices[...], i_{ax+1}..i_{k-1}].
  # `indices` broadcasts to `arr.shape` on non-axis dims; axis dim is free.
  a=asarray(arr)
  idx=asarray(indices)
  if axis is None:
    return take(ravel(a), ravel(idx))
  n=a.ndim
  if idx.ndim!=n:raise ValueError("take_along_axis: indices.ndim must match arr.ndim")
  ax=_axis_int(axis, n, "take_along_axis")
  out_shape=[0]*n
  for d in range(n):
    if d==ax:out_shape[d]=idx.shape[d]
    elif a.shape[d]==idx.shape[d] or idx.shape[d]==1:out_shape[d]=a.shape[d]
    elif a.shape[d]==1:out_shape[d]=idx.shape[d]
    else:raise ValueError("take_along_axis: shape mismatch on non-axis dim")
  out_size=math.prod(out_shape) if out_shape else 1
  out_vals=[]
  for i in range(out_size):
    coords=_flat_to_multi(i, tuple(out_shape))
    idx_coords=tuple(0 if idx.shape[d]==1 else coords[d] for d in range(n))
    iv=_normalize_axis_index(idx._native.get_scalar(_multi_to_flat(idx_coords, idx.shape)), a.shape[ax], ax)
    a_coords=list(coords)
    a_coords[ax]=iv
    a_coords_b=tuple(0 if a.shape[d]==1 and d!=ax else a_coords[d] for d in range(n))
    out_vals.append(a._native.get_scalar(_multi_to_flat(a_coords_b, a.shape)))
  return ndarray(_native.from_flat(out_vals, tuple(out_shape), a.dtype.code))

def put_along_axis(arr:ndarray, indices:object, values:object, axis:object)->None:
  # Scatter: arr[i_0..i_{ax-1}, indices[...], i_{ax+1}..i_{k-1}] = values[...].
  if not isinstance(arr, ndarray):raise TypeError("put_along_axis: arr must be ndarray")
  idx=asarray(indices)
  vals=asarray(values)
  if axis is None:
    flat_i=_flat_values(ravel(idx))
    flat_v=_flat_values(ravel(vals))
    if not flat_v:raise ValueError("put_along_axis: empty values")
    if len(flat_v)<len(flat_i):flat_v=(flat_v*((len(flat_i)+len(flat_v)-1)//len(flat_v)))[:len(flat_i)]
    for i, v in builtins.zip(flat_i, flat_v):
      mi=_flat_to_multi(_normalize_axis_index(i, arr.size, 0), arr.shape)
      arr[mi]=v
    return
  n=arr.ndim
  if idx.ndim!=n:raise ValueError("put_along_axis: indices.ndim must match arr.ndim")
  ax=_axis_int(axis, n, "put_along_axis")
  iter_shape=[0]*n
  for d in range(n):
    if d==ax:iter_shape[d]=idx.shape[d]
    elif arr.shape[d]==idx.shape[d] or idx.shape[d]==1:iter_shape[d]=arr.shape[d]
    elif arr.shape[d]==1:iter_shape[d]=idx.shape[d]
    else:raise IndexError("shape mismatch: indexing arrays could not be broadcast")
  iter_shape_t=tuple(iter_shape)
  v_size=math.prod(iter_shape_t) if iter_shape_t else 1
  for i in range(v_size):
    coords=_flat_to_multi(i, iter_shape_t)
    idx_coords=tuple(0 if idx.shape[d]==1 else coords[d] for d in range(n))
    iv=_normalize_axis_index(idx._native.get_scalar(_multi_to_flat(idx_coords, idx.shape)), arr.shape[ax], ax)
    a_coords=list(coords)
    a_coords[ax]=iv
    a_coords_b=tuple(0 if arr.shape[d]==1 and d!=ax else a_coords[d] for d in range(n))
    if vals.shape:
      vc=_broadcast_coord(coords, vals.shape, iter_shape_t, "put_along_axis")
      v=vals._native.get_scalar(_multi_to_flat(vc, vals.shape))
    else:
      v=vals._native.get_scalar(0)
    arr[a_coords_b]=v

def put(arr:ndarray, indices:object, values:object, mode:str="raise")->None:
  # Scatter into arr.flat at positions `indices`. Modifies arr in place.
  if mode!="raise":raise NotImplementedError("put mode != 'raise'")
  if not isinstance(arr, ndarray):raise TypeError("put: arr must be ndarray")
  flat_i=_flat_values(ravel(asarray(indices)))
  flat_v=_flat_values(ravel(asarray(values)))
  n=len(flat_i)
  if not flat_v:raise ValueError("put: cannot use empty values array")
  if len(flat_v)<n:flat_v=(flat_v*((n+len(flat_v)-1)//len(flat_v)))[:n]
  for i, v in builtins.zip(flat_i, flat_v):
    mi=_flat_to_multi(_normalize_axis_index(i, arr.size, 0), arr.shape)
    arr[mi]=v

def block(arrays:object)->ndarray:
  # Recursive block-matrix construction. Innermost lists concat along
  # axis=-1; one level out, axis=-2; …; outermost, axis=-max_depth.
  def _md(x:object)->int:
    if isinstance(x, (list, tuple)):
      return 1+builtins.max((_md(e) for e in x), default=0)
    return 0
  d=_md(arrays)
  if d==0:return typing.cast(ndarray, atleast_1d(asarray(arrays)))
  def _atleast(x:object, n:int)->ndarray:
    a=asarray(x)
    while a.ndim<n:a=expand_dims(a, 0)
    return a
  def _walk(x:object, depth:int)->ndarray:
    if depth==0:return _atleast(x, d)
    pieces=[_walk(e, depth-1) for e in _iter(x)]
    return concatenate(pieces, axis=-depth)
  return _walk(arrays, d)

def repeat(a:object, repeats:object, axis:object=None)->ndarray:
  arr=asarray(a)
  if axis is None:
    flat=_flat_values(ravel(arr))
    if isinstance(repeats, builtins.int):
      out=[v for v in flat for _ in range(repeats)]
    else:
      r=_flat_values(asarray(repeats))
      if len(r)!=len(flat):raise ValueError("repeat: length mismatch")
      out=[]
      for v, k in builtins.zip(flat, r):
        for _ in range(builtins.int(k)):out.append(v)
    return ndarray(_native.from_flat(out, (len(out),), arr.dtype.code))
  n=arr.ndim
  ax=_axis_int(axis, n, "repeat")
  moved=ascontiguousarray(moveaxis(arr, ax, -1))
  pref=moved.shape[:-1]
  axis_size=moved.shape[-1]
  rows=math.prod(pref) if pref else 1
  flat=moved.reshape((rows, axis_size))
  if isinstance(repeats, builtins.int):reps=[repeats]*axis_size
  else:reps=[builtins.int(v) for v in _flat_values(asarray(repeats))]
  if len(reps)!=axis_size:raise ValueError("repeat: repeats length must match axis")
  total=builtins.sum(reps)
  out_vals=[]
  for r in range(rows):
    row=[flat._native.get_scalar(r*axis_size+i) for i in range(axis_size)]
    for i, k in enumerate(reps):
      for _ in range(k):out_vals.append(row[i])
  out_shape=pref+(total,)
  res=ndarray(_native.from_flat(out_vals, out_shape, arr.dtype.code))
  return res.transpose(_inverse_moveaxis_perm(ax, -1, len(out_shape))) if len(out_shape)>1 else res

def tile(A:object, reps:int|Sequence[int])->ndarray:
  arr=asarray(A)
  reps_t=(reps,) if isinstance(reps, builtins.int) else tuple(reps)
  while arr.ndim<len(reps_t):arr=expand_dims(arr, 0)
  while len(reps_t)<arr.ndim:reps_t=(1,)+reps_t
  out_shape=tuple(s*r for s, r in builtins.zip(arr.shape, reps_t))
  flat=_flat_values(ravel(arr))
  total=math.prod(out_shape)
  in_strides=[1]*arr.ndim
  for d in range(arr.ndim-2, -1, -1):in_strides[d]=in_strides[d+1]*arr.shape[d+1]
  out=[]
  for i in range(total):
    rem=i
    src_idx=0
    for d in range(arr.ndim):
      cur_size=out_shape[d]
      out_stride=math.prod(out_shape[d+1:]) if d<arr.ndim-1 else 1
      coord=rem//out_stride
      rem=rem-coord*out_stride
      src_coord=coord%arr.shape[d]
      src_idx+=src_coord*in_strides[d]
    out.append(flat[src_idx])
  return ndarray(_native.from_flat(out, out_shape, arr.dtype.code))

def roll(a:object, shift:int|Sequence[int], axis:object=None)->ndarray:
  arr=asarray(a)
  if axis is None:
    if not isinstance(shift, builtins.int):raise ValueError("roll: shift must be int when axis=None")
    n=arr.size
    if n==0:return arr
    s=shift%n
    flat=_flat_values(ravel(arr))
    rolled=flat[-s:]+flat[:-s] if s else flat
    return ndarray(_native.from_flat(rolled, arr.shape, arr.dtype.code))
  axes=_axis_tuple(axis)
  shifts=(shift,) if isinstance(shift, builtins.int) else tuple(_py_int(s) for s in _iter(shift))
  if len(axes)!=len(shifts):raise ValueError("roll: shift/axis length mismatch")
  out=arr
  for ax, sh in builtins.zip(axes, shifts):
    n=out.shape[ax]
    if n==0:continue
    s=sh%n
    if s==0:continue
    a_part=take(out, asarray(list(range(n-s, n))+list(range(n-s)), dtype=int64), axis=ax)
    out=a_part
  return out

def append(arr:object, values:object, axis:object=None)->ndarray:
  if axis is None:return concatenate([ravel(asarray(arr)), ravel(asarray(values))])
  base=asarray(arr)
  return concatenate([base, asarray(values)], axis=_axis_int(axis, base.ndim, "append"))

def insert(arr:object, obj:object, values:object, axis:object=None)->ndarray:
  base=asarray(arr)
  if axis is None:
    base=ravel(base)
    ax=0
  else:ax=_axis_int(axis, base.ndim, "insert")
  if isinstance(obj, builtins.int):positions=[obj]
  else:positions=[_py_int(p) for p in _iter(obj)]
  vals=asarray(values)
  if vals.ndim==0:vals=expand_dims(vals, 0)
  pieces=[]
  prev=0
  flat_pos=sorted(positions)
  if vals.shape[0]==1 and len(flat_pos)>1:vals=tile(vals, len(flat_pos))
  for i, p in enumerate(flat_pos):
    key=tuple(slice(prev, p) if d==ax else slice(None) for d in range(base.ndim))
    pieces.append(base[key])
    val_slice_key=tuple(slice(i, i+1) if d==ax else slice(None) for d in range(vals.ndim))
    if vals.ndim==base.ndim:pieces.append(vals[val_slice_key])
    else:
      bcast_shape=list(base.shape)
      bcast_shape[ax]=1
      pieces.append(broadcast_to(vals.reshape((1,)*(base.ndim-vals.ndim)+vals.shape) if vals.ndim<base.ndim else vals[val_slice_key], tuple(bcast_shape)))
    prev=p
  end_key=tuple(slice(prev, None) if d==ax else slice(None) for d in range(base.ndim))
  pieces.append(base[end_key])
  return concatenate(pieces, axis=ax)

def delete(arr:object, obj:object, axis:object=None)->ndarray:
  base=asarray(arr)
  if axis is None:
    base=ravel(base)
    ax=0
  else:ax=_axis_int(axis, base.ndim, "delete")
  if isinstance(obj, builtins.int):to_drop={obj if obj>=0 else obj+base.shape[ax]}
  elif isinstance(obj, slice):to_drop=set(range(*obj.indices(base.shape[ax])))
  else:
    to_drop=set()
    for p in _iter(obj):
      pp=_py_int(p)
      to_drop.add(pp if pp>=0 else pp+base.shape[ax])
  keep=[i for i in range(base.shape[ax]) if i not in to_drop]
  if not keep:
    new_shape=list(base.shape)
    new_shape[ax]=0
    return zeros(tuple(new_shape), dtype=base.dtype)
  return take(base, asarray(keep, dtype=int64), axis=ax)

def trim_zeros(filt:object, trim:str="fb")->ndarray:
  arr=asarray(filt)
  if arr.ndim!=1:raise ValueError("trim_zeros: input must be 1D")
  flat=_flat_values(arr)
  n=len(flat)
  start, end=0, n
  if "f" in trim.lower():
    while start<end and not builtins.bool(flat[start]):start+=1
  if "b" in trim.lower():
    while end>start and not builtins.bool(flat[end-1]):end-=1
  out=flat[start:end]
  return ndarray(_native.from_flat(out, (len(out),), arr.dtype.code))

def broadcast_arrays(*args:object)->list[ndarray]:
  arrs=[asarray(a) for a in args]
  if not arrs:return []
  shape=arrs[0].shape
  for a in arrs[1:]:shape=_broadcast_pair_shape(shape, a.shape)
  return [broadcast_to(a, shape) for a in arrs]

def _broadcast_pair_shape(s1:tuple[int, ...], s2:tuple[int, ...])->tuple[int, ...]:
  ml=builtins.max(len(s1), len(s2))
  p1=(1,)*(ml-len(s1))+s1
  p2=(1,)*(ml-len(s2))+s2
  out=[]
  for a, b in builtins.zip(p1, p2):
    if a==b:out.append(a)
    elif a==1:out.append(b)
    elif b==1:out.append(a)
    else:raise ValueError(f"shapes {s1} and {s2} cannot be broadcast")
  return tuple(out)

def _pad_axis_indices(n:int, pre:int, post:int, mode:str)->list[int]:
  if pre<0 or post<0:raise ValueError("pad: index can't contain negative values")
  if pre==0 and post==0:return list(range(n))
  if n<=0:raise ValueError(f"{mode}: can't extend empty axis")
  if mode=="edge":return [0]*pre+list(range(n))+[n-1]*post
  if mode=="wrap":return [p%n for p in range(-pre, n+post)]
  if mode=="symmetric":
    period=2*n
    return [(m if m<n else period-1-m) for m in (p%period for p in range(-pre, n+post))]
  if mode=="reflect":
    if n<=1:raise ValueError("reflect: axis size must be >= 2")
    period=2*n-2
    return [(m if m<n else period-m) for m in (p%period for p in range(-pre, n+post))]
  raise NotImplementedError(f"pad mode={mode!r} not implemented")

def _pad_axis(arr:ndarray, axis:int, pre:int, post:int, mode:str)->ndarray:
  if pre==0 and post==0:return arr
  idx=_pad_axis_indices(arr.shape[axis], pre, post, mode)
  return take(arr, ndarray(_native.from_flat(idx, (len(idx),), int64.code)), axis=axis)

def pad(array:object, pad_width:object, mode:str="constant", **kwargs:object)->ndarray:
  arr=ascontiguousarray(asarray(array))
  if isinstance(pad_width, builtins.int):pw=[(pad_width, pad_width)]*arr.ndim
  else:
    items=list(_iter(pad_width))
    if items and not isinstance(items[0], (list, tuple)):pw=[(_py_int(items[0]), _py_int(items[1]))]*arr.ndim
    else:pw=[(_py_int(typing.cast(Sequence[object], p)[0]), _py_int(typing.cast(Sequence[object], p)[1])) for p in items]
  if len(pw)!=arr.ndim:raise ValueError("pad: pad_width must match ndim")
  pad_before=tuple(p[0] for p in pw)
  pad_after=tuple(p[1] for p in pw)
  if mode=="constant":
    cv=kwargs.get("constant_values", 0)
    return ndarray(_native.pad_constant(arr._native, pad_before, pad_after, _py_float(cv)))
  if mode in ("edge", "reflect", "symmetric", "wrap"):
    out=arr
    for ax in range(arr.ndim):
      out=_pad_axis(out, ax, pad_before[ax], pad_after[ax], mode)
    return out
  raise NotImplementedError(f"pad mode={mode!r} not implemented in monpy v1")

def tril(m:object, k:int=0)->ndarray:
  arr=ascontiguousarray(asarray(m))
  if arr.ndim<2:raise ValueError("tril: input must be 2D")
  return ndarray(_native.tril(arr._native, builtins.int(k)))

def triu(m:object, k:int=0)->ndarray:
  arr=ascontiguousarray(asarray(m))
  if arr.ndim<2:raise ValueError("triu: input must be 2D")
  return ndarray(_native.triu(arr._native, builtins.int(k)))


def diag_indices(n:int, ndim:int=2)->tuple[ndarray, ...]:
  idx=arange(n, dtype=int64)
  return tuple(idx for _ in range(ndim))

def tril_indices(n:int, k:int=0, m:int|None=None)->tuple[ndarray, ndarray]:
  M=n if m is None else m
  rows=[]
  cols=[]
  for i in range(n):
    for j in range(builtins.min(M, i+k+1)):
      rows.append(i)
      cols.append(j)
  return ndarray(_native.from_flat(rows, (len(rows),), int64.code)), ndarray(_native.from_flat(cols, (len(cols),), int64.code))

def triu_indices(n:int, k:int=0, m:int|None=None)->tuple[ndarray, ndarray]:
  M=n if m is None else m
  rows=[]
  cols=[]
  for i in range(n):
    for j in range(builtins.max(0, i+k), M):
      rows.append(i)
      cols.append(j)
  return ndarray(_native.from_flat(rows, (len(rows),), int64.code)), ndarray(_native.from_flat(cols, (len(cols),), int64.code))

def asfortranarray(a:object, *, dtype:object=None)->ndarray:
  # monpy stores row-major only; for v1 we accept the API but the result is
  # still C-order. Tests should not rely on column-major byte layout.
  return ascontiguousarray(a, dtype=dtype)

def frombuffer(buffer:object, dtype:object=float64, count:int=-1, offset:int=0)->ndarray:
  t=_resolve_dtype(dtype)
  try:return ndarray(_native.frombuffer(buffer, t.code, int(count), int(offset)), owner=buffer)
  except Exception as exc:raise ValueError(str(exc)) from exc

def fromiter(iter_:object, dtype:object, count:int=-1)->ndarray:
  t=_resolve_dtype(dtype)
  if count<0:vals=list(_iter(iter_))
  else:
    it=iter(_iter(iter_))
    vals=[next(it) for _ in range(count)]
  return asarray(vals, dtype=t)

def _is_advanced_index(k:object)->builtins.bool:
  if isinstance(k, ndarray) and (k.dtype==bool or k.dtype in(int8, int16, int32, int64)):return True
  if isinstance(k, list) and k and builtins.all(isinstance(v, builtins.int) for v in k):return True
  return False

def _to_int_indices(k:object)->ndarray:
  if isinstance(k, ndarray):return k
  return asarray(k, dtype=int64)

def _advanced_getitem(self:ndarray, k:object)->ndarray:
  if isinstance(k, ndarray) and k.dtype==bool:
    if k.shape!=self.shape[:k.ndim]:raise IndexError("boolean index shape mismatch")
    flat_arr=_flat_values(ravel(self))
    flat_mask=_flat_values(ravel(k))
    # Each True selects a sub-block of size prod(self.shape[k.ndim:]).
    rest=self.shape[k.ndim:]
    rest_size=math.prod(rest) if rest else 1
    indices=[i for i, b in enumerate(flat_mask) if builtins.bool(b)]
    out_vals=[]
    for i in indices:
      base=i*rest_size
      for j in range(rest_size):out_vals.append(flat_arr[base+j])
    out_shape=(len(indices),)+rest if rest else (len(indices),)
    return ndarray(_native.from_flat(out_vals, out_shape, self.dtype.code))
  idx=_to_int_indices(k)
  flat=_flat_values(self.reshape((self.shape[0], -1) if self.ndim>1 else (self.shape[0],)))
  rest=self.shape[1:]
  rest_size=math.prod(rest) if rest else 1
  idx_flat=_flat_values(ravel(idx))
  out_vals=[]
  d=self.shape[0]
  for i in idx_flat:
    ii=builtins.int(i)
    if ii<0:ii+=d
    if ii<0 or ii>=d:raise IndexError("fancy index out of bounds")
    base=ii*rest_size
    for j in range(rest_size):out_vals.append(flat[base+j])
  out_shape=idx.shape+rest if rest else idx.shape
  return ndarray(_native.from_flat(out_vals, out_shape, self.dtype.code))

def _advanced_getitem_tuple(self:ndarray, k:tuple)->ndarray:
  parts=list(k)
  if Ellipsis in parts:
    ea=parts.index(Ellipsis)
    missing=self.ndim-(len(parts)-1)
    parts=parts[:ea]+[slice(None)]*missing+parts[ea+1:]
  while len(parts)<self.ndim:parts.append(slice(None))
  if builtins.any(p is None for p in parts):raise NotImplementedError("newaxis with fancy indexing not supported in v1")
  if builtins.any(isinstance(p, ndarray) and p.dtype==bool for p in parts):
    raise NotImplementedError("boolean mask in tuple indexing not supported in v1")
  # Pure integer-array fancy indexing along multiple axes; broadcast index arrays.
  idx_arrays=[]
  for p in parts:
    if isinstance(p, builtins.int):idx_arrays.append(asarray([p], dtype=int64))
    elif isinstance(p, (list, ndarray)):idx_arrays.append(_to_int_indices(p))
    elif isinstance(p, slice):
      a, b, c=p.indices(self.shape[len(idx_arrays)])
      idx_arrays.append(asarray(list(range(a, b, c)), dtype=int64))
    else:raise NotImplementedError(f"unsupported index part: {p!r}")
  # Broadcast all idx arrays to a common shape.
  bcast=broadcast_arrays(*idx_arrays)
  flat_idxs=[_flat_values(ravel(b)) for b in bcast]
  total=len(flat_idxs[0])
  out=[]
  flat_self=_flat_values(ravel(self))
  strides=[1]*self.ndim
  for d in range(self.ndim-2, -1, -1):strides[d]=strides[d+1]*self.shape[d+1]
  for i in range(total):
    phys=0
    for d in range(self.ndim):phys+=builtins.int(flat_idxs[d][i])*strides[d]
    out.append(flat_self[phys])
  out_shape=bcast[0].shape
  return ndarray(_native.from_flat(out, out_shape, self.dtype.code))

def _advanced_setitem(self:ndarray, k:object, v:object)->None:
  if isinstance(k, ndarray) and k.dtype==bool:
    if k.shape!=self.shape[:k.ndim]:raise IndexError("boolean index shape mismatch")
    flat_mask=_flat_values(ravel(k))
    indices=[i for i, b in enumerate(flat_mask) if builtins.bool(b)]
    rest=self.shape[k.ndim:]
    rest_size=math.prod(rest) if rest else 1
    val=asarray(v) if not isinstance(v, ndarray) else v
    val_flat=_flat_values(ravel(val))
    is_scalar=val.size==1
    cursor=0
    for idx in indices:
      base=idx*rest_size
      for j in range(rest_size):
        write=val_flat[0] if is_scalar else val_flat[cursor]
        # Write back via __setitem__ — translate flat into multi-index.
        coord=[]
        rem=base+j
        st=self.shape
        strides=[1]*self.ndim
        for d in range(self.ndim-2, -1, -1):strides[d]=strides[d+1]*st[d+1]
        for d in range(self.ndim):
          c=rem//strides[d]
          rem-=c*strides[d]
          coord.append(c)
        # Build a key that the existing _view_for_key can handle.
        key=tuple(builtins.int(c) for c in coord)
        view=self._view_for_key(key)
        _native.fill(view._native, write)
        cursor+=1
    return
  idx=_to_int_indices(k)
  idx_flat=_flat_values(ravel(idx))
  d0=self.shape[0]
  val=asarray(v) if not isinstance(v, ndarray) else v
  val_flat=_flat_values(ravel(val))
  is_scalar=val.size==1
  rest=self.shape[1:]
  rest_size=math.prod(rest) if rest else 1
  cursor=0
  for ii in idx_flat:
    i=builtins.int(ii)
    if i<0:i+=d0
    for j in range(rest_size):
      write=val_flat[0] if is_scalar else val_flat[cursor]
      # Translate i,j → multi-index.
      coord=[i]
      rem=j
      rs=[1]*len(rest)
      for d in range(len(rest)-2, -1, -1):rs[d]=rs[d+1]*rest[d+1]
      for d in range(len(rest)):
        c=rem//rs[d]
        rem-=c*rs[d]
        coord.append(c)
      view=self._view_for_key(tuple(coord))
      _native.fill(view._native, write)
      cursor+=1

def _advanced_setitem_tuple(self:ndarray, k:tuple, v:object)->None:
  raise NotImplementedError("setitem with tuple of fancy indices not implemented in v1")

def matmul(x1:object, x2:object)->ndarray:
  if _is_kernel_value(x1) or _is_kernel_value(x2):
    from .kernels import dsl as _kernel_dsl
    return typing.cast(ndarray, _kernel_dsl.matmul(x1, x2))
  if type(x1) is ndarray and type(x2) is ndarray:                                                                             # ndarray×ndarray fast path; mojo handles promotion
    return ndarray._wrap(_native.matmul(x1._native, x2._native))
  l=_mat(_av(x1))
  r=_mat(_av(x2))
  l, r=_coerce(l, r, OP_MUL)
  return ndarray._wrap(_native.matmul(l._native, r._native))

def diagonal(a:object, offset:int=0, axis1:int=0, axis2:int=1)->ndarray:
  arr=asarray(a)
  d=getattr(_native, "diagonal", None)                                                                           # native impl is feature-gated; fall back if absent
  return ndarray(d(arr._native, int(offset), int(axis1), int(axis2))) if d is not None else _diag_fallback(arr, int(offset), int(axis1), int(axis2))

def trace(a:object, offset:int=0, axis1:int=0, axis2:int=1, dtype:object=None, out:ndarray|None=None)->object:
  arr=asarray(a)
  tr=getattr(_native, "trace", None)
  if tr is not None:
    dc=-1 if dtype is None else _resolve_dtype(dtype).code
    r=ndarray(tr(arr._native, int(offset), int(axis1), int(axis2), dc))
    v=r._scalar() if r.ndim==0 else r
  else:
    v=_trace_fallback(arr, int(offset), int(axis1), int(axis2), dtype)
  if out is not None:
    o=asarray(out)
    o[...]=v
    return out
  return v

def astype(x:object, dtype:object, /, *, copy:builtins.bool=True, device:object=None)->ndarray:return asarray(x).astype(dtype, copy=copy, device=device)

def copy(a:object, order:str="K", subok:builtins.bool=False)->ndarray:
  _check_order(order)
  del subok
  arr=asarray(a)
  return arr.astype(arr.dtype, copy=True)

def ascontiguousarray(a:object, dtype:object=None, *, device:object=None)->ndarray:
  _check_cpu(device)
  if _is_scalar(a):
    return full((1,), a, dtype=_resolve_dtype(dtype) if dtype is not None else _isd(a), device=device)
  arr=asarray(a, dtype=dtype, copy=None)
  if arr.ndim==0:return arr.reshape((1,))
  if arr._native.is_c_contiguous():return arr
  return arr.astype(arr.dtype, copy=True)

def from_dlpack(x:object, /, *, device:object=None, copy:builtins.bool|None=None)->ndarray:
  if device is not None and device!="cpu":raise NotImplementedError("monpy v1 only supports cpu arrays")
  if runtime.ops_numpy.is_array_input(x):
    try:return runtime.ops_numpy._from_numpy_unchecked(x, copy=copy, device=device)
    except ValueError as exc:
      if copy is False and "readonly" in str(exc):raise BufferError(str(exc)) from exc
      raise
  from . import _dlpack
  return _dlpack.from_dlpack(x, copy)

def __array_namespace_info__()->object:
  def dts(*, device:object=None, kind:object=None)->dict[str, DType]:
    _check_cpu(device)
    return {d.name:d for d in _DT}
  def ddts(*, device:object=None)->dict[str, DType]:
    _check_cpu(device)
    return{"integral":int64, "real floating":float64, "bool":bool}
  return SimpleNamespace(default_device=lambda:"cpu", devices=lambda:["cpu"], dtypes=dts, default_dtypes=ddts, capabilities=lambda:{"boolean indexing":False, "data-dependent shapes":False})


def _binary(x1:object, x2:object, op:int, *, out:ndarray|None=None)->object:
  if out is not None:                                                                                                         # `out=` skips the deferred path; everything materialises into out._native
    if type(x1) is ndarray and type(x2) is ndarray:                                                                           # ndarray×ndarray fast path; mojo handles promotion in binary_into
      _native.binary_into(out._native, x1._native, x2._native, op)
      return out
    l=_mat(_av(x1))
    r=_mat(_av(x2))
    l, r=_coerce(l, r, op)
    _native.binary_into(out._native, l._native, r._native, op)
    return out
  f=_fuse(x1, x2, op)                                                                                                           # try sin(x)+scalar*y fusion before falling through
  if f is not None:return f
  if _isarrv(x1):return _binary_from_array(x1, x2, op, scalar_on_left=False)
  if _isarrv(x2):return _binary_from_array(x2, x1, op, scalar_on_left=True)
  l=asarray(x1)
  r=asarray(x2)
  l, r=_coerce(l, r, op)
  return ndarray(_native.binary(l._native, r._native, op))

def _binary_from_array(arr:ndarray|_DeferredArray, other:object, op:int, *, scalar_on_left:builtins.bool)->object:
  # called when `arr` is known array-like; `other` may be array, scalar, or convertible.
  if isinstance(arr, ndarray) and isinstance(other, ndarray):
    l, r=(other, arr) if scalar_on_left else (arr, other)
    if l.dtype is r.dtype:return ndarray(_native.binary(l._native, r._native, op))
    l, r=_coerce(l, r, op)
    return ndarray(_native.binary(l._native, r._native, op))
  if _isarrv(other):
    l=_mat(other) if scalar_on_left else _mat(arr)
    r=_mat(arr) if scalar_on_left else _mat(other)
    l, r=_coerce(l, r, op)
    return ndarray(_native.binary(l._native, r._native, op))
  if _is_scalar(other):
    sd=_isd_arr(arr, other)
    if op==OP_MUL and _can_def_sb(arr, sd):return _ScalarBinaryExpression(arr, other, sd, op, scalar_on_left)                       # defer (only mul, only floats) so a downstream sin(...)+(scalar*y) can fuse
    av=_mat(arr)
    return ndarray(_native.binary_scalar(av._native, other, sd.code, op, scalar_on_left))
  oa=asarray(other)
  if scalar_on_left:
    l, r=_coerce(oa, _mat(arr), op)
    return ndarray(_native.binary(l._native, r._native, op))
  l, r=_coerce(_mat(arr), oa, op)
  return ndarray(_native.binary(l._native, r._native, op))

def _unary(x:object, op:int)->object:
  a=_av(x)
  if op==UNARY_SIN and a.dtype in(float32, float64):return _UnaryExpression(a, op)                                              # only defer sin on floats — that's what sin_add_mul fuses
  av=_mat(a)
  return ndarray(_native.unary(av._native, op))

def _reduce(x:object, axis:object, op:int)->object:
  if axis is not None:raise NotImplementedError("axis-specific reductions are not implemented in monpy v1")
  a=_mat(_av(x))
  return _native.reduce(a._native, op).get_scalar(0)

def _isarrv(v:object)->typing.TypeGuard[ndarray|_DeferredArray]:return isinstance(v, (ndarray, _DeferredArray))                 # is "array value": ours, materialised or not
def _av(v:object)->ndarray|_DeferredArray:return v if isinstance(v, (ndarray, _DeferredArray)) else asarray(v)                  # as-array-value: convert non-arrays via asarray
def _mat(v:ndarray|_DeferredArray)->ndarray:return v._materialize() if isinstance(v, _DeferredArray) else v                    # materialize: collapse a deferred node to a concrete ndarray

def _coerce(l:ndarray, r:ndarray, op:int)->tuple[ndarray, ndarray]:
  # cast both operands to the common result dtype before handing to native binary kernel.
  # Hot path: both dtypes in the original 4-set covered by _BR's table; fall
  # back to mojo `result_dtype_for_binary` for any uint/f16/cross-kind pair.
  try:t=_BR[op][(l.dtype, r.dtype)]
  except KeyError:t=_DTC[_native._result_dtype_for_binary(l.dtype.code, r.dtype.code, op)]
  if l.dtype!=t:l=l.astype(t)
  if r.dtype!=t:r=r.astype(t)
  return l, r

def _result_dtype_for_binary(l:DType, r:DType, op:int)->DType:
  try:return _BR[op][(l, r)]
  except KeyError:return _DTC[_native._result_dtype_for_binary(l.code, r.code, op)]
def _result_dtype_for_unary(d:DType)->DType:
  if d in _UR:return _UR[d]
  return float64 if d.kind=="i" or d.kind=="u" or d==bool else d

def _can_def_sb(v:ndarray|_DeferredArray, sd:DType)->builtins.bool:return v.dtype in(float32, float64) and sd in(float32, float64)  # defer scalar binary only on float×float — int paths must materialise

def _fuse(x1:object, x2:object, op:int)->ndarray|None:
  # try to recognise sin(x)+scalar*y in either argument order.
  if op!=OP_ADD:return None
  r=_match_sam(x1, x2)
  return r if r is not None else _match_sam(x2, x1)

def _match_sam(x1:object, x2:object)->ndarray|None:
  # sam = sin(...) + scalar*(...). returns None unless both operands are the right deferred kinds.
  if not isinstance(x1, _UnaryExpression) or x1._op!=UNARY_SIN:return None
  if not isinstance(x2, _ScalarBinaryExpression) or x2._op!=OP_MUL:return None
  l=_mat(x1._base)
  r=_mat(x2._array)
  return ndarray(_native.sin_add_mul(l._native, r._native, x2._scalar, x2._scalar_dtype.code))

def _diag_fallback(arr:ndarray, offset:int, axis1:int, axis2:int)->ndarray:
  # python-side diagonal extraction used when the native diagonal op is unavailable.
  if arr.ndim<2:raise ValueError("diag requires an array of at least two dimensions")
  axis1=_norm_axis(axis1, arr.ndim)
  axis2=_norm_axis(axis2, arr.ndim)
  if axis1==axis2:raise ValueError("axis1 and axis2 cannot be the same")
  rs=builtins.max(-offset, 0)
  cs=builtins.max(offset, 0)                      # row/col start: positive offset slides diagonal up
  dl=builtins.max(0, builtins.min(arr.shape[axis1]-rs, arr.shape[axis2]-cs))  # diagonal length
  ra=tuple(a for a in range(arr.ndim) if a not in(axis1, axis2))             # remaining axes (kept as outer loop)
  rsh=tuple(arr.shape[a] for a in ra)
  out_shape=rsh+(dl,)
  flat:list[object]=[]
  for prefix in _iter_idx(rsh):
    key:list[object]=[0]*arr.ndim
    for a, i in zip(ra, prefix, strict=True):key[a]=i
    for di in range(dl):
      key[axis1]=rs+di
      key[axis2]=cs+di
      flat.append(arr[tuple(key)])
  return ndarray(_native.from_flat(flat, out_shape, arr.dtype.code))

def _trace_fallback(arr:ndarray, offset:int, axis1:int, axis2:int, dt:object)->object:
  # python-side trace when native is missing — reduces along the diagonal.
  d=diagonal(arr, offset=offset, axis1=axis1, axis2=axis2)
  t=_resolve_dtype(dt) if dt is not None else (int64 if d.dtype==bool else d.dtype)  # bool inputs accumulate as int64 to match numpy
  if d.dtype!=t:d=d.astype(t)
  if d.ndim==1:return sum(d)
  out_shape=d.shape[:-1]
  flat:list[object]=[]
  for prefix in _iter_idx(out_shape):
    total:builtins.float|builtins.int=0.0 if t in(float32, float64) else 0
    for di in range(d.shape[-1]):total+=typing.cast(_Scalar, d[prefix+(di,)])
    flat.append(total)
  return ndarray(_native.from_flat(flat, out_shape, t.code))

def _is_scalar(v:object)->builtins.bool:return isinstance(v, (builtins.bool, builtins.int, builtins.float))                      # python scalar (bool/int/float)

def _isd(v:object)->DType:                                                                                                    # infer scalar dtype from a python scalar alone
  if isinstance(v, builtins.bool):return bool
  if isinstance(v, builtins.int):return int64
  return float64

def _scalar_dtype_for_array_dtype(ad:DType, v:object)->DType:
  # NumPy 2.x keeps python scalars weak around arrays: f32+1 and f32+1.5 stay f32, while int64+1.5 promotes to f64.
  if ad in(float32, float64) and isinstance(v, (builtins.bool, builtins.int, builtins.float)):return ad
  if ad==int64 and isinstance(v, (builtins.bool, builtins.int)):return int64
  return _isd(v)

def _isd_arr(arr:ndarray|_DeferredArray, v:object)->DType:
  # like _isd but biased toward the array's float dtype, so `f32_array * 2` stays f32 instead of upcasting via int64.
  return _scalar_dtype_for_array_dtype(arr.dtype, v)

def _dtype_for_result_type_arg(v:object)->tuple[DType, builtins.bool]:
  if isinstance(v, (ndarray, _DeferredArray)):return v.dtype, True
  if _is_scalar(v):return _isd(v), False
  if isinstance(v, DType):return v, True
  if isinstance(v, str):return _resolve_dtype(v), True
  if _has_ai(v):return _dtype_from_typestr(_ai_typestr(_array_interface(v))), True
  try:return _resolve_dtype(v), True
  except NotImplementedError:pass
  try:return _infer_dtype(_flat(v)[1]), True
  except NotImplementedError as exc:raise NotImplementedError(f"unsupported result_type argument: {v!r}") from exc

def _dtype_for_can_cast_arg(v:object)->DType:
  if _is_scalar(v):raise TypeError("can_cast() does not support Python ints, floats, and complex because the result used to depend on the value.")
  if isinstance(v, (ndarray, _DeferredArray)):return v.dtype
  if _has_ai(v):return _dtype_from_typestr(_ai_typestr(_array_interface(v)))
  return _resolve_dtype(v)

def _abstract_dtype_set(v:object, *, for_isdtype:builtins.bool)->set[DType]|None:
  if isinstance(v, str):
    if not for_isdtype:return None
    if v not in _ISDTYPE_KINDS:raise ValueError(f"kind argument is a string, but {v!r} is not a known kind name.")
    return _ISDTYPE_KINDS[v]
  np_abstract=runtime.ops_numpy.abstract_dtype_set(v)
  if np_abstract is not None:return np_abstract
  if not for_isdtype:
    if v is builtins.bool:return{bool}
    if v is builtins.int:return{int64}
    if v is builtins.float:return{float64}
  return None

def _resolve_dtype(v:object)->DType:
  if v is None:                return float64
  if isinstance(v, DType):      return v
  if isinstance(v, str):
    if v in _DTN:              return _DTN[v]
    if v in _DTBT:             return _DTBT[v]
    if v in _DTF:              return _DTF[v]
    raise NotImplementedError(f"unsupported dtype: {v}")
  if v is builtins.bool:       return bool
  if v is builtins.int:        return int64
  if v is builtins.float:      return float64
  if v is builtins.complex:    return complex128
  if runtime.ops_numpy.is_dtype_input(v): return runtime.ops_numpy.resolve_dtype(v)
  raise NotImplementedError(f"unsupported dtype: {v!r}")

def _dtype_from_typestr(typestr:str)->DType:
  try:return _DTBT[typestr]
  except KeyError as exc:raise NotImplementedError(f"unsupported array-interface typestr: {typestr}") from exc

def _array_interface(o:object)->dict[str, object]:
  try:i=getattr(o, "__array_interface__")
  except Exception as exc:raise NotImplementedError("object does not expose __array_interface__") from exc
  if not isinstance(i, dict):raise NotImplementedError("object __array_interface__ must be a dict")
  return i

def _ai_shape(iface:dict[str, object])->tuple[int, ...]:
  sh=iface.get("shape")
  if not isinstance(sh, tuple) or not builtins.all(isinstance(d, builtins.int) for d in sh):raise NotImplementedError("array interface shape must be a tuple of ints")
  return typing.cast(tuple[int, ...], sh)

def _ai_strides(iface:dict[str, object])->tuple[int, ...]|None:
  st=iface.get("strides")
  if st is None:return None
  if not isinstance(st, tuple) or not builtins.all(isinstance(s, builtins.int) for s in st):raise NotImplementedError("array interface strides must be a tuple of ints or None")
  return typing.cast(tuple[int, ...], st)

def _ai_typestr(iface:dict[str, object])->str:
  ts=iface.get("typestr")
  if not isinstance(ts, str):raise NotImplementedError("array interface typestr must be a string")
  return ts

def _ai_data(iface:dict[str, object])->tuple[int, builtins.bool]:
  data=iface.get("data")
  if not isinstance(data, tuple) or len(data)!=2 or not isinstance(data[0], builtins.int):raise NotImplementedError("array interface data must be an address tuple")
  return data[0], builtins.bool(data[1])

def _has_ai(o:object)->builtins.bool:                                                                                         # has __array_interface__ dict — the universal zero-copy hand-off
  try:i=getattr(o, "__array_interface__")
  except Exception:return False
  return isinstance(i, dict)

def _ai_asarray(obj:object, *, dtype:object, copy:builtins.bool|None)->ndarray:
  # generic array-interface ingest. Handles dtype conversion, readonly, and copy semantics
  # before deciding between zero-copy view (_ext_from_ai) and explicit copy (_copy_from_ai).
  a=typing.cast(_ArrayInterfaceLike, obj)
  iface=a.__array_interface__
  sd=_dtype_from_typestr(_ai_typestr(iface))                                                                                   # source dtype
  t=_resolve_dtype(dtype) if dtype is not None else None
  if t is not None and t!=sd:
    if copy is False:raise ValueError(_CFE)
    return _copy_from_ai(a, dtype_value=sd, iface=iface).astype(t)
  da, ro=_ai_data(iface)                                                                                                        # data_address, readonly flag
  if copy is True:return _copy_from_ai(a, dtype_value=sd, iface=iface)
  if ro:
    if copy is False:raise ValueError("readonly array requires copy=True")
    return _copy_from_ai(a, dtype_value=sd, iface=iface)
  return _ext_from_ai(a, sd, iface=iface, data_address=da)

def _ext_from_ai(a:object, d:DType, *, iface:dict[str, object]|None=None, data_address:int|None=None)->ndarray:
  # zero-copy borrow: native side wraps the foreign buffer; we keep `a` as owner so it isn't GC'd.
  if iface is None:iface=_array_interface(a)
  sh=_ai_shape(iface)
  rs=_ai_strides(iface)
  i=d.itemsize
  if rs is None:bs=_csb(sh, i)
  else:
    bs=rs
    for s in bs:
      if s%i!=0:raise NotImplementedError("array interface strides must align to dtype itemsize")
  es=tuple(s//i for s in bs)                                                                                                   # convert byte strides → element strides for native side
  if data_address is None:data_address=_ai_data(iface)[0]
  bl=math.prod(sh)*i                                                                                                           # byte length (used by native bounds check)
  return ndarray(_native.from_external(data_address, sh, es, d.code, bl), owner=a)

def _copy_from_ai(a:object, *, dtype_value:DType|None=None, iface:dict[str, object]|None=None)->ndarray:
  # forced-copy ingest from any array-interface source.
  if iface is None:iface=_array_interface(a)
  if dtype_value is None:dtype_value=_dtype_from_typestr(_ai_typestr(iface))
  sh=_ai_shape(iface)
  rs=_ai_strides(iface)
  i=dtype_value.itemsize
  if rs is None:es=_cse(sh)
  else:
    for s in rs:
      if s%i!=0:raise NotImplementedError("array interface strides must align to dtype itemsize")
    es=tuple(s//i for s in rs)
  da=_ai_data(iface)[0]
  bl=math.prod(sh)*i
  return ndarray(_native.copy_from_external(da, sh, es, dtype_value.code, bl))

def _cse(sh:tuple[int, ...])->tuple[int, ...]:                                                                                  # c-contiguous strides in elements
  st=[1]*len(sh)
  s=1
  for a in range(len(sh)-1, -1, -1):
    st[a]=s
    s*=sh[a]
  return tuple(st)

def _infer_dtype(flat:Sequence[object])->DType:
  # promote up the ladder bool→int→float→complex depending on what we see.
  if not flat:return float64
  hc=False
  hf=False
  hi=False
  hb=True                                                                                           # has_complex / float / int / bool (so far still all-bool)
  for v in flat:
    if isinstance(v, builtins.bool):continue
    hb=False
    if isinstance(v, builtins.int):
      hi=True
      continue
    if isinstance(v, builtins.float):
      hf=True
      continue
    if isinstance(v, builtins.complex):
      hc=True
      continue
    raise NotImplementedError(f"unsupported array value type: {type(v).__name__}")
  if hc:return complex128
  if hf:return float64
  if hi or not hb:return int64
  return bool

def _norm_shape(shape:int|Sequence[int])->tuple[int, ...]:
  if isinstance(shape, builtins.int):
    if shape<0:raise ValueError("negative dimensions are not allowed")
    return(shape,)
  out=tuple(int(d) for d in shape)
  if builtins.any(d<0 for d in out):raise ValueError("negative dimensions are not allowed")
  return out

def _csb(sh:tuple[int, ...], i:int)->tuple[int, ...]:                                                                            # c-contiguous strides in bytes (itemsize=i)
  st=[0]*len(sh)
  s=i
  for a in range(len(sh)-1, -1, -1):
    st[a]=s
    s*=sh[a]
  return tuple(st)

def _is_c_contig(sh:tuple[int, ...], st:tuple[int, ...], i:int)->builtins.bool:
  # walk axes right-to-left checking strides match c-contiguous expectation; size-0 axes short-circuit true.
  e=i
  for a in range(len(sh)-1, -1, -1):
    if sh[a]==0:return True
    if sh[a]!=1 and st[a]!=e:return False
    e*=sh[a]
  return True

def _iter_idx(sh:tuple[int, ...])->Iterable[tuple[int, ...]]:
  if not sh:
    yield()
    return
  yield from itertools.product(*(range(d) for d in sh))

def _shape_args(sh:Sequence[int|Sequence[int]])->tuple[int, ...]:
  # accept both `reshape(2,3)` and `reshape((2,3))` calling conventions.
  if len(sh)==1 and not isinstance(sh[0], builtins.int):return _norm_shape(sh[0])
  return _norm_shape(typing.cast(Sequence[int], sh))

def _flat(obj:object)->tuple[tuple[int, ...], list[object]]:
  # depth-first flatten of nested list/tuple → (shape, flat values). Rejects ragged sequences.
  if isinstance(obj, (list, tuple)):
    if not obj:return(0,), []
    cs:list[tuple[int, ...]]=[]
    fl:list[object]=[]
    for it in obj:
      s, f=_flat(it)
      cs.append(s)
      fl.extend(f)
    fs=cs[0]
    if builtins.any(s!=fs for s in cs):raise ValueError("cannot create monpy array from ragged nested sequences")
    return(len(obj),)+fs, fl
  if isinstance(obj, ndarray):return obj.shape, [obj._native.get_scalar(i) for i in range(obj.size)]
  if isinstance(obj, (builtins.bool, builtins.int, builtins.float, builtins.complex)):return(), [obj]
  raise NotImplementedError(f"unsupported array input type: {type(obj).__name__}")

def _unflat(flat:Sequence[object], sh:tuple[int, ...])->object:
  if not sh:return flat[0]
  if len(sh)==1:return list(flat[:sh[0]])
  step=math.prod(sh[1:])
  return[_unflat(flat[i*step:(i+1)*step], sh[1:]) for i in range(sh[0])]

def _expand_key(key:object, ndim:int)->tuple[object, ...]:
  # normalise an indexing key, expanding Ellipsis while preserving None axes.
  if key==():
    if ndim!=0:raise IndexError("empty index is only valid for zero-dimensional arrays")
    return()
  parts=key if isinstance(key, tuple) else(key,)
  if parts.count(Ellipsis)>1:raise IndexError("an index can only have a single ellipsis")
  used=builtins.sum(1 for p in parts if p is not None and p is not Ellipsis)
  if used>ndim:raise IndexError("too many indices for array")
  if Ellipsis in parts:
    ea=parts.index(Ellipsis)
    missing=ndim-used
    parts=parts[:ea]+(slice(None),)*missing+parts[ea+1:]
  used=builtins.sum(1 for p in parts if p is not None)
  if used>ndim:raise IndexError("too many indices for array")
  return parts+(slice(None),)*(ndim-used)

def _norm_idx(i:object, d:int)->int:
  if not isinstance(i, builtins.int):raise NotImplementedError("monpy v1 supports only integer and slice indexing")
  if i<0:i+=d
  if i<0 or i>=d:raise IndexError("index out of bounds")
  return i

def _norm_axis(a:int, n:int)->int:
  if a<0:a+=n
  if a<0 or a>=n:raise ValueError("axis out of bounds")
  return a

def _norm_axes(axes:Sequence[int], n:int)->tuple[int, ...]:
  r=tuple(a+n if a<0 else a for a in axes)
  if sorted(r)!=list(range(n)):raise ValueError("axes must be a permutation of dimensions")
  return r

def _check_cpu(d:object)->None:
  if d not in(None, "cpu"):raise NotImplementedError("monpy v1 only supports cpu arrays")

def _check_order(order:str)->None:
  if order not in("C", "A", "K"):raise NotImplementedError("monpy v1 only supports c-contiguous materialization")

def _is_kernel_value(v:object)->builtins.bool:return builtins.bool(getattr(v, "__monpy_kernel_tensor__", False))
def _has_kernel_arg(args:Sequence[object])->builtins.bool:return builtins.any(_is_kernel_value(arg) for arg in args)

runtime=importlib.import_module(f"{__name__}.runtime")
linalg=importlib.import_module(f"{__name__}.linalg")
nn=importlib.import_module(f"{__name__}.nn")

# Top-level numpy aliases for the linalg surface; numpy exposes both `numpy.dot` and `numpy.linalg` paths.
dot=linalg.dot
vdot=linalg.vdot
inner=linalg.inner
outer=linalg.outer
tensordot=linalg.tensordot
kron=linalg.kron
cross=linalg.cross
matvec=linalg.matvec
vecmat=linalg.vecmat
vecdot=linalg.vecdot
einsum=linalg.einsum

# monpy extensions
layer_norm=nn.layer_norm
scaled_masked_softmax=nn.scaled_masked_softmax
softmax=nn.softmax
from_numpy=runtime.ops_numpy.from_numpy

_KERNEL_LAZY_EXPORTS={"jit", "Tensor", "TensorSpec", "LayoutSpec", "TileSpec", "DTypeSpec", "DeviceSpec", "SymbolicDim"}

def __getattr__(name:str)->object:
  if name in _KERNEL_LAZY_EXPORTS:
    kernels=importlib.import_module(f"{__name__}.kernels")
    return getattr(kernels, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__=["DType", "Ufunc", "abs", "absolute", "add", "all", "any", "append", "arange", "arccos", "arcsin", "arctan", "arctan2", "argpartition", "argsort", "argwhere", "array", "array_split", "asarray", "ascontiguousarray", "argmax", "argmin", "asfortranarray", "astype", "atleast_1d", "atleast_2d", "atleast_3d", "average", "bincount", "block", "bool", "bool_", "broadcast_arrays", "broadcast_to", "can_cast", "cbrt", "cdouble", "ceil", "clongdouble", "column_stack", "complex_", "complex64", "complex128", "concatenate", "conj", "conjugate", "copy", "copysign", "cos", "cosh", "count_nonzero", "cross", "csingle", "cummax", "cummin", "cumprod", "cumsum", "deg2rad", "degrees", "delete", "diagonal", "diag_indices", "digitize", "divide", "dot", "dsplit", "dstack", "dtype", "einsum", "e", "empty", "empty_like", "equal", "exp", "exp2", "expm1", "expand_dims", "eye", "fabs", "finfo", "fix", "flatnonzero", "flatten", "flip", "fliplr", "flipud", "float_", "float16", "float32", "float64", "floor", "floor_divide", "fmax", "fmin", "frombuffer", "from_dlpack", "fromiter", "full", "full_like", "geomspace", "greater", "greater_equal", "half", "hsplit", "hstack", "hypot", "identity", "iinfo", "imag", "indices", "inf", "inner", "insert", "int_", "int8", "int16", "int32", "int64", "intersect1d", "intp", "isdtype", "isfinite", "isinf", "isin", "isnan", "issubdtype", "ix_", "kron", "less", "less_equal", "lexsort", "linalg", "layer_norm", "linspace", "log", "log2", "log10", "log1p", "logical_and", "logical_not", "logical_or", "logical_xor", "logspace", "matmul", "matrix_transpose", "matvec", "max", "maximum", "mean", "median", "meshgrid", "min", "minimum", "mod", "moveaxis", "multiply", "nan", "nanargmax", "nanargmin", "nancumprod", "nancumsum", "nanmax", "nanmean", "nanmedian", "nanmin", "nanpercentile", "nanprod", "nanquantile", "nanstd", "nansum", "nanvar", "ndarray", "negative", "newaxis", "nn", "nonzero", "not_equal", "ones", "ones_like", "outer", "pad", "partition", "percentile", "pi", "positive", "power", "prod", "promote_types", "ptp", "put", "put_along_axis", "quantile", "radians", "rad2deg", "ravel", "ravel_multi_index", "real", "reciprocal", "remainder", "repeat", "reshape", "result_type", "rint", "roll", "rot90", "scaled_masked_softmax", "searchsorted", "setdiff1d", "setxor1d", "sign", "signbit", "sin", "sin_add_mul", "sinh", "softmax", "sort", "split", "sqrt", "square", "squeeze", "stack", "std", "subtract", "sum", "swapaxes", "take", "take_along_axis", "tan", "tanh", "angle", "tensordot", "tile", "trace", "transpose", "tri", "trim_zeros", "triu", "tril", "triu_indices", "tril_indices", "true_divide", "trunc", "ubyte", "ufunc", "uint8", "uint16", "uint32", "uint64", "uintc", "ulonglong", "union1d", "unique", "unravel_index", "ushort", "var", "vdot", "vecdot", "vecmat", "vsplit", "vstack", "where", "zeros", "zeros_like"]+sorted(_KERNEL_LAZY_EXPORTS)
