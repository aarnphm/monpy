# fmt: off # ruff: noqa
"""JAX-shaped explicit keys plus NumPy-shaped random conveniences."""

from __future__ import annotations

import builtins, time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import ClassVar, TypeAlias, overload, cast

from . import (
  DType,
  _native,
  _py_float,
  _py_int,
  _resolve_dtype,
  asarray,
  float32,
  float64,
  int8,
  int16,
  int32,
  int64,
  ndarray,
  uint8,
  uint16,
  uint32,
  uint64,
)

_DEFAULT_IMPL="threefry2x32"
_FLOAT_DTYPES=(float32, float64)
_BITS_DTYPES=(uint32, uint64)
_INT_DTYPES=(int64, int32, int16, int8, uint64, uint32, uint16, uint8)
_FloatOrArray:TypeAlias=builtins.float|ndarray
_IntOrArray:TypeAlias=builtins.int|ndarray


def _check_impl(impl:str)->str:
  if impl!=_DEFAULT_IMPL:raise NotImplementedError("monpy.random v1 only supports threefry2x32")
  return impl


def _shape(size:object)->tuple[int, ...]:
  if size is None:return ()
  if isinstance(size, builtins.int):return (builtins.int(size),)
  if _seq(size):return tuple(_py_int(dim) for dim in cast(Sequence[object], size))
  raise TypeError("size/shape must be None, an int, or a sequence of ints")


def _seq(value:object)->bool:
  return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _fresh_seed()->int:
  return time.time_ns() & ((1<<63)-1)


def _as_scalar_if(arr:ndarray, scalar:bool)->object:
  return arr._scalar() if scalar else arr


def _dtype(dtype:object, choices:tuple[DType, ...], message:str)->DType:
  resolved=_resolve_dtype(dtype)
  if resolved not in choices:raise TypeError(message)
  return resolved


def _float_dtype(dtype:object)->DType:return _dtype(dtype, _FLOAT_DTYPES, "random floating dtype must be float32 or float64")
def _bits_dtype(dtype:object)->DType:return _dtype(dtype, _BITS_DTYPES, "bits dtype must be uint32 or uint64")
def _int_dtype(dtype:object)->DType:return _dtype(dtype, _INT_DTYPES, "random integer dtype must be an integer dtype")


@dataclass(frozen=True, slots=True)
class Key:
  word0:int
  word1:int
  impl:str=_DEFAULT_IMPL
  __monpy_random_key__:ClassVar[bool]=True

  def __post_init__(self)->None:
    _check_impl(self.impl)
    if not (0<=self.word0<=0xFFFFFFFF and 0<=self.word1<=0xFFFFFFFF):
      raise ValueError("random key words must be uint32 values")

  @property
  def words(self)->tuple[int, int]:return (self.word0, self.word1)


@dataclass(frozen=True, slots=True)
class KeyBatch:
  _data:ndarray
  impl:str=_DEFAULT_IMPL
  __monpy_random_key_batch__:ClassVar[bool]=True

  def __post_init__(self)->None:
    _check_impl(self.impl)
    if self._data.dtype!=uint32 or self._data.ndim<1 or self._data.shape[-1]!=2:
      raise ValueError("KeyBatch data must have dtype uint32 and trailing dimension 2")

  @property
  def shape(self)->tuple[int, ...]:return self._data.shape[:-1]

  @property
  def ndim(self)->int:return len(self.shape)

  def __len__(self)->int:
    if not self.shape:raise TypeError("len() of unsized KeyBatch")
    return self.shape[0]

  def __iter__(self)->Iterator[Key]:
    for index in range(len(self)):yield self[index]

  @overload
  def __getitem__(self, index:int)->Key:...
  @overload
  def __getitem__(self, index:object)->Key|KeyBatch:...
  def __getitem__(self, index:object)->Key|KeyBatch:
    item=cast(ndarray, self._data[index])
    if item.shape==(2,):return _key_from_words(_py_int(item[0]), _py_int(item[1]), self.impl)
    return KeyBatch(asarray(item, dtype=uint32, copy=True), self.impl)


def _key_from_words(word0:int, word1:int, impl:str)->Key:
  return Key(builtins.int(word0), builtins.int(word1), impl)


def _key_from_data(data:ndarray, impl:str)->Key:
  if data.dtype!=uint32 or data.shape!=(2,):raise ValueError("key data must have dtype uint32 and shape (2,)")
  return _key_from_words(_py_int(data[0]), _py_int(data[1]), impl)


def _coerce_key(value:object)->Key:
  if isinstance(value, Key):return value
  raise TypeError("expected a monpy.random.Key")


def _next_key()->Key:
  global _global_key
  _global_key, out=_split2(_global_key)
  return out


def _split2(source:Key)->tuple[Key, Key]:
  pair=split(source, 2)
  return pair[0], pair[1]


def _parse(name:str, args:tuple[object, ...], key:Key|None, size:object, shape_first:bool=True)->tuple[list[object], Key|None, object]:
  values=list(args)
  if values and isinstance(values[0], Key):
    if key is not None:raise TypeError(f"{name}() got multiple keys")
    key=cast(Key, values.pop(0))
  if shape_first and key is not None and values and _seq(values[0]) and size is None:size=values.pop(0)
  return values, key, size


def _scaled(base:ndarray, loc:object, scale:object)->ndarray:
  return cast(ndarray, cast(ndarray, base*_py_float(scale))+_py_float(loc))


def _maybe_scalar(arr:ndarray, key:Key|None, size:object)->object:
  return arr if key is not None or size is not None else arr._scalar()


def key(seed:int, *, impl:str=_DEFAULT_IMPL)->Key:
  impl=_check_impl(impl)
  return _key_from_data(ndarray(_native._random_key(builtins.int(seed))), impl)


def split(key:Key, num:int=2)->KeyBatch:
  source=_coerce_key(key)
  count=builtins.int(num)
  if count<0:raise ValueError("split num must be non-negative")
  data=ndarray(_native._random_split(source.word0, source.word1, count))
  return KeyBatch(data, source.impl)


def fold_in(key:Key, data:int)->Key:
  source=_coerce_key(key)
  return _key_from_data(ndarray(_native._random_fold_in(source.word0, source.word1, builtins.int(data))), source.impl)


def key_data(key_or_batch:Key|KeyBatch)->ndarray:
  if isinstance(key_or_batch, Key):
    return ndarray(_native._random_key_data(key_or_batch.word0, key_or_batch.word1))
  if isinstance(key_or_batch, KeyBatch):
    return asarray(key_or_batch._data, dtype=uint32, copy=True)
  raise TypeError("key_data expects a Key or KeyBatch")


def wrap_key_data(data:object, *, impl:str=_DEFAULT_IMPL)->Key|KeyBatch:
  impl=_check_impl(impl)
  arr=asarray(data, dtype=uint32, copy=True)
  if arr.shape==(2,):return _key_from_data(arr, impl)
  if arr.ndim>=2 and arr.shape[-1]==2:return KeyBatch(arr, impl)
  raise ValueError("key data must have shape (2,) or (..., 2)")


def bits(key:Key, shape:object=(), dtype:object=uint32)->ndarray:
  source=_coerce_key(key)
  t=_bits_dtype(dtype)
  return ndarray(_native._random_bits(source.word0, source.word1, _shape(shape), t.code))


def _uniform_array(source:Key, size:object, dtype:object, low:object, high:object)->ndarray:
  t=_float_dtype(dtype)
  return ndarray(_native._random_uniform(source.word0, source.word1, _shape(size), t.code, _py_float(low), _py_float(high)))


def _normal_array(source:Key, size:object, dtype:object)->ndarray:
  t=_float_dtype(dtype)
  return ndarray(_native._random_normal(source.word0, source.word1, _shape(size), t.code))


def _randint_array(source:Key, low:object, high:object, size:object, dtype:object)->ndarray:
  t=_int_dtype(dtype)
  lo=_py_int(low)
  hi=_py_int(high)
  return ndarray(_native._random_randint(source.word0, source.word1, _shape(size), lo, hi, t.code))


@overload
def random(key:Key, size:object=None, dtype:object=float64)->ndarray:...
@overload
def random(key:None=None, size:None=None, dtype:object=float64)->builtins.float:...
@overload
def random(key:int|Sequence[int], size:None=None, dtype:object=float64)->ndarray:...
@overload
def random(key:None=None, size:int|Sequence[int]=..., dtype:object=float64)->ndarray:...
@overload
def random(key:object=None, size:object=None, dtype:object=float64)->_FloatOrArray:...
def random(key:object=None, size:object=None, dtype:object=float64)->_FloatOrArray:
  if isinstance(key, Key):
    return _uniform_array(key, size, dtype, 0.0, 1.0)
  if key is not None:
    if size is not None:raise TypeError("random() got both positional size and size=")
    size=key
  return cast(_FloatOrArray, _as_scalar_if(_uniform_array(_next_key(), size, dtype, 0.0, 1.0), size is None))


@overload
def random_sample(size:None=None)->builtins.float:...
@overload
def random_sample(size:int|Sequence[int])->ndarray:...
def random_sample(size:object=None)->_FloatOrArray:return random(None, size=size, dtype=float64)
@overload
def sample(size:None=None)->builtins.float:...
@overload
def sample(size:int|Sequence[int])->ndarray:...
def sample(size:object=None)->_FloatOrArray:return random(None, size=size, dtype=float64)
@overload
def ranf(size:None=None)->builtins.float:...
@overload
def ranf(size:int|Sequence[int])->ndarray:...
def ranf(size:object=None)->_FloatOrArray:return random(None, size=size, dtype=float64)


@overload
def standard_normal(size:Key, dtype:object=float64, *, key:None=None)->ndarray:...
@overload
def standard_normal(size:None=None, dtype:object=float64, *, key:Key)->ndarray:...
@overload
def standard_normal(size:int|Sequence[int], dtype:object=float64, *, key:Key|None=None)->ndarray:...
@overload
def standard_normal(size:None=None, dtype:object=float64, *, key:None=None)->builtins.float:...
def standard_normal(size:object=None, dtype:object=float64, *, key:Key|None=None)->_FloatOrArray:
  if isinstance(size, Key):
    if key is not None:raise TypeError("standard_normal() got multiple keys")
    key=size
    if _seq(dtype):
      size=dtype
      dtype=float64
    else:size=None
  return cast(_FloatOrArray, _maybe_scalar(_normal_array(key if key is not None else _next_key(), size, dtype), key, size))


def uniform(*args:object, key:Key|None=None, low:object=0.0, high:object=1.0, size:object=None, dtype:object=float64)->_FloatOrArray:
  values, key, size=_parse("uniform", args, key, size)
  if values:low=values.pop(0)
  if values:high=values.pop(0)
  if values:size=values.pop(0)
  if values:raise TypeError("uniform() takes at most key, low, high, and size positional arguments")
  return cast(_FloatOrArray, _maybe_scalar(_uniform_array(key if key is not None else _next_key(), size, dtype, low, high), key, size))


def normal(*args:object, key:Key|None=None, loc:object=0.0, scale:object=1.0, size:object=None, dtype:object=float64)->_FloatOrArray:
  values, key, size=_parse("normal", args, key, size)
  if values:loc=values.pop(0)
  if values:scale=values.pop(0)
  if values:size=values.pop(0)
  if values:raise TypeError("normal() takes at most key, loc, scale, and size positional arguments")
  return cast(_FloatOrArray, _maybe_scalar(_scaled(_normal_array(key if key is not None else _next_key(), size, dtype), loc, scale), key, size))


def randint(*args:object, key:Key|None=None, low:object|None=None, high:object|None=None, size:object=None, dtype:object=int64)->_IntOrArray:
  values, key, size=_parse("randint", args, key, size, False)
  if low is None:
    if not values:raise TypeError("randint() missing required argument 'low'")
    low=values.pop(0)
  if values:high=values.pop(0)
  if values:size=values.pop(0)
  if values:raise TypeError("randint() takes at most key, low, high, and size positional arguments")
  lo:object
  hi:object
  if high is None:
    lo=0
    hi=low
  else:
    lo=low
    hi=high
  return cast(_IntOrArray, _maybe_scalar(_randint_array(key if key is not None else _next_key(), lo, hi, size, dtype), key, size))


@overload
def rand()->builtins.float:...
@overload
def rand(dim:int, *dims:int)->ndarray:...
def rand(*dims:int)->_FloatOrArray:return random(None, size=dims if dims else None, dtype=float64)
@overload
def randn()->builtins.float:...
@overload
def randn(dim:int, *dims:int)->ndarray:...
def randn(*dims:int)->_FloatOrArray:return standard_normal(size=dims if dims else None, dtype=float64)


class Generator:
  __slots__=("_key",)

  def __init__(self, seed:object=None)->None:
    if isinstance(seed, Key):self._key=seed
    elif seed is None:self._key=key(_fresh_seed())
    else:self._key=key(_py_int(seed))

  def _draw_key(self)->Key:
    self._key, out=_split2(self._key)
    return out

  @overload
  def random(self, size:None=None, dtype:object=float64)->builtins.float:...
  @overload
  def random(self, size:int|Sequence[int], dtype:object=float64)->ndarray:...
  def random(self, size:object=None, dtype:object=float64)->_FloatOrArray:
    return cast(_FloatOrArray, _as_scalar_if(_uniform_array(self._draw_key(), size, dtype, 0.0, 1.0), size is None))

  @overload
  def uniform(self, low:object=0.0, high:object=1.0, size:None=None)->builtins.float:...
  @overload
  def uniform(self, low:object=0.0, high:object=1.0, size:int|Sequence[int]=...)->ndarray:...
  def uniform(self, low:object=0.0, high:object=1.0, size:object=None)->_FloatOrArray:
    return cast(_FloatOrArray, _as_scalar_if(_uniform_array(self._draw_key(), size, float64, low, high), size is None))

  @overload
  def standard_normal(self, size:None=None, dtype:object=float64)->builtins.float:...
  @overload
  def standard_normal(self, size:int|Sequence[int], dtype:object=float64)->ndarray:...
  def standard_normal(self, size:object=None, dtype:object=float64)->_FloatOrArray:
    return cast(_FloatOrArray, _as_scalar_if(_normal_array(self._draw_key(), size, dtype), size is None))

  @overload
  def normal(self, loc:object=0.0, scale:object=1.0, size:None=None)->builtins.float:...
  @overload
  def normal(self, loc:object=0.0, scale:object=1.0, size:int|Sequence[int]=...)->ndarray:...
  def normal(self, loc:object=0.0, scale:object=1.0, size:object=None)->_FloatOrArray:
    result=_scaled(_normal_array(self._draw_key(), size, float64), loc, scale)
    return result if size is not None else result._scalar()

  @overload
  def integers(self, low:object, high:object|None=None, size:None=None, dtype:object=int64, endpoint:bool=False)->builtins.int:...
  @overload
  def integers(self, low:object, high:object|None=None, size:int|Sequence[int]=..., dtype:object=int64, endpoint:bool=False)->ndarray:...
  def integers(self, low:object, high:object|None=None, size:object=None, dtype:object=int64, endpoint:bool=False)->_IntOrArray:
    if high is None:
      lo, hi=0, low
    else:lo, hi=low, high
    hi=_py_int(hi)+(1 if endpoint else 0)
    return cast(_IntOrArray, _as_scalar_if(_randint_array(self._draw_key(), lo, hi, size, dtype), size is None))


def default_rng(seed:object=None)->Generator:
  return seed if isinstance(seed, Generator) else Generator(seed)


def seed(seed:int|None=None)->None:
  global _global_key
  _global_key=key(_fresh_seed() if seed is None else _py_int(seed))


_global_key=key(0)

__all__=["Generator", "Key", "KeyBatch", "bits", "default_rng", "fold_in", "key", "key_data", "normal", "rand", "randint", "randn", "random", "random_sample", "ranf", "sample", "seed", "split", "standard_normal", "uniform", "wrap_key_data"]
