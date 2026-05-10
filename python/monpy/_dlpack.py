from __future__ import annotations

import ctypes
import math
import typing

if typing.TYPE_CHECKING:
  import monpy as mp

_DLPACK_MAJOR = 1
_DLPACK_MINOR = 0
_DLPACK_CPU = 1
_FLAG_READ_ONLY = 1 << 0
_FLAG_IS_COPIED = 1 << 1
_DLC_INT = 0
_DLC_UINT = 1
_DLC_FLOAT = 2
_DLC_COMPLEX = 5
_DLC_BOOL = 6
_NAME_LEGACY = b"dltensor"
_NAME_LEGACY_USED = b"used_dltensor"
_NAME_VERSIONED = b"dltensor_versioned"
_NAME_VERSIONED_USED = b"used_dltensor_versioned"


class _DLDevice(ctypes.Structure):
  _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
  _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
  _fields_ = [
    ("data", ctypes.c_void_p),
    ("device", _DLDevice),
    ("ndim", ctypes.c_int),
    ("dtype", _DLDataType),
    ("shape", ctypes.POINTER(ctypes.c_int64)),
    ("strides", ctypes.POINTER(ctypes.c_int64)),
    ("byte_offset", ctypes.c_uint64),
  ]


class _DLManagedTensor(ctypes.Structure):
  pass


class _DLPackVersion(ctypes.Structure):
  _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class _DLManagedTensorVersioned(ctypes.Structure):
  pass


_LegacyDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_VersionedDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensorVersioned))
_CapsuleDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

_DLManagedTensor._fields_ = [
  ("dl_tensor", _DLTensor),
  ("manager_ctx", ctypes.c_void_p),
  ("deleter", ctypes.c_void_p),
]
_DLManagedTensorVersioned._fields_ = [
  ("version", _DLPackVersion),
  ("manager_ctx", ctypes.c_void_p),
  ("deleter", ctypes.c_void_p),
  ("flags", ctypes.c_uint64),
  ("dl_tensor", _DLTensor),
]

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
_PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_IsValid.restype = ctypes.c_int
_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_SetName = ctypes.pythonapi.PyCapsule_SetName
_PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_SetName.restype = ctypes.c_int
_CAPSULE_API = ctypes.PyDLL(None)
_PyCapsule_IsValidPtr = _CAPSULE_API.PyCapsule_IsValid
_PyCapsule_IsValidPtr.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_PyCapsule_IsValidPtr.restype = ctypes.c_int
_PyCapsule_GetPointerPtr = _CAPSULE_API.PyCapsule_GetPointer
_PyCapsule_GetPointerPtr.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_PyCapsule_GetPointerPtr.restype = ctypes.c_void_p

_EXPORT_STATES: dict[int, _ExportState] = {}


def _c_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
  strides = [1] * len(shape)
  running = 1
  for axis in range(len(shape) - 1, -1, -1):
    strides[axis] = running
    running *= shape[axis]
  return tuple(strides)


def _dtype_to_dlpack(dtype: object) -> tuple[int, int]:
  import monpy as mp

  table: dict[mp.DType, tuple[int, int]] = {mp.bool: (_DLC_BOOL, 8)}
  for code, dtypes in (
    (_DLC_INT, (mp.int8, mp.int16, mp.int32, mp.int64)),
    (_DLC_UINT, (mp.uint8, mp.uint16, mp.uint32, mp.uint64)),
    (_DLC_FLOAT, (mp.float16, mp.float32, mp.float64)),
    (_DLC_COMPLEX, (mp.complex64, mp.complex128)),
  ):
    table.update((d, (code, d.bits)) for d in dtypes)
  try:
    return table[typing.cast(mp.DType, dtype)]
  except KeyError as exc:
    raise BufferError(f"unsupported dtype for DLPack export: {dtype!r}") from exc


def _dtype_from_dlpack(dl_dtype: _DLDataType) -> object:
  import monpy as mp

  code, bits, lanes = int(dl_dtype.code), int(dl_dtype.bits), int(dl_dtype.lanes)
  if lanes != 1:
    raise BufferError("monpy DLPack import only supports lanes=1")
  table = {
    (_DLC_BOOL, 8): mp.bool,
    (_DLC_INT, 8): mp.int8,
    (_DLC_INT, 16): mp.int16,
    (_DLC_INT, 32): mp.int32,
    (_DLC_INT, 64): mp.int64,
    (_DLC_UINT, 8): mp.uint8,
    (_DLC_UINT, 16): mp.uint16,
    (_DLC_UINT, 32): mp.uint32,
    (_DLC_UINT, 64): mp.uint64,
    (_DLC_FLOAT, 16): mp.float16,
    (_DLC_FLOAT, 32): mp.float32,
    (_DLC_FLOAT, 64): mp.float64,
    (_DLC_COMPLEX, 64): mp.complex64,
    (_DLC_COMPLEX, 128): mp.complex128,
  }
  try:
    return table[(code, bits)]
  except KeyError as exc:
    raise BufferError(f"unsupported DLPack dtype: code={code}, bits={bits}, lanes={lanes}") from exc


class _ExportState:
  def __init__(self, arr: object, *, versioned: bool, copied: bool) -> None:
    import monpy as mp

    self.arr = typing.cast(mp.ndarray, arr)
    self.versioned = versioned
    self.copied = copied
    self.shape = self.arr.shape
    self.strides = tuple(int(s // self.arr.itemsize) for s in self.arr.strides)
    self.shape_buf = (ctypes.c_int64 * len(self.shape))(*self.shape) if self.shape else None
    self.strides_buf = (ctypes.c_int64 * len(self.strides))(*self.strides) if self.strides else None
    dtype_code, dtype_bits = _dtype_to_dlpack(self.arr.dtype)
    data_ptr = 0 if self.arr.size == 0 else int(self.arr._native.data_address())
    tensor = _DLTensor(
      ctypes.c_void_p(data_ptr),
      _DLDevice(_DLPACK_CPU, 0),
      len(self.shape),
      _DLDataType(dtype_code, dtype_bits, 1),
      self.shape_buf,
      self.strides_buf,
      0,
    )
    if versioned:
      flags = _FLAG_IS_COPIED if copied else 0
      self.managed = _DLManagedTensorVersioned(
        _DLPackVersion(_DLPACK_MAJOR, _DLPACK_MINOR),
        ctypes.c_void_p(id(self)),
        ctypes.cast(_VERSIONED_DELETER_CB, ctypes.c_void_p),
        flags,
        tensor,
      )
    else:
      self.managed = _DLManagedTensor(
        tensor,
        ctypes.c_void_p(id(self)),
        ctypes.cast(_LEGACY_DELETER_CB, ctypes.c_void_p),
      )

  @property
  def address(self) -> int:
    return ctypes.addressof(self.managed)


def _pop_export_state(address: int) -> None:
  _EXPORT_STATES.pop(address, None)


@_LegacyDeleter
def _legacy_deleter(ptr: typing.Any) -> None:
  if not ptr:
    return
  _pop_export_state(ctypes.addressof(ptr.contents))


@_VersionedDeleter
def _versioned_deleter(ptr: typing.Any) -> None:
  if not ptr:
    return
  _pop_export_state(ctypes.addressof(ptr.contents))


_LEGACY_DELETER_CB = _legacy_deleter
_VERSIONED_DELETER_CB = _versioned_deleter


def _call_legacy_deleter(address: int) -> None:
  ptr = ctypes.cast(address, ctypes.POINTER(_DLManagedTensor))
  deleter = ptr.contents.deleter
  if deleter:
    _LegacyDeleter(deleter)(ptr)


def _call_versioned_deleter(address: int) -> None:
  ptr = ctypes.cast(address, ctypes.POINTER(_DLManagedTensorVersioned))
  deleter = ptr.contents.deleter
  if deleter:
    _VersionedDeleter(deleter)(ptr)


@_CapsuleDestructor
def _capsule_destructor(capsule: int) -> None:
  try:
    if _PyCapsule_IsValidPtr(capsule, _NAME_LEGACY_USED) or _PyCapsule_IsValidPtr(capsule, _NAME_VERSIONED_USED):
      return
    if _PyCapsule_IsValidPtr(capsule, _NAME_VERSIONED):
      address = int(_PyCapsule_GetPointerPtr(capsule, _NAME_VERSIONED))
      if address:
        _call_versioned_deleter(address)
      return
    if _PyCapsule_IsValidPtr(capsule, _NAME_LEGACY):
      address = int(_PyCapsule_GetPointerPtr(capsule, _NAME_LEGACY))
      if address:
        _call_legacy_deleter(address)
  except Exception:
    return


_CAPSULE_DESTRUCTOR_CB = _capsule_destructor


def export_array(arr: object, max_version: tuple[int, int] | None = None, *, copied: bool = False) -> object:
  versioned = max_version is not None and tuple(max_version) >= (1, 0)
  state = _ExportState(arr, versioned=versioned, copied=copied)
  address = state.address
  _EXPORT_STATES[address] = state
  name = _NAME_VERSIONED if versioned else _NAME_LEGACY
  try:
    return _PyCapsule_New(ctypes.c_void_p(address), name, ctypes.cast(_CAPSULE_DESTRUCTOR_CB, ctypes.c_void_p))
  except Exception:
    _EXPORT_STATES.pop(address, None)
    raise


def _request_capsule(obj: object, copy: bool | None) -> object:
  device_fn = getattr(obj, "__dlpack_device__", None)
  if callable(device_fn):
    device = device_fn()
    if device != (_DLPACK_CPU, 0):
      raise BufferError(f"monpy only imports CPU DLPack tensors, got {device!r}")
  dlpack_attr = getattr(obj, "__dlpack__", None)
  if not callable(dlpack_attr):
    raise BufferError("object does not expose __dlpack__")
  dlpack = typing.cast(typing.Callable[..., object], dlpack_attr)
  kwargs: dict[str, object] = {
    "stream": None,
    "max_version": (_DLPACK_MAJOR, _DLPACK_MINOR),
    "dl_device": (_DLPACK_CPU, 0),
  }
  if copy is not None:
    kwargs["copy"] = copy
  try:
    return dlpack(**kwargs)
  except TypeError:
    kwargs.pop("copy", None)
    try:
      return dlpack(**kwargs)
    except TypeError:
      try:
        return dlpack(stream=None)
      except TypeError:
        return dlpack()


def _capsule_pointer(capsule: object) -> tuple[int, bool, bytes, bytes]:
  if _PyCapsule_IsValid(capsule, _NAME_VERSIONED):
    return int(_PyCapsule_GetPointer(capsule, _NAME_VERSIONED)), True, _NAME_VERSIONED, _NAME_VERSIONED_USED
  if _PyCapsule_IsValid(capsule, _NAME_LEGACY):
    return int(_PyCapsule_GetPointer(capsule, _NAME_LEGACY)), False, _NAME_LEGACY, _NAME_LEGACY_USED
  raise BufferError("invalid or already-consumed DLPack capsule")


def _managed_tensor(address: int, versioned: bool) -> tuple[_DLTensor, int]:
  if versioned:
    managed = ctypes.cast(address, ctypes.POINTER(_DLManagedTensorVersioned)).contents
    if int(managed.version.major) != _DLPACK_MAJOR:
      raise BufferError(f"unsupported DLPack major version: {int(managed.version.major)}")
    return managed.dl_tensor, int(managed.flags)
  managed = ctypes.cast(address, ctypes.POINTER(_DLManagedTensor)).contents
  return managed.dl_tensor, 0


def _shape_and_strides(tensor: _DLTensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
  ndim = int(tensor.ndim)
  if ndim < 0:
    raise BufferError("DLPack tensor rank must be non-negative")
  if ndim == 0:
    return (), ()
  if not tensor.shape:
    raise BufferError("DLPack tensor is missing shape")
  shape = tuple(int(tensor.shape[i]) for i in range(ndim))
  if any(d < 0 for d in shape):
    raise BufferError("DLPack tensor shape contains a negative dimension")
  strides = tuple(int(tensor.strides[i]) for i in range(ndim)) if tensor.strides else _c_strides(shape)
  return shape, strides


def _consume_capsule(capsule: object, used_name: bytes) -> None:
  if _PyCapsule_SetName(capsule, used_name) != 0:
    raise BufferError("failed to mark DLPack capsule as consumed")


def _release(address: int, versioned: bool) -> None:
  if versioned:
    _call_versioned_deleter(address)
  else:
    _call_legacy_deleter(address)


class _DLPackOwner:
  def __init__(self, address: int, versioned: bool, capsule: object) -> None:
    self.address = address
    self.versioned = versioned
    self.capsule = capsule
    self.released = False

  def release(self) -> None:
    if self.released:
      return
    self.released = True
    _release(self.address, self.versioned)

  def __del__(self) -> None:
    try:
      self.release()
    except Exception:
      return


def from_dlpack(obj: object, copy: bool | None) -> mp.ndarray:
  return from_dlpack_capsule(_request_capsule(obj, copy), copy)


def from_dlpack_capsule(capsule: object, copy: bool | None) -> mp.ndarray:
  import monpy as mp

  address, versioned, _name, used_name = _capsule_pointer(capsule)
  tensor, flags = _managed_tensor(address, versioned)
  if int(tensor.device.device_type) != _DLPACK_CPU:
    raise BufferError("monpy only imports CPU DLPack tensors")
  dtype = typing.cast(mp.DType, _dtype_from_dlpack(tensor.dtype))
  shape, strides = _shape_and_strides(tensor)
  size = math.prod(shape) if shape else 1
  readonly = bool(flags & _FLAG_READ_ONLY)
  data = int(tensor.data or 0) + int(tensor.byte_offset)
  byte_len = size * dtype.itemsize
  if size == 0:
    _consume_capsule(capsule, used_name)
    _release(address, versioned)
    return mp.ndarray(mp._native.empty(shape, dtype.code))
  if data == 0:
    raise BufferError("non-empty DLPack tensor has null data pointer")
  if readonly and copy is False:
    raise BufferError("readonly DLPack tensor requires copy=True")
  if copy is True or readonly:
    native = mp._native.copy_from_external(data, shape, strides, dtype.code, byte_len)
    _consume_capsule(capsule, used_name)
    _release(address, versioned)
    return mp.ndarray(native)
  native = mp._native.from_external(data, shape, strides, dtype.code, byte_len)
  owner = _DLPackOwner(address, versioned, capsule)
  _consume_capsule(capsule, used_name)
  return mp.ndarray(native, owner=owner)


__all__ = ["export_array", "from_dlpack", "from_dlpack_capsule"]
