# where the time goes (after typed kernels)

there are three main smells:
- `asarray_zero_copy` at 35x
- `strided_view` at 16x
- `array_copy` at 7x

this is mostly marshaling tax.

it's worth retracing the kernel passes to make this concrete. some recap:
- bucket B for the transposed-binary case, plus a fix to the strided walker that previously fell off into a divmod-per-element scalar walk.
- slimmed the python `ndarray` wrapper from six slots to three and added a `_wrap` classmethod that bypasses `__init__`'s kwarg parsing.
- typed-kernel pass

look at what `asarray(np_arr, copy=False)` actually does on the mojo side.
- the python wrapper hands the numpy array to `_native.asarray_from_numpy(obj, dtype_code, copy_flag)`.
- inside, mojo reads numpy's `__array_interface__`.
- the interface is a python dict.
- to populate the mojo-side `Array`, the kernel walks through this:

```
obj.__array_interface__   →  attribute access (numpy constructs the dict)
iface["typestr"]          →  dict lookup
String(py=...)            →  python string → mojo string
iface["shape"]            →  dict lookup
iface["strides"]          →  dict lookup (or None)
iface["data"]             →  dict lookup → tuple
data_obj[0]               →  tuple index → int (data pointer)
data_obj[1]               →  tuple index → bool (readonly flag)
plus a per-dim Int(py=...) for each shape entry
```

eight or nine python interactions per array crossing. each one is 30–100 nanoseconds depending on cache state. a 1d numpy array trip through this path costs 700–1000 ns of pure marshaling. numpy's own `np.asarray(np_arr)` on the same input is around 50 ns; a refcount bump in C, plus a type check, and it's done.

- every `iface["typestr"]` lowers to a `PyObject_GetAttr` followed by a `PyDict_GetItem` followed by a return-value rebind into mojo.
- cpython interop is good enough that the code reads naturally and compiles.
- every attribute read still pays the full cpython call cost on the way through.

TODO:
- one is the cpython buffer protocol. numpy arrays implement `Py_buffer`, which returns the data pointer, shape, strides, and itemsize through a single cpython call. one `PyObject_GetBuffer(obj, &view, flags)` returns everything currently fetched with eight separate attribute reads. cost on numpy's end is unchanged; on our end it becomes a single ffi call. mojo can speak the buffer protocol through the same `_get_dylib_function` machinery already in `accelerate.mojo` for blas, pointing at `libpython3.11.dylib` instead.
- the other is numpy's c api: `PyArray_DATA(obj)`, `PyArray_NDIM(obj)`, `PyArray_DIMS(obj)`, `PyArray_STRIDES(obj)`. these macros expand to direct struct-field reads on `PyArrayObject` and cost single-digit nanoseconds each. faster than the buffer protocol, but the cost is linking against numpy's runtime dylib and tracking abi (numpy promises api stability across minor versions, abi only across patch versions). the buffer protocol is the portable answer; the numpy c api is the fast answer; do whichever the use case warrants.

either path takes asarray from 35× to somewhere around 1.5–3×. the residual after that is the cost of allocating a mojo `Array` record on the new path, which is real but small.

> the same shape of problem shows up in the other three cases.
>
> `from_dlpack` goes through `__dlpack__` and `__dlpack_device__` instead of `__array_interface__`
>
> `strided_view` runs `obj[::-2]` through a python `__getitem__` slot that currently builds a slice object, normalizes start/stop/step on the python side, and crosses ffi twice.
>
> `array_copy` routes through `asarray_from_numpy` and then `copy_c_contiguous`, so it pays the marshaling cost of asarray followed by a memcpy that's essentially free.

## scope

- moving to the buffer protocol will change how all numpy interop works in monpy. someone hands monpy a numpy array; that array becomes a fast view. the whole point of zero-copy interop.
  - right now it's measurably not zero-cost.
- it doesn't fall out of more typed-kernel work
- the cost is fixed per array, instead of per element.
  - asarray of a 1M-element numpy array still pays the same 700 ns of marshaling that an 8-element array does.
  - small arrays show the gap nakedly
    - with larger array it will be buried under the kernel
