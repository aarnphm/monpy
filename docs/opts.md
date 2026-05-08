## Local Optimization Log

### 2026-05-08 rank-3 transposed add dispatch

The first post-refactor hot spot was the `strides/rank3_transpose_add_f32`
case. The input shape is `32x32x32`; both operands are c-contiguous arrays
viewed as `.transpose((2, 0, 1))`, so the logical output is contiguous but the
inputs have a stride-1 axis that is not the innermost logical axis.

`src/elementwise/__init__.mojo` now routes matching same-dtype rank-3 layouts
through `maybe_binary_rank3_axis0_tile`, the batched form of the existing rank-2
transpose tile. The kernel processes each middle-axis slice as a 4x4 register
tile:

1. Load four stride-1 SIMD vectors from each operand.
2. Apply the binary op in registers.
3. Transpose the 4x4 tile with `shuffle`.
4. Store four contiguous rows into the c-contiguous result.

Focused local result on the M3 Pro checkout:

| run                                                               | monpy us | numpy us | monpy/numpy |
| ----------------------------------------------------------------- | -------: | -------: | ----------: |
| `results/local-sweep-20260508-pass0/results.json`                 |   30.653 |    8.404 |      3.648x |
| `results/local-sweep-20260508-rank3-source-dispatch/results.json` |   19.064 |    8.210 |      2.263x |

That is a 1.61:1 reduction in monpy wall time for this row. The row still does
not beat numpy, so the next optimization pass should use sampled profiles and
hardware counters before adding another kernel. The likely split to verify is
Python wrapper time versus tile shuffle/store pressure versus library-assisted
iterator behavior in numpy.

## Profiling

`monpy-profile` profiles one benchmark case for enough wall time that OS
profilers can see native Mojo, C library, and Python frames. It is intentionally
separate from `monpy-bench`: benchmark runs stay low-overhead and comparable,
while profile runs collect heavier evidence.

```bash
monpy-profile \
  --types strides \
  --case rank3_transpose_add_f32 \
  --candidate monpy \
  --duration 8 \
  --output-dir results/profile-rank3
```

Every run writes:

- `manifest.json`: command, case, child-loop timing, profiler outputs.
- `measurement.json`: long-loop wall time, calls per second, resource usage,
  peak RSS, and native backend flags. This pass does not enable `tracemalloc`,
  because tracing allocations perturbs the CPU timing.
- `allocation-measurement.json`: a shorter allocation pass with Python
  `tracemalloc` enabled.
- `numpy-config.txt`: numpy build and BLAS/LAPACK configuration.

On macOS, the command captures a `sample(1)` stack report by default:

```text
results/profile-rank3/sample.txt
```

Use xctrace when the question needs Instruments data rather than text stacks:

```bash
monpy-profile \
  --types strides \
  --case rank3_transpose_add_f32 \
  --duration 8 \
  --xctrace time,counters,allocations \
  --output-dir results/profile-rank3-xctrace
```

The `time`, `counters`, and `allocations` aliases map to the macOS templates
`Time Profiler`, `CPU Counters`, and `Allocations`. The output `.trace`
bundles are designed for Instruments inspection.

On Linux, `monpy-profile` runs `perf stat` by default when `perf` is available:

```bash
monpy-profile \
  --types strides \
  --case rank3_transpose_add_f32 \
  --duration 8 \
  --perf-events cycles,instructions,cache-references,cache-misses
```

This gives a hardware-counter pass that can distinguish a Python wrapper
regression from a cache-miss, instruction-count, or branch-miss regression.

Run the same case with `--candidate numpy` when the question is whether monpy is
losing to NumPy wrapper overhead, iterator behavior, or a lower-level library
call:

```bash
monpy-profile \
  --types strides \
  --case rank3_transpose_add_f32 \
  --candidate numpy \
  --duration 8 \
  --output-dir results/profile-rank3-numpy
```

### 2026-05-08 scalar ascontiguousarray fast path

`array/copy/ascontiguousarray_scalar_f64` was spending two native crossings on a
scalar: build a rank-0 array via `asarray(3.5)`, then reshape it to NumPy's
required shape `(1,)`. The Python facade now handles Python bool/int/float
inputs directly as `full((1,), scalar, dtype=...)`.

Focused local result:

| run                                                         | monpy us | numpy us | monpy/numpy |
| ----------------------------------------------------------- | -------: | -------: | ----------: |
| `results/local-sweep-20260508-pass0/results.json`           |    7.808 |    2.216 |      3.524x |
| `results/local-sweep-20260508-scalar-ascontig/results.json` |    3.498 |    2.164 |      1.618x |

That is a 2.23:1 reduction in monpy wall time for the scalar row. Dense and
transpose `ascontiguousarray` rows stayed within noise, which is the important
guardrail for this wrapper-only change.

### 2026-05-08 native squeeze view path

`array/views/squeeze_axis0_f32` was the top remaining ratio in the next
`array,strides` sweep. The benchmark still includes tiny-array construction on
both sides, but the monpy half was also doing squeeze metadata in Python:
fetching shape through the extension boundary, normalizing axes, building a
drop set, then crossing back into native `reshape`.

`src/create/__init__.mojo` now owns `squeeze_all` and `squeeze_axes`, so
singleton-axis validation and view shape/stride construction happen in one
native call. The Python facade is reduced to `asarray` plus one native view
constructor.

Focused local result:

| run                                                        | monpy us | numpy us | monpy/numpy |
| ---------------------------------------------------------- | -------: | -------: | ----------: |
| `results/local-sweep-20260508-heartbeat1/results.json`     |   12.494 |    2.436 |      5.205x |
| `results/local-sweep-20260508-native-squeeze/results.json` |    8.182 |    2.429 |      3.390x |

That is a 1.53:1 reduction in monpy wall time for this row. The residual gap is
mostly outside squeeze itself: a direct microbench of an existing monpy array
showed `mnp.asarray(np.zeros(...))` around 4.2 us and native-backed squeeze
around the low single-digit microsecond range. The next wrapper-pass target is
therefore the NumPy-input marshaling family (`from_dlpack`, `asarray_zero_copy`,
and small `array_copy`), not another squeeze-specialized kernel.

### 2026-05-08 NumPy DLPack fast path

`array/interop/from_dlpack_f32` was routing NumPy inputs through the generic
Python DLPack capsule parser. That path remains necessary for arbitrary DLPack
producers, but NumPy ndarrays already expose the buffer metadata monpy needs
through the existing array-interface ingest. The top-level `from_dlpack` facade
now recognizes NumPy ndarray inputs and delegates to the same zero-copy
`from_numpy` path used by `asarray`.

The fast path preserves the important DLPack copy-policy behavior: `copy=True`
detaches, writable `copy=False` shares storage, and readonly `copy=False` still
raises `BufferError`.

Focused local result:

| run                                                               | monpy us | numpy us | monpy/numpy |
| ----------------------------------------------------------------- | -------: | -------: | ----------: |
| `results/local-sweep-20260508-native-squeeze/results.json`        |    8.824 |    2.182 |      4.052x |
| `results/local-sweep-20260508-dlpack-numpy-fastpath/results.json` |    5.669 |    2.172 |      2.587x |

That is a 1.56:1 reduction in monpy wall time for NumPy-backed DLPack imports.
The row now sits with the rest of the small NumPy-input marshaling family rather
than standing out as a separate capsule-parser tax.

### 2026-05-08 NumPy input detector fast path

The remaining `array/interop/asarray_zero_copy_*` rows all enter through the
NumPy-input detector. The old detector scanned the Python MRO looking for a base
named `numpy.ndarray`; useful before NumPy is imported, but wasteful in the
benchmark path where NumPy is already live. The detector now checks
`sys.modules["numpy"].ndarray` with `isinstance` when available, then falls back
to the import-free MRO scan. The hot `asarray` and NumPy-backed `from_dlpack`
facades also use an unchecked internal converter after the detector succeeds,
so the same ndarray is not classified twice.

Focused local result:

| row                                    | previous monpy us | new monpy us | previous ratio | new ratio |
| -------------------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/interop/asarray_zero_copy_f32`  |             6.097 |        5.207 |         3.104x |    2.699x |
| `array/interop/asarray_zero_copy_f64`  |             5.955 |        5.213 |         3.024x |    2.644x |
| `array/interop/asarray_zero_copy_bool` |             6.082 |        5.229 |         3.062x |    2.697x |
| `array/interop/asarray_zero_copy_i64`  |             5.960 |        5.199 |         3.062x |    2.654x |
| `array/interop/from_dlpack_f32`        |             5.669 |        4.963 |         2.587x |    2.290x |

The direct detector microbench moved from about 0.34 us to about 0.09 us while
`tests/python/test_no_numpy_core.py` still verifies that importing the core
package does not import NumPy.

### 2026-05-08 native stack-axis-0 path

The small join rows had a shape where native `concatenate` was already cheap,
but `stack` and 1D `vstack` still paid Python shape validation, one native
concatenate call, and then a second native reshape call. `src/create/__init__.mojo`
now exposes `stack_axis0`, a single native entrypoint that validates identical
input shape/dtype, builds the `[n_arrays] + input_shape` result, and copies each
contiguous input slab directly. The Python `stack(axis=0)` and rank-1 `vstack`
facades optimistically use it when no dtype override is requested, then fall
back to the promotion/general-axis path on mismatch.

Focused local result:

| row                           | previous monpy us | new monpy us | previous ratio | new ratio |
| ----------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/views/stack_axis0_f32` |             9.191 |        3.731 |         2.610x |    1.051x |
| `array/views/vstack_f32`      |             9.678 |        3.700 |         2.926x |    1.127x |

That is a 2.46:1 reduction for `stack_axis0_f32` and a 2.62:1 reduction for
`vstack_f32`. `hstack` and plain `concatenate` did not move, as expected; those
already route through the native concatenate leaf and need a different target if
they become important.

### 2026-05-08 rank-1 atleast_2d expand view

`array/creation/atleast_2d_f32` was still paying for Python shape reads plus a
native reshape when the input was already a monpy rank-1 array. NumPy models
this case as a view (`arr[None, :]`), so monpy now uses the same shape operation:
rank-1 `atleast_2d` calls native `expand_dims(axis=0)` directly and keeps the
original array as the base owner.

Focused local result:

| row                             | previous monpy us | new monpy us | previous ratio | new ratio |
| ------------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/creation/atleast_2d_f32` |             6.087 |        3.526 |         2.664x |    1.519x |

The direct microbench moved `mnp.atleast_2d(existing_vector)` from about 4.1 us
to about 1.4 us. The remaining gap is the generic Python facade and one native
call, not data movement.

### 2026-05-08 exact middle-newaxis view

`array/views/newaxis_middle_f32` uses the concrete key `arr[:, None, :]`. The
generic indexing path expands the key, fetches full shape metadata, validates
the full slices, then calls native `expand_dims`. For this exact rank-2 full
slice pattern, none of the shape metadata is needed: the result is just
`expand_dims(axis=1)`.

`ndarray.__getitem__` now recognizes the exact `(:, None, :)` rank-2 pattern
before entering the generic slice path. Mixed keys such as `arr[:, None, ::-1]`
still fall through to the existing generic slice machinery.

Focused local result:

| row                              | previous monpy us | new monpy us | previous ratio | new ratio |
| -------------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/views/newaxis_middle_f32` |             5.743 |        3.138 |         2.734x |    1.535x |

The direct microbench moved the exact `helper[:, None, :]` view from about 3.6
us to about 1.1 us. The row is still not NumPy-fast, but it no longer pays the
full generic slice tax for a fixed full-slice newaxis.

### 2026-05-08 native swapaxes view

`array/views/swapaxes_f32` was still implemented as Python permutation
construction followed by generic `transpose`. For a pure axis swap, the native
side can avoid layout selection and Python tuple normalization entirely: clone
the shape and stride lists, swap two slots, then return a view with the original
storage.

Focused local result:

| row                        | previous monpy us | new monpy us | previous ratio | new ratio |
| -------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/views/swapaxes_f32` |             5.534 |        3.072 |         2.550x |    1.379x |

The direct microbench moved `mnp.swapaxes(existing_rank3, 0, 2)` from about 3.3
us to about 0.87 us. The residual benchmark time is mostly the Python facade and
benchmark harness overhead, not data movement.

### 2026-05-08 direct NumPy ndarray ingest

The NumPy-input rows were still entering through the generic
`__array_interface__` parser even after the detector fast path. That generic
path is required for non-NumPy producers, but real NumPy arrays already expose
the same fields directly as attributes. `runtime.ops_numpy._from_numpy_unchecked`
now reads `dtype.str`, `shape`, `strides`, `ctypes.data`, and `flags.writeable`
directly, then calls native `from_external` / `copy_from_external` with the same
copy and readonly policy as before.

Focused local result:

| row                                    | previous monpy us | new monpy us | previous ratio | new ratio |
| -------------------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/interop/asarray_zero_copy_f32`  |             5.095 |        4.923 |         2.628x |    2.532x |
| `array/interop/asarray_zero_copy_f64`  |             5.155 |        4.826 |         2.654x |    2.486x |
| `array/interop/asarray_zero_copy_bool` |             5.134 |        4.834 |         2.630x |    2.528x |
| `array/interop/asarray_zero_copy_i64`  |             5.138 |        4.819 |         2.654x |    2.494x |
| `array/interop/from_dlpack_f32`        |             4.900 |        4.596 |         2.262x |    2.173x |

This is a small per-call win, but it lands on every NumPy ndarray import path.
The remaining cost is mostly Python attribute access plus the native array-view
constructor.

### 2026-05-08 single-axis squeeze view

`array/views/squeeze_axis0_f32` calls `mnp.squeeze(..., axis=0)`, but the native
implementation previously treated scalar axes like the general multi-axis case:
Python built a one-element tuple, Mojo parsed the sequence, allocated a boolean
drop list, then built the view. Scalar-axis squeeze now has a direct
`squeeze_axis` native entrypoint. It normalizes and validates one axis, then
copies every non-dropped shape/stride slot into the result view.

Focused local result:

| row                             | previous monpy us | new monpy us | previous ratio | new ratio |
| ------------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/views/squeeze_axis0_f32` |             7.106 |        6.912 |         2.969x |    2.821x |

The row still includes `np.zeros(...)` plus NumPy-to-monpy wrapping, so the
benchmark movement is smaller than the isolated operation movement. Direct
microbenchmarks moved `mnp.squeeze(existing_array, axis=0)` from about 1.10 us
to about 0.83 us, and moved the native view operation from about 0.79 us to
about 0.54 us.

### 2026-05-08 native logspace fill

`array/creation/logspace_50` was mostly Python object churn: the facade built a
Python list of 50 exponents, computed 50 Python scalar powers, then copied that
list back into native storage through `asarray`. NumPy documents scalar-base
`logspace` as `linspace(start, stop)` followed by `power(base, y)`, so monpy now
does the same shape of work inside one native creator: allocate the output, walk
the linear exponent range, and store `pow(base, exponent)` directly.

The first native version used Mojo's SIMD scalar `pow`, which was fast but
missed the existing `1e-12` NumPy parity test by about `7.6e-10` relative on
the 50-point `0..3` span. The committed path calls platform `libm` `pow`
instead, preserving the strict parity test while still avoiding Python-list
materialization.

Focused local result:

| row                          | previous monpy us | new monpy us | previous ratio | new ratio |
| ---------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/creation/logspace_50` |            22.809 |        4.494 |         4.156x |    0.836x |

Direct microbenchmarks moved `mnp.logspace(0.0, 1.0, num=50)` from about 20.3
us to about 2.3 us. The full benchmark row now beats NumPy for this case.

### 2026-05-08 native pinv via least squares

`array/decomp/pinv_8_f64` was spending most of its time outside the final
numeric result. A `sample(1)` profile of `pinv_8_f64` measured about 56 us per
call with a 64 MB physical footprint; the hot stack included the LAPACK SVD
(`DGESDD` / `DGEBRD` / BLAS `DGEMV`) and a separate Python composition layer
that built `s_inv`, transposed `U`/`VT`, broadcast-multiplied, and matmul'd the
pieces back together. The `xctrace` time and CPU-counter captures completed,
but the allocations template timed out after 63 seconds in Instruments startup.

For the matrix `A`, `pinv(A)` is the minimum-norm solution operator for
`A X = I`. Monpy now calls the existing LAPACK `gelsd` least-squares path with
an identity right-hand side inside one native `linalg_pinv` entrypoint. This
keeps the existing monpy cutoff convention (`eps(dtype) * max(m, n)`) while
removing the Python SVD post-processing graph.

Focused local result:

| row                        | previous monpy us | new monpy us | previous ratio | new ratio |
| -------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/decomp/pinv_2_f64`  |            33.683 |        6.824 |         2.145x |    0.437x |
| `array/decomp/pinv_4_f64`  |            39.188 |        8.349 |         2.252x |    0.481x |
| `array/decomp/pinv_8_f64`  |            54.852 |       12.864 |         2.811x |    0.655x |
| `array/decomp/pinv_32_f64` |           136.756 |       54.235 |         2.424x |    0.963x |
| `array/decomp/pinv_8_f32`  |            48.527 |       10.937 |         2.033x |    0.451x |

All benchmarked pseudo-inverse rows now beat NumPy on this machine. The largest
remaining deficit at this checkpoint was no longer `pinv`; it was the
view/import-heavy `squeeze_axis0_f32` row.

### 2026-05-08 NumPy buffer-protocol ingest

The direct NumPy ndarray ingest path still parsed NumPy metadata in Python:
`dtype.str`, `shape`, `strides`, `ctypes.data`, `size`, and `flags.writeable`.
The Python C API buffer protocol is a better contract for this job: one
`PyObject_GetBuffer(..., PyBUF_RECORDS_RO)` request exposes the raw pointer,
item size, shape, strides, readonly bit, and PEP-3118 format string. NumPy's
own ndarray model is the same strided-memory contract, with `shape`, byte
`strides`, `dtype`, and the data buffer as intrinsic array attributes.

`runtime.ops_numpy._from_numpy_unchecked` now keeps only the NumPy-specific
classification and dtype-request policy in Python, then delegates the actual
borrow/copy/cast decision to the existing native `asarray_from_buffer` bridge.
That removes the `ctypes.data` property and per-field Python tuple/int
normalization from every NumPy ndarray import.

Focused local result:

| row                                    | previous monpy us | new monpy us | previous ratio | new ratio |
| -------------------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/interop/asarray_zero_copy_f32`  |             5.042 |        3.792 |         2.593x |    1.941x |
| `array/interop/asarray_zero_copy_f64`  |             5.026 |        3.746 |         2.560x |    1.928x |
| `array/interop/asarray_zero_copy_bool` |             4.978 |        3.741 |         2.524x |    1.949x |
| `array/interop/asarray_zero_copy_i64`  |             5.011 |        3.745 |         2.539x |    1.925x |
| `array/interop/array_copy_f32`         |             5.496 |        4.320 |         2.326x |    1.875x |
| `array/interop/array_copy_f64`         |             5.626 |        4.448 |         2.324x |    1.896x |
| `array/interop/array_copy_bool`        |             5.364 |        4.238 |         2.342x |    1.911x |
| `array/interop/array_copy_i64`         |             5.602 |        4.434 |         2.339x |    1.882x |
| `array/interop/from_dlpack_f32`        |             4.815 |        3.660 |         2.212x |    1.711x |

The direct wrapper microbench for `mnp.asarray(np.arange(1024, dtype=float32),
dtype=mnp.float32, copy=False)` moved from about 2.58 us to about 1.62 us, a
1.59:1 reduction. The unchecked converter itself moved from about 2.37 us to
about 1.37 us.

Profile artifacts:

- `results/profile-20260508-asarray-zero-copy-f32-buffer-bridge/manifest.json`
  measured the patched monpy row at 4.030 us/call in the long child loop, with
  84.8 MB max RSS, 63.2 MB physical footprint in `sample.txt`, and a 1,718 byte
  traced Python allocation peak in the allocation pass.
- `results/profile-20260508-asarray-zero-copy-f32-numpy/manifest.json` measured
  the NumPy row at 1.993 us/call, with 87.4 MB max RSS, 63.2 MB physical
  footprint in `sample.txt`, and a 1,550 byte traced Python allocation peak.
- The macOS `sample(1)` stack for the patched monpy row still shows repeated
  `std::ffi::_DLHandle::get_symbol` / dyld `dlsym` work under
  `asarray_from_buffer_ops` (428 of 2,269 samples in this short capture). The
  next native interop pass should remove that dynamic symbol lookup from the
  per-call path before chasing smaller Python wrapper costs.

References used for the direction:

- [Python C API buffer protocol](https://docs.python.org/3/c-api/buffer.html):
  `Py_buffer` carries `buf`, `itemsize`, `format`, `ndim`, `shape`, `strides`,
  and `readonly`, and `PyBUF_RECORDS_RO` requests shape, strides, format, and
  read-only-compatible export.
- [NumPy ndarray reference](https://numpy.org/doc/stable/reference/arrays.ndarray.html):
  ndarray memory layout is defined by a data buffer, shape, dtype/itemsize, and
  byte strides; NumPy also notes that strided views are first-class and
  algorithms may handle arbitrary strides.

### 2026-05-08 cached CPython buffer functions

The previous buffer-protocol pass moved NumPy import metadata extraction into
native code, but the macOS `sample(1)` report showed that the native bridge was
still resolving `PyObject_GetBuffer` and `PyBuffer_Release` through
`ExternalFunction.load(...)` on every call. In the short capture, dyld symbol
lookup accounted for 428 of 2,269 samples under `asarray_from_buffer_ops`.

`src/buffer.mojo` now stores both CPython buffer function pointers in one
`_Global` value, `MONPY_BUFFER_FUNCTIONS`, initialized once from Mojo's existing
CPython handle. The hot path still calls the official buffer protocol, but it
does not ask dyld for the same two symbols every time an ndarray crosses the
boundary.

Focused local result:

| row                                    | previous monpy us | new monpy us | previous ratio | new ratio |
| -------------------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/interop/asarray_zero_copy_f32`  |             3.792 |        3.026 |         1.941x |    1.529x |
| `array/interop/asarray_zero_copy_f64`  |             3.746 |        3.068 |         1.928x |    1.541x |
| `array/interop/asarray_zero_copy_bool` |             3.741 |        3.080 |         1.949x |    1.561x |
| `array/interop/asarray_zero_copy_i64`  |             3.745 |        3.209 |         1.925x |    1.558x |
| `array/interop/array_copy_f32`         |             4.320 |        3.663 |         1.875x |    1.553x |
| `array/interop/array_copy_f64`         |             4.448 |        3.748 |         1.896x |    1.526x |
| `array/interop/array_copy_bool`        |             4.238 |        3.579 |         1.911x |    1.551x |
| `array/interop/array_copy_i64`         |             4.434 |        3.657 |         1.882x |    1.517x |
| `array/interop/from_dlpack_f32`        |             3.660 |        2.927 |         1.711x |    1.365x |

The direct native-buffer microbench for
`_native.asarray_from_buffer(np.arange(1024, dtype=float32), float32.code, 0)`
moved from about 1.35 us to about 0.56 us. The Python unchecked converter moved
from about 1.37 us to about 0.63 us, and
`mnp.asarray(..., dtype=mnp.float32, copy=False)` moved from about 1.62 us to
about 0.84 us.

Profile artifacts:

- `results/profile-20260508-asarray-zero-copy-f32-buffer-fn-cache/manifest.json`
  measured the patched monpy row at 3.276 us/call in the long child loop, down
  from 4.030 us/call in
  `results/profile-20260508-asarray-zero-copy-f32-buffer-bridge/manifest.json`.
- The same profile reports 84.7 MB max RSS, 63.2 MB physical footprint in
  `sample.txt`, and the same 1,718 byte traced Python allocation peak as the
  previous buffer bridge pass.
- Grepping the new `sample.txt` for `get_symbol` and `dlsym` returns no matches,
  so the sampled dynamic-loader hotspot is gone. The remaining time is now in
  the Python extension wrapper, native `Array` construction, and Python-side
  `ndarray` wrapper allocation.

### 2026-05-08 native concatenate dtype inference

`array/native_kernels/concatenate_axis0_8x128_f64` copies only 8 KiB of data
total: eight contiguous 128-element `float64` inputs. The native side was
already doing one 1 KiB `memcpy` per input, and Mojo's `std.memory.memcpy`
already lowers byte copies through width-32 vectorized chunks for this size.
The slow part was Python preflight: before calling native concat, the facade
crossed the extension boundary once per input for `dtype_code()` and once per
input for `is_c_contiguous()`. A direct microbench measured that metadata
preflight at about 1.94 us for the eight-input benchmark, while the raw native
concat was about 1.68 us.

`concatenate(..., dtype=None)` now optimistically calls native concat with
`dtype_code=-1`. The native entrypoint infers the first dtype, validates equal
dtypes and c-contiguous inputs, and raises a narrow fallback error when the
Python facade needs the old promotion or materialization path. Shape and axis
errors still propagate directly.

Focused local result:

| row                                                | previous monpy us | new monpy us | previous ratio | new ratio |
| -------------------------------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/native_kernels/concatenate_axis0_8x128_f64` |             6.956 |        5.093 |         2.252x |    1.657x |
| `array/views/concatenate_axis0_f32`                |             4.231 |        3.528 |         1.684x |    1.448x |
| `array/views/hstack_f32`                           |             4.827 |        4.120 |         1.596x |    1.402x |

Direct microbenchmarks for the benchmark inputs:

| operation                 | previous us | new us |
| ------------------------- | ----------: | -----: |
| `mnp.concatenate(inp_mp)` |       4.791 |  2.842 |
| raw native concat         |       1.684 |  1.676 |

Profile artifacts:

- `results/profile-20260508-concat-axis0-8x128-native-infer/manifest.json`
  measured the patched row at 5.243 us/call in the long child loop, down from
  7.268 us/call in
  `results/profile-20260508-concat-axis0-8x128-baseline/manifest.json`.
- Max RSS dropped from 87.4 MB to 85.0 MB in the profile child. The traced
  Python allocation peak stayed at 1,718 bytes, so this pass removed call
  overhead rather than changing allocation shape.

The next concatenate-specific lever is a narrower axis-0 vector/list ABI that
avoids Python list construction and repeated `downcast_value_ptr` inside native.
That is probably smaller than the remaining linalg/view rows, so the next broad
target should move to `cholesky_32_f64` or the view wrapper cluster.

### 2026-05-08 Cholesky typed writeback

`array/decomp/cholesky_32_f64` was not losing because it missed LAPACK.
`sample(1)` on
`results/profile-20260508-cholesky-32-f64-baseline-monpy/manifest.json`
showed the native path already inside Accelerate `DPOTRF`; the largest local
symbol was `array::__init__::physical_offset`, with 686 samples in the report.
Netlib documents `DPOTRF` as a blocked Cholesky routine that calls Level-3 BLAS,
so reimplementing the factorization in Mojo was the wrong lever for 32x32. The
right lever was the row-major result copy after the vendor call.

`lapack_cholesky_{f32,f64}_into` now writes the lower-triangular result through
typed contiguous pointers instead of `set_logical_from_f64` for every output
element. This removes the per-element shape/stride divmod from a result that is
freshly allocated, c-contiguous, and already has the exact dtype.

Focused local result:

| row                            | previous monpy us | new monpy us | previous ratio | new ratio |
| ------------------------------ | ----------------: | -----------: | -------------: | --------: |
| `array/decomp/cholesky_32_f64` |            16.887 |        9.776 |         2.201x |    1.281x |
| `array/decomp/cholesky_32_f32` |            16.500 |        9.211 |         1.886x |    1.042x |

Profile artifacts:

- `results/profile-20260508-cholesky-32-f64-direct-writeback/manifest.json`
  measured the patched monpy row at 9.744 us/call, down from 16.546 us/call in
  `results/profile-20260508-cholesky-32-f64-baseline-monpy/manifest.json`.
- The patched sample moved the stack back to Accelerate `DPOTRF`; `physical_offset`
  no longer appears in the Cholesky hot-path grep. The remaining local staging
  is `transpose_to_col_major_f64`, so any further Cholesky win needs a layout or
  `UPLO` policy change rather than another result writeback rewrite.
- The CPU Counters xctrace artifact was captured under
  `results/profile-20260508-cholesky-32-f64-direct-writeback/xctrace-cpu-counters.trace`.
  The child loop in that trace measured 9.838 us/call with 3.0 s of user CPU and
  no major faults; the profile manifest stays the durable scalar record.
- Traced Python allocation peak stayed at 110,752 bytes. This pass removed
  native arithmetic and branch overhead, not Python allocation churn.

The new focused top deficits are view-wrapper and stride rows:
`reversed_add_f32`, `flatten_f32`, `ravel_f32`, `moveaxis_f32`, and
`empty_like_shape_override_f32`. Cholesky is no longer the next broad target.

### 2026-05-08 native ravel and flatten views

`array/views/ravel_f32` and `array/views/flatten_f32` are tiny shape helpers:
the benchmark input is a `2x3x4` float32 array, so the operation itself touches
only 24 elements. The old monpy path crossed into native for contiguity checks,
then crossed again through the generic reshape path. It also accidentally made
`flatten` a view for already-contiguous arrays, while NumPy documents
[`flatten`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html)
as copy-returning and
[`ravel`](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html)
as copy-only-when-needed.

`src/create/__init__.mojo` now exposes native `ravel` and `flatten` entrypoints.
`ravel` returns a one-dimensional view for c-contiguous inputs and materializes
a contiguous copy otherwise. `flatten` always materializes a contiguous copy,
then exposes that copy through a one-dimensional view. This preserves the
existing Python order surface while moving shape/stride construction and copy
semantics into one native call.

Focused local result:

| row                       | previous monpy us | new monpy us | previous ratio | new ratio |
| ------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/views/flatten_f32` |             4.940 |        3.357 |         2.135x |    1.493x |
| `array/views/ravel_f32`   |             4.424 |        2.731 |         1.995x |    1.242x |

Direct microbenchmarks on an existing `2x3x4` array measured
`mnp.ravel(s_mp)` at about 0.615 us and `mnp.flatten(s_mp)` at about 1.115 us.
The full benchmark rows are higher because the benchmark includes the facade
and harness path, which is now the remaining dominant tax.

Profile artifacts:

- `results/profile-20260508-ravel-f32-native/manifest.json` measured the patched
  monpy row at 2.886 us/call, with 85.6 MB max RSS, no major faults, and a
  traced Python allocation peak of 1,718 bytes.
- `results/profile-20260508-flatten-f32-native/manifest.json` measured the
  patched monpy row at 3.545 us/call, with 85.6 MB max RSS, no major faults,
  and the same 1,718 byte traced Python allocation peak.
- The `sample(1)` reports show the new `create::__init__::ravel_ops` and
  `create::__init__::flatten_ops` frames. The old generic `reshape_ops` path no
  longer appears in the hot wrapper stack for these cases.

The top rows after this pass are `reversed_add_f32`, `moveaxis_f32`,
`empty_like_shape_override_f32`, and `transpose_add_f32`. The next high-upside
view target is therefore negative-stride elementwise dispatch or moveaxis
permutation metadata, not further `ravel`/`flatten` work.

### 2026-05-08 F-order transposed binary output

The live `array,strides` frontier put `strides/elementwise/transpose_add_f32`
back at the top. The inputs are both positive-stride transposed views:
`a_mp.T + b_mp.T`, shape `256x256`, element strides `[1, 256]`. The old fused
path treated the output as c-contiguous, loaded each 4x4 tile contiguously from
the inputs, transposed the tile in registers, then stored c-contiguous rows.

NumPy's ufunc contract defaults `order` to `K`, so the output should match the
input element order as closely as possible. For two dense F-order inputs, NumPy
returns an F-contiguous result. Monpy now does the same for this dense
positive-stride rank-2 case: it adds the physically contiguous input buffers in
linear SIMD order, writes the managed result buffer linearly, then marks the
result strides as `[1, rows]`. That removes the in-register transpose and the
row-store reshaping work.

Focused local result:

| row                                     | previous monpy us | new monpy us | previous ratio | new ratio |
| --------------------------------------- | ----------------: | -----------: | -------------: | --------: |
| `strides/elementwise/transpose_add_f32` |            30.212 |       12.184 |         2.776x |    1.147x |

Direct microbenchmarks for the same `256x256` benchmark input moved
`a_mp.T + b_mp.T` from about 27.6 us to about 9.8 us. NumPy measured about
9.6 us in the same local loop; the saved benchmark row is higher because it
runs through the full sweep harness and round aggregation.

Profile artifacts:

- `results/profile-20260508-strides-transpose-add-f32-pre/manifest.json`
  measured the pre-change row at 30.318 us/call. `sample(1)` put 2,019 samples
  in `maybe_binary_rank2_transposed_tile`.
- `results/profile-20260508-strides-transpose-add-f32-forder/manifest.json`
  measured the patched row at 13.265 us/call, with 51.9 MB max RSS, no major
  faults, and a traced Python allocation peak of 1,856 bytes.
- The patched sample still spends most native time in
  `maybe_binary_rank2_transposed_tile`, but it is now the linear F-order branch
  rather than the 4x4 shuffle/store branch.

The new combined `array,strides` frontier is led by
`strides/elementwise/rank3_transpose_add_f32`, `array/views/reversed_add_f32`,
and `array/views/moveaxis_f32`. The next broad target should be the rank-3
transpose path's output order: it likely has the same "doing extra layout work
to force c-contiguous output" smell, only with one more axis in the metadata.

### 2026-05-08 F-order rank-3 transposed binary output

`strides/elementwise/rank3_transpose_add_f32` uses dense C-order `32x32x32`
buffers viewed as `.transpose((2, 0, 1))`. The resulting logical shape is
`[rows, batches, cols]`, but the positive dense element strides are
`[1, rows * cols, rows]`. NumPy ufuncs default `order` to `K`, so NumPy keeps
the output in that input element order instead of forcing a c-contiguous result.

The previous monpy rank-3 fused path still used the 4x4 tile shuffle path that
stores a c-contiguous result. `maybe_binary_rank3_axis0_tile` now detects this
specific dense positive-stride layout, adds the two physical buffers with a
linear SIMD walk, and marks the result strides as `[1, rows * cols, rows]`.
That removes the unnecessary layout conversion while preserving NumPy's result
stride contract.

Focused local result:

| row                                           | previous monpy us | new monpy us | previous ratio | new ratio |
| --------------------------------------------- | ----------------: | -----------: | -------------: | --------: |
| `strides/elementwise/rank3_transpose_add_f32` |            18.486 |       14.113 |         2.385x |    1.672x |

The combined `array,strides` frontier moved the same row to 13.344 us against
NumPy's 7.961 us, a 1.662x ratio. A direct microbench for the benchmark shape
measured monpy at about 11.0 us and NumPy at about 5.6 us; the full sweep is
higher because it includes harness and round aggregation overhead.

Profile artifacts:

- `results/profile-20260508-rank3-transpose-add-f32-forder/manifest.json`
  measured the patched monpy row at 14.080 us/call, with 53.2 MB max RSS, no
  major faults, 8 minor faults, 2.925 s user CPU, and 0.029 s system CPU in the
  3 s child loop.
- The allocation pass traced a 131,312 byte Python peak and reported 52.9 MB
  max RSS. This pass changed native layout work, not Python allocation shape.
- The `sample(1)` report still puts the native time in
  `maybe_binary_rank3_axis0_tile`, now the linear F-order branch. It also shows
  allocator frames under `tc_memalign`, so the next stride-kernel pass should
  separate result allocation cost from pure arithmetic before adding another
  shuffle kernel.

The new combined frontier is led by `array/views/reversed_add_f32` at 2.157x,
`array/views/moveaxis_f32` at 1.898x, and
`array/creation/empty_like_shape_override_f32` at 1.875x. The next broad target
should be negative-stride elementwise dispatch for `reversed_add_f32`, with
`moveaxis` metadata construction as the next view-wrapper fallback if the
negative-stride profile points mostly at allocation.

### 2026-05-08 exact rank-1 reverse slice view

The fresh frontier still put `array/views/reversed_add_f32` first, but direct
decomposition showed the fused reversed-add arithmetic was not the main cost:

| operation                    | monpy us | numpy us |
| ---------------------------- | -------: | -------: |
| `x[::-1]` view construction  |    1.569 |    0.092 |
| prebuilt reversed view add   |    0.979 |    0.741 |
| full `x[::-1] + y[::-1]` row |    4.294 |    0.876 |

NumPy's ndarray model treats slicing as a view over the same memory with a
different strided indexing scheme, so the exact `[::-1]` case is just
`shape=[n]`, `stride=-old_stride`, and `offset=old_offset + (n - 1) * old_stride`.
Monpy already had a native 1-D slice path, but the exact reverse case still paid
Python shape access, Python `int` argument materialization, and native conversion
of `(start, stop, step)`.

`src/create/shape_ops.mojo` now exposes `reverse_1d_ops`, and
`ndarray.__getitem__` dispatches exact rank-1 `[::-1]` to the new no-argument
native method before the generic 1-D slice path. The generic path also reads the
rank-1 length through `shape_at(0)` instead of materializing a one-element Python
shape tuple.

Focused local result:

| row                                      | previous monpy us | new monpy us | previous ratio | new ratio |
| ---------------------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/views/reversed_add_f32`           |             6.655 |        4.616 |         2.188x |    1.512x |
| `strides/elementwise/reverse_1d_add_f32` |             7.219 |        5.192 |         1.633x |    1.218x |
| `array/views/strided_view_f32`           |             3.717 |        3.431 |         1.799x |    1.609x |

Direct post-change microbenchmarks measured `x[::-1]` at about 0.696 us and
the full `x[::-1] + y[::-1]` row at about 2.385 us. That is a 2.25:1 reduction
for exact reverse view construction and a 1.80:1 reduction for the full direct
row.

Profile artifacts:

- `results/profile-20260508-reversed-add-f32-monpy-pre2/manifest.json`
  measured the old monpy row at 6.685 us/call, 86.3 MB max RSS, no major
  faults, and a 91,700 byte traced Python allocation peak.
- `results/profile-20260508-reversed-add-f32-reverse1d-fastpath/manifest.json`
  measured the patched row at 4.789 us/call, 87.2 MB max RSS, no major faults,
  and a 1,856 byte traced Python allocation peak.
- The post-change `sample(1)` report shows `reverse_1d_ops` instead of the old
  `slice_1d_ops` wrapper stack for the exact `[::-1]` construction. The
  remaining cost is now mostly Python wrapper/object allocation plus the
  existing reversed-add native loop.

The new broad frontier is `empty_like_shape_override_f32` at 1.905x,
`moveaxis_f32` at 1.891x, and `squeeze_axis0_f32` at 1.849x. The next target
should be the view-wrapper cluster only after a quick profile decides whether
`moveaxis` or `empty_like(shape=...)` has the larger removable Python metadata
tax.

### 2026-05-08 attention scalar and row-kernel recovery

The linked reference commit, `e2ed21d`, only added `pyyaml`; it was not the
attention regression point. A detached baseline at that commit measured the
attention rows as:

| row | `e2ed21d` monpy us | current-start monpy us | post-fix monpy us | post-fix ratio |
| --- | -----------------: | ---------------------: | ----------------: | -------------: |
| `attention/softmax/causal_scores_t32_f32` | 57.121 | 47.262 | 21.292 | 1.988x |
| `attention/attention/causal_attention_t32_d32_f32` | 133.539 | 121.953 | 43.121 | 1.907x |
| `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32` | 1606.192 | 1534.498 | 165.275 | 1.569x |

The bad current-start profile was still real. It just came from the attention
stack's generic paths, not the PyYAML commit:

- `reduce_axis_ops` used the coordinate-list f64 walker for every
  `axis=-1, keepdims=True` row reduction. Direct softmax decomposition showed
  row `max` at 11.492 us and row `sum` at 11.272 us on a 32x32 float32 score
  matrix.
- `where(cond, fill, scores)` paid the full broadcast `MultiLayoutIter`, taking
  about 55.867 us for the causal 32x32 mask.
- `mnp.power(f32, 3.0)` used the ufunc array/0-d-array path, upcast to
  float64, and turned the GPT MLP into mixed `float64 @ float32` fallback
  matmuls. The two rectangular matmuls were about 565 us each before fixing
  weak scalar handling.

This pass added three general fast paths rather than benchmark-only branches:

- C-contiguous last-axis reductions for float32/float64 `sum`, `mean`, `prod`,
  `min`, and `max`. The sum/mean path reuses the existing 4-accumulator SIMD
  reducer per row.
- C-contiguous `(rows, cols)` by `(rows, 1)` binary broadcasting for float32 and
  float64. This covers softmax row shifts/divides and layer-norm centering.
- Same-shape bool-mask `where` for contiguous float32/float64 arrays, plus weak
  Python-scalar ufunc dispatch so `mnp.power(float32_array, 3.0)` stays float32.
  Scalar power `x**2` and `x**3` now lower to multiplication instead of libm
  `pow`.

Direct microbenchmarks for the attention shapes moved as follows:

| operation | before us | after us |
| --- | --------: | -------: |
| `mnp.max(scores, axis=-1, keepdims=True)` | 11.492 | 2.658 |
| `mnp.sum(scores, axis=-1, keepdims=True)` | 11.272 | 2.326 |
| `scores - row_max` with a prebuilt `(32, 1)` row max | 8.7-ish | 4.858 |
| `mnp.where(causal_mask, fill, scores)` | 55.867 | 3.443 |
| `mnp.power(x_f32, 3.0)` for a 32x128 MLP activation | 64.073 | 2.379 |
| `_gelu_monpy(32x128)` | about 61.4 after mask work | 37.245 |

Post-fix decomposition now has the GPT row dominated by compositional layer
norm and attention overhead, not BLAS misses:

| operation | monpy us | numpy us |
| --- | -------: | -------: |
| layer norm, one 32x32 block | 23.699 | 10.844 |
| causal attention on normalized input | 39.176 | 18.755 |
| GELU on 32x128 MLP activation | 31.778 | 39.254 |
| MLP output matmul | 1.674 | n/a |
| final `lm_head` matmul | 1.624 | n/a |
| full tiny GPT logits | 160.262 | 100.863 |

Profile artifacts:

- `results/local-profile-20260508-attention-softmax-current/manifest.json`
  measured the pre-fix softmax row at 47.898 us/call, 54.4 MB max RSS, and a
  96,504 byte traced allocation peak. The `sample(1)` report put most samples
  inside `create::reduction_ops::reduce_axis_ops`.
- `results/local-profile-20260508-attention-softmax-axislast-reduce/manifest.json`
  measured the row-reduction-only softmax at 30.499 us/call with similar memory
  shape, confirming the first bottleneck was the axis reducer.
- `results/local-profile-20260508-attention-causal-weak-scalar-power/manifest.json`
  measured the post-fix causal-attention row at 44.168 us/call, 54.1 MB max RSS,
  and a 205,720 byte traced allocation peak.
- `results/local-profile-20260508-attention-gpt-weak-scalar-power/manifest.json`
  measured the post-fix GPT row at 171.755 us/call, 54.6 MB max RSS, 54.9 MB
  allocation-pass max RSS, and a 202,800 byte traced allocation peak. The sample
  is now dominated by repeated f32 binary kernels, row broadcasts, and wrapper
  allocation around layer norm / softmax composition.

Next target: fused row softmax and fused row layer norm. The current arithmetic
is no longer absurd, but the benchmark still makes several Python/native
round-trips and temporary arrays per row-normalization step. A single-pass
last-axis layer norm and a stable row softmax would attack the remaining
1.57x GPT gap directly.

### 2026-05-08 `monpy.nn` compile fix

The first `src/nn` split failed at the Mojo extension boundary:

```text
src/lib.mojo:84:16: error: package 'nn' does not contain 'layer_norm_last_axis_ops'
```

The source file was present, but the name `nn` is already claimed by Modular/MAX
kernel packages in the compiler search path. Importing `from nn import ...` did
not resolve to monpy's local directory. A second attempt to import `nn.kernels`
hit the same owner. The working shape is:

- keep the Mojo loop bodies in `src/elementwise/kernels/nn.mojo`
- keep the PythonObject bridge functions in `src/create/ops/nn.mojo`
- re-export the bridge functions from `src/create/__init__.mojo`
- import the bridge functions in `src/lib.mojo` with the rest of `from create import ...`
- keep `python/monpy/nn` as the public Python import scope

This keeps the public Python surface as `monpy.nn` while avoiding collision with
MAX's own `nn` package. Focused verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_numeric.py::test_fused_layer_norm_matches_numpy_formula tests/python/numpy_compat/test_numeric.py::test_fused_softmax_matches_numpy_formula tests/python/numpy_compat/test_numeric.py::test_fused_scaled_masked_softmax_matches_numpy_formula -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python - <<'PY'
import monpy
import monumpy as mnp
print(monpy.nn.softmax is monpy.softmax)
print(mnp.nn.layer_norm is monpy.nn.layer_norm)
PY
```

The editable build completed with the existing ld64 target-version warning, the
three focused tests passed, and both import-scope identity checks printed
`True`. A one-round attention benchmark smoke also wrote
`results/local-sweep-20260508-nn-compile-smoke/manifest.json`; treat that as an
import/runner smoke only, not a stable performance sample.

Follow-up naming consolidation:

- `src/create/ops/elementwise.mojo` now owns the elementwise PythonObject bridge
  functions that used to sit in `src/create/elementwise_ops.mojo`
- `src/create/ops/nn.mojo` owns the public `monpy.nn` bridge functions
- `src/elementwise/kernels/nn.mojo` owns the row-wise NN loop bodies
- `src/elementwise/kernels/matmul.mojo` owns the matmul loop/BLAS dispatchers

Focused verification after the role-first package move:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
find src/create src/elementwise -name '*.mojo' -print0 | xargs -0 /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_numeric.py::test_fused_layer_norm_matches_numpy_formula tests/python/numpy_compat/test_numeric.py::test_fused_softmax_matches_numpy_formula tests/python/numpy_compat/test_numeric.py::test_fused_scaled_masked_softmax_matches_numpy_formula -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types attention --loops 1 --repeats 1 --rounds 1 --format json --sort ratio --output-dir results/local-sweep-20260508-role-first-smoke --no-progress --no-stdout
```

The build completed in 38.95 s, formatter rewrote three files, and the three
focused tests passed. The one-round role-first smoke wrote
`results/local-sweep-20260508-role-first-smoke/manifest.json`; single-sample
medians: softmax 15.584 us vs NumPy 20.375 us (0.765x), attention 34.125 us vs
NumPy 36.250 us (0.941x), GPT 111.917 us vs NumPy 140.750 us (0.795x). Treat
these as import/runner signal only.

### 2026-05-08 binary `out=` ufunc fast path

The heartbeat target sweep
`results/local-sweep-20260508-heartbeat-all-targets/manifest.json` showed that
attention rows were already ahead of NumPy on the current machine, while the
wrapper-bound `array/elementwise/binary_add_out_f32` row still cost 2.126x NumPy
time. The benchmark already had a lower-level extension comparison,
`binary_add_extension_out_f32`, at 1.128x, which isolated most of the remaining
gap to Python ufunc dispatch rather than the native loop.

`python/monpy/__init__.py` now gives the exact `ndarray, ndarray, out=ndarray`
binary ufunc case a direct `_native.binary_into(...)` path before the staged
kernel probe and promotion machinery. This is intentionally narrow: default
`where=True`, default `casting="same_kind"`, no `dtype=`, exact `ndarray` inputs,
and exact `ndarray` output.

Focused verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_numeric.py::test_binary_out_writes_existing_destination tests/python/numpy_compat/test_ufunc.py::test_ufunc_out_kwarg_writes_in_place tests/python/numpy_compat/test_ufunc.py::test_ufunc_dtype_kwarg_casts -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types array --loops 50 --repeats 5 --rounds 3 --vector-size 1024 --vector-sizes 16384 --matrix-sizes 16 --linalg-sizes 2 --format json --sort ratio --output-dir results/local-sweep-20260508-ufunc-out-fastpath --no-progress --no-stdout
```

The focused tests passed. Before/after for the wrapper-bound rows:

| row | before | after | delta |
| --- | ---: | ---: | ---: |
| `binary_add_out_f32` | 5.103 us, 2.116x | 2.955 us, 1.263x | 1.73x faster |
| `binary_add_f32` | 3.126 us, 1.248x | 3.050 us, 1.247x | neutral |
| `binary_add_extension_out_f32` | 2.694 us, 1.128x | 2.667 us, 1.135x | neutral |

Next target remains `complex/matmul_64_complex64`: it already reports
`used_accelerate=True`, so the investigation should focus on BLAS function
lookup/call overhead, row-major complex GEMM calling convention, and whether a
small-N Mojo complex microkernel can beat the framework call for 64x64.
