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

### 2026-05-08 macOS ILP64 complex GEMM

The refreshed heartbeat slice confirmed attention was no longer the hot
regression on this machine:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32` | 81.206 | 109.272 | 0.743x |
| `attention/softmax/causal_scores_t32_f32` | 8.144 | 10.272 | 0.814x |
| `attention/attention/causal_attention_t32_d32_f32` | 20.717 | 22.987 | 0.914x |

The same run left `complex/matmul_64_complex64` as the largest direct blocker:
16.257 us for monpy vs 7.221 us for NumPy, a 2.318x ratio. `used_accelerate()`
was already true, so the failure was not a missed BLAS dispatch.

The useful profiler fact was the symbol, not just the wall clock:

- pre-fix monpy sample:
  `maybe_matmul_contiguous -> cblas_sgemm` inside `libBLAS.dylib`
- NumPy sample:
  `CFLOAT_matmul_matrixmatrix -> cblas_cgemm$NEWLAPACK$ILP64`
- post-fix monpy sample:
  `maybe_matmul_contiguous -> cblas_cgemm$NEWLAPACK$ILP64`

The SDK headers explain the split. The old `cblas.h` complex GEMM takes `int`
sizes and is deprecated with guidance to compile against the updated CBLAS
interface; `cblas_new.h` routes `cblas_cgemm` through `__LAPACK_ALIAS`, and
`lapack_types.h` makes `__LAPACK_int` a `long` under ILP64. The local
`libBLAS.tbd` exposes both `_cblas_cgemm$NEWLAPACK` and
`_cblas_cgemm$NEWLAPACK$ILP64`, matching the symbol NumPy samples.

`src/accelerate.mojo` now uses macOS-only ILP64 function signatures for
`cblas_cgemm$NEWLAPACK$ILP64` and `cblas_zgemm$NEWLAPACK$ILP64`, while leaving
Linux on the existing LP64/OpenBLAS-compatible symbols. A raw ctypes probe on
the exact 64x64 complex64 workload measured the old symbol at about 17.020
us/call and the ILP64 symbol at about 8.490 us/call, which matched the monpy
delta after rebuilding.

Focused verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_einsum.py::test_complex64_matmul_via_cgemm tests/python/numpy_compat/test_einsum.py::test_complex128_matmul_via_zgemm tests/python/numpy_compat/test_numeric.py::test_matmul_matches_numpy_for_1d_and_2d -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260508-complex-ilp64-cgemm --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/matmul/matmul_64_complex64 --types complex --matrix-sizes 64 --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260508-complex-matmul64-ilp64-monpy --sample --no-perf-stat
```

Post-fix benchmark results:

| row | before | after | delta |
| --- | ---: | ---: | ---: |
| `complex/matmul_64_complex64` | 16.257 us, 2.318x | 7.277 us, 1.042x | 2.23x faster |

The post-fix profile loop measured 7.535 us/call, 49.7 MB max RSS, no major
faults, and a 1,590 byte traced allocation peak. Next target moves to the
complex conversion/view cluster: `astype_complex64_to_complex128` remains
2.008x NumPy, followed by `reversed_add_complex64` at 1.871x.

### 2026-05-08 complex width-cast lane fast path

The next live complex rerun after the ILP64 GEMM fix ranked the remaining
frontier as:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/casts/astype_complex64_to_complex128` | 5.289 | 2.579 | 2.051x |
| `complex/views/reversed_add_complex64` | 5.885 | 3.057 | 1.929x |
| `complex/interop/asarray_complex64` | 3.052 | 1.958 | 1.556x |
| `complex/interop/array_copy_complex128` | 3.807 | 2.443 | 1.547x |

The old cast path treated all complex casts as the general fallback:
per-logical-element, fetch real/imag via accessors, widen to `Float64`, then
write through complex setters. That is correct but silly for complex64 ↔
complex128 when both arrays are C-contiguous, because the storage is just
interleaved real lanes.

`src/array/cast.mojo` now has `_complex_contig_lane_cast[src_dt, dst_dt]`, which
walks the `2N` real lanes with SIMD loads, vector casts, and SIMD stores. It is
intentionally narrow: only complex64 ↔ complex128 width changes enter this path.
Complex → real still drops imaginary values through the existing fallback, and
real → complex still writes zero imaginary lanes through the existing setter
logic.

Focused verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_astype_between_widths_matches_numpy tests/python/numpy_compat/test_complex.py::test_complex_astype_drops_imag_to_real_target tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_round_trip -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260508-complex-cast-width-fastpath --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/casts/astype_complex64_to_complex128 --types complex --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260508-complex-astype-width-fastpath --sample --no-perf-stat
```

Post-fix benchmark results:

| row | before | after | delta |
| --- | ---: | ---: | ---: |
| `complex/casts/astype_complex64_to_complex128` | 5.289 us, 2.051x | 3.298 us, 1.260x | 1.60x faster |

The post-fix profile loop measured 3.442 us/call, 49.7 MB max RSS, no major
faults, and a 1,590 byte traced allocation peak. The sample now puts the hot
native frame in `_complex_contig_lane_cast[f32,f64]`, which is the expected
storage-level path. The new top complex deficit is `reversed_add_complex64` at
1.891x, then complex ingress/copy rows around 1.5x.

### 2026-05-08 reversed complex add native rank-1 path

The next complex slice kept `complex/views/reversed_add_complex64` as the most
visible view/kernel row:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/views/reversed_add_complex64` | 5.807 | 3.011 | 1.929x |
| `complex/interop/array_copy_complex128` | 3.791 | 2.443 | 1.551x |
| `complex/interop/asarray_complex64` | 3.023 | 1.958 | 1.543x |
| `complex/elementwise/binary_mul_complex64` | 3.488 | 2.547 | 1.369x |

The pre-fix backend probe reported `used_accelerate=True`; the sample made the
problem concrete. For a 1024-element `complex64[::-1] + complex64[::-1]`, monpy
was issuing two small strided `vDSP_vadd` calls, one for real lanes and one for
imaginary lanes. At this size the library-call and strided-dispatch overhead
beat the useful arithmetic. The scalar generic fallback was not the right shape
either, because it pays `physical_offset` per complex element.

`src/elementwise/kernels/complex.mojo` now has a rank-1 strided complex kernel
that walks physical indexes incrementally. For the hot `complex64` reversed
ADD/SUB case it uses a small SIMD pair-reversal path: load two adjacent complex
values from the reversed physical span, reverse pair order in-register, then
store four float lanes to the contiguous result. MUL/DIV also use the
incremental-index rank-1 path, with the existing Smith division logic, so
negative-stride complex arithmetic no longer falls back to `physical_offset`.

`python/monpy/__init__.py` also trims the exact one-dimensional `[::-1]` path:
it calls the native `reverse_1d_method()` directly before the extra `ndim()`
probe, and only falls back to the general slice machinery when the native method
reports a non-rank-1 array. This preserves the multidimensional `a[::-1]`
behavior while shaving the benchmark's two view constructions.

Focused verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_strided_arithmetic_preserves_imaginary_part tests/python/numpy_compat/test_indexing.py::test_rank1_full_reverse_slice_matches_numpy_and_shares_storage -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-reversed-simd-fast-slice --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/views/reversed_add_complex64 --types complex --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-reversed-add-simd-fast-slice --sample --no-perf-stat
```

Post-fix benchmark results:

| row | before | after | delta |
| --- | ---: | ---: | ---: |
| `complex/views/reversed_add_complex64` | 5.807 us, 1.929x | 4.387 us, 1.361x | 1.32x faster |

The profile loop measured 4.277 us/call, 49.6 MB max RSS, no major faults, and
a 1,728 byte traced allocation peak. Backend reporting flipped from Accelerate
to FUSED for the result, which is expected: the hot path now avoids the two
strided vDSP calls. A direct local timing split also showed the slice change:
one `complex64` reverse view fell from about 0.639 us to 0.499 us, and the full
`a[::-1] + b[::-1]` expression fell from about 2.58 us to 1.99 us in the
micro-timer.

The updated complex frontier is now the ingress/copy cluster plus one arithmetic
kernel:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/array_copy_complex128` | 4.036 | 2.659 | 1.518x |
| `complex/interop/asarray_complex64` | 3.104 | 2.055 | 1.511x |
| `complex/elementwise/binary_mul_complex64` | 3.577 | 2.615 | 1.368x |
| `complex/views/reversed_add_complex64` | 4.387 | 3.224 | 1.361x |

Next target should be complex ingress/copy. The likely win is keeping NumPy
complex buffers on a narrow typed copy path instead of round-tripping through
generic scalar extraction.

### 2026-05-08 direct NumPy buffer ingress

The typed-buffer hypothesis was tested first and rejected. A native
`PyBUF_STRIDES` bridge that skipped `PyBUF_FORMAT` made the raw native call a
little faster, but the Python-side `dtype.str` proof cost more than the native
PEP-3118 decode it was trying to avoid. The official complex slice moved the
wrong way, so that path was cut before committing.

The smaller, useful fix is simpler: when `monpy.asarray()` sees a NumPy ndarray,
it now calls the existing native `asarray_from_buffer` bridge directly instead
of routing through `runtime.ops_numpy.is_array_input()` and
`runtime.ops_numpy._from_numpy_unchecked()`. That keeps dtype/copy semantics in
the same native buffer decoder, avoids a Python module wrapper hop on the hot
path, and still falls back to `runtime.ops_numpy` for any future exotic NumPy
case the local probe misses.

Focused verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_copy_false_shares_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_copy_true_detaches_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_readonly_copy_false_raises_and_copy_none_detaches tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_round_trip tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_false_shares_storage tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_true_detaches_storage -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-direct-numpy-buffer --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/asarray_complex64 --types complex --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-asarray-complex64-direct-numpy-buffer --sample --no-perf-stat
```

Post-fix benchmark results:

| row | before | after | delta |
| --- | ---: | ---: | ---: |
| `complex/interop/asarray_complex64` | 3.005 us, 1.549x | 2.882 us, 1.484x | 1.04x faster |
| `complex/interop/array_copy_complex128` | 3.758 us, 1.535x | 3.647 us, 1.475x | 1.03x faster |

The profile loop for `complex/interop/asarray_complex64` moved from 3.242
us/call to 2.951 us/call, with max RSS essentially flat at about 49 MB and the
same 1,590 byte traced allocation peak. The `sample` run measured 3.418 us/call
before and 3.128 us/call after in the child process. The hot native frame is
still `buffer::asarray_from_buffer_ops`, which is expected: this patch removes
Python wrapper overhead, not memory traffic or SIMD work. No hardware PMU
counters were collected in this run because `--no-perf-stat` was used.

The remaining complex frontier after this slice:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/elementwise/binary_mul_complex64` | 3.455 | 2.538 | 1.364x |
| `complex/elementwise/binary_add_complex64` | 3.082 | 2.438 | 1.265x |
| `complex/elementwise/binary_add_complex128` | 3.253 | 2.619 | 1.248x |
| `complex/matmul_64_complex64` | 7.517 | 7.124 | 1.055x |

Next target should be `complex/elementwise/binary_mul_complex64`. It is no
longer an ingress problem; the likely work is inside the complex multiply
kernel, specifically redundant real/imag lane loads, scalar temporary pressure,
and whether a wider interleaved-lane SIMD path beats the current per-element
Smith-compatible shape for multiplication.

### 2026-05-08 complex64 multiply interleaved SIMD

The next refreshed complex slice ranked the live deficits as:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.922 | 1.953 | 1.501x |
| `complex/interop/array_copy_complex128` | 3.679 | 2.463 | 1.491x |
| `complex/views/reversed_add_complex64` | 4.209 | 3.065 | 1.371x |
| `complex/elementwise/binary_mul_complex64` | 3.497 | 2.547 | 1.368x |

The multiply row was the useful kernel target. It was still using the scalar
contiguous complex loop: two loads from each input, four multiplies, two
add/sub operations, and two stores per complex element. That is correct, but it
does not use the fact that complex64 storage is interleaved float lanes.

`src/elementwise/kernels/complex.mojo` now adds a `float32x4` path for
contiguous complex multiply. Each vector load covers two complex values:
`[a0,b0,a1,b1]` and `[c0,d0,c1,d1]`. The kernel broadcasts real and imaginary
rhs lanes with shuffles, computes real and imaginary candidate lanes, then
interleaves `[real0, imag0, real1, imag1]` back into the output vector. The
scalar tail remains for odd element counts, and the existing Smith division path
is untouched.

`src/elementwise/binary_dispatch.mojo` now also marks the non-Accelerate
contiguous complex path as `BackendKind.FUSED`, so benchmark/profile manifests
report the native fused kernel instead of the default backend code.

Focused verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_arithmetic_add_sub_mul_div_match_numpy tests/python/numpy_compat/test_complex.py::test_complex_strided_arithmetic_preserves_imaginary_part tests/python/numpy_compat/test_complex.py::test_complex_scalar_mul_with_complex_constant tests/python/numpy_compat/test_complex.py::test_complex64_contiguous_multiply_uses_fused_kernel tests/python/numpy_compat/test_complex.py::test_complex_scalar_mul_with_real_int -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-mul-simd-fused --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/elementwise/binary_mul_complex64 --types complex --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-complex-mul-simd-fused --sample --no-perf-stat
```

Post-fix benchmark results:

| row | before | after | delta |
| --- | ---: | ---: | ---: |
| `complex/elementwise/binary_mul_complex64` | 3.497 us, 1.368x | 3.210 us, 1.180x | 1.09x faster |

The post-fix profile manifest reports `used_fused=True`, `backend_code=2`, no
major faults, a 1,590 byte traced allocation peak, and max RSS around 49.6 MB.
The profile loop moved from 3.593 us/call to 3.275 us/call; the `sample` child
loop moved from 3.745 us/call to 3.424 us/call. Hardware PMU counters were not
collected in this run because `--no-perf-stat` was used.

The remaining complex frontier after this slice:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/array_copy_complex128` | 3.849 | 2.590 | 1.454x |
| `complex/interop/asarray_complex64` | 3.002 | 2.078 | 1.449x |
| `complex/views/reversed_add_complex64` | 4.324 | 3.169 | 1.356x |
| `complex/casts/astype_complex64_to_complex128` | 3.387 | 2.704 | 1.272x |
| `complex/elementwise/binary_add_complex64` | 3.270 | 2.575 | 1.255x |
| `complex/elementwise/binary_add_complex128` | 3.330 | 2.751 | 1.211x |
| `complex/elementwise/binary_mul_complex64` | 3.210 | 2.638 | 1.180x |
| `complex/matmul_64_complex64` | 7.355 | 7.147 | 1.055x |

Next target should move back to the interop/copy cluster, but the likely owner
is allocation and Python/native wrapper overhead rather than another SIMD
kernel. The useful question is whether `array(copy=True)` can call a narrower
native copy entrypoint that combines buffer import, allocation, and memcpy with
fewer Python object transitions.

### 2026-05-09 direct contiguous buffer copy and small complex add

The refreshed complex slice ranked `array_copy_complex128` and
`asarray_complex64` above the arithmetic rows:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/array_copy_complex128` | 3.603 | 2.438 | 1.477x |
| `complex/interop/asarray_complex64` | 2.885 | 1.990 | 1.450x |
| `complex/views/reversed_add_complex64` | 4.142 | 3.039 | 1.364x |
| `complex/elementwise/binary_add_complex64` | 3.083 | 2.516 | 1.245x |
| `complex/elementwise/binary_add_complex128` | 3.271 | 2.673 | 1.238x |
| `complex/elementwise/binary_mul_complex64` | 3.085 | 2.543 | 1.204x |

The first fix keeps same-dtype C-contiguous buffer copies inside
`asarray_from_buffer_ops`: allocate the destination array, compute the storage
byte count, and `memcpy` directly from `Py_buffer.buf`. The old path wrapped the
source as an external `Array`, cloned shape/stride metadata, then called
`copy_c_contiguous`. The direct copy leaf preserves the fallback path for
strided inputs, dtype conversion, and readonly/copy policy failures.

The second fix changes the small complex ADD/SUB cost model. Contiguous
complex64/complex128 ADD/SUB was going through Accelerate vDSP even for the
1024-element benchmark row. At that size the framework call toll is larger than
the existing typed SIMD loop, so `maybe_complex_binary_contiguous_accelerate`
now only uses vDSP at 4096 complex elements and above. The 1024-element row
stays in the Mojo fused kernel (`backend_code=2`, `used_fused=True`,
`used_accelerate=False`).

Focused verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_arithmetic_add_sub_mul_div_match_numpy tests/python/numpy_compat/test_complex.py::test_complex_strided_arithmetic_preserves_imaginary_part tests/python/numpy_compat/test_complex.py::test_complex64_contiguous_multiply_uses_fused_kernel tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_true_detaches_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_copy_true_detaches_storage -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-small-mojo-add --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/elementwise/binary_add_complex64 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-binary-add-complex64-mojo-small --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/elementwise/binary_add_complex64 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-binary-add-complex64-numpy --sample --no-perf-stat
```

Post-fix benchmark results:

| row | before | after | delta |
| --- | ---: | ---: | ---: |
| `complex/interop/array_copy_complex128` | 3.603 us, 1.477x | 3.539 us, 1.421x | 1.02x faster |
| `complex/elementwise/binary_add_complex64` | 3.083 us, 1.245x | 2.981 us, 1.216x | 1.03x faster |
| `complex/elementwise/binary_add_complex128` | 3.271 us, 1.238x | 3.227 us, 1.226x | 1.01x faster |
| `complex/elementwise/binary_mul_complex64` | 3.085 us, 1.204x | 3.067 us, 1.196x | flat/noise |

Direct microbenchmarks show the copy leaf moving even though the full
benchmark is mostly wrapper-bound: native complex128 buffer copy moved from
1.002 us to 0.872 us, while `monpy.array(..., copy=True)` moved from 1.434 us
to 1.313 us. NumPy stayed around 0.42 us for the same copy.

The profile comparison for `complex/elementwise/binary_add_complex64` reports:

| candidate | us/call | max RSS | traced peak | backend |
| --- | ---: | ---: | ---: | --- |
| monpy | 3.200 | 49.7 MB | 1,590 B | fused Mojo, no Accelerate |
| numpy | 2.643 | 49.5 MB | 17,608 B | n/a |

No PMU counters were collected in this run because `--no-perf-stat` was used.
The macOS `sample(1)` stacks were written under
`results/local-profile-20260509-binary-add-complex64-*`; they mostly show the
benchmark harness and Python call/attribute machinery, so the next useful
profile pass should use Instruments CPU Counters or a Linux `perf stat` run
when we want instruction/cache ratios rather than wall-clock deltas.

Remaining frontier after this slice:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.896 | 1.972 | 1.484x |
| `complex/interop/array_copy_complex128` | 3.539 | 2.481 | 1.421x |
| `complex/views/reversed_add_complex64` | 4.167 | 3.068 | 1.361x |
| `complex/casts/astype_complex64_to_complex128` | 3.267 | 2.576 | 1.265x |
| `complex/elementwise/binary_add_complex128` | 3.227 | 2.636 | 1.226x |
| `complex/elementwise/binary_add_complex64` | 2.981 | 2.451 | 1.216x |
| `complex/elementwise/binary_mul_complex64` | 3.067 | 2.571 | 1.196x |
| `complex/matmul_64_complex64` | 7.576 | 7.161 | 1.061x |

Next target should stay on the interop cluster. The direct copy leaf helped the
native portion, but the row is still 1.42x NumPy because object creation and
buffer classification dominate the remaining cost.

### 2026-05-09 specialized complex buffer wrappers

The next pass kept the existing benchmark harness fixed and refreshed the
complex slice:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.911 | 1.963 | 1.484x |
| `complex/interop/array_copy_complex128` | 3.479 | 2.456 | 1.420x |
| `complex/views/reversed_add_complex64` | 4.134 | 3.044 | 1.358x |
| `complex/casts/astype_complex64_to_complex128` | 3.258 | 2.557 | 1.270x |

One important measurement wrinkle: `time_call()` still enters
`warnings.catch_warnings()` through `call_bench_fn()` for every timed loop
iteration. A raw microbench put that context-manager tax at about 2.1 us/call
on this machine. That overhead applies to both monpy and NumPy rows, so the
official ratio remains useful for the current campaign, but raw microbenchmarks
are better for isolating the actual ingress leaf.

The raw ingress timings showed the target clearly:

| path | before | after |
| --- | ---: | ---: |
| generic native complex64 view | 0.461 us | 0.461 us |
| specialized native complex64 view | n/a | 0.366 us |
| `monumpy.asarray(..., complex64, copy=False)` | 0.838 us | 0.665 us |
| generic native complex128 copy | 0.828 us | 0.828 us |
| specialized native complex128 copy | n/a | 0.729 us |
| `monumpy.array(..., complex128, copy=True)` | 1.277 us | 1.051 us |

`src/buffer.mojo` now splits the old Python-object argument decoder from the
actual buffer implementation. The generic public function still accepts
`requested_dtype_obj` and `copy_obj`, but the hot wrappers call the shared
implementation with native `Int` constants:

- `asarray_complex64_view_from_buffer(obj)` for the complex64 zero-copy path.
- `asarray_complex128_copy_from_buffer(obj)` for the complex128 forced-copy
  path.

The top-level NumPy-array branch in `python/monpy/__init__.py` uses those
wrappers when the caller passes the exact monpy dtype singleton and copy policy
used by the benchmark. Everything else falls back to the generic buffer bridge,
so dtype conversion, readonly handling, strided copies, and exotic dtype errors
stay centralized.

Focused verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_round_trip tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_false_shares_storage tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_true_detaches_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_copy_false_shares_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_copy_true_detaches_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_readonly_copy_false_raises_and_copy_none_detaches tests/python/numpy_compat/test_array_interface.py::test_ops_numpy_from_numpy_dtype_and_copy_arguments -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-specialized-buffer-wrappers-final --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/asarray_complex64 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-asarray-complex64-specialized-buffer --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/array_copy_complex128 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-array-copy-complex128-specialized-buffer --sample --no-perf-stat
```

Official post-fix benchmark results:

| row | before | after | delta |
| --- | ---: | ---: | ---: |
| `complex/interop/asarray_complex64` | 2.911 us, 1.484x | 2.817 us, 1.417x | 1.03x faster |
| `complex/interop/array_copy_complex128` | 3.479 us, 1.420x | 3.312 us, 1.318x | 1.05x faster |

The profile manifests reported `complex/interop/asarray_complex64` at 2.996
us/call and `complex/interop/array_copy_complex128` at 3.470 us/call, both with
max RSS around 49-50 MB, 1,590 byte traced allocation peaks, no major faults,
and default backend metadata. No hardware PMU counters were collected because
`--no-perf-stat` was used; the `sample(1)` stacks were captured under the two
`results/local-profile-20260509-*-specialized-buffer` directories.

Remaining frontier after this slice:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.817 | 1.999 | 1.417x |
| `complex/views/reversed_add_complex64` | 4.131 | 3.121 | 1.359x |
| `complex/interop/array_copy_complex128` | 3.312 | 2.547 | 1.318x |
| `complex/casts/astype_complex64_to_complex128` | 3.355 | 2.613 | 1.282x |
| `complex/elementwise/binary_add_complex128` | 3.446 | 2.687 | 1.258x |

Next target should probably be `complex/views/reversed_add_complex64` again.
The interop rows still lead, but their remaining raw leaf is already
sub-microsecond; the official row is now dominated by harness and Python wrapper
constant factors. The reversed-add row has real native work left in negative
stride handling and may have more actual loot per line changed.

### 2026-05-09 reverse view wrapper fast path

This pass started by trying a four-complex unroll inside the negative-stride
`complex64` ADD/SUB kernel. That did not move the leaf: presliced monpy add
stayed around 0.94 us and the official `reversed_add_complex64` ratio stayed at
1.359x. The patch was reverted before landing. The useful deficit was still the
view creation path around the already-fused native kernel.

Raw timing before the wrapper patch put a single `z[::-1]` at about 0.456 us
and the full `z[::-1] + w[::-1]` expression at about 1.932 us. The bound native
reverse method itself was only about 0.230 us, so roughly half the slice cost was
Python dispatch and wrapper construction.

`src/array/accessors.mojo` now owns the bound `Array.reverse_1d_method()` path
directly. The public module-level `_native.reverse_1d()` function still uses the
generic shape helper, but the method used by `ndarray.__getitem__` builds the
rank-1 reverse view from the receiver fields without re-entering the generic
factory. `python/monpy/__init__.py` also inlines the wrapper construction for
the exact `[::-1]` case, avoiding the extra `ndarray._wrap(...)` staticmethod
dispatch in this tiny hot path.

Raw post-fix timing:

| path | before | after |
| --- | ---: | ---: |
| native `reverse_1d_method()` | 0.230 us | 0.228 us |
| `ndarray._wrap(reverse_1d_method())` | 0.342 us | 0.337 us |
| `z[::-1]` | 0.456 us | 0.409 us |
| `z[::-1] + w[::-1]` | 1.932 us | 1.819 us |

Focused verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_strided_arithmetic_preserves_imaginary_part tests/python/numpy_compat/test_complex.py::test_complex_arithmetic_add_sub_mul_div_match_numpy tests/python/numpy_compat/test_complex.py::test_complex64_contiguous_multiply_uses_fused_kernel -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_strided_arithmetic_preserves_imaginary_part tests/python/numpy_compat/test_complex.py::test_complex_arithmetic_add_sub_mul_div_match_numpy tests/python/numpy_compat/test_indexing.py -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_strided_arithmetic_preserves_imaginary_part tests/python/numpy_compat/test_indexing.py -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-reverse-wrapper-fastpath --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/views/reversed_add_complex64 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-reversed-add-wrapper-fastpath-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/views/reversed_add_complex64 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-reversed-add-wrapper-fastpath-numpy --sample --no-perf-stat
```

Official post-fix benchmark results:

| row | before | after | delta |
| --- | ---: | ---: | ---: |
| `complex/views/reversed_add_complex64` | 4.131 us, 1.359x | 4.076 us, 1.335x | 1.01x faster |

The profile manifests reported `complex/views/reversed_add_complex64` at 4.210
us/call for monpy and 3.162 us/call for NumPy. Monpy stayed on backend code 2,
the fused Mojo path, with max RSS around 49.3 MB and a 1,728 byte traced peak.
NumPy reported max RSS around 49.3 MB and a 17,760 byte traced peak. No
hardware PMU counters were collected because `--no-perf-stat` was used. The
macOS `sample(1)` captures were written under
`results/local-profile-20260509-reversed-add-wrapper-fastpath-*`; the visible
stacks are still dominated by CPython frame evaluation, attribute lookup, and
benchmark harness work around the native kernel.

Remaining frontier after this slice:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.790 | 1.971 | 1.408x |
| `complex/interop/array_copy_complex128` | 3.280 | 2.441 | 1.339x |
| `complex/views/reversed_add_complex64` | 4.076 | 3.054 | 1.335x |
| `complex/casts/astype_complex64_to_complex128` | 3.301 | 2.582 | 1.279x |
| `complex/elementwise/binary_add_complex128` | 3.225 | 2.684 | 1.225x |

Next target should move to `complex/interop/array_copy_complex128` or the
attention rows. For complex reverse views, the easy wrapper win is spent and the
remaining ratio is mostly Python object overhead around two view creations plus
one already-fused native add.

### 2026-05-09 direct `array(..., complex128, copy=True)` buffer path

The next live refresh included both complex and attention rows:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.815 | 1.996 | 1.409x |
| `complex/interop/array_copy_complex128` | 3.271 | 2.478 | 1.324x |
| `complex/views/reversed_add_complex64` | 4.078 | 3.110 | 1.312x |
| `attention/attention/causal_attention_t32_d32_f32` | 20.488 | 21.857 | 0.937x |
| `attention/softmax/causal_scores_t32_f32` | 7.995 | 10.007 | 0.799x |
| `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32` | 78.485 | 107.275 | 0.733x |

The attention rows are already ahead of NumPy in this slice, so this pass stayed
on `complex/interop/array_copy_complex128`. The raw copy leaf showed a facade
gap: `_native.asarray_complex128_copy_from_buffer(src)` was about 0.705 us,
wrapping it as `ndarray(...)` was about 0.801 us, but `monpy.array(src,
dtype=complex128, copy=True)` was about 1.051 us because it bounced through
`asarray(...)` before reaching the exact native wrapper.

`python/monpy/__init__.py` now gives `array(..., dtype=complex128, copy=True)`
a direct native-buffer path. It still falls back to `asarray(...)` for lists,
non-exact dtype aliases, `copy=None`/`False`, existing monpy arrays, and the
general object-coercion paths.

Raw post-fix timing:

| path | before | after |
| --- | ---: | ---: |
| native complex128 copy wrapper | 0.711 us | 0.705 us |
| `ndarray(native complex128 copy)` | 0.808 us | 0.801 us |
| `monpy.asarray(..., complex128, copy=True)` | 1.025 us | 1.029 us |
| `monpy.array(..., complex128, copy=True)` | 1.051 us | 0.876 us |
| `numpy.array(..., complex128, copy=True)` | 0.461 us | 0.453 us |

Focused verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_round_trip tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_true_detaches_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_copy_true_detaches_storage tests/python/numpy_compat/test_array_coercion.py::test_array_and_asarray_copy_rules_for_existing_monpy_arrays -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_array_coercion.py::test_explicit_supported_dtype_casts_match_numpy tests/python/numpy_compat/test_array_coercion.py::test_array_and_asarray_copy_rules_for_existing_monpy_arrays tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_round_trip tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_true_detaches_storage -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-array-c128-direct-copy --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/array_copy_complex128 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-array-copy-complex128-direct-array-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/array_copy_complex128 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-array-copy-complex128-direct-array-numpy --sample --no-perf-stat
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python -m monpy._bench.mojo_sweep --format json --sort ratio --output-dir results/local-sweep-20260509-mojo-stdlib-refresh --no-stdout --timeout 300
```

Official post-fix benchmark results:

| row | before | after | delta |
| --- | ---: | ---: | ---: |
| `complex/interop/array_copy_complex128` | 3.271 us, 1.324x | 3.115 us, 1.268x | 1.05x faster |

The profile manifests reported `complex/interop/array_copy_complex128` at 3.245
us/call for monpy and 2.647 us/call for NumPy. Monpy used the generic backend
code 0 copy path, max RSS was about 49.6 MB, and the traced peak was 1,590
bytes. NumPy's traced peak was 33,992 bytes. No hardware PMU counters were
collected because `--no-perf-stat` was used; the `sample(1)` captures live under
`results/local-profile-20260509-array-copy-complex128-direct-array-*`.

The pure-Mojo stdlib sweep also completed. The largest candidate/stdlib ratios
were small: `small_matmul_f32_8` at 1.078x, `add_f32_1m` at 1.068x,
`sum_f32_1k` at 1.044x, and `prod_f64_64k` at 1.044x. That says the current
high-ratio NumPy-facing rows are mostly facade/object-bound, not a clear signal
that a production kernel should be replaced with a stdlib primitive today.

Remaining frontier after this slice:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.793 | 1.963 | 1.419x |
| `complex/views/reversed_add_complex64` | 4.086 | 3.072 | 1.329x |
| `complex/casts/astype_complex64_to_complex128` | 3.323 | 2.588 | 1.286x |
| `complex/interop/array_copy_complex128` | 3.115 | 2.468 | 1.268x |
| `complex/elementwise/binary_add_complex128` | 3.253 | 2.658 | 1.224x |

Next target should be `complex/interop/asarray_complex64`. It remains the top
ratio row, and the stdlib sweep does not point to a lower-level Mojo kernel as
the immediate blocker.

### 2026-05-09 direct complex64 copy-false buffer entry

The next refresh kept `complex/interop/asarray_complex64` at the top of the
complex frontier:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.841 | 2.063 | 1.410x |
| `complex/views/reversed_add_complex64` | 4.257 | 3.252 | 1.312x |
| `complex/casts/astype_complex64_to_complex128` | 3.437 | 2.737 | 1.258x |
| `complex/interop/array_copy_complex128` | 3.203 | 2.592 | 1.233x |
| `complex/elementwise/binary_add_complex128` | 3.398 | 2.780 | 1.220x |

Raw split before the patch:

| path | time |
| --- | ---: |
| `_is_numpy_array(src)` | 0.077 us |
| native complex64 view wrapper | 0.349 us |
| `ndarray(native, owner=src)` | 0.487 us |
| `monpy.asarray(..., complex64, copy=False)` | 0.665 us |
| `numpy.asarray(..., complex64)` | 0.073 us |

I first tried specializing `asarray_complex64_view_from_buffer_ops` in Mojo so
the wrapper would skip generic copy/cast branch machinery. That did not help:
the native wrapper stayed around 0.35 us and public `asarray` regressed to about
0.684 us. The patch was reverted before landing.

The useful change is in `python/monpy/__init__.py`: exact
`asarray(obj, dtype=complex64, copy=False)` now tries the complex64 native
buffer view before asking whether `obj` is a NumPy array. For the benchmark's
NumPy input this removes the `_is_numpy_array(...)` detector from the hot
success path. Error mapping remains the same for wrong dtype, read-only arrays,
and unsupported NumPy dtypes.

Raw post-fix timing:

| path | before | after |
| --- | ---: | ---: |
| native complex64 view wrapper | 0.349 us | 0.350 us |
| `ndarray(native, owner=src)` | 0.487 us | 0.487 us |
| `monpy.asarray(..., complex64, copy=False)` | 0.665 us | 0.587 us |
| `numpy.asarray(..., complex64)` | 0.073 us | 0.074 us |

Focused verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_round_trip tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_false_shares_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_copy_false_shares_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_readonly_copy_false_raises_and_copy_none_detaches tests/python/numpy_compat/test_array_interface.py::test_ops_numpy_from_numpy_dtype_and_copy_arguments -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-asarray-c64-direct --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/asarray_complex64 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-asarray-complex64-direct-buffer-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/asarray_complex64 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-asarray-complex64-direct-buffer-numpy --sample --no-perf-stat
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python -m monpy._bench.mojo_sweep --format json --sort ratio --output-dir results/local-sweep-20260509-mojo-stdlib-after-asarray-c64 --no-stdout --timeout 300
```

Official post-fix benchmark results:

| row | before | after | delta |
| --- | ---: | ---: | ---: |
| `complex/interop/asarray_complex64` | 2.841 us, 1.410x | 2.632 us, 1.365x | 1.08x faster |

The profile manifests reported `complex/interop/asarray_complex64` at 2.797
us/call for monpy and 2.085 us/call for NumPy. Monpy used backend code 0, max
RSS was about 49.0 MB, and the traced peak was 1,590 bytes. NumPy's traced peak
was 1,406 bytes. No hardware PMU counters were collected because
`--no-perf-stat` was used; the `sample(1)` captures live under
`results/local-profile-20260509-asarray-complex64-direct-buffer-*`.

The pure-Mojo stdlib sweep still does not implicate a production kernel. Its top
candidate/stdlib ratios were `small_matmul_f32_8` at 1.076x,
`small_matmul_f64_8` at 1.042x, and `min_f64_64k` at 1.040x.

Remaining frontier after this slice:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.632 | 1.947 | 1.365x |
| `complex/views/reversed_add_complex64` | 4.036 | 2.994 | 1.342x |
| `complex/casts/astype_complex64_to_complex128` | 3.263 | 2.534 | 1.287x |
| `complex/interop/array_copy_complex128` | 3.024 | 2.405 | 1.254x |
| `complex/elementwise/binary_add_complex128` | 3.203 | 2.597 | 1.231x |

Next target should move back to `complex/views/reversed_add_complex64` or
`complex/casts/astype_complex64_to_complex128`. The remaining asarray gap is now
mostly wrapper construction versus NumPy returning its input object.

### 2026-05-09 cached native reverse views

The next complex refresh put `asarray_complex64` and `reversed_add_complex64`
near the top again:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.808 | 2.060 | 1.365x |
| `complex/views/reversed_add_complex64` | 4.308 | 3.216 | 1.330x |
| `complex/casts/astype_complex64_to_complex128` | 3.457 | 2.681 | 1.282x |
| `complex/interop/array_copy_complex128` | 3.246 | 2.643 | 1.241x |
| `complex/elementwise/binary_add_complex64` | 3.207 | 2.625 | 1.222x |

The remaining reverse-add cost was not the fused complex64 add. A raw split
showed presliced monpy add around 0.96 us, while the full expression paid for
two `[::-1]` calls every iteration. `ndarray` now caches the native reverse view
object in a private `_reverse_native` slot, but still returns a fresh Python
wrapper for each `a[::-1]` call. That avoids the repeated Mojo view construction
without making `a[::-1] is a[::-1]` true.

Raw post-fix timing:

| path | before | after |
| --- | ---: | ---: |
| `z[::-1]` | 0.409 us | 0.185 us |
| `z[::-1] is z[::-1]` | false | false |
| presliced monpy add | 0.948 us | 0.961 us |
| `z[::-1] + w[::-1]` | 1.819 us | 1.361 us |
| NumPy `z[::-1] + w[::-1]` | 0.971 us | 0.983 us |

Focused verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_indexing.py tests/python/numpy_compat/test_complex.py::test_complex_strided_arithmetic_preserves_imaginary_part tests/python/numpy_compat/test_array_coercion.py::test_astype_supported_cast_matrix_matches_numpy -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-reverse-native-cache --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/views/reversed_add_complex64 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-reversed-add-native-cache-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/views/reversed_add_complex64 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-reversed-add-native-cache-numpy --sample --no-perf-stat
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python -m monpy._bench.mojo_sweep --format json --sort ratio --output-dir results/local-sweep-20260509-mojo-stdlib-refresh-0244 --no-stdout --timeout 300
```

Official post-fix benchmark results:

| row | before | after | delta |
| --- | ---: | ---: | ---: |
| `complex/views/reversed_add_complex64` | 4.308 us, 1.330x | 3.591 us, 1.172x | 1.20x faster |

The profile manifests reported `complex/views/reversed_add_complex64` at 3.728
us/call for monpy and 3.208 us/call for NumPy. Monpy stayed on backend code 2,
the fused Mojo path, with max RSS about 49.5 MB and a 1,598 byte traced peak.
NumPy's traced peak was 17,760 bytes. No hardware PMU counters were collected
because `--no-perf-stat` was used; the `sample(1)` captures live under
`results/local-profile-20260509-reversed-add-native-cache-*`.

The pure-Mojo stdlib sweep stayed quiet. The largest candidate/stdlib ratios
were `small_matmul_f32_8` at 1.077x, `small_matmul_f64_8` at 1.035x,
`sum_f64_1k` at 1.022x, and `sum_f32_1k` at 1.016x.

Remaining frontier after this slice:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.708 | 1.975 | 1.369x |
| `complex/casts/astype_complex64_to_complex128` | 3.322 | 2.574 | 1.283x |
| `complex/interop/array_copy_complex128` | 3.132 | 2.487 | 1.255x |
| `complex/elementwise/binary_add_complex64` | 3.042 | 2.485 | 1.224x |
| `complex/elementwise/binary_add_complex128` | 3.278 | 2.678 | 1.223x |
| `complex/views/reversed_add_complex64` | 3.591 | 3.057 | 1.172x |

Next target should be `complex/casts/astype_complex64_to_complex128` or another
interop facade cut. `asarray_complex64` is still the top ratio, but most of its
remaining gap is the unavoidable wrapper around a zero-copy external buffer.

### 2026-05-09 exact-DType `astype` facade fast path

The next complex refresh made the cast row the best target with actual native
work behind it:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.853 | 2.074 | 1.361x |
| `complex/casts/astype_complex64_to_complex128` | 3.511 | 2.763 | 1.281x |
| `complex/interop/array_copy_complex128` | 3.432 | 2.608 | 1.280x |
| `complex/elementwise/binary_add_complex64` | 3.186 | 2.590 | 1.231x |
| `complex/elementwise/binary_add_complex128` | 3.448 | 2.809 | 1.228x |

The raw split showed the interleaved-lane cast was already close to NumPy:
native monpy `astype` was about 0.645 us and NumPy's cast was about 0.543 us.
The public method was 1.067 us because `ndarray.astype(...)` always resolved the
dtype and then read `self.dtype`, even when `copy=True` meant the identity check
could not return.

`ndarray.astype` now special-cases exact monpy `DType` objects. For
`copy=True`, it dispatches straight to `_native.astype(self._native, dtype.code)`.
For `copy=False`, it still returns `self` when the source dtype already matches.
The generic resolver path remains for NumPy dtype aliases, Python scalar types,
and strings.

Raw post-fix timing:

| path | before | after |
| --- | ---: | ---: |
| native `_native.astype(..., complex128)` | 0.645 us | 0.671 us |
| `ndarray(native astype)` | 0.744 us | 0.759 us |
| `z.astype(complex128)` | 1.067 us | 0.898 us |
| `z.astype(complex128, copy=False)` | n/a | 1.011 us |
| NumPy `z.astype(complex128)` | 0.543 us | 0.540 us |

Focused verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_astype_between_widths_matches_numpy tests/python/numpy_compat/test_complex.py::test_complex_astype_drops_imag_to_real_target tests/python/numpy_compat/test_array_coercion.py::test_astype_supported_cast_matrix_matches_numpy tests/python/numpy_compat/test_array_coercion.py::test_astype_copy_false_keeps_identity_for_same_dtype -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-astype-dtype-fastpath --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/casts/astype_complex64_to_complex128 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-astype-complex64-to-complex128-dtype-fastpath-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/casts/astype_complex64_to_complex128 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-astype-complex64-to-complex128-dtype-fastpath-numpy --sample --no-perf-stat
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python -m monpy._bench.mojo_sweep --format json --sort ratio --output-dir results/local-sweep-20260509-mojo-stdlib-refresh-0304 --no-stdout --timeout 300
```

Official post-fix benchmark results:

| row | before | after | delta |
| --- | ---: | ---: | ---: |
| `complex/casts/astype_complex64_to_complex128` | 3.511 us, 1.281x | 3.035 us, 1.178x | 1.16x faster |

The profile manifests reported `complex/casts/astype_complex64_to_complex128`
at 3.253 us/call for monpy and 2.774 us/call for NumPy. Monpy used backend code
0, max RSS was about 49.5 MB, and the traced peak was 1,598 bytes. NumPy's
traced peak was 33,992 bytes. No hardware PMU counters were collected because
`--no-perf-stat` was used; the `sample(1)` captures live under
`results/local-profile-20260509-astype-complex64-to-complex128-dtype-fastpath-*`.

The pure-Mojo stdlib sweep's largest candidate/stdlib ratio this pass was
`scalar_mul_f64_64k` at 1.149x, followed by `small_matmul_f32_8` at 1.080x.
That is worth a later kernel pass, but it is not the cause of this complex cast
row: the cast row moved by removing Python facade work while the native
interleaved-lane cast stayed effectively flat.

Remaining frontier after this slice:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.678 | 1.965 | 1.365x |
| `complex/interop/array_copy_complex128` | 3.030 | 2.467 | 1.237x |
| `complex/elementwise/binary_add_complex128` | 3.225 | 2.637 | 1.226x |
| `complex/elementwise/binary_add_complex64` | 2.970 | 2.463 | 1.213x |
| `complex/elementwise/binary_mul_complex64` | 3.070 | 2.557 | 1.193x |
| `complex/casts/astype_complex64_to_complex128` | 3.035 | 2.576 | 1.178x |
| `complex/views/reversed_add_complex64` | 3.525 | 3.011 | 1.172x |

Next target should move off the complex view/cast rows and either revisit
interop copy/view facade costs or investigate the pure-Mojo `scalar_mul_f64_64k`
stdlib deficit as a separate kernel-level pass.

### 2026-05-09 direct static typed elementwise ops

The next refresh split the frontier into two different problems. The public
complex rows were still led by facade and complex-buffer cases:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.667 | 1.951 | 1.370x |
| `complex/interop/array_copy_complex128` | 3.087 | 2.427 | 1.268x |
| `complex/elementwise/binary_add_complex64` | 2.975 | 2.418 | 1.229x |
| `complex/elementwise/binary_add_complex128` | 3.234 | 2.643 | 1.224x |
| `complex/elementwise/binary_mul_complex64` | 3.072 | 2.531 | 1.206x |

The pure-Mojo stdlib comparison exposed a cleaner kernel-level miss:

| row | candidate | stdlib | ratio |
| --- | --------: | -----: | ----: |
| `elementwise/add_f64_64k` | 13.79 us | 12.30 us | 1.121x |
| `reductions/prod_f32_64k` | 3.24 us | 3.07 us | 1.057x |
| `elementwise/sin_f64_1k` | 2.58 us | 2.48 us | 1.040x |

The static typed binary kernels were still going through
`apply_binary_typed_vec_static` inside the SIMD loop. Even though `op` is a
comptime parameter, the stdlib baseline was spelling the ADD loop directly:
`load[width]`, add, `store`. The kernel now emits direct SIMD loops for static
ADD/SUB/MUL/DIV in `binary_same_shape_contig_typed_static`, plus direct scalar
ADD/MUL loops in `binary_scalar_contig_typed_static` where operand order does
not affect the result. The generic helper remains for the wider binary-op set.

Focused verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/elementwise/kernels/typed.mojo
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python -m monpy._bench.mojo_sweep --format json --sort ratio --output-dir results/local-sweep-20260509-mojo-static-direct-0324 --no-stdout --timeout 300
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_numeric.py::test_broadcasted_binary_ops_match_numpy tests/python/numpy_compat/test_numeric.py::test_scalar_binary_ops_match_numpy tests/python/numpy_compat/test_numeric.py::test_binary_out_writes_existing_destination tests/python/numpy_compat/test_ufunc.py::test_power_scalar_square_and_cube_match_numpy tests/python/numpy_compat/test_complex.py::test_complex_arithmetic_add_sub_mul_div_match_numpy -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-static-direct-0324 --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types array --loops 50 --repeats 7 --rounds 5 --vector-sizes 65536 --matrix-sizes 64 --linalg-sizes 8 --format json --sort ratio --output-dir results/local-sweep-20260509-array-static-direct-0324 --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/elementwise/binary_add_complex64 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-binary-add-complex64-static-direct-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/elementwise/binary_add_complex64 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-binary-add-complex64-static-direct-numpy --sample --no-perf-stat
```

Post-fix pure-Mojo stdlib result:

| row | before | after | delta |
| --- | -----: | ----: | ----: |
| `elementwise/add_f64_64k` | 13.79 us vs 12.30 us, 1.121x | 16.21 us vs 16.18 us, 1.002x | ratio gap removed |

The absolute nanoseconds moved with machine noise on this run, but the
candidate and stdlib paths now track each other at the row level. This is the
right signal for this harness because both sides are measured back-to-back in
the same generated Mojo binary.

Public complex refresh after rebuilding:

| row | before | after | delta |
| --- | -----: | ----: | ----: |
| `complex/elementwise/binary_add_complex64` | 2.975 us vs 2.418 us, 1.229x | 2.988 us vs 2.484 us, 1.201x | 1.02x ratio improvement |
| `complex/interop/array_copy_complex128` | 3.087 us vs 2.427 us, 1.268x | 3.057 us vs 2.473 us, 1.235x | 1.03x ratio improvement |

The public array slice did not show a real-valued elementwise fallout. The
bandwidth-size `array/bandwidth/binary_add_65536_f32` row stayed comfortably
ahead of NumPy at 6.930 us vs 11.633 us, 0.607x. The wrapper-size
`array/elementwise/binary_add_f32` row was 3.036 us vs 2.407 us, 1.263x, which
keeps the next public array target in facade/orchestration territory rather
than this SIMD loop.

The `complex/elementwise/binary_add_complex64` profiles reported 3.146 us/call
for monpy and 2.616 us/call for NumPy. Monpy used backend code 2, the fused
native path, with max RSS about 49.3 MB and a traced allocation peak of 1,598
bytes. NumPy's traced allocation peak was 17,608 bytes. No PMU counters were
collected because the profile command used `--no-perf-stat`; the `sample(1)`
captures live under
`results/local-profile-20260509-binary-add-complex64-static-direct-*`.

Remaining frontier after this slice:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.671 | 1.974 | 1.354x |
| `complex/interop/array_copy_complex128` | 3.057 | 2.473 | 1.235x |
| `complex/elementwise/binary_add_complex128` | 3.239 | 2.649 | 1.232x |
| `complex/elementwise/binary_add_complex64` | 2.988 | 2.484 | 1.201x |
| `complex/elementwise/binary_mul_complex64` | 3.080 | 2.567 | 1.197x |

Next target should either specialize complex128 add/mul at the fused-kernel
level, or attack the `asarray_complex64` facade only if a profile shows more
than wrapper creation and dtype lookup left to remove.

### 2026-05-09 specialized complex64 zero-copy buffer view

The next live refresh put `asarray_complex64` back at the top of the public
complex frontier:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 3.063 | 2.123 | 1.443x |
| `complex/interop/array_copy_complex128` | 3.378 | 2.814 | 1.251x |
| `complex/elementwise/binary_add_complex64` | 3.270 | 2.647 | 1.228x |
| `complex/elementwise/binary_mul_complex64` | 3.345 | 2.753 | 1.212x |
| `complex/elementwise/binary_add_complex128` | 3.519 | 2.866 | 1.209x |

The same pass refreshed the pure-Mojo stdlib comparison. The largest
candidate/stdlib ratio was `elementwise/scalar_mul_f32_64k` at 1.095x, followed
by `small_matmul_f32_8` at 1.086x. The scalar-MUL source already spells the
direct SIMD multiply in the static kernel, so this pass targeted the public
interop row where the generic buffer bridge was still doing tri-state
copy/cast bookkeeping for an exact zero-copy complex64 view.

`asarray_complex64_view_from_buffer_ops` now owns a narrow fast path:

- one `PyObject_GetBuffer(..., PyBUF_RECORDS_RO)` call;
- direct itemsize/format validation for complex64 (`F` or NumPy's `Z`, 8
  bytes);
- readonly and shape/stride validation;
- direct `make_external_array(COMPLEX64, ...)`;
- guaranteed `PyBuffer_Release` before returning or raising on validation
  failures.

This avoids the generic `requested_code`/`copy_flag` branch tree, dtype decode,
`must_copy` calculation, and copy/cast fallback setup in the hot
`copy=False, dtype=complex64` path.

Focused verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/buffer.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_false_shares_storage tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_round_trip tests/python/test_buffer_core.py::test_asarray_from_writable_buffer_shares_storage tests/python/test_buffer_core.py::test_asarray_from_readonly_buffer_copies_by_default_and_rejects_copy_false tests/python/test_buffer_core.py::test_asarray_buffer_dtype_mismatch_copy_policy -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-c64-view-format-fastcheck-0344 --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/asarray_complex64 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-asarray-complex64-format-fastcheck-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/asarray_complex64 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-asarray-complex64-format-fastcheck-numpy --sample --no-perf-stat
```

Post-fix public complex result:

| row | before | after | delta |
| --- | -----: | ----: | ----: |
| `complex/interop/asarray_complex64` | 3.063 us vs 2.123 us, 1.443x | 2.681 us vs 1.973 us, 1.360x | 1.14x faster monpy side |

The final profiles reported `asarray_complex64` at 2.742 us/call for monpy and
2.062 us/call for NumPy. Monpy used backend code 0, as expected for a zero-copy
external view. Max RSS was about 49.2 MB for monpy and 49.3 MB for NumPy.
Traced allocation peaks were 1,598 bytes for monpy and 1,406 bytes for NumPy.
No hardware PMU counters were collected because the profile command used
`--no-perf-stat`; the `sample(1)` captures live under
`results/local-profile-20260509-asarray-complex64-format-fastcheck-*`.

Remaining frontier after this slice:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.681 | 1.973 | 1.360x |
| `complex/elementwise/binary_add_complex128` | 3.290 | 2.668 | 1.248x |
| `complex/interop/array_copy_complex128` | 3.070 | 2.478 | 1.247x |
| `complex/elementwise/binary_add_complex64` | 2.998 | 2.459 | 1.220x |
| `complex/elementwise/binary_mul_complex64` | 3.094 | 2.552 | 1.210x |

Next target should move back to complex128 add/copy. The remaining
`asarray_complex64` gap is now mostly the cost of allocating a monpy wrapper
around a borrowed Python buffer; there is less generic bridge code left to cut.

### 2026-05-09 specialized complex128 copy buffer path

The next refresh kept the public interop and complex128 rows at the top:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.657 | 1.961 | 1.355x |
| `complex/interop/array_copy_complex128` | 3.046 | 2.441 | 1.253x |
| `complex/elementwise/binary_add_complex128` | 3.221 | 2.619 | 1.228x |
| `complex/elementwise/binary_add_complex64` | 2.968 | 2.423 | 1.222x |
| `complex/elementwise/binary_mul_complex64` | 3.056 | 2.552 | 1.200x |

The pure-Mojo stdlib refresh was quieter than the public facade rows:
`small_matmul_f32_8` led at 1.073x, `sin_f64_1k` was 1.064x, and the previous
`scalar_mul_f32_64k` candidate had settled to 1.014x. That made the
complex128 copy bridge the better target.

`asarray_complex128_copy_from_buffer_ops` now has its own exact-copy path for
complex128 buffers. It skips the generic requested-dtype/copy-policy machinery,
validates complex128 format directly (`D` or NumPy's `Z`, 16 bytes), uses
`memcpy` for c-contiguous sources, and falls back to `copy_c_contiguous` for
strided sources. While testing the strided fallback, a real old bug showed up:
`copy_c_contiguous`'s scalar fallback was reading complex arrays through the
f64 accessor, which preserved only the real lane and zeroed the imaginary lane.
The shared copy fallback now copies complex64 and complex128 with their
dedicated interleaved real/imag accessors.

Focused verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/buffer.mojo src/array/cast.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_true_detaches_storage tests/python/numpy_compat/test_complex.py::test_complex_array_from_strided_numpy_copy_true_detaches_storage tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_round_trip tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_false_shares_storage tests/python/test_buffer_core.py::test_asarray_from_writable_buffer_shares_storage tests/python/test_buffer_core.py::test_asarray_buffer_dtype_mismatch_copy_policy -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-c128-copy-specialized-final-0404 --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/array_copy_complex128 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-array-copy-complex128-specialized-final-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/array_copy_complex128 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-array-copy-complex128-specialized-final-numpy --sample --no-perf-stat
```

Post-fix public complex result:

| row | before | after | delta |
| --- | -----: | ----: | ----: |
| `complex/interop/array_copy_complex128` | 3.046 us vs 2.441 us, 1.253x | 2.961 us vs 2.429 us, 1.220x | 1.03x faster monpy side |

The final profiles reported `array_copy_complex128` at 3.140 us/call for monpy
and 2.627 us/call for NumPy. Monpy used backend code 0, as expected for a copy
bridge. Max RSS was about 49.3 MB for both candidates. Traced allocation peaks
were 1,598 bytes for monpy and 33,992 bytes for NumPy. No hardware PMU counters
were collected because the profile command used `--no-perf-stat`; the
`sample(1)` captures live under
`results/local-profile-20260509-array-copy-complex128-specialized-final-*`.

Remaining frontier after this slice:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.660 | 1.939 | 1.359x |
| `complex/elementwise/binary_add_complex128` | 3.221 | 2.595 | 1.241x |
| `complex/interop/array_copy_complex128` | 2.961 | 2.429 | 1.220x |
| `complex/elementwise/binary_add_complex64` | 2.948 | 2.424 | 1.216x |
| `complex/elementwise/binary_mul_complex64` | 3.057 | 2.523 | 1.204x |

Next target should be the complex128 ADD fused path. The copy bridge has less
generic machinery left, and the live add row now sits above it.

### 2026-05-09 Float32 softmax row kernel

The first refresh for this heartbeat checked both the public complex frontier
and the attention slice. The complex rows were still slower than NumPy, led by
`asarray_complex64` and the complex add/copy cluster:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.653 | 1.945 | 1.363x |
| `complex/elementwise/binary_add_complex64` | 2.977 | 2.444 | 1.230x |
| `complex/elementwise/binary_add_complex128` | 3.198 | 2.607 | 1.223x |
| `complex/elementwise/binary_mul_complex64` | 3.042 | 2.512 | 1.217x |
| `complex/interop/array_copy_complex128` | 2.952 | 2.443 | 1.216x |

The attention rows, however, were all already ahead of NumPy:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `attention/attention/causal_attention_t32_d32_f32` | 20.681 | 22.056 | 0.931x |
| `attention/softmax/causal_scores_t32_f32` | 8.506 | 10.500 | 0.807x |
| `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32` | 81.256 | 105.450 | 0.777x |

The useful attention miss was still local: the f32 softmax kernels were doing
row max, exp, denominator accumulation, and normalization through `Float64`.
The local Mojo stdlib `std.math.exp` overload returns the same floating type it
receives, so the float32 row kernel can stay in `Float32` end to end. This
matches NumPy's f32 softmax path more closely and removes two conversions per
lane in the hot row loops.

`src/elementwise/kernels/nn.mojo` now has narrow Float32 paths for plain
last-axis softmax and scaled masked last-axis softmax. Float64 keeps the
previous accumulation path.

Focused verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/elementwise/kernels/nn.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_numeric.py::test_fused_softmax_matches_numpy_formula tests/python/numpy_compat/test_numeric.py::test_fused_scaled_masked_softmax_matches_numpy_formula tests/python/numpy_compat/test_numeric.py::test_fused_layer_norm_matches_numpy_formula -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types attention --loops 20 --repeats 7 --rounds 5 --vector-size 1024 --matrix-sizes 32 --format json --sort ratio --output-dir results/local-sweep-20260509-attention-f32-softmax --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case attention/softmax/causal_scores_t32_f32 --types attention --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-attention-softmax-f32-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case attention/softmax/causal_scores_t32_f32 --types attention --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-attention-softmax-f32-numpy --sample --no-perf-stat
```

Post-fix attention result:

| row | before | after | delta |
| --- | -----: | ----: | ----: |
| `attention/softmax/causal_scores_t32_f32` | 8.506 us vs 10.500 us, 0.807x | 6.869 us vs 10.052 us, 0.683x | 1.24x faster monpy side |
| `attention/attention/causal_attention_t32_d32_f32` | 20.681 us vs 22.056 us, 0.931x | 19.256 us vs 22.242 us, 0.864x | 1.07x faster monpy side |
| `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32` | 81.256 us vs 105.450 us, 0.777x | 80.921 us vs 108.756 us, 0.745x | 1.00x wall-clock, better ratio |

The profile manifests reported `attention/softmax/causal_scores_t32_f32` at
6.978 us/call for monpy and 10.278 us/call for NumPy. Monpy used backend code
2, the fused native path. Max RSS was about 51.7 MB for monpy and 51.5 MB for
NumPy. Traced allocation peaks were 1,694 bytes for monpy and 23,400 bytes for
NumPy. The monpy `sample(1)` call graph put 1,033 of 2,289 samples directly in
`elementwise::kernels::nn::_softmax_last_axis_f32`, while NumPy's sample showed
most softmax time in ufunc reduction machinery (`PyUFunc_Reduce`,
`FLOAT_maximum`, and `FLOAT_pairwise_sum`). No PMU counters were collected
because the profile command used `--no-perf-stat`; the `sample(1)` captures live
under `results/local-profile-20260509-attention-softmax-f32-*`.

Next target should return to complex128 add/mul, or add a larger attention
matrix-size row so the attention benchmark can expose when softmax stops being
wrapper-sized and starts becoming cache/bandwidth sized.

### 2026-05-09 static unary typed kernels

The next refresh split the frontier again. The public NumPy-facing complex
slice was still led by small wrapper/facade rows:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `complex/interop/asarray_complex64` | 2.880 | 2.129 | 1.348x |
| `complex/elementwise/binary_add_complex64` | 3.306 | 2.512 | 1.258x |
| `complex/elementwise/binary_add_complex128` | 3.249 | 2.969 | 1.223x |
| `complex/interop/array_copy_complex128` | 3.390 | 2.661 | 1.207x |
| `complex/elementwise/binary_mul_complex64` | 3.075 | 2.550 | 1.207x |

The attention slice stayed ahead of NumPy after the Float32 softmax patch:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `attention/attention/causal_attention_t32_d32_f32` | 20.904 | 22.658 | 0.887x |
| `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32` | 81.802 | 107.356 | 0.750x |
| `attention/softmax/causal_scores_t32_f32` | 7.181 | 10.331 | 0.694x |

The pure-Mojo stdlib comparison exposed the real leaf-level miss:

| row | candidate | stdlib | ratio |
| --- | --------: | -----: | ----: |
| `elementwise/sin_f64_64k` | 394342 ns | 187974 ns | 2.098x |
| `elementwise/add_f64_1k` | 222.7 ns | 185.0 ns | 1.204x |
| `matmul/small_matmul_f32_16` | 452.9 ns | 401.5 ns | 1.128x |

`unary_contig_typed` was still calling `apply_unary_typed_vec(..., op)` inside
the vector loop. For common unary ops that means each SIMD chunk entered the
runtime `op` dispatcher before reaching `sin`, `cos`, `exp`, etc. The stdlib
baseline spells the operation directly as `std_sin(load[width])`. The typed
kernel now mirrors the binary-kernel pattern: `try_unary_contig_typed_static`
routes common ops (`SIN`, `COS`, `EXP`, `LOG`, `TANH`, `SQRT`, `NEGATE`,
`POSITIVE`, `SQUARE`) into a comptime-`op` static loop, removing the per-vector
runtime branch.

Focused verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/elementwise/kernels/typed.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_umath.py::test_unary_math_matches_numpy_float_dtypes tests/python/numpy_compat/test_umath.py::test_unary_math_preserves_float32_result_dtype tests/python/numpy_compat/test_umath.py::test_unary_math_is_a_full_numpy_ufunc tests/python/numpy_compat/test_numeric.py::test_fused_sin_add_mul_matches_numpy tests/python/numpy_compat/test_numeric.py::test_numpy_shaped_expression_lowers_to_fused_kernel -q
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python -m monpy._bench.mojo_sweep --format json --sort ratio --output-dir results/local-sweep-20260509-mojo-unary-static-0444 --no-stdout --timeout 300
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types array --loops 50 --repeats 7 --rounds 5 --vector-sizes 65536 --matrix-sizes 64 --linalg-sizes 8 --format json --sort ratio --output-dir results/local-sweep-20260509-array-unary-static-0444 --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case array/bandwidth/unary_sin_65536_f32 --types array --vector-sizes 65536 --matrix-sizes 64 --linalg-sizes 8 --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-unary-sin-static-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case array/bandwidth/unary_sin_65536_f32 --types array --vector-sizes 65536 --matrix-sizes 64 --linalg-sizes 8 --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-unary-sin-static-numpy --sample --no-perf-stat
```

Post-fix pure-Mojo stdlib result:

| row | before | after | delta |
| --- | -----: | ----: | ----: |
| `elementwise/sin_f64_64k` | 394342 ns vs 187974 ns, 2.098x | 167833 ns vs 167000 ns, 1.005x | ratio gap removed |
| `elementwise/sin_f32_64k` | 118877 ns vs 116339 ns, 1.022x | below top deficits | no visible remaining gap |

The remaining stdlib frontier is much flatter: `scalar_mul_f64_64k` leads at
1.086x, `sum_f32_1k` is 1.033x, and `small_matmul_f32_16` is 1.029x. That is a
clean drop from a 2.10x top leaf to low-single-digit misses.

The public NumPy-facing array sweep still uses the macOS Accelerate path for
large f32 sine (`backend_code = 1` in profile), so this patch mainly closes the
pure Mojo leaf gap rather than the public facade row. The post-fix public
bandwidth row was still comfortably ahead of NumPy:

| row | monpy us | numpy us | ratio |
| --- | -------: | -------: | ----: |
| `array/bandwidth/unary_sin_65536_f32` | 32.139 | 101.894 | 0.313x |
| `array/bandwidth/fused_sin_add_mul_65536_f32` | 39.788 | 115.449 | 0.347x |
| `array/elementwise/unary_sin_f32` | 4.523 | 3.792 | 1.191x |

The sample profile for `array/bandwidth/unary_sin_65536_f32` reported monpy at
32.483 us/call and NumPy at 102.304 us/call. Monpy used backend code 1, the
Accelerate path, with a 1,656 byte traced allocation peak; NumPy peaked at
525,512 bytes. The sample call graph showed monpy spending the hot loop in
`VVSINF` through `create::ops::elementwise::unary_ops`, while NumPy spent the
hot path in `_multiarray_umath`'s `simd_sincos_f32` ufunc loop. No PMU counters
were collected because the profile command used `--no-perf-stat`; captures live
under `results/local-profile-20260509-unary-sin-static-*`.

Next target should be the new pure-Mojo `scalar_mul_f64_64k` stdlib gap or the
public complex64 add/asarray wrapper cluster. The former is a cleaner leaf
kernel target; the latter is the larger NumPy-facing ratio.

### 2026-05-09 scalar static commutative loop split

Refresh before editing showed attention still ahead of NumPy and the pure-Mojo
stdlib comparison led by scalar multiply:

| slice | row | candidate | baseline | ratio |
| --- | --- | --------: | -------: | ----: |
| attention | `attention/attention/causal_attention_t32_d32_f32` | 20.483 us | 23.750 us | 0.843x |
| attention | `attention/softmax/causal_scores_t32_f32` | 7.846 us | 11.367 us | 0.690x |
| attention | `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32` | 71.900 us | 113.706 us | 0.619x |
| pure Mojo | `elementwise/scalar_mul_f64_64k` | 11843.5 ns | 10464.7 ns | 1.132x |
| pure Mojo | `elementwise/add_f32_1m` | 131810 ns | 120804 ns | 1.091x |

`binary_scalar_contig_typed_static` already specialized ADD/MUL with comptime
branches, but those branches lived inside the shared scalar-left-aware kernel
body. The stdlib scalar-multiply baseline is just a direct SIMD load, multiply,
store, tail loop. I split ADD and MUL into compile-time early-return loops so
the commutative hot path has no `scalar_on_left` branch or fallback
`apply_binary_typed_vec_static` scaffolding in its body.

Focused verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/elementwise/kernels/typed.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_numeric.py::test_scalar_binary_ops_match_numpy tests/python/numpy_compat/test_numeric.py::test_python_scalars_are_weak_for_float32_arrays tests/python/numpy_compat/test_numeric.py::test_binary_out_writes_existing_destination tests/python/numpy_compat/test_ufunc.py::test_floor_divide_and_remainder_match_numpy tests/python/numpy_compat/test_ufunc.py::test_power_scalar_square_and_cube_match_numpy -q
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python -m monpy._bench.mojo_sweep --format json --sort ratio --output-dir results/local-sweep-20260509-scalar-static-0553 --no-stdout --timeout 300
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python -m monpy._bench.mojo_sweep --format json --sort ratio --output-dir results/local-sweep-20260509-scalar-static-rerun-0553 --no-stdout --timeout 300
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types array --loops 50 --repeats 7 --rounds 5 --vector-sizes 65536 --matrix-sizes 64 --linalg-sizes 8 --format json --sort ratio --output-dir results/local-sweep-20260509-array-scalar-static-0553 --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case array/bandwidth/binary_add_65536_f32 --types array --vector-sizes 65536 --matrix-sizes 64 --linalg-sizes 8 --candidate monpy --duration 2 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-binary-add-static-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case array/bandwidth/binary_add_65536_f32 --types array --vector-sizes 65536 --matrix-sizes 64 --linalg-sizes 8 --candidate numpy --duration 2 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-binary-add-static-numpy --sample --no-perf-stat
```

The first post-patch pure-Mojo sweep moved `scalar_mul_f64_64k` from 1.132x to
0.987x. The rerun put it at 1.002x, so the practical conclusion is "closed to
parity" rather than "reliably faster":

| row | before | after | rerun |
| --- | -----: | ----: | ----: |
| `elementwise/scalar_mul_f64_64k` | 11843.5 ns vs 10464.7 ns, 1.132x | 10527.7 ns vs 10663.8 ns, 0.987x | 11772.2 ns vs 11753.8 ns, 1.002x |
| `elementwise/scalar_mul_f32_64k` | 5932.2 ns vs 5831.3 ns, 1.017x | 5583.4 ns vs 5503.7 ns, 1.014x | 5857.9 ns vs 5544.1 ns, 1.057x |

One sweep produced a false-looking `sum_f32_1m` spike at 4.965x; the immediate
rerun put the same row back at 0.994x. That row is unrelated to the scalar patch
and should be treated as benchmark noise unless it reproduces in a third run
with a longer minimum runtime.

The NumPy-facing guardrail stayed healthy for the bandwidth row:

| row | monpy | NumPy | ratio |
| --- | ----: | ----: | ----: |
| `array/bandwidth/binary_add_65536_f32` | 7.334 us | 11.278 us | 0.670x |
| `array/bandwidth/reversed_add_65536_f32` | 13.884 us | 30.888 us | 0.450x |
| `array/bandwidth/fused_sin_add_mul_65536_f32` | 39.779 us | 116.539 us | 0.342x |

The `sample(1)` CPU profile for `array/bandwidth/binary_add_65536_f32` showed
monpy spending the main hot path under `maybe_binary_same_shape_contiguous`
inside `libvDSP`, while NumPy spent the hot loop in `_multiarray_umath`
`FLOAT_add` plus allocation and ufunc dispatch. Wall-clock profile measurement
was 7.932 us/call for monpy versus 11.433 us/call for NumPy. Tracemalloc peaks
were 1,598 bytes for monpy and 525,512 bytes for NumPy, a 329:1 peak-allocation
ratio. No PMU counters were collected on macOS because this run used
`--no-perf-stat`; the stack samples and manifests live under
`results/local-profile-20260509-binary-add-static-*`.

Next target should be either the pure-Mojo `add_f32_64k`/`add_f32_1m` variance
with a longer stdlib harness runtime, or the larger public wrapper cluster
(`moveaxis_f32`, `empty_like_shape_override_f32`, `transpose_add_f32`) where the
NumPy-facing ratios are still 1.7x-1.9x.
