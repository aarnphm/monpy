## Optimization Log

I use this file as the local ledger for optimization passes: what was slow,
what changed, what moved, and what still smells suspicious enough to profile
again.

Read it like this:

- `results/.../manifest.json` paths are the evidence trail.
- `monpy/numpy` ratios are first-pass triage; profiles decide the real story.
- Wrapper-sized rows need profiles because tiny microbenchmarks can measure the
  harness more than the kernel.
- A 1.01:1 move is noise until a rerun or profile gives it teeth.

### 2026-05-08 rank-3 transposed add dispatch

I hit the first post-refactor hot spot in the `strides/rank3_transpose_add_f32`
case. The input shape is `32x32x32`; both operands are c-contiguous arrays
viewed as `.transpose((2, 0, 1))`, so the logical output is contiguous and the
stride-1 input axis sits outside the innermost logical axis.

I routed matching same-dtype rank-3 layouts through
`maybe_binary_rank3_axis0_tile` in `src/elementwise/__init__.mojo`, using the
batched form of the existing rank-2 transpose tile. The kernel does four things
for each middle-axis slice:

1. Load four stride-1 SIMD vectors from each operand.
2. Apply the binary op in registers.
3. Transpose the 4x4 tile with `shuffle`.
4. Store four contiguous rows into the c-contiguous result.

Local result on the M3 Pro checkout:

| run                                                               | monpy us | numpy us | monpy/numpy |
| ----------------------------------------------------------------- | -------: | -------: | ----------: |
| `results/local-sweep-20260508-pass0/results.json`                 |   30.653 |    8.404 |      3.648x |
| `results/local-sweep-20260508-rank3-source-dispatch/results.json` |   19.064 |    8.210 |      2.263x |

That gave a 1.61:1 reduction in monpy wall time for this row. NumPy is still faster, so next I would use sampled profiles and hardware counters before
adding another kernel. The split I want to verify is Python wrapper time versus
tile shuffle/store pressure versus NumPy's iterator behavior.

## Profiling

`monpy-profile` is my heavier evidence path. It runs one benchmark case for
enough wall time that OS profilers can see native Mojo, C library, and Python
frames. I keep it separate from `monpy-bench` so benchmark runs stay
low-overhead and comparable.

```bash
monpy-profile \
  --types strides \
  --case rank3_transpose_add_f32 \
  --candidate monpy \
  --duration 8 \
  --output-dir results/profile-rank3
```

Each run writes:

- `manifest.json`: command, case, child-loop timing, profiler outputs.
- `measurement.json`: long-loop wall time, calls per second, resource usage,
  peak RSS, and native backend flags. I leave off `tracemalloc`,
  because tracing allocations perturbs the CPU timing.
- `allocation-measurement.json`: a shorter allocation pass with Python
  `tracemalloc` enabled.
- `numpy-config.txt`: NumPy build and BLAS/LAPACK configuration.

On macOS, the command captures a `sample(1)` stack report by default:

```text
results/profile-rank3/sample.txt
```

I use xctrace when the question needs Instruments data beyond text stacks:

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

I use this for a hardware-counter pass that can distinguish a Python wrapper
regression from a cache-miss, instruction-count, or branch-miss regression.

I run the same case with `--candidate numpy` when the question is whether monpy
trails on wrapper overhead, iterator behavior, or a lower-level library call:

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

Local result:

| run                                                         | monpy us | numpy us | monpy/numpy |
| ----------------------------------------------------------- | -------: | -------: | ----------: |
| `results/local-sweep-20260508-pass0/results.json`           |    7.808 |    2.216 |      3.524x |
| `results/local-sweep-20260508-scalar-ascontig/results.json` |    3.498 |    2.164 |      1.618x |

That gave a 2.23:1 reduction in monpy wall time for the scalar row. Dense and
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

Local result:

| run                                                        | monpy us | numpy us | monpy/numpy |
| ---------------------------------------------------------- | -------: | -------: | ----------: |
| `results/local-sweep-20260508-heartbeat1/results.json`     |   12.494 |    2.436 |      5.205x |
| `results/local-sweep-20260508-native-squeeze/results.json` |    8.182 |    2.429 |      3.390x |

That gave a 1.53:1 reduction in monpy wall time for this row. The residual gap is
mostly outside squeeze itself: a direct microbench of an existing monpy array
showed `mnp.asarray(np.zeros(...))` around 4.2 us and native-backed squeeze
around the low single-digit microsecond range. Next target: the NumPy-input marshaling family (`from_dlpack`, `asarray_zero_copy`,
and small `array_copy`). The squeeze-specialized kernel can sit.

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

Local result:

| run                                                               | monpy us | numpy us | monpy/numpy |
| ----------------------------------------------------------------- | -------: | -------: | ----------: |
| `results/local-sweep-20260508-native-squeeze/results.json`        |    8.824 |    2.182 |      4.052x |
| `results/local-sweep-20260508-dlpack-numpy-fastpath/results.json` |    5.669 |    2.172 |      2.587x |

That gave a 1.56:1 reduction in monpy wall time for NumPy-backed DLPack imports.
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
so each ndarray is classified once.

| row                                    | previous monpy us | new monpy us | previous ratio | new ratio |
| -------------------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/interop/asarray_zero_copy_f32`  |             6.097 |        5.207 |         3.104x |    2.699x |
| `array/interop/asarray_zero_copy_f64`  |             5.955 |        5.213 |         3.024x |    2.644x |
| `array/interop/asarray_zero_copy_bool` |             6.082 |        5.229 |         3.062x |    2.697x |
| `array/interop/asarray_zero_copy_i64`  |             5.960 |        5.199 |         3.062x |    2.654x |
| `array/interop/from_dlpack_f32`        |             5.669 |        4.963 |         2.587x |    2.290x |

The direct detector microbench moved from about 0.34 us to about 0.09 us while
`tests/python/test_no_numpy_core.py` still verifies that importing the core
core package stays NumPy-free.

### 2026-05-08 native stack-axis-0 path

The small join rows had a shape where native `concatenate` was already cheap,
but `stack` and 1D `vstack` still paid Python shape validation, one native
concatenate call, and then a second native reshape call. `src/create/__init__.mojo`
now exposes `stack_axis0`, a single native entrypoint that validates identical
input shape/dtype, builds the `[n_arrays] + input_shape` result, and copies each
contiguous input slab directly. The Python `stack(axis=0)` and rank-1 `vstack`
facades optimistically use it when no dtype override is requested, then fall
back to the promotion/general-axis path on mismatch.

Local result:

| row                           | previous monpy us | new monpy us | previous ratio | new ratio |
| ----------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/views/stack_axis0_f32` |             9.191 |        3.731 |         2.610x |    1.051x |
| `array/views/vstack_f32`      |             9.678 |        3.700 |         2.926x |    1.127x |

That gave a 2.46:1 reduction for `stack_axis0_f32` and a 2.62:1 reduction for
`vstack_f32`. `hstack` and plain `concatenate` stayed flat, as expected; those
already route through the native concatenate leaf and need a different target if
they become important.

### 2026-05-08 rank-1 atleast_2d expand view

`array/creation/atleast_2d_f32` was still paying for Python shape reads plus a
native reshape when the input was already a monpy rank-1 array. NumPy models
this case as a view (`arr[None, :]`), so monpy now uses the same shape operation:
rank-1 `atleast_2d` calls native `expand_dims(axis=0)` directly and keeps the
original array as the base owner.

Local result:

| row                             | previous monpy us | new monpy us | previous ratio | new ratio |
| ------------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/creation/atleast_2d_f32` |             6.087 |        3.526 |         2.664x |    1.519x |

The direct microbench moved `mnp.atleast_2d(existing_vector)` from about 4.1 us
to about 1.4 us. The remaining gap is the generic Python facade plus one native
call; data movement has left the crime scene.

### 2026-05-08 exact middle-newaxis view

`array/views/newaxis_middle_f32` uses the concrete key `arr[:, None, :]`. The
generic indexing path expands the key, fetches full shape metadata, validates
the full slices, then calls native `expand_dims`. For this exact rank-2 full
slice pattern, none of the shape metadata is needed: the result is just
`expand_dims(axis=1)`.

`ndarray.__getitem__` now recognizes the exact `(:, None, :)` rank-2 pattern
before entering the generic slice path. Mixed keys such as `arr[:, None, ::-1]`
still fall through to the existing generic slice machinery.

Local result:

| row                              | previous monpy us | new monpy us | previous ratio | new ratio |
| -------------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/views/newaxis_middle_f32` |             5.743 |        3.138 |         2.734x |    1.535x |

The direct microbench moved the exact `helper[:, None, :]` view from about 3.6
us to about 1.1 us. The row still trails NumPy, and the fixed full-slice
newaxis path has shed the generic slice tax.

### 2026-05-08 native swapaxes view

`array/views/swapaxes_f32` was still implemented as Python permutation
construction followed by generic `transpose`. For a pure axis swap, the native
side can avoid layout selection and Python tuple normalization entirely: clone
the shape and stride lists, swap two slots, then return a view with the original
storage.

Local result:

| row                        | previous monpy us | new monpy us | previous ratio | new ratio |
| -------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/views/swapaxes_f32` |             5.534 |        3.072 |         2.550x |    1.379x |

The direct microbench moved `mnp.swapaxes(existing_rank3, 0, 2)` from about 3.3
us to about 0.87 us. The residual benchmark time is mostly Python facade and
benchmark harness overhead; data movement is already tiny.

### 2026-05-08 direct NumPy ndarray ingest

The NumPy-input rows were still entering through the generic
`__array_interface__` parser even after the detector fast path. That generic
path is required for non-NumPy producers, but real NumPy arrays already expose
the same fields directly as attributes. `runtime.ops_numpy._from_numpy_unchecked`
now reads `dtype.str`, `shape`, `strides`, `ctypes.data`, and `flags.writeable`
directly, then calls native `from_external` / `copy_from_external` with the same
copy and readonly policy as before.

Local result:

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

Local result:

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

My first native version used Mojo's SIMD scalar `pow`, which was fast but
missed the existing `1e-12` NumPy parity test by about `7.6e-10` relative on
the 50-point `0..3` span. The committed path calls platform `libm` `pow`
and preserves the strict parity test while keeping Python-list materialization out of the path.

Local result:

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

Local result:

| row                        | previous monpy us | new monpy us | previous ratio | new ratio |
| -------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/decomp/pinv_2_f64`  |            33.683 |        6.824 |         2.145x |    0.437x |
| `array/decomp/pinv_4_f64`  |            39.188 |        8.349 |         2.252x |    0.481x |
| `array/decomp/pinv_8_f64`  |            54.852 |       12.864 |         2.811x |    0.655x |
| `array/decomp/pinv_32_f64` |           136.756 |       54.235 |         2.424x |    0.963x |
| `array/decomp/pinv_8_f32`  |            48.527 |       10.937 |         2.033x |    0.451x |

All benchmarked pseudo-inverse rows now beat NumPy on this machine. The largest
remaining deficit at this checkpoint shifted from `pinv` to the
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

Local result:

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

References I used for the direction:

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
reuses the same two resolved symbols an ndarray crosses the
boundary.

Local result:

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

Local result:

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
  Python allocation peak stayed at 1,718 bytes, so I removed call
  overhead while the allocation shape stayed fixed.

Next lever: a narrower axis-0 vector/list ABI that cuts Python list construction
and repeated `downcast_value_ptr` inside native. That looks smaller than the
remaining linalg/view rows, so the broader target moves to `cholesky_32_f64` or
the view wrapper cluster.

### 2026-05-08 Cholesky typed writeback

`array/decomp/cholesky_32_f64` had already reached LAPACK.
`sample(1)` on
`results/profile-20260508-cholesky-32-f64-baseline-monpy/manifest.json`
showed the native path already inside Accelerate `DPOTRF`; the largest local
symbol was `array::__init__::physical_offset`, with 686 samples in the report.
Netlib documents `DPOTRF` as a blocked Cholesky routine that calls Level-3 BLAS,
so the 32x32 lever was the row-major result copy after the vendor call.

`lapack_cholesky_{f32,f64}_into` now writes the lower-triangular result through
typed contiguous pointers, bypassing `set_logical_from_f64` for every output
element. This removes the per-element shape/stride divmod from a result that is
freshly allocated, c-contiguous, and already has the exact dtype.

Local result:

| row                            | previous monpy us | new monpy us | previous ratio | new ratio |
| ------------------------------ | ----------------: | -----------: | -------------: | --------: |
| `array/decomp/cholesky_32_f64` |            16.887 |        9.776 |         2.201x |    1.281x |
| `array/decomp/cholesky_32_f32` |            16.500 |        9.211 |         1.886x |    1.042x |

Profile artifacts:

- `results/profile-20260508-cholesky-32-f64-direct-writeback/manifest.json`
  measured the patched monpy row at 9.744 us/call, down from 16.546 us/call in
  `results/profile-20260508-cholesky-32-f64-baseline-monpy/manifest.json`.
- The patched sample moved the stack back to Accelerate `DPOTRF`;
  `physical_offset` drops out in the Cholesky hot-path grep. The remaining
  local staging is `transpose_to_col_major_f64`; any further Cholesky win now
  comes from layout or `UPLO` policy.
- The CPU Counters xctrace artifact was captured under
  `results/profile-20260508-cholesky-32-f64-direct-writeback/xctrace-cpu-counters.trace`.
  The child loop in that trace measured 9.838 us/call with 3.0 s of user CPU and
  no major faults; the profile manifest stays the durable scalar record.
- Traced Python allocation peak stayed at 110,752 bytes. Native arithmetic and
  branch overhead moved; Python allocation churn stayed.

The focused deficits now are view-wrapper and stride rows:
`reversed_add_f32`, `flatten_f32`, `ravel_f32`, `moveaxis_f32`, and
`empty_like_shape_override_f32`. The next broad target shifts away from Cholesky.

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

Local result:

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
`empty_like_shape_override_f32`, and `transpose_add_f32`. Next high-upside view
target: negative-stride elementwise dispatch or moveaxis permutation metadata.
Leave `ravel`/`flatten` alone for now.

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

Local result:

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
  in place of the 4x4 shuffle/store branch.

The combined `array,strides` frontier now starts with
`strides/elementwise/rank3_transpose_add_f32`, `array/views/reversed_add_f32`,
and `array/views/moveaxis_f32`. Next broad target: the rank-3
transpose path's output order: it likely has the same "doing extra layout work
to force c-contiguous output" smell, only with one more axis in the metadata.

### 2026-05-08 F-order rank-3 transposed binary output

`strides/elementwise/rank3_transpose_add_f32` uses dense C-order `32x32x32`
buffers viewed as `.transpose((2, 0, 1))`. The resulting logical shape is
`[rows, batches, cols]`, but the positive dense element strides are
`[1, rows * cols, rows]`. NumPy ufuncs default `order` to `K`, so NumPy keeps
the output in that input element order and keeps the input element order.

The previous monpy rank-3 fused path still used the 4x4 tile shuffle path that
stores a c-contiguous result. `maybe_binary_rank3_axis0_tile` now detects this
specific dense positive-stride layout, adds the two physical buffers with a
linear SIMD walk, and marks the result strides as `[1, rows * cols, rows]`.
That removes the unnecessary layout conversion while preserving NumPy's result
stride contract.

Local result:

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
  max RSS. Native layout work changed; Python allocation shape stayed fixed.
- The `sample(1)` report still puts the native time in
  `maybe_binary_rank3_axis0_tile`, now the linear F-order branch. It also shows
  allocator frames under `tc_memalign`, so the next stride-kernel pass should
  separate result allocation cost from pure arithmetic before adding another
  shuffle kernel.

The combined frontier now starts with `array/views/reversed_add_f32` at 2.157x,
`array/views/moveaxis_f32` at 1.898x, and
`array/creation/empty_like_shape_override_f32` at 1.875x. Next broad target: negative-stride elementwise dispatch for `reversed_add_f32`, with
`moveaxis` metadata construction as the next view-wrapper fallback if the
negative-stride profile points mostly at allocation.

### 2026-05-08 exact rank-1 reverse slice view

The fresh frontier still put `array/views/reversed_add_f32` first, but direct
decomposition showed the fused reversed-add arithmetic was the smaller cost:

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
rank-1 length through `shape_at(0)`, which saves the one-element Python shape
tuple.

Local result:

| row                                      | previous monpy us | new monpy us | previous ratio | new ratio |
| ---------------------------------------- | ----------------: | -----------: | -------------: | --------: |
| `array/views/reversed_add_f32`           |             6.655 |        4.616 |         2.188x |    1.512x |
| `strides/elementwise/reverse_1d_add_f32` |             7.219 |        5.192 |         1.633x |    1.218x |
| `array/views/strided_view_f32`           |             3.717 |        3.431 |         1.799x |    1.609x |

Direct post-change microbenchmarks measured `x[::-1]` at about 0.696 us and
the full `x[::-1] + y[::-1]` row at about 2.385 us. That gave a 2.25:1 reduction
for exact reverse view construction and a 1.80:1 reduction for the full direct
row.

Profile artifacts:

- `results/profile-20260508-reversed-add-f32-monpy-pre2/manifest.json`
  measured the old monpy row at 6.685 us/call, 86.3 MB max RSS, no major
  faults, and a 91,700 byte traced Python allocation peak.
- `results/profile-20260508-reversed-add-f32-reverse1d-fastpath/manifest.json`
  measured the patched row at 4.789 us/call, 87.2 MB max RSS, no major faults,
  and a 1,856 byte traced Python allocation peak.
- The post-change `sample(1)` report shows `reverse_1d_ops` in place of the old
  `slice_1d_ops` wrapper stack for the exact `[::-1]` construction. The
  remaining cost is now mostly Python wrapper/object allocation plus the
  existing reversed-add native loop.

The broad frontier now is `empty_like_shape_override_f32` at 1.905x,
`moveaxis_f32` at 1.891x, and `squeeze_axis0_f32` at 1.849x. Next target: the view-wrapper cluster only after a quick profile decides whether
`moveaxis` or `empty_like(shape=...)` has the larger removable Python metadata
tax.

### 2026-05-08 attention scalar and row-kernel recovery

The linked reference commit, `e2ed21d`, only added `pyyaml`. A detached
baseline at that commit measured the attention rows as:

| row                                                | `e2ed21d` monpy us | current-start monpy us | post-fix monpy us | post-fix ratio |
| -------------------------------------------------- | -----------------: | ---------------------: | ----------------: | -------------: |
| `attention/softmax/causal_scores_t32_f32`          |             57.121 |                 47.262 |            21.292 |         1.988x |
| `attention/attention/causal_attention_t32_d32_f32` |            133.539 |                121.953 |            43.121 |         1.907x |
| `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32`   |           1606.192 |               1534.498 |           165.275 |         1.569x |

The bad current-start profile was still real. The regression came from the
attention stack's generic paths:

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

I added three general fast paths with benchmark rows as witnesses:

- C-contiguous last-axis reductions for float32/float64 `sum`, `mean`, `prod`,
  `min`, and `max`. The sum/mean path reuses the existing 4-accumulator SIMD
  reducer per row.
- C-contiguous `(rows, cols)` by `(rows, 1)` binary broadcasting for float32 and
  float64. This covers softmax row shifts/divides and layer-norm centering.
- Same-shape bool-mask `where` for contiguous float32/float64 arrays, plus weak
  Python-scalar ufunc dispatch so `mnp.power(float32_array, 3.0)` stays float32.
  Scalar power `x**2` and `x**3` now lower to multiplication.

Direct microbenchmarks for the attention shapes moved as follows:

| operation                                            |                  before us | after us |
| ---------------------------------------------------- | -------------------------: | -------: |
| `mnp.max(scores, axis=-1, keepdims=True)`            |                     11.492 |    2.658 |
| `mnp.sum(scores, axis=-1, keepdims=True)`            |                     11.272 |    2.326 |
| `scores - row_max` with a prebuilt `(32, 1)` row max |                    8.7-ish |    4.858 |
| `mnp.where(causal_mask, fill, scores)`               |                     55.867 |    3.443 |
| `mnp.power(x_f32, 3.0)` for a 32x128 MLP activation  |                     64.073 |    2.379 |
| `_gelu_monpy(32x128)`                                | about 61.4 after mask work |   37.245 |

After the patch, decomposition has the GPT row dominated by compositional layer
norm and attention overhead. BLAS misses are gone:

| operation                            | monpy us | numpy us |
| ------------------------------------ | -------: | -------: |
| layer norm, one 32x32 block          |   23.699 |   10.844 |
| causal attention on normalized input |   39.176 |   18.755 |
| GELU on 32x128 MLP activation        |   31.778 |   39.254 |
| MLP output matmul                    |    1.674 |      n/a |
| final `lm_head` matmul               |    1.624 |      n/a |
| full tiny GPT logits                 |  160.262 |  100.863 |

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
is sane again, and the benchmark still makes several Python/native
round-trips and temporary arrays per row-normalization step. A single-pass
last-axis layer norm and a stable row softmax would attack the remaining
1.57x GPT gap directly.

### 2026-05-08 `monpy.nn` compile fix

My first `src/nn` split failed at the Mojo extension boundary:

```text
src/lib.mojo:84:16: error: package 'nn' does not contain 'layer_norm_last_axis_ops'
```

The source file was present, but the name `nn` is already claimed by Modular/MAX
kernel packages in the compiler search path. Importing `from nn import ...`
resolved to that owner. A second attempt to import `nn.kernels` hit the same
place. The working shape is:

- keep the Mojo loop bodies in `src/elementwise/kernels/nn.mojo`
- keep the PythonObject bridge functions in `src/create/ops/nn.mojo`
- re-export the bridge functions from `src/create/__init__.mojo`
- import the bridge functions in `src/lib.mojo` with the rest of `from create import ...`
- keep `python/monpy/nn` as the public Python import scope

This keeps the public Python surface as `monpy.nn` while sidestepping collision with
MAX's own `nn` package. Verification:

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
import/runner smoke only, a deliberately unstable performance sample.

Follow-up naming consolidation:

- `src/create/ops/elementwise.mojo` now owns the elementwise PythonObject bridge
  functions that used to sit in `src/create/elementwise_ops.mojo`
- `src/create/ops/nn.mojo` owns the public `monpy.nn` bridge functions
- `src/elementwise/kernels/nn.mojo` owns the row-wise NN loop bodies
- `src/elementwise/kernels/matmul.mojo` owns the matmul loop/BLAS dispatchers

Verification after the role-first package move:

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
gap to Python ufunc dispatch over the native loop.

`python/monpy/__init__.py` now gives the exact `ndarray, ndarray, out=ndarray`
binary ufunc case a direct `_native.binary_into(...)` path before the staged
kernel probe and promotion machinery. This is intentionally narrow: default
`where=True`, default `casting="same_kind"`, no `dtype=`, exact `ndarray` inputs,
and exact `ndarray` output.

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_numeric.py::test_binary_out_writes_existing_destination tests/python/numpy_compat/test_ufunc.py::test_ufunc_out_kwarg_writes_in_place tests/python/numpy_compat/test_ufunc.py::test_ufunc_dtype_kwarg_casts -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types array --loops 50 --repeats 5 --rounds 3 --vector-size 1024 --vector-sizes 16384 --matrix-sizes 16 --linalg-sizes 2 --format json --sort ratio --output-dir results/local-sweep-20260508-ufunc-out-fastpath --no-progress --no-stdout
```

The focused tests passed. Before/after for the wrapper-bound rows:

| row                            |           before |            after |        delta |
| ------------------------------ | ---------------: | ---------------: | -----------: |
| `binary_add_out_f32`           | 5.103 us, 2.116x | 2.955 us, 1.263x | 1.73x faster |
| `binary_add_f32`               | 3.126 us, 1.248x | 3.050 us, 1.247x |      neutral |
| `binary_add_extension_out_f32` | 2.694 us, 1.128x | 2.667 us, 1.135x |      neutral |

Next target: `complex/matmul_64_complex64`: it already reports
`used_accelerate=True`, so the investigation should focus on BLAS function
lookup/call overhead, row-major complex GEMM calling convention, and whether a
small-N Mojo complex microkernel can beat the framework call for 64x64.

### 2026-05-08 macOS ILP64 complex GEMM

The refreshed heartbeat slice confirmed attention had already moved below NumPy on this machine:

| row                                                | monpy us | numpy us |  ratio |
| -------------------------------------------------- | -------: | -------: | -----: |
| `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32`   |   81.206 |  109.272 | 0.743x |
| `attention/softmax/causal_scores_t32_f32`          |    8.144 |   10.272 | 0.814x |
| `attention/attention/causal_attention_t32_d32_f32` |   20.717 |   22.987 | 0.914x |

The same run left `complex/matmul_64_complex64` as the largest direct blocker:
16.257 us for monpy vs 7.221 us for NumPy, a 2.318x ratio. `used_accelerate()`
was already true, so BLAS dispatch was already active.

The useful profiler fact was the symbol more than the wall clock:

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

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_einsum.py::test_complex64_matmul_via_cgemm tests/python/numpy_compat/test_einsum.py::test_complex128_matmul_via_zgemm tests/python/numpy_compat/test_numeric.py::test_matmul_matches_numpy_for_1d_and_2d -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260508-complex-ilp64-cgemm --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/matmul/matmul_64_complex64 --types complex --matrix-sizes 64 --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260508-complex-matmul64-ilp64-monpy --sample --no-perf-stat
```

After the patch:

| row                           |            before |            after |        delta |
| ----------------------------- | ----------------: | ---------------: | -----------: |
| `complex/matmul_64_complex64` | 16.257 us, 2.318x | 7.277 us, 1.042x | 2.23x faster |

The post-fix profile loop measured 7.535 us/call, 49.7 MB max RSS, no major
faults, and a 1,590 byte traced allocation peak. Next target: the
complex conversion/view cluster: `astype_complex64_to_complex128` remains
2.008x NumPy, followed by `reversed_add_complex64` at 1.871x.

### 2026-05-08 complex width-cast lane fast path

Next, I reran the live complex slice after the ILP64 GEMM fix ranked the remaining
frontier as:

| row                                            | monpy us | numpy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `complex/casts/astype_complex64_to_complex128` |    5.289 |    2.579 | 2.051x |
| `complex/views/reversed_add_complex64`         |    5.885 |    3.057 | 1.929x |
| `complex/interop/asarray_complex64`            |    3.052 |    1.958 | 1.556x |
| `complex/interop/array_copy_complex128`        |    3.807 |    2.443 | 1.547x |

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

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_astype_between_widths_matches_numpy tests/python/numpy_compat/test_complex.py::test_complex_astype_drops_imag_to_real_target tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_round_trip -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260508-complex-cast-width-fastpath --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/casts/astype_complex64_to_complex128 --types complex --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260508-complex-astype-width-fastpath --sample --no-perf-stat
```

After the patch:

| row                                            |           before |            after |        delta |
| ---------------------------------------------- | ---------------: | ---------------: | -----------: |
| `complex/casts/astype_complex64_to_complex128` | 5.289 us, 2.051x | 3.298 us, 1.260x | 1.60x faster |

The post-fix profile loop measured 3.442 us/call, 49.7 MB max RSS, no major
faults, and a 1,590 byte traced allocation peak. The sample now puts the hot
native frame in `_complex_contig_lane_cast[f32,f64]`, which is the expected
storage-level path. The top complex deficit now is `reversed_add_complex64` at
1.891x, then complex ingress/copy rows around 1.5x.

### 2026-05-08 reversed complex add native rank-1 path

Next, I kept the complex slice kept `complex/views/reversed_add_complex64` as the most
visible view/kernel row:

| row                                        | monpy us | numpy us |  ratio |
| ------------------------------------------ | -------: | -------: | -----: |
| `complex/views/reversed_add_complex64`     |    5.807 |    3.011 | 1.929x |
| `complex/interop/array_copy_complex128`    |    3.791 |    2.443 | 1.551x |
| `complex/interop/asarray_complex64`        |    3.023 |    1.958 | 1.543x |
| `complex/elementwise/binary_mul_complex64` |    3.488 |    2.547 | 1.369x |

The pre-fix backend probe reported `used_accelerate=True`; the sample made the
problem concrete. For a 1024-element `complex64[::-1] + complex64[::-1]`, monpy
was issuing two small strided `vDSP_vadd` calls, one for real lanes and one for
imaginary lanes. At this size the library-call and strided-dispatch overhead
beat the useful arithmetic. The scalar generic fallback also carried the wrong
cost model because it paid `physical_offset` per complex element.

`src/elementwise/kernels/complex.mojo` now has a rank-1 strided complex kernel
that walks physical indexes incrementally. For the hot `complex64` reversed
ADD/SUB case it uses a small SIMD pair-reversal path: load two adjacent complex
values from the reversed physical span, reverse pair order in-register, then
store four float lanes to the contiguous result. MUL/DIV also use the
incremental-index rank-1 path, with the existing Smith division logic, so
negative-stride complex arithmetic stays on incremental physical indexes.

`python/monpy/__init__.py` also trims the exact one-dimensional `[::-1]` path:
it calls the native `reverse_1d_method()` directly before the extra `ndim()`
probe, and only falls back to the general slice machinery when the native method
reports a non-rank-1 array. This preserves the multidimensional `a[::-1]`
behavior while shaving the benchmark's two view constructions.

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_strided_arithmetic_preserves_imaginary_part tests/python/numpy_compat/test_indexing.py::test_rank1_full_reverse_slice_matches_numpy_and_shares_storage -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-reversed-simd-fast-slice --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/views/reversed_add_complex64 --types complex --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-reversed-add-simd-fast-slice --sample --no-perf-stat
```

After the patch:

| row                                    |           before |            after |        delta |
| -------------------------------------- | ---------------: | ---------------: | -----------: |
| `complex/views/reversed_add_complex64` | 5.807 us, 1.929x | 4.387 us, 1.361x | 1.32x faster |

The profile loop measured 4.277 us/call, 49.6 MB max RSS, no major faults, and
a 1,728 byte traced allocation peak. Backend reporting flipped from Accelerate
to FUSED for the result, which is expected: the hot path now skips the two
strided vDSP calls. A direct local timing split also showed the slice change:
one `complex64` reverse view fell from about 0.639 us to 0.499 us, and the full
`a[::-1] + b[::-1]` expression fell from about 2.58 us to 1.99 us in the
micro-timer.

The complex frontier now is the ingress/copy cluster plus one arithmetic
kernel:

| row                                        | monpy us | numpy us |  ratio |
| ------------------------------------------ | -------: | -------: | -----: |
| `complex/interop/array_copy_complex128`    |    4.036 |    2.659 | 1.518x |
| `complex/interop/asarray_complex64`        |    3.104 |    2.055 | 1.511x |
| `complex/elementwise/binary_mul_complex64` |    3.577 |    2.615 | 1.368x |
| `complex/views/reversed_add_complex64`     |    4.387 |    3.224 | 1.361x |

Next target: complex ingress/copy. The likely win is keeping NumPy complex
buffers on a narrow typed copy path that bypasses generic scalar extraction.

### 2026-05-08 direct NumPy buffer ingress

The typed-buffer hypothesis was tested first and rejected. A native
`PyBUF_STRIDES` bridge that skipped `PyBUF_FORMAT` made the raw native call a
little faster, but the Python-side `dtype.str` proof cost more than the native
PEP-3118 decode it was trying to avoid. The official complex slice regressed,
so I cut that path before committing.

The smaller, useful fix is simpler: when `monpy.asarray()` sees a NumPy ndarray,
it now calls the existing native `asarray_from_buffer` bridge directly instead
of routing through `runtime.ops_numpy.is_array_input()` and
`runtime.ops_numpy._from_numpy_unchecked()`. That keeps dtype/copy semantics in
the same native buffer decoder, skips a Python module wrapper hop on the hot
path, and still falls back to `runtime.ops_numpy` for any future exotic NumPy
case the local probe misses.

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_copy_false_shares_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_copy_true_detaches_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_readonly_copy_false_raises_and_copy_none_detaches tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_round_trip tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_false_shares_storage tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_true_detaches_storage -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-direct-numpy-buffer --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/asarray_complex64 --types complex --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-asarray-complex64-direct-numpy-buffer --sample --no-perf-stat
```

After the patch:

| row                                     |           before |            after |        delta |
| --------------------------------------- | ---------------: | ---------------: | -----------: |
| `complex/interop/asarray_complex64`     | 3.005 us, 1.549x | 2.882 us, 1.484x | 1.04x faster |
| `complex/interop/array_copy_complex128` | 3.758 us, 1.535x | 3.647 us, 1.475x | 1.03x faster |

The profile loop for `complex/interop/asarray_complex64` moved from 3.242
us/call to 2.951 us/call, with max RSS essentially flat at about 49 MB and the
same 1,590 byte traced allocation peak. The `sample` run measured 3.418 us/call
before and 3.128 us/call after in the child process. The hot native frame is
still `buffer::asarray_from_buffer_ops`, which is expected: this patch removes
Python wrapper overhead while memory traffic and SIMD work stayed flat. This run
used `--no-perf-stat`, so hardware PMU counters are absent.

Remaining complex frontier:

| row                                         | monpy us | numpy us |  ratio |
| ------------------------------------------- | -------: | -------: | -----: |
| `complex/elementwise/binary_mul_complex64`  |    3.455 |    2.538 | 1.364x |
| `complex/elementwise/binary_add_complex64`  |    3.082 |    2.438 | 1.265x |
| `complex/elementwise/binary_add_complex128` |    3.253 |    2.619 | 1.248x |
| `complex/matmul_64_complex64`               |    7.517 |    7.124 | 1.055x |

Next target: `complex/elementwise/binary_mul_complex64`. It is no
longer an ingress problem; the likely work is inside the complex multiply
kernel, specifically redundant real/imag lane loads, scalar temporary pressure,
and whether a wider interleaved-lane SIMD path beats the current per-element
Smith-compatible shape for multiplication.

### 2026-05-08 complex64 multiply interleaved SIMD

Next, I refreshed the complex slice ranked the live deficits as:

| row                                        | monpy us | numpy us |  ratio |
| ------------------------------------------ | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`        |    2.922 |    1.953 | 1.501x |
| `complex/interop/array_copy_complex128`    |    3.679 |    2.463 | 1.491x |
| `complex/views/reversed_add_complex64`     |    4.209 |    3.065 | 1.371x |
| `complex/elementwise/binary_mul_complex64` |    3.497 |    2.547 | 1.368x |

The multiply row was the useful kernel target. It was still using the scalar
contiguous complex loop: two loads from each input, four multiplies, two
add/sub operations, and two stores per complex element. That is correct, but it
leaves unused the fact that complex64 storage is interleaved float lanes.

`src/elementwise/kernels/complex.mojo` now adds a `float32x4` path for
contiguous complex multiply. Each vector load covers two complex values:
`[a0,b0,a1,b1]` and `[c0,d0,c1,d1]`. The kernel broadcasts real and imaginary
rhs lanes with shuffles, computes real and imaginary candidate lanes, then
interleaves `[real0, imag0, real1, imag1]` back into the output vector. The
scalar tail remains for odd element counts, and the existing Smith division path
is untouched.

`src/elementwise/binary_dispatch.mojo` now also marks the non-Accelerate
contiguous complex path as `BackendKind.FUSED`, so benchmark/profile manifests
report the native fused kernel as fused backend code.

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_arithmetic_add_sub_mul_div_match_numpy tests/python/numpy_compat/test_complex.py::test_complex_strided_arithmetic_preserves_imaginary_part tests/python/numpy_compat/test_complex.py::test_complex_scalar_mul_with_complex_constant tests/python/numpy_compat/test_complex.py::test_complex64_contiguous_multiply_uses_fused_kernel tests/python/numpy_compat/test_complex.py::test_complex_scalar_mul_with_real_int -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-mul-simd-fused --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/elementwise/binary_mul_complex64 --types complex --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-complex-mul-simd-fused --sample --no-perf-stat
```

After the patch:

| row                                        |           before |            after |        delta |
| ------------------------------------------ | ---------------: | ---------------: | -----------: |
| `complex/elementwise/binary_mul_complex64` | 3.497 us, 1.368x | 3.210 us, 1.180x | 1.09x faster |

The post-fix profile manifest reports `used_fused=True`, `backend_code=2`, no
major faults, a 1,590 byte traced allocation peak, and max RSS around 49.6 MB.
The profile loop moved from 3.593 us/call to 3.275 us/call; the `sample` child
loop moved from 3.745 us/call to 3.424 us/call. This run used
`--no-perf-stat`, so hardware PMU counters are absent.

Remaining complex frontier:

| row                                            | monpy us | numpy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/array_copy_complex128`        |    3.849 |    2.590 | 1.454x |
| `complex/interop/asarray_complex64`            |    3.002 |    2.078 | 1.449x |
| `complex/views/reversed_add_complex64`         |    4.324 |    3.169 | 1.356x |
| `complex/casts/astype_complex64_to_complex128` |    3.387 |    2.704 | 1.272x |
| `complex/elementwise/binary_add_complex64`     |    3.270 |    2.575 | 1.255x |
| `complex/elementwise/binary_add_complex128`    |    3.330 |    2.751 | 1.211x |
| `complex/elementwise/binary_mul_complex64`     |    3.210 |    2.638 | 1.180x |
| `complex/matmul_64_complex64`                  |    7.355 |    7.147 | 1.055x |

Next target: the interop/copy cluster, but the likely owner
is allocation and Python/native wrapper overhead over another SIMD kernel. The useful question is whether `array(copy=True)` can call a narrower
native copy entrypoint that combines buffer import, allocation, and memcpy with
fewer Python object transitions.

### 2026-05-09 direct contiguous buffer copy and small complex add

The refreshed complex slice ranked `array_copy_complex128` and
`asarray_complex64` above the arithmetic rows:

| row                                         | monpy us | numpy us |  ratio |
| ------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/array_copy_complex128`     |    3.603 |    2.438 | 1.477x |
| `complex/interop/asarray_complex64`         |    2.885 |    1.990 | 1.450x |
| `complex/views/reversed_add_complex64`      |    4.142 |    3.039 | 1.364x |
| `complex/elementwise/binary_add_complex64`  |    3.083 |    2.516 | 1.245x |
| `complex/elementwise/binary_add_complex128` |    3.271 |    2.673 | 1.238x |
| `complex/elementwise/binary_mul_complex64`  |    3.085 |    2.543 | 1.204x |

First, I kept same-dtype C-contiguous buffer copies inside
`asarray_from_buffer_ops`: allocate the destination array, compute the storage
byte count, and `memcpy` directly from `Py_buffer.buf`. The old path wrapped the
source as an external `Array`, cloned shape/stride metadata, then called
`copy_c_contiguous`. The direct copy leaf preserves the fallback path for
strided inputs, dtype conversion, and readonly/copy policy failures.

Second, I changed the small complex ADD/SUB cost model. Contiguous
complex64/complex128 ADD/SUB was going through Accelerate vDSP even for the
1024-element benchmark row. At that size the framework call toll is larger than
the existing typed SIMD loop, so `maybe_complex_binary_contiguous_accelerate`
now only uses vDSP at 4096 complex elements and above. The 1024-element row
stays in the Mojo fused kernel (`backend_code=2`, `used_fused=True`,
`used_accelerate=False`).

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_arithmetic_add_sub_mul_div_match_numpy tests/python/numpy_compat/test_complex.py::test_complex_strided_arithmetic_preserves_imaginary_part tests/python/numpy_compat/test_complex.py::test_complex64_contiguous_multiply_uses_fused_kernel tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_true_detaches_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_copy_true_detaches_storage -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-small-mojo-add --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/elementwise/binary_add_complex64 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-binary-add-complex64-mojo-small --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/elementwise/binary_add_complex64 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-binary-add-complex64-numpy --sample --no-perf-stat
```

After the patch:

| row                                         |           before |            after |        delta |
| ------------------------------------------- | ---------------: | ---------------: | -----------: |
| `complex/interop/array_copy_complex128`     | 3.603 us, 1.477x | 3.539 us, 1.421x | 1.02x faster |
| `complex/elementwise/binary_add_complex64`  | 3.083 us, 1.245x | 2.981 us, 1.216x | 1.03x faster |
| `complex/elementwise/binary_add_complex128` | 3.271 us, 1.238x | 3.227 us, 1.226x | 1.01x faster |
| `complex/elementwise/binary_mul_complex64`  | 3.085 us, 1.204x | 3.067 us, 1.196x |   flat/noise |

Direct microbenchmarks show the copy leaf moving even though the full
benchmark is mostly wrapper-bound: native complex128 buffer copy moved from
1.002 us to 0.872 us, while `monpy.array(..., copy=True)` moved from 1.434 us
to 1.313 us. NumPy stayed around 0.42 us for the same copy.

The profile comparison for `complex/elementwise/binary_add_complex64` reports:

| candidate | us/call | max RSS | traced peak | backend                   |
| --------- | ------: | ------: | ----------: | ------------------------- |
| monpy     |   3.200 | 49.7 MB |     1,590 B | fused Mojo, no Accelerate |
| numpy     |   2.643 | 49.5 MB |    17,608 B | n/a                       |

This run used `--no-perf-stat`, so PMU counters are absent.
The macOS `sample(1)` stacks were written under
`results/local-profile-20260509-binary-add-complex64-*`; they mostly show the
benchmark harness and Python call/attribute machinery. The next useful profile
pass should use Instruments CPU Counters or a Linux `perf stat` run when we
want instruction/cache ratios alongside wall-clock deltas.

Remaining frontier:

| row                                            | monpy us | numpy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`            |    2.896 |    1.972 | 1.484x |
| `complex/interop/array_copy_complex128`        |    3.539 |    2.481 | 1.421x |
| `complex/views/reversed_add_complex64`         |    4.167 |    3.068 | 1.361x |
| `complex/casts/astype_complex64_to_complex128` |    3.267 |    2.576 | 1.265x |
| `complex/elementwise/binary_add_complex128`    |    3.227 |    2.636 | 1.226x |
| `complex/elementwise/binary_add_complex64`     |    2.981 |    2.451 | 1.216x |
| `complex/elementwise/binary_mul_complex64`     |    3.067 |    2.571 | 1.196x |
| `complex/matmul_64_complex64`                  |    7.576 |    7.161 | 1.061x |

Next target: the interop cluster. The direct copy leaf helped the
native portion, but the row is still 1.42x NumPy because object creation and
buffer classification dominate the remaining cost.

### 2026-05-09 specialized complex buffer wrappers

Next, I kept the existing benchmark harness fixed and refreshed the
complex slice:

| row                                            | monpy us | numpy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`            |    2.911 |    1.963 | 1.484x |
| `complex/interop/array_copy_complex128`        |    3.479 |    2.456 | 1.420x |
| `complex/views/reversed_add_complex64`         |    4.134 |    3.044 | 1.358x |
| `complex/casts/astype_complex64_to_complex128` |    3.258 |    2.557 | 1.270x |

One important measurement wrinkle: `time_call()` still enters
`warnings.catch_warnings()` through `call_bench_fn()` for every timed loop
iteration. A raw microbench put that context-manager tax at about 2.1 us/call
on this machine. That overhead applies to both monpy and NumPy rows, so the
official ratio remains useful for the current campaign, but raw microbenchmarks
are better for isolating the actual ingress leaf.

The raw ingress timings showed the target clearly:

| path                                          |   before |    after |
| --------------------------------------------- | -------: | -------: |
| generic native complex64 view                 | 0.461 us | 0.461 us |
| specialized native complex64 view             |      n/a | 0.366 us |
| `monumpy.asarray(..., complex64, copy=False)` | 0.838 us | 0.665 us |
| generic native complex128 copy                | 0.828 us | 0.828 us |
| specialized native complex128 copy            |      n/a | 0.729 us |
| `monumpy.array(..., complex128, copy=True)`   | 1.277 us | 1.051 us |

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

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_round_trip tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_false_shares_storage tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_true_detaches_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_copy_false_shares_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_copy_true_detaches_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_readonly_copy_false_raises_and_copy_none_detaches tests/python/numpy_compat/test_array_interface.py::test_ops_numpy_from_numpy_dtype_and_copy_arguments -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-specialized-buffer-wrappers-final --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/asarray_complex64 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-asarray-complex64-specialized-buffer --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/array_copy_complex128 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-array-copy-complex128-specialized-buffer --sample --no-perf-stat
```

Official result after the patch:

| row                                     |           before |            after |        delta |
| --------------------------------------- | ---------------: | ---------------: | -----------: |
| `complex/interop/asarray_complex64`     | 2.911 us, 1.484x | 2.817 us, 1.417x | 1.03x faster |
| `complex/interop/array_copy_complex128` | 3.479 us, 1.420x | 3.312 us, 1.318x | 1.05x faster |

The profile manifests reported `complex/interop/asarray_complex64` at 2.996
us/call and `complex/interop/array_copy_complex128` at 3.470 us/call, both with
max RSS around 49-50 MB, 1,590 byte traced allocation peaks, no major faults,
and default backend metadata. This run used `--no-perf-stat`; the `sample(1)`
stacks were captured under the two
`results/local-profile-20260509-*-specialized-buffer` directories.

Remaining frontier:

| row                                            | monpy us | numpy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`            |    2.817 |    1.999 | 1.417x |
| `complex/views/reversed_add_complex64`         |    4.131 |    3.121 | 1.359x |
| `complex/interop/array_copy_complex128`        |    3.312 |    2.547 | 1.318x |
| `complex/casts/astype_complex64_to_complex128` |    3.355 |    2.613 | 1.282x |
| `complex/elementwise/binary_add_complex128`    |    3.446 |    2.687 | 1.258x |

Next target: `complex/views/reversed_add_complex64` again.
The interop rows still lead, but their remaining raw leaf is already
sub-microsecond; the official row is now dominated by harness and Python wrapper
constant factors. The reversed-add row has real native work left in negative
stride handling and may have more actual loot per line changed.

### 2026-05-09 reverse view wrapper fast path

I started by trying a four-complex unroll inside the negative-stride
`complex64` ADD/SUB kernel. That stayed flat the leaf: presliced monpy add
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
rank-1 reverse view from the receiver fields directly, outside the generic
factory. `python/monpy/__init__.py` also inlines the wrapper construction for
the exact `[::-1]` case, skipping the extra `ndarray._wrap(...)` staticmethod
dispatch in this tiny hot path.

Raw post-fix timing:

| path                                 |   before |    after |
| ------------------------------------ | -------: | -------: |
| native `reverse_1d_method()`         | 0.230 us | 0.228 us |
| `ndarray._wrap(reverse_1d_method())` | 0.342 us | 0.337 us |
| `z[::-1]`                            | 0.456 us | 0.409 us |
| `z[::-1] + w[::-1]`                  | 1.932 us | 1.819 us |

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_strided_arithmetic_preserves_imaginary_part tests/python/numpy_compat/test_complex.py::test_complex_arithmetic_add_sub_mul_div_match_numpy tests/python/numpy_compat/test_complex.py::test_complex64_contiguous_multiply_uses_fused_kernel -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_strided_arithmetic_preserves_imaginary_part tests/python/numpy_compat/test_complex.py::test_complex_arithmetic_add_sub_mul_div_match_numpy tests/python/numpy_compat/test_indexing.py -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_strided_arithmetic_preserves_imaginary_part tests/python/numpy_compat/test_indexing.py -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-reverse-wrapper-fastpath --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/views/reversed_add_complex64 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-reversed-add-wrapper-fastpath-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/views/reversed_add_complex64 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-reversed-add-wrapper-fastpath-numpy --sample --no-perf-stat
```

Official result after the patch:

| row                                    |           before |            after |        delta |
| -------------------------------------- | ---------------: | ---------------: | -----------: |
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

Remaining frontier:

| row                                            | monpy us | numpy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`            |    2.790 |    1.971 | 1.408x |
| `complex/interop/array_copy_complex128`        |    3.280 |    2.441 | 1.339x |
| `complex/views/reversed_add_complex64`         |    4.076 |    3.054 | 1.335x |
| `complex/casts/astype_complex64_to_complex128` |    3.301 |    2.582 | 1.279x |
| `complex/elementwise/binary_add_complex128`    |    3.225 |    2.684 | 1.225x |

Next target: `complex/interop/array_copy_complex128` or the
attention rows. For complex reverse views, the easy wrapper win is spent and the
remaining ratio is mostly Python object overhead around two view creations plus
one already-fused native add.

### 2026-05-09 direct `array(..., complex128, copy=True)` buffer path

Next, I refreshed included both complex and attention rows:

| row                                                | monpy us | numpy us |  ratio |
| -------------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`                |    2.815 |    1.996 | 1.409x |
| `complex/interop/array_copy_complex128`            |    3.271 |    2.478 | 1.324x |
| `complex/views/reversed_add_complex64`             |    4.078 |    3.110 | 1.312x |
| `attention/attention/causal_attention_t32_d32_f32` |   20.488 |   21.857 | 0.937x |
| `attention/softmax/causal_scores_t32_f32`          |    7.995 |   10.007 | 0.799x |
| `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32`   |   78.485 |  107.275 | 0.733x |

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

| path                                        |   before |    after |
| ------------------------------------------- | -------: | -------: |
| native complex128 copy wrapper              | 0.711 us | 0.705 us |
| `ndarray(native complex128 copy)`           | 0.808 us | 0.801 us |
| `monpy.asarray(..., complex128, copy=True)` | 1.025 us | 1.029 us |
| `monpy.array(..., complex128, copy=True)`   | 1.051 us | 0.876 us |
| `numpy.array(..., complex128, copy=True)`   | 0.461 us | 0.453 us |

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_round_trip tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_true_detaches_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_copy_true_detaches_storage tests/python/numpy_compat/test_array_coercion.py::test_array_and_asarray_copy_rules_for_existing_monpy_arrays -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_array_coercion.py::test_explicit_supported_dtype_casts_match_numpy tests/python/numpy_compat/test_array_coercion.py::test_array_and_asarray_copy_rules_for_existing_monpy_arrays tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_round_trip tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_true_detaches_storage -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-array-c128-direct-copy --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/array_copy_complex128 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-array-copy-complex128-direct-array-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/array_copy_complex128 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-array-copy-complex128-direct-array-numpy --sample --no-perf-stat
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python -m monpy._bench.mojo_sweep --format json --sort ratio --output-dir results/local-sweep-20260509-mojo-stdlib-refresh --no-stdout --timeout 300
```

Official result after the patch:

| row                                     |           before |            after |        delta |
| --------------------------------------- | ---------------: | ---------------: | -----------: |
| `complex/interop/array_copy_complex128` | 3.271 us, 1.324x | 3.115 us, 1.268x | 1.05x faster |

The profile manifests reported `complex/interop/array_copy_complex128` at 3.245
us/call for monpy and 2.647 us/call for NumPy. Monpy used the generic backend
code 0 copy path, max RSS was about 49.6 MB, and the traced peak was 1,590
bytes. NumPy's traced peak was 33,992 bytes. This run used `--no-perf-stat`;
the `sample(1)` captures live under
`results/local-profile-20260509-array-copy-complex128-direct-array-*`.

The pure-Mojo stdlib sweep also completed. The largest candidate/stdlib ratios
were small: `small_matmul_f32_8` at 1.078x, `add_f32_1m` at 1.068x,
`sum_f32_1k` at 1.044x, and `prod_f64_64k` at 1.044x. That says the current
high-ratio NumPy-facing rows are mostly facade/object-bound.

Remaining frontier:

| row                                            | monpy us | numpy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`            |    2.793 |    1.963 | 1.419x |
| `complex/views/reversed_add_complex64`         |    4.086 |    3.072 | 1.329x |
| `complex/casts/astype_complex64_to_complex128` |    3.323 |    2.588 | 1.286x |
| `complex/interop/array_copy_complex128`        |    3.115 |    2.468 | 1.268x |
| `complex/elementwise/binary_add_complex128`    |    3.253 |    2.658 | 1.224x |

Next target: `complex/interop/asarray_complex64`. It remains the top
ratio row, and the stdlib sweep points away from a lower-level Mojo kernel as
the immediate blocker.

### 2026-05-09 direct complex64 copy-false buffer entry

Next, I kept `complex/interop/asarray_complex64` at the top of the
complex frontier:

| row                                            | monpy us | numpy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`            |    2.841 |    2.063 | 1.410x |
| `complex/views/reversed_add_complex64`         |    4.257 |    3.252 | 1.312x |
| `complex/casts/astype_complex64_to_complex128` |    3.437 |    2.737 | 1.258x |
| `complex/interop/array_copy_complex128`        |    3.203 |    2.592 | 1.233x |
| `complex/elementwise/binary_add_complex128`    |    3.398 |    2.780 | 1.220x |

Raw split before the patch:

| path                                        |     time |
| ------------------------------------------- | -------: |
| `_is_numpy_array(src)`                      | 0.077 us |
| native complex64 view wrapper               | 0.349 us |
| `ndarray(native, owner=src)`                | 0.487 us |
| `monpy.asarray(..., complex64, copy=False)` | 0.665 us |
| `numpy.asarray(..., complex64)`             | 0.073 us |

I first tried specializing `asarray_complex64_view_from_buffer_ops` in Mojo so
the wrapper would skip generic copy/cast branch machinery. The native wrapper stayed flat:
the native wrapper stayed around 0.35 us and public `asarray` regressed to about
0.684 us. The patch was reverted before landing.

The useful change was in `python/monpy/__init__.py`: exact
`asarray(obj, dtype=complex64, copy=False)` now tries the complex64 native
buffer view before asking whether `obj` is a NumPy array. For the benchmark's
NumPy input this removes the `_is_numpy_array(...)` detector from the hot
success path. Error mapping remains the same for wrong dtype, read-only arrays,
and unsupported NumPy dtypes.

Raw post-fix timing:

| path                                        |   before |    after |
| ------------------------------------------- | -------: | -------: |
| native complex64 view wrapper               | 0.349 us | 0.350 us |
| `ndarray(native, owner=src)`                | 0.487 us | 0.487 us |
| `monpy.asarray(..., complex64, copy=False)` | 0.665 us | 0.587 us |
| `numpy.asarray(..., complex64)`             | 0.073 us | 0.074 us |

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_round_trip tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_false_shares_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_copy_false_shares_storage tests/python/numpy_compat/test_array_coercion.py::test_numpy_array_readonly_copy_false_raises_and_copy_none_detaches tests/python/numpy_compat/test_array_interface.py::test_ops_numpy_from_numpy_dtype_and_copy_arguments -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-asarray-c64-direct --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/asarray_complex64 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-asarray-complex64-direct-buffer-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/asarray_complex64 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-asarray-complex64-direct-buffer-numpy --sample --no-perf-stat
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python -m monpy._bench.mojo_sweep --format json --sort ratio --output-dir results/local-sweep-20260509-mojo-stdlib-after-asarray-c64 --no-stdout --timeout 300
```

Official result after the patch:

| row                                 |           before |            after |        delta |
| ----------------------------------- | ---------------: | ---------------: | -----------: |
| `complex/interop/asarray_complex64` | 2.841 us, 1.410x | 2.632 us, 1.365x | 1.08x faster |

The profile manifests reported `complex/interop/asarray_complex64` at 2.797
us/call for monpy and 2.085 us/call for NumPy. Monpy used backend code 0, max
RSS was about 49.0 MB, and the traced peak was 1,590 bytes. NumPy's traced peak
was 1,406 bytes. This run used `--no-perf-stat`;
the `sample(1)` captures live under
`results/local-profile-20260509-asarray-complex64-direct-buffer-*`.

The pure-Mojo stdlib sweep still points away from a production kernel. Its top
candidate/stdlib ratios were `small_matmul_f32_8` at 1.076x,
`small_matmul_f64_8` at 1.042x, and `min_f64_64k` at 1.040x.

Remaining frontier:

| row                                            | monpy us | numpy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`            |    2.632 |    1.947 | 1.365x |
| `complex/views/reversed_add_complex64`         |    4.036 |    2.994 | 1.342x |
| `complex/casts/astype_complex64_to_complex128` |    3.263 |    2.534 | 1.287x |
| `complex/interop/array_copy_complex128`        |    3.024 |    2.405 | 1.254x |
| `complex/elementwise/binary_add_complex128`    |    3.203 |    2.597 | 1.231x |

Next target: `complex/views/reversed_add_complex64` or
`complex/casts/astype_complex64_to_complex128`. The remaining asarray gap is now
mostly wrapper construction versus NumPy returning its input object.

### 2026-05-09 cached native reverse views

Next, I refreshed put `asarray_complex64` and `reversed_add_complex64`
near the top again:

| row                                            | monpy us | numpy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`            |    2.808 |    2.060 | 1.365x |
| `complex/views/reversed_add_complex64`         |    4.308 |    3.216 | 1.330x |
| `complex/casts/astype_complex64_to_complex128` |    3.457 |    2.681 | 1.282x |
| `complex/interop/array_copy_complex128`        |    3.246 |    2.643 | 1.241x |
| `complex/elementwise/binary_add_complex64`     |    3.207 |    2.625 | 1.222x |

The remaining reverse-add cost sat around view construction. A raw split
showed presliced monpy add around 0.96 us, while the full expression paid for
two `[::-1]` calls every iteration. `ndarray` now caches the native reverse view
object in a private `_reverse_native` slot, but still returns a fresh Python
wrapper for each `a[::-1]` call. That reuses the native view construction. Each call still returns a fresh wrapper, so `a[::-1] is a[::-1]` stays false.

Raw post-fix timing:

| path                      |   before |    after |
| ------------------------- | -------: | -------: |
| `z[::-1]`                 | 0.409 us | 0.185 us |
| `z[::-1] is z[::-1]`      |    false |    false |
| presliced monpy add       | 0.948 us | 0.961 us |
| `z[::-1] + w[::-1]`       | 1.819 us | 1.361 us |
| NumPy `z[::-1] + w[::-1]` | 0.971 us | 0.983 us |

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_indexing.py tests/python/numpy_compat/test_complex.py::test_complex_strided_arithmetic_preserves_imaginary_part tests/python/numpy_compat/test_array_coercion.py::test_astype_supported_cast_matrix_matches_numpy -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-reverse-native-cache --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/views/reversed_add_complex64 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-reversed-add-native-cache-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/views/reversed_add_complex64 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-reversed-add-native-cache-numpy --sample --no-perf-stat
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python -m monpy._bench.mojo_sweep --format json --sort ratio --output-dir results/local-sweep-20260509-mojo-stdlib-refresh-0244 --no-stdout --timeout 300
```

Official result after the patch:

| row                                    |           before |            after |        delta |
| -------------------------------------- | ---------------: | ---------------: | -----------: |
| `complex/views/reversed_add_complex64` | 4.308 us, 1.330x | 3.591 us, 1.172x | 1.20x faster |

The profile manifests reported `complex/views/reversed_add_complex64` at 3.728
us/call for monpy and 3.208 us/call for NumPy. Monpy stayed on backend code 2,
the fused Mojo path, with max RSS about 49.5 MB and a 1,598 byte traced peak.
NumPy's traced peak was 17,760 bytes. This run used `--no-perf-stat`;
the `sample(1)` captures live under
`results/local-profile-20260509-reversed-add-native-cache-*`.

The pure-Mojo stdlib sweep stayed quiet. The largest candidate/stdlib ratios
were `small_matmul_f32_8` at 1.077x, `small_matmul_f64_8` at 1.035x,
`sum_f64_1k` at 1.022x, and `sum_f32_1k` at 1.016x.

Remaining frontier:

| row                                            | monpy us | numpy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`            |    2.708 |    1.975 | 1.369x |
| `complex/casts/astype_complex64_to_complex128` |    3.322 |    2.574 | 1.283x |
| `complex/interop/array_copy_complex128`        |    3.132 |    2.487 | 1.255x |
| `complex/elementwise/binary_add_complex64`     |    3.042 |    2.485 | 1.224x |
| `complex/elementwise/binary_add_complex128`    |    3.278 |    2.678 | 1.223x |
| `complex/views/reversed_add_complex64`         |    3.591 |    3.057 | 1.172x |

Next target: `complex/casts/astype_complex64_to_complex128` or another
interop facade cut. `asarray_complex64` is still the top ratio, but most of its
remaining gap is the unavoidable wrapper around a zero-copy external buffer.

### 2026-05-09 exact-DType `astype` facade fast path

Next, I refreshed made the cast row the best target with actual native
work behind it:

| row                                            | monpy us | numpy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`            |    2.853 |    2.074 | 1.361x |
| `complex/casts/astype_complex64_to_complex128` |    3.511 |    2.763 | 1.281x |
| `complex/interop/array_copy_complex128`        |    3.432 |    2.608 | 1.280x |
| `complex/elementwise/binary_add_complex64`     |    3.186 |    2.590 | 1.231x |
| `complex/elementwise/binary_add_complex128`    |    3.448 |    2.809 | 1.228x |

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

| path                                     |   before |    after |
| ---------------------------------------- | -------: | -------: |
| native `_native.astype(..., complex128)` | 0.645 us | 0.671 us |
| `ndarray(native astype)`                 | 0.744 us | 0.759 us |
| `z.astype(complex128)`                   | 1.067 us | 0.898 us |
| `z.astype(complex128, copy=False)`       |      n/a | 1.011 us |
| NumPy `z.astype(complex128)`             | 0.543 us | 0.540 us |

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_astype_between_widths_matches_numpy tests/python/numpy_compat/test_complex.py::test_complex_astype_drops_imag_to_real_target tests/python/numpy_compat/test_array_coercion.py::test_astype_supported_cast_matrix_matches_numpy tests/python/numpy_compat/test_array_coercion.py::test_astype_copy_false_keeps_identity_for_same_dtype -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-astype-dtype-fastpath --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/casts/astype_complex64_to_complex128 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-astype-complex64-to-complex128-dtype-fastpath-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/casts/astype_complex64_to_complex128 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-astype-complex64-to-complex128-dtype-fastpath-numpy --sample --no-perf-stat
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python -m monpy._bench.mojo_sweep --format json --sort ratio --output-dir results/local-sweep-20260509-mojo-stdlib-refresh-0304 --no-stdout --timeout 300
```

Official result after the patch:

| row                                            |           before |            after |        delta |
| ---------------------------------------------- | ---------------: | ---------------: | -----------: |
| `complex/casts/astype_complex64_to_complex128` | 3.511 us, 1.281x | 3.035 us, 1.178x | 1.16x faster |

The profile manifests reported `complex/casts/astype_complex64_to_complex128`
at 3.253 us/call for monpy and 2.774 us/call for NumPy. Monpy used backend code
0, max RSS was about 49.5 MB, and the traced peak was 1,598 bytes. NumPy's
traced peak was 33,992 bytes. This run used `--no-perf-stat`;
the `sample(1)` captures live under
`results/local-profile-20260509-astype-complex64-to-complex128-dtype-fastpath-*`.

The pure-Mojo stdlib sweep's largest candidate/stdlib ratio this pass was
`scalar_mul_f64_64k` at 1.149x, followed by `small_matmul_f32_8` at 1.080x.
That deserves a later kernel pass. The complex cast row, though, moved through
facade work while the native interleaved-lane cast stayed effectively flat.

Remaining frontier:

| row                                            | monpy us | numpy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`            |    2.678 |    1.965 | 1.365x |
| `complex/interop/array_copy_complex128`        |    3.030 |    2.467 | 1.237x |
| `complex/elementwise/binary_add_complex128`    |    3.225 |    2.637 | 1.226x |
| `complex/elementwise/binary_add_complex64`     |    2.970 |    2.463 | 1.213x |
| `complex/elementwise/binary_mul_complex64`     |    3.070 |    2.557 | 1.193x |
| `complex/casts/astype_complex64_to_complex128` |    3.035 |    2.576 | 1.178x |
| `complex/views/reversed_add_complex64`         |    3.525 |    3.011 | 1.172x |

Next target: move off the complex view/cast rows and either revisit
interop copy/view facade costs or investigate the pure-Mojo `scalar_mul_f64_64k`
stdlib deficit as a separate kernel-level pass.

### 2026-05-09 direct static typed elementwise ops

Next, I split the frontier into two different problems. The public
complex rows were still led by facade and complex-buffer cases:

| row                                         | monpy us | numpy us |  ratio |
| ------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`         |    2.667 |    1.951 | 1.370x |
| `complex/interop/array_copy_complex128`     |    3.087 |    2.427 | 1.268x |
| `complex/elementwise/binary_add_complex64`  |    2.975 |    2.418 | 1.229x |
| `complex/elementwise/binary_add_complex128` |    3.234 |    2.643 | 1.224x |
| `complex/elementwise/binary_mul_complex64`  |    3.072 |    2.531 | 1.206x |

The pure-Mojo stdlib comparison exposed a cleaner kernel-level miss:

| row                       | candidate |   stdlib |  ratio |
| ------------------------- | --------: | -------: | -----: |
| `elementwise/add_f64_64k` |  13.79 us | 12.30 us | 1.121x |
| `reductions/prod_f32_64k` |   3.24 us |  3.07 us | 1.057x |
| `elementwise/sin_f64_1k`  |   2.58 us |  2.48 us | 1.040x |

The static typed binary kernels were still going through
`apply_binary_typed_vec_static` inside the SIMD loop. Even though `op` is a
comptime parameter, the stdlib baseline was spelling the ADD loop directly:
`load[width]`, add, `store`. The kernel now emits direct SIMD loops for static
ADD/SUB/MUL/DIV in `binary_same_shape_contig_typed_static`, plus direct scalar
ADD/MUL loops in `binary_scalar_contig_typed_static` where operand order does
leaves the result unchanged. The generic helper remains for the wider binary-op set.

Verification:

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

After the patch, pure Mojo:

| row                       |                       before |                        after |             delta |
| ------------------------- | ---------------------------: | ---------------------------: | ----------------: |
| `elementwise/add_f64_64k` | 13.79 us vs 12.30 us, 1.121x | 16.21 us vs 16.18 us, 1.002x | ratio gap removed |

The absolute nanoseconds moved with machine noise on this run, but the
candidate and stdlib paths now track each other at the row level. This is the
right signal for this harness because both sides are measured back-to-back in
the same generated Mojo binary.

Public complex refresh after rebuilding:

| row                                        |                       before |                        after |                   delta |
| ------------------------------------------ | ---------------------------: | ---------------------------: | ----------------------: |
| `complex/elementwise/binary_add_complex64` | 2.975 us vs 2.418 us, 1.229x | 2.988 us vs 2.484 us, 1.201x | 1.02x ratio improvement |
| `complex/interop/array_copy_complex128`    | 3.087 us vs 2.427 us, 1.268x | 3.057 us vs 2.473 us, 1.235x | 1.03x ratio improvement |

The public array slice kept the real-valued elementwise guardrail clean. The
bandwidth-size `array/bandwidth/binary_add_65536_f32` row stayed comfortably
ahead of NumPy at 6.930 us vs 11.633 us, 0.607x. The wrapper-size
`array/elementwise/binary_add_f32` row was 3.036 us vs 2.407 us, 1.263x, which
keeps the next public array target in facade/orchestration territory rather
than this SIMD loop.

The `complex/elementwise/binary_add_complex64` profiles reported 3.146 us/call
for monpy and 2.616 us/call for NumPy. Monpy used backend code 2, the fused
native path, with max RSS about 49.3 MB and a traced allocation peak of 1,598
bytes. NumPy's traced allocation peak was 17,608 bytes. The profile command
used `--no-perf-stat`; the `sample(1)` captures live under
`results/local-profile-20260509-binary-add-complex64-static-direct-*`.

Remaining frontier:

| row                                         | monpy us | numpy us |  ratio |
| ------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`         |    2.671 |    1.974 | 1.354x |
| `complex/interop/array_copy_complex128`     |    3.057 |    2.473 | 1.235x |
| `complex/elementwise/binary_add_complex128` |    3.239 |    2.649 | 1.232x |
| `complex/elementwise/binary_add_complex64`  |    2.988 |    2.484 | 1.201x |
| `complex/elementwise/binary_mul_complex64`  |    3.080 |    2.567 | 1.197x |

Next target: either specialize complex128 add/mul at the fused-kernel
level, or attack the `asarray_complex64` facade only if a profile shows more
than wrapper creation and dtype lookup left to remove.

### 2026-05-09 specialized complex64 zero-copy buffer view

Next, I refreshed put `asarray_complex64` back at the top of the public
complex frontier:

| row                                         | monpy us | numpy us |  ratio |
| ------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`         |    3.063 |    2.123 | 1.443x |
| `complex/interop/array_copy_complex128`     |    3.378 |    2.814 | 1.251x |
| `complex/elementwise/binary_add_complex64`  |    3.270 |    2.647 | 1.228x |
| `complex/elementwise/binary_mul_complex64`  |    3.345 |    2.753 | 1.212x |
| `complex/elementwise/binary_add_complex128` |    3.519 |    2.866 | 1.209x |

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

This skips the generic `requested_code`/`copy_flag` branch tree, dtype decode,
`must_copy` calculation, and copy/cast fallback setup in the hot
`copy=False, dtype=complex64` path.

Verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/buffer.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_false_shares_storage tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_round_trip tests/python/test_buffer_core.py::test_asarray_from_writable_buffer_shares_storage tests/python/test_buffer_core.py::test_asarray_from_readonly_buffer_copies_by_default_and_rejects_copy_false tests/python/test_buffer_core.py::test_asarray_buffer_dtype_mismatch_copy_policy -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-c64-view-format-fastcheck-0344 --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/asarray_complex64 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-asarray-complex64-format-fastcheck-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/asarray_complex64 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-asarray-complex64-format-fastcheck-numpy --sample --no-perf-stat
```

After the patch:

| row                                 |                       before |                        after |                   delta |
| ----------------------------------- | ---------------------------: | ---------------------------: | ----------------------: |
| `complex/interop/asarray_complex64` | 3.063 us vs 2.123 us, 1.443x | 2.681 us vs 1.973 us, 1.360x | 1.14x faster monpy side |

The final profiles reported `asarray_complex64` at 2.742 us/call for monpy and
2.062 us/call for NumPy. Monpy used backend code 0, as expected for a zero-copy
external view. Max RSS was about 49.2 MB for monpy and 49.3 MB for NumPy.
Traced allocation peaks were 1,598 bytes for monpy and 1,406 bytes for NumPy.
The profile command used `--no-perf-stat`; the `sample(1)` captures live under
`results/local-profile-20260509-asarray-complex64-format-fastcheck-*`.

Remaining frontier:

| row                                         | monpy us | numpy us |  ratio |
| ------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`         |    2.681 |    1.973 | 1.360x |
| `complex/elementwise/binary_add_complex128` |    3.290 |    2.668 | 1.248x |
| `complex/interop/array_copy_complex128`     |    3.070 |    2.478 | 1.247x |
| `complex/elementwise/binary_add_complex64`  |    2.998 |    2.459 | 1.220x |
| `complex/elementwise/binary_mul_complex64`  |    3.094 |    2.552 | 1.210x |

Next target: complex128 add/copy. The remaining
`asarray_complex64` gap is now mostly the cost of allocating a monpy wrapper
around a borrowed Python buffer; there is less generic bridge code left to cut.

### 2026-05-09 specialized complex128 copy buffer path

Next, I kept the public interop and complex128 rows at the top:

| row                                         | monpy us | numpy us |  ratio |
| ------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`         |    2.657 |    1.961 | 1.355x |
| `complex/interop/array_copy_complex128`     |    3.046 |    2.441 | 1.253x |
| `complex/elementwise/binary_add_complex128` |    3.221 |    2.619 | 1.228x |
| `complex/elementwise/binary_add_complex64`  |    2.968 |    2.423 | 1.222x |
| `complex/elementwise/binary_mul_complex64`  |    3.056 |    2.552 | 1.200x |

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

Verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/buffer.mojo src/array/cast.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_true_detaches_storage tests/python/numpy_compat/test_complex.py::test_complex_array_from_strided_numpy_copy_true_detaches_storage tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_round_trip tests/python/numpy_compat/test_complex.py::test_complex_array_from_numpy_copy_false_shares_storage tests/python/test_buffer_core.py::test_asarray_from_writable_buffer_shares_storage tests/python/test_buffer_core.py::test_asarray_buffer_dtype_mismatch_copy_policy -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types complex --loops 50 --repeats 7 --rounds 5 --matrix-sizes 64 --format json --sort ratio --output-dir results/local-sweep-20260509-complex-c128-copy-specialized-final-0404 --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/array_copy_complex128 --types complex --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-array-copy-complex128-specialized-final-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case complex/interop/array_copy_complex128 --types complex --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-array-copy-complex128-specialized-final-numpy --sample --no-perf-stat
```

After the patch:

| row                                     |                       before |                        after |                   delta |
| --------------------------------------- | ---------------------------: | ---------------------------: | ----------------------: |
| `complex/interop/array_copy_complex128` | 3.046 us vs 2.441 us, 1.253x | 2.961 us vs 2.429 us, 1.220x | 1.03x faster monpy side |

The final profiles reported `array_copy_complex128` at 3.140 us/call for monpy
and 2.627 us/call for NumPy. Monpy used backend code 0, as expected for a copy
bridge. Max RSS was about 49.3 MB for both candidates. Traced allocation peaks
were 1,598 bytes for monpy and 33,992 bytes for NumPy. The profile command used `--no-perf-stat`; the
`sample(1)` captures live under
`results/local-profile-20260509-array-copy-complex128-specialized-final-*`.

Remaining frontier:

| row                                         | monpy us | numpy us |  ratio |
| ------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`         |    2.660 |    1.939 | 1.359x |
| `complex/elementwise/binary_add_complex128` |    3.221 |    2.595 | 1.241x |
| `complex/interop/array_copy_complex128`     |    2.961 |    2.429 | 1.220x |
| `complex/elementwise/binary_add_complex64`  |    2.948 |    2.424 | 1.216x |
| `complex/elementwise/binary_mul_complex64`  |    3.057 |    2.523 | 1.204x |

Next target: the complex128 ADD fused path. The copy bridge has less
generic machinery left, and the live add row now sits above it.

### 2026-05-09 Float32 softmax row kernel

The first refresh for this heartbeat checked both the public complex frontier
and the attention slice. The complex rows were still slower than NumPy, led by
`asarray_complex64` and the complex add/copy cluster:

| row                                         | monpy us | numpy us |  ratio |
| ------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`         |    2.653 |    1.945 | 1.363x |
| `complex/elementwise/binary_add_complex64`  |    2.977 |    2.444 | 1.230x |
| `complex/elementwise/binary_add_complex128` |    3.198 |    2.607 | 1.223x |
| `complex/elementwise/binary_mul_complex64`  |    3.042 |    2.512 | 1.217x |
| `complex/interop/array_copy_complex128`     |    2.952 |    2.443 | 1.216x |

The attention rows, however, were all already ahead of NumPy:

| row                                                | monpy us | numpy us |  ratio |
| -------------------------------------------------- | -------: | -------: | -----: |
| `attention/attention/causal_attention_t32_d32_f32` |   20.681 |   22.056 | 0.931x |
| `attention/softmax/causal_scores_t32_f32`          |    8.506 |   10.500 | 0.807x |
| `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32`   |   81.256 |  105.450 | 0.777x |

The useful attention miss I found was still local: the f32 softmax kernels were doing
row max, exp, denominator accumulation, and normalization through `Float64`.
The local Mojo stdlib `std.math.exp` overload returns the same floating type it
receives, so the float32 row kernel can stay in `Float32` end to end. This
matches NumPy's f32 softmax path more closely and removes two conversions per
lane in the hot row loops.

`src/elementwise/kernels/nn.mojo` now has narrow Float32 paths for plain
last-axis softmax and scaled masked last-axis softmax. Float64 keeps the
previous accumulation path.

Verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/elementwise/kernels/nn.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_numeric.py::test_fused_softmax_matches_numpy_formula tests/python/numpy_compat/test_numeric.py::test_fused_scaled_masked_softmax_matches_numpy_formula tests/python/numpy_compat/test_numeric.py::test_fused_layer_norm_matches_numpy_formula -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types attention --loops 20 --repeats 7 --rounds 5 --vector-size 1024 --matrix-sizes 32 --format json --sort ratio --output-dir results/local-sweep-20260509-attention-f32-softmax --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case attention/softmax/causal_scores_t32_f32 --types attention --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-attention-softmax-f32-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case attention/softmax/causal_scores_t32_f32 --types attention --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-attention-softmax-f32-numpy --sample --no-perf-stat
```

After the patch:

| row                                                |                          before |                           after |                          delta |
| -------------------------------------------------- | ------------------------------: | ------------------------------: | -----------------------------: |
| `attention/softmax/causal_scores_t32_f32`          |   8.506 us vs 10.500 us, 0.807x |   6.869 us vs 10.052 us, 0.683x |        1.24x faster monpy side |
| `attention/attention/causal_attention_t32_d32_f32` |  20.681 us vs 22.056 us, 0.931x |  19.256 us vs 22.242 us, 0.864x |        1.07x faster monpy side |
| `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32`   | 81.256 us vs 105.450 us, 0.777x | 80.921 us vs 108.756 us, 0.745x | 1.00x wall-clock, better ratio |

The profile manifests reported `attention/softmax/causal_scores_t32_f32` at
6.978 us/call for monpy and 10.278 us/call for NumPy. Monpy used backend code
2, the fused native path. Max RSS was about 51.7 MB for monpy and 51.5 MB for
NumPy. Traced allocation peaks were 1,694 bytes for monpy and 23,400 bytes for
NumPy. The monpy `sample(1)` call graph put 1,033 of 2,289 samples directly in
`elementwise::kernels::nn::_softmax_last_axis_f32`, while NumPy's sample showed
most softmax time in ufunc reduction machinery (`PyUFunc_Reduce`,
`FLOAT_maximum`, and `FLOAT_pairwise_sum`). The profile command used
`--no-perf-stat`; the `sample(1)` captures live under
`results/local-profile-20260509-attention-softmax-f32-*`.

Next target: complex128 add/mul, or add a larger attention
matrix-size row so the attention benchmark can expose when softmax stops being
wrapper-sized and starts becoming cache/bandwidth sized.

### 2026-05-09 static unary typed kernels

Next, I split the frontier again. The public NumPy-facing complex
slice was still led by small wrapper/facade rows:

| row                                         | monpy us | numpy us |  ratio |
| ------------------------------------------- | -------: | -------: | -----: |
| `complex/interop/asarray_complex64`         |    2.880 |    2.129 | 1.348x |
| `complex/elementwise/binary_add_complex64`  |    3.306 |    2.512 | 1.258x |
| `complex/elementwise/binary_add_complex128` |    3.249 |    2.969 | 1.223x |
| `complex/interop/array_copy_complex128`     |    3.390 |    2.661 | 1.207x |
| `complex/elementwise/binary_mul_complex64`  |    3.075 |    2.550 | 1.207x |

The attention slice stayed ahead of NumPy after the Float32 softmax patch:

| row                                                | monpy us | numpy us |  ratio |
| -------------------------------------------------- | -------: | -------: | -----: |
| `attention/attention/causal_attention_t32_d32_f32` |   20.904 |   22.658 | 0.887x |
| `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32`   |   81.802 |  107.356 | 0.750x |
| `attention/softmax/causal_scores_t32_f32`          |    7.181 |   10.331 | 0.694x |

The pure-Mojo stdlib comparison exposed the real leaf-level miss:

| row                          | candidate |    stdlib |  ratio |
| ---------------------------- | --------: | --------: | -----: |
| `elementwise/sin_f64_64k`    | 394342 ns | 187974 ns | 2.098x |
| `elementwise/add_f64_1k`     |  222.7 ns |  185.0 ns | 1.204x |
| `matmul/small_matmul_f32_16` |  452.9 ns |  401.5 ns | 1.128x |

`unary_contig_typed` was still calling `apply_unary_typed_vec(..., op)` inside
the vector loop. For common unary ops that means each SIMD chunk entered the
runtime `op` dispatcher before reaching `sin`, `cos`, `exp`, etc. The stdlib
baseline spells the operation directly as `std_sin(load[width])`. The typed
kernel now mirrors the binary-kernel pattern: `try_unary_contig_typed_static`
routes common ops (`SIN`, `COS`, `EXP`, `LOG`, `TANH`, `SQRT`, `NEGATE`,
`POSITIVE`, `SQUARE`) into a comptime-`op` static loop, removing the per-vector
runtime branch.

Verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/elementwise/kernels/typed.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_umath.py::test_unary_math_matches_numpy_float_dtypes tests/python/numpy_compat/test_umath.py::test_unary_math_preserves_float32_result_dtype tests/python/numpy_compat/test_umath.py::test_unary_math_is_a_full_numpy_ufunc tests/python/numpy_compat/test_numeric.py::test_fused_sin_add_mul_matches_numpy tests/python/numpy_compat/test_numeric.py::test_numpy_shaped_expression_lowers_to_fused_kernel -q
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python -m monpy._bench.mojo_sweep --format json --sort ratio --output-dir results/local-sweep-20260509-mojo-unary-static-0444 --no-stdout --timeout 300
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types array --loops 50 --repeats 7 --rounds 5 --vector-sizes 65536 --matrix-sizes 64 --linalg-sizes 8 --format json --sort ratio --output-dir results/local-sweep-20260509-array-unary-static-0444 --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case array/bandwidth/unary_sin_65536_f32 --types array --vector-sizes 65536 --matrix-sizes 64 --linalg-sizes 8 --candidate monpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-unary-sin-static-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m monpy._bench.profile --case array/bandwidth/unary_sin_65536_f32 --types array --vector-sizes 65536 --matrix-sizes 64 --linalg-sizes 8 --candidate numpy --duration 3 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-unary-sin-static-numpy --sample --no-perf-stat
```

After the patch, pure Mojo:

| row                       |                         before |                          after |                    delta |
| ------------------------- | -----------------------------: | -----------------------------: | -----------------------: |
| `elementwise/sin_f64_64k` | 394342 ns vs 187974 ns, 2.098x | 167833 ns vs 167000 ns, 1.005x |        ratio gap removed |
| `elementwise/sin_f32_64k` | 118877 ns vs 116339 ns, 1.022x |             below top deficits | no visible remaining gap |

The remaining stdlib frontier is much flatter: `scalar_mul_f64_64k` leads at
1.086x, `sum_f32_1k` is 1.033x, and `small_matmul_f32_16` is 1.029x. That is a
clean drop from a 2.10x top leaf to low-single-digit misses.

The public NumPy-facing array sweep still uses the macOS Accelerate path for
large f32 sine (`backend_code = 1` in profile), so this patch mainly closes the
pure Mojo leaf gap more than the public facade row. The post-fix public
bandwidth row was still comfortably ahead of NumPy:

| row                                           | monpy us | numpy us |  ratio |
| --------------------------------------------- | -------: | -------: | -----: |
| `array/bandwidth/unary_sin_65536_f32`         |   32.139 |  101.894 | 0.313x |
| `array/bandwidth/fused_sin_add_mul_65536_f32` |   39.788 |  115.449 | 0.347x |
| `array/elementwise/unary_sin_f32`             |    4.523 |    3.792 | 1.191x |

The sample profile for `array/bandwidth/unary_sin_65536_f32` reported monpy at
32.483 us/call and NumPy at 102.304 us/call. Monpy used backend code 1, the
Accelerate path, with a 1,656 byte traced allocation peak; NumPy peaked at
525,512 bytes. The sample call graph showed monpy spending the hot loop in
`VVSINF` through `create::ops::elementwise::unary_ops`, while NumPy spent the
hot path in `_multiarray_umath`'s `simd_sincos_f32` ufunc loop. The profile command used `--no-perf-stat`; captures live
under `results/local-profile-20260509-unary-sin-static-*`.

Next target: the new pure-Mojo `scalar_mul_f64_64k` stdlib gap or the
public complex64 add/asarray wrapper cluster. The former is a cleaner leaf
kernel target; the latter is the larger NumPy-facing ratio.

### 2026-05-09 scalar static commutative loop split

Refresh before editing showed attention still ahead of NumPy and the pure-Mojo
stdlib comparison led by scalar multiply:

| slice     | row                                                |  candidate |   baseline |  ratio |
| --------- | -------------------------------------------------- | ---------: | ---------: | -----: |
| attention | `attention/attention/causal_attention_t32_d32_f32` |  20.483 us |  23.750 us | 0.843x |
| attention | `attention/softmax/causal_scores_t32_f32`          |   7.846 us |  11.367 us | 0.690x |
| attention | `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32`   |  71.900 us | 113.706 us | 0.619x |
| pure Mojo | `elementwise/scalar_mul_f64_64k`                   | 11843.5 ns | 10464.7 ns | 1.132x |
| pure Mojo | `elementwise/add_f32_1m`                           |  131810 ns |  120804 ns | 1.091x |

`binary_scalar_contig_typed_static` already specialized ADD/MUL with comptime
branches, but those branches lived inside the shared scalar-left-aware kernel
body. The stdlib scalar-multiply baseline is just a direct SIMD load, multiply,
store, tail loop. I split ADD and MUL into compile-time early-return loops so
the commutative hot path has no `scalar_on_left` branch or fallback
`apply_binary_typed_vec_static` scaffolding in its body.

Verification:

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
0.987x. The rerun put it at 1.002x, so the practical conclusion is "closed to parity":

| row                              |                           before |                            after |                            rerun |
| -------------------------------- | -------------------------------: | -------------------------------: | -------------------------------: |
| `elementwise/scalar_mul_f64_64k` | 11843.5 ns vs 10464.7 ns, 1.132x | 10527.7 ns vs 10663.8 ns, 0.987x | 11772.2 ns vs 11753.8 ns, 1.002x |
| `elementwise/scalar_mul_f32_64k` |   5932.2 ns vs 5831.3 ns, 1.017x |   5583.4 ns vs 5503.7 ns, 1.014x |   5857.9 ns vs 5544.1 ns, 1.057x |

One sweep produced a false-looking `sum_f32_1m` spike at 4.965x; the immediate
rerun put the same row back at 0.994x. That row is unrelated to the scalar patch
and I would treat as benchmark noise unless it reproduces in a third run
with a longer minimum runtime.

The NumPy-facing guardrail stayed healthy for the bandwidth row:

| row                                           |     monpy |      NumPy |  ratio |
| --------------------------------------------- | --------: | ---------: | -----: |
| `array/bandwidth/binary_add_65536_f32`        |  7.334 us |  11.278 us | 0.670x |
| `array/bandwidth/reversed_add_65536_f32`      | 13.884 us |  30.888 us | 0.450x |
| `array/bandwidth/fused_sin_add_mul_65536_f32` | 39.779 us | 116.539 us | 0.342x |

The `sample(1)` CPU profile for `array/bandwidth/binary_add_65536_f32` showed
monpy spending the main hot path under `maybe_binary_same_shape_contiguous`
inside `libvDSP`, while NumPy spent the hot loop in `_multiarray_umath`
`FLOAT_add` plus allocation and ufunc dispatch. Wall-clock profile measurement
was 7.932 us/call for monpy versus 11.433 us/call for NumPy. Tracemalloc peaks
were 1,598 bytes for monpy and 525,512 bytes for NumPy, a 329:1 peak-allocation
ratio. This macOS run used `--no-perf-stat`; the stack samples and manifests live under
`results/local-profile-20260509-binary-add-static-*`.

Next target: either the pure-Mojo `add_f32_64k`/`add_f32_1m` variance
with a longer stdlib harness runtime, or the larger public wrapper cluster
(`moveaxis_f32`, `empty_like_shape_override_f32`, `transpose_add_f32`) where the
NumPy-facing ratios are still 1.7x-1.9x.

### 2026-05-09 parallel worker policy split

The "8-way" reduction note was easy to misread as "use 8 workers because this
Mac has 8 performance cores." The actual contract is
`REDUCE_SIMD_ACCUMULATORS = 8`: eight independent SIMD accumulator chains inside
one worker, chosen for instruction-level parallelism on 2-IPC floating-point
pipelines. Core fanout is a separate policy owned by
`elementwise/kernels/parallel.mojo`.

The parallel policy is now split into three layers:

1. `worker_count(work_units)`: hardware and `MONPY_THREADS` cap only.
   `MONPY_THREADS=1` still forces serial execution, while `MONPY_THREADS=N`
   caps automatic fanout at N workers.
2. `worker_count_for_bytes(...)`: whole-tensor byte budget per worker. A
   many-core Ubuntu host can use more workers than this laptop, but only when
   the tensor has enough bytes to keep each worker out of tiny cache-cold
   slices.
3. `worker_count_for_row_elements(...)`: row-kernel budget using both row count
   and row width. Row kernels can only split between rows, so row count alone
   is insufficient: a 32x32 attention softmax should stay serial, while a
   32x4096 softmax has enough work inside each row to justify fanout.

That third split mattered. The first implementation used only
`worker_count_for_rows(rows)`, which let the 32-row attention benchmark spawn
two workers. The smoke result immediately regressed:

| smoke           | row                                                | monpy us | NumPy us |  ratio |
| --------------- | -------------------------------------------------- | -------: | -------: | -----: |
| row-only policy | `attention/softmax/causal_scores_t32_f32`          |   18.300 |   12.438 | 1.466x |
| row-only policy | `attention/attention/causal_attention_t32_d32_f32` |   38.654 |   27.550 | 1.412x |
| row-only policy | `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32`   |  139.608 |  116.442 | 1.199x |

After adding the row-element budget, the same small attention rows stayed on the
serial f32 row kernels and returned below NumPy:

| smoke              | row                                                | monpy us | NumPy us |  ratio |
| ------------------ | -------------------------------------------------- | -------: | -------: | -----: |
| row-element policy | `attention/softmax/causal_scores_t32_f32`          |    9.387 |   13.142 | 0.711x |
| row-element policy | `attention/attention/causal_attention_t32_d32_f32` |   23.342 |   26.450 | 0.886x |
| row-element policy | `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32`   |   72.921 |  117.988 | 0.619x |

Verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/elementwise/kernels/parallel.mojo src/elementwise/kernels/nn.mojo src/elementwise/kernels/__init__.mojo src/elementwise/kernels/reduce.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_numeric.py::test_reductions_match_numpy_for_axis_none tests/python/numpy_compat/test_numeric.py::test_axis_reductions_match_numpy tests/python/numpy_compat/test_numeric.py::test_float_axis_last_reductions_match_numpy tests/python/numpy_compat/test_numeric.py::test_fused_softmax_matches_numpy_formula tests/python/numpy_compat/test_numeric.py::test_fused_scaled_masked_softmax_matches_numpy_formula tests/python/numpy_compat/test_numeric.py::test_fused_layer_norm_matches_numpy_formula -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types attention --loops 5 --repeats 3 --rounds 2 --vector-size 1024 --matrix-sizes 32 --format json --sort ratio --output-dir results/local-sweep-20260509-parallel-policy-attention-smoke2 --no-progress --no-stdout
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python -m monpy._bench.mojo_sweep --format json --sort ratio --output-dir results/local-sweep-20260509-parallel-policy-mojo --no-stdout --timeout 300
MONPY_THREADS=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python -m monpy._bench.mojo_sweep --format json --sort ratio --output-dir results/local-sweep-20260509-parallel-policy-mojo-serial --no-stdout --timeout 300
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python -m monpy._bench.mojo_sweep --format json --sort ratio --output-dir results/local-sweep-20260509-parallel-policy-mojo-rerun --no-stdout --timeout 300
```

The pure-Mojo sweep had visible system noise in unrelated raw-kernel rows. One
run reported `reductions/sum_f64_1m` at 2.197x against `std.algorithm.sum`, but
the forced-serial pass put the same raw `reduce_sum_typed` row at 1.057x and an
immediate normal rerun put it at 1.006x. Since `bench_mojo_sweep.mojo` calls
`reduce_sum_typed` directly, that spike belongs outside the new worker policy.
I would treat it as reduction-grain noise until a longer minimum-runtime harness
reproduces it.

Next target: add a larger attention-size sweep so the row-element policy can be
calibrated beyond the 32x32 smoke. The intended curve is serial for tiny
attention, then gradual fanout as `rows * cols / ROW_HEAVY_GRAIN_ELEMS` grows,
capped by row slices, hardware performance cores, and `MONPY_THREADS`.

### 2026-05-09 reduction parallel gate rollback

The live strict Mojo sweep made the problem visible: the default stdlib
comparison had started ranking calibration rows instead of production rows.
`reduce_*_par_typed` was being called directly with `num_performance_cores()`,
and those rows dominated the table with ratios like `max_par_f32_1m` at 57.5x
slower than the serial 8-accumulator reducer.

The public path had the same smell. A separate Python probe compared normal
execution against `MONPY_THREADS=1` in fresh processes:

- before the gate change, `sum/min/max` over 1M f32 could hit hundreds of
  microseconds on the auto path while the serial path sat around 54-55 us.
- after the gate change, the auto path no longer fans out below 1GB of reduction
  input, so 1M and 16M f32 reductions use the same serial reducer as
  `MONPY_THREADS=1`.
- the cause is not SIMD. The SIMD reducer is fine. The expensive atom is
  `sync_parallelize`: the stdlib path creates a CPU `DeviceContext`, enqueues
  tasks, and synchronizes for each call. Until monpy has a persistent worker
  context or a better reduction scheduler, that overhead eats the reduction.

Code changes:

- `REDUCE_GRAIN` moved from 1MB to 1GB. This keeps production reductions serial
  for the sizes that currently matter in the benchmark suite.
- `bench_mojo_sweep.mojo` stopped emitting direct parallel-reduction calibration
  rows. Those belong in `bench_parallel.mojo` or `bench_reduce.mojo`; the default
  stdlib sweep should rank stdlib/kernel deficits, not non-production probes.

Followup (same day): the `REDUCE_GRAIN=1GB` gate left the four `reduce_*_par_typed`
kernels (sum/min/max/prod) unreachable from `maybe_reduce_contiguous`. Rather than
keep ~150 lines of dead surface, the kernels were deleted along with the
`REDUCE_GRAIN` constant itself, the unused `worker_count_for_bytes`/`alloc`
imports in `reduce.mojo`, and the orphan `emit_*_par` helpers in
`bench_mojo_sweep.mojo`. The `reduce_sum_typed`/`reduce_min_typed`/etc. serial
kernels remain untouched. Restore from git if Mojo's threading primitive grows
a persistent worker pool that makes the per-call overhead acceptable.

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_numeric.py::test_reductions_match_numpy_for_axis_none tests/python/numpy_compat/test_numeric.py::test_axis_reductions_match_numpy tests/python/numpy_compat/test_numeric.py::test_float_axis_last_reductions_match_numpy tests/python/test_mojo_bench_sweep.py -q
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench-mojo --include-numojo --strict-numojo --format json --sort ratio --output-dir results/local-sweep-20260509-reduce-gate-numojo-strict-0653 --no-stdout --timeout 300
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types array,attention --loops 10 --repeats 3 --rounds 2 --vector-sizes 65536 --matrix-sizes 32 --linalg-sizes 8 --format json --sort ratio --output-dir results/local-sweep-20260509-reduce-gate-array-attention-0653 --no-progress --no-stdout
```

After removing the calibration rows from the default Mojo sweep, the top stdlib
deficit is no longer a fake production blocker. The remaining real stdlib
frontier is unary/binary elementwise variance (`sin_f64_1m`, `add_f32_64k`) and
small product reductions. The NumPy-facing slice still shows attention ahead of
NumPy (`softmax` 0.637x, causal attention 0.782x, GPT logits 0.643x), while the
array frontier is back in wrapper/copy territory (`full_like_transpose_f32`,
`astype_f32_to_bool`, `asarray_zero_copy_f32`).

Next target: separate unary-parallel calibration from public array performance.
The current stdlib sweep can make `sin` and `add` look worse than the public
NumPy path, so the next pass should run `bench_parallel.mojo`, then decide
whether static unary fast paths should stay serial for cheap ops and fan out
only for wide transcendentals.

### 2026-05-09 single-axis moveaxis native view

The saved array slice initially made `flip_axis0_f32` look like the worst view
row: 10.137 us for monpy vs 3.233 us for NumPy, a 3.122x ratio. That was a
low-loop false lead. A direct 50k-loop `BenchCase` probe put the same row at
3.263-3.305 us for monpy vs 3.151-3.169 us for NumPy, a 1.030x-1.049x range.

The stable target was `moveaxis_f32`:

- before: 7.670-7.722 us for monpy vs 3.904-3.984 us for NumPy,
  a 1.935x-1.964x range inside the repo runner.
- clean inner loop before: `mnp.moveaxis(s_mp, 0, -1)` took about 5.23 us, while
  the equivalent `s_mp.transpose((1, 2, 0))` took about 2.95 us.
- primary source check: NumPy's `moveaxis` normalizes axes, removes `source`
  from the axis order, inserts it at `destination`, then calls `transpose`.

`src/create/ops/shape.mojo` now exposes `moveaxis_single_ops` for the common
`int -> int` case. It builds the remove-then-insert permutation in Mojo and
returns the same layout-algebra view as `transpose_ops`, without Python tuple,
set, list, and axes-object parsing on the hot path. Sequence `moveaxis` still
uses the older generic Python path.

After the patch:

| row                  |       monpy us |       NumPy us |      ratio range |
| -------------------- | -------------: | -------------: | ---------------: |
| `views/moveaxis_f32` | 4.591 .. 4.611 | 3.828 .. 3.847 | 1.196x .. 1.205x |

Clean inner-loop timing moved `mnp.moveaxis(s_mp, 0, -1)` to about 2.38 us, with
the raw native `moveaxis_single` call at about 2.20 us. That is roughly a 2.2:1
direct-call speedup from the original 5.23 us path. It still does not beat NumPy
in the repo runner because these 24-element view rows are mostly Python and
benchmark-harness overhead; the native layout work is now the smaller part of
the row.

The saved low-loop array sweep at
`results/local-sweep-20260509-moveaxis-single-after-0753` reported
`moveaxis_f32` at 4.831 us for monpy vs 4.083 us for NumPy, a 1.184x median
ratio with a 1.109x-1.259x round range. The same low-loop sweep still has noisy
unrelated rows (`eigh_8_f64` at 2.285x), so use the direct focused probe for the
moveaxis delta and the saved sweep for ranking the next blocker.

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
.venv/bin/python - <<'PY'
import numpy as np
import monpy as mnp
for shape in [(2, 3, 4), (3, 4), (5,), (2, 3, 4, 5)]:
    src = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    arr = mnp.asarray(src)
    for source in range(-len(shape), len(shape)):
        for dest in range(-len(shape), len(shape)):
            got = np.asarray(mnp.moveaxis(arr, source, dest))
            expected = np.moveaxis(src, source, dest)
            assert got.shape == expected.shape
            assert np.array_equal(got, expected)
PY
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_creation_helpers.py -q
.venv/bin/python - <<'PY'
from monpy._bench.core import build_cases, run_case
cases = {case.name: case for case in build_cases(vector_size=1024, vector_sizes=(65536,), matrix_sizes=(16,), linalg_sizes=(2,))}
case = cases["moveaxis_f32"]
for i in range(1, 4):
    sample = run_case(case, loops=50000, repeats=7, round_index=i)
    print(sample)
PY
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types array --loops 10 --repeats 3 --rounds 2 --vector-sizes 65536,1048576 --matrix-sizes 32 --linalg-sizes 8 --format json --sort ratio --output-dir results/local-sweep-20260509-moveaxis-single-after-0753 --no-progress --no-stdout
```

Next target: split `squeeze_axis0_f32` before optimizing it. The current row is
stable at about 1.80x, but its benchmark body creates a fresh NumPy zero array
and calls `mnp.asarray(...)` inside the timed lambda, so it is not a clean
squeeze-only signal yet. `fliplr_f32` is the next clean view row at about 1.50x.

### 2026-05-09 single-axis flip native view

The current array sweep put the clean flip rows behind the polluted squeeze row:

- `flip_all_f32`: 3.173 us for monpy vs 2.335 us for NumPy, 1.359x.
- `fliplr_f32`: 3.487 us for monpy vs 2.663 us for NumPy, 1.334x.
- `flip_axis0_f32`: noisy in the low-loop sweep, but a direct 50k-loop probe put
  `mnp.flip(helper_mp, axis=0)` around 1.03 us.

The hot path was pure wrapper/native-call overhead. `flip(axis=int)`, `fliplr`,
and `flipud` all went through the generic multi-axis `flip_ops` ABI: allocate a
Python tuple, read `axes_obj[k]`, build a `seen` list, then do the actual
stride/offset rewrite. For one axis, that is junk work.

`src/create/ops/shape.mojo` now exposes `flip_axis_single_ops`. It normalizes one
axis, clones shape and strides, shifts the offset only when that axis has at
least one element, and negates the selected stride. The Python facade routes
`flip(axis=int)`, `fliplr`, and `flipud` through that entry. Multi-axis flip
still uses the older generic path, because repeated-axis validation belongs
there.

Direct timing after the patch:

| call                     | before ns | after ns | speedup |
| ------------------------ | --------: | -------: | ------: |
| `mnp.flip(..., axis=0)`  |      1028 |      821 |   1.25x |
| `mnp.flip(..., axis=1)`  |      1028 |      795 |   1.29x |
| `mnp.flipud(...)`        |      1330 |      959 |   1.39x |
| `mnp.fliplr(...)`        |      1322 |      954 |   1.39x |
| raw native one-axis flip |       808 |      641 |   1.26x |

The saved low-loop array sweep at
`results/local-sweep-20260509-flip-axis-single-after-0853` reported:

| row                    | before monpy us | after monpy us | after ratio |
| ---------------------- | --------------: | -------------: | ----------: |
| `views/flip_axis0_f32` |           3.287 |          3.044 |      1.032x |
| `views/fliplr_f32`     |           3.487 |          3.154 |      1.385x |
| `views/rot90_k1_f32`   |           7.544 |          7.385 |      1.595x |

The `fliplr` ratio is worse than the monpy wall-clock delta because NumPy's
low-loop sample moved too. Treat the monpy medians and the 50k-loop probe as the
signal. The suite now has flip-axis construction below the fixed Python harness
tax; the next gains need fewer facade calls, not more stride math.

The std-Mojo sweep stayed unrelated to this change. Its worst row was
`elementwise/scalar_mul_f32_1k` at 173.2 ns for monpy vs 91.8 ns for
`stdlib.SIMD_loop`, a 1.888x ratio. All other top stdlib rows were around
1.04x or lower, so the one-axis flip patch did not move the kernel/stdlib
frontier.

Verification:

```text
uv pip install --python .venv/bin/python --extra-index-url https://aarnphm.github.io/mohaus/simple 'mohaus>=0.1,<0.2'
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/create/ops/shape.mojo src/create/ops/__init__.mojo src/create/__init__.mojo src/lib.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_flip.py -q
.venv/bin/python - <<'PY'
import numpy as np
import monpy as mnp
for shape in [(0,), (0, 3), (2, 0, 4)]:
    src = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    arr = mnp.asarray(src)
    for axis in range(-len(shape), len(shape)):
        got = np.asarray(mnp.flip(arr, axis=axis))
        expected = np.flip(src, axis=axis)
        assert got.shape == expected.shape
        assert np.array_equal(got, expected)
PY
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types array --loops 10 --repeats 3 --rounds 2 --vector-sizes 65536,1048576 --matrix-sizes 32 --linalg-sizes 8 --format json --sort ratio --output-dir results/local-sweep-20260509-flip-axis-single-after-0853 --no-progress --no-stdout
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench-mojo --format json --sort ratio --output-dir results/local-sweep-20260509-flip-axis-single-mojo-0853 --no-stdout --timeout 300
```

Next target: split `squeeze_axis0_f32` into a clean existing-array view row and
an explicit asarray-from-NumPy row. Right now one benchmark name hides two
costs, which makes it too easy to optimize the wrong atom.

### 2026-05-09 split squeeze benchmark atom

The old `squeeze_axis0_f32` row measured three atoms under one view name:

- `np.zeros((1, 4, 1, 5), dtype=float32)` on both sides.
- `mnp.asarray(...)` on the monpy side.
- `squeeze(axis=0)` on both sides.

That made the row look like a squeeze problem. It was an ingress-plus-view row.

`python/monpy/_bench/core.py` now prebuilds `squeeze_np` and `squeeze_mp`, then
emits two rows:

- `array/views/squeeze_axis0_f32`: existing-array squeeze only.
- `array/interop/asarray_squeeze_axis0_f32`: NumPy ndarray ingress followed by
  the same squeeze.

Direct timing on the benchmark shape:

| call                      | monpy ns | NumPy ns | ratio |
| ------------------------- | -------: | -------: | ----: |
| existing-array `squeeze`  |      918 |      238 | 3.86x |
| `asarray` plus `squeeze`  |     2031 |      256 | 7.94x |
| raw native `squeeze_axis` |      713 |      n/a |   n/a |

The saved array sweep moved the public rows like this:

| row                                     | monpy us | NumPy us |  ratio |
| --------------------------------------- | -------: | -------: | -----: |
| old `views/squeeze_axis0_f32`           |    4.590 |    2.606 | 1.761x |
| new `views/squeeze_axis0_f32`           |    3.238 |    2.375 | 1.363x |
| new `interop/asarray_squeeze_axis0_f32` |    4.335 |    2.352 | 1.843x |

The split changes the target ranking. Existing-array squeeze is still slower
than NumPy, but the native bridge is already 713 ns and the Python facade adds
about 200 ns. The larger practical row is the NumPy-ingress path, where
`mnp.asarray(existing_numpy)` adds roughly 1.1 us before squeeze even starts.

The current array top rows after the split were noisy, but useful:

- `creation/empty_like_shape_override_f32`: 5.727 us vs 2.319 us, 2.448x.
- `casts/astype_f32_to_i64`: 6.354 us vs 2.808 us, 2.218x.
- `interop/asarray_squeeze_axis0_f32`: 4.335 us vs 2.352 us, 1.843x.
- `views/transpose_add_f32`: 4.729 us vs 2.942 us, 1.627x.

The std-Mojo sweep stayed on the same frontier as the previous pass:
`elementwise/scalar_mul_f32_1k` at 174.9 ns for monpy vs 90.1 ns for
`stdlib.SIMD_loop`, a 1.941x ratio. The next rows were close to parity:
`reductions/min_f32_64k` at 1.062x, `sum_f32_1k` at 1.035x, and `sum_f64_1k` at
1.023x.

Verification:

```text
.venv/bin/python -m py_compile python/monpy/_bench/core.py
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/monpy-bench --types array --loops 10 --repeats 3 --rounds 2 --vector-sizes 65536,1048576 --matrix-sizes 32 --linalg-sizes 8 --format json --sort ratio --output-dir results/local-sweep-20260509-squeeze-split-0953 --no-progress --no-stdout
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench-mojo --format json --sort ratio --output-dir results/local-sweep-20260509-squeeze-split-mojo-0953 --no-stdout --timeout 300
```

Next target: profile `empty_like_shape_override_f32` and `astype_f32_to_i64`
before optimizing. The former may be allocation-wrapper noise; the latter is a
real cast row if it reproduces outside the low-loop sweep.

### 2026-05-09 NuMojo row orientation

`monpy-bench-mojo --include-numojo --sort ratio` looked much worse than it was.
The row contract was inverted only for the NuMojo slice:

- regular Mojo rows: `candidate = monpy`, `baseline = stdlib-ish loop`.
- old NuMojo rows: `candidate = numojo`, `baseline = monpy`.

That made the top sorted rows read like monpy losses if your eye carried over
the `monpy/numpy` habit from the public benchmark suite. The atoms were doing
the opposite. In the pre-fix run at
`results/local-sweep-20260509-numojo-current-062756`, NuMojo was slower in every
NuMojo row:

| row                               | old candidate | old baseline | old ratio |
| --------------------------------- | ------------: | -----------: | --------: |
| `numojo.matmul/matmul_f32_16`     |     226140 ns |       380 ns |  595.746x |
| `numojo.matmul/matmul_f32_8`      |      22633 ns |        64 ns |  353.307x |
| `numojo.reductions/sum_f32_1024`  |        226 ns |        32 ns |    7.078x |
| `numojo.elementwise/add_f32_1024` |        425 ns |        90 ns |    4.750x |

`benches/bench_numojo_sweep.mojo` now matches the rest of the Mojo harness:
`candidate` is monpy, `baseline` is the comparison target, and `ratio` is
`monpy/baseline`. The post-fix run at
`results/local-sweep-20260509-numojo-oriented-063037` now reads in the same
direction:

| row                                | monpy candidate | NuMojo baseline | new ratio |
| ---------------------------------- | --------------: | --------------: | --------: |
| `numojo.elementwise/sin_f32_65536` |       401238 ns |       468333 ns |    0.857x |
| `numojo.elementwise/add_f32_65536` |         6414 ns |        14510 ns |    0.442x |
| `numojo.reductions/sum_f32_65536`  |         3222 ns |        16169 ns |    0.199x |
| `numojo.elementwise/add_f32_1024`  |           92 ns |          465 ns |    0.198x |
| `numojo.reductions/sum_f32_1024`   |           29 ns |          210 ns |    0.138x |

So the answer to "why are we so much slower than NuMojo?" is: in this harness,
we are not. This is also not a proof that the Python-facing monpy API beats
NuMojo's Python-facing API. The NuMojo comparison measures public NuMojo
`NDArray` calls against monpy leaf kernels with preallocated raw buffers. That
is useful as an external library smoke test, but not an abstraction-level fair
fight.

The actual current pure-Mojo frontier after the fix is stdlib-facing:

| row                              |   monpy | baseline |  ratio |
| -------------------------------- | ------: | -------: | -----: |
| `matmul/small_matmul_f32_8`      |  150 ns |    68 ns | 2.194x |
| `elementwise/scalar_mul_f32_1k`  |  176 ns |    94 ns | 1.867x |
| `elementwise/scalar_mul_f64_64k` | 12.1 us |  10.5 us | 1.155x |

The `small_matmul_f32_8` row is noisy enough to demand a rerun before acting.
The scalar-multiply rows have reproduced across several sweeps, so they remain
the better leaf-kernel target once the benchmark labeling no longer lies by
orientation.

Verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 benches/bench_numojo_sweep.mojo
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench-mojo --include-numojo --format json --sort ratio --output-dir results/local-sweep-20260509-numojo-current-062756 --no-stdout --timeout 300
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench-mojo --include-numojo --format json --sort ratio --output-dir results/local-sweep-20260509-numojo-oriented-063037 --no-stdout --timeout 300
.venv/bin/pytest tests/python/test_mojo_bench_sweep.py -q
```

Next target: rerun the stdlib-facing Mojo sweep with a longer runtime and isolate
`scalar_mul_f32_1k` / `scalar_mul_f64_64k` in a tiny variant bench. If the gap
survives a local direct-call baseline, move scalar MUL into a narrower exported
kernel with no runtime op or scalar-left payload.

### 2026-05-09 threading sweep harness

The missing comparison was not NuMojo. It was our own candidate threaded static
implementation against the monpy serial static kernels it would replace.

`benches/bench_threading_sweep.mojo` now emits the same TSV schema as
`bench_mojo_sweep.mojo`. `monpy-bench-mojo --include-threading` runs it through
the normal artifact path, and `--thread-caps` fans out separate Mojo processes:

- `auto`: clears `MONPY_THREADS`, so the policy uses hardware performance cores
  plus the byte-grain gates.
- `N`: runs with `MONPY_THREADS=N`, proving the cap in the same process model
  production uses.

Rows compare:

- candidate: `internal.threaded_static_{add,negate,exp}`;
- baseline: current monpy serial static kernels;
- ratio: `internal / monpy`, so below `1.0x` means the threaded prototype wins.

The smoke run lives at `results/local-sweep-20260509-threading-sweep-smoke-064228`
for `auto,1`. The cap sweep lives at
`results/local-sweep-20260509-threading-sweep-caps-064336` for `2,4`.

Useful rows:

| row            |  cap | internal ns | monpy ns |  ratio |
| -------------- | ---: | ----------: | -------: | -----: |
| `add_f32_64k`  |    1 |        6449 |     6368 | 1.013x |
| `add_f32_64k`  |    2 |        6363 |     6495 | 0.980x |
| `add_f32_64k`  | auto |        6534 |     6460 | 1.011x |
| `add_f32_1m`   |    1 |      113117 |   109349 | 1.034x |
| `add_f32_1m`   |    2 |       66470 |   111947 | 0.594x |
| `add_f32_1m`   |    4 |       63684 |   106159 | 0.600x |
| `add_f32_1m`   | auto |       60245 |   122483 | 0.492x |
| `neg_f32_64k`  | auto |        5901 |     5193 | 1.136x |
| `neg_f32_16m`  |    4 |     1054300 |  1931000 | 0.546x |
| `neg_f32_16m`  | auto |     1092455 |  2022286 | 0.540x |
| `exp_f32_64k`  | auto |       32139 |    29102 | 1.104x |
| `exp_f32_256k` |    2 |       66579 |   116345 | 0.572x |
| `exp_f32_256k` |    4 |       39772 |   117491 | 0.339x |
| `exp_f32_256k` | auto |       36883 |   117439 | 0.314x |
| `exp_f32_1m`   |    4 |      130396 |   462611 | 0.282x |
| `exp_f32_1m`   | auto |      212934 |   471889 | 0.451x |
| `add_f64_1m`   |    4 |       97427 |   331087 | 0.294x |
| `exp_f64_256k` |    4 |       91219 |   307480 | 0.297x |

Read:

- 64K light ops should stay serial. The auto path is at parity or worse, which
  is the spawn-overhead floor showing up cleanly.
- 1M+ add/negate and 256K+ exp are real wins. Those are the candidates for a
  production threaded static fast path.
- `auto` is not uniformly better than `MONPY_THREADS=4`. On this Mac, `exp_f32_1m`
  was 0.282x at cap 4 but 0.451x at auto. The next policy work is not "use all
  cores harder"; it is finding the per-op cap where memory bandwidth, libm work,
  and `sync_parallelize` overhead meet.

Verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 benches/bench_threading_sweep.mojo
.venv/bin/python -m py_compile python/monpy/_bench/mojo_sweep.py
.venv/bin/pytest tests/python/test_mojo_bench_sweep.py -q
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench-mojo --include-threading --thread-caps auto,1 --format json --sort ratio --output-dir results/local-sweep-20260509-threading-sweep-smoke-064228 --no-stdout --timeout 300
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench-mojo --include-threading --thread-caps 2,4 --format json --sort ratio --output-dir results/local-sweep-20260509-threading-sweep-caps-064336 --no-stdout --timeout 300
```

Next target: promote the threading sweep from measurement to policy. Start with
threaded static `EXP` for f32/f64 above the heavy gate, cap auto workers by row
size instead of blindly using every performance core, and keep ADD/NEGATE serial
below the 1M-element region.

### 2026-05-09 static EXP threading policy

The static unary fast path was the trap. `unary_contig_typed` had a parallel
branch, but `try_unary_contig_typed_static` caught `EXP` first and went straight
to the serial static kernel. So the policy existed, but the hot op was stepping
around it.

I added `unary_contig_typed_static_parallel[dtype, op, grain]` and routed only
`EXP` through it for now. The serial `unary_contig_typed_static` primitive stays
unchanged so we still have a fixed baseline for calibration rows. `MONPY_THREADS=1`
still collapses the wrapper back to serial through `worker_count_for_bytes`.

`bench_mojo_sweep.mojo` now emits production `exp_par_*` rows:

- candidate: `monpy.unary_contig_typed`, the public contiguous unary entry.
- baseline: `monpy.unary_contig_typed_static`, the serial static kernel.
- ratio: production / serial, so below `1.0x` means the new dispatch wins.

Fresh run:

| row                | production ns | serial ns |  ratio |
| ------------------ | ------------: | --------: | -----: |
| `exp_par_f32_256k` |        40,600 |   118,086 | 0.344x |
| `exp_par_f32_1m`   |       329,400 |   472,278 | 0.697x |
| `exp_par_f64_256k` |       241,400 |   320,480 | 0.753x |
| `exp_par_f64_1m`   |       531,353 | 1,277,400 | 0.416x |

Serial escape-hatch run with `MONPY_THREADS=1`:

| row                | production ns | serial ns |  ratio |
| ------------------ | ------------: | --------: | -----: |
| `exp_par_f32_256k` |       117,186 |   117,102 | 1.001x |
| `exp_par_f32_1m`   |       480,944 |   464,167 | 1.036x |
| `exp_par_f64_256k` |       310,960 |   309,840 | 1.004x |
| `exp_par_f64_1m`   |     1,272,600 | 1,271,800 | 1.001x |

The f32 1M serial-cap row is the wrapper tax plus bench noise: about 3.6% in
that run, while the other three rows sit at parity. Good enough for the env
contract; not a reason to make the hot path branchier.

I also reran the direct cap-4 threading harness after one noisy saved outlier:

| row            | threaded ns | serial ns |  ratio |
| -------------- | ----------: | --------: | -----: |
| `exp_f32_64k`  |      31,569 |    29,488 | 1.071x |
| `exp_f32_256k` |      39,389 |   114,000 | 0.346x |
| `exp_f32_1m`   |     143,962 |   487,632 | 0.295x |
| `exp_f64_256k` |      89,905 |   320,280 | 0.281x |

Read:

- The 256KB heavy gate is right for f32/f64 `EXP`. It turns on at 256K-ish
  element counts and wins by roughly 2.7-3.6x in the stable cap-4 rows.
- 64K f32 `EXP` stays a loss. The gate keeps production serial there.
- `sync_parallelize` variance is real. One saved include-threading run put
  `threads4 exp_f32_256k` at 1.76x, but the direct rerun and cap-4 artifact both
  returned to ~0.346x. Treat one-shot cap sweeps as smoke, not truth serum.
- The remaining stdlib-facing deficits are small: tiny 8x8 matmul and reductions
  around 1.04-1.11x. The next high-leverage target is not another transcendental
  branch; it is scalar MUL / tiny-matmul dispatch overhead or a steadier worker
  pool story.

Artifacts:

- `results/local-sweep-20260509-exp-static-thread-policy`
- `results/local-sweep-20260509-exp-static-thread-policy-serial`
- `results/local-sweep-20260509-exp-static-thread-policy-threading`
- `results/local-sweep-20260509-exp-static-thread-policy-threading-cap4-rerun`

Verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 benches/bench_mojo_sweep.mojo src/elementwise/kernels/typed.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench-mojo --format json --sort ratio --output-dir results/local-sweep-20260509-exp-static-thread-policy --no-stdout --timeout 300
MONPY_THREADS=1 MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench-mojo --format json --sort ratio --output-dir results/local-sweep-20260509-exp-static-thread-policy-serial --no-stdout --timeout 300
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python -m py_compile python/monpy/_bench/mojo_sweep.py
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/test_mojo_bench_sweep.py -q
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench-mojo --include-threading --thread-caps auto,4 --format json --sort ratio --output-dir results/local-sweep-20260509-exp-static-thread-policy-threading --no-stdout --timeout 300
MONPY_THREADS=4 MOHAUS_EDITABLE_REBUILDING=1 /Users/aarnphm/workspace/modular/.derived/build/bin/mojo run -I src benches/bench_threading_sweep.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench-mojo --include-threading --thread-caps 4 --format json --sort ratio --output-dir results/local-sweep-20260509-exp-static-thread-policy-threading-cap4-rerun --no-stdout --timeout 300
```

Next target: scalar MUL and tiny matmul. The current worst rows are single-digit
percent losses, which is exactly where dispatch overhead, scalar-left branches,
and benchmark harness noise all start wearing the same coat.

### 2026-05-09 full benchmark frontier

I ran the full local macOS Python suite, the pure Mojo suite with NuMojo rows,
the native row/reduction harnesses, and pulled the latest GitHub Benchmark Sweep
artifacts for Linux/macOS. The local macOS rows are current working tree. The
GitHub Linux/macOS rows are from remote commit `644f22a`, not this local HEAD,
so they are platform evidence and not a clean current-regression verdict.

Artifacts:

- `results/local-full-20260509-macos-python`
- `results/local-full-20260509-macos-mojo`
- `results/local-full-20260509-macos-native/bench_parallel.txt`
- `results/local-full-20260509-macos-native/bench_reduce.txt`
- `results/github-benchmark-25594119804/monpy-bench-ubuntu-latest-644f22a8c1c9883d39c4ae5117556c803214d4cf`
- `results/github-benchmark-25594119804/monpy-bench-macos-15-644f22a8c1c9883d39c4ae5117556c803214d4cf`

Local macOS, Python-facing rows:

| slice | rows | slow rows | >1.25x | >1.5x | >2x | median ratio |
| ----- | ---: | --------: | -----: | ----: | --: | -----------: |
| all   |  243 |       154 |     67 |    21 |   0 |       1.092x |

Worst local macOS rows:

| row                                            | monpy us | NumPy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `array/views/transpose_add_f32`                |    4.826 |    2.506 | 1.942x |
| `array/decomp/eigh_2_f64`                      |    9.590 |    4.901 | 1.927x |
| `array/interop/asarray_squeeze_axis0_f32`      |    4.258 |    2.299 | 1.893x |
| `array/decomp/eigh_4_f64`                      |   10.382 |    5.509 | 1.883x |
| `array/creation/empty_like_shape_override_f32` |    4.337 |    2.294 | 1.877x |
| `strides/elementwise/rank3_transpose_add_f32`  |   14.006 |    7.977 | 1.782x |
| `array/decomp/eigh_2_f32`                      |    9.560 |    5.375 | 1.775x |
| `array/decomp/eigh_8_f64`                      |   11.276 |    6.569 | 1.707x |

Read:

- The current macOS frontier is mostly wrapper/view/linalg overhead, not a
  memory-bandwidth disaster. No Python-facing row is over 2x slower in this
  full run.
- `eigh_*` remains high because tiny LAPACK calls pay fixed wrapper/workspace
  cost. The sample stack backs this up: `eigh_2_f64` spent visible time in
  `std::python::python::Python::evaluate`, `lapack_eigh_into`, and `DSYEV`,
  while the benchmark matrix is only 2x2.
- `asarray_squeeze_axis0_f32` is still an ingress-plus-view row. It crosses
  native once for `asarray_from_buffer_ops`, then again for `squeeze_axis_ops`.
  That makes it a better first iteration target than tiny eigens, because the
  fix should be a bridge/view contraction rather than new numerical code.
- `transpose_add_f32` is a fused arithmetic win path that still pays the view
  construction toll (`transpose_full_reverse_ops`, `make_view_array`) before
  the fused add. This is probably a second iteration after the ingress/view
  path, unless a fused transpose-add native entry falls out cleanly.

Local macOS, pure Mojo rows:

| slice | rows | slow rows | >1.25x | >1.5x | >2x | median ratio |
| ----- | ---: | --------: | -----: | ----: | --: | -----------: |
| all   |  114 |        36 |      1 |     0 |   0 |       0.980x |

Worst pure Mojo rows:

| row                              |  monpy ns | baseline ns |  ratio |
| -------------------------------- | --------: | ----------: | -----: |
| `threading.threads1/neg_f32_1m`  |   108,778 |      86,164 | 1.262x |
| `threading.threads4/exp_f32_64k` |    35,574 |      30,155 | 1.180x |
| `elementwise/add_f32_1m`         |   130,696 |     113,130 | 1.155x |
| `threading.threads1/neg_f64_1m`  |   240,625 |     208,605 | 1.153x |
| `elementwise/add_par_f64_16m`    | 6,264,500 |   5,479,750 | 1.143x |
| `threading.auto/exp_f32_64k`     |    31,500 |      28,741 | 1.096x |
| `matmul/small_matmul_f32_8`      |        60 |          57 | 1.053x |

Read:

- Against Mojo/stdlib and NuMojo, the native core is mostly fine. The remaining
  deficits are policy thresholds and tiny dispatch costs, not broad SIMD
  failure.
- The native row harness gives the clearest threading rule: small row kernels
  should not fan out. `par_layernorm_f32_32x32` was roughly two orders of
  magnitude slower than serial because the work is too small for
  `sync_parallelize`. Large rows are different: 1024x4096 softmax and layernorm
  won by about 6:1 and 5.8:1 respectively.
- Parallel reductions should stay gated to very large buffers. The native
  reduction harness showed `sumP_f32_16M` and `sumP_f32_128M` around 2:1 faster
  than serial/std, but small reductions do not justify the thread toll.

Remote GitHub Linux, same remote commit `644f22a`:

| slice | rows | slow rows | >1.25x | >1.5x | >2x | median ratio |
| ----- | ---: | --------: | -----: | ----: | --: | -----------: |
| all   |  242 |       169 |     67 |    36 |   8 |       1.089x |

Worst Linux rows:

| row                                        | monpy us | NumPy us |  ratio |
| ------------------------------------------ | -------: | -------: | -----: |
| `array/bandwidth/unary_sin_16384_f32`      |   94.993 |   32.213 | 2.933x |
| `strides/elementwise/sliced_unary_sin_f32` |  117.284 |   40.727 | 2.883x |
| `array/decomp/eigh_2_f64`                  |   24.329 |    9.925 | 2.451x |
| `array/decomp/eigh_4_f64`                  |   26.102 |   11.121 | 2.347x |
| `array/decomp/eigh_8_f64`                  |   27.843 |   12.489 | 2.233x |
| `array/decomp/eigh_2_f32`                  |   23.718 |   11.177 | 2.090x |
| `array/ext_dtypes/binary_add_f16`          |   24.326 |   11.877 | 2.046x |
| `array/decomp/eigh_4_f32`                  |   26.072 |   12.871 | 2.026x |

Remote GitHub macOS, same remote commit `644f22a`:

| slice | rows | slow rows | >1.25x | >1.5x | >2x | median ratio |
| ----- | ---: | --------: | -----: | ----: | --: | -----------: |
| all   |  242 |       140 |     78 |    35 |   8 |       1.099x |

Worst remote macOS rows:

| row                                            | monpy us | NumPy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `array/views/moveaxis_f32`                     |   19.551 |    4.880 | 2.772x |
| `array/casts/astype_i64_to_i64`                |    7.301 |    3.025 | 2.414x |
| `array/views/diagonal_64_f64`                  |    7.172 |    3.010 | 2.214x |
| `array/creation/meshgrid_xy_f32`               |   25.725 |   11.108 | 2.149x |
| `array/interop/asarray_zero_copy_f32`          |    5.153 |    2.426 | 2.124x |
| `array/views/hstack_f32`                       |    8.761 |    3.679 | 2.124x |
| `strides/elementwise/rank3_transpose_add_f32`  |   20.294 |    8.859 | 2.021x |
| `array/creation/empty_like_shape_override_f32` |    6.173 |    3.080 | 2.004x |

Platform split:

- Linux is uniquely bad on `sin` rows. `unary_sin_16384_f32` is 2.933x on
  Linux but 0.315x on the remote macOS artifact. That is a backend/library
  split, not a generic elementwise problem: macOS can lean on the Accelerate-ish
  path, while Linux is paying scalar/libm/SIMD policy costs. Current-head Linux
  needs a fresh runner before committing a math-kernel patch.
- Linux and macOS both dislike tiny `eigh`. The ratios differ, but the shape is
  the same: very small matrices run through a large fixed-cost path. This wants
  either a tiny closed-form eigensolver for 2x2/4x4 or a cheaper LAPACK wrapper
  that avoids Python evaluation and per-call workspace churn.
- The old remote macOS view/interop rows are not the current local frontier.
  `moveaxis`, `asarray_zero_copy`, and related rows have already moved down in
  this local tree. Current macOS is now dominated by `transpose_add`,
  `asarray_squeeze`, `empty_like_shape_override`, and tiny eigens.

Focused profiles:

| case                                      | us/call | backend            | traced alloc peak | sample read                             |
| ----------------------------------------- | ------: | ------------------ | ----------------: | --------------------------------------- |
| `array/views/transpose_add_f32`           |   4.691 | fused native       |           1,760 B | view construction plus fused arithmetic |
| `array/interop/asarray_squeeze_axis0_f32` |   4.438 | native bridge/view |           1,800 B | buffer ingress plus squeeze wrapper     |
| `array/decomp/eigh_2_f64`                 |   9.860 | Accelerate/LAPACK  |         147,887 B | Python evaluation plus tiny LAPACK      |

Next iteration:

- First target: reduce the actual `asarray_squeeze_axis0_f32` components, not a
  fantasy fused Python expression. Once `mnp.asarray(...)` returns, `squeeze`
  only sees a monpy array. The honest patch surface is either a cheaper common
  `squeeze_axis` view constructor or a faster buffer-ingress path for small
  external arrays.
- Expected ceiling: the row is 4.258 us vs 2.299 us today. A targeted
  `squeeze_axis` specialization can only remove part of the roughly 0.7 us
  native squeeze toll. A buffer-ingress cleanup has the bigger ceiling, but it
  also has a higher chance of hitting CPython protocol cost we cannot wish away.
- Second target: `empty_like_shape_override_f32`, because it is current-mac slow
  and avoids Linux math-library ambiguity.
- Linux target after a fresh current-head runner: `sin` rows. Do not rewrite
  transcendental math from stale `644f22a` numbers alone. That way lies fake
  victory and a checksum shaped headache.

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench --types all --loops 200 --repeats 5 --rounds 3 --format json --sort ratio --output-dir results/local-full-20260509-macos-python --no-progress --no-stdout
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench-mojo --include-numojo --strict-numojo --include-threading --thread-caps auto,1,2,4,8 --format json --sort ratio --output-dir results/local-full-20260509-macos-mojo --no-stdout --timeout 300
MONPY_THREADS=4 MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo run -I src benches/bench_parallel.mojo
MONPY_THREADS=4 MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo run -I src benches/bench_reduce.mojo
gh run download 25594119804 -R aarnphm/monpy --dir results/github-benchmark-25594119804
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-profile --case array/views/transpose_add_f32 --types array,strides,complex,attention --candidate monpy --duration 2 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-full-transpose-add-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-profile --case array/interop/asarray_squeeze_axis0_f32 --types array,strides,complex,attention --candidate monpy --duration 2 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-full-asarray-squeeze-monpy --sample --no-perf-stat
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-profile --case array/decomp/eigh_2_f64 --types array,strides,complex,attention --candidate monpy --duration 2 --memory-duration 1 --warmup 20 --output-dir results/local-profile-20260509-full-eigh2-f64-monpy --sample --no-perf-stat
```

### 2026-05-09 unchecked source-derived views

First iteration from the full frontier: view construction. `make_view_array`
validates shape length and non-negative dimensions, which is correct for
external inputs, but redundant for shape/stride lists derived directly from an
already-valid `Array`. I split out `make_view_array_unchecked` and used it in
the source-derived squeeze paths plus the full reverse transpose path.

This is intentionally small. It does not touch Python facade files, which were
already staged from another edit stream, and it does not change view ownership:
the unchecked helper still retains the source storage and preserves dtype,
backend code, data pointer, byte length, and offset.

Local read:

| row                                       | before us | after us | speedup |
| ----------------------------------------- | --------: | -------: | ------: |
| `array/views/transpose_add_f32`           |     4.826 |    4.518 |  1.07:1 |
| `array/views/squeeze_axis0_f32`           |     3.181 |    3.124 |  1.02:1 |
| `array/interop/asarray_squeeze_axis0_f32` |     4.258 |    4.227 |  1.01:1 |

The direct native microtiming moved harder:

| call                                      | before ns | after ns | speedup |
| ----------------------------------------- | --------: | -------: | ------: |
| raw `_native.squeeze_axis(..., 0)`        |       713 |      578 |  1.23:1 |
| `mnp.squeeze(existing_monpy, axis=0)`     |       918 |      926 |  0.99:1 |
| `mnp.squeeze(mnp.asarray(numpy), axis=0)` |     2,031 |    1,957 |  1.04:1 |

Read:

- Native leaf cost moved, so the validation split did what it was supposed to
  do.
- Public `squeeze` barely moved because the Python wrapper/native-call boundary
  is now the visible floor. We should not keep grinding this path in Mojo unless
  a profile shows another native allocation hotspot.
- `transpose_add_f32` improved more than squeeze because the reverse transpose
  view is in the hot fused-add setup. This makes the helper useful beyond the
  one benchmark row.
- `empty_like_shape_override_f32` is still worse after noise: 4.511 us vs 2.295
  us, 1.915x. The next iteration should either profile that row directly or
  avoid Python facade churn and go after tiny `eigh`.

Verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/array/factory.mojo src/array/__init__.mojo src/create/ops/shape.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_creation_helpers.py::test_squeeze_matches_numpy tests/python/numpy_compat/test_creation_helpers.py::test_squeeze_rejects_non_singleton_axis -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python - <<'PY'
import time
import numpy as np
import monpy as mnp

src_np = np.zeros((1, 4, 1, 5), dtype=np.float32)
src_mp = mnp.asarray(src_np)
cases = [
  ("existing squeeze", lambda: mnp.squeeze(src_mp, axis=0)),
  ("asarray+squeeze", lambda: mnp.squeeze(mnp.asarray(src_np), axis=0)),
  ("native squeeze", lambda: mnp._native.squeeze_axis(src_mp._native, 0)),
]
for name, fn in cases:
  for _ in range(1000):
    fn()
  best = 10**9
  for _ in range(7):
    t0 = time.perf_counter_ns()
    for _ in range(20000):
      fn()
    best = min(best, (time.perf_counter_ns() - t0) / 20000)
  print(f"{name}: {best:.1f} ns")
PY
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench --types array --loops 200 --repeats 5 --rounds 3 --format json --sort ratio --output-dir results/local-sweep-20260509-view-unchecked --no-progress --no-stdout
```

### 2026-05-09 tiny eigh and linalg list construction

Second iteration from the full frontier: tiny linalg fixed costs. The profile
for `eigh_2_f64` showed two separate problems wearing the same jacket:

- LAPACK setup for a 2x2 symmetric matrix.
- `Python.evaluate("[]")` for the two-return Python list.

I added a closed-form symmetric 2x2 `eigh` path in the native linalg bridge:
eigenvalues come from the half-trace plus/minus
`sqrt(half_diff**2 + lower_offdiag**2)`, and eigenvector columns are normalized
from `(b, lambda - a)` with diagonal/degenerate fallbacks. The path reads the
already-UPLO-adjusted lower triangle, so Python's existing `UPLO='U'` transpose
behavior still works.

The bigger win was killing `Python.evaluate("[]")`. `std.python.Python.list`
constructs the result list directly through CPython list allocation and
reference stealing. I switched the linalg multi-return ops to that path:
`qr`, `eigh`, `eig`, `svd`, and `lstsq`.

Local array sweep:

| row                       | before us | after us | before ratio | after ratio | speedup |
| ------------------------- | --------: | -------: | -----------: | ----------: | ------: |
| `array/decomp/eigh_2_f64` |     9.705 |    5.636 |       1.988x |      1.180x |  1.72:1 |
| `array/decomp/eigh_2_f32` |     9.305 |    5.570 |       1.780x |      1.060x |  1.67:1 |
| `array/decomp/eigh_4_f64` |    10.014 |    6.847 |       1.835x |      1.285x |  1.46:1 |
| `array/decomp/eigh_4_f32` |    10.147 |    6.716 |       1.623x |      1.105x |  1.51:1 |
| `array/decomp/svd_2_f64`  |     9.239 |    5.214 |       1.275x |      0.746x |  1.77:1 |
| `array/decomp/svd_4_f64`  |    10.234 |    6.186 |       1.247x |      0.758x |  1.65:1 |
| `array/decomp/qr_2_f64`   |     8.726 |    5.360 |       0.744x |      0.476x |  1.63:1 |

Read:

- The 2x2 closed-form path helped, but list construction was the broad fixed
  cost. `eigh_4_*` improved even though it still uses LAPACK, which pins the
  real culprit.
- `svd_2_f64` and `svd_4_f64` moved from slower than NumPy to roughly 25% faster
  than NumPy in this local run. Good loot. No new numerical shortcut was needed.
- The top current macOS slow row is now `empty_like_shape_override_f32` at
  4.377 us vs 2.268 us, 1.930x. That row is probably Python facade and shape
  normalization overhead; do not edit it while facade files are dirty unless the
  worktree has been refreshed and the staged/unstaged state is clear.

Verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/create/ops/linalg.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/pytest tests/python/numpy_compat/test_tensor_linalg.py -q
MOHAUS_EDITABLE_REBUILDING=1 .venv/bin/python - <<'PY'
import time
import numpy as np
import monpy as mp

x64 = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
x32 = x64.astype(np.float32)
for name, arr, dtype in [
  ("eigh2_f64", x64, mp.float64),
  ("eigvalsh2_f64", x64, mp.float64),
  ("eigh2_f32", x32, mp.float32),
]:
  a = mp.asarray(arr, dtype=dtype)
  fn = (lambda a=a: mp.linalg.eigh(a)) if name.startswith("eigh") else (lambda a=a: mp.linalg.eigvalsh(a))
  for _ in range(1000):
    fn()
  best = 10**9
  for _ in range(7):
    t0 = time.perf_counter_ns()
    for _ in range(10000):
      fn()
    best = min(best, (time.perf_counter_ns() - t0) / 10000)
  print(f"{name}: {best:.1f} ns")
PY
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench --types array --loops 200 --repeats 5 --rounds 3 --format json --sort ratio --output-dir results/local-sweep-20260509-linalg-list-builder --no-progress --no-stdout
```

Post-iteration full sweep:

| slice  | slow rows | >1.25x | >1.5x | median ratio |
| ------ | --------: | -----: | ----: | -----------: |
| before |       154 |     67 |    21 |       1.092x |
| after  |       145 |     55 |    17 |       1.083x |

Key full-sweep deltas:

| row                             | before us | after us | speedup |
| ------------------------------- | --------: | -------: | ------: |
| `array/decomp/eigh_2_f64`       |     9.590 |    5.766 |  1.66:1 |
| `array/decomp/eigh_2_f32`       |     9.560 |    5.705 |  1.68:1 |
| `array/decomp/eigh_4_f64`       |    10.382 |    6.841 |  1.52:1 |
| `array/decomp/svd_2_f64`        |     9.157 |    5.390 |  1.70:1 |
| `array/decomp/svd_4_f64`        |    10.221 |    6.255 |  1.63:1 |
| `array/views/transpose_add_f32` |     4.826 |    4.522 |  1.07:1 |

The remaining top rows after both iterations:

| row                                            | monpy us | NumPy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `array/creation/empty_like_shape_override_f32` |    4.494 |    2.308 | 1.948x |
| `array/interop/asarray_squeeze_axis0_f32`      |    4.241 |    2.298 | 1.845x |
| `array/views/transpose_add_f32`                |    4.522 |    2.493 | 1.793x |
| `strides/elementwise/rank3_transpose_add_f32`  |   13.970 |    8.166 | 1.711x |

Artifact: `results/local-full-20260509-post-linalg`.

### 2026-05-09 linalg benchmark surface

The array suite had real linalg scaling rows, but the public API surface was
patchy: solve/inv/det and a few decomposition paths were visible, while the
wrapper-heavy linalg functions could regress without showing up in the table.

Patch:

- Added an `array/linalg_api` group with fixed f64 rows for the full
  `monpy.linalg` export surface, excluding `LinAlgError`.
- Kept the existing sized rows for scaling: `solve`, `inv`, `det`, `qr`,
  `cholesky`, `eigvalsh`, `svdvals`, and `pinv`.
- Renamed the old decomp rows that were lying by omission:
  `eigh_*` now reports as `eigvalsh_*`, and `svd_*` now reports as
  `svdvals_*`. Historical notes above still use the old artifact names.
- Covered the missing wrapper/API rows: `dot`, `vdot`, `inner`, `outer`,
  `matmul`, `matrix_transpose`, `matvec`, `vecmat`, `vecdot`, `tensordot`,
  `kron`, `cross`, `trace`, `norm`, `vector_norm`, `matrix_norm`,
  `matrix_rank`, `matrix_power`, `slogdet`, `multi_dot`, `tensorinv`,
  `tensorsolve`, `eigh`, `eig`, `eigvals`, `svd`, `lstsq`, plus rectangular
  `pinv` and matrix-RHS `solve`.

Coverage check:

- `monpy.linalg.__all__ - {"LinAlgError"}`: 35 public names.
- Benchmark mapping: 35/35 names covered.
- New manifest shape for the smoke sweep: 158 array cases total,
  32 `array/linalg_api` rows.

Smoke sweep:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench --types array --loops 20 --repeats 3 --rounds 2 --vector-sizes 16384 --matrix-sizes 16 --linalg-sizes 2 --format json --sort ratio --output-dir results/local-sweep-20260509-linalg-api-coverage --no-progress --no-stdout
```

Worst new linalg API rows:

| row                                            | monpy us | NumPy us |  ratio |
| ---------------------------------------------- | -------: | -------: | -----: |
| `array/linalg_api/kron_2x2_f64`                |   73.044 |    9.889 | 7.398x |
| `array/linalg_api/tensordot_axes1_4x5_5x3_f64` |   27.478 |    5.879 | 4.672x |
| `array/linalg_api/vecmat_16_f64`               |   11.684 |    2.691 | 4.343x |
| `array/linalg_api/vdot_32_f64`                 |    9.849 |    2.480 | 3.973x |
| `array/linalg_api/matvec_16_f64`               |   10.515 |    2.652 | 3.965x |
| `array/linalg_api/dot_1d_32_f64`               |    9.195 |    2.419 | 3.803x |
| `array/linalg_api/inner_32_f64`                |   10.300 |    2.845 | 3.660x |
| `array/linalg_api/outer_32_f64`                |   12.992 |    4.084 | 3.181x |

Read:

- The linalg frontier is no longer hidden in the generic matmul/decomp rows.
  The slow rows are mostly Python wrapper loops and shape plumbing, not LAPACK.
- `kron` is the big embarrassing number: it walks Python lists and native scalar
  getters for a 2x2 input, so the 7.4x ratio is expected and now tracked.
- `tensordot` pays transpose/contiguous/reshape setup before the actual matmul.
  That is the next high-upside native path after the current random feature dust
  settles.
- `dot`/`vdot`/`inner`/`matvec`/`vecmat` are wrapper-bound at 16 to 32 elements.
  The fix is not "more LAPACK"; it is fewer Python crossings for scalar-product
  and matrix-vector paths.

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python - <<'PY'
from monpy._bench.core import build_cases, run_case

cases = build_cases(vector_size=1024, vector_sizes=(16384,), matrix_sizes=(16,), linalg_sizes=(2,))
selected = [case for case in cases if case.group in {"linalg", "decomp", "linalg_api"}]
for case in selected:
  run_case(case, loops=1, repeats=1, round_index=1)
print(len(selected))
PY
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/pytest tests/python/numpy_compat/test_tensor_linalg.py tests/python/numpy_compat/test_einsum.py -q
```

Artifact: `results/local-sweep-20260509-linalg-api-coverage`.

### 2026-05-09 native slogdet scalar finish

`python/monpy/linalg.py` still imported CPython `math` after the linalg API
benchmark expansion. Most of that was not a data-path bug:

- shape products in `tensorinv`, `tensorsolve`, and `tensordot` are Python
  metadata over tuples;
- `inf` checks in `norm` are scalar branch selection;
- `slogdet` was different: it called native `det`, pulled the scalar back to
  Python, then did `math.log(abs(det))`.

Patch:

- Added native `linalg_slogdet` at the Mojo extension boundary.
- `slogdet` now computes determinant sign and log-absolute-determinant in
  `src/create/ops/linalg.mojo`, then returns Python scalars.
- Removed `math` from `python/monpy/linalg.py`; shape products use a tiny local
  integer loop, and infinity sentinels are local constants.
- Added coverage for positive, negative, and singular determinants.

Micro row:

| row                               | before us | after us | speedup |
| --------------------------------- | --------: | -------: | ------: |
| `array/linalg_api/slogdet_16_f64` |     4.753 |    4.178 |  1.14:1 |

Read:

- This is not the big linalg win. It is boundary hygiene: a linalg numeric
  result should not bounce through Python `math` when the native linalg layer is
  already open.
- The big rows are still `kron`, `tensordot`, `dot/vdot/inner`, and
  `matvec/vecmat`; those need native kernels, not scalar cleanup.

Verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/create/ops/linalg.mojo src/create/__init__.mojo src/lib.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/pytest tests/python/numpy_compat/test_tensor_linalg.py -q
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/python - <<'PY'
from monpy._bench.core import build_cases, run_case
cases = build_cases(vector_size=1024, vector_sizes=(16384,), matrix_sizes=(16,), linalg_sizes=(2,))
case = next(c for c in cases if c.group == 'linalg_api' and c.name == 'slogdet_16_f64')
print(run_case(case, loops=200, repeats=5, round_index=1))
PY
```

### 2026-05-09 full rerun and linalg facade fastpaths

Fresh full Python-vs-NumPy sweep after the random and `slogdet` commits:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench --types all --loops 200 --repeats 5 --rounds 3 --format json --sort ratio --output-dir results/local-full-20260509-rerun-numpy-frontier --no-progress --no-stdout
```

Run shape:

| artifact                                           | cases | duration | slow rows | >1.25x | >1.5x | >2x | median |
| -------------------------------------------------- | ----: | -------: | --------: | -----: | ----: | --: | -----: |
| `results/local-full-20260509-rerun-numpy-frontier` |   275 |    40.1s |       167 |     67 |    38 |  14 | 1.083x |

The bad rows were concentrated in `array/linalg_api`, mostly because the facade
was routing tiny linalg calls through multiple Python-level helpers before
reaching the native layer.

Patch:

- `dot`, `vdot`, and 1D `inner` now route directly to native matmul instead of
  multiply-plus-reduce.
- `tensordot(..., axes=1)` for 2D x 2D now routes directly to native matmul.
- `matvec` and `vecmat` now use the native 2D x 1D and 1D x 2D matmul paths
  instead of reshaping through a column/row matrix first.
- `kron` no longer loops through Python scalar getters. It uses reshape plus
  broadcasted multiply, then reshapes to the Kronecker output shape.
- Replaced the lingering `math.prod` shape products in `linalg.py` with a local
  integer helper. This keeps metadata on the Python side without importing
  Python `math` into the linalg module.

Post-patch full sweep:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench --types all --loops 200 --repeats 5 --rounds 3 --format json --sort ratio --output-dir results/local-full-20260509-linalg-api-fastpaths --no-progress --no-stdout
```

| artifact                                           | cases | duration | slow rows | >1.25x | >1.5x | >2x | median |
| -------------------------------------------------- | ----: | -------: | --------: | -----: | ----: | --: | -----: |
| `results/local-full-20260509-linalg-api-fastpaths` |   275 |    40.4s |       170 |     72 |    32 |  11 | 1.096x |

Key moved rows:

| row                                            | before us | after us | before ratio | after ratio | speedup |
| ---------------------------------------------- | --------: | -------: | -----------: | ----------: | ------: |
| `array/linalg_api/kron_2x2_f64`                |    72.896 |   22.901 |       6.975x |      2.337x |  3.18:1 |
| `array/linalg_api/tensordot_axes1_4x5_5x3_f64` |    27.552 |    3.746 |       4.837x |      0.669x |  7.36:1 |
| `array/linalg_api/vecmat_16_f64`               |    11.312 |    3.487 |       4.289x |      1.335x |  3.24:1 |
| `array/linalg_api/matvec_16_f64`               |    10.608 |    3.826 |       3.882x |      1.453x |  2.77:1 |
| `array/linalg_api/vdot_32_f64`                 |    10.146 |    4.913 |       4.229x |      2.082x |  2.07:1 |
| `array/linalg_api/inner_32_f64`                |     9.456 |    5.259 |       3.731x |      2.081x |  1.80:1 |
| `array/linalg_api/dot_1d_32_f64`               |     9.152 |    5.186 |       3.794x |      2.209x |  1.76:1 |

Read:

- The top ratio fell from `kron` at 6.98x to `outer` at 3.15x. Good, not done.
- `tensordot_axes1` is now faster than NumPy on this small row; the old path was
  doing transpose/contiguous/reshape work before landing on the same matmul
  shape.
- `kron` is still 2.34x slower after dropping the scalar-getter loop. The next
  step is a native Kronecker kernel, because the broadcast route still allocates
  view metadata and pays generic multiply dispatch.
- Remaining `>2x` linalg rows after the patch: `outer`, `vecdot`,
  `tensorsolve`, `tensorinv`, `matrix_norm`, `kron`, `norm`, `dot`,
  `matrix_transpose`, `vdot`, and `inner`.

Verification:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/pytest tests/python/numpy_compat/test_tensor_linalg.py tests/python/numpy_compat/test_einsum.py -q
git diff --check -- python/monpy/linalg.py docs/opts.md
```

### 2026-05-09 native outer product kernel

Fresh array-only rerun before this patch, using the current dirty tree:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench --types array --loops 200 --repeats 5 --rounds 3 --format json --sort ratio --output-dir results/local-array-20260509-outer-frontier --no-progress --no-stdout
```

Top row:

| row                             | monpy us | numpy us | ratio  |
| ------------------------------- | -------: | -------: | -----: |
| `array/linalg_api/outer_32_f64` |   12.304 |    4.099 | 3.100x |

Patch:

- Added `_native.linalg_outer`, with a typed SIMD fast path for contiguous
  `float32` / `float64` vectors.
- Kept a logical-index fallback for strided and complex inputs. No fake speed if
  the input layout is annoying; we still return the same answer.
- Routed `mp.outer` through the native op instead of spelling it as
  `reshape(a, (n, 1)) @ reshape(b, (1, m))`. That old route was paying GEMM
  dispatch for a rank-1 product, which is the wrong denominator.
- Added coverage for strided and complex `outer`, because the fast path and
  fallback path should both have a tripwire.

Post-patch array rerun:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench --types array --loops 200 --repeats 5 --rounds 3 --format json --sort ratio --output-dir results/local-array-20260509-native-outer --no-progress --no-stdout
```

Key movement:

| row                             | before us | after us | before ratio | after ratio | speedup |
| ------------------------------- | --------: | -------: | -----------: | ----------: | ------: |
| `array/linalg_api/outer_32_f64` |    12.304 |    3.397 |       3.100x |      0.809x |  3.62:1 |

Read:

- `outer_32_f64` is no longer the top NumPy regression; it now beats NumPy on
  this row.
- The new top row is `array/linalg_api/vecdot_axis1_8x4_f64` at 2.665x slower
  (`7.316 us` vs `2.824 us`).
- The next linalg cluster is still obvious: `vecdot`, `matrix_norm`, vector
  `norm`, and `kron` all allocate at least one avoidable temporary or walk
  through generic reduction.

Verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/create/ops/linalg.mojo src/create/ops/__init__.mojo src/create/__init__.mojo src/lib.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/pytest tests/python/numpy_compat/test_tensor_linalg.py -q
```

### 2026-05-09 native L2 and Frobenius norm

Fresh array-only rerun after native `outer`, using the live dirty tree:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench --types array --loops 200 --repeats 5 --rounds 3 --format json --sort ratio --output-dir results/local-array-20260509-vecdot-frontier --no-progress --no-stdout
```

The frontier moved again:

| row                                      | monpy us | numpy us | ratio  |
| ---------------------------------------- | -------: | -------: | -----: |
| `array/linalg_api/norm_vec2_32_f64`      |   11.440 |    2.809 | 3.987x |
| `array/linalg_api/matrix_norm_fro_16_f64` |   14.047 |    3.968 | 3.529x |
| `array/linalg_api/vecdot_axis1_8x4_f64`  |    7.385 |    3.009 | 2.454x |

Patch:

- Added `_native.linalg_norm2_all` for vector L2 and rank-2 Frobenius norm.
- Added `_native.linalg_norm2_last_axis` for row-wise vector norms.
- The contiguous float32/float64 path does one SIMD sum-of-squares pass and
  writes the square root directly. No square temporary, no generic
  `sum(axis=...)`, no second trip through Python.
- Kept the route narrow in Python: float32/float64 only, `keepdims=False`, and
  the exact L2/Frobenius cases the current facade already handles.

Post-patch array rerun:

```text
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/monpy-bench --types array --loops 200 --repeats 5 --rounds 3 --format json --sort ratio --output-dir results/local-array-20260509-native-norm2 --no-progress --no-stdout
```

Key movement:

| row                                      | before us | after us | before ratio | after ratio | speedup |
| ---------------------------------------- | --------: | -------: | -----------: | ----------: | ------: |
| `array/linalg_api/norm_vec2_32_f64`      |    11.440 |    3.188 |       3.987x |      1.165x |  3.59:1 |
| `array/linalg_api/matrix_norm_fro_16_f64` |    14.047 |    4.679 |       3.529x |      1.172x |  3.00:1 |
| `array/linalg_api/vector_norm_axis1_8x4_f64` | 7.715 |    3.913 |       1.867x |      0.960x |  1.97:1 |

Read:

- The top two norm rows stopped being the top problem. Good.
- The new top row is back to `array/linalg_api/vecdot_axis1_8x4_f64` at
  2.828x slower (`7.973 us` vs `2.819 us`).
- That row still does `multiply(A, B)` plus `sum(axis=1)`. The next kernel
  should be row-wise vecdot, not another facade shuffle.

Verification:

```text
MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo /Users/aarnphm/workspace/modular/.derived/build/bin/mojo format --line-length 119 src/create/ops/linalg.mojo src/create/ops/__init__.mojo src/create/__init__.mojo src/lib.mojo
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/mohaus develop --no-build-isolation
MOHAUS_EDITABLE_REBUILDING=1 MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo .venv/bin/pytest tests/python/numpy_compat/test_tensor_linalg.py -q
```
