# Benchmarks

The benchmark suite compares monpy against numpy. Lower `monpy/numpy` ratios are better for monpy; ratios below `1.0x` mean monpy beat numpy for that case.

## Entry points

From a checkout, run the suite directly:

```bash
MOHAUS_MOJO=/path/to/mojo python -m monpy._bench.sweep --types all --format csv --no-progress
```

After `mohaus develop` or an editable install, use the installed command:

```bash
monpy-bench --types all --format csv --no-progress
```

The command is installed from `pyproject.toml` and imports
`monpy._bench.sweep` from the editable package. No repository-root path
injection is required.

## Suites

`monpy._bench.sweep` is the only benchmark runner. It loads case families from
`monpy._bench.types`:

- `array`: broad array-core coverage, including dtype metadata, creation, casts, interop, elementwise, reductions, matmul, linalg, and bandwidth cases.
- `strides`: non-contiguous and view-heavy cases, including transpose, broadcasting, reverse strides, slicing, rank-3 transposes, flips, `rot90`, and contiguous copies from views.
- `complex`: complex dtype interop, casts, elementwise arithmetic, reversed views, and complex matmul.
- `attention`: tiny transformer kernels, including causal-score softmax, causal self-attention, and a one-block GPT-style logits pass.

Use `--types` to control the surface:

```bash
monpy-bench --types array
monpy-bench --types strides,complex
monpy-bench --types attention
monpy-bench --types all
```

## Output

Use the same output formats locally and in CI:

```bash
monpy-bench --types all --format table
monpy-bench --types all --format csv --sort ratio
monpy-bench --types all --format json --no-progress
monpy-bench --types all --format markdown --sort ratio
```

Each run saves the rendered output by default:

```text
results/yyyy-mm-dd/manifest.json
results/yyyy-mm-dd/results.<format>
```

The runner still writes the rendered results to stdout unless `--no-stdout` is passed.
Use `--no-save` for the old stdout-only behavior, or `--output-dir` to pick a stable artifact directory:

```bash
monpy-bench --types all --format json --no-progress --no-stdout
monpy-bench --types all --format csv --output-dir results/local-smoke
monpy-bench --types complex --format table --no-save
```

`manifest.json` records the command, cwd, platform, python/numpy/monpy/mojo versions when detectable, suite selection, vector sizes, matrix sizes, linalg sizes, case list, output file path, byte count, and sha256.

The json output includes:

- `candidate`: `monpy`
- `baseline`: `numpy`
- `comparison`: `monpy_us / numpy_us`
- `types`: enabled suite families
- `vector_size`, `vector_sizes`, `matrix_sizes`, and `linalg_sizes`: the exact
  shape sweep

Each result row includes median, min, and max timings for monpy and numpy, plus the `monpy/numpy` ratio.

## CI Comments

Posting is not part of the benchmark runner. Generate benchmark json first, then render or post through the workflow script:

```bash
monpy-bench --types all --format json --no-progress --no-stdout
python .github/scripts/posts.py \
  results/$(date +%F)/results.json \
  --comment-output benchmark-comment.md
```

CI passes `--post` on macOS to upsert the commit comment:

```bash
python .github/scripts/posts.py \
  results/$(date +%F)/results.json \
  --comment-output benchmark-comment.md \
  --post
```

The comment includes a `winner` column. `monpy` means the median ratio was below `0.995x`; `numpy` means it was above `1.005x`; values inside that band are reported as `tie`.

The post helper also accepts `manifest.json` when the manifest points at json results:

```bash
python .github/scripts/posts.py \
  results/$(date +%F)/manifest.json \
  --comment-output benchmark-comment.md
```

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

| run | monpy us | numpy us | monpy/numpy |
| --- | ---: | ---: | ---: |
| `results/local-sweep-20260508-pass0/results.json` | 30.653 | 8.404 | 3.648x |
| `results/local-sweep-20260508-rank3-source-dispatch/results.json` | 19.064 | 8.210 | 2.263x |

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

| run | monpy us | numpy us | monpy/numpy |
| --- | ---: | ---: | ---: |
| `results/local-sweep-20260508-pass0/results.json` | 7.808 | 2.216 | 3.524x |
| `results/local-sweep-20260508-scalar-ascontig/results.json` | 3.498 | 2.164 | 1.618x |

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

| run | monpy us | numpy us | monpy/numpy |
| --- | ---: | ---: | ---: |
| `results/local-sweep-20260508-heartbeat1/results.json` | 12.494 | 2.436 | 5.205x |
| `results/local-sweep-20260508-native-squeeze/results.json` | 8.182 | 2.429 | 3.390x |

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

| run | monpy us | numpy us | monpy/numpy |
| --- | ---: | ---: | ---: |
| `results/local-sweep-20260508-native-squeeze/results.json` | 8.824 | 2.182 | 4.052x |
| `results/local-sweep-20260508-dlpack-numpy-fastpath/results.json` | 5.669 | 2.172 | 2.587x |

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

| row | previous monpy us | new monpy us | previous ratio | new ratio |
| --- | ---: | ---: | ---: | ---: |
| `array/interop/asarray_zero_copy_f32` | 6.097 | 5.207 | 3.104x | 2.699x |
| `array/interop/asarray_zero_copy_f64` | 5.955 | 5.213 | 3.024x | 2.644x |
| `array/interop/asarray_zero_copy_bool` | 6.082 | 5.229 | 3.062x | 2.697x |
| `array/interop/asarray_zero_copy_i64` | 5.960 | 5.199 | 3.062x | 2.654x |
| `array/interop/from_dlpack_f32` | 5.669 | 4.963 | 2.587x | 2.290x |

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

| row | previous monpy us | new monpy us | previous ratio | new ratio |
| --- | ---: | ---: | ---: | ---: |
| `array/views/stack_axis0_f32` | 9.191 | 3.731 | 2.610x | 1.051x |
| `array/views/vstack_f32` | 9.678 | 3.700 | 2.926x | 1.127x |

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

| row | previous monpy us | new monpy us | previous ratio | new ratio |
| --- | ---: | ---: | ---: | ---: |
| `array/creation/atleast_2d_f32` | 6.087 | 3.526 | 2.664x | 1.519x |

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

| row | previous monpy us | new monpy us | previous ratio | new ratio |
| --- | ---: | ---: | ---: | ---: |
| `array/views/newaxis_middle_f32` | 5.743 | 3.138 | 2.734x | 1.535x |

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

| row | previous monpy us | new monpy us | previous ratio | new ratio |
| --- | ---: | ---: | ---: | ---: |
| `array/views/swapaxes_f32` | 5.534 | 3.072 | 2.550x | 1.379x |

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

| row | previous monpy us | new monpy us | previous ratio | new ratio |
| --- | ---: | ---: | ---: | ---: |
| `array/interop/asarray_zero_copy_f32` | 5.095 | 4.923 | 2.628x | 2.532x |
| `array/interop/asarray_zero_copy_f64` | 5.155 | 4.826 | 2.654x | 2.486x |
| `array/interop/asarray_zero_copy_bool` | 5.134 | 4.834 | 2.630x | 2.528x |
| `array/interop/asarray_zero_copy_i64` | 5.138 | 4.819 | 2.654x | 2.494x |
| `array/interop/from_dlpack_f32` | 4.900 | 4.596 | 2.262x | 2.173x |

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

| row | previous monpy us | new monpy us | previous ratio | new ratio |
| --- | ---: | ---: | ---: | ---: |
| `array/views/squeeze_axis0_f32` | 7.106 | 6.912 | 2.969x | 2.821x |

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

| row | previous monpy us | new monpy us | previous ratio | new ratio |
| --- | ---: | ---: | ---: | ---: |
| `array/creation/logspace_50` | 22.809 | 4.494 | 4.156x | 0.836x |

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

| row | previous monpy us | new monpy us | previous ratio | new ratio |
| --- | ---: | ---: | ---: | ---: |
| `array/decomp/pinv_2_f64` | 33.683 | 6.824 | 2.145x | 0.437x |
| `array/decomp/pinv_4_f64` | 39.188 | 8.349 | 2.252x | 0.481x |
| `array/decomp/pinv_8_f64` | 54.852 | 12.864 | 2.811x | 0.655x |
| `array/decomp/pinv_32_f64` | 136.756 | 54.235 | 2.424x | 0.963x |
| `array/decomp/pinv_8_f32` | 48.527 | 10.937 | 2.033x | 0.451x |

All benchmarked pseudo-inverse rows now beat NumPy on this machine. The largest
remaining deficit is no longer `pinv`; it is the view/import-heavy
`squeeze_axis0_f32` row.
