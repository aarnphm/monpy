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
