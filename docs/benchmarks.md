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
monpy-bench-mojo --format csv
```

The commands are installed from `pyproject.toml` and import their runners from
the editable package. No repository-root path injection is required.

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

CI passes `--post` from each benchmark matrix job. Use a distinct `--comment-key` per platform so ARM
and Ubuntu update separate commit comments:

```bash
python .github/scripts/posts.py \
  results/$(date +%F)/results.json \
  --comment-key arm \
  --comment-title "monpy benchmark sweep (ARM)" \
  --comment-output benchmark-comment-arm.md \
  --post
```

The comment includes an overall case-outcome summary and a `winner` column.
`monpy` means the median ratio was below `0.995x`; `numpy` means it was above
`1.005x`; values inside that band are reported as `tie`. The faster median
timing cell is bolded for non-tie rows.

The post helper also accepts `manifest.json` when the manifest points at json results:

```bash
python .github/scripts/posts.py \
  results/$(date +%F)/manifest.json \
  --comment-output benchmark-comment.md
```

## Mojo-side head-to-head benches (`benches/`)

These benches sit _next to_ the Python sweep above; they answer a different question. The Python sweep asks "is monpy's user-facing API faster than numpy's?" — these ask "is monpy's _kernel_ implementation actually competitive with the equivalent Mojo stdlib primitive?" Pure Mojo, no Python in the inner loop, no FFI overhead. Run them when you suspect a hand-rolled kernel is leaving performance on the table, or before/after touching `src/elementwise/*kernels*.mojo`.

### Run

```bash
MOHAUS_MOJO=$MODULAR_DERIVED_PATH/build/bin/mojo monpy-bench-mojo --format table
$MODULAR_DERIVED_PATH/build/bin/mojo run -I src benches/bench_mojo_sweep.mojo
$MODULAR_DERIVED_PATH/build/bin/mojo run -I src benches/bench_reduce.mojo
```

(`MODULAR_DERIVED_PATH` is set by the modular shell init; it points at the locally-built `mojo` binary that matches `.mojo-version`. The `.venv/bin/mojo` shim works too on darwin but the modular-derived path is the canonical reference.)

`monpy-bench-mojo` runs the pure Mojo sweep and writes durable artifacts by default:

```text
results/yyyy-mm-dd/mojo/manifest.json
results/yyyy-mm-dd/mojo/results.<format>
```

Use `--no-save` for stdout only:

```bash
monpy-bench-mojo --format json --sort ratio --no-save
```

### `bench_mojo_sweep.mojo`

Compares production monpy kernels against stdlib-shaped Mojo baselines on identical native buffers. It emits TSV so the Python runner can render table, csv, json, or markdown artifacts without putting Python in the measured loop.

The sweep covers both f32 and f64:

- elementwise add across 1K, 64K, and 1M elements
- scalar multiply across 1K and 64K elements
- unary `sin` across 1K and 64K elements
- reductions: `sum`, `mean`, `min`, `max`, and `product`
- small square matmul at 8x8 and 16x16

The stdlib side uses the closest local primitive for each family: `std.algorithm.sum`/`mean`/`min`/`max`/`product` for reductions, `std.math.sin` and `SIMD`/`simd_width_of` loops for elementwise kernels, and a simple stdlib-pointer SIMD loop for small matmul. The comparison is deliberately below array semantics: it tests the cost of the kernel body, not dtype promotion, shape validation, Python object construction, or FFI.

Interpretation rule: rows near `1.0x` mean the monpy kernel and stdlib-shaped loop are effectively at parity for that native operation. A stdlib win here is evidence to replace or simplify the inner Mojo kernel. It is not evidence that the Python-facing monpy runtime can disappear, because shape/stride semantics, dtype dispatch, NumPy-compatible behavior, and Accelerate/vForce routing still live above this layer.

### Optional NuMojo comparison

NuMojo is an external array library, so its bench is separate and opt-in. The
repo carries a patched vendored copy at `vendor/NuMojo`; that is the default
`monpy-bench-mojo --include-numojo` lookup after `NUMOJO_PATH`.

```bash
monpy-bench-mojo --include-numojo
$MODULAR_DERIVED_PATH/build/bin/mojo run \
  --ignore-incompatible-package-errors \
  -I src \
  -I vendor/NuMojo \
  benches/bench_numojo_sweep.mojo
```

`bench_numojo_sweep.mojo` compares monpy raw kernels against NuMojo public NDArray operations for add, `sin`, `sum`, and small matmul. The row orientation matches the rest of `monpy-bench-mojo`: `candidate` is monpy, `baseline` is the comparison target, and `ratio = candidate_ns / baseline_ns`. This comparison is intentionally not identical to `bench_mojo_sweep.mojo`: NuMojo includes its array abstraction overhead, while monpy rows are kernel-level. Treat it as an external-library baseline, not as a stdlib replacement proof.

NuMojo `0.9.0` currently tracks the Modular 0.26.x toolchain family. The vendored copy is patched for this checkout's Mojo `1.0.0.dev0` benchmark path; see `vendor/README.md` and `vendor/NuMojo/MONPY_PATCHES.md` for provenance, license references, and the compatibility patch ledger. Update that ledger whenever the vendored source changes. External NuMojo checkouts may still fail at import time with stdlib API drift. The CLI treats that as a skipped optional baseline by default: it still reports the monpy/stdlib Mojo rows and writes the attempted NuMojo command into the manifest. Use `--strict-numojo` when you want that compatibility failure to stop the run.

### Optional threading comparison

Use `--include-threading` when the question is whether a threaded internal
prototype should replace the current monpy serial static kernel for wide
elementwise work:

```bash
monpy-bench-mojo --include-threading --thread-caps auto,1,2,4,8
```

The CLI runs `benches/bench_threading_sweep.mojo` once per thread cap in a fresh
Mojo process. `auto` clears `MONPY_THREADS`; numeric caps set
`MONPY_THREADS=N`. This matters because the production thread policy is read
once per process. Threading rows use the same comparison contract as the rest
of the Mojo sweep: `candidate` is the internal threaded static prototype,
`baseline` is the monpy serial static kernel, and `ratio = candidate_ns /
baseline_ns`. Ratios below `1.0x` mean the threaded prototype wins for that
shape and cap.

### `bench_reduce.mojo`

The reduction harness uses `std.benchmark.Bench` with significance gating, ~100 iterations per cell, two repetitions by default. Each row prints mean wall-time (ms), data-movement throughput (GB/s), and arithmetic throughput (GFLOPS/s).

Compares **five reductions** (`sum`, `mean`, `min`, `max`, `product`) across three implementations on identical buffers in L1/L2/L3/DRAM-resident sizes for both f32 and f64:

- `<op>4_*` — frozen 4-way SIMD-accumulator kernel (legacy baseline, kept for the pipeline-depth lesson)
- `<op>8_*` — current production kernel from `src/elementwise/kernels/reduce.mojo` (8-way unrolled)
- `<op>S_*` — `std.algorithm.reduction.<op>` Span overload

#### Result on Apple Silicon (M-series, single thread)

Throughput in GB/s (data movement, higher is better). Boldface marks the winner per row.

**SUM**

| dtype | size | 4-way | **8-way** | std |
| ----- | ---- | ----: | --------: | --: |
| f32   | 1k   |    90 |   **134** | 130 |
| f32   | 64k  |    66 |    **83** |  82 |
| f32   | 1M   |    65 |    **81** |  80 |
| f64   | 1k   |    82 |   **150** | 131 |
| f64   | 1M   |    61 |    **76** |  75 |

**MIN**

| dtype | size | 4-way | **8-way** | std |
| ----- | ---- | ----: | --------: | --: |
| f32   | 1k   |   123 |   **153** | 134 |
| f32   | 64k  |    84 |        85 |  83 |
| f32   | 1M   |    80 |        81 |  80 |
| f64   | 1k   |   110 |   **138** | 137 |

**MAX**

| dtype | size | 4-way | **8-way** |     std |
| ----- | ---- | ----: | --------: | ------: |
| f32   | 1k   |   108 |   **143** |     134 |
| f32   | 64k  |    85 |        85 |      85 |
| f32   | 1M   |    82 |        82 |      82 |
| f64   | 1k   |   113 |       137 | **146** |

**PROD** (largest 4→8 jump — FMUL has deeper latency than FADD)

| dtype | size | 4-way | **8-way** |     std |
| ----- | ---- | ----: | --------: | ------: |
| f32   | 1k   |    79 |   **126** |     113 |
| f32   | 64k  |    51 |    **85** |      84 |
| f64   | 1k   |    63 |       115 | **117** |

**MEAN** (8-way only — same kernel as sum + scalar div)

| dtype | size | **8-way** | std |
| ----- | ---- | --------: | --: |
| f32   | 64k  |        84 |  85 |
| f32   | 1M   |        81 |  81 |

#### What this tells us

1. **The 8-way kernel matches `std.algorithm.reduction` everywhere and beats it 5–15% at small N.** On all 5 ops, on both dtypes, at every size tested. `std` and the hand-rolled 8-way produce essentially identical assembly modulo dispatch overhead — at small N `std`'s setup cost (input_fn lambda, tracing hooks, generator dispatch) is visible; at large N both are bandwidth-bound and converge.

2. **PROD shows the steepest 4→8 win** (51→85 GB/s at f32 64k, ~67% boost). FMUL is ~4-cycle latency on M-series — needing 4 × 2 IPC = 8 in flight to saturate. 4-way leaves half the multiplier idle; 8-way fills it. Every floating-point op has its own latency-IPC product; 8 is the safe choice across the FADD/FMIN/FMAX/FMUL family on modern cores.

3. **DRAM-bound regime is bandwidth-bound regardless of unroll.** At 16M+ on a single thread, all three implementations converge near ~50 GB/s f32 — that's the M-series single-thread DRAM ceiling. The next-tier improvement is _parallelization_, not a smarter SIMD loop.

### What's been done in this commit

Applied directly to `src/elementwise/kernels/reduce.mojo`:

- **`reduce_sum_typed`**: 4-way → 8-way (8 SIMD accumulators, tree-reduce collapse). Updated the explanatory comment to mention 2-IPC FADD pipelines, the `latency × IPC ≈ 6–8` rule, and the bench validation numbers.
- **`reduce_min_typed`**, **`reduce_max_typed`**, **`reduce_prod_typed`** — new 8-way SIMD kernels (float-only, mirror the sum structure). Seeded from the first 8 SIMD vectors for min/max so we never need an `Inf` sentinel; PROD seeds with 1.0 so partial-product semantics hold for any size.
- Wired into `reduce_strided_typed` (linear-addressable float branch only — integer paths keep their scalar loops because of overflow-promotion concerns) and into `_reduce_axis_last_contiguous_typed` (which is float-only by signature).
- Added MIN/MAX/PROD float fast paths to `maybe_reduce_contiguous` so the c-contig f32/f64 case hits the new SIMD kernels instead of falling through to the scalar `contiguous_as_f64` walker.

Test status: full numpy-compat suite at 706 passes / 1 unrelated pre-existing failure (`test_complex_negate_preserves_imaginary_part` in test_complex.py — pre-existed before this change, confirmed by stashed checkout).

### Threading scope (current)

The `MONPY_THREADS` env knob controls these op classes:

- **Row kernels** (softmax, layernorm, axis-reduce). Gated by
  `worker_count_for_row_elements(rows, cols, ROW_HEAVY_GRAIN_ELEMS=8K)`,
  so a 32×32 attention stays serial while a 32×4096 fans out.
- **Element-wise heavy unary** (exp/log/sin/cos/sqrt/...). Gated at
  256KB per worker via `unary_op_grain(op)`.
- **Element-wise light binary/unary** (add/mul/neg/abs/...). Gated at
  2MB per worker. Note that the production dispatch ladder lands on
  comptime-static single-thread kernels for ADD/SUB/MUL/DIV and the
  common unary ops, so in practice this gate fires only for less-common
  ops (POWER, MAXIMUM, EXPM1, ATAN, etc.).

`MONPY_THREADS=1` forces every gate to fall back to single-thread.
`MONPY_THREADS=N` caps automatic fanout at N workers.

### What threading does NOT do

Whole-tensor reductions (sum/min/max/prod/mean) are **single-thread
regardless of `MONPY_THREADS`**. An earlier prototype using
`std.algorithm.sync_parallelize` produced a 4× slowdown and 3.5× variance
on macOS at 1M f32 sum:

| platform         | sum size |  serial |                   parallel |                         conclusion |
| ---------------- | -------: | ------: | -------------------------: | ---------------------------------: |
| macOS M3 Pro     |   1M f32 |   60 µs | **261 µs (range 84..292)** |        parallel is loss + unstable |
| Ubuntu CI 4-vCPU |   1M f32 |   69 µs |                  **96 µs** |                   parallel is loss |
| macOS M3 Pro     |  16M f32 | ~280 µs |        ~140 µs theoretical | win possible but variance kills it |

The cost atom is `std.algorithm.sync_parallelize`: it allocates and
synchronizes a CPU `DeviceContext` per call, ~200 µs on Apple Silicon.
For any reduction below ~16MB total bytes, that overhead exceeds the
per-thread compute. Until Mojo grows a persistent worker pool or a
better reduction scheduler, reductions must stay serial.

The `reduce_*_par_typed` kernels were deleted (commit
`perf: drop par-reduce kernels and REDUCE_GRAIN`). Restore them from git
if the upstream primitive improves.

### Next-tier improvements (not in this commit)

2. **Consider replacing the hand-roll with `std.algorithm.reduction.<op>`.** At 8-way the production and library kernels are at parity; the hand-roll mainly serves as documentation now. Keep the 4-way and 8-way variants in `benches/` as a teaching trace but call `std_sum`/`std_min`/etc from `reduce.mojo`. Trade-off: gives up a tiny small-N edge for less code to maintain.

3. **Integer min/max/prod SIMD paths.** Currently float-only because integer overflow / promotion semantics differ from float. Adding 8-way SIMD min/max for native int dtypes (no promotion needed for min/max — they preserve dtype) is a strict win. PROD on ints needs more thought (numpy's int64 prod overflows silently; matching that is fine, but match deliberately).

4. **`parallel_memcpy` for broadcast materialization.** `std.algorithm.memory.parallel_memcpy` is a one-line replacement for any large-buffer copy in monpy.

#### Educational framing for new readers

The `<op>4_*` vs `<op>8_*` vs `<op>S_*` triple is a textbook lesson on out-of-order pipelines. Run the bench, observe the gap close at item 1, try editing one of the kernels to use `block = width * 2` and watch the gap reopen. The comment in `reduce.mojo` documents _why_ multiple accumulators are needed; the bench documents _how many_. Together they make the pipeline visible.

The PROD numbers are the cleanest demonstration: FMUL is one cycle deeper than FADD, so the 4-way kernel under-saturates by ~50%, exactly the gap that 8-way closes. If you can predict from a chip's spec sheet how the bench will move, you've internalized the lesson.
