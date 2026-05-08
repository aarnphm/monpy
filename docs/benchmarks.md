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
| `results/local-sweep-20260508-rank3-dispatch/results.json` | 18.544 | 7.931 | 2.373x |

That is a 1.65:1 reduction in monpy wall time for this row. The row still does
not beat numpy, so the next optimization pass should use sampled profiles and
hardware counters before adding another kernel. The likely split to verify is
Python wrapper time versus tile shuffle/store pressure versus library-assisted
iterator behavior in numpy.
