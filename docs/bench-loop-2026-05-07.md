# performance progress, 2026-05-07

this file tracks the local perf pass that started from `results/local-sweep-20260507-141544/results.json`.

## baseline sweep

command:

```bash
env MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo \
  .venv/bin/python -m monpy._bench.sweep \
  --format json --no-progress --no-stdout \
  --output-dir results/local-sweep-20260507-141544 \
  --types all --loops 200 --repeats 5 --rounds 3
```

summary:

- rows: 239
- geomean ratio: 1.501x monpy/numpy
- median ratio: 1.236x
- p90 ratio: 4.914x
- wins / ties / numpy wins: 62 / 2 / 175

highest-value deficits:

- `strides/copy::ascontiguousarray_transpose_f32`: 820.619 us vs 48.967 us, 16.759x
- `strides/elementwise::sliced_unary_sin_f32`: 684.099 us vs 37.736 us, 18.128x
- `strides/elementwise::transpose_add_f32`: 269.125 us vs 11.068 us, 24.315x
- `strides/elementwise::broadcast_row_add_f32`: 257.526 us vs 14.934 us, 17.112x
- `strides/elementwise::rank3_transpose_add_f32`: 142.867 us vs 8.006 us, 17.846x
- `array/creation::indices_4x4`: 126.044 us vs 4.073 us, 30.947x
- `array/decomp::pinv_32_f64`: 158.244 us vs 59.767 us, 2.630x

## fix log

### pass 1, rank-2 copy + native indices

target:

- remove the `physical_offset` div/mod loop from rank-2 materialization of
  transposed views.
- replace python-level `indices` composition with one native int64 kernel.

changed:

- `src/array.mojo`: added rank-2 typed strided materialization for real,
  complex, and bool arrays. `copy_c_contiguous` now uses this before the
  generic logical-index fallback.
- `src/create.mojo`, `src/lib.mojo`, `python/monpy/__init__.py`,
  `python/monpy/_native.pyi`: added native int64 `indices`.

verification:

- `env MOHAUS_MOJO=... .venv/bin/mohaus develop`
- python parity smoke for `ascontiguousarray(a.T)` and `indices((4, 4))`
- `.venv/bin/python -m pytest tests/python/numpy_compat/test_creation_helpers.py -q`

focused sweep:

```bash
env MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo \
  .venv/bin/python -m monpy._bench.sweep \
  --format json --no-progress --no-stdout \
  --output-dir results/local-sweep-20260507-pass1 \
  --types strides,array --loops 200 --repeats 5 --rounds 3
```

movement:

- `strides/copy::ascontiguousarray_transpose_f32`: 820.619 us / 16.759x → 34.095 us / 0.703x.
- `array/creation::indices_4x4`: 126.044 us / 30.947x → 3.489 us / 0.951x.
- `strides` geomean: 4.834x → 3.563x.
- `array` geomean: 1.390x → 1.349x.

### pass 2, strided elementwise add + rank-2 unary

target:

- remove dynamic-op overhead from plain-add fast paths.
- avoid the generic `LayoutIter` unary fallback for rank-2 strided f32/f64.

changed:

- `src/elementwise.mojo`: specialized `op == OP_ADD` inside row-broadcast,
  rank-2 transposed tile, and general strided binary walkers.
- `src/elementwise.mojo`, `src/create.mojo`: added `maybe_unary_rank2_strided`
  and call it from `unary_ops` before the generic fallback.

verification:

- `env MOHAUS_MOJO=... .venv/bin/mohaus develop`
- python parity smoke for `a.T + b.T`, `a + row`, and `sin(a[::2, 1::2])`
- `.venv/bin/python -m pytest tests/python/numpy_compat/test_ufunc.py tests/python/numpy_compat/test_creation_helpers.py -q`

focused sweep artifact:

- `results/local-sweep-20260507-pass2/results.json`

movement:

- `strides/elementwise::broadcast_row_add_f32`: 257.526 us / 17.112x → 10.898 us / 0.752x.
- `strides/elementwise::transpose_add_f32`: 269.125 us / 24.315x → 31.258 us / 2.943x.
- `strides/elementwise::rank3_transpose_add_f32`: 142.867 us / 17.846x → 29.104 us / 3.670x.
- `strides/elementwise::sliced_unary_sin_f32`: 684.099 us / 18.128x → 57.520 us / 1.501x.
- `strides` geomean: 4.834x → 1.393x.

### pass 3, typed eye/tri/tril/triu

target:

- remove per-element setter overhead from hot f32/f64 triangular constructors.

changed:

- `src/create.mojo`: added direct f32/f64 pointer fill/copy paths for
  `eye`, `tri`, `tril`, and `triu`.

verification:

- `env MOHAUS_MOJO=... .venv/bin/mohaus develop`
- python parity smoke for core casts, `eye`, `tri`, `tril`, and `triu`
- `.venv/bin/python -m pytest tests/python/numpy_compat/test_array_coercion.py tests/python/numpy_compat/test_creation_helpers.py tests/python/numpy_compat/test_numeric.py -q`

focused sweep artifact:

- `results/local-sweep-20260507-pass3/results.json`

movement:

- `array/creation::eye_64_f32`: 28.171 us / 7.360x → 3.322 us / 0.881x.
- `array/creation::identity_64_f32`: 28.358 us / 6.893x → 3.400 us / 0.869x.
- `array/creation::tri_64_f32`: 30.589 us / 4.660x → 4.458 us / 0.695x.
- `array/native_kernels::eye_64_native_f64`: 28.371 us / 7.464x → 3.372 us / 0.900x.
- `array/native_kernels::tril_64_native_f64`: 43.139 us / 5.426x → 7.003 us / 0.891x.
- `array/native_kernels::triu_64_native_f64`: 43.384 us / 5.506x → 6.727 us / 0.859x.
- `array` geomean: 1.390x → 1.265x.

### pass 4, typed contiguous casts

target:

- replace the benchmarked contiguous scalar casts with explicit pointer loops
  for `bool`, `int64`, `float32`, and `float64`.

changed:

- `src/array.mojo`: added `_maybe_cast_contiguous_core_dtypes` and route
  `cast_copy_array` through it before the generic physical/logical fallback.

verification:

- `env MOHAUS_MOJO=... .venv/bin/mohaus develop`
- python parity smoke for the 12 cross-dtype cast pairs covered by the bench
- `.venv/bin/python -m pytest tests/python/numpy_compat/test_array_coercion.py -q`

focused sweep artifact:

- `results/local-sweep-20260507-pass4/results.json`

movement:

- cast family moved from 5.175-8.057x baseline rows to 1.234-1.515x rows.
- `array/casts::astype_bool_to_i64`: 20.063 us / 8.057x → 3.643 us / 1.447x.
- `array/casts::astype_i64_to_bool`: 20.778 us / 8.002x → 3.610 us / 1.463x.
- `array/casts::astype_f64_to_f32`: 13.995 us / 5.733x → 3.651 us / 1.515x.
- `array` geomean: 1.390x → 1.182x.
- `strides` geomean: 4.834x → 1.393x.

remaining top debts after pass 4:

- `array/native_kernels::concatenate_axis0_8x128_f64`: 15.438 us / 4.799x.
- `array/creation::{zeros_like,ones_like,full_like}_transpose_f32`: 11.5-12.3 us / 4.0-4.35x.
- `strides/elementwise::rank3_transpose_add_f32`: 30.021 us / 3.650x.
- `array/decomp::pinv_32_f64`: 159.000 us / 2.681x.

## final all-suite sweep

command:

```bash
env MOHAUS_MOJO=/Users/aarnphm/workspace/modular/.derived/build/bin/mojo \
  .venv/bin/python -m monpy._bench.sweep \
  --format json --no-progress --no-stdout \
  --output-dir results/local-sweep-20260507-final \
  --types all --loops 200 --repeats 5 --rounds 3
```

artifact:

- `results/local-sweep-20260507-final/results.json`
- `results/local-sweep-20260507-final/manifest.json`
- `results/local-sweep-20260507-final/benchmark-comment.md`

headline:

- overall geomean: 1.501x → 1.226x.
- overall median: 1.236x → 1.172x.
- p90: 4.948x → 2.360x.
- wins / ties / numpy wins: 62 / 2 / 175 → 76 / 1 / 162.

suite movement:

- `array`: geomean 1.390x → 1.185x, median 1.183x → 1.152x.
- `strides`: geomean 4.834x → 1.377x, median 16.759x → 1.334x.
- `complex`: geomean 2.894x → 2.718x, median 3.108x → 2.300x. Complex improved only indirectly; it still needs its own pass.

largest fixed rows:

- `strides/copy::ascontiguousarray_transpose_f32`: 820.619 us / 16.759x → 33.487 us / 0.688x.
- `strides/elementwise::broadcast_row_add_f32`: 257.526 us / 17.112x → 11.738 us / 0.800x.
- `strides/elementwise::sliced_unary_sin_f32`: 684.099 us / 18.128x → 57.387 us / 1.519x.
- `array/creation::indices_4x4`: 126.044 us / 30.947x → 3.613 us / 0.988x.
- `array/creation::tri_64_f32`: 30.589 us / 4.660x → 4.712 us / 0.673x.
- `array/native_kernels::triu_64_native_f64`: 43.384 us / 5.506x → 6.872 us / 0.856x.
- `array/casts::astype_bool_to_i64`: 20.063 us / 8.057x → 3.558 us / 1.459x.

remaining highest-ratio rows:

- `complex/views::reversed_add_complex64`: 26.940 us / 7.469x.
- `array/native_kernels::concatenate_axis0_8x128_f64`: 16.042 us / 4.869x.
- `complex/elementwise::binary_add_complex128`: 12.460 us / 4.344x.
- `array/creation::zeros_like_transpose_f32`: 12.624 us / 4.339x.
- `array/views::vstack_f32`: 14.509 us / 4.092x.
- `array/creation::{ones_like,full_like}_transpose_f32`: about 12 us / 4.0x.
- `strides/elementwise::rank3_transpose_add_f32`: 29.704 us / 3.624x.
- `array/decomp::pinv_32_f64`: 159.928 us / 2.764x.
