---
title: "recent implementation field notes"
date: 2026-05-09
---

# recent implementation field notes

_the latest full local macOS sweep has 243 Python-facing rows, and none of them are more than 2x slower than NumPy._

this is the overdue overview for the May 8-9 implementation run. it is not a
replacement for `docs/opts.md`, which is the raw ledger of rows, commands, and
artifacts. this note is the field layer: what each recent item taught us, which
surface owns the next move, and which benchmark rows fold multiple costs into
one name.

## 1. the current frontier is fixed cost, not bandwidth

the full local macOS Python-facing sweep had 243 rows. 154 were slower than
NumPy, 67 were above `1.25x`, 21 were above `1.5x`, and none were above `2x`.
the median ratio was `1.092x`. after the tiny-linalg/list-construction pass,
the same shape moved to 145 slow rows, 55 above `1.25x`, 17 above `1.5x`, and
median `1.083x`.

that denominator matters. the working set is no longer "monpy is generally
slow." it is "monpy has a few fixed per-call costs." the worst local rows after
the linalg pass were:

| row                                            |     monpy |    numpy |  ratio |
| ---------------------------------------------- | --------: | -------: | -----: |
| `array/creation/empty_like_shape_override_f32` |  4.494 us | 2.308 us | 1.948x |
| `array/interop/asarray_squeeze_axis0_f32`      |  4.241 us | 2.298 us | 1.845x |
| `array/views/transpose_add_f32`                |  4.522 us | 2.493 us | 1.793x |
| `strides/elementwise/rank3_transpose_add_f32`  | 13.970 us | 8.166 us | 1.711x |

none of these point first at "write a wider SIMD loop." the rows are small,
wrapper-heavy, or view-heavy. if a row touches 24 elements and costs 4 us, the
machine is telling us about Python/native crossings, object construction, shape
normalization, or harness overhead. the bytes are not the problem yet.

## 2. interop moved from dict archaeology to the buffer protocol

the old NumPy ingress path read `__array_interface__`, then walked a Python dict:
`typestr`, `shape`, `strides`, `data`, and per-dimension integer conversion.
that was eight or nine Python interactions for one array crossing. for tiny
arrays, the metadata walk was the benchmark.

`src/buffer.mojo` moved the hot path to CPython's buffer protocol. one
`PyObject_GetBuffer(..., PyBUF_RECORDS_RO)` call gives the pointer, item size,
format, shape, strides, and readonly bit. caching `PyObject_GetBuffer` and
`PyBuffer_Release` removed per-call dyld symbol lookup.

the progress is concrete: `asarray_zero_copy_f32` went from a
35x-class row to about `1.5x` in the array sweep. the residual gap is now
wrapper and array-record construction, not Python dict scraping. this also
explains why specialized complex buffer wrappers only moved official rows by
low single digits even when raw leaf timings improved: the leaf is already
sub-microsecond, and the public row still allocates a Python `ndarray` wrapper
around a borrowed buffer.

rule: if the row name begins with `asarray`, `from_dlpack`, or `array_copy`, the
first profile question is "how many Python object transitions happen before the
first byte of user data moves?"

## 3. view rows need atomized benchmarks

the old `squeeze_axis0_f32` row was bad evidence. it created a NumPy array,
converted it to monpy, and squeezed it inside one timed lambda. the name said
"squeeze"; the row measured ingress plus squeeze plus harness.

splitting it made the atoms visible:

| row                                     |    monpy |    numpy |  ratio |
| --------------------------------------- | -------: | -------: | -----: |
| old `views/squeeze_axis0_f32`           | 4.590 us | 2.606 us | 1.761x |
| new `views/squeeze_axis0_f32`           | 3.238 us | 2.375 us | 1.363x |
| new `interop/asarray_squeeze_axis0_f32` | 4.335 us | 2.352 us | 1.843x |

the direct native squeeze leaf moved from about 713 ns to 578 ns after
unchecked source-derived views, but the public `mnp.squeeze(existing, axis=0)`
barely moved. that is useful measurement. the native validation split did its
job, and now the Python boundary is the visible floor.

the same pattern showed up in one-axis view helpers:

| operation                            |  before |   after | read                                             |
| ------------------------------------ | ------: | ------: | ------------------------------------------------ |
| `mnp.moveaxis(s_mp, 0, -1)`          | 5.23 us | 2.38 us | native single-axis view path                     |
| `mnp.flip(..., axis=0)`              | 1028 ns |  821 ns | tuple/axis parsing removed                       |
| `z[::-1]` with cached native reverse |  409 ns |  185 ns | native view reused, Python wrapper remains fresh |

the important point is not that every view row is solved. the point is that a
view row can contain zero data movement and still lose. a view is a storage
retention contract plus a shape/stride object plus a Python wrapper. optimize
the atom the benchmark actually measures.

## 4. complex kernels are now mostly a boundary story

the complex research note already made the core representation decision:
interleaved real and imaginary lanes, C99-compatible, BLAS-friendly. the recent
work tested that decision against the benchmark frontier.

the biggest complex win was not a new complex multiply formula. it was calling
the right Apple BLAS symbol. `complex/matmul_64_complex64` went from
16.257 us to 7.277 us after routing macOS complex GEMM through
`cblas_cgemm$NEWLAPACK$ILP64`, a 2.23:1 monpy-side speedup. NumPy's sample was
already in the ILP64 symbol; monpy was using the wrong ABI lane.

the smaller wins were storage-level and wrapper-level:

| item                                    |                                       movement | read                                                  |
| --------------------------------------- | ---------------------------------------------: | ----------------------------------------------------- |
| complex64 to complex128 contiguous cast |                           5.289 us to 3.298 us | lane-wise interleaved cast                            |
| reversed complex64 add                  |     5.807 us to 3.591 us over the later passes | fused negative-stride kernel plus cached reverse view |
| complex64 multiply                      |                           3.497 us to 3.210 us | float32x4 interleaved SIMD pair kernel                |
| `asarray_complex64`                     | 2.841 us to 2.632 us in one direct-buffer pass | removed a detector hop, wrapper still dominates       |

the remaining complex frontier around `1.2x` to `1.4x` is mostly ingress,
copy, and Python wrapper allocation. do not invent a new complex arithmetic
algorithm to fix a row whose leaf already spends sub-microseconds in native
code. the leaf is no longer the largest cost.

## 5. pure Mojo rows say the kernel core is close

the full pure-Mojo macOS sweep had 114 rows. 36 were slower than baseline,
only one was above `1.25x`, none were above `1.5x`, and the median ratio was
`0.980x`. that means most monpy kernels are already at or ahead of the
stdlib-shaped baselines.

NuMojo needed a labeling fix before it could say anything useful. the old
NuMojo rows inverted the contract: `candidate = numojo`, `baseline = monpy`.
after orientation was repaired, the rows read like the rest of the harness:
`candidate = monpy`, `baseline = numojo`. the result was not "monpy is slower
than NuMojo." for the kernel-level rows we measured, monpy was faster:

| row                                | monpy candidate | NuMojo baseline |  ratio |
| ---------------------------------- | --------------: | --------------: | -----: |
| `numojo.elementwise/add_f32_1024`  |           92 ns |          465 ns | 0.198x |
| `numojo.reductions/sum_f32_1024`   |           29 ns |          210 ns | 0.138x |
| `numojo.elementwise/sin_f32_65536` |       401238 ns |       468333 ns | 0.857x |

that does not prove a Python-facing API win over NuMojo. it proves something
narrower and more useful: monpy's raw kernel path is not obviously behind the
external Mojo array baseline. the public benchmark frontier is therefore more
likely to live in facade and ownership machinery than in the arithmetic leaf.

## 6. threading is a threshold policy

the 8-way reducer note was easy to misread. `REDUCE_SIMD_ACCUMULATORS = 8` is
instruction-level parallelism inside one worker, not "spawn eight workers." the
recent threading passes forced the distinction.

`sync_parallelize` has a real per-call toll. direct parallel reduction rows
like `max_par_f32_1m` were up to 57x slower than the serial reducer when they
were emitted as calibration rows in the default Mojo sweep. the fix was to
remove those rows from the production sweep and delete unreachable parallel
reduction kernels after the gate made them dead. serial reductions remain the
right production answer for the current suite.

threading does win where the row is wide enough:

| row            |  cap | threaded |    serial |  ratio |
| -------------- | ---: | -------: | --------: | -----: |
| `add_f32_1m`   | auto | 60245 ns | 122483 ns | 0.492x |
| `exp_f32_256k` |    4 | 39389 ns | 114000 ns | 0.346x |
| `exp_f64_256k` |    4 | 89905 ns | 320280 ns | 0.281x |

but 64K light ops stay serial. `exp_f32_64k` was still a loss at cap 4
(31569 ns vs 29488 ns, `1.071x`). the policy that survives is:
small rows stay serial, wide transcendentals can fan out, and reductions need
a persistent worker story before they deserve production threads.

## 7. attention is ahead locally, but it needs larger rows

the attention stack recovered from generic reduction, `where`, and weak-scalar
slow paths. after the Float32 softmax row kernel, the small local attention rows
were below NumPy:

| row                                                |                          before |                           after |
| -------------------------------------------------- | ------------------------------: | ------------------------------: |
| `attention/softmax/causal_scores_t32_f32`          |   8.506 us vs 10.500 us, 0.807x |   6.869 us vs 10.052 us, 0.683x |
| `attention/attention/causal_attention_t32_d32_f32` |  20.681 us vs 22.056 us, 0.931x |  19.256 us vs 22.242 us, 0.864x |
| `attention/gpt/tiny_gpt_logits_t32_d32_v128_f32`   | 81.256 us vs 105.450 us, 0.777x | 80.921 us vs 108.756 us, 0.745x |

the row-element threading policy exists because this surface is brittle. a
32x32 attention softmax should not spawn workers. a 1024x4096 row softmax
probably should. the benchmark suite now needs that larger attention row so the
policy is calibrated on the curve, not on one tiny point.

## 8. linalg needs coverage before cleverness

tiny `eigh` made two fixed costs visible: LAPACK setup for a 2x2 matrix and
`Python.evaluate("[]")` for two-return values. the closed-form 2x2 symmetric
path helped, but replacing `Python.evaluate("[]")` with `std.python.Python.list`
was the broad win. it moved `eigh_4_*`, `svd_2_*`, `svd_4_*`, and `qr_2_*`
without inventing new numeric shortcuts.

the important followup was benchmark coverage. after adding an `array/linalg_api`
group, the suite mapped 35 of 35 public `monpy.linalg` names except
`LinAlgError`. the worst newly visible rows were not LAPACK:

| row                                            |     monpy |    numpy |  ratio |
| ---------------------------------------------- | --------: | -------: | -----: |
| `array/linalg_api/kron_2x2_f64`                | 73.044 us | 9.889 us | 7.398x |
| `array/linalg_api/tensordot_axes1_4x5_5x3_f64` | 27.478 us | 5.879 us | 4.672x |
| `array/linalg_api/vecmat_16_f64`               | 11.684 us | 2.691 us | 4.343x |
| `array/linalg_api/vdot_32_f64`                 |  9.849 us | 2.480 us | 3.973x |

this is the new linalg frontier. `kron` walks Python lists and scalar getters
for a 2x2 input. `tensordot` pays transpose, contiguous materialization, reshape,
then matmul. `dot` and friends are small scalar-product paths crossing too much
Python for too little arithmetic.

`slogdet` got a native finish because pulling a determinant scalar back through
Python `math.log(abs(det))` was boundary nonsense. it moved
`slogdet_16_f64` from 4.753 us to 4.178 us. useful hygiene, not the main prize.

## 9. compat surface moved too

not every recent item was a perf row.

`monpy.random` now has the first native slice: explicit immutable keys,
`split`, `fold_in`, module helpers, and a minimal `default_rng` wrapper over
native samplers. the open work is full NumPy random parity: `RandomState`,
state serialization, `choice`, permutation/shuffle, and the distribution zoo.

`vmap` now exists as an eager JAX-shaped wrapper for flat positional axes,
keyword axis-0 mapping, tuple/list/mapping outputs, and `out_axes` placement.
the limit matters as much as the call shape: eager `vmap` is a behavior pin.
graph `vmap` is still the performance model, with batching rules over
primitives.

the SciPy/JAX planning note now has the next pressure map: finish the array-api
namespace, use eager `vmap` tests as a guardrail, land a first `special` subset,
then add linalg wrappers that reuse the dense core. that order matters because
SciPy is a 5-to-1 integration problem. five existing array families have to act
like one substrate before stats and special functions stop being a pile of
private walkers.

## next notes to write

the formal research corpus still lacks notes for several recent seams:

- cpython buffer ingress and why the buffer protocol is the portable midpoint
  between `__array_interface__` and the NumPy C API.
- view construction economics: storage retention, source-derived unchecked
  views, wrapper allocation, and why a no-copy operation can still lose by 2 us.
- threading policy: row-element budgets, `sync_parallelize` fixed toll,
  persistent-worker prerequisites, and why reductions stayed serial.
- linalg small-matrix fixed costs: LAPACK workspace/query overhead, Python list
  construction, closed-form 2x2 eigens, and the newly exposed `kron` /
  `tensordot` / dot-family frontier.
- random and transform surface: explicit-key RNG as the first real state model,
  eager `vmap` as behavior pin, graph batching as the real compiler contract.

## references

1. `docs/opts.md`, the optimization ledger for the May 8-9 passes.
2. `docs/benchmarks.md`, benchmark runner contracts, Mojo/NuMojo orientation,
   threading harness, and artifact layout.
3. `docs/ffi-marshaling.md`, the buffer-protocol explanation and residual
   interop-cost model.
4. `docs/architecture.md`, current layer ownership across `src/buffer.mojo`,
   `src/create/`, `src/elementwise/`, `python/monpy/linalg.py`, and
   `python/monpy/random.py`.
5. `docs/numpy-port-gaps.md` and `docs/scipy-jax-port-gaps.md`, the compat
   surface maps for NumPy, SciPy, and JAX.
