---
title: matrix power dispatch
date: 2026-05-09
---

_for `2x2`, `n == 3`, the extension boundary cost more than the two multiplies._

## anchor

The local slow row was `array/linalg_api/matrix_power_2_n3_f64`:

- before: `7.161 us` monpy, `3.966 us` NumPy, `1.809x`
- after native float path + native shape guard: `4.564 us` monpy, `4.024 us` NumPy, `1.170x`
- monpy-side cut: `1.57:1`

The profiler said the same thing louder:

- before: `20,000` calls, `840,004` Python calls, `0.465 s`
- after native matrix-power path: `480,004` Python calls, `0.261 s`
- after native shape guard: `220,004` Python calls, `0.117 s`

The hot path was not burning cycles on arithmetic. It was building Python
shape tuples and calling `matmul` twice through the extension boundary.

## source contract

- NumPy documents `matrix_power(a, n)` for `(..., M, M)` arrays. Positive
  powers use repeated squaring plus matrix multiplication; `n == 0` returns an
  identity stack with the input dtype; negative powers invert first and then use
  `abs(n)` ([NumPy docs](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_power.html)).
- NumPy source coerces `n` with `operator.index`, raises a dedicated integer
  exponent `TypeError`, handles object arrays separately, then special-cases
  `n == 0`, `n == 1`, `n == 2`, and `n == 3` before binary decomposition
  ([NumPy source](https://github.com/numpy/numpy/blob/main/numpy/linalg/_linalg.py#L3269-L3452)).
- JAX mirrors the same public shape contract and names repeated squaring as the
  implementation strategy ([JAX docs](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.matrix_power.html)).
- SciPy sparse is a different object: it accepts non-negative sparse powers and
  warns that recursive sparse powers can lose to direct products depending on
  fill-in ([SciPy sparse docs](https://scipy.github.io/devdocs/reference/generated/scipy.sparse.linalg.matrix_power.html)).
  That is not the dense CPU target here.

## algorithm

For dense arrays, the robust positive-power algorithm is:

```text
if n == 0: return I
if n == 1: return A
if n == 2: return A @ A
if n == 3: return (A @ A) @ A

result = none
z = A
while n > 0:
  n, bit = divmod(n, 2)
  if bit:
    result = z if result is none else result @ z
  if n > 0:
    z = z @ z
return result
```

**Proposition.** For `n > 3`, binary decomposition uses at most
`floor(log2(n)) + popcount(n) - 1` dense matrix multiplications.

Sketch:

- Squaring creates the powers `A^(2^i)` needed by each binary digit.
- There are `floor(log2(n))` squarings up to the highest set bit.
- Multiplying selected powers into `result` costs `popcount(n) - 1` because the
  first selected power initializes `result`.
- Total: `floor(log2(n)) + popcount(n) - 1`.

Examples:

|   n | linear multiplies | binary multiplies | linear/binary |
| --: | ----------------: | ----------------: | ------------: |
|   3 |                 2 |                 2 |        1.00:1 |
|   8 |                 7 |                 3 |        2.33:1 |
|  16 |                15 |                 4 |        3.75:1 |

For the benchmark row (`n == 3`), both algorithms do two multiplies. The
measured win is boundary deletion: Python calls fell `840,004 -> 220,004`,
cProfile wall fell `0.465 s -> 0.117 s`, and the `2x2` multiply stayed local
instead of paying a BLAS frame for four output cells.

## monpy boundary

The landed path is deliberately a float try-fast:

- accepts `float32` and `float64`
- accepts rank-2 and stacked `(..., M, M)` square arrays
- supports `n == 0`, `n == 2`, `n == 3`, and larger positive powers
- returns `None` for `n < 0`, `n == 1`, unsupported dtypes, and invalid shapes
- keeps Python responsible for NumPy-compatible exponent coercion and fallback

Why this boundary:

- Positive float powers preserve dtype and are safe to compute with the current
  real `get_logical_as_f64` / `set_logical_from_f64` helpers.
- Integer powers need exact overflow semantics, not a float64 accumulator.
- Complex positive powers need the complex multiply path, not real-only helpers.
- Negative powers inherit `inv`; complex negative-power dtype is still a known
  gap in the inverse path.

The small-matrix cutoff is `N <= 16`.

- For `2x2`, BLAS setup dominates arithmetic. Four cells and eight multiply-add
  terms do not need the vendor library ceremony.
- For larger rank-2 matrices, the path defers to `maybe_matmul_contiguous`, so
  Accelerate/OpenBLAS still owns the dense GEMM case.
- For stacked matrices, the current path loops per matrix. That is correct and
  avoids needing a general batched matmul refactor inside this increment.

## tests

The regression fence is:

- float32/float64 powers for `n in [0, 1, 2, 3, 5, 8]`
- `numpy.int64(3)` exponent coercion
- `1.0` exponent rejection
- `LinAlgError` for rank-1 and non-square inputs
- stacked float64 `(..., 2, 2)` positive power
- strided float64 `2x2` positive power

## next

- Implement complex positive `matrix_power` with `_complex_real`,
  `_complex_imag`, and `_complex_store`.
- Decide whether integer positive powers should stay Python fallback or get a
  typed exact-overflow path.
- Fix complex inverse dtype before claiming negative complex parity.
