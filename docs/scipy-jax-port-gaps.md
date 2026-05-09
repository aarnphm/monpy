# scipy and jax port gap notes

monpy's numpy work is the base layer. scipy and jax add pressure from two
directions: scipy wants a broad numerical library surface that can consume array
api namespaces, while jax wants transformable primitives (`vmap`, `jit`, and the
autodiff family) rather than ad hoc eager loops.

these notes are a planning map for the next compatibility slice. they are not a
claim that monpy implements the whole scipy or jax surface.

## upstream shape

snapshot targets:

- scipy source inspected at `/Users/aarnphm/workspace/scratchpad/scipy`, commit
  `771d831ee90710631b82978a7964dff2264e429f`.
- jax source inspected at `/Users/aarnphm/workspace/scratchpad/jax`, commit
  `da271dd6f4cfa0fe45cce44a06d74cc8fec0f814`.
- exact module counts are drift-prone. refresh them before using the counts as
  release evidence.

observed scipy tree shape:

| scipy module    | impl files | test files | array-api refs |
| --------------- | ---------: | ---------: | -------------: |
| `stats`         |        144 |         72 |            159 |
| `signal`        |         71 |         27 |            123 |
| `special`       |        146 |        187 |             71 |
| `interpolate`   |         52 |         18 |             44 |
| `fft`           |         25 |          9 |             42 |
| `integrate`     |         56 |         11 |             24 |
| `cluster`       |         14 |          0 |             19 |
| `sparse`        |        377 |         19 |             12 |
| `ndimage`       |         48 |         16 |             12 |
| `optimize`      |        190 |         49 |              8 |
| `spatial`       |         54 |         41 |              8 |
| `constants`     |          8 |          4 |              7 |
| `datasets`      |          7 |          3 |              5 |
| `differentiate` |          4 |          3 |              3 |
| `linalg`        |         85 |         50 |              3 |
| `io`            |         81 |         96 |              0 |
| `fftpack`       |         19 |         16 |              0 |
| `odr`           |         15 |          3 |              0 |

the array-api reference count is a useful pressure signal, not an implementation
rank by itself. `stats`, `signal`, and `special` have the hottest array-api
contact surface. `linalg`, `sparse`, `optimize`, and `io` are still large even
when they do not carry many array-api decorators.

## scipy triage

### array api gateway

scipy's `_lib/_array_api.py` is the first seam to satisfy. it handles namespace
selection, backend checks (`jax`, `torch`, `cupy`, `numpy`), device helpers,
copy-to-numpy fallbacks, and test support. monpy already exposes
`__array_namespace__`, but the next useful scipy pass needs more namespace
completeness:

- `xp.asarray(..., copy=...)`, `astype`, `concat`, `permute_dims`,
  `matrix_transpose`, `vecdot`, `isdtype`, and strict dtype metadata.
- device reporting that scipy can interrogate without falling back to numpy.
- stable shape, dtype, and scalar conversion semantics for scipy's array-api
  test helpers.
- no implicit numpy execution inside operations that are marked as implemented.

### linalg

monpy already has the dense core: `solve`, `inv`, `det`, QR, Cholesky, eig,
SVD, least squares, pseudo-inverse, matrix rank, powers, slogdet, norms, and
tensor solve helpers. scipy's linalg namespace is much wider.

near-term wrappers:

- `solve_triangular`
- `block_diag`
- `toeplitz`, `circulant`, `hankel`
- `cho_factor`, `cho_solve`
- `lu_factor`, `lu_solve`

later subsystems:

- banded and tridiagonal solvers
- Schur, QZ, Hessenberg, and polar decompositions
- matrix functions: `expm`, `logm`, `sqrtm`, trig/hyperbolic matrix functions
- Sylvester, Lyapunov, and Riccati equation solvers
- low-level BLAS/LAPACK facade functions

### special

`scipy.special` is mostly ufunc-shaped. that makes it a good monpy target once
the ufunc executor and dtype policy are less fragmented.

first subset, using jax's scipy subset as the practical guide:

- logistic and entropy helpers: `expit`, `logit`, `entr`, `rel_entr`,
  `kl_div`, `xlogy`, `xlog1py`
- error and normal helpers: `erf`, `erfc`, `ndtr`, `ndtri`, `log_ndtr`
- gamma family: `gamma`, `gammaln`, `gammasgn`, `digamma`, `betaln`
- neural-network staples: `logsumexp`, `softmax`, `log_softmax`
- Bessel tail needed by common kernels: `i0`, `i0e`, `i1`, `i1e`

the hard part is not just formula code. these functions need broadcasted ufunc
semantics, dtype resolution, error-state behavior, and transformability under
future graph-level `vmap`.

### stats

`scipy.stats` has the strongest array-api pressure in the inspected tree. the
first monpy slice should avoid the whole distribution zoo and target operations
that mostly reuse existing ndarray primitives:

- `rankdata`, `mode`, `sem`
- `entropy`, `zscore`, `gzscore`
- simple hypothesis-test kernels after `special` lands enough CDF/log-CDF
  support
- selected distribution `pdf`, `cdf`, `logpdf`, and `logcdf` methods only after
  random and special-function contracts are stable

this pulls on `sort`, `unique`, `take_along_axis`, `put_along_axis`, reductions,
nan-aware reductions, and special functions. it is a 5-to-1 integration problem:
five existing array families need to behave as one statistics substrate.

### signal, fft, sparse, and the rest

- `signal`: windows are the low-cost first slice. convolution, correlation, and
  filtering need a real FFT/convolution subsystem.
- `fft`: blocked on an FFT backend and dtype/device policy.
- `sparse`: separate storage model, classes, indexing, conversions, and sparse
  matmul. treat this as its own project: COO/CSR/CSC first, graph/linalg later.
- `optimize`, `integrate`, `interpolate`, `spatial`, `ndimage`, `io`: useful, but
  mostly separate algorithm systems rather than ndarray-core compatibility.

## jax transform target

jax's top-level api exports `jit`, `grad`, `value_and_grad`, `jacfwd`, `jacrev`,
`hessian`, `jvp`, `vjp`, `vmap`, `pmap`, `shard_map`, `custom_jvp`,
`custom_vjp`, device helpers, `lax`, `random`, and `scipy`.

the actionable monpy slice is `vmap` first, but there are two different things
called `vmap`:

- eager `vmap`: a correctness wrapper that slices mapped arguments, calls the
  function repeatedly, and stacks results. this is useful for tests and user
  code on small arrays.
- graph `vmap`: a batching transform over traced primitives with a batching rule
  per primitive. this is the real jax model and the only version that belongs on
  performance-sensitive code paths.

monpy now has the eager call shape at top level:

```python
import monpy as mp

f = mp.vmap(lambda x, y: x + y, in_axes=(0, None), out_axes=0)
out = f(mp.asarray([1, 2, 3]), mp.asarray(10))
```

minimum contract for the current eager path:

- `vmap(fun, in_axes=0, out_axes=0, axis_name=None, axis_size=None,
spmd_axis_name=None, sum_match=False)`
- flat positional `in_axes` as `int`, `None`, or a flat sequence matching the
  positional argument count
- keyword arguments mapped over axis 0
- inferred or explicit `axis_size`
- equal mapped-axis sizes across arguments
- tuple, list, and mapping outputs with matching `out_axes` structure
- `out_axes=None` only for outputs that do not vary across mapped calls

explicit non-goals for the eager path:

- pytree prefix matching beyond flat argument and output structures
- named collectives via `axis_name`
- `pmap`, `shard_map`, device sharding, or mesh semantics
- autodiff transforms (`grad`, `jvp`, `vjp`, custom rules)
- zero-size mapped axes, because the loop cannot infer output structure without
  executing the function body

## graph batching backlog

the graph-level version should follow jax's primitive-batcher model:

1. add a `BatchTrace`-like layer for monpy kernel tensors: each traced value is
   either unmapped or carries one mapped dimension.
2. register batching rules for elementwise, broadcast, reshape, squeeze,
   transpose, concatenate, split, pad, reduce, sort, gather, scatter, and
   `dot_general`/matmul.
3. make batched output axes explicit, then lower axis moves and broadcasts into
   graph nodes rather than python loops.
4. add generalized ufunc signatures, for example `(n,n),(n)->(n)`, so scipy and
   jax scipy wrappers can express batched linalg without private walkers.
5. only then consider autodiff transforms. `vmap` and `grad` compose in jax, but
   they are different systems.

## port order

1. keep the numpy ledger green while finishing dtype, ufunc, reduction, and
   indexing tails.
2. complete the array-api namespace slice needed by scipy's `_array_api.py`.
3. use the eager `vmap` tests as a behavior pin while designing graph batching.
4. land the first `special` subset, because it unlocks stats and common ML math.
5. add linalg wrappers around existing dense kernels: triangular, block/special
   matrices, Cholesky solve, and LU solve.
6. add stats operations that reuse existing primitives before distributions.
7. add signal windows, then FFT-backed convolution/correlation.
8. defer sparse, full optimize, integrate, interpolate, ndimage, random, and io
   until the core array and transform contracts stop moving.

## coverage homes

future code should add focused parity tests under:

- `tests/python/scipy_compat`
- `tests/python/jax_compat`

test rows should cite upstream provenance the same way the numpy compat suite
does: adapted source, upstream commit, and the local reason for every xfail.
