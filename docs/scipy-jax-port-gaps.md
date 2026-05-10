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

## performance lesson from jax

jax does not get linalg performance from Python wrappers. the public functions in
`jax.numpy.linalg` and `jax.scipy.linalg` mostly normalize dtype/shape/api
contracts, then lower into `jax._src.lax.linalg` primitives such as `cholesky_p`,
`lu_p`, `qr_p`, `svd_p`, `eig_p`, and `triangular_solve_p`. those primitives
register abstract eval, autodiff, batching, and platform lowerings. on cpu, many
lowerings dispatch through jaxlib LAPACK FFI/custom calls; on gpu they dispatch
through solver libraries or HLO where that wins.

the monpy analogue should be:

1. Python facade: numpy/scipy/jax-compatible argument handling, namedtuple
   returns, mode flags, and clear errors.
2. native primitive: one monpy operation per actual kernel family, with shape,
   dtype, batch, and backend metadata.
3. kernel backend: typed Mojo for small shapes, BLAS/LAPACK for dense numeric
   kernels, and explicit fallback only when the fast path is unavailable.
4. batching rule: graph-level `vmap` moves the mapped dimension through the
   primitive. it must not loop in Python for performance paths.

monpy already has most of layer 3 for dense rank-2 cpu linalg:

- BLAS/LAPACK dylib resolution lives in `src/accelerate.mojo`.
- matmul dispatch already chooses GEMV, complex GEMM, small typed SIMD, GEMM,
  then scalar fallback.
- dense linalg already reaches LAPACK for `solve`, `inv`, `det`, `qr`,
  `cholesky`, `eig`, `eigh`, `svd`, `lstsq`, and `pinv`.
- the layout-copy tax is now a real perf target. contiguous row-major to
  column-major copy paths are roughly 10-to-1 faster than generic
  physical-offset copies at `n=128`, so dense linalg work should keep reusable
  contiguous scratch buffers and fast copy paths close to the kernel boundary.

therefore the next performance work is not "port jax code." it is to expose the
missing primitives, remove Python-side loops from SciPy-shaped wrappers, and make
batched core dimensions first-class.

jax's useful surface trick is generalized-function signatures, for example
`(m,m),(m)->(m)` and `(m,m),(m,n)->(m,n)` around `solve`. monpy should copy that
signature model before trying to copy the whole transformation stack: it gives a
precise way to say which axes are core linalg axes and which axes are batch axes.
autodiff hooks (`custom_jvp`, `custom_linear_solve`) are later work, after the
primitive contracts are stable.

## scipy patterns

steal these:

- high-level API normalization that ends in compiled kernels. scipy's public
  linalg functions validate shapes, dtype, structure flags, overwrite/check
  options, and batch dimensions before calling lower-level BLAS/LAPACK-backed
  code.
- driver awareness. `lstsq`/SVD APIs should say which LAPACK driver monpy uses
  first instead of pretending to expose every scipy driver at once.
- array-api namespace checks and backend capability tables.

avoid these for performance paths:

- fallback copies to numpy/scipy for unsupported backends. that is acceptable as
  correctness glue in scipy, but it is not a monpy performance design.
- facade-level batching loops. batch dimensions belong in native primitive
  batching rules or generated gufunc kernels.
- sparse, banded, tridiagonal, and special-function fallback systems pretending
  to be dense linalg wrappers. those are separate subsystems.

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

near-term performance work:

- `solve_triangular`: bind a real triangular-solve primitive (`trsm`/`trsv` or
  LAPACK `trtrs`) instead of reducing through general `solve`.
- `cho_factor`, `cho_solve`: expose `potrf` as a factorization result, then solve
  with triangular solves. this reuses the factorization the same way jax's
  `cho_solve` composes two `triangular_solve` calls.
- `lu_factor`, `lu_solve`: expose the existing `getrf` path and add `getrs`.
  current `solve` can use `gesv`, but repeated solves should not refactor `a`.
- `block_diag`: native pad/concat writer. jax implements this by padding each
  block then concatenating; monpy should lower it to one allocation and copy
  loop.
- `toeplitz`, `circulant`, `hankel`: native constructor kernels over strided
  index formulas. scipy uses stride tricks plus copy; jax uses vectorized
  gather-style construction. monpy should write directly into row-major output.
- batched `solve`, `cholesky`, `qr`, `svd`, and `eig`: add graph-level `vmap`
  batching rules over core shapes `(n,n)`, `(n)`, and `(n,k)`, not Python loops.
- `multi_dot`: replace left-fold matmul with dynamic-programmed chain ordering.
  jax documents cases where the chosen ordering changes work by 20-to-1, for
  example `600000` vs `30000` scalar multiply/add operations.
- `inv`: eventually route through the same factorization/RHS batching path as
  `solve(A, I)`, so improvements compound instead of splitting implementation
  paths.

later subsystems:

- banded and tridiagonal solvers
- Schur, QZ, Hessenberg, and polar decompositions
- matrix functions: `expm`, `logm`, `sqrtm`, trig/hyperbolic matrix functions
- Sylvester, Lyapunov, and Riccati equation solvers
- low-level BLAS/LAPACK facade functions

function backlog:

| priority | surface                                         | implementation note                                                                                    |
| -------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| p0       | `solve_triangular`                              | add triangular solve primitive and tests for vector/matrix rhs, trans/conj, lower/upper, unit diagonal |
| p0       | `lu_factor`, `lu_solve`, `lu`                   | expose `getrf` pivots and bind `getrs`; preserve pivot-index convention at the Python facade           |
| p0       | `cho_factor`, `cho_solve`                       | expose `potrf`; implement solve as two triangular solves                                               |
| p1       | gufunc batch signatures                         | represent core signatures such as `(n,n),(n)->(n)` and `(n,n),(n,k)->(n,k)`                            |
| p1       | batched dense linalg                            | graph `vmap` rules for solve/cholesky/qr/svd/eig over leading batch axes                               |
| p1       | `multi_dot`, `cond`                             | chain-order optimization and condition-number facade over existing SVD/norm pieces                     |
| p1       | `block_diag`, `toeplitz`, `circulant`, `hankel` | native output constructors; no Python element loops                                                    |
| p1       | `qr_multiply`                                   | reuse QR factor and apply Q through native matmul/ormqr-style primitive                                |
| p2       | `eigh_tridiagonal`, banded solvers              | requires dedicated tridiagonal/banded storage and LAPACK bindings                                      |
| p2       | `schur`, `hessenberg`, `polar`                  | requires new LAPACK primitive families and return-shape contracts                                      |
| p3       | `expm`, `sqrtm`, `solve_sylvester`              | algorithm-heavy matrix functions; do after Schur/triangular solves                                     |

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

monpy's matching public shape is now:

```python
import monpy as mp

@mp.jit
def f(x, w, bias):
  return mp.where(mp.einsum("ij,jk->ik", x, w) > 0, bias, 0)
```

The important part is the function boundary. `mp.jit` traces the Python
function body with abstract inputs. Every NumPy-compatible public operation used
inside that body must either bind a monpy primitive directly or decompose into
primitive calls before any eager ndarray work happens. `monpy.lax` stays the
primitive/spec namespace; top-level `monpy` is the user-facing transform and
NumPy facade.

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
3. expose missing dense linalg primitives: triangular solve, reusable LU and
   Cholesky factorizations, and factor-solve entrypoints.
4. add gufunc-style core signatures for linalg wrappers, then graph batching
   rules for those primitives.
5. use the eager `vmap` tests as a behavior pin while designing graph batching.
6. land the first `special` subset, because it unlocks stats and common ML math.
7. add linalg wrappers around existing dense kernels: triangular, block/special
   matrices, Cholesky solve, LU solve, condition number, and chain-ordered
   `multi_dot`.
8. add stats operations that reuse existing primitives before distributions.
9. add signal windows, then FFT-backed convolution/correlation.
10. defer sparse, full optimize, integrate, interpolate, ndimage, full random
    distributions, and io until the core array and transform contracts stop moving.

## coverage homes

future code should add focused parity tests under:

- `tests/python/scipy_compat`
- `tests/python/jax_compat`

test rows should cite upstream provenance the same way the numpy compat suite
does: adapted source, upstream commit, and the local reason for every xfail.
