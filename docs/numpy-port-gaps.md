# numpy port gap notes

monpy is currently a mojo-native array runtime with a numpy-shaped python facade.
the missing numpy surface is not just a list of public functions. numpy's public
api sits on a large internal library: dtype descriptors, scalar types, coercion,
casting, strided iteration, ufunc dispatch, reduction machinery, indexing,
linear algebra, random number generators, io helpers, and compatibility modules.

these notes are a planning map for porting that library shape into monpy without
copying numpy's c-extension layout literally.

## upstream shape

snapshot target:

- public compat baseline in the local test suite: numpy `v2.4.4`, commit
  `be93fe2960dbf49b4647f5783c66d967fb2c65b5`.
- upstream source map inspected from
  <https://github.com/numpy/numpy/tree/main/numpy>.
- exact counts from `main` are drift-prone. refresh them before using the counts
  as release evidence.

observed upstream tree shape:

| upstream path                 |                               entries |
| ----------------------------- | ------------------------------------: |
| `numpy/`                      |  50 entries: 22 directories, 28 files |
| `numpy/_core/`                |   71 entries: 4 directories, 67 files |
| `numpy/_core/src/multiarray/` | 120 entries: 2 directories, 118 files |
| `numpy/_core/src/umath/`      |                              60 files |
| `numpy/lib/`                  |                            65 entries |
| `numpy/random/`               |                            34 entries |
| `numpy/linalg/`               |                            11 entries |

the top-level public namespace is a poor implementation guide by itself. in the
local numpy `2.4.4` install, numpy exposes 495 public top-level names; monpy
currently exports 396 public names, with 252 overlapping numpy names. that means
the raw namespace gap is 243 names, roughly 49 percent of numpy's public
top-level namespace. the public-name count now includes monpy-native dtype,
kernel, and neural-network helpers, so overlap is the useful compatibility
number. the real gap is larger than the raw missing-name set, because numpy's
internals also define behavior behind those names.

## current monpy surface

covered or partially covered today:

- array construction from python scalars, lists, tuples, supported numpy arrays,
  array-interface objects, and cpu dlpack producers.
- dtype set with allocation, promotion, casting metadata, and arithmetic:
  `bool`, signed integers (`int8`/`int16`/`int32`/`int64`), unsigned integers
  (`uint8`/`uint16`/`uint32`/`uint64`), `float16`/`float32`/`float64`, and
  `complex64`/`complex128`.
- storage ownership for managed arrays, borrowed external arrays, and views.
- shape and stride metadata, c/f-contiguous checks, negative strides, zero
  strides, basic view safety.
- indexing for integers, slices, reverse slices, ellipsis, zero-dimensional
  scalar access, `newaxis` / `None` insertion, boolean masks, and
  integer-array fancy indexing, including assignment.
- creation: `empty`, `zeros`, `ones`, `full`, `empty_like`, `zeros_like`,
  `ones_like`, `full_like`, `copy`, `ascontiguousarray`, `arange`, `linspace`,
  `eye`, `identity`, `tri`, `atleast_1d`/`atleast_2d`/`atleast_3d`,
  `logspace`, `geomspace`, `meshgrid`, `indices`, `ix_`.
- shape/view operations:
  - layout-only views: `reshape`, `transpose`, `matrix_transpose`,
    `broadcast_to`, `expand_dims`, `flip`, `fliplr`, `flipud`, `rot90`,
    `squeeze`, `moveaxis`, `swapaxes`, `diagonal`, `trace`.
  - copy-out: `ravel`, `flatten`, `concatenate`, `stack`, `hstack`,
    `vstack`, `dstack`, `column_stack`, `split`, `array_split`, `hsplit`,
    `vsplit`, `dsplit`, `block`, `roll`, `repeat`, `tile`, `pad`,
    `append`, `insert`, `delete`, `trim_zeros`, and `broadcast_arrays`.
- elementwise functions: arithmetic, comparisons, logical operations,
  predicates, common transcendentals, complex conjugate/real/imag/angle, and
  the explicit `sin_add_mul` fused kernel.
- reductions and statistics: `sum`, `mean`, `min`, `max`, `prod`, `all`, `any`,
  `argmin`, `argmax`, cumulative variants, nan-aware variants, `std`, `var`,
  `median`, `quantile`, `percentile`, `average`, and `count_nonzero`.
- matmul for 1-d and 2-d operands, including dense, transposed,
  f-contiguous, offset, negative-stride, matrix-vector, vector-matrix, and
  complex `cgemm`/`zgemm` cases.
- tensor operations: `dot`, `vdot`, `inner`, `outer`, `tensordot`, `einsum`,
  `kron`, `cross`, `matvec`, `vecmat`, and `vecdot` for the adapted v1
  behavior.
- linalg namespace: `matmul`, `matrix_transpose`, `solve`, `inv`, `det`, `qr`,
  `cholesky`, `eig`, `eigh`, `eigvals`, `eigvalsh`, `svd`, `svdvals`, `lstsq`,
  `pinv`, `matrix_rank`, `matrix_power`, `slogdet`, `multi_dot`, norm helpers,
  `tensorinv`, `tensorsolve`, and `LinAlgError`.
- sorting, searching, indexing, and set-operation helpers: `sort`, `argsort`,
  `partition`, `searchsorted`, `digitize`, `unique`, `bincount`, `isin`,
  `intersect1d`, `union1d`, `setdiff1d`, `setxor1d`, `nonzero`, `argwhere`,
  `flatnonzero`, `ravel_multi_index`, `unravel_index`, `tril`, `triu`,
  `tril_indices`, and `triu_indices`.
- internal infrastructure (no public surface): vendored CuTe-style
  layout algebra at `src/cute/` — `IntTuple`, `Layout`, `composition`,
  `coalesce`, `complement`, `select`/`transpose`, `logical_divide`,
  `LayoutIter`, `MultiLayoutIter`. used by the proof-of-concept
  unary-strided migration; available for future kernel rewrites.

major local blockers already called out in compatibility coverage:

- unsupported dtype families: object, string, structured, datetime, and
  timedelta.
- higher-rank / batched matmul and stacked linalg semantics.
- full numpy random, fft, masked-array, string, polynomial, io, testing, and
  f2py subsystems.
- deep ufunc/reduction tail: `where=` keyword support on ufuncs,
  `reduce(initial=...)`, `reduceat`, complete floating-point error-state
  behavior, bitwise integer ufuncs, and the c-api-level machinery behind
  numpy's extension ecosystem.
- memory aliasing and formatting helpers: `shares_memory`, `may_share_memory`,
  `array2string`, `array_repr`, print options, and the wider family of
  convenience wrappers that do not belong in core kernels.

## missing core library

### dtype and scalar system

numpy's dtype layer is the root of most behavior. monpy should grow this before
wide api expansion.

missing pieces:

- dtype registry for the full numpy dtype universe. the supported v1 dtypes now
  have native registry-backed metadata and cast/promotion query helpers.
- numpy scalar class hierarchy and abstract scalar aliases: `generic`,
  `number`, `integer`, `signedinteger`, `unsignedinteger`, `floating`,
  `complexfloating`, `flexible`, and friends. monpy has `DType` descriptors and
  dtype aliases, not numpy-compatible scalar classes.
- dtype metadata for later families: native-endian checks, field descriptors,
  subarray descriptors, canonical names beyond the v1 set, and format strings.
- scalar extraction and python scalar conversion per dtype.
- cast safety, cast loops, and astype behavior across every supported dtype pair.
  the current numeric surface has native cast-copy paths and adapted numpy
  cast-matrix tests; object, string, datetime, timedelta, structured, and record
  families are still missing.
- promotion and type discovery beyond the supported v1 set:
  `min_scalar_type`, `common_type`, and wider-family behavior for
  `result_type`, `promote_types`, `can_cast`, `issubdtype`, `isdtype`, `finfo`,
  and `iinfo`.

monpy implementation note:

- add a native dtype layer, likely `src/dtype.mojo`, and make python's dtype
  registry a mirror of that table.
- keep dtype ids compact, but stop treating them as the whole dtype model.
- keep widening dtype support in batches. signed integers, unsigned integers,
  float16, complex64, and complex128 are already in the public surface. the
  low-precision native storage dtypes need explicit interchange semantics, while
  string, object, datetime, timedelta, and structured dtypes should wait until
  the core loop, ownership, and casting contracts are stable.

### coercion and construction

numpy's array construction machinery handles nested python objects, dtype
discovery, array-like protocols, copy rules, memory ownership, and cast decisions
as one pipeline.

missing pieces:

- a single coercion engine for python objects, buffers, dlpack, numpy arrays,
  array-interface objects, nested sequences, scalars, and existing monpy arrays.
- robust shape discovery, including ragged rejection and scalar-vs-sequence
  handling per numpy's rules.
- dtype discovery and requested-dtype resolution in the same pass.
- copy policy that is shared by `array`, `asarray`, `copy`, `astype`, dlpack,
  buffer protocol, and numpy interop.
- buffer-protocol ingestion as the fast portable path. `frombuffer` exists and
  the current supported dtypes use the native dtype format decoder shared with
  the registry helpers, but this is not yet one unified coercion planner.
- optional numpy c-api ingestion as the fastest numpy-specific path if abi
  coupling is worth it.

monpy implementation note:

- replace the python-only `_flat` / `_infer_dtype` path over time with a native
  coercion plan that returns shape, dtype, copy/view decision, and source
  ownership.
- keep numpy as an interchange boundary, not as an internal implementation
  dependency for implemented operations.

### strided iteration

numpy's internals rely heavily on generalized iteration over shapes, strides,
broadcasted operands, and typed inner loops. monpy needs this layer before the
public api can grow cleanly.

missing pieces:

- n-dimensional broadcast iterator for arbitrary operand counts that every
  operation can share. today some families have their own walkers or
  python-level coordinate loops.
- output allocation and `out=` validation against shape, dtype, casting, and
  writeability across all elementwise, reduction, copy, cast, and assignment
  paths.
- masked loop execution for `where=` and scatter/gather operations.
- native join/split/copy kernels for paths that currently flatten and rewrite
  element by element in python.
- one shared executor for unary, binary, ternary, reduction, copy, cast, and
  assignment loops instead of family-specific dispatch code.

monpy implementation note:

- build the iterator and loop executor before adding dozens of new functions.
  otherwise every new function will invent a private stride walker.

### ufunc machinery

monpy now has `Ufunc` objects for the core elementwise surface, but numpy's
ufunc contract is still wider than the current wrapper.

missing pieces:

- complete dtype resolution and loop selection per ufunc for every supported
  dtype family.
- `where=`, `order=`, and `subok=` handling where relevant, plus stricter
  `out=`, `casting=`, and `dtype=` validation.
- method tail: `reduce(initial=...)`, `reduceat`, and more exhaustive `at`
  semantics. `reduce`, `accumulate`, `outer`, and a minimal `at` exist.
- identities and reduction dtype policy across the wider dtype set.
- floating-point error handling for divide/invalid/overflow/underflow.
- bitwise integer ufunc families and the remaining floating-point helper tail.

high-priority ufunc families:

- arithmetic tail: `fmod`, `divmod`, `float_power`, `ldexp`, `frexp`,
  `nextafter`, `spacing`, and `heaviside`.
- bitwise integer ops: `bitwise_and`, `bitwise_or`, `bitwise_xor`,
  `bitwise_not`, shifts, and bit counts.
- floating tests and comparisons: `isclose`, `allclose`, `array_equal`,
  `array_equiv`, and tolerance helpers.
- clipping/convolution/difference helpers: `clip`, `convolve`, `correlate`,
  `diff`, `ediff1d`, and gradient-style wrappers.

monpy implementation note:

- keep the `log(-inf)` parity fix pinned while filling this layer. do not grow
  the long tail by adding private one-off walkers where the shared ufunc
  executor should own the behavior.

### reductions and statistics

axis-none scalar reductions, axis reductions, keepdims, dtype controls, and many
statistics helpers exist for the adapted v1 surface.

missing pieces:

- `where` and `initial` support across reductions and ufunc reductions.
- fuller `out=` validation and dtype policy for mixed integer/float/complex
  reductions.
- axis-aware `quantile` / `nanquantile` tails and more exact numpy method
  coverage.
- histograms, correlation/covariance helpers, and window/statistics utilities
  outside the current ndarray core.

monpy implementation note:

- implement reduction shape planning and axis iteration once. reuse it for ufunc
  reductions, `sum`, `prod`, `any`, `all`, `min`, `max`, arg reductions, and
  statistics.

### indexing and assignment

basic indexing, boolean masks, integer-array fancy indexing, assignment,
`take`, `put`, `take_along_axis`, `put_along_axis`, `nonzero`, `argwhere`,
`flatnonzero`, `where(condition)` form, `indices`, `ix_`,
`ravel_multi_index`, `unravel_index`, `diag_indices`, `tril_indices`, and
`triu_indices` exist for the adapted v1 surface.

missing pieces:

- the full numpy mixed basic/advanced indexing contract for every exotic
  broadcast placement.
- advanced indexing performance work that keeps gathers/scatters inside native
  typed loops instead of Python-level coordinate walks.

monpy implementation note:

- fancy indexing should use gather/scatter machinery shared with `take` and
  `put`; do not grow it out of the current slice-only view code.

### shape manipulation and array creation helpers

many missing functions can become python-level wrappers after core view and
iterator semantics are correct.

shipped (phase 6a / 6b plus the 2026-05-07 tail):

- creation helpers: `eye`, `identity`, `tri`, `meshgrid`, `logspace`,
  `geomspace`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `indices`, `ix_`.
- manipulation helpers: `flip`, `fliplr`, `flipud`, `rot90`,
  `squeeze`, `moveaxis`, `swapaxes`, `ravel`, `flatten`,
  `concatenate`, `stack`, `hstack`, `vstack`, `dstack`,
  `column_stack`, `block`, `split`, `array_split`, `hsplit`,
  `vsplit`, `dsplit`, `roll`, `repeat`, `tile`, `pad`, `append`,
  `insert`, `delete`, `trim_zeros`, and `broadcast_arrays`.

still missing:

- memory checks: `shares_memory`, `may_share_memory`, contiguity helpers.
- convenience helpers: `broadcast_shapes`, `diag`, `diagflat`,
  `diag_indices_from`, `fill_diagonal`, `fromfunction`, `choose`, `compress`,
  `select`, `piecewise`, `place`, `apply_along_axis`, and `apply_over_axes`.
- keyword tails such as `concatenate(out=...)`, `stack(out=...)`, sparse
  `meshgrid`, sparse `indices`, and non-default order modes.

implementation notes:

- `flip`/`fliplr`/`flipud`/`rot90` are zero-cost layout views (stride
  negation + offset bump). bench shows ~0.8–1.7× monpy/numpy with
  python wrapper as the only overhead.
- `concatenate`/`stack`/`hstack`/`vstack` currently go through a
  python-level per-element flat-write into a fresh array. that's
  correct for any rank/dtype but hits ~58000× monpy/numpy on N=256
  inputs. the recorded perf follow-up is a native concatenate kernel
  walking output positions via `LayoutIter`, copying source slices
  with `memcpy` along the trailing axis when both sides are
  c-contiguous along that axis.

monpy implementation note:

- implement wrappers only when the backing primitives preserve numpy copy/view
  semantics. view-producing functions must not silently copy unless numpy does.

### linear algebra and tensor operations

monpy has a useful base, including LAPACK-backed decompositions and tensor
wrappers, but numpy's stacked/batched semantics and parser tails are still much
wider.

missing pieces:

- higher-rank / batched `matmul`, `dot`, `inner`, `cross`, `matvec`, and
  `vecmat` semantics.
- stacked-matrix semantics for `solve`, `inv`, decompositions, rank, norms,
  tensor helpers, and error cases.
- full `einsum`: ellipsis, output arrays, optimize planning, path reporting,
  and `einsum_path`.
- matrix norm `ord` variants, `multi_dot` cost optimization, and full
  hermitian/symmetric-specialized behavior where numpy exposes it.
- dtype/backend widening for every linalg routine, including complex and
  lower-precision paths where LAPACK or Mojo fallbacks differ.

monpy implementation note:

- keep blas/lapack-backed paths for dense f32/f64 cpu arrays.
- use mojo fallback kernels for portability, small shapes, and dtype paths not
  covered by the linked backend.
- add batched shape semantics before adding the higher-level linalg wrappers.

### sorting, searching, and set operations

the first slice exists: `sort`, `argsort`, `partition`, `searchsorted`,
`digitize`, `unique`, `bincount`, `isin`, `intersect1d`, `union1d`,
`setdiff1d`, and `setxor1d` are covered for adapted v1 behavior.

missing pieces:

- `argpartition` and `lexsort` coverage for the implemented subset.
- axis/order/stability variants across sort-like routines.
- numpy 2 unique-family helpers: `unique_all`, `unique_counts`,
  `unique_inverse`, and `unique_values`.
- structured dtype sorting/searching behavior, object/string comparisons, and
  performance work beyond python-level loops.

monpy implementation note:

- start with contiguous 1-d numeric arrays and explicit stable/unstable policy.
  widen to axes and structured dtypes later.

### random

numpy's `random` module is its own library, not a thin helper file.

missing pieces:

- bit generators: `PCG64`, `PCG64DXSM`, `MT19937`, `Philox`, `SFC64`.
- `Generator`, legacy `RandomState`, seeding, state serialization, and pickle
  behavior.
- distributions: uniform, normal, integers, binomial, poisson, exponential,
  gamma, beta, chisquare, dirichlet, multinomial, choice, permutation, shuffle,
  and the rest of numpy's distribution surface.

monpy implementation note:

- this should be a separate subsystem after core ndarray semantics are stable.
  use numpy's distribution tests as behavioral references, but avoid copying the
  random module's cython layout.

### fft

missing pieces:

- `fft`, `ifft`, `rfft`, `irfft`, multidimensional variants, hermitian variants,
  frequency helpers, normalization modes, dtype policy, and axis handling.

monpy implementation note:

- this likely wants a backend decision: bind a system/library fft or write a
  small educational fallback. do not block core ndarray parity on fft.

### strings, char, object, structured, and records

missing pieces:

- `numpy.strings`, `numpy.char`, string dtype operations, bytes/string scalar
  behavior.
- object arrays and reference-count-aware storage.
- structured dtypes, record arrays, field indexing, nested fields, and void
  scalars.

monpy implementation note:

- these features require dtype metadata and ownership semantics that do not
  exist yet. keep them late.

### masked arrays

missing pieces:

- `numpy.ma`, `MaskedArray`, mask propagation, fill values, masked reductions,
  and masked indexing behavior.

monpy implementation note:

- treat this as an optional high-level module after ndarray core behavior is
  stable. it should not leak into the core array representation unless there is
  a measured reason.

### io, printing, and compatibility modules

missing pieces:

- `load`, `save`, `savez`, `loadtxt`, `genfromtxt`, text parsing, `.npy` and
  `.npz` format support.
- formatting and printing: `array2string`, `array_repr`, `array_str`,
  print options, float formatting.
- modules and helpers: `ctypeslib`, `typing`, `testing`, `f2py`, `matlib`,
  `matrixlib`, `rec`, `polynomial`, `lib` wrappers.

monpy implementation note:

- `.npy` format support is probably the first useful io target.
- testing helpers should stay out of the runtime path.
- `f2py` is out of scope unless monpy intentionally becomes a numpy-compatible
  build ecosystem, which is a different project.

## implementation order

the fastest route to meaningful numpy parity is not to add public names first.
it is to add the machinery that makes new names cheap and correct.

1. keep the coverage and gap ledgers synchronized with runtime behavior.
2. finish the shared strided-loop executor: broadcast, output, axis, mask,
   casting, and gather/scatter planning.
3. close ufunc and reduction keyword tails: `where=`, `initial=`, `reduceat`,
   stricter `out=`, and floating-point error-state behavior.
4. add higher-rank / batched matmul and stacked linalg semantics.
5. move python-level join/split/sort/index loops into native kernels where they
   are now correctness-only and visibly slow.
6. fill the numpy scalar hierarchy and later dtype families only when ownership,
   casting, and scalar extraction contracts are explicit.
7. add cheap convenience wrappers once copy/view/error semantics are enforceable.
8. treat `random`, `fft`, `ma`, `strings`, `records`, `polynomial`, and io as
   later subsystems, not prerequisites for ndarray core parity.

## tracking rule

every time a missing family moves from deferred to implemented, update
`tests/python/numpy_compat/COVERAGE.md` with one of:

- `covered`, when monpy matches numpy for the adapted behavior.
- `blocked`, when monpy intentionally raises for a v1 gap.
- `xfail`, when the test suite has a strict failing parity target.
- `deferred`, when the behavior is outside the current implementation surface.

do not mark a namespace as covered because the import exists. coverage means
behavioral tests against numpy exist for the relevant dtype, shape, stride, copy,
and error cases.
