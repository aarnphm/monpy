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
currently exports 45 public names, with 43 overlapping numpy names. that means
the raw namespace gap is 452 names, roughly 91 percent of numpy's public
top-level namespace. the real gap is larger than that, because numpy's internals
also define behavior behind those names.

## current monpy surface

covered or partially covered today:

- array construction from python scalars, lists, tuples, supported numpy arrays,
  array-interface objects, and cpu dlpack producers.
- dtype set with full kernel support: `bool`, `int64`, `float32`, `float64`.
- dtype set with metadata only (allocation raises `NotImplementedError`
  with `"phase-5a kernel work"` until kernels land): `int32`, `int16`,
  `int8`, plus `intp` alias.
- storage ownership for managed arrays, borrowed external arrays, and views.
- shape and stride metadata, c/f-contiguous checks, negative strides, zero
  strides, basic view safety.
- indexing for integers, slices, reverse slices, ellipsis, and zero-dimensional
  scalar access, plus `newaxis` / `None` insertion.
- creation: `empty`, `zeros`, `ones`, `full`, `empty_like`, `zeros_like`,
  `ones_like`, `full_like`, `copy`, `ascontiguousarray`, `arange`, `linspace`,
  `eye`, `identity`, `tri`, `atleast_1d`/`atleast_2d`/`atleast_3d`,
  `logspace`, `geomspace`, `meshgrid`, `indices`, `ix_`.
- shape/view operations:
  - layout-only views: `reshape`, `transpose`, `matrix_transpose`,
    `broadcast_to`, `expand_dims`, `flip`, `fliplr`, `flipud`, `rot90`,
    `squeeze`, `moveaxis`, `swapaxes`, `diagonal`, `trace`.
  - copy-out: `ravel`, `flatten`, `concatenate`, `stack`, `hstack`,
    `vstack`, `dstack`, `column_stack` (current implementation goes
    through python-level per-element flat-write; native LayoutIter
    kernel is the recorded perf follow-up).
- elementwise functions: `add`, `subtract`, `multiply`, `divide`, `sin`, `cos`,
  `exp`, `log`, `where`, and the explicit `sin_add_mul` fused kernel.
- reductions with `axis=None`: `sum`, `mean`, `min`, `max`, `argmax`.
- matmul for 1-d and 2-d operands, including several dense and strided layout
  cases.
- linalg namespace: `matmul`, `matrix_transpose`, `solve`, `inv`, `det`,
  `LinAlgError`.
- internal infrastructure (no public surface): vendored CuTe-style
  layout algebra at `src/cute/` — `IntTuple`, `Layout`, `composition`,
  `coalesce`, `complement`, `select`/`transpose`, `logical_divide`,
  `LayoutIter`, `MultiLayoutIter`. used by the proof-of-concept
  unary-strided migration; available for future kernel rewrites.

major local blockers already called out in compatibility coverage:

- unsupported dtype families: complex, object, string, structured, unsigned,
  datetime.
- phase-5a "registered but unallocatable" dtypes: int32/int16/int8 raise
  `NotImplementedError` on allocation; metadata layer (kind, itemsize,
  alignment, byteorder, format-char, iinfo, can_cast, result_type,
  promote_types, isdtype) is fully wired.
- boolean indexing and integer-array fancy indexing.
- axis-aware reductions and reduction keyword controls.
- higher-rank matmul.
- full ufunc objects and ufunc methods.

## missing core library

### dtype and scalar system

numpy's dtype layer is the root of most behavior. monpy should grow this before
wide api expansion.

missing pieces:

- dtype registry for the full numpy dtype universe. the supported v1 dtypes now
  have native registry-backed metadata and cast/promotion query helpers.
- scalar aliases and scalar classes for signed integers, unsigned integers,
  floating types, complex types, strings, object, datetime, timedelta, and void.
- dtype metadata for later families: native-endian checks, field descriptors,
  subarray descriptors, canonical names beyond the v1 set, and format strings.
- scalar extraction and python scalar conversion per dtype.
- cast safety, cast loops, and astype behavior across every supported dtype pair.
  the current four-dtype surface has a native cast-copy path and an adapted
  numpy cast-matrix test; the wider dtype families are still missing.
- promotion and type discovery beyond the supported v1 set:
  `min_scalar_type`, `common_type`, and wider-family behavior for
  `result_type`, `promote_types`, `can_cast`, `issubdtype`, `isdtype`, `finfo`,
  and `iinfo`.

monpy implementation note:

- add a native dtype layer, likely `src/dtype.mojo`, and make python's dtype
  registry a mirror of that table.
- keep dtype ids compact, but stop treating them as the whole dtype model.
- widen dtype support in batches. signed integers first, then unsigned integers,
  then smaller floats if the mojo/runtime support is sane, then complex. string,
  object, datetime, and structured dtypes should wait until the core loop and
  casting contracts are stable.

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
- buffer-protocol ingestion as the fast portable path. the current supported
  dtypes use the native dtype format decoder shared with the registry helpers.
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

- n-dimensional broadcast iterator for arbitrary operand counts.
- output allocation and `out=` validation against shape, dtype, and writeability.
- inner-loop selection by dtype, contiguity, alignment, and stride pattern.
- negative-stride and zero-stride correctness.
- axis iteration for reductions and cumulative operations.
- boolean mask iteration and indexed gathers/scatters.
- a shared executor for unary, binary, ternary, reduction, copy, cast, and
  assignment loops.

monpy implementation note:

- build the iterator and loop executor before adding dozens of new functions.
  otherwise every new function will invent a private stride walker.

### ufunc machinery

today monpy has plain python functions for a small set of elementwise operations.
numpy has `ufunc` objects with methods and dispatch rules.

missing pieces:

- ufunc-like objects with call semantics.
- dtype resolution and loop selection per ufunc.
- `out=`, `where=`, `casting=`, `order=`, `dtype=`, and `subok=` handling where
  relevant.
- methods: `reduce`, `accumulate`, `reduceat`, `outer`, and `at`.
- identities and reduction dtype policy.
- floating-point error handling for divide/invalid/overflow/underflow.
- broad math, comparison, logical, and bitwise function families.

high-priority ufunc families:

- arithmetic: `negative`, `positive`, `absolute`, `power`, `floor_divide`,
  `remainder`, `mod`, `fmod`, `divmod`, `reciprocal`, `square`, `sqrt`.
- comparisons: `equal`, `not_equal`, `less`, `less_equal`, `greater`,
  `greater_equal`.
- logical: `logical_and`, `logical_or`, `logical_xor`, `logical_not`.
- bitwise integer ops: `bitwise_and`, `bitwise_or`, `bitwise_xor`,
  `bitwise_not`, shifts, and bit counts.
- floating tests: `isnan`, `isinf`, `isfinite`, `isclose`, `signbit`.
- min/max: `minimum`, `maximum`, `fmin`, `fmax`.
- transcendental: `sqrt`, `tan`, inverse trig, hyperbolic functions, `log1p`,
  `log2`, `log10`, `exp2`, `expm1`.

monpy implementation note:

- make existing `add`, `subtract`, `multiply`, `divide`, `sin`, `cos`, `exp`,
  and `log` ufunc-like before adding the long tail.
- keep the `log(-inf)` parity fix pinned while building this layer; full ufunc
  objects are still missing.

### reductions and statistics

axis-none scalar reductions are only the first case.

missing pieces:

- `axis` as an int, negative int, tuple of ints, and `None`.
- `keepdims`, `out`, `where`, `initial`, and dtype controls.
- result-shape calculation for reductions.
- cumulative reductions over axes.
- `prod`, `all`, `any`, `argmin`, `nan*` reductions, `std`, `var`, `median`,
  `quantile`, `percentile`, `average`, `count_nonzero`, histograms, and
  correlation/covariance helpers.

monpy implementation note:

- implement reduction shape planning and axis iteration once. reuse it for ufunc
  reductions, `sum`, `prod`, `any`, `all`, `min`, `max`, arg reductions, and
  statistics.

### indexing and assignment

basic indexing exists. advanced indexing is still missing.

missing pieces:

- boolean indexing and assignment.
- integer-array fancy indexing and assignment.
- mixed basic and advanced indexing semantics.
- `take`, `put`, `take_along_axis`, `put_along_axis`, `nonzero`, `argwhere`,
  `flatnonzero`, `where(condition)` form.
- helpers: `indices`, `ix_`, `ravel_multi_index`, `unravel_index`,
  `diag_indices`, `tril_indices`, `triu_indices`.

monpy implementation note:

- fancy indexing should use gather/scatter machinery shared with `take` and
  `put`; do not grow it out of the current slice-only view code.

### shape manipulation and array creation helpers

many missing functions can become python-level wrappers after core view and
iterator semantics are correct.

shipped (phase 6a / 6b):

- creation helpers: `eye`, `identity`, `tri`, `meshgrid`, `logspace`,
  `geomspace`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `indices`, `ix_`.
- manipulation helpers: `flip`, `fliplr`, `flipud`, `rot90`,
  `squeeze`, `moveaxis`, `swapaxes`, `ravel`, `flatten`,
  `concatenate`, `stack`, `hstack`, `vstack`, `dstack`,
  `column_stack`.

still missing:

- creation helpers: `frombuffer`, `fromiter`, `asfortranarray`
  (needs F-order support that monpy v1 doesn't have), `tril`, `triu`.
- manipulation helpers: `block`, `split`, `array_split`, `hsplit`,
  `vsplit`, `dsplit`, `roll`, `repeat`, `tile`, `pad`, `append`,
  `insert`, `delete`, `trim_zeros`, `broadcast_arrays`.
- memory checks: `shares_memory`, `may_share_memory`, contiguity helpers.

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

monpy has a useful base, but numpy's linear algebra and tensor operation surface
is much wider.

missing pieces:

- tensor ops: `dot`, `vdot`, `vecdot`, `inner`, `outer`, `tensordot`, `einsum`,
  `einsum_path`, `kron`, `cross`, batched `matmul`, `matvec`, `vecmat`.
- linalg: `norm`, `vector_norm`, `matrix_norm`, `matrix_rank`, `matrix_power`,
  `slogdet`, `qr`, `cholesky`, `eig`, `eigh`, `eigvals`, `eigvalsh`, `svd`,
  `svdvals`, `pinv`, `lstsq`, `multi_dot`, `tensorinv`, `tensorsolve`.
- broadcasting and stacked-matrix semantics for linalg functions.
- hermitian/symmetric-specialized paths where numpy exposes them.

monpy implementation note:

- keep blas/lapack-backed paths for dense f32/f64 cpu arrays.
- use mojo fallback kernels for portability, small shapes, and dtype paths not
  covered by the linked backend.
- add batched shape semantics before adding the higher-level linalg wrappers.

### sorting, searching, and set operations

missing pieces:

- `sort`, `argsort`, `partition`, `argpartition`, `lexsort`,
  `searchsorted`, `unique`, `unique_all`, `unique_counts`, `unique_inverse`,
  `unique_values`, `isin`, `intersect1d`, `union1d`, `setdiff1d`, `setxor1d`,
  `bincount`, `digitize`.

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

1. fix current editable/import stability before expanding the surface.
2. introduce a first-class dtype registry and casting table.
3. build a shared strided-loop executor with broadcast, output, and axis
   planning.
4. turn existing elementwise functions into ufunc-like objects.
5. implement axis-aware reductions on the shared executor.
6. add advanced indexing using shared gather/scatter machinery.
7. add cheap creation and shape-manipulation wrappers once copy/view semantics
   are enforceable.
8. widen dtypes in batches and update promotion/casting tests each time.
9. add tensor/linalg functions once batched shape semantics exist.
10. treat `random`, `fft`, `ma`, `strings`, `records`, `polynomial`, and io as
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
