---
title: JAX-first architecture for monpy
date: 2026-05-10
---

# JAX-first architecture for monpy

_JAX traces Python into typed primitive equations before it commits to an
execution backend._

This note is the architecture research pass for a serious monpy restructure.
The target is not "clone JAX" and not "be NumPy with Mojo kernels." The target
is:

1. a JAX-shaped primitive and transform architecture as the spine;
2. an eager interpreter that preserves NumPy-compatible ndarray layout,
   broadcasting, dtype promotion, and view semantics;
3. a Mojo runtime that plans kernels from primitive + dtype + layout metadata
   instead of accumulating one-off `maybe_*` fast paths.

That means NumPy compatibility becomes an interpreter contract over monpy
primitives, not the architecture itself.

## What the current tree says

The repo already contains three partial architectures:

| surface              | current evidence                                                                                                                                                                           | architectural read                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Python eager ndarray | `python/monpy/__init__.py` is 5336 lines and owns dtype objects, ndarray methods, coercion, scalar policy, ufunc spelling, creation, indexing, and many dispatch calls.                    | This is the NumPy compatibility interpreter, but it is too close to being the source of truth. |
| Python graph path    | `python/monpy/_src/core.py`, `api.py`, `lax/tensor.py`, and `interpreters/tracing.py` define `Primitive`, `GraphIR`, `TensorSpec`, `jit`, eager `vmap`, custom calls, and layout metadata. | This is the JAX-shaped spine, but eager APIs mostly bypass it.                                 |
| Mojo runtime         | `src/` is about 18.5k Mojo lines split across storage, array, domain, buffer, create ops, elementwise kernels, linalg, rng, and Accelerate FFI.                                            | The kernel core is real, but the planner is implicit in hand-ordered dispatcher cascades.      |

There was also doc drift worth fixing before the restructure got bigger.
`docs/architecture.md` now names the live NumPy interop boundary
(`python/monpy/numpy/ops.py`), the 21-dtype registry, and the top-level
`jit`/`vmap` transform front doors. `README.md` still needs a later pass for
older monolithic path references such as `src/create.mojo` and
`src/elementwise.mojo`, while the live tree has split packages.

The problem is not "no structure." The problem is a 2:1 truth problem:
eager NumPy-shaped dispatch and graph primitive dispatch can describe the same
operation through different paths. That is where special cases breed.

## Upstream lessons

### JAX

JAX architecture has a clean spine:

```text
Python function
  -> trace against abstract values
  -> jaxpr equations over primitives
  -> lower to backend IR
  -> compile for target
  -> execute with concrete arrays
```

For monpy, the user-facing spelling should be:

```python
import monpy as mp

@mp.jit
def f(x, w):
  return mp.einsum("ij,jk->ik", x, w)
```

The staged unit is the whole Python function. `einsum`, `where`, `sum`, and the
rest of the NumPy-compatible API are not individually "jitted" functions. They
are ordinary public front doors that must be traceable when the surrounding
function runs under `jit`. During tracing they can inspect static metadata such
as shape, dtype, rank, layout, axes, and literal kwargs. They cannot inspect
array values or call `_native`.

The important part is not XLA itself. The important part is the primitive rule
table. A primitive has to know how to:

- execute eagerly;
- compute output abstract values from input abstract values;
- lower to a backend;
- batch under `vmap`;
- differentiate later, if autodiff is in scope;
- represent effects and higher-order subgraphs where needed.

JAX's own docs are blunt about this: JAX-traceable functions compose because
their runtime behavior is either shape/type inspection or primitive calls.
`jax.numpy` is transformable because it lowers NumPy-looking functions into that
primitive layer.

The implication for monpy:

- `monpy.add`, `ndarray.__add__`, `monpy.lax.add`, and `GraphIR(add)` should be
  four spellings over one primitive, not four call paths.
- The eager ndarray path should be just one interpreter of primitives.
- `jit` should not need a parallel shadow API. It should trace the same public
  operations the eager runtime executes.

### NumPy

NumPy's useful lesson is the ndarray and ufunc contract:

- an array is storage plus metadata: dtype, shape, strides, offset/base, flags;
- broadcasting is a view/iteration rule, not a data-copy rule;
- ufuncs wrap type-specific 1-D loops with setup logic for broadcasting,
  casting, output arguments, buffering, and error handling;
- that setup overhead is real, so small arrays can lose to simpler specialized
  paths.

The implication for monpy:

- Keep NumPy-compatible storage/layout semantics, including arbitrary strides,
  negative strides, zero strides, base-owner lifetime, and `copy=` behavior.
- Do not implement every primitive as bespoke shape arithmetic plus bespoke
  strided loops. Build a reusable iterator/planner substrate.
- Keep explicit small-array escape hatches, but hang them off a planner, not
  ad hoc dispatch order embedded in each op.

### PyTorch

PyTorch's useful lesson is `layout, device, dtype` as extension axes, plus
`TensorIterator` for regular elementwise kernels. The public tensor surface can
stay dynamic, but kernels should receive a normalized iteration plan that has
already decided broadcasting, promotion, layout, output allocation, and whether
vectorized access is legal.

The implication for monpy:

- The monpy equivalent of `TensorIterator` / `NpyIter` should be a first-class
  Mojo concept.
- Elementwise kernels should accept an `IterationPlan` and typed pointers, not
  re-derive broadcast and stride cases in every dispatcher.
- Device is v1 `cpu`, but it should already be present in abstract values and
  primitive signatures so future GPU/Metal work does not mutate every API.

### Array API and SciPy

The Array API standard exists because downstream libraries need a portable
namespace and sharper semantics than NumPy's whole historical surface. SciPy's
array API support uses the principle "array type in equals array type out" and
strictly rejects some NumPy long-tail inputs.

The implication for monpy:

- Treat `monpy.array_api` as a strict interpreter over monpy primitives.
- Treat `monpy.numpy` / `monumpy` as compatibility wrappers with explicit
  deviations.
- Do not let SciPy/JAX pressure force the eager compatibility layer to become
  the compiler architecture.

### Mojo

Mojo's useful substrate is already aligned with this: `SIMD[DType, width]`,
`vectorize`, `parallelize`, reductions, and `LayoutTensor` all point toward
typed, planned kernels. `LayoutTensor` also teaches the right separation:
layout metadata plus pointer, with tiling/vectorization/distribution layered on
top.

The implication for monpy:

- Keep the dynamic `Array` record for Python-owned/eager ndarray semantics.
- Convert dynamic arrays into static-enough execution plans at the primitive
  boundary.
- Use parametric Mojo kernels for the inner loops, selected by a planner.
- Use `LayoutTensor`-style ideas for tile kernels and future GPU work, but do
  not force the entire eager ndarray runtime into static layouts.

### Nabla

Nabla is a useful nearby project because it is also Python on MAX/Mojo, but it
chooses a tensor-framework contract rather than NumPy layout compatibility.
The repo is pinned locally at:

```text
/Users/aarnphm/workspace/scratchpad/nabla
commit 08618486b05a92270e7ead4f357180dd90e77389
```

Useful pieces to steal conceptually:

- One `Operation` base owns the operation lifecycle: argument validation,
  sharding propagation, structural cache identity, graph node creation, and
  autodiff hooks. Monpy's equivalent should be primitive binding plus an
  execution plan, not class-per-op inheritance as the main runtime shape.
- Nabla has a small JAX-style pytree module with `PyTreeDef`, stable dict-key
  ordering, flatten/unflatten/map, and a custom-node registry. That is almost
  exactly the transform-boundary utility monpy needs.
- Its `compile` transform keys caches by tensor signatures, static values, and
  pytree structure. That is the right cache shape for monpy too, with layout
  and primitive graph hash added.
- It records lightweight `OpNode`s holding op, args, kwargs, output treedef,
  and structural hash. Monpy's `GraphIR` is already cleaner for compiler
  interchange, but Nabla is a good reminder that output structure must be
  stored outside raw tensor nodes.
- Custom Mojo kernels enter through a named operation hook. Monpy should expose
  this through `monpy.extend` / `custom_call`, with abstract eval and lowering
  rules required up front.

Cautions:

- Nabla's global `GRAPH` and environment-controlled eager/deferred MAX graph
  mode are convenient, but monpy should avoid making hidden global graph state
  the core transform semantics. `@monpy.jit` should own an explicit trace
  context and cache object.
- Nabla propagates sharding and partial reductions deeply inside each
  `Operation`. Monpy should not import that complexity before the CPU primitive
  spine is stable.
- Nabla is not trying to preserve NumPy view semantics, arbitrary strides,
  `copy=` behavior, and ufunc tail keywords. Monpy's extra work belongs in
  `LayoutSpec`, `IterationPlan`, and eager interpreter rules.

## Target architecture

The spine should look like this:

```text
Public API spellings
  monpy.*, ndarray methods, monpy.numpy, monpy.array_api, monpy.lax

Primitive layer
  PrimitiveSpec(name, arity, attrs, effects)
  abstract_eval(inputs, attrs) -> AbstractValue(s)
  dtype_rule(inputs, attrs) -> dtype(s)
  eager_impl(inputs, attrs) -> ndarray/scalar
  batching_rule(inputs, axes, attrs) -> batched primitive graph
  lowering_rule(target, node) -> backend program/custom call

Interpreters
  eager ndarray interpreter
  tracing interpreter -> GraphIR
  batching interpreter -> GraphIR
  lowering interpreter -> MAX/Mojo custom calls

Mojo runtime
  ArrayRecord(storage, dtype, shape, strides, offset, device, flags)
  ExecutionPlan(primitive, operands, outputs, layout class, backend)
  IterationPlan(coalesced axes, byte cursors, broadcast flags, alignment)
  KernelCandidate(predicate, cost, typed entry point)
  Backend(cpu scalar, cpu SIMD, Accelerate/OpenBLAS, future GPU)
```

The single most important invariant:

> Every public array operation binds exactly one primitive before it executes.

For eager execution, the primitive runs immediately. For `jit`, it becomes a
node. For `vmap`, it runs through a batching rule. For future autodiff, it gets
JVP/VJP rules. NumPy compatibility lives in argument normalization and result
wrapping, not in alternate kernels.

## Proposed Python layout

```text
python/monpy/
  __init__.py                 # re-export facade only
  _api/
    ndarray.py                # ndarray wrapper and dunders
    creation.py               # array/asarray/zeros/ones/etc
    elementwise.py            # add/sin/where/etc spelling
    reductions.py
    indexing.py
    linalg.py
    random.py
  _core/
    abstract.py               # AbstractValue, Shape, DType, Device, Layout
    primitive.py              # PrimitiveSpec and registry
    dispatch.py               # bind primitive in eager or tracing context
    dtypes.py                 # generated or schema-backed dtype metadata
    layout.py                 # Python layout spec for GraphIR and ndarray views
    pytrees.py                # tree flattening for transforms
  _interop/
    numpy.py                  # only NumPy-aware ingress/egress
    dlpack.py
    buffer.py
  _transforms/
    trace.py
    jit.py
    batching.py
    autodiff.py               # later, mostly stubs at first
  _lowering/
    max.py
    mojo.py
```

The current `python/monpy/_src` is already close to `_core` plus `_transforms`.
The main move is to stop making `python/monpy/__init__.py` the operating system
of the project. A 5336-line facade file is carrying about 2.8x the line count of
the whole current `_src` compiler-ish layer. That is upside down for
JAX-first monpy.

## Pytrees

Monpy needs pytrees, but only as a small Python-side transform utility. It does
not need a full JAX-compatible custom pytree registry before the primitive spine
is stable.

The useful v1 contract is:

- `tree_flatten(x) -> (leaves, treedef)` for tuples, lists, dicts with stable key
  order, and `None` as a no-leaf node.
- `tree_unflatten(treedef, leaves)` to rebuild public return structure.
- `tree_map(fn, x)` for transform plumbing and tests.
- `jit.compile(...)` stores input and output `PyTreeDef`s on
  `CompiledFunction` while `GraphIR.inputs` and `GraphIR.outputs` still see
  only tensor leaves.
- `vmap` uses the same treedef for `in_axes` / `out_axes` prefix validation
  instead of hand-rolled tuple/list/mapping logic.

That gives the important JAX property: user-facing Python structure is separate
from tensor leaves. It also keeps Mojo out of the business of Python containers.
Mojo should receive primitive operands, layout metadata, and execution plans,
not pytrees.

Defer:

- user-registered custom pytree nodes;
- dataclasses, namedtuples, and implicit object traversal;
- pytree-aware autodiff rules;
- full JAX error compatibility.

The practical reason to add this early is robustness. Eager `vmap` and
`jit.compile` both need to preserve user-facing Python structure while lowering
only tensor leaves. Centralizing this as a tiny pytree module removes duplicated
structure logic before batching and linalg gufuncs make the duplication harder
to unwind.

## Proposed Mojo layout

```text
src/
  lib.mojo                    # CPython extension registration only
  runtime/
    array.mojo                # ArrayRecord, view construction
    storage.mojo              # ownership, refcount, external finalizer
    dtype.mojo                # generated dtype table and casting rules
    layout.mojo               # dynamic layout predicates and adapters
    buffer.mojo               # CPython buffer bridge
  primitives/
    elementwise.mojo          # primitive eager entry points
    reduction.mojo
    matmul.mojo
    linalg.mojo
    random.mojo
    shape.mojo
  execute/
    planner.mojo              # KernelCandidate selection
    iteration.mojo            # NpyIter/TensorIterator analogue
    outputs.mojo              # allocation, out=, copy policy
    errors.mojo
  kernels/
    elementwise/
      scalar.mojo
      simd.mojo
      strided.mojo
      fused.mojo
    reductions/
    linalg/
    random/
  backends/
    cpu/
      generic.mojo
      parallel.mojo
    apple/
      accelerate.mojo
      vforce.mojo
    linux/
      openblas.mojo
```

This is not a demand to move files first. It is a target ownership map. The
first implementation step should introduce the missing concepts while keeping
most current kernels in place.

## Core contracts

### `AbstractValue`

Python side:

```python
@dataclass(frozen=True, slots=True)
class AbstractValue:
  shape: tuple[int | SymbolicDim, ...]
  dtype: DTypeSpec
  device: Device
  layout: LayoutSpec
  weak_type: bool = False
```

This is the equivalent of JAX avals plus the NumPy layout contract monpy needs.
It must be cheap to construct from an eager `ndarray`, a `TensorSpec`, or a
Python scalar.

### `DTypeSpec`

There should be one canonical dtype object, but it should not absorb every
interface's metadata. Core dtype identity and NumPy dtype presentation are
different jobs.

Core:

```python
@dataclass(frozen=True, slots=True)
class DTypeSpec:
  name: str
  code: int
  kind: DTypeKind
  bits: int
  storage: StorageKind
  storage_bits: int
```

This is the dtype value used by eager arrays, `TensorSpec`, primitive rules,
GraphIR, and Mojo planning. It carries stable monpy semantics: identity,
logical kind, logical precision, storage class, and native domain code. It does
not directly carry NumPy's `typestr`, PEP-3118 format, byte-order spelling,
NumPy scalar class, or buffer-export policy.

Interface-specific metadata should be functions over `DTypeSpec`:

```python
@dataclass(frozen=True, slots=True)
class NumpyDTypeInfo:
  itemsize: int
  alignment: int
  byteorder: str
  typestr: str
  format: str
  scalar_type: type[object]
  buffer_exportable: bool

def numpy_dtype_info(dtype: DTypeSpec) -> NumpyDTypeInfo: ...
def array_interface_typestr(dtype: DTypeSpec) -> str: ...
def buffer_format(dtype: DTypeSpec) -> str: ...
```

Those functions belong to the NumPy/interchange boundary, not the core dtype
record. The same pattern applies to backend-specific metadata:
`max_dtype(dtype)`, `safetensors_names(dtype)`, and future device-specific
lowering helpers should live at their target boundary.

The public ergonomics can still keep familiar names:

```python
DType = DTypeSpec
mp.dtype("float32") is mp.float32
mp.numpy.dtype_info(mp.float32).typestr == "<f4"
```

That gives monpy one dtype identity without making the primitive spine depend
on NumPy's historical descriptor fields.

### `PrimitiveSpec`

```python
@dataclass(frozen=True, slots=True)
class PrimitiveSpec:
  name: str
  arity: int
  effects: frozenset[str]
  abstract_eval: Callable[..., tuple[AbstractValue, ...]]
  eager_impl: Callable[..., object]
  batch: Callable[..., object] | None
  lowerings: Mapping[str, Callable[..., object]]
```

The current `Primitive` class already has several of these fields. It should
become the source of truth for eager and traced operation binding.

### `ExecutionPlan`

Mojo side:

```text
ExecutionPlan
  primitive_code
  input array records
  output array records
  dtype policy
  layout class
  iteration plan
  selected backend
  selected typed kernel
```

The planner replaces the current repeated cascade shape:

```text
try same-shape contiguous
try scalar
try row broadcast
try rank-1 vDSP
try tile
try strided
fallback f64 walker
```

That order may still be the policy. It just should live in data and predicates,
not be reimplemented as if/else prose across `binary_dispatch`, `unary_dispatch`,
reductions, `where`, and future gufuncs.

### `IterationPlan`

This is the big one for NumPy-compatible layout:

```text
IterationPlan
  ndim after broadcasting
  coalesced loop rank
  output shape
  per-operand base pointer
  per-operand byte strides
  zero-stride broadcast flags
  negative-stride flags
  alignment class
  contiguous inner-loop span
  may_vectorize
  may_parallelize
```

This is where the NumPy compatibility belongs. Once the plan exists, a typed
kernel should see a small loop problem, not an ndarray semantics problem.

## SIMD and kernel policy

The SIMD policy should become explicit:

- Use Mojo `SIMD[DType, width]` and `vectorize` for contiguous inner loops.
- Keep scalar loops as the correctness oracle for every primitive.
- Use an alignment-aware allocator so aligned loads are a property of storage,
  not an optimistic guess in each kernel.
- Use one coalesced strided iterator for broadcasted and transposed views.
- Use fixed thresholds for parallelization until a persistent worker/runtime
  exists. Prior local notes showed small reductions can lose by orders of
  magnitude when `parallelize` setup dominates the work.
- Special kernels such as `sin_add_mul`, small matmul, row softmax, and LAPACK
  calls should register as higher-priority `KernelCandidate`s with clear
  predicates.

The current hand-ordered fast paths should be preserved as candidates, then
deleted from the call graph once the planner is trustworthy. No breadcrumbs.

## Random

The current explicit-key `monpy.random` direction matches JAX's architecture.
Keep it:

- key values are arrays/state tokens passed explicitly;
- `split` and `fold_in` are primitive operations;
- distributions are primitives or primitive compositions;
- NumPy legacy helpers wrap the explicit-key core and document state behavior.

This keeps random compatible with `jit`, `vmap`, and future autodiff instead of
installing hidden global state into the transform system.

## Linalg and gufuncs

JAX's linalg performance comes from primitives such as Cholesky, LU, QR, SVD,
eigen, triangular solve, and `dot_general`, not from Python wrappers doing loops.

Monpy should add generalized core signatures before expanding the SciPy/JAX
surface:

```text
matmul:              (..., m, k), (..., k, n) -> (..., m, n)
solve:               (..., n, n), (..., n)    -> (..., n)
solve_matrix_rhs:    (..., n, n), (..., n, k) -> (..., n, k)
cholesky:            (..., n, n)              -> (..., n, n)
svd:                 (..., m, n)              -> tuple outputs
```

Then `vmap` over linalg can become a batching rule, not a Python for-loop. This
is the bridge from NumPy layout compatibility to JAX architecture.

## Migration plan

### Phase 0: pin the live architecture

- Fix doc drift in `docs/architecture.md` and `README.md`.
- Write down current dtype schema as generated data or one canonical table.
- Add a test that public docs and `_native._domain_codes()` agree on dtype names.

### Phase 1: primitive spine owns eager elementwise

- Move primitive definitions for `add`, `sub`, `mul`, `div`, unary ops,
  comparisons, `where`, and reductions into one Python registry.
- Make public eager functions bind primitives, then call the current native
  eager implementation.
- Make tracing reuse the same primitive objects.
- Keep behavior tests unchanged. This should be a structural migration.

Live trace for the first slice:

- `python/monpy/__init__.py:708` defines `Ufunc`; it already owns NumPy-facing
  call semantics for `out=`, `where=`, `casting=`, dtype override, scalar
  scalarization, and reduce methods.
- `python/monpy/__init__.py:766` detects traced operands with
  `_has_kernel_arg(args)` and routes them through `._src.lax.primitives.ufunc`.
- `python/monpy/__init__.py:820`, `:822`, `:857`, `:870`, `:886`, and `:888`
  are the eager native exits for unary-preserve, unary, scalar-binary,
  `binary_into`, and binary.
- `python/monpy/_src/core.py:44` defines `Primitive`, and
  `python/monpy/_src/core.py:173` already registers `add_p`, `sub_p`, `mul_p`,
  `div_p`, `matmul_p`, `reduce_p`, and `where_p`.
- `python/monpy/_src/interpreters/tracing.py:34` has a separate name-to-primitive
  table for traced binary ops, while eager `Ufunc` stores only `_kind` and `_op`.

The safe first patch is therefore:

1. Extend `Primitive` with lightweight eager metadata: native kind, native op
   code, ufunc arity, reduction op, and identity. Do this without changing
   `GraphIR` serialization, which should still depend on primitive names.
2. Construct the public ufunc objects from primitive handles instead of raw
   `(name, kind, op)` tuples. `Ufunc` can keep the same public attributes, but
   internally it should store `self._primitive`.
3. Replace the tracing branch's separate `_PRIMITIVE_BY_NAME` dictionary with
   `get_primitive(name)` plus explicit aliases (`subtract -> sub`,
   `multiply -> mul`, `divide -> div`) in one registry-owned place.
4. Leave eager native exits alone for the first patch. The behavior change is
   structural: eager and traced paths now agree on primitive identity before
   execution, but still call the same current Mojo functions.
5. Add tests that `mp.add._primitive is lax.add_p` and that a traced
   `mp.add(x, y)` and eager `mp.add(a, b)` resolve the same primitive name. This
   catches future drift without touching kernel performance.

This creates a 1:1 primitive identity bridge while preserving the current
fast eager leaf calls. It should be basically performance-neutral: one Python
attribute read per ufunc call, paid before the same native transition already
present today.

Implementation checkpoint, 2026-05-10:

- `Primitive` now carries optional ufunc metadata (`ufunc_kind`, `ufunc_op`,
  arity, identity, and reduction op).
- `PrimitiveRegistry` now owns aliases such as `subtract -> sub`,
  `multiply -> mul`, and `divide -> div`.
- `Ufunc` instances for `add`, `subtract`, `multiply`, and `divide` now point at
  the same primitive handles exported through `monpy.lax`.
- The tracing interpreter resolves binary names through the primitive registry
  instead of a local table.

This is still only the identity bridge. Most public ufuncs do not have
first-class primitive handles yet, and eager execution still exits through the
same `_native.*` functions.

Second checkpoint, 2026-05-10:

- `where` now stages as `where_p` instead of forcing eager `asarray(...)`.
- This is the correct direction for NumPy-compatible operations: they should
  either bind a primitive directly or decompose into primitive calls before
  tracing. The Python wrapper may parse arguments, normalize axes, and choose a
  contraction plan, but it must not execute array work when any operand is a
  traced value.

The bigger unresolved example is `einsum`. JAX's `einsum` is not "jitted"
because `einsum` itself is magic; it is transformable because the implementation
lowers the contraction to primitive operations, with `dot_general` as the core
workhorse. Monpy's current `python/monpy/linalg.py` implementation parses the
subscript string, folds operands left-to-right through `_einsum_pair_contract`,
and calls `tensordot` / `matmul` eagerly. That is fine as a NumPy-compatible
eager implementation, but it is not enough for the JAX-first path.

The target split for `einsum`:

1. Parse subscripts and choose a contraction path in Python. The path is static
   metadata, like JAX's `optimize` handling.
2. For eager arrays, execute the path through the existing native `transpose`,
   `reshape`, `matmul`, reductions, and elementwise primitives.
3. For traced tensors, emit the same path as `GraphIR` nodes. No eager
   materialization, no Python scalar reads, no `_native` calls.
4. Later, recognize common signatures directly: elementwise product,
   reduction, dot, matvec, matmul, batched matmul, and general pair contraction.
   These become batching-rule-friendly primitive patterns rather than opaque
   Python helper calls.

Near-term patch after `where`: add staged `tensordot` / `einsum` decomposition
for the binary matmul-shaped cases first (`ij,jk->ik`, `...ij,...jk->...ik`,
`i,i->`, `ij,ij->`). That gives a real transformable path for the ML-shaped
80% before tackling general contraction path optimization.

Public API checkpoint, 2026-05-10:

- `monpy.jit` now exists as the top-level transform entry point, while
  `monpy.lax` remains the namespace for `TensorSpec`, `GraphIR`, primitive
  handles, and lower-level transform machinery.
- The API surface map now lives in `docs/api-surface.md`, with NumPy, SciPy,
  and JAX families classified by whether they are primitives, composites,
  static helpers, eager-only, or deferred subsystems.
- This is a contract change from "top-level monpy is only NumPy-shaped" to
  "top-level monpy is NumPy-shaped plus JAX-style transforms." That matches the
  actual goal better: users write `mp.einsum` inside `@mp.jit`, and the public
  wrapper lowers into GraphIR.

Reduction checkpoint, 2026-05-10:

- Public reductions now have a staged path. `mp.sum(x, axis=..., keepdims=...)`
  and supported `Ufunc.reduce(...)` calls see traced tensors before `_mat(_av(x))`
  can materialize them, then emit `reduce_p` with `axes`, `keepdims`, and
  `reduce_op` attrs.
- `dtype=` on staged reductions lowers as an explicit `cast_p` before
  `reduce_p`, so dtype intent is visible in GraphIR instead of hiding in a
  native reducer.
- This clears the immediate blocker for staged `einsum` decomposition: dot
  products and final contraction-label sums can now be expressed as
  elementwise/matmul nodes followed by `reduce_p`.

Tensor contraction checkpoint, 2026-05-10:

- `tensordot` now detects traced operands before eager `_array(...)` coercion
  and lowers pair contractions through the existing view and matmul primitives.
- `einsum` now has a staged binary path for no-ellipsis, no-self-diagonal
  expressions. `ij,jk->ik` traces as a direct `matmul`; `i,i->` traces as
  `mul` followed by `reduce_p`; pair contractions with reshaping lower through
  transpose/reshape/matmul/reshape.

Comparison checkpoint, 2026-05-10:

- Comparison ufuncs now share primitive identity with `monpy.lax`:
  `equal_p`, `not_equal_p`, `less_p`, `less_equal_p`, `greater_p`, and
  `greater_equal_p`.
- Staged `Tensor` comparison dunders lower through those primitives, and
  comparison nodes carry bool dtype instead of inheriting the lhs dtype.
- `Tensor.__bool__` now raises during tracing, which prevents Python control
  flow from silently branching on a traced value.
- Non-scalar captured constants are rejected until explicit graph constants or
  external weights exist. This blocks a silent graph-corruption path where a
  captured array could previously become a scalar `constant` node.

The next correctness/performance edge is batched matmul signatures such as
`bij,bjk->bik`. They cannot be flattened into one ordinary GEMM without mixing
batch slices, and Apple Accelerate does not provide the batched GEMM surface
monpy would want. This needs either a native batch loop with explicit scheduling
policy or a typed Mojo batched kernel.

### Phase 2: Mojo execution planner for elementwise

- Introduce `ExecutionPlan` and `IterationPlan` in Mojo.
- Register current contiguous, scalar, strided, complex, vDSP, tile, and fallback
  kernels as candidates.
- Convert binary elementwise first because it has the worst special-case fanout.
- Then port unary, `where`, and reductions.

Success condition: adding a new binary op should touch the primitive table and
the scalar/typed kernel body, not five dispatch files.

### Phase 3: shared dtype/layout schema

- Generate Python dtype specs and Mojo dtype codes from one schema file.
- Keep the canonical dtype schema small: name, code, kind, logical bits, storage
  kind, and storage bits.
- Generate interface tables separately: NumPy dtype info, PEP-3118/buffer
  format, array-interface `typestr`, safetensors names, MAX dtype names, and
  Array API support flags.
- Keep promotion rows keyed by canonical `DTypeSpec` codes.
- Do the same for primitive op codes.

This removes the "Python table says X, Mojo table says Y" class of bugs.

### Phase 4: graph batching

- Add a batching interpreter over `GraphIR`.
- Implement batching rules for elementwise, broadcast, reshape, transpose,
  reduction, `where`, stack/concat, matmul, and random key batching.
- Keep eager `vmap` as a correctness fallback and behavior pin.

### Phase 5: linalg primitive cleanup

- Add core signatures and batched shape rules.
- Add missing primitive families: triangular solve, LU factor/solve, Cholesky
  factor/solve.
- Move Python `linalg.py` wrappers toward argument normalization and named
  return construction only.

### Phase 6: lowering targets

- Decide whether the first compiler backend is MAX custom calls or a Mojo-native
  graph executor.
- Do not pick this before the primitive/eager contracts are stable. Backend
  work before primitive stability will cause churn with a 1:many blast radius.

## What not to do

- Do not expand `python/monpy/__init__.py` further. It should become a facade.
- Do not make `monpy.lax` a separate mini-library from eager monpy.
- Do not port JAX internals wholesale. Copy the rule-table architecture, not
  the implementation.
- Do not let NumPy compatibility decide transform architecture. NumPy layout and
  dtype behavior are interpreter semantics.
- Do not start with autodiff. Without primitive batching and layout planning,
  autodiff rules will be dead weight.
- Do not make GPU/Metal part of the first restructure. The architecture should
  leave a `device` field everywhere, but v1 execution is CPU.

## Near-term file moves

The smallest high-signal slice:

1. Create `python/monpy/_core/primitive.py` by moving and extending the current
   `Primitive` / `PrimitiveRegistry`.
2. Create `python/monpy/_core/abstract.py` from the existing `TensorSpec`,
   `DeviceSpec`, `DTypeSpec`, and `LayoutSpec` pieces.
3. Add `bind_primitive(...)` and make two or three public eager ops call it.
4. Make tracing use the same primitive objects.
5. Add `src/execute/iteration.mojo` with a first `IterationPlan` for binary
   elementwise same-shape / broadcast / strided cases.
6. Convert one binary op path to the planner, with the current kernels left in
   place.

That slice proves the architecture without burning the forest down.

## Sources

1. JAX AOT lowering and compilation:
   <https://docs.jax.dev/en/latest/aot.html>
2. JAX primitives and transformation rules:
   <https://docs.jax.dev/en/latest/jax-primitives.html>
3. JAX jaxpr internals:
   <https://docs.jax.dev/en/latest/jaxpr.html>
4. JAX `vmap` contract:
   <https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html>
5. JAX `where`:
   <https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html>
6. JAX `einsum`:
   <https://docs.jax.dev/en/latest/_autosummary/jax.numpy.einsum.html>
7. JAX pytrees:
   <https://docs.jax.dev/en/latest/pytrees.html>
8. JAX PRNG design:
   <https://docs.jax.dev/en/latest/jep/263-prng.html>
9. NumPy C internals and ufunc setup:
   <https://numpy.org/doc/stable/dev/internals.code-explanations.html>
10. NumPy iterator API:
    <https://numpy.org/doc/1.25/reference/c-api/iterator.html>
11. NumPy C structures:
    <https://numpy.org/doc/stable/reference/c-api/types-and-structures.html>
12. NEP 50 scalar promotion:
    <https://numpy.org/neps/nep-0050-scalar-promotion.html>
13. Array API standard background:
    <https://data-apis.org/blog/array_api_v2023_release/>
14. SciPy Array API support:
    <https://scipy.github.io/devdocs/dev/api-dev/array_api.html>
15. XLA architecture:
    <https://github.com/openxla/xla/blob/main/docs/architecture.md>
16. XLA GPU architecture:
    <https://openxla.org/xla/gpu_architecture?hl=en>
17. PyTorch internals:
    <https://blog.ezyang.com/2019/05/pytorch-internals/>
18. Mojo `SIMD` and `DType`:
    <https://docs.modular.com/mojo/manual/types/>
19. Mojo `vectorize`:
    <https://mojolang.org/docs/std/algorithm/backend/vectorize/>
20. Mojo `algorithm` package:
    <https://mojolang.org/docs/std/algorithm/>
21. Mojo `LayoutTensor`:
    <https://mojolang.org/docs/manual/layout/tensors/>
22. Nabla source:
    <https://github.com/nabla-ml/nabla>
