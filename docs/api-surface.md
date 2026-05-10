# monpy api surface contract

monpy has three public pressure fronts:

1. NumPy compatibility for eager ndarray programs.
2. SciPy compatibility for dense numerical library code on top of arrays.
3. JAX-style transforms for whole Python functions.

The architecture must not make these three fronts compete. The shared contract is:

```text
Python API call
  -> parse static arguments, shapes, axes, and dtype options
  -> bind a monpy primitive or decompose into monpy primitives
  -> eager interpreter executes primitives on ndarray storage
  -> jit interpreter records GraphIR primitive nodes
  -> Mojo/MAX consumes GraphIR or the primitive execution plan
```

`monpy.jit` is the transform front door. `monpy.lax` is the primitive/spec
namespace. The public NumPy-shaped API should work inside `@monpy.jit` whenever
the operation has static output shape and can be expressed through primitives.

```python
import monpy as mp

@mp.jit
def block(x, w, bias):
  y = mp.einsum("ij,jk->ik", x, w)
  return mp.where(y > 0, y + bias, 0)
```

The example above should trace the function body. It should not treat
`einsum`, `where`, `add`, or comparison as special independent compilers. Those
calls are normal API functions that lower into the same primitive spine.

## traceability classes

Every public array operation should be classified into exactly one class.

| class              | meaning                                      | examples                                              | required machinery                              |
| ------------------ | -------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------- |
| primitive          | one public call binds one primitive          | `add`, `matmul`, `where`, `reshape`, `transpose`      | eager rule, abstract eval, lowering rule        |
| composite          | public call lowers into multiple primitives  | `einsum`, `tensordot`, `norm`, `softmax`              | static parser/plan plus primitive decomposition |
| static helper      | Python-only metadata helper                  | dtype normalization, axis parsing, shape products     | must not inspect traced array values            |
| eager-only         | NumPy-compatible but not currently traceable | data-dependent shape ops, mutation-heavy APIs         | loud staged error plus eager tests              |
| deferred subsystem | outside current project slice                | sparse, full FFT, masked arrays, object/string dtypes | separate design doc before implementation       |

The important line is "must not inspect traced array values." Shape, dtype,
rank, device, layout, and static kwargs are compile-time data. Array contents
are runtime data.

## module layout target

| namespace         | role                                                      | current state                                            | target                                                      |
| ----------------- | --------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------- |
| `monpy`           | NumPy-shaped top-level facade plus transform entry points | broad eager surface, top-level `jit`/`vmap` shim         | main user import for NumPy-compatible code and `@monpy.jit` |
| `monumpy`         | compatibility shim                                        | re-exports `monpy`                                       | keep as import alias only                                   |
| `monpy.numpy`     | explicit NumPy interchange boundary                       | `monpy.numpy.ops` owns NumPy import-dependent conversion | never needed for core execution                             |
| `monpy.linalg`    | NumPy linalg and tensor contraction surface               | dense rank-2 kernels plus Python wrappers                | facade over linalg primitives and contraction plans         |
| `monpy.random`    | explicit-key random plus NumPy-style helpers              | native key/sampler slice exists                          | random primitives with legacy wrappers                      |
| `monpy.lax`       | primitive/spec/transform namespace                        | `jit`, `vmap`, `TensorSpec`, `GraphIR`, primitives       | small, stable, JAX-shaped core namespace                    |
| `monpy.extend`    | backend registration and lowering hooks                   | MAX/Mojo lowering skeleton exists                        | extension registry, custom calls, target lowerings          |
| `monpy.scipy`     | SciPy-compatible modules                                  | not implemented                                          | later facade packages over existing primitives              |
| `monpy.array_api` | Python Array API namespace                                | partial re-export                                        | SciPy array-api gateway compatibility                       |

## numpy api ledger

The NumPy front is the eager interpreter contract. The detailed gap ledger lives
in `docs/numpy-port-gaps.md`; this table is the architectural status.

| family                                          | status                                                                                         | trace target                                                                     |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| dtype objects, promotion, casting               | broad eager support, no full scalar hierarchy                                                  | dtype rules on primitives, no array-value reads                                  |
| array creation                                  | broad eager support                                                                            | mostly eager-only until constants and allocation nodes are designed              |
| buffer, DLPack, NumPy interop                   | explicit CPU boundary                                                                          | not a traced operation                                                           |
| ndarray storage, shape, strides, views          | broad eager support                                                                            | layout metadata on `TensorSpec` and view primitives                              |
| elementwise ufuncs                              | many eager ufuncs, primitive identity started for `add`/`sub`/`mul`/`div`                      | every ufunc gets a primitive or a composite lowering                             |
| comparisons and predicates                      | broad eager support                                                                            | comparison/predicate primitives                                                  |
| `where`                                         | eager and staged primitive                                                                     | direct `where_p`                                                                 |
| reductions and accumulations                    | eager support with known keyword tail gaps; staged `sum`/ufunc reductions now emit `reduce_p`  | `reduce_p` with axes, keepdims, dtype, and reducer attrs                         |
| shape manipulation                              | eager support for views/copies                                                                 | `reshape_p`, `transpose_p`, `broadcast_to_p`, copy/materialize primitive         |
| indexing and gather/scatter                     | broad eager support, mutation tails remain                                                     | gather/scatter primitives, update primitives later                               |
| sorting/search/setops                           | eager slices exist                                                                             | mostly eager-only until sort/gather primitives exist                             |
| linalg rank-2 dense                             | eager BLAS/LAPACK support                                                                      | linalg primitives with gufunc core signatures                                    |
| tensor contractions                             | eager `einsum`/`tensordot` slices; staged binary contractions cover matmul and dot-style cases | composite lowering through transpose/reshape/matmul/reduce, then `dot_general_p` |
| random                                          | explicit-key native slice plus helpers                                                         | key primitives and distribution composites                                       |
| FFT                                             | deferred                                                                                       | separate FFT backend and primitive family                                        |
| strings, object, structured, datetime           | deferred                                                                                       | out of current primitive spine                                                   |
| masked arrays, records, polynomial, testing, IO | deferred                                                                                       | separate namespaces, not ndarray core                                            |

## scipy api ledger

SciPy should be a library facade over monpy primitives. It should not call back
into NumPy/SciPy for implemented performance paths.

| scipy family        | priority | target architecture                                                           |
| ------------------- | -------- | ----------------------------------------------------------------------------- |
| `scipy.linalg`      | p0       | dense linalg primitives, gufunc signatures, BLAS/LAPACK or typed Mojo kernels |
| `scipy.special`     | p1       | ufunc-shaped primitives and composites, dtype/error policy first              |
| `scipy.stats`       | p1       | reuse sort, unique, reductions, and special functions before distribution zoo |
| `scipy.fft`         | p2       | FFT primitive family plus backend policy                                      |
| `scipy.signal`      | p2       | windows first, convolution/filtering after FFT                                |
| `scipy.sparse`      | p3       | separate COO/CSR/CSC storage project                                          |
| `scipy.optimize`    | p3       | algorithm package over dense primitives and autodiff later                    |
| `scipy.integrate`   | p3       | algorithm package, not ndarray core                                           |
| `scipy.interpolate` | p3       | algorithm package plus indexing/gather kernels                                |
| `scipy.spatial`     | p3       | distance/tree kernels, separate from dense ndarray core                       |
| `scipy.ndimage`     | p3       | stencil/filter subsystem                                                      |
| `scipy.io`          | p4       | file formats and interchange, not primitive spine                             |

First SciPy slice:

1. `solve_triangular`
2. `lu_factor`, `lu_solve`, `lu`
3. `cho_factor`, `cho_solve`
4. gufunc batch signatures for dense linalg
5. `block_diag`, `toeplitz`, `circulant`, `hankel`

Those pay down real performance debt. A facade-level Python loop is not a
valid implementation for the performance path.

## jax api ledger

The JAX front is the transform contract. It defines how function staging works,
not a promise to clone all of JAX.

| family                                 | status                                                                 | target                                                                       |
| -------------------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `jit`                                  | graph compile boundary exists under `monpy.lax`, top-level shim exists | trace whole Python functions into `GraphIR`                                  |
| `vmap`                                 | eager correctness wrapper exists                                       | graph batching transform with primitive batching rules                       |
| `lax` primitives                       | small primitive registry exists                                        | complete primitive rule table                                                |
| `jax.numpy` equivalent                 | represented by public `monpy` API                                      | every static-shape function is traceable or loudly eager-only                |
| `random`                               | explicit-key slice exists                                              | key/distribution primitives and batching rules                               |
| pytrees                                | not required for ndarray core                                          | small tree flatten utility for transform args/results, custom registry later |
| autodiff (`grad`, `jvp`, `vjp`)        | deferred                                                               | add after primitive abstract eval, batching, and lowering stabilize          |
| control flow (`cond`, `while`, `scan`) | deferred                                                               | higher-order primitives with subgraphs                                       |
| sharding/pjit/pmap                     | out of v1                                                              | device mesh after CPU primitive spine is stable                              |
| custom primitives                      | registry exists in embryo                                              | public extension API with abstract eval, lowering, batching hooks            |

### jax interface inventory

The useful JAX surface for monpy is a smaller compatibility target than the
full JAX package. Treat it as five layers:

| layer                      | JAX spelling                                                          | monpy spelling                                                                   | v1 contract                                                                                          |
| -------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| whole-function staging     | `jax.jit`, `jax.make_jaxpr`, `jax.eval_shape`                         | `monpy.jit`, `monpy.lax.GraphIR`, future `monpy.make_graph` / `monpy.eval_shape` | trace Python functions into typed primitive nodes without reading array values                       |
| automatic batching         | `jax.vmap`                                                            | `monpy.vmap`                                                                     | eager fallback now, primitive batching rules later                                                   |
| array facade               | `jax.numpy`                                                           | top-level `monpy` plus `monumpy` alias                                           | NumPy-compatible eager API where every static-shape function is primitive-backed or composite-staged |
| primitive namespace        | `jax.lax`                                                             | `monpy.lax`                                                                      | stable primitive handles, abstract specs, GraphIR, and low-level transform entry points              |
| random                     | `jax.random.key`, `split`, distributions                              | `monpy.random`                                                                   | explicit keys, no hidden global RNG in staged code                                                   |
| pytrees                    | `jax.tree`, `jax.tree_util`                                           | future `monpy.tree` or private `_src.tree_util` first                            | flatten/unflatten/map for transform args/results; custom registry after the spine is stable          |
| autodiff                   | `grad`, `value_and_grad`, `jvp`, `vjp`, `jacfwd`, `jacrev`, `hessian` | deferred                                                                         | only after primitives have abstract eval, batching, and lowering rules                               |
| custom rules               | `custom_jvp`, `custom_vjp`, `jax.extend`                              | future `monpy.extend`                                                            | extension hooks around primitive registration and lowering, not internal object exposure             |
| control flow               | `lax.cond`, `lax.while_loop`, `lax.scan`                              | deferred `monpy.lax` higher-order primitives                                     | subgraph-carrying primitives with static shape checks                                                |
| devices/sharding           | `device_put`, `devices`, `sharding`, `pjit`, `pmap`, `shard_map`      | out of v1                                                                        | keep `device` in specs, defer mesh APIs until CPU primitive spine is stable                          |
| debugging/callbacks/export | `debug`, callbacks, export/stages                                     | out of v1                                                                        | later, once GraphIR lowering and cache identity are real                                             |

So, yes, monpy needs pytrees. It does not need JAX's entire pytree surface
early. The needed piece is structural: transforms must preserve Python
argument and return structure while `GraphIR` only sees tensor leaves,
constants, and static attrs. Without a small pytree module, `jit`, `vmap`, and
future autodiff each grow their own tuple/list/dict handling, which is exactly
how the special-case sludge returns.

## operation record template

Each public function should eventually have a row in a generated or maintained
API ledger:

| field             | meaning                                                                             |
| ----------------- | ----------------------------------------------------------------------------------- |
| public name       | `monpy.einsum`, `monpy.linalg.solve`, `monpy.scipy.linalg.solve_triangular`         |
| upstream match    | NumPy, SciPy, JAX, Array API, or monpy-native                                       |
| class             | primitive, composite, static helper, eager-only, deferred subsystem                 |
| primitive spine   | exact primitive(s) or planned primitive(s)                                          |
| eager behavior    | ndarray implementation and copy/view/error semantics                                |
| staged behavior   | GraphIR nodes and static argument requirements                                      |
| Mojo/MAX lowering | backend kernel or explicit blocker                                                  |
| tests             | eager parity, staged GraphIR, lower/execute, benchmark row if performance-sensitive |

No operation should be called "done" because the name exists. It is done when
the eager and staged contracts are both pinned, or when the staged gap is
documented as intentional.

## current repo facts

As of 2026-05-10:

- `monpy.__all__` exports 275 names.
- `monpy.linalg` exposes 36 public names.
- `docs/numpy-port-gaps.md` owns the NumPy surface gap ledger.
- `docs/scipy-jax-port-gaps.md` owns the SciPy/JAX pressure map.
- `docs/research/jax-first-architecture.md` owns the primitive spine migration
  plan.
- The traceable public-operation slice is intentionally small: arithmetic
  ufunc identity for `add`/`subtract`/`multiply`/`divide`, comparison ufuncs
  for `equal`/`not_equal`/`less`/`less_equal`/`greater`/`greater_equal`,
  `matmul`, `reshape`, `transpose`, `broadcast_to`, `cast`, `custom_call`,
  `where`, and `reduce_p` for public reductions.
- Binary staged `einsum` now handles the common no-diagonal, no-ellipsis cases:
  matrix contraction lowers to `matmul`; dot-style full contractions lower to
  `mul` plus `reduce`; non-batched pair contractions can use
  transpose/reshape/matmul/reshape.
- `mp.where(y > 0, ...)` now traces through comparison dunders on staged
  `Tensor`, bool-typed comparison primitives, and a traced truthiness guard.

The next patch should add the missing batched contraction case (`bij,bjk->bik`)
without pretending it is one flat GEMM. On Apple Accelerate that means either a
native loop over batch slices or a dedicated Mojo batched kernel, not the
current eager `tensordot` flattening shortcut.

## upstream anchors

- JAX primitives and traceability:
  <https://docs.jax.dev/en/latest/jax-primitives.html>
- JAX tracing:
  <https://docs.jax.dev/en/latest/tracing.html>
- `jax.numpy` API:
  <https://docs.jax.dev/en/latest/jax.numpy.html>
- NumPy reference:
  <https://numpy.org/doc/stable/reference/index.html>
- SciPy public API:
  <https://docs.scipy.org/doc/scipy/reference/>
- SciPy Array API support:
  <https://scipy.github.io/devdocs/tutorial/array_api.html>
