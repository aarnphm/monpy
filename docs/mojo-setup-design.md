---
title: Mojo setup and runtime contract
date: 2026-05-10
---

# Mojo setup and runtime contract

_The Mojo runtime is the execution engine for monpy primitives, not a second
public array API._

This document pins the local setup and runtime ownership rules for the
JAX-first architecture. The Python side owns public API spelling, primitive
binding, transforms, pytrees, the lazy interpreter, and GraphIR. The Mojo side
owns storage, layout-aware execution planning, SIMD kernels, vendor library
calls, and the CPython bridge.

The intended path is:

```text
Python facade
  -> static argument normalization
  -> primitive bind
  -> lazy primitive interpreter
  -> GraphIR
  -> Mojo runtime / MAX lowering at materialization boundaries
```

StableHLO is not part of this contract. Monpy needs a small typed GraphIR over
its own primitives before it needs compiler interchange.

`monpy.numpy` is intentionally out of the initial architecture. A NumPy
namespace can be added later as a compatibility adapter over the same primitive
spine. The first contract should keep one public `monpy` surface and one lazy
interpreter so operation semantics do not fork before the runtime is stable.

## toolchain modes

There are two supported ways to build the native extension.

### local development

Local development uses an explicit Mojo binary:

```bash
export MOHAUS_MOJO="$MODULAR_DERIVED_PATH/build/bin/mojo"
uv venv --python 3.11 --managed-python --clear
uv pip install mohaus --index https://aarnphm.github.io/mohaus/simple
uv pip install --no-build-isolation -e ".[dev]"
```

The current local build expects Mojo `1.0.0.dev0`. Keep the native source
limited to the Mojo standard library in this mode. Do not add permanent
out-of-tree include paths.

### package build

Package builds are driven by `pyproject.toml` and `mohaus.backend`. The build
metadata may pin published Mojo beta wheels while local development points at a
custom Mojo checkout. Treat those as two explicit modes, not one hidden
environment accident.

Optional dependencies stay optional:

- `kernels` may install MAX-facing Python packages.
- `vendor/NuMojo` is only an optional benchmark baseline.
- BLAS/LAPACK libraries are platform dependencies for vendor-backed linalg, not
  Python API dependencies.

## smoke commands

Use the configured Mojo binary for native verification:

```bash
MOHAUS_MOJO="$MODULAR_DERIVED_PATH/build/bin/mojo" \
  uv run --no-sync python -c "import monpy; import monpy._native"

MOHAUS_MOJO="$MODULAR_DERIVED_PATH/build/bin/mojo" \
  uv run --no-sync pytest tests/python -q
```

Pure Mojo benchmark smoke should use the same source root:

```bash
"$MOHAUS_MOJO" run -I src benches/bench_mojo_sweep.mojo
"$MOHAUS_MOJO" run -I src benches/bench_reduce.mojo
```

NuMojo comparisons may add the vendor include path in benchmark-only commands.
They must not become runtime imports.

## source ownership

`src/lib.mojo` is the CPython boundary. It marshals Python objects, calls the
runtime, and returns Python-visible values. It should not own algorithms.

`src/domain.mojo` owns compact runtime domain values: dtype tags, operation
tags, promotion hooks, and closed enums. Python's conceptual dtype source of
truth remains `DTypeSpec`; Mojo tags are dispatch encodings, not a second
public dtype model.

`src/storage.mojo`, `src/metadata.mojo`, `src/buffer.mojo`, and `src/array/`
own the ndarray substrate: storage, ownership, shape, strides, offsets, flags,
Python owner pinning, accessors, casting, factory helpers, and result dtype
selection.

`src/cute/` owns target-neutral layout algebra. It should keep array traversal
expressible without hard-coding CPU-only pointer arithmetic into every kernel.

`src/elementwise/` owns operation dispatch. `src/elementwise/kernels/` owns
typed compute loops. Linalg kernels may live there while the tree is small, but
the long-term ownership boundary is a dedicated linalg layer over primitives,
not an elementwise subfeature.

`src/accelerate.mojo` is a vendor seam. Apple Accelerate, vForce, OpenBLAS,
LAPACK, and future MAX custom ops are backend choices selected below the
primitive layer.

## runtime planning

The runtime should converge on a small planner:

```text
primitive tag + operands + requested output
  -> ExecutionPlan
  -> IterationPlan
  -> KernelCandidate
  -> typed kernel entry
```

An `ExecutionPlan` should decide dtype, broadcast shape, output allocation,
layout class, aliasing, backend, and whether materialization is required.

An `IterationPlan` should coalesce axes, normalize zero-stride broadcasts,
track offsets and alignment, expose row-contiguous inner loops, and preserve
negative-stride correctness.

Kernel selection should follow this order:

1. contiguous typed SIMD;
2. row-contiguous or tiled strided SIMD;
3. vendor BLAS/LAPACK/Accelerate where the primitive shape matches;
4. parallel reduction or tiled reduction;
5. generic layout fallback.

Each public operation should bind a primitive before it can execute. Lazy
execution is the first interpreter of primitives. NumPy compatibility can later
be an adapter around that interpreter, not the owner of native algorithms.

## performance policy

The first implemented backend is CPU, but kernel signatures should not make CPU
the final architecture. Layout algebra and primitive lowerings must leave room
for MAX, GPU, or Metal later.

SIMD policy:

- choose logical SIMD width by dtype, with 32 bytes as the default target;
- specialize typed kernels at compile time where Mojo allows it;
- keep unroll policy centralized;
- avoid gather-heavy Apple NEON paths;
- prefer row-flattening, tiling, or materialization when strided inputs are
  large enough to repay the copy.

Reduction policy:

- reductions are not elementwise kernels with a different loop counter;
- use block or pairwise accumulation where NumPy-like numerical behavior
  matters;
- keep parallel scheduling behind one policy module.

Mojo stdlib policy:

- use `std.algorithm` where it expresses the intended kernel shape directly,
  especially `vectorize`, `parallelize`, row parallel helpers, and reduction
  scaffolding;
- keep handwritten kernels when the project is learning something specific:
  stride coalescing, negative-stride correctness, layout tiling, aliasing,
  materialization thresholds, or vendor dispatch;
- benchmark stdlib and handwritten versions side by side before deleting either
  path;
- do not wrap `std.algorithm` so heavily that the generated loop shape becomes
  harder to inspect than a local implementation.

Interop policy:

- Python buffer protocol ingress stays in `src/buffer.mojo`;
- NumPy C API is a later optimization only if fixed overhead remains visible;
- external CPU storage is non-owning and must pin the Python producer.

## migration sequence

1. Pin this setup/runtime contract and keep README paths current.
2. Add one primitive `bind` path so eager and tracing share operation identity.
3. Make the lazy primitive interpreter the primary Python execution model.
4. Add the Mojo planner for one narrow family, binary elementwise with
   broadcasting.
5. Move contractions toward a `dot_general` primitive, then lower `einsum` and
   `tensordot` through it.

## upstream anchors

- JAX primitives explain why transformable functions need primitive rules:
  <https://docs.jax.dev/en/latest/jax-primitives.html>
- JAX jaxpr documents the typed primitive IR shape:
  <https://docs.jax.dev/en/latest/jaxpr.html>
- NumPy internals document ndarray metadata, strides, and views:
  <https://numpy.org/doc/stable/dev/internals.html>
- Mojo SIMD is the native vector substrate:
  <https://docs.modular.com/mojo/std/builtin/simd/SIMD/>
- Mojo CPU parallelization is the scheduling substrate for parallel kernels:
  <https://docs.modular.com/mojo/std/algorithm/backend/cpu/parallelize/>
