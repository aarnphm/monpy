---
title: "formal/empirical verification strategies for monpy"
date: 2026-05-10
---

# formal/empirical verification strategies for monpy

The pipeline being verified is `public API -> primitive bind -> eager interp ∥ tracer -> GraphIR -> MAX/Mojo lowering -> SIMD/BLAS kernels`. Each arrow is a refinement step; bugs hide in the diff between the source and target semantics of each arrow. Stack the cheap checks at every arrow and the expensive ones at the two arrows that matter most (eager≡tracer, kernel≡scalar oracle). See [[jax-first-architecture]] for the spine this lives on.

## what's load-bearing vs theatre

| tier | technique                                                                           | load-bearing?                                                                   |
| ---- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 0    | scalar reference vs SIMD kernel, on the same inputs, differential                   | yes — catches alignment, tail-loop, dtype-promotion bugs that nothing else does |
| 0    | NumPy oracle + per-dtype ULP tolerance for every primitive                          | yes — this is the only thing that catches branch-cut and transcendental drift   |
| 1    | abstract_eval ≡ shape(eager_impl) round-trip for every primitive                    | yes — single cheapest catch for the "spec drifted, kernel didn't" class         |
| 1    | full NxN promotion table walk (golden table)                                        | yes — JAX has shipped one for 6 years and it pays back every release            |
| 1    | hypothesis-driven shape/stride fuzz against NumPy on a small core                   | yes — finds view-aliasing and broadcast bugs that fixed cases miss              |
| 2    | StableHLO emission for the GraphIR + `stablehlo-translate --interpret` ground truth | high leverage, one-time cost                                                    |
| 2    | MLIR `op.verify()` at lowering time (MAX already does this)                         | free, on by default once you hit `max.graph`                                    |
| 2    | process_replay-style kernel-diff between commits                                    | cheap, high catch-rate on canonicalizer changes                                 |
| 3    | Alive2-style SMT equivalence for canonicalizers / view-fusion rewrites              | optional, high effort, only if rewrite layer grows teeth                        |
| 3    | full formal proof of any kernel (Lean / Coq / FPTaylor on transcendentals)          | theatre at monpy's scale; FPTaylor on one log1p kernel maybe, GEMM no           |

The categorisation above is the takeaway. Everything below is the work to get tiers 0–2 ship-shape.

## boundary-by-boundary table

| #   | boundary                           | what goes wrong                                                                                                                                                                          | what real labs do                                                                                                                                                                                                                                                                                                                                                                                                                                          | monpy's cheapest hook                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| --- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Python API ↔ primitive bind        | overload spelling drift; `ndarray.__add__`, `mp.add`, `lax.add_p` resolve to different ops; `out=` / `where=` / `casting=` keyword fanout silently divergent across spellings            | PyTorch `OpInfo` — one dataclass per op declares `ref`, `dtypes`, `sample_inputs_func`, `error_inputs_func`, `supports_out`, `supports_autograd`, `aliases`, `decorators=(toleranceOverride(...))`; >2.5k entries, drives ~50 generic test classes ([common_methods_invocations.py:12525](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_methods_invocations.py#L12525))                                                      | declarative `PrimitiveTestSpec(name, ref, dtypes, sample_inputs, tolerance)` next to each `Primitive` definition in `python/monpy/_src/core.py`; one generic test reads the registry and asserts identity across `mp.<name>`, `ndarray.__op__`, `mp.lax.<name>_p`                                                                                                                                                                                                             |
| 2   | eager ≡ tracer                     | jit-traced graph computes different result from eager; subtle when reductions, `where`, or comparisons enter (you already hit this in your May-10 checkpoint with comparison-bool dtype) | JAX `_CheckAgainstNumpy` + `_CompileAndCheck` run the same op eagerly and under `jit`, with per-op `tol` records keyed by `(op, dtype, device)` ([lax_test.py:118-138](https://github.com/jax-ml/jax/blob/main/tests/lax_test.py#L118)); tinygrad `process_replay` SQLite-captures HEAD's compiled kernels and diffs against master ([process_replay.py](https://github.com/tinygrad/tinygrad/blob/master/test/external/process_replay/process_replay.py)) | a `assert_eager_traced_agree(primitive, args)` helper invoked once per primitive in CI; uses the existing `Primitive.eager_impl` and `jit(lambda: primitive.bind(...))` exits — already plumbed                                                                                                                                                                                                                                                                               |
| 3   | GraphIR / IR level                 | unrepresentable graph passes structural hash; `abstract_eval` claims a shape the eager path doesn't produce; layout invariants drift                                                     | StableHLO ships a spec with `(C1)`-style constraints and a `stablehlo-translate --interpret` reference interpreter; tests use `check.expect_almost_eq` with a uniform $10^{-4}$ tolerance for ~1-ULP claims[^stablehlo-tol]; MLIR `op.verify()` runs traits + custom verifiers on every op insertion, `mlir-opt --verify-each` runs after every pass                                                                                                       | nabla's `verify_eager_shapes()` is the pattern: compare `compute_physical_shape()` prediction vs actual MAX-execution shape, gated by `VERIFY_EAGER_SHAPES=1` env ([nabla/config.py](https://github.com/nabla-ml/nabla/blob/main/nabla/config.py), [nabla/ops/utils.py:578](https://github.com/nabla-ml/nabla/blob/main/nabla/ops/utils.py#L578)). monpy clone: a `_verify_shapes` context var that asserts `Node.spec.shape == eager_impl(...).shape` after every eager bind |
| 4   | lowering → MAX/Mojo                | MLIR rejects the lowered graph; lowering produces semantically different op                                                                                                              | MAX's `Graph._verify_op` calls `op.verify()` after every `_add_op_generated`, with a `_capturing_mlir_diagnostics` context that surfaces the MLIR diagnostic stream as a Python `ValueError` ([max/python/max/graph/graph.py:926](https://github.com/modular/modular/blob/main/max/python/max/graph/graph.py#L926))                                                                                                                                        | nothing to add — you get this for free the moment a lowering uses `max.graph.Graph`. Add one CI test that constructs a deliberately invalid lowering (mismatched dtype on `add`) and asserts MAX raises                                                                                                                                                                                                                                                                       |
| 5   | Mojo kernel ↔ scalar oracle        | SIMD tail loop misses 1-3 elements; broadcast stride zero wrongly vectorised; alignment fault on unaligned start                                                                         | Triton tests every kernel against `torch.matmul` eager; Halide uses `can_prove()` (empirical, 100-iter random probe, not formal — Simplify.cpp:628); kernel autotuners verify each candidate against a reference before promoting                                                                                                                                                                                                                          | one Mojo test per typed-vec dispatcher: scalar-loop expected value vs SIMD result on shape sweep `[1, 7, 8, 9, 31, 32, 33, 1024, 1027]` (powers of two ± 1, ± `lane_count` ± 1) — this is the highest-yield test you can write                                                                                                                                                                                                                                                |
| 6   | floating-point oracles & ULP       | "almost equal" tolerance too loose hides 2-ULP drift, too tight breaks on TPU/fp8; branch cuts misplaced on `clog`, `casin`, `csqrt`                                                     | JAX `_default_tolerance` is the canonical table: `f64=1e-15, f32=1e-6, bf16=1e-2, f16=1e-3, fp8_*=1e-1, fp4_e2m1fn=1e0` ([public_test_util.py:46](https://github.com/jax-ml/jax/blob/main/jax/_src/public_test_util.py#L46)); StableHLO uses uniform $10^{-4}$ as a coarse-grain "within ~1 ULP"; FPTaylor only used in production for verifying single FMA chains (cordic, exp), not whole kernels                                                        | adopt JAX's table verbatim as `monpy/_src/testing/tolerance.py`; per-op overrides via `PrimitiveTestSpec.tol_overrides` mirroring `toleranceOverride({torch.chalf: tol(atol=1e-2)})` — see [[complex-kernels]] for the branch-cut surface that needs the override                                                                                                                                                                                                             |
| 7   | shape / stride / layout invariants | view-of-view-of-broadcast wrongly aliases; `Layout` strides claim 0 where storage isn't broadcast; `offset_elems` off-by-one                                                             | Hypothesis `arrays(dtype, shape)` strategy + `array-api-tests` (driven by `ARRAY_API_TESTS_MODULE` env var); jaxtyping + beartype enforces `Float[Tensor, "b n d"]` at runtime; PyTorch's internal `_strided` invariant checker                                                                                                                                                                                                                            | one Hypothesis test that fuzzes `LayoutSpec.broadcast_to(...).reshape(...).permute(...)` and asserts `is_broadcast_from`, `permutation_from`, and `element_count` are mutually consistent; this lives at `python/monpy/_src/layout.py` so the cost is one file                                                                                                                                                                                                                |
| 8   | dtype promotion & casting          | NEP 50 weak-scalar dispatch silently disagrees between eager and traced; promotion table not a lattice and the associativity gap drifts                                                  | JAX `testObservedPromotionTable` walks an 18×18 typecode matrix (including weak-typed `i*`, `f*`, `c*`) against a hard-coded golden ([dtypes_test.py:976](https://github.com/jax-ml/jax/blob/main/tests/dtypes_test.py#L976)); the test is verbatim "this table does not change over time"                                                                                                                                                                 | walk monpy's 21-dtype × 21-dtype matrix once via `from_monpy_dtype + result_type` against a golden written next to the table. you already have a left-fold non-associative case documented in [[dtype-promotion-casting]] — capture the golden _now_ before it drifts                                                                                                                                                                                                         |
| 9   | nabla-style cross-check            | sharding propagation, structural hash collisions, op lifecycle inconsistency                                                                                                             | nabla's verification stack: `verify_eager_shapes()`, `compute_structural_hash()` (caches op identity), `verify_custom_op()` (MAX-side trait check), context-var `VERIFY_EAGER_SHAPES` gate. test posture is _exclusively_ `_close(nb_val, jax_val)` against JAX with `rtol=5e-4, atol=5e-4` ([nabla/tests/unit/test_transforms_composition.py:25](https://github.com/nabla-ml/nabla/blob/main/tests/unit/test_transforms_composition.py#L25))              | clone the gate: a `verify_shapes_context()` ContextVar in monpy that asserts `TensorSpec.shape == eager_result.shape` post-bind; failing CI test on every primitive when enabled                                                                                                                                                                                                                                                                                              |

The recurring pattern: real array-library labs maintain _one declarative op registry_ and let _generic tests_ fan out from it.[^opinfo-scale] The alternative — one bespoke test file per op — is what creates the silent drift everyone fears.

## monpy-specific proposals

The current `Primitive` ([python/monpy/\_src/core.py:51](../../python/monpy/_src/core.py)) already has the right shape: `abstract_eval`, `dtype_rule`, `eager_impl`, `batching_rule`, `target_lowerings`. Three slots and three CI changes get you tiers 0–2.

### proposal 1: add `reference_impl` and `tolerance` to `Primitive`

```python
# python/monpy/_src/core.py — extend the existing __slots__
__slots__ = (
    # ... existing ...
    "reference_impl",       # Callable[..., np.ndarray] — NumPy oracle
    "tolerance",            # Mapping[DTypeSpec, tuple[float, float]] — (rtol, atol)
    "sample_inputs",        # Callable[[Rng], Iterator[tuple[ndarray, ...]]]
)
```

At registration:

```python
add_p = define_primitive(
    "add",
    abstract_eval=_broadcast_av,
    dtype_rule=_promote_av,
    eager_impl=_native_add,
    reference_impl=lambda x, y: np.add(x, y),
    tolerance={F32: (1e-6, 0), F64: (1e-15, 0), BF16: (1e-2, 0)},
    sample_inputs=_binary_broadcast_samples,
)
```

Cost: one extra line per primitive. Buy: one generic test (`test_primitive_against_reference`) iterates `PRIMITIVES.names()` and runs `(eager, reference, jit-traced)` cross-checks with the right tolerance. This is the PyTorch OpInfo pattern at 1/30th the surface area.

### proposal 2: round-trip `abstract_eval ∘ eager_impl ≡ eager_impl ∘ spec`

```python
# python/monpy/_src/testing/roundtrip.py — new file, ~30 lines
def assert_abstract_eval_matches_eager(prim: Primitive, sample_args):
    eager_out = prim.eager_impl(*sample_args)
    eager_spec = TensorSpec(eager_out.shape, eager_out.dtype, ...)
    inferred = prim.abstract_eval(*[_as_spec(a) for a in sample_args])
    assert eager_spec.shape == inferred.shape, f"{prim.name}: shape drift"
    assert eager_spec.dtype == inferred.dtype, f"{prim.name}: dtype drift"
```

Cost: one new file, one test that loops over `PRIMITIVES.names()` × the per-primitive `sample_inputs` strategy. Buy: every shape/dtype rule is mechanically pinned to the kernel it claims to describe. This is the single most leveraged $20-of-effort hook in the whole stack; copy nabla's `verify_eager_shapes` posture and gate it under `MONPY_VERIFY_SHAPES=1` for CI.

### proposal 3: 21×21 promotion golden, generated once, asserted in CI

```python
# tests/python/test_dtype_promotion_golden.py — new file
def test_promotion_table_unchanged():
    golden = json.loads(Path("tests/data/promotion_golden.json").read_text())
    actual = {
        f"{a}+{b}": result_type(a, b).name
        for a in ALL_21_DTYPES for b in ALL_21_DTYPES
    }
    assert actual == golden  # 441 entries
```

Cost: one regen script, one ~30KB JSON. Buy: NEP 50 weak-scalar dispatch and the documented non-lattice associativity gap (in [[dtype-promotion-casting]]) are mechanically pinned. JAX's version of this test has caught the i4×u4 promotion change between `enable_x64=True` and `ExplicitX64Mode.ALLOW` modes — exactly the kind of silent regression you can't afford. Worth doing this week.

## prioritised adoption plan

**this week (1-2 days each, all in `python/monpy/_src/testing/`):**

1. Generate the 21×21 promotion golden JSON and add the CI assertion. Caches the table _before_ further refactoring happens.
2. Wire `MONPY_VERIFY_SHAPES=1` into the eager `bind_primitive` path. Run the round-trip assert on every primitive call in CI; runtime cost is negligible.
3. Ship the JAX `_default_tolerance` table verbatim as `_src/testing/tolerance.py`. No tests need it yet — just have it ready.

**this month:**

4. Add `reference_impl`, `tolerance`, `sample_inputs` slots to `Primitive`. Backfill the ~20 registered primitives. Single generic test (`test_primitive_against_numpy_reference`) reads the registry, fans out via Hypothesis over shapes, calls `eager`, `jit(eager)`, and `reference_impl`; asserts within tolerance.
5. One Hypothesis fuzz test for `LayoutSpec` algebraic invariants (`broadcast_to ∘ reshape ∘ permute` round-trip consistency). Lives in `tests/python/test_layout_invariants.py`. Surfaces the view-aliasing class of bugs that hand-written tests miss.
6. One Mojo-side scalar-vs-SIMD differential test per typed-vec dispatcher (binary, unary_preserve, complex). Shape sweep: $\{1, 7, 8, 9, 31, 32, 33, 1024, 1027\}$ — powers of two ± 1, lane-count ± 1, exact-multiple. This catches more 1-ULP drift than any property test will.

**this quarter:**

7. Wire `array-api-tests` as a CI job with `ARRAY_API_TESTS_MODULE=monpy.array_api`. It's an external test suite that gives you a few thousand Hypothesis-driven assertions for free; expect ~30% to fail on first run and use the failures as a roadmap.
8. Run monpy's `GraphIR` through StableHLO emission and ground-truth against `stablehlo-translate --interpret` for a small kernel suite (`add`, `matmul`, `where`, `reduce`). One-time effort, $\sim 200$ LOC of glue. Pins the lowering to a spec'd interpreter, not just MAX's MLIR verifier.
9. Steal nabla's `process_replay` posture: capture compiled MAX programs in a SQLite DB on `main`, diff on PR. Cost: ~150 LOC. Catches codegen drift that no semantic test will, because the semantics didn't change — the kernel choice did.

**not on the roadmap:**

10. Lean/Coq proofs of GEMM, full formal verification of any kernel, Alive2 against monpy's GraphIR. None of these earn their cost at monpy's scale; you'd be the only person in the array-library world doing them, which is a signal not in your favor.[^alive2-scope]

## sources

1. JAX `_default_tolerance` dict: <https://github.com/jax-ml/jax/blob/main/jax/_src/public_test_util.py#L46-L76> — the canonical per-dtype ULP-tolerance table; verbatim adoption recommended for monpy.
2. JAX `testObservedPromotionTable`: <https://github.com/jax-ml/jax/blob/main/tests/dtypes_test.py#L970-L1083> — 18×18 typecode matrix asserted against a hard-coded golden, three configs (enable_x64, ExplicitX64Mode.ALLOW, default).
3. JAX `testOpAgainstNumpy` + `lax_reference`: <https://github.com/jax-ml/jax/blob/main/tests/lax_test.py#L118-L138> — every lax op has a NumPy reference; tolerance is per-op + per-dtype + per-device.
4. JAX `check_grads`: <https://github.com/jax-ml/jax/blob/main/jax/_src/public_test_util.py#L326> — fwd/rev/JVP gradient consistency to a chosen order.
5. PyTorch `OpInfo("add", ...)`: <https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_methods_invocations.py#L12525-L12559> — BinaryUfuncInfo with `ref=lambda input, other, *, alpha=1: ...`, `dtypes=all_types_and_complex_and(...)`, `decorators=(DecorateInfo(toleranceOverride({torch.chalf: tol(atol=1e-2, rtol=0)}), 'TestBinaryUfuncs', 'test_reference_numerics'))`. 63 `BinaryUfuncInfo` registrations alone.
6. PyTorch OpInfo `dataclass`: <https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/opinfo/core.py> — 50+ fields covering eager / method / inplace / out / jit / autograd / sparse variants.
7. nabla's `verify_eager_shapes`: <https://github.com/nabla-ml/nabla/blob/main/nabla/ops/utils.py#L578-L625> + `VERIFY_EAGER_SHAPES` ContextVar in `nabla/config.py` — direct precedent for what monpy should adopt.
8. nabla's JAX cross-check posture: `_close(nb_val, jax_val, rtol=5e-4, atol=5e-4)` throughout `tests/unit/test_transforms_composition.py`.
9. MAX Graph `_verify_op`: <https://github.com/modular/modular/blob/main/max/python/max/graph/graph.py#L926-L951> — calls `op.verify()` after every op insertion with MLIR diagnostic capture; free for any monpy lowering that uses `max.graph`.
10. StableHLO spec constraint notation `(C1)`, `(C2)`: <https://github.com/openxla/stablehlo/blob/main/docs/spec.md> — every op has formal constraints that double as verifier rules.
11. StableHLO reference interpreter: <https://github.com/openxla/stablehlo/blob/main/docs/reference.md> — `stablehlo-translate --interpret` + `check.expect_almost_eq` with uniform $10^{-4}$ tolerance.
12. MLIR `op.verify()` mechanism: <https://mlir.llvm.org/docs/OpDefinitions/> — verifier traits + custom `verify()` / `verifyRegions()` per op, `mlir-opt --verify-each` after every pass.
13. Alive2 LLVM equivalence: <https://github.com/AliveToolkit/alive2> — SMT-based refinement check for peephole + pass transformations; useful only if monpy grows a serious GraphIR rewrite layer.
14. Halide `can_prove()`: <https://github.com/halide/Halide/blob/main/src/Simplify.cpp> — empirical 100-iteration random probing, not formal proof; this is the practitioner's compromise.
15. tinygrad `process_replay`: <https://github.com/tinygrad/tinygrad/blob/master/test/external/process_replay/process_replay.py> — SQLite-captured kernel diffs between master and HEAD; catches codegen drift in CI.
16. `array-api-tests`: <https://github.com/data-apis/array-api-tests> — Hypothesis-driven Array API conformance suite, plug in via `ARRAY_API_TESTS_MODULE` env var.
17. jaxtyping runtime shape annotations: <https://github.com/patrick-kidger/jaxtyping> — `Float[Tensor, "b n d"]`-style annotations enforced at runtime via beartype or typeguard.
18. Hypothesis `extra.numpy.arrays`: <https://github.com/HypothesisWorks/hypothesis/blob/master/hypothesis-python/src/hypothesis/extra/numpy.py> — `arrays(dtype, shape, elements=...)` strategy; the basis of all property-based array testing.
19. monpy `Primitive` extension surface: `python/monpy/_src/core.py:51-101`.
20. monpy `LayoutSpec` algebra: `python/monpy/_src/layout.py:72-211`.

[^stablehlo-tol]: StableHLO's $10^{-4}$ tolerance is interesting in how _coarse_ it is — they only claim 1-ULP for basic arithmetic, and a single threshold for everything. JAX's per-dtype table is much sharper: $10^{-15}$ for f64, $10^{-6}$ for f32, $10^{-2}$ for bf16, $10^0$ for fp4*e2m1fn (i.e., basically zero useful tolerance — fp4 has 2 mantissa bits, so "within 1" \_is* the precision). The fp4 entry is a great forcing function: it makes you state explicitly that you've thought about each dtype, rather than reusing a default.

[^opinfo-scale]: For scale: PyTorch's `common_methods_invocations.py` is 27,301 lines, mostly OpInfo entries. JAX's `lax_test.py` is 5,566 lines; the test classes are smaller because the op record is more abstract. nabla's `tests/unit/` is 35 files, ~250KB, every test cross-checks against JAX. monpy can target the nabla density: ~5-10 lines of registry metadata per primitive, generic tests in the hundreds of lines. The leverage ratio is ~30:1 (lines-of-test-coverage : lines-of-registry-metadata) when done right.

[^alive2-scope]: Alive2 works on LLVM IR because LLVM IR has well-defined semantics that don't admit too much UB at the relevant abstraction level — but even Alive2 had to invent "refinement" (target may be _more_ defined than source) because LLVM `undef`/`poison` makes equivalence undecidable. GraphIR's semantics are nowhere near as nailed-down; the cost of formalising them in SMT would dominate any debugging benefit. The actual high-yield rewrite checker for monpy is empirical: emit before/after-rewrite graphs, run both on a Hypothesis-fuzzed input batch, assert bitwise (integer) or within-ULP (float) equality. The same machinery already in place for proposal 1 covers this case; you don't need SMT until you have rewrites with non-obvious soundness conditions, which monpy doesn't.
