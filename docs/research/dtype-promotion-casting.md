---
title: "dtype promotion and casting in monpy"
date: 2026-05-07
---

# dtype promotion and casting in monpy

_the promotion table is a lattice you wish were a lattice; the honest description is a partial order with engineered closure under a join operation that fails associativity for one well-known triple — and pretending otherwise is what got pre-NEP-50 NumPy into trouble._

---

this note formalises monpy's dtype system over its 14 numeric types — `bool`, `int{8,16,32,64}`, `uint{8,16,32,64}`, `float{16,32,64}`, `complex{64,128}` — under NEP 50 semantics. we work at the level of the algebraic structure $(D, \sqcup, \leq_C)$, state where the structure is genuinely lattice-shaped, identify the one place it isn't, and connect the abstract result back to the actual dispatch path through `src/domain.mojo` and `python/monpy/__init__.py`.

the reader is assumed comfortable with posets, semilattices, and the distinction between a partial order embedded in an order-extending join versus a lattice closed under that join. we do not labour the basics. Davey & Priestley chapters 1–2 are the standard reference if anything below feels unfamiliar.[^davey]

[^davey]: Davey & Priestley, _Introduction to Lattices and Order_, 2nd ed., Cambridge University Press 2002. the relevant structural theorems for our purposes are: (1.34) every join-semilattice with a top element is automatically a complete lattice when finite, (2.10) order-isomorphisms preserve joins, and (2.27) sublattice closure.

## 1. NEP 50: motivation and history

pre-NEP-50 NumPy had **value-based casting**. the promotion of a Python `int` against a NumPy array depended on the runtime _value_ of the int, not just its type. the canonical pathology:

```python
# old (legacy) behaviour
np.array([1], dtype=np.int8) + 200      # returns int16 array
np.array([1], dtype=np.int8) + 100      # returns int8  array
np.array([1], dtype=np.int8) + 127      # returns int8  array
np.array([1], dtype=np.int8) + 128      # returns int16 array  (!!)
```

the output dtype branched on whether the right-hand scalar fit in `int8`. this had three knock-on consequences. first, the type of an expression depended on data, which broke type-stability for any user trying to reason about kernel dispatch. second, JIT compilers and ahead-of-time tracers (Numba, JAX, PyTorch's tracer) had to either replicate the value-dependent logic or diverge from NumPy. third — the one that finally killed it — perfectly innocent code silently widened: a histogram pipeline written for `int8` arrays would produce `int16` arrays whenever input bins exceeded 127, blowing memory budgets without warning.

NEP 50, accepted in 2023 and shipped as default in NumPy 2.0 (2024), wipes value-based logic.[^nep50] two rules survive:

[^nep50]: NumPy Enhancement Proposal 50, _Promotion Rules for Python Scalars_, https://numpy.org/neps/nep-0050-scalar-promotion.html. the proposal was authored by Sebastian Berg; the discussion thread on the Scientific Python forum captures the practitioner reaction (https://discuss.scientific-python.org/t/nep-50-promotion-rules-for-python-scalars/280).

1. **array-array promotion** is a fixed function of the two input dtypes. no dependence on contents.
2. **Python scalars** are _weakly_ typed: they carry a kind (`bool`, `int`, `float`, `complex`) but no precision, and they defer to the array dtype when mixed.

the post-NEP-50 version of the pathology becomes:

```python
np.array([1], dtype=np.int8) + 200      # int8 array, with overflow warning
                                        # value 1 + 200 = -55 (wrap)
```

the overflow is now the user's problem — and that is the whole point. NEP 50 trades a hidden silent widen for a loud, deterministic, type-stable wrap. monpy ships exactly this contract.

the cost paid for type stability is that mixed-precision arithmetic with Python literals is no longer "safe" in the colloquial sense. `np.uint8(100) + 200` raises `OverflowWarning` and returns `np.uint8(44)` rather than promoting to `int16`. code that previously relied on automatic widening must now annotate intent explicitly via `np.array(...).astype` or by using `np.int16` literals. the trade is worth it because the alternative — value-dependent dispatch — is incompatible with any compiler stack that wants to specialise kernels at trace time.

## 2. promotion as a join, and where the lattice claim breaks

let $D$ be the 14-element set of monpy dtypes. define the binary operation $\sqcup : D \times D \to D$ by `dtype_result_for_binary(a, b)` in `src/domain.mojo`. the intuition is "promote both to a common dtype that holds both ranges." we catalogue the algebraic properties.

**Proposition 2.1 (commutativity).** $a \sqcup b = b \sqcup a$ for all $a, b \in D$.

_Proof._ the implementation in `src/domain.mojo` decomposes by ordered kind pair $(K(a), K(b))$ but symmetrises by canonicalising the smaller-kind argument first. the width-promote helper is symmetric in its two width arguments. direct inspection of the 25 kind-pair branches verifies that swapping arguments lands in the same branch with equal output. $\square$

**Proposition 2.2 (idempotence).** $a \sqcup a = a$.

_Proof._ the kind pair $(K(a), K(a))$ resolves to "same kind," and the width-promote helper returns $\max(w(a), w(a)) = w(a)$. $\square$

**Proposition 2.3 (top element).** `complex128` is a top element: $a \sqcup \text{complex128} = \text{complex128}$ for all $a \in D$.

_Proof._ the complex absorption rule: $K(\text{cfloat}) \sqcup K(x) = \text{cfloat}$ in all five kind cases, with width $\max(w(\text{complex128}), 2 w(x))$. since $w(\text{complex128}) = 128$ and $\max_{x \in D} 2 w(x) = 128$ (achieved at `float64`), the result width stays at 128. $\square$

so far so good. now the load-bearing question.

**the associativity claim is FALSE.** the standard NumPy-shaped promotion rules — including monpy's — admit the following counter-example, identified canonically in the JAX type-promotion design document:[^jaxlat]

[^jaxlat]: JAX type promotion JEP, _Design of Type Promotion Semantics for JAX_, https://docs.jax.dev/en/latest/jep/9407-type-promotion.html. JAX deliberately abandoned NumPy's table in favour of a sparser DAG precisely because the NumPy rules cannot be made associative without redesign. the JEP is the cleanest articulation of why a true lattice would be preferable, and what NumPy gives up by not having one.

let $a = \text{int8}$, $b = \text{uint8}$, $c = \text{float16}$. apply monpy's rules:

$$
\begin{aligned}
(a \sqcup b) \sqcup c &= \text{int16} \sqcup \text{float16} = \text{float32} \\
a \sqcup (b \sqcup c)  &= \text{int8} \sqcup \text{float16} = \text{float16}
\end{aligned}
$$

the first line: signed × unsigned at equal width promotes to the next signed width up (section 3, rule 3), so int8 ⊔ uint8 = int16. then int16's range $[-2^{15}, 2^{15})$ exceeds float16's exact-integer range $[-2^{11}, 2^{11}]$, so we escalate to float32. the second line: uint8's range $[0, 2^{8})$ fits inside float16's exact-integer range, so uint8 ⊔ float16 = float16. then int8's range $[-2^7, 2^7)$ also fits, so int8 ⊔ float16 = float16. the two associations disagree.

**Theorem 2.4 (the structure of $(D, \sqcup)$).** the pair $(D, \sqcup)$ is a commutative idempotent magma with a top element. it is _not_ a join-semilattice. equivalently, the relation $a \leq b \iff a \sqcup b = b$ is not a partial order: transitivity is broken by the section-2 counter-example.[^antisym]

[^antisym]: one can salvage a partial order by _defining_ $\leq_C$ via `can_cast(., ., "safe")` directly (see section 4) rather than via the join. the two relations agree on most pairs but diverge on pairs like (int16, float16) where int16 ≤_C float64 holds in the safe-cast poset but `int16 ⊔ float16 = float32` in the join — note these aren't logically incompatible, but they mean the join is not the least upper bound under ≤_C either.

this is a real cost of the NEP 50 design. it is what you get when you (a) want kind escalation to be "minimal" and (b) want the table to be a function of dtype pairs only. the cost is legible: programs that fold a sequence of arrays via `np.add.reduce` or via explicit binary chaining can produce results that depend on the _associativity grouping_, even though `+` is mathematically associative on the underlying values.

monpy's `result_type(*args)` resolves this in practice by **fixing a left-fold**: the implementation in `python/monpy/__init__.py` reduces the argument list left-to-right against `_DT_BINARY[a][b]`, falling through to `_native._result_dtype_for_binary` for triples not in the cache. this is a deterministic choice but a choice — a different left/right convention would produce different dtypes for the JAX-style triple. we document this so callers can reason about it.

**what is true.** restrict $D$ to a _single kind_ — say all signed ints, or all floats. within a kind, $\sqcup$ is just $\max$ on width, which trivially makes each kind-restricted set a totally ordered chain (hence a lattice). the cross-kind structure is where associativity fails. most production code stays inside one kind and never sees the failure; the failure mode emerges in mixed-precision libraries (mixed-int autodiff, fp16 training with int8 quantisation tables) where the JAX team felt enough pain to redesign the whole thing.[^jaxchoice]

[^jaxchoice]: JAX's solution: insert _weak_ nodes (`i*`, `f*`, `c*`) for Python scalars, redraw the graph as a true join-semilattice, and accept that some NumPy-compatible promotions change. NumPy itself, and monpy, opted for table compatibility over algebraic cleanliness.

## 3. the 14×14 table, decomposed by kind

a 196-entry table is opaque. the actual structure is much smaller. define the kind function $K : D \to \{\text{bool}, \text{sint}, \text{uint}, \text{float}, \text{cfloat}\}$ and the width function $w : D \to \{1, 8, 16, 32, 64, 128\}$. the promotion algorithm is then a 5×5 dispatch on kind pairs, with each branch consulting widths.

| ↓ a / b →  | bool        | sint                       | uint                       | float                      | cfloat                     |
| ---------- | ----------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- |
| **bool**   | bool        | sint(w_b)                  | uint(w_b)                  | float(w_b)                 | cfloat(w_b)                |
| **sint**   | sint(w_a)   | sint(max)                  | sint(max(w_a, 2 w_b)) ¹    | float(promote_int_float) ² | cfloat(promote_int_cfloat) |
| **uint**   | uint(w_a)   | sint(max(2 w_a, w_b)) ¹    | uint(max)                  | float(promote_int_float) ² | cfloat(promote_int_cfloat) |
| **float**  | float(w_a)  | float(promote_int_float) ² | float(promote_int_float) ² | float(max)                 | cfloat(max(w_a·2, w_b))    |
| **cfloat** | cfloat(w_a) | cfloat(promote_int_cfloat) | cfloat(promote_int_cfloat) | cfloat(max(2 w_b, w_a))    | cfloat(max)                |

¹ capped at 64 — there is no signed int wider than 64. the (uint64, int64) and (int64, uint64) cells fall through to **float64** (see section 7a). ² `promote_int_float(w_int, w_float)` returns 64 if `w_int ≥ 32`, else $\max(w_\text{float}, 32)$ if `w_int = 16`, else $\max(w_\text{float}, 16)$ if `w_int = 8`. the 32-bit threshold is the float32 exact-integer boundary at $2^{24}$ — int32's full range is $2^{31}$, exceeding float32's exact-integer regime, so escalation to float64 is mandatory.

this decomposition collapses 196 entries to 25 kind-pair branches plus three width-promotion helpers (`promote_int_float`, `promote_int_cfloat`, signed-unsigned widening). the implementation in `src/domain.mojo` follows this structure literally — a `match` over the kind pair, followed by an arithmetic computation on widths — which is why the code reads as roughly 80 lines rather than a 196-cell static table.

**same-kind promotion** is just $\max$ on width (rows/columns 1–4 of the diagonal). a pair of int32s gives int32; int8 ⊔ int32 = int32. there is nothing structural to say.

**bool with anything** returns the other type. `bool` sits below all integer kinds in the precision order. `bool ⊔ uint8 = uint8`, `bool ⊔ float64 = float64`. the implementation treats bool as a 1-bit subset of any wider integer kind.

**signed × unsigned at width $w$** returns signed int of width $\min(2w, 64)$ when the unsigned operand could overflow signed at that width. the clean case: `int16 ⊔ uint8 = int16` because $\text{uint8}_\max = 255 < \text{int16}_\max = 32767$. the collision case: `int16 ⊔ uint16 = int32` because $\text{uint16}_\max = 65535 > \text{int16}_\max$. the pathological case: `int64 ⊔ uint64 = float64` (section 7a).

**integer × float** is the rule that breaks "smallest containing." it must escalate width when the int's magnitude exceeds the float's mantissa.[^ieee] float16 has an 11-bit mantissa (exact integers up to $2^{11}$); float32 has 24 bits ($2^{24}$); float64 has 53 bits ($2^{53}$). the rule:

[^ieee]: IEEE 754-2008 / 2019 specifies the binary16 (1+5+10), binary32 (1+8+23), binary64 (1+11+52) layouts. the "+1" hidden bit on the mantissa is what gives 11/24/53 bits of integer precision rather than 10/23/52.

| int width | × float16 | × float32 | × float64 |
| --------- | --------- | --------- | --------- |
| 8         | float16   | float32   | float64   |
| 16        | float32   | float32   | float64   |
| 32        | float64   | float64   | float64   |
| 64        | float64   | float64   | float64   |

note that `int16 + float16 = float32` rather than `float16` — int16's range $[-32768, 32767]$ exceeds float16's exact range $[-2048, 2048]$. and `int32 + float32 = float64` rather than `float32` — int32 exceeds float32's $2^{24}$.

**complex absorption.** any cfloat ⊔ x returns cfloat with width $\max(w(\text{cfloat}), 2 \cdot w_\text{required}(x))$ where $w_\text{required}$ is the float width that would be selected if the cfloat were replaced with a float. complex64 is two float32s; complex128 is two float64s. so `complex64 ⊔ int32 = complex128` (because int32 forces float64) but `complex64 ⊔ int8 = complex64`.

## 4. can_cast as a partial order

define $a \leq_C b$ iff `dtype_can_cast(a, b, "safe") == True`. NumPy's documentation phrases "safe" as "casts which can preserve values."[^cancast] concretely: every value representable in $a$ is exactly representable in $b$.

[^cancast]: NumPy reference, _numpy.can_cast_, https://numpy.org/doc/stable/reference/generated/numpy.can_cast.html. note the NumPy-2.0 change: "does not support Python scalars anymore and does not apply any value-based logic for 0-D arrays and NumPy scalars" — the same NEP 50 cleanup that affected promotion.

**Theorem 4.1.** $\leq_C$ is a partial order on $D$.

_Reflexivity_: every dtype casts to itself with no value loss.

_Antisymmetry_: $a \leq_C b$ and $b \leq_C a$ require that values flow both ways losslessly. this forces equal widths, equal kinds (a uint16 cannot losslessly hold all int16 values nor vice versa), and equal sign-handling. in monpy's registry, $(K, w)$ uniquely identifies a dtype, so $a = b$.

_Transitivity_: if every $a$-value is an exact $b$-value, and every $b$-value is an exact $c$-value, then every $a$-value is an exact $c$-value by composition. the case structure is finite — bool ↪ uint ↪ float, bool ↪ sint ↪ float, float ↪ cfloat — and each link preserves the embedding. $\square$

the Hasse diagram of $\leq_C$ is the standard "kind chain" picture: bool at the bottom, then four parallel chains (signed-int chain, unsigned-int chain, float chain, complex chain) with cross-edges where exact-value containment holds.

```
                          complex128
                         /          \
                  complex64        float64
                  /   |   \      /   |   \
            float32  ... int32  uint32 ...
            /  |
       float16 ...
              ...
           bool
```

(schematic — the actual edges include int8 ↪ int16 ↪ int32 ↪ int64 within signed, plus int{8,16} ↪ float32, int8 ↪ float16, uint{8,16} ↪ float{16,32}, all ints ↪ float64, all floats ↪ complex of double width.)

**the five casting modes** form a stack of progressively weaker relations:

| mode          | semantics                                          | structure                                     |
| ------------- | -------------------------------------------------- | --------------------------------------------- |
| `"no"`        | `a == b` exactly                                   | identity (the discrete order)                 |
| `"equiv"`     | only byte-order differs; `a == b` in monpy         | identity (same as `"no"` since we're LE-only) |
| `"safe"`      | every value of `a` is exactly representable in `b` | the partial order $\leq_C$                    |
| `"same_kind"` | same `K(a) = K(b)`, or safe                        | union of $\leq_C$ with within-kind equality   |
| `"unsafe"`    | any cast                                           | the total relation $D \times D$               |

`"same_kind"` is the casting mode used inside ufuncs by default — a `np.add` on two float arrays will silently truncate float64 → float32 if the destination is f32, but will refuse a float → int truncation. monpy follows this exactly.

**Proposition 4.2 (safe ⊆ promotion).** if $a \leq_C b$ then $a \sqcup b = b$.

_Proof._ if every $a$-value embeds losslessly into $b$, then $b$ is itself a "common type containing both," and the algorithm's minimisation rule selects $b$ (or something even larger, but the algorithm's specific design picks the minimum). inspect the cases: (i) bool ≤_C anything → join with anything returns anything; (ii) intN ≤_C int{N+k} → join is the wider; (iii) intN ≤_C floatM with M sufficient → join is floatM; (iv) floatN ≤_C floatM (M ≥ N) → join is floatM; (v) anything ≤_C cfloat → join is cfloat. $\square$

the converse fails — `int16 ⊔ float16 = float32` even though int16 is _not_ safely castable to float16. the join is constructed to be at least as large as both inputs, but it goes further when neither input contains the other.

## 5. reduction dtype rules

`sum`, `prod`, and `cumsum/cumprod` follow a separate rule: small integer kinds are widened to 64-bit before accumulation. the function `dtype_result_for_reduction(dt, op)` in `src/domain.mojo` encodes:

| input kind                | sum / prod / cumsum / cumprod | min / max / mean / std / var       |
| ------------------------- | ----------------------------- | ---------------------------------- |
| bool, int8, int16, int32  | int64                         | input dtype (mean accumulates f64) |
| uint8, uint16, uint32     | uint64                        | input dtype                        |
| int64, uint64             | input                         | input                              |
| float16, float32, float64 | input                         | input                              |
| complex64, complex128     | input                         | input                              |

the motivation is overflow protection. summing $N$ int8 values in int8 overflows after roughly $N = 2^7$ in the worst case (all 127s); summing in int64 buys $2^{56}$ headroom over the worst-case partial sum. the engineering choice is that this widening is **automatic** — the user does not opt in via `dtype=`; it is the default behaviour of `sum` over small ints. the dtype keyword can override.

`mean`, `std`, `var` accumulate in float64 internally but cast back. this is the "Kahan-friendly intermediate" pattern: the _output_ dtype matches the input (so a float32 array's mean is float32), but the accumulation runs at higher precision to keep round-off tolerable. this means `mean` is not strictly `sum / N` at output dtype; the equality only holds approximately. monpy preserves this contract.

`min` and `max` cannot overflow (they reduce range), so they preserve dtype unconditionally.

for complex, no widening — complex64 sum stays complex64, complex128 stays complex128. the argument for not widening: complex arithmetic is already expensive enough that automatic widening would surprise users in a different direction.

## 6. weak-scalar dispatch

a Python `int(3)` does not have a fixed dtype until it meets an array. NEP 50 calls this _weak_ and defines:

- `int` is weak with default kind `int64` (or `int32` on Windows for historical reasons; monpy uses `int64` uniformly across platforms — the Mojo native side has no Windows-specific divergence).
- `float` is weak with default `float64`.
- `complex` is weak with default `complex128`.
- `bool` is weak with default `bool`.

`_coerce(value)` in `python/monpy/__init__.py` performs the Python → dtype mapping. the promotion against an array follows:

$$
\text{array}_t \oplus \text{weak}_w =
\begin{cases}
\text{array}_t & \text{if } K(t) \succeq K(w) \\
\text{array}_{P(t, w)} & \text{otherwise}
\end{cases}
$$

where $\succeq$ is the kind ordering bool $\preceq$ int $\preceq$ float $\preceq$ complex, and $P(t, w)$ is "promote $t$ to the smallest dtype of kind $K(w)$ that contains $t$."

concrete cases:

| array dtype $t$ | weak scalar  | result    | comment                                      |
| --------------- | ------------ | --------- | -------------------------------------------- |
| int8            | weak-int     | int8      | kinds equal, defer                           |
| int8            | weak-float   | float16   | kind escalates; float16 contains int8        |
| int8            | weak-complex | complex64 | kind escalates; complex64 = 2×float32 ⊃ int8 |
| float32         | weak-int     | float32   | kinds match (both ≼ float), defer            |
| bool            | weak-int     | int64     | kind escalates to int64 (the weak default)   |
| bool            | weak-float   | float64   | kind escalates to float64                    |

the asymmetry to flag is the last two rows. when meeting a `bool` array, a Python `int` does _not_ defer to bool's width — there is no "smallest int containing bool" rule that would give int8. instead it uses the weak default `int64`. the reason is that bool sits at the top of bool-kind, has nowhere wider to go within its own kind, so the kind-escalation defaults to the weak dtype's full precision.

this makes the weak rule asymmetric in a useful way: it preserves array dtype precision when the array's kind dominates, but uses the weak default precision when the kind escalates. code reasoning: if you wrote `bool_array + 1`, you almost certainly mean "treat this as integer addition with a default integer," not "find a 1-bit container for the answer."

NEP 50 explicitly does not define behaviours that depend on the _value_ of the weak scalar. `bool_array + 1` and `bool_array + 1000000` both produce int64 arrays (with overflow on the second if forced to a narrower dtype downstream). weak scalars are pure type tokens.

## 7. subtleties and footguns

### 7a. signed × unsigned at width 64

`int64 ⊔ uint64 = float64`. there is no signed integer of width 128 in monpy (or in NumPy proper, since `int128` is not standard), so the algorithm cannot produce a signed integer wide enough to hold both ranges. the fallback is float64.

this is **lossy**. uint64 values above $2^{53}$ lose precision in float64 because float64's mantissa is 52+1 = 53 bits. the promotion is a deliberate engineering compromise: the alternative is to error out, which would be more correct but would break too much existing code. NumPy chose float64; monpy inherits the choice for compatibility.[^int128]

[^int128]: there has been recurring discussion in numpy/numpy issues about adding `int128` for exactly this case (see issues around #22624, #23102, and the longstanding "uint64 promotion is bad" thread). the blocker is hardware: x86-64 has no 128-bit integer arithmetic in baseline ISA, and software emulation is slow enough that the dtype would be a footgun of a different kind.

the practical advice: if you find yourself mixing int64 and uint64, something has probably gone wrong upstream. pick one signedness and stay there.

### 7b. the "smallest containing" principle is sometimes violated

`float16 + int32 = float64`, not float32. walking through the rule: int32's range $[-2^{31}, 2^{31})$ exceeds float32's exact-integer range $[-2^{24}, 2^{24}]$. a float32 cannot exactly represent every int32, so the algorithm escalates to float64.

this violates a naive reading of "smallest type containing both" because float64 is two steps up from float16, not one. but under the correct reading — "smallest type containing both _exactly_" — float64 is correct: it is the smallest standard float that contains int32 exactly, and any float type wide enough to contain int32 trivially contains float16.

`int16 + float16 = float32` follows the same logic: int16 exceeds float16's $2^{11}$ exact range, so escalate to float32, which contains both.

### 7c. the non-associativity already discussed

restating section 2's counter-example for emphasis. with $a = \text{int8}$, $b = \text{uint8}$, $c = \text{float16}$:

```python
result_type(result_type(int8, uint8), float16)   # → float32
result_type(int8, result_type(uint8, float16))   # → float16
```

the outer `result_type` is the same function in both calls, but the _intermediate_ type carries information the inputs did not. `int8 ⊔ uint8 = int16`, and int16 is "more precise" in width than either input, which then forces the cross to float32. in the right-associated version, the cross-kind step happens before the signed-unsigned crossing, and uint8 is small enough to live inside float16.

monpy's `result_type(*args)` left-folds. this is a deterministic resolution. it is not a "fix" — the structure remains non-associative, and any code that relies on a different fold order will diverge.

### 7d. Python float vs array of int

`int8_array + 1.5`. the Python float is weak-float64 by default. under NEP 50:

1. the kind escalates from int (the array's kind) to float (the scalar's kind).
2. the rule for kind escalation with a weak scalar uses the _array_'s precision class when promoting in-kind, but uses the _minimum precision needed to contain the array_ when escalating across kinds.
3. for int8, the minimum float that contains it exactly is float16.

so `int8_array + 1.5` produces a float16 array. the Python float's "default" precision (float64) is _not_ used — the kind matters but the precision defers to whatever the array can be safely lifted into. this is the asymmetry that makes weak scalars interesting: kind from the scalar, precision from the array.

compare with `int32_array + 1.5`: int32 cannot live in float16 or float32 exactly, so the result is float64. the "default float64" precision of the Python literal becomes operative precisely because int32 forces float64 anyway.

the reader uncomfortable with this should run the table out for all (int kind, weak-float) pairs and confirm: weak-float defers to the array's required float precision, never overrides it upward.

### 7e. bool overflow asymmetry

`bool_array + 1` is `int64`, but `bool_array + np.int8(1)` is `int8` (because the right operand is now a _strong_ dtype). one uses the weak default; the other uses the explicit dtype. the algebra is different in each case. this catches people who try to "lock in a small dtype" by writing `bool_array + 1` and discover their memory budget tripled.

the fix: `bool_array + np.int8(1)` or `bool_array.astype(np.int8) + 1`.

## 8. monpy implementation specifics

the dispatch path for `result_type` in monpy:

1. **`python/monpy/__init__.py`: `_coerce(value)`** — accepts any Python object and returns a dtype handle. for Python `bool`, `int`, `float`, `complex` it returns the corresponding weak dtype. for NumPy scalars and arrays it extracts the strong dtype. for monpy arrays it reads `arr.dtype` directly.

2. **`_DT`** — registry mapping dtype name (`"int8"`, `"float64"`, etc.) to a struct carrying `(kind, width, native_handle)`. the native handle is what gets passed to Mojo when we cross the FFI boundary.

3. **`result_type(*args)`** — folds left over the argument list. for each pair `(a, b)`, it consults `_DT_BINARY[a][b]`, a Python-side cache of the 14×14 table. cache misses (which should be rare) call into `_native._result_dtype_for_binary(a, b)`.

4. **`src/domain.mojo: dtype_result_for_binary(a, b)`** — the source of truth. implemented as a `match` over the kind pair $(K(a), K(b))$, with each branch computing a target width via the rules in section 3. the function is total and pure — same inputs, same output, no side effects.

5. **`dtype_can_cast(src, dst, casting)`** — implements the five casting modes by switching on the mode and consulting the kind/width relation. `"safe"` checks the precision-containment partial order $\leq_C$; `"same_kind"` falls through to a kind-equality check; `"unsafe"` returns True unconditionally.

6. **`dtype_result_for_reduction(dt, op)`** — implements section 5's reduction widening. a switch on the op enum (`SUM`, `PROD`, `CUMSUM`, `CUMPROD`, `MEAN`, `STD`, `VAR`, `MIN`, `MAX`) and the input dtype's kind. small ints widen for sum/prod; everything else preserves.

7. **`dtype_result_for_unary(dt, op)`** — handles unary operations. `abs` on signed int preserves; `abs` on complex returns the corresponding float (complex64 → float32, complex128 → float64). `-x` preserves dtype on signed/float/complex but raises on unsigned (or wraps; current monpy wraps with a warning to match NumPy).

the split between Python-side caching and Mojo-side authority is deliberate: the cache amortises FFI cost for hot paths (`result_type` is called in every elementwise op), while Mojo retains the canonical implementation for any pair the cache hasn't seen, including future dtype additions.

## 9. memory alignment relevance

promotion has direct consequences for storage layout. the output array of an elementwise op must satisfy alignment requirements for its dtype's SIMD path. f64 wants 8-byte alignment for AVX-512 `vmovapd`; f32 wants 4-byte alignment for `vmovaps`; complex128 wants 16-byte alignment so that the real and imaginary halves can be loaded as a single 128-bit lane on platforms with vector complex support. monpy's allocator overaligns to 64 bytes for cache-line friendliness regardless, so the dtype-specific alignment is always satisfied by construction — but the **kernel dispatch** must still select the right instruction set based on the promoted dtype.

the relevant invariant is: after `result_type(a, b)` selects dtype $d$, the elementwise kernel receives a freshly allocated output buffer of dtype $d$, and that buffer is guaranteed by the allocator to be aligned to $\max(64, w(d)/8)$. the kernel can therefore unconditionally use the most aligned load/store variant for $d$. there is no fallback path for "dtype is f64 but buffer is misaligned" — that case is unreachable.

`astype` is the more interesting case for allocation. `astype(src, dst_dtype)` always returns a new array, even when `src.dtype == dst_dtype` (NumPy's `copy=False` extension is supported but defaults to copy). the reason: the contract of `astype` is value-preservation under the cast rules, and downstream code assumes the result is independently owned. for `f64 → f32`, this means walking the source buffer and producing a new buffer of half the size. for `f32 → f64`, doubling the size.

a subtle alignment consequence: if `src` is `f64` with 8-byte stride and the cast target is `f32`, one might imagine writing the f32 result _into_ `src`'s buffer, since each f32 fits in the low 4 bytes of each f64 slot. monpy does not do this — `astype` always allocates fresh — but a future `astype_inplace` operation could exploit the fact that the new dtype's width is ≤ the old dtype's width. the constraint is alignment: the new buffer's stride must be a multiple of the new dtype's width, which is automatic when downcasting (4 | 8) but not when crosscasting (e.g., f32 → complex64 needs stride doubled, allocation forced).

for complex casts, the stride relationship is the inverse. `astype(complex64, float32)` must extract the real part and produce a buffer half the size; the source buffer's f32-component alignment is automatically satisfied because complex64 is two f32s back-to-back, so reading every other f32 is well-aligned. `astype(float32, complex64)` must produce a buffer twice the size with the imaginary parts zero-filled; this is a fresh allocation by necessity.

the cross-reference here is `memory-alignment.md`, which lays out the full proposal for monpy's allocator including overalignment, page-aligned bulk allocations for arrays > 2 MiB, and the SIMD alignment invariants per dtype. the dtype promotion logic does not interact with most of those concerns directly — it determines _which_ dtype the output should be, after which the allocator handles alignment uniformly.

the one place promotion does affect allocation strategy is **temporary buffers in chained operations**. consider `(f32_array + f64_array) * f32_array`. the intermediate `f32 + f64` allocates a temporary f64 buffer; the outer multiply with f32 promotes again to f64, and the temporary's alignment must support f64 SIMD. the allocator's overalignment policy makes this trivially correct, at the cost of a few wasted bytes per array on the small end.

## references

1. NumPy Enhancement Proposal 50, _Promotion Rules for Python Scalars_, https://numpy.org/neps/nep-0050-scalar-promotion.html
2. NumPy reference manual, _Data type promotion in NumPy_, https://numpy.org/doc/stable/reference/arrays.promotion.html
3. NumPy reference manual, _numpy.can_cast_, https://numpy.org/doc/stable/reference/generated/numpy.can_cast.html
4. JAX project, _Design of Type Promotion Semantics for JAX_ (JEP 9407), https://docs.jax.dev/en/latest/jep/9407-type-promotion.html
5. Davey, B. A. & Priestley, H. A., _Introduction to Lattices and Order_, 2nd ed., Cambridge University Press 2002.
6. IEEE Std 754-2019, _IEEE Standard for Floating-Point Arithmetic_. the 2008 revision is functionally equivalent for the binary{16,32,64} formats relevant here.
7. NumPy issue tracker discussion on int64+uint64 promotion: https://github.com/numpy/numpy/issues/22624 and the longer thread it links to.
8. Sebastian Berg's PR introducing optional NEP 50 logic to NumPy: https://github.com/numpy/numpy/pull/21626 — the implementation history is useful for tracking which behaviours are NEP 50 strictly versus pragmatic legacy hold-outs.
