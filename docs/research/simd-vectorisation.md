---
title: "SIMD vectorisation strategy in monpy"
date: 2026-05-07
---

# SIMD vectorisation strategy in monpy

_the inner loop is memory-bound by 24× on M3; every vectorisation decision that doesn't address memory traffic is theatre._

monpy's elementwise kernels live in `src/elementwise.mojo`; the dispatch happens through `apply_binary_typed_vec[dtype, width]` and `apply_unary_preserve_typed_vec[dtype, width]`, with strided fallbacks in `src/array.mojo`. the width parameter is always chosen so that `width * sizeof(dtype) == 32`. that single rule does most of the work in this document. the reason it is the right rule on Apple Silicon is _not_ that 32 bytes wins on instruction throughput — it loses to 16-byte NEON in a register-pressure sense and to 64-byte AVX-512 in a per-instruction-FLOPs sense. it wins because the load/store pipeline on Firestorm-derived cores absorbs two 16-byte μops per cycle without breaking a sweat, and because every elementwise kernel here is _memory-bound by a factor of ~24×_ on M3, so the only thing the inner loop must not do is stall on alignment, branches, or strided gathers.

## 1. the vector-width selection rule

for a 32-byte target the lane count is $\lceil 32 / s \rceil$ where $s$ is `dtype.sizeof()`:

| dtype     | $s$ (B) | lanes | NEON regs touched | AVX2 regs |
| --------- | ------- | ----- | ----------------- | --------- |
| f64 / i64 | 8       | 4     | 2 × `q`           | 1 × `ymm` |
| f32 / i32 | 4       | 8     | 2 × `q`           | 1 × `ymm` |
| f16 / i16 | 2       | 16    | 2 × `q`           | 1 × `ymm` |
| i8        | 1       | 32    | 2 × `q`           | 1 × `ymm` |

this is exactly the table monpy hard-codes through `comptime` width parameters at the call site. three things to notice:

1. the lane count on integer paths matches the float path of the same size, so a single `apply_binary_typed_vec` template covers both. the template only differs in which intrinsic gets selected (e.g. `add` vs `fadd`) — Mojo's `SIMD[T, w]` dispatches on `T`.
2. NEON only has 128-bit Q-registers, so a 32-byte logical SIMD step decomposes into a pair of `ldp q0, q1, [x0]` plus two arithmetic ops. Firestorm-class cores have 4 NEON pipes each capable of one 128-bit FMA per cycle (latency 4c on `fmla.4s`, throughput 1c)[^firestorm-tables]; issuing two ops per logical step uses half the SIMD bandwidth, leaving the other two pipes free for an interleaved unrolled iteration.
3. on AVX2 the same 32B logical step is one `vmovaps ymm` plus one `vfmadd*ps`. AVX2 has typically 2 FMA units (Skylake-class) at 4c latency, throughput 1c per port — so the same pseudocode lights up half the FP issue width at full rate.

### why not 16B (one NEON Q-register)?

smaller vectors lower register pressure to almost zero (one source pair + one destination per iteration) and look attractive for memory-bound code where you don't want to spill. but the OoO core behind NEON has more than enough rename registers (≥ 128 architectural-equivalent slots on Firestorm); the actual ceiling on 16B kernels is not register pressure but _front-end issue rate_. with 16B you need twice as many μops to consume the same number of bytes, which means twice the decode/dispatch traffic. on Apple Silicon the front end is 8-wide[^firestorm-frontend], so the ceiling is high, but you still leave perf on the floor by under-vectorising.

### why not 64B (AVX-512-style)?

two reasons.

**frequency scaling.** pre-Ice-Lake Intel parts entered an "AVX-512 license" on heavy 512-bit ops, dropping single-core turbo by 200-300 MHz on Skylake-X. Ice Lake reduced this to ~100 MHz at single-core[^icl-freq] and Rocket Lake/Sapphire Rapids effectively eliminated it. but the cost is asymmetric: the throttle hits _all_ cores in a domain when one core executes 512-bit ops, so a 5% kernel speedup can cost 10% on the rest of the workload. monpy can't gate this per-process.

**ISA availability.** Apple Silicon has no AVX-512 analogue. SVE2 hardware only exists at 128b on currently-shipping cores (Neoverse V1 is 256b but rare), and Apple has not implemented SVE at all. picking 64B as the canonical width would mean two completely separate kernel variants per dtype, doubling code size and i-cache footprint for a target that's mostly running on M-series laptops.

32B threads the needle: it is exactly two NEON ops per step (no waste), exactly one AVX2 op per step (full ISA on x86), and never triggers AVX-512 frequency licensing. the "wrong" decomposition on NEON — issuing two 16B μops when a single 32B intrinsic would be cleaner — is invisible to the OoO core, which already sees the 4-wide NEON port array as the bottleneck and pairs the two halves trivially.

[^firestorm-tables]: Dougall Johnson's Firestorm tables remain the best public source. M3 microarchitecture is incremental over Firestorm/Avalanche; the per-pipe latencies have not budged since the A14, though M3 Pro NEON throughput on dot-product loops is ~140% of M1 Max [(Eclectic Light Co., Dec 2023)](https://eclecticlight.co/2023/12/05/evaluating-m3-pro-cpu-cores-4-vector-processing-in-neon/), most of which is frequency + scheduling rather than pipe count.

[^firestorm-frontend]: 8-wide decode on Firestorm is widely cited as "the widest commercialised design"; the dispatch unit can sustain ~8 μops/cycle into the schedulers when the front end is hot [(AnandTech M1 deep-dive)](https://www.anandtech.com/show/16226/apple-silicon-m1-a14-deep-dive/2).

[^icl-freq]: Travis Downs' [Ice Lake AVX-512 frequency post](https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html) measured a 100 MHz single-core drop (3.7 → 3.6 GHz) on i5-1035G4. multi-core workloads showed no width-related delta. this is the death of the "AVX-512 always throttles" folklore on modern parts.

## 2. the compute-bound vs memory-bound regime — and why it dominates

for binary elementwise over $N$ elements the cost model is:

$$
\text{bytes moved} = 3 N s, \qquad \text{FLOPs} = N \cdot k
$$

with $k=1$ for `add/sub/mul/div`, $k=2$ for FMA-fusable patterns. arithmetic intensity:

$$
I = \frac{N k}{3 N s} = \frac{k}{3 s}.
$$

plugging in:

| dtype | $k=1$ | $k=2$ (FMA) |
| ----- | ----- | ----------- |
| f64   | 0.042 | 0.083       |
| f32   | 0.083 | 0.167       |
| f16   | 0.167 | 0.333       |
| i32   | 0.083 | —           |

the Roofline is $P_{\text{achieved}} = \min(P_{\text{peak}}, B \cdot I)$ where $B$ is the achievable bandwidth (Williams/Waterman/Patterson, CACM 52(4), 2009). on M3 we have:

- $P_{\text{peak}} \approx 200$ GFLOP/s f32 per p-core (4 FMA pipes × 8 lanes × ~3.5 GHz boost / 2 because FMA is 2 FLOP/op gives ~224, derate for issue stalls).
- $B \approx 100$ GB/s sustained (STREAM Triad on M-series ranges from ~59 GB/s on M1 to ~103 GB/s on M4[^stream]; M3 sits around 80 GB/s achieved on a single thread, ~100 GB/s with multiple threads).
- ridge point $I_{\text{ridge}} = P/B \approx 200 / 100 = 2$ FLOP/B.

f32 add intensity is 0.083 FLOP/B → **24× below the ridge**. that number is what governs every other decision in this document. doubling SIMD width, doubling FMA pipes, or doubling clock all hit the same bandwidth wall. the only knobs that _do_ shift achieved performance are:

1. cache reuse (turning the kernel from "stream from DRAM" into "stream from L2" — L2 on M3 p-clusters is 16 MB, peak ~250 GB/s read).
2. prefetch coverage (M3 has a strong stride prefetcher; we don't have to do much, but we must not break it with strange access patterns).
3. alignment (split loads across 64B cache lines cost extra cycles and double the load-port count).
4. avoiding redundant traffic (e.g. fusing `a*b + c` instead of writing the temporary).

the SIMD-width discussion in §1 is mostly about correctness and code size. the runtime impact on a fully-DRAM-resident array is single-digit percent.

[^stream]: [(Arxiv 2502.05317, Apple Silicon for HPC, 2025)](https://arxiv.org/html/2502.05317v1) — STREAM measurements; ~85% of theoretical peak across the lineup.

### what changes in cache?

run the same f32 add over $N=2{\cdot}10^5$ (= 800 KB working set, fits in M3's 16 MB L2):

- bytes from L2: $3 \cdot 800{\rm KB} = 2.4$ MB at ~250 GB/s = **9.6 µs**.
- FLOPs: $2{\cdot}10^5 / 2{\cdot}10^{11}$ = **1 µs**.

now we're at $I_{\text{eff}} = 0.083$ FLOP/B against an L2 roofline ridge of $250/200 = 1.25$ FLOP/B → still 15× below ridge but the cliff is shorter and we hit it faster. _in-cache elementwise can run an order of magnitude faster than DRAM-resident, even though SIMD width didn't change._ this is why blocking matters more than vectorisation for medium-sized arrays.

## 3. loop unrolling and software pipelining

for binary-typed kernels, Mojo's `@parameter for` over a small unroll factor lets us issue $u$ independent SIMD computations per loop body. why this helps on an OoO core that already reorders:

- the scheduler needs _visible_ independent ops to overlap. loop-carried dependencies (the loop counter) chain everything to a single critical path. unrolling severs the chain by $u$.
- FMA latency is 4c, throughput 1c. to saturate one pipe you need 4 in-flight FMAs at all times; with 4 pipes that's 16. an f32 binary kernel with a single FMA per iteration plus tight loop control may only present 6-8 in-flight ops to the scheduler, leaving 2 pipes idle. unroll by 4 doubles the in-flight budget.
- load latency on M3 L1 is ~4 cycles too.[^l1-lat] the prefetcher hides DRAM latency, but the load-to-use latency on the _first cache line_ still needs unrolling to overlap.

[^l1-lat]: ~4c L1d hit latency is the consensus from `lmbench`-style probes on Firestorm; no published M3-specific number disagrees.

### diminishing returns past 4×

above 4× unroll on f32 binary, three costs compound:

1. **register pressure.** 4 unrolled f32 iterations need 4 source pairs + 4 destinations + 1 loop counter + a few scratch = ~13 vector regs. plus loop-pipelined prefetches you're at ~17. the 32 NEON architectural registers absorb that, but the rename pool starts to fill. above 8× you spill.
2. **i-cache.** Mojo emits aggressive inlining; an 8× unrolled f32 binary kernel with all the type-specific intrinsics expands to ~400 bytes per dtype. with monpy supporting ~10 dtypes that's already 4 KB per _operation_; multiply by ~50 ops and we're well past the 192 KB i-cache.
3. **tail handling.** larger $u$ means longer scalar tails; for $u=8$ on f32 ($w=8$) the tail can reach $u \cdot w - 1 = 63$ scalar elements.

the sweet spot for monpy's elementwise binaries is $u=4$ on f32/i32 and $u=2$ on f64/i64 (same total in-flight register count). for complex multiply (which has a 4-op shuffle pattern per element, see `complex-kernels.md`), $u=4$ is essentially mandatory to amortise shuffle latency across pipes.

### the inner loop, schematically

```
  for i in range(0, n_aligned, u * w):
      @parameter
      for j in range(u):
          let a_v = load[w](a_ptr + i + j*w)   # SIMD[T, w]
          let b_v = load[w](b_ptr + i + j*w)
          store(c_ptr + i + j*w, fma(a_v, b_v, c_v))
```

this is what monpy's typed-vec kernels expand to after `@parameter` unrolling. the OoO core sees `u` independent dataflow chains and pipelines them across the 4 NEON pipes.

## 4. the strided-load story

strided arrays — produced by slicing, transpose, broadcasting — break the contiguous-load assumption. three tactics:

**gather instructions.** AVX2's `vpgatherdd` / `vgatherdps` and SVE2's `ld1*` with index vector. the cost on Skylake-X is ~5 cycles latency + 1/cycle throughput per _element_, so a gather of 8 lanes is ~13 cycles vs. 1 cycle for an aligned contiguous load — a 13× hit. NEON has no native gather; you build it from per-lane `ld1` with `INS`. SVE2 gather exists but executes scalar in the LSU and degrades linearly with vector width.[^sve2-gather]

[^sve2-gather]: [Zingaburga's SVE2 critique](https://gist.github.com/zingaburga/805669eb891c820bd220418ee3f0d6bd) is the practitioner reference for this.

**per-lane scalar load + insert.** this is the SIMD-emulated gather. ~$w$ cycles for the loads + $w$ for inserts — strictly worse than a hardware gather where one exists.

**materialise contiguous.** allocate a scratch buffer, copy strided → contig (one stride-aware loop), run the contig kernel, copy back if the destination is also strided. total cost: $2N s$ (copy in/out) + the contig kernel itself. the contig kernel is ~$3Ns/B$ time (memory bound), plus $2Ns/B$ for the copies. so **5/3× the time of a pure contig kernel**, but the kernel itself runs at full SIMD width.

this is a clean win over scalar fallback whenever $N$ is large enough that the SIMD speedup on the kernel exceeds the copy overhead. for monpy's f32 path with a ~6-8× SIMD speedup over scalar, the breakeven is around $N=64$ elements; for f16 the breakeven is closer to $N=16$ because the relative SIMD speedup is higher (16 lanes).

monpy's current strided kernels go scalar — verifiably leaving 3-5× perf on the table for arrays where only the outer axes are strided. the right fix is the one already sketched: detect `strides[-1] == itemsize` (innermost is contiguous) and treat the array as a stack of contiguous rows. the NumPy nditer optimisation chain calls this "axis flattening" — it is the single biggest perf bug in monpy's strided story right now.

## 5. predication and tail handling

when $N \mod w \neq 0$, the last vector is partial. three strategies:

| strategy               | ISA cost                                                                                                     | safety                | monpy uses |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------- | ---------- |
| pad to multiple of $w$ | needs scratch, adds $\le w-1$ wasted ops                                                                     | needs allocator hooks | no         |
| mask the tail          | AVX-512: `kmov` + masked op (1c). NEON: build a mask vector by hand. SVE2: `whilelo` + predicated op (free). | safe with ISA support | no         |
| scalar tail loop       | $\le w-1$ scalar iterations                                                                                  | always safe           | yes        |

the scalar-tail strategy is the worst per-iteration but the simplest to code. costs:

- for $N \gg w$: tail is amortised out. a 10⁶-element f32 add has at most 7 scalar tail iterations vs. 125,000 vector ones — 0.006%. invisible.
- for $N$ comparable to $w$: tail dominates. an $N=32$ f32 add with $w=8$ executes 3 vector ops + up to 7 scalar ops; the scalar tail is 30% of the work and runs at 1/8 the throughput → roughly 70% of total time.

the fix on AVX-512 / SVE2 is masked tails. monpy doesn't have SVE2 hardware to test on, and AVX-512 mask code paths would need their own per-dtype kernels. the right intermediate move is **vector-from-scalar broadcasts for very small N** — fall through to a different kernel below ~64 elements that does one vector load with a final masked store. this is on the table but not yet implemented.

SVE2's `whilelo` + predicated body folds the tail into the main loop, which is the cleanest model and a reason to keep an eye on Cobalt 100 / Graviton 4 / Apple's eventual SVE story. until then, 32B contiguous + scalar tail is the pragmatic choice.

## 6. f16, partial Mojo support, and the libm hole

Apple Silicon (Armv8.2-A) added native f16 NEON: `fmla.8h` is 8-lane f16 FMA at full throughput. so pure-arithmetic f16 kernels run at 16 lanes per 32B step and beat f32 _per byte of bandwidth_. for monpy's `add/mul/sub/fma/abs/neg` etc., we want f16 to use 16-lane SIMD.

the hole is libm. Apple's libm exposes `atan2f`, `hypotf`, `copysignf` for f32 but no f16 variants — there is no `atan2f16`. Mojo's standard library can't lower these symbols on macOS and emits a linker error when we try. the practical implication for `apply_binary_typed_vec[DType.float16, 16]`: we can't dispatch transcendentals.

monpy gates this with a `comptime if dtype != DType.float16` guard around the affected ops. the fallback is "promote f16 → f32, do op, demote back". two implementation choices:

**in-loop scalar promote-demote.** each element: load f16, cast to f32, call libm f32 transcendental, cast back, store. per-element cost ≈ 2 casts + 1 scalar libm call (scalar `atan2f` is ~30-50c on Apple Silicon depending on operand). throughput ~1 element per ~50c = abysmal.

**block-wise vec promote-demote.** cast a full f16 vector to two f32 vectors (the f32 half of a 32B vec is 8 lanes; cast to 16 f16 lanes splits into 2 × 8 f32 vecs), run the f32 vector kernel, demote both halves to one f16 vec, store. per-vec cost ≈ 2 cast ops + 2 f32 kernel ops + 1 demote. throughput close to f32 throughput, divided by 2 (since one f16 vec covers the same byte budget as two f32 vecs).

the second is what monpy should do — and the mechanism we already use for the gated path is structured to allow it (call `apply_binary_typed_vec[float32, 8]` twice on a buffer of cast values, then demote). the cast cost is modest: `fcvtl2 v0.4s, v1.8h` is 1c throughput per Q-register on Firestorm. overall, gated-via-f32 f16 transcendentals run at roughly 70-80% of native f32 transcendental throughput, which is correct given the cast overhead.

## 7. branchless SIMD for piecewise functions

`sign(x)`, `clip`, `abs` and friends are nominally piecewise. the dispatch options:

**branchy.** lane-wise scalar branch — destroys SIMD because branches don't vectorise.

**mask-and-blend.** compute both branches as full SIMD vectors, then select with `bsl` (NEON) / `vblendvps` (AVX2) / `vmovaps zmm{k}` (AVX-512). cost: $2 \cdot \text{op cost} + 1 \cdot \text{select cost}$. the select itself is 1c throughput on every modern ISA.

**arithmetic identity.** encode the conditional as arithmetic. examples that monpy uses:

- `sign(x) = (x > 0) - (x < 0)` where comparisons return 0/-1 ints and we negate to get 0/1. SIMD-clean: 2 compares + 1 sub. no branches, no blend. ~3 cycles dependency chain.
- `abs(x)` for floats: `x & 0x7FFFFFFF` on the bit pattern (bitwise AND with sign-bit-cleared mask). 1 bitwise op. for ints: `(x ^ (x >> 31)) - (x >> 31)` — 3 ops.
- `clip(x, lo, hi) = max(min(x, hi), lo)`. two SIMD ops, no branches. note the NaN propagation depends on which `min`/`max` semantics — see §8.
- `copysign(x, y)`: clear sign of `x` (AND), extract sign of `y` (AND with sign-mask), OR them. 3 bitwise ops.

the arithmetic-identity approach beats mask-and-blend for short branches because it skips the blend port and avoids the redundant computation of the unused branch. for long branches (e.g. a Taylor series that diverges) you have to take the mask-and-blend hit because re-computing both is more expensive than blending. monpy uses arithmetic identities everywhere it can, falls through to mask-and-blend for ops like `where`, and never branches.

a subtlety: Mojo's autovectoriser sometimes fails to generate the branchless form when given naive `if`/`else` Mojo code with a SIMD comparison. the safe move is to write the bitwise/arithmetic form explicitly using `select` or `bitcast` — this is what `src/elementwise.mojo` does for `sign` etc.

## 8. NaN, infinity, and the min/max minefield

IEEE 754-2008 + the C99/C++ amendments specify:

- NaN propagates through arithmetic.
- `==`, `<`, `>` involving NaN return `false`.
- `min`/`max` of (NaN, x) is implementation-defined.

NumPy's resolution:

| op                      | semantics                                               |
| ----------------------- | ------------------------------------------------------- |
| `np.minimum(a, b)`      | NaN-propagating: NaN if either operand is NaN           |
| `np.fmin(a, b)`         | NaN-quieting: returns the non-NaN if exactly one is NaN |
| `np.min(a)` (reduction) | NaN-propagating                                         |
| `np.nanmin(a)`          | explicitly skips NaN                                    |

monpy must implement both families. NEON has both flavours: `fminnm.4s` (NaN-quieting, ARM "number" variant — propagates only if both are NaN) and `fmin.4s` (NaN-propagating in default rounding mode). AVX has `vminps` (NaN-propagating from second operand under specific conditions — actually the IEEE semantics here are subtle: `vminps a, b` returns `b` if any operand is NaN, which is _operand-order dependent_).

the right pattern for `np.minimum`:

1. compute `m = native_min(a, b)` (NEON `fmin`, AVX `vminps`).
2. compute `nan_mask = (a != a) | (b != b)`.
3. result = `select(nan_mask, NaN_constant, m)`.

this costs 2 extra ops per lane vs. raw `fmin`. monpy implements explicit NaN-mask handling in min/max kernels for the NumPy-compatible paths.

for `np.fmin`, the NEON `fminnm` instruction is exactly the right semantics — one instruction, no extra masks. AVX has no direct equivalent before AVX-512's `vrange` family, so we synthesise it: `select(a != a, b, select(b != b, a, native_min(a, b)))`. three extra ops, still cheap relative to memory traffic.

a practitioner detail not always documented: comparison of NaN with itself (`x != x`) is the canonical NaN test and _does_ SIMD-vectorise cleanly to `vcmpneqps` / `vfcmne` on every modern ISA. don't be tempted by `isnan` library calls in vector code; the bit-pattern test is faster and inlines.

## 9. reduction patterns and stability

reductions break the elementwise model because they introduce a loop-carried accumulator. variants:

**linear with horizontal-reduce.** maintain one SIMD accumulator (e.g. `acc = SIMD[f32, 8]`); each iteration `acc = acc + load(...)`. after the main loop, reduce the 8 lanes to one scalar via $\log_2 w$ pairwise reductions (`vaddv.4s` on NEON, `vextractf128 + vaddps` on AVX2). this is what monpy does.

**tree.** pair-wise sum along the array, halving each pass. $\log_2 N$ passes, $N/2$ ops each. same FLOPs total, $\log_2 N$ better numerical stability. used in pandas' Kahan path and in stable-summation libraries.

**pairwise (NumPy default for `np.sum` since 2014).** divide the array into blocks of ~128 elements, sum each block linearly, then sum the partial sums pairwise. stability $O(\log N \cdot \varepsilon)$ vs. linear's $O(N \varepsilon)$. the block size is chosen to fit in registers.

**Welford's online algorithm.** maintains running mean and $M_2$ (sum of squared deviations from running mean), updating one element at a time:

$$
\mu_n = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n}, \qquad M_2^{(n)} = M_2^{(n-1)} + (x_n - \mu_{n-1})(x_n - \mu_n).
$$

numerically stable and stream-amenable. the downside is the per-element division — about 10× more expensive than a multiply on most cores. for batched/SIMD Welford you maintain $w$ parallel running stats and combine them at the end via the standard parallel-Welford merge formula ([Chan/Golub/LeVeque, 1983](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm)). monpy currently uses a two-pass naive variance (mean then $\Sigma (x-\mu)^2$), which is stable enough for IEEE-canonical inputs but pays double the bandwidth.

for `np.sum` the linear path's relative error is bounded by $N \varepsilon$, which for f32 ($\varepsilon \approx 1.2 \times 10^{-7}$) and $N = 10^7$ gives $1.2 \times 10^0$ — i.e. you can lose all precision on adversarial inputs. NumPy's pairwise scheme bounds it at $\log_2 N \cdot \varepsilon \approx 23 \varepsilon \approx 3 \times 10^{-6}$, six orders of magnitude better. monpy needs the pairwise path for `np.sum` to match NumPy semantics on large arrays; the SIMD-friendly form is "linear within block of B, pairwise across blocks", which preserves vectorisation while bounding error.

## 10. codegen quality: where Mojo helps and where it doesn't

the good:

- `SIMD[T, w]` lowers to clean platform vector intrinsics. the IR generated for `a + b` on `SIMD[float32, 8]` on aarch64 is `fadd v0.4s, v1.4s, v2.4s` paired with a second `fadd` on the upper half — exactly what we want.
- `@parameter for` over a comptime range unrolls cleanly without leaving the scheduler with conditional control flow.
- static alignment hints via `simd_load[align=32]` propagate to the LLVM `align` attribute, generating `vld1.32` rather than the unaligned variant where applicable.

the mixed:

- LLVM's autovectoriser handles simple loops well but bails on complex patterns. the complex multiply

  $$
  (a + bi)(c + di) = (ac - bd) + (ad + bc)i
  $$

  requires a lane shuffle (interleave the real and imaginary parts) that LLVM doesn't recognise unless you write it with explicit `__builtin_shufflevector` (or Mojo's `SIMD.shuffle`). Mojo's autovectoriser inherits this limitation. monpy's complex paths are hand-written SIMD with explicit shuffle patterns.

- branchy code with SIMD comparisons sometimes generates per-lane scalar code where SIMD `select` would work. the fix is to write `select(mask, then_val, else_val)` explicitly rather than `if mask: then_val else else_val`.
- `f16` lowering on macOS hits the libm hole described in §6. workaround is the f32 cast path.

the not-yet:

- no analogue to NumPy's "ufunc loop" — a generic mechanism that takes a per-dtype scalar function and synthesises the SIMD loop. monpy hand-writes the SIMD bodies. this is fine for the ~50 ops in scope but doesn't scale.
- no SVE2 codegen path. when SVE2 hardware becomes meaningful (Cobalt 100, future Apple), monpy needs a vector-length-agnostic pass that emits `whilelo`-driven loops instead of fixed-width 32B.
- no alignment-aware allocator integration with monpy's array constructor — see §12.

## 11. concrete benchmark expectations on M3

given everything above, what should a 1M-element f32 binary add cost on M3?

- memory traffic: $3 \cdot 10^6 \cdot 4 = 12$ MB.
- sustained DRAM bandwidth, single-thread: ~80 GB/s.
- time = $12 / 80{,}000$ ms = **150 µs**.
- compute: $10^6$ FLOPs at 200 GFLOP/s = **5 µs**. 30:1 memory:compute ratio.
- bound: memory at 150 µs.

run with the same array in L2 (e.g. $N=2 \cdot 10^5$, 800 KB working set):

- L2 BW ~250 GB/s.
- time = $2.4 / 250{,}000$ ms = **9.6 µs**.
- compute: $10^6 \cdot 0.2 = 1$ µs.
- 10:1 ratio. still memory-bound, but the gap has tightened.

run in L1 ($N=20{,}000$, 80 KB — slightly above L1d's 128 KB / 4 = 32 KB usable per stream after 2 sources + 1 dest contention; in practice L1 hits dominate at $N \le 5{,}000$):

- L1 BW per port ~1 load/cycle × 32B × 3.5 GHz = 112 GB/s per port; with 3 ports actively servicing 2 loads + 1 store, sustained ~300 GB/s for elementwise.
- time = $0.06 / 300{,}000$ ms = **0.2 µs**.

compare to a strided f32 binary add (stride 2, elements still f32) of the same 1M elements:

- effective bandwidth halved (every other element loaded): real bytes touched ~doubles per useful element.
- monpy's current scalar fallback runs ~6× slower than vector → ~900 µs.
- the proposed row-flattening fix would bring this to ~300 µs (memory-bound on the strided traffic).

for matmul, monpy delegates to BLAS (Accelerate on macOS, MKL on x86), which gives vendor-tuned performance — typically 80-90% of theoretical peak FMA throughput on the relevant tile sizes. worth noting because it removes matmul from the SIMD discussion entirely: monpy's contribution is plumbing, not kernel work.

## 12. memory alignment, re-examined

the 32B vector strategy presumes 32B-aligned base addresses. what goes wrong if the base is, say, only 16B-aligned?

**aligned 32B AVX2 load (`vmovaps`):** 1 cycle per load on Skylake-class. misaligned input on `vmovaps` is a #GP fault — the unaligned variant is `vmovups` (1 cycle on aligned addresses, 1 cycle on aligned-but-explicitly-unaligned-instr, **5+ cycles on cache-line-crossing**).

**cache-line crossing.** a 32B load at offset 48 within a 64B cache line straddles two lines. on Skylake the load port re-issues internally and the cost is 4-6 extra cycles + uses two LFB entries, halving load throughput in throughput-bound kernels. on Apple Silicon the penalty is smaller (~1-2 cycles documented, undocumented officially) because the M-series LSU has wider internal datapaths. either way, it's a per-load multiplicative cost.

**probability of a split.** random 32B loads cross a 64B line with probability 32/64 = **50%**. aligned loads cross with probability 0%. the naïve allocator (e.g. `malloc` on glibc) gives 16B alignment by default; first 32B load is fine, but subsequent loads stride 32B and alternate between aligned and 16-off-line — half of them split.

**stride alignment.** even with a 32B-aligned base, a stride of 24B (3 floats — odd in any sane model but possible from broadcasting) means every load is misaligned. monpy detects this in the strided path; the contiguous path assumes stride = itemsize.

the fix is allocator integration. monpy's array constructor should request 32B (or 64B for cacheline-aligned) base addresses via `posix_memalign(64, ...)` or `aligned_alloc(64, ...)`. the cost is negligible — a few extra fragmentation bytes per allocation — and the win is roughly **15-25% on memory-bound elementwise kernels** because every other vector load avoids the split-line penalty. this is the single highest-leverage change still on the table.

a subtle case: views into a parent array may not preserve the parent's alignment. a `[1:]` slice of a 32B-aligned f32 array starts at a 4B offset → 4B-aligned only. the kernel has two options: (a) detect misalignment and use `vmovups` from the start, eating the split-load cost, (b) prologue: do up to 7 scalar elements until aligned, then run the aligned vector loop, then the scalar tail. option (b) is what high-performance libraries (Eigen, OpenBLAS) do. monpy currently picks (a) — fine for now, leaves a small percent on the floor for sliced arrays.

## 13. things that are not the bottleneck — and a closing list

what is _not_ worth optimising in monpy's elementwise kernels:

- the choice between `fmla` and `fmul + fadd` (rounding-mode debate aside, throughput is identical on all modern cores).
- SIMD width within the 16-64B band — all variants are bandwidth-limited.
- branch prediction in tight kernels — no branches.
- compiler intrinsic vs. inline assembly — Mojo's lowering is competitive.

what _is_ worth optimising:

1. **allocator alignment** to 32B/64B (§12). 15-25% on hot kernels.
2. **strided-but-inner-contig flattening** (§4). 3-5× on a common case.
3. **pairwise reduction** for `sum`/`mean` (§9). numerical correctness for $N > 10^5$.
4. **block-wise f32-promote f16 transcendentals** (§6). 5-10× over the scalar path.
5. **small-N masked tail** (§5). major win on $N < 64$ kernels currently dominated by tail scalar loop.
6. **cache blocking for fused chains** — anywhere we can fuse `a*b+c` rather than materialising `tmp = a*b` then `tmp+c`, we halve memory traffic.

the pattern across all six: alignment, traffic reduction, and edge cases. none are about peak SIMD throughput.

---

## References

1. Williams, Waterman, Patterson. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." _CACM_ 52(4), 2009. the intensity/bandwidth ridge model.
2. Hennessy, Patterson. _Computer Architecture: A Quantitative Approach_, 6th ed., 2017. chapters on SIMD, OoO execution, and memory hierarchy.
3. Arm. _ARMv8/v9 Architecture Reference Manual._ NEON and SVE2 chapters; instruction encodings and latency caveats.
4. Intel. _Software Developer's Manual, Volume 1._ vector extensions, AVX/AVX2/AVX-512.
5. Williams et al., [STREAM-on-Apple-Silicon measurements](https://arxiv.org/html/2502.05317v1), 2025.
6. Dougall Johnson, [Firestorm μarch tables](https://dougallj.github.io/applecpu/firestorm.html), unofficial but the most-cited public source on M-series instruction-level metrics.
7. Travis Downs, [Ice Lake AVX-512 Frequency](https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html), 2020.
8. Eclectic Light Co., [Evaluating M3 Pro CPU cores: NEON](https://eclecticlight.co/2023/12/05/evaluating-m3-pro-cpu-cores-4-vector-processing-in-neon/), Dec 2023.
9. Welford. "Note on a method for calculating corrected sums of squares and products." _Technometrics_ 4(3), 1962. online variance formula.
10. Goldberg. "What Every Computer Scientist Should Know About Floating-Point Arithmetic." _ACM Computing Surveys_ 23(1), 1991. the reference on FP accuracy.
11. Chan, Golub, LeVeque. "Algorithms for Computing the Sample Variance: Analysis and Recommendations." _American Statistician_ 37(3), 1983. parallel-Welford merge.
12. Apple, "Optimizing for Apple Silicon," WWDC sessions across 2020-2024. patchy documentation; Dougall's tables remain more useful for instruction-level work.
13. Zingaburga, [SVE2 critical look](https://gist.github.com/zingaburga/805669eb891c820bd220418ee3f0d6bd), practitioner notes on SVE2 vs NEON on shipping hardware.
