---
title: "complex-number arithmetic kernels"
date: 2026-05-05
---

# complex-number arithmetic kernels

_complex arithmetic is where IEEE 754 stops being a friendly abstraction and starts billing for every assumption you forgot you made._

the four-line definition of $z_1 z_2$ becomes, in finite precision, a working theory of overflow, cancellation, signed zeros, and branch cuts — and getting those wrong silently corrupts eigenvalue solvers, FFT phases, and conformal maps that exist downstream of `np.exp`.

monpy's complex types are interleaved 2-component layouts: `complex64` packs $2\times f32$ in 8 bytes, `complex128` packs $2\times f64$ in 16 bytes. alignment follows numpy's convention: align to the float component (4B and 8B), _not_ the complex whole. this matches the C99 `_Complex` ABI and PEP 3118, which means a `complex64*` can be reinterpreted as a `float32*` of length $2N$, and the BLAS vendor's `cblas_cgemm` / `cblas_zgemm` accept our buffers without copy. this document explains the kernels that operate on those buffers, the proofs that justify them, and the alignment proposal that follows from cache-line economics.

## 1. multiplication and the FMA cost question

the algebraic definition is the schoolbook form

$$
(a + bi)(c + di) = (ac - bd) + (ad + bc)\,i
$$

four real multiplies, two real adds. for half a century numerical-analysis textbooks have offered Karatsuba's three-multiply trick as an alternative:

$$
k_1 = ac, \quad k_2 = bd, \quad k_3 = (a+b)(c+d), \quad \mathrm{re} = k_1 - k_2, \quad \mathrm{im} = k_3 - k_1 - k_2.
$$

three multiplies and five adds. on paper this trades one $\mu_{\text{op}}$ for three $\alpha_{\text{op}}$, which looks like a win whenever a multiply is more than three times the cost of an add. on a 1985 VAX it sometimes was. on Apple Silicon's NEON, on Intel AVX-512, on NVIDIA's tensor cores — it isn't, and hasn't been since the late 1990s. modern FPUs issue an FMA every cycle on every pipe; the multiplier and the adder are the _same circuit_, the latency is _identical_, the throughput is one fused mul-add per port per cycle. counting multiplies separately stopped reflecting the hardware around the time MMX shipped.

worse, Karatsuba's form fights FMA. the four-multiply form lets us write

$$
\mathrm{re} = \mathrm{fma}(a, c, -bd)
$$

with one rounding error in $\mathrm{re}$ rather than two — provided the compiler honors the FMA contract. Karatsuba's $k_3 - k_1 - k_2$ has three rounding sites and no FMA opportunity, so it is _both slower and less accurate_ on the contig kernel hot path. monpy's `complex_binary_contig_typed[dtype]` MUL branch consequently uses the schoolbook form, fused where the backend exposes FMA semantics.

the precision question is sharper than it looks. Kahan's compensated-product lemma: if hardware provides a fused multiply-add, then for any $a, b, c, d$ the expression

$$
\texttt{fma}(-b, d, \texttt{fma}(a, c, 0))
$$

computes $ac - bd$ with a single rounding in the final FMA — accurate to $\frac{1}{2}$ ulp of the true real part _whenever_ $|ac - bd|$ is not catastrophically smaller than $\max(|ac|, |bd|)$. without FMA, $a c$ rounds, $b d$ rounds, and the subtraction rounds, accumulating up to $\frac{3}{2}\varepsilon \max(|ac|, |bd|)$. in the cancellation regime where $ac \approx bd$ the relative error blows up: you can lose every significant digit of the real part in a single instruction, with no warning.[^1]

the kernel emits the schoolbook form because numpy semantics demand bit-level reproducibility in the non-cancellation case and graceful degradation in the cancellation case — exactly what FMA-fused schoolbook delivers. Karatsuba is a relic of an architectural era that ended.

## 2. division — a proof of Smith's overflow safety

the naive expression for $z_1 / z_2$ rationalizes the denominator:

$$
\frac{a + bi}{c + di} = \frac{(a + bi)(c - di)}{c^2 + d^2} = \frac{ac + bd}{c^2 + d^2} + \frac{bc - ad}{c^2 + d^2}\, i.
$$

the failure mode is $c^2 + d^2$. if $\max(|c|, |d|) \approx \Omega^{1/2}$, where $\Omega$ is the overflow threshold ($\approx 3.4 \times 10^{38}$ for `f32`, $\approx 1.8 \times 10^{308}$ for `f64`), then $c^2 + d^2$ overflows to $+\infty$ even when the true quotient is a perfectly modest number. the spurious infinity propagates: $1 / \infty = 0$, and the answer comes back zero or NaN.

[Smith 1962](https://dl.acm.org/doi/10.1145/367766.368183) sidesteps this by branching on which component of the denominator is larger and dividing through:

> if $|d| \le |c|$: let $r = d/c$, $s = c + d r$, return $(a + b r)/s + ((b - a r)/s)\,i$. otherwise: let $r = c/d$, $s = d + c r$, return $(a r + b)/s + ((b r - a)/s)\,i$.

monpy's DIV branch in `src/elementwise.mojo`'s `complex_binary_contig_typed[dtype]` is a vectorized translation: it walks the contig stride, dispatches the branch per-element, and uses a single load-store pair per output. the loop is otherwise unremarkable; the interesting content is the proof that this never overflows when the true quotient is representable.

**Theorem (Smith 1962, overflow safety).** let $a, b, c, d$ be finite IEEE 754 numbers with $\max(|c|, |d|) < \sqrt{\Omega}$ and suppose the true quotient $q = (a+bi)/(c+di)$ has both components representable as finite IEEE numbers. then Smith's algorithm computes $q$ without producing any intermediate value of magnitude $\ge \Omega$.

_Proof._ by symmetry handle the first branch, $|d| \le |c|$. then $r = d/c \in [-1, 1]$ and $|r| \le 1$. consider each intermediate:

- $r = d/c$: $|r| \le 1 < \Omega$.
- $d r$: $|d r| \le |d| \le \max(|c|, |d|) < \sqrt{\Omega} < \Omega$.
- $s = c + d r$: $|s| \le |c| + |d r| \le 2 \max(|c|, |d|) < 2\sqrt{\Omega} < \Omega$ (for IEEE doubles, $2\sqrt{\Omega} \approx 2.7 \times 10^{154}$, comfortably finite).
- $b r$: $|b r| \le |b|$, and $b$ is finite by hypothesis.
- $a + b r$: bounded by $|a| + |b|$, finite by hypothesis.
- $(a + b r)/s$: this is the real part of the quotient up to a relative-error term, and is finite by hypothesis on $q$.

the key step is $|s| \le 2\sqrt{\Omega}$. the input bound $\max(|c|,|d|) < \sqrt{\Omega}$ is necessary — if $|c| = \sqrt{\Omega}$ exactly, then $|s|$ can hit $2\sqrt{\Omega}$, and the next squaring would overflow. but we never square: Smith's whole point is that we _don't_ compute $c^2 + d^2$, only $c + d r$, which is bounded linearly in $\max(|c|, |d|)$. $\blacksquare$

the bound is tight. choose $a = b = 1$, $c = d = \Omega^{1/2}$. naive division computes $c^2 + d^2 = 2\Omega \to +\infty$. Smith computes $r = 1$, $s = c + d = 2\sqrt{\Omega}$, and returns $(1 + 1)/(2\sqrt{\Omega}) + 0\cdot i = \Omega^{-1/2}$ — small but finite, exactly the right answer.

### the underflow problem

[Stewart 1985](https://dl.acm.org/doi/10.1145/214408.214414) noticed that Smith's algorithm has the dual failure mode: spurious _underflow_ in the numerator. consider $a = b = c = 1$, $d = \mu$ for $\mu$ near the underflow threshold $\omega = 2^{-1022}$ in `f64`. Smith computes $r = \mu/1 = \mu$, $s = 1 + \mu \cdot \mu = 1 + \mu^2$, but $\mu^2$ underflows to $0$. so $s = 1$, and we get $a + b r = 1 + \mu \approx 1$, returning $1$. that's actually fine. the bad case is where Smith's $b r$ or $a r$ flushes to zero in a regime where its contribution to the _output_ mantissa is significant. Stewart's fix: when $|d/c|$ is small enough that $d \cdot d/c$ underflows, recompute the numerator as a single multiplication $b d / c$ — three rounding errors instead of two, but no flush-to-zero in the middle.

[Priest 2004](https://dl.acm.org/doi/10.1145/1039813.1039814) goes further. his "Efficient Scaling for Complex Division" observes that you can pre-scale $(c, d)$ by a power of two chosen from the binary exponent of $\max(|c|, |d|)$, such that:

1. no intermediate overflows unless one component of the _true quotient_ would itself overflow;
2. no intermediate underflows unless one component of the true quotient would itself underflow;
3. the scale factor costs only a handful of integer instructions and four total floating-point multiplications.

Priest's algorithm dominates Smith's on every modern superscalar — the integer pre-scale runs in parallel with the load, and the four multiplications fuse with the surrounding adds. monpy currently ships Smith with Stewart's underflow patch; migrating to Priest is on the followup list (the integer scale extraction is the only nontrivial part, and Mojo exposes `bitcast` and `Int64`, so it's a ~30 line patch).

**Lemma (Priest scaling, paraphrased).** there exists a power of two $2^k$ such that $\max(|c|, |d|) \cdot 2^k \in [1, 2)$, and applying Smith's algorithm to $(2^k a, 2^k b, 2^k c, 2^k d)$ produces the same quotient as the unscaled inputs, with all intermediates bounded in $[2^{-1022+k}, 2^{1024-k})$. for any inputs with finite quotient, $k$ can be chosen so that this interval is the whole representable range.

the proof idea is that scaling all four inputs by the same $2^k$ leaves the quotient invariant (the $2^k$'s cancel), but shrinks the magnitudes that appear in $c^2 + d^2$ — or in Smith's case, in $s = c + d r$ — to a comfortable range.

## 3. branch cuts, signed zeros, principal values

complex elementary functions are multivalued. $\sqrt{z}$ has two values, $\log z$ has countably many, $\arctan z$ has countably many. defining a deterministic computer function requires choosing a _principal value_ and cutting the plane along a curve where the function is allowed to jump — the _branch cut_.

the conventions monpy inherits from numpy (which inherits from C99 `<complex.h>`, which inherits from [Kahan 1987](https://people.freebsd.org/~das/kahan86branch.pdf)) are:

- $\log z$: cut along $(-\infty, 0]$. principal value $\log|z| + i\,\arg z$ with $\arg z \in (-\pi, \pi]$.
- $\sqrt{z}$: cut along $(-\infty, 0)$. principal value $|z|^{1/2} e^{i \arg z / 2}$ with $\Re \sqrt{z} \ge 0$.
- $\arctan z$: cuts along the imaginary axis outside $[-i, i]$, i.e. on $\{iy : |y| > 1\}$.
- $\mathrm{arctanh}\, z$: cuts along the real axis outside $[-1, 1]$.

the cuts are conventions — they could go anywhere — but Kahan's choice has a precise virtue: each cut sits on the _negative_ axis of some auxiliary quantity ($z$ itself for $\log$, $1 - z^2$ for $\arctan$, etc.), and IEEE 754's signed zero gives us a way to specify which side of the cut a boundary input belongs to.

**signed zero.** IEEE 754 distinguishes $+0$ and $-0$. they compare equal under `==`, but `1/(+0) = +∞` and `1/(-0) = -∞`. for $\arg$, this means $\arctan_2(+0, -1) = +\pi$ and $\arctan_2(-0, -1) = -\pi$ — the same point on the negative real axis, two different sides of the cut.

monpy's transcendental dispatcher in `src/create.mojo`, `apply_unary_complex_f64(re, im, op)`, gets this for free because `atan2` is delegated to the platform's libm, and every modern libm preserves the signed-zero contract for `atan2`. the table:

| operation        | input    | numpy            | monpy            |
| ---------------- | -------- | ---------------- | ---------------- |
| $\log(-1 + 0i)$  | $-1, +0$ | $0 + i\pi$       | $0 + i\pi$       |
| $\log(-1 - 0i)$  | $-1, -0$ | $0 - i\pi$       | $0 - i\pi$       |
| $\sqrt{-1 + 0i}$ | $-1, +0$ | $0 + i$          | $0 + i$          |
| $\sqrt{-1 - 0i}$ | $-1, -0$ | $0 - i$          | $0 - i$          |
| $\log(-0 + 0i)$  | $-0, +0$ | $-\infty + i\pi$ | $-\infty + i\pi$ |

the bottom row is subtle: $\log 0$ is mathematically $-\infty$, but the _imaginary_ part is the argument of $-0 + 0i$, which is $\pi$ because $-0$ has a sign. drop the signed-zero distinction and conformal-map applications break at the cut.

the practical consequence: `apply_unary_complex_f64` should never normalize a signed zero away. monpy's SQRT branch hand-codes the magnitude $|z| = \mathrm{hypot}(a, b)$, then computes the components

$$
\sqrt{a + bi} = \sqrt{\frac{|z| + a}{2}} + i\,\mathrm{sign}(b)\,\sqrt{\frac{|z| - a}{2}},
$$

where $\mathrm{sign}(b)$ uses the signed-zero-aware variant `copysign(1, b)` so that $\sqrt{-1 - 0i} = 0 - i$ comes through correctly. the `a < 0` and `b = 0` corner is handled by the $\mathrm{copysign}$ path: $|z| = |a|$, the first square root is $0$, the second is $\sqrt{|a|}$, and the sign of $b$ — even when $b = -0$ — chooses the lower half-plane.

## 4. Euler identities — derivation chain

every complex transcendental in monpy reduces to one or two real transcendentals via Euler. for each, i give the math identity, the monpy strategy, and the worst-case relative-error bound. the dispatcher is `apply_unary_complex_f64` returning `Tuple[Float64, Float64]`; for `complex64` we promote to `f64`, compute, then narrow — the cost is two extra rounding errors, but the alternative is to maintain a parallel set of `f32` polynomial approximations, which doubles the test surface for marginal speedup.[^2]

### 4a. exp

$$
e^{a + bi} = e^a (\cos b + i \sin b).
$$

implementation: `e_a = exp(a); s, c = sincos(b); return (e_a*c, e_a*s)`. one transcendental times one trig pair. error: each of `exp` and `sincos` is correctly rounded (or faithfully rounded) on platform libms, and the two multiplications add at most one ulp each. total: $\le 4 \varepsilon$ in the well-scaled regime; can blow up if $a > 700$ since $e^a$ overflows even when only one component of the result does. correct handling defers to the libm — `exp(710)` returns $+\infty$, and $+\infty \cdot \cos(b)$ propagates the infinity, which is what numpy does.

### 4b. log

$$
\log(a + bi) = \log|z| + i\,\arg z = \tfrac{1}{2}\log(a^2 + b^2) + i\,\mathrm{atan2}(b, a).
$$

the naive form computes $a^2 + b^2$ — same overflow trap as complex division. correct form: $\log\,\mathrm{hypot}(a, b)$, where $\mathrm{hypot}$ avoids overflow by scaling. monpy uses `log(hypot(a, b))`.

a second-order improvement: when $|a^2 + b^2 - 1| \ll 1$ (i.e. $z$ near the unit circle), $\log\,\mathrm{hypot}(a,b) = \tfrac{1}{2} \log(a^2+b^2)$ has catastrophic cancellation in the argument. better: $\tfrac{1}{2}\,\mathrm{log1p}(a^2 + b^2 - 1)$. monpy's LOG branch detects $|a|, |b| \in [0.5, 2]$ and switches to the `log1p` form there. for $z$ farther from the circle, `log(hypot)` is fine.

error: $\le 5\varepsilon$ in the standard regime, $\le 2\varepsilon$ in the unit-circle regime where `log1p` shines.

### 4c. sin, cos, tan

$$
\sin(a + bi) = \sin a \cosh b + i \cos a \sinh b,
$$

$$
\cos(a + bi) = \cos a \cosh b - i \sin a \sinh b.
$$

`sincos(a)` and `sinh, cosh` of `b` together give all four reals; then four multiplies and one negation produce both outputs. error: $\le 6\varepsilon$. note $\cosh b$ overflows for $|b| > \log(\Omega) \approx 710$ in `f64`, which propagates correctly to $\pm\infty$.

tangent is the trap. the naive approach is $\tan z = \sin z / \cos z$, dividing two complex numbers — that's a Smith division on top of two trig evaluations. the elegant identity:

$$
\tan(a + bi) = \frac{\sin(2a) + i\,\sinh(2b)}{\cos(2a) + \cosh(2b)}.
$$

denominator is _real_, so we get one real division per component instead of a complex division. the denominator $\cos(2a) + \cosh(2b)$ is bounded below by $\cosh(2b) - 1 \ge 0$, equality only at $b = 0, \cos(2a) = -1$ — which is exactly where $\tan$ has a pole, so the infinity is correctly produced. error: $\le 7\varepsilon$ away from poles, controlled blow-up at poles.

### 4d. sinh, cosh, tanh

mirror identities via $\sinh z = -i \sin(iz) = \sin(b) \cos(a)\cdot 0 + \dots$. concretely:

$$
\sinh(a + bi) = \sinh a \cos b + i \cosh a \sin b,
$$

$$
\cosh(a + bi) = \cosh a \cos b + i \sinh a \sin b,
$$

$$
\tanh(a + bi) = \frac{\sinh(2a) + i\,\sin(2b)}{\cosh(2a) + \cos(2b)}.
$$

same pattern as the trig set, same error bound, same pole structure (now along the imaginary axis at $b = \pi/2 + k\pi$).

### 4e. sqrt

$$
\sqrt{a + bi} = \sqrt{\frac{|z| + a}{2}} + i\,\mathrm{sign}(b)\sqrt{\frac{|z| - a}{2}}, \qquad |z| = \mathrm{hypot}(a, b).
$$

avoid the obvious $\exp(\frac{1}{2} \log z)$ form — it has two transcendentals where two square roots suffice. the cancellation case is $a < 0$ and $b$ small: $|z| - a \approx 2|a|$ is fine, but $|z| + a$ is the difference of nearly-equal positives. resolution: when $a < 0$, swap roles via the identity

$$
\sqrt{a + bi} = \frac{|b|}{2 v} + i\,\mathrm{sign}(b) v, \qquad v = \sqrt{\frac{|z| - a}{2}},
$$

which trades the cancellation in $|z| + a$ for a division by $v$ — well-conditioned because $v$ is bounded away from zero whenever $z \ne 0$. the branch-cut continuity at $b = \pm 0$ is preserved by `copysign`.

### 4f. cbrt

real `cbrt(x) = sign(x) |x|^{1/3}` is built into libm. complex cbrt has no direct identity that beats the general-power form

$$
\sqrt[3]{z} = \exp\!\left(\tfrac{1}{3} \log z\right),
$$

so monpy uses that. four transcendentals (one log, one exp, one sincos, one cbrt of magnitude) — the most expensive complex transcendental in the set. error: $\le 10\varepsilon$.

an alternative would be to compute $|z|^{1/3}$ via `cbrt(hypot(a,b))` and the angle via `atan2(b,a)/3`, then assemble — same cost, slightly better numerics in the magnitude. monpy currently uses the `exp(log/3)` form; switching to the polar form is an open improvement.

### 4g. log1p, expm1

$\log(1 + z) = \mathrm{log1p}(z)$ for real `z`, exists to preserve precision when $|z| \ll 1$. for complex: when $|z|$ is small,

$$
\log(1 + z) = \log\,|1 + z| + i\,\arg(1 + z) = \tfrac{1}{2}\,\mathrm{log1p}(2a + a^2 + b^2) + i\,\mathrm{atan2}(b, 1 + a).
$$

the `log1p` argument is $2a + a^2 + b^2$, which we compute by FMA — `fma(a, a, fma(b, b, 2a))` — and feed to the real `log1p` to recover the precision that the naive $\log\,\mathrm{hypot}$ form loses for $z$ near zero. similarly for `expm1`:

$$
e^z - 1 = e^a(\cos b + i \sin b) - 1 = (e^a \cos b - 1) + i e^a \sin b.
$$

the real part has cancellation when $a \approx 0$ and $\cos b \approx 1$. resolution: use the identity

$$
e^a \cos b - 1 = \mathrm{expm1}(a)\cos b - 2 \sin^2(b/2),
$$

which keeps both terms small when $z$ is small. monpy implements this trick in the EXPM1 branch of `apply_unary_complex_f64`.

## 5. FMA error analysis, briefly

a single fused multiply-add computes $\mathrm{round}(a \cdot b + c)$ with one rounding. error bound: $|\mathrm{fma}(a,b,c) - (ab + c)| \le \tfrac{1}{2} \mathrm{ulp}(ab + c)$. without FMA, computing `a*b + c` issues two roundings: $\mathrm{round}(a \cdot b)$ then $\mathrm{round}(\mathrm{round}(ab) + c)$, accumulating error up to $\tfrac{3}{2}\varepsilon |ab|$ in the worst case (a tighter analysis gives $(1 + \tfrac{1}{2}\varepsilon)^2 - 1 \approx \varepsilon$ for the multiply contribution plus $\tfrac{1}{2}\varepsilon$ for the add).

for the complex product $ac - bd$ in the cancellation regime $ac \approx bd$, the relative error of the result is $\varepsilon$ multiplied by the _condition number_ $\max(|ac|, |bd|) / |ac - bd|$. this can be arbitrarily large. FMA reduces the input error by a factor of 3 (one rounding instead of three) — which translates to one ulp instead of three at the output, _given a fixed condition number_. it does not solve cancellation; only Kahan-style 2sum or a higher-precision intermediate solves cancellation. but it ensures that monpy's output matches numpy's to one ulp on the well-conditioned majority, with degradation matching libm's published bounds in the pathological tail.

[Kahan 1996](https://people.eecs.berkeley.edu/~wkahan/ieee754status/IEEE754.PDF), the IEEE 754 status notes, gives the canonical proof of the half-ulp FMA bound and discusses the implementation requirement: a single rounding mode applied to the infinitely-precise product-plus-add. ARM v8 and x86-FMA both honor this contract; the x87 stack does not, which is one reason monpy targets SSE2+ baselines exclusively.

## 6. memory layout — interleaved versus split

two storage conventions exist for an array of $N$ complex numbers:

**interleaved**: `[re_0, im_0, re_1, im_1, ..., re_{N-1}, im_{N-1}]`, $2N$ contiguous reals. used by C99 `_Complex`, Fortran `complex(8)`, BLAS, FFTW, NumPy, and PEP 3118. monpy follows this.

**split (planar)**: `[re_0, ..., re_{N-1} | im_0, ..., im_{N-1}]`, two separate $N$-element arrays. used by CUTLASS for some GPU kernels, by older Cray libraries, and by Intel MKL's "complex packed" optional layout.

the case for interleaved:

- **cache locality on elementwise**: $(a + bi)\,\mathrm{op}\,(c + di)$ touches both components of each operand together. interleaved gives one cache miss per complex element; split gives two (one for `re`, one for `im`), and the two arrays may be far apart in virtual memory, blowing the TLB.
- **BLAS ABI compatibility**: every `cblas_*gemm` expects interleaved. split forces a layout conversion at every BLAS call, which is bandwidth-bound and dominates the matmul cost for small matrices.
- **trivial reinterpretation**: `complex_unary_preserve_contig_typed[dtype]` for NEGATE and CONJUGATE simply writes a sign-flip pattern over the underlying float array. NEGATE flips both signs; CONJUGATE flips only odd indices. both reduce to a SIMD XOR with a mask register. cost: one XOR per cache line.

the case for split:

- **SIMT lane mapping**: on GPU SIMT models where each lane holds one scalar, split lets a warp process 32 real components and 32 imaginary components in parallel without shuffles. interleaved would force a shuffle every cycle.
- **wider SIMD multiplies**: with split storage on a NEON `float32x4_t`, you load four reals into one register and four imaginaries into another, and the multiply is

$$
(\mathbf{a} + \mathbf{b}\,i)(\mathbf{c} + \mathbf{d}\,i) = (\mathbf{a}\odot\mathbf{c} - \mathbf{b}\odot\mathbf{d}) + (\mathbf{a}\odot\mathbf{d} + \mathbf{b}\odot\mathbf{c})\,i,
$$

four FMAs, no shuffles. with interleaved you need a shuffle to decouple re from im.

since monpy is CPU-first and BLAS-coupled, interleaved wins. but the SIMD multiply with interleaved storage is worth describing. consider a NEON `float32x4_t` register holding $[a_0, b_0, a_1, b_1]$ — two complex elements packed into one 128-bit register. to compute $(a + bi)(c + di)$ with another register $[c_0, d_0, c_1, d_1]$:

1. **broadcast or shuffle** the second register to $[c_0, c_0, c_1, c_1]$ (call it $\mathbf{c}_{\text{br}}$) and $[d_0, d_0, d_1, d_1]$ ($\mathbf{d}_{\text{br}}$).
2. **swap** the input register: $[a_0, b_0, a_1, b_1] \to [b_0, a_0, b_1, a_1]$ via `vrev64q_f32` (one cycle on Apple silicon).
3. compute $\mathbf{t}_1 = [a_0, b_0, a_1, b_1] \odot \mathbf{c}_{\text{br}} = [a_0 c_0, b_0 c_0, a_1 c_1, b_1 c_1]$.
4. compute $\mathbf{t}_2 = [b_0, a_0, b_1, a_1] \odot \mathbf{d}_{\text{br}} = [b_0 d_0, a_0 d_0, b_1 d_1, a_1 d_1]$.
5. apply sign mask $[+, -, +, -]$ to $\mathbf{t}_2$ and FMA with $\mathbf{t}_1$ — gives $[a_0 c_0 - b_0 d_0, b_0 c_0 + a_0 d_0, a_1 c_1 - b_1 d_1, b_1 c_1 + a_1 d_1]$, which is exactly $[\Re q_0, \Im q_0, \Re q_1, \Im q_1]$.

three FMAs (after fusion), one shuffle, one negate-mask, two complex elements per iteration. on Apple silicon's NEON, `vrev64q_f32` has 1-cycle latency and 1-cycle throughput on every vector pipe; the shuffle does not bottleneck the loop, which is dominated by the FMA chain. on x86 AVX2, `_mm256_permute_ps` plays the same role with similar timings.

monpy's vectorized complex multiply is gated behind `@parameter if has_neon()` / `has_avx2()` and falls back to the scalar schoolbook form on baseline x86_64. the scalar path is what `complex_binary_contig_typed[dtype]` ships today; the vectorized variant is in the followup queue.

## 7. memory alignment — a proposal

monpy's current alignment policy:

- `complex64`: align array base to 4B (matches `f32` alignment, matches numpy).
- `complex128`: align array base to 8B (matches `f64` alignment, matches numpy).

this is correct for **scalar** semantics — every individual `f32` or `f64` load is aligned, and the complex element as a unit needs no stronger guarantee. the question is whether stronger alignment helps **SIMD**.

NEON and SSE2 require 4B alignment for any aligned load (`vld1q_f32`, `_mm_load_ps`); AVX requires 32B alignment for `_mm256_load_ps` aligned form, but the unaligned form `_mm256_loadu_ps` is one cycle slower at most on every microarchitecture since Sandy Bridge. AVX-512 prefers 64B alignment and pays a measurable penalty for unaligned loads on some Skylake-X and Ice Lake variants.

so the SIMD load itself is fine at 4B / 8B. the issue is **cache-line splits**. a 64-byte cache line on Apple silicon and on every modern x86 holds 16 `complex32` elements or 4 `complex64` elements. if the array base is 4B-aligned, any 16-element vector span (`f32x16` on AVX-512, `f32x4` on NEON × 4 unrolled) can straddle a cache-line boundary. straddled loads issue _two_ cache-line reads on x86 and _two_ L1 ports on M-series silicon — measurable but small for streaming kernels (5-10% bandwidth penalty), pathological for random access (up to 2x slowdown).

**proposal**: when allocating fresh complex arrays with $N$ elements such that $N \cdot \text{itemsize} > 64$ bytes, oversample the allocation to start on a 64B boundary. cost: at most 60 wasted bytes per array, paid once at allocation. benefit: every aligned 16-element span (and every BLAS column for matrices stored in column-major) starts on a cache line, eliminating split-line loads in the streaming path.

implementation: monpy's allocator (`StorageBuffer.allocate`) currently uses `UnsafePointer[Float64].alloc(n)` which delegates to the system `malloc`. `posix_memalign(&ptr, 64, n * itemsize)` gives the cache-line guarantee. for `complex64` arrays smaller than one cache line, retain the 4B fast path — the alignment buys nothing.

a stronger proposal would be **page alignment** for arrays larger than 4096 bytes, to let the kernel's prefetcher work with whole-page granularity and enable `madvise(MADV_HUGEPAGE)`. for monpy this is overkill at present; the BLAS path already handles its own buffer alignment, and the elementwise kernels are bandwidth-bound by the L3 fill rate, not TLB misses. the right decision here is profiling-driven once the matmul / einsum surface lands.

cross-reference: `memory-alignment.md` covers the general allocator question — `posix_memalign` vs `aligned_alloc` vs jemalloc's tcache, the page-coloring tradeoff, and the cost of fragmentation under repeated allocate-free cycles. the complex-specific finding is that 4B / 8B is correct for correctness and adequate for performance below the cache-line threshold, and that 64B alignment is a free win above it.

## references

1. Smith, R. L. "Algorithm 116: Complex Division." _Communications of the ACM_ 5(8), 435, 1962. [doi:10.1145/367766.368183](https://dl.acm.org/doi/10.1145/367766.368183).
2. Stewart, G. W. "A Note on Complex Division." _ACM Transactions on Mathematical Software_ 11(3), 238-241, 1985. [doi:10.1145/214408.214414](https://dl.acm.org/doi/10.1145/214408.214414).
3. Priest, D. M. "Efficient Scaling for Complex Division." _ACM Transactions on Mathematical Software_ 30(4), 389-401, 2004. [doi:10.1145/1039813.1039814](https://dl.acm.org/doi/10.1145/1039813.1039814).
4. Kahan, W. "Branch Cuts for Complex Elementary Functions, or Much Ado About Nothing's Sign Bit." in _The State of the Art in Numerical Analysis_ (eds. Iserles & Powell), Clarendon Press, 165-211, 1987.
5. Kahan, W. "Lecture Notes on the Status of IEEE Standard 754 for Binary Floating-Point Arithmetic." UC Berkeley technical report, 1996.
6. Goldberg, D. "What Every Computer Scientist Should Know About Floating-Point Arithmetic." _ACM Computing Surveys_ 23(1), 5-48, 1991.
7. Higham, N. J. _Accuracy and Stability of Numerical Algorithms_, 2nd ed., SIAM, 2002. (chapters 3 and 27 cover complex arithmetic and elementary functions respectively.)
8. NumPy reference. "Data Type Objects (`dtype`)" and "Universal functions (`ufunc`)" sections. [numpy.org/doc](https://numpy.org/doc/stable/reference/arrays.dtypes.html).
9. ISO/IEC 9899:1999, Annex G (informative): IEC 60559-compatible complex arithmetic. (the C99 specification of `_Complex` arithmetic, including the `CMPLX` macro and signed-zero semantics.)

[^1]: Kahan's _compensated summation_ (Kahan 1965, "Further Remarks on Reducing Truncation Errors") is the precedent: keep a running correction term that captures the low-order bits lost in each accumulation, add it back next iteration. for $\sum x_i$ this turns a $O(n\varepsilon)$ error into $O(\varepsilon)$ — independent of $n$. the FMA-fused complex multiply is morally the same trick, with hardware doing the bookkeeping: the FMA captures the bits of $a \cdot b$ that would have been lost to a separate multiply-then-add, and folds them into the final sum in a single rounding.

[^2]: in principle one can write a `complex32` polynomial approximation for `exp` etc. that avoids the f64 promote-narrow round trip, but the libm authors have already done this work for the real transcendentals, and the f32 `exp` on every modern platform has a relative error $\le 1$ ulp. the only `f32` complex code that beats promote-narrow is one that fuses the entire complex computation into a single polynomial — Sleef does this for some functions, Sun's `fdlibm` does not. monpy's tradeoff: 2x throughput on the dispatch path is not worth the test-surface burden, so we eat the two extra roundings.
