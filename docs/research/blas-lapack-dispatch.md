---
title: "BLAS/LAPACK dispatch in monpy"
date: 2026-05-07
---

# BLAS/LAPACK dispatch in monpy

_The wrappers in `src/accelerate.mojo` are thin by design: monpy hands every numerically heavy routine to a hand-tuned F77 ABI binary and confines itself to argument marshalling, workspace probing, and the unpacking of one inconvenient output format — the conjugate-pair compressed eigenvectors from real-input `geev`._

This field note documents where the speed comes from within `sgemm` / `dgemm`, per GIL and Grand Central Dispatch on Darwin interactions, and what we should probably change about workspace allocation.

## 1. Fortran-77 ABI

every BLAS/LAPACK routine monpy calls uses the LAPACK F77 calling convention.

**1.1 pass-by-reference for everything.**

Given that classical Fortran has no value semantics for procedure arguments, the standard requires an `INTENT(IN)` scalar to be addressable so that the callee can dereference it.

the compiler is free to emit a copy-in/copy-out around the call when the source operand isn't a simple variable, but the _callee_ always sees a pointer. so a call like

```text
SUBROUTINE DGEMM(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
```

becomes, in C-style declaration form,

```c
void dgemm_(const char *transa, const char *transb,
            const int *m, const int *n, const int *k,
            const double *alpha, const double *A, const int *lda,
            const double *B, const int *ldb,
            const double *beta, double *C, const int *ldc);
```

- the trailing underscore follows gfortran/OpenBLAS name-mangling convention
- `cblas_*` is a thin C-shim that hides this and reorders some scalars, but Accelerate exposes the underscored symbols natively as well, and monpy binds those directly.
- every `Int32` and `Float64` we pass therefore has to be **materialised** into stack-resident storage and addressed
  - in Mojo we do this with `var alpha: Float64 = ...; UnsafePointer.address_of(alpha)`.

**1.2 character flags as one-byte pointers.**

original F77 character arguments were dope-vectors: a string pointer plus a hidden trailing length argument appended at the end of the call. most modern BLAS/LAPACK ABIs (Accelerate, OpenBLAS, MKL, Reference BLAS built with `gfortran -fno-f2c -fno-second-underscore`) implement the "BLAS character argument" convention: the caller passes a pointer to a single byte, the routine inspects exactly the first byte, and the hidden length is silently ignored. this is why monpy's wrappers materialise `'N'`, `'T'`, `'C'`, `'U'`, `'L'`, `'V'` as `Int8` stack variables and pass `UnsafePointer.address_of(...)` rather than dealing with descriptor strings. the full mnemonic table monpy actually uses:

| flag  | meaning in BLAS Level-3            | meaning in LAPACK                                    |
| ----- | ---------------------------------- | ---------------------------------------------------- |
| `'N'` | no-op (`op(A) = A`)                | "do not compute the eigenvectors" in `*syev`/`*geev` |
| `'T'` | transpose (`op(A) = Aᵀ`)           | —                                                    |
| `'C'` | conjugate transpose (`op(A) = A*`) | —                                                    |
| `'U'` | —                                  | use upper triangle (`*potrf`, `*syev`)               |
| `'L'` | —                                  | use lower triangle                                   |
| `'V'` | —                                  | "compute the eigenvectors"                           |
| `'S'` | —                                  | "thin" SVD in `gesdd`                                |
| `'A'` | —                                  | "all" left/right singular vectors                    |

some implementations (notably OpenBLAS built with the Fortran-2003 `BIND(C)` interface) require the hidden length anyway. in practice we get away with not passing it because the routines never read past byte zero of the flag, but a platform with a stricter ABI would need an `Int64` length argument after each character pointer.[^1]

**1.3 leading dimension as a stride.** `LDA` exists because Fortran 2-D arrays are stored in column-major contiguous form, but a routine often wants to operate on a _sub-matrix_ of a larger allocation. the contract is: element $A_{ij}$ (zero-indexed in C, one-indexed in Fortran) lives at byte offset $\mathrm{sizeof}(\text{dtype}) \cdot (j \cdot \text{lda} + i)$ for an $M \times N$ submatrix embedded in an $\text{lda} \times \text{ncols}$ allocation, with the requirement $\text{lda} \geq M$. equivalently, `lda` is the stride between successive _columns_.

NumPy and monpy are row-major by default. the standard workaround is the GEMM transposition identity:

$$
C = \alpha A B + \beta C \;\Longleftrightarrow\; C^\top = \alpha B^\top A^\top + \beta C^\top,
$$

which lets us pretend our row-major buffers are column-major versions of the transposed matrices. concretely, to compute `C += A @ B` on row-major $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$, $C \in \mathbb{R}^{m \times n}$, monpy invokes

```
dgemm('N', 'N', n, m, k, 1.0, B, n, A, k, 1.0, C, n)
```

i.e. it swaps $A \leftrightarrow B$ and `M \leftrightarrow N`, sets the leading dimensions to the row strides, and the routine internally produces the transpose-of-the-transpose. the "transposition trick" is a direct consequence of the column-major-vs-row-major isomorphism, not a hack.

**1.4 the `INFO` argument.** always the last argument, always written by the routine, never read. the semantics are universal across LAPACK:

- `INFO = 0`: success.
- `INFO < 0`: argument number `-INFO` (1-indexed) had an illegal value. a programmer error — usually a wrong leading dimension or a malformed character flag — and the routine _did not run_.
- `INFO > 0`: numerical failure, with per-routine semantics:
  - `*potrf`: `INFO = k` means the leading principal minor of order $k$ is not positive definite, so $A \not\succeq 0$ and the Cholesky factor doesn't exist.
  - `*gesv`/`*getrf`: `INFO = k` means $U_{kk} = 0$ exactly, the matrix is singular, and the LU factor is unusable for back-substitution (though it is a valid factorisation in the $PA = LU$ sense).
  - `*geev`/`*gesdd`: `INFO = k` means the QR/divide-and-conquer iteration failed to converge; eigenvalues `WR[INFO+1:n], WI[INFO+1:n]` (or singular values `S[INFO+1:n]`) are valid but those at indices `1:INFO` are not.
  - `*syev`: `INFO = k` means $k$ off-diagonal elements of the intermediate tridiagonal failed to converge. in practice, with double precision and the modern divide-and-conquer driver, this almost never happens.

`INFO < 0` should be a panic in monpy because it indicates a wrapper bug. `INFO > 0` should be raised as a `LinAlgError` with a routine-specific message. today `python/monpy/linalg.py` does the right thing in most cases but the `geev` non-convergence path is silent.

## 2. the workspace query pattern

every nontrivial LAPACK routine takes a `WORK` array and a length `LWORK`. the minimum legal `LWORK` (e.g. $\max(1, 3n - 1)$ for `dsyev`) is documented in the header, but the _optimal_ `LWORK` depends on the block size returned by `ILAENV`, which in turn probes the cache hierarchy at link time on Accelerate, and at runtime on OpenBLAS via environment variables and `cpuid`. the optimum is generally $n \cdot \text{nb}$ for some block size $\text{nb} \in [32, 256]$.

the two-call workspace query convention:

```
# query
work = [0.0]
lwork = -1
dgeqrf_(m, n, A, lda, tau, work, lwork, info)   # writes optimal lwork into work[0]
lwork_opt = int(work[0])

# allocate and run
work = alloc(lwork_opt)
lwork = lwork_opt
dgeqrf_(m, n, A, lda, tau, work, lwork, info)
```

three subtleties.

**2.1 `WORK[0]` is `Float64`, not `Int32`.** this is a Fortran legacy choice: rather than use a separate output port, the workspace itself returns its own optimum. the caller has to truncate to integer carefully — for very large $n$, the optimum can exceed $2^{31}$, in which case the `INTEGER*4` interface is broken and we need `*64_` ILP64 builds. Accelerate exposes ILP64 via the `$NEWLAPACK$ILP64` symbol suffix on macOS 13.3+; OpenBLAS via `OPENBLAS_USE64BITINT=1`. monpy currently uses LP64 throughout, which caps us at $n \lesssim 46\,000$ for `dgesdd`. we should plumb ILP64 detection through `accelerate.mojo` before this becomes a real ceiling.

**2.2 optimum vs minimum.** LAPACK accepts any $\text{LWORK} \geq \text{LWORK}_{\min}$ and adapts: with less than the optimum it falls back to an unblocked variant that does the same FLOPs but with lower arithmetic intensity. the ratio is typically 1.3–2× slower at the minimum than at the optimum on a modern machine. doing the query is essentially free (one nanosecond of overhead) compared to the $O(n^3)$ payload.

**2.3 `gelsd` queries two scratch arrays.** the integer workspace `IWORK` is sized via the documented closed-form $\text{liwork} = 3 \cdot \min(m,n) \cdot \mathrm{nlvl} + 11 \cdot \min(m,n)$ where $\mathrm{nlvl} = \max(0, \lceil \log_2(\min(m,n)/(\text{smlsiz}+1)) \rceil + 1)$ and `smlsiz` is typically 25. the query path also writes the optimal real workspace to `WORK[0]` and the optimal integer workspace to `IWORK[0]`, so we should read both even though only `WORK[0]` is documented prominently.

## 3. the Goto–van de Geijn microkernel

the reason `sgemm` is fast — and the reason monpy never tries to beat Accelerate at large $m,n,k$ — is the five-loop blocking structure formalised by Goto and van de Geijn in 2008.[^goto] every modern dense GEMM (Accelerate, OpenBLAS, MKL, BLIS, Eigen, NVIDIA cuBLAS) is a variant on this skeleton.

**3.1 the five loops.** read these outside-in:

```
for jc in 0..n step nc:                  # L3-resident   (B panel)
  for pc in 0..k step kc:                # L3-resident
    pack B[pc:pc+kc, jc:jc+nc] -> Bp     # contiguous, panel-major
    for ic in 0..m step mc:              # L2-resident   (A block)
      pack A[ic:ic+mc, pc:pc+kc] -> Ap   # contiguous, panel-major
      for jr in 0..nc step nr:           # register
        for ir in 0..mc step mr:         # register
          microkernel(Ap, Bp, C, mr, nr, kc)
```

the block sizes $(m_c, n_c, k_c)$ are tuned to fit:

- the packed $A$-block $A_p \in \mathbb{R}^{m_c \times k_c}$ in L2,
- the packed $B$-panel $B_p \in \mathbb{R}^{k_c \times n_c}$ in L3 (or in main memory streamed predictably),
- the running $C$-tile $C_t \in \mathbb{R}^{m_r \times n_r}$ in registers.

on Apple M-series, BLIS-style settings are roughly $m_c = 144$, $k_c = 256$, $n_c = 4096$ for `dgemm`, with $m_r \times n_r = 8 \times 6$ or $4 \times 12$ depending on the NEON register file. Accelerate uses different (and undocumented) tile sizes targeting the AMX coprocessor; reverse-engineering work[^amx] suggests $32 \times 32$ AMX tiles with $k_r$-blocked outer products, which is why Accelerate wins decisively at $k \gtrsim 64$ — the AMX `MAC16` instruction emits a $16\!\times\!16$ outer product per cycle, about $8\times$ the throughput of the equivalent NEON FMA loop.

**3.2 packing.** why pack at all? the $A$-block as it lives in user memory is strided (some `lda` away from contiguous), so each microkernel iteration would issue cache-line-misaligned, gather-style loads. packing copies it once into a contiguous, cache-line-aligned, panel-major buffer — costing $O(m_c k_c)$ memory traffic per outer iteration. that block then participates in $n_c / n_r$ microkernel calls, each of which streams it into the FMA pipeline at peak bandwidth. packed-buffer reuse is the central trick, and it's why reaching 90%+ of peak FLOPs requires panel-major data layout that no naïve C kernel will produce.

the packed layout for $A_p$ looks like this (each $\square$ is $m_r$ contiguous floats, panels stack vertically, panels of $k_c$ columns are concatenated):

```
                  k_c columns
        +----+----+----+----+----+
panel 0 | □  | □  | □  | □  | □  |   m_r rows
        +----+----+----+----+----+
panel 1 | □  | □  | □  | □  | □  |   m_r rows
        +----+----+----+----+----+
   ...
        +----+----+----+----+----+
        |    Ap memory order    |
        +----+----+----+----+----+
              row-major within an m_r-stripe;
              stripe-major across m_c
```

**3.3 the microkernel as a rank-1 update.** inside the microkernel we accumulate

$$
C*t \mathrel{+}= \sum*{p=0}^{k_c - 1} a_p b_p^\top,
$$

where $a_p \in \mathbb{R}^{m_r}$ is the $p$-th column of the packed $A$-block (in registers across $m_r/V$ vector registers, where $V$ is the SIMD width — 4 for NEON `f64`, 16 for AVX-512 `f32`) and $b_p \in \mathbb{R}^{n_r}$ is the $p$-th row of the packed $B$-panel (broadcast-loaded one element at a time). each iteration of the inner $p$-loop emits $m_r n_r / V$ FMA instructions and performs $2 m_r n_r$ FLOPs; with NEON's 1-cycle FMA throughput per pipe and 4-pipe issue on M4[^applem4], the kernel hits ~$2 \cdot 8 \cdot 6 \cdot 4 = 384$ FLOPs/cycle at peak.

**3.4 arithmetic intensity argument.** GEMM moves $m k + k n + 2 m n$ words and performs $2 m n k$ FLOPs, giving an arithmetic intensity of

$$
\mathrm{AI} = \frac{2 m n k}{m k + k n + 2 m n}.
$$

for square $m = n = k = N$, $\mathrm{AI} = N/2$ words, growing linearly. above the roofline kink — about 32 FLOPs/word on Apple M4 with 100 GB/s LPDDR5X and ~3 TFLOPs FP32 — GEMM is compute-bound. below it (small $N$, or rectangular cases with one small dimension), it's bandwidth-bound and packing overhead becomes a real cost. this is why monpy maintains a native small-matrix path in `maybe_matmul_*`: for $N \lesssim 32$ the dispatch overhead and packing cost outweigh any benefit from the optimised kernel.

**3.5 complex GEMM.** the complex multiplication $(a + bi)(c + di) = (ac - bd) + (ad + bc)i$ requires 4 real multiplies and 2 real adds per complex multiply (the Karatsuba 3-multiply variant trades 1 multiply for 5 adds and is rarely worthwhile in vector kernels because the adds also cost cycles), so

$$
\mathrm{FLOPs}(\text{cgemm}) = 8 m n k.
$$

the interleaved memory layout `(re, im, re, im, …)` aligns naturally with `_Complex` on every C ABI we care about (x86-64 SysV, AArch64 AAPCS, Apple ARM64), so monpy can pass a `Float32 *` cast of its `complex64` buffer with no marshalling. `alpha` and `beta` are passed as 2-element float arrays (real then imaginary part); we materialise them as `var alpha_re_im: SIMD[DType.float32, 2]` and pass `UnsafePointer.address_of(alpha_re_im).bitcast[Float32]()`.

the `'C'` flag means $\mathrm{op}(A) = A^*$, the conjugate transpose, so `cgemm('C', 'N', ...)` computes $\alpha A^* B + \beta C$. the identity $A^{**} = A$ means we never need a separate "untranspose" pass, but we do need to remember that for real types `'C' == 'T'` and most reference implementations special-case this.

## 4. per-routine deep dives

### 4.1 `getrf` — LU with partial pivoting

the factorisation $PA = LU$ exists for any nonsingular $A$, with $P$ a permutation matrix (encoded by `IPIV`), $L$ unit lower-triangular, and $U$ upper-triangular. the blocked algorithm partitions $A$ into panels of width $n_b$ and alternates _panel factorisation_ (unblocked LU on the current $n_b$ columns, finding pivots and applying them) with _trailing matrix update_ (a `dgemm` of the panel against the rest of $A$). the unblocked panel costs $O(m n_b^2)$; the trailing update costs $O((m - n_b)(n - n_b) n_b)$ and is the whole point — it dominates for $n_b \ll n$ and runs at full GEMM throughput.

total cost: $\frac{2}{3} n^3 + O(n^2)$ for square $A$.

`IPIV[i]` is 1-indexed (Fortran convention): row $i$ was swapped with row `IPIV[i] - 1` (0-indexed) at step $i$. monpy converts to 0-indexed when building the Python-visible `pivots` array.

**stability.** backward stable in the sense that the computed $\hat L \hat U$ satisfies $\hat L \hat U = A + \Delta A$ with $\|\Delta A\| / \|A\| = O(\rho_n \, \varepsilon_{\text{mach}})$, where $\rho_n$ is the growth factor — the ratio of the largest entry of $U$ to the largest entry of $A$. Trefethen and Schreiber[^growth] showed that $\rho_n$ can in principle reach $2^{n-1}$ (Wilkinson's example), but in practice on random matrices grows like $n^{2/3}$ and is essentially never a problem. the pathological matrices form a measure-zero set; if you somehow construct one (the classical example is $A_{ij} = -1$ for $i > j$, $A_{ii} = 1$, $A_{in} = 1$), `getrf` will silently produce a numerically useless factor. this is why iterative refinement (`*gerfs`) exists.

### 4.2 `geqrf` + `orgqr` — Householder QR

the factorisation $A = QR$ with $Q$ orthogonal and $R$ upper-triangular is computed by applying $\min(m, n)$ Householder reflectors

$$
H_k = I - \tau_k v_k v_k^\*, \qquad v_k \in \mathbb{C}^{m - k + 1},
$$

each of which zeros the entries below the diagonal in column $k$ of the (transformed) matrix. the reflectors are stored in compact form: $v_k$'s normalisation is chosen so that $v_k(1) = 1$ implicitly, freeing the diagonal slot and letting the _strictly lower triangular_ part of $A$ overwrite-store $v_2, v_3, \ldots$ the scalars $\tau_k$ go in a separate `TAU` array of length $\min(m, n)$.

`geqrf` does the factorisation in this compact form and never materialises $Q$; `orgqr` reads the compact form back and constructs $Q$ explicitly column by column. the blocked variant uses the WY representation[^wy]:

**Proposition (WY representation).** given $k$ Householder reflectors $H_1, \ldots, H_k$ stored as $(v_i, \tau_i)$, there exist a unit-lower-trapezoidal $V \in \mathbb{R}^{m \times k}$ (column $i$ is $v_i$ padded with zeros above) and an upper-triangular $T \in \mathbb{R}^{k \times k}$ such that

$$
H_1 H_2 \cdots H_k = I - V T V^\*.
$$

$T$ is determined by the recursion $T_{ii} = \tau_i$, $T_{ij} = -\tau_i (V^* V)_{ij} \tau_j$ for $i < j$, $T_{ij} = 0$ for $i > j$.

the point of WY is that applying $I - V T V^*$ to the trailing matrix is two `gemm` calls — $W = T (V^* B)$, then $B \mathrel{-}= V W$ — which run at peak bandwidth, replacing $k$ rank-1 reflector updates that would not.

cost: $\frac{4}{3} n^3$ for the full square $Q$, $\frac{2}{3} n^3$ for the thin economy form. backward error $\|\hat Q \hat R - A\|/\|A\| = O(m \varepsilon_{\text{mach}})$, dimension-independent in the sense that there is no $\kappa(A)$ factor — Householder QR is unconditionally stable.

### 4.3 `potrf` — Cholesky

for $A \succ 0$ symmetric (or Hermitian), the factorisation $A = L L^\top$ exists uniquely with $L$ lower-triangular and $L_{ii} > 0$. the right-looking blocked variant computes block-column $L_{:,j:j+n_b}$ then updates the trailing submatrix as $A_{j+n_b:, j+n_b:} \mathrel{-}= L_{j+n_b:, j:j+n_b} L_{j+n_b:, j:j+n_b}^\top$ via `syrk` (symmetric rank-$k$ update, half the FLOPs of a full GEMM). cost $\frac{1}{3} n^3$, exactly half of `getrf` because the symmetric structure cuts the work in two.

`INFO > 0` returns the index $k$ such that the leading principal minor $A_{1:k,1:k}$ failed positive-definiteness (i.e. $A_{kk} - \sum_{j < k} L_{kj}^2 \leq 0$ when we tried to take the square root). this is monpy's primary positive-definiteness check; we surface it as `LinAlgError("Matrix is not positive definite")`.

backward stable: $\hat L \hat L^\top = A + \Delta A$ with $\|\Delta A\|_2 \leq c_n \varepsilon_{\text{mach}} \|A\|_2$, with $c_n$ growing only polynomially in $n$ and _no $\kappa(A)$ factor_ — much better than LU, and one reason solving normal equations $A^\top A x = A^\top b$ via Cholesky is sometimes acceptable despite the conditioning squaring.

### 4.4 `syev` — symmetric eigendecomposition

the two-stage algorithm:

1. **tridiagonalisation.** apply Householder reflectors from both sides to reduce $A$ to a symmetric tridiagonal $T$: $A = Q T Q^\top$, where $Q$ is the product of reflectors. cost $\frac{4}{3} n^3$, dominated by `syr2k` (symmetric rank-$2k$ update) trailing-matrix updates.
2. **tridiagonal eigensolve.** factor $T = U \Lambda U^\top$ using either:
   - QR iteration with implicit Wilkinson shifts: $O(n^2)$ per sweep, $O(n)$ sweeps, $O(n^3)$ total but with a small constant.
   - divide-and-conquer (`stedc`): $O(n^{2.5})$ in the typical case, $O(n^3)$ worst case, but with much faster constants; this is what Accelerate uses by default.
   - MRRR (`stemr`): $O(n^2)$, but numerically tricky and not always default.
3. **back-transformation.** compute $A$'s eigenvectors as $V = Q U$ via `gemm`. cost $2 n^3$.

total: $O(n^3)$, with the Householder tridiagonalisation and the back-transformation each costing more than the eigensolve itself for $n \gtrsim 100$.

forward error in the eigenvalues: $|\hat \lambda_k - \lambda_k| \leq c_n \varepsilon_{\text{mach}} \|A\|_2$. note this is $\|A\|$, _not_ $\|A\| \kappa(A)$ — symmetric eigenvalues are well-conditioned absolutely. eigenvectors, however, suffer when eigenvalues cluster: the perturbation of an eigenvector is bounded by $\|\Delta A\| / \mathrm{gap}(\lambda_k)$ where $\mathrm{gap}(\lambda_k) = \min_{j \neq k} |\lambda_k - \lambda_j|$.

### 4.5 `geev` — non-symmetric eigendecomposition (the interesting one)

the general $A \in \mathbb{C}^{n \times n}$ admits a Schur decomposition $A = Q T Q^*$ with $Q$ unitary and $T$ upper-triangular over $\mathbb{C}$. real-input `dgeev` produces a real Schur form: $A = Q T Q^\top$ with $Q$ orthogonal and $T$ _quasi-triangular_ — block upper-triangular with $1 \times 1$ blocks for real eigenvalues and $2 \times 2$ blocks for complex conjugate pairs.

the algorithm:

1. **hessenberg reduction.** $A = Q_1 H Q_1^\top$ via Householder reflectors, $H$ upper Hessenberg ($H_{ij} = 0$ for $i > j+1$). cost $\frac{10}{3} n^3$.
2. **QR iteration with shifts.** iterate $H \to RQ + \mu I$ where $H - \mu I = QR$ is a QR factorisation. the shifts $\mu$ are chosen as the eigenvalues of the trailing $2 \times 2$ block (Wilkinson double-shift for real arithmetic preserving real Schur form), which gives quadratic convergence near the true eigenvalues. cost $O(n^2)$ per iteration, $O(n)$ iterations on average, so $O(n^3)$ total with a constant factor of perhaps 25–35 — much larger than `syev`'s.
3. **back-transformation.** eigenvectors of $T$ via back-substitution ($O(n^3)$), then transform back via $Q_1$ ($O(n^3)$).

**the conjugate-pair compressed format.** real $A$ can have complex eigenvalues, but only in conjugate pairs: $\lambda_k, \lambda_{k+1} = a \pm bi$. the corresponding right eigenvectors are also complex conjugates: $v_k, v_{k+1} = u \pm i w$ with $u, w \in \mathbb{R}^n$. storing them as complex would double the output size for what is essentially redundant information; LAPACK's solution is to pack:

$$
\mathrm{WR}[k] = a, \quad \mathrm{WI}[k] = b, \quad \mathrm{WR}[k+1] = a, \quad \mathrm{WI}[k+1] = -b,
$$

and

$$
\mathrm{VR}[:, k] = u, \quad \mathrm{VR}[:, k+1] = w,
$$

with the convention that the eigenvalue with positive imaginary part comes first. the actual complex eigenvectors are reconstructed as

$$
v*k = u + i w, \qquad v*{k+1} = u - i w = \overline{v_k}.
$$

**Theorem (correctness of conjugate-pair unpacking).** let $A \in \mathbb{R}^{n \times n}$ have a complex eigenvalue pair $\lambda_\pm = a \pm bi$ with $b > 0$, and let $u, w \in \mathbb{R}^n$ satisfy $A u = a u - b w$ and $A w = b u + a w$ (equivalently, $u + iw$ is in the kernel of $A - \lambda_+ I$ over $\mathbb{C}$). then $v_+ = u + iw$ and $v_- = u - iw$ are right eigenvectors of $A$ with eigenvalues $\lambda_+$ and $\lambda_-$ respectively.

_Proof._ $A v_+ = A(u + iw) = Au + i \, Aw = (au - bw) + i(bu + aw) = (a + bi)(u + iw) = \lambda_+ v_+$. conjugate to obtain $A v_- = \lambda_- v_-$ (using $A$ real). $\blacksquare$

the unpacking algorithm in monpy's `eig` (in `python/monpy/linalg.py`) walks `WI` left-to-right and emits:

```
i = 0
while i < n:
    if WI[i] == 0:                    # real eigenvalue, real eigenvector
        eigvals[i]    = WR[i] + 0j
        eigvecs[:, i] = VR[:, i].astype(complex128)
        i += 1
    else:                             # complex conjugate pair
        assert WI[i] > 0 and WI[i+1] == -WI[i]
        a, b = WR[i], WI[i]
        u, w = VR[:, i], VR[:, i+1]
        eigvals[i]      = a + b*1j
        eigvals[i+1]    = a - b*1j
        eigvecs[:, i]   = u + 1j*w
        eigvecs[:, i+1] = u - 1j*w
        i += 2
```

two correctness traps. first, the sentinel `WI[i] != 0` test is exact-zero, not a tolerance — LAPACK guarantees that real eigenvalues land with `WI[i] == 0.0` _bitwise_, because they come out of $1 \times 1$ Schur blocks that are never re-aggregated. second, the assertion `WI[i] > 0 and WI[i+1] == -WI[i]` documents an ordering invariant we rely on: positive-imaginary first. if we ever switch to `geevx` (the expert driver with balancing) we should re-check this; the standard driver always satisfies it.

the complex driver `zgeev` uses none of this and returns `complex128` directly — simpler but allocates $2\times$ the output for real-valued spectra. monpy uses the real driver for real input as a memory and dispatch-cost win.

### 4.6 `gesdd` — divide-and-conquer SVD

the factorisation $A = U \Sigma V^\top$ with $\Sigma$ diagonal non-negative and $U, V$ orthogonal. `gesdd`:

1. **bidiagonalisation.** $A = U_1 B V_1^\top$ where $B$ is bidiagonal (nonzero on diagonal and superdiagonal). cost $\frac{4}{3}(2 m n^2 - \frac{2}{3} n^3)$ for $m \geq n$.
2. **divide-and-conquer on the bidiagonal SVD.** recursively split $B$ into halves coupled by a rank-1 update, solve the secular equation at the merge step. cost $O(n^2 \log n)$ with small constants — usually 5–10× faster than `gesvd`'s QR-based bidiagonal SVD.
3. **back-transformation.** $U = U_1 \tilde U$, $V = V_1 \tilde V$. cost $O(m n^2)$.

total: $O(m n^2 + n^3)$ for $m \geq n$. singular values are returned in `S[0:n]` sorted descending. the `JOBZ` flag controls how much of $U$ and $V^\top$ is computed:

| `JOBZ` | $U$ shape            | $V^\top$ shape                      | use                  |
| ------ | -------------------- | ----------------------------------- | -------------------- |
| `'A'`  | $m \times m$         | $n \times n$                        | full SVD             |
| `'S'`  | $m \times \min(m,n)$ | $\min(m,n) \times n$                | thin/economy SVD     |
| `'O'`  | overwrites $A$       | $n \times n$ (in $A$ for $m\geq n$) | in-place             |
| `'N'`  | not computed         | not computed                        | singular values only |

monpy uses `'S'` by default, matching `numpy.linalg.svd(full_matrices=False)`.

numerical caveat: the divide-and-conquer SVD has been observed to fail to converge on rare structured inputs, with `INFO > 0` indicating which superdiagonal bisection didn't bracket the singular value. the fallback is to call `gesvd`, which uses the slower but more conservative implicit-zero-shift QR algorithm. monpy doesn't currently implement that fallback — we just propagate the failure — and we should.

### 4.7 `gelsd` — SVD-based least squares

the minimum-norm least-squares solution to $\min_x \|Ax - b\|_2$ for $A \in \mathbb{R}^{m \times n}$ is $x = V \Sigma^+ U^\top b$, where $\Sigma^+$ is the Moore–Penrose pseudo-inverse — diagonal entries are $1/\sigma_i$ for $\sigma_i > \mathrm{rcond} \cdot \sigma_1$, zero otherwise. `gelsd`:

1. calls `gesdd` internally to get $U, \Sigma, V^\top$.
2. computes $c = U^\top b$ via `gemm`.
3. truncates: $\hat c_i = c_i / \sigma_i$ if $\sigma_i > \mathrm{rcond} \cdot \sigma_1$ else $\hat c_i = 0$. the threshold determines `RANK`.
4. computes $x = V \hat c$ via `gemm`.

total cost dominated by the SVD: $O(m n^2 + n^3)$ for full rank. for rank-deficient or short-fat matrices, the truncation gives the unique minimum-norm solution and the back-substitution skips the zeroed rows.

`RCOND` defaults to machine epsilon if set to a negative value (monpy passes `-1.0` to invoke this). the returned `RANK` is the number of singular values exceeding the threshold and is the most useful single output for diagnosing rank-deficient problems.

## 5. Accelerate vs OpenBLAS — runtime dispatch

**5.1 detection.** monpy decides at module init via `dlopen`:

```
if dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate", RTLD_LAZY):
    use accelerate
elif dlopen("libopenblas.so.0", RTLD_LAZY) or dlopen("libopenblas.dylib", RTLD_LAZY):
    use openblas
else:
    fall back to native panel-tile in maybe_matmul_*
```

the test for Accelerate is on Apple Silicon only (we check `uname` first); on Intel macOS we prefer OpenBLAS because the legacy Accelerate vecLib pre-Ventura is single-threaded for `dgemm`.

**5.2 threading models.** Accelerate dispatches work to a libdispatch (Grand Central Dispatch) global queue: each `dgemm` call splits into tiles internally, each tile is enqueued as a `dispatch_async` block, and the calling thread waits on a `dispatch_group`. the thread pool is shared with all other libdispatch consumers in the process. OpenBLAS uses either pthreads (built with `USE_OPENMP=0`) or OpenMP (the default `USE_OPENMP=1`); thread pool size is controlled by `OPENBLAS_NUM_THREADS` or `OMP_NUM_THREADS`.

**5.3 GIL and the deadlock.** the classical Python-extension idiom is to release the GIL around long C calls so other threads can make progress. with Accelerate this is _especially_ important because of a specific deadlock pattern: if thread $A$ holds the GIL and calls `dgemm`, and a libdispatch worker thread $B$ tries to acquire the GIL (e.g. because some Python callback was registered), $B$ blocks on the GIL while $A$ blocks on the dispatch group waiting for $B$. both threads are stuck. the standard mitigation is `Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS` (or pybind11's `py::gil_scoped_release`). in monpy's PyO3-style wrapper this is `Python::allow_threads(|| ...)` around the BLAS call.

the safety condition: the released-GIL block must not touch any Python object. concretely, by the time we release the GIL we have already extracted raw `UnsafePointer`-typed slabs from the `PythonObject` buffers and we don't reach back through the buffer protocol until the BLAS call returns. this is enforced by structure: `accelerate.mojo` only sees raw pointers and ints.

**5.4 performance.** on M-series, Accelerate beats OpenBLAS on `sgemm`/`dgemm` by 2–4× at $n \geq 512$ because it dispatches to AMX (M1–M3) or SME (M4) coprocessor instructions that NEON-only OpenBLAS doesn't see. Frosner's 2024 benchmarks[^frosner] show Accelerate at 1.8 TFLOPs `sgemm` on M1 vs 0.6 TFLOPs for OpenBLAS NEON — a 3× ratio — and Apple's HPC paper[^applem4] reports M4 hitting 2.9 TFLOPs `sgemm`. below $n \approx 256$, the gap narrows and OpenBLAS occasionally wins on tiny matrices because Accelerate's libdispatch overhead dominates. we should benchmark monpy's crossover and tune `MIN_BLAS_DIM` accordingly — currently it's set to 32, which is probably too low.

for `dgetrf`, `dgeqrf`, and `dpotrf`, the gap shrinks to 1.3–1.8× because the LAPACK driver is bound by panel-factorisation latency more than trailing-matrix throughput.

## 6. memory alignment proposal

workspace allocation in monpy today is per-call: `UnsafePointer.alloc[Float64](lwork)` immediately before the LAPACK call, free immediately after. this is correct but wasteful — for repeated solves of the same shape (the common case in iterative ML pipelines) we allocate, query, allocate again, and free, every iteration. the proposal:

**6.1 why 64-byte alignment matters.** Apple Silicon caches are 128-byte lines (M1–M4); AMX wants its operand pointers aligned to 64 bytes for the `LDX`/`LDY` register-file loads. NEON `LD1` is alignment-tolerant but penalised by ~10% on misaligned 16-byte loads, and AVX-512 (when we eventually port to x86-64 with OpenBLAS) requires 64-byte alignment for `vmovaps`. the cost of misalignment in the workspace specifically is bounded — workspace is used once per call and refilled — but the user-data buffers we hand to BLAS are streamed millions of times in inner loops, and forcing them to a 64-byte boundary at allocation time pays for itself the first time we run a large `gemm`.

**6.2 allocation API alignment guarantees.**

| API                                            | default alignment      | notes                                                                 |
| ---------------------------------------------- | ---------------------- | --------------------------------------------------------------------- |
| `malloc` (glibc)                               | 16 bytes (8 on 32-bit) | insufficient for AMX and AVX-512                                      |
| `posix_memalign(&ptr, 64, n)`                  | requested              | must `free`; works on Linux + Darwin                                  |
| `aligned_alloc(64, n)`                         | requested              | C11; `n` must be multiple of alignment on glibc but not on macOS libc |
| `_aligned_malloc(n, 64)`                       | requested              | Windows-only; `_aligned_free` to release                              |
| Mojo `UnsafePointer[T].alloc(n)`               | `alignof(T)`           | currently 8 bytes for `Float64` — _not enough for AMX_                |
| Mojo `UnsafePointer[T].alloc(n, alignment=64)` | 64 bytes               | what we should use for BLAS-bound buffers                             |

Mojo's `alloc(n, alignment=...)` calls `posix_memalign` on POSIX and `_aligned_malloc` on Windows under the hood; we should standardise on this for every buffer that's destined to be passed to BLAS or LAPACK.

**6.3 LDA padding.** when the row stride `lda` is a multiple of the cache line, every column of $A$ starts on a cache-line boundary, and successive columns hit different sets in the cache (reducing conflict misses). the classical advice is to pad `lda` to a multiple of the SIMD width but _not_ to a power of two: powers of two on a typical 8-way set-associative L1 cause associativity blowups (every column maps to the same set, evicting itself). the Goldilocks zone is "next multiple of 8 (for `f32`) or 4 (for `f64`) that is _not_ a power of two and is _not_ a multiple of the page size 4096". for monpy, the practical rule:

| dtype                 | minimum LDA | preferred padding                      |
| --------------------- | ----------- | -------------------------------------- |
| `f16`                 | 16          | next multiple of 16, avoid powers of 2 |
| `f32`                 | 8           | next multiple of 8, avoid powers of 2  |
| `f64`                 | 4           | next multiple of 4, avoid powers of 2  |
| `c64` (`complex64`)   | 4           | next multiple of 4                     |
| `c128` (`complex128`) | 2           | next multiple of 2                     |

this costs at most $\text{LDA}_{\text{padded}} - M$ wasted entries per column, i.e. $O(N)$ memory overhead — negligible against the $O(MN)$ cost of the matrix itself.

**6.4 concrete proposal: per-thread, per-routine workspace arenas.** cache an aligned LAPACK-workspace arena per thread, keyed on `(routine_tag, dtype, max_size_seen)`:

```
struct WorkspaceArena {
    var routine: WorkspaceRoutine     # GETRF, GEQRF, POTRF, SYEV, ...
    var dtype: DType                  # f32, f64, c64, c128
    var capacity_bytes: Int           # current allocated size
    var ptr: UnsafePointer[Byte]      # 64-byte aligned

    fn ensure(self: mut, needed_bytes: Int):
        if self.capacity_bytes >= needed_bytes:
            return
        if self.ptr:
            self.ptr.free()
        # round up to 4 KiB to amortise reallocations
        self.capacity_bytes = (needed_bytes + 4095) & ~4095
        self.ptr = UnsafePointer[Byte].alloc(self.capacity_bytes, alignment=64)
}
```

storage: a `ThreadLocal[Dict[(WorkspaceRoutine, DType), WorkspaceArena]]`. on each LAPACK call, the wrapper does the workspace query, asks the arena for `ensure(lwork * sizeof(dtype))`, and uses the arena pointer. the arena grows monotonically — if we ever needed an `n=10000` factorisation we keep the buffer around for the next time.

trade-offs:

- **memory.** worst case we hold one arena per `(routine, dtype, thread)` cell. with 7 routines, 4 dtypes, and 8 threads, that's 224 arenas. if each grows to 100 MB (pathological) we waste 22 GB — a real concern. mitigation: a global LRU cap on total arena bytes, evicting the least-recently-touched arena when exceeded. a 1 GB cap is plenty for ML workloads.
- **correctness.** `posix_memalign` returns memory with undefined contents; LAPACK doesn't read uninitialised workspace, but valgrind will flag it. fine in production, annoying for testing.
- **threading.** each thread's arena is private, so no locking. a single-threaded program with one arena per (routine, dtype) is the common case and trivially correct.
- **implementation cost.** ~150 lines of Mojo. the interface to `accelerate.mojo` changes from "wrapper allocates and frees" to "wrapper takes an arena parameter".

the expected speedup on a tight `solve(A, b)` loop with fixed-size $A$: amortising allocation should save ~5% on $n = 1024$ and 15–20% on $n = 64$ (where the BLAS call itself is shorter and allocation overhead is proportionally larger). on the dominant ML use case — repeated calls in a training loop with batched matmuls of the same shape — this is the single highest-leverage perf change available short of writing our own GEMM.

## References

[1] K. Goto and R. A. van de Geijn, "Anatomy of High-Performance Matrix Multiplication," _ACM Transactions on Mathematical Software_ 34(3), Article 12, May 2008.

[2] T. M. Smith, R. A. van de Geijn, M. Smelyanskiy, J. R. Hammond, F. G. Van Zee, "Anatomy of High-Performance Many-Threaded Matrix Multiplication," _Proc. IPDPS 2014_, IEEE, 2014.

[3] E. Anderson, Z. Bai, C. Bischof, et al., _LAPACK Users' Guide_, 3rd ed., SIAM, Philadelphia, 1999.

[4] J. W. Demmel, _Applied Numerical Linear Algebra_, SIAM, 1997.

[5] N. J. Higham, _Accuracy and Stability of Numerical Algorithms_, 2nd ed., SIAM, 2002.

[6] L. N. Trefethen and D. Bau III, _Numerical Linear Algebra_, SIAM, 1997.

[7] L. S. Blackford et al., "An Updated Set of Basic Linear Algebra Subprograms (BLAS)," _ACM Transactions on Mathematical Software_ 28(2), pp. 135–151, 2002 (BLAS Technical Forum standard).

[8] F. G. Van Zee and R. A. van de Geijn, "BLIS: A Framework for Rapidly Instantiating BLAS Functionality," _ACM TOMS_ 41(3), Article 14, June 2015.

[9] J. Zhou, _Performance Analysis of the Apple AMX Matrix Accelerator_, MIT SB Thesis, 2025. <https://commit.csail.mit.edu/papers/2025/Jonathan_Zhou_SB_Thesis.pdf>

[10] D. T. Popovici et al., "Apple vs. Oranges: Evaluating the Apple Silicon M-Series SoCs for HPC Performance and Efficiency," arXiv:2502.05317, 2025. <https://arxiv.org/abs/2502.05317>

[11] R. Schreiber and L. N. Trefethen, "Average-Case Stability of Gaussian Elimination," _SIAM J. Matrix Anal. Appl._ 11(3), pp. 335–360, 1990.

[12] C. Bischof and C. van Loan, "The WY Representation for Products of Householder Matrices," _SIAM J. Sci. Stat. Comput._ 8(1), pp. s2–s13, 1987.

[13] LAPACK source for `dgeev`: <https://www.netlib.org/lapack/explore-html/d9/d28/dgeev_8f_source.html>

[14] F. Frosner, "Comparing OpenBLAS and Accelerate on Apple Silicon for BLAS Routines," dev.to, 2024. <https://dev.to/frosnerd/comparing-openblas-and-accelerate-on-apple-silicon-for-blas-routines-2pb9>

[15] T. Zakharko, "Exploring the scalable matrix extension of the Apple M4 processor," 2024. <https://github.com/tzakharko/m4-sme-exploration>

[16] BLIS Kernels HOWTO. <https://github.com/flame/blis/blob/master/docs/KernelsHowTo.md>

---

[^1]: the hidden length bug bit NumPy in 2018 when LAPACK 3.8 added stricter Fortran-2008 character interfaces; the fix was to either use the `LAPACKE` C API (which inserts the lengths automatically) or to pass `1` as the trailing length. monpy has avoided this by binding directly to the underscored F77 symbols, which on Accelerate, OpenBLAS, and MKL all use the byte-pointer convention without the trailing length. if we ever link against a `gfortran -finit-character=...`-built reference LAPACK on Linux we will need to revisit.

[^goto]: reference [1]. the 2008 paper formalises the model, but the real gold mine is the BLISlab tutorial (FLAME working note 80) and the BLIS source itself, which lets you read the actual microkernel for any architecture. the 5-loop structure is so universal that _every_ high-performance GEMM in the wild — Eigen, libxsmm, Intel MKL, NVIDIA cuBLAS for tile sizes that fit in shared memory — is some variation on it, with different choices for $(m_c, n_c, k_c)$ and how packing and prefetching interleave.

[^amx]: references [9, 15]. the AMX/SME instruction set is technically undocumented (Apple has not published an architectural manual), but Zakharko's 2024 reverse-engineering and Zhou's 2025 thesis pin down the relevant micro-arch details: 8 X-registers and 8 Y-registers of 64 bytes each, 64 Z-registers of 64 bytes (so 4 KiB of accumulator state), and `MAC16` outer-product instructions that emit a $16 \times 16$ FMA per cycle in `f32`. this is what makes Accelerate's `sgemm` so much faster than NEON-only OpenBLAS on M-series.

[^applem4]: M4 added SME (Scalable Matrix Extension), the Arm-standardised cousin of Apple's proprietary AMX, with similar semantics but a public ISA. existing Accelerate binaries dispatch to AMX on M1–M3 and SME on M4 transparently; the same monpy wrapper picks up both.

[^growth]: reference [11]. Trefethen and Schreiber's empirical study on uniformly distributed random matrices showed average growth factor $\rho_n \sim n^{2/3}$, vastly better than the $2^{n-1}$ worst case. the takeaway: for matrices that come from "typical" applications, partial pivoting is fine. for adversarial or carefully engineered linear systems (some PDE discretisations, certain optimisation problems), use full pivoting (`*getrfp` — though LAPACK doesn't ship a stock blocked driver) or iterative refinement.

[^wy]: reference [12]. Bischof and van Loan's 1987 paper introduced WY, replacing $k$ rank-1 updates with two `gemm`s and turning blocked QR from a 2× speedup over unblocked into a 5–10× speedup at $n = 1000$. the compact-WY variant ($Q = I - V T V^*$ rather than $Q = I - V W^*$) due to Schreiber and van Loan (1989) further halved the storage for $T$.

[^frosner]: reference [14]. Frosner's benchmarks aren't peer-reviewed and the methodology is sloppy, but the gap is large enough (3–4× on `sgemm`) that the qualitative conclusion is robust. the MIT thesis [9] confirms it with much more rigorous instrumentation.
