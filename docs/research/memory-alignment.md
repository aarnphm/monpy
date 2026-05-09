---
title: "memory alignment for monpy"
date: 2026-05-07
---

# memory alignment for monpy

_the cache hierarchy is the true bottleneck, and alignment is the cheapest way to stop fighting it._

monpy currently leans on whatever Mojo's default allocator hands back (`alignof(T)` on a 64-bit target, so 8 B for `f64`, 4 B for `f32`), and our SIMD fast paths uniformly emit unaligned loads. that works correctly. it also leaves on the table somewhere between five and twenty percent on memory-bound elementwise kernels and a much larger margin on the awkward shapes — tall-thin matrices, complex arithmetic on AoS layouts, anything with row sizes that aren't a SIMD-multiple. the rest of this document is the gears: why the gap exists, how big it is on the hardware we actually run on, what allocator strategy we should adopt, and a staged rollout that doesn't blow up the codebase.

the audience is someone who has read Drepper end to end, knows what MESI is, and has spent a Tuesday afternoon staring at `perf stat`. we do not redefine cache coherence here. we do reproduce the actual cycle costs because the literature is full of folk wisdom that hasn't been true since Haswell.

---

## 1. the cache hierarchy you're actually targeting

monpy ships on two host families: x86_64 servers and Apple Silicon laptops. both are 64-byte cache-line, both are heavily out-of-order, and both will eat unaligned loads without faulting. they differ in everything else.

### 1.1 Apple Silicon (the dev machine)

the numbers below are for the M3 / M4 generation, which is what the in-tree benchmarks are calibrated against. sizes are per-core unless noted otherwise.

|          level |               M3 P-core               |        M3 E-core        |            M4 P-core             |        M4 E-core        |             line              | latency (typical) |
| -------------: | :-----------------------------------: | :---------------------: | :------------------------------: | :---------------------: | :---------------------------: | :---------------: |
|            L1d |                128 KB                 |          64 KB          |              128 KB              |          64 KB          | 128 B logical, 64 B coherence |       3 cyc       |
|            L1i |                192 KB                 |         128 KB          |              192 KB              |         128 KB          |             64 B              |        n/a        |
|             L2 |       16 MB shared / P-cluster        | 4 MB shared / E-cluster |     16 MB shared / P-cluster     | 4 MB shared / E-cluster |             128 B             |     14–18 cyc     |
|            SLC |   8 MB / 12 MB (Pro) / 48 MB (Max)    |          same           | 8 MB / 24 MB (Pro) / 48 MB (Max) |          same           |             128 B             |      ~80 cyc      |
| DRAM (LPDDR5X) | UMA, ~100 GB/s base, 200+ GB/s on Max |                         |                                  |                         |                               |    100–130 ns     |

a few points worth flagging because they trip people coming from x86:

1. **Apple's L1 line is 128 B externally but 64 B for coherence**. the data array is fetched in 128-byte sectors but the coherence granule (the unit of MESI traffic) remains 64 B. false-sharing analysis stays a 64-byte question; spatial-locality analysis becomes a 128-byte question.[^line]
2. **the SLC is not an LLC in the x86 sense**. it sits between the CPU and the unified memory controller and is shared with the GPU and Neural Engine. hits there don't behave like an Intel L3 hit; they behave like a fast DRAM. treat it as bandwidth amplification, not as a latency hider.
3. **there is no NUMA story on Apple Silicon**. unified memory is genuinely unified. you can stop thinking about node-affinity allocation entirely on this target.[^numa]

[^line]: this sectoring matters for prefetch tuning more than for monpy's elementwise loops. we mention it only because every "Apple has 128-byte cache lines" claim and every "no, it's 64" claim are simultaneously correct and people argue about it on Hacker News every six months.

[^numa]: the flip side: you cannot pin work to a memory controller. on the M3 Max with two memory dies, the SoC fabric handles routing; you have no API to influence it.

### 1.2 x86 servers (the deployment target)

Skylake-X / Ice Lake-SP / Sapphire Rapids / EPYC Genoa all share the basic shape: 32–48 KB L1d per core, 1–2 MB L2 per core, 32–256 MB shared L3. the line is 64 B. DRAM latency runs 80–110 ns; bandwidth is 25–50 GB/s per socket on consumer parts, 200–400 GB/s on aggressive server SKUs. AVX-512 server cores have a third execution port for vector loads, which is where the L1 throughput jump from "two loads / cycle" to "two loads + two stores / cycle on Ice Lake" comes from.[^downs]

[^downs]: see Travis Downs, ["Performance Speed Limits"](https://travisdowns.github.io/blog/2019/06/11/speed-limits.html), 2019, which walks through the actual port counts and the sneaky two-loads-or-three-loads ambiguity on different generations.

### 1.3 the cost of crossing a line

this is the load-bearing fact for everything below. a SIMD load that crosses a 64-byte cache-line boundary fetches _two_ lines, and the OoO engine accounts for it as two micro-ops against the load throughput limit. the cycle penalty has shrunk dramatically:

- **pre-Nehalem**: split-line loads stalled for tens of cycles.
- **Nehalem / Sandy Bridge / Ivy Bridge**: 4–6 extra cycles in the bad case, 2× throughput cost when the kernel is L1-bound.
- **Haswell onward (x86)**: aligned and unaligned variants of `MOV` are identical; the _only_ remaining cost is split-line, ~3–5 cycles on a hit, plus the throughput impact (split counts as 2).
- **page split** (a load crossing a 4 KB boundary): was 100 cycles on Broadwell, fell to ~5 cycles on Skylake.[^page-split]
- **Apple M-series**: micro-benchmarks show split-line loads cost 1–2 cycles extra and consume an additional load slot, so on a kernel that's already pegged on L1 throughput (two 128-bit loads / cycle on Firestorm and successors), splits halve effective bandwidth.

[^page-split]: this Broadwell→Skylake fix is the reason the literature is so confused. half the alignment advice on the internet predates it and is about a problem that no longer exists at full magnitude on a modern OoO chip.

the sharp form of the rule: **if the start address is 64-byte aligned and the load width is ≤ 64 bytes, the load cannot split a line, period.** no alignment check needed; the geometry forbids it. this is the property we are trying to manufacture for monpy's kernels.

for a 32 B AVX2 `vmovupd` starting at byte 48 of a line, the load spans bytes 48–79: two lines. for the same load starting at byte 0, 32, or any multiple of 32, you span one line. that's why even on architectures where unaligned-mode loads are "free", we still want the data to start on a SIMD-multiple boundary. the cost we eliminate is geometric, not microarchitectural.

---

## 2. SIMD alignment, ISA by ISA

### 2.1 NEON (ARMv8 / Apple Silicon)

128-bit registers (`float32x4_t`, `int8x16_t`, etc). the `vld1q_*` family of intrinsics has _no_ alignment requirement at the ISA level — ARMv8 dropped the architectural distinction between `LD1` and `LD1.aligned` that ARMv7 had. behaviorally:

- a 16 B `vld1q` starting on any 4 B boundary works.
- the clang front-end emits a 128-bit alignment hint when fed a `__builtin_assume_aligned(p, 16)` or when the type has `alignas(16)`. this hint flows into LLVM's machine model but on Apple Silicon does not change the issued instruction; it only enables LLVM's downstream optimizer to remove conservative scaffolding (mostly: combining adjacent loads into pair forms `LDP`).
- cache-line splits still exist and still cost an extra load-unit cycle.
- SVE / SVE2 (not yet on Apple Silicon, present on Neoverse and Graviton 3+) widens registers to 128–2048 bits and re-introduces an alignment requirement: a register-width load needs alignment to that width, otherwise you must use the unpredicated unaligned form, which on first-generation SVE costs ~2× throughput for misaligned. future-proofing argument for keeping a 64-byte alignment policy.

the Apple-Silicon-specific implication: aligned and unaligned `vld1q` are indistinguishable in the issued instruction stream. the only thing alignment buys us is geometric: avoiding line splits. that alone is worth ~5–10% on memory-bound code.

### 2.2 AVX2 (Intel Haswell+, AMD Zen+)

256-bit registers. two relevant load variants:

- `_mm256_load_pd(p)` → `VMOVAPD ymm, [p]`. faults if `p` is not 32-byte aligned.
- `_mm256_loadu_pd(p)` → `VMOVUPD ymm, [p]`. tolerates any 8-byte alignment.

pre-Haswell, `VMOVUPD` was 2–4× slower than `VMOVAPD` even when the address happened to be aligned. Haswell unified the two: at L1, both decode to the same internal micro-op when the address is aligned, and both pay the split-line penalty when it isn't. modern code uses `VMOVUPD` exclusively unless the alignment is statically known and the codegen wants to assert it (helpful for the compiler to elide subsequent runtime alignment checks in vectorized loops).

### 2.3 AVX-512 (Skylake-X, Ice Lake, Sapphire Rapids; AMD Zen 4+)

512-bit (64 B) registers. same dichotomy: `_mm512_load_pd` requires 64-byte alignment; `_mm512_loadu_pd` does not. the difference here is that **AVX-512 misalignment is much more painful** because the geometric chance of a split hits hard:

- a randomly-aligned 64 B load splits the 64 B cache line **>98% of the time**.
- a 32 B load splits 50% of the time.
- a 16 B load splits 25% of the time.

at 512-bit width, the question is not "does my load split occasionally" but "do I split _every iteration_ unless I align the base". the throughput cost on Skylake-X is roughly halved by uniform misalignment: you go from 1 load/cycle to 0.5. this is enormous and is the single best argument for monpy adopting an alignment policy at the high end.

### 2.4 Apple AMX (M1/M2/M3, undocumented public ISA)

the Accelerate framework's matmul kernels target a coprocessor that was reverse-engineered by Dougall Johnson and documented in [corsix/amx](https://github.com/corsix/amx). relevant facts for our alignment model:

- **single-register loads (`ldx`, `ldy`, `ldz`)** transfer 64 B and **do not require alignment**.
- **pair loads** transfer 128 B and **must be 128 B aligned**.
- the instruction encoding contains an alignment-hint flag; with the flag set, hardware assumes alignment and short-circuits some boundary checking.

we don't issue AMX directly today (Mojo doesn't expose it; we'd go through Accelerate via FFI), but the design constraint propagates: if we ever pass a buffer to `cblas_dgemm` and the underlying implementation uses pair loads, we want the buffer to be 128-byte aligned. 64 B is the floor for SIMD; 128 B is the ceiling we can usefully shoot for on Apple, and it costs nothing extra given the page geometry below.

### 2.5 what this means for monpy's typed-vec widths

our current `apply_binary_typed_vec[dtype, width]` selections always produce 32 B loads:

|   dtype | width | bytes |
| ------: | :---: | :---: |
|     f32 |   8   |  32   |
|     f64 |   4   |  32   |
|     f16 |  16   |  32   |
|    bf16 |  16   |  32   |
|   i8/u8 |  32   |  32   |
| i32/u32 |   8   |  32   |
| i64/u64 |   4   |  32   |

a 32 B load starting on a 32 B boundary cannot split a 64 B line: it spans either bytes 0–31 or bytes 32–63 of the line. for the _current_ SIMD width selection, **a 32 B alignment policy is sufficient to eliminate splits entirely**. a 64 B policy is forward-looking — it covers AVX-512 and any future doubling of NEON width. we pay essentially nothing for the stronger guarantee, so we take it.

---

## 3. layout policies: contiguous, strided, padded

NumPy's strided memory model has four layout regimes that matter for alignment.

### 3.1 C-contiguous (row-major)

`stride[i] = stride[i+1] * shape[i+1]`. for an `(M, N)` `f32`, the byte stride of the leading axis is `4·N`. the whole buffer is a single C array of `M·N` `f32`s. this is monpy's hot path; every elementwise kernel assumes it (or falls back to a strided iterator).

### 3.2 F-contiguous (column-major)

`stride[0] = itemsize`, `stride[i] = stride[i-1] * shape[i-1]`. comes from `np.asfortranarray`, BLAS `LAPACK_COL_MAJOR`, scientific Python codebases predating modern NumPy. same alignment story as C-contig but with axes swapped.

### 3.3 strided non-contiguous

slices, transposes, broadcast results. strides may not be a multiple of itemsize; they may be zero (broadcast) or negative (reverse). the SIMD path is generally not taken here — we either copy to a contiguous temporary or fall to a scalar walker — but when we _do_ take SIMD on a strided view, alignment of the base address still matters for the first load.

### 3.4 padded contiguous

the interesting case. the shape and strides disagree about contiguity: `stride[i-1] > stride[i] · shape[i]`. there is wasted space at the end of each row. NumPy users almost never construct these directly; they arise from manual layout choices in performance-critical code, from CUDA `cudaMalloc3D` (which pads pitch), and from BLAS Level-3 packed buffers.

the win: a row of 5 `f32`s is 20 bytes. stride that row to 32 bytes — pad with 12 bytes — and the row stride becomes a SIMD multiple. the next row starts at the same SIMD lane position as the first. per-row alignment shuffles disappear; the kernel can pretend each row is a full 8-element SIMD vec with a tail mask of `[1,1,1,1,1,0,0,0]`. the mask predicate costs less than the shuffle (one vector compare versus a full register shuffle plus a permute).

this is the trick GotoBLAS has used for decades on packed $\tilde{A}$ and $\tilde{B}$ panels. the Goto/Van Zee paper makes the case: packing into aligned panels pays for itself within the K-loop's first iteration, because the macro-kernel's inner triple-loop runs hundreds of times against the same panel.[^goto]

[^goto]: Goto & Van de Geijn, "Anatomy of High-Performance Matrix Multiplication", ACM TOMS 34(3), 2008. the packing strategy is the load-bearing performance win in Goto/OpenBLAS/BLIS, more than the micro-kernel itself.

---

## 4. AoS vs SoA for complex types

monpy stores `complex64` and `complex128` as Array-of-Structures: interleaved real and imaginary parts, `[re_0, im_0, re_1, im_1, …]`. the alternative is Structure-of-Arrays (sometimes called planar or split): `[re_0, re_1, …]` followed by `[im_0, im_1, …]`. each has a clean argument.

### 4.1 the AoS multiply, mechanically

for a NEON kernel multiplying complex64 × complex64 in AoS:

```
load a:  vec_a = [re_a0, im_a0, re_a1, im_a1]   ; vld1q_f32(a)
load b:  vec_b = [re_b0, im_b0, re_b1, im_b1]   ; vld1q_f32(b)
swap a's halves within pairs:
         vec_a' = [im_a0, re_a0, im_a1, re_a1]  ; vrev64q_f32 + lane permute
duplicate b's lanes:
         vec_br = [re_b0, re_b0, re_b1, re_b1]  ; vdupq_lane_f32 form (or vtrn)
         vec_bi = [im_b0, im_b0, im_b1, im_b1]
multiply:
         vec_out = vec_a * vec_br
         vec_out = fma(±vec_a', vec_bi, vec_out)  ; fmla / fmls per lane
```

six vector ops (2 loads, 1 reverse, 2 mul/fma, 1 store) for two complex multiplies. ~3 ops per complex.

### 4.2 the SoA multiply

```
load a_re, a_im: 2 loads of [re_a0..re_a3], [im_a0..im_a3]
load b_re, b_im: 2 loads
out_re = a_re*b_re - a_im*b_im   ; 1 mul + 1 fnma
out_im = a_re*b_im + a_im*b_re   ; 1 mul + 1 fma
store out_re, out_im
```

eight vector ops for four complex multiplies. ~2 ops per complex. SoA is roughly 30% denser in vector slots.

### 4.3 why monpy keeps AoS anyway

three reasons, and they compound:

1. **BLAS interop is free with AoS, expensive with SoA.** `cblas_zgemm` takes `complex double*` in interleaved layout. with SoA we'd repack on every BLAS call, eating any win on the elementwise path.
2. **elementwise dominates.** the fraction of monpy code that's complex multiply specifically — vs complex copy, complex add, complex conjugate, complex absolute value — is small. add and conj are _trivial_ on AoS (you just treat the buffer as `2N` reals) and require split loads on SoA.
3. **the shuffle is cheap on Apple Silicon**: `vrev64q_f32` is a single-cycle, dual-issue op on Firestorm and later. the ratio penalty above (3 ops vs 2) shrinks to maybe 5–8% on real kernels because the shuffle issues in parallel with the loads.

the estimated end-to-end cost of staying AoS, for the _complex multiply_ kernel specifically, is 5–10%. for everything else, AoS is at least neutral and often better. we pay 5–10% on a sliver of the API to keep BLAS interop and to keep `np.real(x)` from being a copy.[^soa-escape]

[^soa-escape]: there's an escape valve worth noting: a future `monpy.fft`-style API that operates on long complex arrays can transiently transpose to SoA, run the SIMD-dense kernel, and transpose back. the amortization works for `O(N log N)` ops on `O(N)` data; doesn't work for `O(N)` ops.

---

## 5. false sharing and the multi-thread alignment trap

monpy is currently single-threaded at the Python frontier. we release the GIL during BLAS calls, but our own kernels don't spawn threads. that's about to change when threaded scratch lands, and when it does, false sharing becomes a first-order concern.

### 5.1 the mechanics

two threads writing to addresses that share a 64-byte coherence granule cause the line to ping-pong between the cores' L1s via MESI invalidations. each invalidate-and-refetch costs 50–200 cycles on the core that lost the line. if the pattern is symmetric (each thread writes equally often), both threads stall and aggregate throughput collapses by 10×.

the famous bad pattern:

```mojo
struct ThreadState:
    var acc: Float64       # 8 B
    # implicit padding to alignof(struct) = 8 B; total 8 B
var states: ThreadState[N]
```

`states[0].acc` and `states[1].acc` sit in the same cache line for any N up to 8. eight threads can sustain false sharing on the entire array. the fix is overpadding to 64 B (or 128 B for Apple's logical line):

```mojo
struct ThreadState:
    var acc: Float64
    var _pad: SIMD[DType.uint8, 56]   # bring total to 64 B
```

Mojo doesn't yet have a stable `@align(64)` decorator on aggregate types; the explicit-padding approach works today and survives layout tooling.

### 5.2 the diagonal version

the subtler version is the _partial_ false share: a `ThreadState` of size 24 B straddles a cache line boundary every two-and-a-bit elements, and threads `i` and `i+2` collide depending on the base alignment. this is how tracing-tool overhead leaks into hot kernels — the trace state struct has a counter and a few pointers, total 40 B, and you hit pathological behavior at exactly the worker count where you'd expect linear scaling.[^mckenney]

[^mckenney]: McKenney's _Is Parallel Programming Hard, And, If So, What Can You Do About It?_ (2nd ed., 2021) has a chapter on this with concrete reproduction recipes; chapter 6 ("Partitioning and Synchronization Design") is the relevant one. the book is free at <https://mirrors.edge.kernel.org/pub/linux/kernel/people/paulmck/perfbook/perfbook.html>.

### 5.3 what monpy's threaded design needs

when we land the worker pool, each worker gets:

- a 64-byte-aligned scratch pointer, allocated through `monpy_aligned_alloc`.
- a `WorkerState` aggregate padded to 64 B (or 128 B if we want Apple-strict).
- all cross-worker communication via per-worker output slots that are themselves cache-line-padded; reductions happen at the end with a single-threaded pass.

the cost is constant memory overhead: 64 B per worker for the state, maybe 128 B per output slot. negligible against any real workload.

---

## 6. allocation primitives and what they actually guarantee

this is where the literature is the loosest. the allocator landscape:

### 6.1 `malloc`

**glibc ptmalloc2**: 16 B alignment guarantee on 64-bit (chunks store 2-pointer headers, the smallest aligned chunk is 16 B). for allocations ≥ `M_MMAP_THRESHOLD` (default 128 KB, but adapts upward dynamically), glibc switches to `mmap` and returns page-aligned (4 KB) pointers. **musl** and **macOS libsystem_malloc**: 16 B alignment (8 B was the old guarantee; macOS hardened this around 10.14). for large allocations they too switch to `mmap`-backed and you get page alignment for free.

so: if your array is bigger than ~128 KB on Linux or ~32 KB on macOS, you probably _already_ get page-aligned pointers from `malloc`. the alignment problem is acute only for arrays in the 64 B–32 KB range, which is a substantial fraction of monpy's small-tensor traffic but not the bulk-data hot path.

### 6.2 `posix_memalign(ptr, alignment, size)`

POSIX standard. `alignment` must be a power of two ≥ `sizeof(void*)`. returns 0 on success. error reporting via return value, not `errno`. available on macOS and every Linux libc.

### 6.3 `aligned_alloc(alignment, size)` (C11)

same semantics as `posix_memalign` minus the by-reference-pointer ergonomics. **critical gotcha**: C11 says `size` must be a multiple of `alignment` or behavior is undefined. glibc 2.16+ relaxed this; some implementations didn't. macOS 10.15+ has it. Mojo's runtime today calls into the system allocator; on macOS that's `aligned_alloc` with the relaxed-size assumption.

### 6.4 `mmap` (anonymous)

page-aligned by definition: 4 KB on x86, 16 KB on Apple Silicon. use `MAP_ANONYMOUS | MAP_PRIVATE`, optionally `MAP_HUGETLB` on Linux to request a 2 MB or 1 GB huge page. macOS has no analog of `MAP_HUGETLB`; macOS does not support transparent huge pages or super-pages on Apple Silicon at all.[^macos-thp] you get 16 KB pages, period.

[^macos-thp]: this is not a Modular limitation; it's a kernel limitation. the kernel-level discussion is on the Asahi Linux side: Apple's IOMMU and the SLC's coherence boundaries make THP semantics expensive in ways the ARM Cortex side doesn't share. source: Asahi Linux notes circa 2023, and the [HN thread](https://news.ycombinator.com/item?id=34476480) where Hector Martin (marcan) clarifies the kernel side.

### 6.5 `madvise(MADV_HUGEPAGE)` (Linux)

a hint to the kernel that this region is a candidate for THP backing. the kernel may or may not honor it depending on `transparent_hugepage` mode (`always`, `madvise`, `never`). worth setting on x86 server deployments for arrays > 2 MB; harmless elsewhere.

### 6.6 the Mojo-specific reality

Mojo's `UnsafePointer.alloc[T]` accepts an `alignment` parameter: `UnsafePointer[T].alloc(count, alignment=64)`. under the hood this currently routes to the system aligned-allocator (`aligned_alloc` on POSIX, `_aligned_malloc` on Windows). the corresponding free path is `UnsafePointer.free`, which Mojo wires to the matching deallocator. we don't need to roll our own bookkeeping header; the Mojo runtime already does the right thing. **alignment is one parameter away.**

if we hit a Mojo version where `alignment` isn't honored or doesn't exist on the target backend, the over-allocate-and-round pattern is six lines:

```mojo
fn aligned_alloc[T: AnyTrivialRegType](n: Int, alignment: Int) -> UnsafePointer[T]:
    let raw_bytes = n * sizeof[T]() + alignment + sizeof[Int]()
    let raw = UnsafePointer[UInt8].alloc(raw_bytes)
    let raw_addr = Int(raw)
    let aligned_addr = (raw_addr + alignment + sizeof[Int]() - 1) & ~(alignment - 1)
    let header_addr = aligned_addr - sizeof[Int]()
    UnsafePointer[Int](address=header_addr).store(raw_addr)
    return UnsafePointer[T](address=aligned_addr)

fn aligned_free[T: AnyTrivialRegType](p: UnsafePointer[T]):
    let header_addr = Int(p) - sizeof[Int]()
    let raw_addr = UnsafePointer[Int](address=header_addr).load()
    UnsafePointer[UInt8](address=raw_addr).free()
```

we use this only as fallback; default path goes through Mojo's native alignment parameter.

---

## 7. page size, huge pages, NUMA

### 7.1 page sizes in practice

- **x86 default**: 4 KB. THP up to 2 MB; rare 1 GB pages (need root, kernel boot config).
- **Apple Silicon**: 16 KB. no THP. no 1 GB pages on macOS. Asahi Linux on M-series has experimental 4 KB and 16 KB modes; mainstream macOS is 16 KB.
- **Linux on ARM64 Neoverse / Graviton**: typically 4 KB but kernel build-config can be 16 KB or 64 KB. THP works.

### 7.2 what page size buys monpy

for an array large enough to span multiple pages, the TLB becomes the bottleneck before anything else. a single 4 KB page in an L1 TLB serves 4 KB of data; a 2 MB huge page serves 500× as much. on a kernel that streams through 100 MB of `f32`s, we're looking at:

- 4 KB pages: 25,000 TLB entries of work, each ~3 cycles to walk → 75K cycles of pure TLB walking on a cold pass, negligible on a warm pass.
- 2 MB pages: 50 entries of work; everything fits in the L1 dTLB (typically 64–128 entries on modern x86, 192 on Apple).

the win is real but only for big arrays on x86. Apple Silicon's 16 KB native page already gives 4× the TLB reach of 4 KB, and we get no THP benefit on top. the huge-page argument is x86-server-specific.

### 7.3 NUMA

multi-socket x86 has memory-controller affinity: `numa_alloc_local`, `numa_set_policy`, etc. we don't currently do anything NUMA-aware and we shouldn't add it speculatively. when/if we run on a 2-socket EPYC and we see the 30% bandwidth tax of cross-socket access in profiles, we'll thread through `libnuma`. until then, premature optimization.

---

## 8. the allocator landscape, ranked for monpy's workloads

the 2025-vintage benchmarks paint a workload-dependent picture, not a clear winner. distilled from recent comparisons:[^bench]

[^bench]: sources include the [Microsoft mimalloc bench page](https://microsoft.github.io/mimalloc/bench.html), [Small Datum, "Battle of the Mallocators", 2025](http://smalldatum.blogspot.com/2025/04/battle-of-mallocators.html), and recent allocator papers (SpeedMalloc, Exgen-Malloc) on arXiv. the takeaways below are workload-specific, not absolute.

| allocator       | small alloc latency                      | large alloc throughput            | multi-thread               | verdict for monpy                    |
| :-------------- | :--------------------------------------- | :-------------------------------- | :------------------------- | :----------------------------------- |
| glibc ptmalloc2 | mediocre                                 | mediocre                          | poor under contention      | default; fine for single-thread      |
| jemalloc        | good (P99 wins)                          | good                              | excellent (per-CPU arenas) | strong choice for multi-threaded     |
| tcmalloc        | good                                     | excellent (50× libmalloc at 4 MB) | excellent                  | best for large allocs and throughput |
| mimalloc        | excellent (15% lower P99)                | suffers above ~1 MB               | good                       | best for small allocs only           |
| custom arena    | best for the lifecycle it's designed for | n/a                               | n/a                        | what we want for LAPACK workspaces   |

the actionable conclusion: monpy's allocations are bimodal. we have many big array allocations (10 KB – 10 GB) where tcmalloc or jemalloc would win meaningfully if anyone were stressing them, and a long tail of metadata allocations (shape vectors, stride vectors, ufunc dispatch tables) where allocator choice barely matters because they're short-lived and small. the high-leverage move is **not** "swap the allocator", it's "introduce arena allocators for the patterns we hit repeatedly":

- LAPACK workspace arena (one per dtype, one per routine).
- per-thread scratch arena, lazily grown.
- a small-object pool for sub-cache-line metadata (shape/stride descriptors).

the system allocator can stay default; the workloads that matter sit on top of arenas we control.

---

## 9. monpy's memory model — current and proposed

### 9.1 current state, audited

- `PhysicalStorage` in `src/array.mojo` owns a single `UnsafePointer[Scalar[T]]` and a `(shape, strides, dtype)` triple.
- `Array` is a view (`storage`, `offset`, optional override of shape/strides for slices/transposes).
- all allocation goes through Mojo's default `UnsafePointer[T].alloc(n)` — no alignment parameter, no slack.
- resulting alignment: `alignof(T)`, which is `sizeof(T)` for the trivial scalar types we ship. so `f64` arrays are 8-byte aligned; `f32`, 4-byte; `f16`/`i16`, 2-byte. effectively _no_ SIMD-favorable alignment guarantee.
- SIMD kernels in `src/elementwise.mojo` use unaligned loads (`simd_load[unaligned]`) uniformly.
- no padding inserted on non-multiple row sizes; strides match the natural product everywhere C-contig is requested.
- no huge-page hint, no NUMA hint, no per-thread scratch.

### 9.2 the proposed model

#### 9.2.1 an aligned base allocator

```mojo
# src/internal/aligned_alloc.mojo
fn monpy_aligned_alloc[T: DType](n: Int, alignment: Int = 64) -> UnsafePointer[Scalar[T]]:
    # Prefer Mojo's native alignment parameter where available.
    return UnsafePointer[Scalar[T]].alloc(n, alignment=alignment)

fn monpy_aligned_free[T: DType](p: UnsafePointer[Scalar[T]]):
    p.free()
```

default alignment of 64 B because:

- it eliminates split-line loads for any SIMD width ≤ 64 B (covers AVX-512 trivially).
- it is also a multiple of 32 B (AVX2) and 16 B (NEON, SSE).
- it's the largest cache-coherence granule in current production hardware (x86 64 B; Apple 64 B coherence even with 128 B logical line).
- it's free with `aligned_alloc` for all but the smallest allocations.

#### 9.2.2 tiered alignment policy

|      array size | alignment                          | mechanism                           |
| --------------: | :--------------------------------- | :---------------------------------- |
|          < 64 B | natural `alignof(T)`               | default `UnsafePointer.alloc`       |
|     64 B – 4 KB | 64 B                               | `monpy_aligned_alloc(n, 64)`        |
|     4 KB – 2 MB | page (4 KB on x86, 16 KB on Apple) | `monpy_aligned_alloc(n, page_size)` |
|   > 2 MB on x86 | page + `madvise(MADV_HUGEPAGE)`    | `mmap` + advise                     |
| > 2 MB on Apple | 16 KB page (no THP)                | `monpy_aligned_alloc(n, 16384)`     |

the page-aligned tier is essentially free: `aligned_alloc` for those sizes already routes to `mmap` in glibc, which gives page alignment by construction. on Apple, we explicitly request 16 KB to match the system page so we don't fight the VM subsystem.

the < 64 B tier exists because forcing 64 B alignment on a 16-byte array wastes 48 B per allocation; for the long tail of tiny arrays in unit tests and shape descriptors, that adds up. the threshold was picked at 64 B specifically so that anything large enough to _use_ SIMD gets 64 B alignment and anything smaller doesn't pay the slack tax.

#### 9.2.3 stride padding for 2D

when constructing an `(M, N)` C-contiguous array with `N · itemsize % 64 != 0`, pad the leading-axis stride up to the next multiple of 64. concretely, for `(M, N) = (100, 7)` `f32`:

- natural row stride: `7 · 4 = 28` bytes.
- padded row stride: `32` bytes (next multiple of 32, which is enough for 32 B SIMD; or 64 if we want to match the cache-line tier).
- total overhead: `100 · (32 - 28) = 400` bytes against a `2800` byte payload, ~14% slack.
- win: every row's first element is at an aligned offset. per-row alignment shuffle eliminated.

the slack scales as `O(M · padding_per_row)`, where `padding_per_row` is at most `align - 1`. for tall-thin matrices (large M, small N) this looks bad as a percentage but is small in absolute bytes.

the implementation is a one-liner in `creation.mojo`:

```mojo
fn pad_row_stride(n_cols: Int, itemsize: Int, alignment: Int = 64) -> Int:
    let natural = n_cols * itemsize
    let padded = (natural + alignment - 1) & ~(alignment - 1)
    return padded
```

this has to be opt-in or behind a heuristic: there's a (real) cost in compatibility — a buffer with non-natural strides loses some ABI properties (NumPy's `.flags.c_contiguous` returns False for it, even though semantically the data is row-major). we can expose this via a `align` keyword on `monpy.empty`, default off, and turn it on automatically for arrays produced by intermediate kernel computations where ABI compat is moot.

#### 9.2.4 workspace arena for LAPACK

LAPACK routines like `?gesvd`, `?syevd`, `?gels` request a workspace through a two-call protocol: query with `lwork = -1` to get the optimal size, then allocate, then call again. the optimal size depends on the matrix shape, and naive code allocates fresh on every call.

the arena pattern:

```mojo
struct LapackWorkspace[T: DType]:
    var buf: UnsafePointer[Scalar[T]]
    var capacity_bytes: Int
    var alignment: Int

    fn ensure(mut self, requested_bytes: Int):
        if requested_bytes <= self.capacity_bytes:
            return
        # round up to next power of 2 to amortize.
        let new_cap = next_pow2(requested_bytes)
        self.buf.free()
        self.buf = monpy_aligned_alloc[T](new_cap // sizeof[Scalar[T]](), self.alignment)
        self.capacity_bytes = new_cap
```

keyed by `(routine, dtype)`. the amortization argument: `?gesvd` on an `(M, N)` matrix needs roughly `O(M·N)` workspace; over a sequence of calls with growing `(M, N)`, the arena grows monotonically and the total allocation cost is `O(N_final · log(N_final))` instead of `O(N_total)`.

GC heuristic: every 1000 calls without a hit at the current size, halve the capacity. prevents long-running processes from holding workspace pinned forever after a single big call.

#### 9.2.5 per-thread scratch

already covered in §5.3. each worker gets a `monpy_aligned_alloc(SCRATCH_SIZE, 64)`. the `WorkerState` aggregate is hand-padded to 64 B. scratch is reused across iterations; the worker loop never allocates.

#### 9.2.6 aligned-load fast path in SIMD kernels

the dispatch:

```mojo
fn elementwise_add_f32(a: UnsafePointer[Float32], b: UnsafePointer[Float32],
                      out: UnsafePointer[Float32], n: Int):
    let a_aligned = (Int(a) & 63) == 0
    let b_aligned = (Int(b) & 63) == 0
    let out_aligned = (Int(out) & 63) == 0
    if a_aligned and b_aligned and out_aligned:
        elementwise_add_f32_aligned(a, b, out, n)
    else:
        elementwise_add_f32_unaligned(a, b, out, n)
```

the check is three address-and-compares, total ~1 cycle. the aligned variant uses `simd_load[alignment=64]` and can permit the codegen to issue `VMOVAPS`/`VMOVAPD` on x86; on NEON the issued instruction is identical but the alignment hint passes through to the LLVM optimizer.

**the single most important property**: when `monpy_aligned_alloc` is the _only_ allocator path used for `PhysicalStorage`, the alignment check is statically true for every freshly-allocated array. the dispatch becomes dead-code-eliminable in many call sites. we get the benefit of aligned loads without the overhead of runtime checks for the common path; the fallback handles slices, views, and externally-provided buffers.

#### 9.2.7 stride alignment annotations

add an optional field to `Array`:

```mojo
struct Array[T: DType]:
    ...
    var stride_alignment: Int  # 0 = unknown; otherwise GCD of all stride components
```

set at array creation time (cheap GCD over the strides). the SIMD dispatcher reads it: if `stride_alignment >= 64` and base address is aligned, the inner loop can elide per-iteration alignment checks entirely.

this is the bookkeeping that enables static aligned dispatch. without it we're back to runtime checks.

### 9.3 quantitative estimates

#### elementwise f32 add, $N = 10^6$, L1-resident output

per element:

- two 32 B loads, one 32 B store, one fused add.
- throughput-bound on L1: 2 loads/cycle limit on most x86 since Sandy Bridge, 2 loads + 1 store on Ice Lake+, 2×128-bit loads/cycle on Apple Firestorm.

misalignment cost on Apple Silicon (worst case, every load split): each split eats one extra load slot. with ~12.5% of loads splitting on random alignment for 32 B loads on 64 B lines, at 3-cycle penalty per split and ~125,000 split events for $N = 10^6$, that's ~375,000 cycles. against a ~1 ms kernel (~3 M cycles at ~3 GHz Apple P-core), that's a 12% speedup just from killing splits.

on AVX-512 the split rate is ~98%, so the speedup from alignment is closer to 30–40% on 512-bit kernels. on AVX2 (32 B loads on 64 B lines): splits are 50% by default; ~5–10% kernel speedup.

#### 2048×2048 f64 matrix add

`2048 · 8 = 16384` bytes per row, already a multiple of 64. no padding needed; alignment policy doesn't help on the row geometry, only on the base. win is the base-alignment +5% bucket.

#### 100×7 f32 matrix add with row padding

without padding: 100 rows of 28 B each. each row's start address is offset by 28 B from the previous, so seven of every eight row-starts are misaligned. SIMD fast path is not taken; we fall to scalar.

with padding to 32 B rows: every row-start is 32 B aligned. SIMD path with mask `[1,1,1,1,1,1,1,0,0]` (or 8-elem load with the high lane masked) takes over. rough speedup: 4–6× on this kernel (going from 7 scalar ops/row to 1 vector op/row).

this is the case where alignment policy pays the most. tall-thin matrices arise from chained `[..., None]` broadcasting, from RNN hidden states reshaped into batch×features, from convolution lowering. we see them.

#### complex64 multiply, $N = 10^6$

AoS, currently unaligned, see §4.1. with base alignment guaranteed, we shave the line-split tax (~5%) and gain nothing on the shuffle. kernel speedup ~5%.

#### aggregated rollup

Aligned allocation plus aligned-load fast path:

- f32 elementwise: +8–12%
- f64 elementwise: +5–8%
- complex elementwise: +5–7%
- AVX-512 paths (when we add them): +25–35%

Stride padding:

- tall-thin matrix kernels: +4–6× speedup on the worst shapes, no impact on aligned-natural shapes.

Workspace arenas and threaded scratch:

- LAPACK-heavy code paths: +10–20% on average, dominated by the workspace alloc cost going to zero on subsequent calls.
- threaded reductions: cleared the false-share footgun, no positive perf number to attribute (the wins are in _not_ having a 10× regression).

---

## 10. Mojo-specific quirks

a handful of details that bit early prototypes:

1. **`UnsafePointer[T].alloc(count, alignment=k)`** is the canonical API. pass `alignment=64`. free with `.free()`; the runtime knows it was alignment-allocated.
2. **`@register_passable("trivial")` types have no internal padding.** our complex types and SIMD vectors are register-passable; they pack as you'd expect. this is good — you get exactly `2·sizeof(f64) = 16 B` for `Complex[f64]`.
3. **`simd_load[alignment=K]` and `simd_store[alignment=K]`** are the parameterized intrinsics. `simd_load[unaligned]` is `simd_load[alignment=1]`; `simd_load` (no qualifier) currently defaults to `alignment=alignof(T)`, _not_ `alignment=widthbytes`. be explicit. for our 32-byte SIMD loads of `f32`, set `alignment=32` (and the codegen uses `VMOVAPS` on x86 if the pointer is statically known to be aligned, otherwise an aligned-mode load that _would_ fault on misalignment in debug builds).
4. **the Mojo standard library does not yet expose `madvise` directly**; we'd FFI through libc on Linux. trivial but adds a platform shim.
5. **Mojo's `Span[T]`** types pass alignment information through generics. when we change `PhysicalStorage` to wrap an aligned pointer, downstream `Span[T]` materializations should pick up the alignment via type-level metadata in upcoming Mojo versions. right now we hand-thread it.
6. **`@parameter`-known alignment specialization**: writing `@parameter if alignment >= 64:` in the kernel lets us monomorphize an aligned vs unaligned variant at compile time when the alignment is parameter-known. this is the cleanest expression of the dispatch in §9.2.6.

---

## 11. the buffer-protocol contract

monpy exposes its buffers to NumPy via PEP 3118. the relevant alignment claims of the buffer protocol:

- `itemsize`: the dtype's size in bytes.
- `strides`: tuple of byte counts.
- the base address must be at least `itemsize`-aligned. (NumPy's allocator gives 16 B regardless; CPython's buffer protocol does not require more.)

if we over-align — give NumPy a 64-byte-aligned buffer — NumPy doesn't notice and doesn't care. NumPy reads buffers with its own alignment guards (it'll memcpy to an aligned temp internally if it needs strict alignment for SIMD; otherwise unaligned-load through the strided iterator). so our aggressive alignment policy is safe to expose; it never breaks an external consumer.

the flip side: if we hand NumPy a buffer with **stride padding** (§9.2.3), NumPy will report `c_contiguous=False` because the strides don't match the natural product. code that hits a `np.ascontiguousarray` then triggers a copy. this is correct behavior; the cost is a copy on the NumPy side that we knew we were eating in exchange for SIMD speedup on the monpy side. the right policy is: **don't pad strides on arrays that we expect to leave the monpy boundary**. padding is for kernels' own intermediate buffers and for arrays explicitly tagged as monpy-internal. the user-facing default constructors (`monpy.empty`, `monpy.zeros`) produce natural strides and rely on base alignment alone. internal codegen uses `monpy.empty_aligned`-style helpers when the win is large enough to justify the c_contiguous loss.

---

## 12. staged rollout

a concrete plan with believable timelines and exit criteria.

### Step 1: aligned base allocator (1–2 days)

**land**:

- `src/internal/aligned_alloc.mojo` with `monpy_aligned_alloc` / `monpy_aligned_free`.
- `PhysicalStorage` constructor defaults to 64 B alignment for sizes ≥ 64 B; falls back to natural alignment for smaller.
- microbenchmark in `benchmarks/bench_alignment.py` measuring base-address alignment distribution before and after.

**exit**: all `PhysicalStorage` allocations on the default path produce `(addr & 63) == 0` for sizes ≥ 64 B. no measurable regression on existing benchmarks; expect +0–5% from the L1 throughput improvement on incidentally-aligned code.

### Step 2: aligned-load fast path (1 week)

**land**:

- `Array.stride_alignment` field, computed at construction.
- aligned-mode SIMD intrinsic dispatch in `apply_binary_typed_vec` and `apply_unary_preserve_typed_vec`.
- static parameter specialization where alignment is known at compile time.
- benchmarks against bench_array_core.py for f32/f64/complex elementwise.

**exit**: aligned variant fires on ≥ 95% of dispatches against monpy-allocated arrays. +5–15% on memory-bound elementwise across f32, f64. AVX-512 path (when introduced) shows the bigger lift.

### Step 3: stride padding (3–5 days)

**land**:

- `monpy.empty(shape, align=True)` constructor variant.
- internal kernels that produce intermediate 2D arrays use `align=True` by default.
- stride padding heuristic: pad if `n_cols · itemsize % 64 != 0` AND `n_rows ≥ 8`.
- benchmark suite for tall-thin shapes: `(N, 3)`, `(N, 5)`, `(N, 7)` for `N ∈ {100, 1000, 100000}`.

**exit**: tall-thin elementwise kernels run on the SIMD path (currently fall to scalar). 4–6× speedup on those shapes, ~10% memory overhead acceptable.

### Step 4: threaded scratch + false-share avoidance (concurrent with parallelization landing)

**land**:

- worker pool with per-worker 64 B aligned scratch.
- `WorkerState` padded to 64 B (or 128 B for Apple-strict mode).
- a test that intentionally constructs the false-share case to ensure we _don't_ hit it.

**exit**: multi-threaded reductions and elementwise kernels scale linearly with worker count up to physical-core count. no false-share regressions in microbenchmarks.

### Step 5: workspace arenas (1 week)

**land**:

- `LapackWorkspace[T]` keyed by `(routine_id, dtype)`.
- wired through `linalg` module's LAPACK call sites.
- GC heuristic: idle decay every 1000 calls.

**exit**: repeated LAPACK calls on growing-but-bounded shapes show ≤ 1 alloc cost in profiles after warmup. +10–20% on LAPACK-bound benchmarks.

---

## 13. what we are _not_ doing (and why)

some discarded ideas, recorded so we don't relitigate them:

- **switching system allocators wholesale (jemalloc, mimalloc, tcmalloc)**: the wins are workload-specific and small relative to the arena-allocator wins, and we'd take on a non-trivial dependency. we can revisit if a profile shows allocator lock contention as a top-3 hotspot. right now it isn't.
- **SoA layout for complex**: covered in §4. the 5–10% win on complex multiply isn't worth the BLAS-interop cost or the doubled cost on real-only ops.
- **page-locked memory (`mlock`)**: relevant for GPU DMA paths, not for CPU-only kernels. defer until we have an actual GPU offload path.
- **1 GB huge pages**: useful for terabyte-scale arrays; not our target.
- **NUMA pinning**: defer until we see cross-socket bandwidth tax in profiles. probably never on Apple Silicon; possible-but-not-priority on multi-socket EPYC.
- **`@align(N)` on every aggregate type**: tempting and clean syntactically, but Mojo's support is uneven across versions. hand-padding `WorkerState` is uglier and bulletproof.

---

## 14. references (the actually-load-bearing ones)

cited and worth reading in full:

1. Drepper, U. "What Every Programmer Should Know About Memory." LWN.net, 2007. <https://www.akkadia.org/drepper/cpumemory.pdf>. the foundational text. most of the cycle counts in §1.3 trace back here, updated against modern hardware.
2. Goto, K. & Van de Geijn, R. "Anatomy of High-Performance Matrix Multiplication." ACM TOMS 34(3), 2008. the packing-into-aligned-panels argument behind §3.4.
3. McKenney, P. "Is Parallel Programming Hard, And, If So, What Can You Do About It?" 2nd ed., 2021. <https://mirrors.edge.kernel.org/pub/linux/kernel/people/paulmck/perfbook/perfbook.html>. false-sharing chapter is the §5 reference.
4. Intel, "Intel 64 and IA-32 Architectures Optimization Reference Manual," current ed. authoritative source for AVX/AVX-512 alignment-mode penalties; consult §15.7 ("Alignment").
5. Apple, WWDC sessions on optimizing for Apple Silicon, 2020–2024 (especially "Tune your Core ML models for the Apple Neural Engine," 2023, which has the most concrete public data on M-series cache behavior).
6. Hennessy, J. & Patterson, D. "Computer Architecture: A Quantitative Approach." 6th ed., 2017. Chapter 2 on memory hierarchy is the reference for everything in §1.
7. Bryant, R. & O'Hallaron, D. "Computer Systems: A Programmer's Perspective." 3rd ed., 2015. Chapter 6 ("Memory Hierarchy") is the gentler companion to Hennessy/Patterson.
8. Travis Downs, "Performance Speed Limits," 2019. <https://travisdowns.github.io/blog/2019/06/11/speed-limits.html>. modern x86 throughput limits, including the Ice Lake load+store-per-cycle figures cited in §1.2.
9. PEP 3118 — Revising the buffer protocol. <https://peps.python.org/pep-3118/>. the contract surface we expose to NumPy.
10. corsix/amx, <https://github.com/corsix/amx>. the reverse-engineered AMX ISA reference, covering pair-load alignment requirements (§2.4).
11. mimalloc bench, <https://microsoft.github.io/mimalloc/bench.html>. micro vs macro allocator benchmarks; the source for the table in §8.
12. Small Datum, "Battle of the Mallocators," 2025. <http://smalldatum.blogspot.com/2025/04/battle-of-mallocators.html>. the 2025 update to the allocator landscape.
13. Asahi Linux notes on Apple's MMU and 16 KB page handling. <https://asahilinux.org/>. background for the THP-on-Apple discussion in §6.4 / §7.

---

## 15. closing read

the alignment work for monpy is shaped like a hill, not a cliff. the aligned allocator lands in a day and gets us a few percent. the aligned-load fast path is the big monomorphization payoff and lands in a week. stride padding unlocks the tall-thin shapes that are silently scalar today. workspace arenas and threaded scratch are downstream of the threading work and should not be done before it.

the single thing to internalize: **alignment is a geometric property as much as a microarchitectural one**. modern hardware has narrowed the cycle penalty of misaligned-mode loads to near zero, and this fact has propagated through the literature as "alignment doesn't matter anymore". it does. the penalty just moved: it's no longer in the load decode, it's in the line crossings and the throughput-port accounting. a 64-byte-aligned base address forecloses on a whole class of split-line losses without any extra bookkeeping at runtime.

the other thing: false sharing is the alignment problem that is least about alignment and most about **separation**. padding to 64 B is not making your data go faster; it's making your data not slow other threads' data down. different problem, same byte count, opposite intuition.

the work itself is mostly mechanical: route allocations through one new function, propagate one new field, branch on one new predicate. the reasoning takes a long note. the code is short.
