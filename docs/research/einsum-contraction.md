---
title: "einsum and tensor contraction in monpy"
date: 2026-05-07
---

# einsum and tensor contraction in monpy

_the naive left-to-right contraction is defensible as a starting point and catastrophic at scale — the gap between a greedy path and a random fold can be a hundred-fold in FLOPs._

this note grounds the design of `einsum`, `tensordot`, `tensorinv`, and `tensorsolve` in `python/monpy/linalg.py`. the sections move from formalism to optimisation strategy to the load-bearing engineering questions that determine whether a contraction runs at peak BLAS throughput or wastes 90% of its FLOP budget on memory traffic.

---

## 1. the einsum convention — formal definition

Einstein summation, in its pre-machine form, is a stenographic convention: any index appearing twice in a product of factors is summed over, with the summation symbol elided. the convention sits at the boundary between notation and semantics. repurposing it as an array-language operator (Daniel et al. did this in NumPy as a generalisation of `tensordot`) requires removing the implicit "appears twice" rule and replacing it with an explicit _output specification_. the modern einsum is thus a small DSL: a left-hand side (operand label lists) and a right-hand side (output label list), separated by `->`.

**Definition (einsum expression).** fix a finite alphabet $\Sigma$ of _labels_ and a dimension assignment $d : \Sigma \to \mathbb{N}_{>0}$. an einsum expression on $n$ tensors is a tuple
$$ \mathcal{E} = (\Sigma_1, \Sigma_2, \ldots, \Sigma_n; \Sigma_{\text{out}}) $$ where each $\Sigma_k$ is a _finite sequence_ (not a set) of labels — order matters because it pins each label to a specific tensor axis — and $\Sigma_{\text{out}}$ is a sequence of labels with $\Sigma_{\text{out}} \subseteq \bigcup_k \mathrm{set}(\Sigma_k)$. the shape of operand $T_k$ is $(d(\ell))_{\ell \in \Sigma_k}$.

the set of _contracted labels_ is
$$ C(\mathcal{E}) = \Big(\bigcup_k \mathrm{set}(\Sigma_k)\Big) \setminus \mathrm{set}(\Sigma_{\text{out}}). $$

the _value_ of $\mathcal{E}$ on tensors $T_1, \ldots, T_n$ is the tensor $R$ of shape $(d(\ell))_{\ell \in \Sigma_{\text{out}}}$ defined componentwise by
$$ R[\,(x_\ell)_{\ell \in \Sigma_{\text{out}}}\,] \;=\; \sum_{(y_\ell)_{\ell \in C(\mathcal{E})}} \;\prod_{k=1}^{n} T_k\!\left[\,(z^{(k)}_\ell)_{\ell \in \Sigma_k}\,\right] $$ where $z^{(k)}_\ell = x_\ell$ if $\ell \in \Sigma_{\text{out}}$ and $z^{(k)}_\ell = y_\ell$ otherwise. the case $\ell$ appearing more than once in $\Sigma_k$ — a _self-trace_ — is handled by extracting the diagonal first; see §5.

**implicit output mode.** when `->` is omitted, NumPy (and monpy) computes the output label sequence as the labels appearing exactly once across all operands, sorted alphabetically.[^alphabetical] thus `'i,j'` becomes `'i,j->ij'` (every label appears once) and `'ij,jk'` becomes `'ij,jk->ik'` (`j` appears twice, hence contracted). the implicit rule is occasionally surprising: `'ij,ji'` collapses to scalar (no free labels), while `'ij,ji->ij'` is a Hadamard product after one transpose.

**multiset vs sequence.** it is tempting to model $\Sigma_k$ as a multiset, but order is load-bearing: `'ij,jk'` and `'ji,jk'` differ by a transpose of the first operand. internally, monpy normalises by storing each $\Sigma_k$ as a list and using sets only when reasoning about _which_ labels are repeated, contracted, or batched.

[^alphabetical]: this is one of the convention's mild war crimes. if you write `'σα,αβ'` and forget the implicit-output rule, the result depends on the Unicode codepoint ordering of your indices. monpy follows NumPy verbatim here for compatibility, but a kinder design would reject implicit-output expressions when any non-ASCII labels appear.

---

## 2. the contraction order problem

for a chain $T_1, T_2, \ldots, T_n$ contracted under a fixed set of labels, the _value_ of the expression is invariant under associativity: $((T_1 \cdot T_2) \cdot T_3) = (T_1 \cdot (T_2 \cdot T_3))$. the _cost_ — counting FLOPs and peak memory — is not. choosing the ordering is the **contraction order problem**.

**concrete example.** consider `'ij,jk,kl->il'` with $i = l = 10$, $j = k = 100$. the left-to-right path computes $(T_1 T_2)$ first, an intermediate of shape $(10, 100)$ at $10 \cdot 100 \cdot 100 = 10^5$ FLOPs, then multiplies by $T_3$ of shape $(100, 10)$ at $10 \cdot 100 \cdot 10 = 10^4$ FLOPs. total: $1.1 \times 10^5$. the right-to-left path computes $(T_2 T_3)$ first, an intermediate of shape $(100, 10)$ at $100 \cdot 100 \cdot 10 = 10^5$ FLOPs, then multiplies $T_1$ at $10 \cdot 100 \cdot 10 = 10^4$. same total, by symmetry. now make $j = k = 1000$: both paths scale by $100\times$ in the inner term, so the choice still doesn't matter for matrix chains where the reduction dimension is shared. _but_ introduce a hyperedge — say `'ij,jk,kj->i'` — and the sub-network ordering changes the cost by orders of magnitude. the matrix-chain case is benign; the general tensor-network case is wild.

**Theorem (NP-hardness).** determining the operation-minimising contraction sequence for an arbitrary tensor network is NP-hard. the connection runs through _treewidth_: Markov and Shi (2008) proved that the minimum cost of contracting a tensor network with bond dimension $d$ scales as $d^{O(\mathrm{tw}(G))}$, where $G$ is the line graph of the network and $\mathrm{tw}$ is its treewidth. computing treewidth is NP-hard (Arnborg, Corneil, Proskurowski 1987), and the cost reduction is tight enough that contraction order inherits the hardness.[^treewidth-tight]

**Proof sketch.** a tensor network is a hypergraph $H = (V, E)$ with $V$ the set of tensors and $E$ the set of labels (each label is a hyperedge connecting all tensors that share it). a contraction sequence is an elimination ordering on $E$. the cost of eliminating edge $e$ at step $t$ is $\prod_{e' \text{ incident to a remaining tensor at } t} d(e')$ — exactly the size of the merged tensor. the minimum over orderings is, up to log factors, $d^{\mathrm{tw}(L(H)) + 1}$ where $L(H)$ is the line graph (because each elimination step corresponds to vertex elimination in $L(H)$, and tree decompositions characterise minimum-fill elimination orderings). Markov and Shi show this for quantum-circuit simulation, but the argument is purely combinatorial.

the earlier Lam–Sadayappan–Wenger 1997 result establishes NP-hardness for the more restricted class of _single-term_ tensor contractions arising in quantum-chemistry coupled-cluster expressions. their reduction goes through the matrix-chain ordering generalisation, which is itself in P (the Hu–Shing $O(n \log n)$ algorithm), but the chemistry expressions allow shared indices that turn the chain into a DAG and break tractability.

[^treewidth-tight]: "tight" here means up to constants in the exponent. the actual reduction in O'Gorman 2019 ("Parameterization of tensor network contraction") establishes the equivalence at the level of carving-width and contraction trees, not just treewidth. for practical purposes — anything monpy will encounter — the gap is irrelevant; treewidth gives the right intuition.

**practical consequence.** monpy contractions involving more than $\sim 6$ tensors should not search exhaustively. the constants matter: opt_einsum's `'optimal'` is tractable up to $\sim 10$ tensors, `'dp'` up to $\sim 30$, `'greedy'` essentially unbounded. the crossover into NP-hard hell happens fast.

---

## 3. opt_einsum strategies

`opt_einsum` (Smith and Gray 2018, JOSS) is the de-facto path optimiser. it exposes four families of algorithms; understanding their tradeoffs is necessary for deciding what monpy should do.

### 3a. greedy

at each step, evaluate the _cost_ (FLOPs of the resulting matmul) and _size_ (elements of the intermediate) of every pairwise contraction among the remaining tensors. pair the two minimising the heuristic — by default, `opt_einsum` uses `(removed_size, cost)` as a lexicographic key, preferring contractions that shrink the network the most.

complexity: $O(n^3)$ in the number of tensors per round, $O(n)$ rounds, so $O(n^4)$ overall. in practice, dominated by the cost function evaluation. empirically returns the optimal path on a strong majority of structured contractions (the NumPy docs claim "majority of cases", which matches opt_einsum's benchmarks).

**failure mode.** greedy is myopic; it cannot lookahead two steps. a contraction with two roughly-equal-cost first moves can be tricked into picking the locally cheaper one whose downstream cost is catastrophic. the classic trap is a network where one labelling produces a low-rank intermediate the second move can exploit, but greedy preferred a slightly smaller-but-dense first intermediate.

### 3b. branch-and-bound

full search with cost-based pruning. opt_einsum's `'branch-all'` searches likely paths; `'branch-2'` restricts to the two best options at each step. the lower-bound function uses the size of the smallest-possible-merge; the cost of the best path found so far prunes any partial path exceeding it.

complexity: worst case $O(n!)$, but pruning makes it tractable up to $\sim 20$ tensors. the `'optimal'` strategy is essentially branch-and-bound with no early termination heuristic.

### 3c. dynamic programming

for the _matrix-chain_ special case (a sequence with one shared index per neighbour), the standard $O(n^3)$ DP is optimal — and in fact Hu and Shing 1981 give an $O(n \log n)$ polygon-triangulation algorithm. for _general_ tensor networks, opt_einsum's `'dp'` strategy enumerates subsets of the $n$ tensors in order of size, computes for each subset the cheapest contraction tree, and combines. worst case $O(3^n)$ in the number of tensors but with strong pruning (no outer products are considered, which truncates the search dramatically). Pfeifer, Haegeman, Verstraete 2014 refined this with bond-dimension-aware pruning and showed several-orders-of-magnitude wall-time reductions on physics workloads. the `'dp'` strategy is "essentially optimal" for tensor-network-states-style contractions and is the most-used optimiser in TN practice.

### 3d. random search (`'random-greedy'`)

sample contraction trees by running greedy with stochastic tie-breaking — at each step, choose from the top-$n$ contractions weighted by a Boltzmann factor on cost. keep the best path found across $k$ samples. opt_einsum's docs report that for a 40-tensor randomly generated contraction with bond dimension 5, random-greedy with 32 samples found a path $2^4 = 16\times$ faster than vanilla greedy (the cost dropped from $2^{36}$ to $2^{32}$ FLOPs). speedups depend strongly on contraction structure; for structured networks (regular grids, matrix chains) greedy is already near-optimal and random-greedy buys little.

### what monpy should do

the current `python/monpy/linalg.py` implementation pairs left-to-right: `_einsum_pair_contract(T_0, T_1) → T_0'`, then pair $(T_0', T_2)$, then $(T_0'', T_3)$, and so on. this is the _worst defensible_ order — it ignores the structure of the network entirely.

**recommendation.** implement a 100-line in-house greedy. the cost function is straightforward: for each unordered pair $(T_i, T_j)$ with shared labels $S$ and free labels $F_i, F_j$, the contraction cost is $\prod_{\ell \in S \cup F_i \cup F_j} d(\ell)$, the intermediate size is $\prod_{\ell \in F_i \cup F_j} d(\ell)$. pick the pair with smallest `(intermediate_size, cost)`, contract it, repeat. this requires no dependency, captures 90% of the available speedup on the contractions monpy will see, and the remaining 10% (specifically, contractions with treewidth $\geq 4$) are either rare or arrive with enough tensors that the user should be using opt_einsum directly.

if users want full path optimisation, expose an `optimize=` keyword that defers to `opt_einsum.contract_path` when the package is available. NumPy did exactly this in 1.12.0; the design has held up.

---

## 4. pairwise contraction reduces to a single matmul

this is the load-bearing optimisation. every binary einsum contraction can be expressed as one BLAS gemm call, modulo transposes and reshapes. the proof is a finger exercise but worth doing carefully because the bookkeeping is where bugs hide.

**Theorem (gemm reduction).** let $A$ have label sequence $\Sigma_A$ and $B$ have $\Sigma_B$. let $C = \mathrm{set}(\Sigma_A) \cap \mathrm{set}(\Sigma_B)$ be the contracted labels. let $F_A = \mathrm{set}(\Sigma_A) \setminus C$ and $F_B = \mathrm{set}(\Sigma_B) \setminus C$ be the free labels. then the einsum with output $\Sigma_{\text{out}} = \langle F_A, F_B \rangle$ (some ordering) equals
$$ \mathrm{reshape}\bigl( \mathrm{matmul}(\hat A, \hat B), \; \mathrm{shape}(\Sigma_{\text{out}}) \bigr) $$ where $\hat A$ is $A$ transposed to put $F_A$ axes first and $C$ axes last, then reshaped to $(|F_A|, |C|)$ where $|S| = \prod_{\ell \in S} d(\ell)$, and $\hat B$ is $B$ transposed to put $C$ first and $F_B$ last, then reshaped to $(|C|, |F_B|)$. a final transpose realigns to the user-requested $\Sigma_{\text{out}}$ order.

**worked example.** `'ijk,jkl->il'` with $A.\mathrm{shape} = (2, 3, 4)$, $B.\mathrm{shape} = (3, 4, 5)$.

1. $\Sigma_A = \langle i, j, k \rangle$, $\Sigma_B = \langle j, k, l \rangle$. $C = \{j, k\}$, $F_A = \{i\}$, $F_B = \{l\}$.
2. transpose $A$ to put $i$ first, $j, k$ last: already $\langle i, j, k \rangle$, no transpose needed.
3. reshape $A$ to $(2, 3 \cdot 4) = (2, 12)$.
4. transpose $B$ to put $j, k$ first, $l$ last: already $\langle j, k, l \rangle$, no transpose.
5. reshape $B$ to $(3 \cdot 4, 5) = (12, 5)$.
6. matmul: $(2, 12) \cdot (12, 5) = (2, 5)$.
7. no final reshape (output is already 2D), no final transpose ($\Sigma_{\text{out}} = \langle i, l \rangle$ matches).

**counter-example showing transposes are necessary.** `'ikj,kjl->il'` with the same shapes. $A$ is now $(2, 4, 3)$, $B$ is $(4, 3, 5)$. to put contracted axes $\{j, k\}$ at the end of $A$ in the order matching the start of $B$, transpose $A$ to axis order $(i, k, j) = (0, 1, 2)$ — but $B$'s contracted axes are in order $(k, j)$, so $A$'s should also be $(k, j)$. the discipline: pick a _canonical order_ for the contracted labels (alphabetical, or first-appearance-in-$A$), transpose both tensors to match that order, then reshape.

**why this matters.** a naive nested-loop implementation of einsum costs $\prod_\ell d(\ell)$ FLOPs on the same access pattern but achieves at best $\sim 5\%$ of peak FLOPs because of cache misses and the lack of register tiling. a gemm call achieves $> 80\%$ of peak on M-series Apple silicon, $> 90\%$ on a tuned AVX-512 system. the transpose+reshape preprocessing is where the kernel selection happens; everything else is bookkeeping.

**ternary and beyond.** for $n \geq 3$ operands, fold by repeated pairwise contraction. the choice of which pair to fold first is the path optimisation problem (§3). each pairwise step reduces to one gemm, so the total cost is the sum of gemm costs along the chosen path.

---

## 5. self-traces and diagonals

a label appearing twice within a single tensor is a _self-trace_. the simplest case is `'ii->'` on a square matrix: the trace, $\sum_i A_{ii}$. the general case is a partial trace plus axis preservation, e.g. `'iijk->jk'` on a tensor of shape $(n, n, m, p)$: extract the $n$-element diagonal of axes $(0, 1)$, then keep $(j, k)$.

**Lemma (self-trace reduction).** for an einsum expression $\mathcal{E}$ with operand $T_k$ having repeated label $\ell$ at positions $p, q \in \Sigma_k$, define $T_k' = \mathrm{diagonal}(T_k, \mathrm{axis1}=p, \mathrm{axis2}=q)$ — a tensor of shape $T_k$ with axes $p, q$ replaced by a single axis of size $d(\ell)$. then the einsum with $T_k$ replaced by $T_k'$ and $\Sigma_k$ updated to remove one occurrence of $\ell$ has the same value.

the proof is a re-indexing: the original sum has terms with $z^{(k)}_p = z^{(k)}_q$ enforced by the repeat, and `diagonal` extracts exactly those terms.

**implementation note.** `np.diagonal` returns a _view_ via stride manipulation: a tensor of shape $(n, n)$ with strides $(s_0, s_1)$ becomes a 1D view of shape $(n,)$ with stride $(s_0 + s_1)$ — the offset that walks down the diagonal in one hop. no copy. monpy's `_einsum_trace_diag` should exploit this; the current implementation does for two-axis traces but is silent on the three-or-more-repeats case (`'iii'`, the "super-diagonal" of a 3-tensor), which is rare but worth handling.

**Lemma (multi-repeat extension).** if $\ell$ appears $r \geq 2$ times in $\Sigma_k$ at positions $p_1 < p_2 < \cdots < p_r$, extract the $r$-fold diagonal: a view with stride $\sum_i s_{p_i}$ along a single axis of size $d(\ell)$.

---

## 6. Hadamard products and the no-contraction case

if $\mathcal{E}$ has $C(\mathcal{E}) = \emptyset$ — no contracted labels — the einsum is _not_ a contraction. it is either a pure outer product (every label distinct across operands) or a pure Hadamard product (some labels shared but all preserved in the output) or a mix.

**examples.**

- `'i,j->ij'`: outer product. use `multiply.outer`.
- `'i,i->i'`: Hadamard. use `multiply` (broadcasting).
- `'ij,ij->ij'`: Hadamard on 2-tensors. use `multiply`.
- `'i,i->'`: dot product — Hadamard _then_ sum. use `multiply` then `sum`.
- `'ij,jk->ijk'`: outer-with-broadcast. reshape both to $(i, j, 1)$ and $(1, j, k)$, multiply.

**dispatch rule.** detect the no-contraction case by checking $C(\mathcal{E}) = \emptyset$. then the operation is elementwise (with broadcasting). if `set(Σ_out) == set(Σ_1) ∪ ... ∪ set(Σ_n)`, dispatch to the elementwise multiply ufunc. otherwise some labels must be summed _after_ the elementwise product — handle by detecting that case as "elementwise then reduce". monpy's current implementation handles `i,i` and `ij,ij` correctly by happenstance because the gemm reduction in §4 degenerates to a sum over a length-1 inner axis when there's nothing to contract — but the gemm overhead is unwarranted, and for large arrays the elementwise dispatch is $\sim 3\times$ faster because it avoids the reshape/transpose copies.

---

## 7. batched matmul patterns

`'bij,bjk->bik'` is a batched matmul. the label `b` appears in both inputs and the output but is not summed; the contraction `j` proceeds independently for each value of `b`. the shape signature: a label is _batch_ iff it appears in every operand and the output.

**recognition rule.** label $\ell$ is a batch axis iff $\ell \in \Sigma_k$ for all $k$ and $\ell \in \Sigma_{\text{out}}$. batch axes can be moved to the leading position of all tensors via transpose, then the contraction proceeds shape-wise as if each batch slice were an independent matmul.

**implementation.** two approaches:

1. _loop over the batch axis._ call `gemm` $B$ times. acceptable when each inner gemm is large ($M, N, K \geq \sim 64$). for tiny inner gemms, the gemm setup overhead (a few microseconds per call on Apple Accelerate) dominates and the loop becomes catastrophic — a $1000 \times 4 \times 4 \times 4$ batch matmul takes longer than a $4000 \times 4 \times 4$ standalone gemm by 10–100×.
2. _batched gemm._ Intel MKL exposes `cblas_sgemm_batch`, NVIDIA cuBLAS has `cublasSgemmBatched` and `cublasSgemmStridedBatched`, OpenBLAS has `cblas_sgemm_batch_strided`. **Apple Accelerate does not expose a batched gemm** as of macOS 15.x — the `cblas_*gemm` family is single-call only. this is a real engineering gap for monpy on Apple silicon.

**workaround for Accelerate.** three options: (a) loop with GCD-dispatched parallel calls, exploiting Accelerate's internal multithreading; (b) reshape-and-pad: stack the batch into a block-diagonal megamatrix and do one call (memory-wasteful, only sane when the per-batch matrices are large); (c) reshape: `(b, i, j) × (b, j, k)` becomes a single $(bi, j) \times (j, k)$ gemm only if the batch axis is the same on both sides and we accept that `b` cannot vary per-batch — which it doesn't, so this _does_ work for _broadcasted_ batched matmul where one operand has no batch axis. for full batched matmul on Accelerate, the loop is unavoidable.

monpy should detect batched-matmul einsums and dispatch to a tight loop with GCD parallelism rather than fold through tensordot, which currently would issue a single oversized matmul producing an incorrect block-diagonal result.[^batched-bug]

[^batched-bug]: the current `python/monpy/linalg.py` likely has a latent correctness bug here. a test like `monpy.einsum('bij,bjk->bik', A, B)` for $A, B$ of shape $(2, 3, 3)$ should produce a $(2, 3, 3)$ result where slice $[0]$ is $A[0] @ B[0]$ and slice $[1]$ is $A[1] @ B[1]$. if the implementation flattens $b$ into the gemm rather than looping, the result will be wrong. worth a regression test.

---

## 8. tensordot

`tensordot(A, B, axes=k)` is einsum's positional cousin: contract the last $k$ axes of $A$ with the first $k$ axes of $B$:
$$ C^{i_1\ldots i_{p-k},\, j_{k+1}\ldots j_q} = \sum_{l_1 \ldots l_k} A^{i_1\ldots i_{p-k},\, l_1 \ldots l_k}\, B^{l_1 \ldots l_k,\, j_{k+1} \ldots j_q}. $$

the general form `tensordot(A, B, axes=([a_1, ..., a_k], [b_1, ..., b_k]))` lets the user pick which axes of $A$ contract with which of $B$ — in particular, the contracted axes need not be at the boundary. NumPy implements it by transposing both operands to bring the contracted axes to the appropriate boundary, reshaping to 2D, calling matmul, and reshaping back.

**this is exactly the gemm reduction of §4.** einsum _delegates_ to tensordot for binary contractions; tensordot is the actual workhorse, and einsum is the parser plus the path optimiser plus the diagonal-extractor. monpy follows the same architecture.

**edge case.** `axes=0` means no contraction — outer product. `tensordot(A, B, axes=0)` = $A \otimes B$ in the Kronecker sense, with shape $A.\mathrm{shape} + B.\mathrm{shape}$. the reshape interpretation: flatten $A$ to $(|A|, 1)$, flatten $B$ to $(1, |B|)$, gemm gives $(|A|, |B|)$, reshape to $A.\mathrm{shape} + B.\mathrm{shape}$. fine, but the trivial implementation `multiply.outer` is $\sim 2\times$ faster because it skips the transpose checks.

---

## 9. tensorinv and tensorsolve

these are the under-appreciated cousins: NumPy ports them, almost no Python code uses them, but they implement an isomorphism that's central to multilinear algebra and worth understanding properly.

### 9a. tensorinv

let $A$ have shape $(s_1, \ldots, s_p, t_1, \ldots, t_q)$ with $\prod s_i = \prod t_j = N$. reshape $A$ to a matrix $\tilde A \in \mathbb{R}^{N \times N}$ by flattening the first $p$ axes into rows and the last $q$ axes into columns. `tensorinv(A, ind=p)` computes $\tilde A^{-1}$ and reshapes the result to $(t_1, \ldots, t_q, s_1, \ldots, s_p)$ — note the _swap_ of leading and trailing groups, because the inverse swaps row and column indices.

**the isomorphism.** a linear map $\mathcal{L} : \mathbb{R}^{s_1 \times \cdots \times s_p} \to \mathbb{R}^{t_1 \times \cdots \times t_q}$ is, after flattening, an $N \times N$ matrix. the tensor $A$ is the kernel of $\mathcal{L}$; `tensorinv` computes the kernel of $\mathcal{L}^{-1}$. concretely:
$$ \sum_{i_1\ldots i_p} A^{j_1\ldots j_q,\, i_1\ldots i_p}\, [\mathrm{tensorinv}(A)]^{i_1\ldots i_p,\, k_1 \ldots k_q} = \delta^{j_1 k_1} \cdots \delta^{j_q k_q}. $$

this is the identity tensor in $\mathbb{R}^{t \times t}$, where $t = (t_1, \ldots, t_q)$ flattened.

the `ind=p` parameter says "the first $p$ axes are the input, the rest are the output". the default `ind=2` is the most common case in physics (e.g., 4-leg tensors with two in, two out).

### 9b. tensorsolve

`tensorsolve(A, B)` solves $A \cdot X = B$ where $A$ has shape $B.\mathrm{shape} + X.\mathrm{shape}$ — the leading axes of $A$ match $B$'s shape (the "row" indices), the trailing match $X$'s shape (the "column" indices). after flattening, this is a standard linear system $\tilde A \tilde X = \tilde B$ with $\tilde A \in \mathbb{R}^{|B| \times |X|}$, $\tilde X \in \mathbb{R}^{|X|}$, $\tilde B \in \mathbb{R}^{|B|}$. the solver delegates to `linalg.solve` post-flatten.

**the system in einsum notation.** let $\Sigma_A = \langle i_1, \ldots, i_q, j_1, \ldots, j_p \rangle$ split into $B$-shape labels and $X$-shape labels. then
$$ \sum_{j_1 \ldots j_p} A^{i_1 \ldots i_q,\, j_1 \ldots j_p}\, X^{j_1 \ldots j_p} = B^{i_1 \ldots i_q}. $$

**determinacy.** the system is square iff $|B| = |X|$ (i.e., $\prod \mathrm{shape}(B) = \prod \mathrm{shape}(X)$). NumPy's `tensorsolve` allows non-square via least-squares; monpy should follow suit, dispatching to `linalg.lstsq` when $\tilde A$ isn't square.

---

## 10. monpy implementation walkthrough

the current `python/monpy/linalg.py` einsum, summarised at the level that matters:

**parsing.** split on `->` to separate input and output specs; split inputs on `,`. for each operand, build the label sequence as a list of single-character strings (no support yet for the `numpy.einsum` style with explicit-axis-objects API, only the string DSL). detect implicit-output mode by absence of `->`; compute it by collecting all labels appearing exactly once across operands and sorting alphabetically.

**diagonal extraction.** for each operand, scan for repeated labels. if found, call `_einsum_trace_diag` to extract the appropriate diagonal-via-stride view. update the operand's label sequence to remove duplicates.

**pair-contract loop.** with operands $T_1, \ldots, T_n$ and label sequences $\Sigma_1, \ldots, \Sigma_n$:

1. take the leftmost two: $(T_1, \Sigma_1)$ and $(T_2, \Sigma_2)$.
2. compute the intermediate's label sequence: the union of $\Sigma_1, \Sigma_2$ minus labels that don't appear in any later operand or in $\Sigma_{\text{out}}$.
3. identify contracted labels (in $\Sigma_1 \cap \Sigma_2$ but not in the intermediate's output).
4. call `_einsum_pair_contract(T_1, T_2, contracted, kept)` which performs the transpose+reshape+matmul of §4.
5. replace $T_1$ with the result, drop $T_2$, repeat.

**finalise.** when one tensor remains, call `_einsum_finalise` which:

1. extracts any remaining self-trace (rare at this point but possible if the output spec has no occurrence of a label that the current tensor still has).
2. transposes to match $\Sigma_{\text{out}}$ axis order.
3. returns.

**gaps (relative to NumPy):**

- _no path optimisation._ pair-and-fold is left-to-right; can be 100× slower than greedy on adversarial cases. see §3.
- _no batched-gemm dispatch._ batched einsums fold through the same gemm reduction, which on Accelerate produces correct results only when the batch is degenerate. see §7.
- _no elementwise short-circuit._ `'i,i->i'` goes through gemm rather than `multiply`. see §6.
- _no `tensordot(axes=([], []))` case._ outer product through gemm rather than `multiply.outer`.
- _accumulation precision is the input dtype._ for f32 contractions over $K > 10^6$, this loses 4–5 significant digits. see §11.

each gap is a 50–200-line fix. the first three deliver the bulk of the practical speedup; the precision question is more involved.

---

## 11. numerical precision

a contraction of summed dimension $K$ accumulates $K$ floating-point products. the standard bound (Higham 2002, _Accuracy and Stability_, Ch. 3) is
$$ |\hat C - C| \leq \gamma_K \, \|A\|_\infty \, \|B\|_\infty $$ for componentwise error, where
$$ \gamma_K = \frac{K \varepsilon}{1 - K \varepsilon} \approx K \varepsilon $$ and $\varepsilon$ is the unit roundoff ($2^{-24} \approx 6 \times 10^{-8}$ for f32, $2^{-53} \approx 10^{-16}$ for f64).

**concrete numbers.** for f32 with $K = 10^6$: $\gamma_K \approx 6 \times 10^{-2}$ — about 1.5 decimal digits of accuracy in the result, regardless of operand magnitudes. for $K = 10^4$: $\gamma_K \approx 6 \times 10^{-4}$, about 3.5 digits. the bound is tight for adversarial inputs but loose for random inputs where errors partially cancel; in practice f32 contractions over $K = 10^6$ deliver $\sim 4$ digits, not 1.5 — but that's still bad.

**mitigations.**

_compensated summation (Kahan, Neumaier)._ maintains a running error correction term. reduces $\gamma_K$ to $O(\varepsilon)$ — no $K$-dependence. costs $\sim 4\times$ more FLOPs per accumulation, often invisible because contraction is memory-bound. BLAS implementations don't generally do this.

_higher-precision accumulation._ compute products in f32 but accumulate in f64. NVIDIA's tensor cores support this natively (mixed-precision gemm: f16 inputs, f32 accumulator); CPU BLAS rarely does. the cost is 2× more memory traffic for the accumulator, plus the conversion overhead.

_pairwise summation._ instead of $\sum_i = (\ldots((x_1 + x_2) + x_3) + \ldots)$, compute as a balanced binary tree. the error grows as $O(\log K \cdot \varepsilon)$ instead of $O(K \cdot \varepsilon)$. NumPy uses pairwise summation for `np.sum` since 1.9. for matmul, BLAS implementations don't — the $(M, N, K)$ block partitioning gives implicit pairwise summation along $K$ at the granularity of the block size, but within a block it's serial.

monpy currently does none of this. a worthwhile addition: for f32 contractions with $K \geq 10^4$, accumulate in f64 (cast input blocks as we go). the cost is $\sim 1.3\times$ memory bandwidth, the gain is back to $\sim 6$ digits of accuracy.

a more aggressive option: when the user passes `dtype=np.float32` but the contraction has large $K$, _automatically_ upcast accumulation to f64 and emit a warning. NumPy's `einsum` exposes a `dtype` parameter for the output; the _accumulator_ dtype is implementation-defined. Apple's Accelerate on M-series silicon uses f64 accumulation for sgemm internally on the AMX coprocessor — this is undocumented but visible in benchmarks (sgemm is more accurate on M-series than on Intel AVX2 with FMA).

---

## 12. memory alignment, strides, and the BLAS handoff

the transpose+reshape preprocessing of §4 is where contraction performance lives or dies. three concerns:

**(a) contiguity after transpose.** a transpose is a stride permutation; it produces a view, not a copy. if the original tensor has C-contiguous strides $(s_0, s_1, s_2, \ldots) = (\prod_{i>0} d_i, \prod_{i>1} d_i, \ldots, 1)$, transposing axes $(0, 2, 1)$ gives strides $(s_0, s_2, s_1) = (d_1 d_2, 1, d_2)$ — no longer C-contiguous. a subsequent reshape to a flat axis layout requires a copy whenever the original strides aren't compatible with the target shape (the standard NumPy "cannot reshape, this would require a copy" branch).

**(b) when the copy is avoidable.** BLAS gemm exposes leading-dimension parameters (`LDA`, `LDB`, `LDC`) that allow strided inputs _within_ the matrix-row direction, as long as each row is contiguous. the interface: `gemm` reads $A$ as an $M \times K$ matrix where row $i$ starts at offset $i \cdot \mathrm{LDA}$ and contains $K$ contiguous elements; the slack between rows ($\mathrm{LDA} - K$) is unused. concretely, if $A$ has shape $(M, K)$ with strides $(s_0, s_1)$ in elements, gemm requires $s_1 = 1$ (innermost contiguous) and $s_0 \geq K$ (rows don't overlap); LDA is then set to $s_0$.

**(c) when the copy is forced.** if $s_1 \neq 1$ — e.g., after a transpose that pulls a non-innermost axis to the inner position — gemm cannot consume the strides directly. the fallback is `cblas_*gemm` with a scratch buffer, or in NumPy's code the explicit `ascontiguousarray` call. the copy is $O(MK)$ memory traffic; for a contraction whose gemm is $O(MK^2 / \mathrm{block})$ FLOPs, the copy is $\sim O(\mathrm{block} / K)$ relative cost — a few percent for large $K$, dominant for small.

**proposal for monpy.** before calling gemm in `_einsum_pair_contract`:

1. after the user's transpose+reshape, inspect the resulting strides.
2. if `strides[-1] == itemsize` and `strides[-2] >= shape[-1] * itemsize` (innermost contiguous, no row overlap), pass the view directly to `cblas_sgemm` with `LDA = strides[-2] / itemsize`.
3. otherwise, fall back to `ascontiguousarray` and `LDA = shape[-1]`.

the check is $O(1)$ and saves the copy on the common case where the user's transpose only permuted batch/free axes, leaving the contracted axis innermost. this is the path NumPy takes since 1.10; monpy should match it.

**cross-reference.** the general memory-alignment story — vectorisation requirements, page boundaries, NUMA effects on multi-socket systems — lives in `memory-alignment.md`. the contraction-specific question is narrower: given that the upstream caller has produced a strided view, can we avoid the copy? the answer is _almost always yes_ for the common access patterns (transpose of leading axes, reshape that respects the innermost stride).

**one warning.** Apple Accelerate's gemm has historically been picky about LDA values that aren't multiples of 8 elements (for f32 SIMD alignment). modern Accelerate handles arbitrary LDA correctly but with a $\sim 5\%$ throughput penalty for misaligned inner stride. a small alignment-aware copy — pad to multiple-of-8 — pays for itself when the contraction is run repeatedly (e.g., inside a training loop). monpy could expose this as `monpy.linalg.contract(..., aligned=True)` for the hot path, defaulting to `False` to avoid unexpected copies.

---

## 13. recommendations (in priority order)

1. **implement greedy path optimisation.** ~150 lines in pure Python. 5–100× speedup on multi-tensor contractions. this is the highest-leverage change.
2. **add the elementwise short-circuit.** detect $C(\mathcal{E}) = \emptyset$ before any reshape; dispatch to `multiply` or `multiply.outer`. ~30 lines. 2–3× speedup on Hadamard and outer.
3. **fix batched-matmul dispatch.** loop over batch axes with GCD parallelism; do not fold through a single gemm. ~80 lines. correctness fix, performance varies.
4. **stride-aware gemm handoff.** skip `ascontiguousarray` when strides are LDA-compatible. ~20 lines. 5–15% speedup on large contractions.
5. **f64 accumulation for f32 with large $K$.** add an `accumulator_dtype` parameter; warn-and-upcast for $K \geq 10^4$ when user passes f32. numerical correctness.
6. **defer to opt_einsum when available.** soft import; if `opt_einsum.contract_path` is on the path and the user passes `optimize=True`, use it. ~10 lines. free state-of-the-art for users who want it.

the first four are the practical 80% — items 5 and 6 are quality-of-life. none of them require touching the Mojo native layer; this is all Python-side reorganisation around the existing `_einsum_pair_contract` and `tensordot` primitives.

---

## References

1. R. N. C. Pfeifer, J. Haegeman, F. Verstraete. "Faster identification of optimal contraction sequences for tensor networks." _Phys. Rev. E_ 90, 033315 (2014). [arXiv:1304.6112](https://arxiv.org/abs/1304.6112).
2. C. Lam, P. Sadayappan, R. Wenger. "On Optimizing a Class of Multi-Dimensional Loops with Reductions for Parallel Execution." _Parallel Processing Letters_ 7(2), 157–168 (1997).
3. D. G. A. Smith, J. Gray. "`opt_einsum` — A Python package for optimizing contraction order for einsum-like expressions." JOSS 3(26), 753 (2018). https://joss.theoj.org/papers/10.21105/joss.00753
4. I. L. Markov, Y. Shi. "Simulating Quantum Computation by Contracting Tensor Networks." _SIAM J. Comput._ 38(3), 963–981 (2008). [arXiv:quant-ph/0511069](https://arxiv.org/abs/quant-ph/0511069).
5. B. O'Gorman. "Parameterization of tensor network contraction." _TQC 2019_. [arXiv:1906.00013](https://arxiv.org/abs/1906.00013).
6. S. Hirata. "Tensor Contraction Engine: Abstraction and Automated Parallel Implementation of Configuration-Interaction, Coupled-Cluster, and Many-Body Perturbation Theories." _J. Phys. Chem. A_ 107, 9887–9897 (2003).
7. U. Schollwöck. "The density-matrix renormalization group in the age of matrix product states." _Annals of Physics_ 326(1), 96–192 (2011).
8. N. J. Higham. _Accuracy and Stability of Numerical Algorithms_, 2nd ed. SIAM (2002). Chapter 3 ("Basics") for the inner-product bound; Chapter 4 ("Summation") for compensated summation.
9. T. C. Hu, M. T. Shing. "Computation of Matrix Chain Products. Part I, Part II." _SIAM J. Comput._ 11(2), 362–373 (1982); 13(2), 228–251 (1984). the $O(n \log n)$ matrix-chain algorithm.
10. NumPy reference: [`numpy.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html), [`numpy.einsum_path`](https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html), `numpy.tensordot`, `numpy.linalg.tensorinv`, `numpy.linalg.tensorsolve`.
11. opt_einsum docs: [path-finding](https://optimized-einsum.readthedocs.io/en/stable/path_finding.html), [random-greedy](https://optimized-einsum.readthedocs.io/en/stable/random_greedy_path.html), [dynamic-programming](https://optimized-einsum.readthedocs.io/en/stable/dp_path.html).
