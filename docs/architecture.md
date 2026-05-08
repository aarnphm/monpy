# monpy architecture notes

monpy should be a mojo array library with numpy-shaped python APIs.

## layers

- `src/lib.mojo` is only the cpython extension boundary. it owns `PyInit__native`, builds `PythonModuleBuilder("_native")`, registers `Array`, and binds python-facing function names into `monpy._native`.
- `src/domain.mojo` owns compact dtype, op, reduction, unary-op, casting, and backend codes, plus the 14-dtype registry, the NxN promotion table, and the can_cast rules.
- `src/storage.mojo` owns the storage record, refcounting, managed allocation, and external non-owning allocation records.
- `src/buffer.mojo` is the single-FFI cpython buffer-protocol bridge (`asarray_from_buffer_ops`). one `PyObject_GetBuffer` call replaces the older eight-step `__array_interface__` walk, and the CPython buffer function pointers are cached in `MONPY_BUFFER_FUNCTIONS` so hot imports do not call dyld symbol lookup per array crossing.
- `src/cute/` is the vendorred CuTe-style layout algebra package (cpu-only subset of NVIDIA CUTLASS's `cute/`). split into:
  - `int_tuple.mojo` — recursive `IntTuple` ADT (leaf or list of IntTuples), traversal helpers (`flatten`, `product`, `prefix_product`, `inner_product`), `crd2idx`/`idx2crd`. `layout.mojo` — `Layout = (shape, stride)` struct, ctors (`row_major`, `col_major`, `strided`, `ordered`), basic queries (`__call__`, `idx2crd`, `size`, `cosize`, `__getitem__`). `functional.mojo` — algebra (`coalesce`, `select`, `transpose`, `composition`, `complement`, `logical_divide`). `iter.mojo` — `LayoutIter` (single layout) and `MultiLayoutIter` (N broadcasted operands), with both byte-cursor and `element_index()` accessors.
  - named `cute` to avoid collision w/ `std.algorithm` and `max/kernels/src/layout` on Mojo's import path. see also [[cute-layout]]
- `src/array.mojo` owns the `Array` record, dtype-typed scalar accessors (`get_physical_*` / `set_physical_*` per dtype, including the complex re/imag pairs), metadata methods, shape/stride helpers, c/f-contig probes, native cast-copy dispatch for supported dtype pairs, dynamic-rank fallback addressing, and the `Array ↔ Layout` adapter (`as_layout`, `array_with_layout`) that bridges `Array` to the `cute` package's primitives.
  - dtype metadata and promotion rules delegate back to `domain.mojo`.
  - the `Layout` is linear (no constant offset);
  - the offset rides on `Array.offset_elems`, mirroring CuTe's `Layout` vs `Tensor` split.
- `src/create.mojo` owns every python-callable op dispatcher: creation (`empty` / `full` / `arange` / `linspace` / `eye` / `tri` / `concatenate` / `pad_constant`), elementwise (`unary_ops` / `unary_preserve_ops` / `binary_ops` / `binary_into_ops` / `binary_scalar_ops` / `compare_ops` / `logical_ops` / `predicate_ops`), reductions (`reduce_ops` / `reduce_axis_ops`), matmul (`matmul_ops`), linalg (`solve_ops` / `inv_ops` / `det_ops` / `qr_ops` / `cholesky_ops` / `eigh_ops` / `eig_ops` / `svd_ops` / `lstsq_ops`), shape ops (`reshape` / `transpose` / `slice` / `broadcast_to` / `expand_dims` / `flip` / `diagonal` / `trace`), casting (`astype_ops` / `materialize_c_contiguous_ops`), python-side dtype helpers. a top-level `linalg.mojo` would shadow Mojo's stdlib `linalg`, so the linalg dispatchers live here.
- `src/elementwise.mojo` owns numeric loops, contiguous fast paths via the typed-vec dispatcher (`apply_binary_typed_vec[dtype, width]`, `apply_unary_preserve_typed_vec[dtype, width]`) covering f16 / f32 / f64 and the eight integer dtypes, strided fallbacks, fused elementwise kernels (`sin_add_mul`), the complex kernels (schoolbook FMA multiply, Smith-algorithm divide, conjugate / negate / square preserve, plus `apply_unary_complex_f64` for transcendentals via Euler identities), reductions, matmul dispatch helpers, and the LAPACK call sites (`lapack_qr_reduced_*_into`, `lapack_cholesky_*_into`, `lapack_eigh_*_into`, `lapack_eig_*_real_into`, `lapack_svd_*_into`, `lapack_lstsq_*_into`).
- `src/accelerate.mojo` owns Apple Accelerate ffi shims only. BLAS: `cblas_sgemm` / `dgemm` / `cgemm` / `zgemm`. LAPACK: `getrf` / `gesv` / `geqrf` + `orgqr` / `potrf` / `syev` / `geev` / `gesdd` / `gelsd` for f32 and f64. each wrapper follows the F77 calling convention — pointers everywhere, character flags as `Int8` byte pointers, workspace queries via `LWORK = -1`.
- `python/monpy` is the python API facade. it parses python objects, cpython buffers, array-interface exporters, dlpack cpu producers, keyword arguments, and numpy-shaped ergonomics, then delegates implemented work into mojo. NEP 50 weak-scalar dispatch lives here; the registry mirrors numpy's dtype metadata (`kind`, `itemsize`, `alignment`, `byteorder`, `format`, `scalar_type`).
- `python/monpy/runtime/ops_numpy.py` is the explicit numpy interop boundary. core `monpy` does not import numpy; numpy dtype aliases, `to_numpy(...)`, and numpy-aware `asarray(...)` live in this module.
- `python/monpy/linalg.py` is the numpy-shaped linear algebra namespace. wraps the LAPACK dispatchers and adds python-level `pinv` / `matrix_rank` / `einsum` / `tensorinv` / `tensorsolve`. `einsum` parses the subscript string, folds pairwise contractions through `tensordot` (transpose + reshape + matmul), and unpacks LAPACK's compressed conjugate-pair `WR` / `WI` representation when `eig` returns a complex spectrum.
- `python/monpy/array_api.py` is the standards-shaped namespace and re-exports the same `linalg` module for the currently supported surface.
- `python/monumpy` is a compatibility shim that re-exports `monpy`.

## supported surface

| surface   | coverage                                                                                                                                                                        |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| dtypes    | `bool`, `int{8,16,32,64}`, `uint{8,16,32,64}`, `float{16,32,64}`, `complex{64,128}` (14 total)                                                                                  |
| promotion | NEP 50 deterministic; weak-scalar dispatch in python; native NxN table with kind-decomposed structure                                                                           |
| ufuncs    | full elementwise (real + complex), reductions, comparisons, predicates, `where`, fused `sin_add_mul`                                                                            |
| matmul    | `sgemm` / `dgemm` Accelerate path for positive-stride dense f32/f64 rank-2 (c-contig + f-contig + transposed views); `cgemm` / `zgemm` for complex; scalar mojo fallback        |
| linalg    | `solve` / `inv` / `det` / `qr` / `cholesky` / `eigh` / `eig` / `svd` / `lstsq` via Accelerate LAPACK; `pinv` / `matrix_rank` / `einsum` / `tensorinv` / `tensorsolve` in python |
| creation  | `empty` / `full` / `zeros` / `ones` / `arange` / `linspace` / `eye` / `tri` / `tril` / `triu` / `concatenate` / `pad` (constant mode) native                                    |
| views     | slicing, reshape, transpose, broadcast, expand_dims, flip, diagonal — all stride-only, no copies                                                                                |
| io        | `__array_interface__` export, dlpack round-trips, explicit `runtime.ops_numpy` conversion, and cpython buffer protocol fast paths via `buffer.mojo`                             |

v1 non-goals: `numpy.random`, `numpy.fft`, `numpy.ma`, `numpy.strings`, `numpy.io`. see [[numpy-port-gaps]].

## policy

- implemented array operations must not call numpy internally.
- numpy is allowed as a test oracle and inside the explicit `runtime.ops_numpy` cpu interchange boundary. core array import uses the cpython buffer protocol first, then direct array-interface parsing, and does not call numpy internally.
- core dtype resolution may lazy-load `runtime.ops_numpy` when the caller passes
  a concrete numpy dtype object or dtype class. that preserves existing
  frontend code like `asarray(x, dtype=np.float32)` while keeping plain
  `import monpy` and non-numpy code paths numpy-free.
- `copy=False` at an interchange boundary means storage sharing or a loud error; `copy=True` means a detached monpy-owned allocation; `copy=None` may copy only when dtype conversion or readonly memory makes zero-copy unsafe.
- views retain storage instead of copying raw pointers. external storage is non-owning in mojo and is kept alive by python owner slots.
- inserted-axis views use stride-zero native metadata and retain the same storage owner.
- unsupported numpy long-tail features should fail loudly with `NotImplementedError`, `BufferError`, or a narrow runtime error.
- cpu-only is the v1 device model.

## performance notes

- generic paths preserve dynamic-rank correctness with shape and stride metadata.
- fast paths should be added only when a dtype/layout/rank predicate makes the cheaper path obvious.
- the typed-vec dispatcher uses 32 B SIMD vectors uniformly — f64 lanes=4, f32 lanes=8, f16 lanes=16, i8 lanes=32. that's the compromise between NEON's 16 B and AVX-512's 64 B; pipelines as two NEON ops per logical step on Apple Silicon and sidesteps AVX-512 frequency throttling on x86.
- complex multiplication uses schoolbook FMA. complex division uses Smith's algorithm so $|c|^2 + |d|^2$ overflow doesn't poison representable quotients. complex transcendentals use Euler identities through `apply_unary_complex_f64`. proofs and branch-cut tables in [[research/complex-kernels]].
- f16 transcendentals (`atan2` / `hypot` / `copysign`) are gated off because Mojo can't legalise the half-precision libm symbols on Apple; the fallback is scalar f32-promote-demote. arithmetic (add / mul / fma) runs natively on Armv8.2-A NEON.
- `sin_add_mul(x, y, scalar)` is the first explicit fused expression kernel. the numpy-shaped `sin(x) + y * scalar` pattern lowers through a private python expression object and materializes through the same mojo fused kernel. benchmarks must force materialization so this does not become a fake python-only win.
- matmul uses Apple Accelerate for positive-stride dense macos f32/f64 rank-2 arrays, including c-contiguous and f-contiguous/transposed views, with scalar mojo as the portable fallback. `cgemm` / `zgemm` cover complex.
- `linalg` exposes aliases for matmul and matrix transpose, plus native `solve` / `inv` / `det` / `qr` / `cholesky` / `eigh` / `eig` / `svd` / `lstsq`. f32/f64 macos inputs use Accelerate LAPACK; unsupported accelerated paths fall back to portable partial-pivot LU. `eig` with real input + complex spectrum unpacks LAPACK's compressed `WR` / `WI` representation into `complex128` results in `linalg.py`.
- backend markers on native arrays (`used_accelerate`, `used_fused`, `backend_code`) let tests and benchmarks assert that specialized kernels actually ran.
- next levers: allocation reuse, `out=`, expression fusion, wider SIMD coverage on strided paths, aligned-load fast paths against a 64 B-aligned allocator. proposal in [[research/memory-alignment]].

see [[apple-backends]] for the apple silicon backend split. see [[ffi-marshaling]] for why the residual `asarray` / `from_dlpack` / `strided_view` / `array_copy` ratios are marshaling tax rather than kernel cost, and the two paths out (cpython buffer protocol or numpy c api). see [[cute-layout]] for the `src/cute/` package — vendored CuTe-style layout algebra used by view operations and (eventually) by tiled-kernel migrations off `physical_offset`. see [[research/index|research notes]] for write-ups on BLAS / LAPACK dispatch, complex kernels, CuTe layout algebra, NEP 50 promotion (and the not-a-lattice gotcha it surfaced), memory alignment, einsum contraction order, and SIMD vectorisation.
