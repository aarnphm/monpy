# numpy compatibility coverage

Pinned upstream baseline: NumPy `v2.4.4`, commit
`be93fe2960dbf49b4647f5783c66d967fb2c65b5`.

Statuses:

- `covered`: monpy matches NumPy for the adapted v1 behavior.
- `blocked`: monpy intentionally raises for a v1 gap.
- `xfail`: roadmap behavior is tracked with `pytest.mark.xfail(strict=True)`.
- `deferred`: upstream coverage is outside the current monpy surface.

| area | status | local coverage |
| --- | --- | --- |
| list, tuple, scalar, empty, and nested array construction | covered | `test_array_coercion.py` |
| supported dtype discovery, explicit casts, and astype cast matrix | covered | `test_array_coercion.py`, `test_promotions.py` |
| supported dtype metadata, promotion helpers, cast queries, kind queries, `finfo`, and `iinfo` | covered | `test_promotions.py` |
| unsupported dtype families: complex, object, string, structured, unsigned, narrow ints | blocked | `test_array_coercion.py`, `test_promotions.py` |
| importing NumPy arrays into monpy with `copy=` rules | covered | `test_array_coercion.py` |
| array interface export, NumPy conversion, and view-owner safety | covered | `test_array_interface.py` |
| dlpack cpu import/export with `copy=` rules | covered | `test_array_interface.py` |
| integer, slice, reverse-slice, ellipsis, `newaxis`, and 0-d indexing | covered | `test_indexing.py` |
| boolean indexing and integer-array fancy indexing | blocked | `test_indexing.py` |
| array creation: `empty`, `zeros`, `ones`, `full`, `*_like`, `arange`, `linspace`, `copy`, `ascontiguousarray` | covered | `test_numeric.py` |
| broadcasting, `where`, reshape, transpose, matrix transpose, `expand_dims`, view safety | covered | `test_numeric.py`, `test_indexing.py` |
| axis-none reductions | covered | `test_numeric.py` |
| axis/out/keepdims/where reductions | blocked | `test_numeric.py` |
| 1-d, 2-d, dense-transposed, f-contiguous, offset, and negative-stride matmul | covered | `test_numeric.py`, `test_linalg.py` |
| `diagonal` and `trace` for rank-2 arrays, including offset sweeps | covered | `test_linalg.py` |
| supported `linalg` namespace surface: matmul, matrix transpose, solve, inv, det with dtype/size sweeps | covered | `test_linalg.py`, `test_import_smoke.py` |
| higher-rank matmul | blocked | `test_numeric.py` |
| float32/int64 mixed array promotion | covered | `test_promotions.py` |
| `log(-inf)` NumPy parity | covered | `test_umath.py` |
| ufunc objects, `out=`, `where=`, casting controls, reductions, accumulations | deferred | `test_umath.py` |
| broad dtype machinery, string/object/structured arrays, datetime, c-api tests | deferred | none |
| import smoke for `monpy`, `monumpy`, `monpy.array_api`, `monpy.linalg`, and star import | covered | `test_import_smoke.py` |
