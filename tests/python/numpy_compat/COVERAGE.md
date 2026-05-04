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
| supported dtype discovery and explicit casts | covered | `test_array_coercion.py`, `test_promotions.py` |
| unsupported dtype families: complex, object, string, structured, unsigned, narrow ints | blocked | `test_array_coercion.py`, `test_promotions.py` |
| importing NumPy arrays into monpy | blocked | `test_array_coercion.py` |
| array interface export and NumPy conversion | covered | `test_array_interface.py` |
| dlpack import/export | blocked | `test_array_interface.py` |
| integer, slice, reverse-slice, ellipsis, and 0-d indexing | covered | `test_indexing.py` |
| `newaxis`, boolean indexing, and integer-array fancy indexing | blocked | `test_indexing.py` |
| array creation: `empty`, `zeros`, `ones`, `full`, `arange`, `linspace` | covered | `test_numeric.py` |
| broadcasting, `where`, reshape, transpose, matrix transpose | covered | `test_numeric.py` |
| axis-none reductions | covered | `test_numeric.py` |
| axis/out/keepdims/where reductions | blocked | `test_numeric.py` |
| 1-d and 2-d matmul | covered | `test_numeric.py` |
| higher-rank matmul | blocked | `test_numeric.py` |
| float32/int64 mixed array promotion | xfail | `test_promotions.py` |
| `log(-inf)` NumPy parity | xfail | `test_umath.py` |
| ufunc objects, `out=`, `where=`, casting controls, reductions, accumulations | deferred | `test_umath.py` |
| broad dtype machinery, string/object/structured arrays, datetime, c-api tests | deferred | none |
