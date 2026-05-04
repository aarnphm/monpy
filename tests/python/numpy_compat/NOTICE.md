# numpy-derived compatibility fixtures

The tests in this directory are adapted behavioral fixtures inspired by NumPy's
public test suite. They are intentionally not vendored wholesale. Each test is
rewritten around monpy's v1 scope: cpu-only arrays, bool/int64/float32/float64,
array-api-shaped semantics, NumPy interop as a conversion target, and explicit
blockers for unsupported NumPy long-tail behavior.

The pinned upstream baseline is NumPy `v2.4.4`, release commit
`be93fe2960dbf49b4647f5783c66d967fb2c65b5`. The main donor files are:

- `numpy/_core/tests/test_array_coercion.py`
- `numpy/_core/tests/test_array_interface.py`
- `numpy/_core/tests/test_indexing.py`
- `numpy/_core/tests/test_multiarray.py`
- `numpy/_core/tests/test_numeric.py`
- `numpy/_core/tests/test_nep50_promotions.py`
- `numpy/_core/tests/test_ufunc.py`
- `numpy/_core/tests/test_umath.py`
- `numpy/_core/tests/test_dlpack.py`

NumPy is distributed under the BSD 3-Clause license. The full license text for
the pinned upstream baseline is copied in `LICENSE.numpy.txt`.
