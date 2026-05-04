# numpy-derived compatibility fixtures

The tests in this directory are adapted behavioral fixtures inspired by NumPy's
public test suite, especially:

- `numpy/_core/tests/test_array_coercion.py`
- `numpy/_core/tests/test_array_interface.py`
- `numpy/_core/tests/test_indexing.py`
- `numpy/_core/tests/test_multiarray.py`
- `numpy/_core/tests/test_numeric.py`
- `numpy/_core/tests/test_ufunc.py`
- `numpy/_core/tests/test_umath.py`
- `numpy/_core/tests/test_dlpack.py`

They are intentionally not vendored wholesale. Each test is rewritten around
monpy's v1 scope: cpu-only arrays, bool/int64/float32/float64, array-api-shaped
semantics, numpy interop as a conversion target, and explicit skips/blockers for
unsupported numpy long-tail behavior.

NumPy is distributed under the BSD 3-Clause license:

```text
Copyright (c) 2005-2025, NumPy Developers.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name of the NumPy Developers nor the names of any contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.
```
