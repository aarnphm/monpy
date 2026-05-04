# monpy [ALPHA]

_not ready for production_

Mojo-native array runtime with a NumPy-API interface.

See [docs/architecture.md](docs/architecture.md) for the layer split. Short
version: `src/native.mojo` is the Mojo array library, `src/lib.mojo` is only the
CPython binding module, and `python/monpy`/`python/monumpy` provide the
NumPy-flavored Python surface.

Instead of

```python
import numpy as np

x = np.asarray([1, 2, 3], dtype=np.float32)
y = np.sin(x) + 2
```

You can do

```python
import monumpy as np
```

and enjoy Mojo perf!

## notes

The project is intentionally not a drop-in NumPy replacement. The v1 target is a
CPU-only, array-api-shaped subset with Mojo-owned storage and native kernels for
creation, views, broadcasting, elementwise math, reductions, and matmul.
NumPy is used by the tests as an oracle and by `ndarray.__array__` as an explicit conversion target.

## local development

You should use [mohaus](https://github.com/aarnphm/mohaus), and install [Mojo](https://docs.modular.com/mojo/manual/install/)

```bash
uv venv --python 3.11 --managed-python --clear
uv pip install -e git+https://github.com/aarnphm/mohaus
uv pip install --no-build-isolation -e ".[dev]"
python -m pytest tests/python
python benchmarks/bench_array_core.py
```

If you have a custom built Mojo, set `MOHAUS_MOJO` envvar:

```bash
MOHAUS_MOJO=/path/bin/mojo
```
