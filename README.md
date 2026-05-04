# monpy [ALPHA]

_not ready for production_

Mojo-native array runtime with a NumPy-API interface.

See [docs/architecture.md](docs/architecture.md) for the layer split.

## features

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

The project is intentionally not a drop-in NumPy replacement.

This is educational first, as a part for me to learn Mojo.

The current targets includes CPU-only, array-API-shaped subset with Mojo-owned storage and native kernels for creation, views, broadcasting, elementwise math, reductions, and matmul.

NumPy is used by the tests as an oracle and by `ndarray.__array__` as an explicit conversion target.

There is a small [layout.mojo](src/layout.mojo) where it applies some heuristics and algebra.

majority of logics can be found inside [create.mojo](src/create.mojo), and an accelerate backend for MacOS can be found under [accelerate.mojo](src/accelerate.mojo)

Some of the major elementwise ops can be found under [elementwise.mojo](src/elementwise.mojo)

## local development

You should use [mohaus](https://github.com/aarnphm/mohaus), and install [Mojo](https://docs.modular.com/mojo/manual/install/)

```bash
uv venv --python 3.11 --managed-python --clear
uv pip install mohaus --index https://aarnphm.github.io/mohaus/simple
uv pip install --no-build-isolation -e ".[dev]"
```

For running verification and benchmarks

```bash
python -m pytest tests/python
python benchmarks/bench_array_core.py --rounds 3 --loop 3
python benchmarks/bench_array_core.py --format csv --no-progress
```


## notes

If you have a custom built Mojo, set `MOHAUS_MOJO` envvar:

```bash
MOHAUS_MOJO=/path/bin/mojo
```

## acknowledgement

- CuTEDSL Layout algebra
- Modular's `LayoutTensor`
- [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo)
