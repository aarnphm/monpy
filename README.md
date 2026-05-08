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

and enjoy Mojo perf! (hopefully)

## notes

The project is intentionally yet to be a drop-in NumPy replacement.

This is educational first, as a part for me to learn Mojo.

The current targets includes CPU-only, array-API-shaped subset with Mojo-owned storage and native kernels for creation, views, broadcasting, elementwise math, reductions, and matmul.

NumPy is used by the tests as an oracle and by `ndarray.__array__` as an explicit conversion target.

There is a small [layout.mojo](src/cute/layout.mojo) where it applies some CuTe-inspired layout design and relevant algebra.

majority of logics can be found inside [create.mojo](src/create.mojo), and the BLAS/LAPACK backend lives in [accelerate.mojo](src/accelerate.mojo) (it routes to Apple Accelerate on macOS and OpenBLAS / netlib on Linux based on the comptime target)

Some of the major elementwise ops can be found under [elementwise.mojo](src/elementwise.mojo)

We also have a few storage and relevant domains that can be found under [storage.mojo](src/storage.mojo)

## supported platforms

| platform        | BLAS / LAPACK     | vector math (sin/cos/exp/log) |
| --------------- | ----------------- | ----------------------------- |
| macOS (arm64)   | Apple Accelerate  | Apple vForce (vvsinf etc.)    |
| Linux (x86_64)  | OpenBLAS / netlib | SIMD via std.math (libm)      |
| Linux (aarch64) | OpenBLAS / netlib | SIMD via std.math (libm)      |

On Linux, install OpenBLAS and LAPACK before building:

```bash
# Ubuntu / Debian
sudo apt install libopenblas-dev liblapack-dev
# Fedora / RHEL
sudo dnf install openblas-devel lapack-devel
# Arch
sudo pacman -S openblas lapack
```

## local development

You should use [mohaus](https://github.com/aarnphm/mohaus), and install [Mojo](https://docs.modular.com/mojo/manual/install/)

```bash
uv venv --python 3.11 --managed-python --clear
uv pip install mohaus --index https://aarnphm.github.io/mohaus/simple
uv pip install --no-build-isolation -e ".[dev]"
```

For running verification and benchmarks

```bash
# to run the tiny monpy GPT example
python examples/tiny_gpt.py
# to run benchmark
monpy-bench --types all --format csv --loops 3 --no-progress
# to run tests
python -m pytest tests/python
```

Benchmark runs write `results/yyyy-mm-dd/results.<format>` and `results/yyyy-mm-dd/manifest.json` by default.
See [docs/benchmarks.md](docs/benchmarks.md) for suite types, output formats, saved manifests, and CI comment posting.

## notes

If you have a custom built Mojo, set `MOHAUS_MOJO` envvar:

```bash
MOHAUS_MOJO=/path/bin/mojo
```

## acknowledgement

- CuTEDSL Layout algebra
- Modular's `LayoutTensor`
- [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo)
- [numpy](https://github.com/numpy/numpy)
