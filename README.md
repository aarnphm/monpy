# monpy [ALPHA]

_not ready for production_

Mojo-native array runtime with a NumPy-compatible Python interface.

See [docs/architecture.md](docs/architecture.md) for the layer split.
See [docs/api-surface.md](docs/api-surface.md) for the JAX-first primitive
surface, and [docs/mojo-setup-design.md](docs/mojo-setup-design.md) for the
Mojo setup and runtime contract.

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

The current target is a CPU-first, array-API-shaped subset with Mojo-owned
storage and native kernels for creation, views, broadcasting, elementwise math,
reductions, and matmul. The architecture keeps the Python API on a single
lazy primitive spine so ordinary calls and `@monpy.jit` do not grow separate
meanings for the same operation.

NumPy is used by the tests as an oracle and by `ndarray.__array__` as an explicit conversion target.

There is a small CuTe-inspired layout algebra under [src/cute/](src/cute/).

Creation and high-level operation entrypoints live under [src/create/](src/create/)
and [src/array/](src/array/). Elementwise dispatch lives under
[src/elementwise/](src/elementwise/), with typed kernels in
[src/elementwise/kernels/](src/elementwise/kernels/). The BLAS/LAPACK backend
lives in [accelerate.mojo](src/accelerate.mojo), which routes to Apple
Accelerate on macOS and OpenBLAS / netlib on Linux based on the comptime
target.

Storage, metadata, buffer ingress, and runtime domain tags live in
[storage.mojo](src/storage.mojo), [metadata.mojo](src/metadata.mojo),
[buffer.mojo](src/buffer.mojo), and [domain.mojo](src/domain.mojo).

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

You should use [mohaus](https://github.com/aarnphm/mohaus) with Modular's Python wheel suite.

```bash
uv venv --python 3.11 --managed-python --clear
uv pip install mohaus modular \
  --index-url https://whl.modular.com/nightly/simple/ \
  --extra-index-url https://aarnphm.github.io/mohaus/simple \
  --extra-index-url https://pypi.org/simple \
  --prerelease allow
uv pip install -e ".[dev]" \
  --index-url https://whl.modular.com/nightly/simple/ \
  --extra-index-url https://aarnphm.github.io/mohaus/simple \
  --extra-index-url https://pypi.org/simple \
  --prerelease allow
```

For running verification and benchmarks

```bash
# to run the tiny monpy GPT example
python examples/tiny_gpt.py
# to run benchmark
monpy-bench --types all --format csv --loops 3 --no-progress
# to run tests
uv run --no-sync pytest tests/python
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
- [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo), vendored under `vendor/NuMojo` for optional benchmarks under `Apache-2.0 WITH LLVM-exception`
- [numpy](https://github.com/numpy/numpy)
