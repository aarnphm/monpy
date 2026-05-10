---
title: meshgrid stride semantics
date: 2026-05-09
---

# meshgrid stride semantics

`meshgrid` is a shape/stride constructor with an optional materialisation step.
The May 9 benchmark caught the mismatch: monpy paid more Python overhead than
NumPy while still returning views for `copy=True`.

## contract

NumPy's public API says `meshgrid(*xi, copy=True, sparse=False, indexing="xy")`
accepts coordinate vectors and returns coordinate matrices. For two input
vectors with lengths `M` and `N`, `indexing="xy"` returns `(N, M)` outputs, and
`indexing="ij"` returns `(M, N)` outputs. `copy=False` is allowed to return
non-contiguous views, including dense broadcast views where several logical
indices hit the same memory address. `copy=True` returns independent arrays.

The source path is compact:

- reshape each input into an open-grid shape
- swap the first two open-grid shapes for `xy`
- broadcast unless `sparse=True`
- copy the outputs if `copy=True`

Use the same boundary in monpy: the native layer builds the view geometry; the
Python facade holds owner lifetimes and dispatches the two-vector fast path.

## dense two-vector proof

Let rank-1 `x` have length `M`, element stride `sx`, and offset `ox`. Let
rank-1 `y` have length `N`, element stride `sy`, and offset `oy`.

For `indexing="xy"`, dense `meshgrid(x, y, copy=False)` should expose:

| output | shape  | strides | element map                        |
| ------ | ------ | ------- | ---------------------------------- |
| `X`    | `N, M` | `0, sx` | `X[i, j] = storage_x[ox + j * sx]` |
| `Y`    | `N, M` | `sy, 0` | `Y[i, j] = storage_y[oy + i * sy]` |

For `indexing="ij"`, the natural axes stay in order:

| output | shape  | strides | element map                        |
| ------ | ------ | ------- | ---------------------------------- |
| `X`    | `M, N` | `sx, 0` | `X[i, j] = storage_x[ox + i * sx]` |
| `Y`    | `M, N` | `0, sy` | `Y[i, j] = storage_y[oy + j * sy]` |

**Proposition.** These formulas produce the same shape, stride, and
index-to-storage mapping as NumPy's open-grid reshape plus broadcast path for
all rank-1 inputs.

**Proof.** In `xy`, NumPy reshapes `x` to shape `(1, M)` with strides
`(M * sx, sx)` and `y` to `(N, 1)` with strides `(sy, sy)`. Broadcasting both
to `(N, M)` sets only the newly expanded axes to zero, giving `x -> (0, sx)`
and `y -> (sy, 0)`. `ij` is the same construction without swapping the first
two open-grid shapes. Negative strides survive because the formulas copy `sx`
and `sy` directly. Sliced inputs survive because each view inherits its input
data pointer and offset before reshaping.

## sparse two-vector proof

Sparse mode stops before dense broadcasting. For rank-1 inputs, it is just
reshape:

| indexing | output | shape  | strides      |
| -------- | ------ | ------ | ------------ |
| `xy`     | `X`    | `1, M` | `M * sx, sx` |
| `xy`     | `Y`    | `N, 1` | `sy, sy`     |
| `ij`     | `X`    | `M, 1` | `sx, sx`     |
| `ij`     | `Y`    | `1, N` | `N * sy, sy` |

The stride on any length-1 axis is not semantically observable for indexing,
but matching NumPy's normal reshape convention helps tests catch wrong view
construction.

## materialisation policy

`copy=True` should be implemented as `copy_c_contiguous(view)`, not as a
separate fill kernel unless profiling proves the copy loop is hot. This gives
three useful properties:

- one view formula covers contiguous, sliced, reversed, integer, float, and
  complex inputs
- dense `copy=True` gets independent C-contiguous outputs
- dense `copy=False` keeps zero-stride aliasing and external-buffer owner
  lifetimes through the Python wrapper base

For the benchmark row, this removes four public shape calls (`reshape` twice,
`broadcast_to` twice) and two Python shape normalisation passes. The native
helper constructs two views and copies them in one extension call.

## references

1. NumPy manual, `numpy.meshgrid`, stable API contract for `copy`, `sparse`,
   and `indexing`: <https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html>
2. NumPy source, `meshgrid` implementation in `numpy/lib/_function_base_impl.py`
   at v2.4.0: <https://github.com/numpy/numpy/blob/v2.4.0/numpy/lib/_function_base_impl.py#L5058-L5196>
