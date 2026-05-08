"""monpy's CuTe-style layout algebra.

scope:
- a subset of CUTLASS's CuTe layout algebra.
- **not** an import of `max/kernels/src/layout/` but built so the same algebra extends in that direction once we ship GPU kernels.
- powers numpy view operations where layout transformations apply:
  - reshape (composition)
  - transpose (select)
  - broadcast (stride-zero injection + coalesce)
  - slice (composition + offset delta)
  - tiled reductions / matmul (logical_divide)

design goal:
- CPU first, GPU-portable foundation
- `IntTuple`, `Layout`, `composition`, `coalesce`, `complement`, `select`, `logical_divide` is direction-agnostic by construction
  - same primitives target CPU SIMD lanes, GPU threads, or TPU lanes.
  - kernel authors should write against `Layout` / `LayoutTensor`-shaped operands, instead of raw `Array` byte offsets.
- things deferred from CUTLASS that come back when we add GPU:
  - `RuntimeLayout` (shape erasure for many comptime-known kernels)
  - `Swizzle<B, M, S>` (shared-memory bank conflict avoidance)
  - `tiled_mma` / `tiled_copy` / TMA primitives,
  - address spaces
  -  tensor core fragments.

package layout:
- `int_tuple`:
  - recursive `IntTuple` value type (leaf or list of IntTuples)
  - traversal helpers
  - `crd2idx` / `idx2crd`
- `layout`:
  - `Layout = (shape: IntTuple, stride: IntTuple)` struct
  - ctors (`row_major`, `col_major`, `strided`, `ordered`)
  - basic queries (`__call__`, `idx2crd`, `size`, `cosize`, `__getitem__`)
- `functional`
  - `coalesce`, `select`, `transpose`, `composition`, `complement`, `logical_divide`
  - mirrors CUTLASS's `cute/algorithm/functional.hpp`
- `iter`
  - `LayoutIter` (single layout)
  - `MultiLayoutIter` (N broadcasted operands).
  - amortizes the `crd2idx` divmod across the whole walk.

naming notes:
- package is `cute` because:
  - `algorithm` collides with `std.algorithm` on Mojo's import path.
  - `layout` collides with `max/kernels/src/layout` on the toolchain import path.
  - `cute` matches CUTLASS provenance and is collision-free.

Mojo 1.0 patterns worth noting:
- `IntTuple` declares `Copyable` and provides `def __init__(out self, *, copy: Self)`).
  - the manual constructor breaks the synthesis cycle between `IntTuple: Copyable` and `List[IntTuple]: Copyable`.
- we do not declare `ImplicitlyCopyable`
  - because field-wise synthesis through `List[Self]` fails.
  - consequence: reads of `_children[i]` outside `__eq__` require explicit `.copy()`.
- factories that take variadic ints use `List[Int]` instead of `*values: Int`
  - because mohaus stub generation refuses `*name` Python identifiers when emitting `_native.pyi`.
  - need fixing

references:
- CUTLASS `media/docs/cute/01_layout.md` and `02_layout_algebra.md`;
- algorithm skeletons in `include/cute/layout.hpp` and `cute/algorithm/functional.hpp`
- Jay Shah's "CuTe Layout Algebra" series on research.colfax-intl.com;
- Modular's `max/kernels/src/layout/`.

detailed design notes in `docs/cute-layout.md`.
"""

from .int_tuple import IntTuple
from .layout import (
    Layout,
    make_layout_col_major,
    make_layout_row_major,
    make_layout_strided,
    make_ordered_layout,
)
from .functional import (
    coalesce,
    complement,
    composition,
    logical_divide,
    select,
    transpose,
)
from .iter import LayoutIter, MultiLayoutIter
