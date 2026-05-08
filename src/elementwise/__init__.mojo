"""monpy elementwise — typed kernels, dispatch, accelerate paths.

Package layout (post-split). Each sub-module has its own header that goes
deeper; this is just a map.

  - `apply_scalar` — scalar f64 fallback bodies (`apply_binary_f64`,
    `apply_unary_f64`); the universal fallback every typed dispatcher
    walks to when no fast path applies.
  - `predicates` — small `is_*` and `Rank2BlasLayout` helpers shared
    across kernels and dispatch.
  - `accelerate_dispatch` — macOS Accelerate FFI glue
    (`maybe_*_accelerate`, `maybe_*_rank1_strided_accelerate`).

Typed kernels (`[dt: DType]` parametric, comptime SIMD width):
  - `kernels/typed` — `apply_*_typed_vec`, `*_static`, `*_contig_typed`
    family (binary, scalar, row-broadcast, unary, unary-preserve,
    rank-2-strided unary).
  - `kernels/complex` — interleaved (re, im) variants: unary preserve,
    binary strided, complex×complex / complex×real scalar broadcast,
    `complex_binary_contig_typed`, vDSP fast path.
  - `kernels/reduce` — `reduce_sum_typed`, `reduce_strided_typed`,
    `maybe_argmax_contiguous`.
  - `kernels/linalg` — LAPACK-backed kernels (qr/cholesky/eigh/eig/svd/
    lstsq) and the LU pure-Mojo fallback.
  - `kernels/matmul` — `matmul_small_typed`, `maybe_matmul_*` family.
  - `kernels/nn` — row-wise layer norm / softmax bodies consumed by
    `create/ops/nn.mojo`.

Strided / tile / fused fast paths:
  - `strided_walkers` — `strided_binary_walk_typed`,
    `binary_strided_walk_typed`, `maybe_binary_strided_typed`.
  - `kernels/tile` — `StridedInnerChoice`, 4×4 transposed tiles,
    rank-3 axis-0 tile, column-broadcast tile dispatch.
  - `kernels/fused` — `maybe_sin_add_mul_contiguous`.

Dispatch:
  - `dispatch_helpers` — comptime-fn-parametric 11-way dtype monomorphizers
    (`dispatch_real_typed_simd_binary/unary`).
  - `binary_dispatch` — `maybe_binary_*` entry points called by
    `src/create/ops/elementwise.mojo`.
  - `unary_dispatch` — `maybe_unary_*` entry points.

Public re-exports below cover the union all upstream callers (lib,
create/*/ops) consume. Kept as a flat re-export so existing `from elementwise
import X` lines keep working.
"""

from .accelerate_dispatch import (
    maybe_binary_accelerate,
    maybe_binary_rank1_strided_accelerate,
    maybe_complex_binary_rank1_strided_accelerate,
    maybe_unary_accelerate,
)
from .apply_scalar import apply_binary_f64, apply_unary_f64
from .binary_dispatch import (
    maybe_binary_contiguous,
    maybe_binary_row_broadcast_contiguous,
    maybe_binary_same_shape_contiguous,
    maybe_binary_same_shape_strided,
    maybe_binary_scalar_contiguous,
    maybe_binary_scalar_value_contiguous,
)
from .kernels.complex import (
    complex_binary_contig_typed,
    complex_binary_same_shape_strided_typed,
    complex_scalar_complex_contig_typed,
    complex_scalar_real_contig_typed,
    complex_unary_preserve_contig_typed,
    maybe_complex_binary_contiguous_accelerate,
    maybe_complex_binary_same_shape_strided,
)
from .dispatch_helpers import (
    BinaryContigKernel,
    UnaryContigKernel,
    dispatch_real_typed_simd_binary,
    dispatch_real_typed_simd_unary,
)
from .kernels.fused import maybe_sin_add_mul_contiguous
from .kernels.linalg import (
    abs_f64,
    copy_rhs_to_col_major,
    lapack_cholesky_into,
    lapack_eig_real_into,
    lapack_eigh_into,
    lapack_lstsq_into,
    lapack_pivot_sign,
    lapack_qr_r_only_into,
    lapack_qr_reduced_into,
    lapack_svd_into,
    load_square_matrix_f64,
    lu_decompose_partial_pivot,
    lu_det,
    lu_det_into,
    lu_inverse_into,
    lu_solve_into,
    make_lu_pivots,
    maybe_lapack_det,
    maybe_lapack_inverse,
    maybe_lapack_solve,
    solve_lu_factor_into,
    swap_lu_rows,
    swap_rhs_rows,
    transpose_to_col_major,
    transpose_to_col_major_rect,
    write_cholesky_lower,
    write_col_major_to_array,
    write_solve_result,
)
from .kernels.nn import (
    layer_norm_last_axis_typed,
    scaled_masked_softmax_last_axis_typed,
    softmax_last_axis_typed,
)
from .kernels.matmul import (
    matmul_small_typed,
    maybe_matmul_complex_accelerate,
    maybe_matmul_contiguous,
    maybe_matmul_f32_small,
    maybe_matmul_vector_accelerate,
)
from .predicates import (
    Rank2BlasLayout,
    is_contiguous_float_array,
    is_contiguous_typed_simd_array,
    is_float_dtype,
    is_typed_simd_dtype,
    max_int,
    rank2_blas_layout,
)
from .kernels.reduce import (
    maybe_argmax_contiguous,
    maybe_reduce_axis_last_contiguous,
    maybe_reduce_contiguous,
    maybe_reduce_strided_typed,
    reduce_strided_typed,
    reduce_sum_typed,
)
from .strided_walkers import (
    binary_strided_walk_typed,
    maybe_binary_strided_typed,
    strided_binary_walk_typed,
)
from .kernels.tile import (
    StridedInnerChoice,
    maybe_binary_column_broadcast_dispatch,
    maybe_binary_rank2_transposed_tile,
    maybe_binary_rank2_transposed_tile_bcast_1d,
    maybe_binary_rank3_axis0_tile,
    pick_inner_axis_for_strided_binary,
)
from .kernels.typed import (
    apply_binary_typed_vec,
    apply_binary_typed_vec_static,
    apply_unary_preserve_typed_vec,
    apply_unary_typed_vec,
    binary_column_broadcast_contig_typed,
    binary_row_broadcast_contig_typed,
    binary_same_shape_contig_typed,
    binary_same_shape_contig_typed_static,
    binary_scalar_contig_typed,
    binary_scalar_contig_typed_static,
    try_binary_same_shape_contig_typed_static,
    try_binary_scalar_contig_typed_static,
    unary_contig_typed,
    unary_preserve_contig_typed,
    unary_rank2_strided_typed,
)
from .unary_dispatch import (
    maybe_unary_contiguous,
    maybe_unary_preserve_contiguous,
    maybe_unary_rank2_strided,
)
