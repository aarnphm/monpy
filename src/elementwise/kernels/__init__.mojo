"""Typed compute kernels grouped by item."""

from .complex import (
    complex_binary_contig_typed,
    complex_binary_same_shape_strided_typed,
    complex_scalar_complex_contig_typed,
    complex_scalar_real_contig_typed,
    complex_unary_preserve_contig_typed,
    maybe_complex_binary_contiguous_accelerate,
    maybe_complex_binary_same_shape_strided,
)
from .fused import maybe_sin_add_mul_contiguous
from .linalg import (
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
from .matmul import (
    matmul_small_typed,
    maybe_matmul_complex_accelerate,
    maybe_matmul_contiguous,
    maybe_matmul_f32_small,
    maybe_matmul_vector_accelerate,
)
from .nn import layer_norm_last_axis_typed, scaled_masked_softmax_last_axis_typed, softmax_last_axis_typed
from .reduce import (
    maybe_argmax_contiguous,
    maybe_reduce_axis_last_contiguous,
    maybe_reduce_contiguous,
    maybe_reduce_strided_typed,
    reduce_strided_typed,
    reduce_sum_typed,
)
from .tile import (
    StridedInnerChoice,
    maybe_binary_column_broadcast_dispatch,
    maybe_binary_rank2_transposed_tile,
    maybe_binary_rank2_transposed_tile_bcast_1d,
    maybe_binary_rank3_axis0_tile,
    pick_inner_axis_for_strided_binary,
)
from .typed import (
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
