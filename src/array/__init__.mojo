"""monpy array — Array struct, accessors, factories, dispatch, casting.

Package layout (post-split). Sub-modules use relative imports
(`from .accessors import Array, …`); none of them re-enter the package
through `from array import …`, so there's no init-time cycle.

  - `accessors` — Array struct (with 16 `*_py` static methods that
    `lib.mojo` registers on the python side), plus the small shape
    probes and parametric leaf accessors the methods call.
  - `factory` — `make_*_array` constructors, Layout adapters
    (`as_layout`, `as_broadcast_layout`, `array_with_layout`),
    `int_list_from_py`.
  - `dispatch` — runtime → comptime dtype dispatchers
    (`dispatch_real_to_f64`, `dispatch_real_write_f64`,
    `dispatch_int_write_i64`, the from-py variants), plus the universal
    `get_physical_as_f64` / `set_*_from_f64` / `set_logical_from_py` /
    `fill_all_from_py` / `contiguous_as_f64` paths.
  - `cast` — pairwise cast dispatchers (real-real 11×11, bool↔real),
    rank-2 strided copy tile family, `copy_c_contiguous`,
    `cast_copy_array`.
  - `result_dtypes` — `result_dtype_for_*` wrappers and `broadcast_shape`.

Public re-exports below cover every name that callers consume; existing
`from array import X` lines keep working without churn at the call sites.
"""

from .accessors import (
    Array,
    clone_int_list,
    contiguous_ptr,
    get_physical,
    get_physical_bool,
    get_physical_c128_imag,
    get_physical_c128_real,
    get_physical_c64_imag,
    get_physical_c64_real,
    get_physical_e8m0fnu,
    get_physical_fp4_e2m1fn,
    has_negative_strides,
    has_zero_strides,
    is_c_contiguous,
    is_f_contiguous,
    is_linearly_addressable,
    item_size,
    make_c_strides,
    physical_offset,
    same_shape,
    set_physical,
    set_physical_c128,
    set_physical_c64,
    set_physical_e8m0fnu,
    set_physical_fp4_e2m1fn,
    shape_size,
    slice_length,
    validate_shape,
)
from .cast import (
    cast_copy_array,
    copy_c_contiguous,
    dispatch_real_pair_cast,
    dispatch_real_typed_contig_pair,
)
from .dispatch import (
    contiguous_as_f64,
    dispatch_int_write_i64,
    dispatch_real_contig_fill_from_py,
    dispatch_real_to_f64,
    dispatch_real_write_f64,
    dispatch_real_write_from_py,
    fill_all_from_py,
    get_logical_as_f64,
    get_physical_as_f64,
    scalar_py_as_f64,
    set_contiguous_from_f64,
    set_logical_from_f64,
    set_logical_from_i64,
    set_logical_from_py,
    set_physical_from_f64,
)
from .factory import (
    array_with_layout,
    as_broadcast_layout,
    as_layout,
    int_list_from_py,
    make_empty_array,
    make_external_array,
    make_view_array,
)
from .result_dtypes import (
    broadcast_shape,
    result_dtype_for_binary,
    result_dtype_for_linalg,
    result_dtype_for_linalg_binary,
    result_dtype_for_reduction,
    result_dtype_for_unary,
    result_dtype_for_unary_preserve,
)
