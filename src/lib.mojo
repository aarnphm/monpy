from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from create import (
    arange_ops,
    array_add_method_ops,
    array_div_method_ops,
    array_matmul_method_ops,
    array_mul_method_ops,
    array_sub_method_ops,
    binary_ops,
    binary_into_ops,
    det_ops,
    empty_ops,
    from_external_ops,
    from_flat_ops,
    full_ops,
    inv_ops,
    linspace_ops,
    asarray_from_numpy_ops,
    binary_scalar_ops,
    copy_from_external_ops,
    result_dtype_for_binary_py_ops,
    result_dtype_for_unary_py_ops,
    solve_ops,
    sin_add_mul_ops,
    transpose_full_reverse_ops,
    unary_ops,
    where_ops,
)
from array import Array
from matmul import matmul
from reductions import reduce, result_dtype_for_reduction_py
from views import (
    astype,
    broadcast_to,
    copyto,
    diagonal,
    fill,
    materialize_c_contiguous,
    reshape,
    slice,
    slice_1d,
    trace,
    transpose,
)


# Keep this file as the CPython extension boundary. Mojo modules stay flat and
# compile into this one private Python extension chunk.
@export
def PyInit__native() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_native")
        _ = (
            module.add_type[Array]("Array")
            .def_method[Array.dtype_code_py]("dtype_code")
            .def_method[Array.ndim_py]("ndim")
            .def_method[Array.size_py]("size")
            .def_method[Array.shape_at_py]("shape_at")
            .def_method[Array.stride_at_py]("stride_at")
            .def_method[Array.item_size_py]("item_size")
            .def_method[Array.data_address_py]("data_address")
            .def_method[Array.is_c_contiguous_py]("is_c_contiguous")
            .def_method[Array.is_f_contiguous_py]("is_f_contiguous")
            .def_method[Array.has_negative_strides_py]("has_negative_strides")
            .def_method[Array.has_zero_strides_py]("has_zero_strides")
            .def_method[Array.storage_refcount_py]("storage_refcount")
            .def_method[Array.used_accelerate_py]("used_accelerate")
            .def_method[Array.used_fused_py]("used_fused")
            .def_method[Array.backend_code_py]("backend_code")
            .def_method[Array.get_scalar_py]("get_scalar")
            .def_method[array_add_method_ops]("add")
            .def_method[array_sub_method_ops]("sub")
            .def_method[array_mul_method_ops]("mul")
            .def_method[array_div_method_ops]("div")
            .def_method[array_matmul_method_ops]("matmul")
            .def_method[slice_1d]("slice_1d_method")
            .def_method[transpose_full_reverse_ops]("transpose_full_reverse_method")
        )
        module.def_function[empty_ops]("empty")
        module.def_function[full_ops]("full")
        module.def_function[from_flat_ops]("from_flat")
        module.def_function[from_external_ops]("from_external")
        module.def_function[copy_from_external_ops]("copy_from_external")
        module.def_function[asarray_from_numpy_ops]("asarray_from_numpy")
        module.def_function[arange_ops]("arange")
        module.def_function[linspace_ops]("linspace")
        module.def_function[reshape]("reshape")
        module.def_function[transpose]("transpose")
        module.def_function[transpose_full_reverse_ops]("transpose_full_reverse")
        module.def_function[slice]("slice")
        module.def_function[broadcast_to]("broadcast_to")
        module.def_function[astype]("astype")
        module.def_function[materialize_c_contiguous]("materialize_c_contiguous")
        module.def_function[diagonal]("diagonal")
        module.def_function[trace]("trace")
        module.def_function[unary_ops]("unary")
        module.def_function[binary_ops]("binary")
        module.def_function[binary_into_ops]("binary_into")
        module.def_function[binary_scalar_ops]("binary_scalar")
        module.def_function[sin_add_mul_ops]("sin_add_mul")
        module.def_function[where_ops]("where")
        module.def_function[reduce]("reduce")
        module.def_function[result_dtype_for_unary_py_ops]("_result_dtype_for_unary")
        module.def_function[result_dtype_for_binary_py_ops]("_result_dtype_for_binary")
        module.def_function[result_dtype_for_reduction_py]("_result_dtype_for_reduction")
        module.def_function[matmul]("matmul")
        module.def_function[solve_ops]("linalg_solve")
        module.def_function[inv_ops]("linalg_inv")
        module.def_function[det_ops]("linalg_det")
        module.def_function[fill]("fill")
        module.def_function[copyto]("copyto")
        module.def_function[slice_1d]("slice_1d")
        return module.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
