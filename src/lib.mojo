from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from create import (
    arange,
    binary,
    binary_into,
    det,
    empty,
    from_external,
    from_flat,
    full,
    inv,
    linspace,
    binary_scalar,
    copy_from_external,
    result_dtype_for_binary_py,
    result_dtype_for_unary_py,
    solve,
    sin_add_mul,
    transpose_full_reverse,
    unary,
    where,
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
            .def_method[Array.has_negative_strides_py](
                "has_negative_strides"
            )
            .def_method[Array.has_zero_strides_py]("has_zero_strides")
            .def_method[Array.storage_refcount_py]("storage_refcount")
            .def_method[Array.used_accelerate_py]("used_accelerate")
            .def_method[Array.used_fused_py]("used_fused")
            .def_method[Array.backend_code_py]("backend_code")
            .def_method[Array.get_scalar_py]("get_scalar")
        )
        module.def_function[empty]("empty")
        module.def_function[full]("full")
        module.def_function[from_flat]("from_flat")
        module.def_function[from_external]("from_external")
        module.def_function[copy_from_external]("copy_from_external")
        module.def_function[arange]("arange")
        module.def_function[linspace]("linspace")
        module.def_function[reshape]("reshape")
        module.def_function[transpose]("transpose")
        module.def_function[transpose_full_reverse]("transpose_full_reverse")
        module.def_function[slice]("slice")
        module.def_function[broadcast_to]("broadcast_to")
        module.def_function[astype]("astype")
        module.def_function[materialize_c_contiguous](
            "materialize_c_contiguous"
        )
        module.def_function[diagonal]("diagonal")
        module.def_function[trace]("trace")
        module.def_function[unary]("unary")
        module.def_function[binary]("binary")
        module.def_function[binary_into]("binary_into")
        module.def_function[binary_scalar]("binary_scalar")
        module.def_function[sin_add_mul]("sin_add_mul")
        module.def_function[where]("where")
        module.def_function[reduce]("reduce")
        module.def_function[result_dtype_for_unary_py](
            "_result_dtype_for_unary"
        )
        module.def_function[result_dtype_for_binary_py](
            "_result_dtype_for_binary"
        )
        module.def_function[result_dtype_for_reduction_py](
            "_result_dtype_for_reduction"
        )
        module.def_function[matmul]("matmul")
        module.def_function[solve]("linalg_solve")
        module.def_function[inv]("linalg_inv")
        module.def_function[det]("linalg_det")
        module.def_function[fill]("fill")
        module.def_function[copyto]("copyto")
        module.def_function[slice_1d]("slice_1d")
        return module.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
