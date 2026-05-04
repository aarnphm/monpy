from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from native import (
    native_add,
    native_add_into,
    native_add_f32_into,
    native_arange,
    native_astype,
    native_binary,
    native_binary_into,
    native_broadcast_to,
    native_copyto,
    native_empty,
    native_fill,
    native_from_external,
    native_from_flat,
    native_full,
    native_diagonal,
    native_linalg_det,
    native_linalg_inv,
    native_linalg_solve,
    native_linspace,
    native_matmul,
    native_materialize_c_contiguous,
    native_binary_scalar,
    native_reduce,
    native_result_dtype_for_binary,
    native_result_dtype_for_reduction,
    native_result_dtype_for_unary,
    native_reshape,
    native_sin_add_mul,
    native_slice,
    native_slice_1d,
    native_trace,
    native_transpose,
    native_unary,
    native_where,
)
from native_types import NativeArray


# Keep this file as the CPython extension boundary. The array runtime lives in
# native.mojo so Mojo can grow as a library without dragging Python module setup
# through every storage or kernel change.
@export
def PyInit__native() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_native")
        _ = (
            module.add_type[NativeArray]("NativeArray")
            .def_method[NativeArray.dtype_code_py]("dtype_code")
            .def_method[NativeArray.ndim_py]("ndim")
            .def_method[NativeArray.size_py]("size")
            .def_method[NativeArray.shape_at_py]("shape_at")
            .def_method[NativeArray.stride_at_py]("stride_at")
            .def_method[NativeArray.item_size_py]("item_size")
            .def_method[NativeArray.data_address_py]("data_address")
            .def_method[NativeArray.is_c_contiguous_py]("is_c_contiguous")
            .def_method[NativeArray.is_f_contiguous_py]("is_f_contiguous")
            .def_method[NativeArray.has_negative_strides_py](
                "has_negative_strides"
            )
            .def_method[NativeArray.has_zero_strides_py]("has_zero_strides")
            .def_method[NativeArray.storage_refcount_py]("storage_refcount")
            .def_method[NativeArray.used_accelerate_py]("used_accelerate")
            .def_method[NativeArray.used_fused_py]("used_fused")
            .def_method[NativeArray.backend_code_py]("backend_code")
            .def_method[NativeArray.get_scalar_py]("get_scalar")
        )
        module.def_function[native_empty]("empty")
        module.def_function[native_full]("full")
        module.def_function[native_from_flat]("from_flat")
        module.def_function[native_from_external]("from_external")
        module.def_function[native_arange]("arange")
        module.def_function[native_linspace]("linspace")
        module.def_function[native_reshape]("reshape")
        module.def_function[native_transpose]("transpose")
        module.def_function[native_slice]("slice")
        module.def_function[native_broadcast_to]("broadcast_to")
        module.def_function[native_astype]("astype")
        module.def_function[native_materialize_c_contiguous](
            "materialize_c_contiguous"
        )
        module.def_function[native_diagonal]("diagonal")
        module.def_function[native_trace]("trace")
        module.def_function[native_unary]("unary")
        module.def_function[native_add]("add")
        module.def_function[native_add_into]("add_into")
        module.def_function[native_add_f32_into]("add_f32_into")
        module.def_function[native_binary]("binary")
        module.def_function[native_binary_into]("binary_into")
        module.def_function[native_binary_scalar]("binary_scalar")
        module.def_function[native_sin_add_mul]("sin_add_mul")
        module.def_function[native_where]("where")
        module.def_function[native_reduce]("reduce")
        module.def_function[native_result_dtype_for_unary](
            "_result_dtype_for_unary"
        )
        module.def_function[native_result_dtype_for_binary](
            "_result_dtype_for_binary"
        )
        module.def_function[native_result_dtype_for_reduction](
            "_result_dtype_for_reduction"
        )
        module.def_function[native_matmul]("matmul")
        module.def_function[native_linalg_solve]("linalg_solve")
        module.def_function[native_linalg_inv]("linalg_inv")
        module.def_function[native_linalg_det]("linalg_det")
        module.def_function[native_fill]("fill")
        module.def_function[native_copyto]("copyto")
        module.def_function[native_slice_1d]("slice_1d")
        return module.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
