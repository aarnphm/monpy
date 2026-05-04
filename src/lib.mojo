from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from native import (
    NativeArray,
    native_arange,
    native_astype,
    native_binary,
    native_broadcast_to,
    native_copyto,
    native_empty,
    native_fill,
    native_from_flat,
    native_full,
    native_layout_smoke,
    native_linspace,
    native_matmul,
    native_reduce,
    native_reshape,
    native_slice,
    native_transpose,
    native_unary,
    native_where,
)


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
            .def_method[NativeArray.used_layout_tensor_py]("used_layout_tensor")
            .def_method[NativeArray.get_scalar_py]("get_scalar")
        )
        module.def_function[native_empty]("empty")
        module.def_function[native_full]("full")
        module.def_function[native_from_flat]("from_flat")
        module.def_function[native_arange]("arange")
        module.def_function[native_linspace]("linspace")
        module.def_function[native_reshape]("reshape")
        module.def_function[native_transpose]("transpose")
        module.def_function[native_slice]("slice")
        module.def_function[native_broadcast_to]("broadcast_to")
        module.def_function[native_astype]("astype")
        module.def_function[native_unary]("unary")
        module.def_function[native_binary]("binary")
        module.def_function[native_where]("where")
        module.def_function[native_reduce]("reduce")
        module.def_function[native_matmul]("matmul")
        module.def_function[native_fill]("fill")
        module.def_function[native_copyto]("copyto")
        module.def_function[native_layout_smoke]("layout_smoke")
        return module.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
