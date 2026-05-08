"""Elementwise `_ops` Python-bridge entry points.

Hosts the elementwise compute surface — unary preserve, comparison, logical,
predicate. Each one walks via `LayoutIter` (single-source) or `MultiLayoutIter`
(broadcast pair) to amortize the divmod across iteration; typed contiguous
fast paths route through `maybe_*_contiguous` upstream in elementwise/.

Bool-output ops (compare/logical/predicate) write to a fresh BOOL array;
unary_preserve resolves output dtype via `result_dtype_for_unary_preserve`
(input dtype with bool→int64 promotion).
"""

from std.math import (
    atan2 as _atan2,
    cos as _cos,
    cosh as _cosh,
    exp as _exp,
    isinf,
    isnan,
    log as _log,
    sin as _sin,
    sinh as _sinh,
    sqrt as _sqrt,
)
from std.python import PythonObject

from array import (
    Array,
    as_broadcast_layout,
    as_layout,
    broadcast_shape,
    clone_int_list,
    get_logical_as_f64,
    get_physical_as_f64,
    get_physical_bool,
    item_size,
    make_empty_array,
    result_dtype_for_binary,
    result_dtype_for_unary,
    result_dtype_for_unary_preserve,
    same_shape,
    scalar_py_as_f64,
    set_logical_from_f64,
    set_physical_from_f64,
)
from cute.iter import LayoutIter, MultiLayoutIter
from cute.layout import Layout
from domain import ArrayDType, BackendKind, BinaryOp, CompareOp, LogicalOp, PredicateOp, UnaryOp
from elementwise import (
    apply_binary_f64,
    apply_unary_f64,
    is_contiguous_float_array,
    maybe_binary_contiguous,
    maybe_binary_same_shape_contiguous,
    maybe_binary_scalar_value_contiguous,
    maybe_binary_strided_typed,
    maybe_matmul_contiguous,
    maybe_sin_add_mul_contiguous,
    maybe_unary_contiguous,
    maybe_unary_preserve_contiguous,
    maybe_unary_rank2_strided,
)

from ._complex_helpers import _complex_imag, _complex_real, _complex_store


def unary_preserve_ops(array_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    """Preserve-dtype unary ops (negate/abs/square/positive/floor/ceil/ trunc/rint/logical_not).
    Output dtype = input dtype (with bool→int64 promotion).
    Typed-vec contig fast path for f32/f64/i32/i64/u32/u64; other paths fall through to the f64 round-trip.
    """
    var src = array_obj.downcast_value_ptr[Array]()
    var op = UnaryOp.from_int(Int(py=op_obj)).value
    var shape = clone_int_list(src[].shape)
    var dtype_code = result_dtype_for_unary_preserve(src[].dtype_code)
    var result = make_empty_array(dtype_code, shape^)
    if maybe_unary_preserve_contiguous(src[], result, op):
        return PythonObject(alloc=result^)
    var src_layout = as_layout(src[])
    var dst_layout = as_layout(result)
    var src_item = item_size(src[].dtype_code)
    var dst_item = item_size(result.dtype_code)
    var src_iter = LayoutIter(src_layout, src_item, src[].offset_elems * src_item)
    var dst_iter = LayoutIter(dst_layout, dst_item, result.offset_elems * dst_item)
    while src_iter.has_next():
        var value = get_physical_as_f64(src[], src_iter.element_index())
        var output = apply_unary_f64(value, op)
        set_physical_from_f64(result, dst_iter.element_index(), output)
        src_iter.step()
        dst_iter.step()
    return PythonObject(alloc=result^)


def compare_ops(
    lhs_obj: PythonObject,
    rhs_obj: PythonObject,
    op_obj: PythonObject,
) raises -> PythonObject:
    """Elementwise comparison; returns a bool array. Operands broadcast.
    Walks via MultiLayoutIter so the broadcast divmod amortizes."""
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
    var op = CompareOp.from_int(Int(py=op_obj)).value
    var same = same_shape(lhs[].shape, rhs[].shape)
    var shape: List[Int]
    if same:
        shape = clone_int_list(lhs[].shape)
    else:
        shape = broadcast_shape(lhs[], rhs[])
    var result = make_empty_array(ArrayDType.BOOL.value, shape^)
    var lhs_layout = as_broadcast_layout(lhs[], result.shape)
    var rhs_layout = as_broadcast_layout(rhs[], result.shape)
    var out_layout = as_layout(result)
    var item_lhs = item_size(lhs[].dtype_code)
    var item_rhs = item_size(rhs[].dtype_code)
    var item_out = item_size(result.dtype_code)
    var operand_layouts = List[Layout]()
    operand_layouts.append(lhs_layout^)
    operand_layouts.append(rhs_layout^)
    operand_layouts.append(out_layout^)
    var item_sizes = List[Int]()
    item_sizes.append(item_lhs)
    item_sizes.append(item_rhs)
    item_sizes.append(item_out)
    var base_offsets = List[Int]()
    base_offsets.append(lhs[].offset_elems * item_lhs)
    base_offsets.append(rhs[].offset_elems * item_rhs)
    base_offsets.append(result.offset_elems * item_out)
    var iter = MultiLayoutIter(result.shape, operand_layouts^, item_sizes^, base_offsets^)
    while iter.has_next():
        var lval = get_physical_as_f64(lhs[], iter.element_index(0))
        var rval = get_physical_as_f64(rhs[], iter.element_index(1))
        var output: Bool
        if op == CompareOp.EQ.value:
            output = lval == rval
        elif op == CompareOp.NE.value:
            output = lval != rval
        elif op == CompareOp.LT.value:
            output = lval < rval
        elif op == CompareOp.LE.value:
            output = lval <= rval
        elif op == CompareOp.GT.value:
            output = lval > rval
        elif op == CompareOp.GE.value:
            output = lval >= rval
        else:
            raise Error("unknown comparison op")
        set_physical_from_f64(result, iter.element_index(2), 1.0 if output else 0.0)
        iter.step()
    return PythonObject(alloc=result^)


def logical_ops(
    lhs_obj: PythonObject,
    rhs_obj: PythonObject,
    op_obj: PythonObject,
) raises -> PythonObject:
    """Elementwise logical_and / or / xor. Operates on truthiness of any
    numeric input; result is bool. Walks via MultiLayoutIter."""
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
    var op = LogicalOp.from_int(Int(py=op_obj)).value
    var same = same_shape(lhs[].shape, rhs[].shape)
    var shape: List[Int]
    if same:
        shape = clone_int_list(lhs[].shape)
    else:
        shape = broadcast_shape(lhs[], rhs[])
    var result = make_empty_array(ArrayDType.BOOL.value, shape^)
    var lhs_layout = as_broadcast_layout(lhs[], result.shape)
    var rhs_layout = as_broadcast_layout(rhs[], result.shape)
    var out_layout = as_layout(result)
    var item_lhs = item_size(lhs[].dtype_code)
    var item_rhs = item_size(rhs[].dtype_code)
    var item_out = item_size(result.dtype_code)
    var operand_layouts = List[Layout]()
    operand_layouts.append(lhs_layout^)
    operand_layouts.append(rhs_layout^)
    operand_layouts.append(out_layout^)
    var item_sizes = List[Int]()
    item_sizes.append(item_lhs)
    item_sizes.append(item_rhs)
    item_sizes.append(item_out)
    var base_offsets = List[Int]()
    base_offsets.append(lhs[].offset_elems * item_lhs)
    base_offsets.append(rhs[].offset_elems * item_rhs)
    base_offsets.append(result.offset_elems * item_out)
    var iter = MultiLayoutIter(result.shape, operand_layouts^, item_sizes^, base_offsets^)
    while iter.has_next():
        var l_truthy = get_physical_as_f64(lhs[], iter.element_index(0)) != 0.0
        var r_truthy = get_physical_as_f64(rhs[], iter.element_index(1)) != 0.0
        var output: Bool
        if op == LogicalOp.AND.value:
            output = l_truthy and r_truthy
        elif op == LogicalOp.OR.value:
            output = l_truthy or r_truthy
        elif op == LogicalOp.XOR.value:
            output = l_truthy != r_truthy
        else:
            raise Error("unknown logical op")
        set_physical_from_f64(result, iter.element_index(2), 1.0 if output else 0.0)
        iter.step()
    return PythonObject(alloc=result^)


def unary_ops(array_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    """Generic unary apply: handles complex (Euler-form) + real (typed-vec
    contig fast path → rank-2 strided fast path → LayoutIter fallback).
    Output dtype = `result_dtype_for_unary(input)` (e.g. log of int → f64,
    sqrt of int → f64, transcendental of bool → f64).
    """
    var src = array_obj.downcast_value_ptr[Array]()
    var op = UnaryOp.from_int(Int(py=op_obj)).value
    var shape = clone_int_list(src[].shape)
    var result = make_empty_array(result_dtype_for_unary(src[].dtype_code), shape^)
    # Complex transcendentals: sin/cos/exp/log/sqrt etc. via Euler identities.
    # Output dtype = input dtype (preserved by result_dtype_for_unary).
    if src[].dtype_code == ArrayDType.COMPLEX64.value or src[].dtype_code == ArrayDType.COMPLEX128.value:
        for i in range(src[].size_value):
            var re = _complex_real(src[], i)
            var im = _complex_imag(src[], i)
            var out_re: Float64
            var out_im: Float64
            (out_re, out_im) = apply_unary_complex_f64(re, im, op)
            _complex_store(result, i, out_re, out_im)
        return PythonObject(alloc=result^)
    if maybe_unary_contiguous(src[], result, op):
        return PythonObject(alloc=result^)
    if maybe_unary_rank2_strided(src[], result, op):
        return PythonObject(alloc=result^)
    # Strided fallback: walk via LayoutIter so the divmod amortizes
    # across the iteration instead of paying physical_offset per element.
    var src_layout = as_layout(src[])
    var dst_layout = as_layout(result)
    var src_item = item_size(src[].dtype_code)
    var dst_item = item_size(result.dtype_code)
    var src_iter = LayoutIter(src_layout, src_item, src[].offset_elems * src_item)
    var dst_iter = LayoutIter(dst_layout, dst_item, result.offset_elems * dst_item)
    while src_iter.has_next():
        var value = get_physical_as_f64(src[], src_iter.element_index())
        var output = apply_unary_f64(value, op)
        set_physical_from_f64(result, dst_iter.element_index(), output)
        src_iter.step()
        dst_iter.step()
    return PythonObject(alloc=result^)


def apply_unary_complex_f64(re: Float64, im: Float64, op: Int) raises -> Tuple[Float64, Float64]:
    """Complex unary transcendentals via Euler identities. Operates on
    (re, im) Float64 pairs and returns the new pair. The python-level
    `unary_ops` walks complex arrays element-by-element through this.
    """
    if op == UnaryOp.EXP.value:
        # exp(a+bi) = exp(a) * (cos(b) + i sin(b))
        var ea = _exp(re)
        return (ea * _cos(im), ea * _sin(im))
    if op == UnaryOp.LOG.value:
        # log(z) = log|z| + i arg(z)
        var modulus = _sqrt(re * re + im * im)
        return (_log(modulus), _atan2(im, re))
    if op == UnaryOp.SIN.value:
        # sin(a+bi) = sin(a)cosh(b) + i cos(a)sinh(b)
        return (_sin(re) * _cosh(im), _cos(re) * _sinh(im))
    if op == UnaryOp.COS.value:
        # cos(a+bi) = cos(a)cosh(b) - i sin(a)sinh(b)
        return (_cos(re) * _cosh(im), -_sin(re) * _sinh(im))
    if op == UnaryOp.SINH.value:
        # sinh(a+bi) = sinh(a)cos(b) + i cosh(a)sin(b)
        return (_sinh(re) * _cos(im), _cosh(re) * _sin(im))
    if op == UnaryOp.COSH.value:
        # cosh(a+bi) = cosh(a)cos(b) + i sinh(a)sin(b)
        return (_cosh(re) * _cos(im), _sinh(re) * _sin(im))
    if op == UnaryOp.TANH.value:
        # tanh(z) = sinh(z) / cosh(z) — use Euler-form identities and divide.
        var s_re = _sinh(re) * _cos(im)
        var s_im = _cosh(re) * _sin(im)
        var c_re = _cosh(re) * _cos(im)
        var c_im = _sinh(re) * _sin(im)
        var denom = c_re * c_re + c_im * c_im
        return ((s_re * c_re + s_im * c_im) / denom, (s_im * c_re - s_re * c_im) / denom)
    if op == UnaryOp.TAN.value:
        # tan(z) = sin(z) / cos(z)
        var s_re = _sin(re) * _cosh(im)
        var s_im = _cos(re) * _sinh(im)
        var c_re = _cos(re) * _cosh(im)
        var c_im = -_sin(re) * _sinh(im)
        var denom = c_re * c_re + c_im * c_im
        return ((s_re * c_re + s_im * c_im) / denom, (s_im * c_re - s_re * c_im) / denom)
    if op == UnaryOp.SQRT.value:
        # sqrt(z): principal branch. z = r * exp(i*theta), sqrt(z) = sqrt(r) * exp(i*theta/2).
        var modulus = _sqrt(re * re + im * im)
        var arg = _atan2(im, re)
        var s = _sqrt(modulus)
        return (s * _cos(arg / 2.0), s * _sin(arg / 2.0))
    if op == UnaryOp.LOG2.value:
        # log2(z) = log(z) / log(2)
        var modulus = _sqrt(re * re + im * im)
        var ln2 = 0.6931471805599453
        return (_log(modulus) / ln2, _atan2(im, re) / ln2)
    if op == UnaryOp.LOG10.value:
        var modulus = _sqrt(re * re + im * im)
        var ln10 = 2.302585092994046
        return (_log(modulus) / ln10, _atan2(im, re) / ln10)
    if op == UnaryOp.LOG1P.value:
        # log(1 + z)
        var nre = 1.0 + re
        var modulus = _sqrt(nre * nre + im * im)
        return (_log(modulus), _atan2(im, nre))
    if op == UnaryOp.EXPM1.value:
        # exp(z) - 1 — use exp formula then subtract 1 from real part.
        var ea = _exp(re)
        return (ea * _cos(im) - 1.0, ea * _sin(im))
    if op == UnaryOp.RECIPROCAL.value:
        # 1 / (a + bi) = (a - bi) / (a² + b²)
        var denom = re * re + im * im
        return (re / denom, -im / denom)
    if op == UnaryOp.EXP2.value:
        # 2^z = exp(z * log(2))
        var ln2 = 0.6931471805599453
        var nre = re * ln2
        var nim = im * ln2
        var ea = _exp(nre)
        return (ea * _cos(nim), ea * _sin(nim))
    if op == UnaryOp.CBRT.value:
        # cbrt(z) — principal branch.
        var modulus = _sqrt(re * re + im * im)
        var arg = _atan2(im, re)
        var s = modulus.__pow__(1.0 / 3.0)
        return (s * _cos(arg / 3.0), s * _sin(arg / 3.0))
    raise Error("unary op not implemented for complex inputs")


def predicate_ops(array_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    """Unary predicate (isnan / isinf / isfinite / signbit). Returns
    bool. Walks via LayoutIter so the divmod amortizes across the
    iteration."""
    var src = array_obj.downcast_value_ptr[Array]()
    var op = PredicateOp.from_int(Int(py=op_obj)).value
    var shape = clone_int_list(src[].shape)
    var result = make_empty_array(ArrayDType.BOOL.value, shape^)
    var src_layout = as_layout(src[])
    var dst_layout = as_layout(result)
    var src_item = item_size(src[].dtype_code)
    var dst_item = item_size(result.dtype_code)
    var src_iter = LayoutIter(src_layout, src_item, src[].offset_elems * src_item)
    var dst_iter = LayoutIter(dst_layout, dst_item, result.offset_elems * dst_item)
    while src_iter.has_next():
        var value = get_physical_as_f64(src[], src_iter.element_index())
        var output: Bool
        if op == PredicateOp.ISNAN.value:
            output = isnan(value)
        elif op == PredicateOp.ISINF.value:
            output = isinf(value)
        elif op == PredicateOp.ISFINITE.value:
            output = not (isnan(value) or isinf(value))
        elif op == PredicateOp.SIGNBIT.value:
            # numpy.signbit returns True for -0.0, so use bitcast.
            output = value < 0.0
            if value == 0.0:
                # negative-zero → True via bitcast.
                var bits = SIMD[DType.float64, 1](value).cast[DType.uint64]()[0]
                output = (bits >> 63) != 0
        else:
            raise Error("unknown predicate op")
        set_physical_from_f64(result, dst_iter.element_index(), 1.0 if output else 0.0)
        src_iter.step()
        dst_iter.step()
    return PythonObject(alloc=result^)


def binary_dispatch_ops(lhs: Array, rhs: Array, op: Int) raises -> Array:
    # Internal binary_ops dispatch core. Allocates the result and runs the
    # contiguous / strided fast paths; if all fail, walks via
    # MultiLayoutIter so the broadcast divmod amortizes across the whole
    # iteration instead of paying per element.
    var dtype_code = result_dtype_for_binary(lhs.dtype_code, rhs.dtype_code, op)
    var same = same_shape(lhs.shape, rhs.shape)
    var shape: List[Int]
    if same:
        shape = clone_int_list(lhs.shape)
    else:
        shape = broadcast_shape(lhs, rhs)
    var result = make_empty_array(dtype_code, shape^)
    if same and maybe_binary_same_shape_contiguous(lhs, rhs, result, op):
        return result^
    if maybe_binary_contiguous(lhs, rhs, result, op):
        return result^
    # Typed strided walker: same-dtype broadcasts skip the f64 dispatch
    # cascade entirely. Most ops in practice (arithmetic on matching
    # dtypes) hit this. Mixed-dtype broadcasts fall through.
    if maybe_binary_strided_typed(lhs, rhs, result, op):
        return result^
    # Strided fallback: cursor walk via MultiLayoutIter. Each operand
    # carries stride-zero modes for broadcast dimensions; the output is
    # contiguous (we just allocated it).
    var lhs_layout = as_broadcast_layout(lhs, result.shape)
    var rhs_layout = as_broadcast_layout(rhs, result.shape)
    var out_layout = as_layout(result)
    var item_lhs = item_size(lhs.dtype_code)
    var item_rhs = item_size(rhs.dtype_code)
    var item_out = item_size(result.dtype_code)
    var operand_layouts = List[Layout]()
    operand_layouts.append(lhs_layout^)
    operand_layouts.append(rhs_layout^)
    operand_layouts.append(out_layout^)
    var item_sizes = List[Int]()
    item_sizes.append(item_lhs)
    item_sizes.append(item_rhs)
    item_sizes.append(item_out)
    var base_offsets = List[Int]()
    base_offsets.append(lhs.offset_elems * item_lhs)
    base_offsets.append(rhs.offset_elems * item_rhs)
    base_offsets.append(result.offset_elems * item_out)
    var iter = MultiLayoutIter(result.shape, operand_layouts^, item_sizes^, base_offsets^)
    while iter.has_next():
        var lval = get_physical_as_f64(lhs, iter.element_index(0))
        var rval = get_physical_as_f64(rhs, iter.element_index(1))
        set_physical_from_f64(result, iter.element_index(2), apply_binary_f64(lval, rval, op))
        iter.step()
    return result^


def binary_op_method_ops(py_self: PythonObject, other_obj: PythonObject, op: Int) raises -> PythonObject:
    # Method-style entrypoint shared by Array.add/sub/mul/div. The Mojo
    # PythonObject method dispatch is measurably tighter than def_function
    # for arg-marshal-bound calls; per-op trampolines (add_method_py etc.)
    # bake `op` into the dispatch so Python doesn't need to pass it.
    var self_ptr = py_self.downcast_value_ptr[Array]()
    var other_ptr = other_obj.downcast_value_ptr[Array]()
    var result = binary_dispatch_ops(self_ptr[], other_ptr[], op)
    return PythonObject(alloc=result^)


def array_add_method_ops(py_self: PythonObject, other_obj: PythonObject) raises -> PythonObject:
    return binary_op_method_ops(py_self, other_obj, BinaryOp.ADD.value)


def array_sub_method_ops(py_self: PythonObject, other_obj: PythonObject) raises -> PythonObject:
    return binary_op_method_ops(py_self, other_obj, BinaryOp.SUB.value)


def array_mul_method_ops(py_self: PythonObject, other_obj: PythonObject) raises -> PythonObject:
    return binary_op_method_ops(py_self, other_obj, BinaryOp.MUL.value)


def array_div_method_ops(py_self: PythonObject, other_obj: PythonObject) raises -> PythonObject:
    return binary_op_method_ops(py_self, other_obj, BinaryOp.DIV.value)


def array_matmul_method_ops(py_self: PythonObject, other_obj: PythonObject) raises -> PythonObject:
    var self_ptr = py_self.downcast_value_ptr[Array]()
    var other_ptr = other_obj.downcast_value_ptr[Array]()
    var lhs_ndim = len(self_ptr[].shape)
    var rhs_ndim = len(other_ptr[].shape)
    if lhs_ndim < 1 or lhs_ndim > 2 or rhs_ndim < 1 or rhs_ndim > 2:
        raise Error("matmul() only supports 1d and 2d arrays")
    var m = 1
    var k_lhs = self_ptr[].shape[0]
    if lhs_ndim == 2:
        m = self_ptr[].shape[0]
        k_lhs = self_ptr[].shape[1]
    var k_rhs = other_ptr[].shape[0]
    var n = 1
    if rhs_ndim == 2:
        n = other_ptr[].shape[1]
    if k_lhs != k_rhs:
        raise Error("matmul() dimension mismatch")
    var out_shape = List[Int]()
    if lhs_ndim == 2 and rhs_ndim == 2:
        out_shape.append(m)
        out_shape.append(n)
    elif lhs_ndim == 2 and rhs_ndim == 1:
        out_shape.append(m)
    elif lhs_ndim == 1 and rhs_ndim == 2:
        out_shape.append(n)
    var result = make_empty_array(
        result_dtype_for_binary(self_ptr[].dtype_code, other_ptr[].dtype_code, BinaryOp.MUL.value),
        out_shape^,
    )
    if lhs_ndim == 1 and rhs_ndim == 1:
        var total = 0.0
        for k in range(k_lhs):
            total += get_logical_as_f64(self_ptr[], k) * get_logical_as_f64(other_ptr[], k)
        set_logical_from_f64(result, 0, total)
        return PythonObject(alloc=result^)
    if maybe_matmul_contiguous(self_ptr[], other_ptr[], result, m, n, k_lhs):
        return PythonObject(alloc=result^)
    for i in range(m):
        for j in range(n):
            var total = 0.0
            for k in range(k_lhs):
                var lhs_index = k
                if lhs_ndim == 2:
                    lhs_index = i * k_lhs + k
                var rhs_index = k
                if rhs_ndim == 2:
                    rhs_index = k * n + j
                total += get_logical_as_f64(self_ptr[], lhs_index) * get_logical_as_f64(other_ptr[], rhs_index)
            var out_index = j
            if lhs_ndim == 2:
                out_index = i * n + j
            set_logical_from_f64(result, out_index, total)
    return PythonObject(alloc=result^)


def binary_ops(lhs_obj: PythonObject, rhs_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
    var op = BinaryOp.from_int(Int(py=op_obj)).value
    var result = binary_dispatch_ops(lhs[], rhs[], op)
    return PythonObject(alloc=result^)


def binary_into_ops(
    dst_obj: PythonObject,
    lhs_obj: PythonObject,
    rhs_obj: PythonObject,
    op_obj: PythonObject,
) raises -> PythonObject:
    var dst = dst_obj.downcast_value_ptr[Array]()
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
    var op = BinaryOp.from_int(Int(py=op_obj)).value
    var dtype_code = result_dtype_for_binary(lhs[].dtype_code, rhs[].dtype_code, op)
    if dst[].dtype_code != dtype_code:
        raise Error("out dtype does not match binary result dtype")
    if same_shape(lhs[].shape, rhs[].shape) and same_shape(lhs[].shape, dst[].shape):
        if maybe_binary_same_shape_contiguous(lhs[], rhs[], dst[], op):
            return PythonObject(None)
    else:
        var shape = broadcast_shape(lhs[], rhs[])
        if not same_shape(shape, dst[].shape):
            raise Error("out shape does not match binary result shape")
        if maybe_binary_contiguous(lhs[], rhs[], dst[], op):
            return PythonObject(None)
    var lhs_layout = as_broadcast_layout(lhs[], dst[].shape)
    var rhs_layout = as_broadcast_layout(rhs[], dst[].shape)
    var dst_layout = as_layout(dst[])
    var item_lhs = item_size(lhs[].dtype_code)
    var item_rhs = item_size(rhs[].dtype_code)
    var item_dst = item_size(dst[].dtype_code)
    var operand_layouts = List[Layout]()
    operand_layouts.append(lhs_layout^)
    operand_layouts.append(rhs_layout^)
    operand_layouts.append(dst_layout^)
    var item_sizes = List[Int]()
    item_sizes.append(item_lhs)
    item_sizes.append(item_rhs)
    item_sizes.append(item_dst)
    var base_offsets = List[Int]()
    base_offsets.append(lhs[].offset_elems * item_lhs)
    base_offsets.append(rhs[].offset_elems * item_rhs)
    base_offsets.append(dst[].offset_elems * item_dst)
    var iter = MultiLayoutIter(dst[].shape, operand_layouts^, item_sizes^, base_offsets^)
    while iter.has_next():
        var lval = get_physical_as_f64(lhs[], iter.element_index(0))
        var rval = get_physical_as_f64(rhs[], iter.element_index(1))
        set_physical_from_f64(dst[], iter.element_index(2), apply_binary_f64(lval, rval, op))
        iter.step()
    return PythonObject(None)


def binary_scalar_ops(
    array_obj: PythonObject,
    scalar_obj: PythonObject,
    scalar_dtype_obj: PythonObject,
    op_obj: PythonObject,
    scalar_on_left_obj: PythonObject,
) raises -> PythonObject:
    var array = array_obj.downcast_value_ptr[Array]()
    var scalar_dtype = Int(py=scalar_dtype_obj)
    var op = BinaryOp.from_int(Int(py=op_obj)).value
    var scalar_on_left = Bool(py=scalar_on_left_obj)
    var shape = clone_int_list(array[].shape)
    var dtype_code = result_dtype_for_binary(array[].dtype_code, scalar_dtype, op)
    var result = make_empty_array(dtype_code, shape^)
    var scalar_value = scalar_py_as_f64(scalar_obj, scalar_dtype)
    if maybe_binary_scalar_value_contiguous(array[], scalar_value, result, op, scalar_on_left):
        return PythonObject(alloc=result^)
    var src_layout = as_layout(array[])
    var dst_layout = as_layout(result)
    var src_item = item_size(array[].dtype_code)
    var dst_item = item_size(result.dtype_code)
    var src_iter = LayoutIter(src_layout, src_item, array[].offset_elems * src_item)
    var dst_iter = LayoutIter(dst_layout, dst_item, result.offset_elems * dst_item)
    while src_iter.has_next():
        var lhs = get_physical_as_f64(array[], src_iter.element_index())
        var rhs = scalar_value
        if scalar_on_left:
            lhs = scalar_value
            rhs = get_physical_as_f64(array[], src_iter.element_index())
        set_physical_from_f64(result, dst_iter.element_index(), apply_binary_f64(lhs, rhs, op))
        src_iter.step()
        dst_iter.step()
    return PythonObject(alloc=result^)


def sin_add_mul_ops(
    lhs_obj: PythonObject,
    rhs_obj: PythonObject,
    scalar_obj: PythonObject,
    scalar_dtype_obj: PythonObject,
) raises -> PythonObject:
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
    var scalar_dtype = Int(py=scalar_dtype_obj)
    var scalar_value = scalar_py_as_f64(scalar_obj, scalar_dtype)
    if (
        same_shape(lhs[].shape, rhs[].shape)
        and lhs[].dtype_code == rhs[].dtype_code
        and lhs[].dtype_code == scalar_dtype
        and is_contiguous_float_array(lhs[])
        and is_contiguous_float_array(rhs[])
    ):
        var fast_shape = clone_int_list(lhs[].shape)
        var fast_result = make_empty_array(lhs[].dtype_code, fast_shape^)
        if maybe_sin_add_mul_contiguous(lhs[], rhs[], scalar_value, fast_result):
            return PythonObject(alloc=fast_result^)
    var shape = broadcast_shape(lhs[], rhs[])
    var rhs_mul_dtype = result_dtype_for_binary(rhs[].dtype_code, scalar_dtype, BinaryOp.MUL.value)
    var dtype_code = result_dtype_for_binary(
        result_dtype_for_unary(lhs[].dtype_code), rhs_mul_dtype, BinaryOp.ADD.value
    )
    var result = make_empty_array(dtype_code, shape^)
    if maybe_sin_add_mul_contiguous(lhs[], rhs[], scalar_value, result):
        return PythonObject(alloc=result^)
    var lhs_layout = as_broadcast_layout(lhs[], result.shape)
    var rhs_layout = as_broadcast_layout(rhs[], result.shape)
    var out_layout = as_layout(result)
    var item_lhs = item_size(lhs[].dtype_code)
    var item_rhs = item_size(rhs[].dtype_code)
    var item_out = item_size(result.dtype_code)
    var operand_layouts = List[Layout]()
    operand_layouts.append(lhs_layout^)
    operand_layouts.append(rhs_layout^)
    operand_layouts.append(out_layout^)
    var item_sizes = List[Int]()
    item_sizes.append(item_lhs)
    item_sizes.append(item_rhs)
    item_sizes.append(item_out)
    var base_offsets = List[Int]()
    base_offsets.append(lhs[].offset_elems * item_lhs)
    base_offsets.append(rhs[].offset_elems * item_rhs)
    base_offsets.append(result.offset_elems * item_out)
    var iter = MultiLayoutIter(result.shape, operand_layouts^, item_sizes^, base_offsets^)
    while iter.has_next():
        var output = _sin(get_physical_as_f64(lhs[], iter.element_index(0))) + (
            get_physical_as_f64(rhs[], iter.element_index(1)) * scalar_value
        )
        set_physical_from_f64(result, iter.element_index(2), output)
        iter.step()
    result.backend_code = BackendKind.FUSED.value
    return PythonObject(alloc=result^)


def where_ops(cond_obj: PythonObject, lhs_obj: PythonObject, rhs_obj: PythonObject) raises -> PythonObject:
    var cond = cond_obj.downcast_value_ptr[Array]()
    var lhs = lhs_obj.downcast_value_ptr[Array]()
    var rhs = rhs_obj.downcast_value_ptr[Array]()
    var partial_shape = broadcast_shape(cond[], lhs[])
    var tmp = make_empty_array(lhs[].dtype_code, partial_shape^)
    var shape = broadcast_shape(tmp, rhs[])
    var dtype_code = result_dtype_for_binary(lhs[].dtype_code, rhs[].dtype_code, BinaryOp.ADD.value)
    var result = make_empty_array(dtype_code, shape^)
    var cond_layout = as_broadcast_layout(cond[], result.shape)
    var lhs_layout = as_broadcast_layout(lhs[], result.shape)
    var rhs_layout = as_broadcast_layout(rhs[], result.shape)
    var out_layout = as_layout(result)
    var item_cond = item_size(cond[].dtype_code)
    var item_lhs = item_size(lhs[].dtype_code)
    var item_rhs = item_size(rhs[].dtype_code)
    var item_out = item_size(result.dtype_code)
    var operand_layouts = List[Layout]()
    operand_layouts.append(cond_layout^)
    operand_layouts.append(lhs_layout^)
    operand_layouts.append(rhs_layout^)
    operand_layouts.append(out_layout^)
    var item_sizes = List[Int]()
    item_sizes.append(item_cond)
    item_sizes.append(item_lhs)
    item_sizes.append(item_rhs)
    item_sizes.append(item_out)
    var base_offsets = List[Int]()
    base_offsets.append(cond[].offset_elems * item_cond)
    base_offsets.append(lhs[].offset_elems * item_lhs)
    base_offsets.append(rhs[].offset_elems * item_rhs)
    base_offsets.append(result.offset_elems * item_out)
    var iter = MultiLayoutIter(result.shape, operand_layouts^, item_sizes^, base_offsets^)
    while iter.has_next():
        var cond_phys = iter.element_index(0)
        var picked: Float64
        if cond[].dtype_code == ArrayDType.BOOL.value:
            picked = 1.0 if get_physical_bool(cond[], cond_phys) else 0.0
        else:
            picked = get_physical_as_f64(cond[], cond_phys)
        if picked != 0.0:
            set_physical_from_f64(result, iter.element_index(3), get_physical_as_f64(lhs[], iter.element_index(1)))
        else:
            set_physical_from_f64(result, iter.element_index(3), get_physical_as_f64(rhs[], iter.element_index(2)))
        iter.step()
    return PythonObject(alloc=result^)
