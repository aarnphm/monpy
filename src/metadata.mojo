from std.python import Python, PythonObject

from domain import (
    ArrayDType,
    BackendKind,
    BinaryOp,
    CastingRule,
    CompareOp,
    DTypeKind,
    LogicalOp,
    PredicateOp,
    ReduceOp,
    UnaryOp,
)


def _dtype_codes() raises -> PythonObject:
    var codes = Python.dict()
    codes["BOOL"] = ArrayDType.BOOL.value
    codes["INT64"] = ArrayDType.INT64.value
    codes["FLOAT32"] = ArrayDType.FLOAT32.value
    codes["FLOAT64"] = ArrayDType.FLOAT64.value
    codes["INT32"] = ArrayDType.INT32.value
    codes["INT16"] = ArrayDType.INT16.value
    codes["INT8"] = ArrayDType.INT8.value
    codes["UINT64"] = ArrayDType.UINT64.value
    codes["UINT32"] = ArrayDType.UINT32.value
    codes["UINT16"] = ArrayDType.UINT16.value
    codes["UINT8"] = ArrayDType.UINT8.value
    codes["FLOAT16"] = ArrayDType.FLOAT16.value
    codes["COMPLEX64"] = ArrayDType.COMPLEX64.value
    codes["COMPLEX128"] = ArrayDType.COMPLEX128.value
    return codes^


def _dtype_kind_codes() raises -> PythonObject:
    var codes = Python.dict()
    codes["BOOL"] = DTypeKind.BOOL.value
    codes["SIGNED_INT"] = DTypeKind.SIGNED_INT.value
    codes["REAL_FLOAT"] = DTypeKind.REAL_FLOAT.value
    codes["UNSIGNED_INT"] = DTypeKind.UNSIGNED_INT.value
    codes["COMPLEX_FLOAT"] = DTypeKind.COMPLEX_FLOAT.value
    return codes^


def _casting_codes() raises -> PythonObject:
    var codes = Python.dict()
    codes["NO"] = CastingRule.NO.value
    codes["EQUIV"] = CastingRule.EQUIV.value
    codes["SAFE"] = CastingRule.SAFE.value
    codes["SAME_KIND"] = CastingRule.SAME_KIND.value
    codes["UNSAFE"] = CastingRule.UNSAFE.value
    return codes^


def _binary_codes() raises -> PythonObject:
    var codes = Python.dict()
    codes["ADD"] = BinaryOp.ADD.value
    codes["SUB"] = BinaryOp.SUB.value
    codes["MUL"] = BinaryOp.MUL.value
    codes["DIV"] = BinaryOp.DIV.value
    codes["FLOOR_DIV"] = BinaryOp.FLOOR_DIV.value
    codes["MOD"] = BinaryOp.MOD.value
    codes["POWER"] = BinaryOp.POWER.value
    codes["MAXIMUM"] = BinaryOp.MAXIMUM.value
    codes["MINIMUM"] = BinaryOp.MINIMUM.value
    codes["FMIN"] = BinaryOp.FMIN.value
    codes["FMAX"] = BinaryOp.FMAX.value
    codes["ARCTAN2"] = BinaryOp.ARCTAN2.value
    codes["HYPOT"] = BinaryOp.HYPOT.value
    codes["COPYSIGN"] = BinaryOp.COPYSIGN.value
    return codes^


def _unary_codes() raises -> PythonObject:
    var codes = Python.dict()
    codes["SIN"] = UnaryOp.SIN.value
    codes["COS"] = UnaryOp.COS.value
    codes["EXP"] = UnaryOp.EXP.value
    codes["LOG"] = UnaryOp.LOG.value
    codes["TAN"] = UnaryOp.TAN.value
    codes["ARCSIN"] = UnaryOp.ARCSIN.value
    codes["ARCCOS"] = UnaryOp.ARCCOS.value
    codes["ARCTAN"] = UnaryOp.ARCTAN.value
    codes["SINH"] = UnaryOp.SINH.value
    codes["COSH"] = UnaryOp.COSH.value
    codes["TANH"] = UnaryOp.TANH.value
    codes["LOG1P"] = UnaryOp.LOG1P.value
    codes["LOG2"] = UnaryOp.LOG2.value
    codes["LOG10"] = UnaryOp.LOG10.value
    codes["EXP2"] = UnaryOp.EXP2.value
    codes["EXPM1"] = UnaryOp.EXPM1.value
    codes["SQRT"] = UnaryOp.SQRT.value
    codes["CBRT"] = UnaryOp.CBRT.value
    codes["DEG2RAD"] = UnaryOp.DEG2RAD.value
    codes["RAD2DEG"] = UnaryOp.RAD2DEG.value
    codes["RECIPROCAL"] = UnaryOp.RECIPROCAL.value
    codes["NEGATE"] = UnaryOp.NEGATE.value
    codes["POSITIVE"] = UnaryOp.POSITIVE.value
    codes["ABS"] = UnaryOp.ABS.value
    codes["SQUARE"] = UnaryOp.SQUARE.value
    codes["SIGN"] = UnaryOp.SIGN.value
    codes["FLOOR"] = UnaryOp.FLOOR.value
    codes["CEIL"] = UnaryOp.CEIL.value
    codes["TRUNC"] = UnaryOp.TRUNC.value
    codes["RINT"] = UnaryOp.RINT.value
    codes["LOGICAL_NOT"] = UnaryOp.LOGICAL_NOT.value
    codes["CONJUGATE"] = UnaryOp.CONJUGATE.value
    return codes^


def _compare_codes() raises -> PythonObject:
    var codes = Python.dict()
    codes["EQ"] = CompareOp.EQ.value
    codes["NE"] = CompareOp.NE.value
    codes["LT"] = CompareOp.LT.value
    codes["LE"] = CompareOp.LE.value
    codes["GT"] = CompareOp.GT.value
    codes["GE"] = CompareOp.GE.value
    return codes^


def _logical_codes() raises -> PythonObject:
    var codes = Python.dict()
    codes["AND"] = LogicalOp.AND.value
    codes["OR"] = LogicalOp.OR.value
    codes["XOR"] = LogicalOp.XOR.value
    return codes^


def _predicate_codes() raises -> PythonObject:
    var codes = Python.dict()
    codes["ISNAN"] = PredicateOp.ISNAN.value
    codes["ISINF"] = PredicateOp.ISINF.value
    codes["ISFINITE"] = PredicateOp.ISFINITE.value
    codes["SIGNBIT"] = PredicateOp.SIGNBIT.value
    return codes^


def _reduce_codes() raises -> PythonObject:
    var codes = Python.dict()
    codes["SUM"] = ReduceOp.SUM.value
    codes["MEAN"] = ReduceOp.MEAN.value
    codes["MIN"] = ReduceOp.MIN.value
    codes["MAX"] = ReduceOp.MAX.value
    codes["ARGMAX"] = ReduceOp.ARGMAX.value
    codes["PROD"] = ReduceOp.PROD.value
    codes["ALL"] = ReduceOp.ALL.value
    codes["ANY"] = ReduceOp.ANY.value
    codes["ARGMIN"] = ReduceOp.ARGMIN.value
    return codes^


def _backend_codes() raises -> PythonObject:
    var codes = Python.dict()
    codes["GENERIC"] = BackendKind.GENERIC.value
    codes["ACCELERATE"] = BackendKind.ACCELERATE.value
    codes["FUSED"] = BackendKind.FUSED.value
    return codes^


def domain_codes_py_ops() raises -> PythonObject:
    var codes = Python.dict()
    codes["dtype"] = _dtype_codes()
    codes["dtype_kind"] = _dtype_kind_codes()
    codes["casting"] = _casting_codes()
    codes["binary"] = _binary_codes()
    codes["unary"] = _unary_codes()
    codes["compare"] = _compare_codes()
    codes["logical"] = _logical_codes()
    codes["predicate"] = _predicate_codes()
    codes["reduce"] = _reduce_codes()
    codes["backend"] = _backend_codes()
    return codes^
