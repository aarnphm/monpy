@fieldwise_init
struct ArrayDType(Equatable, ImplicitlyCopyable, Movable):
    var value: Int

    comptime BOOL = Self(0)
    comptime INT64 = Self(1)
    comptime FLOAT32 = Self(2)
    comptime FLOAT64 = Self(3)
    comptime INT32 = Self(4)
    comptime INT16 = Self(5)
    comptime INT8 = Self(6)
    comptime UINT64 = Self(7)
    comptime UINT32 = Self(8)
    comptime UINT16 = Self(9)
    comptime UINT8 = Self(10)
    comptime FLOAT16 = Self(11)
    comptime COMPLEX64 = Self(12)
    comptime COMPLEX128 = Self(13)

    @staticmethod
    def from_int(value: Int) raises -> Self:
        if value >= Self.BOOL.value and value <= Self.COMPLEX128.value:
            return Self(value)
        raise Error("unsupported dtype code")


@fieldwise_init
struct DTypeKind(Equatable, ImplicitlyCopyable, Movable):
    var value: Int

    comptime BOOL = Self(0)
    comptime SIGNED_INT = Self(1)
    comptime REAL_FLOAT = Self(2)
    comptime UNSIGNED_INT = Self(3)
    comptime COMPLEX_FLOAT = Self(4)

    @staticmethod
    def from_int(value: Int) raises -> Self:
        if value >= Self.BOOL.value and value <= Self.COMPLEX_FLOAT.value:
            return Self(value)
        raise Error("unsupported dtype kind code")


@fieldwise_init
struct CastingRule(Equatable, ImplicitlyCopyable, Movable):
    var value: Int

    comptime NO = Self(0)
    comptime EQUIV = Self(1)
    comptime SAFE = Self(2)
    comptime SAME_KIND = Self(3)
    comptime UNSAFE = Self(4)

    @staticmethod
    def from_int(value: Int) raises -> Self:
        if value >= Self.NO.value and value <= Self.UNSAFE.value:
            return Self(value)
        raise Error("unsupported casting code")


@fieldwise_init
struct BinaryOp(Equatable, ImplicitlyCopyable, Movable):
    var value: Int

    comptime ADD = Self(0)
    comptime SUB = Self(1)
    comptime MUL = Self(2)
    comptime DIV = Self(3)
    comptime FLOOR_DIV = Self(4)
    comptime MOD = Self(5)
    comptime POWER = Self(6)
    comptime MAXIMUM = Self(7)
    comptime MINIMUM = Self(8)
    comptime FMIN = Self(9)
    comptime FMAX = Self(10)
    comptime ARCTAN2 = Self(11)
    comptime HYPOT = Self(12)
    comptime COPYSIGN = Self(13)

    @staticmethod
    def from_int(value: Int) raises -> Self:
        if value >= Self.ADD.value and value <= Self.COPYSIGN.value:
            return Self(value)
        raise Error("unsupported binary op")


@fieldwise_init
struct UnaryOp(Equatable, ImplicitlyCopyable, Movable):
    var value: Int

    comptime SIN = Self(0)
    comptime COS = Self(1)
    comptime EXP = Self(2)
    comptime LOG = Self(3)
    comptime TAN = Self(4)
    comptime ARCSIN = Self(5)
    comptime ARCCOS = Self(6)
    comptime ARCTAN = Self(7)
    comptime SINH = Self(8)
    comptime COSH = Self(9)
    comptime TANH = Self(10)
    comptime LOG1P = Self(11)
    comptime LOG2 = Self(12)
    comptime LOG10 = Self(13)
    comptime EXP2 = Self(14)
    comptime EXPM1 = Self(15)
    comptime SQRT = Self(16)
    comptime CBRT = Self(17)
    comptime DEG2RAD = Self(18)
    comptime RAD2DEG = Self(19)
    comptime RECIPROCAL = Self(20)
    comptime NEGATE = Self(30)
    comptime POSITIVE = Self(31)
    comptime ABS = Self(32)
    comptime SQUARE = Self(33)
    comptime SIGN = Self(34)
    comptime FLOOR = Self(35)
    comptime CEIL = Self(36)
    comptime TRUNC = Self(37)
    comptime RINT = Self(38)
    comptime LOGICAL_NOT = Self(39)
    comptime CONJUGATE = Self(40)

    @staticmethod
    def from_int(value: Int) raises -> Self:
        if (value >= Self.SIN.value and value <= Self.RECIPROCAL.value) or (
            value >= Self.NEGATE.value and value <= Self.CONJUGATE.value
        ):
            return Self(value)
        raise Error("unsupported unary op")


@fieldwise_init
struct CompareOp(Equatable, ImplicitlyCopyable, Movable):
    var value: Int

    comptime EQ = Self(0)
    comptime NE = Self(1)
    comptime LT = Self(2)
    comptime LE = Self(3)
    comptime GT = Self(4)
    comptime GE = Self(5)

    @staticmethod
    def from_int(value: Int) raises -> Self:
        if value >= Self.EQ.value and value <= Self.GE.value:
            return Self(value)
        raise Error("unsupported comparison op")


@fieldwise_init
struct LogicalOp(Equatable, ImplicitlyCopyable, Movable):
    var value: Int

    comptime AND = Self(0)
    comptime OR = Self(1)
    comptime XOR = Self(2)

    @staticmethod
    def from_int(value: Int) raises -> Self:
        if value >= Self.AND.value and value <= Self.XOR.value:
            return Self(value)
        raise Error("unsupported logical op")


@fieldwise_init
struct PredicateOp(Equatable, ImplicitlyCopyable, Movable):
    var value: Int

    comptime ISNAN = Self(0)
    comptime ISINF = Self(1)
    comptime ISFINITE = Self(2)
    comptime SIGNBIT = Self(3)

    @staticmethod
    def from_int(value: Int) raises -> Self:
        if value >= Self.ISNAN.value and value <= Self.SIGNBIT.value:
            return Self(value)
        raise Error("unsupported predicate op")


@fieldwise_init
struct ReduceOp(Equatable, ImplicitlyCopyable, Movable):
    var value: Int

    comptime SUM = Self(0)
    comptime MEAN = Self(1)
    comptime MIN = Self(2)
    comptime MAX = Self(3)
    comptime ARGMAX = Self(4)
    comptime PROD = Self(5)
    comptime ALL = Self(6)
    comptime ANY = Self(7)
    comptime ARGMIN = Self(8)

    @staticmethod
    def from_int(value: Int) raises -> Self:
        if value >= Self.SUM.value and value <= Self.ARGMIN.value:
            return Self(value)
        raise Error("unsupported reduction op")


@fieldwise_init
struct BackendKind(Equatable, ImplicitlyCopyable, Movable):
    var value: Int

    comptime GENERIC = Self(0)
    comptime ACCELERATE = Self(1)
    comptime FUSED = Self(2)

    @staticmethod
    def from_int(value: Int) raises -> Self:
        if value >= Self.GENERIC.value and value <= Self.FUSED.value:
            return Self(value)
        raise Error("unsupported backend code")


def dtype_item_size(dtype_code: Int) raises -> Int:
    if (
        dtype_code == ArrayDType.BOOL.value
        or dtype_code == ArrayDType.INT8.value
        or dtype_code == ArrayDType.UINT8.value
    ):
        return 1
    if (
        dtype_code == ArrayDType.INT16.value
        or dtype_code == ArrayDType.UINT16.value
        or dtype_code == ArrayDType.FLOAT16.value
    ):
        return 2
    if (
        dtype_code == ArrayDType.INT32.value
        or dtype_code == ArrayDType.FLOAT32.value
        or dtype_code == ArrayDType.UINT32.value
    ):
        return 4
    if (
        dtype_code == ArrayDType.INT64.value
        or dtype_code == ArrayDType.FLOAT64.value
        or dtype_code == ArrayDType.UINT64.value
    ):
        return 8
    if dtype_code == ArrayDType.COMPLEX64.value:
        return 8  # 2 × float32
    if dtype_code == ArrayDType.COMPLEX128.value:
        return 16  # 2 × float64
    raise Error("unsupported dtype code")


def dtype_alignment(dtype_code: Int) raises -> Int:
    return dtype_item_size(dtype_code)


def _is_signed_int_code(code: Int) -> Bool:
    return (
        code == ArrayDType.INT64.value
        or code == ArrayDType.INT32.value
        or code == ArrayDType.INT16.value
        or code == ArrayDType.INT8.value
    )


def _is_unsigned_int_code(code: Int) -> Bool:
    return (
        code == ArrayDType.UINT64.value
        or code == ArrayDType.UINT32.value
        or code == ArrayDType.UINT16.value
        or code == ArrayDType.UINT8.value
    )


def _is_float_code(code: Int) -> Bool:
    return code == ArrayDType.FLOAT64.value or code == ArrayDType.FLOAT32.value or code == ArrayDType.FLOAT16.value


def _is_complex_code(code: Int) -> Bool:
    return code == ArrayDType.COMPLEX64.value or code == ArrayDType.COMPLEX128.value


def dtype_kind_code(dtype_code: Int) raises -> Int:
    if dtype_code == ArrayDType.BOOL.value:
        return DTypeKind.BOOL.value
    if _is_signed_int_code(dtype_code):
        return DTypeKind.SIGNED_INT.value
    if _is_unsigned_int_code(dtype_code):
        return DTypeKind.UNSIGNED_INT.value
    if _is_float_code(dtype_code):
        return DTypeKind.REAL_FLOAT.value
    if _is_complex_code(dtype_code):
        return DTypeKind.COMPLEX_FLOAT.value
    raise Error("unsupported dtype code")


def dtype_code_from_format_char(c: Int, itemsize: Int) raises -> Int:
    # Python buffers expose PEP-3118 format chars. Keep this compact decode
    # next to dtype metadata so every import bridge resolves the same dtype ids.
    if c == 0x3F and itemsize == 1:  # '?'
        return ArrayDType.BOOL.value
    if c == 0x62 and itemsize == 1:  # 'b'
        return ArrayDType.INT8.value
    if c == 0x42 and itemsize == 1:  # 'B'
        return ArrayDType.UINT8.value
    if c == 0x68 and itemsize == 2:  # 'h'
        return ArrayDType.INT16.value
    if c == 0x48 and itemsize == 2:  # 'H'
        return ArrayDType.UINT16.value
    if c == 0x69 and itemsize == 4:  # 'i'
        return ArrayDType.INT32.value
    if c == 0x49 and itemsize == 4:  # 'I'
        return ArrayDType.UINT32.value
    if (c == 0x6C or c == 0x71) and itemsize == 8:  # 'l' or 'q'
        return ArrayDType.INT64.value
    if (c == 0x4C or c == 0x51) and itemsize == 8:  # 'L' or 'Q'
        return ArrayDType.UINT64.value
    # 'l'/'L' on 32-bit Linux is 4 bytes — interpret as int32/uint32.
    if c == 0x6C and itemsize == 4:
        return ArrayDType.INT32.value
    if c == 0x4C and itemsize == 4:
        return ArrayDType.UINT32.value
    if c == 0x65 and itemsize == 2:  # 'e' (float16, PEP-3118 + numpy convention)
        return ArrayDType.FLOAT16.value
    if c == 0x66 and itemsize == 4:  # 'f'
        return ArrayDType.FLOAT32.value
    if c == 0x64 and itemsize == 8:  # 'd'
        return ArrayDType.FLOAT64.value
    if c == 0x46 and itemsize == 8:  # 'F' (complex64 = 2 × float32 = 8 bytes)
        return ArrayDType.COMPLEX64.value
    if c == 0x44 and itemsize == 16:  # 'D' (complex128 = 2 × float64 = 16 bytes)
        return ArrayDType.COMPLEX128.value
    if c == 0x5A and itemsize == 8:  # 'Z' = numpy struct typestr; resolves complex64
        return ArrayDType.COMPLEX64.value
    if c == 0x5A and itemsize == 16:  # 'Z' for complex128
        return ArrayDType.COMPLEX128.value
    raise Error("buffer format unsupported by monpy")


def dtype_result_for_unary(dtype_code: Int) -> Int:
    if dtype_code == ArrayDType.FLOAT16.value:
        return ArrayDType.FLOAT16.value
    if dtype_code == ArrayDType.FLOAT32.value:
        return ArrayDType.FLOAT32.value
    if dtype_code == ArrayDType.COMPLEX64.value:
        return ArrayDType.COMPLEX64.value
    if dtype_code == ArrayDType.COMPLEX128.value:
        return ArrayDType.COMPLEX128.value
    return ArrayDType.FLOAT64.value


def dtype_result_for_unary_preserve(dtype_code: Int) -> Int:
    """
    - preserve-dtype unary ops (negate/abs/square/positive/floor/ceil/trunc/rint).
    - bool inputs get promoted to int64 because numpy treats `~bool_arr` style ops as integer transforms;
    - same pattern for negate.
    """
    if dtype_code == ArrayDType.BOOL.value:
        return ArrayDType.INT64.value
    return dtype_code


def _is_int_code(code: Int) -> Bool:
    return _is_signed_int_code(code) or _is_unsigned_int_code(code)


def _signed_int_size_to_code(size: Int) raises -> Int:
    if size == 1:
        return ArrayDType.INT8.value
    if size == 2:
        return ArrayDType.INT16.value
    if size == 4:
        return ArrayDType.INT32.value
    return ArrayDType.INT64.value


def _unsigned_int_size_to_code(size: Int) raises -> Int:
    if size == 1:
        return ArrayDType.UINT8.value
    if size == 2:
        return ArrayDType.UINT16.value
    if size == 4:
        return ArrayDType.UINT32.value
    return ArrayDType.UINT64.value


def _wider_signed_int(a: Int, b: Int) raises -> Int:
    var sa = dtype_item_size(a)
    var sb = dtype_item_size(b)
    if sa >= sb:
        return a
    return b


def _wider_unsigned_int(a: Int, b: Int) raises -> Int:
    var sa = dtype_item_size(a)
    var sb = dtype_item_size(b)
    if sa >= sb:
        return a
    return b


def _signed_unsigned_promote(signed_code: Int, unsigned_code: Int) raises -> Int:
    # numpy 2.x: int_n + uint_n → next-wider signed (e.g. int8 + uint8 →
    # int16). int64 + uint64 → float64 because no integer holds both ranges.
    var ss = dtype_item_size(signed_code)
    var us = dtype_item_size(unsigned_code)
    if us < ss:
        return signed_code  # signed already wider; result stays signed
    if us == 8 and ss == 8:
        return ArrayDType.FLOAT64.value
    var promoted_size = us * 2
    if promoted_size >= 8:
        return ArrayDType.INT64.value
    return _signed_int_size_to_code(promoted_size)


def dtype_result_for_binary(lhs_dtype: Int, rhs_dtype: Int, op: Int) -> Int:
    """
    Numpy 2.x binary promotion.
    - 13-dtype matrix plus any operator-specific overrides (always-float transcendentals, division).
    """
    # Complex will absorbs everything: any pair with at least
    # one complex side promotes to the wider of the two complex types
    # (or complex64 for complex64+anything-non-complex128).
    if _is_complex_code(lhs_dtype) or _is_complex_code(rhs_dtype):
        if lhs_dtype == ArrayDType.COMPLEX128.value or rhs_dtype == ArrayDType.COMPLEX128.value:
            return ArrayDType.COMPLEX128.value
        # The other side may be a real that can't fit in complex64's f32 mantissa.
        var other = rhs_dtype if _is_complex_code(lhs_dtype) else lhs_dtype
        if other == ArrayDType.FLOAT64.value or other == ArrayDType.INT64.value or other == ArrayDType.UINT64.value:
            return ArrayDType.COMPLEX128.value
        if other == ArrayDType.INT32.value or other == ArrayDType.UINT32.value:
            return ArrayDType.COMPLEX128.value
        return ArrayDType.COMPLEX64.value
    # Always-float binary transcendentals.
    if op == BinaryOp.ARCTAN2.value or op == BinaryOp.HYPOT.value or op == BinaryOp.COPYSIGN.value:
        if lhs_dtype == ArrayDType.FLOAT32.value and rhs_dtype == ArrayDType.FLOAT32.value:
            return ArrayDType.FLOAT32.value
        if lhs_dtype == ArrayDType.FLOAT16.value and rhs_dtype == ArrayDType.FLOAT16.value:
            return ArrayDType.FLOAT16.value
        return ArrayDType.FLOAT64.value
    # Division: always float, with size-aware promotion.
    if op == BinaryOp.DIV.value:
        if lhs_dtype == ArrayDType.FLOAT64.value or rhs_dtype == ArrayDType.FLOAT64.value:
            return ArrayDType.FLOAT64.value
        if lhs_dtype == ArrayDType.FLOAT32.value or rhs_dtype == ArrayDType.FLOAT32.value:
            var other = rhs_dtype if lhs_dtype == ArrayDType.FLOAT32.value else lhs_dtype
            if (
                other == ArrayDType.FLOAT32.value
                or other == ArrayDType.BOOL.value
                or other == ArrayDType.FLOAT16.value
            ):
                return ArrayDType.FLOAT32.value
            if (
                other == ArrayDType.INT8.value
                or other == ArrayDType.INT16.value
                or other == ArrayDType.UINT8.value
                or other == ArrayDType.UINT16.value
            ):
                return ArrayDType.FLOAT32.value
        if lhs_dtype == ArrayDType.FLOAT16.value or rhs_dtype == ArrayDType.FLOAT16.value:
            var other = rhs_dtype if lhs_dtype == ArrayDType.FLOAT16.value else lhs_dtype
            if other == ArrayDType.FLOAT16.value or other == ArrayDType.BOOL.value:
                return ArrayDType.FLOAT16.value
            if other == ArrayDType.INT8.value or other == ArrayDType.UINT8.value:
                return ArrayDType.FLOAT16.value
            return ArrayDType.FLOAT32.value
        return ArrayDType.FLOAT64.value
    # Non-division arithmetic.
    if lhs_dtype == ArrayDType.FLOAT64.value or rhs_dtype == ArrayDType.FLOAT64.value:
        return ArrayDType.FLOAT64.value
    if lhs_dtype == ArrayDType.FLOAT32.value or rhs_dtype == ArrayDType.FLOAT32.value:
        var other = rhs_dtype if lhs_dtype == ArrayDType.FLOAT32.value else lhs_dtype
        # int32+/int64/uint32+/uint64 + float32 → float64.
        if other == ArrayDType.INT64.value or other == ArrayDType.INT32.value:
            return ArrayDType.FLOAT64.value
        if other == ArrayDType.UINT64.value or other == ArrayDType.UINT32.value:
            return ArrayDType.FLOAT64.value
        return ArrayDType.FLOAT32.value
    if lhs_dtype == ArrayDType.FLOAT16.value or rhs_dtype == ArrayDType.FLOAT16.value:
        var other = rhs_dtype if lhs_dtype == ArrayDType.FLOAT16.value else lhs_dtype
        if other == ArrayDType.FLOAT16.value or other == ArrayDType.BOOL.value:
            return ArrayDType.FLOAT16.value
        if other == ArrayDType.INT8.value or other == ArrayDType.UINT8.value:
            return ArrayDType.FLOAT16.value
        if other == ArrayDType.INT16.value or other == ArrayDType.UINT16.value:
            return ArrayDType.FLOAT32.value
        # bigger ints with f16 → f64 (not enough mantissa).
        return ArrayDType.FLOAT64.value
    # All-integer / bool path.
    if _is_int_code(lhs_dtype) or _is_int_code(rhs_dtype):
        var l_signed = _is_signed_int_code(lhs_dtype)
        var l_unsigned = _is_unsigned_int_code(lhs_dtype)
        var r_signed = _is_signed_int_code(rhs_dtype)
        var r_unsigned = _is_unsigned_int_code(rhs_dtype)
        # bool propagates to whatever the other side is.
        if not _is_int_code(lhs_dtype):
            return rhs_dtype
        if not _is_int_code(rhs_dtype):
            return lhs_dtype
        # both integer.
        try:
            if l_signed and r_signed:
                return _wider_signed_int(lhs_dtype, rhs_dtype)
            if l_unsigned and r_unsigned:
                return _wider_unsigned_int(lhs_dtype, rhs_dtype)
            if l_signed and r_unsigned:
                return _signed_unsigned_promote(lhs_dtype, rhs_dtype)
            return _signed_unsigned_promote(rhs_dtype, lhs_dtype)
        except:
            return ArrayDType.INT64.value
    return ArrayDType.BOOL.value


def dtype_result_for_reduction(dtype_code: Int, op: Int) -> Int:
    if op == ReduceOp.MEAN.value:
        if dtype_code == ArrayDType.FLOAT16.value or dtype_code == ArrayDType.FLOAT32.value:
            return ArrayDType.FLOAT32.value
        return ArrayDType.FLOAT64.value
    if op == ReduceOp.SUM.value and dtype_code == ArrayDType.BOOL.value:
        return ArrayDType.INT64.value
    if op == ReduceOp.SUM.value or op == ReduceOp.PROD.value:
        # numpy: small-int reductions accumulate in int64/uint64 to avoid overflow.
        if _is_signed_int_code(dtype_code):
            return ArrayDType.INT64.value
        if _is_unsigned_int_code(dtype_code):
            return ArrayDType.UINT64.value
    return dtype_code


def dtype_result_for_linalg(dtype_code: Int) -> Int:
    if dtype_code == ArrayDType.FLOAT32.value:
        return ArrayDType.FLOAT32.value
    return ArrayDType.FLOAT64.value


def dtype_result_for_linalg_binary(lhs_dtype: Int, rhs_dtype: Int) -> Int:
    if lhs_dtype == ArrayDType.FLOAT32.value and rhs_dtype == ArrayDType.FLOAT32.value:
        return ArrayDType.FLOAT32.value
    return ArrayDType.FLOAT64.value


def dtype_promote_types(lhs_dtype: Int, rhs_dtype: Int) -> Int:
    return dtype_result_for_binary(lhs_dtype, rhs_dtype, BinaryOp.ADD.value)


def dtype_can_cast(from_dtype: Int, to_dtype: Int, casting: Int) raises -> Bool:
    var from_kind = dtype_kind_code(from_dtype)
    var to_kind = dtype_kind_code(to_dtype)
    if from_dtype == to_dtype:
        return True
    if casting == CastingRule.NO.value or casting == CastingRule.EQUIV.value:
        return False
    if casting == CastingRule.UNSAFE.value:
        return True
    # Complex has its own casting rules: real → complex always safe;
    # complex → real never safe (lossy); complex → complex widens.
    if from_kind == DTypeKind.COMPLEX_FLOAT.value:
        if to_kind == DTypeKind.COMPLEX_FLOAT.value:
            if casting == CastingRule.SAFE.value:
                # complex64 → complex128 safe; reverse not.
                return from_dtype == ArrayDType.COMPLEX64.value and to_dtype == ArrayDType.COMPLEX128.value
            return casting == CastingRule.SAME_KIND.value
        return False
    if to_kind == DTypeKind.COMPLEX_FLOAT.value:
        if casting == CastingRule.SAFE.value:
            # Real → complex safe if real fits the complex's float component.
            if to_dtype == ArrayDType.COMPLEX128.value:
                return True  # any real fits in complex128's f64 component
            # to_dtype == ArrayDType.COMPLEX64.value
            if from_kind == DTypeKind.BOOL.value:
                return True
            if from_dtype == ArrayDType.FLOAT32.value or from_dtype == ArrayDType.FLOAT16.value:
                return True
            if from_dtype == ArrayDType.INT8.value or from_dtype == ArrayDType.INT16.value:
                return True
            if from_dtype == ArrayDType.UINT8.value or from_dtype == ArrayDType.UINT16.value:
                return True
            return False
        return casting == CastingRule.SAME_KIND.value
    if casting == CastingRule.SAFE.value:
        # Bool fits in any numeric type.
        if from_kind == DTypeKind.BOOL.value:
            return True
        if from_kind == DTypeKind.SIGNED_INT.value:
            if to_kind == DTypeKind.SIGNED_INT.value:
                var fs = dtype_item_size(from_dtype)
                var ts = dtype_item_size(to_dtype)
                return ts >= fs
            if to_kind == DTypeKind.REAL_FLOAT.value:
                var fs = dtype_item_size(from_dtype)
                if to_dtype == ArrayDType.FLOAT64.value:
                    return True
                if to_dtype == ArrayDType.FLOAT32.value:
                    return fs <= 2
                if to_dtype == ArrayDType.FLOAT16.value:
                    return False  # int → f16 not safe (mantissa = 10 bits)
            # signed → unsigned never safe (negative values lose).
            return False
        if from_kind == DTypeKind.UNSIGNED_INT.value:
            if to_kind == DTypeKind.UNSIGNED_INT.value:
                var fs = dtype_item_size(from_dtype)
                var ts = dtype_item_size(to_dtype)
                return ts >= fs
            if to_kind == DTypeKind.SIGNED_INT.value:
                # uint_n fits safely in int_{2n}: uint8→int16, uint16→int32, uint32→int64.
                # uint64 → no signed int holds full range.
                var fs = dtype_item_size(from_dtype)
                var ts = dtype_item_size(to_dtype)
                return ts > fs
            if to_kind == DTypeKind.REAL_FLOAT.value:
                if to_dtype == ArrayDType.FLOAT64.value:
                    return True
                if to_dtype == ArrayDType.FLOAT32.value:
                    return dtype_item_size(from_dtype) <= 2
                if to_dtype == ArrayDType.FLOAT16.value:
                    return False
            return False
        if from_kind == DTypeKind.REAL_FLOAT.value:
            if to_kind == DTypeKind.REAL_FLOAT.value:
                var fs = dtype_item_size(from_dtype)
                var ts = dtype_item_size(to_dtype)
                return ts >= fs
            return False
        return False
    if casting == CastingRule.SAME_KIND.value:
        if from_kind == DTypeKind.BOOL.value:
            return True
        if from_kind == DTypeKind.SIGNED_INT.value:
            return (
                to_kind == DTypeKind.SIGNED_INT.value
                or to_kind == DTypeKind.UNSIGNED_INT.value
                or to_kind == DTypeKind.REAL_FLOAT.value
            )
        if from_kind == DTypeKind.UNSIGNED_INT.value:
            return (
                to_kind == DTypeKind.UNSIGNED_INT.value
                or to_kind == DTypeKind.SIGNED_INT.value
                or to_kind == DTypeKind.REAL_FLOAT.value
            )
        if from_kind == DTypeKind.REAL_FLOAT.value:
            return to_kind == DTypeKind.REAL_FLOAT.value
        return False
    raise Error("unknown casting policy")
