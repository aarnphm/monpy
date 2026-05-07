@fieldwise_init
struct ArrayDType(ImplicitlyCopyable, Movable):
    var value: Int


@fieldwise_init
struct BinaryOp(ImplicitlyCopyable, Movable):
    var value: Int


@fieldwise_init
struct UnaryOp(ImplicitlyCopyable, Movable):
    var value: Int


@fieldwise_init
struct ReduceOp(ImplicitlyCopyable, Movable):
    var value: Int


@fieldwise_init
struct BackendKind(ImplicitlyCopyable, Movable):
    var value: Int


# Python bindings still pass compact integer codes. The strongly named wrappers
# above are the internal migration target; these constants keep the existing
# private wire contract stable while the operation modules move over.
comptime DTYPE_BOOL = 0
comptime DTYPE_INT64 = 1
comptime DTYPE_FLOAT32 = 2
comptime DTYPE_FLOAT64 = 3
# Phase-5a signed ints.
comptime DTYPE_INT32 = 4
comptime DTYPE_INT16 = 5
comptime DTYPE_INT8 = 6
# Phase-5b unsigned ints. Allocation + arithmetic via the f64
# round-trip path; promotion rules follow numpy 2.x.
comptime DTYPE_UINT64 = 7
comptime DTYPE_UINT32 = 8
comptime DTYPE_UINT16 = 9
comptime DTYPE_UINT8 = 10
# Phase-5c float16. Stored as 2-byte half; arithmetic delegates through
# the f64 round-trip path until a SIMD float16 kernel lands.
comptime DTYPE_FLOAT16 = 11
# Phase-5d complex. Interleaved (real, imag) storage per numpy convention.
# complex64 = 2 × float32 = 8 bytes. complex128 = 2 × float64 = 16 bytes.
comptime DTYPE_COMPLEX64 = 12
comptime DTYPE_COMPLEX128 = 13

comptime DTYPE_KIND_BOOL = 0
comptime DTYPE_KIND_SIGNED_INT = 1
comptime DTYPE_KIND_REAL_FLOAT = 2
comptime DTYPE_KIND_UNSIGNED_INT = 3
comptime DTYPE_KIND_COMPLEX_FLOAT = 4

comptime CASTING_NO = 0
comptime CASTING_EQUIV = 1
comptime CASTING_SAFE = 2
comptime CASTING_SAME_KIND = 3
comptime CASTING_UNSAFE = 4

comptime OP_ADD = 0
comptime OP_SUB = 1
comptime OP_MUL = 2
comptime OP_DIV = 3
# binary ops (all numeric; promotion handled in dtype_result_for_binary).
comptime OP_FLOOR_DIV = 4
comptime OP_MOD = 5
comptime OP_POWER = 6
comptime OP_MAXIMUM = 7
comptime OP_MINIMUM = 8
comptime OP_FMIN = 9
comptime OP_FMAX = 10
comptime OP_ARCTAN2 = 11
comptime OP_HYPOT = 12
comptime OP_COPYSIGN = 13

comptime UNARY_SIN = 0
comptime UNARY_COS = 1
comptime UNARY_EXP = 2
comptime UNARY_LOG = 3
# unary transcendentals (all float-only; promote int → float64).
comptime UNARY_TAN = 4
comptime UNARY_ARCSIN = 5
comptime UNARY_ARCCOS = 6
comptime UNARY_ARCTAN = 7
comptime UNARY_SINH = 8
comptime UNARY_COSH = 9
comptime UNARY_TANH = 10
comptime UNARY_LOG1P = 11
comptime UNARY_LOG2 = 12
comptime UNARY_LOG10 = 13
comptime UNARY_EXP2 = 14
comptime UNARY_EXPM1 = 15
comptime UNARY_SQRT = 16
comptime UNARY_CBRT = 17
comptime UNARY_DEG2RAD = 18
comptime UNARY_RAD2DEG = 19
comptime UNARY_RECIPROCAL = 20
# unary arith (preserves dtype kind: int → int, float → float).
comptime UNARY_NEGATE = 30
comptime UNARY_POSITIVE = 31
comptime UNARY_ABS = 32
comptime UNARY_SQUARE = 33
comptime UNARY_SIGN = 34
comptime UNARY_FLOOR = 35
comptime UNARY_CEIL = 36
comptime UNARY_TRUNC = 37
comptime UNARY_RINT = 38
comptime UNARY_LOGICAL_NOT = 39
# complex-only unary ops. CONJ flips imag sign; REAL/IMAG/ANGLE
# return a real-valued result (handled at python level).
comptime UNARY_CONJUGATE = 40

comptime CMP_EQ = 0
comptime CMP_NE = 1
comptime CMP_LT = 2
comptime CMP_LE = 3
comptime CMP_GT = 4
comptime CMP_GE = 5

comptime LOGIC_AND = 0
comptime LOGIC_OR = 1
comptime LOGIC_XOR = 2

comptime PRED_ISNAN = 0
comptime PRED_ISINF = 1
comptime PRED_ISFINITE = 2
comptime PRED_SIGNBIT = 3

comptime REDUCE_SUM = 0
comptime REDUCE_MEAN = 1
comptime REDUCE_MIN = 2
comptime REDUCE_MAX = 3
comptime REDUCE_ARGMAX = 4
comptime REDUCE_PROD = 5
comptime REDUCE_ALL = 6
comptime REDUCE_ANY = 7
comptime REDUCE_ARGMIN = 8

comptime BACKEND_GENERIC = 0
comptime BACKEND_ACCELERATE = 1
comptime BACKEND_FUSED = 2


def dtype_item_size(dtype_code: Int) raises -> Int:
    if dtype_code == DTYPE_BOOL or dtype_code == DTYPE_INT8 or dtype_code == DTYPE_UINT8:
        return 1
    if dtype_code == DTYPE_INT16 or dtype_code == DTYPE_UINT16 or dtype_code == DTYPE_FLOAT16:
        return 2
    if dtype_code == DTYPE_INT32 or dtype_code == DTYPE_FLOAT32 or dtype_code == DTYPE_UINT32:
        return 4
    if dtype_code == DTYPE_INT64 or dtype_code == DTYPE_FLOAT64 or dtype_code == DTYPE_UINT64:
        return 8
    if dtype_code == DTYPE_COMPLEX64:
        return 8  # 2 × float32
    if dtype_code == DTYPE_COMPLEX128:
        return 16  # 2 × float64
    raise Error("unsupported dtype code")


def dtype_alignment(dtype_code: Int) raises -> Int:
    return dtype_item_size(dtype_code)


def _is_signed_int_code(code: Int) -> Bool:
    return code == DTYPE_INT64 or code == DTYPE_INT32 or code == DTYPE_INT16 or code == DTYPE_INT8


def _is_unsigned_int_code(code: Int) -> Bool:
    return code == DTYPE_UINT64 or code == DTYPE_UINT32 or code == DTYPE_UINT16 or code == DTYPE_UINT8


def _is_float_code(code: Int) -> Bool:
    return code == DTYPE_FLOAT64 or code == DTYPE_FLOAT32 or code == DTYPE_FLOAT16


def _is_complex_code(code: Int) -> Bool:
    return code == DTYPE_COMPLEX64 or code == DTYPE_COMPLEX128


def dtype_kind_code(dtype_code: Int) raises -> Int:
    if dtype_code == DTYPE_BOOL:
        return DTYPE_KIND_BOOL
    if _is_signed_int_code(dtype_code):
        return DTYPE_KIND_SIGNED_INT
    if _is_unsigned_int_code(dtype_code):
        return DTYPE_KIND_UNSIGNED_INT
    if _is_float_code(dtype_code):
        return DTYPE_KIND_REAL_FLOAT
    if _is_complex_code(dtype_code):
        return DTYPE_KIND_COMPLEX_FLOAT
    raise Error("unsupported dtype code")


def dtype_code_from_format_char(c: Int, itemsize: Int) raises -> Int:
    # Python buffers expose PEP-3118 format chars. Keep this compact decode
    # next to dtype metadata so every import bridge resolves the same dtype ids.
    if c == 0x3F and itemsize == 1:  # '?'
        return DTYPE_BOOL
    if c == 0x62 and itemsize == 1:  # 'b'
        return DTYPE_INT8
    if c == 0x42 and itemsize == 1:  # 'B'
        return DTYPE_UINT8
    if c == 0x68 and itemsize == 2:  # 'h'
        return DTYPE_INT16
    if c == 0x48 and itemsize == 2:  # 'H'
        return DTYPE_UINT16
    if c == 0x69 and itemsize == 4:  # 'i'
        return DTYPE_INT32
    if c == 0x49 and itemsize == 4:  # 'I'
        return DTYPE_UINT32
    if (c == 0x6C or c == 0x71) and itemsize == 8:  # 'l' or 'q'
        return DTYPE_INT64
    if (c == 0x4C or c == 0x51) and itemsize == 8:  # 'L' or 'Q'
        return DTYPE_UINT64
    # 'l'/'L' on 32-bit Linux is 4 bytes — interpret as int32/uint32.
    if c == 0x6C and itemsize == 4:
        return DTYPE_INT32
    if c == 0x4C and itemsize == 4:
        return DTYPE_UINT32
    if c == 0x65 and itemsize == 2:  # 'e' (float16, PEP-3118 + numpy convention)
        return DTYPE_FLOAT16
    if c == 0x66 and itemsize == 4:  # 'f'
        return DTYPE_FLOAT32
    if c == 0x64 and itemsize == 8:  # 'd'
        return DTYPE_FLOAT64
    if c == 0x46 and itemsize == 8:  # 'F' (complex64 = 2 × float32 = 8 bytes)
        return DTYPE_COMPLEX64
    if c == 0x44 and itemsize == 16:  # 'D' (complex128 = 2 × float64 = 16 bytes)
        return DTYPE_COMPLEX128
    if c == 0x5A and itemsize == 8:  # 'Z' = numpy struct typestr; resolves complex64
        return DTYPE_COMPLEX64
    if c == 0x5A and itemsize == 16:  # 'Z' for complex128
        return DTYPE_COMPLEX128
    raise Error("buffer format unsupported by monpy")


def dtype_result_for_unary(dtype_code: Int) -> Int:
    if dtype_code == DTYPE_FLOAT16:
        return DTYPE_FLOAT16
    if dtype_code == DTYPE_FLOAT32:
        return DTYPE_FLOAT32
    if dtype_code == DTYPE_COMPLEX64:
        return DTYPE_COMPLEX64
    if dtype_code == DTYPE_COMPLEX128:
        return DTYPE_COMPLEX128
    return DTYPE_FLOAT64


def dtype_result_for_unary_preserve(dtype_code: Int) -> Int:
    """
    - preserve-dtype unary ops (negate/abs/square/positive/floor/ceil/trunc/rint).
    - bool inputs get promoted to int64 because numpy treats `~bool_arr` style ops as integer transforms;
    - same pattern for negate.
    """
    if dtype_code == DTYPE_BOOL:
        return DTYPE_INT64
    return dtype_code


def _is_int_code(code: Int) -> Bool:
    return _is_signed_int_code(code) or _is_unsigned_int_code(code)


def _signed_int_size_to_code(size: Int) raises -> Int:
    if size == 1:
        return DTYPE_INT8
    if size == 2:
        return DTYPE_INT16
    if size == 4:
        return DTYPE_INT32
    return DTYPE_INT64


def _unsigned_int_size_to_code(size: Int) raises -> Int:
    if size == 1:
        return DTYPE_UINT8
    if size == 2:
        return DTYPE_UINT16
    if size == 4:
        return DTYPE_UINT32
    return DTYPE_UINT64


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
        return DTYPE_FLOAT64
    var promoted_size = us * 2
    if promoted_size >= 8:
        return DTYPE_INT64
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
        if lhs_dtype == DTYPE_COMPLEX128 or rhs_dtype == DTYPE_COMPLEX128:
            return DTYPE_COMPLEX128
        # The other side may be a real that can't fit in complex64's f32 mantissa.
        var other = rhs_dtype if _is_complex_code(lhs_dtype) else lhs_dtype
        if other == DTYPE_FLOAT64 or other == DTYPE_INT64 or other == DTYPE_UINT64:
            return DTYPE_COMPLEX128
        if other == DTYPE_INT32 or other == DTYPE_UINT32:
            return DTYPE_COMPLEX128
        return DTYPE_COMPLEX64
    # Always-float binary transcendentals.
    if op == OP_ARCTAN2 or op == OP_HYPOT or op == OP_COPYSIGN:
        if lhs_dtype == DTYPE_FLOAT32 and rhs_dtype == DTYPE_FLOAT32:
            return DTYPE_FLOAT32
        if lhs_dtype == DTYPE_FLOAT16 and rhs_dtype == DTYPE_FLOAT16:
            return DTYPE_FLOAT16
        return DTYPE_FLOAT64
    # Division: always float, with size-aware promotion.
    if op == OP_DIV:
        if lhs_dtype == DTYPE_FLOAT64 or rhs_dtype == DTYPE_FLOAT64:
            return DTYPE_FLOAT64
        if lhs_dtype == DTYPE_FLOAT32 or rhs_dtype == DTYPE_FLOAT32:
            var other = rhs_dtype if lhs_dtype == DTYPE_FLOAT32 else lhs_dtype
            if other == DTYPE_FLOAT32 or other == DTYPE_BOOL or other == DTYPE_FLOAT16:
                return DTYPE_FLOAT32
            if other == DTYPE_INT8 or other == DTYPE_INT16 or other == DTYPE_UINT8 or other == DTYPE_UINT16:
                return DTYPE_FLOAT32
        if lhs_dtype == DTYPE_FLOAT16 or rhs_dtype == DTYPE_FLOAT16:
            var other = rhs_dtype if lhs_dtype == DTYPE_FLOAT16 else lhs_dtype
            if other == DTYPE_FLOAT16 or other == DTYPE_BOOL:
                return DTYPE_FLOAT16
            if other == DTYPE_INT8 or other == DTYPE_UINT8:
                return DTYPE_FLOAT16
            return DTYPE_FLOAT32
        return DTYPE_FLOAT64
    # Non-division arithmetic.
    if lhs_dtype == DTYPE_FLOAT64 or rhs_dtype == DTYPE_FLOAT64:
        return DTYPE_FLOAT64
    if lhs_dtype == DTYPE_FLOAT32 or rhs_dtype == DTYPE_FLOAT32:
        var other = rhs_dtype if lhs_dtype == DTYPE_FLOAT32 else lhs_dtype
        # int32+/int64/uint32+/uint64 + float32 → float64.
        if other == DTYPE_INT64 or other == DTYPE_INT32:
            return DTYPE_FLOAT64
        if other == DTYPE_UINT64 or other == DTYPE_UINT32:
            return DTYPE_FLOAT64
        return DTYPE_FLOAT32
    if lhs_dtype == DTYPE_FLOAT16 or rhs_dtype == DTYPE_FLOAT16:
        var other = rhs_dtype if lhs_dtype == DTYPE_FLOAT16 else lhs_dtype
        if other == DTYPE_FLOAT16 or other == DTYPE_BOOL:
            return DTYPE_FLOAT16
        if other == DTYPE_INT8 or other == DTYPE_UINT8:
            return DTYPE_FLOAT16
        if other == DTYPE_INT16 or other == DTYPE_UINT16:
            return DTYPE_FLOAT32
        # bigger ints with f16 → f64 (not enough mantissa).
        return DTYPE_FLOAT64
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
            return DTYPE_INT64
    return DTYPE_BOOL


def dtype_result_for_reduction(dtype_code: Int, op: Int) -> Int:
    if op == REDUCE_MEAN:
        if dtype_code == DTYPE_FLOAT16 or dtype_code == DTYPE_FLOAT32:
            return DTYPE_FLOAT32
        return DTYPE_FLOAT64
    if op == REDUCE_SUM and dtype_code == DTYPE_BOOL:
        return DTYPE_INT64
    if op == REDUCE_SUM or op == REDUCE_PROD:
        # numpy: small-int reductions accumulate in int64/uint64 to avoid overflow.
        if _is_signed_int_code(dtype_code):
            return DTYPE_INT64
        if _is_unsigned_int_code(dtype_code):
            return DTYPE_UINT64
    return dtype_code


def dtype_result_for_linalg(dtype_code: Int) -> Int:
    if dtype_code == DTYPE_FLOAT32:
        return DTYPE_FLOAT32
    return DTYPE_FLOAT64


def dtype_result_for_linalg_binary(lhs_dtype: Int, rhs_dtype: Int) -> Int:
    if lhs_dtype == DTYPE_FLOAT32 and rhs_dtype == DTYPE_FLOAT32:
        return DTYPE_FLOAT32
    return DTYPE_FLOAT64


def dtype_promote_types(lhs_dtype: Int, rhs_dtype: Int) -> Int:
    return dtype_result_for_binary(lhs_dtype, rhs_dtype, OP_ADD)


def dtype_can_cast(from_dtype: Int, to_dtype: Int, casting: Int) raises -> Bool:
    var from_kind = dtype_kind_code(from_dtype)
    var to_kind = dtype_kind_code(to_dtype)
    if from_dtype == to_dtype:
        return True
    if casting == CASTING_NO or casting == CASTING_EQUIV:
        return False
    if casting == CASTING_UNSAFE:
        return True
    # Complex has its own casting rules: real → complex always safe;
    # complex → real never safe (lossy); complex → complex widens.
    if from_kind == DTYPE_KIND_COMPLEX_FLOAT:
        if to_kind == DTYPE_KIND_COMPLEX_FLOAT:
            if casting == CASTING_SAFE:
                # complex64 → complex128 safe; reverse not.
                return from_dtype == DTYPE_COMPLEX64 and to_dtype == DTYPE_COMPLEX128
            return casting == CASTING_SAME_KIND
        return False
    if to_kind == DTYPE_KIND_COMPLEX_FLOAT:
        if casting == CASTING_SAFE:
            # Real → complex safe if real fits the complex's float component.
            if to_dtype == DTYPE_COMPLEX128:
                return True  # any real fits in complex128's f64 component
            # to_dtype == DTYPE_COMPLEX64
            if from_kind == DTYPE_KIND_BOOL:
                return True
            if from_dtype == DTYPE_FLOAT32 or from_dtype == DTYPE_FLOAT16:
                return True
            if from_dtype == DTYPE_INT8 or from_dtype == DTYPE_INT16:
                return True
            if from_dtype == DTYPE_UINT8 or from_dtype == DTYPE_UINT16:
                return True
            return False
        return casting == CASTING_SAME_KIND
    if casting == CASTING_SAFE:
        # Bool fits in any numeric type.
        if from_kind == DTYPE_KIND_BOOL:
            return True
        if from_kind == DTYPE_KIND_SIGNED_INT:
            if to_kind == DTYPE_KIND_SIGNED_INT:
                var fs = dtype_item_size(from_dtype)
                var ts = dtype_item_size(to_dtype)
                return ts >= fs
            if to_kind == DTYPE_KIND_REAL_FLOAT:
                var fs = dtype_item_size(from_dtype)
                if to_dtype == DTYPE_FLOAT64:
                    return True
                if to_dtype == DTYPE_FLOAT32:
                    return fs <= 2
                if to_dtype == DTYPE_FLOAT16:
                    return False  # int → f16 not safe (mantissa = 10 bits)
            # signed → unsigned never safe (negative values lose).
            return False
        if from_kind == DTYPE_KIND_UNSIGNED_INT:
            if to_kind == DTYPE_KIND_UNSIGNED_INT:
                var fs = dtype_item_size(from_dtype)
                var ts = dtype_item_size(to_dtype)
                return ts >= fs
            if to_kind == DTYPE_KIND_SIGNED_INT:
                # uint_n fits safely in int_{2n}: uint8→int16, uint16→int32, uint32→int64.
                # uint64 → no signed int holds full range.
                var fs = dtype_item_size(from_dtype)
                var ts = dtype_item_size(to_dtype)
                return ts > fs
            if to_kind == DTYPE_KIND_REAL_FLOAT:
                if to_dtype == DTYPE_FLOAT64:
                    return True
                if to_dtype == DTYPE_FLOAT32:
                    return dtype_item_size(from_dtype) <= 2
                if to_dtype == DTYPE_FLOAT16:
                    return False
            return False
        if from_kind == DTYPE_KIND_REAL_FLOAT:
            if to_kind == DTYPE_KIND_REAL_FLOAT:
                var fs = dtype_item_size(from_dtype)
                var ts = dtype_item_size(to_dtype)
                return ts >= fs
            return False
        return False
    if casting == CASTING_SAME_KIND:
        if from_kind == DTYPE_KIND_BOOL:
            return True
        if from_kind == DTYPE_KIND_SIGNED_INT:
            return (
                to_kind == DTYPE_KIND_SIGNED_INT
                or to_kind == DTYPE_KIND_UNSIGNED_INT
                or to_kind == DTYPE_KIND_REAL_FLOAT
            )
        if from_kind == DTYPE_KIND_UNSIGNED_INT:
            return (
                to_kind == DTYPE_KIND_UNSIGNED_INT
                or to_kind == DTYPE_KIND_SIGNED_INT
                or to_kind == DTYPE_KIND_REAL_FLOAT
            )
        if from_kind == DTYPE_KIND_REAL_FLOAT:
            return to_kind == DTYPE_KIND_REAL_FLOAT
        return False
    raise Error("unknown casting policy")
