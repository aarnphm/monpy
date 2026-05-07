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
# Phase-5a registrations. Metadata is live (itemsize/kind/format-char
# resolve) so finfo/iinfo/can_cast/result_type can reason about them,
# but allocation and arithmetic kernels still error on these codes
# until the kernel work in elementwise.mojo lands.
comptime DTYPE_INT32 = 4
comptime DTYPE_INT16 = 5
comptime DTYPE_INT8 = 6

comptime DTYPE_KIND_BOOL = 0
comptime DTYPE_KIND_SIGNED_INT = 1
comptime DTYPE_KIND_REAL_FLOAT = 2

comptime CASTING_NO = 0
comptime CASTING_EQUIV = 1
comptime CASTING_SAFE = 2
comptime CASTING_SAME_KIND = 3
comptime CASTING_UNSAFE = 4

comptime OP_ADD = 0
comptime OP_SUB = 1
comptime OP_MUL = 2
comptime OP_DIV = 3
# Phase-3 binary ops (all numeric; promotion handled in dtype_result_for_binary).
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
# Phase-3 unary transcendentals (all float-only; promote int → float64).
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
# Phase-3 unary arith (preserves dtype kind: int → int, float → float).
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
    if dtype_code == DTYPE_BOOL or dtype_code == DTYPE_INT8:
        return 1
    if dtype_code == DTYPE_INT16:
        return 2
    if dtype_code == DTYPE_INT32 or dtype_code == DTYPE_FLOAT32:
        return 4
    if dtype_code == DTYPE_INT64 or dtype_code == DTYPE_FLOAT64:
        return 8
    raise Error("unsupported dtype code")


def dtype_alignment(dtype_code: Int) raises -> Int:
    return dtype_item_size(dtype_code)


def dtype_kind_code(dtype_code: Int) raises -> Int:
    if dtype_code == DTYPE_BOOL:
        return DTYPE_KIND_BOOL
    if (
        dtype_code == DTYPE_INT64
        or dtype_code == DTYPE_INT32
        or dtype_code == DTYPE_INT16
        or dtype_code == DTYPE_INT8
    ):
        return DTYPE_KIND_SIGNED_INT
    if dtype_code == DTYPE_FLOAT32 or dtype_code == DTYPE_FLOAT64:
        return DTYPE_KIND_REAL_FLOAT
    raise Error("unsupported dtype code")


def dtype_code_from_format_char(c: Int, itemsize: Int) raises -> Int:
    # Python buffers expose PEP-3118 format chars. Keep this compact decode
    # next to dtype metadata so every import bridge resolves the same dtype ids.
    if c == 0x3F and itemsize == 1:  # '?'
        return DTYPE_BOOL
    if c == 0x62 and itemsize == 1:  # 'b'
        return DTYPE_INT8
    if c == 0x68 and itemsize == 2:  # 'h'
        return DTYPE_INT16
    if c == 0x69 and itemsize == 4:  # 'i'
        return DTYPE_INT32
    if (c == 0x6C or c == 0x71) and itemsize == 8:  # 'l' or 'q'
        return DTYPE_INT64
    # 'l' on 32-bit Linux is 4 bytes — interpret as int32.
    if c == 0x6C and itemsize == 4:
        return DTYPE_INT32
    if c == 0x66 and itemsize == 4:  # 'f'
        return DTYPE_FLOAT32
    if c == 0x64 and itemsize == 8:  # 'd'
        return DTYPE_FLOAT64
    raise Error("buffer format unsupported by monpy")


def dtype_result_for_unary(dtype_code: Int) -> Int:
    if dtype_code == DTYPE_FLOAT32:
        return DTYPE_FLOAT32
    return DTYPE_FLOAT64


def dtype_result_for_unary_preserve(dtype_code: Int) -> Int:
    """Preserve-dtype unary ops (negate/abs/square/positive/floor/ceil/
    trunc/rint). bool inputs get promoted to int64 because numpy treats
    `~bool_arr` style ops as integer transforms; same pattern for negate.
    """
    if dtype_code == DTYPE_BOOL:
        return DTYPE_INT64
    return dtype_code


def _is_int_code(code: Int) -> Bool:
    return (
        code == DTYPE_INT64
        or code == DTYPE_INT32
        or code == DTYPE_INT16
        or code == DTYPE_INT8
    )


def _wider_int(a: Int, b: Int) raises -> Int:
    """Return the wider of two signed-integer dtype codes."""
    var sa = dtype_item_size(a)
    var sb = dtype_item_size(b)
    if sa >= sb:
        return a
    return b


def dtype_result_for_binary(lhs_dtype: Int, rhs_dtype: Int, op: Int) -> Int:
    """Numpy 2.x binary promotion. The original 4-dtype if-chain is
    extended to cover int8/16/32 by reducing through size-aware cases."""
    # Always-float binary transcendentals.
    if op == OP_ARCTAN2 or op == OP_HYPOT or op == OP_COPYSIGN:
        if lhs_dtype == DTYPE_FLOAT32 and rhs_dtype == DTYPE_FLOAT32:
            return DTYPE_FLOAT32
        return DTYPE_FLOAT64
    # Division always promotes to float; small-int + small-float can
    # stay in float32, but large-int (int32+) needs float64 for safety.
    if op == OP_DIV:
        # If either side is float64 → float64.
        if lhs_dtype == DTYPE_FLOAT64 or rhs_dtype == DTYPE_FLOAT64:
            return DTYPE_FLOAT64
        # Both float32 (or float32 + bool/small-int): float32.
        if lhs_dtype == DTYPE_FLOAT32 or rhs_dtype == DTYPE_FLOAT32:
            var other = rhs_dtype if lhs_dtype == DTYPE_FLOAT32 else lhs_dtype
            if other == DTYPE_FLOAT32 or other == DTYPE_BOOL:
                return DTYPE_FLOAT32
            if other == DTYPE_INT8 or other == DTYPE_INT16:
                return DTYPE_FLOAT32
        return DTYPE_FLOAT64
    # Non-division arithmetic.
    if lhs_dtype == DTYPE_FLOAT64 or rhs_dtype == DTYPE_FLOAT64:
        return DTYPE_FLOAT64
    if lhs_dtype == DTYPE_FLOAT32 or rhs_dtype == DTYPE_FLOAT32:
        # int32+/int64 + float32 → float64 (precision loss otherwise).
        if (lhs_dtype == DTYPE_INT64 or lhs_dtype == DTYPE_INT32) or (
            rhs_dtype == DTYPE_INT64 or rhs_dtype == DTYPE_INT32
        ):
            return DTYPE_FLOAT64
        return DTYPE_FLOAT32
    if _is_int_code(lhs_dtype) or _is_int_code(rhs_dtype):
        # both integer (or one bool one int): widest signed-integer wins.
        if not _is_int_code(lhs_dtype):
            return rhs_dtype
        if not _is_int_code(rhs_dtype):
            return lhs_dtype
        try:
            return _wider_int(lhs_dtype, rhs_dtype)
        except:
            return DTYPE_INT64
    return DTYPE_BOOL


def dtype_result_for_reduction(dtype_code: Int, op: Int) -> Int:
    if op == REDUCE_MEAN:
        if dtype_code == DTYPE_FLOAT32:
            return DTYPE_FLOAT32
        return DTYPE_FLOAT64
    if op == REDUCE_SUM and dtype_code == DTYPE_BOOL:
        return DTYPE_INT64
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
    if casting == CASTING_SAFE:
        # Bool fits in any numeric type.
        if from_kind == DTYPE_KIND_BOOL:
            return True
        # Signed-integer widening: smaller fits in larger.
        if from_kind == DTYPE_KIND_SIGNED_INT:
            if to_kind == DTYPE_KIND_SIGNED_INT:
                var fs = dtype_item_size(from_dtype)
                var ts = dtype_item_size(to_dtype)
                return ts >= fs
            if to_kind == DTYPE_KIND_REAL_FLOAT:
                # int8/int16 fit in float32 mantissa; int32 needs float64;
                # int64 is technically lossy in f64 but numpy says safe.
                var fs = dtype_item_size(from_dtype)
                if to_dtype == DTYPE_FLOAT64:
                    return True
                if to_dtype == DTYPE_FLOAT32:
                    return fs <= 2
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
            return to_kind == DTYPE_KIND_SIGNED_INT or to_kind == DTYPE_KIND_REAL_FLOAT
        if from_kind == DTYPE_KIND_REAL_FLOAT:
            return to_kind == DTYPE_KIND_REAL_FLOAT
        return False
    raise Error("unknown casting policy")
