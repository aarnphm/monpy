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

comptime UNARY_SIN = 0
comptime UNARY_COS = 1
comptime UNARY_EXP = 2
comptime UNARY_LOG = 3

comptime REDUCE_SUM = 0
comptime REDUCE_MEAN = 1
comptime REDUCE_MIN = 2
comptime REDUCE_MAX = 3
comptime REDUCE_ARGMAX = 4

comptime BACKEND_GENERIC = 0
comptime BACKEND_ACCELERATE = 1
comptime BACKEND_FUSED = 2


def dtype_item_size(dtype_code: Int) raises -> Int:
    if dtype_code == DTYPE_BOOL:
        return 1
    if dtype_code == DTYPE_INT64:
        return 8
    if dtype_code == DTYPE_FLOAT32:
        return 4
    if dtype_code == DTYPE_FLOAT64:
        return 8
    raise Error("unsupported dtype code")


def dtype_alignment(dtype_code: Int) raises -> Int:
    return dtype_item_size(dtype_code)


def dtype_kind_code(dtype_code: Int) raises -> Int:
    if dtype_code == DTYPE_BOOL:
        return DTYPE_KIND_BOOL
    if dtype_code == DTYPE_INT64:
        return DTYPE_KIND_SIGNED_INT
    if dtype_code == DTYPE_FLOAT32 or dtype_code == DTYPE_FLOAT64:
        return DTYPE_KIND_REAL_FLOAT
    raise Error("unsupported dtype code")


def dtype_code_from_format_char(c: Int, itemsize: Int) raises -> Int:
    # Python buffers expose PEP-3118 format chars. Keep this compact decode
    # next to dtype metadata so every import bridge resolves the same dtype ids.
    if c == 0x3F and itemsize == 1:  # '?'
        return DTYPE_BOOL
    if (c == 0x6C or c == 0x71) and itemsize == 8:  # 'l' or 'q'
        return DTYPE_INT64
    if c == 0x66 and itemsize == 4:  # 'f'
        return DTYPE_FLOAT32
    if c == 0x64 and itemsize == 8:  # 'd'
        return DTYPE_FLOAT64
    raise Error("buffer format unsupported by monpy")


def dtype_result_for_unary(dtype_code: Int) -> Int:
    if dtype_code == DTYPE_FLOAT32:
        return DTYPE_FLOAT32
    return DTYPE_FLOAT64


def dtype_result_for_binary(lhs_dtype: Int, rhs_dtype: Int, op: Int) -> Int:
    if op == OP_DIV:
        if lhs_dtype == DTYPE_FLOAT32 and (
            rhs_dtype == DTYPE_FLOAT32 or rhs_dtype == DTYPE_BOOL
        ):
            return DTYPE_FLOAT32
        if rhs_dtype == DTYPE_FLOAT32 and (
            lhs_dtype == DTYPE_FLOAT32 or lhs_dtype == DTYPE_BOOL
        ):
            return DTYPE_FLOAT32
        return DTYPE_FLOAT64
    if lhs_dtype == DTYPE_FLOAT64 or rhs_dtype == DTYPE_FLOAT64:
        return DTYPE_FLOAT64
    if (lhs_dtype == DTYPE_INT64 and rhs_dtype == DTYPE_FLOAT32) or (
        lhs_dtype == DTYPE_FLOAT32 and rhs_dtype == DTYPE_INT64
    ):
        return DTYPE_FLOAT64
    if lhs_dtype == DTYPE_FLOAT32 or rhs_dtype == DTYPE_FLOAT32:
        return DTYPE_FLOAT32
    if lhs_dtype == DTYPE_INT64 or rhs_dtype == DTYPE_INT64:
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
        if from_kind == DTYPE_KIND_BOOL:
            return True
        if from_dtype == DTYPE_INT64:
            return to_dtype == DTYPE_FLOAT64
        if from_dtype == DTYPE_FLOAT32:
            return to_dtype == DTYPE_FLOAT64
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
