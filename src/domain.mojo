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
