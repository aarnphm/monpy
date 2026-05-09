"""Native random helpers for monpy.random.
"""

from std.collections import List
from std.math import cos, log, sqrt
from std.python import PythonObject

from array import Array, contiguous_ptr, int_list_from_py, make_empty_array
from domain import ArrayDType


comptime GOLDEN64 = UInt64(11400714819323198485)
comptime MIX64_A = UInt64(13787848793156543929)
comptime MIX64_B = UInt64(10723151780598845931)
comptime MASK32 = UInt64(4294967295)
comptime THREEFRY_C240 = UInt32(466688986)
comptime TWO_NEG_53 = 1.1102230246251565e-16
comptime TWO_PI = 6.2831853071795864769


def _positive_seed_bits(seed: Int) -> UInt64:
    if seed >= 0:
        return UInt64(seed)
    return UInt64(-seed) ^ UInt64(0xD1B54A32D192ED03)


def _mix64(value: UInt64) -> UInt64:
    var z = value + GOLDEN64
    z = (z ^ (z >> 30)) * MIX64_A
    z = (z ^ (z >> 27)) * MIX64_B
    return z ^ (z >> 31)


def _u32_high(value: UInt64) -> UInt32:
    return UInt32((value >> 32) & MASK32)


def _u32_low(value: UInt64) -> UInt32:
    return UInt32(value & MASK32)


def _key_base(key0: UInt32, key1: UInt32) -> UInt64:
    return (UInt64(key0) << 32) | UInt64(key1)


def _rotl32(value: UInt32, amount: Int) -> UInt32:
    return (value << UInt32(amount)) | (value >> UInt32(32 - amount))


def _threefry_rot(round_index: Int) -> Int:
    var index = round_index % 8
    if index == 0:
        return 13
    if index == 1:
        return 15
    if index == 2:
        return 26
    if index == 3:
        return 6
    if index == 4:
        return 17
    if index == 5:
        return 29
    if index == 6:
        return 16
    return 24


def _threefry_key_word(index: Int, key0: UInt32, key1: UInt32, key2: UInt32) -> UInt32:
    var wrapped = index % 3
    if wrapped == 0:
        return key0
    if wrapped == 1:
        return key1
    return key2


def _threefry2x32_u64(key0: UInt32, key1: UInt32, counter0: UInt32, counter1: UInt32) -> UInt64:
    var key2 = THREEFRY_C240 ^ key0 ^ key1
    var x0 = counter0 + key0
    var x1 = counter1 + key1
    for round_index in range(20):
        x0 = x0 + x1
        x1 = _rotl32(x1, _threefry_rot(round_index)) ^ x0
        if round_index % 4 == 3:
            var schedule = round_index // 4 + 1
            x0 = x0 + _threefry_key_word(schedule, key0, key1, key2)
            x1 = x1 + _threefry_key_word(schedule + 1, key0, key1, key2) + UInt32(schedule)
    return (UInt64(x0) << 32) | UInt64(x1)


def _random_u64(key0: UInt32, key1: UInt32, counter: Int) -> UInt64:
    var ctr = UInt64(counter)
    return _threefry2x32_u64(key0, key1, _u32_low(ctr), _u32_high(ctr))


def _unit_f64(bits: UInt64) -> Float64:
    var mantissa = (bits >> 11) & UInt64(9007199254740991)
    return Float64(Int(mantissa)) * TWO_NEG_53


def _unit_f64_open(bits: UInt64) -> Float64:
    var mantissa = (bits >> 11) & UInt64(9007199254740991)
    return (Float64(Int(mantissa)) + 1.0) / 9007199254740993.0


def _shape_from_size(size_obj: PythonObject) raises -> List[Int]:
    return int_list_from_py(size_obj)


def _write_key_words(mut result: Array, word0: UInt32, word1: UInt32):
    var out = contiguous_ptr[DType.uint32](result)
    out[0] = word0
    out[1] = word1


def random_key_ops(seed_obj: PythonObject) raises -> PythonObject:
    var shape = List[Int]()
    shape.append(2)
    var result = make_empty_array(ArrayDType.UINT32.value, shape^)
    var seed = _positive_seed_bits(Int(py=seed_obj))
    var first = _mix64(seed)
    var second = _mix64(first)
    _write_key_words(result, _u32_high(first), _u32_low(second))
    return PythonObject(alloc=result^)


def random_key_data_ops(key0_obj: PythonObject, key1_obj: PythonObject) raises -> PythonObject:
    var shape = List[Int]()
    shape.append(2)
    var result = make_empty_array(ArrayDType.UINT32.value, shape^)
    _write_key_words(result, UInt32(Int(py=key0_obj)), UInt32(Int(py=key1_obj)))
    return PythonObject(alloc=result^)


def random_split_ops(key0_obj: PythonObject, key1_obj: PythonObject, num_obj: PythonObject) raises -> PythonObject:
    var num = Int(py=num_obj)
    if num < 0:
        raise Error("split num must be non-negative")
    var shape = List[Int]()
    shape.append(num)
    shape.append(2)
    var result = make_empty_array(ArrayDType.UINT32.value, shape^)
    var out = contiguous_ptr[DType.uint32](result)
    var key0 = UInt32(Int(py=key0_obj))
    var key1 = UInt32(Int(py=key1_obj))
    for i in range(num):
        var first = _random_u64(key0, key1, i * 2)
        var second = _random_u64(key0, key1, i * 2 + 1)
        out[i * 2] = _u32_high(first)
        out[i * 2 + 1] = _u32_low(second)
    return PythonObject(alloc=result^)


def random_fold_in_ops(key0_obj: PythonObject, key1_obj: PythonObject, data_obj: PythonObject) raises -> PythonObject:
    var shape = List[Int]()
    shape.append(2)
    var result = make_empty_array(ArrayDType.UINT32.value, shape^)
    var key0 = UInt32(Int(py=key0_obj))
    var key1 = UInt32(Int(py=key1_obj))
    var data_bits = _positive_seed_bits(Int(py=data_obj))
    var first = _mix64(_key_base(key0, key1) ^ _mix64(data_bits))
    var second = _mix64(first)
    _write_key_words(result, _u32_high(first), _u32_low(second))
    return PythonObject(alloc=result^)


def random_bits_ops(
    key0_obj: PythonObject,
    key1_obj: PythonObject,
    shape_obj: PythonObject,
    dtype_obj: PythonObject,
) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var shape = _shape_from_size(shape_obj)
    var result = make_empty_array(dtype_code, shape^)
    var key0 = UInt32(Int(py=key0_obj))
    var key1 = UInt32(Int(py=key1_obj))
    if dtype_code == ArrayDType.UINT32.value:
        var out32 = contiguous_ptr[DType.uint32](result)
        for i in range(result.size_value):
            out32[i] = _u32_low(_random_u64(key0, key1, i))
        return PythonObject(alloc=result^)
    if dtype_code == ArrayDType.UINT64.value:
        var out64 = contiguous_ptr[DType.uint64](result)
        for i in range(result.size_value):
            out64[i] = _random_u64(key0, key1, i)
        return PythonObject(alloc=result^)
    raise Error("bits dtype must be uint32 or uint64")


def random_uniform_ops(
    key0_obj: PythonObject,
    key1_obj: PythonObject,
    shape_obj: PythonObject,
    dtype_obj: PythonObject,
    low_obj: PythonObject,
    high_obj: PythonObject,
) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var low = Float64(py=low_obj)
    var high = Float64(py=high_obj)
    var span = high - low
    var shape = _shape_from_size(shape_obj)
    var result = make_empty_array(dtype_code, shape^)
    var key0 = UInt32(Int(py=key0_obj))
    var key1 = UInt32(Int(py=key1_obj))
    if dtype_code == ArrayDType.FLOAT32.value:
        var out32 = contiguous_ptr[DType.float32](result)
        for i in range(result.size_value):
            out32[i] = Float32(low + span * _unit_f64(_random_u64(key0, key1, i)))
        return PythonObject(alloc=result^)
    if dtype_code == ArrayDType.FLOAT64.value:
        var out64 = contiguous_ptr[DType.float64](result)
        for i in range(result.size_value):
            out64[i] = low + span * _unit_f64(_random_u64(key0, key1, i))
        return PythonObject(alloc=result^)
    raise Error("uniform dtype must be float32 or float64")


def random_normal_ops(
    key0_obj: PythonObject,
    key1_obj: PythonObject,
    shape_obj: PythonObject,
    dtype_obj: PythonObject,
) raises -> PythonObject:
    var dtype_code = Int(py=dtype_obj)
    var shape = _shape_from_size(shape_obj)
    var result = make_empty_array(dtype_code, shape^)
    var key0 = UInt32(Int(py=key0_obj))
    var key1 = UInt32(Int(py=key1_obj))
    if dtype_code == ArrayDType.FLOAT32.value:
        var out32 = contiguous_ptr[DType.float32](result)
        for i in range(result.size_value):
            var u1 = _unit_f64_open(_random_u64(key0, key1, i * 2))
            var u2 = _unit_f64(_random_u64(key0, key1, i * 2 + 1))
            out32[i] = Float32(sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2))
        return PythonObject(alloc=result^)
    if dtype_code == ArrayDType.FLOAT64.value:
        var out64 = contiguous_ptr[DType.float64](result)
        for i in range(result.size_value):
            var u1 = _unit_f64_open(_random_u64(key0, key1, i * 2))
            var u2 = _unit_f64(_random_u64(key0, key1, i * 2 + 1))
            out64[i] = sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2)
        return PythonObject(alloc=result^)
    raise Error("normal dtype must be float32 or float64")


def _sample_bounded(key0: UInt32, key1: UInt32, counter: Int, span: UInt64) -> UInt64:
    var threshold = (UInt64(0) - span) % span
    var sample = _random_u64(key0, key1, counter)
    var offset = 1
    while sample < threshold:
        sample = _random_u64(key0, key1, counter + offset)
        offset += 1
    return sample % span


def random_randint_ops(
    key0_obj: PythonObject,
    key1_obj: PythonObject,
    shape_obj: PythonObject,
    low_obj: PythonObject,
    high_obj: PythonObject,
    dtype_obj: PythonObject,
) raises -> PythonObject:
    var low = Int(py=low_obj)
    var high = Int(py=high_obj)
    if high <= low:
        raise Error("high must be greater than low")
    var dtype_code = Int(py=dtype_obj)
    var span = UInt64(high - low)
    var shape = _shape_from_size(shape_obj)
    var result = make_empty_array(dtype_code, shape^)
    var key0 = UInt32(Int(py=key0_obj))
    var key1 = UInt32(Int(py=key1_obj))
    if dtype_code == ArrayDType.INT64.value:
        var out_i64 = contiguous_ptr[DType.int64](result)
        for i in range(result.size_value):
            out_i64[i] = Int64(low + Int(_sample_bounded(key0, key1, i * 2, span)))
        return PythonObject(alloc=result^)
    if dtype_code == ArrayDType.INT32.value:
        var out_i32 = contiguous_ptr[DType.int32](result)
        for i in range(result.size_value):
            out_i32[i] = Int32(low + Int(_sample_bounded(key0, key1, i * 2, span)))
        return PythonObject(alloc=result^)
    if dtype_code == ArrayDType.INT16.value:
        var out_i16 = contiguous_ptr[DType.int16](result)
        for i in range(result.size_value):
            out_i16[i] = Int16(low + Int(_sample_bounded(key0, key1, i * 2, span)))
        return PythonObject(alloc=result^)
    if dtype_code == ArrayDType.INT8.value:
        var out_i8 = contiguous_ptr[DType.int8](result)
        for i in range(result.size_value):
            out_i8[i] = Int8(low + Int(_sample_bounded(key0, key1, i * 2, span)))
        return PythonObject(alloc=result^)
    if low < 0:
        raise Error("unsigned randint low must be non-negative")
    if dtype_code == ArrayDType.UINT64.value:
        var out_u64 = contiguous_ptr[DType.uint64](result)
        for i in range(result.size_value):
            out_u64[i] = UInt64(low) + _sample_bounded(key0, key1, i * 2, span)
        return PythonObject(alloc=result^)
    if dtype_code == ArrayDType.UINT32.value:
        var out_u32 = contiguous_ptr[DType.uint32](result)
        for i in range(result.size_value):
            out_u32[i] = UInt32(UInt64(low) + _sample_bounded(key0, key1, i * 2, span))
        return PythonObject(alloc=result^)
    if dtype_code == ArrayDType.UINT16.value:
        var out_u16 = contiguous_ptr[DType.uint16](result)
        for i in range(result.size_value):
            out_u16[i] = UInt16(UInt64(low) + _sample_bounded(key0, key1, i * 2, span))
        return PythonObject(alloc=result^)
    if dtype_code == ArrayDType.UINT8.value:
        var out_u8 = contiguous_ptr[DType.uint8](result)
        for i in range(result.size_value):
            out_u8[i] = UInt8(UInt64(low) + _sample_bounded(key0, key1, i * 2, span))
        return PythonObject(alloc=result^)
    raise Error("randint dtype must be an integer dtype")
