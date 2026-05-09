"""Pure-Mojo kernel sweep: monpy kernels against stdlib primitives.

This bench intentionally stays below the Python facade. It answers a narrower
question than `monpy-bench`: when the data is already in native Mojo memory,
are monpy's hand-rolled kernels better than the equivalent stdlib building
blocks we could bind directly?

Run directly:

    mojo run -I src benches/bench_mojo_sweep.mojo

The output is TSV so `python -m monpy._bench.mojo_sweep` can parse it and save
the same kind of durable benchmark artifact as the Python sweep.
"""

from std.algorithm import (
    max as std_max,
    mean as std_mean,
    min as std_min,
    product as std_product,
    sum as std_sum,
)
from std.benchmark import keep, run
from std.math import sin as std_sin
from std.memory import Span
from std.memory.unsafe_pointer import alloc
from std.sys import simd_width_of, size_of

from domain import BinaryOp, UnaryOp
from elementwise.kernels.matmul import matmul_small_typed
from elementwise.kernels.reduce import (
    reduce_max_typed,
    reduce_min_typed,
    reduce_prod_typed,
    reduce_sum_typed,
)
from elementwise.kernels.typed import (
    binary_same_shape_contig_typed,
    binary_same_shape_contig_typed_static,
    binary_scalar_contig_typed_static,
    unary_contig_typed,
    unary_contig_typed_static,
)


def fill_buffer[dtype: DType](ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], n: Int):
    for i in range(n):
        ptr[i] = Scalar[dtype](Float64(i % 257) * 0.00390625 + 0.25)


def fill_matrix[dtype: DType](ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], n: Int):
    for i in range(n * n):
        ptr[i] = Scalar[dtype](Float64(i % 31) * 0.03125 + 0.5)


def emit_header():
    print("group\tname\tcandidate\tbaseline\tcandidate_ns\tbaseline_ns\tratio\tbytes\tflops")


def emit_result(
    group: String,
    name: String,
    candidate: String,
    baseline: String,
    candidate_ns: Float64,
    baseline_ns: Float64,
    bytes: Int,
    flops: Int,
) raises:
    print(
        String("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}").format(
            group,
            name,
            candidate,
            baseline,
            candidate_ns,
            baseline_ns,
            candidate_ns / baseline_ns,
            bytes,
            flops,
        )
    )


def bench_configured[func: def() raises capturing[_] -> None]() raises -> Float64:
    var report = run[func3=func](
        num_warmup_iters=1,
        max_iters=100_000,
        min_runtime_secs=0.005,
        max_runtime_secs=0.05,
        max_batch_size=0,
    )
    return report.mean("ns")


def std_binary_same_shape[
    dtype: DType
](
    lhs: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    dst: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
):
    comptime width = simd_width_of[dtype]()
    var i = 0
    while i + width <= size:
        dst.store(i, lhs.load[width=width](i) + rhs.load[width=width](i))
        i += width
    while i < size:
        dst[i] = lhs[i] + rhs[i]
        i += 1


def std_binary_scalar_mul[
    dtype: DType
](
    src: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    scalar: Scalar[dtype],
    dst: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
):
    comptime width = simd_width_of[dtype]()
    var scalar_vec = SIMD[dtype, width](scalar)
    var i = 0
    while i + width <= size:
        dst.store(i, src.load[width=width](i) * scalar_vec)
        i += width
    while i < size:
        dst[i] = src[i] * scalar
        i += 1


def std_unary_sin[
    dtype: DType
](
    src: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    dst: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
) where dtype.is_floating_point():
    comptime width = simd_width_of[dtype]()
    var i = 0
    while i + width <= size:
        dst.store(i, std_sin(src.load[width=width](i)))
        i += width
    while i < size:
        dst[i] = std_sin(src[i])
        i += 1


def std_matmul_small[
    dtype: DType
](
    lhs: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    dst: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    m: Int,
    n: Int,
    k_lhs: Int,
) where dtype.is_floating_point():
    comptime width = simd_width_of[dtype]()
    for row in range(m):
        var col = 0
        while col + width <= n:
            var acc = SIMD[dtype, width](0)
            for k in range(k_lhs):
                acc += SIMD[dtype, width](lhs[row * k_lhs + k]) * rhs.load[width=width](k * n + col)
            dst.store(row * n + col, acc)
            col += width
        while col < n:
            var total = Scalar[dtype](0)
            for k in range(k_lhs):
                total += lhs[row * k_lhs + k] * rhs[k * n + col]
            dst[row * n + col] = total
            col += 1


def dtype_name[dtype: DType]() -> String:
    comptime if dtype == DType.float32:
        return "f32"
    else:
        return "f64"


def size_name[n: Int]() -> String:
    comptime if n == 1024:
        return "1k"
    else:
        comptime if n == 65536:
            return "64k"
        else:
            comptime if n == 1048576:
                return "1m"
            else:
                comptime if n == 16777216:
                    return "16m"
                else:
                    comptime if n == 134217728:
                        return "128m"
                    else:
                        return String(n)


def emit_binary_add[dtype: DType, n: Int]() raises:
    var lhs = alloc[Scalar[dtype]](n)
    var rhs = alloc[Scalar[dtype]](n)
    var out = alloc[Scalar[dtype]](n)
    fill_buffer[dtype](lhs, n)
    fill_buffer[dtype](rhs, n)

    @parameter
    def monpy_call() raises:
        binary_same_shape_contig_typed_static[dtype, BinaryOp.ADD.value](lhs, rhs, out, n)
        keep(out[0])

    @parameter
    def std_call() raises:
        std_binary_same_shape[dtype](lhs, rhs, out, n)
        keep(out[0])

    emit_result(
        "elementwise",
        String("add_{}_{}").format(dtype_name[dtype](), size_name[n]()),
        "monpy.binary_same_shape_contig_typed_static",
        "stdlib.SIMD_loop",
        bench_configured[monpy_call](),
        bench_configured[std_call](),
        n * size_of[Scalar[dtype]]() * 3,
        n,
    )
    lhs.free()
    rhs.free()
    out.free()


def emit_scalar_mul[dtype: DType, n: Int]() raises:
    var src = alloc[Scalar[dtype]](n)
    var out = alloc[Scalar[dtype]](n)
    fill_buffer[dtype](src, n)
    var scalar = Scalar[dtype](3.0)

    @parameter
    def monpy_call() raises:
        binary_scalar_contig_typed_static[dtype, BinaryOp.MUL.value](src, scalar, out, n, False)
        keep(out[0])

    @parameter
    def std_call() raises:
        std_binary_scalar_mul[dtype](src, scalar, out, n)
        keep(out[0])

    emit_result(
        "elementwise",
        String("scalar_mul_{}_{}").format(dtype_name[dtype](), size_name[n]()),
        "monpy.binary_scalar_contig_typed_static",
        "stdlib.SIMD_loop",
        bench_configured[monpy_call](),
        bench_configured[std_call](),
        n * size_of[Scalar[dtype]]() * 2,
        n,
    )
    src.free()
    out.free()


def emit_unary_sin[dtype: DType, n: Int]() raises where dtype.is_floating_point():
    var src = alloc[Scalar[dtype]](n)
    var out = alloc[Scalar[dtype]](n)
    fill_buffer[dtype](src, n)

    @parameter
    def monpy_call() raises:
        unary_contig_typed[dtype](src, out, n, UnaryOp.SIN.value)
        keep(out[0])

    @parameter
    def std_call() raises:
        std_unary_sin[dtype](src, out, n)
        keep(out[0])

    emit_result(
        "elementwise",
        String("sin_{}_{}").format(dtype_name[dtype](), size_name[n]()),
        "monpy.unary_contig_typed",
        "stdlib.SIMD_loop",
        bench_configured[monpy_call](),
        bench_configured[std_call](),
        n * size_of[Scalar[dtype]]() * 2,
        n,
    )
    src.free()
    out.free()


def emit_unary_exp_par[dtype: DType, n: Int]() raises where dtype.is_floating_point():
    # Production `EXP` through the public unary entry point. Baseline keeps the
    # serial static kernel fixed so this row isolates the static parallel gate.
    var src = alloc[Scalar[dtype]](n)
    var out = alloc[Scalar[dtype]](n)
    fill_buffer[dtype](src, n)

    @parameter
    def monpy_call() raises:
        unary_contig_typed[dtype](src, out, n, UnaryOp.EXP.value)
        keep(out[0])

    @parameter
    def baseline_call() raises:
        unary_contig_typed_static[dtype, UnaryOp.EXP.value](src, out, n)
        keep(out[0])

    emit_result(
        "elementwise",
        String("exp_par_{}_{}").format(dtype_name[dtype](), size_name[n]()),
        "monpy.unary_contig_typed",
        "monpy.unary_contig_typed_static",
        bench_configured[monpy_call](),
        bench_configured[baseline_call](),
        n * size_of[Scalar[dtype]]() * 2,
        n,
    )
    src.free()
    out.free()


def emit_sum[dtype: DType, n: Int]() raises where dtype.is_floating_point():
    var src = alloc[Scalar[dtype]](n)
    fill_buffer[dtype](src, n)

    @parameter
    def monpy_call() raises:
        var result = reduce_sum_typed[dtype](src, n)
        keep(result)

    @parameter
    def std_call() raises:
        var span = Span[Scalar[dtype], MutExternalOrigin](ptr=src, length=n)
        var result = std_sum(span)
        keep(result)

    emit_result(
        "reductions",
        String("sum_{}_{}").format(dtype_name[dtype](), size_name[n]()),
        "monpy.reduce_sum_typed",
        "std.algorithm.sum",
        bench_configured[monpy_call](),
        bench_configured[std_call](),
        n * size_of[Scalar[dtype]](),
        n,
    )
    src.free()


def emit_mean[dtype: DType, n: Int]() raises where dtype.is_floating_point():
    var src = alloc[Scalar[dtype]](n)
    fill_buffer[dtype](src, n)

    @parameter
    def monpy_call() raises:
        var result = reduce_sum_typed[dtype](src, n) / Float64(n)
        keep(result)

    @parameter
    def std_call() raises:
        var span = Span[Scalar[dtype], MutExternalOrigin](ptr=src, length=n)
        var result = std_mean(span)
        keep(result)

    emit_result(
        "reductions",
        String("mean_{}_{}").format(dtype_name[dtype](), size_name[n]()),
        "monpy.reduce_sum_typed/n",
        "std.algorithm.mean",
        bench_configured[monpy_call](),
        bench_configured[std_call](),
        n * size_of[Scalar[dtype]](),
        n,
    )
    src.free()


def emit_min[dtype: DType, n: Int]() raises where dtype.is_floating_point():
    var src = alloc[Scalar[dtype]](n)
    fill_buffer[dtype](src, n)

    @parameter
    def monpy_call() raises:
        var result = reduce_min_typed[dtype](src, n)
        keep(result)

    @parameter
    def std_call() raises:
        var span = Span[Scalar[dtype], MutExternalOrigin](ptr=src, length=n)
        var result = std_min(span)
        keep(result)

    emit_result(
        "reductions",
        String("min_{}_{}").format(dtype_name[dtype](), size_name[n]()),
        "monpy.reduce_min_typed",
        "std.algorithm.min",
        bench_configured[monpy_call](),
        bench_configured[std_call](),
        n * size_of[Scalar[dtype]](),
        n,
    )
    src.free()


def emit_max[dtype: DType, n: Int]() raises where dtype.is_floating_point():
    var src = alloc[Scalar[dtype]](n)
    fill_buffer[dtype](src, n)

    @parameter
    def monpy_call() raises:
        var result = reduce_max_typed[dtype](src, n)
        keep(result)

    @parameter
    def std_call() raises:
        var span = Span[Scalar[dtype], MutExternalOrigin](ptr=src, length=n)
        var result = std_max(span)
        keep(result)

    emit_result(
        "reductions",
        String("max_{}_{}").format(dtype_name[dtype](), size_name[n]()),
        "monpy.reduce_max_typed",
        "std.algorithm.max",
        bench_configured[monpy_call](),
        bench_configured[std_call](),
        n * size_of[Scalar[dtype]](),
        n,
    )
    src.free()


def emit_prod[dtype: DType, n: Int]() raises where dtype.is_floating_point():
    var src = alloc[Scalar[dtype]](n)
    fill_buffer[dtype](src, n)

    @parameter
    def monpy_call() raises:
        var result = reduce_prod_typed[dtype](src, n)
        keep(result)

    @parameter
    def std_call() raises:
        var span = Span[Scalar[dtype], MutExternalOrigin](ptr=src, length=n)
        var result = std_product(span)
        keep(result)

    emit_result(
        "reductions",
        String("prod_{}_{}").format(dtype_name[dtype](), size_name[n]()),
        "monpy.reduce_prod_typed",
        "std.algorithm.product",
        bench_configured[monpy_call](),
        bench_configured[std_call](),
        n * size_of[Scalar[dtype]](),
        n,
    )
    src.free()


def emit_binary_add_par[dtype: DType, n: Int]() raises:
    # Same-shape binary `add` through the parallel-aware entry point.
    # At n*size_of[dtype]() >= ELEMENTWISE_LIGHT_GRAIN (2MB) the kernel
    # fans out via sync_parallelize; below it stays serial. Baseline is
    # the static-op kernel (no parallel gate) so we can read the spawn cost.
    var lhs = alloc[Scalar[dtype]](n)
    var rhs = alloc[Scalar[dtype]](n)
    var out = alloc[Scalar[dtype]](n)
    fill_buffer[dtype](lhs, n)
    fill_buffer[dtype](rhs, n)

    @parameter
    def monpy_call() raises:
        binary_same_shape_contig_typed[dtype](lhs, rhs, out, n, BinaryOp.ADD.value)
        keep(out[0])

    @parameter
    def baseline_call() raises:
        binary_same_shape_contig_typed_static[dtype, BinaryOp.ADD.value](lhs, rhs, out, n)
        keep(out[0])

    emit_result(
        "elementwise",
        String("add_par_{}_{}").format(dtype_name[dtype](), size_name[n]()),
        "monpy.binary_same_shape_contig_typed",
        "monpy.binary_same_shape_contig_typed_static",
        bench_configured[monpy_call](),
        bench_configured[baseline_call](),
        n * size_of[Scalar[dtype]]() * 3,
        n,
    )
    lhs.free()
    rhs.free()
    out.free()


def emit_matmul_small[dtype: DType, n: Int]() raises where dtype.is_floating_point():
    var lhs = alloc[Scalar[dtype]](n * n)
    var rhs = alloc[Scalar[dtype]](n * n)
    var out = alloc[Scalar[dtype]](n * n)
    fill_matrix[dtype](lhs, n)
    fill_matrix[dtype](rhs, n)

    @parameter
    def monpy_call() raises:
        matmul_small_typed[dtype](lhs, rhs, out, n, n, n)
        keep(out[0])

    @parameter
    def std_call() raises:
        std_matmul_small[dtype](lhs, rhs, out, n, n, n)
        keep(out[0])

    emit_result(
        "matmul",
        String("small_matmul_{}_{}").format(dtype_name[dtype](), n),
        "monpy.matmul_small_typed",
        "std.pointer_loop",
        bench_configured[monpy_call](),
        bench_configured[std_call](),
        n * n * size_of[Scalar[dtype]]() * 3,
        2 * n * n * n,
    )
    lhs.free()
    rhs.free()
    out.free()


def emit_family[dtype: DType]() raises where dtype.is_floating_point():
    emit_binary_add[dtype, 1024]()
    emit_binary_add[dtype, 65536]()
    emit_binary_add[dtype, 1048576]()
    emit_scalar_mul[dtype, 1024]()
    emit_scalar_mul[dtype, 65536]()
    emit_unary_sin[dtype, 1024]()
    emit_unary_sin[dtype, 65536]()
    emit_unary_sin[dtype, 1048576]()
    emit_sum[dtype, 1024]()
    emit_sum[dtype, 65536]()
    emit_sum[dtype, 1048576]()
    emit_mean[dtype, 65536]()
    emit_min[dtype, 1024]()
    emit_min[dtype, 65536]()
    emit_max[dtype, 1024]()
    emit_max[dtype, 65536]()
    emit_prod[dtype, 1024]()
    emit_prod[dtype, 65536]()
    # Parallel-vs-serial head-to-head at sizes that exceed the
    # ELEMENTWISE_LIGHT_GRAIN (2MB) gates. Parallel-reduction calibration
    # lives in bench_parallel.mojo / bench_reduce.mojo so stdlib deficit
    # ranking is not dominated by non-production calibration rows.
    # These are the rows that surface multi-thread regressions.
    emit_unary_exp_par[dtype, 262144]()
    emit_unary_exp_par[dtype, 1048576]()
    emit_binary_add_par[dtype, 16777216]()
    emit_matmul_small[dtype, 8]()
    emit_matmul_small[dtype, 16]()


def main() raises:
    emit_header()
    emit_family[DType.float32]()
    emit_family[DType.float64]()
