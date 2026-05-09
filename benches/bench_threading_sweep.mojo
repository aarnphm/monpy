"""Threading sweep: internal threaded prototypes against monpy serial kernels.

This bench is TSV-shaped so `monpy-bench-mojo --include-threading` can save it
beside the regular Mojo and NuMojo rows.

Rows compare a proposed internal threaded static-op implementation against the
serial monpy static kernel it would replace. Run this file in separate
processes with different `MONPY_THREADS` values; `elementwise.kernels.parallel`
reads that env knob once per process.

    MONPY_THREADS=1 mojo run -I src benches/bench_threading_sweep.mojo
    MONPY_THREADS=4 mojo run -I src benches/bench_threading_sweep.mojo
    mojo run -I src benches/bench_threading_sweep.mojo  # auto
"""

from std.algorithm import sync_parallelize
from std.benchmark import keep, run
from std.math import ceildiv
from std.memory.unsafe_pointer import alloc
from std.sys import size_of

from domain import BinaryOp, UnaryOp
from elementwise.kernels.parallel import (
    ELEMENTWISE_HEAVY_GRAIN,
    ELEMENTWISE_LIGHT_GRAIN,
    thread_limit,
    worker_count_for_bytes,
)
from elementwise.kernels.typed import (
    binary_same_shape_contig_typed_static,
    unary_contig_typed_static,
)


def fill_buffer[dtype: DType](ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin], n: Int):
    for i in range(n):
        ptr[i] = Scalar[dtype](Float64(i % 251) * 0.005 + 0.25)


def emit_header():
    print("group\tname\tcandidate\tbaseline\tcandidate_ns\tbaseline_ns\tratio\tbytes\tflops")


def thread_group() raises -> String:
    var limit = thread_limit()
    if limit <= 0:
        return "threading.auto"
    return String("threading.threads{}").format(limit)


def emit_result(
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
            thread_group(),
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


def size_name[n: Int]() -> String:
    comptime if n == 65536:
        return "64k"
    else:
        comptime if n == 262144:
            return "256k"
        else:
            comptime if n == 1048576:
                return "1m"
            else:
                comptime if n == 4194304:
                    return "4m"
                else:
                    comptime if n == 16777216:
                        return "16m"
                    else:
                        return String(n)


def dtype_name[dtype: DType]() -> String:
    comptime if dtype == DType.float32:
        return "f32"
    else:
        return "f64"


def parallel_static_add[
    dtype: DType
](
    lhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
) raises:
    var byte_count = size * size_of[Scalar[dtype]]()
    var nworkers = worker_count_for_bytes(size, byte_count, ELEMENTWISE_LIGHT_GRAIN)
    if nworkers <= 1 or size < nworkers:
        binary_same_shape_contig_typed_static[dtype, BinaryOp.ADD.value](lhs_ptr, rhs_ptr, out_ptr, size)
        return

    var chunk = ceildiv(size, nworkers)

    @parameter
    def chunk_worker(i: Int) raises:
        var start = i * chunk
        var end = start + chunk
        if end > size:
            end = size
        if start >= size:
            return
        binary_same_shape_contig_typed_static[dtype, BinaryOp.ADD.value](
            lhs_ptr + start, rhs_ptr + start, out_ptr + start, end - start
        )

    sync_parallelize[chunk_worker](nworkers)


def parallel_static_unary[
    dtype: DType, op: Int, grain: Int
](
    src_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    out_ptr: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    size: Int,
) raises where dtype.is_floating_point():
    var byte_count = size * size_of[Scalar[dtype]]()
    var nworkers = worker_count_for_bytes(size, byte_count, grain)
    if nworkers <= 1 or size < nworkers:
        unary_contig_typed_static[dtype, op](src_ptr, out_ptr, size)
        return

    var chunk = ceildiv(size, nworkers)

    @parameter
    def chunk_worker(i: Int) raises:
        var start = i * chunk
        var end = start + chunk
        if end > size:
            end = size
        if start >= size:
            return
        unary_contig_typed_static[dtype, op](src_ptr + start, out_ptr + start, end - start)

    sync_parallelize[chunk_worker](nworkers)


def emit_add[dtype: DType, n: Int]() raises:
    var lhs = alloc[Scalar[dtype]](n)
    var rhs = alloc[Scalar[dtype]](n)
    var out = alloc[Scalar[dtype]](n)
    fill_buffer[dtype](lhs, n)
    fill_buffer[dtype](rhs, n)

    @parameter
    def internal_call() raises:
        parallel_static_add[dtype](lhs, rhs, out, n)
        keep(out[0])

    @parameter
    def monpy_call() raises:
        binary_same_shape_contig_typed_static[dtype, BinaryOp.ADD.value](lhs, rhs, out, n)
        keep(out[0])

    emit_result(
        String("add_{}_{}").format(dtype_name[dtype](), size_name[n]()),
        "internal.threaded_static_add",
        "monpy.binary_same_shape_contig_typed_static",
        bench_configured[internal_call](),
        bench_configured[monpy_call](),
        3 * n * size_of[Scalar[dtype]](),
        n,
    )
    lhs.free()
    rhs.free()
    out.free()


def emit_neg[dtype: DType, n: Int]() raises where dtype.is_floating_point():
    var src = alloc[Scalar[dtype]](n)
    var out = alloc[Scalar[dtype]](n)
    fill_buffer[dtype](src, n)

    @parameter
    def internal_call() raises:
        parallel_static_unary[dtype, UnaryOp.NEGATE.value, ELEMENTWISE_LIGHT_GRAIN](src, out, n)
        keep(out[0])

    @parameter
    def monpy_call() raises:
        unary_contig_typed_static[dtype, UnaryOp.NEGATE.value](src, out, n)
        keep(out[0])

    emit_result(
        String("neg_{}_{}").format(dtype_name[dtype](), size_name[n]()),
        "internal.threaded_static_negate",
        "monpy.unary_contig_typed_static",
        bench_configured[internal_call](),
        bench_configured[monpy_call](),
        2 * n * size_of[Scalar[dtype]](),
        n,
    )
    src.free()
    out.free()


def emit_exp[dtype: DType, n: Int]() raises where dtype.is_floating_point():
    var src = alloc[Scalar[dtype]](n)
    var out = alloc[Scalar[dtype]](n)
    fill_buffer[dtype](src, n)

    @parameter
    def internal_call() raises:
        parallel_static_unary[dtype, UnaryOp.EXP.value, ELEMENTWISE_HEAVY_GRAIN](src, out, n)
        keep(out[0])

    @parameter
    def monpy_call() raises:
        unary_contig_typed_static[dtype, UnaryOp.EXP.value](src, out, n)
        keep(out[0])

    emit_result(
        String("exp_{}_{}").format(dtype_name[dtype](), size_name[n]()),
        "internal.threaded_static_exp",
        "monpy.unary_contig_typed_static",
        bench_configured[internal_call](),
        bench_configured[monpy_call](),
        2 * n * size_of[Scalar[dtype]](),
        n,
    )
    src.free()
    out.free()


def main() raises:
    emit_header()
    emit_add[DType.float32, 65536]()
    emit_add[DType.float32, 1048576]()
    emit_add[DType.float32, 16777216]()
    emit_neg[DType.float32, 65536]()
    emit_neg[DType.float32, 1048576]()
    emit_neg[DType.float32, 16777216]()
    emit_exp[DType.float32, 65536]()
    emit_exp[DType.float32, 262144]()
    emit_exp[DType.float32, 1048576]()
    emit_add[DType.float64, 1048576]()
    emit_neg[DType.float64, 1048576]()
    emit_exp[DType.float64, 262144]()
