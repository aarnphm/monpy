"""Optional NuMojo sweep for the pure-Mojo benchmark layer.

This file deliberately imports NuMojo, so it requires a NuMojo checkout or
package on the Mojo include path:

    mojo run -I src -I /path/to/NuMojo benches/bench_numojo_sweep.mojo

NuMojo currently tracks the Modular 0.26.x family. If this checkout is using a
newer dev compiler, run this file with a compatible Mojo toolchain or expect
the import to fail before any benchmark row is emitted.
"""

from std.benchmark import keep, run
from std.memory.unsafe_pointer import alloc
from std.sys import size_of

from numojo.core.dtype.default_dtype import f32
from numojo.core.type_aliases import Shape
from numojo.routines.creation import arange
from numojo.routines.math.sums import sum
from numojo.routines.math.trig import sin

from domain import BinaryOp, UnaryOp
from elementwise.kernels.matmul import matmul_small_typed
from elementwise.kernels.reduce import reduce_sum_typed
from elementwise.kernels.typed import (
    binary_same_shape_contig_typed_static,
    unary_contig_typed,
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


def emit_numojo_add_f32[n: Int]() raises:
    var lhs_nm = arange[f32](0.0, Float32(n))
    var rhs_nm = arange[f32](0.0, Float32(n))
    var lhs = alloc[Scalar[DType.float32]](n)
    var rhs = alloc[Scalar[DType.float32]](n)
    var out = alloc[Scalar[DType.float32]](n)
    fill_buffer[DType.float32](lhs, n)
    fill_buffer[DType.float32](rhs, n)

    @parameter
    def numojo_call() raises:
        var result = lhs_nm + rhs_nm
        keep(result.size)

    @parameter
    def monpy_call() raises:
        binary_same_shape_contig_typed_static[DType.float32, BinaryOp.ADD.value](lhs, rhs, out, n)
        keep(out[0])

    emit_result(
        "numojo.elementwise",
        String("add_f32_{}").format(n),
        "numojo.NDArray.__add__",
        "monpy.binary_same_shape_contig_typed_static",
        bench_configured[numojo_call](),
        bench_configured[monpy_call](),
        n * size_of[Scalar[DType.float32]]() * 3,
        n,
    )
    lhs.free()
    rhs.free()
    out.free()


def emit_numojo_sin_f32[n: Int]() raises:
    var src_nm = arange[f32](0.0, Float32(n))
    var src = alloc[Scalar[DType.float32]](n)
    var out = alloc[Scalar[DType.float32]](n)
    fill_buffer[DType.float32](src, n)

    @parameter
    def numojo_call() raises:
        var result = sin(src_nm)
        keep(result.size)

    @parameter
    def monpy_call() raises:
        unary_contig_typed[DType.float32](src, out, n, UnaryOp.SIN.value)
        keep(out[0])

    emit_result(
        "numojo.elementwise",
        String("sin_f32_{}").format(n),
        "numojo.sin",
        "monpy.unary_contig_typed",
        bench_configured[numojo_call](),
        bench_configured[monpy_call](),
        n * size_of[Scalar[DType.float32]]() * 2,
        n,
    )
    src.free()
    out.free()


def emit_numojo_sum_f32[n: Int]() raises:
    var src_nm = arange[f32](0.0, Float32(n))
    var src = alloc[Scalar[DType.float32]](n)
    fill_buffer[DType.float32](src, n)

    @parameter
    def numojo_call() raises:
        var result = sum(src_nm)
        keep(result)

    @parameter
    def monpy_call() raises:
        var result = reduce_sum_typed[DType.float32](src, n)
        keep(result)

    emit_result(
        "numojo.reductions",
        String("sum_f32_{}").format(n),
        "numojo.sum",
        "monpy.reduce_sum_typed",
        bench_configured[numojo_call](),
        bench_configured[monpy_call](),
        n * size_of[Scalar[DType.float32]](),
        n,
    )
    src.free()


def emit_numojo_matmul_f32[n: Int]() raises:
    var lhs_nm = arange[f32](0.0, Float32(n * n))
    var rhs_nm = arange[f32](0.0, Float32(n * n))
    lhs_nm.resize(Shape(n, n))
    rhs_nm.resize(Shape(n, n))

    var lhs = alloc[Scalar[DType.float32]](n * n)
    var rhs = alloc[Scalar[DType.float32]](n * n)
    var out = alloc[Scalar[DType.float32]](n * n)
    fill_matrix[DType.float32](lhs, n)
    fill_matrix[DType.float32](rhs, n)

    @parameter
    def numojo_call() raises:
        var result = lhs_nm @ rhs_nm
        keep(result.size)

    @parameter
    def monpy_call() raises:
        matmul_small_typed[DType.float32](lhs, rhs, out, n, n, n)
        keep(out[0])

    emit_result(
        "numojo.matmul",
        String("matmul_f32_{}").format(n),
        "numojo.NDArray.__matmul__",
        "monpy.matmul_small_typed",
        bench_configured[numojo_call](),
        bench_configured[monpy_call](),
        n * n * size_of[Scalar[DType.float32]]() * 3,
        2 * n * n * n,
    )
    lhs.free()
    rhs.free()
    out.free()


def main() raises:
    emit_header()
    emit_numojo_add_f32[1024]()
    emit_numojo_add_f32[65536]()
    emit_numojo_sin_f32[1024]()
    emit_numojo_sin_f32[65536]()
    emit_numojo_sum_f32[1024]()
    emit_numojo_sum_f32[65536]()
    emit_numojo_matmul_f32[8]()
    emit_numojo_matmul_f32[16]()
