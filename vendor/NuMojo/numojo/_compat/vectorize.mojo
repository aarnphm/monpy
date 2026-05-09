# ===----------------------------------------------------------------------=== #
# NuMojo compatibility helpers for MonPy's vendored checkout.
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See ../../LICENSE and https://llvm.org/LICENSE.txt.
# ===----------------------------------------------------------------------=== #
"""Compatibility `vectorize` overloads for current Mojo compilers.

NuMojo was written against a closure-capture spelling that current Mojo no
longer accepts. The stdlib `vectorize` overloads still require a non-capturing
callable, so these local overloads keep NuMojo's captured-loop call sites
building while we vendor the library.
"""


@always_inline
def vectorize[
    simd_width: Int,
    func: def[width: Int](idx: Int) capturing -> None,
    /,
    *,
    unroll_factor: Int = 1,
](size: Int):
    comptime assert simd_width > 0, "simd width must be > 0"
    assert size >= 0, "size must be >= 0"

    var simd_end = size - (size % simd_width)
    for simd_idx in range(0, simd_end, simd_width):
        func[simd_width](simd_idx)

    for i in range(simd_end, size):
        func[1](i)


@always_inline
def vectorize[
    simd_width: Int,
    func: def[width: Int](idx: Int, evl: Int) capturing -> None,
    /,
    *,
    unroll_factor: Int = 1,
](size: Int):
    comptime assert simd_width > 0, "simd width must be > 0"
    assert size >= 0, "size must be >= 0"

    var simd_end = size - (size % simd_width)
    for simd_idx in range(0, simd_end, simd_width):
        func[simd_width](simd_idx, simd_width)

    var remainder = size - simd_end
    if remainder > 0:
        func[simd_width](simd_end, remainder)
