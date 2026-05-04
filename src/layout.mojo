from std.sys import simd_width_of
from std.utils import IndexList

from array import Array


@fieldwise_init
struct Shape[rank: Int](Movable, Writable):
    var dims: IndexList[Self.rank]

    def __getitem__(self, axis: Int) -> Int:
        return self.dims[axis]

    def size(self) -> Int:
        var total = 1
        comptime for axis in range(Self.rank):
            total *= self.dims[axis]
        return total


@fieldwise_init
struct Layout[rank: Int](Movable, Writable):
    var shape: Shape[Self.rank]
    var strides: IndexList[Self.rank]
    var offset: Int

    def __call__(self, coord: IndexList[Self.rank]) -> Int:
        var physical = self.offset
        comptime for axis in range(Self.rank):
            physical += coord[axis] * self.strides[axis]
        return physical

    def tile_shape[*sizes: Int](self) -> IndexList[Self.rank]:
        var out = IndexList[Self.rank]()
        comptime for axis in range(Self.rank):
            out[axis] = self.shape[axis]
        comptime for axis in range(len(sizes)):
            if axis < Self.rank:
                out[axis] = sizes[axis]
        return out


@fieldwise_init
struct LayoutTensor[rank: Int](Movable, Writable):
    var data: UnsafePointer[UInt8, MutExternalOrigin]
    var dtype_code: Int
    var layout: Layout[Self.rank]


def layout_from_array[rank: Int](array: Array) raises -> Layout[rank]:
    if len(array.shape) != rank:
        raise Error("array rank does not match requested static layout rank")
    var dims = IndexList[rank]()
    var strides = IndexList[rank]()
    comptime for axis in range(rank):
        dims[axis] = array.shape[axis]
        strides[axis] = array.strides[axis]
    return Layout[rank](Shape[rank](dims), strides, array.offset_elems)


def as_tensor[rank: Int](array: Array) raises -> LayoutTensor[rank]:
    return LayoutTensor[rank](
        array.data,
        array.dtype_code,
        layout_from_array[rank](array),
    )


def vector_width[dtype: DType]() -> Int:
    return simd_width_of[dtype]()
