from std.memory.unsafe_pointer import alloc


@fieldwise_init
struct Storage(Movable, Writable):
    var data: UnsafePointer[UInt8, MutExternalOrigin]
    var byte_len: Int
    var ref_count: Int
    var owns_data: Bool


def make_managed_storage(
    byte_len: Int,
) raises -> UnsafePointer[Storage, MutExternalOrigin]:
    var actual_byte_len = byte_len
    if actual_byte_len < 1:
        actual_byte_len = 1
    var data = alloc[UInt8](actual_byte_len)
    var storage = alloc[Storage](1)
    var value = Storage(data, actual_byte_len, 1, True)
    storage.init_pointee_move(value^)
    return storage


def make_external_storage(
    data: UnsafePointer[UInt8, MutExternalOrigin], byte_len: Int
) raises -> UnsafePointer[Storage, MutExternalOrigin]:
    var actual_byte_len = byte_len
    if actual_byte_len < 1:
        actual_byte_len = 1
    var storage = alloc[Storage](1)
    var value = Storage(data, actual_byte_len, 1, False)
    storage.init_pointee_move(value^)
    return storage


def retain_storage(storage: UnsafePointer[Storage, MutExternalOrigin]) -> UnsafePointer[Storage, MutExternalOrigin]:
    storage[].ref_count += 1
    return storage


def release_storage(storage: UnsafePointer[Storage, MutExternalOrigin]):
    storage[].ref_count -= 1
    if storage[].ref_count == 0:
        if storage[].owns_data:
            storage[].data.free()
        storage.destroy_pointee()
        storage.free()
