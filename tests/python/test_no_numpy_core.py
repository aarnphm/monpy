from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_core_import_and_buffer_paths_do_not_import_numpy() -> None:
  code = textwrap.dedent(
    """
    import array
    import importlib.abc
    import sys

    class BlockNumpy(importlib.abc.MetaPathFinder):
      def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
          raise ModuleNotFoundError("numpy import blocked for core smoke test")
        return None

    sys.meta_path.insert(0, BlockNumpy())

    import monpy
    import monumpy

    assert not hasattr(monpy, "runtime")
    import monpy.numpy as monpy_numpy
    assert set(monpy_numpy.__all__) == {
      "NumpyDTypeInfo",
      "array_interface_typestr",
      "buffer_format",
      "dtype_from_buffer_format",
      "dtype_from_typestr",
      "dtype_info",
      "ops",
    }
    from monpy.numpy import dtype_info, ops
    assert callable(ops.from_numpy)
    assert dtype_info(monpy.float32).typestr == "<f4"
    assert ops.is_array_input([1, 2, 3]) is False
    assert ops.is_dtype_input(int) is False
    assert ops.resolve_dtype(monpy.float32) is monpy.float32
    assert "numpy" not in sys.modules

    arr = monpy.asarray([1, 2, 3], dtype=monpy.int64)
    assert arr.tolist() == [1, 2, 3]
    assert (arr + 2).tolist() == [3, 4, 5]
    assert monumpy.asarray([True, False]).dtype == monpy.bool
    assert monpy.random.key_data(monpy.random.key(0)).shape == (2,)
    assert monpy.random.random(monpy.random.key(1), size=(2,)).shape == (2,)
    assert monumpy.random.key_data is monpy.random.key_data
    assert "numpy" not in sys.modules

    exported = array.array("i", [1, 2, 3])
    view = monpy.asarray(exported, copy=False)
    exported[1] = 99
    assert view.tolist() == [1, 99, 3]

    raw = bytearray([1, 2, 3, 4])
    raw_view = monpy.frombuffer(raw, dtype=monpy.uint8)
    raw_view[2] = 77
    assert raw[2] == 77

    readonly = monpy.frombuffer(bytes([5, 6, 7]), dtype=monpy.uint8)
    readonly[0] = 88
    assert readonly.tolist() == [88, 6, 7]

    memory = monpy.asarray(memoryview(bytearray([8, 9])), copy=False)
    assert memory.tolist() == [8, 9]

    iface = arr.__array_interface__
    assert iface["shape"] == (3,)
    assert iface["typestr"] == "<i8"
    assert not hasattr(arr, "__array__")
    assert "numpy" not in sys.modules
    """
  )
  env = dict(os.environ)
  if "MOHAUS_MOJO" not in env and "MODULAR_DERIVED_PATH" in env:
    env["MOHAUS_MOJO"] = os.path.join(env["MODULAR_DERIVED_PATH"], "build", "bin", "mojo")
  subprocess.run([sys.executable, "-I", "-c", code], check=True, env=env)
