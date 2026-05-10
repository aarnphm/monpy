from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_monpy_lax_and_extend_imports_are_lazy_and_old_kernels_breaks() -> None:
  code = textwrap.dedent(
    """
    import importlib
    import sys

    import monpy

    assert "max" not in sys.modules
    assert "safetensors" not in sys.modules
    assert "numpy" not in sys.modules
    assert "monpy.lax" not in sys.modules
    assert "monpy.extend" not in sys.modules
    assert "monpy.kernels" not in sys.modules

    for name in ("jit", "vmap", "Tensor", "TensorSpec", "LayoutSpec", "TileSpec", "DTypeSpec", "DeviceSpec", "SymbolicDim"):
      assert not hasattr(monpy, name), name

    try:
      importlib.import_module("monpy.kernels")
    except ModuleNotFoundError:
      pass
    else:
      raise AssertionError("monpy.kernels must not be import-compatible")

    import monpy.lax as lax
    assert callable(lax.jit)
    assert callable(lax.vmap)
    assert "max" not in sys.modules
    assert "safetensors" not in sys.modules
    assert "numpy" not in sys.modules

    import monpy.extend as extend
    assert extend.MAX_TARGET == "max"
    assert extend.MOJO_TARGET == "mojo"
    assert "max" not in sys.modules
    assert "safetensors" not in sys.modules
    assert "numpy" not in sys.modules
    """
  )
  env = dict(os.environ)
  if "MOHAUS_MOJO" not in env and "MODULAR_DERIVED_PATH" in env:
    env["MOHAUS_MOJO"] = os.path.join(env["MODULAR_DERIVED_PATH"], "build", "bin", "mojo")
  subprocess.run([sys.executable, "-I", "-c", code], check=True, env=env)


def test_numpy_inputs_enter_through_numpy_ops_boundary() -> None:
  code = textwrap.dedent(
    """
    import sys
    import numpy
    import monpy

    assert "monpy.numpy.ops" not in sys.modules
    arr = monpy.asarray(numpy.asarray([1, 2, 3], dtype=numpy.int64))
    assert arr.tolist() == [1, 2, 3]
    assert "monpy.numpy.ops" in sys.modules
    assert "max" not in sys.modules
    assert "safetensors" not in sys.modules

    assert monpy.dtype(numpy.float32) is monpy.float32
    """
  )
  env = dict(os.environ)
  if "MOHAUS_MOJO" not in env and "MODULAR_DERIVED_PATH" in env:
    env["MOHAUS_MOJO"] = os.path.join(env["MODULAR_DERIVED_PATH"], "build", "bin", "mojo")
  subprocess.run([sys.executable, "-I", "-c", code], check=True, env=env)


def test_layout_spec_native_lowering_decisions() -> None:
  from monpy.extend.mlir_or_max import GraphLowerer, LayoutAction
  from monpy.lax import LayoutSpec, TileSpec

  lowerer = GraphLowerer()
  base = LayoutSpec.row_major((2, 3))

  reshape = LayoutSpec.row_major((3, 2))
  assert lowerer.lower_layout(None, base, reshape).action is LayoutAction.NATIVE_RESHAPE

  permuted = base.permute((1, 0))
  permute_decision = lowerer.lower_layout(None, base, permuted)
  assert permute_decision.action is LayoutAction.NATIVE_PERMUTE
  assert permute_decision.parameters == {"axis_0": 1, "axis_1": 0, "rank": 2}

  broadcast = LayoutSpec.row_major((3,)).broadcast_to((2, 3))
  assert lowerer.lower_layout(None, LayoutSpec.row_major((3,)), broadcast).action is LayoutAction.NATIVE_BROADCAST

  tiled = base.with_tile(TileSpec((1, 3), vector_width=4))
  custom = lowerer.lower_layout(None, base, tiled)
  assert custom.action is LayoutAction.CUSTOM_ATTRS
  assert custom.parameters is not None
  assert custom.parameters["layout_tile_shape_1"] == 3
  assert custom.parameters["layout_tile_vector_width"] == 4


def test_jit_traces_graph_ir_with_layout_metadata() -> None:
  import monpy as mp
  import monpy.lax as lax

  @lax.jit
  def f(x: lax.Tensor, w: lax.Tensor) -> lax.Tensor:
    y = mp.matmul(x, w)
    return mp.transpose(mp.reshape(y, (4, 2)), (1, 0))

  compiled = f.compile(
    lax.TensorSpec((2, 3), mp.float32, "cpu"),
    lax.TensorSpec((3, 4), mp.float32, "cpu"),
  )

  graph = compiled.graph
  assert [node.op for node in graph.nodes] == ["input", "input", "matmul", "reshape", "transpose"]
  assert [node.primitive.name for node in graph.nodes] == ["input", "input", "matmul", "reshape", "transpose"]
  assert graph.outputs == (4,)
  assert graph.nodes[4].spec.shape == (2, 4)
  assert graph.nodes[4].spec.layout.shape == (2, 4)
  assert graph.structural_key


def test_graph_ir_key_changes_when_shape_dtype_params_or_layout_change() -> None:
  import monpy as mp
  import monpy.lax as lax

  @lax.jit
  def plain(x: lax.Tensor) -> lax.Tensor:
    return mp.reshape(x, (3, 2))

  @lax.jit
  def transposed(x: lax.Tensor) -> lax.Tensor:
    return mp.transpose(x, (1, 0))

  spec = lax.TensorSpec((2, 3), mp.float32, "cpu")

  plain_graph = plain.compile(spec).graph
  transposed_graph = transposed.compile(spec).graph
  dtype_graph = plain.compile(lax.TensorSpec((2, 3), mp.float64, "cpu")).graph
  shape_graph = plain.compile(lax.TensorSpec((3, 2), mp.float32, "cpu")).graph

  assert plain_graph.structural_key != transposed_graph.structural_key
  assert plain_graph.structural_key != dtype_graph.structural_key
  assert plain_graph.structural_key != shape_graph.structural_key
