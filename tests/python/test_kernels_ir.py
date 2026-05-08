from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_monpy_import_keeps_kernel_backends_lazy() -> None:
  code = textwrap.dedent(
    """
    import sys

    import monpy

    assert "max" not in sys.modules
    assert "safetensors" not in sys.modules
    assert "xgrammar" not in sys.modules
    assert "monpy.kernels" not in sys.modules

    assert monpy.jit is not None
    assert "monpy.kernels" in sys.modules
    assert "max" not in sys.modules
    assert "safetensors" not in sys.modules
    assert "xgrammar" not in sys.modules
    """
  )
  env = dict(os.environ)
  if "MOHAUS_MOJO" not in env and "MODULAR_DERIVED_PATH" in env:
    env["MOHAUS_MOJO"] = os.path.join(env["MODULAR_DERIVED_PATH"], "build", "bin", "mojo")
  subprocess.run([sys.executable, "-I", "-c", code], check=True, env=env)


def test_layout_spec_native_lowering_decisions() -> None:
  from monpy.kernels.graph import GraphLowerer, LayoutAction
  from monpy.kernels.layout import LayoutSpec, TileSpec

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
  from monpy.kernels.ir import Op

  @mp.jit
  def f(x: mp.Tensor, w: mp.Tensor) -> mp.Tensor:
    y = mp.matmul(x, w)
    return mp.transpose(mp.reshape(y, (4, 2)), (1, 0))

  compiled = f.compile(
    mp.TensorSpec((2, 3), mp.float32, "cpu"),
    mp.TensorSpec((3, 4), mp.float32, "cpu"),
  )

  graph = compiled.graph
  assert [node.op for node in graph.nodes] == [
    Op.INPUT,
    Op.INPUT,
    Op.MATMUL,
    Op.RESHAPE,
    Op.TRANSPOSE,
  ]
  assert graph.outputs == (4,)
  assert graph.nodes[4].spec.shape == (2, 4)
  assert graph.nodes[4].spec.layout.shape == (2, 4)
  assert graph.structural_key


def test_graph_ir_key_changes_when_layout_changes() -> None:
  import monpy as mp

  @mp.jit
  def plain(x: mp.Tensor) -> mp.Tensor:
    return mp.reshape(x, (3, 2))

  @mp.jit
  def transposed(x: mp.Tensor) -> mp.Tensor:
    return mp.transpose(x, (1, 0))

  spec = mp.TensorSpec((2, 3), mp.float32, "cpu")

  plain_graph = plain.compile(spec).graph
  transposed_graph = transposed.compile(spec).graph

  assert plain_graph.structural_key != transposed_graph.structural_key
