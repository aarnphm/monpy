from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


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

    assert callable(monpy.jit)
    assert callable(monpy.vmap)
    assert "monpy.lax" not in sys.modules

    for name in ("Tensor", "TensorSpec", "LayoutSpec", "TileSpec", "DTypeSpec", "DeviceSpec", "SymbolicDim"):
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


def test_public_ufuncs_share_lax_primitive_identity() -> None:
  import monpy as mp
  import monpy.lax as lax

  assert mp.add.primitive is lax.add_p
  assert mp.subtract.primitive is lax.sub_p
  assert mp.multiply.primitive is lax.mul_p
  assert mp.divide.primitive is lax.div_p
  assert mp.equal.primitive is lax.equal_p
  assert mp.not_equal.primitive is lax.not_equal_p
  assert mp.less.primitive is lax.less_p
  assert mp.less_equal.primitive is lax.less_equal_p
  assert mp.greater.primitive is lax.greater_p
  assert mp.greater_equal.primitive is lax.greater_equal_p
  assert lax.get_primitive("subtract") is lax.sub_p
  assert lax.get_primitive("multiply") is lax.mul_p
  assert lax.get_primitive("divide") is lax.div_p

  assert lax.add_p.ufunc_kind == "binary"
  assert lax.add_p.ufunc_op == int(mp.BinaryOp.ADD)
  assert lax.add_p.ufunc_nin == 2
  assert lax.add_p.ufunc_nout == 1
  assert lax.add_p.ufunc_identity == 0
  assert lax.add_p.reduce_op == int(mp.ReduceOp.SUM)
  assert lax.greater_p.ufunc_kind == "compare"
  assert lax.greater_p.ufunc_op == int(mp.CompareOp.GT)
  assert lax.greater_p.ufunc_nin == 2
  assert lax.greater_p.ufunc_nout == 1


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


def test_jit_traces_where_as_primitive() -> None:
  import monpy as mp
  import monpy.lax as lax

  @lax.jit
  def f(mask: lax.Tensor, x: lax.Tensor, y: lax.Tensor) -> lax.Tensor:
    return mp.where(mask, x, y)

  compiled = f.compile(
    lax.TensorSpec((2, 3), mp.bool, "cpu"),
    lax.TensorSpec((2, 3), mp.float32, "cpu"),
    lax.TensorSpec((2, 3), mp.float32, "cpu"),
  )

  graph = compiled.graph
  assert [node.op for node in graph.nodes] == ["input", "input", "input", "where"]
  assert graph.nodes[3].primitive is lax.where_p
  assert graph.nodes[3].spec.shape == (2, 3)
  assert graph.nodes[3].spec.dtype.name == "float32"


def test_jit_traces_comparison_ufunc_as_bool_primitive() -> None:
  import monpy as mp
  import monpy.lax as lax

  @mp.jit
  def f(x: lax.Tensor) -> lax.Tensor:
    return mp.greater(x, 0)

  compiled = f.compile(lax.TensorSpec((2, 3), mp.float32, "cpu"))

  graph = compiled.graph
  assert [node.op for node in graph.nodes] == ["input", "constant", "greater"]
  assert graph.nodes[2].primitive is lax.greater_p
  assert graph.nodes[2].spec.shape == (2, 3)
  assert graph.nodes[2].spec.dtype.name == "bool"


def test_jit_traces_where_comparison_docs_example() -> None:
  import monpy as mp
  import monpy.lax as lax

  @mp.jit
  def f(x: lax.Tensor, w: lax.Tensor, bias: lax.Tensor) -> lax.Tensor:
    y = mp.einsum("ij,jk->ik", x, w)
    return mp.where(y > 0, y + bias, 0)

  compiled = f.compile(
    lax.TensorSpec((2, 3), mp.float32, "cpu"),
    lax.TensorSpec((3, 4), mp.float32, "cpu"),
    lax.TensorSpec((4,), mp.float32, "cpu"),
  )

  graph = compiled.graph
  assert [node.op for node in graph.nodes] == [
    "input",
    "input",
    "input",
    "matmul",
    "constant",
    "greater",
    "add",
    "constant",
    "where",
  ]
  assert graph.nodes[5].primitive is lax.greater_p
  assert graph.nodes[5].spec.dtype.name == "bool"
  assert graph.nodes[8].primitive is lax.where_p
  assert graph.nodes[8].spec.shape == (2, 4)


def test_jit_preserves_structured_output_tree() -> None:
  import monpy as mp
  import monpy.lax as lax
  from monpy._src.tree_util import tree_unflatten

  @mp.jit
  def f(x: lax.Tensor, y: lax.Tensor) -> dict[str, object]:
    return {"z": [x + y, None], "a": (mp.sum(x), y)}

  compiled = f.compile(
    lax.TensorSpec((2, 3), mp.float32, "cpu"),
    lax.TensorSpec((2, 3), mp.float32, "cpu"),
  )

  graph = compiled.graph
  assert [node.op for node in graph.nodes] == ["input", "input", "add", "reduce"]
  assert graph.outputs == (3, 1, 2)
  assert tree_unflatten(compiled.output_tree, graph.outputs) == {"a": (3, 1), "z": [2, None]}


def test_jit_rejects_static_output_leaves() -> None:
  import monpy as mp
  import monpy.lax as lax

  @mp.jit
  def f(x: lax.Tensor) -> dict[str, object]:
    return {"x": x, "static": 1}

  with pytest.raises(TypeError, match="pytree of monpy.Tensor"):
    f.compile(lax.TensorSpec((2, 3), mp.float32, "cpu"))


def test_traced_tensor_truthiness_is_rejected() -> None:
  import monpy as mp
  import monpy.lax as lax

  @mp.jit
  def f(x: lax.Tensor) -> lax.Tensor:
    if x:
      return x
    return x

  with pytest.raises(TypeError, match="truth value of a traced monpy.Tensor"):
    f.compile(lax.TensorSpec((2, 3), mp.float32, "cpu"))


def test_jit_rejects_captured_non_scalar_array_constant() -> None:
  import monpy as mp
  import monpy.lax as lax

  captured = mp.asarray([1, 2, 3], dtype=mp.float32)

  @mp.jit
  def f(x: lax.Tensor) -> lax.Tensor:
    return x + captured

  with pytest.raises(NotImplementedError, match="non-scalar constants"):
    f.compile(lax.TensorSpec((3,), mp.float32, "cpu"))


def test_top_level_jit_traces_public_numpy_api() -> None:
  import monpy as mp
  import monpy.lax as lax

  @mp.jit
  def f(x: lax.Tensor, y: lax.Tensor) -> lax.Tensor:
    return mp.add(x, y)

  compiled = f.compile(
    lax.TensorSpec((2, 3), mp.float32, "cpu"),
    lax.TensorSpec((2, 3), mp.float32, "cpu"),
  )

  graph = compiled.graph
  assert [node.op for node in graph.nodes] == ["input", "input", "add"]
  assert graph.nodes[2].primitive is lax.add_p
  assert graph.nodes[2].spec.shape == (2, 3)


def test_jit_traces_sum_axis_as_reduce_primitive() -> None:
  import monpy as mp
  import monpy.lax as lax

  @mp.jit
  def f(x: lax.Tensor) -> lax.Tensor:
    return mp.sum(x, axis=1, keepdims=True)

  compiled = f.compile(lax.TensorSpec((2, 3), mp.float32, "cpu"))

  graph = compiled.graph
  assert [node.op for node in graph.nodes] == ["input", "reduce"]
  assert graph.nodes[1].primitive is lax.reduce_p
  assert graph.nodes[1].attrs == {"axes": (1,), "keepdims": True, "reduce_op": int(mp.ReduceOp.SUM)}
  assert graph.nodes[1].spec.shape == (2, 1)
  assert graph.nodes[1].spec.dtype.name == "float32"


def test_jit_traces_sum_dtype_cast_before_reduce() -> None:
  import monpy as mp
  import monpy.lax as lax

  @mp.jit
  def f(x: lax.Tensor) -> lax.Tensor:
    return mp.sum(x, dtype=mp.float64)

  compiled = f.compile(lax.TensorSpec((2, 3), mp.float32, "cpu"))

  graph = compiled.graph
  assert [node.op for node in graph.nodes] == ["input", "cast", "reduce"]
  assert graph.nodes[1].attrs == {"dtype": "float64"}
  assert graph.nodes[2].primitive is lax.reduce_p
  assert graph.nodes[2].attrs == {"axes": (0, 1), "keepdims": False, "reduce_op": int(mp.ReduceOp.SUM)}
  assert graph.nodes[2].spec.shape == ()
  assert graph.nodes[2].spec.dtype.name == "float64"


def test_jit_traces_ufunc_reduce_as_reduce_primitive() -> None:
  import monpy as mp
  import monpy.lax as lax

  @mp.jit
  def f(x: lax.Tensor) -> lax.Tensor:
    return mp.add.reduce(x, axis=0)

  compiled = f.compile(lax.TensorSpec((2, 3), mp.float32, "cpu"))

  graph = compiled.graph
  assert [node.op for node in graph.nodes] == ["input", "reduce"]
  assert graph.nodes[1].primitive is lax.reduce_p
  assert graph.nodes[1].attrs == {"axes": (0,), "keepdims": False, "reduce_op": int(mp.ReduceOp.SUM)}
  assert graph.nodes[1].spec.shape == (3,)


def test_jit_traces_einsum_matmul_case_as_matmul() -> None:
  import monpy as mp
  import monpy.lax as lax

  @mp.jit
  def f(x: lax.Tensor, w: lax.Tensor) -> lax.Tensor:
    return mp.einsum("ij,jk->ik", x, w)

  compiled = f.compile(
    lax.TensorSpec((2, 3), mp.float32, "cpu"),
    lax.TensorSpec((3, 4), mp.float32, "cpu"),
  )

  graph = compiled.graph
  assert [node.op for node in graph.nodes] == ["input", "input", "matmul"]
  assert graph.nodes[2].primitive is lax.matmul_p
  assert graph.nodes[2].spec.shape == (2, 4)


def test_jit_traces_einsum_dot_case_as_multiply_reduce() -> None:
  import monpy as mp
  import monpy.lax as lax

  @mp.jit
  def f(x: lax.Tensor, y: lax.Tensor) -> lax.Tensor:
    return mp.einsum("i,i->", x, y)

  compiled = f.compile(
    lax.TensorSpec((3,), mp.float32, "cpu"),
    lax.TensorSpec((3,), mp.float32, "cpu"),
  )

  graph = compiled.graph
  assert [node.op for node in graph.nodes] == ["input", "input", "mul", "reduce"]
  assert graph.nodes[2].primitive is lax.mul_p
  assert graph.nodes[2].spec.shape == (3,)
  assert graph.nodes[3].primitive is lax.reduce_p
  assert graph.nodes[3].attrs == {"axes": (0,), "keepdims": False, "reduce_op": int(mp.ReduceOp.SUM)}
  assert graph.nodes[3].spec.shape == ()


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
