from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from monpy.kernels.api import JittedFunction
from monpy.kernels.ir import Op, TensorSpec as KernelTensorSpec
from monpy.kernels.tensor import Tensor

KernelDecorator = Callable[[Callable[..., object]], JittedFunction]
KernelJitFactory = Callable[..., KernelDecorator]


def test_top_level_jit_compile_usage_traces_graph_model_fragment() -> None:
  import monpy as mp

  jit = cast(KernelJitFactory, mp.jit)
  tensor_spec = cast(type[KernelTensorSpec], mp.TensorSpec)

  def block(x: Tensor, w: Tensor, bias: Tensor) -> Tensor:
    y = (x @ w) + bias
    return y.astype(mp.float32).reshape((2, 4))

  compiled = jit(backend="graph", cache_size=8)(block).compile(
    tensor_spec((2, 3), mp.bfloat16, "gpu:0"),
    tensor_spec((3, 4), mp.bfloat16, "gpu:0"),
    tensor_spec((4,), mp.bfloat16, "gpu:0"),
  )

  graph = compiled.graph
  assert compiled.backend == "graph"
  assert graph.inputs == (0, 1, 2)
  assert graph.outputs == (6,)
  assert [node.op for node in graph.nodes] == [
    Op.INPUT,
    Op.INPUT,
    Op.INPUT,
    Op.MATMUL,
    Op.ADD,
    Op.CAST,
    Op.RESHAPE,
  ]
  assert graph.nodes[0].spec.device.kind == "gpu"
  assert graph.nodes[0].spec.device.index == 0
  assert graph.nodes[0].spec.dtype.name == "bfloat16"
  assert graph.nodes[3].spec.shape == (2, 4)
  assert graph.nodes[4].spec.layout.is_row_major_compact()
  assert graph.nodes[5].attrs == {"dtype": "float32"}
  assert graph.nodes[6].spec.dtype.name == "float32"
  assert graph.nodes[6].spec.layout.shape == (2, 4)
  assert graph.structural_key


def test_monpy_kernels_namespace_usage_for_specs_layouts_and_dtypes() -> None:
  import monpy as mp
  import monpy.kernels as mk

  tiled = mk.LayoutSpec.row_major((8, 16), alignment_bytes=16).with_tile(
    mk.TileSpec((1, 16), logical_order=(0, 1), vector_width=8, warp_shape=(1, 32), swizzle="xor")
  )
  spec = mk.TensorSpec((8, 16), mp.float8_e4m3fn, "gpu:1", tiled)
  packed = mk.TensorSpec((17,), mp.float4_e2m1fn, "gpu:1")

  assert spec.device == mk.DeviceSpec("gpu", 1)
  assert spec.dtype.name == "float8_e4m3fn"
  assert spec.dtype.storage_bits == 8
  assert spec.dtype.byte_width == 1
  assert spec.layout is tiled

  params = spec.layout.as_parameters("x")
  assert params["x_rank"] == 2
  assert params["x_alignment_bytes"] == 16
  assert params["x_tile_rank"] == 2
  assert params["x_tile_shape_1"] == 16
  assert params["x_tile_vector_width"] == 8
  assert params["x_tile_swizzle"] == "xor"

  assert packed.dtype.name == "float4_e2m1fn"
  assert packed.dtype.is_packed
  assert packed.dtype.storage_bits == 4
  with pytest.raises(ValueError, match="integer byte width"):
    _ = packed.dtype.byte_width

  def passthrough_fn(x: Tensor) -> Tensor:
    return x

  passthrough = cast(KernelJitFactory, mk.jit)(backend="graph")(passthrough_fn)
  compiled = passthrough.compile(spec)
  assert compiled.graph.inputs == (0,)
  assert compiled.graph.outputs == (0,)
  assert compiled.graph.nodes[0].spec == spec


def test_custom_call_usage_carries_layout_attrs_for_extension_kernels() -> None:
  import monpy as mp
  import monpy.kernels as mk
  from monpy.kernels.graph import GraphLowerer, LayoutAction

  def rms_norm_fn(x: Tensor, weight: Tensor) -> Tensor:
    tile = mk.TileSpec((1, 128), vector_width=8)
    out = mk.TensorSpec(x.spec.shape, x.spec.dtype, x.spec.device, x.spec.layout.with_tile(tile))
    return x._trace.custom_call("monpy.rms_norm", (x, weight), out)

  rms_norm = cast(KernelJitFactory, mk.jit)(backend="graph")(rms_norm_fn)
  compiled = rms_norm.compile(
    mk.TensorSpec((4, 128), mp.float32, "gpu:0"),
    mk.TensorSpec((128,), mp.float32, "gpu:0"),
  )

  graph = compiled.graph
  assert graph.outputs == (2,)
  assert graph.nodes[2].attrs == {"name": "monpy.rms_norm"}
  assert graph.nodes[2].spec.layout.tile == mk.TileSpec((1, 128), vector_width=8)

  lowered = GraphLowerer().lower(graph)
  assert len(lowered.layout_decisions) == 1
  decision = lowered.layout_decisions[0]
  assert decision.action is LayoutAction.CUSTOM_ATTRS
  assert decision.parameters is not None
  assert decision.parameters["layout_shape_1"] == 128
  assert decision.parameters["layout_tile_shape_1"] == 128
  assert decision.parameters["layout_tile_vector_width"] == 8


def test_jitted_function_call_requires_explicit_compile_boundary() -> None:
  import monpy as mp

  def identity_fn(x: Tensor) -> Tensor:
    return x

  identity = cast(KernelDecorator, mp.jit)(identity_fn)
  with pytest.raises(TypeError, match=r"call \.compile\(\.\.\.\) with TensorSpec inputs"):
    identity(mp.zeros((2,), dtype=mp.float32))


def test_external_weight_binding_is_not_a_numpy_fallback() -> None:
  import monpy as mp

  tensor_spec = cast(type[KernelTensorSpec], mp.TensorSpec)

  def identity_fn(x: Tensor) -> Tensor:
    return x

  identity = cast(KernelDecorator, mp.jit)(identity_fn)
  spec = tensor_spec((2,), mp.float32, "cpu")
  with pytest.raises(NotImplementedError, match="external weight binding"):
    identity.compile(spec, weights={"weight": "model.safetensors"})
