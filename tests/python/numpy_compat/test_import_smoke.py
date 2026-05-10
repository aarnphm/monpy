from __future__ import annotations


def test_monpy_and_monumpy_import_smoke() -> None:
  import monpy
  import monumpy

  assert monpy.asarray([1, 2, 3], dtype=monpy.int64).tolist() == [1, 2, 3]
  assert monumpy.asarray([1, 2, 3], dtype=monumpy.int64).tolist() == [1, 2, 3]
  assert monumpy.float32 is monpy.float32
  assert monumpy.random is monpy.random
  assert callable(monpy.random.default_rng)
  assert callable(monpy.numpy.ops.to_numpy)
  assert callable(monpy.numpy.ops.from_numpy)


def test_array_api_import_smoke() -> None:
  import monpy.array_api as xp

  arr = xp.asarray([1, 2, 3], dtype=xp.int64)

  assert arr.__array_namespace__() is xp
  assert xp.__array_namespace_info__().default_device() == "cpu"


def test_linalg_import_smoke() -> None:
  import monpy.linalg as direct_linalg
  from monpy import linalg as monpy_linalg
  from monpy.array_api import linalg as array_api_linalg
  from monumpy import linalg as monumpy_linalg

  assert direct_linalg.solve is monpy_linalg.solve
  assert monumpy_linalg.matmul is monpy_linalg.matmul
  assert callable(monumpy_linalg.solve)
  assert callable(array_api_linalg.matmul)
  assert callable(array_api_linalg.matrix_transpose)


def test_star_import_surface_smoke() -> None:
  namespace: dict[str, object] = {}

  exec("from monpy import *", namespace)

  assert namespace["asarray"]([1, 2, 3]).tolist() == [1, 2, 3]
  assert callable(namespace["linalg"].solve)
  assert callable(namespace["diagonal"])
  assert callable(namespace["random"].key)
