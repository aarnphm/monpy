from __future__ import annotations

import builtins


def test_key_sampler_is_deterministic_and_non_mutating() -> None:
  import monpy as mp

  k = mp.random.key(123)

  first = mp.random.uniform(k, size=(4,), dtype=mp.float64)
  second = mp.random.uniform(k, size=(4,), dtype=mp.float64)

  assert first.tolist() == second.tolist()
  assert mp.random.key_data(k).tolist() == mp.random.key_data(k).tolist()
  assert not hasattr(mp.random, "PRNGKey")


def test_split_and_fold_in_are_deterministic() -> None:
  import monpy as mp

  k = mp.random.key(0)
  keys = mp.random.split(k, 3)
  keys_again = mp.random.split(k, 3)

  assert len(keys) == 3
  assert mp.random.key_data(keys).shape == (3, 2)
  assert mp.random.key_data(keys).tolist() == mp.random.key_data(keys_again).tolist()
  assert mp.random.key_data(keys[0]).tolist() != mp.random.key_data(keys[1]).tolist()
  fold_9 = mp.random.key_data(mp.random.fold_in(k, 9)).tolist()
  assert fold_9 == mp.random.key_data(mp.random.fold_in(k, 9)).tolist()
  assert fold_9 != mp.random.key_data(mp.random.fold_in(k, 10)).tolist()


def test_key_data_round_trips() -> None:
  import monpy as mp

  k = mp.random.key(77)
  restored = mp.random.wrap_key_data(mp.random.key_data(k))
  batch = mp.random.split(k, 4)
  restored_batch = mp.random.wrap_key_data(mp.random.key_data(batch))

  assert isinstance(restored, mp.random.Key)
  assert isinstance(restored_batch, mp.random.KeyBatch)
  assert mp.random.key_data(restored).tolist() == mp.random.key_data(k).tolist()
  restored_data = mp.random.key_data(restored_batch).tolist()
  assert restored_data == mp.random.key_data(batch).tolist()


def test_sampler_shapes_dtypes_and_ranges() -> None:
  import monpy as mp

  k = mp.random.key(5)
  scalar = mp.random.random(k)
  floats = mp.random.random(k, size=(2, 3), dtype=mp.float32)
  jaxish_uniform = mp.random.uniform(k, (2,), dtype=mp.float32)
  jaxish_normal = mp.random.normal(k, (2,), dtype=mp.float32)
  normals = mp.random.normal(k, loc=10.0, scale=2.0, size=(3,), dtype=mp.float64)
  ints = mp.random.randint(k, 2, 8, size=(32,), dtype=mp.int32)
  raw32 = mp.random.bits(k, shape=(4,), dtype=mp.uint32)
  raw64 = mp.random.bits(k, shape=(4,), dtype=mp.uint64)

  assert scalar.shape == ()
  assert floats.shape == (2, 3)
  assert floats.dtype == mp.float32
  assert jaxish_uniform.shape == (2,)
  assert jaxish_uniform.dtype == mp.float32
  assert jaxish_normal.shape == (2,)
  assert jaxish_normal.dtype == mp.float32
  assert normals.shape == (3,)
  assert ints.dtype == mp.int32
  assert all(2 <= builtins.int(value) < 8 for value in ints.tolist())
  assert raw32.dtype == mp.uint32
  assert raw32.shape == (4,)
  assert raw64.dtype == mp.uint64
  assert raw64.shape == (4,)


def test_stateful_sampler_scalarizes_only_numpy_style_calls() -> None:
  import monpy as mp

  mp.random.seed(42)
  scalar = mp.random.random()
  arr = mp.random.random((2,))

  assert isinstance(scalar, builtins.float)
  assert arr.shape == (2,)


def test_vmap_maps_over_key_batches() -> None:
  import monpy as mp

  keys = mp.random.split(mp.random.key(0), 3)
  out = mp.vmap(lambda k: mp.random.normal(k, size=(2,)))(keys)

  assert out.shape == (3, 2)
