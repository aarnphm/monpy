from __future__ import annotations

import builtins


def test_module_seed_rand_and_random_sample_are_reproducible() -> None:
  import monpy as mp

  mp.random.seed(2026)
  first = mp.random.rand(2, 3)
  first_scalar = mp.random.random_sample()
  mp.random.seed(2026)
  second = mp.random.rand(2, 3)
  second_scalar = mp.random.random_sample()

  assert first.shape == (2, 3)
  assert first.tolist() == second.tolist()
  assert first_scalar == second_scalar
  assert isinstance(mp.random.rand(), builtins.float)
  assert mp.random.sample((2,)).shape == (2,)
  assert mp.random.ranf((2,)).shape == (2,)


def test_module_randn_randint_uniform_and_normal_shapes() -> None:
  import monpy as mp

  mp.random.seed(11)
  gaussian = mp.random.randn(2, 2)
  ints = mp.random.randint(3, 9, size=(64,), dtype=mp.int64)
  uniform = mp.random.uniform(-1.0, 2.0, size=(16,))
  normal = mp.random.normal(5.0, 0.25, size=(8,))

  assert gaussian.shape == (2, 2)
  assert ints.dtype == mp.int64
  assert all(3 <= builtins.int(value) < 9 for value in ints.tolist())
  assert all(-1.0 <= builtins.float(value) < 2.0 for value in uniform.tolist())
  assert normal.shape == (8,)


def test_default_rng_methods_are_reproducible_and_advance_state() -> None:
  import monpy as mp

  left = mp.random.default_rng(99)
  right = mp.random.default_rng(99)

  first = left.random((4,))
  second = left.random((4,))

  assert first.tolist() == right.random((4,)).tolist()
  assert first.tolist() != second.tolist()
  assert left.standard_normal((2,), dtype=mp.float32).dtype == mp.float32
  assert left.uniform(1.0, 2.0, size=(2,)).shape == (2,)
  assert left.normal(0.0, 1.0, size=(2,)).shape == (2,)


def test_generator_integers_endpoint_and_dtype() -> None:
  import monpy as mp

  rng = mp.random.default_rng(7)
  values = rng.integers(1, 3, size=(64,), dtype=mp.int32, endpoint=True)

  assert values.dtype == mp.int32
  assert all(1 <= builtins.int(value) <= 3 for value in values.tolist())


def test_monumpy_random_points_at_monpy_random_surface() -> None:
  import monpy
  import monumpy

  assert monumpy.random is monpy.random
  assert callable(monumpy.random.default_rng)
  assert callable(monumpy.random.seed)
