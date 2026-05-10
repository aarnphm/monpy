from __future__ import annotations

import pytest

from monpy._src.tree_util import tree_flatten, tree_map, tree_unflatten


def test_private_pytree_flattens_dicts_in_stable_key_order() -> None:
  leaves, treedef = tree_flatten({"z": [1, None], "a": (2, 3)})

  assert leaves == (2, 3, 1)
  assert tree_unflatten(treedef, leaves) == {"a": (2, 3), "z": [1, None]}
  assert tree_map(lambda value: value + 10, {"z": [1], "a": (2,)}) == {"a": (12,), "z": [11]}


def test_vmap_maps_the_leading_axis() -> None:
  import monpy as mp
  import monpy.lax as lax

  out = lax.vmap(lambda x: x * 2)(mp.asarray([[1, 2], [3, 4]]))

  assert out.tolist() == [[2, 4], [6, 8]]


def test_vmap_supports_flat_in_axes_and_out_axes() -> None:
  import monpy as mp
  import monpy.lax as lax

  x = mp.asarray([[1, 2, 3], [4, 5, 6]])
  bias = mp.asarray([10, 20])

  out = lax.vmap(lambda column, offset: column + offset, in_axes=(1, None), out_axes=1)(x, bias)

  assert out.tolist() == [[11, 12, 13], [24, 25, 26]]


def test_vmap_maps_keyword_arguments_over_axis_zero() -> None:
  import monpy as mp
  import monpy.lax as lax

  out = lax.vmap(lambda x, *, y: x + y)(mp.asarray([1, 2, 3]), y=mp.asarray([10, 20, 30]))

  assert out.tolist() == [11, 22, 33]


def test_vmap_stacks_tuple_outputs() -> None:
  import monpy as mp
  import monpy.lax as lax

  mapped = lax.vmap(lambda x: (x + 1, mp.sum(x)), out_axes=(0, 0))(mp.asarray([[1, 2], [3, 4]]))

  assert mapped[0].tolist() == [[2, 3], [4, 5]]
  assert mapped[1].tolist() == [3, 7]


def test_vmap_stacks_nested_pytree_outputs() -> None:
  import monpy as mp
  import monpy.lax as lax

  mapped = lax.vmap(
    lambda x: {"z": [x + 1, None], "a": (mp.sum(x), x)},
    out_axes={"a": (0, 0), "z": [0, None]},
  )(mp.asarray([[1, 2], [3, 4]]))

  assert mapped["a"][0].tolist() == [3, 7]
  assert mapped["a"][1].tolist() == [[1, 2], [3, 4]]
  assert mapped["z"][0].tolist() == [[2, 3], [4, 5]]
  assert mapped["z"][1] is None


def test_vmap_out_axes_none_requires_unmapped_output() -> None:
  import monpy as mp
  import monpy.lax as lax

  assert lax.vmap(lambda x: 7, out_axes=None)(mp.asarray([1, 2, 3])) == 7
  with pytest.raises(ValueError, match="out_axes=None"):
    lax.vmap(lambda x: x, out_axes=None)(mp.asarray([1, 2, 3]))


def test_vmap_rejects_changed_pytree_output_structure() -> None:
  import monpy as mp
  import monpy.lax as lax

  with pytest.raises(ValueError, match="pytree structure mismatch"):
    lax.vmap(lambda x: (x if int(x) == 1 else [x]))(mp.asarray([1, 2]))


def test_vmap_rejects_mismatched_axis_sizes() -> None:
  import monpy as mp
  import monpy.lax as lax

  with pytest.raises(ValueError, match="same size"):
    lax.vmap(lambda x, y: x + y)(mp.asarray([1, 2]), mp.asarray([1, 2, 3]))
