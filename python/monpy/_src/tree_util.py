"""Small pytree helpers for transform argument and result structure."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, cast

TreeKind = Literal["leaf", "none", "tuple", "list", "dict"]


@dataclass(frozen=True, slots=True)
class PyTreeDef:
  kind: TreeKind
  metadata: tuple[object, ...] = ()
  children: tuple["PyTreeDef", ...] = ()

  @property
  def num_leaves(self) -> int:
    if self.kind == "leaf":
      return 1
    return sum(child.num_leaves for child in self.children)


def _dict_keys(tree: Mapping[object, object]) -> tuple[object, ...]:
  return tuple(sorted(tree.keys(), key=lambda key: (type(key).__module__, type(key).__qualname__, repr(key))))


def tree_flatten(tree: object) -> tuple[tuple[object, ...], PyTreeDef]:
  if tree is None:
    return (), PyTreeDef("none")
  if isinstance(tree, tuple):
    flattened = tuple(tree_flatten(item) for item in tree)
    leaves = tuple(leaf for item_leaves, _ in flattened for leaf in item_leaves)
    return leaves, PyTreeDef("tuple", (len(tree),), tuple(defn for _, defn in flattened))
  if isinstance(tree, list):
    flattened = tuple(tree_flatten(item) for item in tree)
    leaves = tuple(leaf for item_leaves, _ in flattened for leaf in item_leaves)
    return leaves, PyTreeDef("list", (len(tree),), tuple(defn for _, defn in flattened))
  if isinstance(tree, Mapping):
    mapping = cast(Mapping[object, object], tree)
    keys = _dict_keys(mapping)
    flattened = tuple(tree_flatten(mapping[key]) for key in keys)
    leaves = tuple(leaf for item_leaves, _ in flattened for leaf in item_leaves)
    return leaves, PyTreeDef("dict", keys, tuple(defn for _, defn in flattened))
  return (tree,), PyTreeDef("leaf")


def tree_unflatten(treedef: PyTreeDef, leaves: Iterable[object]) -> object:
  iterator = iter(leaves)
  result = _tree_unflatten(treedef, iterator)
  try:
    next(iterator)
  except StopIteration:
    return result
  raise ValueError("too many leaves for pytree definition")


def _tree_unflatten(treedef: PyTreeDef, iterator: Iterator[object]) -> object:
  if treedef.kind == "leaf":
    try:
      return next(iterator)
    except StopIteration as exc:
      raise ValueError("not enough leaves for pytree definition") from exc
  if treedef.kind == "none":
    return None
  children = tuple(_tree_unflatten(child, iterator) for child in treedef.children)
  if treedef.kind == "tuple":
    return children
  if treedef.kind == "list":
    return list(children)
  if treedef.kind == "dict":
    return dict(zip(treedef.metadata, children, strict=True))
  raise TypeError(f"unknown pytree node kind: {treedef.kind!r}")


def tree_map(fn: Callable[[object], object], tree: object) -> object:
  leaves, treedef = tree_flatten(tree)
  return tree_unflatten(treedef, (fn(leaf) for leaf in leaves))


def assert_same_structure(expected: PyTreeDef, actual: PyTreeDef, context: str) -> None:
  if expected != actual:
    raise ValueError(f"{context}: pytree structure mismatch")
