from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import pytest


def load_posts_module() -> ModuleType:
  path = Path(__file__).parents[2] / ".github" / "scripts" / "posts.py"
  spec = importlib.util.spec_from_file_location("monpy_benchmark_posts", path)
  if spec is None or spec.loader is None:
    raise RuntimeError("could not load benchmark post helper")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def benchmark_payload() -> dict[str, object]:
  return {
    "config": {
      "suite": "array-core",
      "rounds": 1,
      "repeats": 1,
      "loops": 1,
      "unit": "us",
      "types": ["array"],
    },
    "results": [
      {
        "group": "elementwise",
        "name": "binary_add_f32",
        "monpy_median_us": 2.0,
        "numpy_median_us": 4.0,
        "ratio_median": 0.5,
        "monpy_min_us": 2.0,
        "monpy_max_us": 2.0,
        "numpy_min_us": 4.0,
        "numpy_max_us": 4.0,
        "ratio_min": 0.5,
        "ratio_max": 0.5,
        "rounds": 1,
      }
    ],
  }


def test_platform_comment_marker_and_title_are_rendered() -> None:
  posts = load_posts_module()

  marker = posts.comment_marker("ubuntu")
  body = posts.render_comment(
    benchmark_payload(),
    marker=marker,
    title="monpy benchmark sweep (Ubuntu)",
  )

  assert body.startswith(
    "<!-- monpy-bench:ubuntu -->\n\n### monpy benchmark sweep (Ubuntu)"
  )
  assert "<!-- monpy-bench -->" not in body


def test_existing_comment_id_uses_requested_marker_only(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  posts = load_posts_module()
  comments = [
    {"id": 1, "body": "<!-- monpy-bench:arm -->\n\n### monpy benchmark sweep (ARM)"},
    {
      "id": 2,
      "body": "<!-- monpy-bench:ubuntu -->\n\n### monpy benchmark sweep (Ubuntu)",
    },
  ]

  def fake_request_json(
    method: str,
    path: str,
    *,
    token: str,
    body: Mapping[str, object] | None = None,
  ) -> object:
    return comments

  monkeypatch.setattr(posts, "request_json", fake_request_json)

  comment_id = posts.existing_comment_id(
    repo="aarnphm/monpy",
    sha="abc123",
    token="token",
    markers=(posts.comment_marker("ubuntu"),),
  )

  assert comment_id == 2


def test_existing_comment_id_prefers_platform_marker_over_legacy(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  posts = load_posts_module()
  comments = [
    {"id": 1, "body": "<!-- monpy-bench -->\n\n### monpy benchmark sweep"},
    {"id": 2, "body": "<!-- monpy-bench:arm -->\n\n### monpy benchmark sweep (ARM)"},
  ]

  def fake_request_json(
    method: str,
    path: str,
    *,
    token: str,
    body: Mapping[str, object] | None = None,
  ) -> object:
    return comments

  monkeypatch.setattr(posts, "request_json", fake_request_json)

  comment_id = posts.existing_comment_id(
    repo="aarnphm/monpy",
    sha="abc123",
    token="token",
    markers=(posts.comment_marker("arm"), posts.MARKER),
  )

  assert comment_id == 2


def test_existing_comment_id_can_claim_legacy_marker(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  posts = load_posts_module()
  comments = [
    {"id": 1, "body": "<!-- monpy-bench -->\n\n### monpy benchmark sweep"},
  ]

  def fake_request_json(
    method: str,
    path: str,
    *,
    token: str,
    body: Mapping[str, object] | None = None,
  ) -> object:
    return comments

  monkeypatch.setattr(posts, "request_json", fake_request_json)

  comment_id = posts.existing_comment_id(
    repo="aarnphm/monpy",
    sha="abc123",
    token="token",
    markers=(posts.comment_marker("arm"), posts.MARKER, *posts.LEGACY_MARKERS),
  )

  assert comment_id == 1
