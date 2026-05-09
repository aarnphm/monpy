from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import pytest


def benchmark_result(
  *,
  name: str,
  monpy_median_us: float,
  numpy_median_us: float,
  ratio_median: float,
) -> dict[str, object]:
  return {
    "group": "elementwise",
    "name": name,
    "monpy_median_us": monpy_median_us,
    "numpy_median_us": numpy_median_us,
    "ratio_median": ratio_median,
    "monpy_min_us": monpy_median_us,
    "monpy_max_us": monpy_median_us,
    "numpy_min_us": numpy_median_us,
    "numpy_max_us": numpy_median_us,
    "ratio_min": ratio_median,
    "ratio_max": ratio_median,
    "rounds": 1,
  }


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
      benchmark_result(
        name="binary_add_f32",
        monpy_median_us=2.0,
        numpy_median_us=4.0,
        ratio_median=0.5,
      )
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
    marker=posts.comment_marker("ubuntu"),
  )

  assert comment_id == 2


def test_comment_highlights_overall_result_and_winning_timings() -> None:
  posts = load_posts_module()
  payload = benchmark_payload()
  payload["results"] = [
    benchmark_result(
      name="binary_add_f32",
      monpy_median_us=2.0,
      numpy_median_us=4.0,
      ratio_median=0.5,
    ),
    benchmark_result(
      name="binary_add_f64",
      monpy_median_us=3.0,
      numpy_median_us=6.0,
      ratio_median=0.5,
    ),
    benchmark_result(
      name="array_copy_f32",
      monpy_median_us=5.0,
      numpy_median_us=2.5,
      ratio_median=2.0,
    ),
    benchmark_result(
      name="astype_f32",
      monpy_median_us=1.0,
      numpy_median_us=1.0,
      ratio_median=1.0,
    ),
  ]

  body = posts.render_comment(payload)

  assert (
    "overall result: **monpy** (case outcomes: **monpy=2**, numpy=1, tie=1)"
    in body
  )
  assert "| elementwise | binary_add_f32 | **2.000** | 4.000 | 0.500x | monpy |" in body
  assert "| elementwise | array_copy_f32 | 5.000 | **2.500** | 2.000x | numpy |" in body
  assert "| elementwise | astype_f32 | 1.000 | 1.000 | 1.000x | tie |" in body


def test_overall_draw_highlights_both_tied_counts() -> None:
  posts = load_posts_module()
  payload = benchmark_payload()
  payload["results"] = [
    benchmark_result(
      name="binary_add_f32",
      monpy_median_us=2.0,
      numpy_median_us=4.0,
      ratio_median=0.5,
    ),
    benchmark_result(
      name="array_copy_f32",
      monpy_median_us=5.0,
      numpy_median_us=2.5,
      ratio_median=2.0,
    ),
  ]

  body = posts.render_comment(payload)

  assert (
    "overall result: **tie** (case outcomes: **monpy=1**, **numpy=1**, tie=0)"
    in body
  )
