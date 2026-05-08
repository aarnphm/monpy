from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

MARKER = "<!-- monpy-bench -->"
API_VERSION = "2026-03-10"
COMMENT_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def finite_float(value: object) -> float:
  number = float(cast(Any, value))
  if math.isnan(number):
    raise ValueError("benchmark json contains nan")
  return number


def format_us(value: object) -> str:
  number = finite_float(value)
  if math.isinf(number):
    return "inf"
  return f"{number:.3f}"


def format_ratio(value: object) -> str:
  number = finite_float(value)
  if math.isinf(number):
    return "inf"
  return f"{number:.3f}x"


def format_range(min_value: object, max_value: object, *, ratio: bool = False) -> str:
  formatter = format_ratio if ratio else format_us
  return f"{formatter(min_value)}..{formatter(max_value)}"


def winner(value: object) -> str:
  ratio = finite_float(value)
  if math.isinf(ratio):
    return "numpy"
  if ratio < 0.995:
    return "monpy"
  if ratio > 1.005:
    return "numpy"
  return "tie"


def string_value(value: object) -> str:
  if isinstance(value, str):
    return value
  if isinstance(value, int | float):
    return str(value)
  raise TypeError(f"expected scalar string-ish value, got {type(value).__name__}")


def comment_marker(comment_key: str | None) -> str:
  if comment_key is None:
    return MARKER
  normalized = comment_key.strip().lower()
  if not COMMENT_KEY_PATTERN.fullmatch(normalized):
    raise ValueError(
      "comment key must use lowercase letters, digits, '.', '_', or '-' "
      "and start with a letter or digit"
    )
  return f"<!-- monpy-bench:{normalized} -->"


def result_rows(results: Sequence[Mapping[str, object]]) -> list[tuple[str, ...]]:
  rows: list[tuple[str, ...]] = [
    (
      "group",
      "case",
      "monpy us",
      "numpy us",
      "monpy/numpy",
      "winner",
      "monpy range",
      "numpy range",
      "ratio range",
      "rounds",
    )
  ]
  for result in results:
    rows.append((
      string_value(result["group"]),
      string_value(result["name"]),
      format_us(result["monpy_median_us"]),
      format_us(result["numpy_median_us"]),
      format_ratio(result["ratio_median"]),
      winner(result["ratio_median"]),
      format_range(result["monpy_min_us"], result["monpy_max_us"]),
      format_range(result["numpy_min_us"], result["numpy_max_us"]),
      format_range(result["ratio_min"], result["ratio_max"], ratio=True),
      string_value(result["rounds"]),
    ))
  return rows


def render_markdown_table(payload: Mapping[str, object]) -> str:
  config = payload["config"]
  if not isinstance(config, Mapping):
    raise TypeError("benchmark json config must be an object")
  config = cast(Mapping[str, object], config)

  raw_results = payload["results"]
  if not isinstance(raw_results, Sequence):
    raise TypeError("benchmark json results must be a list")

  results: list[Mapping[str, object]] = []
  for index, result in enumerate(raw_results):
    if not isinstance(result, Mapping):
      raise TypeError(f"benchmark result {index} must be an object")
    results.append(cast(Mapping[str, object], result))

  rows = result_rows(results)
  header = "| " + " | ".join(rows[0]) + " |"
  align = "| " + " | ".join([
    "---",
    "---",
    "---:",
    "---:",
    "---:",
    "---",
    "---:",
    "---:",
    "---:",
    "---:",
  ]) + " |"
  body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
  parts = [
    f"suite={config.get('suite', 'array-core')}",
    f"rounds={config['rounds']}",
    f"repeats={config['repeats']}",
    f"loops={config['loops']}",
    f"unit={config['unit']}",
  ]
  types = config.get("types")
  if isinstance(types, Sequence) and not isinstance(types, str):
    parts.append("types=" + ",".join(string_value(value) for value in types))
  candidate = config.get("candidate")
  baseline = config.get("baseline")
  if candidate is not None and baseline is not None:
    parts.append(f"candidate={string_value(candidate)}")
    parts.append(f"baseline={string_value(baseline)}")
  line = " ".join(parts)
  return "\n".join([line, "", header, align, *body])


def command_output(command: Sequence[str]) -> str:
  try:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
  except (FileNotFoundError, subprocess.CalledProcessError):
    return ""
  return completed.stdout.strip()


def detect_mojo_version() -> str:
  mojo = os.environ.get("MOHAUS_MOJO") or shutil.which("mojo")
  if not mojo:
    return ""
  return command_output([mojo, "--version"])


def workflow_url() -> str:
  server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
  repository = os.environ.get("GITHUB_REPOSITORY", "")
  run_id = os.environ.get("GITHUB_RUN_ID", "")
  if not repository or not run_id:
    return ""
  return f"{server_url}/{repository}/actions/runs/{run_id}"


def metadata_rows() -> list[tuple[str, str]]:
  sha = os.environ.get("GITHUB_SHA", "")
  runner_os = os.environ.get("RUNNER_OS", platform.system())
  runner_arch = os.environ.get("RUNNER_ARCH", platform.machine())
  rows = [
    ("commit", f"`{sha}`" if sha else ""),
    ("runner", f"`{runner_os} {runner_arch}`"),
    ("python", f"`{platform.python_version()}`"),
  ]
  mojo_version = detect_mojo_version()
  if mojo_version:
    rows.append(("mojo", f"`{mojo_version}`"))
  url = workflow_url()
  if url:
    rows.append(("workflow", f"[run {os.environ.get('GITHUB_RUN_NUMBER', '')}]({url})"))
  return [(key, value) for key, value in rows if value]


def render_metadata_table(rows: Sequence[tuple[str, str]]) -> str:
  body = ["| item | value |", "| --- | --- |"]
  body.extend(f"| {key} | {value} |" for key, value in rows)
  return "\n".join(body)


def render_comment(
  payload: Mapping[str, object],
  *,
  marker: str = MARKER,
  title: str = "monpy benchmark sweep",
) -> str:
  return "\n\n".join([
    marker,
    f"### {title}",
    render_metadata_table(metadata_rows()),
    (
      "lower `monpy/numpy` is better for monpy; values below `1.000x` mean "
      "monpy beat numpy for that case."
    ),
    render_markdown_table(payload),
  ])


def request_json(
  method: str,
  path: str,
  *,
  token: str,
  body: Mapping[str, object] | None = None,
) -> object:
  api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
  data = None if body is None else json.dumps(body).encode()
  request = urllib.request.Request(
    f"{api_url}{path}",
    data=data,
    method=method,
    headers={
      "Accept": "application/vnd.github+json",
      "Authorization": f"Bearer {token}",
      "Content-Type": "application/json",
      "User-Agent": "monpy-bench",
      "X-GitHub-Api-Version": API_VERSION,
    },
  )
  try:
    with urllib.request.urlopen(request, timeout=30) as response:
      raw = response.read()
  except urllib.error.HTTPError as exc:
    detail = exc.read().decode(errors="replace")
    message = f"github api {method} {path} failed with {exc.code}: {detail}"
    raise RuntimeError(message) from exc
  if not raw:
    return None
  return json.loads(raw)


def existing_comment_id(*, repo: str, sha: str, token: str, marker: str) -> int | None:
  comments = request_json(
    "GET",
    f"/repos/{repo}/commits/{sha}/comments?per_page=100",
    token=token,
  )
  if not isinstance(comments, list):
    raise TypeError("github comments response was not a list")
  for comment in comments:
    if not isinstance(comment, Mapping):
      continue
    comment = cast(Mapping[str, object], comment)
    body = comment.get("body")
    comment_id = comment.get("id")
    if isinstance(body, str) and marker in body and isinstance(comment_id, int):
      return comment_id
  return None


def upsert_commit_comment(body: str, *, marker: str) -> str:
  repo = os.environ["GITHUB_REPOSITORY"]
  sha = os.environ["GITHUB_SHA"]
  token = os.environ["GITHUB_TOKEN"]
  comment_id = existing_comment_id(repo=repo, sha=sha, token=token, marker=marker)
  if comment_id is None:
    response = request_json(
      "POST",
      f"/repos/{repo}/commits/{sha}/comments",
      token=token,
      body={"body": body},
    )
  else:
    response = request_json(
      "PATCH",
      f"/repos/{repo}/comments/{comment_id}",
      token=token,
      body={"body": body},
    )
  if isinstance(response, Mapping):
    response = cast(Mapping[str, object], response)
    url = response.get("html_url")
    if isinstance(url, str):
      return url
  return ""


def load_payload(path: Path) -> Mapping[str, object]:
  with path.open(encoding="utf-8") as f:
    payload = json.load(f)
  if not isinstance(payload, Mapping):
    raise TypeError("benchmark json root must be an object")
  payload = cast(Mapping[str, object], payload)
  if payload.get("kind") == "monpy-bench-manifest":
    return load_payload_from_manifest(path, payload)
  return payload


def load_payload_from_manifest(
  path: Path,
  manifest: Mapping[str, object],
) -> Mapping[str, object]:
  outputs = manifest.get("outputs")
  if not isinstance(outputs, Mapping):
    raise TypeError("benchmark manifest outputs must be an object")
  outputs = cast(Mapping[str, object], outputs)
  results = outputs.get("results")
  if not isinstance(results, Mapping):
    raise TypeError("benchmark manifest outputs.results must be an object")
  results = cast(Mapping[str, object], results)
  result_format = results.get("format")
  if result_format != "json":
    raise TypeError("benchmark manifest must point at json results")
  raw_result_path = results.get("path")
  if not isinstance(raw_result_path, str):
    raise TypeError("benchmark manifest result path must be a string")

  result_path = Path(raw_result_path)
  if result_path.is_absolute():
    candidates = [result_path]
  else:
    candidates = [
      Path.cwd() / result_path,
      path.parent / result_path.name,
    ]
  for candidate in candidates:
    if candidate.exists():
      return load_payload(candidate)
  searched = ", ".join(str(candidate) for candidate in candidates)
  raise FileNotFoundError(f"benchmark results json not found; searched {searched}")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="render and optionally post monpy benchmark sweep results"
  )
  parser.add_argument(
    "results_json",
    type=Path,
    help="benchmark results json or manifest.json",
  )
  parser.add_argument("--comment-output", type=Path)
  parser.add_argument(
    "--comment-key",
    help="stable key for a platform-specific commit comment, for example arm or ubuntu",
  )
  parser.add_argument("--comment-title", default="monpy benchmark sweep")
  parser.add_argument("--post", action="store_true")
  args = parser.parse_args()

  payload = load_payload(args.results_json)
  marker = comment_marker(args.comment_key)
  body = render_comment(payload, marker=marker, title=args.comment_title)
  if args.comment_output is not None:
    args.comment_output.write_text(body + "\n", encoding="utf-8")
  if args.post:
    url = upsert_commit_comment(body, marker=marker)
    if url:
      print(url)


if __name__ == "__main__":
  try:
    main()
  except Exception as exc:
    print(f"error: {exc}", file=sys.stderr)
    raise
