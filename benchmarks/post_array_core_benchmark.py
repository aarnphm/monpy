from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

MARKER = "<!-- monpy-array-core-benchmark -->"
API_VERSION = "2026-03-10"


def finite_float(value: object) -> float:
  number = float(value)
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


def string_value(value: object) -> str:
  if isinstance(value, str):
    return value
  if isinstance(value, int | float):
    return str(value)
  raise TypeError(f"expected scalar string-ish value, got {type(value).__name__}")


def result_rows(results: Sequence[Mapping[str, object]]) -> list[tuple[str, ...]]:
  rows = [
    (
      "group",
      "case",
      "monpy us",
      "numpy us",
      "monpy/numpy",
      "monpy range",
      "numpy range",
      "ratio range",
      "rounds",
    )
  ]
  for result in results:
    rows.append(
      (
        string_value(result["group"]),
        string_value(result["name"]),
        format_us(result["monpy_median_us"]),
        format_us(result["numpy_median_us"]),
        format_ratio(result["ratio_median"]),
        format_range(result["monpy_min_us"], result["monpy_max_us"]),
        format_range(result["numpy_min_us"], result["numpy_max_us"]),
        format_range(result["ratio_min"], result["ratio_max"], ratio=True),
        string_value(result["rounds"]),
      )
    )
  return rows


def render_markdown_table(payload: Mapping[str, object]) -> str:
  config = payload["config"]
  if not isinstance(config, Mapping):
    raise TypeError("benchmark json config must be an object")

  raw_results = payload["results"]
  if not isinstance(raw_results, Sequence):
    raise TypeError("benchmark json results must be a list")

  results = []
  for index, result in enumerate(raw_results):
    if not isinstance(result, Mapping):
      raise TypeError(f"benchmark result {index} must be an object")
    results.append(result)

  rows = result_rows(results)
  header = "| " + " | ".join(rows[0]) + " |"
  align = "| " + " | ".join(["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]) + " |"
  body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
  line = (
    f"rounds={config['rounds']} repeats={config['repeats']} "
    f"loops={config['loops']} unit={config['unit']}"
  )
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


def render_comment(payload: Mapping[str, object]) -> str:
  return "\n\n".join(
    [
      MARKER,
      "### array core benchmark",
      render_metadata_table(metadata_rows()),
      "lower `monpy/numpy` is better for monpy.",
      render_markdown_table(payload),
    ]
  )


def request_json(method: str, path: str, *, token: str, body: Mapping[str, object] | None = None) -> object:
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
      "User-Agent": "monpy-array-core-benchmark",
      "X-GitHub-Api-Version": API_VERSION,
    },
  )
  try:
    with urllib.request.urlopen(request, timeout=30) as response:
      raw = response.read()
  except urllib.error.HTTPError as exc:
    detail = exc.read().decode(errors="replace")
    raise RuntimeError(f"github api {method} {path} failed with {exc.code}: {detail}") from exc
  if not raw:
    return None
  return json.loads(raw)


def existing_comment_id(*, repo: str, sha: str, token: str) -> int | None:
  comments = request_json("GET", f"/repos/{repo}/commits/{sha}/comments?per_page=100", token=token)
  if not isinstance(comments, list):
    raise TypeError("github comments response was not a list")
  for comment in comments:
    if not isinstance(comment, Mapping):
      continue
    body = comment.get("body")
    comment_id = comment.get("id")
    if isinstance(body, str) and MARKER in body and isinstance(comment_id, int):
      return comment_id
  return None


def upsert_commit_comment(body: str) -> str:
  repo = os.environ["GITHUB_REPOSITORY"]
  sha = os.environ["GITHUB_SHA"]
  token = os.environ["GITHUB_TOKEN"]
  comment_id = existing_comment_id(repo=repo, sha=sha, token=token)
  if comment_id is None:
    response = request_json("POST", f"/repos/{repo}/commits/{sha}/comments", token=token, body={"body": body})
  else:
    response = request_json("PATCH", f"/repos/{repo}/comments/{comment_id}", token=token, body={"body": body})
  if isinstance(response, Mapping):
    url = response.get("html_url")
    if isinstance(url, str):
      return url
  return ""


def load_payload(path: Path) -> Mapping[str, object]:
  with path.open(encoding="utf-8") as f:
    payload = json.load(f)
  if not isinstance(payload, Mapping):
    raise TypeError("benchmark json root must be an object")
  return payload


def main() -> None:
  parser = argparse.ArgumentParser(description="render and optionally post monpy array-core benchmark results")
  parser.add_argument("results_json", type=Path)
  parser.add_argument("--comment-output", type=Path)
  parser.add_argument("--post", action="store_true")
  args = parser.parse_args()

  payload = load_payload(args.results_json)
  body = render_comment(payload)
  if args.comment_output is not None:
    args.comment_output.write_text(body + "\n", encoding="utf-8")
  if args.post:
    url = upsert_commit_comment(body)
    if url:
      print(url)


if __name__ == "__main__":
  try:
    main()
  except Exception as exc:
    print(f"error: {exc}", file=sys.stderr)
    raise
