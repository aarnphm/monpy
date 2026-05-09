# fmt: off
# ruff: noqa
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from monpy._bench.core import BenchCase, BenchResult
from monpy._bench.core import (
  parse_sizes,
  positive_int,
  render_csv,
  render_json,
  render_markdown,
  render_table,
)
from monpy._bench.core import run_benchmarks, sorted_results

SUITE_TYPES = ("array", "strides", "complex", "attention")


def local_now() -> datetime:
  return datetime.now().astimezone()


def default_output_dir(now: datetime | None = None) -> Path:
  run_date = (now or local_now()).date().isoformat()
  return Path("results") / run_date


def parse_types(value: str) -> tuple[str, ...]:
  raw_types = tuple(part.strip() for part in value.split(",") if part.strip())
  if not raw_types:
    raise argparse.ArgumentTypeError("must include at least one benchmark type")
  if "all" in raw_types:
    return SUITE_TYPES
  unknown = sorted(set(raw_types) - set(SUITE_TYPES))
  if unknown:
    raise argparse.ArgumentTypeError(f"unknown benchmark type(s): {', '.join(unknown)}")
  return tuple(dict.fromkeys(raw_types))


def package_version(name: str) -> str:
  try:
    return importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:
    return ""


def command_output(command: Sequence[str]) -> str:
  try:
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
  except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
    return ""
  return completed.stdout.strip()


def mojo_metadata() -> dict[str, str]:
  mojo = os.environ.get("MOHAUS_MOJO") or shutil.which("mojo")
  if not mojo:
    return {}
  metadata = {"path": mojo}
  version = command_output([mojo, "--version"])
  if version:
    metadata["version"] = version
  return metadata


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def display_path(path: Path) -> str:
  resolved = path.resolve()
  try:
    return str(resolved.relative_to(Path.cwd().resolve()))
  except ValueError:
    return str(resolved)


def build_cases(
  *,
  types: Sequence[str],
  vector_size: int,
  vector_sizes: Sequence[int],
  matrix_sizes: Sequence[int],
  linalg_sizes: Sequence[int],
) -> list[BenchCase]:
  cases: list[BenchCase] = []
  for suite_type in types:
    module = importlib.import_module(f"monpy._bench.types.{suite_type}")
    cases.extend(
      module.build_cases(
        vector_size=vector_size,
        vector_sizes=vector_sizes,
        matrix_sizes=matrix_sizes,
        linalg_sizes=linalg_sizes,
      )
    )
  return cases


def sweep_config(args: argparse.Namespace) -> dict[str, object]:
  return {
    "suite": "sweep",
    "types": list(args.types),
    "candidate": "monpy",
    "baseline": "numpy",
    "comparison": "monpy_us / numpy_us",
    "vector_size": args.vector_size,
    "vector_sizes": list(args.vector_sizes),
    "matrix_sizes": list(args.matrix_sizes),
    "linalg_sizes": list(args.linalg_sizes),
  }


def render_sweep_json(results: Sequence[BenchResult], *, args: argparse.Namespace) -> str:
  payload = json.loads(render_json(results, rounds=args.rounds, loops=args.loops, repeats=args.repeats))
  payload["config"].update(sweep_config(args))
  return json.dumps(payload, indent=2, sort_keys=True)


def render_sweep_markdown(results: Sequence[BenchResult], *, args: argparse.Namespace) -> str:
  header = (
    f"suite=sweep types={','.join(args.types)} candidate=monpy baseline=numpy "
    "comparison=monpy_us/numpy_us"
  )
  return "\n\n".join([
    header,
    render_markdown(results, rounds=args.rounds, loops=args.loops, repeats=args.repeats),
  ])


def render_results(results: Sequence[BenchResult], *, args: argparse.Namespace) -> str:
  ordered = sorted_results(results, sort=args.sort)
  if args.format == "csv":
    return render_csv(ordered)
  if args.format == "json":
    return render_sweep_json(ordered, args=args)
  if args.format == "markdown":
    return render_sweep_markdown(ordered, args=args)
  return render_table(ordered, rounds=args.rounds, loops=args.loops, repeats=args.repeats)


def case_manifest(case: BenchCase) -> dict[str, object]:
  return {
    "group": case.group,
    "name": case.name,
    "check_values": case.check_values,
    "check_dtype": case.check_dtype,
    "rtol": case.rtol,
    "atol": case.atol,
  }


def group_counts(cases: Sequence[BenchCase]) -> dict[str, int]:
  counts: dict[str, int] = {}
  for case in cases:
    counts[case.group] = counts.get(case.group, 0) + 1
  return dict(sorted(counts.items()))


def build_manifest(
  *,
  args: argparse.Namespace,
  cases: Sequence[BenchCase],
  results: Sequence[BenchResult],
  results_path: Path,
  manifest_path: Path,
  started_at: datetime,
  finished_at: datetime,
  invocation: Sequence[str],
) -> dict[str, object]:
  return {
    "schema_version": 1,
    "kind": "monpy-bench-manifest",
    "created_at": finished_at.isoformat(),
    "started_at": started_at.isoformat(),
    "finished_at": finished_at.isoformat(),
    "duration_s": (finished_at - started_at).total_seconds(),
    "commands": {
      "argv": list(invocation),
      "shell": shlex.join(invocation),
      "cwd": str(Path.cwd()),
    },
    "environment": {
      "platform": platform.platform(),
      "machine": platform.machine(),
      "python": platform.python_version(),
      "monpy": package_version("monpy"),
      "numpy": package_version("numpy"),
      "mojo": mojo_metadata(),
      "env": {
        "MOHAUS_MOJO": os.environ.get("MOHAUS_MOJO", ""),
      },
    },
    "parameters": {
      "rounds": args.rounds,
      "loops": args.loops,
      "repeats": args.repeats,
      "format": args.format,
      "sort": args.sort,
      "progress": args.progress,
      "output_dir": display_path(args.output_dir),
      "save": args.save,
      "stdout": args.stdout,
      **sweep_config(args),
    },
    "matrices": {
      "matrix_sizes": list(args.matrix_sizes),
      "linalg_sizes": list(args.linalg_sizes),
    },
    "vectors": {
      "vector_size": args.vector_size,
      "vector_sizes": list(args.vector_sizes),
    },
    "cases": {
      "count": len(cases),
      "groups": group_counts(cases),
      "items": [case_manifest(case) for case in cases],
    },
    "results": {
      "count": len(results),
    },
    "outputs": {
      "manifest": {
        "path": display_path(manifest_path),
      },
      "results": {
        "path": display_path(results_path),
        "format": args.format,
        "bytes": results_path.stat().st_size,
        "sha256": sha256_file(results_path),
      },
    },
  }


def write_run_outputs(
  rendered: str,
  *,
  args: argparse.Namespace,
  cases: Sequence[BenchCase],
  results: Sequence[BenchResult],
  started_at: datetime,
  finished_at: datetime,
  invocation: Sequence[str],
) -> tuple[Path, Path]:
  output_dir = args.output_dir
  output_dir.mkdir(parents=True, exist_ok=True)
  results_path = output_dir / f"results.{args.format}"
  manifest_path = output_dir / "manifest.json"
  results_path.write_text(rendered.rstrip() + "\n", encoding="utf-8")
  manifest = build_manifest(
    args=args,
    cases=cases,
    results=results,
    results_path=results_path,
    manifest_path=manifest_path,
    started_at=started_at,
    finished_at=finished_at,
    invocation=invocation,
  )
  manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return manifest_path, results_path


def main(argv: Sequence[str] | None = None) -> None:
  parser = argparse.ArgumentParser(
    description="benchmark monpy against numpy across typed benchmark suites"
  )
  parser.add_argument(
    "--loops",
    type=positive_int,
    default=200,
    help="inner calls per timing sample",
  )
  parser.add_argument(
    "--repeats",
    type=positive_int,
    default=5,
    help="timing samples per case per round",
  )
  parser.add_argument(
    "--rounds",
    type=positive_int,
    default=3,
    help="full benchmark passes to aggregate",
  )
  parser.add_argument(
    "--types",
    type=parse_types,
    default=parse_types("array,strides,complex"),
    help="comma-separated benchmark types: array,strides,complex,attention,all",
  )
  parser.add_argument(
    "--vector-size",
    type=positive_int,
    default=1024,
    help="size for wrapper-bound elementwise cases (default 1024)",
  )
  parser.add_argument(
    "--vector-sizes",
    type=parse_sizes,
    default=parse_sizes("16384,262144,1048576"),
    help="sizes for bandwidth-regime cases (default 16K,256K,1M)",
  )
  parser.add_argument(
    "--matrix-sizes",
    type=parse_sizes,
    default=parse_sizes("16,64,128,256"),
    help="square matrix sizes for matmul/matvec/vecmat (default 16,64,128,256)",
  )
  parser.add_argument(
    "--linalg-sizes",
    type=parse_sizes,
    default=parse_sizes("2,4,8,32,128"),
    help="square matrix sizes for solve/inv/det (default 2,4,8,32,128)",
  )
  parser.add_argument("--format", choices=("table", "csv", "json", "markdown"), default="table")
  parser.add_argument(
    "--sort",
    choices=("input", "name", "fastest", "slowest", "monpy", "ratio"),
    default="input",
    help=(
      "row order. 'fastest' = monpy/numpy ascending (cases where monpy beats numpy "
      "the most on top). 'slowest' = monpy/numpy descending (regressions on top). "
      "'ratio' is a back-compat alias for 'slowest'. 'monpy' sorts by absolute "
      "monpy timing descending (largest cells first)."
    ),
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="directory for saved outputs (default results/YYYY-MM-DD)",
  )
  parser.add_argument(
    "--save",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="write results.<format> and manifest.json to --output-dir",
  )
  parser.add_argument(
    "--stdout",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="write rendered benchmark results to stdout",
  )
  parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
  args = parser.parse_args(argv)
  if args.output_dir is None:
    args.output_dir = default_output_dir()
  if not args.save and not args.stdout:
    parser.error("at least one of --save or --stdout must be enabled")
  invocation = sys.argv[:] if argv is None else [sys.argv[0], *argv]
  started_at = local_now()

  cases = build_cases(
    types=args.types,
    vector_size=args.vector_size,
    vector_sizes=args.vector_sizes,
    matrix_sizes=args.matrix_sizes,
    linalg_sizes=args.linalg_sizes,
  )
  results = run_benchmarks(
    cases,
    rounds=args.rounds,
    loops=args.loops,
    repeats=args.repeats,
    progress=args.progress,
  )
  rendered = render_results(results, args=args)
  finished_at = local_now()
  if args.save:
    manifest_path, results_path = write_run_outputs(
      rendered,
      args=args,
      cases=cases,
      results=results,
      started_at=started_at,
      finished_at=finished_at,
      invocation=invocation,
    )
    print(f"wrote benchmark results to {display_path(results_path)}", file=sys.stderr)
    print(f"wrote benchmark manifest to {display_path(manifest_path)}", file=sys.stderr)
  if args.stdout:
    print(rendered)


if __name__ == "__main__":
  raise SystemExit(main())
