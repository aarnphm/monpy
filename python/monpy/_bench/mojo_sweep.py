# fmt: off
# ruff: noqa
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from monpy._bench.core import positive_int
from monpy._bench.sweep import (
  default_output_dir,
  display_path,
  local_now,
  package_version,
  sha256_file,
)

TSV_HEADER = (
  "group",
  "name",
  "candidate",
  "baseline",
  "candidate_ns",
  "baseline_ns",
  "ratio",
  "bytes",
  "flops",
)


@dataclass(frozen=True, slots=True)
class MojoBenchRow:
  group: str
  name: str
  candidate: str
  baseline: str
  candidate_ns: float
  baseline_ns: float
  ratio: float
  bytes: int
  flops: int


@dataclass(frozen=True, slots=True)
class MojoBenchCollection:
  rows: list[MojoBenchRow]
  commands: list[list[str]]
  warnings: list[str]


def repo_root_from_package() -> Path:
  return Path(__file__).resolve().parents[3]


def default_mojo_output_dir(now: datetime | None = None) -> Path:
  return default_output_dir(now) / "mojo"


def default_numojo_path(repo_root: Path) -> Path | None:
  env_value = os.environ.get("NUMOJO_PATH")
  if env_value:
    return Path(env_value).expanduser()
  vendor = repo_root / "vendor" / "NuMojo"
  if vendor.exists():
    return vendor
  scratchpad = Path.home() / "workspace" / "scratchpad" / "NuMojo"
  return scratchpad if scratchpad.exists() else None


def find_mojo(explicit: Path | None = None) -> Path:
  if explicit is not None:
    return explicit.expanduser()

  env_value = os.environ.get("MOHAUS_MOJO")
  if env_value:
    return Path(env_value).expanduser()

  derived = os.environ.get("MODULAR_DERIVED_PATH")
  if derived:
    candidate = Path(derived).expanduser() / "build" / "bin" / "mojo"
    if candidate.exists():
      return candidate

  which = shutil.which("mojo")
  if which:
    return Path(which)

  raise FileNotFoundError(
    "could not find mojo; pass --mojo, set MOHAUS_MOJO, or set MODULAR_DERIVED_PATH"
  )


def standard_command(*, mojo: Path, repo_root: Path) -> list[str]:
  return [str(mojo), "run", "-I", str(repo_root / "src"), str(repo_root / "benches" / "bench_mojo_sweep.mojo")]


def numojo_command(*, mojo: Path, repo_root: Path, numojo_path: Path) -> list[str]:
  return [
    str(mojo), "run", "--ignore-incompatible-package-errors", "-I", str(repo_root / "src"), "-I", str(numojo_path),
    str(repo_root / "benches" / "bench_numojo_sweep.mojo"),
  ]


def threading_command(*, mojo: Path, repo_root: Path) -> list[str]:
  return [str(mojo), "run", "-I", str(repo_root / "src"), str(repo_root / "benches" / "bench_threading_sweep.mojo")]


def parse_thread_caps(value: str) -> tuple[str, ...]:
  caps: list[str] = []
  seen: set[str] = set()
  for raw in value.split(","):
    cap = raw.strip().lower()
    if not cap:
      continue
    if cap != "auto":
      try:
        parsed = int(cap)
      except ValueError as exc:
        raise argparse.ArgumentTypeError(
          f"thread cap {raw!r} must be 'auto' or a positive integer"
        ) from exc
      if parsed <= 0:
        raise argparse.ArgumentTypeError(
          f"thread cap {raw!r} must be 'auto' or a positive integer"
        )
      normalized = str(parsed)
    else:
      normalized = "auto"
    if normalized not in seen:
      caps.append(normalized)
      seen.add(normalized)
  if not caps:
    raise argparse.ArgumentTypeError("at least one thread cap is required")
  return tuple(caps)


def parse_mojo_tsv(output: str) -> list[MojoBenchRow]:
  rows: list[MojoBenchRow] = []
  seen_header = False
  for line in output.splitlines():
    stripped = line.strip()
    if not stripped:
      continue
    fields = tuple(stripped.split("\t"))
    if fields == TSV_HEADER:
      seen_header = True
      continue
    if not seen_header:
      continue
    if len(fields) != len(TSV_HEADER):
      raise ValueError(f"malformed Mojo benchmark row: {line!r}")
    rows.append(MojoBenchRow(fields[0], fields[1], fields[2], fields[3], float(fields[4]), float(fields[5]), float(fields[6]), int(fields[7]), int(fields[8])))
  if not seen_header:
    raise ValueError("Mojo benchmark output did not include the TSV header")
  return rows


def run_mojo_command(
  command: Sequence[str],
  *,
  timeout: int,
  env_overrides: dict[str, str] | None = None,
  unset_env: Sequence[str] = (),
) -> str:
  env = mojo_subprocess_env(Path(command[0]))
  for name in unset_env:
    env.pop(name, None)
  if env_overrides is not None:
    env.update(env_overrides)
  try:
    completed = subprocess.run(
      command,
      check=True,
      capture_output=True,
      text=True,
      timeout=timeout,
      env=env,
    )
  except FileNotFoundError as exc:
    raise RuntimeError(f"could not execute {command[0]!r}") from exc
  except subprocess.TimeoutExpired as exc:
    raise RuntimeError(f"command timed out after {timeout}s: {shlex.join(command)}") from exc
  except subprocess.CalledProcessError as exc:
    stderr = exc.stderr.strip()
    stdout = exc.stdout.strip()
    details = "\n".join(part for part in (stdout, stderr) if part)
    raise RuntimeError(f"command failed: {shlex.join(command)}\n{details}") from exc
  return completed.stdout


def summarize_failure(exc: BaseException, *, max_lines: int = 3) -> str:
  lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
  if not lines:
    return exc.__class__.__name__
  diagnostic_lines = [line for line in lines if "error:" in line or "warning:" in line]
  selected = diagnostic_lines[:max_lines] or lines[:max_lines]
  suffix = "" if len(selected) == len(diagnostic_lines or lines) else " ..."
  return "; ".join(selected) + suffix


def prepend_env_path(env: dict[str, str], name: str, path: Path) -> None:
  current = env.get(name, "")
  env[name] = str(path) if not current else f"{path}{os.pathsep}{current}"


def mojo_subprocess_env(mojo: Path) -> dict[str, str]:
  env = os.environ.copy()
  venv_root = mojo.expanduser().resolve().parent.parent
  site_packages = venv_root / "lib"
  if not site_packages.exists():
    return env

  modular_libs = sorted(site_packages.glob("python*/site-packages/modular/lib"))
  if not modular_libs:
    return env

  modular_lib = modular_libs[0]
  prepend_env_path(env, "DYLD_LIBRARY_PATH", modular_lib)
  prepend_env_path(env, "LD_LIBRARY_PATH", modular_lib)
  return env


def mojo_binary_metadata(mojo: Path) -> dict[str, str]:
  metadata = {"path": str(mojo)}
  try:
    completed = subprocess.run(
      [str(mojo), "--version"],
      check=True,
      capture_output=True,
      text=True,
      timeout=5,
      env=mojo_subprocess_env(mojo),
    )
  except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
    return metadata
  version = completed.stdout.strip()
  if version:
    metadata["version"] = version
  return metadata


def sorted_rows(rows: Sequence[MojoBenchRow], *, sort: str) -> list[MojoBenchRow]:
  if sort == "name":
    return sorted(rows, key=lambda row: (row.name, row.group, row.candidate, row.baseline))
  if sort == "candidate":
    return sorted(rows, key=lambda row: row.candidate_ns, reverse=True)
  if sort == "fastest":
    # Smallest candidate/baseline first: cases where the candidate beats the
    # baseline by the largest margin.
    return sorted(rows, key=lambda row: row.ratio)
  if sort in ("slowest", "ratio"):
    return sorted(rows, key=lambda row: row.ratio, reverse=True)
  return list(rows)


def format_ns(value: float) -> str:
  if value >= 1000.0:
    return f"{value / 1000.0:.3f} us"
  return f"{value:.3f} ns"


def format_ratio(value: float) -> str:
  return f"{value:.3f}x"


def table_rows(rows: Sequence[MojoBenchRow]) -> list[tuple[str, ...]]:
  rendered: list[tuple[str, ...]] = [
    (
      "group",
      "case",
      "candidate",
      "baseline",
      "candidate ns",
      "baseline ns",
      "ratio",
      "bytes",
      "flops",
    )
  ]
  for row in rows:
    rendered.append((
      row.group,
      row.name,
      row.candidate,
      row.baseline,
      format_ns(row.candidate_ns),
      format_ns(row.baseline_ns),
      format_ratio(row.ratio),
      str(row.bytes),
      str(row.flops),
    ))
  return rendered


def render_table(rows: Sequence[MojoBenchRow]) -> str:
  table = table_rows(rows)
  widths = [max(len(row[index]) for row in table) for index in range(len(table[0]))]
  numeric_columns = {4, 5, 6, 7, 8}

  def render_row(row: Sequence[str]) -> str:
    cells = [cell.rjust(widths[index]) if index in numeric_columns else cell.ljust(widths[index]) for index, cell in enumerate(row)]
    return " | ".join(cells)

  separator = "-+-".join("-" * width for width in widths)
  return "\n".join([
    "unit=ns/call comparison=candidate_ns/baseline_ns",
    render_row(table[0]),
    separator,
    *[render_row(row) for row in table[1:]],
  ])


def row_record(row: MojoBenchRow) -> dict[str, object]:
  return asdict(row)


def render_csv(rows: Sequence[MojoBenchRow]) -> str:
  output = io.StringIO()
  writer = csv.DictWriter(output, fieldnames=TSV_HEADER, lineterminator="\n")
  writer.writeheader()
  writer.writerows(row_record(row) for row in rows)
  return output.getvalue().rstrip()


def render_json(rows: Sequence[MojoBenchRow], *, args: argparse.Namespace) -> str:
  payload = {
    "config": mojo_sweep_config(args),
    "results": [row_record(row) for row in rows],
  }
  return json.dumps(payload, indent=2, sort_keys=True)


def render_markdown(rows: Sequence[MojoBenchRow], *, args: argparse.Namespace) -> str:
  table = table_rows(rows)
  header = "| " + " | ".join(table[0]) + " |"
  align = "| " + " | ".join(["---", "---", "---", "---", "---:", "---:", "---:", "---:", "---:"]) + " |"
  body = ["| " + " | ".join(row) + " |" for row in table[1:]]
  config = mojo_sweep_config(args)
  summary = (
    f"suite={config['suite']} include_numojo={config['include_numojo']} "
    "unit=ns/call comparison=candidate_ns/baseline_ns"
  )
  return "\n".join([summary, "", header, align, *body])


def render_results(rows: Sequence[MojoBenchRow], *, args: argparse.Namespace) -> str:
  ordered = sorted_rows(rows, sort=args.sort)
  if args.format == "csv":
    return render_csv(ordered)
  if args.format == "json":
    return render_json(ordered, args=args)
  if args.format == "markdown":
    return render_markdown(ordered, args=args)
  return render_table(ordered)


def mojo_sweep_config(args: argparse.Namespace) -> dict[str, object]:
  numojo_path = "" if args.numojo_path is None else display_path(args.numojo_path)
  return {
    "suite": "mojo-kernel-sweep",
    "unit": "ns/call",
    "candidate": "row.candidate",
    "baseline": "row.baseline",
    "comparison": "candidate_ns / baseline_ns",
    "include_numojo": args.include_numojo,
    "include_threading": args.include_threading,
    "numojo_path": numojo_path,
    "strict_numojo": getattr(args, "strict_numojo", False),
    "thread_caps": list(args.thread_caps),
    "sort": args.sort,
    "format": args.format,
  }


def build_manifest(
  *,
  args: argparse.Namespace,
  rows: Sequence[MojoBenchRow],
  results_path: Path,
  manifest_path: Path,
  started_at: datetime,
  finished_at: datetime,
  invocation: Sequence[str],
  commands: Sequence[Sequence[str]],
  warnings: Sequence[str],
  mojo: Path,
) -> dict[str, object]:
  return {
    "schema_version": 1,
    "kind": "monpy-mojo-bench-manifest",
    "created_at": finished_at.isoformat(),
    "started_at": started_at.isoformat(),
    "finished_at": finished_at.isoformat(),
    "duration_s": (finished_at - started_at).total_seconds(),
    "commands": {
      "argv": list(invocation),
      "shell": shlex.join(invocation),
      "cwd": str(Path.cwd()),
      "mojo": [
        {
          "argv": list(command),
          "shell": shlex.join(command),
        }
        for command in commands
      ],
    },
    "environment": {
      "platform": platform.platform(),
      "machine": platform.machine(),
      "python": platform.python_version(),
      "monpy": package_version("monpy"),
      "mojo": mojo_binary_metadata(mojo),
      "env": {
        "MOHAUS_MOJO": os.environ.get("MOHAUS_MOJO", ""),
        "MODULAR_DERIVED_PATH": os.environ.get("MODULAR_DERIVED_PATH", ""),
        "NUMOJO_PATH": os.environ.get("NUMOJO_PATH", ""),
      },
    },
    "parameters": {
      "repo_root": display_path(args.repo_root),
      "output_dir": display_path(args.output_dir),
      "save": args.save,
      "stdout": args.stdout,
      "timeout": args.timeout,
      **mojo_sweep_config(args),
    },
    "rows": {
      "count": len(rows),
      "groups": group_counts(rows),
      "items": [row_record(row) for row in rows],
    },
    "warnings": list(warnings),
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


def group_counts(rows: Sequence[MojoBenchRow]) -> dict[str, int]:
  counts: dict[str, int] = {}
  for row in rows:
    counts[row.group] = counts.get(row.group, 0) + 1
  return dict(sorted(counts.items()))


def write_run_outputs(
  rendered: str,
  *,
  args: argparse.Namespace,
  rows: Sequence[MojoBenchRow],
  started_at: datetime,
  finished_at: datetime,
  invocation: Sequence[str],
  commands: Sequence[Sequence[str]],
  warnings: Sequence[str],
  mojo: Path,
) -> tuple[Path, Path]:
  output_dir = args.output_dir
  output_dir.mkdir(parents=True, exist_ok=True)
  results_path = output_dir / f"results.{args.format}"
  manifest_path = output_dir / "manifest.json"
  results_path.write_text(rendered.rstrip() + "\n", encoding="utf-8")
  manifest = build_manifest(
    args=args,
    rows=rows,
    results_path=results_path,
    manifest_path=manifest_path,
    started_at=started_at,
    finished_at=finished_at,
    invocation=invocation,
    commands=commands,
    warnings=warnings,
    mojo=mojo,
  )
  manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return manifest_path, results_path


def collect_rows(
  *,
  mojo: Path,
  repo_root: Path,
  include_numojo: bool,
  numojo_path: Path | None,
  strict_numojo: bool,
  include_threading: bool = False,
  thread_caps: Sequence[str] = (),
  timeout: int,
) -> MojoBenchCollection:
  commands: list[list[str]] = []
  warnings: list[str] = []
  std_command = standard_command(mojo=mojo, repo_root=repo_root)
  commands.append(std_command)
  rows = parse_mojo_tsv(run_mojo_command(std_command, timeout=timeout))

  if include_numojo:
    if numojo_path is None:
      raise RuntimeError("NuMojo comparison requested but --numojo-path was not provided")
    nm_command = numojo_command(mojo=mojo, repo_root=repo_root, numojo_path=numojo_path)
    commands.append(nm_command)
    try:
      rows.extend(parse_mojo_tsv(run_mojo_command(nm_command, timeout=timeout)))
    except (RuntimeError, ValueError) as exc:
      if strict_numojo:
        raise
      warnings.append(
        "skipping NuMojo comparison because the selected checkout did not run with "
        f"this Mojo toolchain: {summarize_failure(exc)}"
      )

  if include_threading:
    th_command = threading_command(mojo=mojo, repo_root=repo_root)
    for cap in thread_caps:
      if cap == "auto":
        commands.append(["env", "-u", "MONPY_THREADS", *th_command])
        rows.extend(
          parse_mojo_tsv(
            run_mojo_command(th_command, timeout=timeout, unset_env=("MONPY_THREADS",))
          )
        )
      else:
        commands.append(["env", f"MONPY_THREADS={cap}", *th_command])
        rows.extend(
          parse_mojo_tsv(
            run_mojo_command(th_command, timeout=timeout, env_overrides={"MONPY_THREADS": cap})
          )
        )

  return MojoBenchCollection(rows=rows, commands=commands, warnings=warnings)


def main(argv: Sequence[str] | None = None) -> None:
  parser = argparse.ArgumentParser(
    description="benchmark monpy Mojo kernels against Mojo stdlib, NuMojo, and threading baselines"
  )
  parser.add_argument("--mojo", type=Path, default=None, help="path to the mojo executable")
  parser.add_argument(
    "--repo-root",
    type=Path,
    default=repo_root_from_package(),
    help="monpy checkout root (default: inferred from package path)",
  )
  parser.add_argument(
    "--include-numojo",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="also run benches/bench_numojo_sweep.mojo against a NuMojo checkout",
  )
  parser.add_argument(
    "--numojo-path",
    type=Path,
    default=None,
    help=(
      "path to a NuMojo checkout or package root "
      "(default: NUMOJO_PATH, vendor/NuMojo, or ~/workspace/scratchpad/NuMojo)"
    ),
  )
  parser.add_argument(
    "--strict-numojo",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="fail instead of warning when the optional NuMojo comparison cannot compile or run",
  )
  parser.add_argument(
    "--include-threading",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
      "also run benches/bench_threading_sweep.mojo, comparing internal threaded "
      "static kernels against monpy serial static kernels"
    ),
  )
  parser.add_argument(
    "--thread-caps",
    type=parse_thread_caps,
    default=("auto", "1"),
    help=(
      "comma-separated MONPY_THREADS caps for --include-threading "
      "(default: auto,1; use e.g. auto,1,2,4,8)"
    ),
  )
  parser.add_argument("--timeout", type=positive_int, default=300, help="seconds per Mojo command")
  parser.add_argument("--format", choices=("table", "csv", "json", "markdown"), default="table")
  parser.add_argument(
    "--sort",
    choices=("input", "name", "fastest", "slowest", "candidate", "ratio"),
    default="input",
    help=(
      "row order. 'fastest' = candidate/baseline ascending (cases where the "
      "candidate beats the baseline the most on top). 'slowest' = descending. "
      "'ratio' is a back-compat alias for 'slowest'. 'candidate' sorts by "
      "absolute candidate timing descending."
    ),
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="directory for saved outputs (default results/YYYY-MM-DD/mojo)",
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
  args = parser.parse_args(argv)
  if args.output_dir is None:
    args.output_dir = default_mojo_output_dir()
  args.repo_root = args.repo_root.expanduser().resolve()
  if args.numojo_path is None:
    args.numojo_path = default_numojo_path(args.repo_root)
  if args.numojo_path is not None:
    args.numojo_path = args.numojo_path.expanduser().resolve()
  if args.include_numojo and args.numojo_path is None:
    parser.error("--include-numojo requires --numojo-path or NUMOJO_PATH")
  if not args.save and not args.stdout:
    parser.error("at least one of --save or --stdout must be enabled")

  invocation = sys.argv[:] if argv is None else [sys.argv[0], *argv]
  try:
    mojo = find_mojo(args.mojo)
    started_at = local_now()
    collection = collect_rows(
      mojo=mojo,
      repo_root=args.repo_root,
      include_numojo=args.include_numojo,
      numojo_path=args.numojo_path,
      strict_numojo=args.strict_numojo,
      include_threading=args.include_threading,
      thread_caps=args.thread_caps,
      timeout=args.timeout,
    )
    for warning in collection.warnings:
      print(warning, file=sys.stderr)
    rows = collection.rows
    rendered = render_results(rows, args=args)
    finished_at = local_now()
    if args.save:
      manifest_path, results_path = write_run_outputs(
        rendered,
        args=args,
        rows=rows,
        started_at=started_at,
        finished_at=finished_at,
        invocation=invocation,
        commands=collection.commands,
        warnings=collection.warnings,
        mojo=mojo,
      )
      print(f"wrote Mojo benchmark results to {display_path(results_path)}", file=sys.stderr)
      print(f"wrote Mojo benchmark manifest to {display_path(manifest_path)}", file=sys.stderr)
    if args.stdout:
      print(rendered)
  except (FileNotFoundError, RuntimeError, ValueError) as exc:
    parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
  raise SystemExit(main())
