# fmt: off
# ruff: noqa
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from monpy._bench.core import (
  BenchCase,
  call_bench_fn,
  force_monpy,
  parse_sizes,
  positive_int,
  verify_same,
  verify_shape_dtype,
)
from monpy._bench.sweep import build_cases, display_path, local_now, parse_types

DEFAULT_PERF_EVENTS = "cycles,instructions,cache-references,cache-misses,branches,branch-misses"


def positive_float(value: str) -> float:
  parsed = float(value)
  if parsed <= 0.0: raise argparse.ArgumentTypeError("must be > 0")
  return parsed


def default_profile_dir(now: datetime | None = None) -> Path:
  stamp = (now or local_now()).strftime("%Y-%m-%d-%H%M%S")
  return Path("results") / f"profile-{stamp}"


def case_id(case: BenchCase) -> str:
  return f"{case.group}/{case.name}"


def select_case(cases: Sequence[BenchCase], requested: str) -> BenchCase:
  exact = [case for case in cases if case_id(case) == requested]
  if exact:
    return exact[0]
  by_name = [case for case in cases if case.name == requested]
  if len(by_name) == 1:
    return by_name[0]
  if by_name:
    names = ", ".join(case_id(case) for case in by_name)
    raise SystemExit(f"ambiguous case {requested!r}; use one of: {names}")
  known = "\n".join(f"  {case_id(case)}" for case in cases)
  raise SystemExit(f"unknown case {requested!r}; known cases:\n{known}")


def ru_maxrss_bytes(value: int) -> int:
  return value if platform.system() == "Darwin" else value * 1024


def usage_record(before: resource.struct_rusage, after: resource.struct_rusage) -> dict[str, float | int]:
  return {
    "user_cpu_s": after.ru_utime - before.ru_utime,
    "system_cpu_s": after.ru_stime - before.ru_stime,
    "max_rss_bytes": ru_maxrss_bytes(after.ru_maxrss),
    "minor_faults": after.ru_minflt - before.ru_minflt,
    "major_faults": after.ru_majflt - before.ru_majflt,
    "voluntary_context_switches": after.ru_nvcsw - before.ru_nvcsw,
    "involuntary_context_switches": after.ru_nivcsw - before.ru_nivcsw,
  }


def backend_records(value: object) -> list[dict[str, object]]:
  records: list[dict[str, object]] = []

  def visit(current: object, path: str) -> None:
    native = getattr(current, "_native", None)
    if native is not None:
      record: dict[str, object] = {"path": path, "type": type(current).__name__}
      for name in ("backend_code", "used_accelerate", "used_fused"):
        attr = getattr(native, name, None)
        if callable(attr):
          try:
            record[name] = attr()
          except Exception as exc:  # noqa: BLE001
            record[name] = f"error: {exc}"
      records.append(record)
    if isinstance(current, (tuple, list)):
      for index, item in enumerate(current):
        visit(item, f"{path}[{index}]")

  visit(value, "result")
  return records


def numpy_config() -> str:
  buffer = io.StringIO()
  with contextlib.redirect_stdout(buffer):
    np.show_config()
  return buffer.getvalue().rstrip() + "\n"


def child_command(
  args: argparse.Namespace,
  output_path: Path,
  *,
  duration: float | None = None,
  trace_allocations: bool = False,
) -> list[str]:
  command = [
    sys.executable,
    "-m",
    "monpy._bench.profile",
    "--child",
    "--case",
    args.case,
    "--candidate",
    args.candidate,
    "--types",
    ",".join(args.types),
    "--vector-size",
    str(args.vector_size),
    "--vector-sizes",
    ",".join(str(size) for size in args.vector_sizes),
    "--matrix-sizes",
    ",".join(str(size) for size in args.matrix_sizes),
    "--linalg-sizes",
    ",".join(str(size) for size in args.linalg_sizes),
    "--duration",
    str(duration if duration is not None else args.duration),
    "--warmup",
    str(args.warmup),
    "--tracemalloc-frames",
    str(args.tracemalloc_frames),
    "--child-output",
    str(output_path),
  ]
  if trace_allocations:
    command.append("--trace-allocations")
  return command


def child_env() -> dict[str, str]:
  env = os.environ.copy()
  for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__"):
    env.pop(name, None)
  return env


def run_child_measurement(
  args: argparse.Namespace,
  output_path: Path,
  *,
  duration: float | None = None,
  trace_allocations: bool = False,
) -> dict[str, object]:
  command = child_command(args, output_path, duration=duration, trace_allocations=trace_allocations)
  output_path.with_suffix(".command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
  completed = subprocess.run(command, check=False, capture_output=True, text=True, env=child_env())
  output_path.with_suffix(".stdout.txt").write_text(completed.stdout, encoding="utf-8")
  output_path.with_suffix(".stderr.txt").write_text(completed.stderr, encoding="utf-8")
  if completed.returncode != 0:
    raise SystemExit(f"child measurement failed with exit code {completed.returncode}: {display_path(output_path)}")
  return json.loads(output_path.read_text(encoding="utf-8"))


def run_sample(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
  sample = shutil.which("sample")
  if sample is None:
    return {"available": False, "reason": "sample not found"}
  child_json = output_dir / "sample-child.json"
  command = child_command(args, child_json, duration=args.duration + 1.0)
  proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=child_env())
  time.sleep(min(0.5, args.duration / 4.0))
  sample_path = output_dir / "sample.txt"
  sample_cmd = [
    sample,
    str(proc.pid),
    str(args.duration),
    str(args.sample_interval_ms),
    "-mayDie",
    "-file",
    str(sample_path),
  ]
  sample_completed = subprocess.run(
    sample_cmd,
    check=False,
    capture_output=True,
    text=True,
    timeout=args.duration + 20.0,
  )
  stdout, stderr = proc.communicate(timeout=args.duration + 20.0)
  (output_dir / "sample-command.stdout.txt").write_text(sample_completed.stdout, encoding="utf-8")
  (output_dir / "sample-command.stderr.txt").write_text(sample_completed.stderr, encoding="utf-8")
  child_json.with_suffix(".stdout.txt").write_text(stdout, encoding="utf-8")
  child_json.with_suffix(".stderr.txt").write_text(stderr, encoding="utf-8")
  return {
    "available": True,
    "command": sample_cmd,
    "returncode": sample_completed.returncode,
    "path": display_path(sample_path),
    "child_returncode": proc.returncode,
    "child_json": display_path(child_json),
  }


def run_perf_stat(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
  perf = shutil.which("perf")
  if perf is None:
    return {"available": False, "reason": "perf not found"}
  child_json = output_dir / "perf-child.json"
  command = [
    perf,
    "stat",
    "-x",
    ",",
    "-e",
    args.perf_events,
    "--",
    *child_command(args, child_json),
  ]
  completed = subprocess.run(
    command, check=False, capture_output=True, text=True, timeout=args.duration + 30.0, env=child_env()
  )
  (output_dir / "perf-stat.stdout.txt").write_text(completed.stdout, encoding="utf-8")
  (output_dir / "perf-stat.stderr.txt").write_text(completed.stderr, encoding="utf-8")
  return {
    "available": True,
    "command": command,
    "returncode": completed.returncode,
    "stderr_path": display_path(output_dir / "perf-stat.stderr.txt"),
    "child_json": display_path(child_json),
  }


def parse_xctrace_templates(value: str) -> tuple[str, ...]:
  if not value: return ()
  aliases = {
    "time": "Time Profiler",
    "counters": "CPU Counters",
    "allocations": "Allocations",
  }
  return tuple(aliases.get(item, item) for raw in value.split(",") if (item := raw.strip()))


def xctrace_duration(value: float) -> str:
  return f"{max(1, math.ceil(value))}s"


def run_xctrace(args: argparse.Namespace, output_dir: Path) -> list[dict[str, object]]:
  xctrace = shutil.which("xctrace")
  if xctrace is None:
    return [{"available": False, "reason": "xctrace not found"}]
  records: list[dict[str, object]] = []
  for template in args.xctrace:
    safe_name = template.lower().replace(" ", "-")
    child_json = output_dir / f"xctrace-{safe_name}-child.json"
    trace_path = output_dir / f"xctrace-{safe_name}.trace"
    command = [
      xctrace,
      "record",
      "--template",
      template,
      "--time-limit",
      xctrace_duration(args.duration + 2.0),
      "--output",
      str(trace_path),
      "--target-stdout",
      str(output_dir / f"xctrace-{safe_name}.stdout.txt"),
      "--launch",
      "--",
      *child_command(args, child_json),
    ]
    completed = subprocess.run(
      command, check=False, capture_output=True, text=True, timeout=args.duration + 60.0, env=child_env()
    )
    (output_dir / f"xctrace-{safe_name}-command.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / f"xctrace-{safe_name}-command.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    records.append({
      "available": True,
      "template": template,
      "command": command,
      "returncode": completed.returncode,
      "trace_path": display_path(trace_path),
      "child_json": display_path(child_json),
    })
  return records


def child_main(args: argparse.Namespace) -> None:
  cases = build_cases(
    types=args.types,
    vector_size=args.vector_size,
    vector_sizes=args.vector_sizes,
    matrix_sizes=args.matrix_sizes,
    linalg_sizes=args.linalg_sizes,
  )
  case = select_case(cases, args.case)
  monpy_result = call_bench_fn(case.monpy_fn)
  numpy_result = call_bench_fn(case.numpy_fn)
  if case.check_values:
    verify_same(monpy_result, numpy_result, rtol=case.rtol, atol=case.atol)
  else:
    verify_shape_dtype(monpy_result, numpy_result, check_dtype=case.check_dtype)
  bench_fn = case.monpy_fn if args.candidate == "monpy" else case.numpy_fn
  for _ in range(args.warmup):
    result = call_bench_fn(bench_fn)
    if args.candidate == "monpy":
      force_monpy(result)
  if args.trace_allocations and not tracemalloc.is_tracing():
    tracemalloc.start(args.tracemalloc_frames)
  if args.trace_allocations:
    tracemalloc.reset_peak()
  before = resource.getrusage(resource.RUSAGE_SELF)
  started = time.perf_counter()
  deadline = started + args.duration
  iterations = 0
  while time.perf_counter() < deadline:
    result = call_bench_fn(bench_fn)
    if args.candidate == "monpy":
      force_monpy(result)
    iterations += 1
  finished = time.perf_counter()
  after = resource.getrusage(resource.RUSAGE_SELF)
  current_bytes: int | None = None
  peak_bytes: int | None = None
  if args.trace_allocations:
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
  payload = {
    "case": case_id(case),
    "candidate": args.candidate,
    "duration_s": finished - started,
    "iterations": iterations,
    "us_per_call": ((finished - started) / iterations) * 1_000_000 if iterations else None,
    "backend": backend_records(monpy_result) if args.candidate == "monpy" else [],
    "resource": usage_record(before, after),
    "tracemalloc": {
      "current_bytes": current_bytes,
      "peak_bytes": peak_bytes,
      "frames": args.tracemalloc_frames,
      "enabled": args.trace_allocations,
    },
  }
  args.child_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description="profile one monpy benchmark case with OS profilers")
  parser.add_argument("--case", required=True, help="case name or full group/name")
  parser.add_argument("--candidate", choices=("monpy", "numpy"), default="monpy")
  parser.add_argument("--types", type=parse_types, default=parse_types("strides"))
  parser.add_argument("--vector-size", type=positive_int, default=1024)
  parser.add_argument("--vector-sizes", type=parse_sizes, default=parse_sizes("16384,262144,1048576"))
  parser.add_argument("--matrix-sizes", type=parse_sizes, default=parse_sizes("16,64,128,256"))
  parser.add_argument("--linalg-sizes", type=parse_sizes, default=parse_sizes("2,4,8,32,128"))
  parser.add_argument("--duration", type=positive_float, default=8.0)
  parser.add_argument("--memory-duration", type=positive_float, default=2.0)
  parser.add_argument("--warmup", type=positive_int, default=10)
  parser.add_argument("--tracemalloc-frames", type=positive_int, default=25)
  parser.add_argument("--output-dir", type=Path, default=None)
  parser.add_argument(
    "--sample",
    action=argparse.BooleanOptionalAction,
    default=platform.system() == "Darwin",
    help="on macOS, capture a sample(1) stack report",
  )
  parser.add_argument("--sample-interval-ms", type=positive_int, default=1)
  parser.add_argument(
    "--perf-stat",
    action=argparse.BooleanOptionalAction,
    default=platform.system() == "Linux",
    help="on Linux, run perf stat hardware counters around the child loop",
  )
  parser.add_argument("--perf-events", default=DEFAULT_PERF_EVENTS)
  parser.add_argument(
    "--xctrace",
    type=parse_xctrace_templates,
    default=(),
    help="comma-separated macOS xctrace templates or aliases: time,counters,allocations",
  )
  parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
  parser.add_argument("--child-output", type=Path, default=Path("profile-child.json"), help=argparse.SUPPRESS)
  parser.add_argument("--trace-allocations", action="store_true", help=argparse.SUPPRESS)
  args = parser.parse_args(argv)

  if args.child:
    child_main(args)
    return

  if args.output_dir is None:
    args.output_dir = default_profile_dir()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  (args.output_dir / "numpy-config.txt").write_text(numpy_config(), encoding="utf-8")
  baseline = run_child_measurement(args, args.output_dir / "measurement.json")
  allocation = run_child_measurement(
    args,
    args.output_dir / "allocation-measurement.json",
    duration=args.memory_duration,
    trace_allocations=True,
  )
  manifest: dict[str, Any] = {
    "schema_version": 1,
    "kind": "monpy-profile-manifest",
    "created_at": local_now().isoformat(),
    "command": sys.argv[:] if argv is None else [sys.argv[0], *argv],
    "cwd": str(Path.cwd()),
    "case": baseline["case"],
    "candidate": args.candidate,
    "duration_s": args.duration,
    "measurement": baseline,
    "allocation_measurement": allocation,
    "outputs": {
      "measurement": display_path(args.output_dir / "measurement.json"),
      "allocation_measurement": display_path(args.output_dir / "allocation-measurement.json"),
      "numpy_config": display_path(args.output_dir / "numpy-config.txt"),
    },
    "profilers": {},
  }
  if args.sample:
    manifest["profilers"]["sample"] = run_sample(args, args.output_dir)
  if args.perf_stat:
    manifest["profilers"]["perf_stat"] = run_perf_stat(args, args.output_dir)
  if args.xctrace:
    manifest["profilers"]["xctrace"] = run_xctrace(args, args.output_dir)
  manifest_path = args.output_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(f"wrote profile manifest to {display_path(manifest_path)}")


if __name__ == "__main__":
  main()
