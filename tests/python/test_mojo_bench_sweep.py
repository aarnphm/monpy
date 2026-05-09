from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from monpy._bench import mojo_sweep


def test_parse_mojo_tsv_ignores_preheader_noise() -> None:
  output = "\n".join([
    "compiler warning on stdout would be rude, but parse around it",
    "\t".join(mojo_sweep.TSV_HEADER),
    "elementwise\tadd_f32_1k\tmonpy.add\tstd.add\t12.5\t10.0\t1.25\t12288\t1024",
    "reductions\tsum_f32_64k\tmonpy.sum\tstd.sum\t90.0\t100.0\t0.9\t262144\t65536",
  ])

  rows = mojo_sweep.parse_mojo_tsv(output)

  assert [row.name for row in rows] == ["add_f32_1k", "sum_f32_64k"]
  assert rows[0].candidate_ns == 12.5
  assert rows[0].baseline_ns == 10.0
  assert rows[1].ratio == 0.9


def test_standard_numojo_and_threading_commands_use_repo_include_paths() -> None:
  repo_root = Path("/repo/monpy")
  mojo = Path("/toolchain/mojo")
  numojo = Path("/deps/NuMojo")

  standard = mojo_sweep.standard_command(mojo=mojo, repo_root=repo_root)
  optional = mojo_sweep.numojo_command(mojo=mojo, repo_root=repo_root, numojo_path=numojo)
  threading = mojo_sweep.threading_command(mojo=mojo, repo_root=repo_root)

  assert standard == [
    "/toolchain/mojo",
    "run",
    "-I",
    "/repo/monpy/src",
    "/repo/monpy/benches/bench_mojo_sweep.mojo",
  ]
  assert "--ignore-incompatible-package-errors" in optional
  assert optional == [
    "/toolchain/mojo",
    "run",
    "--ignore-incompatible-package-errors",
    "-I",
    "/repo/monpy/src",
    "-I",
    "/deps/NuMojo",
    "/repo/monpy/benches/bench_numojo_sweep.mojo",
  ]
  assert threading == [
    "/toolchain/mojo",
    "run",
    "-I",
    "/repo/monpy/src",
    "/repo/monpy/benches/bench_threading_sweep.mojo",
  ]


def test_parse_thread_caps_accepts_auto_and_positive_integer_caps() -> None:
  assert mojo_sweep.parse_thread_caps("auto,1,2,4,auto") == ("auto", "1", "2", "4")

  with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
    mojo_sweep.parse_thread_caps("auto,0")


def test_default_numojo_path_prefers_vendor_checkout(tmp_path: Path) -> None:
  repo_root = tmp_path / "monpy"
  vendor = repo_root / "vendor" / "NuMojo"
  vendor.mkdir(parents=True)

  assert mojo_sweep.default_numojo_path(repo_root) == vendor


def test_mojo_subprocess_env_adds_packaged_modular_libs(tmp_path: Path) -> None:
  mojo = tmp_path / ".venv" / "bin" / "mojo"
  modular_lib = tmp_path / ".venv" / "lib" / "python3.11" / "site-packages" / "modular" / "lib"
  mojo.parent.mkdir(parents=True)
  modular_lib.mkdir(parents=True)
  mojo.write_text("", encoding="utf-8")

  env = mojo_sweep.mojo_subprocess_env(mojo)

  assert str(modular_lib) in env["DYLD_LIBRARY_PATH"].split(":")
  assert str(modular_lib) in env["LD_LIBRARY_PATH"].split(":")


def test_render_json_preserves_row_level_candidate_and_baseline() -> None:
  args = argparse.Namespace(
    format="json",
    sort="input",
    include_numojo=True,
    include_threading=True,
    numojo_path=Path("/deps/NuMojo"),
    thread_caps=("auto", "1"),
  )
  rows = [
    mojo_sweep.MojoBenchRow(
      group="numojo.reductions",
      name="sum_f32_1024",
      candidate="monpy.reduce_sum_typed",
      baseline="numojo.sum",
      candidate_ns=80.0,
      baseline_ns=120.0,
      ratio=2.0 / 3.0,
      bytes=4096,
      flops=1024,
    )
  ]

  payload = json.loads(mojo_sweep.render_results(rows, args=args))

  assert payload["config"]["suite"] == "mojo-kernel-sweep"
  assert payload["config"]["include_numojo"] is True
  assert payload["config"]["include_threading"] is True
  assert payload["config"]["comparison"] == "candidate_ns / baseline_ns"
  assert payload["config"]["strict_numojo"] is False
  assert payload["config"]["thread_caps"] == ["auto", "1"]
  assert payload["results"][0]["candidate"] == "monpy.reduce_sum_typed"
  assert payload["results"][0]["baseline"] == "numojo.sum"


def _row(name: str, candidate_ns: float, ratio: float = 1.0) -> mojo_sweep.MojoBenchRow:
  return mojo_sweep.MojoBenchRow(
    group="elementwise",
    name=name,
    candidate="monpy",
    baseline="std",
    candidate_ns=candidate_ns,
    baseline_ns=candidate_ns / ratio,
    ratio=ratio,
    bytes=0,
    flops=0,
  )


def test_sorted_rows_fastest_orders_smallest_ratio_first() -> None:
  # ratio = candidate/baseline. fastest = "candidate beats baseline by the most"
  # = smallest ratio first.
  rows = [_row("loss", 100.0, ratio=2.0), _row("tie", 100.0, ratio=1.0), _row("win", 100.0, ratio=0.4)]
  ordered = mojo_sweep.sorted_rows(rows, sort="fastest")
  assert [row.name for row in ordered] == ["win", "tie", "loss"]


def test_sorted_rows_slowest_matches_legacy_ratio_alias() -> None:
  # slowest = ratio descending = "regressions on top". Should match the
  # historical `--sort ratio` semantics byte-for-byte.
  rows = [_row("loss", 100.0, ratio=2.0), _row("tie", 100.0, ratio=1.0), _row("win", 100.0, ratio=0.4)]
  by_slowest = [row.name for row in mojo_sweep.sorted_rows(rows, sort="slowest")]
  by_ratio = [row.name for row in mojo_sweep.sorted_rows(rows, sort="ratio")]
  assert by_slowest == by_ratio == ["loss", "tie", "win"]


def test_sorted_rows_candidate_still_orders_by_absolute_timing() -> None:
  # `candidate` keeps its old absolute-timing semantic for back-compat.
  rows = [_row("slow", 1000.0), _row("medium", 100.0), _row("fast", 10.0)]
  ordered = mojo_sweep.sorted_rows(rows, sort="candidate")
  assert [row.name for row in ordered] == ["slow", "medium", "fast"]


def _standard_tsv() -> str:
  return "\n".join([
    "\t".join(mojo_sweep.TSV_HEADER),
    "elementwise\tadd_f32_1024\tmonpy.add\tstd.add\t12.5\t10.0\t1.25\t12288\t1024",
  ])


def _run_standard_then_fail_numojo(
  command: Sequence[str],
  *,
  timeout: int,
  env_overrides: dict[str, str] | None = None,
  unset_env: Sequence[str] = (),
) -> str:
  del timeout
  del env_overrides
  del unset_env
  if command[-1].endswith("bench_numojo_sweep.mojo"):
    raise RuntimeError(
      "command failed: mojo run bench_numojo_sweep.mojo\n"
      "/deps/NuMojo/numojo/routines/math/sums.mojo:51:37: "
      "error: unknown function effect 'unified'"
    )
  return _standard_tsv()


def test_collect_rows_warns_and_keeps_standard_rows_when_numojo_fails(
  monkeypatch: MonkeyPatch,
) -> None:
  monkeypatch.setattr(mojo_sweep, "run_mojo_command", _run_standard_then_fail_numojo)

  collection = mojo_sweep.collect_rows(
    mojo=Path("/toolchain/mojo"),
    repo_root=Path("/repo/monpy"),
    include_numojo=True,
    numojo_path=Path("/deps/NuMojo"),
    strict_numojo=False,
    timeout=5,
  )

  assert [row.name for row in collection.rows] == ["add_f32_1024"]
  assert len(collection.commands) == 2
  assert len(collection.warnings) == 1
  assert "skipping NuMojo comparison" in collection.warnings[0]
  assert "unknown function effect 'unified'" in collection.warnings[0]


def test_collect_rows_can_fail_strictly_for_numojo_failure(monkeypatch: MonkeyPatch) -> None:
  monkeypatch.setattr(mojo_sweep, "run_mojo_command", _run_standard_then_fail_numojo)

  with pytest.raises(RuntimeError, match="unknown function effect 'unified'"):
    mojo_sweep.collect_rows(
      mojo=Path("/toolchain/mojo"),
      repo_root=Path("/repo/monpy"),
      include_numojo=True,
      numojo_path=Path("/deps/NuMojo"),
      strict_numojo=True,
      timeout=5,
    )


def test_collect_rows_runs_threading_sweep_in_fresh_thread_cap_processes(
  monkeypatch: MonkeyPatch,
) -> None:
  calls: list[tuple[tuple[str, ...], dict[str, str] | None, tuple[str, ...]]] = []

  def run_with_thread_caps(
    command: Sequence[str],
    *,
    timeout: int,
    env_overrides: dict[str, str] | None = None,
    unset_env: Sequence[str] = (),
  ) -> str:
    del timeout
    calls.append((tuple(command), env_overrides, tuple(unset_env)))
    if command[-1].endswith("bench_threading_sweep.mojo"):
      return "\n".join([
        "\t".join(mojo_sweep.TSV_HEADER),
        "threading.auto\tadd_f32_1m\tinternal.threaded\tmonpy.serial\t80.0\t100.0\t0.8\t12582912\t1048576",
      ])
    return _standard_tsv()

  monkeypatch.setattr(mojo_sweep, "run_mojo_command", run_with_thread_caps)

  collection = mojo_sweep.collect_rows(
    mojo=Path("/toolchain/mojo"),
    repo_root=Path("/repo/monpy"),
    include_numojo=False,
    numojo_path=None,
    strict_numojo=False,
    include_threading=True,
    thread_caps=("auto", "1", "4"),
    timeout=5,
  )

  assert [row.group for row in collection.rows] == [
    "elementwise",
    "threading.auto",
    "threading.auto",
    "threading.auto",
  ]
  assert calls[1][1] is None
  assert calls[1][2] == ("MONPY_THREADS",)
  assert calls[2][1] == {"MONPY_THREADS": "1"}
  assert calls[3][1] == {"MONPY_THREADS": "4"}
  assert collection.commands[1][0:3] == ["env", "-u", "MONPY_THREADS"]
  assert collection.commands[2][0:2] == ["env", "MONPY_THREADS=1"]
