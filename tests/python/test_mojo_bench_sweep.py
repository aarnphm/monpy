from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def test_standard_and_numojo_commands_use_repo_include_paths() -> None:
  repo_root = Path("/repo/monpy")
  mojo = Path("/toolchain/mojo")
  numojo = Path("/deps/NuMojo")

  standard = mojo_sweep.standard_command(mojo=mojo, repo_root=repo_root)
  optional = mojo_sweep.numojo_command(mojo=mojo, repo_root=repo_root, numojo_path=numojo)

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
    numojo_path=Path("/deps/NuMojo"),
  )
  rows = [
    mojo_sweep.MojoBenchRow(
      group="numojo.reductions",
      name="sum_f32_1024",
      candidate="numojo.sum",
      baseline="monpy.reduce_sum_typed",
      candidate_ns=120.0,
      baseline_ns=80.0,
      ratio=1.5,
      bytes=4096,
      flops=1024,
    )
  ]

  payload = json.loads(mojo_sweep.render_results(rows, args=args))

  assert payload["config"]["suite"] == "mojo-kernel-sweep"
  assert payload["config"]["include_numojo"] is True
  assert payload["config"]["comparison"] == "candidate_ns / baseline_ns"
  assert payload["results"][0]["candidate"] == "numojo.sum"
  assert payload["results"][0]["baseline"] == "monpy.reduce_sum_typed"
