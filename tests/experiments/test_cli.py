"""CLI tests: argparse + end-to-end run writing artifacts (hermetic)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from npcc.experiments import cli

_TOML = """
[grid]
families = ["clayton"]
tau_scenarios = ["linear"]
normalize = ["none", 2]
n = [30]
n_rep = 1

[[grid.estimators]]
label = "normal-logit"
backend = "normal-quantile"
transform = "logit"
"""

_TABLES = (
  "summary",
  "metrics_by_x",
  "summary_by_x",
  "summary_over_x",
  "selection_summary",
  "projection_summary",
  "tau_summary",
  "diagnostics",
  "quantities",
  "runtime",
  "runtime_summary",
)


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_cli_end_to_end_writes_tables(
  register_hermetic_backends: None, tmp_path: Path, fmt: str
) -> None:
  """A table missing from either format would leave a run unanalyzable."""
  if fmt == "parquet" and importlib.util.find_spec("pyarrow") is None:
    pytest.skip("pyarrow is required for parquet output")
  config = tmp_path / "study.toml"
  config.write_text(_TOML)
  out = tmp_path / "results"

  rc = cli.main(
    ["--config", str(config), "--out", str(out), "--workers", "1"]
    + ["--format", fmt]
  )
  assert rc == 0

  for name in _TABLES:
    path = out / f"{name}.{fmt}"
    assert path.exists(), f"missing {name}.{fmt}"
    assert path.stat().st_size > 0
  echo = json.loads((out / "config.json").read_text())
  assert echo["grid"]["estimators"][0]["label"] == "normal-logit"
  assert echo["run"]["fmt"] == fmt


def test_cli_parser_requires_config_and_out() -> None:
  parser = cli._build_parser()
  ns = parser.parse_args(["--config", "g.toml", "--out", "o", "--workers", "4"])
  assert ns.fmt == "parquet"
  assert ns.workers == 4
  assert ns.resume is False
  assert Path(ns.config).name == "g.toml"
  resumed = parser.parse_args(["--config", "g.toml", "--out", "o", "--resume"])
  assert resumed.resume is True


def test_cli_leaves_the_cuda_allocator_at_its_default(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  """Starting a study must not switch torch to expandable segments.

  They are built on CUDA's virtual-memory API, which a virtualized GPU may
  not offer, and there every CUDA allocation fails with "operation not
  supported" -- so a run on such a box died at its first fit.
  """
  from npcc.experiments import _study

  monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
  args = _study.study_parser("probe", "A probe.").parse_args(
    ["--config", str(tmp_path / "unused.toml"), "--out", str(tmp_path / "o")]
  )

  _study.start(args)

  assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ


def test_sharded_runs_reproduce_the_unsharded_tables(
  register_hermetic_backends: None, tmp_path: Path
) -> None:
  """Shards split the cells between processes without changing any result.

  Each shard computes every K-th cell into the shared folder and writes no
  tables; the pass without ``--shard`` then aggregates. A shard that skipped a
  cell, computed one twice, or seeded a cell by its position in the shard
  would leave tables that differ from a single process's.
  """
  config = tmp_path / "study.toml"
  config.write_text(_TOML.replace("n_rep = 1", "n_rep = 3"))
  common = ["--config", str(config), "--workers", "1", "--format", "csv"]

  single = tmp_path / "single"
  assert cli.main([*common, "--out", str(single)]) == 0

  sharded = tmp_path / "sharded"
  for index in range(2):
    rc = cli.main(
      [*common, "--out", str(sharded), "--resume", "--shard", f"{index}/2"]
    )
    assert rc == 0
  assert not (sharded / "metrics_by_x.csv").exists()
  assert len(list((sharded / "cells").glob("*/DONE"))) == 3

  assert cli.main([*common, "--out", str(sharded), "--resume"]) == 0
  for name in ("metrics_by_x", "summary_over_x", "diagnostics"):
    expected = (single / f"{name}.csv").read_text()
    assert (sharded / f"{name}.csv").read_text() == expected, name


@pytest.mark.parametrize("bad", ["1", "a/2", "-1/2", "1/"])
def test_a_malformed_shard_is_refused_by_the_parser(bad: str) -> None:
  """``--shard`` takes ``I/K`` in whole numbers, and says so at parse time."""
  with pytest.raises(SystemExit):
    cli._build_parser().parse_args(
      ["--config", "c.toml", "--out", "o", "--shard", bad]
    )


@pytest.mark.parametrize("shard", [(2, 2), (3, 2)])
def test_a_shard_outside_its_count_is_refused(
  tmp_path: Path, shard: tuple[int, int]
) -> None:
  """A shard index of ``K`` or more matches no cell, so it would run nothing."""
  from npcc.experiments.config import RunConfig

  with pytest.raises(ValueError, match="0 <= I < K"):
    RunConfig(out=tmp_path, shard=shard)
