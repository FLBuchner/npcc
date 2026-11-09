"""CLI tests: argparse + end-to-end run writing artifacts (hermetic)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from npcc.core import RosenblattBicop
from npcc.experiments import cli

if TYPE_CHECKING:
  from pyvinecopulib.core import ControlsLike

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

#: A second estimator for the grid above, told apart by its transform.
_PROBIT = """
[[grid.estimators]]
label = "normal-probit"
backend = "normal-quantile"
transform = "probit"
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


def test_every_process_of_a_run_records_the_code_it_ran(
  register_hermetic_backends: None, tmp_path: Path
) -> None:
  """Each shard and the aggregation leave their own provenance record.

  The tables carry no commit, and the processes of one run may start from
  different checkouts, so a record per process is what traces every cell back
  to the code that computed it; one record per run would hide a shard that
  ran something else.
  """
  config = tmp_path / "study.toml"
  config.write_text(_TOML)
  out = tmp_path / "run"
  common = ["--config", str(config), "--out", str(out), "--format", "csv"]
  for index in range(2):
    assert cli.main([*common, "--resume", "--shard", f"{index}/2"]) == 0
  assert cli.main([*common, "--resume"]) == 0

  lines = (out / "provenance.jsonl").read_text().splitlines()
  records = [json.loads(line) for line in lines]

  assert [r["options"]["shard"] for r in records] == [[0, 2], [1, 2], None]
  assert all(r["options"]["config"] == str(config) for r in records)
  assert all(
    {"started", "npcc_commit", "uncommitted"} <= set(r) for r in records
  )


def test_a_shard_that_owns_no_cells_exits_cleanly(
  register_hermetic_backends: None, tmp_path: Path
) -> None:
  """More shards than cells leaves a shard with nothing to do, not a crash.

  With one cell and two shards, shard 1 owns no cell. Its tables are empty,
  and sorting an empty table on the study's axes would raise on columns it
  does not have.
  """
  config = tmp_path / "study.toml"
  config.write_text(_TOML)
  out = tmp_path / "run"
  common = ["--config", str(config), "--out", str(out), "--format", "csv"]

  assert cli.main([*common, "--resume", "--shard", "1/2"]) == 0
  assert not list((out / "cells").glob("*/DONE"))


def test_code_provenance_names_this_checkout_and_only_a_checkout(
  tmp_path: Path,
) -> None:
  """The commit is this work tree's ``HEAD``, and nothing outside a checkout.

  A package imported from anywhere but a checkout's ``src`` -- an installed
  wheel inside a repository's virtual environment, say -- must not inherit the
  enclosing repository's commit, or the record would name code that never ran.
  """
  from npcc.experiments import _study

  here = Path(_study.__file__)
  head = subprocess.run(
    ["git", "-C", str(here.parent), "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
    check=False,
  )
  if head.returncode != 0:
    pytest.skip("npcc is not imported from a git checkout")
  record = _study.code_provenance()
  assert record["npcc_commit"] == head.stdout.strip()
  assert isinstance(record["uncommitted"], list)

  stray = tmp_path / "a" / "b" / "c" / "module.py"
  stray.parent.mkdir(parents=True)
  stray.write_text("")
  assert _study.code_provenance(stray) == {
    "npcc_commit": None,
    "uncommitted": None,
  }


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


def test_a_run_restricted_by_labels_fits_as_the_full_run_does(
  register_hermetic_backends: None,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """``--labels`` refits some of a grid's estimators into a fresh folder.

  The analysis swaps such a run's rows in for the full run's, so each kept
  estimator must see the same data and the same torch seed as in the full
  grid. One seeded by its position among the estimators would replace fits
  it never matched.
  """
  fits: list[tuple[str, str, int]] = []
  original_fit = RosenblattBicop.fit

  def record_fit(
    self: RosenblattBicop,
    uv: torch.Tensor,
    /,
    controls: ControlsLike | None = None,
    *,
    var_types: list[str] | None = None,
    x: torch.Tensor | None = None,
  ) -> RosenblattBicop:
    data = uv if x is None else torch.column_stack([uv, x])
    digest = hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest()
    fits.append((self.transform, digest, torch.initial_seed()))
    return original_fit(self, uv, controls, var_types=var_types, x=x)

  monkeypatch.setattr(RosenblattBicop, "fit", record_fit)
  config = tmp_path / "study.toml"
  config.write_text(_TOML.replace("n_rep = 1", "n_rep = 2") + _PROBIT)
  common = ["--config", str(config), "--workers", "1", "--format", "csv"]

  assert cli.main([*common, "--out", str(tmp_path / "full")]) == 0
  full = sorted(f for f in fits if f[0] == "probit")
  fits.clear()
  part = tmp_path / "part"
  assert (
    cli.main([*common, "--out", str(part), "--labels", "normal-probit"]) == 0
  )

  assert full
  assert sorted(fits) == full
  echo = json.loads((part / "config.json").read_text())
  assert [e["label"] for e in echo["grid"]["estimators"]] == ["normal-probit"]
  assert "normal-logit" not in (part / "metrics_by_x.csv").read_text()


def test_an_aggregation_pass_without_the_shards_labels_is_refused(
  register_hermetic_backends: None, tmp_path: Path
) -> None:
  """Shards run under ``--labels`` must be aggregated under the same labels.

  Otherwise the pass would echo the full grid's estimators over a subset's
  cells. The resume signature hashes the kept estimators, so it refuses.
  """
  config = tmp_path / "study.toml"
  config.write_text(_TOML + _PROBIT)
  common = ["--config", str(config), "--out", str(tmp_path / "part")]
  common += ["--workers", "1", "--format", "csv", "--resume"]
  assert cli.main([*common, "--shard", "0/1", "--labels", "normal-probit"]) == 0
  with pytest.raises(ValueError, match="grid signature must match"):
    cli.main(common)
