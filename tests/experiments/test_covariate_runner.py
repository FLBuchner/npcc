"""The covariate study (E9): its config, its runner and its CLI, end to end.

Every fit uses the hermetic ``normal-quantile`` backend, and every external
estimator a Python stand-in for R that predicts the independence copula, so
no third-party model and no R are needed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest
import torch

from npcc.core.errors import EstimatorConfigError
from npcc.experiments import _external, metrics
from npcc.experiments import covariate_scenarios as cs
from npcc.experiments.config import EstimatorSpec, RunConfig, load_grid
from npcc.experiments.covariate_cli import main
from npcc.experiments.covariate_config import (
  CovariateGridConfig,
  load_covariate_grid,
)
from npcc.experiments.covariate_runner import (
  _cell_data,
  cell_seed,
  eval_points,
  grid_signature,
  run_covariate_study,
)

_ROOT = Path(__file__).resolve().parents[2]

# Every grid here is built on the hermetic backend, so its registration
# spans the module.
pytestmark = pytest.mark.usefixtures("register_hermetic_backends")

#: A script speaking E9's exchange for the independence copula. Each run
#: appends its arguments and the eval file's header to a ``.calls`` file.
_STAND_IN = """\
import csv
import sys
from pathlib import Path

train_csv, eval_csv, pred_csv, seed, scenario, family = sys.argv[1:]
with open(train_csv, newline="") as fh:
  train_header = next(csv.reader(fh))
with open(eval_csv, newline="") as fh:
  reader = csv.DictReader(fh)
  header = reader.fieldnames
  points = list(reader)
with open(Path(__file__).with_suffix(".calls"), "a") as fh:
  fh.write(" ".join([scenario, family, ",".join(train_header),
                     ",".join(header)]))
  fh.write("\\n")
timing = {"fit_time": 1.0, "pdf_time": 0.5, "hfunc1_time": 0.25,
          "hfunc2_time": 0.25, "tau_time": 0.0}
rows = [
  {**p, "pdf": 1.0, "hfunc1": p["u2"], "hfunc2": p["u1"], "tau": 0.0, **timing}
  for p in points
]
with open(pred_csv, "w", newline="") as fh:
  writer = csv.DictWriter(fh, [*header, "pdf", "hfunc1", "hfunc2", "tau",
                               *timing])
  writer.writeheader()
  writer.writerows(rows)
"""


def _normal(label: str) -> EstimatorSpec:
  return EstimatorSpec(
    label=label, backend="normal-quantile", transform="probit"
  )


def _grid(*estimators: EstimatorSpec, **kwargs: object) -> CovariateGridConfig:
  options: dict[str, object] = {
    "families": ["clayton"],
    "scenarios": ["gampcc", "constant"],
    "n": [80],
    "n_rep": 1,
    "estimators": list(estimators),
    "n_points": 3,
    "uv_grid_n": 3,
    "holdout_n": 10,
    **kwargs,
  }
  return CovariateGridConfig(**options)  # ty: ignore[invalid-argument-type] - options are typed per field above


@pytest.fixture
def stand_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  """The stand-in script, with the exchange's interpreter pointed at Python."""
  monkeypatch.setattr(_external, "RSCRIPT", sys.executable)
  script = tmp_path / "stand_in.py"
  script.write_text(_STAND_IN)
  return script


def _external_spec(script: Path) -> EstimatorSpec:
  return EstimatorSpec(
    label="independence",
    backend="stand-in",
    transform="identity",
    kind="external_r",
    script=str(script),
  )


def test_shipped_configs_fit_e1s_estimators_on_one_grid() -> None:
  """The R competitors must fit the backends' exact cells, with E1's ids.

  A ``[grid]`` field that drifted between the two configs would score the
  methods on different data or points, and an estimator id other than E1's
  would make "TabPFN v3" another model than E1's.
  """
  backends = load_covariate_grid(_ROOT / "configs" / "e9_covariates.toml")
  external = load_covariate_grid(_ROOT / "configs" / "e9_covariates_r.toml")
  for f in fields(CovariateGridConfig):
    if f.name not in {"estimators", "active_covariates", "without_covariates"}:
      assert getattr(external, f.name) == getattr(backends, f.name), f.name
  assert len(backends.cells()) == 2 * 3 * 2 * 10
  e1 = {
    e.estimator_id
    for e in load_grid(_ROOT / "configs" / "study.toml").estimators
  }
  assert {e.estimator_id for e in backends.estimators} <= e1
  assert [backends.covariates(e) for e in backends.estimators] == [
    "all",
    "all",
    "all",
    "active",
    "none",
  ]
  assert {e.kind for e in external.estimators} == {"external_r"}


@pytest.mark.parametrize(
  ("kwargs", "match"),
  [
    ({"active_covariates": ["missing"]}, "must name estimators of the grid"),
    (
      {"active_covariates": ["a"], "without_covariates": ["a"]},
      "must not share a label",
    ),
    ({"scenarios": ["linear"]}, "Unknown scenarios"),
    ({"families": ["gumbel"]}, "Unknown families"),
    ({"holdout_n": 1}, "holdout_n must be >= 2"),
  ],
)
def test_a_grid_that_cannot_run_is_refused(
  kwargs: dict[str, object], match: str
) -> None:
  """A flag on a missing label would silently fit it on every covariate."""
  with pytest.raises((ValueError, EstimatorConfigError), match=match):
    _grid(_normal("a"), **kwargs)


def test_a_script_cannot_be_fitted_on_a_subset(stand_in: Path) -> None:
  """A script is handed every covariate; a subset flag on it would be ignored."""
  with pytest.raises(EstimatorConfigError, match="kind='backend'"):
    _grid(_external_spec(stand_in), without_covariates=["independence"])


def test_each_estimator_is_fitted_on_its_covariates(
  tmp_path: Path, pair_fit_widths: list[int | None]
) -> None:
  """The no-covariate and active-covariate arms must see what they claim.

  An arm handed every covariate would measure nothing; the active arm has no
  covariate to see under constant, where it is not fitted.
  """
  grid = _grid(
    _normal("all"),
    _normal("active"),
    _normal("none"),
    active_covariates=["active"],
    without_covariates=["none"],
  )
  metric_df, runtime_df, _ = run_covariate_study(grid, RunConfig(out=tmp_path))
  # gampcc: all, active, none; constant: all, none.
  assert pair_fit_widths == [15, len(cs.ACTIVE["gampcc"]), None, 15, None]
  fitted = runtime_df.groupby("scenario")["label"].apply(sorted).to_dict()
  assert fitted == {
    "gampcc": ["active", "all", "none"],
    "constant": ["all", "none"],
  }
  assert set(metric_df["covariates"]) == {"all", "active", "none"}


def test_every_row_scores_a_point(tmp_path: Path) -> None:
  """A missing (quantity, point) row would bias the mean over points."""
  grid = _grid(_normal("all"))
  metric_df, _, _ = run_covariate_study(grid, RunConfig(out=tmp_path))
  counts = metric_df.groupby(["scenario", "quantity"]).size()
  assert (counts == grid.n_points).all()
  pdf = metric_df[metric_df["quantity"] == "pdf"]
  assert pdf["KL_mc"].notna().all()
  assert (pdf["KL_mc_n"] == grid.holdout_n).all()
  assert metric_df.loc[metric_df["quantity"] != "pdf", "KL_mc"].isna().all()


def test_the_points_are_the_scenarios_alone() -> None:
  """Points that moved with the cell would not be shared across reps."""
  grid = _grid(
    _normal("all"), families=["clayton", "gaussian"], n=[40, 80], n_rep=2
  )
  pts = {
    cell: _cell_data(cell, grid, 7).metric_grid.x_axis for cell in grid.cells()
  }
  for cell, x in pts.items():
    assert x is not None
    assert torch.equal(x, eval_points(grid, 7, cell.scenario))
  assert not torch.equal(
    eval_points(grid, 7, "gampcc"), eval_points(grid, 7, "constant")
  )


def test_every_estimator_of_a_cell_scores_its_draws() -> None:
  """The KL_mc pairs estimators only if they score the same draws."""
  grid = _grid(_normal("all"))
  cell = grid.cells()[0]
  a, b = _cell_data(cell, grid, 7), _cell_data(cell, grid, 7)
  assert torch.equal(a.holdout.uv, b.holdout.uv)
  assert a.seed == cell_seed(7, cell)
  other = grid.cells()[1]
  assert not torch.equal(_cell_data(other, grid, 7).holdout.uv, a.holdout.uv)


def test_a_script_gets_every_covariate_the_scenario_and_the_family(
  tmp_path: Path, stand_in: Path
) -> None:
  """The exchange must hand all fifteen covariates, the scenario and family.

  The oracle fits the true model, so it reads both; without them it would
  have to guess the truth from the data.

  The stand-in predicts the independence copula, so its KL_mc at each point
  is the mean true log density at that point's draws, exactly.
  """
  grid = _grid(_external_spec(stand_in))
  metric_df, runtime_df, _ = run_covariate_study(grid, RunConfig(out=tmp_path))
  calls = stand_in.with_suffix(".calls").read_text().splitlines()
  header = ",".join(cs.COVARIATES)
  assert [c.split(" ")[:2] for c in calls] == [
    ["gampcc", "clayton"],
    ["constant", "clayton"],
  ]
  for call in calls:
    _, _, train, evaluation = call.split(" ")
    assert train == f"u1,u2,{header}"
    assert evaluation == f"grid,u1,u2,{header}"

  cell = grid.cells()[0]
  data = _cell_data(cell, grid, RunConfig(out=tmp_path).base_seed)
  log_true = data.holdout_log_true.reshape(data.holdout.shape)
  pdf = metric_df[
    (metric_df["scenario"] == cell.scenario) & (metric_df["quantity"] == "pdf")
  ].sort_values("point")
  for j, kl in enumerate(pdf["KL_mc"]):
    expected = metrics.holdout_kl(log_true[:, j], torch.ones(grid.holdout_n))
    assert kl == pytest.approx(expected["KL_mc"], rel=1e-12)
  assert (runtime_df["fit_time"] == 1.0).all()


def test_the_signature_follows_what_a_cell_computes() -> None:
  """A resume must refuse cells scored otherwise, and keep a grown grid's."""
  run = RunConfig(out=Path("unused"))
  base = grid_signature(_grid(_normal("a")), run)
  assert grid_signature(_grid(_normal("a"), n_rep=3, n=[80, 160]), run) == base
  for kwargs in ({"holdout_n": 20}, {"n_points": 4}, {"uv_grid_n": 4}):
    assert grid_signature(_grid(_normal("a"), **kwargs), run) != base
  flagged = _grid(_normal("a"), without_covariates=["a"])
  assert grid_signature(flagged, run) != base


def test_the_cli_writes_the_tables_points_and_config(tmp_path: Path) -> None:
  """A finished run must leave every table the analysis and the paper read."""
  config = tmp_path / "e9.toml"
  config.write_text(
    """
[grid]
families = ["gaussian"]
scenarios = ["interaction"]
n = [60]
n_rep = 1
n_points = 2
uv_grid_n = 3
holdout_n = 10
without_covariates = ["none"]

[[grid.estimators]]
label = "all"
backend = "normal-quantile"
transform = "probit"

[[grid.estimators]]
label = "none"
backend = "normal-quantile"
transform = "probit"
"""
  )
  out = tmp_path / "out"
  assert (
    main(["--config", str(config), "--out", str(out), "--device", "cpu"]) == 0
  )
  for name in (
    "metrics_by_point",
    "summary_by_point",
    "summary_over_points",
    "runtime",
    "runtime_summary",
    "points",
  ):
    assert (out / f"{name}.parquet").is_file(), name
  points = pd.read_parquet(out / "points.parquet")
  assert list(points.columns[2:17]) == list(cs.COVARIATES)
  echo = json.loads((out / "config.json").read_text())
  assert echo["grid"]["without_covariates"] == ["none"]


def test_a_failed_script_is_a_result_not_a_dropped_row(
  tmp_path: Path, stand_in: Path
) -> None:
  """A fit that fails must stay in the tables, scored as missing, with why.

  Dropped, it would leave the mean over the other repetitions looking like
  the method's whole record; raised, it would stop the shard's other cells.
  """
  stand_in.write_text(
    "import sys\nsys.stderr.write('no family converged')\nsys.exit(1)\n"
  )
  grid = _grid(_external_spec(stand_in), _normal("all"))
  metric_df, runtime_df, _ = run_covariate_study(grid, RunConfig(out=tmp_path))
  failed = metric_df[metric_df["label"] == "independence"]
  assert len(failed) == 2 * 3 * grid.n_points
  assert failed["error"].str.contains("no family converged").all()
  assert failed[["IAE", "ISE", "KL"]].isna().all().all()
  assert failed.loc[failed["quantity"] == "pdf", "KL_mc"].isna().all()
  assert (metric_df.loc[metric_df["label"] == "all", "error"] == "").all()
  assert (
    runtime_df.loc[runtime_df["label"] == "independence", "fit_time"]
    .isna()
    .all()
  )


def test_a_missing_interpreter_stops_the_run(
  tmp_path: Path, stand_in: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A machine without R must not record every R fit as the method failing."""
  monkeypatch.setattr(_external, "RSCRIPT", str(tmp_path / "no-such-rscript"))
  grid = _grid(_external_spec(stand_in))
  with pytest.raises(RuntimeError, match="must be on PATH"):
    run_covariate_study(grid, RunConfig(out=tmp_path / "out"))


def test_predictions_that_break_the_exchange_stop_the_run(
  tmp_path: Path, stand_in: Path
) -> None:
  """A harness or script defect must not be recorded as the method failing.

  Recorded, a broken exchange would turn every fit of the script into a
  ``failed`` count in the tables, and read as the method's record.
  """
  stand_in.write_text(
    stand_in.read_text().replace('"pdf": 1.0', '"pdf": float("nan")')
  )
  grid = _grid(_external_spec(stand_in))
  with pytest.raises(_external.ExchangeError, match="finite pdf"):
    run_covariate_study(grid, RunConfig(out=tmp_path / "out"))


def test_a_script_refusing_its_input_stops_the_run(
  tmp_path: Path, stand_in: Path
) -> None:
  """A script's own check failing is not the method failing on its sample.

  E9's scripts exit with ``REFUSED_STATUS`` when they refuse their input or
  a check of their own fails; recorded as failed fits, a harness defect
  would read as the method's record.
  """
  stand_in.write_text(
    "import sys\nsys.stderr.write('train.csv must have the columns')\n"
    f"sys.exit({_external.REFUSED_STATUS})\n"
  )
  grid = _grid(_external_spec(stand_in))
  with pytest.raises(_external.ExchangeError, match="must have the columns"):
    run_covariate_study(grid, RunConfig(out=tmp_path / "out"))


@pytest.mark.parametrize(
  "script",
  sorted((_ROOT / "r_methods").glob("*_covariates.R"))
  + [
    _ROOT / "r_methods" / f"fit_gamcopula_{s}.R" for s in ("select", "oracle")
  ],
  ids=lambda p: p.name,
)
def test_e9s_scripts_refuse_with_the_refused_status(script: Path) -> None:
  """A check of the script's own must exit with the status the harness reads.

  With ``stop()``, R's status 1, a bad input would be recorded as the
  method failing; the harness's checks call ``refuse``, which quits with
  ``REFUSED_STATUS``.
  """
  text = script.read_text()
  assert f'quit(save = "no", status = {_external.REFUSED_STATUS})' in text
  assert 'refuse("usage: ' in text and "refuse(what, " in text
  assert 'refuse("the family map disagrees' in text


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="needs Rscript")
def test_a_missing_package_stops_the_run_before_its_first_cell(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A package missing from the machine would fail every fit of its script.

  Each failure would be recorded as the method's own, so the run must stop
  before it fits anything.
  """
  monkeypatch.setattr(_external, "RSCRIPT", shutil.which("Rscript"))
  script = tmp_path / "needs_a_package.R"
  script.write_text(
    "suppressPackageStartupMessages({\n  library(npccNoSuchPackage)\n})\n"
  )
  grid = _grid(_external_spec(script))
  out = tmp_path / "out"
  with pytest.raises(RuntimeError, match="npccNoSuchPackage"):
    run_covariate_study(grid, RunConfig(out=out))
  assert not (out / "cells").exists() or not any((out / "cells").iterdir())


def _gamcopula() -> bool:
  rscript = shutil.which("Rscript")
  return rscript is not None and (
    subprocess.run(
      [rscript, "-e", "library(gamCopula)"], capture_output=True, check=False
    ).returncode
    == 0
  )


@pytest.mark.skipif(not _gamcopula(), reason="needs Rscript with gamCopula")
def test_the_oracle_grows_a_basis_as_the_selection_does(tmp_path: Path) -> None:
  """At k = 10 the oracle underfits s3(t) = sin(6 pi t) / 4, three periods.

  It would then lose to the selection arm, whose doubling step grows that
  basis, for a reason unrelated to knowing the truth; the oracle takes the
  same rule, and the doubled basis shows in its formula.
  """
  u, v, x = cs.sample("gaussian", "gampcc", 2000, 3)
  columns = {name: x[:, k].numpy() for k, name in enumerate(cs.COVARIATES)}
  pd.DataFrame({"u1": u.numpy(), "u2": v.numpy(), **columns}).to_csv(
    tmp_path / "train.csv", index=False
  )
  pd.DataFrame(
    {
      "grid": "g",
      "u1": u.numpy()[:3],
      "u2": v.numpy()[:3],
      **{name: c[:3] for name, c in columns.items()},
    }
  ).to_csv(tmp_path / "eval.csv", index=False)
  done = subprocess.run(
    [
      shutil.which("Rscript") or "Rscript",
      str(_ROOT / "r_methods" / "fit_gamcopula_oracle.R"),
      str(tmp_path / "train.csv"),
      str(tmp_path / "eval.csv"),
      str(tmp_path / "pred.csv"),
      "1",
      "gampcc",
      "gaussian",
    ],
    capture_output=True,
    text=True,
    check=True,
  )
  assert 's(t3, k = 20, bs = "cr")' in done.stdout
  assert 's(t1, k = 10, bs = "cr")' in done.stdout
