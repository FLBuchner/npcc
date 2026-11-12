"""E9's analysis: the runs it pairs, the fits it fails, the families it calls true."""

from __future__ import annotations

import importlib.util
import json
import math
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import cast

import pandas as pd
import pytest
import torch

from npcc.experiments import covariate_scenarios as cs
from npcc.experiments.covariate_config import load_covariate_grid

_SCRIPT = (
  Path(__file__).resolve().parents[2]
  / "scripts"
  / "paper"
  / "e9"
  / "analyze.py"
)


def _load_script() -> ModuleType:
  spec = importlib.util.spec_from_file_location("e9_analyze", _SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


analyze = _load_script()

Cell = tuple[str, str, int, int]

#: Every family and scenario at one n, two repetitions.
CELLS: list[Cell] = [
  (family, scenario, 500, rep)
  for family in cs.SCENARIO_FAMILIES
  for scenario in cs.SCENARIOS
  for rep in range(2)
]


def _rows(
  label: str,
  cells: Sequence[Cell],
  *,
  covariates: str = "all",
  seed: int = 7,
  kl_mc: float = 0.1,
  n_points: int = 2,
  tau: float = 0.5,
) -> list[dict[str, object]]:
  rows: list[dict[str, object]] = []
  for family, scenario, n, rep in cells:
    for quantity in ("pdf", "hfunc1", "hfunc2"):
      for point in range(n_points):
        pdf = quantity == "pdf"
        rows.append(
          {
            "family": family,
            "scenario": scenario,
            "n": n,
            "rep": rep,
            "seed": seed,
            "label": label,
            "covariates": covariates,
            "error": "",
            "quantity": quantity,
            "point": point,
            "tau_true": tau,
            "IAE": 0.2,
            "KL_mc": kl_mc if pdf else math.nan,
            "KL_mc_n": 2000.0 if pdf else math.nan,
          }
        )
  return rows


def _run(
  path: Path, rows: list[dict[str, object]], *, uv_grid_n: int = 20
) -> Path:
  """A run folder holding ``rows``, one checkpoint per cell, as CSV."""
  path.mkdir(parents=True)
  grid = {"n_points": 2, "uv_grid_n": uv_grid_n, "holdout_n": 2000}
  (path / "config.json").write_text(json.dumps({"grid": grid}))
  frame = pd.DataFrame(rows)
  for keys, cell in frame.groupby(analyze.CELL):
    family, scenario, n, rep = cast("Cell", keys)
    folder = path / "cells" / f"{family}__{scenario}__n{n}__rep{rep}"
    folder.mkdir(parents=True)
    cell.to_csv(folder / "metrics.csv", index=False)
    (folder / "DONE").touch()
  return path


def test_runs_on_other_seeds_are_refused(tmp_path: Path) -> None:
  """A difference between fits on different data is not a paired one."""
  npcc = _run(tmp_path / "a", _rows(analyze.BASELINE, CELLS))
  r = _run(tmp_path / "b", _rows("gamcopula-select", CELLS, seed=8))
  with pytest.raises(SystemExit, match="one seed"):
    analyze.load([npcc, r])


def test_runs_scored_at_other_points_are_refused(tmp_path: Path) -> None:
  """Two estimators scored at different points do not differ by method alone.

  The same base seed with another ``n_points`` gives the same cells and
  seeds but other points, and other true taus at them.
  """
  npcc = _run(tmp_path / "a", _rows(analyze.BASELINE, CELLS))
  r = _run(tmp_path / "b", _rows("gamcopula-select", CELLS, n_points=1))
  with pytest.raises(SystemExit, match="same points"):
    analyze.load([npcc, r])
  r = _run(tmp_path / "c", _rows("gamcopula-select", CELLS, tau=0.4))
  with pytest.raises(SystemExit, match="one true tau"):
    analyze.load([npcc, r])


def test_runs_scored_on_other_grids_are_refused(tmp_path: Path) -> None:
  """An IAE on a 10 x 10 grid is not comparable to one on a 20 x 20 grid."""
  npcc = _run(tmp_path / "a", _rows(analyze.BASELINE, CELLS))
  r = _run(tmp_path / "b", _rows("gamcopula-select", CELLS), uv_grid_n=10)
  with pytest.raises(SystemExit, match="uv_grid_n"):
    analyze.load([npcc, r])
  (r / "config.json").unlink()
  with pytest.raises(SystemExit, match="config.json"):
    analyze.load([npcc, r])


def test_every_shipped_e9_estimator_has_a_display_name() -> None:
  """An estimator without one would reach the paper's tables as its label."""
  root = Path(__file__).resolve().parents[2] / "configs"
  labels = {
    est.label
    for config in sorted(root.glob("e9_covariates*.toml"))
    for est in load_covariate_grid(config).estimators
  }
  assert "tabpfn-quantiles-v3.5-probit" in labels
  assert labels <= set(analyze.DISPLAY)


def test_a_third_run_pairs_with_the_baseline(tmp_path: Path) -> None:
  """An arm run from its own config must still be paired with TabPFN v3.

  The v3.5 arm comes in through ``--with`` beside the R run; were it read
  as its own baseline, or dropped, its difference would be missing or zero.
  """
  npcc = _run(tmp_path / "a", _rows(analyze.BASELINE, CELLS, kl_mc=0.1))
  r = _run(tmp_path / "b", _rows("gamcopula-select", CELLS, kl_mc=0.3))
  v35 = _run(
    tmp_path / "c", _rows("tabpfn-quantiles-v3.5-probit", CELLS, kl_mc=0.15)
  )
  metrics = analyze.load([npcc, r, v35])
  kl = analyze.paired(
    analyze.per_rep(metrics, "pdf", "KL_mc"), "KL_mc", ["scenario", "n"]
  )
  row = kl[kl["estimator"] == "TabPFN v3.5"]
  assert len(row) == len(cs.SCENARIOS)
  assert row["diff_vs_tabpfn"].to_list() == pytest.approx(
    [0.05] * len(cs.SCENARIOS)
  )
  assert row["reps"].to_list() == [2] * len(cs.SCENARIOS)


def test_an_estimator_without_the_baselines_cells_is_refused(
  tmp_path: Path,
) -> None:
  """A mean over fewer cells than the baseline's is not comparable to it.

  With a family missing, the difference would be one family's mean minus
  both families', and nothing in the tables would say so.
  """
  clayton = [cell for cell in CELLS if cell[0] == "clayton"]
  npcc = _run(tmp_path / "a", _rows(analyze.BASELINE, CELLS))
  r = _run(tmp_path / "b", _rows("gamcopula-select", clayton))
  with pytest.raises(SystemExit, match="must have every cell"):
    analyze.load([npcc, r])


def test_the_active_arm_is_not_expected_where_nothing_is_active(
  tmp_path: Path,
) -> None:
  """The arm on the active covariates is not fitted where there are none."""
  fitted = [cell for cell in CELLS if cs.ACTIVE[cell[1]]]
  rows = [
    *_rows(analyze.BASELINE, CELLS),
    *_rows(f"{analyze.BASELINE}-active", fitted, covariates="active"),
  ]
  metrics = analyze.load([_run(tmp_path / "a", rows)])
  assert set(metrics["scenario"]) == set(cs.SCENARIOS)


@pytest.mark.parametrize("value", [math.nan, -math.inf])
def test_a_non_finite_score_is_a_failed_fit(
  tmp_path: Path, value: float
) -> None:
  """A non-finite score at one point must not leave a mean over the others.

  NaN drops out of a mean silently, and an infinite density's KL_mc is
  ``-inf``, which is not missing; either would score the fit on what it got
  right.
  """
  bad, good = CELLS[0], CELLS[1:]
  rows = [
    *_rows(analyze.BASELINE, CELLS),
    *_rows("lightgbm-probit", good),
    *_rows("lightgbm-probit", [bad], kl_mc=value),
  ]
  metrics = analyze.load([_run(tmp_path / "a", rows)])
  fit = metrics[
    (metrics["label"] == "lightgbm-probit")
    & (metrics["family"] == bad[0])
    & (metrics["scenario"] == bad[1])
    & (metrics["rep"] == bad[3])
  ]
  assert fit["error"].str.startswith("non-finite").all()
  reps = analyze.per_rep(metrics, "pdf", "KL_mc")
  table = analyze.paired(reps, "KL_mc", ["family", "scenario", "n"])
  row = table[
    (table["label"] == "lightgbm-probit")
    & (table["family"] == bad[0])
    & (table["scenario"] == bad[1])
  ].iloc[0]
  assert row["failed"] == 1
  assert row["reps"] == 1
  assert row["mean"] == pytest.approx(0.1)


@pytest.mark.parametrize(
  ("family", "scenario", "expected"),
  [
    ("gaussian", "gampcc", {1}),
    ("gaussian", "constant", {1}),
    ("clayton", "gampcc", {301}),
    ("clayton", "interaction", {301, 302}),
    ("clayton", "constant", {301, 302}),
  ],
)
def test_both_double_claytons_are_true_where_tau_is_positive(
  family: str, scenario: str, expected: set[int]
) -> None:
  """301 and 302 differ only below zero, where they rotate differently.

  Where every true tau is positive both are the Clayton, and a method that
  picks 302 there has picked the true copula; under ``gampcc`` only 301
  rotates by 90 degrees as the truth does.
  """
  assert analyze.true_families(family, scenario) == expected
  if scenario not in analyze.SIGN_CHANGING:
    x = cs.covariates(500, torch.Generator().manual_seed(0))
    assert (cs.kendall_tau(family, scenario, x) > 0).all()
