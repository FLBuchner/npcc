"""Vine config: the shipped TOML loads, and bad grids are refused."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from npcc.experiments.config import load_grid
from npcc.experiments.vine_config import VineGridConfig, load_vine_grid

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_shipped_vine_config_loads() -> None:
  """The committed vine config must load and fit E1's main-text estimator.

  The vine study reports the backend and transform of E1's main text; one
  that drifted from E1's estimator id would describe a different model.
  """
  grid = load_vine_grid(_CONFIGS / "vine_study.toml")
  assert grid.dims == [3, 5]
  assert grid.n_rep == 20
  assert grid.arms == ["oracle", "random", "tll"]
  assert [e.label for e in grid.estimators] == ["tabpfn-quantiles-v3-probit"]
  e1 = load_grid(_CONFIGS / "study.toml")
  assert {e.estimator_id for e in grid.estimators} <= {
    e.estimator_id for e in e1.estimators
  }


def test_the_selected_arm_runs_on_e2s_cells() -> None:
  """The selected arm must fit E2's cells with E2's estimator.

  It runs from its own config and is paired with the oracle and random arms
  of E2's run, so any other drift from ``vine_study.toml`` would score it on
  other data, test points or pair copulas.
  """
  e2 = load_vine_grid(_CONFIGS / "vine_study.toml")
  selected = load_vine_grid(_CONFIGS / "vine_study_selected.toml")
  assert selected.arms == ["selected"]
  for f in fields(VineGridConfig):
    if f.name != "arms":
      assert getattr(selected, f.name) == getattr(e2, f.name), f.name


def test_e2_at_d10_is_e2_at_another_dimension() -> None:
  """E2 at d = 10 must differ from E2 only in its dimension and its arms.

  Any other drift -- the sample sizes, repetitions, evaluation points,
  projection grid or estimator -- would make its numbers incomparable with
  §3.4's at d <= 5.
  """
  e2 = load_vine_grid(_CONFIGS / "vine_study.toml")
  d10 = load_vine_grid(_CONFIGS / "vine_study_d10.toml")
  assert d10.dims == [10]
  assert d10.arms == ["oracle", "random", "selected", "tll"]
  for f in fields(VineGridConfig):
    if f.name not in {"dims", "arms"}:
      assert getattr(d10, f.name) == getattr(e2, f.name), f.name


@pytest.mark.parametrize(
  ("name", "dims", "e2_name"),
  [
    ("vine_study_mixture.toml", [3, 5], "vine_study.toml"),
    ("vine_study_mixture_d10.toml", [10], "vine_study_d10.toml"),
  ],
)
def test_e10_mixtures_run_on_e2s_cells(
  name: str, dims: list[int], e2_name: str
) -> None:
  """E10's mixtures must fit E2's cells with E2's estimator.

  The mixture is read against E2's arms at the same (d, n, rep), and its
  first member against the selected arm there; any other drift would score
  it on other data, test points or pair copulas.
  """
  e2 = load_vine_grid(_CONFIGS / e2_name)
  e10 = load_vine_grid(_CONFIGS / name)
  assert e10.dims == dims == e2.dims
  assert e10.n == [500, 1000, 2000]
  assert e10.arms == ["mixture"]
  assert e10.mixture_k == 5
  for f in fields(VineGridConfig):
    if f.name not in {"n", "arms", "mixture_k"}:
      assert getattr(e10, f.name) == getattr(e2, f.name), f.name


@pytest.mark.parametrize(
  ("overrides", "match"),
  [
    ({"arms": ["oracle", "bogus"]}, "arms must be drawn"),
    ({"arms": ["tll", "tll"]}, "arms must be unique"),
    ({"dims": [1]}, "dims must be"),
    ({"eval_m": 5}, "eval_m must be"),
    ({"mixture_k": 0}, "mixture_k must be"),
  ],
)
def test_bad_grid_is_refused(overrides: dict[str, object], match: str) -> None:
  """An unknown or repeated arm would silently drop or double-count a column."""
  kwargs: dict[str, object] = {
    "dims": [3],
    "n": [40],
    "n_rep": 1,
    "estimators": [],
    "arms": ["tll"],
  }
  kwargs.update(overrides)
  with pytest.raises(ValueError, match=match):
    VineGridConfig(**kwargs)  # ty: ignore[invalid-argument-type] - mistyped values exercise the guard
