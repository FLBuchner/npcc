"""Vine config: the shipped TOML loads, and bad grids are refused."""

from __future__ import annotations

from pathlib import Path

import pytest

from npcc.experiments.vine_config import VineGridConfig, load_vine_grid


def test_shipped_vine_config_loads() -> None:
  """The committed vine config must load and match the study design."""
  cfg_path = Path(__file__).resolve().parents[2] / "configs" / "vine_study.toml"
  grid = load_vine_grid(cfg_path)
  assert grid.dims == [3, 5]
  assert grid.arms == ["oracle", "random", "tll"]
  assert grid.estimators


@pytest.mark.parametrize(
  ("overrides", "match"),
  [
    ({"arms": ["oracle", "bogus"]}, "arms must be drawn"),
    ({"arms": ["tll", "tll"]}, "arms must be unique"),
    ({"dims": [1]}, "dims must be"),
    ({"eval_m": 5}, "eval_m must be"),
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
    VineGridConfig(**kwargs)  # ty: ignore[invalid-argument-type] - a test feeds deliberately wrong values
