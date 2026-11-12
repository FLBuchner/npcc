"""The application's primary summaries, on hand-made station means."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

_SCRIPT = (
  Path(__file__).resolve().parents[2]
  / "scripts"
  / "paper"
  / "application"
  / "paired_summary.py"
)
_EMOS, _TABPFN = "emos-seasonal", "tabpfn-quantiles-all"
_MIXTURE = "npcc-mixture-tabpfn-quantiles-all"


def _module() -> ModuleType:
  spec = importlib.util.spec_from_file_location("paired_summary", _SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _run(tmp_path: Path) -> Path:
  """Three stations; the mixture on the TabPFN margins alone, as decided."""
  means = {
    (_EMOS, "independence"): [7.0, 8.0, 9.0],
    (_EMOS, "gaussian"): [6.0, 8.5, 8.0],
    (_TABPFN, "independence"): [6.5, 7.5, 7.0],
    (_TABPFN, "gaussian"): [6.0, 7.0, 7.5],
    (_TABPFN, _MIXTURE): [5.0, 7.0, 9.5],
  }
  rows = [
    {
      "station": station,
      "model": model,
      "arm": arm,
      "n_cases": 10,
      "joint_nll_modeled": value,
      "energy": value / 10,
      "joint_nll_modeled_gain": 0.0,
      "energy_skill": 0.0,
    }
    for (model, arm), values in means.items()
    for station, value in zip((0, 1, 106), values, strict=True)
  ]
  by_station = pd.DataFrame(rows)
  tables = tmp_path / "run" / "tables"
  tables.mkdir(parents=True)
  by_station.to_csv(tables / "copulas_by_station.csv", index=False)
  pooled = (
    by_station.groupby(["model", "arm"], sort=False)
    .agg(
      n_cases=("n_cases", "sum"),
      n_stations=("station", "nunique"),
      joint_nll_modeled=("joint_nll_modeled", "mean"),
      energy=("energy", "mean"),
    )
    .reset_index()
  )
  pooled.to_csv(tables / "copulas_pooled.csv", index=False)
  return tmp_path / "run"


def test_the_paired_summary_pairs_by_station_against_both_references(
  tmp_path: Path,
) -> None:
  """A pairing on the wrong key, or a sign flipped, would reverse a ranking.

  Each arm's difference is arm minus reference at the same station: against
  the Gaussian copula on its own margins, and against the TabPFN mixture
  across margins; the reference itself is never compared with itself, and
  the robustness check drops station 106 and nothing else.
  """
  run = _run(tmp_path)
  written = _module().build(run, run / "analysis")
  assert {p.name for p in written} == {
    "pooled_means.csv",
    "pooled_means.md",
    "paired_by_station.csv",
    "paired.csv",
  }
  summary = pd.read_csv(run / "analysis" / "paired.csv")
  nll = summary[summary["score"] == "joint_nll_modeled"].set_index(
    ["stations", "model", "arm", "reference"]
  )

  same = nll.loc[("all", _TABPFN, _MIXTURE, "gaussian (same margins)")]
  diffs = [5.0 - 6.0, 7.0 - 7.0, 9.5 - 7.5]
  mean = sum(diffs) / 3
  sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / 2)
  assert same["mean_diff"] == pytest.approx(mean)
  assert same["se"] == pytest.approx(sd / math.sqrt(3))
  assert (same["lower"], same["higher"], same["level"]) == (1, 1, 1)

  across = nll.loc[("all", _EMOS, "gaussian", f"{_MIXTURE} on {_TABPFN}")]
  assert across["mean_diff"] == pytest.approx(
    ((6.0 - 5.0) + (8.5 - 7.0) + (8.0 - 9.5)) / 3
  )
  assert across["n_stations"] == 3

  kept = nll.loc[("without 106", _EMOS, "gaussian", f"{_MIXTURE} on {_TABPFN}")]
  assert kept["n_stations"] == 2
  assert kept["mean_diff"] == pytest.approx(((6.0 - 5.0) + (8.5 - 7.0)) / 2)
  assert (kept["lower"], kept["higher"]) == (0, 2)

  references = summary.groupby("reference")["arm"].apply(set).to_dict()
  assert "gaussian" not in references["gaussian (same margins)"]
  assert ("emos-seasonal", _MIXTURE) not in set(
    zip(summary["model"], summary["arm"], strict=True)
  )
  pairs = set(
    zip(
      summary.loc[summary["reference"] != "gaussian (same margins)", "model"],
      summary.loc[summary["reference"] != "gaussian (same margins)", "arm"],
      strict=True,
    )
  )
  assert (_TABPFN, _MIXTURE) not in pairs
  assert set(summary["score"]) == {"joint_nll_modeled", "energy"}


def test_the_pooled_tables_hold_the_means_and_no_skill_columns(
  tmp_path: Path,
) -> None:
  """A skill or gain column read as a score would be summarized as one."""
  run = _run(tmp_path)
  _module().build(run, run / "analysis")
  pooled = pd.read_csv(run / "analysis" / "pooled_means.csv")
  assert list(pooled.columns) == [
    "model",
    "arm",
    "n_cases",
    "n_stations",
    "joint_nll_modeled",
    "energy",
  ]
  text = (run / "analysis" / "pooled_means.md").read_text()
  assert "### joint_nll_modeled" in text and "### energy" in text
  assert f"| {_MIXTURE} |  | 7.1667 |" in text


def test_a_run_not_yet_evaluated_is_refused(tmp_path: Path) -> None:
  """A summary of a folder without tables must say what to run first."""
  with pytest.raises(FileNotFoundError, match="evaluate"):
    _module().build(tmp_path, tmp_path / "analysis")
