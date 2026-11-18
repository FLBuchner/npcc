"""The application's primary summaries, on hand-made station means and cases."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from npcc.experiments.application.scores import date_block_interval
from npcc.experiments.application.store import copula_dir

_SCRIPT = (
  Path(__file__).resolve().parents[2]
  / "scripts"
  / "paper"
  / "application"
  / "paired_summary.py"
)
_EMOS, _TABPFN = "emos-seasonal", "tabpfn-quantiles-all"
_MIXTURE = "npcc-mixture-tabpfn-quantiles-all"
_STATIONS = (0, 1, 106)
_N_CASES = 10
#: Station 106's test dates start three days later, so the calendar of
#: dates is not any one station's.
_FIRST = {0: "2021-01-02", 1: "2021-01-02", 106: "2021-01-05"}


def _module() -> ModuleType:
  spec = importlib.util.spec_from_file_location("paired_summary", _SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _case_scores(station: int, arm_index: int, mean: float) -> pd.DataFrame:
  """Ten dated test cases whose scores average to ``mean`` and ``mean / 10``.

  Each arm's cases spread around its mean by its own slope, so a paired
  difference varies from date to date; the offsets are multiples of 1/4,
  which sum to zero exactly.
  """
  offsets = 0.25 * (arm_index + 1) * (np.arange(_N_CASES) - 4.5)
  first = np.datetime64(_FIRST[station], "D")
  return pd.DataFrame(
    {
      "row": 1000 * station + np.arange(_N_CASES),
      "time": (first + np.arange(_N_CASES)).astype("datetime64[ns]"),
      "joint_nll_modeled": mean + offsets,
      "energy": (mean + offsets[::-1]) / 10,
    }
  )


def _run(tmp_path: Path) -> Path:
  """Three stations; the mixture on the TabPFN margins alone, as decided.

  The tables hold each station's means, and the copula units each test
  case's scores, as ``evaluate`` leaves a run.
  """
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
      "n_cases": _N_CASES,
      "joint_nll_modeled": value,
      "energy": value / 10,
      "joint_nll_modeled_gain": 0.0,
      "energy_skill": 0.0,
    }
    for (model, arm), values in means.items()
    for station, value in zip(_STATIONS, values, strict=True)
  ]
  for index, ((model, arm), values) in enumerate(means.items()):
    for station, value in zip(_STATIONS, values, strict=True):
      unit = copula_dir(tmp_path / "run", station, model, arm)
      unit.mkdir(parents=True)
      _case_scores(station, index, value).assign(
        station=station, model=model, run_arm=arm, arm=arm
      ).to_parquet(unit / "scores.parquet", index=False)
  by_station = pd.DataFrame(rows)
  tables = tmp_path / "run" / "tables"
  tables.mkdir(parents=True, exist_ok=True)
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
    "paired.md",
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


def test_the_interval_resamples_each_paired_case_on_shared_dates(
  tmp_path: Path,
) -> None:
  """An interval from mispaired cases, or from dates drawn per station, errs.

  Each comparison's interval must be the block bootstrap of its case-by-case
  differences (arm less reference at the same station and date), and the
  columns must follow the summary's own, which keep their values. The
  markdown shows the difference with its interval.
  """
  run = _run(tmp_path)
  _module().build(run, run / "analysis")
  summary = pd.read_csv(run / "analysis" / "paired.csv")
  assert list(summary.columns) == [
    "stations",
    "model",
    "arm",
    "reference",
    "score",
    "n_stations",
    "mean_diff",
    "se",
    "lower",
    "higher",
    "level",
    "boot_low",
    "boot_high",
  ]
  stations, dates, diffs = [], [], []
  for station, (mixture, gaussian) in zip(
    _STATIONS, [(5.0, 6.0), (7.0, 7.0), (9.5, 7.5)], strict=True
  ):
    # The mixture is the fifth arm of _run's table, the TabPFN Gaussian
    # copula the fourth.
    a, r = _case_scores(station, 4, mixture), _case_scores(station, 3, gaussian)
    stations.append(np.full(_N_CASES, station))
    dates.append(a["time"].to_numpy().astype("datetime64[D]"))
    diffs.append(a["joint_nll_modeled"].to_numpy() - r["joint_nll_modeled"])
  expected = {
    label: date_block_interval(
      np.concatenate(stations)[keep],
      np.concatenate(dates)[keep],
      np.concatenate(diffs)[keep],
    )
    for label, keep in (
      ("all", np.full(3 * _N_CASES, True)),
      ("without 106", np.concatenate(stations) != 106),
    )
  }
  rows = summary.set_index(["stations", "model", "arm", "reference", "score"])
  for label, (mean, low, high) in expected.items():
    row = rows.loc[
      (label, _TABPFN, _MIXTURE, "gaussian (same margins)", "joint_nll_modeled")
    ]
    assert row["mean_diff"] == pytest.approx(mean)
    assert (row["boot_low"], row["boot_high"]) == pytest.approx((low, high))
    assert low < high
  assert summary[["boot_low", "boot_high"]].notna().all().all()
  text = (run / "analysis" / "paired.md").read_text()
  low, high = expected["all"][1:]
  assert f"| {_TABPFN} | {_MIXTURE} | gaussian (same margins) | 0.3333 " in text
  assert f"[{low:.4g}, {high:.4g}] | 1 / 1 / 1 of 3 |" in text
  assert "### joint_nll_modeled, without 106" in text


def test_case_scores_that_are_not_the_tables_are_refused(
  tmp_path: Path,
) -> None:
  """An interval of other cases than the summarized ones would misstate it.

  A unit rewritten after ``evaluate``, or one gone, must stop the summary.
  """
  run = _run(tmp_path)
  unit = copula_dir(run, 1, _EMOS, "gaussian") / "scores.parquet"
  cases = pd.read_parquet(unit)
  cases.loc[0, "energy"] += 1.0
  cases.to_parquet(unit, index=False)
  with pytest.raises(ValueError, match="evaluate"):
    _module().build(run, run / "analysis")
  unit.unlink()
  with pytest.raises(FileNotFoundError, match="scores.parquet"):
    _module().build(run, run / "analysis")
