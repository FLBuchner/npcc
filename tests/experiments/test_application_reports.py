"""The application's report script, on a finished synthetic run."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from npcc.experiments.application.cli import main
from npcc.experiments.application.marginal_set import MarginalSet
from tests.experiments._application import (
  MODELS,
  FakeCopulas,
  config_text,
  write_data,
)

_SCRIPT = (
  Path(__file__).resolve().parents[2]
  / "scripts"
  / "paper"
  / "application"
  / "report.py"
)
_CHAIN = "npcc-chain-linear-gaussian-all"
_ARMS = ("independence", "gaussian", _CHAIN, "independence@5d", "gaussian@5d")
# The paper's designs: the triple modeled and two responses derived from it,
# and all five modeled at once.
_RESPONSES = (
  'modeled = ["obs:air_temperature", "obs:relative_humidity", '
  '"obs:surface_air_pressure"]\n'
  'derived = ["obs:dew_point_temperature", "obs:water_vapor_mixing_ratio"]\n'
)
_ALL = (
  "obs:air_temperature",
  "obs:dew_point_temperature",
  "obs:relative_humidity",
  "obs:surface_air_pressure",
  "obs:water_vapor_mixing_ratio",
)
_DESIGNS = (
  f"[copulas.designs.5d]\nmodeled = {json.dumps(list(_ALL))}\nderived = []\n\n"
)


def _report() -> ModuleType:
  spec = importlib.util.spec_from_file_location("application_report", _SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_the_report_reads_a_finished_run_and_fits_nothing(
  tmp_path: Path,
  register_linear_gaussian_backend: None,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A report that breaks on a run's own tables fails once the run is done.

  The run has the paper's layout: the triple with two derived responses
  and a 5d design, a chain arm whose order has a log score alone, and the
  Gaussian copula failed at one station on EMOS's margins, in both designs.
  The paper's table must leave that arm out on those margins, keep every
  other result with the tables' own numbers, and not report the 5d log
  score, which has no density to score. A station the run does not hold is
  refused before anything is written, rather than leaving half a report.
  """
  del register_linear_gaussian_backend
  copulas = FakeCopulas(
    fail={(1, "emos-seasonal", "gaussian")},
    arms=("independence", "gaussian", _CHAIN),
  )
  monkeypatch.setitem(
    sys.modules, "npcc.experiments.application.copulas", copulas.module()
  )
  # The helper's fake stage records `ms.train_pits`, which MarginalSet does
  # not have: its training PITs are out of fold, with no attribute to say so.
  monkeypatch.setattr(MarginalSet, "train_pits", "out_of_fold", raising=False)
  text = config_text(
    write_data(tmp_path / "data"), arms=_ARMS, stations="[0, 1]"
  )
  text = text.replace("[responses]\n", "[responses]\n" + _RESPONSES)
  text = text.replace("[[estimators]]", _DESIGNS + "[[estimators]]", 1)
  config = tmp_path / "app.toml"
  config.write_text(text)
  run = tmp_path / "run"
  for stage in ("margins", "copulas", "evaluate"):
    assert main([stage, "--config", str(config), "--out", str(run)]) == 0

  report = _report()
  bad = tmp_path / "bad"
  with pytest.raises(ValueError, match=r"station must be one of \[0, 1\]"):
    report.main(["--run", str(run), "--out", str(bad), "--station", "7"])
  assert list(bad.iterdir()) == []

  assert report.main(["--run", str(run), "--station", "1"]) == 0
  assert (run / "reports" / "report.pdf").stat().st_size > 100_000
  table = pd.read_csv(run / "reports" / "copula_table.csv")
  results = {*_ARMS, f"{_CHAIN}-m1"}
  assert set(zip(table["model"], table["arm"], strict=True)) == {
    (m, a) for m in MODELS for a in results
  } - {("emos-seasonal", "gaussian"), ("emos-seasonal", "gaussian@5d")}
  assert {
    "joint_nll_modeled_gain_station_mean",
    "energy_skill_low",
    "crps_dew_point_temperature_skill_high",
  } <= set(table.columns)
  assert not any(c.startswith("joint_nll_5d") for c in table.columns)
  pooled = pd.read_csv(run / "tables" / "copulas_pooled.csv")
  merged = table.merge(pooled, on=["model", "arm"], suffixes=("", "_tables"))
  assert len(merged) == len(table)
  for column in ("joint_nll_modeled_gain", "energy_skill"):
    np.testing.assert_allclose(merged[column], merged[f"{column}_tables"])
