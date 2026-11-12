"""The application's report scripts, on a finished synthetic run."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from npcc.experiments.application.cli import main
from npcc.experiments.application.store import read_station_data
from tests.experiments._application import (
  MODELS,
  FakeCopulas,
  write_config,
  write_data,
)

_SCRIPTS = (
  Path(__file__).resolve().parents[2] / "scripts" / "paper" / "application"
)


def _script(name: str) -> ModuleType:
  spec = importlib.util.spec_from_file_location(
    f"application_{name}", _SCRIPTS / f"{name}.py"
  )
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_the_reports_read_a_finished_run_and_fit_nothing(
  tmp_path: Path,
  register_linear_gaussian_backend: None,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A report that breaks on the run's own tables fails only once the run is done.

  Each script reads the tables ``evaluate`` wrote and the stored units, with
  a failed copula arm among them, and writes its files.
  """
  del register_linear_gaussian_backend
  copulas = FakeCopulas(fail={(1, "emos-seasonal", "gaussian")})
  monkeypatch.setitem(
    sys.modules, "npcc.experiments.application.copulas", copulas.module()
  )
  config = write_config(
    tmp_path / "app.toml", write_data(tmp_path / "data"), stations="[0, 1]"
  )
  run = tmp_path / "run"
  for stage in ("margins", "copulas", "evaluate"):
    assert main([stage, "--config", str(config), "--out", str(run)]) == 0

  variables = ["obs:relative_humidity", "obs:air_temperature"]
  margins = _script("report_margins")
  assert margins.main(["--run", str(run), "--variables", *variables]) == 0
  assert (run / "reports" / "marginal_report.pdf").stat().st_size > 10_000
  assert _script("report_copulas").main(["--run", str(run)]) == 0
  assert (run / "reports" / "copula_report.pdf").stat().st_size > 10_000
  table = pd.read_csv(run / "reports" / "copula_table.csv")
  assert set(zip(table["model"], table["arm"], strict=True)) == {
    (m, a) for m in MODELS for a in ("independence", "gaussian")
  } - {("emos-seasonal", "gaussian")}
  assert {"energy_skill_low", "joint_nll_gain_station_mean"} <= set(
    table.columns
  )

  out = tmp_path / "tau"
  rolling = _script("plot_rolling_tau")
  argv = [
    "--run",
    str(run),
    "--station",
    "1",
    "--window",
    "150",
    "--out",
    str(out),
  ]
  argv += ["--models", "emos-seasonal", "--variables", *variables]
  assert rolling.main(argv) == 0
  tau = pd.read_csv(out / "rolling_kendall_tau.csv")
  assert set(tau["series"]) == {"observations", "emos-seasonal"}
  assert tau["tau"].between(-1, 1).all()
  assert len(tau) == 2 * (len(read_station_data(run, 1).row_train) - 149)
  assert (out / "rolling_kendall_tau_comparison.pdf").exists()
  assert len(list((out / "comparison").glob("*.png"))) == 1
