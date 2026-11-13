"""The surface-pressure check, on hand-made stations with a known step."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from npcc.experiments.application.data import StationData
from npcc.experiments.application.store import ensure_station_data

_SCRIPT = (
  Path(__file__).resolve().parents[2]
  / "scripts"
  / "paper"
  / "application"
  / "pressure_shift.py"
)
_OBSERVED = "obs:surface_air_pressure"
_FORECAST = "coe:surface_air_pressure_ensavg"


def _module() -> ModuleType:
  spec = importlib.util.spec_from_file_location("pressure_shift", _SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _store(
  run: Path, station: int, errors: np.ndarray, n_train: int
) -> np.ndarray:
  """A station whose pressure error is ``errors``, in time order.

  The forecast varies and the observation is the forecast plus the error, so
  the error is recovered only by differencing the right two columns. A
  second, unrelated covariate and response sit beside them in sorted order.
  """
  n = len(errors)
  times = np.datetime64("2017-01-01") + np.arange(n).astype("timedelta64[D]")
  rng = np.random.default_rng(station)
  forecast = 950.0 + 10.0 * rng.standard_normal(n)
  temperature = rng.standard_normal(n)
  x = np.column_stack([temperature, forecast])
  y = np.column_stack([temperature + 1.0, forecast + errors])
  data = StationData(
    station=station,
    feature_names=("coe:air_temperature_ensavg", _FORECAST),
    variables=("obs:air_temperature", _OBSERVED),
    x_train=x[:n_train],
    x_test=x[n_train:],
    y_train=y[:n_train],
    y_test=y[n_train:],
    time_train=times[:n_train].astype("datetime64[ns]"),
    time_test=times[n_train:].astype("datetime64[ns]"),
    row_train=np.arange(n_train, dtype=np.int64),
    row_test=np.arange(n_train, n, dtype=np.int64),
  )
  ensure_station_data(run, data, seconds=0.0)
  return times


def test_a_step_inside_the_training_period_is_placed_and_sized(
  tmp_path: Path,
) -> None:
  """A barometer step must be found where it is, not at the split.

  The step falls 40 cases before the end of training, so the train/test
  means differ by less than the step itself; a check that only compared the
  splits would understate it and misdate it.
  """
  ps = _module()
  run = tmp_path / "run"
  n, n_train, at = 400, 300, 260
  rng = np.random.default_rng(0)
  errors = 0.1 * rng.standard_normal(n)
  errors[at:] -= 5.0
  times = _store(run, 106, errors, n_train)
  _store(run, 7, 0.1 * rng.standard_normal(n), n_train)

  table = ps.build(run, tmp_path / "checks").set_index("station")

  row = table.loc[106]
  assert pd.Timestamp(row["step_time"]) == pd.Timestamp(times[at])
  assert row["n_before"] == at
  assert row["n_train_after"] == n_train - at
  assert row["step"] == pytest.approx(errors[at:].mean() - errors[:at].mean())
  assert row["shift"] == pytest.approx(
    errors[n_train:].mean() - errors[:n_train].mean()
  )
  assert abs(row["shift"]) < abs(row["step"])
  assert abs(table.loc[7, "shift"]) < 0.1
  written = pd.read_csv(tmp_path / "checks" / "pressure_shift.csv")
  assert sorted(written["station"]) == [7, 106]


def test_too_few_cases_to_place_a_step_are_refused() -> None:
  """A step needs ``min_side`` cases on each side, or its mean is noise."""
  ps = _module()
  with pytest.raises(ValueError, match="at least 2 \\* min_side = 60"):
    ps.largest_step(np.zeros(59), min_side=30)


def test_a_run_without_stored_stations_is_refused(tmp_path: Path) -> None:
  """An empty run must not yield an empty table that reads as no shift."""
  ps = _module()
  with pytest.raises(FileNotFoundError, match="must hold stored station"):
    ps.build(tmp_path / "run", tmp_path / "checks")
