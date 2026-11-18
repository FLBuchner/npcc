"""Kendall's tau within covariate bins, on hand-made stored margins."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from npcc.experiments.application.data import StationData
from npcc.experiments.application.store import (
  ensure_station_data,
  quantile_columns,
  write_margin,
)

_SCRIPT = (
  Path(__file__).resolve().parents[2]
  / "scripts"
  / "paper"
  / "application"
  / "dependence.py"
)
_T, _RH, _P = (
  "obs:air_temperature",
  "obs:relative_humidity",
  "obs:surface_air_pressure",
)
_FEATURES = (
  "coe:air_temperature_ensavg",
  "coe:relative_humidity_ensavg",
  "coe:surface_air_pressure_ensavg",
)
_MODELS = ("tabpfn-quantiles-all", "emos-seasonal")
_LEVELS = [0.25, 0.5, 0.75]


def _module() -> ModuleType:
  spec = importlib.util.spec_from_file_location("dependence", _SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _chain(rng: np.random.Generator, *rhos: np.ndarray) -> np.ndarray:
  """Uniform PITs whose normal scores each correlate with the previous one.

  Column ``k + 1``'s score correlates with column ``k``'s by ``rhos[k]``,
  case by case.
  """
  scores = [rng.standard_normal(len(rhos[0]))]
  for rho in rhos:
    noise = rng.standard_normal(len(rho))
    scores.append(rho * scores[-1] + np.sqrt(1.0 - rho**2) * noise)
  return stats.norm.cdf(np.column_stack(scores))


def _store(run: Path, station: int, *, out_of_fold: bool = True) -> None:
  """A station whose out-of-fold PITs depend where a bin says, and only there.

  T and RH depend (rho 0.9) in the cases valid in the first fifth of the
  year, and RH and p where RH's ensemble mean is in its top fifth; nowhere
  else, but for T and p where both hold. The in-sample PITs carry the opposite dependence, so reading them
  instead would flip each sign.
  """
  rng = np.random.default_rng(station)
  n_train, n_test = 4 * 365, 40
  n = n_train + n_test
  times = (np.datetime64("2017-01-01") + np.arange(n)).astype("datetime64[ns]")
  x = rng.normal(size=(n, len(_FEATURES)))
  dates = times.astype("datetime64[D]")
  day = (dates - dates.astype("datetime64[Y]")).astype(np.int64) + 1
  winter = day < 367 / 5 + 1
  humid = x[:, 1] > np.quantile(x[:n_train, 1], 0.8)
  oof = _chain(rng, np.where(winter, 0.9, 0.0), np.where(humid, 0.9, 0.0))
  # The same draws with RH's reversed, which reverses both planted pairs.
  in_sample = np.column_stack([oof[:, 0], 1.0 - oof[:, 1], oof[:, 2]])
  y = rng.normal(size=(n, 3))
  ensure_station_data(
    run,
    StationData(
      station=station,
      feature_names=_FEATURES,
      variables=(_T, _RH, _P),
      x_train=x[:n_train],
      x_test=x[n_train:],
      y_train=y[:n_train],
      y_test=y[n_train:],
      time_train=times[:n_train],
      time_test=times[n_train:],
      row_train=np.arange(n_train, dtype=np.int64),
      row_test=np.arange(n_train, n, dtype=np.int64),
    ),
    seconds=0.0,
  )
  for model in _MODELS:
    parts = []
    for j, variable in enumerate((_T, _RH, _P)):
      for split, rows in (
        ("train", slice(0, n_train)),
        ("test", slice(n_train, n)),
      ):
        part = pd.DataFrame(
          {
            "station": station,
            "model": model,
            "variable": variable,
            "split": split,
            "row": np.arange(n)[rows],
            "time": times[rows],
            "pit": in_sample[rows, j],
            "logpdf": np.nan if split == "train" else 0.0,
          }
        )
        if out_of_fold:
          part["pit_out_of_fold"] = oof[rows, j] if split == "train" else np.nan
        for k, column in enumerate(quantile_columns(len(_LEVELS))):
          part[column] = np.nan if split == "train" else float(k)
        parts.append(part)
    write_margin(
      run,
      station,
      model,
      predictions=pd.concat(parts, ignore_index=True),
      scores=pd.DataFrame({"row": [0]}),
      timings=pd.DataFrame({"stage": ["fit"], "seconds": [0.0]}),
      info={"levels": _LEVELS},
    )


def test_tau_is_read_off_the_out_of_fold_pits_in_the_right_bins(
  tmp_path: Path,
) -> None:
  """A bin misplaced, a covariate mismatched or the wrong PITs flip a finding.

  The planted dependence must appear in the first day-of-year bin for T and
  RH, and in the top bin of RH's ensemble mean for RH and p, positive, as
  the out-of-fold PITs carry it; elsewhere tau stays near 0.
  """
  run = tmp_path / "run"
  for station in (3, 8):
    _store(run, station)
  out = tmp_path / "tau"
  assert _module().main(["--run", str(run), "--out", str(out)]) == 0

  frame = pd.read_csv(out / "binned_kendall_tau.csv")
  assert set(frame["model"]) == set(_MODELS)
  assert set(frame["pits"]) == {"out_of_fold"}
  assert set(frame["station"]) == {3, 8}
  keyed = frame.set_index(
    ["station", "model", "variable_a", "variable_b", "binning", "bin"]
  ).sort_index()
  days = keyed.xs((3, _MODELS[0], _T, _RH, "day_of_year"))
  assert days.loc[0, "tau"] > 0.6
  assert (days.loc[1:, "tau"].abs() < 0.15).all()
  assert days["lower"].tolist() == pytest.approx(
    list(np.linspace(1, 367, 6)[:-1])
  )
  forecast = keyed.xs((3, _MODELS[0], _RH, _P, "forecast"))
  assert set(forecast["covariate"]) == {"coe:relative_humidity_ensavg"}
  assert forecast.loc[4, "tau"] > 0.6
  assert (forecast.loc[:3, "tau"].abs() < 0.15).all()
  assert forecast["n_cases"].sum() == 4 * 365
  # The pair (T, RH) is binned by T's own ensemble mean.
  assert set(keyed.xs((3, _MODELS[0], _T, _RH, "forecast"))["covariate"]) == {
    "coe:air_temperature_ensavg"
  }
  assert len(list(out.glob("*.png"))) == 3
  assert (out / "binned_kendall_tau.pdf").stat().st_size > 10_000


def test_a_bin_with_too_few_cases_keeps_its_count_and_no_tau(
  tmp_path: Path,
) -> None:
  """A tau of a handful of cases would read as a finding."""
  run = tmp_path / "run"
  _store(run, 0)
  frame = _module().build(
    run, tmp_path / "tau", models=_MODELS[:1], bins=5, min_pairs=400
  )
  assert frame["tau"].isna().all()
  assert (frame["n_pairs"] == frame["n_cases"]).all()
  assert (frame["n_cases"] > 200).all()


def test_margins_without_out_of_fold_pits_are_refused(tmp_path: Path) -> None:
  """In-sample PITs crowd the middle, and their tau is not the copula's."""
  run = tmp_path / "run"
  _store(run, 0, out_of_fold=False)
  with pytest.raises(ValueError, match="out-of-fold"):
    _module().build(run, tmp_path / "tau")


def test_the_day_of_year_bins_are_one_calendar_for_every_station() -> None:
  """Bins of each station's own quantiles would not line up across stations.

  A station with cases in January and December alone still gets the
  calendar's five bins, its cases in the first and the last.
  """
  times = np.array(
    ["2019-01-03", "2019-12-30", "2020-12-31", "2020-02-29"],
    dtype="datetime64[ns]",
  )
  labels, edges = _module().day_of_year_bins(times, 5)
  assert labels.tolist() == [0, 4, 4, 0]
  np.testing.assert_allclose(edges, np.linspace(1, 367, 6))
