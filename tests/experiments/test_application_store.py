"""The run folder: units written whole, read back exactly, refused if stale."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from npcc.experiments.application.config import ApplicationConfig, load_config
from npcc.experiments.application.data import StationData, load_station
from npcc.experiments.application.margins import StationMargins, fit_station
from npcc.experiments.application.store import (
  copula_dir,
  copula_status,
  ensure_station_data,
  margin_dir,
  read_copula_tables,
  read_margin_table,
  read_marginal_set,
  read_station_data,
  start_run,
  stored_stations,
  write_copula,
  write_copula_failure,
  write_margin,
)
from tests.experiments._application import (
  FakeCopulaFailure,
  FakeCopulas,
  write_config,
  write_data,
)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
  return write_data(tmp_path_factory.mktemp("data"))


@pytest.fixture
def config(
  data_dir: Path, tmp_path: Path, register_linear_gaussian_backend: None
) -> ApplicationConfig:
  del register_linear_gaussian_backend
  return load_config(write_config(tmp_path / "app.toml", data_dir))


@pytest.fixture
def station(data_dir: Path, config: ApplicationConfig) -> StationData:
  return load_station(data_dir, 1, config.selection, sha256=None)


@pytest.fixture
def margins(station: StationData, config: ApplicationConfig) -> StationMargins:
  return fit_station(
    station,
    config.marginal_model("linear-gaussian-all"),
    config=config,
    device="cpu",
  )


def _store(
  out: Path, station: StationData, margins: StationMargins, model: str
) -> None:
  ensure_station_data(out, station, seconds=0.5)
  write_margin(
    out,
    station.station,
    model,
    predictions=margins.predictions,
    scores=margins.scores,
    timings=margins.timings,
    info=margins.info,
  )


def test_a_marginal_set_reads_back_exactly_what_was_fitted(
  tmp_path: Path, station: StationData, margins: StationMargins
) -> None:
  """The copula stage must see the margins' own numbers, bit for bit.

  Every array is aligned on the station's cases in time order and on its
  responses; a value rounded or a row reordered on the way through the
  store would move every copula score.
  """
  _store(tmp_path, station, margins, "linear-gaussian-all")
  ms = read_marginal_set(tmp_path, station.station, "linear-gaussian-all")

  assert ms.station == station.station
  assert ms.model == "linear-gaussian-all"
  assert ms.variables == station.variables
  assert ms.feature_names == station.feature_names
  for name in (
    "x_train",
    "x_test",
    "y_train",
    "y_test",
    "time_train",
    "time_test",
  ):
    np.testing.assert_array_equal(getattr(ms, name), getattr(station, name))
  predictions = margins.predictions
  levels = np.asarray(margins.info["levels"])
  np.testing.assert_array_equal(ms.levels, levels)
  for j, variable in enumerate(station.variables):
    rows = predictions.loc[predictions["variable"] == variable]
    train = rows.loc[rows["split"] == "train"]
    test = rows.loc[rows["split"] == "test"]
    np.testing.assert_array_equal(ms.pit_train[:, j], train["pit"])
    np.testing.assert_array_equal(ms.pit_test[:, j], test["pit"])
    np.testing.assert_array_equal(ms.logpdf_test[:, j], test["logpdf"])
    np.testing.assert_array_equal(
      ms.quantiles_test[:, :, j], test.filter(like="q_").to_numpy()
    )
  assert ms.quantiles_test.shape == (
    len(station.row_test),
    len(levels),
    len(station.variables),
  )


def test_a_unit_is_written_whole_or_not_at_all(
  tmp_path: Path, station: StationData, margins: StationMargins
) -> None:
  """A unit cut short must not read as finished on a resume."""
  ensure_station_data(tmp_path, station, seconds=0.5)
  bad = margins.scores.assign(observed=[object()] * len(margins.scores))

  with pytest.raises(Exception, match="object"):
    write_margin(
      tmp_path,
      station.station,
      "linear-gaussian-all",
      predictions=margins.predictions,
      scores=bad,
      timings=margins.timings,
      info=margins.info,
    )
  parent = margin_dir(tmp_path, station.station, "linear-gaussian-all").parent
  assert list(parent.iterdir()) == []


def test_a_unit_is_never_overwritten(
  tmp_path: Path, station: StationData, margins: StationMargins
) -> None:
  """Two processes on one station must not silently replace each other."""
  _store(tmp_path, station, margins, "linear-gaussian-all")
  with pytest.raises(FileExistsError, match="must not exist"):
    _store(tmp_path, station, margins, "linear-gaussian-all")


def test_cases_that_changed_under_stored_results_are_refused(
  tmp_path: Path, station: StationData
) -> None:
  """A resume on edited data would mix results from two datasets."""
  ensure_station_data(tmp_path, station, seconds=0.5)
  ensure_station_data(tmp_path, station, seconds=0.7)
  y = station.y_test.copy()
  y[4, 0] += 1e-9
  changed = dataclasses.replace(station, y_test=y)

  with pytest.raises(ValueError, match=r"\['y_test'\] differ"):
    ensure_station_data(tmp_path, changed, seconds=0.5)
  assert stored_stations(tmp_path) == (station.station,)
  stored = read_station_data(tmp_path, station.station)
  np.testing.assert_array_equal(stored.y_test, station.y_test)
  np.testing.assert_array_equal(stored.row_train, station.row_train)


def test_a_run_folder_refuses_another_configuration(
  tmp_path: Path, config: ApplicationConfig
) -> None:
  """Results under two configurations in one folder cannot be told apart.

  Each process records when it ran, how, on which commit and on which data.
  """
  start_run(tmp_path, config, {"stage": "margins"})
  start_run(
    tmp_path, dataclasses.replace(config, stations=(0,)), {"stage": "x"}
  )
  with pytest.raises(ValueError, match="fresh --out"):
    start_run(tmp_path, dataclasses.replace(config, base_seed=12), {})

  lines = (tmp_path / "provenance.jsonl").read_text().splitlines()
  assert len(lines) == 2
  record = json.loads(lines[0])
  assert record["options"] == {"stage": "margins"}
  assert set(record["data_sha256"]) == {"x.nc", "y.nc"}
  assert all(len(v) == 64 for v in record["data_sha256"].values())
  assert "npcc_commit" in record
  assert (tmp_path / "COMMIT").read_text().startswith("HEAD ")
  assert "numpy==" in (tmp_path / "env.txt").read_text()
  echo = json.loads((tmp_path / "config.json").read_text())
  assert echo["signature"] == config.signature()


def test_copula_results_and_failures_are_units_of_their_own(
  tmp_path: Path, station: StationData, margins: StationMargins
) -> None:
  """A failed arm is recorded and read back as failed, never as scored.

  Its test rows are the input files' rows, as the margins' are.
  """
  _store(tmp_path, station, margins, "linear-gaussian-all")
  ms = read_marginal_set(tmp_path, station.station, "linear-gaussian-all")
  copulas = FakeCopulas()
  for arm in ("independence", "gaussian"):
    results = copulas.run_copula_arm(
      arm,
      ms,
      estimators={},
      n_draws=4,
      sample_batch_size=10,
      mixture_k=1,
      seed=3,
      device=None,
    )
    write_copula(
      tmp_path,
      ms,
      arm,
      results,
      rows=station.row_test,
      seed=3,
      wall_seconds=0.1,
    )
  write_copula_failure(
    tmp_path,
    ms,
    "vine",
    FakeCopulaFailure("density evaluation failed", [1, 3]),
    rows=station.row_test,
    seed=3,
  )

  assert copula_status(tmp_path, 1, "linear-gaussian-all", "gaussian") == "done"
  assert copula_status(tmp_path, 1, "linear-gaussian-all", "vine") == "failed"
  assert copula_status(tmp_path, 1, "linear-gaussian-all", "other") is None
  failure = json.loads(
    (
      copula_dir(tmp_path, 1, "linear-gaussian-all", "vine") / "failure.json"
    ).read_text()
  )
  assert [c["row"] for c in failure["cases"]] == station.row_test[
    [1, 3]
  ].tolist()
  tables = read_copula_tables(
    tmp_path, [1], ["linear-gaussian-all"], ["independence", "gaussian", "vine"]
  )
  scores = tables["scores"]
  assert set(scores["arm"]) == {"independence", "gaussian"}
  for _, part in scores.groupby("arm"):
    np.testing.assert_array_equal(part["row"], station.row_test)
  assert tables["failures"]["arm"].tolist() == ["vine"]
  assert tables["failures"]["n_cases"].tolist() == [2]
  assert set(tables["timings"]["stage"]) == {"fit", "sample"}
  with pytest.raises(FileNotFoundError, match="missing"):
    read_copula_tables(tmp_path, [1], ["linear-gaussian-all"], ["other"])


def test_a_copula_result_of_the_wrong_shape_is_refused(
  tmp_path: Path, station: StationData, margins: StationMargins
) -> None:
  """A score per draw instead of per case would be read as cases."""
  _store(tmp_path, station, margins, "linear-gaussian-all")
  ms = read_marginal_set(tmp_path, station.station, "linear-gaussian-all")
  (result,) = FakeCopulas().run_copula_arm(
    "gaussian",
    ms,
    estimators={},
    n_draws=4,
    sample_batch_size=10,
    mixture_k=1,
    seed=3,
    device=None,
  )
  short = dataclasses.replace(result, energy=result.energy[:-1])

  with pytest.raises(ValueError, match="energy must be finite with shape"):
    write_copula(
      tmp_path,
      ms,
      "gaussian",
      [short],
      rows=station.row_test,
      seed=3,
      wall_seconds=0.1,
    )
  assert copula_status(tmp_path, 1, "linear-gaussian-all", "gaussian") is None


def test_tables_of_a_missing_unit_are_refused(
  tmp_path: Path, station: StationData, margins: StationMargins
) -> None:
  """Scoring models on different stations would compare different cases."""
  _store(tmp_path, station, margins, "linear-gaussian-all")
  table = read_margin_table(tmp_path, [1], ["linear-gaussian-all"], "scores")
  assert isinstance(table, pd.DataFrame)
  assert len(table) == len(station.variables) * len(station.row_test)
  with pytest.raises(FileNotFoundError, match=r"\(1, 'emos-seasonal'\)"):
    read_margin_table(
      tmp_path, [1], ["linear-gaussian-all", "emos-seasonal"], "scores"
    )
  with pytest.raises(FileNotFoundError, match="run `margins`"):
    read_marginal_set(tmp_path, 1, "emos-seasonal")
