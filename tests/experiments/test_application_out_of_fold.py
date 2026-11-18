"""Out-of-fold training PITs: each from a margin fitted without its year."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from npcc.experiments._study import derive_seed
from npcc.experiments.application.cli import main
from npcc.experiments.application.config import ApplicationConfig, load_config
from npcc.experiments.application.data import StationData, load_station
from npcc.experiments.application.margins import (
  fit_margin,
  fit_station,
  out_of_fold_pits,
  training_years,
)
from npcc.experiments.application.scores import pit_uniformity
from npcc.experiments.application.store import (
  ensure_station_data,
  margin_dir,
  read_marginal_set,
  write_margin,
)
from tests.experiments._application import (
  FakeCopulas,
  config_text,
  write_data,
)

_OUT_OF_FOLD = '[margins]\ntrain_pits = "out_of_fold"\n'


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
  return write_data(tmp_path_factory.mktemp("data"))


def _config(data_dir: Path, tmp_path: Path, *, out_of_fold: bool) -> Path:
  text = config_text(data_dir)
  if out_of_fold:
    text = text.replace("[margins]\n", _OUT_OF_FOLD)
  path = tmp_path / ("oof.toml" if out_of_fold else "in_sample.toml")
  path.write_text(text)
  return path


@pytest.fixture
def configs(
  data_dir: Path, tmp_path: Path, register_linear_gaussian_backend: None
) -> tuple[ApplicationConfig, ApplicationConfig]:
  del register_linear_gaussian_backend
  return (
    load_config(_config(data_dir, tmp_path, out_of_fold=False)),
    load_config(_config(data_dir, tmp_path, out_of_fold=True)),
  )


@pytest.fixture
def station(
  data_dir: Path, configs: tuple[ApplicationConfig, ApplicationConfig]
) -> StationData:
  return load_station(data_dir, 0, configs[0].selection, sha256=None)


class _Recorder:
  """A margin that records what it was fitted on and flags a seen case."""

  def __init__(self, fitted: list[set[float]]) -> None:
    self.fitted = fitted

  def fit(self, y: np.ndarray, *, x: np.ndarray) -> _Recorder:
    del x
    self.seen = set(y.tolist())
    self.fitted.append(self.seen)
    return self

  def cdf(self, y: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    del x
    return np.where([v in self.seen for v in y.tolist()], 0.9, 0.1)

  def logpdf(self, y: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    raise NotImplementedError

  def quantiles(self, levels: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    raise NotImplementedError


class _NearestNeighbor:
  """A margin that centers each case on its nearest training case's response.

  At a training case its nearest neighbor is itself, so the in-sample PIT
  is exactly 0.5: the extreme of an in-context learner's overfit.
  """

  def fit(self, y: np.ndarray, *, x: np.ndarray) -> _NearestNeighbor:
    self.x, self.y = x[:, 0].copy(), y.copy()
    return self

  def cdf(self, y: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    nearest = np.abs(x[:, :1] - self.x[None, :]).argmin(axis=1)
    return stats.norm.cdf(y - self.y[nearest])

  def logpdf(self, y: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    raise NotImplementedError

  def quantiles(self, levels: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def test_each_years_pits_come_from_a_model_that_never_saw_them() -> None:
  """A fold fitted on its own held-out cases would be in sample again.

  Every training case gets exactly one out-of-fold PIT, from the one model
  fitted on the other years, each with its year's seed, and the fold
  timings are recorded.
  """
  years = np.repeat([2017, 2018, 2019, 2020], 5)
  y = np.arange(20, dtype=float)
  x = y[:, None]
  fitted: list[set[float]] = []
  timings: dict[str, float] = {}
  seeds = {2017: 1, 2018: 2, 2019: 3, 2020: 4}

  pit, info = out_of_fold_pits(
    lambda: _Recorder(fitted), y, x, years, seeds=seeds, timings=timings
  )

  np.testing.assert_array_equal(pit, np.full(20, 0.1))
  assert len(fitted) == 4
  for year, seen in zip((2017, 2018, 2019, 2020), fitted, strict=True):
    assert seen == set(y[years != year].tolist())
  folds = info["folds"]
  assert isinstance(folds, list)
  assert [f["year"] for f in folds] == [2017, 2018, 2019, 2020]
  assert [f["seed"] for f in folds] == [1, 2, 3, 4]
  assert [f["n_held_out"] for f in folds] == [5, 5, 5, 5]
  assert set(timings) == {"fold_fit", "fold_pit_train"}


def test_a_single_training_year_leaves_nothing_to_fit_a_fold_on() -> None:
  """With one year, leaving it out would fit a model on no cases."""
  with pytest.raises(ValueError, match="at least two years"):
    out_of_fold_pits(
      lambda: _Recorder([]),
      np.zeros(4),
      np.zeros((4, 1)),
      np.full(4, 2019),
      seeds={2019: 0},
      timings={},
    )


def test_out_of_fold_pits_undo_an_overfit_margins_crowding() -> None:
  """In-sample PITs of an overfit margin crowd 0.5; out-of-fold ones do not.

  This is the defect the folds exist for: a copula fitted to the in-sample
  PITs learns margins far narrower than the test PITs it is scored on.
  """
  rng = np.random.default_rng(0)
  n = 800
  years = np.repeat([2017, 2018, 2019, 2020], n // 4)
  x = rng.uniform(0, 10, (n, 1))
  y = x[:, 0] + rng.normal(0, 1, n)

  in_sample = _NearestNeighbor().fit(y, x=x).cdf(y, x=x)
  oof, _ = out_of_fold_pits(
    _NearestNeighbor,
    y,
    x,
    years,
    seeds=dict.fromkeys((2017, 2018, 2019, 2020), 0),
    timings={},
  )

  np.testing.assert_allclose(in_sample, 0.5)
  assert oof.std() > 0.25
  frame = pd.DataFrame(
    {
      "station": 0,
      "model": "nn",
      "variable": "y",
      "split": "train",
      "pit": in_sample,
      "pit_out_of_fold": oof,
    }
  )
  table = pit_uniformity(frame).set_index("kind")
  assert table.loc["train_in_sample", "sd"] == pytest.approx(0.0)
  assert table.loc["train_in_sample", "central_50"] == 1.0
  assert table.loc["train_out_of_fold", "ks"] < 0.15


def test_the_uniformity_table_reads_a_uniform_sample_as_uniform() -> None:
  """A wrong SD, band or KS formula would misreport every margin."""
  pit = (np.arange(10000) + 0.5) / 10000
  frame = pd.DataFrame(
    {
      "station": np.repeat([0, 1], 5000),
      "model": "m",
      "variable": "v",
      "split": "test",
      "pit": pit,
    }
  )
  row = pit_uniformity(frame).iloc[0]
  assert row["kind"] == "test"
  assert row["sd"] == pytest.approx(1 / np.sqrt(12), abs=1e-4)
  assert row["central_50"] == pytest.approx(0.5, abs=1e-3)
  assert row["outer_10"] == pytest.approx(0.1, abs=1e-3)
  assert row["ks"] == pytest.approx(0.5 / 10000, abs=1e-9)
  assert row["n"] == 10000


def test_a_fold_is_a_forecasts_initialization_year(
  station: StationData,
) -> None:
  """A case's stored time is its valid time, a lead after initialization.

  Folding by valid time would put the forecast initialized on 31 December
  2020 in 2021, a year with no other training case.
  """
  years = training_years(station, 24)
  init = station.time_train - np.timedelta64(24, "h")
  assert set(years.tolist()) == {2017, 2018, 2019, 2020}
  np.testing.assert_array_equal(
    years, init.astype("datetime64[Y]").astype(int) + 1970
  )
  last = np.datetime64("2021-01-01T00:00", "ns")
  shifted = station.time_train.copy()
  shifted[-1] = last
  object.__setattr__(station, "time_train", shifted)
  assert training_years(station, 24)[-1] == 2020


@pytest.mark.parametrize("model", ["emos-seasonal", "linear-gaussian-all"])
def test_folds_leave_the_test_predictions_as_they_were(
  station: StationData,
  configs: tuple[ApplicationConfig, ApplicationConfig],
  model: str,
) -> None:
  """The folds add PITs; they must not move a test prediction or a seed.

  The model fitted on every training case is fitted first, with its own
  seed, so its in-sample and test outputs are those of a run without folds.
  """
  plain, folded = configs
  variable = "obs:air_temperature"
  a = fit_margin(
    station, plain.marginal_model(model), variable, config=plain, device="cpu"
  )
  b = fit_margin(
    station, folded.marginal_model(model), variable, config=folded, device="cpu"
  )

  assert a.pit_train_out_of_fold is None
  assert b.pit_train_out_of_fold is not None
  # Two fits of the hermetic backend agree to rounding, not bit for bit.
  for name in ("pit_train", "pit_test", "logpdf_test", "quantiles_test"):
    np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=1e-12)
  assert not np.allclose(b.pit_train_out_of_fold, b.pit_train)
  assert {"fold_fit", "fold_pit_train"} <= set(b.timings)
  folds = b.info["out_of_fold"]
  assert isinstance(folds, dict)
  records = folds["folds"]
  assert [r["year"] for r in records] == [2017, 2018, 2019, 2020]
  assert sum(r["n_held_out"] for r in records) == len(station.row_train)
  assert records[0]["seed"] == derive_seed(
    folded.base_seed, "margin", 0, model, variable, "fold", 2017
  )


def test_the_stored_unit_carries_both_pits_and_serves_either(
  station: StationData,
  configs: tuple[ApplicationConfig, ApplicationConfig],
  tmp_path: Path,
) -> None:
  """A copula reading the wrong column would be fitted to the wrong PITs.

  The unit stores the in-sample PIT in ``pit`` and the out-of-fold one in
  ``pit_out_of_fold``; asking for out-of-fold PITs from a unit without them
  is refused rather than served the in-sample ones.
  """
  plain, folded = configs
  out = tmp_path / "run"
  ensure_station_data(out, station, seconds=0.0)
  for name, config in (("plain", plain), ("folded", folded)):
    margins = fit_station(
      station,
      config.marginal_model("linear-gaussian-all"),
      config=config,
      device="cpu",
    )
    write_margin(
      out,
      0,
      f"linear-gaussian-all-{name}",
      predictions=margins.predictions,
      scores=margins.scores,
      timings=margins.timings,
      info=margins.info,
    )
  stored = pd.read_parquet(
    margin_dir(out, 0, "linear-gaussian-all-folded") / "predictions.parquet"
  )
  train = stored.loc[stored["split"] == "train"]
  assert train["pit_out_of_fold"].notna().all()
  assert stored.loc[stored["split"] == "test", "pit_out_of_fold"].isna().all()

  in_sample = read_marginal_set(out, 0, "linear-gaussian-all-folded")
  oof = read_marginal_set(
    out, 0, "linear-gaussian-all-folded", train_pits="out_of_fold"
  )
  assert (in_sample.train_pits, oof.train_pits) == ("in_sample", "out_of_fold")
  first = train.loc[train["variable"] == in_sample.variables[0]]
  np.testing.assert_array_equal(in_sample.pit_train[:, 0], first["pit"])
  np.testing.assert_array_equal(oof.pit_train[:, 0], first["pit_out_of_fold"])
  np.testing.assert_array_equal(oof.pit_test, in_sample.pit_test)
  with pytest.raises(ValueError, match="must store out-of-fold"):
    read_marginal_set(
      out, 0, "linear-gaussian-all-plain", train_pits="out_of_fold"
    )


def test_the_setting_is_part_of_the_signature_only_off_its_default(
  data_dir: Path, tmp_path: Path, register_linear_gaussian_backend: None
) -> None:
  """A run folder of in-sample PITs must not resume with out-of-fold ones.

  The default leaves the signature as it was, so a folder started before
  the setting existed still resumes; any other value changes it.
  """
  del register_linear_gaussian_backend
  plain = load_config(_config(data_dir, tmp_path, out_of_fold=False))
  folded = load_config(_config(data_dir, tmp_path, out_of_fold=True))
  explicit = tmp_path / "explicit.toml"
  explicit.write_text(
    config_text(data_dir).replace(
      "[margins]\n", '[margins]\ntrain_pits = "in_sample"\n'
    )
  )
  assert load_config(explicit).signature() == plain.signature()
  assert folded.signature() != plain.signature()
  assert folded.echo()["train_pits"] == "out_of_fold"
  wrong = tmp_path / "wrong.toml"
  wrong.write_text(
    config_text(data_dir).replace(
      "[margins]\n", '[margins]\ntrain_pits = "loo"\n'
    )
  )
  with pytest.raises(ValueError, match="train_pits must be one of"):
    load_config(wrong)


@pytest.fixture
def fake_copulas(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeCopulas]:
  copulas = FakeCopulas()
  monkeypatch.setitem(
    sys.modules, "npcc.experiments.application.copulas", copulas.module()
  )
  yield copulas


def test_the_copula_stage_is_fitted_to_the_out_of_fold_pits(
  data_dir: Path,
  tmp_path: Path,
  register_linear_gaussian_backend: None,
  fake_copulas: FakeCopulas,
) -> None:
  """With ``train_pits = "out_of_fold"``, no arm may see an in-sample PIT.

  End to end through the command line: the margins store both PITs, every
  copula call receives the out-of-fold ones, and evaluate writes the
  uniformity table with both training kinds.
  """
  del register_linear_gaussian_backend
  path = _config(data_dir, tmp_path, out_of_fold=True)
  out = tmp_path / "run"
  args = ["--config", str(path), "--out", str(out)]
  assert main(["margins", *args, "--stations", "0", "--device", "cpu"]) == 0
  assert main(["copulas", *args, "--stations", "0", "--device", "cpu"]) == 0

  stored = pd.read_parquet(
    margin_dir(out, 0, "emos-seasonal") / "predictions.parquet"
  )
  assert fake_copulas.calls
  for call in fake_copulas.calls:
    assert call["train_pits"] == "out_of_fold"
    model = stored if call["model"] == "emos-seasonal" else None
    if model is None:
      continue
    first = model.loc[
      (model["split"] == "train") & (model["variable"] == "obs:air_temperature")
    ]
    np.testing.assert_array_equal(
      np.asarray(call["pit_train"])[:, 0], first["pit_out_of_fold"]
    )
  assert main(["evaluate", *args, "--stations", "0"]) == 0
  table = pd.read_csv(out / "tables" / "margins_pit_uniformity.csv")
  assert set(table["kind"]) == {"train_in_sample", "train_out_of_fold", "test"}
