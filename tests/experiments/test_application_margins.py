"""A marginal model fitted to one station's response, end to end.

Every npcc margin here is the hermetic ``linear-gaussian`` backend of
``tests/conftest.py``, through the registry, exactly as a real backend is
reached; EMOS is the real thing.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from npcc.experiments._study import derive_seed
from npcc.experiments.application.config import ApplicationConfig, load_config
from npcc.experiments.application.data import (
  StationData,
  feature_columns,
  load_station,
)
from npcc.experiments.application.emos import BetaEMOS
from npcc.experiments.application.margins import (
  MarginFit,
  NpccPredictor,
  Predictor,
  build_predictor,
  fit_margin,
  fit_station,
  training_years,
)
from npcc.experiments.application.scores import CLOSED_FORM_CRPS
from tests.conftest import LinearGaussianBackend
from tests.experiments._application import write_config, write_data


def _detail(fit: MarginFit, key: str) -> dict[str, object]:
  value = fit.info[key]
  assert isinstance(value, dict)
  return cast("dict[str, object]", value)


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
  return load_station(data_dir, 0, config.selection, sha256=None)


@pytest.mark.parametrize(
  "model", ["emos-seasonal", "linear-gaussian-all", "linear-gaussian-emos"]
)
def test_every_response_is_predicted_in_shape_and_range(
  station: StationData, config: ApplicationConfig, model: str
) -> None:
  """A PIT outside [0, 1] or a crossing quantile would poison every copula.

  The copula stage reads these arrays as they are, so they must be finite,
  aligned with the cases and ordered along the levels.
  """
  margins = fit_station(
    station, config.marginal_model(model), config=config, device="cpu"
  )
  n_train, n_test = len(station.row_train), len(station.row_test)
  predictions = margins.predictions

  for variable in station.variables:
    rows = predictions.loc[predictions["variable"] == variable]
    train = rows.loc[rows["split"] == "train"]
    test = rows.loc[rows["split"] == "test"]
    np.testing.assert_array_equal(train["row"], station.row_train)
    np.testing.assert_array_equal(test["row"], station.row_test)
    for part in (train, test):
      assert ((part["pit"] >= 0) & (part["pit"] <= 1)).all()
    quantiles = test.filter(like="q_").to_numpy()
    assert quantiles.shape == (n_test, config.n_levels)
    assert np.isfinite(quantiles).all()
    assert (np.diff(quantiles, axis=1) >= 0).all()
    assert np.isfinite(test["logpdf"]).all()
    assert train["logpdf"].isna().all()
  assert len(predictions) == len(station.variables) * (n_train + n_test)
  assert len(margins.scores) == len(station.variables) * n_test
  stages = set(margins.timings["stage"])
  # EMOS alone has a closed-form CRPS, and its own stage for it.
  closed_form = {"crps_closed_form"} if model == "emos-seasonal" else set()
  assert stages == {
    "construction",
    "fit",
    "pit_train",
    "pit_test",
    "logpdf_test",
    "quantiles_test",
    "fold_fit",
    "fold_pit_train",
    *closed_form,
  }
  assert (CLOSED_FORM_CRPS in margins.scores) == bool(closed_form)
  assert (margins.timings["seconds"] >= 0).all()
  assert margins.info["levels"] == pytest.approx(config.levels.tolist())


def test_emos_closed_form_crps_checks_the_quantile_crps_on_its_scale(
  station: StationData, config: ApplicationConfig
) -> None:
  """The closed-form CRPS must be on the response's own scale.

  It exists to check the quantile CRPS's approximation, so one left on the
  model scale (relative humidity as a fraction) would report a gap of a
  hundredfold that is a unit error rather than the approximation's.
  """
  margins = fit_station(
    station, config.marginal_model("emos-seasonal"), config=config, device="cpu"
  )
  pooled = margins.scores.groupby("variable")[["crps", CLOSED_FORM_CRPS]].mean()
  assert np.isfinite(margins.scores[CLOSED_FORM_CRPS]).all()
  assert (margins.scores[CLOSED_FORM_CRPS] > 0).all()
  # Nineteen levels approximate the integral to a few percent.
  np.testing.assert_allclose(pooled["crps"], pooled[CLOSED_FORM_CRPS], rtol=0.1)


@pytest.mark.parametrize("model", ["emos-seasonal", "linear-gaussian-all"])
def test_the_log_density_is_the_derivative_of_the_cdf_on_the_original_scale(
  station: StationData, config: ApplicationConfig, model: str
) -> None:
  """A density left on the model scale would misstate the humidity's by 100.

  The stored log-density must be ``d/dy`` of the stored PIT's CDF in the
  response's own units: for humidity, in percent, though the model sees a
  fraction.
  """
  spec = config.marginal_model(model)
  for variable in station.variables:
    fit = fit_margin(station, spec, variable, config=config, device="cpu")
    response = config.responses[variable]
    j = station.column(variable)
    columns = feature_columns(station.feature_names, spec.scope, variable)
    predictor = build_predictor(
      spec, response, config=config, feature_names=[], device="cpu"
    )
    predictor.fit(
      response.to_model(station.y_train[:, j], variable),
      x=station.x_train[:, columns],
    )
    y = station.y_test[:, j]
    lo, hi = response.support
    inside = (y > lo * response.scale + 1e-3) & (y < hi * response.scale - 1e-3)
    x = station.x_test[inside][:, columns]
    y = y[inside]
    h = 1e-4 * max(1.0, float(np.std(y)))

    def cdf(
      values: np.ndarray,
      x: np.ndarray = x,
      predictor: Predictor = predictor,
      scale: float = response.scale,
    ) -> np.ndarray:
      return predictor.cdf(values / scale, x=x)

    slope = (cdf(y + h) - cdf(y - h)) / (2 * h)
    np.testing.assert_allclose(
      np.exp(fit.logpdf_test[inside]), slope, rtol=1e-4, err_msg=variable
    )
    np.testing.assert_allclose(
      fit.pit_test[inside], cdf(y), rtol=1e-12, err_msg=variable
    )


def test_humidity_is_fitted_as_a_nudged_fraction_and_reported_in_percent(
  station: StationData, config: ApplicationConfig
) -> None:
  """His humidity handling: nudge 0 and 100, divide by 100, scale back.

  An observation at a bound keeps a finite log-density, and the quantiles
  come back in percent.
  """
  variable = "obs:relative_humidity"
  j = station.column(variable)
  fit = fit_margin(
    station,
    config.marginal_model("emos-seasonal"),
    variable,
    config=config,
    device=None,
  )
  model_scale = config.responses[variable].to_model(
    station.y_test[:, j], variable
  )
  x = station.x_test[
    :, feature_columns(station.feature_names, "emos", variable)
  ]
  emos = BetaEMOS().fit(
    config.responses[variable].to_model(station.y_train[:, j], variable),
    x=station.x_train[
      :, feature_columns(station.feature_names, "emos", variable)
    ],
  )

  assert ((station.y_test[:, j] == 0) | (station.y_test[:, j] == 100)).any()
  assert np.isfinite(fit.logpdf_test).all()
  np.testing.assert_allclose(
    fit.logpdf_test, emos.logpdf(model_scale, x=x) - np.log(100.0), rtol=1e-12
  )
  # Rounded to float32 once predicted, as every margin's test quantiles are.
  np.testing.assert_array_equal(
    fit.quantiles_test,
    (100.0 * emos.quantiles(config.levels, x=x)).astype(np.float32),
  )
  assert _detail(fit, "emos")["class"] == "BetaEMOS"


def test_each_support_gets_his_emos_family(
  station: StationData, config: ApplicationConfig
) -> None:
  """His choice by response: beta, truncated normal, else seasonal Gaussian."""
  expected = {
    "obs:air_temperature": "SeasonalGaussianEMOS",
    "obs:relative_humidity": "BetaEMOS",
    "obs:water_vapor_mixing_ratio": "TruncatedNormalEMOS",
  }
  for variable, name in expected.items():
    fit = fit_margin(
      station,
      config.marginal_model("emos-seasonal"),
      variable,
      config=config,
      device=None,
    )
    assert _detail(fit, "emos")["class"] == name
    assert fit.info["features"] == [
      "coe:" + variable.removeprefix("obs:") + "_ensavg",
      "time:sin_dayofyear",
      "time:cos_dayofyear",
    ]


def test_the_quantiles_read_one_table_for_every_level(
  station: StationData,
  config: ApplicationConfig,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """199 levels must not mean 199 predictions of the backend.

  The margin keeps its last table, keyed on the rows, so after the test
  PITs every level of the test quantiles is read from it; on new rows, the
  first level predicts it once and the rest reuse it.
  """
  calls: list[int] = []
  original = LinearGaussianBackend._predict_quantiles

  def counted(
    self: LinearGaussianBackend, x: torch.Tensor, alphas: torch.Tensor
  ) -> torch.Tensor:
    calls.append(x.shape[0])
    return original(self, x, alphas)

  monkeypatch.setattr(LinearGaussianBackend, "_predict_quantiles", counted)
  spec = config.marginal_model("linear-gaussian-all")
  variable = "obs:air_temperature"
  response = config.responses[variable]
  columns = feature_columns(station.feature_names, "all", variable)
  predictor = build_predictor(
    spec, response, config=config, feature_names=[], device="cpu"
  )
  assert isinstance(predictor, NpccPredictor)
  j = station.column(variable)
  predictor.fit(station.y_train[:, j], x=station.x_train[:, columns])
  x_test = station.x_test[:, columns]

  predictor.cdf(station.y_test[:, j], x=x_test)
  before = len(calls)
  quantiles = predictor.quantiles(config.levels, x=x_test)
  assert len(calls) == before
  predictor.quantiles(config.levels, x=station.x_train[:, columns])
  assert len(calls) == before + 1
  assert quantiles.shape == (len(x_test), config.n_levels)


def test_a_fit_is_seeded_by_its_own_station_model_and_response(
  station: StationData,
  config: ApplicationConfig,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A fit seeded by what ran before it would change with the run order.

  So would each out-of-fold fit, which follows with its year's own seed.
  """
  seeds: list[int] = []
  monkeypatch.setattr(torch, "manual_seed", lambda seed: seeds.append(seed))
  spec = config.marginal_model("linear-gaussian-emos")

  fit = fit_margin(
    station, spec, "obs:air_temperature", config=config, device="cpu"
  )
  expected = derive_seed(
    config.base_seed, "margin", 0, "linear-gaussian-emos", "obs:air_temperature"
  )
  assert seeds[0] == expected
  years = sorted({int(y) for y in training_years(station, 24)})
  assert seeds[1:] == [
    derive_seed(
      config.base_seed,
      "margin",
      0,
      "linear-gaussian-emos",
      "obs:air_temperature",
      "fold",
      year,
    )
    for year in years
  ]
  assert fit.info["seed"] == expected
  assert _detail(fit, "estimator")["device"] == "cpu"


def test_non_finite_predictions_are_refused(
  station: StationData,
  config: ApplicationConfig,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A non-finite PIT or density stored now would fail a copula much later."""

  class Broken:
    def fit(self, y: np.ndarray, *, x: np.ndarray) -> Broken:
      return self

    def cdf(self, y: np.ndarray, *, x: np.ndarray) -> np.ndarray:
      return np.full(len(y), 0.5)

    def logpdf(self, y: np.ndarray, *, x: np.ndarray) -> np.ndarray:
      out = np.zeros(len(y))
      out[1] = -np.inf
      return out

    def quantiles(self, levels: np.ndarray, *, x: np.ndarray) -> np.ndarray:
      return np.tile(levels, (len(x), 1))

  monkeypatch.setattr(
    "npcc.experiments.application.margins.build_predictor",
    lambda *args, **kwargs: Broken(),
  )
  with pytest.raises(ValueError, match="logpdf_test must be finite"):
    fit_margin(
      station,
      config.marginal_model("linear-gaussian-all"),
      "obs:air_temperature",
      config=config,
      device="cpu",
    )


def test_crossing_quantiles_are_refused(
  station: StationData,
  config: ApplicationConfig,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Draws mapped through crossing quantiles are draws of no distribution."""

  class Crossing:
    def fit(self, y: np.ndarray, *, x: np.ndarray) -> Crossing:
      return self

    def cdf(self, y: np.ndarray, *, x: np.ndarray) -> np.ndarray:
      return np.full(len(y), 0.5)

    def logpdf(self, y: np.ndarray, *, x: np.ndarray) -> np.ndarray:
      return np.zeros(len(y))

    def quantiles(self, levels: np.ndarray, *, x: np.ndarray) -> np.ndarray:
      return np.tile(levels[::-1], (len(x), 1))

  monkeypatch.setattr(
    "npcc.experiments.application.margins.build_predictor",
    lambda *args, **kwargs: Crossing(),
  )
  with pytest.raises(ValueError, match="nondecreasing"):
    fit_margin(
      station,
      config.marginal_model("emos-seasonal"),
      "obs:air_temperature",
      config=config,
      device=None,
    )


def test_a_response_without_a_scale_is_refused(
  station: StationData, config: ApplicationConfig
) -> None:
  """A response with no declared support has no transform to fit on."""
  import dataclasses

  partial = dataclasses.replace(
    config,
    responses={
      k: v for k, v in config.responses.items() if k != "obs:air_temperature"
    },
  )
  with pytest.raises(ValueError, match="obs:air_temperature"):
    fit_margin(
      station,
      partial.marginal_model("emos-seasonal"),
      "obs:air_temperature",
      config=partial,
      device=None,
    )
