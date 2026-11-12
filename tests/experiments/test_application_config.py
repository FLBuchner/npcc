"""The application's configuration: what it reads, refuses and hashes."""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from pathlib import Path

import numpy as np
import pytest

from npcc.core.errors import EstimatorConfigError
from npcc.experiments.application.config import (
  EMOS_MODEL,
  ApplicationConfig,
  ResponseSpec,
  load_config,
)
from npcc.experiments.application.data import CaseSelection
from npcc.experiments.config import EstimatorSpec, load_grid
from tests.experiments._application import config_text

_ROOT = Path(__file__).resolve().parents[2]
_E1_LABELS = ("lightgbm-probit", "tabpfn-quantiles-v3-probit", "tabicl-probit")


@pytest.mark.parametrize("name", ["application.toml", "application_smoke.toml"])
def test_the_shipped_configs_use_e1s_estimators(name: str) -> None:
  """A copied entry that drifts from E1's would fit another estimator.

  The paper reports the application with the simulation study's estimators;
  an edit to either file that the other does not follow would change one
  study's backend configuration under the same label.
  """
  config = load_config(_ROOT / "configs" / name)
  e1 = {
    e.label: e for e in load_grid(_ROOT / "configs" / "study.toml").estimators
  }

  assert sorted(e.label for e in config.estimators) == sorted(_E1_LABELS)
  for spec in config.estimators:
    assert spec.estimator_id == e1[spec.label].estimator_id, spec.label
  tabpfn = config.estimators_by_backend["tabpfn-quantiles"]
  assert tabpfn.backend_kwargs["model_version"] == "v3"
  assert tabpfn.backend_kwargs["model_kwargs"] == {"softmax_temperature": 1.0}


def test_the_full_config_holds_the_decided_models_and_arms() -> None:
  """A model or arm dropped from the full run would silently go unreported.

  The full run fits his seven marginal models, and the copula arms on the
  EMOS and TabPFN-on-all-covariates margins; the smoke run, on station 0,
  every arm on every margin.
  """
  config = load_config(_ROOT / "configs" / "application.toml")

  assert config.models == (
    "emos-seasonal",
    "lightgbm-all",
    "lightgbm-emos",
    "tabpfn-quantiles-all",
    "tabpfn-quantiles-emos",
    "tabicl-all",
    "tabicl-emos",
  )
  assert config.copula_models == ("emos-seasonal", "tabpfn-quantiles-all")
  assert config.copula_arms == (
    "independence",
    "gaussian",
    "boostcopula-all",
    "boostcopula-time",
    "npcc-random-tabpfn-quantiles-all",
    "npcc-random-tabpfn-quantiles-time",
    "npcc-mixture-tabpfn-quantiles-all",
    "npcc-mixture-tabpfn-quantiles-time",
  )
  assert config.verify_data
  assert (
    config.data_dir == Path("~/npcc-data/meteoswiss_zanetta2023").expanduser()
  )
  assert config.stations is None
  assert config.n_levels == 199
  assert (config.n_draws, config.sample_batch_size, config.mixture_k) == (
    200,
    5000,
    5,
  )
  assert config.device_for("lightgbm") == "cpu"
  assert config.device_for("npcc-tabpfn-quantiles-all") == "cuda"
  smoke = load_config(_ROOT / "configs" / "application_smoke.toml")
  assert smoke.stations == (0,)
  assert smoke.signature() == config.signature()
  assert smoke.copula_models == smoke.models == config.models
  assert len(smoke.copula_arms) == 22
  assert set(config.copula_arms) <= set(smoke.copula_arms)


def test_the_smoke_config_runs_every_arm_the_copula_stage_knows() -> None:
  """An arm left out of the smoke run first costs in the full run."""
  copulas = pytest.importorskip("npcc.experiments.application.copulas")
  smoke = load_config(_ROOT / "configs" / "application_smoke.toml")

  assert smoke.copula_arms == tuple(copulas.COPULA_ARMS)


def test_the_levels_are_the_midpoints() -> None:
  """Levels off the midpoint grid would change every CRPS from quantiles."""
  config = load_config(_ROOT / "configs" / "application.toml")

  np.testing.assert_allclose(
    config.levels, (np.arange(199) + 0.5) / 199, rtol=0, atol=0
  )


def _write(tmp_path: Path, text: str) -> Path:
  path = tmp_path / "app.toml"
  path.write_text(text)
  return path


@pytest.mark.usefixtures("register_linear_gaussian_backend")
@pytest.mark.parametrize(
  ("old", "new", "error", "match"),
  [
    ("[margins]", "[margins]\nbogus = 1", ValueError, "unknown"),
    ("base_seed = 11", "base_seed = 11\n[extra]", ValueError, "unknown"),
    (
      '"emos-seasonal", ',
      '"emos-seasonal", "linear-gaussian-hourly", ',
      ValueError,
      "scope",
    ),
    (
      '"emos-seasonal", ',
      '"emos-seasonal", "lightgbm-all", ',
      ValueError,
      "backend",
    ),
    (
      '"emos-seasonal", ',
      '"emos-seasonal", "emos-seasonal", ',
      ValueError,
      "unique",
    ),
    ('stations = "all"', "stations = [-1]", ValueError, "station ids"),
    ('stations = "all"', "stations = []", ValueError, "station ids"),
    ('stations = "all"', 'stations = "some"', TypeError, "'all'"),
    ("n_levels = 19", "n_levels = 1", ValueError, "n_levels"),
    ("n_draws = 8", "n_draws = 1", ValueError, "n_draws"),
    ("n_levels = 19", "n_levels = 19.5", TypeError, "integer"),
    (
      "test = [2021-01-01, 2021-12-31]",
      "test = [2020-06-01, 2021-12-31]",
      ValueError,
      "before test",
    ),
    (
      "train = [2017-01-01, 2020-12-31]",
      "train = [2017-01-01T00:00:00, 2020-12-31]",
      TypeError,
      "without a time",
    ),
    (
      'transform = "log" }',
      'transform = "cube" }',
      ValueError,
      "transform",
    ),
    (
      "boundary_nudge = 0.05",
      "boundary_nudge = 0.0",
      ValueError,
      "boundary_nudge",
    ),
    (
      '"obs:air_temperature" = { transform = "identity" }',
      '"obs:air_temperature" = { transform = "identity", boundary_nudge = 1.0 }',
      ValueError,
      "no upper bound",
    ),
    (
      '"obs:air_temperature" = { transform = "identity" }',
      '"obs:air_temperature" = { transform = "identity", shift = 1.0 }',
      ValueError,
      "unknown",
    ),
    (
      "[[estimators]]",
      '[devices]\nlightgbm = "cpu"\n\n[[estimators]]',
      ValueError,
      "estimator backends",
    ),
    (
      "[[estimators]]",
      '[devices]\nlinear-gaussian = "gpu-please"\n\n[[estimators]]',
      ValueError,
      "torch device",
    ),
    (
      'arms = ["independence", "gaussian"]',
      'arms = ["independence", "independence"]',
      ValueError,
      "unique",
    ),
    (
      'arms = ["independence", "gaussian"]',
      'arms = ["independence", "gaussian"]\narms_by_margin = { "no-such-model" = ["gaussian"] }',
      ValueError,
      "marginal models",
    ),
    (
      'arms = ["independence", "gaussian"]',
      'arms = ["independence", "gaussian"]\narms_by_margin = { "emos-seasonal" = ["boostcopula-all"] }',
      ValueError,
      "drawn from copulas.arms",
    ),
    (
      'arms = ["independence", "gaussian"]',
      'arms = ["independence", "gaussian"]\narms_by_margin = { "emos-seasonal" = [] }',
      ValueError,
      "at least one arm",
    ),
    (
      'arms = ["independence", "gaussian"]',
      'arms = ["independence", "gaussian"]\narms_by_margin = { "emos-seasonal" = ["gaussian", "gaussian"] }',
      ValueError,
      "unique",
    ),
    (
      'arms = ["independence", "gaussian"]',
      'arms = ["independence", "gaussian"]\narms_by_margin = ["gaussian"]',
      TypeError,
      "table of arm lists",
    ),
  ],
)
def test_a_config_no_run_could_honor_is_refused(
  tmp_path: Path,
  old: str,
  new: str,
  error: type[Exception],
  match: str,
) -> None:
  """A setting the runner cannot honor must fail at load, not mid-run.

  A misspelled key, a model with no estimator, a transform with no support
  or a device torch cannot parse would otherwise be dropped silently or
  surface hours into a run.
  """
  text = config_text(Path("data"))
  assert old in text
  with pytest.raises(error, match=match):
    load_config(_write(tmp_path, text.replace(old, new, 1)))


@pytest.mark.usefixtures("register_linear_gaussian_backend")
def test_a_margin_may_run_only_some_of_the_arms(tmp_path: Path) -> None:
  """A per-margin arm list must narrow that margin's arms and no other's.

  The arms keep ``copulas.arms``'s order whatever order the list gives, the
  echo records each margin's arms, and the signature does not change, since
  the list only selects work.
  """
  plain = config_text(Path("data"))
  arms = 'arms = ["independence", "gaussian"]'
  narrowed = plain.replace(
    arms,
    f'{arms}\narms_by_margin = {{ "linear-gaussian-all" = '
    '["gaussian", "independence"], "emos-seasonal" = ["gaussian"] }',
    1,
  )
  config = load_config(_write(tmp_path, narrowed))

  assert config.copula_arms_for("emos-seasonal") == ("gaussian",)
  assert config.copula_arms_for("linear-gaussian-all") == (
    "independence",
    "gaussian",
  )
  assert config.copula_arms_for("linear-gaussian-emos") == config.copula_arms
  assert config.echo()["copula_arms_by_margin"] == {
    "emos-seasonal": ["gaussian"],
    "linear-gaussian-all": ["independence", "gaussian"],
    "linear-gaussian-emos": ["independence", "gaussian"],
  }
  (tmp_path / "plain").mkdir()
  assert (
    config.signature()
    == load_config(_write(tmp_path / "plain", plain)).signature()
  )
  with pytest.raises(ValueError, match="copula stage's marginal models"):
    config.copula_arms_for("tabpfn-quantiles-all")


@pytest.mark.usefixtures("register_linear_gaussian_backend")
def test_two_estimators_on_one_backend_are_refused(tmp_path: Path) -> None:
  """A model names its estimator by backend; two would make that ambiguous."""
  text = config_text(Path("data")) + (
    '\n[[estimators]]\nlabel = "other"\nbackend = "linear-gaussian"\n'
    'transform = "logit"\n'
  )
  with pytest.raises(EstimatorConfigError, match="each backend once"):
    load_config(_write(tmp_path, text))


@pytest.mark.usefixtures("register_linear_gaussian_backend")
def test_omitted_keys_take_the_dataclass_defaults(tmp_path: Path) -> None:
  """A default restated in the loader could drift from the dataclass's."""
  text = (
    config_text(Path("data"))
    .replace("n_draws = 8\n", "")
    .replace("sample_batch_size = 50\n", "")
  )
  config = load_config(_write(tmp_path, text))
  defaults = {f.name: f.default for f in dataclasses.fields(ApplicationConfig)}

  assert config.n_draws == defaults["n_draws"]
  assert config.sample_batch_size == defaults["sample_batch_size"]
  assert config.mixture_k == defaults["mixture_k"]
  assert config.selection.lead_hours == 24
  assert config.selection.init_hour == 0
  assert config.selection.min_cases == 20


@pytest.mark.usefixtures("register_linear_gaussian_backend")
def test_the_signature_ignores_what_only_selects_work(tmp_path: Path) -> None:
  """A shard or a narrower rerun must be able to resume the same folder.

  Which stations, models and arms run, and where, choose what to compute and
  not what a result is; anything that changes a stored result must change
  the signature, or a resume would mix results of two configurations.
  """
  base = load_config(_write(tmp_path, config_text(Path("data"))))
  same = dataclasses.replace(
    base,
    stations=(1,),
    models=(EMOS_MODEL,),
    copula_arms=("gaussian",),
    devices={"linear-gaussian": "cpu"},
    batch_size=64,
  )
  assert same.signature() == base.signature()
  for change in (
    {"base_seed": 12},
    {"n_levels": 21},
    {"n_draws": 9},
    {
      "responses": {
        **base.responses,
        "obs:air_temperature": ResponseSpec("identity", scale=2.0),
      }
    },
    {
      "selection": dataclasses.replace(base.selection, lead_hours=48),
    },
    {"estimators": (EstimatorSpec("lg", "linear-gaussian", "logit"),)},
  ):
    assert dataclasses.replace(base, **change).signature() != base.signature()


def test_humidity_is_nudged_off_its_bounds_then_scaled() -> None:
  """An exact 0 or 100 has no density under a logit model and would be lost.

  His design moves them to 0.05 and 99.95 percent before dividing by 100.
  """
  spec = ResponseSpec("logit", scale=100.0, boundary_nudge=0.05)
  y = np.array([0.0, 12.5, 100.0, 99.95])

  np.testing.assert_array_equal(
    spec.to_model(y, "rh"), [0.0005, 0.125, 0.9995, 0.9995]
  )
  assert spec.log_scale == pytest.approx(math.log(100.0))
  assert spec.support == (0.0, 1.0)
  np.testing.assert_array_equal(y, [0.0, 12.5, 100.0, 99.95])


@pytest.mark.parametrize(
  ("spec", "values", "match"),
  [
    (ResponseSpec("logit", 100.0, 0.05), [50.0, 100.5], r"\[0, 100.0\]"),
    (ResponseSpec("logit", 100.0, 0.05), [-0.1], r"\[0, 100.0\]"),
    (ResponseSpec("log"), [1.0, 0.0], "positive"),
    (ResponseSpec("log"), [-2.0], "positive"),
  ],
)
def test_a_response_off_its_support_is_refused(
  spec: ResponseSpec, values: list[float], match: str
) -> None:
  """Clamping a response off its support would score a different value."""
  with pytest.raises(ValueError, match=match):
    spec.to_model(np.array(values), "y")


def test_an_identity_response_passes_through() -> None:
  """Temperatures and pressure are modeled in their own units."""
  y = np.array([-3.0, 0.0, 1013.25])

  np.testing.assert_array_equal(ResponseSpec("identity").to_model(y, "t"), y)


@pytest.mark.parametrize(
  ("train", "test", "match"),
  [
    ((dt.date(2018, 1, 1), dt.date(2017, 1, 1)), None, "on or before"),
    (None, (dt.date(2020, 12, 31), dt.date(2021, 2, 1)), "before test"),
  ],
)
def test_a_selection_with_empty_or_shared_splits_is_refused(
  train: tuple[dt.date, dt.date] | None,
  test: tuple[dt.date, dt.date] | None,
  match: str,
) -> None:
  """A case in both splits would be scored on data its model was fitted on."""
  with pytest.raises(ValueError, match=match):
    CaseSelection(
      train=train or (dt.date(2017, 1, 1), dt.date(2020, 12, 31)),
      test=test or (dt.date(2021, 1, 1), dt.date(2021, 12, 31)),
    )
