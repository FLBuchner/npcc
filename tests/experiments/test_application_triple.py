"""The triple design: (T, Td, p) modeled, relative humidity and w derived."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from npcc.experiments._study import derive_seed
from npcc.experiments.application import copulas
from npcc.experiments.application.cli import main
from npcc.experiments.application.config import load_config
from npcc.experiments.application.derived import (
  DEW_POINT,
  MAGNUS,
  METEOSWISS,
  MIXING_RATIO,
  PRESSURE,
  RELATIVE_HUMIDITY,
  SONNTAG_1990,
  TEMPERATURE,
  Derivation,
  derive,
  mixing_ratio,
  nudge_saturated,
  relative_humidity,
  saturation_dew_point,
  saturation_vapor_pressure,
)
from npcc.experiments.application.emos import gaussian_crps
from npcc.experiments.application.joint_scores import (
  energy_score,
  inverse_margins,
  variogram_score,
)
from npcc.experiments.application.marginal_set import MarginalSet
from npcc.experiments.application.store import read_marginal_set
from tests.experiments._application import config_text

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "r_methods" / "fit_boostcopula_vine.R"
_MODELED = (TEMPERATURE, DEW_POINT, PRESSURE)
_DERIVED = (RELATIVE_HUMIDITY, MIXING_RATIO)
_TRIPLE = """
modeled = ["obs:air_temperature", "obs:dew_point_temperature", "obs:surface_air_pressure"]
derived = ["obs:relative_humidity", "obs:water_vapor_mixing_ratio"]
"""


# ---------------------------------------------------------------------------
# The formulas
# ---------------------------------------------------------------------------


def test_the_formulas_give_the_hand_values() -> None:
  """A wrong constant would derive another humidity from every draw.

  By hand: ``e_s(0) = 6.112``; ``e_s(10) = 6.112 exp(176.2 / 253.12) =
  12.2603``; ``e_s(20) = 6.112 exp(352.4 / 263.12) = 23.3263`` hPa; so
  ``RH(20, 10) = 52.560`` and ``w(10, 1000) = 622 x 12.2603 / 987.7397 =
  7.7205`` g/kg.
  """
  np.testing.assert_allclose(
    saturation_vapor_pressure(np.array([0.0, 10.0, 20.0])),
    [6.112, 12.2603, 23.3263],
    rtol=2e-5,
  )
  assert float(relative_humidity(np.array(20.0), np.array(10.0))) == (
    pytest.approx(52.560, rel=1e-4)
  )
  assert float(
    mixing_ratio(np.array(20.0), np.array(10.0), np.array(1000.0))
  ) == pytest.approx(7.7205, rel=1e-4)
  t = np.array([-12.0, 3.5, 31.0])
  np.testing.assert_array_equal(relative_humidity(t, t), [100.0] * 3)


def test_the_meteoswiss_formula_is_piecewise_at_zero() -> None:
  """Zanetta et al.'s constants: one set at and above 0 C, another below.

  By hand: ``6.107 exp(173.68 / 248.83) = 12.2733`` hPa at 10 C and
  ``6.108 exp(-178.56 / 235.52) = 2.86179`` hPa at -10 C.
  """
  np.testing.assert_allclose(
    METEOSWISS.saturation_vapor_pressure(np.array([10.0, -10.0, 0.0])),
    [12.2733, 2.86179, 6.107],
    rtol=2e-5,
  )
  for magnus in (SONNTAG_1990, METEOSWISS):
    t = np.array([-25.0, -0.5, 0.0, 0.5, 30.0])
    np.testing.assert_allclose(
      magnus.dew_point(magnus.saturation_vapor_pressure(t)), t, atol=1e-10
    )
  assert MAGNUS == {"sonntag1990": SONNTAG_1990, "meteoswiss": METEOSWISS}
  rh = relative_humidity(np.array(20.0), np.array(10.0), magnus=METEOSWISS)
  assert float(rh) != pytest.approx(52.560, rel=1e-4)


def test_a_dew_point_above_the_temperature_is_clipped_only_if_asked() -> None:
  """The data never has Td > T; whether a draw that does reads as saturated
  is the clip's choice, off by default."""
  t, p = np.array([5.0, 5.0]), np.array([950.0, 950.0])
  above = np.array([7.0, 5.0])

  np.testing.assert_array_equal(
    relative_humidity(t, above, clip=True), [100.0, 100.0]
  )
  np.testing.assert_array_equal(
    mixing_ratio(t, above, p, clip=True), mixing_ratio(t, t, p)
  )
  unclipped = relative_humidity(t, above)
  assert unclipped[0] > 100.0
  assert unclipped[1] == 100.0
  assert mixing_ratio(t, above, p)[0] > mixing_ratio(t, t, p)[0]


def test_derived_draws_and_the_count_of_td_above_t() -> None:
  """Every draw derives both responses; Td > T is counted, clipped or not."""
  draws = np.array(
    [
      [[10.0, 4.0, 900.0], [10.0, 12.0, 905.0], [0.0, -1.0, 1000.0]],
      [[20.0, 20.5, 950.0], [20.0, 19.0, 950.0], [20.0, 20.0, 950.0]],
    ]
  )
  values, above = derive(draws, _MODELED, _DERIVED)
  clipped, above_too = derive(
    draws, _MODELED, _DERIVED, Derivation(clip_dew_point=True)
  )

  assert values.shape == (2, 3, 2)
  np.testing.assert_array_equal(
    above, [[False, True, False], [True, False, False]]
  )
  np.testing.assert_array_equal(above, above_too)
  np.testing.assert_array_equal(
    values[..., 0], relative_humidity(draws[..., 0], draws[..., 1])
  )
  np.testing.assert_array_equal(
    values[..., 1],
    mixing_ratio(draws[..., 0], draws[..., 1], draws[..., 2]),
  )
  assert values[0, 1, 0] > 100.0
  assert clipped[0, 1, 0] == 100.0
  np.testing.assert_array_equal(clipped[~above], values[~above])
  assert above.mean(axis=1).tolist() == [1 / 3, 1 / 3]
  meteo, _ = derive(draws, _MODELED, _DERIVED, Derivation(magnus="meteoswiss"))
  np.testing.assert_array_equal(
    meteo[..., 0],
    relative_humidity(draws[..., 0], draws[..., 1], magnus=METEOSWISS),
  )


def test_a_derived_response_needs_its_inputs_modeled() -> None:
  """RH without a modeled dew point has nothing to be computed from."""
  draws = np.zeros((1, 2, 2))
  with pytest.raises(ValueError, match="not modeled"):
    derive(draws, (TEMPERATURE, PRESSURE), (RELATIVE_HUMIDITY,))
  with pytest.raises(ValueError, match="derived must be drawn"):
    derive(draws, (TEMPERATURE, DEW_POINT), (PRESSURE,))


def test_the_saturation_nudge_is_his_rh_nudge_on_the_dew_point() -> None:
  """A case with Td = T moves to where the Magnus RH is 99.95 when asked.

  It is counted whether or not it moves.
  """
  t = np.array([-20.0, 0.0, 15.0, 35.0])
  for magnus in (SONNTAG_1990, METEOSWISS):
    td = saturation_dew_point(t, 0.05, magnus)
    np.testing.assert_allclose(
      relative_humidity(t, td, magnus=magnus), 99.95, rtol=1e-12
    )
    assert (td < t).all()
  td = saturation_dew_point(t, 0.05)
  observed = np.array([-20.0, -3.0, 15.0, 30.0])
  moved, count = nudge_saturated(t, observed, 0.05)
  assert count == 2
  np.testing.assert_array_equal(moved, [td[0], -3.0, td[2], 30.0])
  same, counted = nudge_saturated(t, observed, 0.0)
  assert counted == 2
  np.testing.assert_array_equal(same, observed)


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------


def _triple_config(tmp_path: Path, block: str = _TRIPLE) -> Path:
  text = config_text(Path("data")).replace(
    "[responses]\n", "[responses]\n" + block
  )
  path = tmp_path / "triple.toml"
  path.write_text(text)
  return path


@pytest.mark.usefixtures("register_linear_gaussian_backend")
def test_the_triple_config_reads_its_design(tmp_path: Path) -> None:
  """The triple is a choice of the config, recorded in its signature."""
  config = load_config(_triple_config(tmp_path))
  default = load_config(_triple_config(tmp_path, ""))

  assert config.modeled == _MODELED
  assert config.derived == _DERIVED
  assert config.saturation_nudge == 0.0
  assert config.derivation == Derivation()
  assert not config.is_default_design
  assert default.is_default_design
  assert config.signature() != default.signature()
  assert config.echo()["derived"] == list(_DERIVED)
  assert "derived" not in default.echo()


@pytest.mark.parametrize(
  "name", ["application_triple", "application_smoke_triple"]
)
def test_the_shipped_triple_configs(name: str) -> None:
  """The triple runs the full run's margins and arms, on (T, Td, p)."""
  config = load_config(_ROOT / "configs" / f"{name}.toml")

  assert config.modeled == _MODELED
  assert config.derived == _DERIVED
  # Both pending the user's decision, so off.
  assert config.saturation_nudge == 0.0
  assert not config.clip_dew_point
  assert config.magnus == "sonntag1990"
  assert config.models == ("emos-seasonal", "tabpfn-quantiles-all")
  assert config.copula_models == config.models
  full = load_config(_ROOT / "configs" / "application.toml")
  assert config.copula_arms == full.copula_arms
  assert config.stations == ((0,) if "smoke" in name else None)


@pytest.mark.usefixtures("register_linear_gaussian_backend")
@pytest.mark.parametrize(
  ("block", "match"),
  [
    ('\nmodeled = ["obs:air_temperature", "obs:wind"]\n', "subset"),
    ("\nmodeled = []\n", "subset"),
    (
      '\nmodeled = ["obs:air_temperature", "obs:dew_point_temperature"]\n'
      'derived = ["obs:water_vapor_mixing_ratio"]\n',
      "not modeled",
    ),
    (
      '\nmodeled = ["obs:air_temperature", "obs:dew_point_temperature", '
      '"obs:surface_air_pressure", "obs:relative_humidity"]\n'
      'derived = ["obs:relative_humidity"]\n',
      "not both",
    ),
    (
      '\nmodeled = ["obs:air_temperature", "obs:dew_point_temperature", '
      '"obs:surface_air_pressure"]\nderived = ["obs:relative_humidity"]\n',
      "modeled or derived",
    ),
    (
      '\nmodeled = ["obs:air_temperature", "obs:relative_humidity", '
      '"obs:surface_air_pressure", "obs:water_vapor_mixing_ratio"]\n'
      'derived = ["obs:dew_point_temperature"]\n',
      "with a formula",
    ),
    (_TRIPLE + "saturation_nudge = 100.0\n", r"\[0, 100\)"),
    (_TRIPLE + 'saturation_nudge = "0.05"\n', "number"),
    (_TRIPLE + 'magnus = "goff-gratch"\n', "magnus must be one of"),
    (_TRIPLE + 'clip_dew_point = "yes"\n', "true or false"),
    ("\nclip_dew_point = true\n", "unless responses are derived"),
    (
      "\nsaturation_nudge = 0.05\n" + '\nmodeled = ["obs:air_temperature", '
      '"obs:relative_humidity", "obs:surface_air_pressure", '
      '"obs:water_vapor_mixing_ratio", "obs:dew_point_temperature"]\n',
      None,
    ),
    (
      '\nmodeled = ["obs:air_temperature", "obs:relative_humidity", '
      '"obs:surface_air_pressure", "obs:water_vapor_mixing_ratio"]\n'
      "saturation_nudge = 0.05\n",
      "both modeled",
    ),
  ],
)
def test_a_design_that_cannot_be_derived_is_refused(
  tmp_path: Path, block: str, match: str | None
) -> None:
  """A derived response without inputs, or a response left out, must stop.

  The one config here that does load models every response and nudges the
  dew point, which the triple's checks allow.
  """
  path = _triple_config(tmp_path, block)
  if match is None:
    assert load_config(path).saturation_nudge == 0.05
    return
  with pytest.raises((ValueError, TypeError), match=match):
    load_config(path)


# ---------------------------------------------------------------------------
# A copula arm on the triple
# ---------------------------------------------------------------------------


def _triple_set(
  n_train: int = 120, n_test: int = 5, seed: int = 0
) -> MarginalSet:
  """Gaussian margins of (T, Td, p), RH and w derived from the observations."""
  rng = np.random.default_rng(seed)
  mu = np.array([12.0, 6.0, 950.0])
  sigma = np.array([4.0, 3.0, 8.0])
  levels = (np.arange(99) + 0.5) / 99

  def sample(n: int) -> np.ndarray:
    y = mu + sigma * rng.standard_normal((n, 3))
    y[:, 1] = np.minimum(y[:, 1], y[:, 0])
    return y

  y_train, y_test = sample(n_train), sample(n_test)
  derived_train, _ = derive(y_train, _MODELED, _DERIVED)
  derived_test, _ = derive(y_test, _MODELED, _DERIVED)
  quantiles = mu + sigma * stats.norm.ppf(levels)[:, None]
  x = rng.standard_normal((n_train + n_test, 2))
  return MarginalSet(
    station=4,
    model="emos-seasonal",
    variables=_MODELED,
    feature_names=("time:sin_dayofyear", "time:cos_dayofyear"),
    x_train=x[:n_train],
    x_test=x[n_train:],
    y_train=y_train,
    y_test=y_test,
    time_train=np.arange(n_train)
    .astype("datetime64[D]")
    .astype("datetime64[ns]"),
    time_test=np.arange(n_test)
    .astype("datetime64[D]")
    .astype("datetime64[ns]"),
    pit_train=stats.norm.cdf((y_train - mu) / sigma),
    pit_test=stats.norm.cdf((y_test - mu) / sigma),
    logpdf_test=stats.norm.logpdf(y_test, loc=mu, scale=sigma),
    levels=levels,
    quantiles_test=np.broadcast_to(
      quantiles, (n_test, *quantiles.shape)
    ).copy(),
    derived=_DERIVED,
    y_derived_train=derived_train,
    y_derived_test=derived_test,
  )


def _run(arm: str, ms: MarginalSet, n_draws: int = 40) -> copulas.CopulaResult:
  (result,) = copulas.run_copula_arm(
    arm,
    ms,
    estimators={},
    n_draws=n_draws,
    sample_batch_size=50,
    mixture_k=1,
    seed=9,
    device="cpu",
  )
  return result


def test_the_modeled_log_score_adds_the_copula_to_three_margins() -> None:
  """``joint_nll_modeled`` is the copula's log-density plus the triple's."""
  ms = _triple_set()
  for arm in ("independence", "gaussian"):
    result = _run(arm, ms)
    np.testing.assert_allclose(
      result.joint_nll,
      -(result.logcop_test + ms.logpdf_test.sum(axis=1)),
      rtol=1e-14,
    )
    assert result.info["modeled"] == list(_MODELED)
    assert result.info["derived"] == list(_DERIVED)


def test_the_energy_score_covers_all_five_with_the_derived_draws() -> None:
  """The joint scores read RH and w from each draw of the triple.

  Rebuilt from the arm's own draws: the triple's through the quantiles, RH
  and w derived from them, all five divided by their training SD; each
  response's CRPS is the one-dimensional energy score in its units.
  """
  ms = _triple_set()
  result = _run("independence", ms)
  seed = derive_seed(9, ms.station, ms.model, "independence", "draws")
  uniforms = np.random.default_rng(seed).random((len(ms.y_test), 40, 3))
  draws = inverse_margins(uniforms, ms.levels, ms.quantiles_test)
  extra, above = derive(draws, _MODELED, _DERIVED)
  full = np.concatenate([draws, extra], axis=2)
  observed = np.concatenate([ms.y_test, ms.y_derived_test], axis=1)
  scale = np.concatenate(
    [ms.y_train.std(axis=0), ms.y_derived_train.std(axis=0)]
  )

  np.testing.assert_allclose(
    result.energy, energy_score(full / scale, observed / scale), rtol=1e-14
  )
  np.testing.assert_allclose(
    result.variogram,
    variogram_score(full / scale, observed / scale, p=0.5),
    rtol=1e-14,
  )
  assert list(result.sample_crps) == [*_MODELED, *_DERIVED]
  for j, name in enumerate((*_MODELED, *_DERIVED)):
    np.testing.assert_allclose(
      result.sample_crps[name],
      energy_score(full[:, :, j : j + 1], observed[:, j : j + 1]),
      rtol=1e-14,
    )
  assert result.dew_point_above_share is not None
  np.testing.assert_array_equal(
    result.dew_point_above_share, above.mean(axis=1)
  )
  assert result.info["dew_point_above_draws_share"] == pytest.approx(
    above.mean()
  )
  assert result.info["clip_dew_point"] is False
  assert result.info["magnus"] == "sonntag1990"


def test_the_sample_crps_is_the_gaussian_crps() -> None:
  """The CRPS from draws estimates the CRPS, here against its closed form."""
  rng = np.random.default_rng(5)
  mu, sigma = 1.5, 2.0
  y = np.array([[-1.0], [1.5], [6.0]])
  draws = mu + sigma * rng.standard_normal((3, 40_000, 1))

  np.testing.assert_allclose(
    energy_score(draws, y),
    gaussian_crps(y[:, 0], mu, sigma),
    rtol=2e-2,
  )


# ---------------------------------------------------------------------------
# boostCopula at d = 3
# ---------------------------------------------------------------------------


def _initial_matrix(d: int) -> np.ndarray:
  """The R script's ``initial_matrix(d)``, run through R."""
  code = (
    f"src <- readLines('{_SCRIPT}'); "
    "start <- grep('^initial_matrix <- function', src); "
    "end <- start + which(src[start:length(src)] == '}')[1] - 1; "
    "eval(parse(text = src[start:end])); "
    f"m <- initial_matrix({d}); "
    "cat(VineCopula::RVineMatrixCheck(m), '\\n'); write.table(m, sep = ',', "
    "row.names = FALSE, col.names = FALSE)"
  )
  done = subprocess.run(
    ["Rscript", "-e", code], capture_output=True, text=True, check=True
  )
  check, *rows = done.stdout.strip().splitlines()
  assert check.strip() == "1"
  return np.array([[float(v) for v in row.split(",")] for row in rows])


needs_r = pytest.mark.skipif(
  shutil.which("Rscript") is None, reason="needs Rscript with VineCopula"
)


@needs_r
def test_boostcopulas_initial_matrix_at_three_and_five() -> None:
  """d = 5 is his matrix; d = 3 is the same pattern, a valid R-vine."""
  np.testing.assert_array_equal(
    _initial_matrix(5),
    [
      [5, 0, 0, 0, 0],
      [2, 2, 0, 0, 0],
      [3, 3, 3, 0, 0],
      [1, 4, 4, 4, 0],
      [4, 1, 1, 1, 1],
    ],
  )
  np.testing.assert_array_equal(
    _initial_matrix(3), [[3, 0, 0], [1, 2, 0], [2, 1, 1]]
  )
  for d in (2, 4, 6):
    assert _initial_matrix(d).shape == (d, d)


@needs_r
def test_boostcopula_fits_the_triple(monkeypatch: pytest.MonkeyPatch) -> None:
  """boostCopula fits and scores three variables, five derived."""
  try:
    copulas.check_boostcopula(str(_SCRIPT))
  except RuntimeError as exc:
    pytest.skip(f"needs boostCopula: {exc}")
  monkeypatch.setattr(copulas, "BOOSTCOPULA_SCRIPT", str(_SCRIPT))
  ms = _triple_set(n_train=60, n_test=3)
  result = _run("boostcopula-time", ms, n_draws=6)

  assert np.isfinite(result.joint_nll).all()
  assert np.isfinite(result.energy).all()
  assert len(result.sample_crps) == 5
  assert np.asarray(result.info["structure"]).shape == (3, 3)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_the_triple_runs_end_to_end_from_the_command_line(
  tmp_path: Path, register_linear_gaussian_backend: None
) -> None:
  """Margins on the triple, copulas with RH and w derived, and the tables.

  On synthetic data that obeys the identities, with 5% of cases saturated:
  the moved cases are counted, the copula units carry the triple's log
  score and every response's CRPS, and evaluate reports them.
  """
  del register_linear_gaussian_backend
  from npcc.experiments.application.data import write_synthetic

  data = tmp_path / "data"
  write_synthetic(data, n_stations=2, step_days=7, seed=3, saturated_share=0.05)
  text = config_text(data, stations="[0, 1]").replace(
    "[responses]\n", "[responses]\n" + _TRIPLE
  )
  config = tmp_path / "triple.toml"
  config.write_text(text)
  run = tmp_path / "run"
  for stage in ("margins", "copulas", "evaluate"):
    assert main([stage, "--config", str(config), "--out", str(run)]) == 0

  ms = read_marginal_set(run, 0, "linear-gaussian-all")
  assert ms.variables == _MODELED
  assert ms.derived == _DERIVED
  assert ms.quantiles_test.shape[2] == 3
  rh = ms.y_derived_test[:, 0]
  assert ((rh == 100.0) == (ms.y_test[:, 1] == ms.y_test[:, 0])).all()
  info = json.loads(
    (run / "stations/000/margins/emos-seasonal/info.json").read_text()
  )
  assert info["derived"] == list(_DERIVED)
  detail = info["variables"][DEW_POINT]
  assert detail["saturated_train"] > 0
  assert detail["saturated_test"] >= 0
  saturation = pd.read_csv(run / "tables" / "saturation_by_station.csv")
  assert set(saturation["station"]) == {0, 1}
  pooled = pd.read_csv(run / "tables" / "copulas_pooled.csv")
  assert "joint_nll" not in pooled
  for column in (
    "joint_nll_modeled",
    "joint_nll_modeled_gain",
    "crps_relative_humidity",
    "crps_water_vapor_mixing_ratio_skill",
    "dew_point_above_share",
  ):
    assert column in pooled, column
  assert pooled["dew_point_above_share"].between(0, 1).all()
  margins = pd.read_csv(run / "tables" / "margins_pooled.csv")
  assert set(margins["variable"]) == set(_MODELED)
  unit = json.loads(
    (run / "stations/001/copulas/emos-seasonal/gaussian/info.json").read_text()
  )
  assert unit["modeled"] == list(_MODELED)
  assert unit["derived"] == list(_DERIVED)
  assert math.isclose(
    unit["results"]["gaussian"]["info"]["dew_point_above_draws_share"],
    float(
      pd.read_parquet(
        run / "stations/001/copulas/emos-seasonal/gaussian/scores.parquet"
      )["dew_point_above_share"].mean()
    ),
  )
