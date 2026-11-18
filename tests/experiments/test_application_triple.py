"""The triple design: (T, RH, p) modeled, the dew point and w derived."""

from __future__ import annotations

import json
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
from npcc.experiments.application.data import write_synthetic
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
  dew_point,
  mixing_ratio,
  residuals,
  saturation_vapor_pressure,
  vapor_pressure,
)
from npcc.experiments.application.emos import gaussian_crps
from npcc.experiments.application.joint_scores import (
  energy_score,
  inverse_margins,
  variogram_score,
)
from npcc.experiments.application.marginal_set import MarginalSet
from npcc.experiments.application.store import read_marginal_set
from npcc.experiments.config import EstimatorSpec
from tests.experiments._application import config_text

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "r_methods" / "fit_boostcopula_vine.R"
_MODELED = (TEMPERATURE, RELATIVE_HUMIDITY, PRESSURE)
_DERIVED = (DEW_POINT, MIXING_RATIO)
_TRIPLE = (
  'modeled = ["obs:air_temperature", "obs:relative_humidity", '
  '"obs:surface_air_pressure"]\n'
  'derived = ["obs:dew_point_temperature", "obs:water_vapor_mixing_ratio"]\n'
)


# ---------------------------------------------------------------------------
# The formulas
# ---------------------------------------------------------------------------


def test_meteoswiss_magnus_gives_the_hand_values_on_both_sides_of_zero() -> (
  None
):
  """A wrong constant or branch would derive another Td from every draw.

  By hand: ``6.107 exp(173.68 / 248.83) = 12.2733`` hPa at 10 C,
  ``6.108 exp(-178.56 / 235.52) = 2.86179`` hPa at -10 C, and 6.107 at 0.
  """
  np.testing.assert_allclose(
    saturation_vapor_pressure(np.array([10.0, -10.0, 0.0])),
    [12.2733, 2.86179, 6.107],
    rtol=2e-5,
  )
  np.testing.assert_allclose(
    SONNTAG_1990.saturation_vapor_pressure(np.array([0.0, 20.0])),
    [6.112, 23.3263],
    rtol=2e-5,
  )
  assert dict(MAGNUS) == {
    "sonntag1990": SONNTAG_1990,
    "meteoswiss": METEOSWISS,
  }


@pytest.mark.parametrize("name", ["meteoswiss", "sonntag1990"])
def test_the_dew_point_inverts_the_formula_at_saturation(name: str) -> None:
  """At RH = 100 the dew point is the air temperature, on either side of 0.

  A piecewise inverse that took other constants than ``e_s(T)`` did would
  move it, by up to 0.0023 C just below 0.
  """
  formula = MAGNUS[name]
  t = np.array([-30.0, -5.0, -0.3, -0.01, -1e-6, 0.0, 1e-6, 0.3, 5.0, 35.0])
  np.testing.assert_allclose(
    formula.dew_point(formula.saturation_vapor_pressure(t), t), t, atol=1e-9
  )
  np.testing.assert_allclose(
    dew_point(t, np.full_like(t, 100.0), formula), t, atol=1e-9
  )


def test_the_dew_point_takes_the_air_temperatures_constants() -> None:
  """A dew point below 0 under air above it keeps the warm constants.

  That is how MeteoSwiss computed the data's Td. Inverting on the dew
  point's own branch instead, with the cold constants, puts it 0.0045 C
  lower at T = 2 C and RH = 60 %, and 0.027 C lower at T = 0.5 C and
  RH = 30 %: residuals of up to 0.24 C against the stored data.

  By hand, with the warm constants throughout: ``e = 0.6 x 6.107
  exp(34.736 / 240.83) = 4.23271`` hPa, ``g = ln(e / 6.107) = -0.366593``,
  ``Td = 238.83 g / (17.368 - g) = -4.93685`` C; at T = 0.5 C and RH = 30 %,
  -15.0455 C; and at T = -2 C and RH = 60 %, with the cold constants
  throughout, -8.71923 C.
  """
  td = dew_point(np.array([2.0, 0.5, -2.0]), np.array([60.0, 30.0, 60.0]))
  np.testing.assert_allclose(td, [-4.93685, -15.0455, -8.71923], atol=1e-4)
  np.testing.assert_allclose(
    METEOSWISS.saturation_vapor_pressure(td, np.array([2.0, 0.5, -2.0])),
    vapor_pressure(np.array([2.0, 0.5, -2.0]), np.array([60.0, 30.0, 60.0])),
    rtol=1e-12,
  )
  own_branch = METEOSWISS.dew_point(
    vapor_pressure(np.array([2.0]), np.array([60.0])), td[:1]
  )
  assert float(td[0] - own_branch[0]) == pytest.approx(0.0045, abs=1e-4)


def test_the_derivation_reproduces_the_stored_dew_point() -> None:
  """MeteoSwiss's Td is the derivation of its T and RH, to float32 precision.

  A coefficient convention other than the data's leaves residuals of up to
  0.24 C where T and Td straddle 0 C; the data's leaves float32 rounding,
  about 1e-5 C. Skipped without the data.
  """
  path = Path("~/npcc-data/meteoswiss_zanetta2023/y.nc").expanduser()
  if not path.exists():
    pytest.skip(f"the MeteoSwiss data is not at {path}")
  xr = pytest.importorskip("xarray")
  with xr.open_dataset(path) as y:
    rows = slice(None, None, 97)
    t, rh, td = (
      y[name].values[rows].astype(np.float64)
      for name in (TEMPERATURE, RELATIVE_HUMIDITY, DEW_POINT)
    )
  ok = np.isfinite(t) & np.isfinite(rh) & np.isfinite(td) & (rh > 0)
  t, rh, td = t[ok], rh[ok], td[ok]
  straddle = (t >= 0) != (td >= 0)
  assert straddle.sum() > 1000
  residual = dew_point(t, rh) - td
  assert np.abs(residual).max() < 1e-4
  assert np.abs(residual[straddle]).max() < 1e-4


def test_the_mixing_ratio_gives_the_hand_value() -> None:
  """``e = 0.5 x 12.2733 = 6.13665`` hPa; ``w = 622 x 6.13665 / 893.86335``.

  That is 4.27027 g/kg at T = 10 C, RH = 50 % and p = 900 hPa.
  """
  e = vapor_pressure(np.array(10.0), np.array(50.0))
  assert float(e) == pytest.approx(6.13665, rel=2e-5)
  w = mixing_ratio(np.array(10.0), np.array(50.0), np.array(900.0))
  assert float(w) == pytest.approx(4.27027, rel=2e-5)


def test_derived_draws_follow_each_triple() -> None:
  """Every draw of (T, RH, p) gives its own Td and w; RH = 100 gives Td = T."""
  draws = np.array(
    [
      [[10.0, 100.0, 900.0], [10.0, 40.0, 905.0], [-2.0, 85.0, 1000.0]],
      [[0.2, 99.5, 950.0], [-0.2, 100.0, 950.0], [25.0, 30.0, 950.0]],
    ]
  )
  values = derive(draws, _MODELED, _DERIVED)

  assert values.shape == (2, 3, 2)
  np.testing.assert_array_equal(
    values[..., 0], dew_point(draws[..., 0], draws[..., 1])
  )
  np.testing.assert_array_equal(
    values[..., 1], mixing_ratio(draws[..., 0], draws[..., 1], draws[..., 2])
  )
  at_saturation = draws[..., 1] == 100.0
  np.testing.assert_allclose(
    values[..., 0][at_saturation], draws[..., 0][at_saturation], atol=1e-12
  )
  assert (values[..., 0] <= draws[..., 0] + 1e-12).all()
  sonntag = derive(draws, _MODELED, _DERIVED, Derivation("sonntag1990"))
  assert not np.allclose(sonntag[..., 0], values[..., 0])


def test_a_derived_response_needs_its_inputs_modeled() -> None:
  """Td without a modeled RH has nothing to be computed from."""
  draws = np.zeros((1, 2, 2))
  with pytest.raises(ValueError, match="not modeled"):
    derive(draws, (TEMPERATURE, PRESSURE), (DEW_POINT,))
  with pytest.raises(ValueError, match="derived must be drawn"):
    derive(draws, (TEMPERATURE, RELATIVE_HUMIDITY), (PRESSURE,))


def test_the_residual_check_reproduces_observed_values_to_rounding() -> None:
  """On values rounded as the data stores them, the check reports rounding.

  The observed Td and w here are the derivation's own, rounded to 0.1 and
  0.01, so its residuals are rounding errors; another formula's are not.
  """
  rng = np.random.default_rng(2)
  t = np.round(rng.uniform(-15, 30, 500), 1)
  rh = np.round(rng.uniform(20, 100, 500), 1)
  p = np.round(rng.uniform(850, 1030, 500), 1)
  values = np.column_stack([t, rh, p])
  observed = np.column_stack(
    [np.round(dew_point(t, rh), 1), np.round(mixing_ratio(t, rh, p), 2)]
  )

  found = residuals(values, _MODELED, _DERIVED, observed)
  assert found[DEW_POINT]["max_abs"] <= 0.05 + 1e-9
  assert found[MIXING_RATIO]["max_abs"] <= 0.005 + 1e-9
  assert 0 < found[DEW_POINT]["sd"] < 0.035
  assert found[DEW_POINT]["n"] == 500
  wrong = residuals(
    values, _MODELED, _DERIVED, observed, Derivation("sonntag1990")
  )
  assert wrong[DEW_POINT]["max_abs"] > 0.05


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
  assert config.derivation == Derivation("meteoswiss")
  assert not config.is_default_design
  assert default.is_default_design
  assert config.signature() != default.signature()
  assert config.echo()["derived"] == list(_DERIVED)
  assert "derived" not in default.echo()


@pytest.mark.parametrize(
  "name", ["application_triple", "application_smoke_triple"]
)
def test_the_shipped_triple_configs(name: str) -> None:
  """The measured triple, with each run's margins and arms.

  The full run must not fit an npcc copula, vine or chain, to the EMOS
  margins, whose PITs are not uniform given the covariates; the smoke run
  fitted every vine arm to both, and its config must still say so. Both
  fit the copulas to out-of-fold training PITs, never in-sample ones,
  and store every scored result's draws.
  """
  config = load_config(_ROOT / "configs" / f"{name}.toml")

  assert config.modeled == _MODELED
  assert config.derived == _DERIVED
  assert config.magnus == "meteoswiss"
  assert config.models == ("emos-seasonal", "tabpfn-quantiles-all")
  assert config.copula_models == config.models
  classical = (
    "independence",
    "gaussian",
    "boostcopula-all",
    "boostcopula-time",
  )
  mixtures = (
    "npcc-mixture-tabpfn-quantiles-all",
    "npcc-mixture-tabpfn-quantiles-time",
  )
  if "smoke" in name:
    randoms = (
      "npcc-random-tabpfn-quantiles-all",
      "npcc-random-tabpfn-quantiles-time",
    )
    assert config.copula_arms == (*classical, *randoms, *mixtures)
    assert config.copula_arms_for("emos-seasonal") == config.copula_arms
  else:
    chains = (
      "npcc-chain-tabpfn-quantiles-all",
      "npcc-chain-tabpfn-quantiles-time",
    )
    assert config.copula_arms == (*classical, *mixtures, *chains)
    assert config.copula_arms_for("emos-seasonal") == classical
  assert config.copula_arms_for("tabpfn-quantiles-all") == config.copula_arms
  assert config.responses[RELATIVE_HUMIDITY].transform == "logit"
  assert config.responses[RELATIVE_HUMIDITY].boundary_nudge == 0.05
  assert config.device_for("tabpfn-quantiles") == "cuda"
  assert config.stations == ((0,) if "smoke" in name else None)
  assert config.train_pits == "out_of_fold"
  assert config.save_draws


@pytest.mark.usefixtures("register_linear_gaussian_backend")
@pytest.mark.parametrize(
  ("block", "match"),
  [
    ('modeled = ["obs:air_temperature", "obs:wind"]\n', "subset"),
    ("modeled = []\n", "subset"),
    (
      'modeled = ["obs:air_temperature", "obs:relative_humidity"]\n'
      'derived = ["obs:water_vapor_mixing_ratio"]\n',
      "not modeled",
    ),
    (
      'modeled = ["obs:air_temperature", "obs:relative_humidity", '
      '"obs:surface_air_pressure", "obs:dew_point_temperature"]\n'
      'derived = ["obs:dew_point_temperature"]\n',
      "not both",
    ),
    (
      'modeled = ["obs:air_temperature", "obs:relative_humidity", '
      '"obs:surface_air_pressure"]\nderived = ["obs:dew_point_temperature"]\n',
      "modeled or derived",
    ),
    (
      'modeled = ["obs:air_temperature", "obs:dew_point_temperature", '
      '"obs:surface_air_pressure", "obs:water_vapor_mixing_ratio"]\n'
      'derived = ["obs:relative_humidity"]\n',
      "with a formula",
    ),
    (_TRIPLE + 'magnus = "goff-gratch"\n', "magnus must be one of"),
    (_TRIPLE + "saturation_nudge = 0.05\n", "saturation_nudge"),
  ],
)
def test_a_design_that_cannot_be_derived_is_refused(
  tmp_path: Path, block: str, match: str
) -> None:
  """A derived response without inputs, or a response left out, must stop."""
  with pytest.raises((ValueError, TypeError), match=match):
    load_config(_triple_config(tmp_path, block))


# ---------------------------------------------------------------------------
# A copula arm on the triple
# ---------------------------------------------------------------------------


def _triple_set(
  n_train: int = 120, n_test: int = 5, seed: int = 0
) -> MarginalSet:
  """Gaussian margins of (T, RH, p), Td and w derived from the observations."""
  rng = np.random.default_rng(seed)
  mu = np.array([12.0, 70.0, 950.0])
  sigma = np.array([4.0, 8.0, 8.0])
  levels = (np.arange(99) + 0.5) / 99

  def sample(n: int) -> np.ndarray:
    y = mu + sigma * rng.standard_normal((n, 3))
    y[:, 1] = np.clip(y[:, 1], 5.0, 100.0)
    return y

  y_train, y_test = sample(n_train), sample(n_test)
  quantiles = mu + sigma * stats.norm.ppf(levels)[:, None]
  quantiles[:, 1] = np.clip(quantiles[:, 1], 1.0, 100.0)
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
    time_train=np.arange(n_train).astype("datetime64[ns]"),
    time_test=np.arange(n_test).astype("datetime64[ns]"),
    pit_train=stats.norm.cdf((y_train - mu) / sigma),
    pit_test=stats.norm.cdf((y_test - mu) / sigma),
    logpdf_test=stats.norm.logpdf(y_test, loc=mu, scale=sigma),
    levels=levels,
    quantiles_test=np.broadcast_to(
      quantiles, (n_test, *quantiles.shape)
    ).copy(),
    derived=_DERIVED,
    y_derived_train=derive(y_train, _MODELED, _DERIVED),
    y_derived_test=derive(y_test, _MODELED, _DERIVED),
  )


def _run(
  arm: str,
  ms: MarginalSet,
  n_draws: int = 40,
  estimators: dict[str, EstimatorSpec] | None = None,
  mixture_k: int = 1,
) -> list[copulas.CopulaResult]:
  return copulas.run_copula_arm(
    arm,
    ms,
    estimators=estimators or {},
    n_draws=n_draws,
    sample_batch_size=50,
    mixture_k=mixture_k,
    seed=9,
    device="cpu",
  )


def test_the_modeled_log_score_adds_the_copula_to_three_margins() -> None:
  """``joint_nll_modeled`` is the copula's log-density plus the triple's."""
  ms = _triple_set()
  for arm in ("independence", "gaussian"):
    (result,) = _run(arm, ms)
    np.testing.assert_allclose(
      result.joint_nll,
      -(result.logcop_test + ms.logpdf_test.sum(axis=1)),
      rtol=1e-14,
    )
    assert result.info["modeled"] == list(_MODELED)
    assert result.info["derived"] == list(_DERIVED)
    assert result.info["magnus"] == "meteoswiss"


def test_the_energy_score_covers_all_five_with_the_derived_draws() -> None:
  """The joint scores read Td and w from each draw of the triple.

  Rebuilt from the arm's own draws: the triple's through the quantiles, Td
  and w derived from them, all five divided by their training SD; each
  response's CRPS is the one-dimensional energy score in its units.
  """
  ms = _triple_set()
  (result,) = _run("independence", ms)
  seed = derive_seed(9, ms.station, ms.model, "independence", "draws")
  uniforms = np.random.default_rng(seed).random((len(ms.y_test), 40, 3))
  draws = inverse_margins(uniforms, ms.levels, ms.quantiles_test)
  full = np.concatenate([draws, derive(draws, _MODELED, _DERIVED)], axis=2)
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


def test_the_sample_crps_is_the_gaussian_crps() -> None:
  """The CRPS from draws estimates the CRPS, here against its closed form."""
  rng = np.random.default_rng(5)
  mu, sigma = 1.5, 2.0
  y = np.array([[-1.0], [1.5], [6.0]])
  draws = mu + sigma * rng.standard_normal((3, 40_000, 1))

  np.testing.assert_allclose(
    energy_score(draws, y), gaussian_crps(y[:, 0], mu, sigma), rtol=2e-2
  )


def test_three_variables_have_three_structures() -> None:
  """A 3-d R-vine is fixed by the middle variable of its first tree."""
  assert [copulas.n_rvine_structures(d) for d in (2, 3, 4, 5)] == [
    1,
    3,
    24,
    480,
  ]
  edges = [
    frozenset(copulas._external_vine.edges(s))
    for s in copulas.all_structures(3)
  ]
  assert len(set(edges)) == 3
  with pytest.raises(ValueError, match="smaller mixture_k"):
    copulas.all_structures(4)


def test_the_d3_mixture_averages_every_structure(
  register_linear_gaussian_backend: None,
) -> None:
  """At d = 3 the mixture is of all three structures, the selected first.

  Asked for five members, it takes three: TLL's selection, then the other
  two in a fixed order, each also scored on its own.
  """
  del register_linear_gaussian_backend
  spec = EstimatorSpec("lin", "linear-gaussian", "probit")
  ms = _triple_set(n_train=80, n_test=3)
  results = _run(
    "npcc-mixture-tabpfn-quantiles-time",
    ms,
    n_draws=6,
    estimators={"tabpfn-quantiles": spec},
    mixture_k=5,
  )

  assert [r.arm for r in results] == [
    "npcc-mixture-tabpfn-quantiles-time",
    *(f"npcc-mixture-tabpfn-quantiles-time-m{j}" for j in (1, 2, 3)),
  ]
  mixture = results[0]
  assert mixture.info["mixture_k"] == 3
  assert mixture.info["mixture_k_requested"] == 5
  assert mixture.info["every_structure"] is True
  assert mixture.info["n_distinct_structures"] == 3
  members = mixture.info["members"]
  assert isinstance(members, list)
  assert [m["structure_method"] for m in members] == [
    "dissmann-tll",
    "enumerated",
    "enumerated",
  ]
  logs = np.stack([r.logcop_test for r in results[1:]])
  np.testing.assert_allclose(
    mixture.logcop_test, np.log(np.exp(logs).mean(axis=0)), rtol=1e-10
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
  """boostCopula fits three variables; all five are scored."""
  try:
    copulas.check_boostcopula(str(_SCRIPT))
  except RuntimeError as exc:
    pytest.skip(f"needs boostCopula: {exc}")
  monkeypatch.setattr(copulas, "BOOSTCOPULA_SCRIPT", str(_SCRIPT))
  ms = _triple_set(n_train=60, n_test=3)
  (result,) = _run("boostcopula-time", ms, n_draws=6)

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
  """Margins on the triple, copulas with Td and w derived, and the tables.

  On synthetic data that obeys the identities: the RH = 100 cases are
  counted, the derivation reproduces the observed Td and w, the copula
  units carry the triple's log score and every response's CRPS, and
  evaluate reports them.
  """
  del register_linear_gaussian_backend
  data = tmp_path / "data"
  write_synthetic(data, n_stations=2, step_days=7, seed=3, measured=True)
  text = config_text(data, stations="[0, 1]").replace(
    "[responses]\n", "[responses]\n" + _TRIPLE
  )
  config = tmp_path / "triple.toml"
  config.write_text(text)
  run = tmp_path / "run"
  assert main(["preflight", "--config", str(config)]) == 0
  for stage in ("margins", "copulas", "evaluate"):
    assert main([stage, "--config", str(config), "--out", str(run)]) == 0

  ms = read_marginal_set(run, 0, "linear-gaussian-all")
  assert ms.variables == _MODELED
  assert ms.derived == _DERIVED
  assert ms.quantiles_test.shape[2] == 3
  info = json.loads(
    (run / "stations/000/margins/emos-seasonal/info.json").read_text()
  )
  assert info["derived"] == list(_DERIVED)
  detail = info["variables"][RELATIVE_HUMIDITY]
  rh_train = ms.y_train[:, 1]
  assert detail["at_upper_train"] == int((rh_train == 100.0).sum()) > 0
  assert detail["at_lower_train"] == 0
  bounds = pd.read_csv(run / "tables" / "bounds_by_station.csv")
  assert set(bounds["station"]) == {0, 1}
  assert set(bounds["variable"]) == {RELATIVE_HUMIDITY}
  residual = pd.read_csv(run / "tables" / "derivation_residuals.csv")
  assert set(residual["variable"]) == set(_DERIVED)
  assert (residual["max_abs"] < 1e-9).all()
  pooled = pd.read_csv(run / "tables" / "copulas_pooled.csv")
  assert "joint_nll" not in pooled
  for column in (
    "joint_nll_modeled",
    "joint_nll_modeled_gain",
    "crps_dew_point_temperature",
    "crps_water_vapor_mixing_ratio_skill",
  ):
    assert column in pooled, column
  margins = pd.read_csv(run / "tables" / "margins_pooled.csv")
  assert set(margins["variable"]) == set(_MODELED)
  unit = json.loads(
    (run / "stations/001/copulas/emos-seasonal/gaussian/info.json").read_text()
  )
  assert unit["modeled"] == list(_MODELED)
  assert unit["derived"] == list(_DERIVED)
