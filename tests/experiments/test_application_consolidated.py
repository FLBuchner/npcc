"""The consolidated design: the triple and a 5d design in one run.

The margins are fitted once to all five responses; an arm ``<arm>@5d``
models all five at once, an arm without a design models the triple and
derives the dew point and the mixing ratio. Their test quantiles are stored
as float32 and read back as stored, each arm's draws are stored but not its
members', and the paired summaries compare the 5d arms with the triple's.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from npcc.experiments.application.cli import main
from npcc.experiments.application.config import (
  CopulaDesign,
  load_config,
  split_arm,
)
from npcc.experiments.application.data import write_synthetic
from npcc.experiments.application.derived import (
  DEW_POINT,
  MIXING_RATIO,
  PRESSURE,
  RELATIVE_HUMIDITY,
  TEMPERATURE,
)
from npcc.experiments.application.marginal_set import MarginalSet
from npcc.experiments.application.scores import crps_from_quantiles
from npcc.experiments.application.store import (
  draws_frame,
  read_marginal_set,
  write_copula,
)
from npcc.experiments.config import load_grid
from tests.experiments._application import FakeCopulaResult, config_text

_ROOT = Path(__file__).resolve().parents[2]
_ALL = (
  TEMPERATURE,
  DEW_POINT,
  RELATIVE_HUMIDITY,
  PRESSURE,
  MIXING_RATIO,
)
_TRIPLE = (TEMPERATURE, RELATIVE_HUMIDITY, PRESSURE)
_RESPONSES = (
  'modeled = ["obs:air_temperature", "obs:relative_humidity", '
  '"obs:surface_air_pressure"]\n'
  'derived = ["obs:dew_point_temperature", "obs:water_vapor_mixing_ratio"]\n'
)
_DESIGNS = (
  "[copulas.designs.5d]\n"
  "modeled = " + json.dumps(list(_ALL)) + "\n"
  "derived = []\n\n"
)
_ARMS = ("independence", "gaussian", "independence@5d", "gaussian@5d")


def _config(
  tmp_path: Path,
  data: Path,
  *,
  arms: tuple[str, ...] = _ARMS,
  designs: str = _DESIGNS,
  stations: str = "[0, 1]",
  margins: str = 'train_pits = "out_of_fold"\nquantile_dtype = "float32"\n',
) -> Path:
  text = config_text(data, arms=arms, stations=stations)
  text = text.replace("[responses]\n", "[responses]\n" + _RESPONSES)
  text = text.replace("n_levels = 19\n", "n_levels = 19\n" + margins)
  text = text.replace(
    "sample_batch_size = 50\n", "sample_batch_size = 50\nsave_draws = true\n"
  )
  text = text.replace("[[estimators]]", designs + "[[estimators]]", 1)
  path = tmp_path / "consolidated.toml"
  path.write_text(text)
  return path


def _paired_summary() -> ModuleType:
  path = _ROOT / "scripts" / "paper" / "application" / "paired_summary.py"
  spec = importlib.util.spec_from_file_location("paired_summary", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _thin_draws() -> ModuleType:
  path = _ROOT / "scripts" / "paper" / "application" / "thin_draws.py"
  spec = importlib.util.spec_from_file_location("thin_draws", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------


def test_an_arm_splits_into_its_base_and_its_design() -> None:
  """A design suffix must be read off the arm, and a malformed one refused.

  Otherwise ``gaussian@5d`` would reach the copula stage as an unknown arm,
  or ``gaussian@`` silently run on the main design.
  """
  assert split_arm("gaussian") == ("gaussian", None)
  assert split_arm("npcc-chain-tabpfn-quantiles-all@5d") == (
    "npcc-chain-tabpfn-quantiles-all",
    "5d",
  )
  for bad in ("gaussian@", "@5d", "gaussian@5D", "gaussian@5d@x"):
    with pytest.raises(ValueError, match="design"):
      split_arm(bad)


@pytest.mark.usefixtures("register_linear_gaussian_backend")
def test_the_margins_cover_every_designs_responses(tmp_path: Path) -> None:
  """A 5d design must have margins for the two responses the triple derives.

  The margins are fitted once, to every response any design models, so the
  triple's arms and the 5d arms read the same units.
  """
  config = load_config(_config(tmp_path, Path("data")))

  assert config.modeled == _TRIPLE
  assert config.margin_responses == tuple(sorted(_ALL))
  assert config.design(None) == CopulaDesign(
    modeled=_TRIPLE, derived=(DEW_POINT, MIXING_RATIO)
  )
  assert config.design("5d") == CopulaDesign(modeled=tuple(sorted(_ALL)))


@pytest.mark.usefixtures("register_linear_gaussian_backend")
def test_designs_and_float32_enter_the_signature_only_when_set(
  tmp_path: Path,
) -> None:
  """A run folder of another design or dtype must refuse to resume.

  And a configuration without either keeps the signature it had, so every
  earlier run folder still resumes.
  """
  plain = _config(
    tmp_path, Path("data"), designs="", arms=("gaussian",), margins=""
  )
  base = load_config(plain).signature()
  with_designs = load_config(
    _config(tmp_path, Path("data"), arms=("gaussian",), margins="")
  ).signature()
  with_float32 = load_config(
    _config(
      tmp_path,
      Path("data"),
      designs="",
      arms=("gaussian",),
      margins='quantile_dtype = "float32"\n',
    )
  ).signature()
  assert len({base, with_designs, with_float32}) == 3
  explicit = load_config(
    _config(
      tmp_path,
      Path("data"),
      designs="",
      arms=("gaussian",),
      margins='quantile_dtype = "float64"\n',
    )
  ).signature()
  assert explicit == base


@pytest.mark.usefixtures("register_linear_gaussian_backend")
@pytest.mark.parametrize(
  ("designs", "arms", "match"),
  [
    (_DESIGNS, ("gaussian@6d",), "must name a design"),
    (
      "[copulas.designs.5d]\nmodeled = "
      + json.dumps(list(_TRIPLE))
      + "\nderived = []\n\n",
      ("gaussian@5d",),
      "model or derive every response",
    ),
    (
      "[copulas.designs.5D]\nmodeled = " + json.dumps(list(_ALL)) + "\n\n",
      ("gaussian",),
      "lowercase",
    ),
    (
      "[copulas.designs.5d]\nmodeled = "
      + json.dumps([TEMPERATURE, RELATIVE_HUMIDITY, PRESSURE, DEW_POINT])
      + '\nderived = ["obs:dew_point_temperature", '
      '"obs:water_vapor_mixing_ratio"]\n\n',
      ("gaussian@5d",),
      "not both",
    ),
    (
      "[copulas.designs.5d]\nmodeled = "
      + json.dumps(list(_ALL))
      + '\nextra = "x"\n\n',
      ("gaussian@5d",),
      "only modeled and derived",
    ),
  ],
)
def test_a_design_no_run_could_honor_is_refused(
  tmp_path: Path, designs: str, arms: tuple[str, ...], match: str
) -> None:
  """An arm of an unknown design, or a design that scores less, is refused.

  A design that left a response neither modeled nor derived would score
  its arms on fewer responses than the triple's, so their energy scores
  could not be compared.
  """
  with pytest.raises(ValueError, match=match):
    load_config(_config(tmp_path, Path("data"), designs=designs, arms=arms))


@pytest.mark.parametrize(
  "name", ["application_consolidated", "application_consolidated_smoke"]
)
def test_the_shipped_consolidated_configs(name: str) -> None:
  """The rerun's design as decided, before any of it runs.

  The chain must stay off the EMOS margins (section 4: a conditional copula
  on conditionally miscalibrated PITs partly recalibrates them); the triple
  is the main design; the 5d arms are the ones the plausibility comparison
  pairs; every margin is E1's estimator on all five responses, cross-fitted
  and stored as float32; and the draws are stored.
  """
  config = load_config(_ROOT / "configs" / f"{name}.toml")
  e1 = {
    e.label: e for e in load_grid(_ROOT / "configs" / "study.toml").estimators
  }
  for estimator in config.estimators:
    assert estimator.estimator_id == e1[estimator.label].estimator_id

  assert config.modeled == _TRIPLE
  assert config.derived == (DEW_POINT, MIXING_RATIO)
  assert config.margin_responses == tuple(sorted(_ALL))
  assert config.models == (
    "emos-seasonal",
    "lightgbm-all",
    "lightgbm-emos",
    "tabpfn-quantiles-all",
    "tabpfn-quantiles-emos",
    "tabicl-all",
    "tabicl-emos",
  )
  assert config.train_pits == "out_of_fold"
  assert config.quantile_dtype == "float32"
  assert config.save_draws
  assert config.n_draws == 200
  assert config.mixture_k == 5
  assert config.stations == ((0,) if "smoke" in name else None)
  classical = ("independence", "gaussian", "boostcopula-all")
  chain = "npcc-chain-tabpfn-quantiles-all"
  assert config.copula_arms_for("emos-seasonal") == (
    *classical,
    "independence@5d",
    "gaussian@5d",
  )
  for model in config.models[1:]:
    if model == "tabpfn-quantiles-all":
      continue
    assert config.copula_arms_for(model) == (*classical, chain)
  assert config.copula_arms_for("tabpfn-quantiles-all") == (
    *classical,
    "boostcopula-time",
    "npcc-mixture-tabpfn-quantiles-all",
    "npcc-mixture-tabpfn-quantiles-time",
    chain,
    "npcc-chain-tabpfn-quantiles-time",
    "independence@5d",
    "gaussian@5d",
    f"{chain}@5d",
  )
  assert not any(
    a.startswith("npcc-") for a in config.copula_arms_for("emos-seasonal")
  )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _run(tmp_path: Path) -> tuple[Path, Path]:
  data = tmp_path / "data"
  write_synthetic(data, n_stations=2, step_days=7, seed=3, measured=True)
  config = _config(tmp_path, data)
  run = tmp_path / "run"
  assert main(["preflight", "--config", str(config)]) == 0
  for stage in ("margins", "copulas", "evaluate"):
    assert main([stage, "--config", str(config), "--out", str(run)]) == 0
  return config, run


def test_float32_quantiles_are_stored_and_read_exactly_as_scored(
  tmp_path: Path, register_linear_gaussian_backend: None
) -> None:
  """The copula stage must see the quantiles the margin stage scored.

  They are rounded to float32 once, before scoring and storing, so the
  stored file, the marginal scores and the copula stage's set hold the same
  numbers, and a rerun from the files reproduces every score.
  """
  del register_linear_gaussian_backend
  _, run = _run(tmp_path)
  unit = run / "stations" / "000" / "margins" / "linear-gaussian-all"
  predictions = pd.read_parquet(unit / "predictions.parquet")
  q = [c for c in predictions.columns if c.startswith("q_")]
  assert (predictions[q].dtypes == np.float32).all()
  scores = pd.read_parquet(unit / "scores.parquet")
  levels = np.asarray(json.loads((unit / "info.json").read_text())["levels"])
  ms = read_marginal_set(run, 0, "linear-gaussian-all")
  assert ms.variables == tuple(sorted(_ALL))
  for j, variable in enumerate(ms.variables):
    test = predictions[
      (predictions["variable"] == variable) & (predictions["split"] == "test")
    ]
    stored = test[q].to_numpy(np.float64)
    np.testing.assert_array_equal(ms.quantiles_test[:, :, j], stored)
    assert (stored == stored.astype(np.float32)).all()
    crps = scores.loc[scores["variable"] == variable, "crps"].to_numpy()
    # The same quantiles, so the same CRPS to the last bits a summation over
    # a differently laid out array can move (a float64 rounding of a
    # float32's value would move it by 1e-8).
    np.testing.assert_allclose(
      crps, crps_from_quantiles(ms.y_test[:, j], stored, levels), rtol=1e-13
    )


def test_both_designs_run_on_the_same_margins_and_are_compared(
  tmp_path: Path, register_linear_gaussian_backend: None
) -> None:
  """A 5d arm models all five responses, the triple's derives two of them.

  Each design's log score has its own column and reference; both are
  scored on every response's CRPS and the physical bounds; and the paired
  summary compares the 5d arms with the triple's on the scores both draw,
  never on the log score.
  """
  del register_linear_gaussian_backend
  _, run = _run(tmp_path)
  model = "emos-seasonal"
  five = pd.read_parquet(
    run / "stations/000/copulas" / model / "gaussian@5d" / "scores.parquet"
  )
  three = pd.read_parquet(
    run / "stations/000/copulas" / model / "gaussian" / "scores.parquet"
  )
  assert "joint_nll_5d" in five and "joint_nll_modeled" not in five
  assert "joint_nll_modeled" in three and "joint_nll_5d" not in three
  for variable in _ALL:
    column = "crps_" + variable.removeprefix("obs:")
    assert five[column].notna().all() and three[column].notna().all()
  # The triple's draws break no bound by construction; the 5d draws of a
  # Gaussian copula on independent margins put Td above T at some cases.
  assert (three["broken_td_above_t"] == 0).all()
  assert (three["broken_rh_outside_0_100"] == 0).all()
  assert five["broken_td_above_t"].sum() > 0

  draws5 = pd.read_parquet(
    run / "stations/000/copulas" / model / "gaussian@5d" / "draws.parquet"
  )
  draws3 = pd.read_parquet(
    run / "stations/000/copulas" / model / "gaussian" / "draws.parquet"
  )
  for draws in (draws5, draws3):
    assert set(_ALL) <= set(draws.columns)
  assert set(draws5["arm"]) == {"gaussian@5d"}
  np.testing.assert_array_equal(
    (draws3[DEW_POINT] > draws3[TEMPERATURE]).to_numpy(), False
  )

  by_station = pd.read_csv(run / "tables" / "copulas_by_station.csv")
  row = by_station.set_index(["station", "model", "arm"])
  gain = row.loc[(0, model, "gaussian@5d"), "joint_nll_5d_gain"]
  expected = (
    row.loc[(0, model, "independence@5d"), "joint_nll_5d"]
    - row.loc[(0, model, "gaussian@5d"), "joint_nll_5d"]
  )
  assert gain == pytest.approx(expected)
  assert np.isnan(row.loc[(0, model, "gaussian@5d"), "joint_nll_modeled"])
  energy_skill = row.loc[(0, model, "gaussian@5d"), "energy_skill"]
  assert energy_skill == pytest.approx(
    1.0
    - row.loc[(0, model, "gaussian@5d"), "energy"]
    / row.loc[(0, model, "independence@5d"), "energy"]
  )
  validity = pd.read_csv(run / "tables" / "copulas_physical_validity.csv")
  assert {"gaussian", "gaussian@5d"} <= set(validity["arm"])

  ps = _paired_summary()
  ps.build(run, run / "analysis")
  paired = pd.read_csv(run / "analysis" / "paired.csv")
  same = paired[paired["reference"] == "gaussian@5d (same margins)"]
  assert set(same["arm"]) == {"independence@5d"}
  assert "joint_nll_5d" in set(same["score"])
  cross = paired[
    paired["reference"].str.endswith("(main design, same margins)")
  ]
  assert set(cross["arm"]) == {"independence@5d", "gaussian@5d"}
  assert set(cross["reference"]) == {
    "independence (main design, same margins)",
    "gaussian (main design, same margins)",
  }
  assert not cross["score"].str.startswith("joint_nll").any()
  assert {"energy", "variogram", "broken_td_above_t_share"} <= set(
    cross["score"]
  )
  td = cross[
    (cross["stations"] == "all")
    & (cross["arm"] == "gaussian@5d")
    & (cross["model"] == model)
    & (cross["score"] == "broken_td_above_t_share")
  ]
  assert len(td) == 1 and td["mean_diff"].iloc[0] > 0
  per_station = pd.read_csv(run / "analysis" / "paired_by_station.csv")
  one = per_station[
    (per_station["station"] == 0)
    & (per_station["model"] == model)
    & (per_station["arm"] == "gaussian@5d")
    & (per_station["reference"] == "gaussian (main design, same margins)")
    & (per_station["score"] == "energy")
  ]
  assert one["diff"].iloc[0] == pytest.approx(
    row.loc[(0, model, "gaussian@5d"), "energy"]
    - row.loc[(0, model, "gaussian"), "energy"]
  )


def test_a_mixtures_members_draws_are_not_stored(tmp_path: Path) -> None:
  """Only the arm's own draws are stored; its members keep their scores.

  Storing every member's would multiply the disk the mixtures take by
  their number of members.
  """
  rng = np.random.default_rng(0)
  n, m = 4, 3
  ms = MarginalSet(
    station=0,
    model="emos-seasonal",
    variables=(TEMPERATURE,),
    feature_names=("x",),
    x_train=np.zeros((5, 1)),
    x_test=np.zeros((n, 1)),
    y_train=rng.normal(size=(5, 1)),
    y_test=rng.normal(size=(n, 1)),
    time_train=np.arange(5).astype("datetime64[D]").astype("datetime64[ns]"),
    time_test=np.arange(n).astype("datetime64[D]").astype("datetime64[ns]"),
    pit_train=rng.random((5, 1)),
    pit_test=rng.random((n, 1)),
    logpdf_test=np.zeros((n, 1)),
    levels=np.array([0.25, 0.5, 0.75]),
    quantiles_test=np.sort(rng.normal(size=(n, 3, 1)), axis=1),
  )

  def result(arm: str) -> FakeCopulaResult:
    return FakeCopulaResult(
      arm=arm,
      logcop_test=np.zeros(n),
      joint_nll=np.ones(n),
      energy=np.ones(n),
      variogram=np.ones(n),
      draws=rng.normal(size=(n, m, 1)),
    )

  arm = "npcc-mixture-tabpfn-quantiles-all"
  results = [result(arm), result(f"{arm}-m1"), result(f"{arm}-m2")]
  write_copula(
    tmp_path / "run",
    ms,
    arm,
    results,
    rows=np.arange(n),
    seed=1,
    wall_seconds=0.0,
    save_draws=True,
  )
  unit = tmp_path / "run" / "stations/000/copulas/emos-seasonal" / arm
  draws = pd.read_parquet(unit / "draws.parquet")
  assert set(draws["arm"]) == {arm}
  assert len(draws) == n * m
  scores = pd.read_parquet(unit / "scores.parquet")
  assert set(scores["arm"]) == {arm, f"{arm}-m1", f"{arm}-m2"}
  with pytest.raises(ValueError, match="sampled result"):
    draws_frame(ms, arm, [], rows=np.arange(n))


def test_thinning_keeps_the_first_draws_of_the_named_arms_only(
  tmp_path: Path,
) -> None:
  """The disk-full fallback must touch only what it is told to, and say so.

  It keeps draws 0 to k - 1 of each case, leaves every other arm and every
  score as they were, records what it did in the unit and in the run's log,
  and leaves an already thin unit alone.
  """
  td = _thin_draws()
  run = tmp_path / "run"
  frames = {}
  for arm in ("gaussian", "gaussian@5d"):
    unit = run / "stations/000/copulas/emos-seasonal" / arm
    unit.mkdir(parents=True)
    frame = pd.DataFrame(
      {
        "row": np.repeat(np.arange(3), 10),
        "draw": np.tile(np.arange(10, dtype=np.int32), 3),
        TEMPERATURE: np.arange(30, dtype=np.float32),
      }
    )
    frame.to_parquet(unit / "draws.parquet", index=False)
    (unit / "scores.parquet").write_bytes(b"scores")
    frames[arm] = frame

  records = td.thin(run, ["gaussian@5d"], 4)

  assert [r["action"] for r in records] == ["thinned"]
  unit5 = run / "stations/000/copulas/emos-seasonal/gaussian@5d"
  thinned = pd.read_parquet(unit5 / "draws.parquet")
  pd.testing.assert_frame_equal(
    thinned,
    frames["gaussian@5d"][frames["gaussian@5d"]["draw"] < 4].reset_index(
      drop=True
    ),
  )
  assert (unit5 / "scores.parquet").read_bytes() == b"scores"
  assert json.loads((unit5 / "draws_thinned.json").read_text())["keep"] == 4
  untouched = pd.read_parquet(
    run / "stations/000/copulas/emos-seasonal/gaussian/draws.parquet"
  )
  pd.testing.assert_frame_equal(untouched, frames["gaussian"])
  again = td.thin(run, ["gaussian@5d"], 4)
  assert again[0]["action"].startswith("left as it is")
  log = (run / "logs" / "draws_thinned.jsonl").read_text().splitlines()
  assert len(log) == 2
  with pytest.raises(ValueError, match="at least 2"):
    td.thin(run, ["gaussian"], 1)
