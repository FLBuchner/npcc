"""Two designs in one run: the triple and the five responses.

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
from scipy.stats import multivariate_normal, norm

from npcc.experiments.application.cli import main
from npcc.experiments.application.config import (
  CopulaDesign,
  load_config,
  split_arm,
)
from npcc.experiments.application.copulas import PIT_CLIP
from npcc.experiments.application.data import write_synthetic
from npcc.experiments.application.derived import (
  DEW_POINT,
  MIXING_RATIO,
  PRESSURE,
  RELATIVE_HUMIDITY,
  TEMPERATURE,
)
from npcc.experiments.application.joint_scores import (
  energy_score,
  variogram_score,
)
from npcc.experiments.application.marginal_set import MarginalSet
from npcc.experiments.application.scores import crps_from_quantiles
from npcc.experiments.application.store import (
  draws_frame,
  read_marginal_set,
  read_station_data,
  write_copula,
)
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
  margins: str = "",
  copulas: str = "",
  name: str = "consolidated.toml",
) -> Path:
  text = config_text(data, arms=arms, stations=stations)
  text = text.replace("[responses]\n", "[responses]\n" + _RESPONSES)
  text = text.replace("n_levels = 19\n", "n_levels = 19\n" + margins)
  text = text.replace(
    "sample_batch_size = 50\n",
    "sample_batch_size = 50\nsave_draws = true\n" + copulas,
  )
  text = text.replace("[[estimators]]", designs + "[[estimators]]", 1)
  path = tmp_path / name
  path.write_text(text)
  return path


def _paired_summary() -> ModuleType:
  path = _ROOT / "scripts" / "paper" / "application" / "paired_summary.py"
  spec = importlib.util.spec_from_file_location("paired_summary", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("register_linear_gaussian_backend")
def test_a_design_enters_the_signature(tmp_path: Path) -> None:
  """A run folder of another design must refuse to resume."""
  plain = _config(
    tmp_path, Path("data"), designs="", arms=("gaussian",), name="plain.toml"
  )
  designed = _config(tmp_path, Path("data"), arms=("gaussian",))
  assert load_config(plain).signature() != load_config(designed).signature()


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
  # The 5d log score is stored, never summarized: the five responses have
  # no joint density.
  assert "joint_nll_5d" not in set(paired["score"])
  assert "energy" in set(same["score"])
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


def test_subsets_and_projected_log_scores_compare_the_designs(
  tmp_path: Path, register_linear_gaussian_backend: None
) -> None:
  """The 5d arms must be comparable with the triple's on the triple's responses.

  A subset's energy and variogram scores are the full scores' estimators on
  its responses alone, divided by the same scales, so recomputing them from
  the stored draws gives them back, on either design. The projected log
  score of ``independence@5d`` and ``gaussian@5d`` is the exact density of
  the triple's modeled responses under the 5d model, in the triple's
  column: independence's equals the triple's independence, on the same
  margins, and the Gaussian's is the Gaussian copula of its correlation's
  submatrix.
  """
  del register_linear_gaussian_backend
  data = tmp_path / "data"
  write_synthetic(data, n_stations=2, step_days=7, seed=3, measured=True)
  copulas = (
    "subsets = { trhp = " + json.dumps(list(_TRIPLE)) + " }\n"
    "projected_log_score = true\n"
  )
  config = _config(tmp_path, data, copulas=copulas)
  run = tmp_path / "run"
  for stage in ("margins", "copulas", "evaluate"):
    assert main([stage, "--config", str(config), "--out", str(run)]) == 0
  model = "emos-seasonal"
  unit = run / "stations" / "000" / "copulas" / model
  scores = {
    arm: pd.read_parquet(unit / arm / "scores.parquet").set_index("row")
    for arm in _ARMS
  }
  station = read_station_data(run, 0)
  # The stored draws are sorted by row, the station's cases by time.
  order = np.argsort(station.row_test)
  observed = np.stack(
    [station.y_test[order, station.column(v)] for v in _TRIPLE], axis=1
  )
  for arm in ("gaussian", "gaussian@5d"):
    info = json.loads((unit / arm / "info.json").read_text())
    scales = np.array([info["scales"][v] for v in _TRIPLE])
    stored = pd.read_parquet(unit / arm / "draws.parquet").sort_values(
      ["row", "draw"]
    )
    rows = stored["row"].unique()
    np.testing.assert_array_equal(rows, station.row_test[order])
    draws = stored[list(_TRIPLE)].to_numpy(np.float64)
    draws = draws.reshape(len(rows), -1, len(_TRIPLE))
    # The draws are stored as float32, the scores were computed before.
    np.testing.assert_allclose(
      scores[arm].loc[rows, "energy_trhp"],
      energy_score(draws / scales, observed / scales),
      rtol=1e-5,
    )
    np.testing.assert_allclose(
      scores[arm].loc[rows, "variogram_trhp"],
      variogram_score(draws / scales, observed / scales, p=0.5),
      rtol=1e-4,
    )

  np.testing.assert_allclose(
    scores["independence@5d"]["joint_nll_modeled"],
    scores["independence"]["joint_nll_modeled"],
    rtol=1e-12,
  )
  ms = read_marginal_set(run, 0, model)
  keep = [ms.variables.index(v) for v in _TRIPLE]
  info = json.loads((unit / "gaussian@5d" / "info.json").read_text())
  correlation = np.asarray(
    info["results"]["gaussian@5d"]["info"]["correlation"]
  )
  sub = correlation[np.ix_(keep, keep)]
  z = norm.ppf(np.clip(ms.pit_test[:, keep], PIT_CLIP, 1 - PIT_CLIP))
  logcop = multivariate_normal(cov=sub).logpdf(z) - norm.logpdf(z).sum(axis=1)
  expected = -(logcop + ms.logpdf_test[:, keep].sum(axis=1))
  np.testing.assert_allclose(
    scores["gaussian@5d"].loc[station.row_test, "joint_nll_modeled"],
    expected,
    rtol=1e-9,
  )
  # The tables carry both, each 5d arm's gain against independence@5d.
  by_station = pd.read_csv(run / "tables" / "copulas_by_station.csv")
  row = by_station.set_index(["station", "model", "arm"])
  assert row.loc[(0, model, "gaussian@5d"), "joint_nll_modeled_gain"] == (
    pytest.approx(
      row.loc[(0, model, "independence@5d"), "joint_nll_modeled"]
      - row.loc[(0, model, "gaussian@5d"), "joint_nll_modeled"]
    )
  )
  assert "energy_trhp_skill" in row.columns

  # (iii) compares the 5d arms with the triple's on the subset too, and on
  # the log score where it is an exact projection, independence and
  # Gaussian; never on the 5d log score.
  _paired_summary().build(run, run / "analysis")
  paired = pd.read_csv(run / "analysis" / "paired.csv")
  cross = paired[
    paired["reference"].str.endswith("(main design, same margins)")
    & (paired["stations"] == "all")
    & (paired["model"] == model)
  ]
  projected = cross[cross["score"] == "joint_nll_modeled"]
  assert set(projected["arm"]) == {"independence@5d", "gaussian@5d"}
  level = projected.loc[projected["arm"] == "independence@5d", "mean_diff"]
  assert level.iloc[0] == pytest.approx(0.0, abs=1e-12)
  assert {"energy_trhp", "variogram_trhp"} <= set(cross["score"])
  assert "joint_nll_5d" not in set(paired["score"])


@pytest.mark.usefixtures("register_linear_gaussian_backend")
@pytest.mark.parametrize(
  ("copulas", "message"),
  [
    (
      'subsets = { trhp_low = ["obs:air_temperature", '
      '"obs:surface_air_pressure"] }\n',
      "not ending in",
    ),
    ('subsets = { t = ["obs:air_temperature"] }\n', "at least two"),
    (
      'subsets = { tx = ["obs:air_temperature", "obs:wind"] }\n',
      "at least two",
    ),
  ],
)
def test_a_subset_no_score_could_name_is_refused(
  tmp_path: Path, copulas: str, message: str
) -> None:
  """A subset must name its own score columns and responses every draw has.

  A name ending like a table's derived column would be read as a skill or
  an interval of another score, and a response the draws lack has no score.
  """
  with pytest.raises(ValueError, match=message):
    load_config(_config(tmp_path, Path("data"), copulas=copulas))


@pytest.mark.usefixtures("register_linear_gaussian_backend")
def test_a_projected_log_score_needs_another_design(tmp_path: Path) -> None:
  """Without another design there is nothing to project, so it is refused."""
  with pytest.raises(ValueError, match="another design"):
    load_config(
      _config(
        tmp_path,
        Path("data"),
        designs="",
        arms=("gaussian",),
        copulas="projected_log_score = true\n",
      )
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
