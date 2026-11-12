"""The scores and their tables: CRPS from quantiles, coverage, skills."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from npcc.experiments.application.emos import gaussian_crps
from npcc.experiments.application.scores import (
  NOMINAL_COVERAGE,
  copula_tables,
  coverage_column,
  crps_from_quantiles,
  interpolate_quantiles,
  marginal_case_scores,
  marginal_tables,
  skill,
  station_interval,
  summarize_timings,
)

_LEVELS = (np.arange(199) + 0.5) / 199


@pytest.mark.parametrize(
  ("y", "rtol"), [(0.2, 1e-4), (-1.0, 1e-4), (2.0, 1e-4), (5.0, 1e-3)]
)
def test_crps_from_199_midpoint_quantiles_is_the_gaussian_crps(
  y: float, rtol: float
) -> None:
  """The CRPS every model is ranked by must be the CRPS.

  Twice the mean pinball loss over evenly spread levels is a quadrature of
  the CRPS's quantile form; with his 199 midpoints it matches the closed
  form to a few parts in ten thousand, a little less in the far tail.
  """
  mu, sigma = 0.2, 1.3
  quantiles = stats.norm.ppf(_LEVELS, loc=mu, scale=sigma)[None, :]

  got = crps_from_quantiles(np.array([y]), quantiles, _LEVELS)
  expected = gaussian_crps(np.array([y]), mu, sigma)
  np.testing.assert_allclose(got, expected, rtol=rtol)


def test_crps_from_quantiles_is_his_formula() -> None:
  """His report computes ``2 mean(max(a e, (a - 1) e))``; this must too."""
  rng = np.random.default_rng(1)
  quantiles = np.sort(rng.normal(size=(6, 199)), axis=1)
  y = rng.normal(size=6)
  error = y[:, None] - quantiles
  his = 2 * np.mean(np.maximum(_LEVELS * error, (_LEVELS - 1) * error), axis=1)

  np.testing.assert_allclose(crps_from_quantiles(y, quantiles, _LEVELS), his)


@pytest.mark.parametrize("p", [0.0, 0.001, 0.025, _LEVELS[3], 0.5, 0.975, 1.0])
def test_interpolated_quantiles_are_nps_interp(p: float) -> None:
  """Interval bounds and copula draws read quantiles the way he does."""
  rng = np.random.default_rng(2)
  quantiles = np.sort(rng.normal(size=(5, 199)), axis=1)

  np.testing.assert_allclose(
    interpolate_quantiles(quantiles, _LEVELS, p),
    [np.interp(p, _LEVELS, q) for q in quantiles],
    rtol=0,
    atol=1e-15,
  )


def test_coverage_counts_an_observation_inside_each_central_interval() -> None:
  """A coverage read off the wrong tail would report miscalibration."""
  quantiles = stats.norm.ppf(_LEVELS)[None, :].repeat(4, axis=0)
  y = np.array([0.0, 1.0, 1.5, 3.0])
  scores = marginal_case_scores(
    y,
    logpdf=stats.norm.logpdf(y),
    pit=stats.norm.cdf(y),
    quantiles=quantiles,
    levels=_LEVELS,
  )

  np.testing.assert_array_equal(scores[coverage_column(0.5)], [1, 0, 0, 0])
  np.testing.assert_array_equal(scores[coverage_column(0.8)], [1, 1, 0, 0])
  np.testing.assert_array_equal(scores[coverage_column(0.9)], [1, 1, 1, 0])
  np.testing.assert_array_equal(scores[coverage_column(0.95)], [1, 1, 1, 0])
  np.testing.assert_allclose(scores["nll"], -stats.norm.logpdf(y))
  assert set(scores) == {
    "crps",
    "nll",
    "pit",
    *(coverage_column(n) for n in NOMINAL_COVERAGE),
  }


def test_skill_is_a_ratio_of_means() -> None:
  """A mean of per-case ratios would let a near-zero reference dominate."""
  score = np.array([1.0, 3.0])
  reference = np.array([2.0, 2.0])

  assert skill(score, reference) == pytest.approx(0.0)
  assert np.isnan(skill(score, np.zeros(2)))


def _marginal_cases() -> pd.DataFrame:
  rng = np.random.default_rng(3)
  rows = []
  for station in (0, 1, 2):
    for model, scale in (("emos-seasonal", 1.0), ("m", 0.8)):
      for variable in ("a", "b"):
        for row in range(10):
          rows.append(
            {
              "station": station,
              "model": model,
              "variable": variable,
              "row": 100 * station + row,
              "crps": scale * (1.0 + station) * rng.uniform(0.5, 1.5),
              "nll": scale * rng.uniform(1.0, 2.0),
              "pit": rng.uniform(),
              **{
                coverage_column(n): float(rng.uniform() < n)
                for n in NOMINAL_COVERAGE
              },
            }
          )
  return pd.DataFrame(rows)


def test_pooled_crpss_is_one_minus_a_ratio_of_pooled_means() -> None:
  """His pooled CRPSS: ``1 - mean(CRPS) / mean(EMOS CRPS)`` over all cases.

  Averaging per-station skills instead would weight a station by its count
  of stations, not of cases, and is what the equal-station table is for.
  """
  cases = _marginal_cases()
  tables = marginal_tables(cases)
  pooled = tables["pooled"].set_index(["model", "variable"])
  by_station = tables["by_station"].set_index(["station", "model", "variable"])

  for variable in ("a", "b"):
    ours = cases.loc[(cases["model"] == "m") & (cases["variable"] == variable)]
    ref = cases.loc[
      (cases["model"] == "emos-seasonal") & (cases["variable"] == variable)
    ]
    assert pooled.loc[("m", variable), "crpss"] == pytest.approx(
      1 - ours["crps"].mean() / ref["crps"].mean()
    )
    assert pooled.loc[("m", variable), "log_score_gain"] == pytest.approx(
      ref["nll"].mean() - ours["nll"].mean()
    )
    assert pooled.loc[("emos-seasonal", variable), "crpss"] == 0.0
    s1 = ours.loc[ours["station"] == 1]
    r1 = ref.loc[ref["station"] == 1]
    assert by_station.loc[(1, "m", variable), "crpss"] == pytest.approx(
      1 - s1["crps"].mean() / r1["crps"].mean()
    )
  mean = tables["station_mean"].set_index(["model", "variable"])
  per_station = tables["by_station"]
  per_station = per_station.loc[
    (per_station["model"] == "m") & (per_station["variable"] == "a"), "crpss"
  ]
  assert mean.loc[("m", "a"), "crpss"] == pytest.approx(per_station.mean())
  assert mean.loc[("m", "a"), "crpss_low"] <= mean.loc[("m", "a"), "crpss"]
  assert mean.loc[("m", "a"), "n_stations"] == 3


def test_the_case_gains_average_to_the_pooled_crpss() -> None:
  """His boxplots show case gains whose mean is the pooled CRPSS."""
  tables = marginal_tables(_marginal_cases())
  cases = tables["cases"]
  pooled = tables["pooled"].set_index(["model", "variable"])

  gains = cases.loc[(cases["model"] == "m") & (cases["variable"] == "b")]
  assert gains["crps_gain"].mean() == pytest.approx(
    pooled.loc[("m", "b"), "crpss"]
  )


def test_models_scored_on_different_cases_are_refused() -> None:
  """A skill over unpaired cases compares two different test sets."""
  cases = _marginal_cases()
  cases = cases.drop(index=cases.index[(cases["model"] == "m")][0])

  with pytest.raises(ValueError, match="same cases"):
    marginal_tables(cases)


def test_without_emos_the_skills_are_undefined_not_zero() -> None:
  """A skill against a missing baseline must not read as no improvement."""
  cases = _marginal_cases()
  tables = marginal_tables(cases.loc[cases["model"] == "m"])

  assert tables["pooled"]["crpss"].isna().all()
  assert tables["by_station"]["log_score_gain"].isna().all()


def _copula_cases() -> pd.DataFrame:
  rng = np.random.default_rng(4)
  rows = []
  for station in (0, 1):
    for model in ("m1", "m2"):
      for arm, shift in (("independence", 0.0), ("gaussian", -0.2)):
        for row in range(8):
          rows.append(
            {
              "station": station,
              "model": model,
              "arm": arm,
              "row": row,
              "joint_nll": 5.0 + shift + rng.normal(0, 0.1),
              "energy": 2.0 + shift + rng.normal(0, 0.1),
              "variogram": 1.0 + shift + rng.normal(0, 0.1),
            }
          )
  return pd.DataFrame(rows)


def test_copula_skills_are_against_independence_on_the_same_margins() -> None:
  """An arm compared with another margin's independence mixes two effects."""
  cases = _copula_cases()
  tables = copula_tables(cases)
  pooled = tables["pooled"].set_index(["model", "arm"])

  for model in ("m1", "m2"):
    arm = cases.loc[(cases["model"] == model) & (cases["arm"] == "gaussian")]
    ref = cases.loc[
      (cases["model"] == model) & (cases["arm"] == "independence")
    ]
    row = pooled.loc[(model, "gaussian")]
    assert row["joint_nll_gain"] == pytest.approx(
      ref["joint_nll"].mean() - arm["joint_nll"].mean()
    )
    assert row["energy_skill"] == pytest.approx(
      1 - arm["energy"].mean() / ref["energy"].mean()
    )
    assert row["variogram_skill"] == pytest.approx(
      1 - arm["variogram"].mean() / ref["variogram"].mean()
    )
    assert pooled.loc[(model, "independence"), "energy_skill"] == 0.0
  case = tables["cases"]
  gains = case.loc[(case["model"] == "m1") & (case["arm"] == "gaussian")]
  assert gains["energy_gain"].mean() == pytest.approx(
    pooled.loc[("m1", "gaussian"), "energy_skill"]
  )


def test_copula_arms_on_different_cases_are_refused() -> None:
  """Arms compared within a margin must cover the same test cases."""
  cases = _copula_cases()
  cases = cases.loc[~((cases["arm"] == "gaussian") & (cases["row"] == 0))]

  with pytest.raises(ValueError, match="same cases"):
    copula_tables(cases)


def test_the_station_interval_is_a_percentile_bootstrap() -> None:
  """One station has a mean but no interval; a nan poisons nothing silently."""
  mean, low, high = station_interval(np.array([1.0, 2.0, 3.0, 4.0]), seed=0)
  assert mean == pytest.approx(2.5)
  assert low < mean < high
  assert station_interval(np.array([1.0]))[0] == 1.0
  assert np.isnan(station_interval(np.array([1.0]))[1])
  assert all(np.isnan(v) for v in station_interval(np.array([1.0, np.nan])))


def test_timings_sum_within_a_station_then_summarize_over_them() -> None:
  """Each station counts once, whatever its number of responses."""
  timings = pd.DataFrame(
    {
      "station": [0, 0, 1, 1],
      "model": ["m"] * 4,
      "variable": ["a", "b", "a", "b"],
      "stage": ["fit"] * 4,
      "seconds": [1.0, 2.0, 3.0, 5.0],
    }
  )
  summary = summarize_timings(timings, ["model"]).iloc[0]

  assert summary["n_stations"] == 2
  assert summary["total_seconds"] == pytest.approx(11.0)
  assert summary["mean_seconds"] == pytest.approx(5.5)
  assert summarize_timings(timings.iloc[:0], ["model"]).empty
