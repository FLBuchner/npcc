"""Scores, and the tables that summarize them per station and pooled.

Per test case, a margin is scored by the CRPS of its quantiles at the levels
(the mean pinball loss, doubled), the negative log-density (``nll``), its
PIT, and whether each central interval covers the observation. A copula arm
arrives already scored, by joint NLL and the energy and variogram scores.

Skills compare a model with a reference on identical cases: a margin with
seasonal EMOS, a copula arm with independence on the same margins. A score
skill is ``1 - mean(score) / mean(reference)``, a ratio of means rather than
a mean of per-case ratios; a log-score gain is the additive improvement
``mean(reference nll) - mean(nll)``. Pooled, the means run over every case
of every station; per station, over its cases; and the equal-station tables
average the per-station values, with a percentile bootstrap over stations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from npcc.experiments.application.config import EMOS_MODEL

#: The central intervals whose coverage is scored.
NOMINAL_COVERAGE: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)
#: The copula arm skills are relative to.
INDEPENDENCE = "independence"
#: Every per-case copula score column.
COPULA_SCORES: tuple[str, ...] = ("joint_nll", "energy", "variogram")


def coverage_column(nominal: float) -> str:
  """The column name of a central interval's coverage, e.g. ``coverage_90``.

  Parameters
  ----------
  nominal : float
      The interval's nominal coverage.

  Returns
  -------
  str
      ``coverage_<percent>``.
  """
  return f"coverage_{round(100 * nominal)}"


def sample_crps_column(variable: str) -> str:
  """The column of a response's CRPS from copula draws, e.g. ``crps_air_temperature``.

  Parameters
  ----------
  variable : str
      The response, as named in ``y.nc``.

  Returns
  -------
  str
      ``crps_`` and the name without its ``obs:`` prefix.
  """
  return "crps_" + variable.removeprefix("obs:")


def crps_from_quantiles(
  y: np.ndarray, quantiles: np.ndarray, levels: np.ndarray
) -> np.ndarray:
  """CRPS approximated from quantiles: twice the mean pinball loss.

  Parameters
  ----------
  y : numpy.ndarray, shape (n,)
      Observations.
  quantiles : numpy.ndarray, shape (n, L)
      Each case's predictive quantiles at ``levels``.
  levels : numpy.ndarray, shape (L,)
      Probability levels in ``(0, 1)``, evenly spread.

  Returns
  -------
  numpy.ndarray, shape (n,)
      ``2 / L * sum_k max(a_k e_k, (a_k - 1) e_k)`` with ``e_k = y - q_k``.
  """
  levels = np.asarray(levels, dtype=np.float64)
  error = np.asarray(y, dtype=np.float64)[:, None] - quantiles
  return 2.0 * np.mean(np.maximum(levels * error, (levels - 1.0) * error), 1)


def interpolate_quantiles(
  quantiles: np.ndarray, levels: np.ndarray, p: float
) -> np.ndarray:
  """Each case's quantile at ``p``, linear between the levels.

  Parameters
  ----------
  quantiles : numpy.ndarray, shape (n, L)
      Nondecreasing along the level axis.
  levels : numpy.ndarray, shape (L,)
      Increasing probability levels.
  p : float
      A probability; outside the levels, the end quantile.

  Returns
  -------
  numpy.ndarray, shape (n,)
      ``np.interp(p, levels, q)`` for each row ``q``.
  """
  levels = np.asarray(levels, dtype=np.float64)
  k = int(np.clip(np.searchsorted(levels, p), 1, len(levels) - 1))
  lo, hi = levels[k - 1], levels[k]
  w = float(np.clip((p - lo) / (hi - lo), 0.0, 1.0))
  return (1.0 - w) * quantiles[:, k - 1] + w * quantiles[:, k]


def marginal_case_scores(
  y: np.ndarray,
  *,
  logpdf: np.ndarray,
  pit: np.ndarray,
  quantiles: np.ndarray,
  levels: np.ndarray,
) -> dict[str, np.ndarray]:
  """Every per-case score of one margin's test predictions.

  Parameters
  ----------
  y : numpy.ndarray, shape (n,)
      Observations, on the original scale.
  logpdf : numpy.ndarray, shape (n,)
      Log-density at the observations, on the original scale.
  pit : numpy.ndarray, shape (n,)
      The CDF at the observations.
  quantiles : numpy.ndarray, shape (n, L)
      Quantiles at ``levels``, on the original scale.
  levels : numpy.ndarray, shape (L,)
      The probability levels.

  Returns
  -------
  dict of str to numpy.ndarray
      ``crps``, ``nll``, ``pit`` and one ``coverage_<percent>`` per
      :data:`NOMINAL_COVERAGE`, as 0 or 1.
  """
  y = np.asarray(y, dtype=np.float64)
  scores = {
    "crps": crps_from_quantiles(y, quantiles, levels),
    "nll": -np.asarray(logpdf, dtype=np.float64),
    "pit": np.asarray(pit, dtype=np.float64),
  }
  for nominal in NOMINAL_COVERAGE:
    tail = 0.5 * (1.0 - nominal)
    lower = interpolate_quantiles(quantiles, levels, tail)
    upper = interpolate_quantiles(quantiles, levels, 1.0 - tail)
    scores[coverage_column(nominal)] = ((y >= lower) & (y <= upper)).astype(
      np.float64
    )
  return scores


def skill(score: np.ndarray, reference: np.ndarray) -> float:
  """``1 - mean(score) / mean(reference)``; ``nan`` for a non-positive one.

  Parameters
  ----------
  score : numpy.ndarray
      A model's scores, negatively oriented.
  reference : numpy.ndarray
      The reference's scores on the same cases.

  Returns
  -------
  float
      The skill: positive where the model scores lower on average.
  """
  ref = float(np.mean(reference))
  return 1.0 - float(np.mean(score)) / ref if ref > 0.0 else float("nan")


def station_interval(
  values: np.ndarray, *, seed: int = 42, n_boot: int = 1000
) -> tuple[float, float, float]:
  """The equal-station mean and a 95% percentile bootstrap interval.

  Stations are resampled with replacement; the interval ignores any
  dependence between them, so it is exploratory.

  Parameters
  ----------
  values : numpy.ndarray, shape (n_stations,)
      One value per station.
  seed : int, default=42
      Seed of the resampling.
  n_boot : int, default=1000
      Resamples.

  Returns
  -------
  mean, low, high : float
      ``nan`` throughout for a non-finite value, and for the interval with
      fewer than two stations.
  """
  values = np.asarray(values, dtype=np.float64)
  if values.ndim != 1 or n_boot < 1:
    raise ValueError(
      f"values must be one-dimensional and n_boot positive; got shape "
      f"{values.shape} and {n_boot}"
    )
  nan = float("nan")
  if not len(values) or not np.isfinite(values).all():
    return nan, nan, nan
  mean = float(values.mean())
  if len(values) < 2:
    return mean, nan, nan
  rng = np.random.default_rng(seed)
  means = rng.choice(values, size=(n_boot, len(values))).mean(axis=1)
  low, high = np.quantile(means, [0.025, 0.975])
  return mean, float(low), float(high)


def _check_paired(
  cases: pd.DataFrame, by: str, keys: list[str], within: list[str]
) -> None:
  """Refuse ``by`` values that do not cover the same ``keys``."""
  groups = cases.groupby(within, sort=False) if within else [((), cases)]
  for group, frame in groups:
    reference: set[tuple[object, ...]] | None = None
    for name, sub in frame.groupby(by, sort=False):
      covered = set(sub[keys].itertuples(index=False, name=None))
      if len(covered) != len(sub):
        raise ValueError(
          f"cases must hold one row per {keys} for each {by}; {name} repeats "
          f"some"
        )
      if reference is None:
        reference = covered
      elif covered != reference:
        raise ValueError(
          f"every {by} must be scored on the same cases {keys} to be "
          f"compared{f' within {group}' if within else ''}; {name} differs"
        )


def _equal_station(
  by_station: pd.DataFrame,
  group: list[str],
  metrics: Sequence[str],
  seed: int,
) -> pd.DataFrame:
  """Per group, each metric's mean over stations and its bootstrap interval."""
  rows = []
  for key, frame in by_station.groupby(group, sort=False):
    ordered = frame.sort_values("station")
    values = cast("tuple[object, ...]", key)
    row: dict[str, object] = dict(zip(group, values, strict=True))
    row["n_stations"] = len(ordered)
    for metric in metrics:
      mean, low, high = station_interval(
        ordered[metric].to_numpy(dtype=np.float64), seed=seed
      )
      row[metric] = mean
      row[f"{metric}_low"] = low
      row[f"{metric}_high"] = high
    rows.append(row)
  return pd.DataFrame(rows)


def marginal_tables(
  cases: pd.DataFrame,
  *,
  baseline: str = EMOS_MODEL,
  seed: int = 42,
) -> dict[str, pd.DataFrame]:
  """The marginal score tables, per station, pooled and over stations.

  Parameters
  ----------
  cases : pandas.DataFrame
      One row per ``(station, model, variable, row)``, with the columns of
      :func:`marginal_case_scores`. Every model must cover the same cases.
  baseline : str, default="emos-seasonal"
      The reference model; skills are ``nan`` without it.
  seed : int, default=42
      Seed of the station bootstrap.

  Returns
  -------
  dict of str to pandas.DataFrame
      ``by_station``: per station, model and variable, the mean of each
      score and coverage, ``crpss`` and ``log_score_gain``. ``pooled``: the
      same over every station's cases. ``station_mean``: the per-station
      values averaged over stations, with bootstrap intervals. ``cases``:
      the input with each case's ``crps_gain``, its CRPS improvement on the
      baseline over the baseline's pooled mean CRPS (whose mean is the pooled
      CRPSS), and ``log_score_gain``.
  """
  _check_paired(cases, "model", ["station", "variable", "row"], [])
  coverage = [coverage_column(n) for n in NOMINAL_COVERAGE]
  means = ["crps", "nll", *coverage]
  has_baseline = baseline in set(cases["model"])

  by_station = (
    cases.groupby(["station", "model", "variable"], sort=True)
    .agg(n_cases=("row", "size"), **{m: (m, "mean") for m in means})
    .reset_index()
  )
  pooled = (
    cases.groupby(["model", "variable"], sort=True)
    .agg(
      n_cases=("row", "size"),
      n_stations=("station", "nunique"),
      **{m: (m, "mean") for m in means},
    )
    .reset_index()
  )
  wide = cases.copy()
  if has_baseline:
    ref = cases.loc[
      cases["model"] == baseline, ["station", "variable", "row", "crps", "nll"]
    ].rename(columns={"crps": "ref_crps", "nll": "ref_nll"})
    wide = wide.merge(ref, on=["station", "variable", "row"], validate="m:1")
    pooled_ref = ref.groupby("variable")["ref_crps"].mean()
    denominator = wide["variable"].map(pooled_ref)
    wide["crps_gain"] = (wide["ref_crps"] - wide["crps"]) / denominator.where(
      denominator > 0
    )
    wide["log_score_gain"] = wide["ref_nll"] - wide["nll"]
    wide = wide.drop(columns=["ref_crps", "ref_nll"])
    for table, keys in (
      (by_station, ["station", "variable"]),
      (pooled, ["variable"]),
    ):
      reference = table.loc[table["model"] == baseline, [*keys, "crps", "nll"]]
      merged = table.merge(
        reference.rename(columns={"crps": "ref_crps", "nll": "ref_nll"}),
        on=keys,
        how="left",
        validate="m:1",
      )
      table["crpss"] = (
        1.0 - merged["crps"] / merged["ref_crps"].where(merged["ref_crps"] > 0)
      ).to_numpy()
      table["log_score_gain"] = (merged["ref_nll"] - merged["nll"]).to_numpy()
  else:
    wide["crps_gain"] = np.nan
    wide["log_score_gain"] = np.nan
    for table in (by_station, pooled):
      table["crpss"] = np.nan
      table["log_score_gain"] = np.nan
  station_mean = _equal_station(
    by_station,
    ["model", "variable"],
    ["crps", "crpss", "nll", "log_score_gain", *coverage],
    seed,
  )
  return {
    "by_station": by_station,
    "pooled": pooled,
    "station_mean": station_mean,
    "cases": wide,
  }


def copula_score_columns(cases: pd.DataFrame) -> tuple[str, list[str]]:
  """The copula scores present: the log score's column, then every score.

  Parameters
  ----------
  cases : pandas.DataFrame
      Per-case copula scores.

  Returns
  -------
  nll : str
      ``joint_nll``, or ``joint_nll_modeled`` where responses are derived and
      the log score covers the modeled ones only.
  scores : list of str
      ``nll``, ``energy``, ``variogram``, then each response's CRPS from the
      draws (``crps_<name>``) where present.
  """
  nll = "joint_nll_modeled" if "joint_nll_modeled" in cases else "joint_nll"
  crps = [
    c
    for c in cases.columns
    if c.startswith("crps_")
    and not c.endswith(("_skill", "_gain", "_low", "_high"))
  ]
  return nll, [nll, "energy", "variogram", *crps]


def copula_tables(
  cases: pd.DataFrame,
  *,
  reference: str = INDEPENDENCE,
  seed: int = 42,
) -> dict[str, pd.DataFrame]:
  """The copula score tables, each arm against independence on its margins.

  Parameters
  ----------
  cases : pandas.DataFrame
      One row per ``(station, model, arm, row)``, ``model`` being the
      marginal model, with the columns of :func:`copula_score_columns`.
      Within a marginal model, every arm must cover the same cases.
  reference : str, default="independence"
      The reference arm; skills are ``nan`` without it.
  seed : int, default=42
      Seed of the station bootstrap.

  Returns
  -------
  dict of str to pandas.DataFrame
      ``by_station``, ``pooled`` and ``station_mean`` as for
      :func:`marginal_tables`: each score's mean, the log score's gain
      (``joint_nll_gain``, the additive improvement on the reference) and
      every other score's skill (``energy_skill``, ``variogram_skill``, a
      ratio of means); and
      ``cases``, the input with each case's log-score gain and its
      ``energy_gain`` and ``variogram_gain`` over the reference's pooled
      mean.
  """
  _check_paired(cases, "arm", ["station", "row"], ["model"])
  nll, scores = copula_score_columns(cases)
  skilled = scores[1:]
  means = scores
  by_station = (
    cases.groupby(["station", "model", "arm"], sort=True)
    .agg(n_cases=("row", "size"), **{m: (m, "mean") for m in means})
    .reset_index()
  )
  pooled = (
    cases.groupby(["model", "arm"], sort=True)
    .agg(
      n_cases=("row", "size"),
      n_stations=("station", "nunique"),
      **{m: (m, "mean") for m in means},
    )
    .reset_index()
  )
  ref_cases = cases.loc[
    cases["arm"] == reference, ["station", "model", "row", *scores]
  ].rename(columns={m: f"ref_{m}" for m in scores})
  wide = cases.merge(
    ref_cases, on=["station", "model", "row"], how="left", validate="m:1"
  )
  wide[f"{nll}_gain"] = wide[f"ref_{nll}"] - wide[nll]
  for metric in ("energy", "variogram"):
    pooled_ref = ref_cases.groupby("model")[f"ref_{metric}"].mean()
    denominator = wide["model"].map(pooled_ref)
    wide[f"{metric}_gain"] = (wide[f"ref_{metric}"] - wide[metric]) / (
      denominator.where(denominator > 0)
    )
  wide = wide.drop(columns=[f"ref_{m}" for m in scores])
  for table, keys in (
    (by_station, ["station", "model"]),
    (pooled, ["model"]),
  ):
    ref = table.loc[table["arm"] == reference, [*keys, *scores]]
    merged = table.merge(
      ref.rename(columns={m: f"ref_{m}" for m in scores}),
      on=keys,
      how="left",
      validate="m:1",
    )
    table[f"{nll}_gain"] = (merged[f"ref_{nll}"] - merged[nll]).to_numpy()
    for metric in skilled:
      ref_mean = merged[f"ref_{metric}"]
      table[f"{metric}_skill"] = (
        1.0 - merged[metric] / ref_mean.where(ref_mean > 0)
      ).to_numpy()
  station_mean = _equal_station(
    by_station,
    ["model", "arm"],
    [*means, f"{nll}_gain", *(f"{m}_skill" for m in skilled)],
    seed,
  )
  return {
    "by_station": by_station,
    "pooled": pooled,
    "station_mean": station_mean,
    "cases": wide,
  }


def summarize_timings(timings: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
  """Seconds per stage, summed within a station, then summarized over them.

  Parameters
  ----------
  timings : pandas.DataFrame
      One row per measurement, with ``station``, the ``by`` columns,
      ``stage`` and ``seconds``.
  by : sequence of str
      The columns that name what was timed, such as ``["model"]``.

  Returns
  -------
  pandas.DataFrame
      Per ``(*by, stage)``: ``n_stations`` and the total, mean, median and
      quartiles of the per-station seconds, each station weighted equally.
  """
  columns = [*by, "stage"]
  if timings.empty:
    return pd.DataFrame(
      columns=[
        *columns,
        "n_stations",
        "total_seconds",
        "mean_seconds",
        "median_seconds",
        "q25_seconds",
        "q75_seconds",
      ]
    )
  per_station = timings.groupby(["station", *columns], as_index=False)[
    "seconds"
  ].sum()
  summary = (
    per_station.groupby(columns, sort=True)["seconds"]
    .agg(
      n_stations="count",
      total_seconds="sum",
      mean_seconds="mean",
      median_seconds="median",
      q25_seconds=lambda s: s.quantile(0.25),
      q75_seconds=lambda s: s.quantile(0.75),
    )
    .reset_index()
  )
  return cast("pd.DataFrame", summary)


#: The PIT columns :func:`pit_uniformity` reads, by the PITs they hold.
PIT_KINDS: dict[str, tuple[str, str]] = {
  "train_in_sample": ("train", "pit"),
  "train_out_of_fold": ("train", "pit_out_of_fold"),
  "test": ("test", "pit"),
}


def _uniformity(pit: np.ndarray) -> dict[str, float]:
  """How far a sample of PITs is from uniform."""
  u = np.sort(pit)
  n = len(u)
  ranks = np.arange(1, n + 1) / n
  return {
    "n": float(n),
    "sd": float(u.std()),
    "central_50": float(np.mean((u > 0.25) & (u < 0.75))),
    "outer_10": float(np.mean((u < 0.05) | (u > 0.95))),
    "ks": float(np.max(np.maximum(ranks - u, u - (ranks - 1.0 / n)))),
  }


def pit_uniformity(predictions: pd.DataFrame) -> pd.DataFrame:
  """How uniform each marginal model's PITs are, per response and kind.

  Uniform PITs have standard deviation ``1 / sqrt(12) = 0.2887``, half of
  them in (0.25, 0.75) and a tenth below 0.05 or above 0.95. A margin
  evaluated at its own training cases scores them closer than new ones, so
  its in-sample PITs crowd the middle; the out-of-fold ones should not.

  Parameters
  ----------
  predictions : pandas.DataFrame
      The margins' stored predictions: ``station``, ``model``, ``variable``,
      ``split``, ``pit``, and ``pit_out_of_fold`` where it was computed.

  Returns
  -------
  pandas.DataFrame
      One row per ``(model, variable, kind)``, ``kind`` a key of
      :data:`PIT_KINDS` the input holds: over every station's cases pooled,
      ``n``, ``sd``, ``central_50``, ``outer_10`` and ``ks``, the
      Kolmogorov-Smirnov distance to uniform; and ``median_station_sd``,
      the median over stations of each station's standard deviation.
  """
  rows = []
  for kind, (split, column) in PIT_KINDS.items():
    if column not in predictions:
      continue
    part = predictions.loc[predictions["split"] == split]
    for (model, variable), group in part.groupby(
      ["model", "variable"], sort=True
    ):
      pit = group[column].to_numpy(dtype=np.float64)
      if np.isnan(pit).all():
        continue
      by_station = group.groupby("station")[column].std(ddof=0)
      rows.append(
        {
          "model": model,
          "variable": variable,
          "kind": kind,
          **_uniformity(pit),
          "median_station_sd": float(by_station.median()),
        }
      )
  return pd.DataFrame(rows)
