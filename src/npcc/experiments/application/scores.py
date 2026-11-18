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

from npcc.experiments.application.config import (
  ARM_DESIGN,
  EMOS_MODEL,
  split_arm,
)

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


def date_block_counts(
  n_dates: int, *, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
  """How often each date is drawn, per replicate of a circular block bootstrap.

  The dates are taken in calendar order and closed into a circle. Each
  replicate draws ``ceil(n_dates / b)`` blocks of ``b = round(n_dates **
  (1 / 3))`` consecutive dates (at least one), each starting at a uniform
  date and wrapping from the last date to the first, and keeps the first
  ``n_dates`` dates drawn.

  Parameters
  ----------
  n_dates : int
      The number of distinct dates.
  n_boot : int
      Replicates.
  rng : numpy.random.Generator
      The source of the block starts.

  Returns
  -------
  numpy.ndarray, shape (n_boot, n_dates)
      Each replicate's count of each date; every row sums to ``n_dates``.
  """
  if n_dates < 1 or n_boot < 1:
    raise ValueError(
      f"n_dates and n_boot must be positive; got {n_dates} and {n_boot}"
    )
  block = max(1, round(n_dates ** (1.0 / 3.0)))
  n_blocks = -(-n_dates // block)
  starts = rng.integers(n_dates, size=(n_boot, n_blocks))
  drawn = (starts[..., None] + np.arange(block)) % n_dates
  drawn = drawn.reshape(n_boot, -1)[:, :n_dates]
  counts = np.zeros((n_boot, n_dates))
  np.add.at(counts, (np.arange(n_boot)[:, None], drawn), 1.0)
  return counts


def date_block_interval(
  stations: np.ndarray,
  dates: np.ndarray,
  values: np.ndarray,
  *,
  seed: int = 2026,
  n_boot: int = 500,
) -> tuple[float, float, float]:
  """The equal-station mean and a 95% block bootstrap interval over dates.

  Stations share the weather on a date, so the dates are resampled once per
  replicate, for every station at once: circular blocks over the sorted
  distinct dates of all stations (:func:`date_block_counts`). Each replicate
  then resamples the stations with replacement and averages, over them,
  each station's mean over the dates drawn, a date drawn twice counting
  twice. A date a station has no value on contributes nothing to that
  station; a block runs on through it, since the blocks are laid on the
  dates of every station. A station none of whose dates is drawn is left
  out of that replicate. The interval is the replicates' 2.5 % and 97.5 %
  quantiles; it ignores dependence that outlasts a block, so it is
  exploratory.

  Parameters
  ----------
  stations : numpy.ndarray, shape (n,)
      Each value's station.
  dates : numpy.ndarray, shape (n,)
      Each value's date; any type with an order, such as ``datetime64[D]``.
  values : numpy.ndarray, shape (n,)
      One value per station and date, such as a paired score difference.
  seed : int, default=2026
      Seed of the resampling: the dates, then the stations.
  n_boot : int, default=500
      Replicates.

  Returns
  -------
  mean, low, high : float
      ``mean`` is the mean over stations of each station's mean; ``nan``
      throughout for an empty input or a non-finite value.

  Raises
  ------
  ValueError
      If the arrays are not one-dimensional and of one length, or a station
      has two values on one date.
  """
  stations, dates = np.asarray(stations), np.asarray(dates)
  values = np.asarray(values, dtype=np.float64)
  if (
    values.ndim != 1
    or stations.shape != values.shape
    or dates.shape != values.shape
  ):
    raise ValueError(
      f"stations, dates and values must be one-dimensional and of one "
      f"length; got shapes {stations.shape}, {dates.shape} and "
      f"{values.shape}"
    )
  nan = float("nan")
  if not len(values) or not np.isfinite(values).all():
    return nan, nan, nan
  station_ids, station = np.unique(stations, return_inverse=True)
  date_ids, date = np.unique(dates, return_inverse=True)
  n_stations, n_dates = len(station_ids), len(date_ids)
  present = np.zeros((n_stations, n_dates))
  np.add.at(present, (station, date), 1.0)
  if (present > 1.0).any():
    raise ValueError(
      f"values must hold one value per station and date; got "
      f"{int((present > 1.0).sum())} repeated (station, date) pairs"
    )
  table = np.zeros((n_stations, n_dates))
  table[station, date] = values
  mean = float((table.sum(axis=1) / present.sum(axis=1)).mean())
  rng = np.random.default_rng(seed)
  counts = date_block_counts(n_dates, n_boot=n_boot, rng=rng)
  drawn = counts @ present.T
  totals = counts @ table.T
  station_means = np.divide(
    totals, drawn, out=np.zeros_like(totals), where=drawn > 0.0
  )
  picks = rng.integers(n_stations, size=(n_boot, n_stations))
  weight = np.zeros((n_boot, n_stations))
  np.add.at(weight, (np.arange(n_boot)[:, None], picks), 1.0)
  weight *= drawn > 0.0
  kept = weight.sum(axis=1)
  replicates = np.divide(
    (weight * station_means).sum(axis=1),
    kept,
    out=np.full(n_boot, np.nan),
    where=kept > 0.0,
  )
  low, high = np.nanquantile(replicates, [0.025, 0.975])
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


def copula_log_score_columns(cases: pd.DataFrame) -> list[str]:
  """The copula log-score columns present, the main design's first.

  Parameters
  ----------
  cases : pandas.DataFrame
      Per-case copula scores, or a table of their means.

  Returns
  -------
  list of str
      ``joint_nll`` or ``joint_nll_modeled`` (where responses are derived and
      the log score covers the modeled ones only), then each other design's
      ``joint_nll_<design>``, sorted. A table without any gives
      ``["joint_nll"]``.
  """
  main = [c for c in ("joint_nll", "joint_nll_modeled") if c in cases]
  others = sorted(
    c
    for c in cases.columns
    if c.startswith("joint_nll_")
    and c != "joint_nll_modeled"
    and not c.endswith(("_skill", "_gain", "_low", "_high"))
    and not c.startswith("joint_nll_modeled_")
  )
  return [*(main or (["joint_nll"] if not others else [])), *others]


def copula_score_columns(cases: pd.DataFrame) -> tuple[str, list[str]]:
  """The copula scores present: the main log score's column, then every score.

  Parameters
  ----------
  cases : pandas.DataFrame
      Per-case copula scores.

  Returns
  -------
  nll : str
      The first of :func:`copula_log_score_columns`.
  scores : list of str
      Every log score (:func:`copula_log_score_columns`), ``energy``,
      ``variogram``, then each response's CRPS from the draws
      (``crps_<name>``) where present.
  """
  nlls = copula_log_score_columns(cases)
  crps = [
    c
    for c in cases.columns
    if c.startswith("crps_")
    and not c.endswith(("_skill", "_gain", "_low", "_high"))
  ]
  return nlls[0], [*nlls, "energy", "variogram", *crps]


def arm_design(arm: str) -> str:
  """The design an arm or result belongs to; ``""`` for the main one.

  Parameters
  ----------
  arm : str
      An arm or a mixture member, such as ``gaussian@5d``.

  Returns
  -------
  str
      What follows :data:`~npcc.experiments.application.config.ARM_DESIGN`.
  """
  return split_arm(arm)[1] or ""


def _reference_name(reference: str, design: str) -> str:
  """The reference arm of a design: ``independence@5d`` for ``5d``."""
  return f"{reference}{ARM_DESIGN}{design}" if design else reference


def copula_tables(
  cases: pd.DataFrame,
  *,
  reference: str = INDEPENDENCE,
  seed: int = 42,
) -> dict[str, pd.DataFrame]:
  """The copula score tables, each arm against independence on its margins.

  An arm of another design (``<arm>@<design>``) is compared with the
  reference of its own design, ``independence@<design>``, since the two
  designs model different responses; each design's log score has its own
  column (:func:`copula_log_score_columns`).

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
      :func:`marginal_tables`: each score's mean, each log score's gain
      (``<log score>_gain``, the additive improvement on the reference) and
      every other score's skill (``energy_skill``, ``variogram_skill``, a
      ratio of means); and
      ``cases``, the input with each case's log-score gains and its
      ``energy_gain`` and ``variogram_gain`` over the reference's pooled
      mean.
  """
  _check_paired(cases, "arm", ["station", "row"], ["model"])
  _, scores = copula_score_columns(cases)
  nlls = copula_log_score_columns(cases)
  skilled = [s for s in scores if s not in nlls]
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
  design = cases["arm"].map(arm_design)
  is_reference = cases["arm"] == design.map(
    lambda d: _reference_name(reference, d)
  )
  ref_cases = (
    cases.loc[is_reference, ["station", "model", "row", *scores]]
    .assign(_design=design[is_reference])
    .rename(columns={m: f"ref_{m}" for m in scores})
  )
  wide = cases.assign(_design=design).merge(
    ref_cases,
    on=["station", "model", "_design", "row"],
    how="left",
    validate="m:1",
  )
  for nll in nlls:
    wide[f"{nll}_gain"] = wide[f"ref_{nll}"] - wide[nll]
  for metric in ("energy", "variogram"):
    pooled_ref = ref_cases.groupby(["model", "_design"])[f"ref_{metric}"].mean()
    denominator = pd.Series(
      pooled_ref.reindex(
        pd.MultiIndex.from_frame(wide[["model", "_design"]])
      ).to_numpy(),
      index=wide.index,
    )
    wide[f"{metric}_gain"] = (wide[f"ref_{metric}"] - wide[metric]) / (
      denominator.where(denominator > 0)
    )
  wide = wide.drop(columns=[*(f"ref_{m}" for m in scores), "_design"])
  for table, keys in (
    (by_station, ["station", "model"]),
    (pooled, ["model"]),
  ):
    table_design = table["arm"].map(arm_design)
    ref_rows = table["arm"] == table_design.map(
      lambda d: _reference_name(reference, d)
    )
    ref = table.loc[ref_rows, [*keys, *scores]].assign(
      _design=table_design[ref_rows]
    )
    merged = table.assign(_design=table_design).merge(
      ref.rename(columns={m: f"ref_{m}" for m in scores}),
      on=[*keys, "_design"],
      how="left",
      validate="m:1",
    )
    for nll in nlls:
      table[f"{nll}_gain"] = (merged[f"ref_{nll}"] - merged[nll]).to_numpy()
    for metric in skilled:
      ref_mean = merged[f"ref_{metric}"]
      table[f"{metric}_skill"] = (
        1.0 - merged[metric] / ref_mean.where(ref_mean > 0)
      ).to_numpy()
  station_mean = _equal_station(
    by_station,
    ["model", "arm"],
    [
      *means,
      *(f"{nll}_gain" for nll in nlls),
      *(f"{m}_skill" for m in skilled),
    ],
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


def physical_validity(cases: pd.DataFrame, n_draws: int) -> pd.DataFrame:
  """How many draws of each arm break a physical bound, per station.

  Parameters
  ----------
  cases : pandas.DataFrame
      Per-case copula scores, with a ``broken_<check>`` column per physical
      check, each case's count of draws breaking it; ``nan`` for a result
      without draws of its own, which is left out.
  n_draws : int
      Draws per test case.

  Returns
  -------
  pandas.DataFrame
      One row per ``(station, model, arm)``: ``n_cases``, ``n_draws`` (all
      of the cases' draws), and per check its count of draws and their
      ``share``. Empty where no case carries a check.
  """
  checks = [c for c in cases.columns if c.startswith("broken_")]
  if not checks:
    return pd.DataFrame()
  sampled = cases.loc[cases[checks].notna().all(axis=1)]
  if sampled.empty:
    return pd.DataFrame()
  table = cast(
    "pd.DataFrame",
    sampled.groupby(["station", "model", "arm"], sort=True)
    .agg(n_cases=("row", "size"), **{c: (c, "sum") for c in checks})
    .reset_index(),
  )
  table.insert(4, "n_draws", table["n_cases"] * n_draws)
  for c in checks:
    table[c] = table[c].astype(np.int64)
    table[f"{c}_share"] = table[c] / table["n_draws"]
  return table
