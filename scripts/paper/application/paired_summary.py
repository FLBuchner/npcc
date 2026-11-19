"""The application's primary summaries: pooled means, and paired station means.

Fixed before the full run's results were seen:

(i) every copula arm's mean score over all test cases, one table per score:
    the joint NLL of the main design's modeled responses, the energy and
    variogram scores on all responses and on each response subset, and each
    response's sample CRPS. Another design's own log score
    (``joint_nll_<design>``) is stored by the run but neither summarized nor
    ranked: two of the five responses are functions of the other three, so
    the five have no joint density;
(ii) every arm against each reference arm, paired by station: the mean over
    stations of the difference of station means (arm - reference, so
    negative favors the arm), its 95% block bootstrap interval (below), its
    standard error across stations, and the stations where the arm is
    lower, higher or level. The references are the
    Gaussian copula on the arm's own margins and design (``gaussian@5d`` for
    an arm ``<arm>@5d``) and, across margins, the npcc mixture on all
    covariates on the TabPFN margins. Every score is lower-is-better;
(iii) where the run has arms of another design (``<arm>@<design>``), each
    against the same arm of the main design on the same margins, on the
    scores both are drawn on: energy, variogram, each response subset's
    energy and variogram, each response's sample CRPS, and the share of
    draws breaking each physical bound
    (``tables/copulas_physical_validity.csv``). Of the log scores, only the
    main design's enters, where an arm of another design has one: the
    exact projection of its independence or Gaussian copula onto the main
    design's modeled responses.

The paired differences are given on every station and again without the
stations ``--without`` names (station 106 by default, whose pressure shifts
across the train/test split), as a robustness check.

**The interval.** FLBuchner's hierarchical circular-block bootstrap
(npcc-paper PR #6), with the blocks laid on the calendar: each of 500
replicates draws circular blocks of ``round(n_dates ** (1 / 3))``
consecutive test dates, the same dates at every station, since stations
share the weather on a date, and resamples the stations with replacement;
its statistic is the mean over the stations drawn of each one's mean
paired difference over the dates drawn. The interval is the replicates'
2.5 % and 97.5 % quantiles, from a fixed seed
(:func:`npcc.experiments.application.scores.date_block_interval`). A case's
date is that of its valid time. It reads each test case's scores from the
copula units, ``stations/<NNN>/copulas/<model>/<arm>/scores.parquet``, and
refuses units whose means are not the tables'.

Reads ``tables/copulas_by_station.csv`` and ``tables/copulas_pooled.csv`` of a
run ``npcc-application evaluate`` has finished, and writes into ``--out``
(``<run>/analysis`` by default):

- ``pooled_means.csv`` and ``pooled_means.md``: (i), the second one table per
  score with the arms as rows and the margins as columns;
- ``paired_by_station.csv``: each station's difference, per arm, reference
  and score;
- ``paired.csv``: (ii) and (iii), per station set, arm, reference and score,
  with the interval in ``boot_low`` and ``boot_high``;
- ``paired.md``: the same, one table per station set and score, each
  difference with its interval.

Usage::

    python scripts/paper/application/paired_summary.py --run <run folder>
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from npcc.experiments.application.scores import date_block_interval
from npcc.experiments.application.store import station_dir

KEYS = ("station", "model", "arm")
#: The arm every arm is compared with on its own margins.
SAME_MARGINS = "gaussian"
#: The (margins, arm) every arm is compared with across margins.
ANCHOR = ("tabpfn-quantiles-all", "npcc-mixture-tabpfn-quantiles-all")
#: What separates an arm from its design, as in ``gaussian@5d``.
DESIGN = "@"
#: The validity table's columns (iii) compares, by their stem.
SHARE_SUFFIX = "_share"
#: The interval's replicates and seed, FLBuchner's.
N_BOOT = 500
BOOT_SEED = 2026
#: What identifies a comparison in ``paired.csv``.
COMPARISON = ("stations", "model", "arm", "reference", "score")


#: The main design's log-score columns; ``joint_nll_<design>`` is another's.
MAIN_LOG_SCORES = ("joint_nll", "joint_nll_modeled")


def score_columns(by_station: pd.DataFrame) -> list[str]:
  """The scores of a by-station table, in its column order.

  Another design's own log score is left out: it is stored, not summarized.
  """
  return [
    c
    for c in by_station.columns
    if c not in (*KEYS, "n_cases")
    and not c.endswith(("_gain", "_skill"))
    and (not c.startswith("joint_nll") or c in MAIN_LOG_SCORES)
  ]


def design_of(arm: str) -> tuple[str, str]:
  """An arm's base and its design, ``""`` for the main one."""
  base, _, design = arm.partition(DESIGN)
  return base, design


def read_validity(run: Path) -> pd.DataFrame | None:
  """The run's physical-validity table, or ``None`` where it has none."""
  path = Path(run) / "tables" / "copulas_physical_validity.csv"
  return pd.read_csv(path) if path.exists() else None


def read_tables(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
  """A finished run's by-station and pooled copula tables."""
  tables = Path(run) / "tables"
  found = []
  for name in ("copulas_by_station", "copulas_pooled"):
    path = tables / f"{name}.csv"
    if not path.exists():
      raise FileNotFoundError(
        f"{path} must exist; run `npcc-application evaluate` on {run} first"
      )
    found.append(pd.read_csv(path))
  return found[0], found[1]


def paired_by_station(
  by_station: pd.DataFrame,
  scores: Sequence[str],
  *,
  same_margins: str = SAME_MARGINS,
  anchor: tuple[str, str] = ANCHOR,
  within: Sequence[str] = (),
) -> pd.DataFrame:
  """Each station's difference of means, per arm, reference and score.

  Parameters
  ----------
  by_station : pandas.DataFrame
      One row per station, margins and arm, with each score's mean.
  scores : sequence of str
      The score columns.
  same_margins : str, default="gaussian"
      The reference arm on each arm's own margins.
  anchor : tuple of str, default=ANCHOR
      The ``(margins, arm)`` reference across margins.
  within : sequence of str, default=()
      Columns that tell a station's cases apart, such as ``("date",)``,
      for one row per case instead of per station: each case is then
      paired with the reference's at the same station and ``within``.

  Returns
  -------
  pandas.DataFrame
      ``station``, ``model``, ``arm``, the ``within`` columns,
      ``reference``, ``score``, ``arm_mean``, ``reference_mean`` and
      ``diff``, for every arm other than the reference itself; a reference
      absent at a station leaves that station out, and a score either side
      lacks (nan) leaves that comparison out.
  """
  ids = [*KEYS, *within]
  long = by_station.melt(id_vars=ids, value_vars=list(scores), var_name="score")
  parts = []
  # The reference on an arm's own margins is of its own design, which models
  # the same responses: gaussian for the main design, gaussian@5d for 5d.
  reference = long["arm"].map(
    lambda a: (
      f"{same_margins}{DESIGN}{design_of(a)[1]}"
      if design_of(a)[1]
      else same_margins
    )
  )
  long_same = long.assign(_reference=reference)
  ref_same = long_same[long_same["arm"] == long_same["_reference"]].rename(
    columns={"value": "reference_mean"}
  )
  same = long_same.merge(
    ref_same.drop(columns="arm"),
    on=["station", *within, "model", "_reference", "score"],
    validate="m:1",
  )
  same = same[same["arm"] != same["_reference"]]
  parts.append(
    same.assign(reference=same["_reference"] + " (same margins)").drop(
      columns="_reference"
    )
  )
  model, arm = anchor
  ref_anchor = long[(long["model"] == model) & (long["arm"] == arm)].rename(
    columns={"value": "reference_mean"}
  )
  across = long.merge(
    ref_anchor.drop(columns=["model", "arm"]),
    on=["station", *within, "score"],
    validate="m:1",
  )
  across = across[~((across["model"] == model) & (across["arm"] == arm))]
  parts.append(across.assign(reference=f"{arm} on {model}"))
  out = pd.concat(parts, ignore_index=True).rename(
    columns={"value": "arm_mean"}
  )
  out["diff"] = out["arm_mean"] - out["reference_mean"]
  # A chain mixture's order is scored by its log score alone; its sample
  # scores are nan, and a comparison it cannot enter is left out rather than
  # counted. So is a log score of one design against another's.
  out = out[out["diff"].notna()]
  columns = [
    *ids,
    "reference",
    "score",
    "arm_mean",
    "reference_mean",
    "diff",
  ]
  return pd.DataFrame(
    out[columns].sort_values(
      ["reference", "model", "arm", "score", "station", *within]
    )
  )


def design_pairs(
  by_station: pd.DataFrame,
  scores: Sequence[str],
  validity: pd.DataFrame | None = None,
  *,
  within: Sequence[str] = (),
) -> pd.DataFrame:
  """(iii): each arm of another design against its main-design arm.

  Parameters
  ----------
  by_station : pandas.DataFrame
      One row per station, margins and arm, with each score's mean.
  scores : sequence of str
      The score columns. Of the log scores only the main design's
      (:data:`MAIN_LOG_SCORES`) enters, and only where the other design's arm
      has it: a projection onto the main design's modeled responses. Any
      other log score is of different responses and is left out.
  validity : pandas.DataFrame, optional
      ``tables/copulas_physical_validity.csv``: its ``broken_*_share``
      columns are compared too.
  within : sequence of str, default=()
      As for :func:`paired_by_station`.

  Returns
  -------
  pandas.DataFrame
      The columns of :func:`paired_by_station`, ``reference`` reading
      ``<arm> (main design, same margins)``; ``diff`` is the other design's
      mean less the main design's. Empty without arms of another design.
  """
  ids = [*KEYS, *within]
  columns = [*ids, "reference", "score", "arm_mean", "reference_mean", "diff"]
  designed = by_station["arm"].map(lambda a: bool(design_of(a)[1]))
  if not designed.any():
    return pd.DataFrame(columns=columns)
  sample = [
    s for s in scores if not s.startswith("joint_nll") or s in MAIN_LOG_SCORES
  ]
  table = by_station[[*ids, *sample]]
  if validity is not None:
    shares = [c for c in validity.columns if c.endswith(SHARE_SUFFIX)]
    table = table.merge(
      validity[[*KEYS, *shares]], on=list(KEYS), how="left", validate="1:1"
    )
    sample = [*sample, *shares]
  long = table.melt(id_vars=ids, value_vars=sample, var_name="score")
  long["_base"] = long["arm"].map(lambda a: design_of(a)[0])
  other = long[long["arm"] != long["_base"]]
  main = long[long["arm"] == long["_base"]].rename(
    columns={"value": "reference_mean"}
  )
  out = other.merge(
    main.drop(columns="arm"),
    on=["station", *within, "model", "_base", "score"],
    validate="m:1",
  ).rename(columns={"value": "arm_mean"})
  out["reference"] = out["_base"] + " (main design, same margins)"
  out["diff"] = out["arm_mean"] - out["reference_mean"]
  out = out[out["diff"].notna()]
  return pd.DataFrame(
    out[columns].sort_values(
      ["reference", "model", "arm", "score", "station", *within]
    )
  )


def paired(diffs: pd.DataFrame, *, label: str) -> pd.DataFrame:
  """Summarize :func:`paired_by_station`'s differences over stations.

  Parameters
  ----------
  diffs : pandas.DataFrame
      Its output, or a subset of its stations.
  label : str
      The station set, recorded in a ``stations`` column.

  Returns
  -------
  pandas.DataFrame
      Per ``model``, ``arm``, ``reference`` and ``score``: ``n_stations``,
      ``mean_diff``, ``se`` (the standard deviation of the differences over
      the square root of their number), and ``lower``, ``higher`` and
      ``level``, the stations where the arm's mean is below, above or equal
      to the reference's.
  """
  grouped = diffs.groupby(["model", "arm", "reference", "score"], sort=False)[
    "diff"
  ]
  out = grouped.agg(
    n_stations="size",
    mean_diff="mean",
    sd="std",
    lower=lambda d: int((d < 0).sum()),
    higher=lambda d: int((d > 0).sum()),
    level=lambda d: int((d == 0).sum()),
  ).reset_index()
  out["se"] = out["sd"] / out["n_stations"].map(math.sqrt)
  out = out.drop(columns="sd")
  out.insert(0, "stations", label)
  columns = [
    "stations",
    "model",
    "arm",
    "reference",
    "score",
    "n_stations",
    "mean_diff",
    "se",
    "lower",
    "higher",
    "level",
  ]
  return pd.DataFrame(out[columns])


def read_cases(
  run: Path,
  by_station: pd.DataFrame,
  scores: Sequence[str],
  validity: pd.DataFrame | None = None,
) -> pd.DataFrame:
  """Each test case's scores, for every station, margins and arm summarized.

  Parameters
  ----------
  run : pathlib.Path
      The run folder.
  by_station : pandas.DataFrame
      ``tables/copulas_by_station.csv``: which ``(station, model, arm)`` to
      read, and the means their cases must reproduce.
  scores : sequence of str
      The score columns.
  validity : pandas.DataFrame, optional
      ``tables/copulas_physical_validity.csv``: its draws per case turn
      each case's count of draws breaking a check into its share.

  Returns
  -------
  pandas.DataFrame
      One row per ``(station, model, arm)`` and test case: ``row``,
      ``date`` (the day of its valid time), the scores, and each
      ``broken_<check>_share`` of ``validity``.

  Raises
  ------
  FileNotFoundError
      If a summarized ``(station, model, arm)`` has no stored cases.
  ValueError
      If the stored cases' count or means differ from ``by_station``'s:
      the units changed after ``evaluate`` wrote the tables.
  """
  keys = list(KEYS)
  wanted = by_station[keys].drop_duplicates()
  frames = []
  for (station, model), arms in wanted.groupby(["station", "model"]):
    folder = station_dir(run, int(station)) / "copulas" / str(model)
    units = (
      sorted(u for u in folder.iterdir() if not u.name.startswith("."))
      if folder.is_dir()
      else []
    )
    for unit in units:
      path = unit / "scores.parquet"
      if not path.exists():
        continue
      frame = pd.read_parquet(path)
      frames.append(frame[frame["arm"].isin(set(arms["arm"]))])
  broken = sorted(
    {c for f in frames for c in f.columns if c.startswith("broken_")}
  )
  cases = (
    pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
  ).reindex(columns=[*keys, "row", "time", *scores, *broken])
  found = (
    cases[keys].drop_duplicates().merge(wanted, how="right", indicator=True)
  )
  missing = found[found["_merge"] == "right_only"]
  if not missing.empty:
    raise FileNotFoundError(
      f"every (station, model, arm) of {run}/tables/copulas_by_station.csv "
      f"must have its test cases in its unit's scores.parquet; missing "
      f"{list(missing[keys].itertuples(index=False, name=None))[:5]}"
    )
  if cases.duplicated([*keys, "row"]).any():
    raise ValueError(
      "the copula units must hold one row per (station, model, arm) and test "
      "case; some repeat"
    )
  grouped = cases.groupby(keys, sort=True)
  columns = ["n_cases", *scores]
  stored = grouped[list(scores)].mean().join(grouped.size().rename("n_cases"))
  stored = stored[columns]
  table = by_station.set_index(keys)[columns].reindex(stored.index)
  agree = np.isclose(
    stored.to_numpy(dtype=np.float64),
    table.to_numpy(dtype=np.float64),
    rtol=1e-9,
    atol=1e-12,
    equal_nan=True,
  ).all(axis=1)
  if not agree.all():
    raise ValueError(
      f"the copula units' test cases must be those "
      f"tables/copulas_by_station.csv summarizes; {int((~agree).sum())} "
      f"(station, model, arm) differ, such as "
      f"{stored.index[~agree][0]}. Run `npcc-application evaluate` on "
      f"{run} again."
    )
  cases["date"] = pd.to_datetime(cases["time"]).dt.normalize()
  if validity is not None:
    draws = validity[keys].assign(
      _draws=validity["n_draws"] / validity["n_cases"]
    )
    cases = cases.merge(draws, on=keys, how="left", validate="m:1")
    for share in (c for c in validity.columns if c.endswith(SHARE_SUFFIX)):
      cases[share] = cases[share.removesuffix(SHARE_SUFFIX)] / cases["_draws"]
    cases = cases.drop(columns="_draws")
  return cases.drop(columns=["time"])


def block_intervals(
  cases: pd.DataFrame,
  scores: Sequence[str],
  *,
  without: Sequence[int],
  label: str,
  seed: int = BOOT_SEED,
  n_boot: int = N_BOOT,
) -> pd.DataFrame:
  """Each comparison's 95% block bootstrap interval, from its test cases.

  Parameters
  ----------
  cases : pandas.DataFrame
      :func:`read_cases`'s output.
  scores : sequence of str
      The score columns; each ``broken_<check>_share`` of ``cases`` is
      compared too, between designs alone, as in :func:`design_pairs`.
  without : sequence of int
      The stations the robustness check leaves out.
  label : str
      That check's station set, as in ``paired.csv``.
  seed : int, default=2026
      Seed of every comparison's resampling.
  n_boot : int, default=500
      Replicates.

  Returns
  -------
  pandas.DataFrame
      Per comparison of ``paired.csv`` (:data:`COMPARISON`): ``boot_low``
      and ``boot_high``, of
      :func:`~npcc.experiments.application.scores.date_block_interval` on
      its paired differences, case by case.
  """
  shares = [c for c in cases.columns if c.endswith(SHARE_SUFFIX)]
  within = ("date",)
  rows = []
  for score in [*scores, *shares]:
    parts = [design_pairs(cases, [score], within=within)]
    if score in scores:
      parts.insert(0, paired_by_station(cases, [score], within=within))
    # A run of the main design alone pairs no broken-draw share.
    found = [p for p in parts if not p.empty]
    if not found:
      continue
    diffs = pd.concat(found, ignore_index=True)
    for stations, subset in (
      ("all", diffs),
      (label, diffs[~diffs["station"].isin(list(without))]),
    ):
      for key, group in subset.groupby(
        ["model", "arm", "reference", "score"], sort=False
      ):
        _, low, high = date_block_interval(
          group["station"].to_numpy(),
          group["date"].to_numpy(),
          group["diff"].to_numpy(),
          seed=seed,
          n_boot=n_boot,
        )
        rows.append(
          dict(zip(COMPARISON, (stations, *key), strict=True))
          | {"boot_low": low, "boot_high": high}
        )
  return pd.DataFrame(rows, columns=[*COMPARISON, "boot_low", "boot_high"])


def paired_markdown(summary: pd.DataFrame) -> str:
  """(ii) and (iii) as Markdown: per station set and score, every pair."""

  def number(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.4g}"

  blocks = [
    "Arm less reference: the mean over stations of the difference of "
    "station means, negative favoring the arm, and in brackets its 95% "
    f"block bootstrap interval ({N_BOOT} replicates of circular blocks of "
    "test dates shared by every station, and of the stations; seed "
    f"{BOOT_SEED}). Stations: those where the arm is lower / higher / level."
  ]
  for (stations, score), table in summary.groupby(
    ["stations", "score"], sort=False
  ):
    rows = [
      f"| {r.model} | {r.arm} | {r.reference} | {number(r.mean_diff)} "
      f"[{number(r.boot_low)}, {number(r.boot_high)}] | "
      f"{r.lower} / {r.higher} / {r.level} of {r.n_stations} |"
      for r in table.itertuples(index=False)
    ]
    blocks.append(
      "\n".join(
        [
          f"### {score}, {'every station' if stations == 'all' else stations}",
          "",
          "| margins | arm | reference | difference [95% interval] | "
          "stations |",
          "|---|---|---|---|---|",
          *rows,
        ]
      )
    )
  return "\n\n".join(blocks) + "\n"


def pooled_markdown(pooled: pd.DataFrame, scores: Sequence[str]) -> str:
  """(i) as Markdown: per score, the arms as rows and the margins as columns."""
  blocks = []
  for score in scores:
    table = pooled.pivot(index="arm", columns="model", values=score)
    table = table.loc[[a for a in pooled["arm"].unique() if a in table.index]]
    header = "| arm | " + " | ".join(table.columns) + " |"
    rule = "|---|" + "---|" * len(table.columns)
    rows = [
      f"| {arm} | "
      + " | ".join("" if pd.isna(v) else f"{v:.4f}" for v in row)
      + " |"
      for arm, row in table.iterrows()
    ]
    blocks.append("\n".join([f"### {score}", "", header, rule, *rows]))
  return "\n\n".join(blocks) + "\n"


def build(
  run: Path,
  out: Path,
  *,
  without: Sequence[int] = (106,),
  seed: int = BOOT_SEED,
  n_boot: int = N_BOOT,
) -> list[Path]:
  """Write the summaries of a finished run.

  Parameters
  ----------
  run : pathlib.Path
      The run folder.
  out : pathlib.Path
      Where to write.
  without : sequence of int, default=(106,)
      The stations the robustness check leaves out.
  seed : int, default=2026
      Seed of the block bootstrap.
  n_boot : int, default=500
      Its replicates.

  Returns
  -------
  list of pathlib.Path
      The files written.
  """
  by_station, pooled = read_tables(run)
  scores = score_columns(by_station)
  validity = read_validity(run)
  cases = read_cases(run, by_station, scores, validity)
  out.mkdir(parents=True, exist_ok=True)
  written = []

  means = pooled[["model", "arm", "n_cases", "n_stations", *scores]]
  path = out / "pooled_means.csv"
  means.to_csv(path, index=False)
  written.append(path)
  path = out / "pooled_means.md"
  path.write_text(pooled_markdown(means, scores))
  written.append(path)

  diffs = paired_by_station(by_station, scores)
  pairs = design_pairs(by_station, scores, validity)
  if not pairs.empty:
    diffs = pd.concat([diffs, pairs], ignore_index=True)
  path = out / "paired_by_station.csv"
  diffs.to_csv(path, index=False)
  written.append(path)
  kept = diffs[~diffs["station"].isin(list(without))]
  label = "without " + ", ".join(str(s) for s in without)
  summary = pd.concat(
    [paired(diffs, label="all"), paired(kept, label=label)],
    ignore_index=True,
  )
  intervals = block_intervals(
    cases, scores, without=without, label=label, seed=seed, n_boot=n_boot
  )
  summary = summary.merge(
    intervals, on=list(COMPARISON), how="left", validate="1:1", indicator=True
  )
  unmatched = summary["_merge"] != "both"
  if unmatched.any():
    raise ValueError(
      f"every paired comparison must have its test cases' differences for "
      f"its interval; {int(unmatched.sum())} have none, such as "
      f"{tuple(summary.loc[unmatched, list(COMPARISON)].iloc[0])}"
    )
  summary = summary.drop(columns="_merge")
  path = out / "paired.csv"
  summary.to_csv(path, index=False)
  written.append(path)
  path = out / "paired.md"
  path.write_text(paired_markdown(summary))
  written.append(path)
  return written


def main(argv: list[str] | None = None) -> int:
  """Run :func:`build` from the command line.

  Parameters
  ----------
  argv : list of str, optional
      The arguments; ``None`` reads them from ``sys.argv``.

  Returns
  -------
  int
      The exit status.
  """
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--run", required=True, type=Path, help="The run folder.")
  p.add_argument(
    "--out",
    type=Path,
    default=None,
    help="Where to write; <run>/analysis by default.",
  )
  p.add_argument(
    "--without",
    type=int,
    nargs="+",
    default=[106],
    help="The stations the robustness check leaves out.",
  )
  a = p.parse_args(argv)
  out = a.out if a.out is not None else a.run / "analysis"
  for path in build(a.run, out, without=a.without):
    print(path)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
