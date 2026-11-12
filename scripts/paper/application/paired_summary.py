"""The application's primary summaries: pooled means, and paired station means.

Fixed before the full run's results were seen:

(i) every copula arm's mean score over all test cases, one table per score:
    the joint NLL of the modeled responses, the energy and variogram scores on
    all responses, and each response's sample CRPS;
(ii) every arm against each reference arm, paired by station: the mean over
    stations of the difference of station means (arm - reference, so
    negative favors the arm), its standard error across stations, and the
    stations where the arm is lower, higher or level. The references are the
    Gaussian copula on the arm's own margins and, across margins, the npcc
    mixture on all covariates on the TabPFN margins. Every score is
    lower-is-better.

The paired differences are given on every station and again without the
stations ``--without`` names (station 106 by default, whose pressure shifts
across the train/test split), as a robustness check.

Reads ``tables/copulas_by_station.csv`` and ``tables/copulas_pooled.csv`` of a
run ``npcc-application evaluate`` has finished, and writes into ``--out``
(``<run>/analysis`` by default):

- ``pooled_means.csv`` and ``pooled_means.md``: (i), the second one table per
  score with the arms as rows and the margins as columns;
- ``paired_by_station.csv``: each station's difference, per arm, reference
  and score;
- ``paired.csv``: (ii), per station set, arm, reference and score.

Usage::

    python scripts/paper/application/paired_summary.py --run <run folder>
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

KEYS = ("station", "model", "arm")
#: The arm every arm is compared with on its own margins.
SAME_MARGINS = "gaussian"
#: The (margins, arm) every arm is compared with across margins.
ANCHOR = ("tabpfn-quantiles-all", "npcc-mixture-tabpfn-quantiles-all")


def score_columns(by_station: pd.DataFrame) -> list[str]:
  """The scores of a by-station table, in its column order."""
  return [
    c
    for c in by_station.columns
    if c not in (*KEYS, "n_cases") and not c.endswith(("_gain", "_skill"))
  ]


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

  Returns
  -------
  pandas.DataFrame
      ``station``, ``model``, ``arm``, ``reference``, ``score``,
      ``arm_mean``, ``reference_mean`` and ``diff``, for every arm other
      than the reference itself; a reference absent at a station leaves
      that station out.
  """
  long = by_station.melt(
    id_vars=list(KEYS), value_vars=list(scores), var_name="score"
  )
  parts = []
  ref_same = long[long["arm"] == same_margins].rename(
    columns={"value": "reference_mean"}
  )
  same = long.merge(
    ref_same.drop(columns="arm"),
    on=["station", "model", "score"],
    validate="m:1",
  )
  same = same[same["arm"] != same_margins]
  parts.append(same.assign(reference=f"{same_margins} (same margins)"))
  model, arm = anchor
  ref_anchor = long[(long["model"] == model) & (long["arm"] == arm)].rename(
    columns={"value": "reference_mean"}
  )
  across = long.merge(
    ref_anchor.drop(columns=["model", "arm"]),
    on=["station", "score"],
    validate="m:1",
  )
  across = across[~((across["model"] == model) & (across["arm"] == arm))]
  parts.append(across.assign(reference=f"{arm} on {model}"))
  out = pd.concat(parts, ignore_index=True).rename(
    columns={"value": "arm_mean"}
  )
  out["diff"] = out["arm_mean"] - out["reference_mean"]
  columns = [*KEYS, "reference", "score", "arm_mean", "reference_mean", "diff"]
  return pd.DataFrame(
    out[columns].sort_values(["reference", "model", "arm", "score", "station"])
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
  run: Path, out: Path, *, without: Sequence[int] = (106,)
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

  Returns
  -------
  list of pathlib.Path
      The files written.
  """
  by_station, pooled = read_tables(run)
  scores = score_columns(by_station)
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
  path = out / "paired_by_station.csv"
  diffs.to_csv(path, index=False)
  written.append(path)
  kept = diffs[~diffs["station"].isin(list(without))]
  label = "without " + ", ".join(str(s) for s in without)
  summary = pd.concat(
    [paired(diffs, label="all"), paired(kept, label=label)],
    ignore_index=True,
  )
  path = out / "paired.csv"
  summary.to_csv(path, index=False)
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
