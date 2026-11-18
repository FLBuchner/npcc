"""The weather application's report: margins, then copulas, as one PDF.

Reads a run folder ``npcc-application evaluate`` has finished, its
``tables/`` and each unit's stored per-case scores, and fits nothing.
Writes, under ``<run>/reports`` (or ``--out``):

- ``report.pdf``. Part 1, the margins against seasonal EMOS: per response,
  the equal-station table, boxes of the station and the case values and the
  held-out PIT histograms; the coverage; then one station's (``--station``)
  PIT histograms, 30-day rolling scores and coverage. Part 2, each copula
  arm against independence of its design on the same margins and cases:
  per marginal model, the equal-station tables of the gains and skills,
  boxes of the station values and of the case gains; the failed units.
- ``copula_table.csv``, the paper's table: per marginal model and result
  (mixture members and chain orders included), each score's pooled mean,
  the main log score's gain and every other score's skill, each with its
  equal-station mean and interval.

A case's gain is its improvement on the reference over the reference's
pooled mean, so the gains average to the pooled skill. An arm that failed
at any station is left out of its marginal model's comparisons.

Usage (from the repository root)::

    python scripts/paper/application/report.py --run <run> [--station 0]

``dependence.py`` and ``eda_physical_consistency.py`` write their PDFs with
:class:`Report`.
"""

from __future__ import annotations

import argparse
import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from npcc.experiments.application.config import EMOS_MODEL, split_arm
from npcc.experiments.application.scores import (
  INDEPENDENCE,
  NOMINAL_COVERAGE,
  arm_design,
  copula_log_score_columns,
  copula_score_columns,
  copula_tables,
  coverage_column,
  marginal_tables,
)
from npcc.experiments.application.store import read_margin_table

PAGE = (11.7, 8.3)
ROWS_PER_PAGE = 24


def read_table(run: Path, name: str) -> pd.DataFrame:
  """One of ``evaluate``'s tables, refusing a run it has not evaluated."""
  path = Path(run) / "tables" / f"{name}.csv"
  if not path.exists():
    raise FileNotFoundError(
      f"{path} must exist; run `npcc-application evaluate` on {run} first"
    )
  return pd.read_csv(path)


def number(value: float) -> str:
  """Three significant digits, or N/A."""
  return f"{value:.3g}" if np.isfinite(value) else "N/A"


def interval_table(
  frame: pd.DataFrame, first: str, metrics: Sequence[str]
) -> pd.DataFrame:
  """``first``, the stations, and each metric as ``mean [low, high]``."""
  out = {first.title(): frame[first], "Stations": frame["n_stations"]}
  for m in metrics:
    out[m.replace("_", " ")] = [
      f"{number(r[m])} [{number(r[m + '_low'])}, {number(r[m + '_high'])}]"
      for _, r in frame.iterrows()
    ]
  return pd.DataFrame(out)


class Report:
  """A PDF that numbers its pages, written whole or not at all."""

  def __init__(self, path: Path, title: str) -> None:
    self.path = path
    self.title = title
    self.pages = 0

  def __enter__(self) -> Self:
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.partial = self.path.with_name(f".{self.path.name}.partial")
    self.pdf = PdfPages(self.partial)
    self.pdf.infodict()["Title"] = self.title
    return self

  def __exit__(self, kind: object, *_: object) -> None:
    self.pdf.close()
    if kind is None:
      self.partial.replace(self.path)
    else:
      self.partial.unlink(missing_ok=True)

  def save(self, figure: Figure) -> None:
    """Add a page and close its figure."""
    self.pages += 1
    figure.text(0.99, 0.006, str(self.pages), ha="right", fontsize=8)
    self.pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)

  def text(self, title: str, lines: Sequence[str]) -> None:
    """A page of prose."""
    figure = plt.figure(figsize=PAGE)
    figure.text(0.06, 0.9, title, fontsize=18, weight="bold")
    for k, line in enumerate(lines):
      figure.text(0.06, 0.8 - 0.07 * k, line, fontsize=11, va="top", wrap=True)
    self.save(figure)

  def table(self, frame: pd.DataFrame, title: str, note: str = "") -> None:
    """A table over as many pages as it needs, floats to 3 digits."""
    if frame.empty:
      self.text(title, ["No rows.", note])
    for start in range(0, len(frame), ROWS_PER_PAGE):
      part = frame.iloc[start : start + ROWS_PER_PAGE].copy()
      for column in part:
        if pd.api.types.is_float_dtype(part[column]):
          part[column] = part[column].map(number)
      figure, axis = plt.subplots(figsize=PAGE)
      axis.axis("off")
      rows = f" (rows {start + 1}-{start + len(part)})"
      figure.suptitle(title + rows * (len(frame) > ROWS_PER_PAGE), fontsize=13)
      table = axis.table(
        cellText=part.to_numpy(),
        colLabels=list(part.columns),
        cellLoc="center",
        loc="center",
      )
      table.auto_set_font_size(False)
      table.set_fontsize(7)
      table.scale(1, 1.4)
      table.auto_set_column_width(list(range(len(part.columns))))
      figure.text(0.02, 0.02, note, fontsize=8, wrap=True)
      self.save(figure)


def panels(n: int, columns: int = 3) -> tuple[Figure, list[Axes]]:
  """A page of ``n`` panels, ``columns`` to a row."""
  figure, axes = plt.subplots(
    -(-n // columns), columns, figsize=PAGE, squeeze=False, layout="constrained"
  )
  for axis in axes.flat[n:]:
    axis.set_visible(False)
  return figure, list(axes.flat[:n])


def boxes(
  axis: Axes, frame: pd.DataFrame, column: str, groups: Sequence[str], by: str
) -> None:
  """Horizontal boxes of ``column`` per group: quartiles, 1.5 IQR, mean.

  A gain or a skill gets a line at zero, where it ties its reference.
  """
  data = [frame.loc[frame[by] == g, column].dropna() for g in groups]
  kept = [(i, d.to_numpy(float)) for i, d in enumerate(data) if len(d)]
  # `vert` gave way to `orientation` in matplotlib 3.10.
  new = "orientation" in inspect.signature(axis.boxplot).parameters
  if kept:
    axis.boxplot(
      [d for _, d in kept],
      positions=[i for i, _ in kept],
      showfliers=False,
      showmeans=True,
      meanprops={"marker": "D", "markersize": 3},
      **({"orientation": "horizontal"} if new else {"vert": False}),
    )
  axis.set(yticks=range(len(groups)), yticklabels=list(groups))
  axis.set(title=column.replace("_", " "), ylim=(len(groups) - 0.5, -0.5))
  axis.tick_params(labelsize=7)
  axis.grid(axis="x", alpha=0.2)
  if column.endswith(("gain", "skill", "crpss")):
    axis.axvline(0.0, color="black", linewidth=0.6)


def box_page(
  report: Report,
  frame: pd.DataFrame,
  columns: Sequence[str],
  groups: Sequence[str],
  by: str,
  title: str,
) -> None:
  """A page of :func:`boxes`, one panel per column."""
  figure, axes = panels(len(columns), 3 if len(columns) > 4 else 2)
  for axis, column in zip(axes, columns, strict=True):
    boxes(axis, frame, column, groups, by)
  figure.suptitle(title, fontsize=13)
  note = "Boxes: quartiles; whiskers: 1.5 IQR; diamonds: means."
  figure.text(0.01, -0.03, note, fontsize=8)
  report.save(figure)


def pit_page(
  report: Report, cases: pd.DataFrame, models: Sequence[str], title: str
) -> None:
  """Held-out PIT histograms, one panel per model, pooled over ``cases``."""
  figure, axes = panels(len(models))
  for axis, model in zip(axes, models, strict=True):
    pit = cases.loc[cases["model"] == model, "pit"].to_numpy(float)
    axis.hist(pit, bins=np.linspace(0, 1, 11), density=True)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axis.set(xlim=(0, 1), xlabel="PIT", title=f"{model} (n = {len(pit):,})")
  figure.suptitle(f"{title}\nDashed: the uniform density", fontsize=12)
  report.save(figure)


def coverage_page(
  report: Report,
  table: pd.DataFrame,
  models: Sequence[str],
  variables: Sequence[str],
  title: str,
) -> None:
  """Empirical against nominal central-interval coverage, per response."""
  nominal = list(NOMINAL_COVERAGE)
  columns = [coverage_column(n) for n in nominal]
  figure, axes = panels(len(variables))
  for axis, variable in zip(axes, variables, strict=True):
    part = table[table["variable"] == variable].set_index("model")
    axis.plot([0.45, 1], [0.45, 1], "k--", linewidth=0.8)
    for model in models:
      axis.plot(nominal, part.loc[model, columns], "o-", label=model)
    axis.set(title=variable, xlabel="Nominal", ylabel="Empirical")
  axes[0].legend(fontsize=7)
  figure.suptitle(title, fontsize=13)
  report.save(figure)


def time_page(
  report: Report, cases: pd.DataFrame, models: Sequence[str], title: str
) -> None:
  """Trailing 30-day means of the CRPS and NLL over the test year."""
  figure, axes = plt.subplots(2, 1, figsize=PAGE, layout="constrained")
  for model in models:
    part = cases[cases["model"] == model]
    part = part.set_index(pd.to_datetime(part["time"])).sort_index()
    for axis, metric in zip(axes, ("crps", "nll"), strict=True):
      mean = part[metric].rolling("30D", min_periods=10).mean()
      axis.plot(mean.index, mean.to_numpy(), label=model)
      axis.set_ylabel(f"30-day mean {metric.upper()}")
      axis.grid(alpha=0.2)
  axes[0].legend(fontsize="small", ncol=2)
  axes[1].set_xlabel("Valid time; trailing windows of at least 10 cases")
  figure.suptitle(title)
  report.save(figure)


def margin_pages(report: Report, run: Path, station: int) -> None:
  """Part 1: every margin against seasonal EMOS, then one station's."""
  station_mean = read_table(run, "margins_station_mean")
  by_station = read_table(run, "margins_by_station")
  stations = sorted(int(s) for s in set(by_station["station"]))
  if station not in stations:
    raise ValueError(f"station must be one of {stations}; got {station}")
  models = sorted(set(by_station["model"]), key=lambda m: (m != EMOS_MODEL, m))
  variables = sorted(set(by_station["variable"]))
  scores = read_margin_table(run, stations, models, "scores")
  cases = marginal_tables(scores)["cases"]
  report.text(
    "1. Margins",
    [
      f"{len(stations)} station(s), {len(models)} marginal models, "
      f"{len(variables)} responses; trained on 2017-2020, scored on 2021.",
      "CRPSS: 1 - mean(CRPS) / mean(EMOS CRPS). Log-score gain: EMOS's mean "
      "NLL less the model's. Each response in its own units.",
      "Brackets: 95% station-bootstrap intervals (exploratory; spatial "
      "dependence ignored); N/A with one station.",
      "Case boxes: every station's test cases, which are not independent.",
    ],
  )
  for variable in variables:
    report.table(
      interval_table(
        station_mean[station_mean["variable"] == variable],
        "model",
        ["crps", "crpss", "nll", "log_score_gain"],
      ),
      f"{variable}: equal station weights",
    )
    for frame, gain, kind in (
      (by_station, "crpss", "one value per station"),
      (cases, "crps_gain", "every station's test cases"),
    ):
      box_page(
        report,
        frame[frame["variable"] == variable],
        ["crps", "nll", gain, "log_score_gain"],
        models,
        "model",
        f"{variable}: {kind}",
      )
    part = cases[cases["variable"] == variable]
    pit_page(report, part, models, f"{variable}: held-out PIT, every station")
  pooled = read_table(run, "margins_pooled")
  coverage_page(report, pooled, models, variables, "Coverage, pooled")
  for variable in variables:
    part = cases[
      (cases["station"] == station) & (cases["variable"] == variable)
    ]
    title = f"Station {station}, {variable}"
    pit_page(report, part, models, f"{title}: held-out PIT")
    time_page(report, part, models, f"{title}: scores over the test year")
  one = by_station[by_station["station"] == station]
  coverage_page(report, one, models, variables, f"Station {station}: coverage")


def stored_copula_cases(run: Path, tables: pd.DataFrame) -> pd.DataFrame:
  """The stored per-case scores of the arms as run that ``tables`` holds."""
  _, scores = copula_score_columns(tables)
  wanted = ["station", "model", "run_arm", "arm", "row", *scores]
  frames = []
  for path in sorted(run.glob("stations/*/copulas/*/*/scores.parquet")):
    frame = pd.read_parquet(path)
    keep = frame["arm"] == frame["run_arm"]
    frames.append(frame.loc[keep, [c for c in wanted if c in frame]])
  if not frames:
    raise FileNotFoundError(f"{run} must hold copula scores; run `copulas`")
  keys = tables[["station", "model", "arm"]].drop_duplicates()
  cases = pd.concat(frames, ignore_index=True).merge(keys)
  return cases.drop(columns="run_arm")


def paper_table(
  pooled: pd.DataFrame, station_mean: pd.DataFrame
) -> pd.DataFrame:
  """Pooled means, gains and skills, each with its equal-station interval.

  Another design's own log score (``joint_nll_5d``) is left out: two of its
  five responses are functions of the other three, so it has no density.
  """
  nlls = copula_log_score_columns(pooled)
  _, scores = copula_score_columns(pooled)
  metrics = [s for s in scores if s not in nlls[1:]] + [f"{nlls[0]}_gain"]
  metrics += [f"{s}_skill" for s in scores if s not in nlls]
  merged = pooled.merge(
    station_mean,
    on=["model", "arm"],
    suffixes=("", "_stations"),
    validate="1:1",
  )
  out = {k: merged[k] for k in ("model", "arm", "n_stations", "n_cases")}
  for m in metrics:
    out[m] = merged[m]
    out[f"{m}_station_mean"] = merged[f"{m}_stations"]
    out[f"{m}_low"] = merged[f"{m}_low"]
    out[f"{m}_high"] = merged[f"{m}_high"]
  return pd.DataFrame(out)


def arm_key(arm: str) -> tuple[str, bool, str]:
  """The main design's arms first, each design's independence first."""
  return arm_design(arm), split_arm(arm)[0] != INDEPENDENCE, arm


def copula_pages(report: Report, run: Path, table: Path) -> None:
  """Part 2: every arm against independence; writes the paper's ``table``."""
  pooled = read_table(run, "copulas_pooled")
  station_mean = read_table(run, "copulas_station_mean")
  by_station = read_table(run, "copulas_by_station")
  failures = read_table(run, "copula_failures")
  paper_table(pooled, station_mean).to_csv(table, index=False)
  stored = stored_copula_cases(run, by_station)
  cases = copula_tables(stored)["cases"]
  run_arms = set(stored["arm"])
  gain = f"{copula_log_score_columns(pooled)[0]}_gain"
  _, scores = copula_score_columns(pooled)
  skills = [f"{s}_skill" for s in scores if not s.startswith(("joint", "crps"))]
  crps = [f"{s}_skill" for s in scores if s.startswith("crps_")]
  report.text(
    "2. Copulas",
    [
      "Fitted to the out-of-fold training PITs; scored on the 2021 cases.",
      "Each arm against independence of its design, on the same margins.",
      "Log-score gain: independence's mean NLL less the arm's, on the main "
      "design's log score. Skill: 1 - mean / independence's mean.",
      "Energy and variogram scores: each response over its training SD.",
      "Tables: every result, mixture members and chain orders included. "
      "Boxes: the arms as run.",
      f"{len(failures)} failed unit(s), listed after the comparisons.",
    ],
  )
  for model in sorted(set(by_station["model"])):
    mean = station_mean[station_mean["model"] == model]
    mean = mean.set_index("arm").loc[sorted(mean["arm"], key=arm_key)]
    mean = mean.reset_index()
    arms = [a for a in mean["arm"] if a in run_arms]
    for metrics, kind in (
      ([gain, *skills], "gains and skills"),
      (crps, "each response's CRPS skill, from the draws"),
    ):
      if not metrics:
        continue
      report.table(
        interval_table(mean, "arm", metrics),
        f"{model}: {kind}, equal station weights",
        "Brackets: 95% station-bootstrap intervals.",
      )
      box_page(
        report,
        by_station[by_station["model"] == model],
        metrics,
        arms,
        "arm",
        f"{model}: {kind}, one value per station",
      )
    box_page(
      report,
      cases[cases["model"] == model],
      [gain, "energy_gain", "variogram_gain"],
      arms,
      "arm",
      f"{model}: every station's test cases",
    )
  report.table(
    failures,
    "Failed copula units",
    "Left out of their marginal model's comparisons at every station.",
  )


def build(run: Path, out: Path, station: int = 0) -> list[Path]:
  """Write the report of ``run`` and the paper's copula table into ``out``."""
  out.mkdir(parents=True, exist_ok=True)
  path, table = out / "report.pdf", out / "copula_table.csv"
  with Report(path, f"Weather application: {Path(run).name}") as report:
    margin_pages(report, run, station)
    copula_pages(report, run, table)
  return [path, table]


def main(argv: list[str] | None = None) -> int:
  """Write the report from the command line."""
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--run", required=True, type=Path, help="The run folder.")
  p.add_argument("--out", type=Path, help="Where to write; <run>/reports.")
  p.add_argument("--station", type=int, default=0, help="Part 1's station.")
  args = p.parse_args(argv)
  for path in build(args.run, args.out or args.run / "reports", args.station):
    print(f"wrote {path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
