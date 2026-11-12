"""The marginal report: scores, skills, PIT, coverage and timings, as a PDF.

Ports FLBuchner's ``application/report.py``, ``report_pits.py`` and
``report_timing.py`` (npcc-paper PR #4) onto a run folder: it reads the
tables ``npcc-application evaluate`` wrote and the stored per-case scores,
and fits nothing. Writes ``<run>/reports/marginal_report.pdf`` (or
``--out``):

1. every station: per response, the equal-station means with their station
   bootstrap intervals and boxplots of the station values; the pooled means
   and boxplots of the case values; the pooled held-out PIT histograms;
   then the central-interval coverage and the stage timings;
2. one station (``--station``, 0 by default): its scores, PIT histograms,
   coverage curves and 30-day rolling scores.

CRPSS is ``1 - mean(CRPS) / mean(EMOS CRPS)``; a case's CRPS gain is its
improvement on EMOS over EMOS's pooled mean CRPS, so the gains average to
the pooled CRPSS. The log-score gain is EMOS's mean NLL less the model's.

Usage (from the repository root)::

    python scripts/paper/application/report_margins.py --run results/application
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_common as common

from npcc.experiments.application.config import EMOS_MODEL
from npcc.experiments.application.scores import (
  NOMINAL_COVERAGE,
  coverage_column,
  marginal_tables,
)
from npcc.experiments.application.store import read_margin_table

_CASE_METRICS = (
  ("crps", "CRPS (lower is better)", False),
  ("nll", "NLL (lower is better)", False),
  ("crps_gain", "CRPS gain on EMOS / pooled EMOS CRPS", True),
  ("log_score_gain", "Log-score gain on EMOS", True),
)
_STATION_METRICS = (
  ("crps", "CRPS (lower is better)", False),
  ("nll", "NLL (lower is better)", False),
  ("crpss", "CRPSS against EMOS", True),
  ("log_score_gain", "Log-score gain on EMOS", True),
)


def _score_table(frame: pd.DataFrame, *, intervals: bool) -> pd.DataFrame:
  """A model per row: cases, CRPS, CRPSS, NLL and log-score gain."""
  out = pd.DataFrame({"Model": frame["model"]})
  if intervals:
    out["Stations"] = frame["n_stations"]
    for metric, label in (
      ("crps", "CRPS"),
      ("crpss", "CRPSS"),
      ("nll", "NLL"),
      ("log_score_gain", "Log-score gain"),
    ):
      out[label] = common.with_interval(frame, metric)
  else:
    out["Stations"] = frame["n_stations"]
    out["Cases"] = frame["n_cases"]
    for metric, label in (
      ("crps", "CRPS"),
      ("crpss", "CRPSS"),
      ("nll", "NLL"),
      ("log_score_gain", "Log-score gain"),
    ):
      out[label] = frame[metric]
  return out


def _coverage_figure(
  report: common.Report, frame: pd.DataFrame, models: list[str], title: str
) -> None:
  """Empirical against nominal central-interval coverage, per model."""
  figure, axis = plt.subplots(figsize=(8, 6), layout="constrained")
  axis.plot([0, 1], [0, 1], "k--", label="Ideal")
  nominal = list(NOMINAL_COVERAGE)
  for model in models:
    row = frame.loc[frame["model"] == model]
    if row.empty:
      continue
    axis.plot(
      nominal,
      [float(row[coverage_column(n)].iloc[0]) for n in nominal],
      "o-",
      label=model,
    )
  axis.set(
    xlabel="Nominal central-interval coverage",
    ylabel="Empirical coverage",
    xlim=(0.45, 1.0),
    ylim=(0.0, 1.0),
    title=title,
  )
  axis.legend(fontsize="small")
  report.save(figure)


def _time_figure(
  report: common.Report, cases: pd.DataFrame, models: list[str], title: str
) -> None:
  """Trailing 30-day means of the CRPS and NLL over the test year."""
  figure, axes = plt.subplots(2, 1, figsize=common.PAGE, layout="constrained")
  for model in models:
    part = cases.loc[cases["model"] == model].sort_values("time")
    times = pd.to_datetime(part["time"])
    for axis, metric in zip(axes, ("crps", "nll"), strict=True):
      series = pd.Series(part[metric].to_numpy(), index=times)
      axis.plot(
        times, series.rolling("30D", min_periods=10).mean(), label=model
      )
  for axis, label in zip(axes, ("CRPS", "NLL"), strict=True):
    axis.set_ylabel(f"30-day mean {label}")
    axis.grid(alpha=0.2)
  axes[0].legend(fontsize="small", ncol=2)
  axes[-1].set_xlabel("Valid time; trailing windows of at least 10 cases")
  figure.suptitle(title)
  report.save(figure)


def build(
  run: Path, out: Path, station: int, variables: list[str] | None = None
) -> Path:
  """Write the marginal report of ``run`` to ``out``; return its path."""
  station_mean = common.read_table(run, "margins_station_mean")
  pooled = common.read_table(run, "margins_pooled")
  by_station = common.read_table(run, "margins_by_station")
  timings = common.read_table(run, "margins_timings")
  timings_by_station = common.read_table(run, "margins_timings_by_station")
  models = common.ordered(by_station["model"], lambda m: m == EMOS_MODEL)
  stations = sorted(int(s) for s in by_station["station"].unique())
  if variables is None:
    variables = sorted(by_station["variable"].unique())
  cases = marginal_tables(read_margin_table(run, stations, models, "scores"))[
    "cases"
  ]
  path = out / "marginal_report.pdf"
  with common.Report(
    path, "Weather application: marginal evaluation"
  ) as report:
    report.text(
      "1. Every station: marginal evaluation",
      [
        f"{len(stations)} station(s), {len(models)} marginal models, "
        f"{len(variables)} responses.",
        "Trained on 2017-2020, scored on the held-out 2021 cases.",
        "Equal-station pages: one value per station, with 95% station "
        "bootstrap intervals (exploratory; spatial dependence ignored).",
        "Pooled pages: every station's cases, each weighted equally.",
        "CRPSS = 1 - mean(CRPS) / mean(EMOS CRPS); the log-score gain is "
        "additive. Scores are in each response's own units.",
      ],
    )
    for variable in variables:
      mean = station_mean.loc[station_mean["variable"] == variable]
      report.table(
        _score_table(mean, intervals=True),
        f"{variable}: equal station weights",
        "Brackets: 95% station-bootstrap intervals; N/A with one station.",
      )
      common.box_page(
        report,
        by_station.loc[by_station["variable"] == variable],
        _STATION_METRICS,
        models,
        by="model",
        title=f"{variable}: one value per station",
        note="Boxes: quartiles; whiskers: 1.5 IQR; diamonds: means. Spread, "
        "not confidence intervals.",
      )
      pool = pooled.loc[pooled["variable"] == variable]
      report.table(
        _score_table(pool, intervals=False),
        f"{variable}: every station's cases pooled",
      )
      common.box_page(
        report,
        cases.loc[cases["variable"] == variable],
        _CASE_METRICS,
        models,
        by="model",
        title=f"{variable}: pooled test cases",
        note="Case CRPS gains average to the pooled CRPSS; cases are not "
        "independent.",
      )
      common.pit_page(
        report,
        cases.loc[cases["variable"] == variable],
        models,
        title=f"{variable}: held-out PIT, every station",
      )
    coverage = pooled[
      ["model", "variable", *(coverage_column(n) for n in NOMINAL_COVERAGE)]
    ]
    report.table(coverage, "Central-interval coverage, every station pooled")
    common.timing_pages(
      report,
      timings,
      timings_by_station,
      by="model",
      title="Marginal stage timings, every station",
    )
    if station not in stations:
      report.text(
        f"2. Station {station}",
        [f"Station {station} is not among the evaluated stations."],
      )
      return path
    report.text(
      f"2. Station {station}",
      [
        f"Station {station}'s held-out cases only: casewise variability, not "
        "between-station uncertainty.",
      ],
    )
    one = cases.loc[cases["station"] == station]
    tables = marginal_tables(one)
    for variable in variables:
      report.table(
        _score_table(
          tables["pooled"].loc[tables["pooled"]["variable"] == variable],
          intervals=False,
        ),
        f"Station {station}, {variable}",
      )
      part = one.loc[one["variable"] == variable]
      common.box_page(
        report,
        part,
        _CASE_METRICS,
        models,
        by="model",
        title=f"Station {station}, {variable}: test cases",
        note="Boxes: quartiles; whiskers: 1.5 IQR; diamonds: means.",
      )
      common.pit_page(
        report, part, models, title=f"Station {station}, {variable}: PIT"
      )
      _coverage_figure(
        report,
        tables["pooled"].loc[tables["pooled"]["variable"] == variable],
        models,
        f"Station {station}, {variable}: coverage",
      )
      _time_figure(
        report, part, models, f"Station {station}, {variable}: scores over time"
      )
    common.timing_pages(
      report,
      timings_by_station.loc[timings_by_station["station"] == station]
      .groupby(["model", "stage"], as_index=False)["seconds"]
      .sum(),
      timings_by_station.loc[timings_by_station["station"] == station],
      by="model",
      title=f"Station {station}: timings",
    )
  return path


def main(argv: list[str] | None = None) -> int:
  """Write the marginal report."""
  p = common.parser(__doc__.splitlines()[0])
  p.add_argument("--station", type=int, default=0)
  p.add_argument(
    "--variables", nargs="+", default=None, help="Responses; all by default."
  )
  args = p.parse_args(argv)
  out = args.out or args.run / "reports"
  path = build(args.run, out, args.station, args.variables)
  print(f"wrote {path}")
  return 0


if __name__ == "__main__":
  np.seterr(all="ignore")
  raise SystemExit(main())
