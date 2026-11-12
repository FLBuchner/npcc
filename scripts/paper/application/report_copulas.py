"""The copula report: each arm against independence on its own margins.

Ports FLBuchner's ``application/report_copulas.py`` (npcc-paper PR #4) onto
a run folder: it reads the tables ``npcc-application evaluate`` wrote and
the stored per-case copula scores, and fits nothing. Comparisons stay within
a marginal model, on identical cases: an arm that failed at any station is
left out of that model's comparisons and listed instead. Writes, under
``<run>/reports`` (or ``--out``):

- ``copula_report.pdf``: per marginal model, the equal-station table with
  station bootstrap intervals and the pooled table, boxplots of the station
  and case values, then the failures and the copula stage's timings;
- ``copula_table.csv``: the paper's table, one row per marginal model and
  arm, pooled means and skills with their equal-station intervals.

The joint NLL gain is independence's mean less the arm's; the energy and
variogram skills are ``1 - mean(arm) / mean(independence)``. Where responses
are derived from the modeled ones, the log score is ``joint_nll_modeled``,
over the modeled responses only, and each response's CRPS from the draws and
are reported too.

Usage (from the repository root)::

    python scripts/paper/application/report_copulas.py --run results/application
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_common as common

from npcc.experiments.application.scores import (
  INDEPENDENCE,
  copula_score_columns,
  copula_tables,
)

Metric = tuple[str, str, bool]


def metrics(
  columns: pd.Index,
) -> tuple[list[Metric], list[Metric], list[Metric]]:
  """The headline, per-response CRPS and per-case metrics present.

  The log score is ``joint_nll``, or ``joint_nll_modeled`` where responses
  are derived and it covers the modeled ones only; the per-response CRPS
  from the draws exists only then.
  """
  frame = pd.DataFrame(columns=columns)
  nll, scores = copula_score_columns(frame)
  label = "Joint NLL" if nll == "joint_nll" else "Joint NLL, modeled responses"
  headline = [
    (nll, label, False),
    ("energy", "Energy score", False),
    ("variogram", "Variogram score", False),
    (f"{nll}_gain", "NLL gain on independence", True),
    ("energy_skill", "Energy skill", True),
    ("variogram_skill", "Variogram skill", True),
  ]
  crps = [(c, f"CRPS, {c.removeprefix('crps_')}", False) for c in scores[3:]]
  crps += [
    (f"{c}_skill", f"CRPS skill, {c.removeprefix('crps_')}", True)
    for c in scores[3:]
  ]
  cases = [
    *headline[:4],
    ("energy_gain", "Energy gain / pooled independence", True),
    ("variogram_gain", "Variogram gain / pooled independence", True),
  ]
  return headline, crps, cases


def stored_cases(run: Path, failures: pd.DataFrame) -> pd.DataFrame:
  """Every stored copula score, less the arms that failed anywhere."""
  frames = [
    pd.read_parquet(p)
    for p in sorted(run.glob("stations/*/copulas/*/*/scores.parquet"))
  ]
  if not frames:
    raise FileNotFoundError(
      f"{run} must hold copula scores; run `copulas` first"
    )
  cases = pd.concat(frames, ignore_index=True)
  failed = set(zip(failures["model"], failures["arm"], strict=True))
  keep = [
    (m, a) not in failed
    for m, a in zip(cases["model"], cases["run_arm"], strict=True)
  ]
  return cases.loc[keep]


def paper_table(
  pooled: pd.DataFrame, station_mean: pd.DataFrame
) -> pd.DataFrame:
  """Pooled means and skills, each with its equal-station interval."""
  merged = pooled.merge(
    station_mean,
    on=["model", "arm"],
    suffixes=("", "_stations"),
    validate="1:1",
  )
  out = merged[["model", "arm", "n_stations", "n_cases"]].copy()
  headline, crps, _ = metrics(pooled.columns)
  for metric, _, _ in [*headline, *crps]:
    out[metric] = merged[metric]
    out[f"{metric}_station_mean"] = merged[f"{metric}_stations"]
    out[f"{metric}_low"] = merged[f"{metric}_low"]
    out[f"{metric}_high"] = merged[f"{metric}_high"]
  return out


def build(run: Path, out: Path) -> list[Path]:
  """Write the copula report of ``run`` into ``out``; return the paths."""
  pooled = common.read_table(run, "copulas_pooled")
  station_mean = common.read_table(run, "copulas_station_mean")
  by_station = common.read_table(run, "copulas_by_station")
  failures = common.read_table(run, "copula_failures")
  stored = stored_cases(run, failures)
  _, scores = copula_score_columns(stored)
  cases = copula_tables(stored[["station", "model", "arm", "row", *scores]])[
    "cases"
  ]
  headline, crps, case_metrics = metrics(pooled.columns)
  out.mkdir(parents=True, exist_ok=True)
  table_path = out / "copula_table.csv"
  paper_table(pooled, station_mean).to_csv(table_path, index=False)
  path = out / "copula_report.pdf"
  with common.Report(path, "Weather application: copula evaluation") as report:
    report.text(
      "Copula evaluation",
      [
        "Fitted on the 2017-2020 training PITs; scored on the held-out 2021 "
        "cases.",
        "Each arm is compared with independence on the same margins and the "
        "same cases.",
        "Energy and variogram scores: draws mapped through each case's "
        "quantiles, responses divided by their training standard deviation.",
        "NLL gain: independence's mean less the arm's. Skills: 1 - mean / "
        "independence's mean.",
        "Brackets: 95% station-bootstrap intervals (exploratory).",
        f"{len(failures)} failed unit(s), listed after the comparisons.",
      ],
    )
    for model in common.ordered(by_station["model"], lambda m: False):
      arms = common.ordered(
        by_station.loc[by_station["model"] == model, "arm"],
        lambda a: a == INDEPENDENCE,
      )
      mean = station_mean.loc[station_mean["model"] == model]
      table = pd.DataFrame({"Arm": mean["arm"], "Stations": mean["n_stations"]})
      for metric, label, _ in headline:
        table[label] = common.with_interval(mean, metric)
      report.table(table, f"{model}: equal station weights")
      common.box_page(
        report,
        by_station.loc[by_station["model"] == model],
        headline,
        arms,
        by="arm",
        title=f"{model}: one value per station",
        note="Boxes: quartiles; whiskers: 1.5 IQR; diamonds: means.",
      )
      pool = pooled.loc[pooled["model"] == model]
      report.table(
        pool[["arm", "n_stations", "n_cases", *(m for m, _, _ in headline)]],
        f"{model}: every station's cases pooled",
      )
      if crps:
        table = pd.DataFrame({"Arm": mean["arm"]})
        for metric, label, _ in crps:
          table[label] = common.with_interval(mean, metric)
        report.table(
          table,
          f"{model}: each response's CRPS from the draws, equal station "
          "weights",
          "Derived responses are computed from each draw of the modeled ones.",
        )
        report.table(
          pool[["arm", *(m for m, _, _ in crps)]],
          f"{model}: each response's CRPS from the draws, pooled",
        )
        common.box_page(
          report,
          by_station.loc[by_station["model"] == model],
          crps,
          arms,
          by="arm",
          title=f"{model}: each response's CRPS, one value per station",
          note="Skills: 1 - mean / independence's mean.",
        )
      common.box_page(
        report,
        cases.loc[cases["model"] == model],
        case_metrics,
        arms,
        by="arm",
        title=f"{model}: pooled test cases",
        note="Case gains over independence's pooled mean average to the "
        "pooled skill.",
      )
    report.table(
      failures,
      "Failed copula units",
      "A failed arm is left out of its marginal model's comparisons at "
      "every station.",
    )
    timing_path = run / "tables" / "copulas_timings.csv"
    if timing_path.exists():
      timings = pd.read_csv(timing_path)
      per_station = pd.concat(
        [
          pd.read_parquet(p)
          for p in sorted(run.glob("stations/*/copulas/*/*/timings.parquet"))
        ],
        ignore_index=True,
      )
      per_station["label"] = per_station["model"] + " | " + per_station["arm"]
      common.timing_pages(
        report,
        timings,
        per_station,
        by="label",
        title="Copula stage timings, every station",
      )
  return [path, table_path]


def main(argv: list[str] | None = None) -> int:
  """Write the copula report."""
  args = common.parser(__doc__.splitlines()[0]).parse_args(argv)
  out = args.out or args.run / "reports"
  for path in build(args.run, out):
    print(f"wrote {path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
