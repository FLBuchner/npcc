"""Per-cell model selection of the R competitors, joined with their accuracy.

The external R scripts log, per cell, the family each method selected, its
model formula and the smooth's effective degrees of freedom (EDF); the result
tables carry only the metrics. This joins the two into
``<run>/analysis/r_selection.csv``: one row per estimator and cell, with the
chosen family, whether it contains the truth, the EDF, and the mean pdf KL and
IAE over x. The KL is the Monte Carlo KL, through ``analyze.monte_carlo_kl``,
which refuses a run without it.

Usage::

    uv run --no-sync python scripts/paper/e1/r_selection.py \
      results/study_2026-11-05_r
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import pandas as pd
from analyze import monte_carlo_kl

_LINE = re.compile(
  r"(?P<label>[\w-]+) on (?P<family>[\w-]+?)__(?P<tau_scenario>[\w-]+?)"
  r"__n(?P<n>\d+)__rep(?P<rep>\d+): .*?family (?P<chosen>\d+), "
  r"(?P<formula>.*?), EDF (?P<edf>[\d.]+)"
)

#: VineCopula family codes that contain each simulated family.
_CONTAINS = {"clayton": {3, 13, 23, 33, 301, 302, 303, 304}, "gaussian": {1}}


def main() -> None:
  run = Path(sys.argv[1])
  rows = []
  for log in sorted(glob.glob(str(run / "logs" / "shard*.log"))):
    for line in Path(log).read_text().splitlines():
      m = _LINE.search(line)
      if m:
        rows.append(m.groupdict())
  if not rows:
    raise SystemExit(f"no model-selection lines in {run}/logs")
  sel = pd.DataFrame(rows).astype(
    {"n": int, "rep": int, "chosen": int, "edf": float}
  )
  # A cell rerun after an interruption logs twice; the last line is its fit.
  keys = ["label", "family", "tau_scenario", "n", "rep"]
  sel = sel.drop_duplicates(keys, keep="last")
  sel["contains_truth"] = [
    c in _CONTAINS.get(f, set())
    for f, c in zip(sel["family"], sel["chosen"], strict=True)
  ]

  metrics = pd.concat(
    pd.read_parquet(p)
    for p in glob.glob(str(run / "cells" / "*" / "metrics.parquet"))
  )
  metrics = monte_carlo_kl(metrics)
  pdf = metrics[metrics["quantity"] == "pdf"]
  acc = pdf.groupby(keys)[["KL", "IAE"]].mean().reset_index()
  out = sel.merge(acc, on=keys, how="left").sort_values(keys)
  (run / "analysis").mkdir(exist_ok=True)
  out.to_csv(run / "analysis" / "r_selection.csv", index=False)
  print(f"{len(out)} fits -> {run / 'analysis' / 'r_selection.csv'}")


if __name__ == "__main__":
  main()
