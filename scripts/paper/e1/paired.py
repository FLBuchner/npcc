"""Paired differences between estimators on the same cells, with win counts.

Every run on one grid fits the same data in each (family, regime, n, rep)
cell, so two estimators' x-averaged metrics difference cell by cell. Within a
repetition the difference is then averaged over everything not in the
breakdown -- families, regimes and n overall; regimes and n by family;
families and regimes by n -- and its mean, SE and wins (the repetitions in
which the first estimator's metric is the lower) are taken across
repetitions, as in the E1 tables. ``best-gb`` stands for the gradient booster
with the lowest overall pdf KL.

Writes ``<run>/analysis/paired.csv``, one row per (estimator, competitor,
metric, breakdown, level).

Usage (from the npcc repository root)::

    uv run --no-sync python scripts/paper/e1/paired.py \
      results/study_misspec_2026-11-06 \
      --with results/study_misspec_2026-11-06_r \
      --labels tabpfn-quantiles-v3-logit tabicl-logit best-gb \
      --against gamcopula boostcopula-smooth-cv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

# The E1 analysis beside this file: its loader reads the cells the same way.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load

METRICS = ["KL", "IAE"]
CELL = ["family", "tau_scenario", "n", "rep"]


def best_gb(per_cell: pd.DataFrame) -> str:
  """The gradient booster with the lowest overall pdf KL."""
  gb = per_cell[per_cell["group"] == "GB"]
  if gb.empty:
    raise SystemExit("best-gb needs a gradient booster in the runs; got none")
  return str(gb.groupby("label")["KL"].mean().idxmin())


def paired(
  per_cell: pd.DataFrame, label: str, other: str, *, common: bool = False
) -> pd.DataFrame:
  """``label`` minus ``other``, per breakdown and level, over repetitions.

  Both must cover the same cells, unless ``common`` restricts them to the
  cells they share -- an ablation run on a subset of a study's grid, say.
  """
  a = per_cell[per_cell["label"] == label].set_index(CELL)[METRICS]
  b = per_cell[per_cell["label"] == other].set_index(CELL)[METRICS]
  if common:
    shared = a.index.intersection(b.index)
    a, b = a.loc[shared], b.loc[shared]
  if a.empty or b.empty:
    raise SystemExit(f"{label} and {other} must share cells; got none")
  if not a.index.sort_values().equals(b.index.sort_values()):
    missing = a.index.symmetric_difference(b.index)
    raise SystemExit(
      f"{label} and {other} must cover the same cells; got "
      f"{len(missing)} in only one (pass --common-cells to use the shared)"
    )
  diff = (a - b.loc[a.index]).reset_index()
  rows = []
  for breakdown in (None, "family", "n"):
    by = [] if breakdown is None else [breakdown]
    per_rep = diff.groupby([*by, "rep"])[METRICS].mean()
    groups = (
      [("all", per_rep)]
      if breakdown is None
      else list(per_rep.groupby(level=breakdown))
    )
    for level, values in groups:
      for metric in METRICS:
        v = values[metric]
        rows.append(
          {
            "label": label,
            "against": other,
            "metric": metric,
            "breakdown": breakdown or "overall",
            "level": level,
            "mean": float(v.mean()),
            "se": float(v.std(ddof=1) / math.sqrt(len(v))),
            "wins": int((v < 0).sum()),
            "n_rep": len(v),
            "n_cells": len(a),
          }
        )
  return pd.DataFrame(rows)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("run", type=Path, help="The study's output folder.")
  p.add_argument(
    "--with",
    dest="extra",
    type=Path,
    nargs="*",
    default=[],
    help="Further runs on the same grid, e.g. the R competitors.",
  )
  p.add_argument("--labels", nargs="+", required=True)
  p.add_argument("--against", nargs="+", required=True)
  p.add_argument(
    "--common-cells",
    action="store_true",
    help="Compare on the cells both estimators share, not all of each.",
  )
  p.add_argument(
    "--out", type=Path, default=None, help="Default: <run>/analysis."
  )
  a = p.parse_args()

  metrics = load([a.run, *a.extra])["metrics"]
  pdf = metrics[metrics["quantity"] == "pdf"]
  per_cell = (
    pdf.groupby(["label", "group", *CELL])[METRICS].mean().reset_index()
  )
  labels = [best_gb(per_cell) if lab == "best-gb" else lab for lab in a.labels]
  table = pd.concat(
    [
      paired(per_cell, lab, other, common=a.common_cells)
      for lab in labels
      for other in a.against
    ],
    ignore_index=True,
  )
  out = a.out or a.run / "analysis"
  out.mkdir(parents=True, exist_ok=True)
  table.to_csv(out / "paired.csv", index=False)
  show = table[table["metric"] == "KL"]
  print(show.round(4).to_string(index=False))
  print(f"{len(table)} rows -> {out / 'paired.csv'}")


if __name__ == "__main__":
  main()
