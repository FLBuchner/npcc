"""Report chosen read-out arms against a baseline arm, paired by cell and rep.

For each estimator (``label``) and sample size, each listed arm's metrics
minus the baseline arm's on the same fitted model: mean ± SE over the
(cell, rep) pairs, and the share of pairs the arm improves. Reported on the
held-out reps 5-9 and, for reference, on all reps.

Example (the in-context backends' current read-out is the 3-cell average)::

    python scripts/qt_ab/summarize_arms.py --out results/qt_ab/tabpfn \
      --baseline m21:ma1|current \
      --arms m21:wh0.65|current m19:ma1|current m19:wh0.65|current \
             m41:wh0.65|current
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from summarize import KEYS, load, per_rep

METRICS = [
  "pdf_KL",
  "pdf_IAE",
  "hfunc1_IAE",
  "hfunc2_IAE",
  "tau_err",
  "holdout_kl",
]


def parse_arm(text: str) -> tuple[str, str]:
  grid, arm = text.split(":", 1)
  return grid, arm


def report(diff: pd.DataFrame, title: str) -> None:
  print(f"\n== {title} ==")
  for (label, grid, arm), frame in diff.groupby(["label", "grid", "arm"]):
    g = frame.groupby("n")[METRICS]
    mean, se = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
    wins = frame.groupby("n")[METRICS].agg(lambda c: float((c < 0).mean()))
    table = (
      mean.round(4).astype(str)
      + " ± "
      + se.round(4).astype(str)
      + " ("
      + (100 * wins).round(0).astype(int).astype(str)
      + "%)"
    )
    print(f"\n-- {label}: {grid} {arm} minus baseline; (share improved) --")
    print(table.to_string())


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--baseline", type=parse_arm, required=True)
  parser.add_argument("--arms", type=parse_arm, nargs="+", required=True)
  args = parser.parse_args()
  pd.set_option("display.width", 250, "display.max_columns", 30)

  w = per_rep(load(args.out))
  base_grid, base_arm = args.baseline
  base = w[(w.grid == base_grid) & (w.arm == base_arm)].set_index(KEYS)
  print("== Baseline absolute metrics, all reps ==")
  print(base.groupby(["label", "n"])[METRICS].mean().round(4).to_string())

  keep = pd.Series(False, index=w.index)
  for grid, arm in args.arms:
    keep |= (w.grid == grid) & (w.arm == arm)
  arms = w[keep].set_index(KEYS)
  diff = arms[["grid", "arm", *METRICS]].copy()
  diff[METRICS] = diff[METRICS] - base[METRICS].reindex(diff.index)
  diff = diff.reset_index()

  report(diff[diff.rep >= 5], "Held-out reps 5-9")
  report(diff, "All reps 0-9")


if __name__ == "__main__":
  main()
