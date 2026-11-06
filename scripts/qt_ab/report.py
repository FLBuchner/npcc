"""Report a read-out A/B: arms against a baseline arm, paired by fitted model.

Every arm is compared with the baseline on the same fitted model, so
differences are paired by (estimator, cell, rep). Metrics are E1's, averaged
over the uv grid at each x and then over x: pdf KL and IAE, hfunc1 / hfunc2
IAE and the tau error; plus the held-out KL from true to estimated density.

- ``--arms``: each listed arm minus the baseline, mean ± SE over the pairs and
  the share of pairs it improves, on the held-out reps 5-9 by estimator and n.
- ``--select``: for each backend, every arm ranked by its mean paired pdf KL
  on reps 0-4, and the best one reported on reps 5-9, so the reported gain is
  not the largest of many noisy comparisons.
- ``--tau``: tau-hat at the strongest-dependence x of clayton / linear.

Examples::

    # the tree backends; their E1 read-out was the 7-cell average
    python scripts/qt_ab/report.py --out results/qt_ab/gb \
      --baseline 'm21:ma2|current' --select --arms 'm21:wh0.65|current'

    # TabPFN; its E1 read-out was the 3-cell average
    python scripts/qt_ab/report.py --out results/qt_ab/tabpfn \
      --baseline 'm21:ma1|current' --arms 'm21:wh0.65|current' \
      'm19:wh0.65|current'
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["label", "backend", "transform", "family", "tau_scenario", "n", "rep"]
ARM = ["grid", "arm"]
METRICS = [
  "pdf_KL",
  "pdf_IAE",
  "hfunc1_IAE",
  "hfunc2_IAE",
  "tau_err",
  "holdout_kl",
]


def parse_arm(text: str) -> tuple[str, str]:
  """``"m21:wh0.65|current"`` -> ``("m21", "wh0.65|current")``."""
  grid, arm = text.split(":", 1)
  return grid, arm


def per_rep(d: pd.DataFrame) -> pd.DataFrame:
  """One row per (estimator, cell, rep, arm), every metric averaged over x."""
  met = d[d.kind == "metric"]
  wide = (
    met.groupby([*KEYS, *ARM, "quantity"])[["IAE", "KL"]]
    .mean()
    .unstack("quantity")
  )
  wide.columns = [f"{q}_{m}" for m, q in wide.columns]
  wide = wide[["pdf_KL", "pdf_IAE", "hfunc1_IAE", "hfunc2_IAE"]]
  diag = d[d.kind == "diag"]
  wide["tau_err"] = diag.groupby([*KEYS, *ARM]).tau_abs_err.mean()
  strongest = diag.tau_true >= (
    diag.groupby([*KEYS, *ARM]).tau_true.transform("max") - 1e-9
  )
  wide["tau_hat_max"] = diag[strongest].groupby([*KEYS, *ARM]).tau_hat.mean()
  wide["holdout_kl"] = (
    d[d.kind == "holdout"].groupby([*KEYS, *ARM]).holdout_kl.mean()
  )
  return wide.reset_index()


def paired(w: pd.DataFrame, baseline: tuple[str, str]) -> pd.DataFrame:
  """Each arm minus the baseline arm of the same estimator, cell and rep."""
  base = w[(w.grid == baseline[0]) & (w.arm == baseline[1])].set_index(KEYS)
  diff = w.set_index(KEYS)[[*ARM, *METRICS]].copy()
  diff[METRICS] = diff[METRICS] - base[METRICS].reindex(diff.index)
  return diff.reset_index()


def summary(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
  """``mean ± SE (share of pairs improved)`` for every metric."""
  g = frame.groupby(by)[METRICS]
  mean, se = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
  wins = g.agg(lambda c: float((c < 0).mean()))
  return (
    mean.round(4).astype(str)
    + " ± "
    + se.round(4).astype(str)
    + " ("
    + (100 * wins).round(0).astype(int).astype(str)
    + "%)"
  )


def main() -> None:
  """Print the report."""
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--baseline", type=parse_arm, required=True)
  parser.add_argument("--arms", type=parse_arm, nargs="*", default=[])
  parser.add_argument("--select", action="store_true")
  parser.add_argument("--tau", action="store_true")
  args = parser.parse_args()
  pd.set_option("display.width", 250, "display.max_rows", 500)
  pd.set_option("display.max_columns", 30)

  frames = [pd.read_parquet(p) for p in sorted(args.out.glob("*.parquet"))]
  w = per_rep(pd.concat(frames, ignore_index=True))
  diff = paired(w, args.baseline)
  held_out = diff[diff.rep >= 5]
  base = w[(w.grid == args.baseline[0]) & (w.arm == args.baseline[1])]

  print(f"== Baseline {args.baseline}: absolute metrics, all reps ==")
  print(base.groupby(["label", "n"])[METRICS].mean().round(4).to_string())

  for grid, arm in args.arms:
    r = held_out[(held_out.grid == grid) & (held_out.arm == arm)]
    print(f"\n== {grid} {arm} minus baseline, held-out reps 5-9 ==")
    print(summary(r, ["label", "n"]).to_string())

  if args.select:
    ranked = diff[diff.rep <= 4].groupby(["backend", *ARM])[METRICS].mean()
    for backend, frame in ranked.groupby(level="backend"):
      print(f"\n== {backend}: every arm minus baseline, reps 0-4 ==")
      print(frame.droplevel("backend").sort_values("pdf_KL").round(4))
      grid, arm = frame["pdf_KL"].idxmin()[1:]
      r = held_out[
        (held_out.backend == backend)
        & (held_out.grid == grid)
        & (held_out.arm == arm)
      ]
      print(f"\n-- selected {grid} {arm}, reported on reps 5-9 --")
      print(summary(r, ["transform", "n"]).to_string())

  if args.tau:
    a = w[(w.family == "clayton") & (w.tau_scenario == "linear")]
    table = a.groupby(["label", "n", *ARM]).tau_hat_max.mean().unstack(ARM)
    shown = [args.baseline, *args.arms]
    print("\n== tau-hat at the strongest-dependence x, clayton / linear ==")
    print(table[[c for c in shown if c in table.columns]].round(3).to_string())


if __name__ == "__main__":
  main()
