"""Summarize the read-out A/B: paired gains vs the current read-out.

Every arm is compared with the baseline arm A0 (``grid = m21``,
``arm = ma2|current`` -- the gradient-boosting backends' current read-out) on
the same fitted model, so differences are paired by (estimator, cell, rep).

Arms are *selected* on reps 0-4 and their gains *reported* on reps 5-9, so the
reported gain is not the maximum of many noisy comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["label", "backend", "transform", "family", "tau_scenario", "n", "rep"]
ARM = ["grid", "arm"]
BASELINE = ("m21", "ma2|current")
LOWER_IS_BETTER = [
  "pdf_KL",
  "pdf_IAE",
  "hfunc1_IAE",
  "hfunc2_IAE",
  "tau_err",
  "holdout_kl",
]


def load(out: Path) -> pd.DataFrame:
  frames = [pd.read_parquet(p) for p in sorted(out.glob("*.parquet"))]
  return pd.concat(frames, ignore_index=True)


def per_rep(d: pd.DataFrame) -> pd.DataFrame:
  """One row per (estimator, cell, rep, arm): every metric averaged over x."""
  met = d[d.kind == "metric"]
  wide = (
    met.groupby([*KEYS, *ARM, "quantity"])[["IAE", "KL"]]
    .mean()
    .unstack("quantity")
  )
  wide.columns = [f"{q}_{m}" for m, q in wide.columns]
  wide = wide[["pdf_KL", "pdf_IAE", "hfunc1_IAE", "hfunc2_IAE"]]
  diag = d[d.kind == "diag"].groupby([*KEYS, *ARM])
  wide["tau_err"] = diag.tau_abs_err.mean()
  wide["tau_hat_max"] = (
    d[d.kind == "diag"]
    .assign(
      strong=lambda f: (
        f.tau_true >= f.groupby([*KEYS, *ARM]).tau_true.transform("max") - 1e-9
      )
    )
    .query("strong")
    .groupby([*KEYS, *ARM])
    .tau_hat.mean()
  )
  ho = d[d.kind == "holdout"].groupby([*KEYS, *ARM])
  wide["holdout_kl"] = ho.holdout_kl.mean()
  return wide.reset_index()


def paired(w: pd.DataFrame) -> pd.DataFrame:
  """Each arm minus the baseline arm of the same estimator, cell and rep."""
  base = w[(w.grid == BASELINE[0]) & (w.arm == BASELINE[1])].set_index(KEYS)
  cols = [*LOWER_IS_BETTER, "tau_hat_max"]
  diff = w.set_index(KEYS)[[*ARM, *cols]].copy()
  diff[cols] = diff[cols] - base[cols].reindex(diff.index)
  return diff.reset_index()


def mean_se(
  frame: pd.DataFrame, by: list[str], cols: list[str]
) -> pd.DataFrame:
  g = frame.groupby(by)[cols]
  mean, se = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
  return mean.round(4).astype(str) + " ± " + se.round(4).astype(str)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  d = load(args.out)
  w = per_rep(d)
  diff = paired(w)
  pd.set_option(
    "display.width", 250, "display.max_rows", 400, "display.max_columns", 30
  )

  print("== Baseline (A0) absolute metrics, all reps ==")
  base = w[(w.grid == BASELINE[0]) & (w.arm == BASELINE[1])]
  print(
    base.groupby(["label", "n"])[LOWER_IS_BETTER].mean().round(4).to_string()
  )

  sel = diff[diff.rep <= 4]
  rep = diff[diff.rep >= 5]
  print(
    "\n== Selection (reps 0-4): mean paired difference vs A0, pooled over cells =="
  )
  ranked = sel.groupby(["backend", *ARM])[LOWER_IS_BETTER].mean()
  for backend, frame in ranked.groupby(level="backend"):
    print(f"\n-- {backend} (sorted by pdf_KL) --")
    print(frame.droplevel("backend").sort_values("pdf_KL").round(4).to_string())

  best = ranked["pdf_KL"].groupby(level="backend").idxmin()
  print("\n== Selected arm per backend (lowest mean pdf_KL on reps 0-4) ==")
  print(best.to_string())

  print("\n== Report (reps 5-9): selected arm minus A0, mean ± SE over reps ==")
  for backend, (_, grid, arm) in best.items():
    r = rep[(rep.backend == backend) & (rep.grid == grid) & (rep.arm == arm)]
    print(f"\n-- {backend}: {grid} {arm} --")
    print(mean_se(r, ["transform", "n"], LOWER_IS_BETTER).to_string())
    wins = (r[LOWER_IS_BETTER] < 0).mean().round(3)
    print("share of (cell, rep) improved:", wins.to_dict())

  print("\n== Fixed arms of interest (reps 5-9), pooled over transforms ==")
  for grid, arm in [
    ("m21", "ma1|current"),
    ("m21", "ma0|current"),
    ("m21", "wh0.65|current"),
    ("m21", "ma2|modeled"),
    ("m19", "ma2|current"),
    ("m41", "ma2|current"),
  ]:
    r = rep[(rep.grid == grid) & (rep.arm == arm)]
    if r.empty:
      continue
    print(f"\n-- {grid} {arm} minus A0 --")
    print(mean_se(r, ["backend", "n"], LOWER_IS_BETTER).to_string())

  print(
    "\n== Symptom (a): tau_hat at the strongest-dependence x, clayton/linear =="
  )
  a = w[(w.family == "clayton") & (w.tau_scenario == "linear")]
  print(
    a.groupby(["label", "n", *ARM])
    .tau_hat_max.mean()
    .unstack(ARM)[
      [
        BASELINE,
        ("m21", "ma1|current"),
        ("m21", "ma0|current"),
        ("m21", "wh0.65|current"),
      ]
    ]
    .round(3)
    .to_string()
  )

  print("\n== Symptom (b): pdf KL at n <= 500, by arm ==")
  b = w[w.n <= 500]
  print(
    b.groupby(["label", "n", *ARM])
    .pdf_KL.mean()
    .unstack(ARM)[
      [
        BASELINE,
        ("m21", "ma1|current"),
        ("m21", "ma0|current"),
        ("m21", "wh0.65|current"),
      ]
    ]
    .round(3)
    .to_string()
  )

  t = d[d.kind == "fit"].groupby(["backend", "n"])[["fit_s", "record_s"]].mean()
  print("\n== Seconds per fit and per recording pass ==")
  print(t.round(2).to_string())


if __name__ == "__main__":
  main()
