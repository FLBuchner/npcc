"""Summarize E4's coherence CSVs: mean and SE over reps, by estimator and n.

Within a repetition each quantity is averaged over families, regimes and x;
the spread is then taken across repetitions, as in the E1 tables. Writes
``directional.csv``, ``hfunc.csv`` and ``projection.csv`` (and, from a
``--cost-only`` run, ``projection_cost.csv``) next to the inputs.

Usage::

    uv run --no-sync python scripts/paper/e4/summarize.py results/e4_coherence
"""

from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

import pandas as pd


def mean_se(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
  per_rep = df.groupby([*by, "rep"])[cols].mean()
  grouped = per_rep.groupby(level=by)
  mean = grouped.mean().add_suffix("_mean")
  se = (grouped.std(ddof=1) / grouped.count().map(math.sqrt)).add_suffix("_se")
  out = pd.concat([mean, se], axis=1)
  out["n_rep"] = grouped.size()
  return out[sorted(out.columns)].reset_index()


def main() -> None:
  run = Path(sys.argv[1])
  parts = sorted(glob.glob(str(run / "coherence_*.csv")))
  df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
  dirn = df[df["diagnostic"] == "directional"]
  cols = [
    "iae_between_directions",
    "v_given_u_IAE",
    "u_given_v_IAE",
    "average_IAE",
    "v_given_u_KL",
    "u_given_v_KL",
    "average_KL",
  ]
  mean_se(dirn, ["label", "n"], cols).to_csv(
    run / "directional.csv", index=False
  )
  h = df[df["diagnostic"] == "hfunc"]
  mean_se(h, ["label", "n"], ["h1_IAE", "h2_IAE", "h1_max", "h2_max"]).to_csv(
    run / "hfunc.csv", index=False
  )
  proj = df[df["diagnostic"] == "projection"].copy()
  proj["sinkhorn_iters"] = proj["sinkhorn_iters"].astype(str)
  cols = [
    "pdf_IAE",
    "pdf_KL",
    "mass",
    "row_max_abs_err",
    "col_max_abs_err",
    "row_mean_abs_err",
    "col_mean_abs_err",
  ]
  table = mean_se(proj, ["label", "n", "sinkhorn_iters"], cols)
  # One overhead per cell (it covers the whole sweep), so average per cell.
  cells = proj.drop_duplicates(["family", "tau_scenario", "n", "rep", "label"])
  overhead = cells.groupby(["label", "n"])["projection_overhead_s"].mean()
  table = table.merge(overhead.reset_index(), on=["label", "n"], how="left")
  table.to_csv(run / "projection.csv", index=False)
  costs = sorted(glob.glob(str(run / "projection_cost_*.csv")))
  if costs:
    pd.concat((pd.read_csv(p) for p in costs), ignore_index=True).to_csv(
      run / "projection_cost.csv", index=False
    )
  print(f"{len(parts)} part(s), {df['label'].nunique()} estimators -> {run}")


if __name__ == "__main__":
  main()
