"""E12: the mass of E3d's, E3e's and E10's vines, per study, design, d, arm.

Reads ``mass_e3_cells.csv`` (``mass_e3.py``), applies the estimability rule
of ``mass_common`` to each cell, and averages over seeds, per study folder,
design, d and arm (TabPFN v3 and CatBoost shortened to ``tabpfn`` and
``catboost``): the cells and how many pass (``n``, ``n_pass``); over the
passing cells, the mass's mean, min and max, the median cell SE, ``-log
m``'s mean, min and max, the median of means' and the trimmed mean's means,
and ``excess_plus_logm``, the mean of ``excess_nll + log m``, the excess
NLL of the arm renormalized to mass 1; over every cell, the smallest ESS
share, the largest single ratio's share, the excess NLL and the non-finite
points.

A row that fails the rule has the columns that derive from the mass blank
(``estimable`` is False); the ESS and ratio shares stay, as the evidence.

**Outputs**, in ``--out`` (the run folder by default):
``mass_e3_summary.csv``, also printed; and the per-cell tables with the
rule applied, ``mass_e3_cells_reported.csv`` and, when the run holds
``mass_e3ac_cells.csv``, ``mass_e3ac_cells_reported.csv``: each cell's row
with ``estimable`` added and the estimate blank where it fails.

**Usage** (from the repository root)::

    uv run --no-sync python scripts/paper/e12/summarize.py results/e12_<TAG>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from mass_common import blank_rows, mark_cells, printable, shorten

#: The columns that derive from the mass estimate.
IS_COLUMNS = [
  "mass_mean",
  "mass_min",
  "mass_max",
  "se_med",
  "neglogm_mean",
  "neglogm_min",
  "neglogm_max",
  "mom_mean",
  "trim_mean",
  "excess_plus_logm",
]


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("run", type=Path, help="the folder of mass_e3_cells.csv")
  p.add_argument("--out", type=Path, help="default: the run folder")
  a = p.parse_args()
  out = a.out or a.run
  out.mkdir(parents=True, exist_ok=True)

  pd.set_option("display.width", 250)
  pd.set_option("display.max_rows", 400)
  pd.set_option("display.max_columns", 30)
  df = mark_cells(pd.read_csv(a.run / "mass_e3_cells.csv"))
  df.to_csv(out / "mass_e3_cells_reported.csv", index=False)
  e3ac = a.run / "mass_e3ac_cells.csv"
  if e3ac.is_file():
    reported = mark_cells(pd.read_csv(e3ac))
    reported.to_csv(out / "mass_e3ac_cells_reported.csv", index=False)
  df["arm_s"] = df.arm.map(shorten)
  g = df.groupby(["study", "design", "d", "arm_s"])
  s = g.agg(
    n=("mass", "size"),
    n_pass=("estimable", "sum"),
    mass_mean=("mass", "mean"),
    mass_min=("mass", "min"),
    mass_max=("mass", "max"),
    se_med=("mass_se", "median"),
    neglogm_mean=("neglogmass", "mean"),
    neglogm_min=("neglogmass", "min"),
    neglogm_max=("neglogmass", "max"),
    mom_mean=("mass_mom20", "mean"),
    trim_mean=("mass_trim_top0.1pct", "mean"),
    ess_min=("ess_share", "min"),
    top1_max=("top1_share", "max"),
    excess=("excess_nll", "mean"),
    nonfin=("n_nonfinite", "sum"),
  ).reset_index()
  s["excess_plus_logm"] = g.apply(
    lambda x: (x.excess_nll + np.log(x.mass)).mean()
  ).values
  s = blank_rows(s, IS_COLUMNS)
  s.to_csv(out / "mass_e3_summary.csv", index=False)
  print(printable(s, IS_COLUMNS).to_string())


if __name__ == "__main__":
  main()
