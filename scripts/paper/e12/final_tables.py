"""E12's tables: the mass of every arm of E3d, E3e, E10, E3c and E3a.

Each cell first passes through ``mass_common``'s estimability rule: its
mass is kept only if its ESS share is at least ``MIN_ESS_SHARE``, and a row
only if at least ``MIN_PASSING`` of every ``OF_CELLS`` of its cells pass.

**E3d, E3e and E10.** From ``mass_e3_cells.csv`` (``mass_e3.py``), one row
per design, d and arm, with each arm taken from one study folder:

- E3d: every arm of ``results/e3d_unknown_structure``;
- E3e: its three variants, ``tll-s``, ``true-ns`` and ``true-s``, from both
  machines (d = 5 here, d = 10 on gpu053). Its ``tll-ns`` is E3d's arm and
  gpu053's refit of it is left out;
- E10 (arms suffixed `` [mix5]``): the mixtures of this machine's folder,
  except TabPFN at d = 10 on ``decay``, which comes from gpu053's.

Per row: the cells and how many pass (``n``, ``n_pass``); over the passing
cells, the mass's mean, min and max, the median cell SE, the median of
means' mean and ``-log m``'s mean; over every cell, the excess NLL and the
median ESS share.

**E3c.** From ``mass_e3ac_cells.csv`` (``mass_e3ac.py``), one row per
design, d and arm, with the same columns but the ESS share.

**E3a.** From the same file, one row per pair-copula estimator and
``n_train``, over E3a's 45 cells (nine cases, five seeds), with E3c's
columns.

A row that fails the rule has the columns that derive from the mass blank
in the CSVs (``estimable`` is False) and ``not estimable`` in the printed
tables. A cell held twice, or an arm the table does not order, is refused
rather than counted twice or dropped.

**Outputs**, in ``--out`` (the run folder by default):
``final_table_e3d_e3e_e10.csv``, ``final_table_e3c.csv`` and
``final_table_e3a.csv``, each also printed as Markdown, the first per design
and d.

**Usage** (from the repository root)::

    uv run --no-sync python scripts/paper/e12/final_tables.py results/e12_<TAG>
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import pandas as pd
from mass_common import (
  NOT_ESTIMABLE,
  blank_rows,
  mark_cells,
  refuse_duplicates,
  shorten,
)

#: The arms of the first table, in its row order.
ORDER = [
  "TabPFN-tll-ns",
  "TabPFN-random-ns",
  "TabPFN-tll-s",
  "TabPFN-true-ns",
  "TabPFN-true-s",
  "TabPFN [mix5]",
  "CatBoost-tll-ns",
  "CatBoost-random-ns",
  "CatBoost-random-s",
  "CatBoost-true-s",
  "CatBoost [mix5]",
  "parametric",
  "tll",
  "vdc",
  "parametric [mix5]",
  "tll [mix5]",
  "vdc_shared [mix5]",
  "parametric_true",
  "tll_true",
  "vdc_true",
]

#: E3c's consistency checks, left out of its printed table.
E3C_CHECKS = ("rosenblatt-vinecop-check", "tll_cpp")

#: The columns that derive from the mass estimate.
IS_COLUMNS = ["m", "mn", "mx", "se", "mom", "nl"]

#: Every table's columns, as named aggregations of the cells.
COLUMNS = {
  "n": ("mass", "size"),
  "n_pass": ("estimable", "sum"),
  "m": ("mass", "mean"),
  "mn": ("mass", "min"),
  "mx": ("mass", "max"),
  "se": ("mass_se", "median"),
  "mom": ("mass_mom20", "mean"),
  "nl": ("neglogmass", "mean"),
  "ex": ("excess_nll", "mean"),
}


def mass_cells(r: pd.Series, digits: int = 3) -> str:
  """A row's mass columns in Markdown, or ``not estimable`` in each."""
  if not r.estimable:
    return " | ".join([NOT_ESTIMABLE] * 4)
  k = digits
  return (
    f"{r.m:.{k}f} [{r.mn:.{k}f}, {r.mx:.{k}f}] | {r.se:.4f} | {r.mom:.{k}f} "
    f"| {r.nl:+.{k}f}"
  )


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("run", type=Path, help="the folder of the cells CSVs")
  p.add_argument("--out", type=Path, help="default: the run folder")
  a = p.parse_args()
  out = a.out or a.run
  out.mkdir(parents=True, exist_ok=True)
  short = partial(shorten, tabpfn="TabPFN", catboost="CatBoost")

  df = mark_cells(pd.read_csv(a.run / "mass_e3_cells.csv"))
  df["arm_s"] = df.arm.map(short)
  e10 = df[df.study == "E10"]
  e10g = df[(df.study == "E10_gpu053") & (df.arm_s == "TabPFN") & (df.d == 10)]
  here = ~((e10.arm_s == "TabPFN") & (e10.d == 10) & (e10.design == "decay"))
  e10 = pd.concat([e10[here], e10g])
  e10 = e10.assign(arm_s=e10.arm_s + " [mix5]", study="E10")
  e3e = df[df.study.isin(["E3e", "E3e_gpu053"])].assign(study="E3e")
  e3e = e3e[e3e.arm_s != "TabPFN-tll-ns"]
  e3d = df[df.study == "E3d"]
  allr = pd.concat([e3d, e3e, e10])
  refuse_duplicates(
    allr, ["study", "arm_s", "design", "d", "seed"], "The E3d/E3e/E10 table"
  )
  allr["o"] = allr.arm_s.map({arm: i for i, arm in enumerate(ORDER)})
  if allr.o.isna().any():
    raise SystemExit(
      f"every arm must be in ORDER; got {sorted(set(allr.arm_s) - set(ORDER))}"
    )
  g = allr.groupby(["design", "d", "o", "arm_s"])
  t = g.agg(**COLUMNS, ess=("ess_share", "median")).reset_index()
  t = blank_rows(t, IS_COLUMNS)
  t.to_csv(out / "final_table_e3d_e3e_e10.csv", index=False)
  for (des, d), x in t.groupby(["design", "d"]):
    print(f"\n### {des}, d={d}")
    print(
      "| arm | n | pass | mass mean [min, max] | cell SE (median) | "
      "MoM20 mean | -log m (mean) | excess NLL | median ESS share |"
    )
    for _, r in x.iterrows():
      print(
        f"| {r.arm_s} | {r.n} | {r.n_pass} | {mass_cells(r)} | {r.ex:.3f} | "
        f"{r.ess:.2f} |"
      )

  c = mark_cells(pd.read_csv(a.run / "mass_e3ac_cells.csv"))
  cc = c[c.study == "E3c"]
  t2 = cc.groupby(["design", "d", "arm"]).agg(**COLUMNS).reset_index()
  t2 = blank_rows(t2, IS_COLUMNS)
  t2.to_csv(out / "final_table_e3c.csv", index=False)
  print("\n### E3c")
  for _, r in t2.iterrows():
    if r.arm in E3C_CHECKS:
      continue
    print(
      f"| {r.design} | {r.d} | {short(r.arm)} | {r.n} | {r.n_pass} | "
      f"{mass_cells(r)} | {r.ex:.3f} |"
    )

  ca = c[c.study == "E3a"]
  t3 = ca.groupby(["arm", "n_train"]).agg(**COLUMNS).reset_index()
  t3 = blank_rows(t3, IS_COLUMNS)
  t3.to_csv(out / "final_table_e3a.csv", index=False)
  print("\n### E3a")
  for _, r in t3.iterrows():
    print(
      f"| {r.arm} | {r.n_train} | {r.n} | {r.n_pass} | {mass_cells(r, 4)} | "
      f"{r.ex:.4f} |"
    )


if __name__ == "__main__":
  main()
