"""E12's tables: the mass of every arm of E3d, E3e and E10, and of E3c.

**E3d, E3e and E10.** From ``mass_e3_cells.csv`` (``mass_e3.py``), one row
per design, d and arm, with each arm taken from one study folder:

- E3d: every arm of ``results/e3d_unknown_structure``;
- E3e: its three variants, ``tll-s``, ``true-ns`` and ``true-s``, from both
  machines (d = 5 here, d = 10 on gpu053). Its ``tll-ns`` is E3d's arm and
  gpu053's refit of it is left out;
- E10 (arms suffixed `` [mix5]``): the mixtures of this machine's folder,
  except TabPFN at d = 10 on ``decay``, which comes from gpu053's.

Per row: the cells, the mass's mean, min and max, the median cell SE, the
median of means' mean, ``-log m``'s mean, the excess NLL and the median ESS
share.

**E3c.** From ``mass_e3ac_cells.csv`` (``mass_e3ac.py``), one row per
design, d and arm, with the same columns but the ESS share.

At ``mass_common.NOT_ESTIMABLE`` the columns that derive from the mass are
blank in the CSVs (``estimable`` is False) and read ``not estimable`` in the
printed tables. A cell held twice, or an arm the table does not order, is
refused rather than counted twice or dropped.

**Outputs**, in ``--out`` (the run folder by default):
``final_table_e3d_e3e_e10.csv`` and ``final_table_e3c.csv``; both tables
are also printed as Markdown, the first per design and d.

**Usage** (from the repository root)::

    uv run --no-sync python scripts/paper/e12/final_tables.py results/e12_<TAG>
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import pandas as pd
from mass_common import (
  blank_not_estimable,
  estimable,
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

NOT_ESTIMABLE = "not estimable"


def mass_cells(r: pd.Series) -> str:
  """A row's mass columns in Markdown, or ``not estimable`` in each."""
  if not estimable(r.design, r.d):
    return " | ".join([NOT_ESTIMABLE] * 4)
  return (
    f"{r.m:.3f} [{r.mn:.3f}, {r.mx:.3f}] | {r.se:.4f} | {r.mom:.3f} | "
    f"{r.nl:+.3f}"
  )


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("run", type=Path, help="the folder of the cells CSVs")
  p.add_argument("--out", type=Path, help="default: the run folder")
  a = p.parse_args()
  out = a.out or a.run
  out.mkdir(parents=True, exist_ok=True)
  short = partial(shorten, tabpfn="TabPFN", catboost="CatBoost")

  df = pd.read_csv(a.run / "mass_e3_cells.csv")
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
  t = g.agg(
    n=("mass", "size"),
    m=("mass", "mean"),
    mn=("mass", "min"),
    mx=("mass", "max"),
    se=("mass_se", "median"),
    mom=("mass_mom20", "mean"),
    nl=("neglogmass", "mean"),
    ex=("excess_nll", "mean"),
    ess=("ess_share", "median"),
  ).reset_index()
  t = blank_not_estimable(t, IS_COLUMNS)
  t.to_csv(out / "final_table_e3d_e3e_e10.csv", index=False)
  for (des, d), x in t.groupby(["design", "d"]):
    print(f"\n### {des}, d={d}")
    print(
      "| arm | n | mass mean [min, max] | cell SE (median) | MoM20 mean | "
      "-log m (mean) | excess NLL | median ESS share |"
    )
    for _, r in x.iterrows():
      print(
        f"| {r.arm_s} | {r.n} | {mass_cells(r)} | {r.ex:.3f} | {r.ess:.2f} |"
      )

  c = pd.read_csv(a.run / "mass_e3ac_cells.csv")
  cc = c[c.study == "E3c"]
  t2 = (
    cc.groupby(["design", "d", "arm"])
    .agg(
      n=("mass", "size"),
      m=("mass", "mean"),
      mn=("mass", "min"),
      mx=("mass", "max"),
      se=("mass_se", "median"),
      mom=("mass_mom20", "mean"),
      nl=("neglogmass", "mean"),
      ex=("excess_nll", "mean"),
    )
    .reset_index()
  )
  t2 = blank_not_estimable(t2, IS_COLUMNS)
  t2.to_csv(out / "final_table_e3c.csv", index=False)
  print("\n### E3c")
  for _, r in t2.iterrows():
    if r.arm in E3C_CHECKS:
      continue
    print(
      f"| {r.design} | {r.d} | {short(r.arm)} | {r.n} | {mass_cells(r)} | "
      f"{r.ex:.3f} |"
    )


if __name__ == "__main__":
  main()
