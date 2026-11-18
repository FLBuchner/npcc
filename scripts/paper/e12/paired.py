"""E12: paired differences of ``-log m`` and excess NLL between E3 arms.

Two arms fitted on the same cells (design, d, seed) are paired cell by
cell, from ``mass_e3_cells.csv`` (``mass_e3.py``), after ``mass_common``'s
estimability rule has marked each cell. Per comparison ``a - b`` and per
design and d:

- ``n``, ``d_excess`` and ``se_excess``: over every cell both arms hold,
  the mean difference of excess NLL and its SE over cells;
- ``n_pass`` and, over the cells where both arms' cells pass,
  ``d_excess_pass`` (the same difference), ``d_neglogm`` and ``se_neglogm``
  (the same for ``-log m``, with its min and max over cells) and
  ``d_excess_corrected``, the mean of ``d_excess - d_neglogm``: the excess
  NLL difference once both arms are renormalized to mass 1.

The comparisons: within E3d (TabPFN's random against its TLL-selected
structure, TabPFN's ``tll-ns`` against each competitor, CatBoost's
``tll-ns`` against TLL, CatBoost's random structure non-simplified against
simplified); E3e's 2 x 2, whose ``tll-ns`` is E3d's at d = 5 and gpu053's
refit at d = 10; and E10's TabPFN mixture (this machine's folder, and
gpu053's at d = 10) against the ``tll-ns`` arm its member 1 refits, each
competitor's mixture and ``vdc_true``, and CatBoost's mixture against its
``tll-ns``.

A comparison is reported only where both arms' rows pass the rule (each
arm's cells at that design and d); elsewhere the columns over passing cells
are blank (``estimable`` is False). ``d_excess`` stays, since it does not
use the mass.

**Outputs.** ``mass_paired.csv`` in ``--out`` (the run folder by default),
and the same table printed.

**Usage** (from the repository root)::

    uv run --no-sync python scripts/paper/e12/paired.py results/e12_<TAG>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from mass_common import (
  arm_rows_pass,
  mark_cells,
  printable,
  refuse_duplicates,
  shorten,
)

#: The columns that derive from the mass estimate.
IS_COLUMNS = [
  "d_excess_pass",
  "d_neglogm",
  "se_neglogm",
  "d_neglogm_min",
  "d_neglogm_max",
  "d_excess_corrected",
]

KEYS = ["design", "d", "seed"]


def pair(a: pd.DataFrame, b: pd.DataFrame, label: str) -> list[dict]:
  """The paired differences ``a - b`` per design and d, as rows.

  Parameters
  ----------
  a, b : pd.DataFrame
      Cell rows indexed by design, d and seed.
  label : str
      The comparison's name.

  Returns
  -------
  list of dict
      One row per design and d both arms hold.
  """
  refuse_duplicates(a, KEYS, f"{label}: the first arm")
  refuse_duplicates(b, KEYS, f"{label}: the second arm")
  ok_a, ok_b = arm_rows_pass(a), arm_rows_pass(b)
  cols = ["neglogmass", "excess_nll", "estimable"]
  j = a[cols].join(b[cols], lsuffix="_a", rsuffix="_b", how="inner")
  out = []
  for (des, d), g in j.groupby(level=[0, 1]):
    de_all = g.excess_nll_a - g.excess_nll_b
    both = g[g.estimable_a & g.estimable_b]
    dn = both.neglogmass_a - both.neglogmass_b
    de = both.excess_nll_a - both.excess_nll_b
    out.append(
      {
        "comp": label,
        "design": des,
        "d": d,
        "n": len(g),
        "d_excess": de_all.mean(),
        "se_excess": de_all.std(ddof=1) / np.sqrt(len(g)),
        "n_pass": len(both),
        "d_excess_pass": de.mean(),
        "d_neglogm": dn.mean(),
        "se_neglogm": dn.std(ddof=1) / np.sqrt(len(both)),
        "d_neglogm_min": dn.min(),
        "d_neglogm_max": dn.max(),
        "d_excess_corrected": (de - dn).mean(),
        "estimable": ok_a[(des, int(d))] and ok_b[(des, int(d))],
      }
    )
  return out


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("run", type=Path, help="the folder of mass_e3_cells.csv")
  p.add_argument("--out", type=Path, help="default: the run folder")
  a = p.parse_args()
  out = a.out or a.run
  out.mkdir(parents=True, exist_ok=True)

  df = mark_cells(pd.read_csv(a.run / "mass_e3_cells.csv"))
  df["arm_s"] = df.arm.map(shorten)

  def get(study: str, arm: str) -> pd.DataFrame:
    return df[(df.study == study) & (df.arm_s == arm)].set_index(KEYS)

  def e3d(arm: str) -> pd.DataFrame:
    return get("E3d", arm)

  def e3e(arm: str) -> pd.DataFrame:
    return pd.concat([get("E3e", arm), get("E3e_gpu053", arm)])

  def e10(arm: str) -> pd.DataFrame:
    if arm == "tabpfn":
      return pd.concat([get("E10", arm), get("E10_gpu053", arm).query("d==10")])
    return get("E10", arm)

  rows = []
  rows += pair(
    e3d("tabpfn-random-ns"), e3d("tabpfn-tll-ns"), "tabpfn random-ns - tll-ns"
  )
  for c in ["parametric", "vdc", "tll", "vdc_true"]:
    rows += pair(e3d("tabpfn-tll-ns"), e3d(c), f"tabpfn tll-ns - {c}")
  rows += pair(e3d("catboost-tll-ns"), e3d("tll"), "catboost tll-ns - tll")
  rows += pair(
    e3d("catboost-random-ns"),
    e3d("catboost-random-s"),
    "catboost random-ns - random-s",
  )
  # E3e's 2 x 2: tll-ns is E3d's arm at d = 5 and gpu053's refit at d = 10.
  tllns = pd.concat(
    [
      e3d("tabpfn-tll-ns").query("d==5"),
      get("E3e_gpu053", "tabpfn-tll-ns"),
    ]
  )
  rows += pair(e3e("tabpfn-tll-s"), tllns, "tabpfn tll-s - tll-ns")
  rows += pair(
    e3e("tabpfn-true-s"), e3e("tabpfn-true-ns"), "tabpfn true-s - true-ns"
  )
  rows += pair(tllns, e3e("tabpfn-true-ns"), "tabpfn tll-ns - true-ns")
  rows += pair(
    e3e("tabpfn-true-ns"), e3d("vdc_true"), "tabpfn true-ns - vdc_true"
  )
  # E10's mixtures.
  rows += pair(e10("tabpfn"), tllns, "tabpfn mixture - tll-ns (member 1)")
  for c in ["tll", "parametric", "vdc_shared"]:
    rows += pair(e10("tabpfn"), e10(c), f"tabpfn mixture - {c} mixture")
  rows += pair(e10("tabpfn"), e3d("vdc_true"), "tabpfn mixture - vdc_true")
  rows += pair(
    e10("catboost"), e3d("catboost-tll-ns"), "catboost mixture - tll-ns"
  )
  r = pd.DataFrame(rows)
  r.loc[~r.estimable, IS_COLUMNS] = np.nan
  r.to_csv(out / "mass_paired.csv", index=False)
  pd.set_option("display.width", 250)
  pd.set_option("display.max_rows", 300)
  print(printable(r, IS_COLUMNS).to_string())


if __name__ == "__main__":
  main()
