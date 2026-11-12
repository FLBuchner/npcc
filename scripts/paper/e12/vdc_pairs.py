"""E12: NPCC against VDC and parametric, before and after renormalizing.

An arm's excess NLL against the truth is ``-log m`` lower than that of the
same density renormalized to mass 1, so a paired difference of excess NLL
between two arms of different mass mixes their fit with their mass. Per
comparison ``a - b`` (``a`` TabPFN v3's NPCC arm), design and d, over the
cells both arms hold, this gives three mean differences with their SE over
cells:

- ``reported``: of excess NLL, as E3's tables report it;
- ``npcc_renormalized``: with ``a`` renormalized, ``reported + log m_a``;
- ``both_renormalized``: with both renormalized, also ``- log m_b``.

The comparisons: E3d's ``tll-ns``, E3e's ``true-ns`` and E10's mixture
against VDC on the true structure (E3d's ``vdc_true``); E3d's ``tll-ns``
against the parametric vine; and on E3c, TabPFN against VDC's pairs in the
shared vine and against the parametric vine.

At ``mass_common.NOT_ESTIMABLE`` the renormalized differences are blank in
the CSV (``estimable`` is False) and printed as ``not estimable``.

**Inputs.** ``mass_e3_cells.csv`` and ``mass_e3ac_cells.csv`` in the run
folder (``mass_e3.py`` and ``mass_e3ac.py``).

**Outputs.** ``mass_vdc_pairs.csv`` in ``--out`` (the run folder by
default), and one printed line per comparison, design and d.

**Usage** (from the repository root)::

    uv run --no-sync python scripts/paper/e12/vdc_pairs.py results/e12_<TAG>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from mass_common import TABPFN, blank_not_estimable, refuse_duplicates

KEYS = ["design", "d", "seed"]

#: The columns that derive from the mass estimate.
IS_COLUMNS = [
  "npcc_renormalized",
  "npcc_renormalized_se",
  "both_renormalized",
  "both_renormalized_se",
]


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("run", type=Path, help="the folder of the cells CSVs")
  p.add_argument("--out", type=Path, help="default: the run folder")
  a = p.parse_args()
  out = a.out or a.run
  out.mkdir(parents=True, exist_ok=True)

  df = pd.read_csv(a.run / "mass_e3_cells.csv")
  c = pd.read_csv(a.run / "mass_e3ac_cells.csv")

  def g(cells: pd.DataFrame, study: str, arm: str) -> pd.DataFrame:
    rows = cells[(cells.study == study) & (cells.arm == arm)]
    return rows.set_index(KEYS)

  e10 = pd.concat(
    [
      g(df, "E10", TABPFN).query('not (d==10 and design=="decay")'),
      g(df, "E10_gpu053", TABPFN).query("d==10"),
    ]
  )
  e3e = pd.concat(
    [
      g(df, "E3e", TABPFN + "-true-ns"),
      g(df, "E3e_gpu053", TABPFN + "-true-ns"),
    ]
  )
  pairs = [
    (
      "E3d tll-ns - vdc_true",
      g(df, "E3d", TABPFN + "-tll-ns"),
      g(df, "E3d", "vdc_true"),
    ),
    ("E3e true-ns - vdc_true", e3e, g(df, "E3d", "vdc_true")),
    ("E10 mixture - vdc_true", e10, g(df, "E3d", "vdc_true")),
    (
      "E3d tll-ns - parametric",
      g(df, "E3d", TABPFN + "-tll-ns"),
      g(df, "E3d", "parametric"),
    ),
    ("E3c TabPFN - VDC (shared vine)", g(c, "E3c", TABPFN), g(c, "E3c", "vdc")),
    (
      "E3c TabPFN - parametric",
      g(c, "E3c", TABPFN),
      g(c, "E3c", "parametric"),
    ),
  ]
  rows = []
  for lab, left, right in pairs:
    refuse_duplicates(left, KEYS, f"{lab}: the NPCC arm")
    refuse_duplicates(right, KEYS, f"{lab}: the reference")
    j = left[["excess_nll", "mass"]].join(
      right[["excess_nll", "mass"]], lsuffix="_a", rsuffix="_b", how="inner"
    )
    for (des, d), x in j.groupby(level=[0, 1]):
      raw = x.excess_nll_a - x.excess_nll_b
      own = raw + np.log(x.mass_a)
      both = own - np.log(x.mass_b)
      n = len(x)

      def se(s: pd.Series, n: int = n) -> float:
        return s.std(ddof=1) / np.sqrt(n)

      rows.append(
        {
          "comp": lab,
          "design": des,
          "d": d,
          "n": n,
          "reported": raw.mean(),
          "reported_se": se(raw),
          "npcc_renormalized": own.mean(),
          "npcc_renormalized_se": se(own),
          "both_renormalized": both.mean(),
          "both_renormalized_se": se(both),
        }
      )
  r = blank_not_estimable(pd.DataFrame(rows), IS_COLUMNS)
  r.to_csv(out / "mass_vdc_pairs.csv", index=False)
  for x in r.itertuples():
    head = (
      f"{x.comp:34s} {x.design:6s} d={x.d:2d} n={x.n:2d}  reported "
      f"{x.reported:+.4f} ({x.reported_se:.4f})"
    )
    if x.estimable:
      tail = (
        f"NPCC renormalized {x.npcc_renormalized:+.4f} "
        f"({x.npcc_renormalized_se:.4f})  both renormalized "
        f"{x.both_renormalized:+.4f} ({x.both_renormalized_se:.4f})"
      )
    else:
      tail = "NPCC renormalized not estimable  both renormalized not estimable"
    print(f"{head}  {tail}")


if __name__ == "__main__":
  main()
