#!/usr/bin/env python3
"""E3e's 2 x 2: NPCC simplified or not, on TLL's selected or the true structure.

Run in the NPCC environment, from the repository root, once E3e's variants
have run (``e3d_npcc_fit.py``, see this folder's README):

    uv run --no-sync python scripts/paper/e3/e3e_report.py \\
        results/e3d_unknown_structure results/e3e_simplification_2x2 \\
        --dims 5 --out results/e3e_simplification_2x2/analysis_d5
    uv run --no-sync python scripts/paper/e3/e3e_report.py \\
        results/e3e_simplification_2x2_gpu053 \\
        --dims 10 --out results/e3e_simplification_2x2_gpu053/analysis_d10

**Inputs.** Each folder's ``npcc_<estimator>-<variant>.csv`` for the variants
``tll-ns``, ``tll-s``, ``true-ns`` and ``true-s``, E3d's rows only. A later
folder's row for a (design, d, seed, variant) replaces an earlier one's, so
E3d's folder goes first where it supplies ``tll-ns`` (at d = 5, which E3e
did not rerun). The folders must come from one machine: at d = 10 the same
fit on two machines differs per seed by up to about 4e-3 in excess NLL on
``decay`` and 0.07 on ``strong``, the size of some of the effects.

**Output.** Per (design, d), the mean excess NLL of each cell of the 2 x 2
with its standard error over seeds, and four paired effects, each with its
SE and the number of seeds where it is negative:

- simplification on each structure: ``tll-s - tll-ns``, ``true-s - true-ns``;
- the structure under each fit: ``tll-ns - true-ns``, ``tll-s - true-s``.

They are printed, and with ``--out`` written as ``e3e_wide.csv`` (one row per
design, d and seed, a column per variant) and ``e3e_effects.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = ("tll-ns", "tll-s", "true-ns", "true-s")
KEY = ["design", "d", "seed"]
EFFECTS = {
  "simplify | tll (tll-s - tll-ns)": ("tll-s", "tll-ns"),
  "simplify | true (true-s - true-ns)": ("true-s", "true-ns"),
  "structure | ns (tll-ns - true-ns)": ("tll-ns", "true-ns"),
  "structure | s (tll-s - true-s)": ("tll-s", "true-s"),
}


def load(folders: list[Path], estimator: str) -> pd.DataFrame:
  """One row per (design, d, seed), one column per variant's excess NLL."""
  frames = []
  for rank, folder in enumerate(folders):
    for v in VARIANTS:
      path = folder / f"npcc_{estimator}-{v}.csv"
      if not path.exists():
        continue
      t = pd.read_csv(path)
      t = t[t["experiment"] == "E3d"].assign(variant=v, rank=rank)
      frames.append(t[[*KEY, "variant", "rank", "excess_nll"]])
  if not frames:
    raise SystemExit(f"no E3e variant CSV in {[str(f) for f in folders]}")
  long = pd.concat(frames, ignore_index=True).sort_values("rank")
  long = long.drop_duplicates([*KEY, "variant"], keep="last")
  wide = long.pivot_table(index=KEY, columns="variant", values="excess_nll")
  return wide.reset_index()


def mean_se(x: pd.Series) -> tuple[float, float, int]:
  """Mean, its standard error and the count, NaN left out."""
  x = x.dropna()
  se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
  return float(x.mean()), float(se), len(x)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("folders", nargs="+", type=Path)
  p.add_argument("--estimator", default="npcc-tabpfn-quantiles-v3-probit")
  p.add_argument("--dims", nargs="+", type=int, default=None)
  p.add_argument("--out", type=Path, default=None)
  a = p.parse_args()
  wide = load(a.folders, a.estimator)
  if a.dims:
    wide = wide[wide["d"].isin(a.dims)]
  rows = []
  for (design, d), g in wide.groupby(["design", "d"]):
    print(f"\n== {design}, d = {d} ==")
    for v in VARIANTS:
      if v in g:
        m, se, n = mean_se(g[v])
        print(f"  {v:8s} {m:.4f} (se {se:.4f}, n = {n})")
    for label, (x, y) in EFFECTS.items():
      if x not in g or y not in g:
        continue
      diff = (g[x] - g[y]).dropna()
      m, se, n = mean_se(diff)
      below = int((diff < 0).sum())
      print(f"  {label:36s} {m:+.4f} (se {se:.4f}, {below}/{n} < 0)")
      rows.append(
        {
          "design": design,
          "d": d,
          "effect": label,
          "mean": m,
          "se": se,
          "below_zero": below,
          "n": n,
        }
      )
  if a.out is not None:
    a.out.mkdir(parents=True, exist_ok=True)
    wide.to_csv(a.out / "e3e_wide.csv", index=False)
    pd.DataFrame(rows).to_csv(a.out / "e3e_effects.csv", index=False)
    print(f"\nwrote {a.out / 'e3e_wide.csv'} and {a.out / 'e3e_effects.csv'}")


if __name__ == "__main__":
  main()
