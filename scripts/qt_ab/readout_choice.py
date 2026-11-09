"""The read-out choice table: each arm against E1's read-out, held-out reps.

One row per backend x n x arm, on the held-out reps 5-9. Each arm is paired
with A0, the read-out E1 ran for that backend (the 7-cell moving average for
the tree backends, the 3-cell one for the others), on the same fitted model:

- ``base_pdf_KL``: A0's pdf KL;
- ``d_<metric>`` / ``se_<metric>``: the arm minus A0 in pdf KL, pdf IAE and
  held-out KL, mean and standard error over the pairs, one per cell x rep.
  A backend run under several transforms (the tree backends, logit and
  probit) fits them on the same sample, so its differences are averaged over
  the transforms within each cell x rep first: counting them as separate
  pairs would understate the SE;
- ``share_improved``: the share of pairs whose pdf KL the arm lowers;
- ``commit``: the commits the runs were produced from.

With ``--reference``, the sign of every mean difference is compared with that
table's, and the rows whose sign flips are listed.

Example::

    uv run --no-sync python scripts/qt_ab/readout_choice.py \
      --runs results/qt_ab_2026-11/{gb,tabpfn,icl} \
      --csv results/qt_ab_2026-11/readout_choice.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from report import paired, per_rep

TREES = ("lightgbm", "xgboost", "catboost")
#: The read-out each backend ran in E1, the baseline of every comparison.
A0 = {"trees": ("m21", "ma2|current"), "other": ("m21", "ma1|current")}
#: The compared arms: the adopted read-out on each grid, and for the tree
#: backends E1's read-out with modeled-scale tails.
ARMS = [
  ("m21", "wh0.65|current"),
  ("m19", "wh0.65|current"),
  ("m41", "wh0.65|current"),
  ("m21", "ma2|modeled"),
]
METRICS = ["pdf_KL", "pdf_IAE", "holdout_kl"]
#: One sample: what a pair is, once the transforms fitted on it are averaged.
CELL_REP = ["family", "tau_scenario", "rep"]


def table(runs: list[Path]) -> pd.DataFrame:
  """The choice table from the parquet folders ``runs``."""
  frames = [pd.read_parquet(p) for run in runs for p in run.glob("*.parquet")]
  d = pd.concat(frames, ignore_index=True)
  # Outputs written before ab.py stamped its rows carry no commit.
  commits = ";".join(sorted(d.commit.unique())) if "commit" in d else ""
  if "dirty" in d and d.dirty.any():
    raise SystemExit("some rows come from a tree that differed from its commit")
  w = per_rep(d)
  rows = []
  for backend, wb in w.groupby("backend"):
    trees = backend in TREES
    base = A0["trees" if trees else "other"]
    diff = paired(wb, base)
    held = diff[diff.rep >= 5]
    a0 = wb[(wb.grid == base[0]) & (wb.arm == base[1]) & (wb.rep >= 5)]
    for grid, arm in ARMS:
      if arm.endswith("|modeled") and not trees:
        continue
      for n, rt in held[(held.grid == grid) & (held.arm == arm)].groupby("n"):
        r = rt.groupby(CELL_REP)[METRICS].mean()
        row: dict[str, object] = {
          "backend": backend,
          "transforms": "+".join(sorted(rt["transform"].unique())),
          "n": n,
          "grid": grid,
          "arm": arm,
          "a0": f"{base[0]}:{base[1]}",
          "pairs": len(r),
          "base_pdf_KL": a0[a0.n == n].pdf_KL.mean(),
        }
        for m in METRICS:
          row[f"d_{m}"] = r[m].mean()
          row[f"se_{m}"] = r[m].std(ddof=1) / np.sqrt(r[m].count())
        row["share_improved"] = float((r.pdf_KL < 0).mean())
        row["commit"] = commits
        rows.append(row)
  return pd.DataFrame(rows)


def sign_flips(new: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
  """Each (row, metric) whose mean difference changes sign from ``ref``.

  A flip of a reference mean within an SE or two of zero is noise; the
  reference's ``z = d / se`` is listed so it can be told apart.
  """
  keys = ["backend", "n", "grid", "arm"]
  both = new.merge(ref, on=keys, suffixes=("", "_ref"))
  flips = []
  for m in METRICS:
    d, d_ref, se_ref = both[f"d_{m}"], both[f"d_{m}_ref"], both[f"se_{m}_ref"]
    hit = np.sign(d) != np.sign(d_ref)
    flips.append(
      both.loc[hit, keys].assign(
        metric=m,
        d=d[hit].to_numpy(),
        d_ref=d_ref[hit].to_numpy(),
        z_ref=(d_ref / se_ref)[hit].to_numpy(),
      )
    )
  return pd.concat(flips, ignore_index=True)


def main() -> None:
  """Write the table, and compare it with a reference if one is given."""
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--runs", type=Path, nargs="+", required=True)
  parser.add_argument("--csv", type=Path, required=True)
  parser.add_argument("--reference", type=Path, default=None)
  args = parser.parse_args()
  pd.set_option("display.width", 250, "display.max_rows", 500)
  pd.set_option("display.max_columns", 30)

  out = table(args.runs)
  args.csv.parent.mkdir(parents=True, exist_ok=True)
  out.to_csv(args.csv, index=False)
  print(out.round(4).to_string(index=False))
  print(f"wrote {args.csv}")
  if args.reference is not None:
    flips = sign_flips(out, pd.read_csv(args.reference))
    print(f"\n{len(flips)} sign flips against {args.reference}")
    if len(flips):
      print(flips.round(4).to_string(index=False))


if __name__ == "__main__":
  main()
