"""Why the selected arm's first tree differs from the truth's: pooled tau.

Dissmann's algorithm picks the first tree that maximizes the sum of
``|Kendall's tau|`` over its edges, with tau taken over the whole sample,
pooled over ``x``. Where a deeper tree is strong, a pair the truth joins only
in tree 2 can have a larger pooled tau than a true tree-1 edge. The heuristic
then rightly prefers it, and the truth's first tree is not what pooled tau
identifies.

For every cell holding the ``selected`` arm in a vine run, this redraws the
cell's training sample as the runner draws it
(:func:`npcc.experiments.vine_runner.training_sample`), on the CPU. It takes
the true PITs, the truth's own margins, and compares on them:

- ``true_sum``: the sum of ``|tau|`` over the true first tree's edges;
- ``selected_sum``: the same over the selected first tree's, read from the
  arm's runtime row;
- ``mst_sum``: the maximum over all first trees, the spanning tree Dissmann's
  algorithm would pick on these PITs, which bounds the other two.

The selected arm chose its structure on the PITs of its own fitted margins,
not on the true ones, so ``selected_sum`` can fall short of ``mst_sum``.
The data are redrawn here rather than read back, so the numbers are not
paired with the run's metrics. Writes into ``<run>/analysis/``:

- ``tree1_diagnostic_cells.csv``: one row per cell, with the three sums;
- ``tree1_diagnostic.csv``: per (d, n), the mean of each sum, how often
  ``selected_sum >= true_sum``, and the mean ``|tau|`` of the true first
  tree's edges against that of the selected first-tree edges the truth lacks.

Usage (from the npcc repository root)::

    uv run --no-sync python scripts/paper/e2/tree1_diagnostic.py \\
        results/vine_2026-11-11_selected
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyvinecopulib as pv

from npcc.experiments import vine_runner
from npcc.experiments.vine_config import VineCell

Pair = frozenset[int]


def tree1(matrix: list[list[int]]) -> set[Pair]:
  """A structure's first tree, as pairs of 1-based variable labels."""
  array = np.asfortranarray(np.asarray(matrix, dtype=np.uint64))
  first = pv.RVineStructure.from_matrix(array).get_trees()[0]
  return {frozenset((a, b)) for a, b, _ in first}


def abs_tau(u: np.ndarray) -> dict[Pair, float]:
  """``|Kendall's tau|`` of every pair of columns, keyed by 1-based labels.

  pyvinecopulib's own estimator, the one its structure selection uses.
  """
  d = u.shape[1]
  return {
    frozenset((i + 1, j + 1)): abs(
      pv.utils.wdm(
        np.ascontiguousarray(u[:, i]), np.ascontiguousarray(u[:, j]), "tau"
      )
    )
    for i in range(d)
    for j in range(i + 1, d)
  }


def max_spanning_sum(taus: dict[Pair, float], d: int) -> float:
  """The largest sum of ``taus`` over a spanning tree of ``d`` variables."""
  inside, total = {1}, 0.0
  while len(inside) < d:
    best, chosen = -1.0, None
    for pair, tau in taus.items():
      a, b = sorted(pair)
      if (a in inside) != (b in inside) and tau > best:
        best, chosen = tau, pair
    assert chosen is not None
    inside |= chosen
    total += best
  return total


def cell_rows(run: Path) -> pd.DataFrame:
  """One row per cell of the selected arm, with its three first-tree sums."""
  base_seed = json.loads((run / "config.json").read_text())["run"]["base_seed"]
  truths = {
    (t["d"], t["rep"]): t["matrix"]
    for t in json.loads((run / "truths.json").read_text())
  }
  runtime = pd.read_parquet(run / "runtime.parquet")
  selected = runtime[runtime["arm"] == "selected"]
  if selected.empty:
    raise SystemExit(f"{run} must hold the selected arm; it has none")
  rows = []
  for r in selected.itertuples(index=False):
    d, n, rep = int(r.d), int(r.n), int(r.rep)
    truth = vine_runner.truth_for(base_seed, d, rep)
    if np.asarray(truth.structure.matrix).tolist() != truths[(d, rep)]:
      raise SystemExit(
        f"the truth redrawn for d = {d}, rep {rep} must be the run's "
        f"truths.json one; the run used another base seed or truth code"
      )
    _, u, _ = vine_runner.training_sample(base_seed, VineCell(d, n, rep))
    taus = abs_tau(u.numpy())
    true_edges = tree1(truths[(d, rep)])
    chosen = tree1(json.loads(str(r.structure)))
    rows.append(
      {
        "d": d,
        "n": n,
        "rep": rep,
        "true_sum": sum(taus[p] for p in true_edges),
        "selected_sum": sum(taus[p] for p in chosen),
        "mst_sum": max_spanning_sum(taus, d),
        "tree1_shared": len(true_edges & chosen),
        "tree1_edges": len(true_edges),
        "true_edge_taus": json.dumps(
          [taus[p] for p in sorted(true_edges, key=sorted)]
        ),
        "selected_only_taus": json.dumps(
          [taus[p] for p in sorted(chosen - true_edges, key=sorted)]
        ),
      }
    )
  return pd.DataFrame(rows)


def summary(cells: pd.DataFrame) -> pd.DataFrame:
  """Per (d, n): the sums' means, and the true and selected-only edges' taus."""
  rows = []
  for (d, n), g in cells.groupby(["d", "n"]):
    true_taus = [t for s in g["true_edge_taus"] for t in json.loads(s)]
    other = [t for s in g["selected_only_taus"] for t in json.loads(s)]
    rows.append(
      {
        "d": int(d),
        "n": int(n),
        "cells": len(g),
        "true_sum_mean": float(g["true_sum"].mean()),
        "selected_sum_mean": float(g["selected_sum"].mean()),
        "mst_sum_mean": float(g["mst_sum"].mean()),
        "selected_at_least_true": int(
          (g["selected_sum"] >= g["true_sum"]).sum()
        ),
        "tree1_shared_mean": float(g["tree1_shared"].mean()),
        "tree1_edges": int(g["tree1_edges"].iloc[0]),
        "true_edge_tau_mean": float(np.mean(true_taus)),
        "selected_only_tau_mean": (
          float(np.mean(other)) if other else float("nan")
        ),
        "selected_only_edges": len(other),
      }
    )
  return pd.DataFrame(rows)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("run", type=Path, help="A vine run holding the selected arm.")
  p.add_argument(
    "--out", type=Path, default=None, help="Default: <run>/analysis."
  )
  a = p.parse_args()
  out = a.out or a.run / "analysis"
  out.mkdir(parents=True, exist_ok=True)
  cells = cell_rows(a.run)
  cells.to_csv(out / "tree1_diagnostic_cells.csv", index=False)
  table = summary(cells)
  table.to_csv(out / "tree1_diagnostic.csv", index=False)
  pd.set_option("display.width", 200, "display.max_columns", 20)
  print(table.round(4).to_string(index=False))


if __name__ == "__main__":
  main()
