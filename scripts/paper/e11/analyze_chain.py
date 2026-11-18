"""Analyze E11's chain-rule baseline against the vines on E2's cells.

Reads a finished vine run holding the ``chain`` arm
(``configs/vine_study_chain*.toml``): its ``metrics_by_x.parquet`` has a row
per (d, n, rep, x) for the chain mixture and for each order ``chain-m<k>``.
``--with`` takes the runs holding its references on the same cells: E2's
``oracle``, ``random`` and ``selected`` arms, and E10's npcc ``mixture``.
Each is read through ``scripts/paper/e10/analyze_mixture.py``'s ``others``,
so a run that does not pair -- another cell seed or mean true log-density at
any shared (d, n, rep, x) -- is refused, as E10's analysis refuses it.

Per (d, n), the comparison is paired by repetition on the cells every run
holds: a repetition missing from any reference the runs carry at that (d, n)
leaves every comparison there. Writes into ``<run>/analysis/`` (or
``--out``):

- ``chain.csv`` -- one row per (d, n). For the joint and the copula KL, each
  x-averaged within a repetition: the chain mixture's mean and SE over
  repetitions, and for each reference -- ``mixture``, ``selected``,
  ``oracle``, ``random``, the orders' mean ``orders_mean`` and the best order
  ``best_order`` (the order that scores lowest in each repetition, an oracle
  choice no user has) -- its mean, the paired difference chain - reference
  (mean and SE), and in how many repetitions the chain is lower.
- ``chain_cells.csv`` -- the same per cell, and how many distinct orders the
  cell's chain mixed.
- ``chain_checks.json`` -- per (d, n), the references found, the repetitions
  left out, and Jensen's bound: the chain mixture's KL is at most its
  orders' mean in every repetition, so ``jensen_max`` must be at most 0.

Usage (from the npcc repository root)::

    uv run --no-sync python scripts/paper/e11/analyze_chain.py \\
        results/vine_study_chain_<TAG> --with results/vine_2026-11-10 \\
        results/vine_2026-11-11_selected results/vine_study_mixture_<TAG>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e10"))
from analyze_mixture import load, others, stat, wide

METRICS = {"joint": "KL", "copula": "KL_copula"}
REFERENCES = ("mixture", "selected", "oracle", "random")


def order_arms(metrics: pd.DataFrame) -> list[str]:
  """The orders' arm names, ``chain-m<k>``, in order."""
  arms = [a for a in metrics["arm"].unique() if a.startswith("chain-m")]
  if "chain" not in set(metrics["arm"]) or not arms:
    raise SystemExit("the run must hold the chain arm and its orders")
  return sorted(arms, key=lambda a: int(a.removeprefix("chain-m")))


def distinct_orders(run: Path) -> pd.Series:
  """How many distinct variable orders each cell's chain mixed."""
  runtime = pd.read_parquet(run / "runtime.parquet")
  rows = runtime[runtime["arm"] == "chain"]
  counts = {
    (int(r.d), int(r.n), int(r.rep)): len(
      {tuple(o) for o in json.loads(str(r.chain_orders))}
    )
    for r in rows.itertuples(index=False)
  }
  index = pd.MultiIndex.from_tuples(list(counts), names=["d", "n", "rep"])
  return pd.Series(list(counts.values()), index=index, name="n_distinct")


def tables(metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
  """Per metric, one row per cell: every arm, the orders' mean and the best.

  A missing order leaves the orders' mean and the best order ``nan``, so a
  cell is never scored on fewer orders than its mixture mixed.
  """
  orders = order_arms(metrics)
  out = {}
  for name, metric in METRICS.items():
    frame = wide(metrics, metric)
    frame["orders_mean"] = frame[orders].mean(axis=1, skipna=False)
    frame["best_order"] = frame[orders].min(axis=1, skipna=False)
    out[name] = frame
  return out


def table(
  metrics: pd.DataFrame, distinct: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
  """The per-(d, n) comparison, the per-cell values and the checks."""
  orders = order_arms(metrics)
  wides = tables(metrics)
  rows: list[dict[str, object]] = []
  cells: list[pd.DataFrame] = []
  checks: dict[str, object] = {}
  for d, n in wides["joint"].index.droplevel("rep").unique():
    joint = wides["joint"].loc[(d, n)]
    present = [
      r for r in REFERENCES if r in joint.columns and joint[r].notna().any()
    ]
    needed = ["chain", *orders, *present]
    keep = pd.Series(True, index=joint.index)
    for frame in wides.values():
      keep &= frame.loc[(d, n), needed].notna().all(axis=1)
    reps = keep.index[keep]
    row: dict[str, object] = {
      "d": int(d),
      "n": int(n),
      "n_rep": len(reps),
      "k": len(orders),
      "references": ";".join(present),
    }
    cell = pd.DataFrame(index=reps)
    check: dict[str, object] = {
      "references": present,
      "reps_left_out": [int(r) for r in keep.index[~keep]],
    }
    for name, frame in wides.items():
      w = frame.loc[(d, n)].loc[reps]
      chain = w["chain"]
      row[f"{name}_KL_chain_mean"], row[f"{name}_KL_chain_se"] = stat(chain)
      cell[f"{name}_KL_chain"] = chain
      for ref in (*present, "orders_mean", "best_order"):
        diff = chain - w[ref]
        row[f"{name}_KL_{ref}_mean"], row[f"{name}_KL_{ref}_se"] = stat(w[ref])
        (
          row[f"{name}_chain_minus_{ref}_mean"],
          row[f"{name}_chain_minus_{ref}_se"],
        ) = stat(diff)
        row[f"{name}_chain_wins_over_{ref}"] = int((diff < 0).sum())
        cell[f"{name}_KL_{ref}"] = w[ref]
      gap = chain - w["orders_mean"]
      check[f"{name}_jensen_max"] = float(gap.max()) if len(gap) else None
    counts = distinct.reindex(pd.MultiIndex.from_product([[d], [n], reps]))
    row["n_distinct_orders_mean"] = float(counts.mean())
    cell["n_distinct_orders"] = counts.to_numpy()
    cell.insert(0, "n", int(n))
    cell.insert(0, "d", int(d))
    rows.append(row)
    cells.append(cell.rename_axis("rep").reset_index())
    checks[f"d{int(d)}_n{int(n)}"] = check
  return pd.DataFrame(rows), pd.concat(cells, ignore_index=True), checks


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("run", type=Path, help="A vine run holding the chain arm.")
  p.add_argument(
    "--with",
    dest="others",
    type=Path,
    nargs="*",
    default=[],
    help="E2 and E10 runs on the same cells, read against the chain.",
  )
  p.add_argument(
    "--out", type=Path, default=None, help="Default: <run>/analysis."
  )
  a = p.parse_args()
  out = a.out or a.run / "analysis"
  out.mkdir(parents=True, exist_ok=True)
  metrics = load(a.run)
  if a.others:
    extra = others(a.others, metrics, REFERENCES)
    metrics = pd.concat([metrics, extra], ignore_index=True)
  result, cells, checks = table(metrics, distinct_orders(a.run))
  result.to_csv(out / "chain.csv", index=False)
  cells.to_csv(out / "chain_cells.csv", index=False)
  (out / "chain_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
  with pd.option_context("display.width", 200, "display.max_columns", 60):
    print(result.round(4).to_string(index=False))
  print(json.dumps(checks, indent=2))


if __name__ == "__main__":
  main()
