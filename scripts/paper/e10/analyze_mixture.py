"""Analyze E10's conditional structure mixtures against their members.

Reads a finished vine run holding the ``mixture`` arm
(``configs/vine_study_mixture*.toml``): its ``metrics_by_x.parquet`` has a row
per (d, n, rep, x) for the mixture and for each member ``mixture-member-<k>``,
the first of which is on the selected structure. Writes into
``<run>/analysis/`` (or ``--out``):

- ``vine_mixture.csv`` -- one row per (d, n). For the joint and the copula KL,
  each x-averaged within a repetition: the mixture's mean and SE over
  repetitions, its first member's, its members' mean, the paired
  differences mixture - member 1 and mixture - members' mean, and in how
  many repetitions the mixture beats member 1. Then each member's mean joint
  KL. With ``--with``, each other arm's mean joint KL over the repetitions
  both runs hold, and the mixture's paired difference to it.
  The best member's mean KL (the member that scores lowest in each
  repetition, an oracle choice no user has), and how many distinct
  structures the mixtures hold: members' draws can coincide, as they must at
  d = 3, which has three R-vines.
- ``vine_mixture_cells.csv`` -- the same per cell: the mixture's, member 1's,
  the members' mean and the best member's joint and copula KL, and the
  number of distinct structures among the members.
- ``vine_mixture_checks.json`` -- per (d, n), Jensen's bound (the mixture's
  KL is at most its members' mean in every repetition), and where a
  ``selected`` arm comes through ``--with``, how far member 1 is from it.

``--with`` takes E2 runs on the same cells, such as
``results/vine_2026-11-10``, ``results/vine_2026-11-11_selected`` and
``results/vine_n2000_<TAG>``, whose oracle, random, selected and kde-tll arms
the mixture is then read against. Rows they share with the mixture run must
pair: the same cell seed and mean true log-density at every shared
(d, n, rep, x), which pins the training data's cell and the test points.

Usage (from the npcc repository root)::

    uv run --no-sync python scripts/paper/e10/analyze_mixture.py \\
        results/vine_study_mixture_<TAG> --with results/vine_2026-11-10 \\
        results/vine_2026-11-11_selected results/vine_n2000_<TAG>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyvinecopulib as pv

METRICS = {"joint": "KL", "copula": "KL_copula"}
KEY = ["d", "n", "rep", "x"]
OTHER_ARMS = ("oracle", "random", "selected", "tll")


def load(run: Path) -> pd.DataFrame:
  """The run's fixed-x metrics."""
  path = run / "metrics_by_x.parquet"
  if not path.is_file():
    raise SystemExit(f"{path} must exist; run the aggregation pass first")
  return pd.read_parquet(path)


def member_arms(metrics: pd.DataFrame) -> list[str]:
  """The members' arm names, in member order."""
  arms = [a for a in metrics["arm"].unique() if a.startswith("mixture-member-")]
  if "mixture" not in set(metrics["arm"]) or not arms:
    raise SystemExit("the run must hold the mixture arm and its members")
  return sorted(arms, key=lambda a: int(a.rsplit("-", 1)[1]))


Edge = tuple[frozenset[int], frozenset[int]]


def edges(matrix: list[list[int]]) -> frozenset[Edge]:
  """A vine's edges as (conditioned pair, conditioning set), every tree.

  Two matrices can write one vine, so structures compare by their edges.
  """
  array = np.asfortranarray(np.asarray(matrix, dtype=np.uint64))
  # pyvinecopulib's compiled `get_trees` is untyped here.
  trees = cast(
    "list[list[tuple[int, int, list[int]]]]",
    pv.RVineStructure.from_matrix(array).get_trees(),
  )
  return frozenset(
    (frozenset((a, b)), frozenset(conditioning))
    for tree in trees
    for a, b, conditioning in tree
  )


def distinct_structures(run: Path) -> pd.Series:
  """How many distinct structures each cell's mixture holds."""
  runtime = pd.read_parquet(run / "runtime.parquet")
  rows = runtime[runtime["arm"] == "mixture"]
  counts = {
    (int(r.d), int(r.n), int(r.rep)): len(
      {edges(m) for m in json.loads(str(r.mixture_structures))}
    )
    for r in rows.itertuples(index=False)
  }
  index = pd.MultiIndex.from_tuples(list(counts), names=["d", "n", "rep"])
  return pd.Series(list(counts.values()), index=index, name="n_distinct")


def wide(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
  """The metric averaged over x, one column per arm, one row per cell."""
  xavg = metrics.groupby(["d", "n", "rep", "arm"])[metric].mean()
  return cast("pd.DataFrame", xavg.unstack("arm"))


def stat(values: pd.Series) -> tuple[float, float]:
  """Mean and standard error over repetitions."""
  v = values.dropna().to_numpy(dtype=float)
  se = float(np.std(v, ddof=1) / math.sqrt(v.size)) if v.size > 1 else math.nan
  return float(np.mean(v)), se


def others(paths: list[Path], ours: pd.DataFrame) -> pd.DataFrame:
  """The ``--with`` runs' rows for E2's arms, checked to pair with ours."""
  frames = []
  mine = ours.groupby(KEY)[["seed", "loglik_true"]].first()
  for path in paths:
    theirs = load(path)
    theirs = theirs[theirs["arm"].isin(OTHER_ARMS)]
    got = theirs.groupby(KEY)[["seed", "loglik_true"]].first()
    shared = mine.index.intersection(got.index)
    if len(shared) == 0:
      raise SystemExit(f"{path} must share cells with the mixture run")
    if not (
      (mine.loc[shared, "seed"] == got.loc[shared, "seed"]).all()
      and (
        mine.loc[shared, "loglik_true"] == got.loc[shared, "loglik_true"]
      ).all()
    ):
      raise SystemExit(
        f"{path} must be scored on the mixture run's cells and test points, "
        "with the same seed and true log-density at every shared (d, n, rep, x)"
      )
    frames.append(theirs)
  out = cast("pd.DataFrame", pd.concat(frames, ignore_index=True))
  if out.duplicated([*KEY, "arm"]).any():
    raise SystemExit("the --with runs must not hold one arm's cell twice")
  return out


def per_cell(
  tables: dict[str, pd.DataFrame], members: list[str], distinct: pd.Series
) -> pd.DataFrame:
  """Per cell: the mixture, member 1, the members' mean and the best."""
  out = pd.DataFrame(index=tables["joint"].index)
  for name, frame in tables.items():
    out[f"{name}_KL_mixture"] = frame["mixture"]
    out[f"{name}_KL_member1"] = frame[members[0]]
    out[f"{name}_KL_members_mean"] = frame[members].mean(axis=1)
    out[f"{name}_KL_best_member"] = frame[members].min(axis=1)
  out["n_distinct"] = distinct.reindex(out.index)
  return out.reset_index()


def table(
  metrics: pd.DataFrame,
  extra: pd.DataFrame | None,
  distinct: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
  members = member_arms(metrics)
  rows: list[dict[str, object]] = []
  checks: dict[str, object] = {}
  tables = {name: wide(metrics, m) for name, m in METRICS.items()}
  theirs = None if extra is None else wide(extra, "KL")
  cells = tables["joint"].index.droplevel("rep").unique()
  for d, n in cells:
    row: dict[str, object] = {
      "d": int(d),
      "n": int(n),
      "n_rep": len(tables["joint"].loc[(d, n)]),
    }
    check: dict[str, object] = {}
    for name, frame in tables.items():
      w = frame.loc[(d, n)]
      mixture, first = w["mixture"], w[members[0]]
      mean_members = w[members].mean(axis=1)
      for label, values in (
        ("mixture", mixture),
        ("member1", first),
        ("members_mean", mean_members),
        ("best_member", w[members].min(axis=1)),
        ("mixture_minus_member1", mixture - first),
        ("mixture_minus_members_mean", mixture - mean_members),
      ):
        row[f"{name}_KL_{label}_mean"], row[f"{name}_KL_{label}_se"] = stat(
          values
        )
      row[f"{name}_mixture_wins_over_member1"] = int((mixture < first).sum())
      check[f"{name}_jensen_max"] = float((mixture - mean_members).max())
    for k, arm in enumerate(members, start=1):
      row[f"joint_KL_member{k}_mean"] = float(
        tables["joint"].loc[(d, n), arm].mean()
      )
    counts = distinct.loc[(d, n)]
    row["n_distinct_mean"] = float(counts.mean())
    row["n_distinct_min"] = int(counts.min())
    row["n_distinct_max"] = int(counts.max())
    if theirs is not None and (d, n) in theirs.index.droplevel("rep"):
      t = theirs.loc[(d, n)]
      mixture = tables["joint"].loc[(d, n), "mixture"]
      for arm in [a for a in OTHER_ARMS if a in t.columns]:
        both = pd.concat([mixture, t[arm]], axis=1, join="inner").dropna()
        if both.empty:
          continue
        row[f"joint_KL_{arm}_mean"], _ = stat(both.iloc[:, 1])
        diff = both.iloc[:, 0] - both.iloc[:, 1]
        (
          row[f"joint_mixture_minus_{arm}_mean"],
          row[f"joint_mixture_minus_{arm}_se"],
        ) = stat(diff)
        row[f"n_rep_{arm}"] = len(both)
        if arm == "selected":
          first = tables["joint"].loc[(d, n), members[0]]
          gap = (first - t[arm]).dropna().abs()
          check["member1_vs_selected_max_abs"] = float(gap.max())
    rows.append(row)
    checks[f"d{int(d)}_n{int(n)}"] = check
  return pd.DataFrame(rows), per_cell(tables, members, distinct), checks


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("run", type=Path, help="A vine run holding the mixture arm.")
  p.add_argument(
    "--with",
    dest="others",
    type=Path,
    nargs="*",
    default=[],
    help="E2 runs on the same cells, read against the mixture.",
  )
  p.add_argument(
    "--out", type=Path, default=None, help="Default: <run>/analysis."
  )
  a = p.parse_args()
  out = a.out or a.run / "analysis"
  out.mkdir(parents=True, exist_ok=True)
  metrics = load(a.run)
  extra = others(a.others, metrics) if a.others else None
  result, cells, checks = table(metrics, extra, distinct_structures(a.run))
  result.to_csv(out / "vine_mixture.csv", index=False)
  cells.to_csv(out / "vine_mixture_cells.csv", index=False)
  (out / "vine_mixture_checks.json").write_text(
    json.dumps(checks, indent=2) + "\n"
  )
  with pd.option_context("display.width", 200, "display.max_columns", 40):
    print(result.round(4).to_string(index=False))
  print(json.dumps(checks, indent=2))


if __name__ == "__main__":
  main()
