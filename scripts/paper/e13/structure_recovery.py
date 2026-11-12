#!/usr/bin/env python3
"""E13: how much of the true vine structure does selection recover? CPU only.

    cd <repository root>
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    taskset -c 4-11 nice -n 15 uv run --no-sync python \\
        scripts/paper/e13/structure_recovery.py --out <new folder> --workers 6

**Question.** The vine on the true structure beats the chain rule wherever
both exist, and the chain beats the vine mixture, whose first member is the
selected structure. If selection is the bottleneck, the selected structures
are far from the truth, and a criterion that recovers more of it is worth a
GPU run. This script fits nothing on a GPU: it reads the structures the runs
stored and reselects on the CPU.

**Recovery.** An edge is its conditioned pair and its conditioning set, both
unordered (``e3d_structure``); tree ``t``'s recovery is the share of the
truth's tree-``t`` edges the structure has, and ``overall`` the share over
every tree. A structure that misses an edge in tree ``t`` usually misses
every later edge built on it, so deep trees recover little by construction.

**Datasets.** Every cell with a vine mixture:

- the conditional vine study (E2's truths, E10's mixture): d = 3, 5 (run
  ``vine_study_mixture_2026-11-11``) and d = 10 (``..._d10_2026-11-11``),
  n = 500, 1000 and 2000, 20 reps each;
- E3d's unconditional truths (E10 on E3d): ``decay`` and ``strong`` at d = 5
  and 10, seeds 101 to 1010. The mixture's rows are E10's on E3d,
  ``e10_e3d_2026-11-11``, except d = 10 ``decay``, from gpu053
  (``e10_e3d_gpu053_2026-11-11``).

**Sources** (the ``source`` column of ``cells.csv``):

- ``stored``: the structures the runs recorded. ``mixture-member-1`` is the
  mixture's selected structure, ``vine_runner.select_structure`` on the
  probability integral transforms of its own fitted margins (the
  conditional study) or on the copula-scale training sample (E3d), so this
  is exactly its code path's output. ``mixture-member-<j>`` for ``j >= 2``
  are its weighted random spanning trees. ``E2-selected`` (conditional) and
  ``E3d-tll-ns`` (E3d) are the selected structures whose KL the gap uses.
- ``recomputed``: selection rerun here with pyvinecopulib's Dissmann
  algorithm, TLL pairs, one thread and default controls but for
  ``tree_criterion`` (``tau``, the runs' own, then ``rho``, ``hoeffd``,
  ``mcor``, ``cxi`` and ``joe``). On E3d this is the very training sample
  the runs saw. In the conditional study the runs selected on TabPFN's
  fitted transforms, which only a GPU refit reproduces, so this uses the
  true transforms ``u`` that ``vine_runner.training_sample`` returns beside
  ``y``; ``checks.json`` says how often ``tau`` there gives the stored
  structure, and table 1 sets the two side by side.
- ``tree1``: conditional study only, tree 1 only, on the true transforms.
  ``pooled-tau`` is the maximum spanning tree on ``|Kendall's tau|`` over
  every row, which is Dissmann's first tree (``checks.json`` counts the
  agreements); ``binned-tau`` the maximum spanning tree on the mean over
  ``--bins`` equal-count bins of ``x`` of ``|tau|`` within the bin. An edge
  whose tau changes sign or size with ``x`` can look weak pooled over
  ``x``, and not within a bin.

**The gap.** Per cell, the selected structure's KL less the true
structure's, against the selected structure's recovery:

- conditional: E2's ``selected`` arm less its ``oracle`` (joint and copula
  KL averaged over the evaluation ``x``), with ``E2-selected``'s recovery;
  and E10's ``mixture-member-1`` less E2's ``oracle``, with member 1's;
- E3d: E3d's ``tll-ns`` excess NLL less E3e's ``true-ns``, within one
  machine (d = 5 on the main one, where E3e's ``tll-ns`` is E3d's; d = 10
  ``decay`` on gpu053, both E3e's), with ``E3d-tll-ns``'s recovery. d = 10
  ``strong`` has no ``true-ns`` yet.

**Outputs** (under ``--out``, which must not exist): ``cells.csv`` (one row
per cell, source and structure), ``gap_cells.csv``,
``table1_baseline.csv``, ``table2_criteria.csv``,
``table3_tree1_covariate.csv``, ``table4_recovery_gap.csv`` and
``checks.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvinecopulib as pv
import torch
from scipy import stats

from npcc.experiments import vine_runner
from npcc.experiments.vine_config import VineCell

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e3"))
import e3d_structure

Edge = tuple[frozenset, frozenset]

CRITERIA = ("tau", "rho", "hoeffd", "mcor", "cxi", "joe")
COND_RUNS = {
  3: "vine_study_mixture_2026-11-11",
  5: "vine_study_mixture_2026-11-11",
  10: "vine_study_mixture_d10_2026-11-11",
}
# E2's selected and oracle arms, by (d, n): every run with them.
E2_RUNS = (
  "vine_2026-11-10",
  "vine_2026-11-11_selected",
  "vine_n2000_2026-11-11",
  "vine_d10_2026-11-11",
  "vine_d10_n2000_2026-11-11",
)
E3D = "e3d_unknown_structure"
E3D_ESTIMATOR = "npcc-tabpfn-quantiles-v3-probit"
E10_E3D = {
  ("decay", 5): "e10_e3d_2026-11-11",
  ("strong", 5): "e10_e3d_2026-11-11",
  ("strong", 10): "e10_e3d_2026-11-11",
  ("decay", 10): "e10_e3d_gpu053_2026-11-11",
}
# E3e's true-ns, and the tll-ns it is compared with on the same machine.
E3E = {
  ("decay", 5): ("e3e_simplification_2x2", E3D),
  ("strong", 5): ("e3e_simplification_2x2", E3D),
  ("decay", 10): (
    "e3e_simplification_2x2_gpu053",
    "e3e_simplification_2x2_gpu053",
  ),
}
E3D_SEEDS = tuple(range(101, 1011, 101))
MAX_TREES = 9


def tree_edges(matrix: Sequence[Sequence[int]]) -> list[set[Edge]]:
  """Each tree's edges of the vine a pyvinecopulib matrix writes."""
  return e3d_structure.edges_from_matrix(np.asarray(matrix, dtype=int))


def recovery_row(
  matrix: Sequence[Sequence[int]], truth: list[set[Edge]]
) -> dict[str, object]:
  """Per-tree and overall recovery of ``truth`` by the vine ``matrix``."""
  rec = e3d_structure.recovery(tree_edges(matrix), truth)
  row: dict[str, object] = {
    f"tree{t + 1}": (rec["per_tree"][t] if t < len(truth) else math.nan)
    for t in range(MAX_TREES)
  }
  row["overall"] = rec["overall"]
  row["equal"] = bool(rec["equal"])
  row["matrix"] = json.dumps(np.asarray(matrix, dtype=int).tolist())
  return row


def select(u: np.ndarray, criterion: str) -> np.ndarray:
  """Dissmann's selection with TLL pairs and ``tree_criterion=criterion``.

  ``tau`` is :func:`vine_runner.select_structure` itself; every other
  criterion is the same call with only the criterion changed.
  """
  if criterion == "tau":
    return np.asarray(
      vine_runner.select_structure(torch.from_numpy(u)).matrix, dtype=int
    )
  controls = pv.FitControlsVinecop(
    family_set=[pv.BicopFamily.tll], num_threads=1, tree_criterion=criterion
  )
  return np.asarray(pv.Vinecop.from_data(u, controls=controls).structure.matrix)


def abs_tau(u: np.ndarray) -> np.ndarray:
  """``|Kendall's tau|`` of every pair of columns, a symmetric matrix."""
  d = u.shape[1]
  w = np.zeros((d, d))
  for i in range(d):
    for j in range(i + 1, d):
      w[i, j] = w[j, i] = abs(stats.kendalltau(u[:, i], u[:, j]).statistic)
  return w


def binned_abs_tau(u: np.ndarray, x: np.ndarray, bins: int) -> np.ndarray:
  """The mean over ``bins`` equal-count bins of ``x`` of the within-bin |tau|."""
  order = np.argsort(x, kind="stable")
  return np.mean([abs_tau(u[part]) for part in np.array_split(order, bins)], 0)


def max_spanning_tree(weights: np.ndarray) -> set[Edge]:
  """Prim's maximum spanning tree on ``weights``, as tree-1 edges (1-based)."""
  d = weights.shape[0]
  inside = {0}
  edges: set[Edge] = set()
  while len(inside) < d:
    best = max(
      ((i, j) for i in inside for j in range(d) if j not in inside),
      key=lambda e: weights[e],
    )
    edges.add((frozenset((best[0] + 1, best[1] + 1)), frozenset()))
    inside.add(best[1])
  return edges


def tree1_row(edges: set[Edge], truth: list[set[Edge]]) -> dict[str, object]:
  """Tree-1 recovery of a tree-1 edge set; deeper trees left ``nan``."""
  row: dict[str, object] = {f"tree{t + 1}": math.nan for t in range(MAX_TREES)}
  row["tree1"] = len(edges & truth[0]) / len(truth[0])
  row["overall"] = math.nan
  row["equal"] = edges == truth[0]
  row["matrix"] = json.dumps(sorted(sorted(p) for p, _ in edges))
  return row


# ---------------------------------------------------------------- the cells


def e2_selected(results: Path, label: str) -> dict[tuple[int, int, int], list]:
  """E2's ``selected`` structure per (d, n, rep), from every E2 run."""
  out: dict[tuple[int, int, int], list] = {}
  for r in E2_RUNS:
    rt = pd.read_parquet(results / r / "runtime.parquet")
    rt = rt[(rt["arm"] == "selected") & (rt["label"] == label)]
    for row in rt.itertuples(index=False):
      key = (int(row.d), int(row.n), int(row.rep))
      if key in out:
        raise SystemExit(f"E2's runs must not repeat the selected cell {key}")
      out[key] = json.loads(str(row.structure))
  return out


def conditional_cells(results: Path) -> list[dict[str, Any]]:
  """One job per conditional cell with a mixture: its stored structures."""
  jobs: list[dict[str, Any]] = []
  for run_name in sorted(set(COND_RUNS.values())):
    run = results / run_name
    runtime = pd.read_parquet(run / "runtime.parquet")
    config = json.loads((run / "config.json").read_text())
    base_seed = int(config["run"]["base_seed"])
    truths = {
      (int(t["d"]), int(t["rep"])): t["matrix"]
      for t in json.loads((run / "truths.json").read_text())
    }
    rows = runtime[runtime["arm"] == "mixture"]
    for r in rows.sort_values(["d", "n", "rep"]).itertuples(index=False):
      d, n, rep = int(r.d), int(r.n), int(r.rep)
      members = json.loads(str(r.mixture_structures))
      jobs.append(
        {
          "study": "conditional",
          "design": "",
          "d": d,
          "n": n,
          "rep": rep,
          "base_seed": base_seed,
          "label": str(r.label),
          "truth_stored": truths[(d, rep)],
          "stored": {
            f"mixture-member-{j}": m for j, m in enumerate(members, start=1)
          },
        }
      )
  labels = {j["label"] for j in jobs}
  if len(labels) != 1:
    raise SystemExit(f"the mixture runs must hold one estimator; got {labels}")
  selected = e2_selected(results, labels.pop())
  for job in jobs:
    key = (job["d"], job["n"], job["rep"])
    if key in selected:
      job["stored"]["E2-selected"] = selected[key]
  return jobs


def tll_ns_name(design: str, d: int) -> str | None:
  """The selected structure whose gap E3d's cell reports, if it has one."""
  if (design, d) not in E3E:
    return None
  return "E3d-tll-ns" if E3E[(design, d)][1] == E3D else "E3e-tll-ns"


def e3d_cells(results: Path) -> list[dict[str, Any]]:
  """One job per E3d cell with a mixture: its stored structures."""

  def read(folder: str, arm: str) -> pd.DataFrame:
    return pd.read_csv(results / folder / f"npcc_{E3D_ESTIMATOR}-{arm}.csv")

  e3d = read(E3D, "tll-ns")
  e3e = read("e3e_simplification_2x2_gpu053", "tll-ns")
  jobs: list[dict[str, Any]] = []
  for (design, d), folder in sorted(E10_E3D.items()):
    mix = pd.read_csv(results / folder / f"mixture_{E3D_ESTIMATOR}.csv")
    mix = mix[(mix["design"] == design) & (mix["d"] == d)]
    for seed in E3D_SEEDS:
      row = mix[mix["seed"] == seed]
      if len(row) != 1:
        raise SystemExit(
          f"{folder}: {design} d={d} seed {seed} must be one row"
        )
      members = json.loads(str(row["structure_matrices"].iloc[0]))
      stored = {
        f"mixture-member-{j}": m for j, m in enumerate(members, start=1)
      }
      for name, frame in (("E3d-tll-ns", e3d), ("E3e-tll-ns", e3e)):
        hit = frame[
          (frame["design"] == design)
          & (frame["d"] == d)
          & (frame["seed"] == seed)
        ]
        if len(hit) == 1:
          stored[name] = json.loads(str(hit["structure_matrix"].iloc[0]))
      jobs.append(
        {
          "study": "E3d",
          "design": design,
          "d": d,
          "n": int(row["n_train"].iloc[0]),
          "rep": seed,
          "stored": stored,
        }
      )
  return jobs


def run_cell(job: dict[str, Any], results: str, bins: int) -> list[dict]:
  """Every source's recovery in one cell."""
  torch.set_num_threads(1)
  key = {k: job[k] for k in ("study", "design", "d", "n", "rep")}
  rows: list[dict[str, Any]] = []

  def add(source: str, name: str, row: dict, seconds: float = 0.0) -> None:
    rows.append(
      {**key, "source": source, "structure": name, **row, "seconds": seconds}
    )

  t0 = time.perf_counter()
  if job["study"] == "conditional":
    d, n, rep = int(job["d"]), int(job["n"]), int(job["rep"])
    base_seed = int(job["base_seed"])
    truth_matrix = np.asarray(
      vine_runner.truth_for(base_seed, d, rep).structure.matrix, dtype=int
    )
    if truth_matrix.tolist() != job["truth_stored"]:
      raise SystemExit(f"d={d} rep={rep}: the truth must match truths.json")
    _, u_t, x_t = vine_runner.training_sample(base_seed, VineCell(d, n, rep))
    u, x = u_t.numpy(), x_t.numpy().ravel()
  else:
    design, d, seed = str(job["design"]), int(job["d"]), int(job["rep"])
    with np.load(Path(results) / E3D / "samples.npz") as z:
      key_ = f"{design}_d{d}_s{seed}"
      truth_matrix = np.asarray(z[f"{key_}_true_matrix"], dtype=int)
      u = np.ascontiguousarray(z[f"{key_}_train"], dtype=np.float64)
    x = None
  data_seconds = time.perf_counter() - t0
  truth = tree_edges(truth_matrix)

  for name, matrix in job["stored"].items():
    add("stored", name, recovery_row(matrix, truth))

  for criterion in CRITERIA:
    t0 = time.perf_counter()
    matrix = select(u, criterion)
    seconds = time.perf_counter() - t0
    add("recomputed", criterion, recovery_row(matrix, truth), seconds)

  if x is not None:
    for name, weights in (
      ("pooled-tau", lambda: abs_tau(u)),
      ("binned-tau", lambda: binned_abs_tau(u, x, bins)),
    ):
      t0 = time.perf_counter()
      edges = max_spanning_tree(weights())
      add("tree1", name, tree1_row(edges, truth), time.perf_counter() - t0)
  for row in rows:
    row["data_seconds"] = data_seconds
  return rows


def run_cell_star(args: tuple[dict[str, Any], str, int]) -> list[dict]:
  """:func:`run_cell` on one tuple, for :meth:`ProcessPoolExecutor.map`."""
  return run_cell(*args)


# ----------------------------------------------------------------- the gaps


def x_mean_kl(run: Path, arms: Sequence[str], label: str) -> pd.DataFrame:
  """Joint and copula KL averaged over x, one row per (d, n, rep, arm)."""
  m = pd.read_parquet(run / "metrics_by_x.parquet")
  m = m[m["arm"].isin(arms) & (m["label"] == label)]
  return (
    m.groupby(["d", "n", "rep", "arm"])
    .agg(
      KL=("KL", "mean"), KL_copula=("KL_copula", "mean"), seed=("seed", "first")
    )
    .reset_index()
  )


def conditional_gaps(
  results: Path, cells: pd.DataFrame, label: str
) -> pd.DataFrame:
  """Selected-minus-oracle KL per conditional cell, with the recovery.

  ``E2-selected`` is E2's own selected arm; ``mixture-member-1`` is E10's
  first member, scored in E10's run, against E2's oracle in the same cell.
  """
  e2 = pd.concat(
    [x_mean_kl(results / r, ["selected", "oracle"], label) for r in E2_RUNS]
  )
  if e2.duplicated(["d", "n", "rep", "arm"]).any():
    raise SystemExit("E2's runs must not repeat an (arm, cell)")
  e10 = pd.concat(
    [
      x_mean_kl(results / r, ["mixture-member-1"], label)
      for r in sorted(set(COND_RUNS.values()))
    ]
  )
  kl = pd.concat([e2, e10])
  if (kl.groupby(["d", "n", "rep"])["seed"].nunique() > 1).any():
    raise SystemExit("every run must give a cell the same seed")
  wide = kl.pivot_table(
    index=["d", "n", "rep"], columns="arm", values=["KL", "KL_copula"]
  )
  cond = cells[
    (cells["study"] == "conditional") & (cells["source"] == "stored")
  ]
  out = []
  for name, arm in (
    ("E2-selected", "selected"),
    ("mixture-member-1", "mixture-member-1"),
  ):
    rec = cond[cond["structure"] == name].set_index(["d", "n", "rep"])
    for metric in ("KL", "KL_copula"):
      gap = (wide[(metric, arm)] - wide[(metric, "oracle")]).rename("gap")
      frame = rec[["tree1", "overall"]].join(gap, how="inner").reset_index()
      frame["study"], frame["design"] = "conditional", ""
      frame["structure"], frame["metric"] = name, metric
      out.append(frame)
  return pd.concat(out, ignore_index=True)


def excess_nll(folder: Path, arm: str, design: str, d: int) -> pd.Series:
  """An E3d-layout arm's excess NLL per seed, on one design at one d."""
  f = pd.read_csv(folder / f"npcc_{E3D_ESTIMATOR}-{arm}.csv")
  f = f[(f["design"] == design) & (f["d"] == d)]
  return f.set_index("seed")["excess_nll"]


def e3d_gaps(results: Path, cells: pd.DataFrame) -> pd.DataFrame:
  """tll-ns less true-ns excess NLL per E3d cell, within one machine.

  The recovery is that of the tll-ns run's own recorded structure.
  """
  out = []
  e3d = cells[(cells["study"] == "E3d") & (cells["source"] == "stored")]
  for (design, d), (true_dir, sel_dir) in E3E.items():
    name = tll_ns_name(design, d)

    gap = (
      excess_nll(results / sel_dir, "tll-ns", design, d)
      - excess_nll(results / true_dir, "true-ns", design, d)
    ).dropna()
    rec = e3d[
      (e3d["design"] == design) & (e3d["d"] == d) & (e3d["structure"] == name)
    ].set_index("rep")
    frame = rec[["n", "tree1", "overall"]].join(gap.rename("gap"), how="inner")
    frame = frame.reset_index(names="rep")
    frame["study"], frame["design"], frame["d"] = "E3d", design, d
    frame["structure"], frame["metric"] = name, "excess_nll"
    frame["machine"] = "gpu053" if "gpu053" in true_dir else "main"
    out.append(frame)
  return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------- the tables

GROUP = ["study", "design", "d", "n"]


def mean_se(v: pd.Series) -> tuple[float, float]:
  """Mean and standard error, ``nan`` kept out."""
  a = v.dropna().to_numpy(dtype=float)
  se = float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else math.nan
  return (float(a.mean()) if a.size else math.nan), se


def recovery_table(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
  """Mean per-tree and overall recovery, its SE and the share equal to truth."""
  rows = []
  for keys, g in frame.groupby(by, sort=True, dropna=False):
    row = dict(zip(by, keys, strict=True))
    row["n_cells"] = len(g)
    for t in range(MAX_TREES):
      row[f"tree{t + 1}"] = mean_se(g[f"tree{t + 1}"])[0]
    row["overall"], row["overall_se"] = mean_se(g["overall"])
    row["share_equal"] = float(g["equal"].mean())
    rows.append(row)
  return pd.DataFrame(rows)


def baseline_table(cells: pd.DataFrame) -> pd.DataFrame:
  """Table 1: the mixture's selected structure, its other members, the rest."""
  pick = cells.copy()
  member = pick["structure"].str.startswith("mixture-member-")
  pick.loc[member & (pick["structure"] != "mixture-member-1"), "structure"] = (
    "mixture-members-2-to-K"
  )
  keep = (pick["source"] == "stored") | (
    (pick["source"] == "recomputed") & (pick["structure"] == "tau")
  )
  pick = pick[keep]
  pick["structure"] = np.where(
    pick["source"] == "recomputed", "recomputed-tau", pick["structure"]
  )
  return recovery_table(pick, [*GROUP, "structure"])


def criteria_table(cells: pd.DataFrame) -> pd.DataFrame:
  """Table 2: recovery by tree criterion, reselected here."""
  pick = cells[cells["source"] == "recomputed"].rename(
    columns={"structure": "criterion"}
  )
  table = recovery_table(pick, [*GROUP, "criterion"])
  table["u"] = np.where(
    table["study"] == "conditional", "true PITs", "training sample"
  )
  return table


def tree1_table(cells: pd.DataFrame) -> pd.DataFrame:
  """Table 3: tree-1 recovery, pooled against binned |tau| (conditional)."""
  c = cells[cells["study"] == "conditional"]
  wide = c.pivot_table(
    index=[*GROUP, "rep"], columns="structure", values="tree1"
  )
  rows = []
  for keys, g in wide.groupby(level=GROUP, sort=True):
    row = dict(zip(GROUP, keys, strict=True))
    row["n_cells"] = len(g)
    for name in ("mixture-member-1", "tau", "pooled-tau", "binned-tau"):
      row[f"{name}_tree1"], row[f"{name}_tree1_se"] = mean_se(g[name])
    diff = g["binned-tau"] - g["pooled-tau"]
    row["binned_minus_pooled"], row["binned_minus_pooled_se"] = mean_se(diff)
    row["binned_better"] = int((diff > 0).sum())
    row["binned_worse"] = int((diff < 0).sum())
    rows.append(row)
  return pd.DataFrame(rows)


def correlations(frame: pd.DataFrame) -> dict[str, float]:
  """Pearson and Spearman of each recovery against ``gap``, with p-values."""
  row: dict[str, float] = {}
  for rec in ("overall", "tree1"):
    ok = frame[[rec, "gap"]].dropna()
    row[f"{rec}_mean"] = float(ok[rec].mean()) if len(ok) else math.nan
    if len(ok) > 2 and ok[rec].nunique() > 1 and ok["gap"].nunique() > 1:
      p = stats.pearsonr(ok[rec], ok["gap"])
      s = stats.spearmanr(ok[rec], ok["gap"])
      row[f"{rec}_pearson"] = float(p.statistic)
      row[f"{rec}_pearson_p"] = float(p.pvalue)
      row[f"{rec}_spearman"] = float(s.statistic)
      row[f"{rec}_spearman_p"] = float(s.pvalue)
  return row


def gap_table(gaps: pd.DataFrame) -> pd.DataFrame:
  """Table 4: correlation of recovery with the selected-minus-true gap.

  One row per cell group (``scope`` ``cells``), then pooled rows (``scope``
  ``within``) per study and per study and d, where each recovery and the
  gap are first centered within their cell group: recovery and the gap both
  move with ``d`` and ``n``, so pooling the raw values would correlate those
  instead.
  """
  by = ["study", "design", "d", "n", "structure", "metric"]
  rows = []
  for keys, g in gaps.groupby(by, sort=True, dropna=False):
    row = {**dict(zip(by, keys, strict=True)), "scope": "cells"}
    row["n_cells"] = len(g)
    row["gap_mean"], row["gap_se"] = mean_se(g["gap"])
    rows.append({**row, **correlations(g)})
  centered = gaps.copy()
  for col in ("overall", "tree1", "gap"):
    centered[col] = centered[col] - centered.groupby(by, dropna=False)[
      col
    ].transform("mean")
  for pooled in (["study"], ["study", "d"]):
    keys_by = [*pooled, "structure", "metric"]
    for keys, g in centered.groupby(keys_by, sort=True, dropna=False):
      row = dict(zip(keys_by, keys, strict=True))
      row = {
        "study": row["study"],
        "design": "all",
        "d": row.get("d", "all"),
        "n": "all",
        "structure": row["structure"],
        "metric": row["metric"],
        "scope": "within",
        "n_cells": len(g),
      }
      rows.append({**row, **correlations(g)})
  return pd.DataFrame(rows)


def checks(cells: pd.DataFrame) -> dict[str, object]:
  """How the reselection and the tree-1 MST agree with what the runs stored."""
  out: dict[str, object] = {}
  for (study, design, d, n), g in cells.groupby(GROUP, sort=True, dropna=False):
    w = g.pivot_table(
      index="rep", columns="structure", values="matrix", aggfunc="first"
    )
    eq = []
    for _, r in w.iterrows():
      a, b = r.get("mixture-member-1"), r.get("tau")
      if isinstance(a, str) and isinstance(b, str):
        eq.append(
          e3d_structure.edges_from_matrix(np.asarray(json.loads(a)))
          == e3d_structure.edges_from_matrix(np.asarray(json.loads(b)))
        )
    entry: dict[str, object] = {
      "recomputed_tau_equals_stored_member1": f"{sum(eq)}/{len(eq)}",
    }
    if study == "conditional":
      t = g.pivot_table(index="rep", columns="structure", values="tree1")
      tau_tree1 = g[g["structure"] == "tau"].set_index("rep")["matrix"]
      pooled = g[g["structure"] == "pooled-tau"].set_index("rep")["matrix"]
      agree = [
        {(frozenset(p), frozenset()) for p in json.loads(pooled[r])}
        == e3d_structure.edges_from_matrix(
          np.asarray(json.loads(tau_tree1[r]))
        )[0]
        for r in pooled.index
      ]
      entry["pooled_tau_mst_equals_dissmann_tree1"] = (
        f"{sum(agree)}/{len(agree)}"
      )
      entry["tree1_member1_minus_recomputed_tau_mean"] = float(
        (t["mixture-member-1"] - t["tau"]).mean()
      )
    if study == "E3d" and "E3d-tll-ns" in w:
      eq2 = [
        e3d_structure.edges_from_matrix(np.asarray(json.loads(r["E3d-tll-ns"])))
        == e3d_structure.edges_from_matrix(
          np.asarray(json.loads(r["mixture-member-1"]))
        )
        for _, r in w.iterrows()
        if isinstance(r.get("E3d-tll-ns"), str)
      ]
      entry["E3d_tll_ns_equals_member1"] = f"{sum(eq2)}/{len(eq2)}"
    out[f"{study} {design} d={d} n={n}".replace("  ", " ")] = entry
  return out


def main() -> None:
  """Reselect, read the stored structures, and write the tables."""
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument(
    "--results",
    type=Path,
    default=Path(__file__).resolve().parents[3] / "results",
  )
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--workers", type=int, default=1)
  p.add_argument("--bins", type=int, default=5)
  p.add_argument(
    "--limit",
    type=int,
    default=None,
    help="only the first LIMIT cells of each study (a smoke run)",
  )
  args = p.parse_args()
  if args.out.exists():
    raise SystemExit(f"--out must not exist; got {args.out}")
  e3d_structure.self_check(pv)

  jobs = conditional_cells(args.results)
  label = str(jobs[0]["label"])
  e3d_jobs = e3d_cells(args.results)
  if args.limit is not None:
    jobs, e3d_jobs = jobs[: args.limit], e3d_jobs[: args.limit]
  # The slow conditional cells (d = 10, n = 2000) first, so they do not trail.
  jobs = sorted(jobs, key=lambda j: -int(j["d"]) * int(j["n"])) + e3d_jobs
  t0 = time.perf_counter()
  rows: list[dict] = []
  with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for i, out in enumerate(
      pool.map(run_cell_star, [(j, str(args.results), args.bins) for j in jobs])
    ):
      rows += out
      if (i + 1) % 20 == 0 or i + 1 == len(jobs):
        print(
          f"{i + 1}/{len(jobs)} cells, {time.perf_counter() - t0:.0f} s",
          flush=True,
        )
  cells = pd.DataFrame(rows)

  args.out.mkdir(parents=True)
  cells.to_csv(args.out / "cells.csv", index=False)
  baseline_table(cells).to_csv(args.out / "table1_baseline.csv", index=False)
  criteria_table(cells).to_csv(args.out / "table2_criteria.csv", index=False)
  tree1_table(cells).to_csv(
    args.out / "table3_tree1_covariate.csv", index=False
  )
  gaps = pd.concat(
    [
      conditional_gaps(args.results, cells, label),
      e3d_gaps(args.results, cells),
    ],
    ignore_index=True,
  )
  gaps.to_csv(args.out / "gap_cells.csv", index=False)
  gap_table(gaps).to_csv(args.out / "table4_recovery_gap.csv", index=False)
  report = checks(cells)
  report["_run"] = {
    "cells": len(jobs),
    "bins": args.bins,
    "criteria": list(CRITERIA),
    "wall_seconds": time.perf_counter() - t0,
    "workers": args.workers,
    "threads": {
      k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
    },
  }
  (args.out / "checks.json").write_text(json.dumps(report, indent=2) + "\n")
  print(f"done in {time.perf_counter() - t0:.0f} s: {args.out}")


if __name__ == "__main__":
  main()
