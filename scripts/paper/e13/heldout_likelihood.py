#!/usr/bin/env python3
"""E13b: does a simplified vine's held-out likelihood rank the true structure first?

    cd <repository root>
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    taskset -c 4-11 nice -n 15 uv run --no-sync python \\
        scripts/paper/e13/heldout_likelihood.py --out <new folder> --workers 6
    # the same on E3d's test samples (20000 per cell):
    ... --sample test --out <another new folder>

**Question.** E13 found that no pairwise-dependence criterion recovers more
of E3d's true structures than Dissmann's ``tau``, even at n = 20000: every
pair is dependent at a similar strength, so the first tree is not
identifiable from pairs alone. An E3d truth is a simplified vine on its own
structure, though, so a simplified vine should fit best on that structure.
This script asks whether a held-out likelihood finds the truth where
pairwise dependence cannot. CPU only; nothing is fitted on a GPU.

**Cells.** E3d's ``decay`` and ``strong`` designs at d = 5 and 10, seeds
101 to 1010, on each cell's stored sample (``--sample train``, 2000 rows,
the one every E3d arm was fitted on; or ``test``, 20000 rows drawn
independently from the same truth).

**Candidates** (16 per cell): the truth; the mixture's stored members (E10
on E3d), member 1 being Dissmann's selection (``selected``) and 2 to 5 its
weighted random spanning trees; and 10 structures drawn uniformly by
``pyvinecopulib.RVineStructure.sample(d, seeds=[derive_seed("E13b",
"uniform", design, d, seed, j)])``, ``j = 1, ..., 10``.

**Scores.** The sample is split once per cell, by a permutation seeded
with ``derive_seed("E13b", "split", design, d, seed, sample)``: the first
70% fits and the rest is held out. On each candidate, a simplified vine
is fitted on the 70% with pyvinecopulib's compiled
``Vinecop.from_data(..., structure=...)`` on one thread. It is scored by
the mean held-out ``log max(c_hat, 1e-300)`` per observation, the floor
E3d uses, so a higher score is better. There are two scores:

- ``tll``: TLL pairs only, otherwise default controls;
- ``parametric``: ``pyvinecopulib.families.parametric`` (independence,
  Gaussian, Student, Clayton, Gumbel, Frank, Joe, BB1, BB6, BB7, BB8 and
  Tawn), otherwise default controls (AIC selection), since the truth's
  pairs are parametric.

On ``test`` the parametric fits take hours (24 s per d = 10 vine on 1400
rows, about linear in n), so ``--scores tll`` runs the TLL score alone.

**Ranks.** A candidate's rank is 1 plus the number of candidates with a
higher score. Candidates that are the same vine as the ranked one (equal
edge sets, ``e3d_structure``) are left out of its comparison and counted
in ``n_same``.

**References.** E3e's KL gap, true-ns less tll-ns excess NLL, per
observation on the test sample and within one machine (as in E13). Also
E10's member KLs (``member_excess_nll_common``), which say whether the KL
ranks the selected structure above members 2 to 5.

**Outputs** (``--out`` must not exist): ``cells.csv`` (one row per cell,
candidate and score), ``table1_truth_rank.csv`` and
``table2_selected_vs_random.csv``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvinecopulib as pv

from npcc.experiments._study import derive_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import structure_recovery as e13

N_UNIFORM = 10
FIT_SHARE = 0.7
LOG_FLOOR = 1e-300
SCORES = ("tll", "parametric")


def controls(score: str) -> pv.FitControlsVinecop:
  """The simplified vine's controls for ``score``: one family set, one thread."""
  families = {
    "tll": [pv.BicopFamily.tll],
    "parametric": list(pv.families.parametric),
  }[score]
  return pv.FitControlsVinecop(family_set=families, num_threads=1)


def heldout_score(
  fit: np.ndarray, held: np.ndarray, matrix: np.ndarray, score: str
) -> tuple[float, int, float]:
  """Mean held-out log-density of a simplified vine fitted on ``matrix``.

  Returns the score, the number of held-out points whose density is not
  finite and positive (floored at ``LOG_FLOOR``), and the fit's seconds.
  """
  structure = pv.RVineStructure.from_matrix(
    np.asfortranarray(np.asarray(matrix, dtype=np.uint64))
  )
  t0 = time.perf_counter()
  vine = pv.Vinecop.from_data(
    fit, structure=structure, controls=controls(score)
  )
  seconds = time.perf_counter() - t0
  c = np.asarray(vine.pdf(held), dtype=float)
  bad = int((~np.isfinite(c) | (c <= 0)).sum())
  c = np.where(np.isfinite(c), np.maximum(c, LOG_FLOOR), LOG_FLOOR)
  return float(np.mean(np.log(c))), bad, seconds


def candidates(job: dict[str, Any], truth_matrix: np.ndarray) -> list[tuple]:
  """``(name, kind, matrix)`` for the truth, the members and the uniforms."""
  design, d, seed = str(job["design"]), int(job["d"]), int(job["rep"])
  out: list[tuple[str, str, np.ndarray]] = [("truth", "truth", truth_matrix)]
  for j in range(1, 6):
    kind = "selected" if j == 1 else "member"
    matrix = np.asarray(job["stored"][f"mixture-member-{j}"], dtype=int)
    out.append((f"mixture-member-{j}", kind, matrix))
  for j in range(1, N_UNIFORM + 1):
    s = derive_seed("E13b", "uniform", design, d, seed, j)
    matrix = np.asarray(pv.RVineStructure.sample(d, seeds=[s]).matrix)
    out.append((f"uniform-{j}", "uniform", matrix.astype(int)))
  return out


def run_cell(
  job: dict[str, Any], results: str, sample: str, scores: tuple[str, ...]
) -> list[dict]:
  """Every candidate's two held-out scores in one cell."""
  design, d, seed = str(job["design"]), int(job["d"]), int(job["rep"])
  with np.load(Path(results) / e13.E3D / "samples.npz") as z:
    key = f"{design}_d{d}_s{seed}"
    truth_matrix = np.asarray(z[f"{key}_true_matrix"], dtype=int)
    u = np.ascontiguousarray(z[f"{key}_{sample}"], dtype=np.float64)
  perm = np.random.default_rng(
    derive_seed("E13b", "split", design, d, seed, sample)
  ).permutation(u.shape[0])
  n_fit = round(FIT_SHARE * u.shape[0])
  fit = np.ascontiguousarray(u[perm[:n_fit]])
  held = np.ascontiguousarray(u[perm[n_fit:]])
  truth = e13.tree_edges(truth_matrix)
  rows: list[dict[str, Any]] = []
  for name, kind, matrix in candidates(job, truth_matrix):
    rec = e13.recovery_row(matrix, truth)
    for score in scores:
      value, bad, seconds = heldout_score(fit, held, matrix, score)
      rows.append(
        {
          "design": design,
          "d": d,
          "seed": seed,
          "sample": sample,
          "n_fit": n_fit,
          "n_held": held.shape[0],
          "candidate": name,
          "kind": kind,
          "score": score,
          "heldout_loglik": value,
          "n_floored": bad,
          "fit_seconds": seconds,
          "recovery_overall": rec["overall"],
          "recovery_tree1": rec["tree1"],
          "matrix": rec["matrix"],
        }
      )
  return rows


def run_cell_star(
  args: tuple[dict[str, Any], str, str, tuple[str, ...]],
) -> list[dict]:
  """:func:`run_cell` on one tuple, for :meth:`ProcessPoolExecutor.map`."""
  return run_cell(*args)


# --------------------------------------------------------------- the tables


def edges_of(matrix: str) -> list:
  """A stored candidate matrix's edge sets, for comparing vines."""
  return e13.tree_edges(json.loads(matrix))


def add_ranks(cells: pd.DataFrame) -> pd.DataFrame:
  """Each candidate's rank in its cell and score, and how many share its vine."""
  out = []
  for _, g in cells.groupby(["design", "d", "seed", "sample", "score"]):
    g = g.copy()
    vines = [edges_of(m) for m in g["matrix"]]
    values = g["heldout_loglik"].to_numpy()
    ranks, same = [], []
    for i in range(len(g)):
      others = [j for j in range(len(g)) if j != i and vines[j] != vines[i]]
      ranks.append(1 + int(sum(values[j] > values[i] for j in others)))
      same.append(len(g) - 1 - len(others))
    g["rank"], g["n_same"] = ranks, same
    out.append(g)
  return pd.concat(out, ignore_index=True)


def e3e_kl_gap(results: Path) -> pd.Series:
  """E3e's true-ns less tll-ns excess NLL per (design, d, seed), one machine."""
  frames = []
  for (design, d), (true_dir, sel_dir) in e13.E3E.items():
    gap = e13.excess_nll(results / true_dir, "true-ns", design, d) - (
      e13.excess_nll(results / sel_dir, "tll-ns", design, d)
    )
    gap = gap.dropna().rename("kl_gap").reset_index()
    gap["design"], gap["d"] = design, d
    frames.append(gap)
  return pd.concat(frames).set_index(["design", "d", "seed"])["kl_gap"]


def e10_member_kl(results: Path) -> pd.DataFrame:
  """E10's excess NLL of each mixture member on E3d, on the common points."""
  rows = []
  for (design, d), folder in e13.E10_E3D.items():
    mix = pd.read_csv(results / folder / f"mixture_{e13.E3D_ESTIMATOR}.csv")
    mix = mix[(mix["design"] == design) & (mix["d"] == d)]
    for r in mix.itertuples(index=False):
      values = json.loads(str(r.member_excess_nll_common))
      for j, v in enumerate(values, start=1):
        rows.append(
          {
            "design": design,
            "d": d,
            "seed": int(r.seed),
            "candidate": f"mixture-member-{j}",
            "excess_nll": float(v),
          }
        )
  return pd.DataFrame(rows)


def truth_rank_table(ranked: pd.DataFrame, kl_gap: pd.Series) -> pd.DataFrame:
  """Table 1: the truth's rank, and the score gap truth minus selected."""
  rows = []
  for keys, g in ranked.groupby(["design", "d", "sample", "score"], sort=True):
    row = dict(zip(["design", "d", "sample", "score"], keys, strict=True))
    truth = g[g["kind"] == "truth"].set_index("seed")
    sel = g[g["kind"] == "selected"].set_index("seed")
    row["n_cells"] = len(truth)
    row["truth_rank_mean"] = float(truth["rank"].mean())
    row["truth_first_share"] = float((truth["rank"] == 1).mean())
    row["selected_rank_mean"] = float(sel["rank"].mean())
    row["selected_first_share"] = float((sel["rank"] == 1).mean())
    gap = truth["heldout_loglik"] - sel["heldout_loglik"]
    row["score_gap_truth_minus_selected"], row["score_gap_se"] = e13.mean_se(
      gap
    )
    row["truth_beats_selected"] = int((gap > 0).sum())
    best = g.loc[g.groupby("seed")["heldout_loglik"].idxmax(), "kind"]
    for kind in ("truth", "selected", "member", "uniform"):
      row[f"best_is_{kind}"] = int((best == kind).sum())
    row["n_floored_max"] = int(g["n_floored"].max())
    kl = kl_gap.reindex(
      pd.MultiIndex.from_tuples(
        [(keys[0], keys[1], s) for s in truth.index], names=kl_gap.index.names
      )
    ).dropna()
    row["e3e_kl_gap_truens_minus_tllns"], row["e3e_kl_gap_se"] = e13.mean_se(kl)
    row["e3e_n_cells"] = len(kl)
    rows.append(row)
  return pd.DataFrame(rows)


def selected_vs_random_table(
  ranked: pd.DataFrame, member_kl: pd.DataFrame
) -> pd.DataFrame:
  """Table 2: does the score put the selected structure above the others?

  Shares over (cell, other candidate) pairs, leaving out any candidate that
  is the selected structure's own vine; beside them, the share by which
  E10's KL puts the selected member above members 2 to 5.
  """
  rows = []
  for keys, g in ranked.groupby(["design", "d", "sample", "score"], sort=True):
    row = dict(zip(["design", "d", "sample", "score"], keys, strict=True))
    for kind in ("member", "uniform"):
      wins = total = 0
      truth_wins = truth_total = 0
      for _, c in g.groupby("seed"):
        sel = c[c["kind"] == "selected"].iloc[0]
        tru = c[c["kind"] == "truth"].iloc[0]
        sel_vine, tru_vine = edges_of(sel["matrix"]), edges_of(tru["matrix"])
        for o in c[c["kind"] == kind].itertuples(index=False):
          vine = edges_of(str(o.matrix))
          if vine != sel_vine:
            total += 1
            wins += int(sel["heldout_loglik"] > o.heldout_loglik)
          if vine != tru_vine:
            truth_total += 1
            truth_wins += int(tru["heldout_loglik"] > o.heldout_loglik)
      row[f"selected_above_{kind}_share"] = wins / total if total else math.nan
      row[f"selected_vs_{kind}_pairs"] = total
      row[f"truth_above_{kind}_share"] = (
        truth_wins / truth_total if truth_total else math.nan
      )
    kl = member_kl[
      (member_kl["design"] == keys[0]) & (member_kl["d"] == keys[1])
    ]
    wins = total = 0
    for seed, c in kl.groupby("seed"):
      cell = g[g["seed"] == seed]
      sel_vine = edges_of(
        cell.loc[cell["kind"] == "selected", "matrix"].iloc[0]
      )
      first = c[c["candidate"] == "mixture-member-1"]["excess_nll"].iloc[0]
      for o in c[c["candidate"] != "mixture-member-1"].itertuples(index=False):
        m = cell.loc[cell["candidate"] == o.candidate, "matrix"].iloc[0]
        if edges_of(m) != sel_vine:
          total += 1
          wins += int(first < o.excess_nll)
    row["e10_kl_selected_above_member_share"] = (
      wins / total if total else math.nan
    )
    row["e10_kl_pairs"] = total
    rows.append(row)
  return pd.DataFrame(rows)


def main() -> None:
  """Score every candidate in every cell, then write the tables."""
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument(
    "--results",
    type=Path,
    default=Path(__file__).resolve().parents[3] / "results",
  )
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--workers", type=int, default=1)
  p.add_argument("--sample", choices=("train", "test"), default="train")
  p.add_argument(
    "--scores",
    nargs="+",
    choices=SCORES,
    default=list(SCORES),
    help="which scores to compute (parametric fits are the slow ones)",
  )
  p.add_argument(
    "--limit", type=int, default=None, help="only the first LIMIT cells"
  )
  args = p.parse_args()
  if args.out.exists():
    raise SystemExit(f"--out must not exist; got {args.out}")
  jobs = e13.e3d_cells(args.results)
  if args.limit is not None:
    jobs = jobs[: args.limit]
  jobs = sorted(jobs, key=lambda j: -int(j["d"]))
  t0 = time.perf_counter()
  rows: list[dict] = []
  work = [(j, str(args.results), args.sample, tuple(args.scores)) for j in jobs]
  with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for i, out in enumerate(pool.map(run_cell_star, work)):
      rows += out
      if (i + 1) % 5 == 0 or i + 1 == len(jobs):
        elapsed = time.perf_counter() - t0
        print(f"{i + 1}/{len(jobs)} cells, {elapsed:.0f} s", flush=True)
  ranked = add_ranks(pd.DataFrame(rows))
  args.out.mkdir(parents=True)
  ranked.to_csv(args.out / "cells.csv", index=False)
  truth_rank_table(ranked, e3e_kl_gap(args.results)).to_csv(
    args.out / "table1_truth_rank.csv", index=False
  )
  selected_vs_random_table(ranked, e10_member_kl(args.results)).to_csv(
    args.out / "table2_selected_vs_random.csv", index=False
  )
  print(f"done in {time.perf_counter() - t0:.0f} s: {args.out}")


if __name__ == "__main__":
  main()
