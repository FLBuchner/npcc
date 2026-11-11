#!/usr/bin/env python3
"""E10 on E3d: equal-weight mixtures of vines on several structures.

Run in the NPCC environment, from the repository root, after ``e3d_truth.py``:

    cd /home/thibault/npcc
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    M=scripts/paper/e3/e3d_mixture.py
    O=results/e10_e3d_competitors_2026-11-11
    taskset -c 4 nice -n 10 uv run --no-sync python $M --arms tll parametric --out $O
    taskset -c 5 nice -n 10 uv run --no-sync python $M --arms vdc_shared --vdc-core 6 --out $O
    # GPU (main-text TabPFN v3), once allowed:
    uv run --no-sync python $M --arms npcc-tabpfn-quantiles-v3-probit --dims 5 \\
        --out results/e10_e3d_tabpfn --device cuda --gpu-is-free

**Question.** E3d fits one vine on one structure, selected or random. A
non-simplified vine is consistent on any structure, so averaging vines fitted
on several plausible structures should lose little where the selection is
right and gain where it is wrong. The competitors are run the same way, so the
mixture's gain can be read against theirs.

**Structures.** Per cell, ``--k`` (5) structures picked on the training data,
which is already on the copula scale, and shared by every arm:

- member 1: TLL's Dissmann selection (``vine_runner.select_structure``),
  which is the structure of E3d's ``tll`` arm and of its NPCC ``tll-ns``;
- members 2 to K: pyvinecopulib's weighted random spanning trees, weights
  ``|tau|`` (``vine_runner.weighted_random_structure``), each seeded with
  ``derive_seed("E3d", "mixture", d, seed, j)``.

Every process recomputes them, and a cell refuses to run when another arm's
row for it under ``--out`` records other structures.

**Arms.** Each fits one vine per structure and scores their equal-weight
mixture, ``log c_mix = logsumexp_j log c_j - log K``
(``vine_runner.mixture_log_density``):

- ``tll`` and ``parametric``: pyvinecopulib's compiled
  ``Vinecop.from_data(train, structure=..., controls=...)`` with E3d's
  controls for its arms of those names (``e3d_npcc_fit.compiled_controls``);
- ``vdc_shared``: VDC's pair copulas in ``e3_vine.SharedVinecop``, through a
  ``VdcServer``, as E3d's ``vdc_true`` but on each member structure (not
  VDC's own pipeline, E3d's ``vdc``);
- ``npcc-<E1 estimator>``: ``npcc.RosenblattVinecop`` with a
  ``NonSimplifiedContext`` and the estimator's controls, as in
  ``e3d_npcc_fit.py``. Member 1's torch seed is that script's for
  ``tll-ns``, so member 1 is E3d's ``npcc-<estimator>-tll-ns``; member
  ``j >= 2``'s is ``derive_seed("E3d", "mixture", design, d, seed, j,
  estimator_id)``. One untimed warm-up fit precedes the first member.

Members are fitted one at a time, and each is evaluated at the test points and
released before the next is fitted, so one member's pair copulas occupy the
device at a time. A finished member is saved, and an interrupted cell resumes
at its next member.

**Metrics.** As in E3d: excess NLL = mean of ``log c_true - log max(c_hat,
1e-300)`` over the arm's own finite points. Each member's log-densities are
the ones E3d records for that arm (the compiled vine's: the log of the
floored density; the others' raw), NaN kept, and a NaN in any member makes the
mixture NaN at that point. Per cell: the mixture's excess NLL; each member's,
on its own finite points and on the mixture's (where every member is finite);
their mean, member 1's (the selected structure's) and the best member's; and
how many of the K structures are distinct vines, compared as edge sets
(several matrices encode one vine). On the mixture's points, the log of a mean is at
least the mean of the logs, so the mixture's excess NLL is at most the
members' mean. Also the fit and inference seconds of every member, its
structure, and its recovery of the truth (``e3d_structure.recovery``).
``fit_s`` sums the members' structure picks and fits, ``inference_s`` their
evaluations and the mixing.

**Outputs** (under ``--out``, which must differ from E3d's folder): per arm
and cell, ``cells/<arm>/<design>_d<d>_s<seed>.json`` (one row), ``.npy``
(the mixture's test-point log-densities, float32, NaN kept) and
``_members.npy`` (the members', ``(K, n_test)``, float32); each finished
member in ``cells/<arm>/members/<key>_m<j>.json`` and ``.npy`` (float64);
``mixture_<arm>.csv`` (the rows); and ``summary.csv``, one row per cell and
method over every arm under ``--out``: the mixture's, member 1's, the
members' mean and the best member's excess NLL on the mixture's points, and
the number of distinct structures.

**Seeds.** ``--seeds`` takes an explicit list (at d = 10 the GPU runs go in
two batches, 101-505 then 606-1010), and every run resumes by cell: a cell
whose row exists is skipped, and a cell with finished members resumes at the
next one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyvinecopulib as pv
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_common as common  # noqa: E402
import e3_npcc  # noqa: E402
import e3d_structure  # noqa: E402
import npcc  # noqa: E402
from e3_vdc_pair import VdcBicop, VdcServer  # noqa: E402
from e3_vine import SharedVinecop  # noqa: E402
from e3d_npcc_fit import RESULTS as E3D_RESULTS  # noqa: E402
from e3d_npcc_fit import compiled_controls  # noqa: E402
from npcc import FitControlsRosenblattVinecop, RosenblattVinecop  # noqa: E402
from npcc.experiments._study import derive_seed, memory_usage, release_gpu, reset_gpu_peaks  # noqa: E402
from npcc.experiments.vine_runner import (  # noqa: E402
  mixture_log_density,
  select_structure,
  weighted_random_structure,
)
from pyvinecopulib.core import NonSimplifiedContext  # noqa: E402

#: The members of every mixture, as in the vine study's ``mixture`` arm.
K = 5
COMPETITORS = ("tll", "parametric", "vdc_shared")
#: The repository this script runs from, which may be a worktree of
#: ``common.NPCC_REPO``; its commit is recorded beside the provenance's.
SCRIPT_REPO = Path(__file__).resolve().parents[3]


def parse_arm(arm: str) -> str | None:
  """The E1 estimator label of ``npcc-<label>``, or ``None`` for a competitor."""
  if arm in COMPETITORS:
    return None
  label = arm[len("npcc-"):] if arm.startswith("npcc-") else ""
  if label not in e3_npcc.ESTIMATORS:
    raise SystemExit(f"arm must be one of {COMPETITORS} or npcc-<E1 estimator>; got {arm}")
  return label


def structures_for(train: torch.Tensor, d: int, seed: int, k: int) -> tuple[list, list, list]:
  """The cell's ``k`` structures, the seed of each draw, and the seconds each took.

  Member 1 is TLL's Dissmann selection (no seed); member ``j >= 2`` is a
  weighted random draw seeded by ``derive_seed("E3d", "mixture", d, seed, j)``.
  """
  structures, seeds, seconds = [], [], []
  for j in range(1, k + 1):
    t0 = time.perf_counter()
    if j == 1:
      s, s_seed = select_structure(train), None
    else:
      s_seed = derive_seed("E3d", "mixture", d, seed, j)
      s = weighted_random_structure(train, s_seed)
    seconds.append(time.perf_counter() - t0)
    structures.append(s)
    seeds.append(s_seed)
  return structures, seeds, seconds


def member_torch_seed(design: str, d: int, seed: int, j: int, estimator_id: str) -> int:
  """Member ``j``'s torch seed; member 1's is ``e3d_npcc_fit.py``'s for ``tll-ns``."""
  if j == 1:
    return derive_seed("E3d", design, d, seed, "tll-ns", estimator_id)
  return derive_seed("E3d", "mixture", design, d, seed, j, estimator_id)


def excess_nll(lp: np.ndarray, logtrue: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
  """E3d's excess NLL over ``mask``, and its Monte-Carlo standard error."""
  ratio = (logtrue - np.maximum(lp, np.log(common.LOG_FLOOR)))[mask]
  return float(np.mean(ratio)), float(np.std(ratio, ddof=1) / np.sqrt(ratio.size))


def score_members(member_logs: np.ndarray, logtrue: np.ndarray) -> dict:
  """The mixture's log-densities and the excess NLL of it and of each member.

  Parameters
  ----------
  member_logs : ndarray, shape (k, n)
      Each member's log-density at the test points, NaN kept.
  logtrue : ndarray, shape (n,)
      The true log-density there.

  Returns
  -------
  dict
      ``mixture`` (the ``(n,)`` log-densities) and the row's score columns.
  """
  mixture = mixture_log_density(torch.as_tensor(member_logs, dtype=torch.float64)).numpy()
  finite = np.isfinite(mixture)
  own = [excess_nll(lp, logtrue, np.isfinite(lp)) for lp in member_logs]
  common_set = [excess_nll(lp, logtrue, finite)[0] for lp in member_logs]
  mix, mix_se = excess_nll(mixture, logtrue, finite)
  floor = np.log(common.LOG_FLOOR)
  return {
    "mixture": mixture,
    "test_loglik": float(np.mean(np.maximum(mixture, floor)[finite])),
    "excess_nll": mix, "excess_nll_mc_se": mix_se,
    "n_nonfinite": int(np.sum(~finite)), "excess_nll_scope": "own finite points",
    "member_excess_nll": json.dumps([e for e, _ in own]),
    "member_excess_nll_mc_se": json.dumps([s for _, s in own]),
    "member_n_nonfinite": json.dumps([int(np.sum(~np.isfinite(lp))) for lp in member_logs]),
    "members_mean_excess_nll": float(np.mean([e for e, _ in own])),
    "member1_excess_nll": own[0][0],
    "member_excess_nll_common": json.dumps(common_set),
    "members_mean_excess_nll_common": float(np.mean(common_set)),
    "member1_excess_nll_common": common_set[0],
    "best_member_excess_nll_common": float(np.min(common_set)),
    "best_member": int(np.argmin(common_set)) + 1,
    "mixture_minus_members_mean": mix - float(np.mean(common_set)),
    "mixture_minus_member1": mix - common_set[0],
    "mixture_minus_best_member": mix - float(np.min(common_set)),
  }


def check_structures(out: Path, key: str, matrices: list) -> None:
  """Refuse a cell whose other rows under ``out`` record other structures."""
  for path in sorted(out.glob(f"cells/*/{key}.json")):
    if json.loads(json.loads(path.read_text())["structure_matrices"]) != matrices:
      raise SystemExit(f"{key}: the structures must equal those of {path}; they differ")
  for path in sorted(out.glob(f"cells/*/members/{key}_m*.json")):
    j = int(path.stem.rsplit("_m", 1)[1])
    if j <= len(matrices) and json.loads(path.read_text())["structure_matrix"] != matrices[j - 1]:
      raise SystemExit(f"{key}: member {j}'s structure must equal that of {path}; it differs")


def edge_set(matrix: list) -> frozenset:
  """A vine's edges (conditioned pair, conditioning set), as pyvinecopulib reads them.

  Several matrices can encode one vine, but not several edge sets;
  ``scripts/paper/e2/analyze.py`` compares structures the same way.
  """
  array = np.asfortranarray(np.asarray(matrix, dtype=np.uint64))
  return frozenset((frozenset((a, b)), frozenset(cond))
                   for tree in pv.RVineStructure.from_matrix(array).get_trees()
                   for a, b, cond in tree)


def distinct(matrices: list) -> int:
  """How many different vines the matrices encode, compared as edge sets."""
  return len({edge_set(m) for m in matrices})


#: The summary's columns: one row per cell and method, the mixture against its
#: members on the mixture's finite points.
SUMMARY = {"design": "design", "d": "d", "seed": "seed", "method": "method", "k": "k",
           "n_distinct_structures": "n_distinct_structures",
           "mixture_excess_nll": "excess_nll", "member1_excess_nll": "member1_excess_nll_common",
           "members_mean_excess_nll": "members_mean_excess_nll_common",
           "best_member_excess_nll": "best_member_excess_nll_common", "best_member": "best_member",
           "mixture_minus_member1": "mixture_minus_member1",
           "mixture_minus_members_mean": "mixture_minus_members_mean",
           "mixture_minus_best_member": "mixture_minus_best_member",
           "n_excluded": "n_nonfinite", "member1_recovery_overall": None,
           "fit_s": "fit_s", "inference_s": "inference_s", "device": "device"}


def write_summary(out: Path) -> None:
  """``summary.csv``: every finished cell under ``out``, one row per cell and method.

  Cells copied in from another machine are compared here too: every method of
  a cell must have mixed the same structures.
  """
  rows, seen = [], {}
  for path in sorted(out.glob("cells/*/*.json")):
    r = json.loads(path.read_text())
    cell = (r["design"], r["d"], r["seed"])
    first = seen.setdefault(cell, (r["method"], r["structure_matrices"]))
    if json.loads(first[1]) != json.loads(r["structure_matrices"]):
      raise SystemExit(f"{cell}: {r['method']} must have mixed the structures {first[0]} "
                       f"mixed; they differ")
    rows.append({col: (json.loads(r["member_recovery_overall"])[0] if src is None else r[src])
                 for col, src in SUMMARY.items()})
  if not rows:
    return
  rows.sort(key=lambda r: (r["design"], r["d"], r["method"], r["seed"]))
  # Arms run in separate processes; each writes its own file and renames it.
  tmp = out / f"summary.{os.getpid()}.csv"
  common.write_csv(rows, tmp)
  tmp.replace(out / "summary.csv")


def fit_member(arm: str, label: str | None, structure: pv.RVineStructure, tr: np.ndarray,
               te: np.ndarray, *, device: str, torch_seed: int | None,
               server: VdcServer | None) -> tuple[np.ndarray, float, float, dict]:
  """Fit one member and return its test-point log-densities, seconds and extras."""
  tr_t = torch.as_tensor(tr, dtype=torch.float64)
  te_t = torch.as_tensor(te, dtype=torch.float64)
  extra: dict = {}
  if arm in ("tll", "parametric"):
    t0 = time.perf_counter()
    vc = pv.Vinecop.from_data(tr, structure=structure, controls=compiled_controls(arm))
    fit_s = time.perf_counter() - t0
    if not np.array_equal(np.asarray(vc.matrix), np.asarray(structure.matrix)):
      raise RuntimeError(f"{arm}'s fitted vine must keep the structure it was given")
    t0 = time.perf_counter()
    lp = np.log(np.maximum(np.asarray(vc.pdf(te), dtype=np.float64), common.LOG_FLOOR))
    inference_s = time.perf_counter() - t0
    return lp, fit_s, inference_s, extra
  if arm == "vdc_shared":
    assert server is not None
    n_calls = len(server.calls)
    vine = SharedVinecop(structure, device="cpu")
    t0 = time.perf_counter()
    vine.fit(tr_t, fit_edge=lambda t, e, u_e, x_e=None, var_types=("c", "c"): VdcBicop.from_data(u_e))
    fit_s = time.perf_counter() - t0
    with torch.inference_mode():
      t0 = time.perf_counter()
      lp = vine.logpdf(te_t).detach().cpu().numpy().astype(np.float64)
      inference_s = time.perf_counter() - t0
    calls = server.calls[n_calls:]
    extra = {"vdc_server_s": sum(c["server_s"] for c in calls),
             "vdc_projection_s": sum(c["projection_s"] for c in calls),
             "vdc_ipc_s": sum(c["ipc_s"] for c in calls), "vdc_calls": len(calls)}
    return lp, fit_s, inference_s, extra
  assert label is not None and torch_seed is not None
  controls = e3_npcc.spec(label).controls(
    FitControlsRosenblattVinecop, device=device,
    projection_grid_size=e3_npcc.PROJECTION_GRID_SIZE, batch_size=None)
  reset_gpu_peaks()
  torch.manual_seed(torch_seed)
  t0 = time.perf_counter()
  vine = RosenblattVinecop(None, structure, device=device, context=NonSimplifiedContext())
  vine.fit(tr_t, controls)
  if device == "cuda":
    torch.cuda.synchronize()
  fit_s = time.perf_counter() - t0
  with torch.inference_mode():
    t0 = time.perf_counter()
    lp = vine.logpdf(te_t).detach().cpu().numpy().astype(np.float64)
    inference_s = time.perf_counter() - t0
  extra = {"torch_seed": torch_seed, "gpu_peak_reserved_mb": memory_usage()["gpu_peak_reserved_mb"]}
  del vine
  release_gpu()
  return lp, fit_s, inference_s, extra


def warm_up(label: str, tr: np.ndarray, device: str) -> None:
  """One untimed fit on three columns, as ``e3d_npcc_fit.py`` warms up."""
  controls = e3_npcc.spec(label).controls(
    FitControlsRosenblattVinecop, device=device,
    projection_grid_size=e3_npcc.PROJECTION_GRID_SIZE, batch_size=None)
  RosenblattVinecop(None, pv.RVineStructure.from_order([1, 2, 3]), device=device,
                    context=NonSimplifiedContext()).fit(
                      torch.as_tensor(tr[:, :3], dtype=torch.float64), controls)
  release_gpu()


def main() -> None:  # noqa: PLR0915 - one linear protocol
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--folder", type=Path, default=E3D_RESULTS,
                 help="E3d's folder, where truths.json and samples.npz are read from")
  p.add_argument("--out", type=Path, required=True, help="where cells/ and the CSVs are written")
  p.add_argument("--arms", nargs="+", required=True)
  p.add_argument("--designs", nargs="+", default=["decay"], choices=["decay", "strong"])
  p.add_argument("--dims", type=int, nargs="+", default=[5, 10])
  p.add_argument("--seeds", type=int, nargs="+", default=None, help="seeds to run (default: all ten)")
  p.add_argument("--k", type=int, default=K, help="members per mixture")
  p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
  p.add_argument("--gpu-is-free", action="store_true")
  p.add_argument("--vdc-core", type=int, default=None)
  a = p.parse_args()

  labels = {arm: parse_arm(arm) for arm in a.arms}
  for arm, label in labels.items():
    kind = e3_npcc.ESTIMATORS[label][2] if label else "cpu"
    if (kind == "gpu") != (a.device == "cuda"):
      raise SystemExit(f"{arm} must run with --device {'cuda' if kind == 'gpu' else 'cpu'}")
  if a.device == "cuda" and not a.gpu_is_free:
    raise SystemExit("--device cuda requires --gpu-is-free: the GPU is shared")
  if a.k < 1:
    raise SystemExit(f"--k must be at least 1; got {a.k}")
  if a.out.resolve() == a.folder.resolve():
    raise SystemExit(f"--out must differ from {a.folder}, which E3d's rows are cited from")

  e3d_structure.self_check(pv)
  info = json.loads((a.folder / "truths.json").read_text())
  samples = a.folder / "samples.npz"
  if common.sha256_file(samples) != info["samples_sha256"]:
    raise SystemExit(f"{samples} must match the sha256 recorded by e3d_truth.py")
  with np.load(samples) as f:
    data = {k: f[k] for k in f.files}
  prov = common.provenance(__file__, a.device)
  repo = {"script_repo": str(SCRIPT_REPO), "script_repo_commit": common.git_short(SCRIPT_REPO),
          "script_repo_dirty_paths": ";".join(common.git_dirty_paths(SCRIPT_REPO, "src", "scripts")),
          "npcc_module": str(Path(npcc.__file__).resolve().parent)}

  server = None
  if "vdc_shared" in a.arms:
    server = VdcServer(cpu_core=a.vdc_core)
    VdcBicop.server = server
  warmed = {arm: labels[arm] is None for arm in a.arms}
  try:
    for design, d, seed in zip(data["design"], data["d"], data["seed"]):
      design, d, seed = str(design), int(d), int(seed)
      if design not in a.designs or d not in a.dims or (a.seeds and seed not in a.seeds):
        continue
      key = f"{design}_d{d}_s{seed}"
      todo = [arm for arm in a.arms if not (a.out / "cells" / arm / f"{key}.json").exists()]
      if not todo:
        continue
      tr, te, logtrue = data[f"{key}_train"], data[f"{key}_test"], data[f"{key}_logtrue"]
      truth = e3d_structure.edges_from_matrix(data[f"{key}_true_matrix"])
      structures, structure_seeds, selection_s = structures_for(
        torch.as_tensor(tr, dtype=torch.float64), d, seed, a.k)
      matrices = [np.asarray(s.matrix).tolist() for s in structures]
      check_structures(a.out, key, matrices)
      recs = [e3d_structure.recovery(e3d_structure.edges_from_matrix(np.asarray(m)), truth)
              for m in matrices]

      for arm in todo:
        label = labels[arm]
        est = e3_npcc.spec(label) if label else None
        member_dir = a.out / "cells" / arm / "members"
        member_dir.mkdir(parents=True, exist_ok=True)
        load_before = common.load_average()
        logs, members = [], []
        for j, structure in enumerate(structures, start=1):
          stem = member_dir / f"{key}_m{j}"
          if stem.with_suffix(".json").exists():
            member = json.loads(stem.with_suffix(".json").read_text())
            logs.append(np.load(stem.with_suffix(".npy")))
            members.append(member)
            continue
          if not warmed[arm]:
            warm_up(label, tr, a.device)
            warmed[arm] = True
          torch_seed = member_torch_seed(design, d, seed, j, est.estimator_id) if est else None
          lp, fit_s, inference_s, extra = fit_member(
            arm, label, structure, tr, te, device=a.device, torch_seed=torch_seed, server=server)
          member = {"member": j, "structure_matrix": matrices[j - 1],
                    "structure_seed": structure_seeds[j - 1], "selection_s": selection_s[j - 1],
                    "fit_s": fit_s, "inference_s": inference_s, **extra,
                    "excess_nll": excess_nll(lp, logtrue, np.isfinite(lp))[0],
                    "load_avg_1m": common.load_average(), "started_at": prov["started_at"]}
          np.save(stem.with_suffix(".npy"), lp)
          common.write_json(member, stem.with_suffix(".json"))
          logs.append(lp)
          members.append(member)
          print(f"{arm} {key} member {j}/{a.k}: excess {member['excess_nll']:.4f} (tree-1 "
                f"recovery {recs[j - 1]['per_tree'][0]:.2f}) fit {fit_s:.1f} s inf "
                f"{inference_s:.1f} s", flush=True)

        member_logs = np.stack(logs)
        t0 = time.perf_counter()
        scores = score_members(member_logs, logtrue)
        mixing_s = time.perf_counter() - t0
        mixture = scores.pop("mixture")
        path = a.out / "cells" / arm / f"{key}.json"
        np.save(path.with_suffix(".npy"), mixture.astype(np.float32))  # raw, NaN kept
        np.save(path.parent / f"{key}_members.npy", member_logs.astype(np.float32))
        if label is None:
          machinery = ("pyvinecopulib 1.0 compiled Vinecop.from_data, structure given"
                       if arm != "vdc_shared" else
                       "pyvinecopulib 1.0 VinecopBase (e3_vine.SharedVinecop), simplified")
          pair = {"parametric": "parametric family selection (BIC), VDC's fit_bicop set",
                  "tll": "TLL, pyvinecopulib defaults",
                  "vdc_shared": "e3_vdc_pair.VdcBicop (VDC grid, padded, TorchTllBicop norm_maxiter=0)"}[arm]
          extra = {}
          if arm == "vdc_shared":
            extra = {f"{c}": sum(m[c] for m in members)
                     for c in ("vdc_server_s", "vdc_projection_s", "vdc_ipc_s", "vdc_calls")}
            extra["vdc_server_code_sha256"] = json.dumps(server.info["code_sha256"], sort_keys=True)
        else:
          machinery = "npcc.RosenblattVinecop, NonSimplifiedContext"
          pair = f"npcc.RosenblattBicop ({label})"
          extra = {"estimator": label, "estimator_id": est.estimator_id, "backend": est.backend,
                   "transform": est.transform, "model_version": e3_npcc.ESTIMATORS[label][1],
                   "simplification": "non-simplified",
                   "member_torch_seeds": json.dumps([m["torch_seed"] for m in members]),
                   "member_gpu_peak_reserved_mb": json.dumps(
                     [m["gpu_peak_reserved_mb"] for m in members])}
        row = {
          "experiment": "E10", "setting": "E3d", "design": design, "d": d, "seed": seed,
          "n_train": int(tr.shape[0]), "n_test": int(te.shape[0]), "n_edges": d * (d - 1) // 2,
          "method": arm, "role": "mixture", "k": a.k,
          "structure": "mixture: TLL's Dissmann structure, then weighted random (|tau|) structures",
          "structure_matrices": json.dumps(matrices),
          "structure_seeds": json.dumps(structure_seeds),
          "n_distinct_structures": distinct(matrices),
          "member_recovery_per_tree": json.dumps([r["per_tree"] for r in recs]),
          "member_recovery_tree1": json.dumps([r["per_tree"][0] for r in recs]),
          "member_recovery_overall": json.dumps([r["overall"] for r in recs]),
          "member_structure_equals_truth": json.dumps([r["equal"] for r in recs]),
          "vine_machinery": machinery, "pair_copula": pair, "env": "npcc",
          "true_loglik": float(np.mean(logtrue)), **scores,
          "fit_s": sum(m["selection_s"] + m["fit_s"] for m in members),
          "selection_s": sum(m["selection_s"] for m in members),
          "inference_s": sum(m["inference_s"] for m in members) + mixing_s,
          "selection_in_fit_s": True,
          "member_fit_s": json.dumps([m["fit_s"] for m in members]),
          "member_selection_s": json.dumps([m["selection_s"] for m in members]),
          "member_inference_s": json.dumps([m["inference_s"] for m in members]),
          **extra,
          "torch_threads": torch.get_num_threads(), "cpu_cores": len(os.sched_getaffinity(0)),
          "samples_sha256": info["samples_sha256"], "load_avg_1m": load_before, **repo, **prov,
        }
        common.write_json(row, path)
        print(f"{arm} {key}: mixture {row['excess_nll']:.4f}, member 1 "
              f"{row['member1_excess_nll_common']:.4f}, members' mean "
              f"{row['members_mean_excess_nll_common']:.4f}, best member "
              f"{row['best_member_excess_nll_common']:.4f} ({row['n_distinct_structures']} "
              f"distinct structures)", flush=True)
  finally:
    if server is not None:
      server.close()
  for arm in a.arms:
    rows = [json.loads(q.read_text()) for q in sorted((a.out / "cells" / arm).glob("*.json"))]
    if rows:
      common.write_csv(rows, a.out / f"mixture_{arm}.csv")
  write_summary(a.out)


if __name__ == "__main__":
  main()
