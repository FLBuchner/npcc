#!/usr/bin/env python3
"""E11 on E3d: chain-rule copula densities mixed over variable orders.

Run in the NPCC environment, from the repository root, after ``e3d_truth.py``:

    C=scripts/paper/e3/e3d_chain.py
    O=results/e11_e3d_<TAG>
    uv run --no-sync python $C --designs decay strong --dims 5 --out $O \\
        --device cuda --gpu-is-free

**Question.** A referee asks why a vine is needed at all. The backend that
fits one Rosenblatt direction can condition on every earlier score, so the
copula density can be written down by the chain rule along a variable order
``pi``, ``c(u) = prod_{k=2..d} f(u_pi(k) | u_pi(<k))``: ``d - 1`` regressions
and no structure (``npcc.experiments.chain.ChainCopula``). There is no ``x``
here, so the features are the earlier scores alone.

**Orders.** Per cell, ``--k`` (5) orders drawn uniformly from the ``d!``
permutations (``npcc.experiments.chain.draw_order``), order ``j`` from the
seed ``derive_seed("E3d", "chain", design, d, seed, j)``. They are a function
of the cell alone, so every estimator and process mixes the same ones.

**Estimator.** ``--estimator``, an E1 label read from E1's config by
``e3_npcc.spec`` (TabPFN v3 at the probit transform by default). Factor ``f``
of order ``j`` is one Rosenblatt direction of it, with the torch seed
``derive_seed("E3d", "chain", design, d, seed, j, f, estimator_id)``.

**Metrics.** As in E3d, and scored by ``e3d_mixture.score_members``, the
orders taking the place of its members: excess NLL = mean of
``log c_true - log max(c_hat, 1e-300)`` at the 20,000 test points. The
mixture's log-density is ``logsumexp_j log c_j - log K``
(``vine_runner.mixture_log_density``), NaN kept, and a NaN in any order makes
the mixture NaN at that point. Per cell: the mixture's excess NLL on its own
finite points, and on the mixture's finite points each order's, their mean
and the best order's. The log of a mean is at least the mean of the logs, so
the mixture scores at most the orders' mean.

Orders are fitted one at a time, and each is evaluated at the test points and
released before the next is fitted, so one order's factors occupy the device
at a time. A finished order is saved, and an interrupted cell resumes at its
next order. One untimed warm-up fit precedes the first.

**Outputs** (under ``--out``, which must differ from E3d's folder): per cell,
``cells/chain-<estimator>/<design>_d<d>_s<seed>.json`` (one row), ``.npy``
(the mixture's test-point log-densities, float32, NaN kept) and
``_members.npy`` (the orders', ``(K, n_test)``, float32); each finished order
in ``cells/chain-<estimator>/members/<key>_m<j>.json`` and ``.npy``
(float64); ``chain_<estimator>.csv`` (the rows); and
``chain_<estimator>_summary.csv``, per design and d over seeds.

**Seeds.** ``--seeds`` takes an explicit list, and every run resumes by
cell: a cell whose row exists is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pyvinecopulib as pv
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_common as common
import e3_npcc
import e3d_structure
from e3d_mixture import SCRIPT_REPO, excess_nll, score_members
from e3d_npcc_fit import RESULTS as E3D_RESULTS

import npcc
from npcc import FitControlsRosenblattBicop
from npcc.experiments._study import (
  derive_seed,
  memory_usage,
  release_gpu,
  reset_gpu_peaks,
)
from npcc.experiments.chain import ChainCopula, draw_order
from npcc.experiments.config import EstimatorSpec

#: Orders per mixture, as in the vine study's ``chain`` arm.
K = 5
#: The main-text TabPFN, E1's label.
ESTIMATOR = "tabpfn-quantiles-v3-probit"


def orders_for(
  design: str, d: int, seed: int, k: int
) -> tuple[list[list[int]], list[int]]:
  """The cell's ``k`` variable orders, and the seed each was drawn from."""
  seeds = [
    derive_seed("E3d", "chain", design, d, seed, j) for j in range(1, k + 1)
  ]
  return [draw_order(d, s) for s in seeds], seeds


def factor_seeds(
  design: str, d: int, seed: int, j: int, estimator_id: str
) -> list[int]:
  """Order ``j``'s torch seeds, one per factor ``f = 2, ..., d``."""
  return [
    derive_seed("E3d", "chain", design, d, seed, j, f, estimator_id)
    for f in range(2, d + 1)
  ]


def controls(est: EstimatorSpec, device: str) -> FitControlsRosenblattBicop:
  """The estimator's pair controls, as E3d's NPCC arms build them."""
  return est.controls(
    FitControlsRosenblattBicop,
    device=device,
    projection_grid_size=e3_npcc.PROJECTION_GRID_SIZE,
    batch_size=None,
  )


def fit_order(
  est: EstimatorSpec,
  order: list[int],
  tr: np.ndarray,
  te: np.ndarray,
  *,
  device: str,
  seeds: list[int],
) -> tuple[np.ndarray, float, float, dict[str, object]]:
  """Fit one order, evaluate it at ``te``, release it.

  Returns its raw test-point log-densities, its fit and inference seconds,
  and its GPU peak.
  """
  reset_gpu_peaks()
  t0 = time.perf_counter()
  chain = ChainCopula(order, controls(est, device))
  chain.fit(torch.as_tensor(tr, dtype=torch.float64), seeds=seeds)
  if device == "cuda":
    torch.cuda.synchronize()
  fit_s = time.perf_counter() - t0
  with torch.inference_mode():
    t0 = time.perf_counter()
    lp = chain.logpdf(torch.as_tensor(te, dtype=torch.float64))
    lp = lp.detach().cpu().numpy().astype(np.float64)
    inference_s = time.perf_counter() - t0
  extra = {"gpu_peak_reserved_mb": memory_usage()["gpu_peak_reserved_mb"]}
  del chain
  release_gpu()
  return lp, fit_s, inference_s, extra


def warm_up(est: EstimatorSpec, tr: np.ndarray, device: str) -> None:
  """One untimed fit on three columns, as ``e3d_npcc_fit.py`` warms up."""
  ChainCopula([1, 2, 3], controls(est, device)).fit(
    torch.as_tensor(tr[:, :3], dtype=torch.float64)
  )
  release_gpu()


def run_cell(
  est: EstimatorSpec,
  key: str,
  tr: np.ndarray,
  te: np.ndarray,
  logtrue: np.ndarray,
  *,
  k: int,
  device: str,
  cell_dir: Path,
  extra: Mapping[str, object],
) -> dict[str, object]:
  """Fit and score one cell's ``k`` orders; write and return its row.

  ``key`` is ``<design>_d<d>_s<seed>``. Each finished order is saved under
  ``cell_dir/members`` and read back on a rerun, which must have drawn the
  same order. ``extra`` (the provenance) closes the row, which is written
  last, so a cell whose row exists is finished.
  """
  design, dim, s = key.split("_")
  d, seed = int(dim.removeprefix("d")), int(s.removeprefix("s"))
  if tr.shape[1] != d:
    raise ValueError(f"{key} must have {d} columns; got {tr.shape[1]}")
  orders, order_seeds = orders_for(design, d, seed, k)
  member_dir = cell_dir / "members"
  member_dir.mkdir(parents=True, exist_ok=True)
  logs, members = [], []
  for j, order in enumerate(orders, start=1):
    stem = member_dir / f"{key}_m{j}"
    if stem.with_suffix(".json").exists():
      member = json.loads(stem.with_suffix(".json").read_text())
      if member["order"] != order:
        raise SystemExit(
          f"{stem}.json must hold the order {order}, which this cell draws; "
          f"got {member['order']}"
        )
      logs.append(np.load(stem.with_suffix(".npy")))
      members.append(member)
      continue
    seeds = factor_seeds(design, d, seed, j, est.estimator_id)
    lp, fit_s, inference_s, peak = fit_order(
      est, order, tr, te, device=device, seeds=seeds
    )
    member = {
      "member": j,
      "order": order,
      "order_seed": order_seeds[j - 1],
      "factor_seeds": seeds,
      "fit_s": fit_s,
      "inference_s": inference_s,
      **peak,
      "excess_nll": excess_nll(lp, logtrue, np.isfinite(lp))[0],
      "load_avg_1m": common.load_average(),
    }
    np.save(stem.with_suffix(".npy"), lp)
    common.write_json(member, stem.with_suffix(".json"))
    logs.append(lp)
    members.append(member)
    print(
      f"chain {key} order {j}/{k} {order}: excess "
      f"{member['excess_nll']:.4f} fit {fit_s:.1f} s inf {inference_s:.1f} s",
      flush=True,
    )

  member_logs = np.stack(logs)
  t0 = time.perf_counter()
  scores = score_members(member_logs, logtrue)
  mixing_s = time.perf_counter() - t0
  mixture = scores.pop("mixture")
  path = cell_dir / f"{key}.json"
  np.save(path.with_suffix(".npy"), mixture.astype(np.float32))
  np.save(cell_dir / f"{key}_members.npy", member_logs.astype(np.float32))
  row: dict[str, object] = {
    "experiment": "E11",
    "setting": "E3d",
    "design": design,
    "d": d,
    "seed": seed,
    "n_train": int(tr.shape[0]),
    "n_test": int(te.shape[0]),
    "method": cell_dir.name,
    "role": "chain",
    "k": k,
    "n_factors": k * (d - 1),
    "structure": "none: chain-rule densities along uniformly random orders",
    "orders": json.dumps(orders),
    "order_seeds": json.dumps(order_seeds),
    "n_distinct_orders": len({tuple(o) for o in orders}),
    "pair_copula": f"npcc.experiments.chain.ChainCopula ({est.label})",
    "estimator": est.label,
    "estimator_id": est.estimator_id,
    "backend": est.backend,
    "transform": est.transform,
    "feature_transform": est.feature_transform,
    "env": "npcc",
    "true_loglik": float(np.mean(logtrue)),
    **scores,
    "fit_s": sum(m["fit_s"] for m in members),
    "inference_s": sum(m["inference_s"] for m in members) + mixing_s,
    "member_fit_s": json.dumps([m["fit_s"] for m in members]),
    "member_inference_s": json.dumps([m["inference_s"] for m in members]),
    "member_factor_seeds": json.dumps([m["factor_seeds"] for m in members]),
    "member_gpu_peak_reserved_mb": json.dumps(
      [m["gpu_peak_reserved_mb"] for m in members]
    ),
    "torch_threads": torch.get_num_threads(),
    "cpu_cores": len(os.sched_getaffinity(0)),
    **extra,
  }
  common.write_json(row, path)
  return row


#: The summary's columns: per design and d, the mixture against its orders on
#: the mixture's finite points, over seeds.
SUMMARY = (
  "excess_nll",
  "members_mean_excess_nll_common",
  "best_member_excess_nll_common",
  "mixture_minus_members_mean",
  "mixture_minus_best_member",
  "fit_s",
  "inference_s",
)


def write_summary(out: Path, arm: str) -> pd.DataFrame | None:
  """``<arm>.csv`` (the rows) and ``<arm>_summary.csv``, from every cell."""
  rows = [
    json.loads(q.read_text())
    for q in sorted((out / "cells" / arm).glob("*.json"))
  ]
  if not rows:
    return None
  common.write_csv(rows, out / f"{arm.replace('-', '_', 1)}.csv")
  frame = pd.DataFrame(rows)
  grouped = frame.groupby(["design", "d"])
  summary = grouped[list(SUMMARY)].mean()
  for col in ("mixture_minus_members_mean", "mixture_minus_best_member"):
    summary[f"{col}_se"] = grouped[col].sem()
  summary["seeds"] = grouped.size()
  summary["mixture_beats_orders_mean"] = grouped[
    "mixture_minus_members_mean"
  ].apply(lambda v: int((v < 0).sum()))
  summary["n_excluded"] = grouped["n_nonfinite"].sum()
  summary = summary.reset_index()
  summary.to_csv(out / f"{arm.replace('-', '_', 1)}_summary.csv", index=False)
  return summary


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument(
    "--folder",
    type=Path,
    default=E3D_RESULTS,
    help="E3d's folder, where truths.json and samples.npz are read from",
  )
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--estimator", default=ESTIMATOR, help="an E1 label")
  p.add_argument(
    "--designs", nargs="+", default=["decay"], choices=["decay", "strong"]
  )
  p.add_argument("--dims", type=int, nargs="+", default=[5, 10])
  p.add_argument(
    "--seeds", type=int, nargs="+", default=None, help="default: all ten"
  )
  p.add_argument("--k", type=int, default=K, help="orders per mixture")
  p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
  p.add_argument("--gpu-is-free", action="store_true")
  a = p.parse_args()

  if a.estimator not in e3_npcc.ESTIMATORS:
    raise SystemExit(
      f"--estimator must be one of {list(e3_npcc.ESTIMATORS)}; got "
      f"{a.estimator}"
    )
  kind = e3_npcc.ESTIMATORS[a.estimator][2]
  if (kind == "gpu") != (a.device == "cuda"):
    raise SystemExit(
      f"{a.estimator} must run with --device "
      f"{'cuda' if kind == 'gpu' else 'cpu'}"
    )
  if a.device == "cuda" and not a.gpu_is_free:
    raise SystemExit("--device cuda requires --gpu-is-free: the GPU is shared")
  if a.k < 1:
    raise SystemExit(f"--k must be at least 1; got {a.k}")
  if a.out.resolve() == a.folder.resolve():
    raise SystemExit(
      f"--out must differ from {a.folder}, which E3d's rows are cited from"
    )

  e3d_structure.self_check(pv)
  info = json.loads((a.folder / "truths.json").read_text())
  samples = a.folder / "samples.npz"
  if common.sha256_file(samples) != info["samples_sha256"]:
    raise SystemExit(
      f"{samples} must match the sha256 recorded by e3d_truth.py"
    )
  with np.load(samples) as f:
    data = {k: f[k] for k in f.files}
  prov = common.provenance(__file__, a.device)
  repo = {
    "script_repo": str(SCRIPT_REPO),
    "script_repo_commit": common.git_short(SCRIPT_REPO),
    "script_repo_dirty_paths": ";".join(
      common.git_dirty_paths(SCRIPT_REPO, "src", "scripts")
    ),
    "npcc_module": str(Path(npcc.__file__).resolve().parent),
  }

  est = e3_npcc.spec(a.estimator)
  arm = f"chain-{a.estimator}"
  cell_dir = a.out / "cells" / arm
  cell_dir.mkdir(parents=True, exist_ok=True)
  warmed = False
  for design, d, seed in zip(data["design"], data["d"], data["seed"]):
    design, d, seed = str(design), int(d), int(seed)
    if (
      design not in a.designs
      or d not in a.dims
      or (a.seeds and seed not in a.seeds)
    ):
      continue
    key = f"{design}_d{d}_s{seed}"
    path = cell_dir / f"{key}.json"
    if path.exists():
      continue
    tr, te = data[f"{key}_train"], data[f"{key}_test"]
    if not warmed:
      warm_up(est, tr, a.device)
      warmed = True
    row = run_cell(
      est,
      key,
      tr,
      te,
      data[f"{key}_logtrue"],
      k=a.k,
      device=a.device,
      cell_dir=cell_dir,
      extra={
        "model_version": e3_npcc.ESTIMATORS[a.estimator][1],
        "samples_sha256": info["samples_sha256"],
        "load_avg_1m": common.load_average(),
        **repo,
        **prov,
      },
    )
    print(
      f"{arm} {key}: mixture {row['excess_nll']:.4f}, orders' mean "
      f"{row['members_mean_excess_nll_common']:.4f}, best order "
      f"{row['best_member_excess_nll_common']:.4f}",
      flush=True,
    )
  summary = write_summary(a.out, arm)
  if summary is not None:
    with pd.option_context("display.width", 200):
      print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
  main()
