#!/usr/bin/env python3
"""E3d, NPCC side: unconditional vines with an unknown structure.

Run in the NPCC environment, from the repository root, after ``e3d_truth.py``:

    cd /home/thibault/npcc
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c 11 nice -n 10 uv run --no-sync python scripts/paper/e3/e3d_npcc_fit.py \
        --arms parametric tll parametric_true tll_true vdc_true --vdc-core 11
    ... e3d_npcc_fit.py --arms npcc-catboost-logit-random-ns npcc-catboost-logit-tll-ns \
        npcc-catboost-logit-random-s npcc-catboost-logit-true-s
    # GPU (main-text TabPFN v3), once allowed; d = 10 on the decay design and
    # five seeds only (the paper lead's scope):
    ... e3d_npcc_fit.py --arms npcc-tabpfn-quantiles-v3-logit-random-ns ... --dims 5 --device cuda --gpu-is-free
    ... e3d_npcc_fit.py --arms ... --designs decay --dims 10 --seeds 101 202 303 404 505 --device cuda --gpu-is-free

**Competitors**, each with its own default structure selection, as a user
would run them (``vdc``, VDC's own greedy D-vine, is ``e3d_vdc_side.py``):

- ``parametric``: ``pyvinecopulib.Vinecop.from_data`` with Dissmann's
  algorithm and parametric family selection by BIC (VDC's ``fit_bicop``
  family set, rotations), otherwise default controls, one thread;
- ``tll``: the same with TLL as the only family and default controls.

``fit_s`` includes the structure selection.

**NPCC**, with an E1 logit estimator, in ``npcc.RosenblattVinecop``:

- ``npcc-<est>-random-ns``: a random R-vine structure drawn independently of
  the truth (``e3d_truth.py``), *non-simplified* -- each edge conditions on
  its conditioning set (``NonSimplifiedContext``, as E2's ``random`` arm);
- ``npcc-<est>-tll-ns``: the structure ``tll``'s Dissmann selection picks
  (recomputed here from the same data and controls; its seconds are in
  ``fit_s`` and in ``selection_s``), non-simplified;
- ``npcc-<est>-random-s``: the random structure, *simplified* (an ablation).

**References on the true structure** (an oracle no arm has):
``parametric_true`` and ``tll_true`` (the compiled vine with the same
controls and ``structure=``), ``vdc_true`` (VDC's pair copulas in
``e3_vine.SharedVinecop``, as in E3c) and ``npcc-<est>-true-s``
(simplified).

**Metrics.** Excess NLL = mean of ``log c_true - log max(c_hat, 1e-300)``.
A density an arm cannot evaluate (NaN) is not floored: the row's
``excess_nll`` is over the arm's own finite points (``n_nonfinite`` counts
the rest), the raw log-densities are kept, and ``e3_collect.py`` scores every
arm on the test points where all arms and the truth are finite; ``fit_s`` and
``inference_s``; and for every structure an arm used, its distance to the
truth (``e3d_structure.recovery``: the share of true edges recovered in each
tree and overall, and whether it is the truth).

**Outputs** (``results/e3d_unknown_structure/``): one JSON row and one
float32 ``.npy`` of raw test-point log-densities (NaN kept) per arm and cell in
``cells/<arm>/<design>_d<d>_s<seed>``, and ``npcc_<arm>.csv``.
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
from e3_vdc_pair import VdcBicop, VdcServer  # noqa: E402
from e3_vine import SharedVinecop  # noqa: E402
from npcc import FitControlsRosenblattVinecop, RosenblattVinecop  # noqa: E402
from npcc.experiments._study import derive_seed  # noqa: E402
from pyvinecopulib.core import NonSimplifiedContext, SimplifiedContext  # noqa: E402

RESULTS = common.NPCC_REPO / "results" / "e3d_unknown_structure"
#: VDC's ``fit_bicop`` parametric family set.
PARAMETRIC_FAMILIES = ("indep", "gaussian", "student", "clayton", "gumbel", "frank",
                       "joe", "bb1", "bb6", "bb7", "bb8", "tawn")
COMPETITORS = ("parametric", "tll", "parametric_true", "tll_true", "vdc_true")
NPCC_VARIANTS = {"random-ns": ("random", "non-simplified"), "tll-ns": ("tll", "non-simplified"),
                 "random-s": ("random", "simplified"), "true-s": ("true", "simplified")}
ROLES = {"parametric": "arm", "tll": "arm", "parametric_true": "reference",
         "tll_true": "reference", "vdc_true": "reference"}


def compiled_controls(name: str) -> pv.FitControlsVinecop:
  """Default controls but for the family set (and BIC for parametric)."""
  if name.startswith("parametric"):
    return pv.FitControlsVinecop(family_set=[getattr(pv.families, f) for f in PARAMETRIC_FAMILIES],
                                 selection_criterion="bic", num_threads=1)
  return pv.FitControlsVinecop(family_set=[pv.families.tll], num_threads=1)


def parse_npcc_arm(arm: str) -> tuple[str, str] | None:
  """``npcc-<estimator>-<variant>`` into its estimator label and variant."""
  if not arm.startswith("npcc-"):
    return None
  for variant in NPCC_VARIANTS:
    if arm.endswith(f"-{variant}"):
      label = arm[len("npcc-"):-len(variant) - 1]
      if label in e3_npcc.ESTIMATORS:
        return label, variant
  return None


def main() -> None:  # noqa: PLR0915 - one linear protocol
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--folder", type=Path, default=RESULTS)
  p.add_argument("--arms", nargs="+", required=True)
  p.add_argument("--designs", nargs="+", default=["decay", "strong"], choices=["decay", "strong"])
  p.add_argument("--dims", type=int, nargs="+", default=[5, 10])
  p.add_argument("--seeds", type=int, nargs="+", default=None,
                 help="seeds to run (default: all ten); the d = 10 TabPFN arms use five")
  p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
  p.add_argument("--gpu-is-free", action="store_true")
  p.add_argument("--vdc-core", type=int, default=None)
  a = p.parse_args()

  for arm in a.arms:
    parsed = parse_npcc_arm(arm)
    if arm not in COMPETITORS and parsed is None:
      raise SystemExit(f"arm must be one of {COMPETITORS} or npcc-<E1 estimator>-"
                       f"<{'|'.join(NPCC_VARIANTS)}>; got {arm}")
    kind = e3_npcc.ESTIMATORS[parsed[0]][2] if parsed else "cpu"
    if (kind == "gpu") != (a.device == "cuda"):
      raise SystemExit(f"{arm} must run with --device {'cuda' if kind == 'gpu' else 'cpu'}")
  if a.device == "cuda" and not a.gpu_is_free:
    raise SystemExit("--device cuda requires --gpu-is-free: the GPU is shared")

  e3d_structure.self_check(pv)
  info = json.loads((a.folder / "truths.json").read_text())
  samples = a.folder / "samples.npz"
  if common.sha256_file(samples) != info["samples_sha256"]:
    raise SystemExit(f"{samples} must match the sha256 recorded by e3d_truth.py")
  with np.load(samples) as f:
    data = {k: f[k] for k in f.files}
  prov = common.provenance(__file__, a.device)
  floor = np.log(common.LOG_FLOOR)

  server = None
  if "vdc_true" in a.arms:
    server = VdcServer(cpu_core=a.vdc_core)
    VdcBicop.server = server
  try:
    for arm in a.arms:
      out_dir = a.folder / "cells" / arm
      out_dir.mkdir(parents=True, exist_ok=True)
      parsed = parse_npcc_arm(arm)
      warmed = parsed is None
      for design, d, seed in zip(data["design"], data["d"], data["seed"]):
        design, d, seed = str(design), int(d), int(seed)
        if design not in a.designs or d not in a.dims or (a.seeds and seed not in a.seeds):
          continue
        key = f"{design}_d{d}_s{seed}"
        path = out_dir / f"{key}.json"
        if path.exists():
          continue
        tr, te, logtrue = data[f"{key}_train"], data[f"{key}_test"], data[f"{key}_logtrue"]
        s_true = pv.RVineStructure.from_matrix(data[f"{key}_true_matrix"])
        s_rand = pv.RVineStructure.from_matrix(data[f"{key}_random_matrix"])
        truth = e3d_structure.edges_from_matrix(data[f"{key}_true_matrix"])
        tr_t = torch.as_tensor(tr, dtype=torch.float64)
        te_t = torch.as_tensor(te, dtype=torch.float64)
        extra: dict = {}
        selection_s = None
        load_before = common.load_average()

        if parsed is None:
          if arm in ("parametric", "tll", "parametric_true", "tll_true"):
            given = arm.endswith("_true")
            ctl = compiled_controls(arm)
            t0 = time.perf_counter()
            vc = (pv.Vinecop.from_data(tr, structure=s_true, controls=ctl) if given
                  else pv.Vinecop.from_data(tr, controls=ctl))
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            lp = np.log(np.maximum(np.asarray(vc.pdf(te), dtype=np.float64), common.LOG_FLOOR))
            inference_s = time.perf_counter() - t0
            used = np.asarray(vc.matrix)
            structure = "true (given)" if given else "selected: Dissmann (pyvinecopulib default)"
            machinery = "pyvinecopulib 1.0 compiled Vinecop.from_data"
            pair = ("parametric family selection (BIC), VDC's fit_bicop set" if arm.startswith("parametric")
                    else "TLL, pyvinecopulib defaults")
          else:  # vdc_true
            n_calls = len(server.calls)
            vine = SharedVinecop(s_true, device="cpu")
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
                     "vdc_ipc_s": sum(c["ipc_s"] for c in calls), "vdc_calls": len(calls),
                     "vdc_server_code_sha256": json.dumps(server.info["code_sha256"], sort_keys=True)}
            used = np.asarray(s_true.matrix)
            structure = "true (given)"
            machinery = "pyvinecopulib 1.0 VinecopBase (e3_vine.SharedVinecop), simplified"
            pair = "e3_vdc_pair.VdcBicop (VDC grid, padded, TorchTllBicop norm_maxiter=0)"
          role = ROLES[arm]
        else:
          label, variant = parsed
          which, simplification = NPCC_VARIANTS[variant]
          est = e3_npcc.spec(label)
          controls = est.controls(FitControlsRosenblattVinecop, device=a.device,
                                  projection_grid_size=e3_npcc.PROJECTION_GRID_SIZE, batch_size=None)
          context = NonSimplifiedContext() if simplification == "non-simplified" else SimplifiedContext()
          if not warmed:
            RosenblattVinecop(None, pv.RVineStructure.from_order([1, 2, 3]), device=a.device,
                              context=context).fit(tr_t[:, :3], controls)
            warmed = True
          t0 = time.perf_counter()
          if which == "tll":
            structure_used = pv.Vinecop.from_data(tr, controls=compiled_controls("tll")).structure
          else:
            structure_used = s_true if which == "true" else s_rand
          selection_s = time.perf_counter() - t0
          torch.manual_seed(derive_seed("E3d", design, d, seed, variant, est.estimator_id))
          vine = RosenblattVinecop(None, structure_used, device=a.device, context=context)
          vine.fit(tr_t, controls)
          if a.device == "cuda":
            torch.cuda.synchronize()
          fit_s = time.perf_counter() - t0
          with torch.inference_mode():
            t0 = time.perf_counter()
            lp = vine.logpdf(te_t).detach().cpu().numpy().astype(np.float64)
            inference_s = time.perf_counter() - t0
          used = np.asarray(structure_used.matrix)
          structure = {"random": "random (independent of the truth)",
                       "tll": "selected: TLL's Dissmann structure", "true": "true (given)"}[which]
          machinery = f"npcc.RosenblattVinecop, {type(context).__name__}"
          pair = f"npcc.RosenblattBicop ({label})"
          role = "reference" if which == "true" else ("ablation" if variant == "random-s" else "arm")
          extra = {"estimator": label, "estimator_id": est.estimator_id, "backend": est.backend,
                   "transform": est.transform, "model_version": e3_npcc.ESTIMATORS[label][1],
                   "simplification": simplification}

        np.save(path.with_suffix(".npy"), lp.astype(np.float32))  # raw, NaN kept
        finite = np.isfinite(lp)
        n_nonfinite = int(np.sum(~finite))
        ratio = (logtrue - np.maximum(lp, floor))[finite]
        rec = e3d_structure.recovery(e3d_structure.edges_from_matrix(used), truth)
        row = {
          "experiment": "E3d", "design": design, "d": d, "seed": seed,
          "n_train": int(tr.shape[0]), "n_test": int(te.shape[0]), "n_edges": d * (d - 1) // 2,
          "method": arm, "role": role, "structure": structure,
          "structure_matrix": json.dumps(used.tolist()),
          "recovery_per_tree": json.dumps(rec["per_tree"]), "recovery_tree1": rec["per_tree"][0],
          "recovery_overall": rec["overall"], "structure_equals_truth": rec["equal"],
          "vine_machinery": machinery, "pair_copula": pair, "env": "npcc",
          "test_loglik": float(np.mean(np.maximum(lp, floor)[finite])),
          "true_loglik": float(np.mean(logtrue)),
          "excess_nll": float(np.mean(ratio)),
          "excess_nll_mc_se": float(np.std(ratio, ddof=1) / np.sqrt(ratio.size)),
          "n_nonfinite": n_nonfinite, "excess_nll_scope": "own finite points",
          "fit_s": fit_s, "selection_s": selection_s, "inference_s": inference_s,
          "selection_in_fit_s": structure.startswith("selected"), **extra,
          "torch_threads": torch.get_num_threads(), "cpu_cores": len(os.sched_getaffinity(0)),
          "samples_sha256": info["samples_sha256"], "load_avg_1m": load_before, **prov,
        }
        common.write_json(row, path)
        print(f"{arm} {key}: excess {row['excess_nll']:.4f} (tree-1 recovery "
              f"{rec['per_tree'][0]:.2f}) fit {fit_s:.1f} s inf {inference_s:.1f} s", flush=True)
      rows = [json.loads(q.read_text()) for q in sorted(out_dir.glob("*.json"))]
      if rows:
        common.write_csv(rows, a.folder / f"npcc_{arm}.csv")
  finally:
    if server is not None:
      server.close()


if __name__ == "__main__":
  main()
