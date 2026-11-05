#!/usr/bin/env python3
"""E3c, NPCC side: VDC, TLL and NPCC in one vine machinery on the true structure.

Run in the NPCC environment, from the repository root, after
``e3c_vdc_side.py`` has written ``samples.npz``:

    cd /home/thibault/npcc
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c 13 nice -n 10 uv run --no-sync python scripts/paper/e3/e3c_npcc_fit.py \
        --arms vdc tll tll_cpp parametric --vdc-core 12
    ... e3c_npcc_fit.py --arms npcc-lightgbm-logit          # one CPU backend per process
    ... e3c_npcc_fit.py --arms npcc-catboost-logit rosenblatt-vinecop-check

**One vine machinery.** Every arm but the two C++ ones fits and evaluates a
``e3_vine.SharedVinecop``: pyvinecopulib 1.0's ``VinecopBase`` cascades on the
true structure (``RVineStructure.from_matrix`` of the matrix the VDC side
simulated from), simplified, with no structure selection anywhere. The arms
differ only in the pair copula each edge holds:

- ``vdc``: ``e3_vdc_pair.VdcBicop`` -- VDC's 64 x 64 grid from a density
  server in VDC's environment (m = 64, 30 projection iterations, the Table 1
  runner's call), padded and held by ``TorchTllBicop`` with
  ``norm_maxiter=0`` (adapted from pyvinecopulib PR #217);
- ``tll``: ``pyvinecopulib.torch.TorchTllBicop.from_data`` with
  ``FitControlsTorchBicop()`` -- pyvinecopulib's TLL defaults (constant,
  multiplier 1, a 30-point normal grid), no family selection;
- ``npcc-<estimator>``: ``npcc.RosenblattBicop`` with an E1 logit backend,
  no covariate.

And, in pyvinecopulib 1.0's compiled vine (``Vinecop.from_data(train,
structure=..., controls=...)``) on the same structure:

- ``parametric``: parametric family selection by BIC with rotations per edge,
  over VDC's ``fit_bicop`` family set, one thread;
- ``tll_cpp`` (a cross-check, not an arm): the compiled TLL vine, which the
  shared machinery's ``tll`` must reproduce to floating-point tolerance;
- ``rosenblatt-vinecop-check`` (d = 5 only, CatBoost; a cross-check):
  ``npcc.RosenblattVinecop`` with a ``SimplifiedContext``, which must equal
  ``npcc-catboost-logit`` in the shared vine.

**Metrics.** Held-out log-likelihood = mean over the 20,000 test points of
``log c_hat`` (floored at ``log 1e-300``; a density the arm cannot evaluate,
NaN, counts as the floor and is counted in ``n_nonfinite``), and excess NLL =
mean of ``log c_true - log c_hat``, the KL estimate; ``excess_nll_finite`` is
the same over the finite points only. ``fit_s`` is the vine fit along the
fixed structure (every edge's pair fit and the h-functions passed between
trees; nothing is selected); ``inference_s`` the log-density at the test
points. For ``vdc``, the seconds VDC's estimates took inside the server
(``vdc_server_s``, of which ``vdc_projection_s``) and the inter-process
overhead (``vdc_ipc_s``) are summed over the edges and reported separately.

**Outputs** (under ``--folder``): one JSON row per arm and cell in
``npcc_cells/<arm>/`` (so a run resumes), and ``npcc_<arm>.csv``.

**On timings.** VDC's published vine timings compared TLL with R-vine
structure selection on a CPU against VDC on a greedy D-vine on an H100. Here
the structure is fixed and true for every arm, nothing is selected, and each
arm states its device, so a time differs only through the pair-copula
estimator. All arms here run on one pinned CPU core.
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
from e3_vdc_pair import VdcBicop, VdcServer  # noqa: E402
from e3_vine import SharedVinecop  # noqa: E402

SPECIAL = ("vdc", "tll", "tll_cpp", "parametric", "rosenblatt-vinecop-check")
#: VDC's ``fit_bicop`` parametric family set (``vdc.baselines.pyvinecopulib``).
PARAMETRIC_FAMILIES = ("indep", "gaussian", "student", "clayton", "gumbel", "frank",
                       "joe", "bb1", "bb6", "bb7", "bb8", "tawn")
ROLES = {"vdc": "arm", "tll": "arm", "parametric": "arm", "tll_cpp": "cross_check",
         "rosenblatt-vinecop-check": "cross_check"}
MACHINERY = {
  "shared": "pyvinecopulib 1.0 VinecopBase (e3_vine.SharedVinecop), simplified, true structure",
  "cpp": "pyvinecopulib 1.0 compiled Vinecop.from_data, true structure",
  "npcc": "npcc.RosenblattVinecop, SimplifiedContext, true structure",
}


def cells(data: dict) -> list[tuple[int, int]]:
  return [(int(d), int(s)) for d, s in zip(data["d"], data["seed"])]


def main() -> None:  # noqa: PLR0915 - one linear protocol
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--folder", type=Path, default=common.RESULTS / "e3c")
  p.add_argument("--arms", nargs="+", required=True)
  p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
  p.add_argument("--gpu-is-free", action="store_true")
  p.add_argument("--vdc-core", type=int, default=None)
  p.add_argument("--dims", type=int, nargs="*", default=None)
  a = p.parse_args()

  for arm in a.arms:
    label = arm[len("npcc-"):] if arm.startswith("npcc-") else None
    if arm not in SPECIAL and label not in e3_npcc.ESTIMATORS:
      raise SystemExit(f"arm must be one of {SPECIAL} or npcc-<E1 estimator>; got {arm}")
    kind = e3_npcc.ESTIMATORS[label][2] if label else "cpu"
    if (kind == "gpu") != (a.device == "cuda"):
      raise SystemExit(f"{arm} must run with --device {'cuda' if kind == 'gpu' else 'cpu'}")
  if a.device == "cuda" and not a.gpu_is_free:
    raise SystemExit("--device cuda requires --gpu-is-free: the GPU is shared with E1")

  info = json.loads((a.folder / "vdc_side.json").read_text())
  samples = a.folder / "samples.npz"
  if common.sha256_file(samples) != info["samples_sha256"]:
    raise SystemExit(f"{samples} must match the sha256 recorded by the VDC side")
  with np.load(samples) as f:
    data = {k: f[k] for k in f.files}
  prov = common.provenance(__file__, a.device)

  server = None
  if "vdc" in a.arms:
    server = VdcServer(cpu_core=a.vdc_core)
    VdcBicop.server = server
  try:
    for arm in a.arms:
      out_dir = a.folder / "npcc_cells" / arm
      out_dir.mkdir(parents=True, exist_ok=True)
      warmed = arm in ("vdc", "tll_cpp", "parametric", "tll")
      for d, seed in cells(data):
        if a.dims and d not in a.dims:
          continue
        if arm == "rosenblatt-vinecop-check" and d != 5:
          continue
        path = out_dir / f"d{d}_s{seed}.json"
        if path.exists():
          continue
        key = f"d{d}_s{seed}"
        tr, te, logtrue = data[f"{key}_train"], data[f"{key}_test"], data[f"{key}_logtrue"]
        order = [int(o) for o in data[f"{key}_order"]]
        structure = pv.RVineStructure.from_matrix(data[f"{key}_matrix"])
        if not np.array_equal(structure.matrix, pv.RVineStructure.from_order(order).matrix):
          raise RuntimeError(f"{key}: the structure must be the D-vine on order {order}")
        tr_t = torch.as_tensor(tr, dtype=torch.float64)
        te_t = torch.as_tensor(te, dtype=torch.float64)
        extra: dict = {}

        def shared_fit(fit_edge, device="cpu"):  # noqa: ANN001, ANN202
          vine = SharedVinecop(structure, device=device)
          t0 = time.perf_counter()
          vine.fit(tr_t, fit_edge=fit_edge)
          if device.startswith("cuda"):
            torch.cuda.synchronize()
          fit_s = time.perf_counter() - t0
          with torch.inference_mode():
            t0 = time.perf_counter()
            lp = vine.logpdf(te_t).detach().cpu().numpy().astype(np.float64)
            inference_s = time.perf_counter() - t0
          return lp, fit_s, inference_s

        def npcc_edge(est):  # noqa: ANN001, ANN202
          from npcc import FitControlsRosenblattBicop, RosenblattBicop

          controls = est.controls(FitControlsRosenblattBicop, device=a.device,
                                  projection_grid_size=e3_npcc.PROJECTION_GRID_SIZE, batch_size=None)

          def fit_edge(tree, edge, u_e, x_e=None, var_types=("c", "c")):  # noqa: ANN001, ANN202
            if x_e is not None:
              raise ValueError("x_e must be None in a simplified unconditional vine")
            return RosenblattBicop(controls).fit(u_e)

          return fit_edge

        load_before = common.load_average()
        if arm == "vdc":
          n_calls = len(server.calls)
          lp, fit_s, inference_s = shared_fit(lambda t, e, u_e, x_e=None, var_types=("c", "c"): VdcBicop.from_data(u_e))
          calls = server.calls[n_calls:]
          extra = {"vdc_server_s": sum(c["server_s"] for c in calls),
                   "vdc_projection_s": sum(c["projection_s"] for c in calls),
                   "vdc_ipc_s": sum(c["ipc_s"] for c in calls), "vdc_calls": len(calls),
                   "vdc_server_code_sha256": json.dumps(server.info["code_sha256"], sort_keys=True),
                   "vdc_server_versions": json.dumps(server.info["versions"])}
          machinery, pair = "shared", "e3_vdc_pair.VdcBicop (VDC grid, padded, TorchTllBicop norm_maxiter=0)"
        elif arm == "tll":
          from pyvinecopulib.torch import FitControlsTorchBicop, TorchTllBicop

          ctl = FitControlsTorchBicop()
          lp, fit_s, inference_s = shared_fit(
            lambda t, e, u_e, x_e=None, var_types=("c", "c"): TorchTllBicop.from_data(u_e, ctl))
          machinery, pair = "shared", f"pyvinecopulib.torch.TorchTllBicop, {ctl}"
        elif arm in ("parametric", "tll_cpp"):
          fams = PARAMETRIC_FAMILIES if arm == "parametric" else ("tll",)
          ctl = pv.FitControlsVinecop(family_set=[getattr(pv.families, f) for f in fams],
                                      selection_criterion="bic", allow_rotations=True, num_threads=1)
          t0 = time.perf_counter()
          vc = pv.Vinecop.from_data(tr, structure=structure, controls=ctl)
          fit_s = time.perf_counter() - t0
          t0 = time.perf_counter()
          lp = np.log(np.maximum(np.asarray(vc.pdf(te), dtype=np.float64), common.LOG_FLOOR))
          inference_s = time.perf_counter() - t0
          machinery = "cpp"
          pair = f"pyvinecopulib.Bicop, family set {list(fams)}, BIC, rotations, 1 thread"
        elif arm == "rosenblatt-vinecop-check":
          from npcc import FitControlsRosenblattVinecop, RosenblattVinecop
          from pyvinecopulib.core import SimplifiedContext

          est = e3_npcc.spec("catboost-logit")
          torch.manual_seed(common_seed(d, seed, est))
          vine = RosenblattVinecop(None, structure, device="cpu", context=SimplifiedContext())
          controls = est.controls(FitControlsRosenblattVinecop, device="cpu",
                                  projection_grid_size=e3_npcc.PROJECTION_GRID_SIZE, batch_size=None)
          t0 = time.perf_counter()
          vine.fit(tr_t, controls)
          fit_s = time.perf_counter() - t0
          with torch.inference_mode():
            t0 = time.perf_counter()
            lp = vine.logpdf(te_t).detach().cpu().numpy().astype(np.float64)
            inference_s = time.perf_counter() - t0
          machinery, pair = "npcc", "npcc.RosenblattBicop (catboost-logit)"
        else:
          est = e3_npcc.spec(arm[len("npcc-"):])
          if not warmed:
            SharedVinecop(pv.RVineStructure.from_order([1, 2, 3]), device=a.device).fit(
              tr_t[:, :3], fit_edge=npcc_edge(est))
            warmed = True
          torch.manual_seed(common_seed(d, seed, est))
          lp, fit_s, inference_s = shared_fit(npcc_edge(est), device=a.device)
          machinery, pair = "shared", f"npcc.RosenblattBicop ({est.label})"
          extra = {"estimator": est.label, "estimator_id": est.estimator_id,
                   "backend": est.backend, "transform": est.transform,
                   "model_version": e3_npcc.ESTIMATORS[est.label][1]}
        # A density the arm cannot evaluate (NaN) counts as the floor, as a
        # density below it does: every arm is then scored on all the test
        # points. `np.maximum` alone would pass a NaN through. The excess
        # over the finite points is recorded beside it.
        finite = np.isfinite(lp)
        n_nonfinite = int(np.sum(~finite))
        excess_finite = float(np.mean((logtrue - lp)[finite])) if finite.any() else None
        floor = np.log(common.LOG_FLOOR)
        lp = np.where(np.isnan(lp) | (lp < floor), floor, lp)
        ratio = logtrue - lp
        row = {
          "experiment": "E3c", "d": d, "seed": seed, "n_train": int(tr.shape[0]),
          "n_test": int(te.shape[0]), "n_edges": d * (d - 1) // 2, "order": json.dumps(order),
          "method": arm, "role": ROLES.get(arm, "arm"),
          "vine_machinery": MACHINERY[machinery], "pair_copula": pair, "env": "npcc",
          "test_loglik": float(np.mean(lp)), "true_loglik": float(np.mean(logtrue)),
          "excess_nll": float(np.mean(ratio)),
          "excess_nll_mc_se": float(np.std(ratio, ddof=1) / np.sqrt(ratio.size)),
          "n_nonfinite": n_nonfinite, "excess_nll_finite": excess_finite,
          "fit_s": fit_s, "inference_s": inference_s, **extra,
          "structure_selection": "none (true structure given)",
          "torch_threads": torch.get_num_threads(),
          "cpu_cores": len(os.sched_getaffinity(0)),
          "samples_sha256": info["samples_sha256"],
          "load_avg_1m": load_before, **prov,
        }
        np.save(path.with_suffix(".npy"), lp.astype(np.float32))
        common.write_json(row, path)
        print(f"{arm} d={d} s{seed}: ll {row['test_loglik']:.4f} (true {row['true_loglik']:.4f}) "
              f"excess {row['excess_nll']:.4f} fit {fit_s:.1f} s inf {inference_s:.1f} s", flush=True)
      rows = [json.loads(q.read_text()) for q in sorted(out_dir.glob("d*.json"))]
      if rows:
        common.write_csv(rows, a.folder / f"npcc_{arm}.csv")
  finally:
    if server is not None:
      server.close()


def common_seed(d: int, seed: int, est) -> int:  # noqa: ANN001
  """The torch seed of an NPCC fit, salted as E1 salts its cells."""
  from npcc.experiments._study import derive_seed

  return derive_seed("E3c", d, seed, est.estimator_id)


if __name__ == "__main__":
  main()
