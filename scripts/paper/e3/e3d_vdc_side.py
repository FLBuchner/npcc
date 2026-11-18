#!/usr/bin/env python3
"""E3d, VDC side: VDC's own vine pipeline, choosing its own structure.

Run in the VDC environment (Python 3.10, VDC v0.2.0), after ``e3d_truth.py``:

    cd /home/thibault/npcc
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c 10 nice -n 10 /home/thibault/vdc/env/bin/python scripts/paper/e3/e3d_vdc_side.py

**Arm.** ``vdc``: VDC's default vine pipeline as its paper and runners use
it -- the greedy D-vine (``VineCopulaModel(vine_type="dvine")`` with no
order, so VDC picks the order from the data) with the settings of its UCI
vine runner (m = 64, 30 projection iterations, ``hfunc_use_spline=False``,
edge batches of 256), one torch thread, CPU. ``fit_s`` includes the order
selection; ``inference_s`` is ``logpdf`` at the 20,000 test points.

**Structure.** The selected order gives the D-vine VDC fitted;
``e3d_structure.recovery`` scores it against the true structure: per tree,
the share of the true edges recovered, the overall share, and whether it is
the truth.

**Metric.** Excess NLL = mean of ``log c_true - log max(c_hat, 1e-300)``.
A density VDC cannot evaluate (NaN) is not floored: the row's
``excess_nll`` is over VDC's own finite points, and ``e3_collect.py`` scores
every arm on the test points where all arms and the truth are finite.

**Outputs** (``results/e3d_unknown_structure/``): ``cells/vdc/<design>_d<d>_s<seed>.npy``
(raw test-point log-densities, NaN kept), ``vdc_side.csv`` (one row per
design x d x seed) and ``vdc_side.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_common as common  # noqa: E402
import e3d_structure  # noqa: E402
from e3c_vdc_side import VDC_VINE_SETTINGS  # noqa: E402

RESULTS = common.NPCC_REPO / "results" / "e3d_unknown_structure"


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--checkpoint", type=Path, default=common.CHECKPOINT)
  p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
  p.add_argument("--folder", type=Path, default=RESULTS)
  a = p.parse_args()

  import pyvinecopulib as pv
  import torch

  import vdc
  from vdc.pretrained import load_checkpoint_bundle
  from vdc.vine.api import VineCopulaModel

  assert Path(vdc.__file__).resolve().is_relative_to(common.VDC_REPO)
  assert common.sha256_file(a.checkpoint) == common.CHECKPOINT_SHA256
  e3d_structure.self_check(pv)
  info = json.loads((a.folder / "truths.json").read_text())
  samples = a.folder / "samples.npz"
  if common.sha256_file(samples) != info["samples_sha256"]:
    raise SystemExit(f"{samples} must match the sha256 recorded by e3d_truth.py")
  with np.load(samples) as f:
    data = {k: f[k] for k in f.files}
  prov = common.provenance(__file__, a.device)

  torch.set_num_threads(1)
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = True
  torch.backends.cudnn.benchmark = False
  bundle = load_checkpoint_bundle(a.checkpoint, device=a.device)
  ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
  bundle.model.load_state_dict(ck["model_state_dict"], strict=True)
  del ck
  settings = {**VDC_VINE_SETTINGS, "device": a.device}  # no order: VDC selects it

  rows: list[dict] = []
  warmed = False
  floor = np.log(common.LOG_FLOOR)
  for design, d, seed in zip(data["design"], data["d"], data["seed"]):
    design, d, seed = str(design), int(d), int(seed)
    key = f"{design}_d{d}_s{seed}"
    tr, te, logtrue = data[f"{key}_train"], data[f"{key}_test"], data[f"{key}_logtrue"]
    truth = e3d_structure.edges_from_matrix(data[f"{key}_true_matrix"])
    if not warmed:  # VDC's protocol excludes warm-up
      VineCopulaModel(**settings).fit(tr, diffusion_model=bundle.model, diffusion=None, verbose=False)
      warmed = True
    load_before = common.load_average()
    model = VineCopulaModel(**settings)
    t0 = time.perf_counter()
    model.fit(tr, diffusion_model=bundle.model, diffusion=None, verbose=False)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    lp = np.asarray(model.logpdf(te), dtype=np.float64)
    inference_s = time.perf_counter() - t0
    cells = a.folder / "cells" / "vdc"
    cells.mkdir(parents=True, exist_ok=True)
    np.save(cells / f"{key}.npy", lp.astype(np.float32))  # raw, NaN kept
    finite = np.isfinite(lp)
    ratio = (logtrue - np.maximum(lp, floor))[finite]
    order = [int(o) + 1 for o in model.structure.order]
    rec = e3d_structure.recovery(e3d_structure.dvine_edges(order), truth)
    rows.append({
      "experiment": "E3d", "design": design, "d": d, "seed": seed,
      "n_train": int(tr.shape[0]), "n_test": int(te.shape[0]), "n_edges": d * (d - 1) // 2,
      "method": "vdc", "role": "arm",
      "structure": "selected: VDC's greedy D-vine order", "selected_order": json.dumps(order),
      "recovery_per_tree": json.dumps(rec["per_tree"]), "recovery_tree1": rec["per_tree"][0],
      "recovery_overall": rec["overall"], "structure_equals_truth": rec["equal"],
      "vine_machinery": "VDC's own (vdc.vine.api.VineCopulaModel)", "env": "vdc",
      "test_loglik": float(np.mean(np.maximum(lp, floor)[finite])),
      "true_loglik": float(np.mean(logtrue)),
      "excess_nll": float(np.mean(ratio)),
      "excess_nll_mc_se": float(np.std(ratio, ddof=1) / np.sqrt(ratio.size)),
      "n_nonfinite": int(np.sum(~finite)), "excess_nll_scope": "own finite points",
      "fit_s": fit_s, "inference_s": inference_s, "selection_in_fit_s": True,
      "torch_threads": torch.get_num_threads(), "vdc_settings": json.dumps(settings),
      "samples_sha256": info["samples_sha256"], "load_avg_1m": load_before, **prov,
    })
    print(f"{key}: vdc excess {np.mean(ratio):.4f}, tree-1 recovery {rec['per_tree'][0]:.2f}, "
          f"fit {fit_s:.1f} s, inf {inference_s:.1f} s", flush=True)

  common.write_csv(rows, a.folder / "vdc_side.csv")
  common.write_json({
    "experiment": "E3d", "description": "VDC's own vine pipeline with its own structure choice",
    "provenance": prov, "vdc_settings": {**settings, "order": "selected by VDC (greedy)",
                                         "source": "paper/neurips2026/runners/run_uci_maf.py"},
    "samples_sha256": info["samples_sha256"], "log_floor": common.LOG_FLOOR,
  }, a.folder / "vdc_side.json")


if __name__ == "__main__":
  main()
