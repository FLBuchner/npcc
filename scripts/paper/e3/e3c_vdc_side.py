#!/usr/bin/env python3
"""E3c, VDC side: simplified vines on the nine Table 1 pair copulas.

Run in the VDC environment (Python 3.10, pyvinecopulib 0.7.5, VDC v0.2.0):

    cd /home/thibault/npcc
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c 12 nice -n 10 /home/thibault/vdc/env/bin/python \
        scripts/paper/e3/e3c_vdc_side.py --out results/e3_vdc/e3c

**Design.** For ``d`` in ``{5, 10}`` and seeds ``101, 202, 303, 404, 505``:
the true vine is a D-vine whose variable order is a random permutation, and
each of its ``d (d - 1) / 2`` pair copulas is one of the nine cases of VDC's
Table 1 runner (parsed from its source), drawn uniformly and independently
per edge, in every tree. Randomness comes from ``np.random.default_rng([seed,
d])`` (the order first, then the edges tree by tree); the samples from
pyvinecopulib 0.7.5, ``Vinecop.simulate(2000, seeds=[seed, d, 1])`` and
``simulate(20000, seeds=[seed, d, 2])``, clipped to ``[1e-10, 1 - 1e-10]`` as
in the runner. The true log-density of every test point is
``log max(Vinecop.pdf, 1e-300)``.

A D-vine because it is the vine VDC's own implementation takes on a given
order, so VDC can also run *its* vine on the true structure. The structure
matrix is saved, and the NPCC side rebuilds the same structure in
pyvinecopulib 1.0 (``RVineStructure.from_matrix``, checked against
``from_order``).

**Arm here** (a reference, not an arm of the comparison): ``vdc_own``, VDC's
own vine implementation on the true D-vine order with the settings of VDC's
UCI vine runner (``VineCopulaModel(vine_type="dvine", order=..., m=64,
projection_iters=30, hfunc_use_spline=False, batch_edges=True,
edge_batch_size=256)``, one torch thread, CPU). The comparison's arms run in
one vine machinery on the NPCC side (``e3c_npcc_fit.py``); this row shows how
far VDC's own vine machinery is from that.

**Outputs** (under ``--out``): ``vdc_own_cells/d<d>_s<seed>.npy`` (the
reference's raw test-point log-densities), ``samples.npz`` (per cell: train, test, true
log-density, order, structure matrix, the edge-to-case table),
``vdc_side.csv`` (one row per cell) and ``vdc_side.json`` (provenance and the
design).
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
from e3a_vdc_side import case_label, runner_constants  # noqa: E402

DIMS = (5, 10)
SEEDS = (101, 202, 303, 404, 505)
N_TRAIN = 2000
N_TEST = 20000
VDC_VINE_SETTINGS = {"vine_type": "dvine", "m": 64, "projection_iters": 30,
                     "hfunc_use_spline": False, "batch_edges": True, "edge_batch_size": 256}


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--checkpoint", type=Path, default=common.CHECKPOINT)
  p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
  p.add_argument("--out", type=Path, required=True)
  a = p.parse_args()

  import pyvinecopulib as pv
  import torch

  import vdc
  from vdc.pretrained import load_checkpoint_bundle
  from vdc.vine.api import VineCopulaModel

  assert Path(vdc.__file__).resolve().is_relative_to(common.VDC_REPO)
  assert common.sha256_file(a.checkpoint) == common.CHECKPOINT_SHA256
  cases, _ = runner_constants(common.TABLE1_RUNNER.read_text())
  a.out.mkdir(parents=True, exist_ok=True)
  prov = common.provenance(__file__, a.device)

  torch.set_num_threads(1)
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = True
  torch.backends.cudnn.benchmark = False
  bundle = load_checkpoint_bundle(a.checkpoint, device=a.device)
  ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
  bundle.model.load_state_dict(ck["model_state_dict"], strict=True)
  del ck

  def bicop(j: int):  # noqa: ANN202 - a pv.Bicop
    family, params, rotation = cases[j]
    return pv.Bicop(family=getattr(pv.BicopFamily, family), rotation=rotation,
                    **({"parameters": np.asarray(params)} if params else {}))

  arrays: dict[str, list] = {k: [] for k in ("d", "seed")}
  per_cell: dict[str, np.ndarray] = {}
  rows: list[dict] = []
  warmed = False
  for d in DIMS:
    for seed in SEEDS:
      rng = np.random.default_rng([seed, d])
      order = [int(o) + 1 for o in rng.permutation(d)]
      edge_cases = [[int(rng.integers(len(cases))) for _ in range(d - 1 - t)] for t in range(d - 1)]
      structure = pv.DVineStructure(order=order)
      vine = pv.Vinecop.from_structure(
        structure=structure, pair_copulas=[[bicop(j) for j in row] for row in edge_cases])
      tr = np.clip(vine.simulate(N_TRAIN, seeds=[seed, d, 1]), 1e-10, 1 - 1e-10)
      te = np.clip(vine.simulate(N_TEST, seeds=[seed, d, 2]), 1e-10, 1 - 1e-10)
      logtrue = np.log(np.maximum(vine.pdf(te), common.LOG_FLOOR))
      key = f"d{d}_s{seed}"
      per_cell.update({f"{key}_train": tr, f"{key}_test": te, f"{key}_logtrue": logtrue,
                       f"{key}_order": np.asarray(order), f"{key}_matrix": np.asarray(structure.matrix),
                       f"{key}_edge_cases": np.asarray([c for row in edge_cases for c in row])})
      arrays["d"].append(d); arrays["seed"].append(seed)

      settings = {**VDC_VINE_SETTINGS, "order": [o - 1 for o in order], "device": a.device}
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
      ratio = logtrue - lp
      # Raw per-point log-densities (NaN kept), so the collector can score
      # every arm on the test points where all arms are finite.
      cells = a.out / "vdc_own_cells"
      cells.mkdir(exist_ok=True)
      np.save(cells / f"d{d}_s{seed}.npy", lp.astype(np.float32))
      rows.append({
        "experiment": "E3c", "d": d, "seed": seed, "n_train": N_TRAIN, "n_test": N_TEST,
        "n_edges": d * (d - 1) // 2, "order": json.dumps(order),
        "edge_cases": json.dumps([[case_label(*cases[j]) for j in row] for row in edge_cases]),
        "method": "vdc_own", "method_group": "reference", "role": "reference",
        "vine_machinery": "VDC's own (vdc.vine.api.VineCopulaModel, D-vine on the true order)",
        "env": "vdc",
        "test_loglik": float(np.mean(lp)), "true_loglik": float(np.mean(logtrue)),
        "excess_nll": float(np.mean(ratio)),
        "excess_nll_mc_se": float(np.std(ratio, ddof=1) / np.sqrt(ratio.size)),
        "n_nonfinite": int(np.sum(~np.isfinite(lp))),
        "fit_s": fit_s, "inference_s": inference_s,
        "torch_threads": torch.get_num_threads(),
        "vdc_settings": json.dumps(settings),
        "load_avg_1m": load_before, **prov,
      })
      print(f"d={d} seed={seed}: true ll {np.mean(logtrue):.4f}, vdc_own ll {np.mean(lp):.4f}, "
            f"excess {np.mean(ratio):.4f}, fit {fit_s:.1f} s, inf {inference_s:.1f} s", flush=True)

  samples = a.out / "samples.npz"
  common.save_npz(samples, {"d": np.asarray(arrays["d"]), "seed": np.asarray(arrays["seed"]), **per_cell})
  common.write_csv(rows, a.out / "vdc_side.csv")
  common.write_json({
    "experiment": "E3c",
    "description": "simplified D-vines on the nine Table 1 pair copulas; VDC's own vine as a reference",
    "provenance": prov,
    "design": {"dims": list(DIMS), "seeds": list(SEEDS), "n_train": N_TRAIN, "n_test": N_TEST,
               "structure": "D-vine, order = rng.permutation(d) + 1",
               "edges": "case index rng.integers(9) per edge, tree by tree, in every tree",
               "rng": "np.random.default_rng([seed, d]); order first, then edges",
               "simulate_seeds": {"train": "[seed, d, 1]", "test": "[seed, d, 2]"},
               "sample_clip": [1e-10, 1 - 1e-10], "cases": cases},
    "vdc_own_settings": {**VDC_VINE_SETTINGS, "source": "paper/neurips2026/runners/run_uci_maf.py",
                         "order": "the true D-vine order"},
    "log_floor": common.LOG_FLOOR,
    "samples_npz": samples.name, "samples_sha256": common.sha256_file(samples),
  }, a.out / "vdc_side.json")


if __name__ == "__main__":
  main()
