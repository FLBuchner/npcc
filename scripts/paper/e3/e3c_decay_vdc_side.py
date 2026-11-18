#!/usr/bin/env python3
"""E3c, decay design, VDC side: simplified vines whose dependence decays by tree.

Run in the VDC environment (Python 3.10, pyvinecopulib 0.7.5, VDC v0.2.0):

    cd /home/thibault/npcc
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c 10 nice -n 10 /home/thibault/vdc/env/bin/python \
        scripts/paper/e3/e3c_decay_vdc_side.py --out results/e3_vdc/e3c_decay

then fit the arms with ``e3c_npcc_fit.py --folder results/e3_vdc/e3c_decay``,
exactly as for the strong design.

**Two E3c designs.** ``decay`` (this script) is the primary one; ``strong``
(``e3c_vdc_side.py``, every tree as strong as the first) is kept as a stress
variant. Their outputs live in ``results/e3_vdc/e3c_decay/`` and
``results/e3_vdc/e3c/``, and ``e3_collect.py`` labels every merged row with
its ``design``.

**Design.** Everything the strong design draws is drawn here the same way,
from the same random stream: ``d`` in ``{5, 10}``, seeds ``101, 202, 303, 404,
505``, a D-vine on ``rng.permutation(d) + 1`` and one of the nine cases of
VDC's Table 1 runner per edge, tree by tree, with ``rng =
np.random.default_rng([seed, d])``. So each cell has the strong design's order
and the strong design's case on every edge. What changes is the parameter:
an edge in tree ``t`` (``t = 1`` for the first tree) gets the parameter whose
Kendall's tau is ``tau_case * 0.7 ** (t - 1)``, keeping its family, rotation
and sign. Independence stays independence; the Gaussian and the Student t get
``rho = sin(pi * tau / 2)``, the Student keeping ``nu = 5``; Clayton, Gumbel,
Joe and Frank get pyvinecopulib's ``Bicop.tau_to_parameters`` for their
rotation. Every edge's resulting tau is checked against its target.

Samples come from pyvinecopulib 0.7.5, ``Vinecop.simulate(2000, seeds=[seed,
d, 1])`` and ``simulate(20000, seeds=[seed, d, 2])`` -- the strong design's
seeds -- clipped to ``[1e-10, 1 - 1e-10]``, with the true log-density of every
test point.

**Arm here** (a reference, not an arm of the comparison): ``vdc_own``, VDC's
own vine on the true D-vine order with its UCI runner's settings, as in the
strong design.

**Outputs** (under ``--out``): ``vdc_own_cells/d<d>_s<seed>.npy`` (the
reference's raw test-point log-densities), ``samples.npz`` (per cell: train, test, true
log-density, order, structure matrix, the edge-to-case table and each edge's
tau), ``vdc_side.csv`` (``design = "decay"``) and ``vdc_side.json``.
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
from e3c_vdc_side import DIMS, N_TEST, N_TRAIN, SEEDS, VDC_VINE_SETTINGS  # noqa: E402

DESIGN = "decay"
DECAY = 0.7


def decayed_bicop(pv, case: tuple, tree: int):  # noqa: ANN001, ANN201 - pyvinecopulib module, Bicop
  """The case's pair copula with tau scaled by ``DECAY ** tree`` (``tree`` 0-based)."""
  family, params, rotation = case
  if family == "indep":
    return pv.Bicop(), 0.0
  base = pv.Bicop(family=getattr(pv.BicopFamily, family), rotation=rotation,
                  parameters=np.asarray(params))
  tau = float(base.parameters_to_tau(base.parameters)) * DECAY ** tree
  if family in ("gaussian", "student"):
    rho = float(np.sin(np.pi * tau / 2))
    pars = [[rho]] if family == "gaussian" else [[rho], [float(params[1][0])]]
  else:
    pars = np.asarray(base.tau_to_parameters(tau)).tolist()
  bc = pv.Bicop(family=getattr(pv.BicopFamily, family), rotation=rotation,
                parameters=np.asarray(pars))
  got = float(bc.parameters_to_tau(bc.parameters))
  if abs(got - tau) > 1e-6:
    raise RuntimeError(f"{case} in tree {tree + 1} must have tau {tau:.6f}; got {got:.6f}")
  return bc, tau


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--checkpoint", type=Path, default=common.CHECKPOINT)
  p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
  p.add_argument("--out", type=Path, default=common.RESULTS / "e3c_decay")
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

  arrays: dict[str, list] = {k: [] for k in ("d", "seed")}
  per_cell: dict[str, np.ndarray] = {}
  rows: list[dict] = []
  warmed = False
  for d in DIMS:
    for seed in SEEDS:
      # The strong design's stream: the order first, then the cases tree by tree.
      rng = np.random.default_rng([seed, d])
      order = [int(o) + 1 for o in rng.permutation(d)]
      edge_cases = [[int(rng.integers(len(cases))) for _ in range(d - 1 - t)] for t in range(d - 1)]
      fitted = [[decayed_bicop(pv, cases[j], t) for j in row] for t, row in enumerate(edge_cases)]
      structure = pv.DVineStructure(order=order)
      vine = pv.Vinecop.from_structure(
        structure=structure, pair_copulas=[[bc for bc, _ in row] for row in fitted])
      tr = np.clip(vine.simulate(N_TRAIN, seeds=[seed, d, 1]), 1e-10, 1 - 1e-10)
      te = np.clip(vine.simulate(N_TEST, seeds=[seed, d, 2]), 1e-10, 1 - 1e-10)
      logtrue = np.log(np.maximum(vine.pdf(te), common.LOG_FLOOR))
      key = f"d{d}_s{seed}"
      per_cell.update({f"{key}_train": tr, f"{key}_test": te, f"{key}_logtrue": logtrue,
                       f"{key}_order": np.asarray(order), f"{key}_matrix": np.asarray(structure.matrix),
                       f"{key}_edge_cases": np.asarray([c for row in edge_cases for c in row]),
                       f"{key}_edge_taus": np.asarray([tau for row in fitted for _, tau in row])})
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
        "experiment": "E3c", "design": DESIGN, "d": d, "seed": seed,
        "n_train": N_TRAIN, "n_test": N_TEST,
        "n_edges": d * (d - 1) // 2, "order": json.dumps(order),
        "edge_cases": json.dumps([[case_label(*cases[j]) for j in row] for row in edge_cases]),
        "edge_taus": json.dumps([[round(tau, 6) for _, tau in row] for row in fitted]),
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
    "experiment": "E3c", "design": DESIGN,
    "description": "simplified D-vines whose Kendall's tau decays by 0.7 per tree; VDC's own vine as a reference",
    "provenance": prov,
    "design_spec": {"dims": list(DIMS), "seeds": list(SEEDS), "n_train": N_TRAIN, "n_test": N_TEST,
                    "structure": "D-vine, order = rng.permutation(d) + 1 (the strong design's)",
                    "edges": "case index rng.integers(9) per edge, tree by tree (the strong design's)",
                    "rng": "np.random.default_rng([seed, d]); order first, then edges",
                    "parameter": f"tau = tau_case * {DECAY} ** (t - 1), t = 1 for the first tree; family, "
                                 "rotation and sign kept; independence stays independence; Gaussian and "
                                 "Student t: rho = sin(pi tau / 2), Student nu = 5; others: "
                                 "pyvinecopulib Bicop.tau_to_parameters",
                    "simulate_seeds": {"train": "[seed, d, 1]", "test": "[seed, d, 2]"},
                    "sample_clip": [1e-10, 1 - 1e-10], "cases": cases},
    "vdc_own_settings": {**VDC_VINE_SETTINGS, "source": "paper/neurips2026/runners/run_uci_maf.py",
                         "order": "the true D-vine order"},
    "log_floor": common.LOG_FLOOR,
    "samples_npz": samples.name, "samples_sha256": common.sha256_file(samples),
  }, a.out / "vdc_side.json")


if __name__ == "__main__":
  main()
