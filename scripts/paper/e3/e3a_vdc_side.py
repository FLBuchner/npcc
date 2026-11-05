#!/usr/bin/env python3
"""E3a, VDC side: VDC's Table 1 samples, its three fitted methods, timed.

Run in the VDC environment (Python 3.10, pyvinecopulib 0.7.5, VDC v0.2.0):

    cd /home/thibault/npcc
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c 12 nice -n 10 /home/thibault/vdc/env/bin/python \
        scripts/paper/e3/e3a_vdc_side.py \
        --reference-run results/e3_vdc/table1_rerun_cpu \
        --out results/e3_vdc/e3a

**What it does.** For the nine cases and five seeds of VDC's runner
``paper/neurips2026/runners/reproduce_table1_likelihood.py`` it

1. draws the 2,000 training and 20,000 test pseudo-observations and the true
   log-density of every test point exactly as the runner does -- the case
   list and the seed list are parsed from the runner's own source, and the
   four generator lines below are required to appear in it verbatim, so the
   two cannot drift apart;
2. refits VDC (released ``vdc-denoiser-m64-v1``, m = 64, 30 projection
   iterations, bilinear grid lookup), pyvinecopulib TLL and pyvinecopulib
   parametric selection (BIC, rotations), with the runner's calls and
   settings, timing each fit and each evaluation on the 20,000 test points;
   VDC's Sinkhorn projection is timed separately by wrapping
   ``vdc.models.projection.copula_project``, which VDC's estimator imports
   at call time;
3. checks the result against an unmodified run of the runner
   (``--reference-run``): every true log-density must be bitwise equal --
   which pins the test samples -- and the three methods' per-point
   log-densities and KLs are compared and their largest gaps recorded;
4. writes ``samples.npz`` (train, test, true log-density) for the NPCC side,
   ``vdc_side.csv`` (one row per case x seed x method), and
   ``vdc_side.json`` (provenance and the cross-check).

**Metric.** KL = mean over test points of
``log max(c_true, 1e-300) - log max(c_hat, 1e-300)``, the runner's own
definition (it floors densities at 1e-300 and clips the samples to
``[1e-10, 1 - 1e-10]``).

**Timing.** As in VDC's protocol, loading and a warm-up call are excluded:
the model is loaded once and one untimed VDC estimate is made first. VDC's
fit time is the histogram, the denoiser forward pass, the projection and
the construction of the bilinear lookup; its inference time is the lookup
at the test points. One torch thread and one pyvinecopulib thread, as in the
runner. The machine is shared with other jobs, so every row carries the
one-minute load average.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_common as common  # noqa: E402

#: The runner's generator, line for line. ``main`` refuses to run if any of
#: these is missing from the runner's source.
GENERATOR_LINES = (
  "bc = pv.Bicop(family=getattr(pv.BicopFamily,family), rotation=rotation,",
  "**({'parameters':np.asarray(params)} if params else {}))",
  "tr = np.clip(bc.simulate(2000,seeds=[seed,seed+1]),1e-10,1-1e-10)",
  "te = np.clip(bc.simulate(20000,seeds=[seed+777777,seed+777778]),1e-10,1-1e-10)",
  "true = np.log(np.maximum(bc.pdf(te),1e-300))",
)
N_TRAIN = 2000
N_TEST = 20000


def runner_constants(source: str) -> tuple[list, list]:
  """The runner's ``cases`` list and its ``for seed in [...]`` seed list."""
  tree = ast.parse(source)
  cases = seeds = None
  for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and any(
      isinstance(t, ast.Name) and t.id == "cases" for t in node.targets
    ):
      cases = ast.literal_eval(node.value)
    if (
      isinstance(node, ast.For)
      and isinstance(node.target, ast.Name)
      and node.target.id == "seed"
    ):
      seeds = ast.literal_eval(node.iter)
  if cases is None or seeds is None:
    raise RuntimeError("runner source must define `cases` and a seed loop")
  return cases, seeds


def case_label(family: str, params: list, rotation: int) -> str:
  """A readable, unique label such as ``gaussian_rho=-0.7`` or ``clayton_r90``."""
  flat = [p[0] for p in params]
  names = {
    "gaussian": ["rho"],
    "student": ["rho", "nu"],
    "clayton": ["theta"],
    "gumbel": ["theta"],
    "joe": ["theta"],
    "frank": ["theta"],
  }.get(family, [])
  parts = [family] + [f"{n}={v:g}" for n, v in zip(names, flat)]
  if rotation:
    parts.append(f"r{rotation}")
  return "_".join(parts)


class ProjectionTimer:
  """Wrap ``vdc.models.projection.copula_project`` and add up its time."""

  def __init__(self, module) -> None:  # noqa: ANN001 - a module object
    self.module = module
    self.original = module.copula_project
    self.seconds = 0.0
    self.calls = 0
    module.copula_project = self

  def __call__(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN204 - pass-through
    t0 = time.perf_counter()
    out = self.original(*args, **kwargs)
    self.seconds += time.perf_counter() - t0
    self.calls += 1
    return out

  def take(self) -> float:
    """Return the seconds accumulated since the last call, and reset."""
    seconds, self.seconds = self.seconds, 0.0
    return seconds


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--runner", type=Path, default=common.TABLE1_RUNNER)
  p.add_argument("--checkpoint", type=Path, default=common.CHECKPOINT)
  p.add_argument("--reference-run", type=Path, required=True)
  p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
  p.add_argument("--out", type=Path, required=True)
  a = p.parse_args()

  import pyvinecopulib as pv
  import torch

  import vdc
  import vdc.models.projection as projection
  from vdc.baselines.pyvinecopulib import fit_bicop
  from vdc.models.hfunc import HFuncLookup
  from vdc.pretrained import (
    estimate_pair_density_from_samples,
    load_checkpoint_bundle,
  )

  assert Path(vdc.__file__).resolve().is_relative_to(common.VDC_REPO)
  assert common.sha256_file(a.checkpoint) == common.CHECKPOINT_SHA256
  source = a.runner.read_text()
  missing = [line for line in GENERATOR_LINES if line not in source]
  if missing:
    raise RuntimeError(f"runner must contain the generator lines; missing {missing}")
  cases, seeds = runner_constants(source)

  with np.load(a.reference_run / "likelihoods.npz") as ref_file:
    # An NpzFile decompresses an array on every key access: read each once.
    ref_npz = {"logtrue": ref_file["logtrue"], "logpdf": ref_file["logpdf"]}
  ref_json = json.loads((a.reference_run / "results.json").read_text())
  assert ref_json["checkpoint_sha256"] == common.CHECKPOINT_SHA256

  a.out.mkdir(parents=True, exist_ok=True)
  prov = common.provenance(__file__, a.device)

  # The runner's numerical settings.
  torch.set_num_threads(1)
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = True
  torch.backends.cudnn.benchmark = False
  t_load = time.perf_counter()
  bundle = load_checkpoint_bundle(a.checkpoint, device=a.device)
  ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
  bundle.model.load_state_dict(ck["model_state_dict"], strict=True)
  del ck
  load_s = time.perf_counter() - t_load
  timer = ProjectionTimer(projection)

  rows: list[dict] = []
  train, test, logtrue = [], [], []
  meta = {"case_index": [], "case": [], "family": [], "rotation": [],
          "parameters": [], "seed": []}
  max_gap = {"logtrue": 0.0, "vdc": 0.0, "tll": 0.0, "parametric": 0.0}
  max_kl_gap = {"vdc": 0.0, "tll": 0.0, "parametric": 0.0}
  warmed = False
  k = 0
  started = time.perf_counter()
  for i, (family, params, rotation) in enumerate(cases):
    label = case_label(family, params, rotation)
    # --- generator: verbatim from the runner (checked above) ---
    bc = pv.Bicop(family=getattr(pv.BicopFamily,family), rotation=rotation,
        **({'parameters':np.asarray(params)} if params else {}))
    for seed in seeds:
      tr = np.clip(bc.simulate(2000,seeds=[seed,seed+1]),1e-10,1-1e-10)
      te = np.clip(bc.simulate(20000,seeds=[seed+777777,seed+777778]),1e-10,1-1e-10)
      true = np.log(np.maximum(bc.pdf(te),1e-300))
      # --- end of generator ---
      assert tr.shape == (N_TRAIN, 2) and te.shape == (N_TEST, 2)
      ref_row = ref_json["rows"][k]
      assert (ref_row["seed"], ref_row["family"], ref_row["rotation"]) == (
        seed, family, rotation)
      if not np.array_equal(true, ref_npz["logtrue"][k]):
        raise RuntimeError(f"true log-density differs from the runner at {label} seed {seed}")

      if not warmed:
        with torch.no_grad():
          estimate_pair_density_from_samples(bundle, tr, m=64, projection_iters=30)
        timer.take()
        warmed = True

      base = {
        "experiment": "E3a", "case_index": i, "case": label, "family": family,
        "rotation": rotation, "parameters": json.dumps(params), "seed": seed,
        "n_train": N_TRAIN, "n_test": N_TEST,
      }
      logs = {}
      # VDC, as in the runner.
      load_before = common.load_average()
      t0 = time.perf_counter()
      with torch.no_grad():
        grid = np.asarray(estimate_pair_density_from_samples(bundle,tr,m=64,projection_iters=30),dtype=float)
      grid /= grid.mean()
      cop = HFuncLookup(grid,use_spline=False)
      fit_s = time.perf_counter() - t0
      projection_s = timer.take()
      t0 = time.perf_counter()
      logs["vdc"] = np.log(np.maximum(cop.pdf(te[:,0],te[:,1]),1e-300))
      inference_s = time.perf_counter() - t0
      timing = {"vdc": (fit_s, projection_s, inference_s, None, None, None, load_before)}
      # pyvinecopulib TLL and parametric selection, as in the runner.
      for name, mode in (("tll", "nonparametric"), ("parametric", "parametric")):
        load_before = common.load_average()
        t0 = time.perf_counter()
        fitted = fit_bicop(tr,mode=mode,num_threads=1,selection_criterion='bic',allow_rotations=True)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        logs[name] = np.log(np.maximum(fitted.pdf(te),1e-300))
        inference_s = time.perf_counter() - t0
        model = fitted.model
        selected = (
          str(model.family).split(".")[-1],
          int(model.rotation),
          json.dumps(np.asarray(model.parameters).round(6).tolist())
          if str(model.family).split(".")[-1] != "tll" else None,
        )
        timing[name] = (fit_s, None, inference_s, *selected, load_before)

      for j, name in enumerate(("vdc", "tll", "parametric")):
        ratio = true - logs[name]
        kl = float(np.mean(ratio))
        gap = float(np.max(np.abs(logs[name] - ref_npz["logpdf"][k][j])))
        max_gap[name] = max(max_gap[name], gap)
        max_kl_gap[name] = max(max_kl_gap[name], abs(kl - ref_row["kl"][name]))
        fit_s, projection_s, inference_s, fam, rot, par, load_before = timing[name]
        rows.append({
          **base,
          "method": name,
          "method_group": "vdc_runner",
          "conditions_on_u1": None,
          "backend": None, "transform": None, "model_version": None,
          "env": "vdc",
          "kl": kl,
          "kl_mc_se": float(np.std(ratio, ddof=1) / np.sqrt(ratio.size)),
          "kl_runner": ref_row["kl"][name],
          "n_floored": int(np.sum(logs[name] <= np.log(common.LOG_FLOOR))),
          "min_logpdf": float(np.min(logs[name])),
          "max_log_ratio": float(np.max(ratio)),
          "fit_s": fit_s,
          "projection_s": projection_s,
          "inference_s": inference_s,
          "torch_threads": torch.get_num_threads(),
          "pyvinecopulib_threads": 1,
          "selected_family": fam, "selected_rotation": rot,
          "selected_parameters": par,
          "load_avg_1m": load_before,
          **prov,
        })
      train.append(tr); test.append(te); logtrue.append(true)
      for key, value in (("case_index", i), ("case", label), ("family", family),
                         ("rotation", rotation), ("parameters", json.dumps(params)),
                         ("seed", seed)):
        meta[key].append(value)
      k += 1
    print(f"done {label}", flush=True)

  samples = a.out / "samples.npz"
  np.savez_compressed(
    samples,
    train=np.asarray(train), test=np.asarray(test), logtrue=np.asarray(logtrue),
    **{key: np.asarray(value) for key, value in meta.items()},
  )
  common.write_csv(rows, a.out / "vdc_side.csv")
  means = {m: float(np.mean([r["kl"] for r in rows if r["method"] == m]))
           for m in ("vdc", "tll", "parametric")}
  common.write_json({
    "experiment": "E3a",
    "description": "VDC Table 1 samples and VDC/TLL/parametric refits, timed",
    "provenance": prov,
    "runner": str(a.runner), "runner_sha256": common.sha256_file(a.runner),
    "reference_run": str(a.reference_run),
    "reference_results_sha256": common.sha256_file(a.reference_run / "results.json"),
    "reference_arrays_sha256": common.sha256_file(a.reference_run / "likelihoods.npz"),
    "cases": cases, "seeds": seeds, "n_train": N_TRAIN, "n_test": N_TEST,
    "log_floor": common.LOG_FLOOR, "sample_clip": [1e-10, 1 - 1e-10],
    "vdc_settings": {"m": 64, "projection_iters": 30, "lookup": "HFuncLookup(use_spline=False).pdf",
                     "torch_threads": 1, "dtype": "float32"},
    "baseline_settings": {"family_sets": "vdc.baselines.pyvinecopulib", "selection_criterion": "bic",
                          "allow_rotations": True, "num_threads": 1},
    "model_load_s": load_s,
    "wall_s": time.perf_counter() - started,
    "check_true_logdensity_bitwise_equal": True,
    "max_abs_logpdf_gap_vs_runner": max_gap,
    "max_abs_kl_gap_vs_runner": max_kl_gap,
    "mean_kl": means,
    "samples_npz": samples.name, "samples_sha256": common.sha256_file(samples),
  }, a.out / "vdc_side.json")
  print(json.dumps({"mean_kl": means, "max_kl_gap": max_kl_gap}, indent=2))


if __name__ == "__main__":
  main()
