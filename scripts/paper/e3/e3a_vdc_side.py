#!/usr/bin/env python3
"""E3a, VDC side: VDC's Table 1 samples, VDC, TLL and parametric selection, timed.

Run in the VDC environment (Python 3.10, pyvinecopulib 0.7.5, VDC v0.2.0):

    cd /home/thibault/npcc
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c 12 nice -n 10 /home/thibault/vdc/env/bin/python \
        scripts/paper/e3/e3a_vdc_side.py \
        --reference-run results/e3_vdc/table1_rerun_cpu \
        --out results/e3_vdc/e3a

**Samples.** For the nine cases and five seeds of VDC's runner
``paper/neurips2026/runners/reproduce_table1_likelihood.py``, at two training
sizes:

- ``n_train = 2000``: exactly the runner's samples -- the case list and the
  seed list are parsed from the runner's own source, the generator lines
  below must appear in it verbatim, and every true log-density must equal
  the unmodified runner's (``--reference-run``) bitwise;
- ``n_train = 10000``: the same generator with the training size changed and
  nothing else. The training draw keeps the runner's seeds ``[seed, seed +
  1]``, and the test set is the runner's (``n = 20000``, seeds ``[seed +
  777777, seed + 777778]``), identical at both sizes. pyvinecopulib fills
  the uniforms column by column, so the 2,000-row sample is not a prefix of
  the 10,000-row one: the two share their first 2,000 first-column uniforms
  and are otherwise distinct draws.

**Arms** (one torch thread, one pyvinecopulib thread, as in the runner):

- ``vdc``: the released ``vdc-denoiser-m64-v1`` with the runner's call
  (m = 64, 30 projection iterations, ``grid /= grid.mean()``) and its own
  lookup, ``HFuncLookup(use_spline=False).pdf`` -- bilinear between cell
  centers, constant beyond the outer ones. The Sinkhorn projection is timed
  separately by wrapping ``vdc.models.projection.copula_project``, which
  VDC's estimator imports at call time;
- ``tll``: pyvinecopulib's TLL itself, with pyvinecopulib's defaults
  (``FitControlsBicop(family_set=[tll])``: constant method, multiplier 1, a
  30-point grid) and one thread; :func:`check_tll_defaults` confirms these
  are also the settings of VDC's ``fit_bicop`` and records them;
- ``parametric``: parametric family selection by BIC with rotations, the
  runner's call (``vdc.baselines.pyvinecopulib.fit_bicop``);
- ``tll_runner_bic`` (``n_train = 2000`` only; a reproduction check of
  VDC's Table 1, not an arm): the runner's TLL call, a BIC choice between
  independence and TLL.

**Metrics.** KL = mean over the test points of ``log max(c_true, 1e-300) -
log max(c_hat, 1e-300)``, the runner's definition. On the ``200 x 200``
midpoint grid ``((i + 0.5) / 200, (j + 0.5) / 200)`` of the open unit
square: IAE = mean ``|c_hat - c|`` and ISE = mean ``(c_hat - c)^2``, against
the true copula density, plus the largest fitted density on the grid, the
largest ratio ``c_hat / c``, and where each occurs.

**Outputs** (under ``--out``, every ``.npz`` written byte for byte
reproducibly by ``e3_common.save_npz``): ``samples.npz`` (``n_train = 2000``)
and ``samples_n10000.npz``; ``grid_truth.npz`` (the midpoints and the true density
on the grid, per case); ``vdc_grids_n{n}.npz`` (VDC's 64 x 64 grids, for the
NPCC side's cross-check of the adapted VDC pair copula); ``vdc_side.csv``
(one row per n_train x case x seed x method, with a ``role``); and
``vdc_side.json`` (provenance, settings, the reproduction check).

**Timing.** As in VDC's protocol, loading and a warm-up call are excluded.
VDC's fit time is the histogram, the denoiser forward pass, the projection
and the construction of the lookup; inference is the lookup at the test
points; ``grid_s`` is the evaluation on the 200 x 200 grid. The machine is
shared, so every row carries the one-minute load average.
"""

from __future__ import annotations

import argparse
import ast
import inspect
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
N_TRAINS = (2000, 10000)
N_TEST = 20000
GRID_N = 200
#: pyvinecopulib's TLL defaults, which must also be the settings of VDC's
#: ``fit_bicop``; :func:`check_tll_defaults` asserts both.
TLL_DEFAULTS = {"nonparametric_method": "constant", "nonparametric_mult": 1.0,
                "nonparametric_grid_size": 30}
#: Published mean KL of VDC's Table 1 (runner README), nats per edge.
PUBLISHED_MEAN_KL = {"vdc": 0.00611124, "tll_runner_bic": 0.01348007,
                     "parametric": 0.00036485}
#: The runner's methods, in the order of its ``logpdf`` array, and their keys
#: in its ``results.json``.
RUNNER_METHODS = {"vdc": (0, "vdc"), "tll_runner_bic": (1, "tll"),
                  "parametric": (2, "parametric")}


def fit_tll_plain(uv: np.ndarray):  # noqa: ANN201 - a pyvinecopulib Bicop
  """Fit pyvinecopulib's TLL with its defaults and no family selection."""
  import pyvinecopulib as pv

  controls = pv.FitControlsBicop(family_set=[pv.BicopFamily.tll], num_threads=1)
  return pv.Bicop.from_data(np.clip(uv, 1e-10, 1 - 1e-10), controls=controls)


def check_tll_defaults() -> dict:
  """Confirm pyvinecopulib's TLL defaults are VDC's ``fit_bicop`` settings."""
  import pyvinecopulib as pv
  from vdc.baselines.pyvinecopulib import fit_bicop

  controls = pv.FitControlsBicop(family_set=[pv.BicopFamily.tll], num_threads=1)
  resolved = {k: getattr(controls, k) for k in TLL_DEFAULTS}
  params = inspect.signature(fit_bicop).parameters
  vdc_defaults = {k: params[k].default for k in TLL_DEFAULTS}
  if not resolved == TLL_DEFAULTS == vdc_defaults:
    raise RuntimeError(f"TLL defaults must match VDC's fit_bicop; got {resolved} vs {vdc_defaults}")
  return {"family_set": ["tll"], "num_threads": 1, "pyvinecopulib_defaults": resolved,
          "vdc_fit_bicop_defaults": vdc_defaults, "match": True}


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


def midpoint_grid(n: int = GRID_N) -> tuple[np.ndarray, np.ndarray]:
  """The midpoints ``(i + 0.5) / n`` and the ``(n * n, 2)`` grid, ``u`` major."""
  g = (np.arange(n) + 0.5) / n
  uu, vv = np.meshgrid(g, g, indexing="ij")
  return g, np.column_stack([uu.ravel(), vv.ravel()])


def grid_metrics(pdf_hat: np.ndarray, pdf_true: np.ndarray, g: np.ndarray) -> dict:
  """IAE, ISE and the largest density and ratio on the midpoint grid."""
  pdf_hat = np.asarray(pdf_hat, dtype=np.float64).ravel()
  err = pdf_hat - pdf_true
  n = g.size
  i_max = int(np.argmax(pdf_hat))
  ratio = pdf_hat / pdf_true
  i_ratio = int(np.argmax(ratio))
  return {
    "iae": float(np.mean(np.abs(err))), "ise": float(np.mean(err ** 2)),
    "grid_max_pdf": float(pdf_hat[i_max]),
    "grid_argmax_u": float(g[i_max // n]), "grid_argmax_v": float(g[i_max % n]),
    "grid_true_at_argmax": float(pdf_true[i_max]),
    "grid_max_ratio": float(ratio[i_ratio]),
    "grid_argmax_ratio_u": float(g[i_ratio // n]), "grid_argmax_ratio_v": float(g[i_ratio % n]),
    "grid_min_pdf": float(np.min(pdf_hat)),
  }


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


def main() -> None:  # noqa: PLR0915 - one linear protocol
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--runner", type=Path, default=common.TABLE1_RUNNER)
  p.add_argument("--checkpoint", type=Path, default=common.CHECKPOINT)
  p.add_argument("--reference-run", type=Path, required=True)
  p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
  p.add_argument("--n-train", type=int, nargs="+", default=list(N_TRAINS))
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
  tll_check = check_tll_defaults()

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
  with torch.no_grad():
    estimate_pair_density_from_samples(
      bundle, np.random.default_rng(0).uniform(size=(2000, 2)), m=64, projection_iters=30)
  timer.take()

  g, grid_pts = midpoint_grid()
  bicops = [pv.Bicop(family=getattr(pv.BicopFamily, f), rotation=r,
                     **({'parameters': np.asarray(pa)} if pa else {})) for f, pa, r in cases]
  grid_true = np.stack([bc.pdf(grid_pts) for bc in bicops])
  common.save_npz(a.out / "grid_truth.npz", {"midpoints": g, "pdf_true": grid_true,
                  "case": np.asarray([case_label(*c) for c in cases])})

  rows: list[dict] = []
  max_gap = {"logtrue": 0.0, **{m: 0.0 for m in RUNNER_METHODS}}
  max_kl_gap = {m: 0.0 for m in RUNNER_METHODS}
  samples_sha: dict[str, str] = {}
  started = time.perf_counter()
  for n_train in a.n_train:
    train, test, logtrue, grids = [], [], [], []
    meta = {"case_index": [], "case": [], "family": [], "rotation": [],
            "parameters": [], "seed": []}
    k = 0
    for i, (family, params, rotation) in enumerate(cases):
      label = case_label(family, params, rotation)
      # --- generator: verbatim from the runner (checked above) ---
      bc = pv.Bicop(family=getattr(pv.BicopFamily,family), rotation=rotation,
          **({'parameters':np.asarray(params)} if params else {}))
      for seed in seeds:
        if n_train == 2000:
          tr = np.clip(bc.simulate(2000,seeds=[seed,seed+1]),1e-10,1-1e-10)
        else:
          tr = np.clip(bc.simulate(n_train,seeds=[seed,seed+1]),1e-10,1-1e-10)
        te = np.clip(bc.simulate(20000,seeds=[seed+777777,seed+777778]),1e-10,1-1e-10)
        true = np.log(np.maximum(bc.pdf(te),1e-300))
        # --- end of generator ---
        assert tr.shape == (n_train, 2) and te.shape == (N_TEST, 2)
        ref_row = ref_json["rows"][k]
        assert (ref_row["seed"], ref_row["family"], ref_row["rotation"]) == (seed, family, rotation)
        if not np.array_equal(true, ref_npz["logtrue"][k]):
          raise RuntimeError(f"true log-density differs from the runner at {label} seed {seed}")

        base = {
          "experiment": "E3a", "case_index": i, "case": label, "family": family,
          "rotation": rotation, "parameters": json.dumps(params), "seed": seed,
          "n_train": n_train, "n_test": N_TEST,
          "train_seeds": json.dumps([seed, seed + 1]),
          "test_seeds": json.dumps([seed + 777777, seed + 777778]),
        }
        evals: dict[str, dict] = {}

        def evaluate(name: str, logpdf, pdf_grid, fit_s, projection_s, extra, load_before):  # noqa: ANN001, PLR0913
          t0 = time.perf_counter()
          lp = logpdf()
          inference_s = time.perf_counter() - t0
          t0 = time.perf_counter()
          gpdf = pdf_grid()
          grid_s = time.perf_counter() - t0
          evals[name] = {"logpdf": lp, "grid": grid_metrics(gpdf, grid_true[i], g),
                         "fit_s": fit_s, "projection_s": projection_s,
                         "inference_s": inference_s, "grid_s": grid_s,
                         "load_before": load_before, **extra}

        # VDC, as in the runner, through its own lookup.
        load_before = common.load_average()
        t0 = time.perf_counter()
        with torch.no_grad():
          grid = np.asarray(estimate_pair_density_from_samples(bundle,tr,m=64,projection_iters=30),dtype=float)
        grid /= grid.mean()
        cop = HFuncLookup(grid,use_spline=False)
        fit_s = time.perf_counter() - t0
        projection_s = timer.take()
        grids.append(grid)
        evaluate("vdc", lambda: np.log(np.maximum(cop.pdf(te[:,0],te[:,1]),1e-300)),
                 lambda: cop.pdf(grid_pts[:, 0], grid_pts[:, 1]), fit_s, projection_s,
                 {"selected": (None, None, None)}, load_before)

        def describe(model) -> tuple:  # noqa: ANN001
          fam = str(model.family).split(".")[-1]
          pars = None if fam == "tll" else json.dumps(np.asarray(model.parameters).round(6).tolist())
          return fam, int(model.rotation), pars

        # TLL itself, pyvinecopulib's defaults.
        load_before = common.load_average()
        t0 = time.perf_counter()
        tll = fit_tll_plain(tr)
        fit_s = time.perf_counter() - t0
        evaluate("tll", lambda: np.log(np.maximum(tll.pdf(te), 1e-300)),
                 lambda: tll.pdf(grid_pts), fit_s, None, {"selected": describe(tll)}, load_before)
        # Parametric family selection (BIC), and at n = 2000 the runner's TLL.
        methods = [("parametric", "parametric")]
        if n_train == 2000:
          methods.append(("tll_runner_bic", "nonparametric"))
        for name, mode in methods:
          load_before = common.load_average()
          t0 = time.perf_counter()
          fitted = fit_bicop(tr,mode=mode,num_threads=1,selection_criterion='bic',allow_rotations=True)
          fit_s = time.perf_counter() - t0
          evaluate(name, lambda: np.log(np.maximum(fitted.pdf(te),1e-300)),
                   lambda: fitted.pdf(grid_pts), fit_s, None,
                   {"selected": describe(fitted.model)}, load_before)

        for name, ev in evals.items():
          ratio = true - ev["logpdf"]
          kl = float(np.mean(ratio))
          kl_runner = None
          if n_train == 2000 and name in RUNNER_METHODS:
            j, key = RUNNER_METHODS[name]
            gap = float(np.max(np.abs(ev["logpdf"] - ref_npz["logpdf"][k][j])))
            max_gap[name] = max(max_gap[name], gap)
            kl_runner = ref_row["kl"][key]
            max_kl_gap[name] = max(max_kl_gap[name], abs(kl - kl_runner))
          check = name == "tll_runner_bic"
          fam, rot, par = ev["selected"]
          rows.append({
            **base,
            "method": name,
            "method_group": "table1_reproduction_check" if check else "competitor",
            "role": "reproduction_check" if check else "arm",
            "conditions_on_u1": None,
            "backend": None, "transform": None, "model_version": None,
            "env": "vdc",
            "kl": kl,
            "kl_mc_se": float(np.std(ratio, ddof=1) / np.sqrt(ratio.size)),
            "kl_runner": kl_runner,
            "n_floored": int(np.sum(ev["logpdf"] <= np.log(common.LOG_FLOOR))),
            "min_logpdf": float(np.min(ev["logpdf"])),
            "max_log_ratio": float(np.max(ratio)),
            **ev["grid"],
            "grid_n": GRID_N,
            "fit_s": ev["fit_s"], "projection_s": ev["projection_s"],
            "inference_s": ev["inference_s"], "grid_s": ev["grid_s"],
            "torch_threads": torch.get_num_threads(), "pyvinecopulib_threads": 1,
            "selected_family": fam, "selected_rotation": rot, "selected_parameters": par,
            "tll_settings": "pyvinecopulib defaults: constant, mult 1, grid 30" if name == "tll" else None,
            "load_avg_1m": ev["load_before"],
            **prov,
          })
        train.append(tr); test.append(te); logtrue.append(true)
        for key, value in (("case_index", i), ("case", label), ("family", family),
                           ("rotation", rotation), ("parameters", json.dumps(params)),
                           ("seed", seed)):
          meta[key].append(value)
        k += 1
      print(f"n={n_train} done {label}", flush=True)

    samples = a.out / ("samples.npz" if n_train == 2000 else f"samples_n{n_train}.npz")
    common.save_npz(samples, {
      "train": np.asarray(train), "test": np.asarray(test), "logtrue": np.asarray(logtrue),
      **{key: np.asarray(value) for key, value in meta.items()},
    })
    samples_sha[str(n_train)] = common.sha256_file(samples)
    common.save_npz(a.out / f"vdc_grids_n{n_train}.npz", {"grids": np.asarray(grids),
                    "case": np.asarray(meta["case"]), "seed": np.asarray(meta["seed"])})

  common.write_csv(rows, a.out / "vdc_side.csv")
  means = {f"{m}@n={n}": float(np.mean([r["kl"] for r in rows if r["method"] == m and r["n_train"] == n]))
           for n in a.n_train for m in ("vdc", "tll", "parametric", "tll_runner_bic")
           if any(r["method"] == m and r["n_train"] == n for r in rows)}
  common.write_json({
    "experiment": "E3a",
    "description": "VDC Table 1 samples (n_train 2000 and 10000) and VDC/TLL/parametric fits, timed",
    "provenance": prov,
    "runner": str(a.runner), "runner_sha256": common.sha256_file(a.runner),
    "reference_run": str(a.reference_run),
    "reference_results_sha256": common.sha256_file(a.reference_run / "results.json"),
    "reference_arrays_sha256": common.sha256_file(a.reference_run / "likelihoods.npz"),
    "cases": cases, "seeds": seeds, "n_train": list(a.n_train), "n_test": N_TEST,
    "seed_scheme": {
      "n_train_2000": "the runner's: train seeds [seed, seed+1], test seeds [seed+777777, seed+777778]",
      "n_train_10000": "train bc.simulate(10000, seeds=[seed, seed+1]); test identical to n_train 2000",
      "nesting": "not nested: pyvinecopulib fills the uniforms column by column, so only the "
                 "first 2000 first-column uniforms are shared between the two training draws"},
    "log_floor": common.LOG_FLOOR, "sample_clip": [1e-10, 1 - 1e-10],
    "grid": {"n": GRID_N, "points": "midpoints (i+0.5)/200, u major", "file": "grid_truth.npz"},
    "vdc_settings": {"m": 64, "projection_iters": 30, "normalization": "grid /= grid.mean()",
                     "lookup": "HFuncLookup(use_spline=False).pdf", "torch_threads": 1, "dtype": "float32"},
    "parametric_settings": {"call": "vdc.baselines.pyvinecopulib.fit_bicop(mode='parametric')",
                            "selection_criterion": "bic", "allow_rotations": True, "num_threads": 1},
    "tll_settings": tll_check,
    "tll_runner_bic_settings": {"call": "vdc.baselines.pyvinecopulib.fit_bicop(mode='nonparametric')",
                                "family_set": ["indep", "tll"], "selection_criterion": "bic",
                                "role": "Table 1 reproduction check, not an arm", "n_train": 2000},
    "model_load_s": load_s,
    "wall_s": time.perf_counter() - started,
    "check_true_logdensity_bitwise_equal": True,
    "max_abs_logpdf_gap_vs_runner": max_gap,
    "max_abs_kl_gap_vs_runner": max_kl_gap,
    "mean_kl": means,
    "table1_reproduction": {
      m: {"published": PUBLISHED_MEAN_KL[m],
          "runner_rerun": ref_json["means"][RUNNER_METHODS[m][1]],
          "this_rerun": means[f"{m}@n=2000"]} for m in RUNNER_METHODS} if 2000 in a.n_train else None,
    "samples_npz": "samples.npz", "samples_sha256": samples_sha.get("2000"),
    "samples_sha256_by_n": samples_sha,
  }, a.out / "vdc_side.json")
  print(json.dumps({"mean_kl": means, "max_kl_gap": max_kl_gap}, indent=2))


if __name__ == "__main__":
  main()
