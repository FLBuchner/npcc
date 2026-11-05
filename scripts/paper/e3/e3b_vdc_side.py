#!/usr/bin/env python3
"""E3b, VDC side: the non-simplified stress test, data and VDC-env methods.

Run in the VDC environment (Python 3.10, pyvinecopulib 0.7.5, VDC v0.2.0):

    cd /home/thibault/npcc
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c 12 nice -n 10 /home/thibault/vdc/env/bin/python \
        scripts/paper/e3/e3b_vdc_side.py --out results/e3_vdc/e3b

**Design** (VDC's appendix ``e18_nonsimplified_stress``, whose script stays in
the authors' workspace and is not released; this generator is rebuilt from
the saved configuration ``paper/neurips2026/figures/inputs/paper_outputs/
e18_nonsimplified_stress.json`` and the appendix text). ``U1 ~ U(0, 1)``,
and given ``U1`` the pair ``(U2, U3)`` has copula ``C_low`` when
``U1 < 0.5`` and ``C_high`` otherwise:

- ``gaussian_sign_flip``: Gaussian, rho = -0.75 below the split, +0.75 above;
- ``clayton_gumbel_tail_switch``: Clayton (theta = 3) below, Gumbel
  (theta = 2.5) above, both Kendall's tau = 0.6.

2,000 training and 10,000 test observations, seeds 0-9. Draws use
pyvinecopulib 0.7.5 (``U1`` is the first column of an independence-copula
sample), with seeds ``[seed, case, split, stream]``, and are clipped to
``[1e-10, 1 - 1e-10]`` as in VDC's Table 1 runner. The samples are therefore
not VDC's seed-0 sample; the rerun of VDC, TLL and parametric selection on
them is what makes the comparison matched.

**Methods** written here (all with VDC's code and the Table 1 runner's
settings: m = 64, 30 projection iterations, bilinear lookup; pyvinecopulib
TLL and BIC parametric selection with rotations, one thread):

- ``oracle``: the true conditional density ``c(u2, u3 | u1)``;
- ``pooled_oracle``: the true pooled density ``(C_low + C_high) / 2``, the
  lowest NLL any estimator that ignores ``U1`` can reach;
- ``vdc``, ``tll``, ``parametric``: pooled over ``U1``, as in the appendix;
- ``tll_bic``: the Table 1 runner's TLL call, ``fit_bicop(mode=
  "nonparametric")``, which selects between independence and TLL by BIC.
  On the pooled sign-flip sample TLL's effective degrees of freedom are
  about 130, so BIC picks independence on every seed, whereas the appendix
  reports a fitted TLL for this arm. ``tll`` therefore fits TLL alone, with
  the runner's TLL settings otherwise (constant method, multiplier 1, a
  30-point grid, one thread); both are written;
- ``regime_gaussian``: a Gaussian copula fitted in each regime (the
  appendix's conditional reference);
- ``vdc_regime``, ``tll_regime``, ``parametric_regime``: each method fitted
  separately in each regime. They are told the split point, which no
  estimator of the comparison is; they bound what pooling costs each method.

**Metric.** Test NLL = ``-mean log max(c_hat, 1e-300)`` over the test points
and excess NLL = test NLL - oracle NLL, in nats per test pair.

**Outputs** (under ``--out``): ``samples.npz`` (train/test ``(u1, u2, u3)``,
the oracle and pooled-oracle log-densities), ``vdc_side.csv`` (one row per
case x seed x method) and ``vdc_side.json`` (provenance, settings,
per-seed regime diagnostics).
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

N_TRAIN = 2000
N_TEST = 10000
SEEDS = list(range(10))
SPLIT = 0.5
CASES = {
  "gaussian_sign_flip": (("gaussian", -0.75), ("gaussian", 0.75)),
  "clayton_gumbel_tail_switch": (("clayton", 3.0), ("gumbel", 2.5)),
}


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--checkpoint", type=Path, default=common.CHECKPOINT)
  p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
  p.add_argument("--out", type=Path, required=True)
  a = p.parse_args()

  import pyvinecopulib as pv
  import scipy.stats as st
  import torch

  import vdc
  import vdc.models.projection as projection
  from vdc.baselines.pyvinecopulib import fit_bicop
  from vdc.models.hfunc import HFuncLookup
  from vdc.pretrained import (
    estimate_pair_density_from_samples,
    load_checkpoint_bundle,
  )

  sys.path.insert(0, str(Path(__file__).resolve().parent))
  from e3a_vdc_side import ProjectionTimer

  assert Path(vdc.__file__).resolve().is_relative_to(common.VDC_REPO)
  assert common.sha256_file(a.checkpoint) == common.CHECKPOINT_SHA256
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
  timer = ProjectionTimer(projection)

  def bicop(family: str, theta: float):  # noqa: ANN202 - a pv.Bicop
    return pv.Bicop(family=getattr(pv.BicopFamily, family),
                    parameters=np.asarray([[theta]]))

  def draw(c_idx: int, low, high, seed: int, split: int, n: int) -> np.ndarray:  # noqa: ANN001
    indep = pv.Bicop()
    u1 = indep.simulate(n, seeds=[seed, c_idx, split, 0])[:, 0]
    uv_low = low.simulate(n, seeds=[seed, c_idx, split, 1])
    uv_high = high.simulate(n, seeds=[seed, c_idx, split, 2])
    uv = np.where((u1 < SPLIT)[:, None], uv_low, uv_high)
    return np.clip(np.column_stack([u1, uv]), 1e-10, 1 - 1e-10)

  # --- the fitted methods: VDC's calls and the Table 1 runner's settings ---
  def fit_vdc(uv: np.ndarray):  # noqa: ANN202 - an HFuncLookup
    with torch.no_grad():
      grid = np.asarray(estimate_pair_density_from_samples(
        bundle, uv, m=64, projection_iters=30), dtype=float)
    grid /= grid.mean()
    return HFuncLookup(grid, use_spline=False)

  def vdc_logpdf(cop, uv: np.ndarray) -> np.ndarray:  # noqa: ANN001
    return np.log(np.maximum(cop.pdf(uv[:, 0], uv[:, 1]), common.LOG_FLOOR))

  def fit_pv(uv: np.ndarray, mode: str):  # noqa: ANN202 - a BicopFitResult
    return fit_bicop(uv, mode=mode, num_threads=1, selection_criterion="bic",
                     allow_rotations=True)

  def pv_logpdf(fitted, uv: np.ndarray) -> np.ndarray:  # noqa: ANN001
    return np.log(np.maximum(fitted.pdf(uv), common.LOG_FLOOR))

  def fit_gauss(uv: np.ndarray):  # noqa: ANN202 - a pv.Bicop
    controls = pv.FitControlsBicop(family_set=[pv.BicopFamily.gaussian], num_threads=1)
    return pv.Bicop.from_data(uv, controls=controls)

  def fit_tll_only(uv: np.ndarray):  # noqa: ANN202 - a pv.Bicop
    # fit_bicop's TLL controls, without the independence alternative.
    controls = pv.FitControlsBicop(
      family_set=[pv.BicopFamily.tll], selection_criterion="bic",
      allow_rotations=True, num_threads=1, nonparametric_method="constant",
      nonparametric_mult=1.0, nonparametric_grid_size=30)
    return pv.Bicop.from_data(uv, controls=controls)

  def bicop_logpdf(model, uv: np.ndarray) -> np.ndarray:  # noqa: ANN001
    return np.log(np.maximum(model.pdf(uv), common.LOG_FLOOR))

  def describe(model) -> str:  # noqa: ANN001
    fam = str(model.family).split(".")[-1]
    if fam == "tll":
      return "tll"
    pars = np.asarray(model.parameters).round(4).ravel().tolist()
    return f"{fam} r{int(model.rotation)} {pars}"

  methods = {
    "vdc": (fit_vdc, vdc_logpdf, lambda m: "pooled grid"),
    "tll": (fit_tll_only, bicop_logpdf, describe),
    "tll_bic": (lambda uv: fit_pv(uv, "nonparametric"), pv_logpdf, lambda m: describe(m.model)),
    "parametric": (lambda uv: fit_pv(uv, "parametric"), pv_logpdf, lambda m: describe(m.model)),
  }
  regime_methods = {
    "regime_gaussian": (fit_gauss, bicop_logpdf, describe),
    "vdc_regime": methods["vdc"],
    "tll_regime": methods["tll"],
    "parametric_regime": methods["parametric"],
  }

  # Warm-up, untimed: VDC's protocol excludes it.
  with torch.no_grad():
    estimate_pair_density_from_samples(
      bundle, np.random.default_rng(0).uniform(size=(N_TRAIN, 2)), m=64, projection_iters=30)
  timer.take()

  rows: list[dict] = []
  diagnostics: list[dict] = []
  arrays = {"train": [], "test": [], "logtrue": [], "log_pooled_oracle": [],
            "case": [], "seed": []}
  started = time.perf_counter()
  for c_idx, (case, ((f_low, t_low), (f_high, t_high))) in enumerate(CASES.items()):
    low, high = bicop(f_low, t_low), bicop(f_high, t_high)
    for seed in SEEDS:
      tr = draw(c_idx, low, high, seed, 1, N_TRAIN)
      te = draw(c_idx, low, high, seed, 2, N_TEST)
      lo_tr, lo_te = tr[:, 0] < SPLIT, te[:, 0] < SPLIT
      uv_tr, uv_te = tr[:, 1:], te[:, 1:]
      c_low, c_high = low.pdf(uv_te), high.pdf(uv_te)
      logtrue = np.log(np.maximum(np.where(lo_te, c_low, c_high), common.LOG_FLOOR))
      logpool = np.log(np.maximum(0.5 * c_low + 0.5 * c_high, common.LOG_FLOOR))
      oracle_nll = -float(np.mean(logtrue))

      base = {
        "experiment": "E3b", "case": case, "seed": seed,
        "low_copula": f"{f_low}({t_low:g})", "high_copula": f"{f_high}({t_high:g})",
        "split": SPLIT, "n_train": N_TRAIN, "n_test": N_TEST,
        "n_train_low": int(lo_tr.sum()), "n_test_low": int(lo_te.sum()),
        "oracle_nll": oracle_nll,
      }

      def row(method: str, group: str, uses_u1: str, logc: np.ndarray, *,  # noqa: PLR0913
              fit_s=None, projection_s=None, inference_s=None, note=None,  # noqa: ANN001
              load_before=None) -> dict:  # noqa: ANN001
        ratio = logtrue - logc
        return {
          **base, "method": method, "method_group": group,
          "conditions_on_u1": uses_u1, "env": "vdc",
          "backend": None, "transform": None, "model_version": None,
          "test_nll": -float(np.mean(logc)),
          "excess_nll": float(np.mean(ratio)),
          "excess_nll_mc_se": float(np.std(ratio, ddof=1) / np.sqrt(ratio.size)),
          "n_floored": int(np.sum(logc <= np.log(common.LOG_FLOOR))),
          "min_logpdf": float(np.min(logc)),
          "fit_s": fit_s, "projection_s": projection_s, "inference_s": inference_s,
          "torch_threads": torch.get_num_threads(), "pyvinecopulib_threads": 1,
          "note": note, "load_avg_1m": load_before, **prov,
        }

      rows.append(row("oracle", "oracle", "yes", logtrue, note="true conditional"))
      rows.append(row("pooled_oracle", "oracle", "no", logpool,
                      note="true pooled density (best possible without U1)"))

      for name, (fit, logpdf, note) in methods.items():
        load_before = common.load_average()
        timer.take()
        t0 = time.perf_counter()
        model = fit(uv_tr)
        fit_s = time.perf_counter() - t0
        projection_s = timer.take() if name == "vdc" else None
        t0 = time.perf_counter()
        logc = logpdf(model, uv_te)
        inference_s = time.perf_counter() - t0
        rows.append(row(name, "simplified_pooled", "no", logc, fit_s=fit_s,
                        projection_s=projection_s, inference_s=inference_s,
                        note=note(model), load_before=load_before))

      for name, (fit, logpdf, note) in regime_methods.items():
        load_before = common.load_average()
        logc = np.empty(N_TEST)
        fit_s = inference_s = 0.0
        notes = []
        timer.take()
        for mask_tr, mask_te in ((lo_tr, lo_te), (~lo_tr, ~lo_te)):
          t0 = time.perf_counter()
          model = fit(uv_tr[mask_tr])
          fit_s += time.perf_counter() - t0
          t0 = time.perf_counter()
          logc[mask_te] = logpdf(model, uv_te[mask_te])
          inference_s += time.perf_counter() - t0
          notes.append(note(model))
        projection_s = timer.take() if name == "vdc_regime" else None
        rows.append(row(name, "split_known", "split known", logc, fit_s=fit_s,
                        projection_s=projection_s, inference_s=inference_s,
                        note=" / ".join(notes), load_before=load_before))

      tau = lambda m: float(st.kendalltau(uv_tr[m, 0], uv_tr[m, 1])[0])  # noqa: E731
      diagnostics.append({
        "case": case, "seed": seed,
        "tau_low_train": tau(lo_tr), "tau_high_train": tau(~lo_tr),
        "tau_pooled_train": tau(np.ones(N_TRAIN, dtype=bool)),
        "tau_low_true": float(low.parameters_to_tau(low.parameters)),
        "tau_high_true": float(high.parameters_to_tau(high.parameters)),
        "oracle_nll": oracle_nll, "pooled_oracle_nll": -float(np.mean(logpool)),
      })
      for key, value in (("train", tr), ("test", te), ("logtrue", logtrue),
                         ("log_pooled_oracle", logpool), ("case", case), ("seed", seed)):
        arrays[key].append(value)
      print(f"done {case} seed {seed}", flush=True)

  samples = a.out / "samples.npz"
  np.savez_compressed(samples, **{k: np.asarray(v) for k, v in arrays.items()})
  common.write_csv(rows, a.out / "vdc_side.csv")
  summary = {}
  for case in CASES:
    for method in dict.fromkeys(r["method"] for r in rows):
      vals = [r["excess_nll"] for r in rows if r["case"] == case and r["method"] == method]
      summary[f"{case}/{method}"] = {
        "mean": float(np.mean(vals)), "se": float(np.std(vals, ddof=1) / np.sqrt(len(vals)))}
  common.write_json({
    "experiment": "E3b",
    "description": "non-simplified stress test: data and VDC-env methods",
    "provenance": prov,
    "design": {"cases": {k: {"low": list(v[0]), "high": list(v[1])} for k, v in CASES.items()},
               "split": SPLIT, "u1": "U(0,1), first column of pv.Bicop().simulate",
               "seeds": SEEDS, "seed_vector": "[seed, case_index, split(1=train,2=test), stream]",
               "n_train": N_TRAIN, "n_test": N_TEST, "sample_clip": [1e-10, 1 - 1e-10]},
    "generator_note": "VDC's e18 generator is not released (paper workspace); rebuilt from its saved config and appendix.",
    "published_seed0_excess_nll": {
      "gaussian_sign_flip": {"vdc": 0.2793, "parametric": 0.2906, "tll": 0.2970, "regime_gaussian": 0.0018},
      "clayton_gumbel_tail_switch": {"vdc": 0.0759, "parametric": 0.0883, "tll": 0.0895, "regime_gaussian": 0.1173}},
    "vdc_settings": {"m": 64, "projection_iters": 30, "lookup": "HFuncLookup(use_spline=False).pdf",
                     "torch_threads": 1, "dtype": "float32"},
    "log_floor": common.LOG_FLOOR,
    "wall_s": time.perf_counter() - started,
    "summary_excess_nll": summary,
    "diagnostics": diagnostics,
    "samples_npz": samples.name, "samples_sha256": common.sha256_file(samples),
  }, a.out / "vdc_side.json")
  print(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
