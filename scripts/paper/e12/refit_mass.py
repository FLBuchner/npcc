"""E12's check of the truth-only estimate: E3d's CatBoost cells, refitted.

``mass_e3.py`` estimates a fitted vine's mass by importance sampling under
the truth, which only looks where the truth puts its mass. This script
estimates it again, for E3d's CatBoost arms, with a proposal that covers the
whole cube.

**Refit.** For an E3d cell, ``npcc-catboost-probit-<variant>`` is refitted
as ``scripts/paper/e3/e3d_npcc_fit.py`` fitted it: E1's estimator spec, the
variant's structure and context, the same torch seed, on the CPU. Its
log-density at the cell's 20,000 test points is compared with E3d's stored
``.npy`` (``reproduce_max_abs``; the stored values are float32).

**Proposal.** A defensive mixture

    q = (f_true + g_tll + 1) / 3,

with ``f_true`` the true vine, ``g_tll`` pyvinecopulib's TLL with Dissmann
selection fitted on the cell's training data, and ``1`` the uniform density
on [0, 1]^d. ``--n-per`` points are drawn from each component, and

    m_hat = (1/3) sum_j mean_{i in j} f_hat(y_i) / q(y_i)

is stratified multiple importance sampling: unbiased wherever ``q > 0``,
with SE ``sqrt(sum_j var_j / n_j) / 3``. The truth, TLL and the parametric
arm (Dissmann with BIC), all proper densities, are scored the same way as
controls. Each fitted arm is also scored by importance sampling under the
truth on E3d's test points (``mass_is_truth_test``, ``mass_e3.py``'s
estimate) and on the new draws from the truth (``mass_is_truth_new``).

**Code.** The refit must run the code E3d's CatBoost cells were fitted
with, b7702e1. ``--code`` names a checkout of it: its ``scripts/paper/e3``
supplies the estimator spec (``e3_npcc``) and the decayed pair copulas
(``e3c_decay_vdc_side``), and its ``src`` must be the ``npcc`` that is
imported, which ``PYTHONPATH`` decides; the script refuses otherwise.

**Outputs**, under ``--out``:

- ``refit_results.jsonl``, appended to: per cell, one line for the controls
  (``truth``, ``tll``, ``parametric``: the mixture-proposal mass, SE and
  per-component means; the competitors' truth-only estimates; the truth's
  check against the stored log-density) and one per variant (refit check,
  mixture-proposal mass, SE and per-component means, both truth-only
  estimates, excess NLL, seconds). The same lines are printed.
- ``refit/<arm>_<cell>.npz``: the arm's log-density at the pooled draws
  (``lp_y``), their component (``comp``), ``log q`` and the true
  log-density (``lf``), float32.

**Usage** (from the repository root, on the CPU). With three variants, a
d = 5 cell takes about 2 min and a d = 10 cell about 8 min, on two cores'
worth of CPU: torch runs two threads, and CatBoost one per core the process
may use::

    git worktree add --detach ../npcc-b7702e1 b7702e1
    PYTHONPATH=../npcc-b7702e1/src CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 \\
      uv run --no-sync python scripts/paper/e12/refit_mass.py \\
      --code ../npcc-b7702e1 --cells decay_d5_s101 strong_d10_s202 \\
      --variants tll-ns random-ns true-s --out results/e12_<TAG>

``show_refit.py`` prints the JSON lines as a table.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pyvinecopulib as pv
import torch
from pyvinecopulib.core import NonSimplifiedContext, SimplifiedContext

import npcc
from npcc import FitControlsRosenblattVinecop, RosenblattVinecop
from npcc.experiments._study import derive_seed

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results"

#: E3d's estimator label, and the arm names built from it.
LABEL = "catboost-probit"

#: The parametric arm's families, as E3d's ``parametric``.
PARAMETRIC_FAMILIES = (
  "indep",
  "gaussian",
  "student",
  "clayton",
  "gumbel",
  "frank",
  "joe",
  "bb1",
  "bb6",
  "bb7",
  "bb8",
  "tawn",
)

#: E3d's variants: the structure fitted on, and the context (ns or s).
VARIANTS = {
  "tll-ns": ("tll", "ns"),
  "random-ns": ("random", "ns"),
  "true-s": ("true", "s"),
  "random-s": ("random", "s"),
}


def load_code(code: Path) -> tuple[ModuleType, Callable]:
  """E3's estimator spec module and decayed pair copulas, from ``code``.

  Parameters
  ----------
  code : Path
      The npcc checkout whose code the refit runs.

  Returns
  -------
  tuple
      The ``e3_npcc`` module and ``e3c_decay_vdc_side.decayed_bicop``.
  """
  src = (code / "src").resolve()
  if not Path(npcc.__file__).resolve().is_relative_to(src):
    raise SystemExit(
      f"npcc must be imported from {src}, the --code checkout's; got "
      f"{npcc.__file__}. Run with PYTHONPATH={src}."
    )
  sys.path.insert(0, str(code / "scripts" / "paper" / "e3"))
  e3_npcc = importlib.import_module("e3_npcc")
  decay = importlib.import_module("e3c_decay_vdc_side")
  return e3_npcc, decay.decayed_bicop


def compiled_controls(name: str) -> pv.FitControlsVinecop:
  """E3d's controls for its ``parametric`` or ``tll`` arm."""
  if name.startswith("parametric"):
    return pv.FitControlsVinecop(
      family_set=[getattr(pv.families, f) for f in PARAMETRIC_FAMILIES],
      selection_criterion="bic",
      num_threads=1,
    )
  return pv.FitControlsVinecop(family_set=[pv.families.tll], num_threads=1)


def case_bicop(case: tuple) -> pv.Bicop:
  """One of E3d's edge cases, ``(family, parameters, rotation)``."""
  family, params, rotation = case
  if family == "indep":
    return pv.Bicop()
  return pv.Bicop(
    family=getattr(pv.BicopFamily, family),
    rotation=rotation,
    parameters=np.asarray(params),
  )


def truth_vine(
  data: dict[str, np.ndarray],
  info: dict,
  design: str,
  d: int,
  seed: int,
  decayed_bicop: Callable,
) -> pv.Vinecop:
  """The true vine of an E3d cell, from its edge cases and matrix."""
  key = f"{design}_d{d}_s{seed}"
  cases = info["design_spec"]["cases"]
  idx = list(data[f"{key}_edge_cases"])
  rows, k = [], 0
  for t in range(d - 1):
    rows.append(idx[k : k + d - 1 - t])
    k += d - 1 - t
  if design == "decay":
    pcs = [
      [decayed_bicop(pv, cases[j], t)[0] for j in row]
      for t, row in enumerate(rows)
    ]
  else:
    pcs = [[case_bicop(cases[j]) for j in row] for row in rows]
  s = pv.RVineStructure.from_matrix(data[f"{key}_true_matrix"])
  return pv.Vinecop.from_structure(structure=s, pair_copulas=pcs)


def npcc_logpdf(
  vine: RosenblattVinecop, y: np.ndarray, batch: int = 20000
) -> np.ndarray:
  """The vine's log-density at ``y``, in batches, as float64."""
  out = []
  with torch.inference_mode():
    for i in range(0, y.shape[0], batch):
      chunk = torch.as_tensor(y[i : i + batch], dtype=torch.float64)
      out.append(vine.logpdf(chunk).cpu().numpy())
  return np.concatenate(out).astype(np.float64)


def mis_mass(
  logp_hat: np.ndarray, logq: np.ndarray, comp: np.ndarray, n_comp: int
) -> tuple[float, float, list[float], int]:
  """The stratified estimate of mass, its SE, means per component, non-finite.

  Parameters
  ----------
  logp_hat : np.ndarray
      The density's log at the pooled draws.
  logq : np.ndarray
      The proposal's log-density there.
  comp : np.ndarray
      Which component drew each point, ``0`` to ``n_comp - 1``.
  n_comp : int
      The number of components, each weighted ``1 / n_comp`` in ``q``.

  Returns
  -------
  tuple
      The mass, its SE, the mean weight per component and the number of
      non-finite weights, which are left out.
  """
  w = np.exp(logp_hat - logq)
  fin = np.isfinite(w)
  means, vars_ = [], []
  for j in range(n_comp):
    wj = w[(comp == j) & fin]
    means.append(wj.mean())
    vars_.append(wj.var(ddof=1) / wj.size)
  m = float(np.mean(means))
  se = float(np.sqrt(np.sum(vars_)) / n_comp)
  return m, se, [float(x) for x in means], int((~fin).sum())


def is_truth(logp_hat: np.ndarray, logtrue: np.ndarray) -> tuple[float, float]:
  """Truth-only importance-sampling mass and SE, over the finite ratios."""
  r = np.exp(logp_hat - logtrue)
  r = r[np.isfinite(r)]
  return float(r.mean()), float(r.std(ddof=1) / np.sqrt(r.size))


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument(
    "--cells", nargs="+", required=True, help="<design>_d<d>_s<seed>"
  )
  p.add_argument(
    "--variants",
    nargs="+",
    default=["tll-ns", "random-ns"],
    choices=list(VARIANTS),
  )
  p.add_argument("--n-per", type=int, default=40000, help="draws per component")
  p.add_argument(
    "--folder",
    type=Path,
    default=RESULTS / "e3d_unknown_structure",
    help="E3d's results: samples.npz, truths.json and cells/",
  )
  p.add_argument(
    "--code",
    type=Path,
    default=REPO,
    help="the npcc checkout to fit with (E3d's CatBoost cells: b7702e1)",
  )
  p.add_argument("--torch-threads", type=int, default=2)
  p.add_argument("--out", type=Path, required=True)
  a = p.parse_args()
  e3_npcc, decayed_bicop = load_code(a.code)
  torch.set_num_threads(a.torch_threads)
  (a.out / "refit").mkdir(parents=True, exist_ok=True)
  jsonl = a.out / "refit_results.jsonl"

  def emit(row: dict) -> None:
    line = json.dumps(row)
    print(line, flush=True)
    with jsonl.open("a") as f:
      f.write(line + "\n")

  info = json.loads((a.folder / "truths.json").read_text())
  with np.load(a.folder / "samples.npz") as f:
    data = {k: f[k] for k in f.files}
  est = e3_npcc.spec(LABEL)
  controls = est.controls(
    FitControlsRosenblattVinecop,
    device="cpu",
    projection_grid_size=e3_npcc.PROJECTION_GRID_SIZE,
    batch_size=None,
  )
  for cell in a.cells:
    design, dd, ss = cell.split("_")
    d, seed = int(dd[1:]), int(ss[1:])
    key = cell
    tr, te = data[f"{key}_train"], data[f"{key}_test"]
    logtrue = data[f"{key}_logtrue"]
    truth = truth_vine(data, info, design, d, seed, decayed_bicop)
    lt_te = np.log(np.maximum(truth.pdf(te), 1e-300))
    chk_truth = float(np.max(np.abs(lt_te - logtrue)))
    t0 = time.perf_counter()
    tll = pv.Vinecop.from_data(tr, controls=compiled_controls("tll"))
    par = pv.Vinecop.from_data(tr, controls=compiled_controls("parametric"))
    # The proposal's draws: n from each of the truth, TLL and the uniform.
    n = a.n_per
    y_f = np.clip(truth.simulate(n, seeds=[seed, d, 777]), 1e-10, 1 - 1e-10)
    y_g = np.clip(tll.simulate(n, seeds=[seed, d, 778]), 1e-10, 1 - 1e-10)
    y_u = np.clip(
      np.random.default_rng([seed, d, 779]).random((n, d)), 1e-10, 1 - 1e-10
    )
    y = np.vstack([y_f, y_g, y_u])
    comp = np.repeat([0, 1, 2], n)
    lf = np.log(np.asarray(truth.pdf(y), dtype=np.float64))
    lg = np.log(np.asarray(tll.pdf(y), dtype=np.float64))
    lpar = np.log(np.asarray(par.pdf(y), dtype=np.float64))
    logq = np.logaddexp(np.logaddexp(lf, lg), 0.0) - np.log(3.0)
    emit(
      {
        "cell": cell,
        "n_per": n,
        "truth_check_max_abs": chk_truth,
        "truth": mis_mass(lf, logq, comp, 3)[:2],
        "tll": mis_mass(lg, logq, comp, 3)[:3],
        "parametric": mis_mass(lpar, logq, comp, 3)[:3],
        "parametric_is_truth": is_truth(
          np.log(np.asarray(par.pdf(te))), logtrue
        ),
        "tll_is_truth": is_truth(np.log(np.asarray(tll.pdf(te))), logtrue),
        "competitors_s": time.perf_counter() - t0,
      }
    )
    for variant in a.variants:
      arm = f"npcc-{LABEL}-{variant}"
      which, simp = VARIANTS[variant]
      context = NonSimplifiedContext() if simp == "ns" else SimplifiedContext()
      if which == "tll":
        tll_fit = pv.Vinecop.from_data(tr, controls=compiled_controls("tll"))
        s_used = tll_fit.structure
      elif which == "true":
        s_used = pv.RVineStructure.from_matrix(data[f"{key}_true_matrix"])
      else:
        s_used = pv.RVineStructure.from_matrix(data[f"{key}_random_matrix"])
      torch.manual_seed(
        derive_seed("E3d", design, d, seed, variant, est.estimator_id)
      )
      t0 = time.perf_counter()
      vine = RosenblattVinecop(None, s_used, device="cpu", context=context)
      vine.fit(torch.as_tensor(tr, dtype=torch.float64), controls)
      fit_s = time.perf_counter() - t0
      lp_te = npcc_logpdf(vine, te)
      stored = np.load(a.folder / "cells" / arm / f"{key}.npy")
      diff = np.abs(lp_te - stored.astype(np.float64))
      t0 = time.perf_counter()
      lp_y = npcc_logpdf(vine, y)
      eval_s = time.perf_counter() - t0
      m, se, by, nnf = mis_mass(lp_y, logq, comp, 3)
      m_is, se_is = is_truth(lp_te, logtrue)
      # The draws from the truth are a second, independent sample for the
      # truth-only estimate.
      m_is2, se_is2 = is_truth(lp_y[comp == 0], lf[comp == 0])
      emit(
        {
          "cell": cell,
          "arm": arm,
          "fit_s": fit_s,
          "eval_s": eval_s,
          "reproduce_max_abs": float(np.nanmax(diff)),
          "reproduce_median_abs": float(np.nanmedian(diff)),
          "mass_mis": m,
          "mass_mis_se": se,
          "mass_mis_by_component": by,
          "n_nonfinite": nnf,
          "mass_is_truth_test": m_is,
          "mass_is_truth_test_se": se_is,
          "mass_is_truth_new": m_is2,
          "mass_is_truth_new_se": se_is2,
          "excess_nll_test": float(np.mean(logtrue - lp_te)),
        }
      )
      np.savez_compressed(
        a.out / "refit" / f"{arm}_{key}.npz",
        lp_y=lp_y.astype(np.float32),
        comp=comp,
        logq=logq.astype(np.float32),
        lf=lf.astype(np.float32),
      )
      del vine


if __name__ == "__main__":
  main()
