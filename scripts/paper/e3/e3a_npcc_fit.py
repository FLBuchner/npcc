#!/usr/bin/env python3
"""E3a, NPCC side: our estimator, unconditional, on VDC's Table 1 samples.

Run in the NPCC environment, from the repository root, after
``e3a_vdc_side.py`` has written the samples:

    cd /home/thibault/npcc
    # CPU arms (gradient boosting), one pinned core and one BLAS/OpenMP
    # thread, as VDC's runner times VDC, TLL and parametric selection:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c 13 nice -n 10 uv run --no-sync python scripts/paper/e3/e3a_npcc_fit.py \
        --estimators lightgbm-probit --n-train 2000 10000 --device cpu
    # The adapted VDC pair copula (a cross-check of VDC's own lookup):
    ... e3a_npcc_fit.py --estimators vdc-adapted --vdc-core 12 --device cpu
    # GPU arms, once the GPU is free (n_train 10000: main-text TabPFN only):
    uv run --no-sync python scripts/paper/e3/e3a_npcc_fit.py \
        --estimators gpu --n-train 2000 --device cuda --gpu-is-free

**What it does.** For each requested E1 estimator (see ``e3_npcc.py``),
each training size and each of the 45 fits (nine cases x five seeds) it fits
:class:`npcc.RosenblattBicop` with no covariate on the training sample,
evaluates the density at the 20,000 test points and on the ``200 x 200``
midpoint grid, and scores both against the truth the VDC side dumped -- the
very arrays VDC's runner scored at ``n_train = 2000``, so nothing is
re-derived here.

**The ``vdc-adapted`` arm** is VDC's estimate as a pyvinecopulib 1.0 pair
copula (``e3_vdc_pair.VdcBicop``: the 64 x 64 grid from a VDC density server
running in VDC's environment, replicate-padded and held by ``TorchTllBicop``
with ``norm_maxiter=0``, adapted from pyvinecopulib PR #217). It is a
cross-check of VDC's own lookup, not an arm: its grid must equal the VDC
side's bitwise, and its KL, IAE and ISE should agree with the ``vdc`` rows.

**Metrics.** KL = mean over test points of ``log c_true - log max(c_hat,
1e-300)``, VDC's runner's definition, with ``kl_mc_se`` its Monte Carlo
standard error over the test points. On the grid: IAE = mean ``|c_hat - c|``,
ISE = mean ``(c_hat - c)^2``, and the largest density and ratio with their
locations. NPCC's grid density is its own ``pdf_grid`` (both Rosenblatt
directions on the Cartesian product, averaged), spot-checked per fit against
pointwise ``pdf`` at 500 grid points (``grid_spot_check_max_rel``).

**Outputs** (under ``--out``, which defaults to ``--folder``): one JSON row
and one float32 ``.npy`` of test-point log-densities per fit in
``npcc_cells/<estimator>/n<n_train>/`` (so a run resumes), and
``npcc_<estimator>.csv`` with every row of the estimator. Every row carries
the provenance of ``e3_common.provenance``. The samples and truths are always
read from ``--folder``, so a run that must leave it untouched (the qnorm
screen's, with ``--config configs/qnorm_screen.toml``) writes elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_common as common  # noqa: E402
import e3_npcc  # noqa: E402
from e3_vdc_pair import VdcBicop, VdcServer  # noqa: E402
from e3a_vdc_side import grid_metrics  # noqa: E402

N_WARMUP_TEST = 2000
N_SPOT = 500
VDC_ADAPTED = "vdc-adapted"


def resolve(names: list[str]) -> list[str]:
  """Expand ``cpu`` / ``gpu`` / ``all`` into estimator labels."""
  out: list[str] = []
  for name in names:
    if name == "cpu":
      out += e3_npcc.CPU_ESTIMATORS
    elif name == "gpu":
      out += e3_npcc.GPU_ESTIMATORS
    elif name == "all":
      out += list(e3_npcc.ESTIMATORS)
    elif name in e3_npcc.KNOWN or name == VDC_ADAPTED:
      out.append(name)
    else:
      raise SystemExit(f"estimator must be cpu, gpu, all, {VDC_ADAPTED} or one of "
                       f"{list(e3_npcc.KNOWN)}; got {name}")
  return list(dict.fromkeys(out))


def samples_path(folder: Path, n_train: int) -> Path:
  return folder / ("samples.npz" if n_train == 2000 else f"samples_n{n_train}.npz")


class VdcAdapted:
  """The adapted VDC pair copula, fitted through a VDC density server."""

  def __init__(self, cpu_core: int | None) -> None:
    self.server = VdcServer(cpu_core=cpu_core)
    VdcBicop.server = self.server
    self.cls = VdcBicop
    self.pair = None

  def fit(self, train: np.ndarray) -> float:
    t0 = time.perf_counter()
    self.pair = self.cls.from_data(train)
    return time.perf_counter() - t0

  def pdf(self, uv: np.ndarray) -> np.ndarray:
    assert self.pair is not None
    return self.pair.pdf(torch.as_tensor(uv, dtype=torch.float64)).numpy()


def main() -> None:  # noqa: PLR0915 - one linear protocol
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--folder", type=Path, default=common.RESULTS / "e3a",
                 help="the VDC side's samples and truths")
  p.add_argument("--out", type=Path, default=None,
                 help="where the fits go (default: --folder)")
  p.add_argument("--config", type=Path, default=e3_npcc.E1_CONFIG,
                 help="the config the estimators are read from (default: E1's)")
  p.add_argument("--estimators", nargs="+", default=["cpu"])
  p.add_argument("--n-train", type=int, nargs="+", default=[2000, 10000])
  p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
  p.add_argument("--gpu-is-free", action="store_true",
                 help="required for --device cuda; the GPU is shared with E1")
  p.add_argument("--vdc-core", type=int, default=None,
                 help="core to pin the VDC density server to (vdc-adapted only)")
  p.add_argument("--only", type=int, nargs="*", default=None,
                 help="fit indices to run (default: all 45); for smoke tests")
  a = p.parse_args()

  labels = resolve(a.estimators)
  out = a.folder if a.out is None else a.out
  for label in labels:
    kind = "cpu" if label == VDC_ADAPTED else e3_npcc.KNOWN[label][2]
    if kind == "gpu" and a.device != "cuda":
      raise SystemExit(f"{label} must run with --device cuda; got --device {a.device}")
    if kind == "cpu" and a.device != "cpu":
      raise SystemExit(f"{label} runs on the host; pass --device cpu, got {a.device}")
  if a.device == "cuda" and not a.gpu_is_free:
    raise SystemExit("--device cuda requires --gpu-is-free: the GPU is shared with E1")

  info = json.loads((a.folder / "vdc_side.json").read_text())
  with np.load(a.folder / "grid_truth.npz") as f:
    g, grid_true = f["midpoints"], f["pdf_true"]
  uu, vv = np.meshgrid(g, g, indexing="ij")
  grid_pts = np.column_stack([uu.ravel(), vv.ravel()])
  spot = np.random.default_rng(20261105).choice(grid_pts.shape[0], N_SPOT, replace=False)
  prov = common.provenance(__file__, a.device)

  for n_train in a.n_train:
    path = samples_path(a.folder, n_train)
    sha = info["samples_sha256_by_n"][str(n_train)]
    if common.sha256_file(path) != sha:
      raise SystemExit(f"{path} must match the sha256 recorded by the VDC side")
    with np.load(path) as f:
      data = {k: f[k] for k in f.files}
    vdc_grids = None
    if VDC_ADAPTED in labels:
      with np.load(a.folder / f"vdc_grids_n{n_train}.npz") as f:
        vdc_grids = f["grids"]
    n_fits = data["train"].shape[0]
    indices = range(n_fits) if a.only is None else a.only

    for label in labels:
      adapted = label == VDC_ADAPTED
      est = None if adapted else e3_npcc.spec(label, a.config)
      runner = VdcAdapted(a.vdc_core) if adapted else None
      cell_dir = out / "npcc_cells" / label / f"n{n_train}"
      cell_dir.mkdir(parents=True, exist_ok=True)
      warmed = adapted  # the server warms itself up
      try:
        for k in indices:
          case, seed = str(data["case"][k]), int(data["seed"][k])
          i_case = int(data["case_index"][k])
          stem = cell_dir / f"k{k:02d}_{case}_s{seed}"
          if stem.with_suffix(".json").exists():
            continue
          tr, te = data["train"][k], data["test"][k]
          if not warmed:
            t0 = time.perf_counter()
            e3_npcc.fit_and_score(est, tr, te[:N_WARMUP_TEST], device=a.device,
                                  salt=("E3a-warmup",))
            print(f"{label}: warm-up {time.perf_counter() - t0:.1f} s", flush=True)
            warmed = True
          load_before = common.load_average()
          extra: dict = {}
          if adapted:
            fit_s = runner.fit(tr)
            t0 = time.perf_counter()
            log_pdf = np.log(np.maximum(runner.pdf(te), common.LOG_FLOOR))
            inference_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            gpdf = runner.pdf(grid_pts)
            grid_s = time.perf_counter() - t0
            timing = runner.pair.vdc_timing
            extra = {"vdc_server_code_sha256": json.dumps(runner.server.info["code_sha256"], sort_keys=True),
                     "server_s": timing["server_s"], "projection_s": timing["projection_s"],
                     "ipc_s": timing["ipc_s"], "roundtrip_s": timing["roundtrip_s"],
                     "grid_equal_vdc_side": bool(np.array_equal(runner.pair.vdc_grid, vdc_grids[k])),
                     "grid_max_abs_diff_vdc_side": float(np.max(np.abs(runner.pair.vdc_grid - vdc_grids[k])))}
            n_nan = int(np.sum(~np.isfinite(gpdf)))
            spot_rel = None
          else:
            res = e3_npcc.fit_and_score(est, tr, te, device=a.device,
                                        salt=("E3a", i_case, seed, n_train),
                                        grid=g, spot=grid_pts[spot])
            fit_s, inference_s, grid_s = res.fit_s, res.inference_s, res.grid_s
            log_pdf, gpdf, n_nan = res.log_pdf, res.grid_pdf.ravel(), res.n_nan
            spot_rel = float(np.max(np.abs(res.spot_pdf - gpdf[spot]) / np.abs(res.spot_pdf)))
          ratio = data["logtrue"][k] - log_pdf
          row = {
            "experiment": "E3a", "case_index": i_case, "case": case,
            "family": str(data["family"][k]), "rotation": int(data["rotation"][k]),
            "parameters": str(data["parameters"][k]), "seed": seed,
            "n_train": int(tr.shape[0]), "n_test": int(te.shape[0]),
            "method": "vdc_adapted" if adapted else f"npcc-{label}",
            "method_group": "cross_check" if adapted else "npcc",
            "role": "cross_check" if adapted else "arm",
            "conditions_on_u1": None,
            "estimator": label, "estimator_id": None if adapted else est.estimator_id,
            "backend": "vdc" if adapted else est.backend,
            "transform": None if adapted else est.transform,
            "feature_transform": None if adapted else est.feature_transform,
            "model_version": None if adapted else e3_npcc.KNOWN[label][1],
            "env": "npcc",
            "kl": float(np.mean(ratio)),
            "kl_mc_se": float(np.std(ratio, ddof=1) / np.sqrt(ratio.size)),
            "n_floored": int(np.sum(log_pdf <= np.log(common.LOG_FLOOR))),
            "n_nonfinite": n_nan,
            "min_logpdf": float(np.min(log_pdf)),
            "max_log_ratio": float(np.max(ratio)),
            **grid_metrics(gpdf, grid_true[i_case].ravel(), g),
            "grid_n": int(g.size),
            "grid_spot_check_max_rel": spot_rel,
            "fit_s": fit_s, "projection_s": extra.pop("projection_s", None),
            "inference_s": inference_s, "grid_s": grid_s,
            **extra,
            "torch_threads": torch.get_num_threads(),
            "backend_threads": (f"library defaults, confined to {len(os.sched_getaffinity(0))} "
                                "pinned core(s)" if a.device == "cpu" else None),
            "warmup_excluded": True,
            "samples_sha256": sha,
            "load_avg_1m": load_before,
            **prov,
          }
          np.save(stem.with_suffix(".npy"), log_pdf.astype(np.float32))
          common.write_json(row, stem.with_suffix(".json"))
          print(f"{label} n={n_train} k={k:02d} {case} s{seed}: KL {row['kl']:.5f} "
                f"IAE {row['iae']:.4f} fit {fit_s:.2f} s inf {inference_s:.2f} s "
                f"grid {grid_s:.2f} s", flush=True)
      finally:
        if runner is not None:
          runner.server.close()

  for label in labels:
    rows = [json.loads(path.read_text())
            for path in sorted((out / "npcc_cells" / label).glob("n*/k*.json"))]
    if rows:
      common.write_csv(rows, out / f"npcc_{label}.csv")
      for n in sorted({r["n_train"] for r in rows}):
        sel = [r for r in rows if r["n_train"] == n]
        print(f"{label} n={n}: {len(sel)} fits, mean KL {np.mean([r['kl'] for r in sel]):.5f}, "
              f"mean IAE {np.mean([r['iae'] for r in sel]):.4f}", flush=True)


if __name__ == "__main__":
  main()
