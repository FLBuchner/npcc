#!/usr/bin/env python3
"""E3a, NPCC side: our estimator, unconditional, on VDC's Table 1 samples.

Run in the NPCC environment, from the repository root, after
``e3a_vdc_side.py`` has written ``samples.npz``:

    cd /home/thibault/npcc
    # CPU arms (gradient boosting), one pinned core and one BLAS/OpenMP
    # thread, as VDC's runner times VDC, TLL and parametric selection:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c 13 nice -n 10 uv run --no-sync python scripts/paper/e3/e3a_npcc_fit.py \
        --estimators cpu --device cpu
    # GPU arms (TabPFN v2.5 / v3 / v3.5, TabICL, Nori), once the GPU is free:
    uv run --no-sync python scripts/paper/e3/e3a_npcc_fit.py \
        --estimators gpu --device cuda --gpu-is-free

**What it does.** For each requested E1 logit estimator (see ``e3_npcc.py``)
and each of the 45 fits (nine cases x five seeds) it fits
:class:`npcc.RosenblattBicop` with no covariate on the 2,000 training
pseudo-observations, evaluates the density at the 20,000 test points, and
scores it against the true log-density dumped by the VDC side -- the very
arrays VDC's runner scored, so the samples are identical and nothing is
re-derived here.

**Metric.** KL = mean over test points of ``log c_true - log max(c_hat,
1e-300)``, exactly VDC's runner's definition. ``kl_mc_se`` is the Monte Carlo
standard error of that mean over the test points.

**Outputs** (under ``--out``): one JSON row and one float32 ``.npy`` of
per-point log-densities per fit in ``npcc_cells/<estimator>/`` (so a run
resumes), and ``npcc_<estimator>.csv`` with the 45 rows of each estimator.
Every row carries the provenance of ``e3_common.provenance``.
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

N_WARMUP_TEST = 2000


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
    elif name in e3_npcc.ESTIMATORS:
      out.append(name)
    else:
      raise SystemExit(f"estimator must be cpu, gpu, all or one of {list(e3_npcc.ESTIMATORS)}; got {name}")
  return list(dict.fromkeys(out))


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--samples", type=Path, default=common.RESULTS / "e3a" / "samples.npz")
  p.add_argument("--estimators", nargs="+", default=["cpu"])
  p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
  p.add_argument("--gpu-is-free", action="store_true",
                 help="required for --device cuda; the GPU is shared with E1")
  p.add_argument("--out", type=Path, default=common.RESULTS / "e3a")
  p.add_argument("--only", type=int, nargs="*", default=None,
                 help="fit indices to run (default: all 45); for smoke tests")
  a = p.parse_args()

  labels = resolve(a.estimators)
  for label in labels:
    kind = e3_npcc.ESTIMATORS[label][2]
    if kind == "gpu" and a.device != "cuda":
      raise SystemExit(f"{label} must run with --device cuda; got --device {a.device}")
    if kind == "cpu" and a.device != "cpu":
      raise SystemExit(f"{label} runs on the host; pass --device cpu, got {a.device}")
  if a.device == "cuda" and not a.gpu_is_free:
    raise SystemExit("--device cuda requires --gpu-is-free: the GPU is shared with E1")

  sample_info = json.loads((a.samples.parent / "vdc_side.json").read_text())
  if common.sha256_file(a.samples) != sample_info["samples_sha256"]:
    raise SystemExit(f"{a.samples} must match the sha256 recorded by the VDC side")
  with np.load(a.samples) as f:
    data = {k: f[k] for k in f.files}
  n_fits = data["train"].shape[0]
  indices = range(n_fits) if a.only is None else a.only
  prov = common.provenance(__file__, a.device)

  for label in labels:
    est = e3_npcc.spec(label)
    cell_dir = a.out / "npcc_cells" / label
    cell_dir.mkdir(parents=True, exist_ok=True)
    warmed = False
    for k in indices:
      case, seed = str(data["case"][k]), int(data["seed"][k])
      stem = cell_dir / f"k{k:02d}_{case}_s{seed}"
      if stem.with_suffix(".json").exists():
        continue
      if not warmed:
        t0 = time.perf_counter()
        e3_npcc.fit_and_score(
          est, data["train"][k], data["test"][k][:N_WARMUP_TEST],
          device=a.device, salt=("E3a-warmup",))
        warmup_s = time.perf_counter() - t0
        warmed = True
        print(f"{label}: warm-up {warmup_s:.1f} s", flush=True)
      load_before = common.load_average()
      res = e3_npcc.fit_and_score(
        est, data["train"][k], data["test"][k], device=a.device,
        salt=("E3a", int(data["case_index"][k]), seed))
      ratio = data["logtrue"][k] - res.log_pdf
      row = {
        "experiment": "E3a",
        "case_index": int(data["case_index"][k]), "case": case,
        "family": str(data["family"][k]), "rotation": int(data["rotation"][k]),
        "parameters": str(data["parameters"][k]), "seed": seed,
        "n_train": int(data["train"].shape[1]), "n_test": int(data["test"].shape[1]),
        "method": f"npcc-{label}",
        "method_group": "npcc",
        "conditions_on_u1": None,
        "estimator": label, "estimator_id": est.estimator_id,
        "backend": est.backend, "transform": est.transform,
        "model_version": e3_npcc.ESTIMATORS[label][1],
        "env": "npcc",
        "kl": float(np.mean(ratio)),
        "kl_mc_se": float(np.std(ratio, ddof=1) / np.sqrt(ratio.size)),
        "n_floored": int(np.sum(res.log_pdf <= np.log(common.LOG_FLOOR))),
        "n_nonfinite": res.n_nan,
        "min_logpdf": float(np.min(res.log_pdf)),
        "max_log_ratio": float(np.max(ratio)),
        "fit_s": res.fit_s,
        "projection_s": None,
        "inference_s": res.inference_s,
        "torch_threads": torch.get_num_threads(),
        "backend_threads": (f"library defaults, confined to {len(os.sched_getaffinity(0))} pinned core(s)" if a.device == "cpu" else None),
        "warmup_excluded": True,
        "samples_sha256": sample_info["samples_sha256"],
        "load_avg_1m": load_before,
        **prov,
      }
      np.save(stem.with_suffix(".npy"), res.log_pdf.astype(np.float32))
      common.write_json(row, stem.with_suffix(".json"))
      print(f"{label} k={k:02d} {case} s{seed}: KL {row['kl']:.5f} "
            f"fit {res.fit_s:.2f} s inf {res.inference_s:.2f} s", flush=True)

    rows = [json.loads(path.read_text()) for path in sorted(cell_dir.glob("k*.json"))]
    if len(rows) == n_fits:
      common.write_csv(rows, a.out / f"npcc_{label}.csv")
      print(f"{label}: mean KL {np.mean([r['kl'] for r in rows]):.5f} over {len(rows)} fits", flush=True)


if __name__ == "__main__":
  main()
