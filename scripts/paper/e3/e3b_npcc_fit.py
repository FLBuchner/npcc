#!/usr/bin/env python3
"""E3b, NPCC side: our estimator on the non-simplified stress test.

Run in the NPCC environment, from the repository root, after
``e3b_vdc_side.py`` has written ``samples.npz``:

    cd /home/thibault/npcc
    # CPU arms (gradient boosting), one pinned core and one BLAS/OpenMP
    # thread, as VDC's runner times VDC, TLL and parametric selection:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c 13 nice -n 10 uv run --no-sync python scripts/paper/e3/e3b_npcc_fit.py \
        --estimators cpu --device cpu
    # GPU arms, once the GPU is free:
    uv run --no-sync python scripts/paper/e3/e3b_npcc_fit.py \
        --estimators gpu --device cuda --gpu-is-free

**What it does.** For each requested E1 logit estimator (``e3_npcc.py``) and
each of the 20 fits (two cases x seeds 0-9) it fits
:class:`npcc.RosenblattBicop` on the 2,000 training pairs ``(U2, U3)`` in two
modes:

- ``conditional`` (method ``npcc-<estimator>``): with ``x = U1`` as an
  ``(n, 1)`` covariate -- the pair copula of a non-simplified vine's second
  tree, which receives its conditioning variable;
- ``pooled`` (method ``npcc-<estimator>-pooled``): without ``x``, pooling over
  ``U1`` like VDC, TLL and parametric selection. Same backend and samples, so
  the difference between the two modes is what conditioning buys.

The density is evaluated at the 10,000 test points (with their own ``U1`` in
the conditional mode) and scored against the oracle log-density the VDC side
dumped.

**Metric.** Test NLL = ``-mean log max(c_hat, 1e-300)``; excess NLL = test NLL
- oracle NLL = ``mean(log c_true(u2, u3 | u1) - log c_hat)``, in nats.

**Outputs** (under ``--out``): one JSON row and one float32 ``.npy`` of
per-point log-densities per fit in ``npcc_cells/<estimator>/``, and
``npcc_<estimator>.csv`` with every row of that estimator.
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
from e3a_npcc_fit import resolve  # noqa: E402

N_WARMUP_TEST = 2000
MODES = ("conditional", "pooled")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--samples", type=Path, default=common.RESULTS / "e3b" / "samples.npz")
  p.add_argument("--estimators", nargs="+", default=["cpu"])
  p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
  p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
  p.add_argument("--gpu-is-free", action="store_true",
                 help="required for --device cuda; the GPU is shared with E1")
  p.add_argument("--out", type=Path, default=common.RESULTS / "e3b")
  p.add_argument("--only", type=int, nargs="*", default=None,
                 help="fit indices to run (default: all 20); for smoke tests")
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
  design = sample_info["design"]

  for label in labels:
    est = e3_npcc.spec(label)
    cell_dir = a.out / "npcc_cells" / label
    cell_dir.mkdir(parents=True, exist_ok=True)
    warmed = False
    for k in indices:
      case, seed = str(data["case"][k]), int(data["seed"][k])
      tr, te, logtrue = data["train"][k], data["test"][k], data["logtrue"][k]
      for mode in a.modes:
        stem = cell_dir / f"k{k:02d}_{case}_s{seed}_{mode}"
        if stem.with_suffix(".json").exists():
          continue
        x_tr = tr[:, [0]] if mode == "conditional" else None
        x_te = te[:, [0]] if mode == "conditional" else None
        if not warmed:
          t0 = time.perf_counter()
          e3_npcc.fit_and_score(
            est, tr[:, 1:], te[:N_WARMUP_TEST, 1:], device=a.device,
            salt=("E3b-warmup",), x_train=tr[:, [0]], x_test=te[:N_WARMUP_TEST, [0]])
          print(f"{label}: warm-up {time.perf_counter() - t0:.1f} s", flush=True)
          warmed = True
        load_before = common.load_average()
        res = e3_npcc.fit_and_score(
          est, tr[:, 1:], te[:, 1:], device=a.device,
          salt=("E3b", case, seed, mode), x_train=x_tr, x_test=x_te)
        ratio = logtrue - res.log_pdf
        row = {
          "experiment": "E3b", "case": case, "seed": seed,
          "split": design["split"],
          "n_train": int(tr.shape[0]), "n_test": int(te.shape[0]),
          "n_train_low": int(np.sum(tr[:, 0] < design["split"])),
          "n_test_low": int(np.sum(te[:, 0] < design["split"])),
          "oracle_nll": -float(np.mean(logtrue)),
          "method": f"npcc-{label}" + ("" if mode == "conditional" else "-pooled"),
          "method_group": "npcc_conditional" if mode == "conditional" else "npcc_pooled",
          "conditions_on_u1": "yes" if mode == "conditional" else "no",
          "mode": mode,
          "estimator": label, "estimator_id": est.estimator_id,
          "backend": est.backend, "transform": est.transform,
          "model_version": e3_npcc.ESTIMATORS[label][1],
          "env": "npcc",
          "test_nll": -float(np.mean(res.log_pdf)),
          "excess_nll": float(np.mean(ratio)),
          "excess_nll_mc_se": float(np.std(ratio, ddof=1) / np.sqrt(ratio.size)),
          "n_floored": int(np.sum(res.log_pdf <= np.log(common.LOG_FLOOR))),
          "n_nonfinite": res.n_nan,
          "min_logpdf": float(np.min(res.log_pdf)),
          "fit_s": res.fit_s, "projection_s": None, "inference_s": res.inference_s,
          "torch_threads": torch.get_num_threads(),
          "backend_threads": (f"library defaults, confined to {len(os.sched_getaffinity(0))} pinned core(s)" if a.device == "cpu" else None),
          "warmup_excluded": True,
        "samples_sha256": sample_info["samples_sha256"],
          "load_avg_1m": load_before,
          **prov,
        }
        np.save(stem.with_suffix(".npy"), res.log_pdf.astype(np.float32))
        common.write_json(row, stem.with_suffix(".json"))
        print(f"{label} k={k:02d} {case} s{seed} {mode}: excess {row['excess_nll']:.4f} "
              f"fit {res.fit_s:.2f} s inf {res.inference_s:.2f} s", flush=True)

    rows = [json.loads(path.read_text()) for path in sorted(cell_dir.glob("k*.json"))]
    if len(rows) == n_fits * len(MODES):
      common.write_csv(rows, a.out / f"npcc_{label}.csv")
      for mode in MODES:
        vals = [r["excess_nll"] for r in rows if r["mode"] == mode]
        print(f"{label} {mode}: mean excess NLL {np.mean(vals):.4f}", flush=True)


if __name__ == "__main__":
  main()
