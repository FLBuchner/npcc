#!/usr/bin/env python3
"""E6, the §3.3 part: VDC's own vine, timed with and without its order selection.

``e6_timing.sh`` runs this in the VDC environment (Python 3.10, VDC v0.2.0),
from the repository root, pinned to one core:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    taskset -c 4 /home/thibault/vdc/env/bin/python scripts/paper/e3/e6_timing_vdc.py

It times VDC's default vine pipeline as E3d ran it (``e3d_vdc_side.py``): the
D-vine with UCI-runner settings (``VDC_VINE_SETTINGS``), one torch thread,
CPU. The samples are E3d's own (``results/e3d_unknown_structure/``), decay
design by default. Each cell is timed twice:

- *with* selection: no order is given, so VDC picks a greedy order from the
  data, as in E3d;
- *without* selection: the order VDC picked is passed back, so the two fits
  differ by the selection alone.

Inference is ``logpdf`` at the test points. One fit and one evaluation of the
first cell run first, untimed. Every timed cell records its held-out
log-likelihood, which ``e6_timing.py collect`` compares with E3d's row.

**Output.** ``results/e6_timing/e3/e3d/vdc.csv`` (``--out``), one row per cell
and setting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_common as common  # noqa: E402
from e3c_vdc_side import VDC_VINE_SETTINGS  # noqa: E402

E3D = common.NPCC_REPO / "results" / "e3d_unknown_structure"


def main() -> None:  # noqa: PLR0915 - one linear protocol
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--checkpoint", type=Path, default=common.CHECKPOINT)
  p.add_argument("--design", default="decay", choices=["decay", "strong"])
  p.add_argument("--dims", type=int, nargs="+", default=[5, 10])
  p.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
  p.add_argument("--n-test", type=int, default=None,
                 help="evaluate at the first N test points only: smoke tests, which collect refuses")
  p.add_argument("--out", type=Path, default=common.NPCC_REPO / "results" / "e6_timing" / "e3" / "e3d")
  a = p.parse_args()

  import torch

  import vdc
  from vdc.pretrained import load_checkpoint_bundle
  from vdc.vine.api import VineCopulaModel

  assert Path(vdc.__file__).resolve().is_relative_to(common.VDC_REPO)
  assert common.sha256_file(a.checkpoint) == common.CHECKPOINT_SHA256
  info = json.loads((E3D / "truths.json").read_text())
  samples = E3D / "samples.npz"
  if common.sha256_file(samples) != info["samples_sha256"]:
    raise SystemExit(f"{samples} must match the sha256 recorded by e3d_truth.py")
  with np.load(samples) as f:
    data = {k: f[k] for k in f.files}
  keys = [f"{a.design}_d{d}_s{seed}" for d in a.dims for seed in a.seeds]
  prov = common.provenance(__file__, "cpu")

  # E3d's numerical settings (e3d_vdc_side.py).
  torch.set_num_threads(1)
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = True
  torch.backends.cudnn.benchmark = False
  bundle = load_checkpoint_bundle(a.checkpoint, device="cpu")
  ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
  bundle.model.load_state_dict(ck["model_state_dict"], strict=True)
  del ck
  settings = {**VDC_VINE_SETTINGS, "device": "cpu"}

  def run(tr: np.ndarray, te: np.ndarray, order: list[int] | None) -> tuple:
    model = VineCopulaModel(**settings) if order is None else VineCopulaModel(**settings, order=order)
    t0 = time.perf_counter()
    model.fit(tr, diffusion_model=bundle.model, diffusion=None, verbose=False)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    lp = np.asarray(model.logpdf(te), dtype=np.float64)
    return model, fit_s, time.perf_counter() - t0, lp

  rows: list[dict] = []
  out = a.out / "vdc.csv"
  floor = np.log(common.LOG_FLOOR)

  def record(cell: dict, setting: str, order: list[int], timed: tuple, load: float) -> None:
    fit_s, inf_s, lp = timed
    finite = np.isfinite(lp)
    rows.append({
      **cell, "method": "vdc", "setting": setting,
      "selected_order": json.dumps([o + 1 for o in order]),
      "test_loglik": float(np.mean(np.maximum(lp, floor)[finite])),
      "n_nonfinite": int(np.sum(~finite)), "fit_s": fit_s, "inference_s": inf_s,
      "load_avg_1m": load, "env": "vdc", "torch_threads": torch.get_num_threads(),
      "cores": len(os.sched_getaffinity(0)), "warmup_excluded": True,
      "vdc_settings": json.dumps(settings), "samples_sha256": info["samples_sha256"], **prov,
    })
    common.write_csv(rows, out)
    print(f"vdc {cell['design']} d={cell['d']} s{cell['seed']} {setting}: "
          f"fit {fit_s:.1f} s, inference {inf_s:.1f} s", flush=True)

  for i, key in enumerate(keys):
    design, d, seed = key.split("_")
    tr, te = data[f"{key}_train"], data[f"{key}_test"][:a.n_test]
    cell = {"section": "e3d", "design": design, "d": int(d[1:]), "seed": int(seed[1:]),
            "n_train": int(tr.shape[0]), "n_test": int(te.shape[0])}
    if i == 0:  # warm-up, untimed
      model, *_ = run(tr, te, None)
      run(tr, te, [int(o) for o in model.structure.order])
    load = common.load_average()
    model, *timed = run(tr, te, None)
    order = [int(o) for o in model.structure.order]
    record(cell, "with selection (greedy D-vine order)", order, tuple(timed), load)
    load = common.load_average()
    model, *timed = run(tr, te, order)
    record(cell, "without selection (the selected order given)",
           [int(o) for o in model.structure.order], tuple(timed), load)
  print(f"wrote {len(rows)} rows to {out}", flush=True)


if __name__ == "__main__":
  main()
