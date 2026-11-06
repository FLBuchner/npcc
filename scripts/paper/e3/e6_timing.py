#!/usr/bin/env python3
"""E6, the §3.3 part: clean timings of E3a's pair copulas and E3d's vines.

``e6_timing.sh`` runs the whole part, one job at a time, on an idle machine
(E6's ``scripts/paper/e6/run_all.sh`` calls it), and documents the order, the
core pinning and the thread counts. This script has two subcommands, both run
in the NPCC environment from the repository root:

    uv run --no-sync python scripts/paper/e3/e6_timing.py vines --part competitors
    uv run --no-sync python scripts/paper/e3/e6_timing.py vines --part npcc \
        --device cuda --gpu-is-free --dims 5 --seeds 101 202 303
    uv run --no-sync python scripts/paper/e3/e6_timing.py collect

**vines** times E3d's vines on E3d's own samples
(``results/e3d_unknown_structure/samples.npz``, checked against
``truths.json``), decay design by default:

- ``--part competitors``: pyvinecopulib 1.0's compiled vine with E3d's
  controls (``e3d_npcc_fit.compiled_controls``, one thread), for TLL and
  for parametric family selection by BIC. Each is fit *with* structure
  selection (Dissmann's algorithm, ``Vinecop.from_data``) and *without* it
  (the structure that selection returned, passed back as ``structure=``),
  so the two fits differ by the selection alone. Inference is ``pdf`` at the
  test points;
- ``--part npcc``: ``npcc.RosenblattVinecop`` with an E1 logit estimator
  (the main TabPFN v3 by default). It is fit on E3d's random structure,
  non-simplified: arm (a), which selects nothing. Controls and torch seeds
  are E3d's. Inference is ``logpdf`` at the test points.

VDC's own vine, with and without its order selection, is
``e6_timing_vdc.py``.

**Warm-up.** Each method runs once, untimed, before its first timed cell.
A competitor fits and evaluates the first cell. The NPCC vine fits a
three-variable vine on the first cell's first columns and evaluates it at
2,000 test points.

**Same computation.** Every timed cell records its held-out
log-likelihood. ``collect`` compares it with E3d's row for the same cell
and method (``loglik_gap_vs_e3``), so the time is that of the computation
E3d reported.

**collect** writes ``<root>/e3_timing.csv`` (``--root``, by default
``results/e6_timing``, the E6 folder E6's ``consolidate.py`` reads), one row
per section, method, setting, d, n, seed and stage. It reads, under
``<root>/e3/``:

- E3a's timing run in ``e3a/``: ``e3a_vdc_side.py`` and ``e3a_npcc_fit.py``,
  E3a's own scripts with E3a's warm-ups. Only the seeds given by
  ``--e3a-seeds`` are kept, so every method is timed on the same samples;
- the vine parts in ``e3d/``: ``competitors_d<dims>.csv`` and
  ``npcc_<label>_d<dims>.csv`` from ``vines``, and ``vdc.csv`` from
  ``e6_timing_vdc.py``.

Fit and inference are the stages. VDC's pair-copula fit also reports its
Sinkhorn projection as a ``projection`` stage, which is part of ``fit``.
``threads`` is the number of cores the process was pinned to; every BLAS and
OpenMP pool, and torch, is set to that number. ``compute`` is ``gpu`` for a
method that ran on the GPU, else ``cpu``. ``collect`` also writes
``<root>/e3/e3_timing_code_check.csv``: every recorded code hash against git
history and disk, as ``e3_collect.py`` checks E3's own rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_common as common  # noqa: E402

E3D = common.NPCC_REPO / "results" / "e3d_unknown_structure"
ROOT = common.NPCC_REPO / "results" / "e6_timing"
MAIN_TABPFN = "tabpfn-quantiles-v3-logit"
N_WARMUP_TEST = 2000
COLUMNS = ["section", "method", "setting", "d", "n", "seed", "stage", "seconds", "device",
           "compute", "cpu", "threads", "npcc_commit", "case", "design", "n_test", "gpu",
           "torch_threads", "load_avg_1m", "warmup_excluded", "loglik_gap_vs_e3", "script",
           "script_sha256"]


def e3d_cells(design: str, dims: list[int], seeds: list[int]) -> tuple[dict, dict, list[str]]:
  """E3d's samples, checked, and the keys of the requested cells in order."""
  info = json.loads((E3D / "truths.json").read_text())
  path = E3D / "samples.npz"
  if common.sha256_file(path) != info["samples_sha256"]:
    raise SystemExit(f"{path} must match the sha256 recorded by e3d_truth.py")
  with np.load(path) as f:
    data = {k: f[k] for k in f.files}
  keys = [f"{design}_d{d}_s{seed}" for d in dims for seed in seeds]
  missing = [k for k in keys if f"{k}_train" not in data]
  if missing:
    raise SystemExit(f"cells must exist in {path}; got missing {missing}")
  return info, data, keys


def vine_row(key: str, tr: np.ndarray, te: np.ndarray, lp: np.ndarray) -> dict:
  """The cell's identity and its held-out log-likelihood."""
  design, d, seed = key.split("_")
  finite = np.isfinite(lp)
  return {"design": design, "d": int(d[1:]), "seed": int(seed[1:]),
          "n_train": int(tr.shape[0]), "n_test": int(te.shape[0]),
          "test_loglik": float(np.mean(np.maximum(lp, np.log(common.LOG_FLOOR))[finite])),
          "n_nonfinite": int(np.sum(~finite))}


def time_competitors(data: dict, keys: list[str], n_test: int | None, write) -> None:  # noqa: ANN001
  import pyvinecopulib as pv
  from e3d_npcc_fit import compiled_controls

  def run(tr: np.ndarray, te: np.ndarray, ctl, structure=None) -> tuple:  # noqa: ANN001
    t0 = time.perf_counter()
    vc = (pv.Vinecop.from_data(tr, controls=ctl) if structure is None
          else pv.Vinecop.from_data(tr, structure=structure, controls=ctl))
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    lp = np.log(np.maximum(np.asarray(vc.pdf(te), dtype=np.float64), common.LOG_FLOOR))
    return vc, fit_s, time.perf_counter() - t0, lp

  for name in ("tll", "parametric"):
    ctl = compiled_controls(name)
    for i, key in enumerate(keys):
      tr, te = data[f"{key}_train"], data[f"{key}_test"][:n_test]
      if i == 0:  # warm-up, untimed
        vc, *_ = run(tr, te, ctl)
        run(tr, te, ctl, vc.structure)
      load = common.load_average()
      vc, fit_s, inf_s, lp = run(tr, te, ctl)
      write({**vine_row(key, tr, te, lp), "method": name,
             "setting": "with selection (Dissmann), pyvinecopulib 1.0",
             "fit_s": fit_s, "inference_s": inf_s, "load_avg_1m": load,
             "structure_matrix": json.dumps(np.asarray(vc.matrix).tolist())})
      load = common.load_average()
      given, fit_s, inf_s, lp = run(tr, te, ctl, vc.structure)
      write({**vine_row(key, tr, te, lp), "method": name,
             "setting": "without selection (the selected structure given), pyvinecopulib 1.0",
             "fit_s": fit_s, "inference_s": inf_s, "load_avg_1m": load,
             "structure_matrix": json.dumps(np.asarray(given.matrix).tolist())})
      print(f"{name} {key}: done", flush=True)


def time_npcc(data: dict, keys: list[str], label: str, device: str, n_test: int | None,
              write) -> None:  # noqa: ANN001
  import pyvinecopulib as pv
  import torch
  from pyvinecopulib.core import NonSimplifiedContext

  import e3_npcc
  from npcc import FitControlsRosenblattVinecop, RosenblattVinecop
  from npcc.experiments._study import derive_seed

  est = e3_npcc.spec(label)
  controls = est.controls(FitControlsRosenblattVinecop, device=device,
                          projection_grid_size=e3_npcc.PROJECTION_GRID_SIZE, batch_size=None)
  context = NonSimplifiedContext()

  def sync() -> None:
    if device == "cuda":
      torch.cuda.synchronize()

  for i, key in enumerate(keys):
    design, d, seed = key.split("_")
    tr = torch.as_tensor(data[f"{key}_train"], dtype=torch.float64)
    te = torch.as_tensor(data[f"{key}_test"][:n_test], dtype=torch.float64)
    if i == 0:  # warm-up, untimed: a three-variable vine, fit and evaluated
      warm = RosenblattVinecop(None, pv.RVineStructure.from_order([1, 2, 3]), device=device,
                               context=context)
      warm.fit(tr[:, :3], controls)
      with torch.inference_mode():
        warm.logpdf(te[:N_WARMUP_TEST, :3]).detach().cpu()
      sync()
    structure = pv.RVineStructure.from_matrix(data[f"{key}_random_matrix"])
    load = common.load_average()
    torch.manual_seed(derive_seed("E3d", design, int(d[1:]), int(seed[1:]), "random-ns",
                                  est.estimator_id))
    t0 = time.perf_counter()
    vine = RosenblattVinecop(None, structure, device=device, context=context)
    vine.fit(tr, controls)
    sync()
    fit_s = time.perf_counter() - t0
    with torch.inference_mode():
      t0 = time.perf_counter()
      lp = vine.logpdf(te).detach().cpu().numpy().astype(np.float64)
      inf_s = time.perf_counter() - t0
    write({**vine_row(key, tr.numpy(), te.numpy(), lp), "method": f"npcc-{label}-random-ns",
           "setting": "random structure, non-simplified (arm (a); no selection)",
           "fit_s": fit_s, "inference_s": inf_s, "load_avg_1m": load,
           "estimator": label, "estimator_id": est.estimator_id,
           "model_version": e3_npcc.ESTIMATORS[label][1]})
    print(f"npcc-{label} {key}: fit {fit_s:.1f} s, inference {inf_s:.1f} s", flush=True)


def vines(a: argparse.Namespace) -> None:
  import torch

  # Imported before ``provenance`` so that their hashes are recorded: the
  # competitors' controls are E3d's, and so are the estimators.
  import e3_npcc
  import e3d_npcc_fit  # noqa: F401

  if a.part == "npcc":
    if a.estimator not in e3_npcc.ESTIMATORS:
      raise SystemExit(f"estimator must be one of {list(e3_npcc.ESTIMATORS)}; got {a.estimator}")
    kind = e3_npcc.ESTIMATORS[a.estimator][2]
    if (kind == "gpu") != (a.device == "cuda"):
      raise SystemExit(f"{a.estimator} must run with --device {'cuda' if kind == 'gpu' else 'cpu'}")
  elif a.device != "cpu":
    raise SystemExit(f"the competitors run on the host; pass --device cpu, got {a.device}")
  if a.device == "cuda" and not a.gpu_is_free:
    raise SystemExit("--device cuda requires --gpu-is-free: E6 runs alone on the machine")
  info, data, keys = e3d_cells(a.design, a.dims, a.seeds)
  prov = common.provenance(__file__, a.device)
  name = "competitors" if a.part == "competitors" else f"npcc_{a.estimator}"
  out = a.out / f"{name}_d{'-'.join(str(d) for d in a.dims)}.csv"
  rows: list[dict] = []

  def write(row: dict) -> None:
    rows.append({"section": "e3d", **row, "env": "npcc",
                 "torch_threads": torch.get_num_threads(),
                 "cores": len(os.sched_getaffinity(0)), "warmup_excluded": True,
                 "samples_sha256": info["samples_sha256"], **prov})
    common.write_csv(rows, out)

  if a.part == "competitors":
    time_competitors(data, keys, a.n_test, write)
  else:
    time_npcc(data, keys, a.estimator, a.device, a.n_test, write)
  print(f"wrote {len(rows)} rows to {out}", flush=True)


def e3a_rows(folder: Path, seeds: set[int]) -> list[dict]:
  """E3a's timing rows, from its VDC side and its NPCC cells, long format."""
  rows: list[dict] = []
  side = pd.read_csv(folder / "vdc_side.csv")
  side = side[side["seed"].isin(seeds) & side["method"].isin(["vdc", "tll", "parametric"])]
  settings = {"vdc": "vdc-denoiser-m64-v1, m = 64, 30 projection iterations, its own lookup",
              "tll": "pyvinecopulib 0.7.5 TLL, defaults, no family selection",
              "parametric": "pyvinecopulib 0.7.5, parametric family selection by BIC"}
  for r in side.to_dict("records"):
    stages = {"fit": r["fit_s"], "inference": r["inference_s"]}
    if r["method"] == "vdc":
      stages["projection"] = r["projection_s"]
    for stage, seconds in stages.items():
      rows.append({**r, "setting": settings[r["method"]], "stage": stage, "seconds": seconds})
  for path in sorted(folder.glob("npcc_cells/*/n2000/k*.json")):
    r = json.loads(path.read_text())
    if r["seed"] not in seeds:
      continue
    for stage in ("fit", "inference"):
      rows.append({**r, "setting": f"{r['estimator']} (E1 logit arm)", "stage": stage,
                   "seconds": r[f"{stage}_s"]})
  for r in rows:
    r.update({"section": "e3a", "d": 2, "warmup_excluded": True})
  return rows


def collect(a: argparse.Namespace) -> None:
  import e3_collect

  e3a_dir, e3d_dir = a.root / "e3" / "e3a", a.root / "e3" / "e3d"
  paper_e3a = json.loads((common.RESULTS / "e3a" / "vdc_side.json").read_text())
  timed_e3a = json.loads((e3a_dir / "vdc_side.json").read_text())
  if timed_e3a["samples_sha256_by_n"]["2000"] != paper_e3a["samples_sha256_by_n"]["2000"]:
    raise SystemExit("E6's E3a samples must be E3a's (sha256 of samples.npz)")
  raw_a = e3a_rows(e3a_dir, set(a.e3a_seeds))
  raw_d = [r for path in sorted(e3d_dir.glob("*.csv")) for r in pd.read_csv(path).to_dict("records")]
  if not raw_a or not raw_d:
    raise SystemExit(f"timings must exist under {a.root / 'e3'}; got {len(raw_a)} E3a and "
                     f"{len(raw_d)} vine rows")

  # The same computation: E3a's KL, and E3d's held-out log-likelihood.
  e3a = pd.read_csv(common.RESULTS / "e3a_long.csv")
  e3a = e3a[e3a["n_train"] == 2000].set_index(["method", "case", "seed"])["kl"]
  for r in raw_a:
    key = (r["method"], r["case"], r["seed"])
    r["loglik_gap_vs_e3"] = abs(float(r["kl"]) - float(e3a[key])) if key in e3a.index else None
  e3d = pd.read_csv(E3D / "e3d_long.csv").set_index(["method", "design", "d", "seed"])["test_loglik_row"]
  for r in raw_d:
    key = (r["method"], r["design"], int(r["d"]), int(r["seed"]))
    r["loglik_gap_vs_e3"] = abs(float(r["test_loglik"]) - float(e3d[key])) if key in e3d.index else None
    if int(r["n_test"]) != 20000:
      raise SystemExit(f"vine rows must use E3d's 20,000 test points; got {r['n_test']} in {r['method']}")

  rows = list(raw_a)
  for r in raw_d:
    for stage in ("fit", "inference"):
      rows.append({**r, "stage": stage, "seconds": r[f"{stage}_s"]})
  for r in rows:
    # ``cpu_affinity`` is the comma-joined list of the cores the process was pinned to.
    r.update({"n": r["n_train"], "cpu": r["cpu_model"],
              "compute": "gpu" if str(r["device"]).startswith("cuda") else "cpu",
              "threads": len(str(r["cpu_affinity"]).split(","))})
  table = pd.DataFrame(rows).reindex(columns=COLUMNS)
  table.to_csv(a.root / "e3_timing.csv", index=False)

  frames = {"E6-e3a": pd.DataFrame(raw_a), "E6-e3d": pd.DataFrame(raw_d)}
  check = e3_collect.code_check(frames)
  check.to_csv(a.root / "e3" / "e3_timing_code_check.csv", index=False)
  stale = check[~check["match"]]
  with pd.option_context("display.width", 200, "display.max_columns", 20):
    summary = (table.groupby(["section", "method", "setting", "d", "stage"], sort=False)["seconds"]
               .agg(["count", "median", "min", "max"]))
    print(summary.to_string())
  gaps = pd.to_numeric(table["loglik_gap_vs_e3"], errors="coerce")
  print(f"rows {len(table)}; largest gap to E3's own log-likelihood or KL: {gaps.max():.3g}; "
        f"rows without an E3 counterpart: {int(gaps.isna().sum())}")
  print("code check:", "every row was produced by committed code or the code on disk" if stale.empty
        else f"{len(stale)} stale entries (see e3_timing_code_check.csv)")
  if not stale.empty and not a.allow_stale:
    sys.exit("e6_timing: rows must come from committed code or the code on disk")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  sub = p.add_subparsers(dest="command", required=True)
  v = sub.add_parser("vines", help="time E3d's vines")
  v.add_argument("--part", choices=["competitors", "npcc"], required=True)
  v.add_argument("--estimator", default=MAIN_TABPFN)
  v.add_argument("--design", default="decay", choices=["decay", "strong"])
  v.add_argument("--dims", type=int, nargs="+", default=[5, 10])
  v.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
  v.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
  v.add_argument("--gpu-is-free", action="store_true")
  v.add_argument("--n-test", type=int, default=None,
                 help="evaluate at the first N test points only: smoke tests, which collect refuses")
  v.add_argument("--out", type=Path, default=ROOT / "e3" / "e3d")
  c = sub.add_parser("collect", help="write e3_timing.csv")
  c.add_argument("--root", type=Path, default=ROOT, help="the E6 folder")
  c.add_argument("--e3a-seeds", type=int, nargs="+", default=[202])
  c.add_argument("--allow-stale", action="store_true")
  a = p.parse_args()
  if a.command == "vines":
    vines(a)
  else:
    collect(a)


if __name__ == "__main__":
  main()
