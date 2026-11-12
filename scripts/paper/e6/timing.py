"""E6: clean fit and inference timings, for a machine with nothing else on it.

**Bivariate.** For every estimator label, n and repetition it fits the pair
copula on an E1 cell as the E1 runner does -- the same cell seed, data and
per-estimator torch seed -- and times each Rosenblatt direction separately:
the fit of its conditional margin, the evaluation of its density, and its
h-function, each on E1's evaluation grid. The pair's averaged density is timed
too, as E1's ``pdf_time`` is. One untimed fit and evaluation of the same
estimator on the smallest cell comes first, so loading a model or compiling a
kernel is charged to no row.

**Vines.** For every d, n and repetition, the vine runner's own per-arm fit
and density-evaluation times (``fit_time``, ``pdf_time``), after one untimed
cell.

Every row records the device, where the estimator computes (``compute``), the
GPU and CPU models, the torch thread count, the OpenMP and BLAS thread
settings, the CPUs the process may use, and the npcc commit. Times are seconds
of wall clock with the device synchronized on both sides.

Usage (from the npcc repository root, with nothing else running)::

    uv run --no-sync python scripts/paper/e6/timing.py bivariate \
      --out results/e6_timing
    uv run --no-sync python scripts/paper/e6/timing.py vine \
      --out results/e6_timing
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import platform
import subprocess
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from npcc.core import FitControlsRosenblattBicop, RosenblattBicop
from npcc.experiments import scenarios
from npcc.experiments._study import derive_seed, release_gpu
from npcc.experiments.config import Cell, load_grid
from npcc.experiments.runner import _cell_seed
from npcc.experiments.vine_config import VineCell, load_vine_grid
from npcc.experiments.vine_runner import summarize_one_cell

#: The main text's estimators, one per backend, as E1 labels them.
MAIN_LABELS = [
  "tabpfn-quantiles-v3-probit",
  "tabicl-probit",
  "nori-probit",
  "lightgbm-probit",
  "xgboost-probit",
  "catboost-probit",
]

#: Backends whose model runs on the torch device; the gradient boosters
#: compute on the CPU whatever device their tensors are placed on.
DEVICE_BACKENDS = {"tabpfn-quantiles", "tabicl", "nori"}

#: Each gradient booster's own thread-count argument.
THREAD_KWARG = {
  "lightgbm": "n_jobs",
  "xgboost": "n_jobs",
  "catboost": "thread_count",
}


def _sync() -> None:
  if torch.cuda.is_available():
    torch.cuda.synchronize()


def _timed(call: Callable[[], object]) -> float:
  """Seconds one call takes, with the device synchronized on both sides."""
  _sync()
  t0 = perf_counter()
  call()
  _sync()
  return perf_counter() - t0


def hardware() -> dict[str, object]:
  """What the timings ran on: devices, thread settings and the commit."""
  cpu = platform.processor() or platform.machine()
  info = Path("/proc/cpuinfo")
  if info.exists():
    for line in info.read_text().splitlines():
      if line.startswith("model name"):
        cpu = line.split(":", 1)[1].strip()
        break
  commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True,
    text=True,
    check=False,
  ).stdout.strip()
  return {
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    "cpu": cpu,
    "cpus_usable": len(os.sched_getaffinity(0)),
    "torch_threads": torch.get_num_threads(),
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
    "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
    "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
    "npcc_commit": commit,
  }


def time_pair(
  model: RosenblattBicop,
  uv: torch.Tensor,
  x: torch.Tensor,
  grid: scenarios.EvalGrid,
) -> dict[str, float]:
  """Fit ``model`` on ``(uv, x)``; time each direction's fit and reads.

  The pair is fitted through :meth:`RosenblattBicop.fit`, exactly as
  anywhere else; only its two margins' ``fit`` are wrapped, so that each
  direction is charged its own call.
  """
  fits: dict[str, float] = {}
  for name in ("v_given_ux_", "u_given_vx_"):
    margin = getattr(model, name)

    def timed_fit(
      *args: object,
      _fit: Callable[..., object] = margin.fit,
      _name: str = name,
      **kwargs: object,
    ) -> object:
      _sync()
      t0 = perf_counter()
      out = _fit(*args, **kwargs)
      _sync()
      fits[_name] = perf_counter() - t0
      return out

    margin.fit = timed_fit
  model.fit(uv, x=x)

  if grid.x_flat is None:
    raise ValueError("time_pair needs a conditional grid; got x_flat=None")
  device = model.v_given_ux_._device
  u, v, xg = (t.to(device) for t in (grid.u_flat, grid.v_flat, grid.x_flat))
  with torch.inference_mode():
    return {
      "fit_v_given_u_s": fits["v_given_ux_"],
      "fit_u_given_v_s": fits["u_given_vx_"],
      "pdf_v_given_u_s": _timed(
        lambda: model.v_given_ux_.pdf(v, x=torch.column_stack([u, xg]))
      ),
      "pdf_u_given_v_s": _timed(
        lambda: model.u_given_vx_.pdf(u, x=torch.column_stack([v, xg]))
      ),
      "hfunc1_s": _timed(lambda: model.hfunc1(grid.uv, x=grid.x_flat)),
      "hfunc2_s": _timed(lambda: model.hfunc2(grid.uv, x=grid.x_flat)),
      "pdf_pair_s": _timed(lambda: model.pdf(grid.uv, x=grid.x_flat)),
    }


def bivariate(a: argparse.Namespace) -> pd.DataFrame:
  """One row per estimator, n and repetition; the warm-up is dropped."""
  grid_cfg = load_grid(a.config)
  by_label = {e.label: e for e in grid_cfg.estimators}
  missing = sorted(set(a.labels) - set(by_label))
  if missing:
    raise SystemExit(f"labels not in {a.config}: {missing}")
  eval_grid = scenarios.eval_grid(
    a.tau_scenario,
    conditional_uv_grid_n=grid_cfg.conditional_uv_grid_n,
    conditional_x_grid_n=grid_cfg.conditional_x_grid_n,
  )
  hw = hardware()
  plan = [(min(a.n), 0, True)]
  plan += [(n, rep, False) for n in a.n for rep in range(a.reps)]
  rows = []
  for label in a.labels:
    est = by_label[label]
    # A thread count is the boosters' own argument; the seed stays E1's.
    spec, threads = est, ""
    if est.backend in THREAD_KWARG:
      threads = "all"
      if a.gb_threads is not None:
        kwargs = {**est.backend_kwargs, THREAD_KWARG[est.backend]: a.gb_threads}
        spec, threads = (
          dataclasses.replace(est, backend_kwargs=kwargs),
          a.gb_threads,
        )
    for n, rep, warm_up in plan:
      cell = Cell(family=a.family, tau_scenario=a.tau_scenario, n=n, rep=rep)
      seed = _cell_seed(a.base_seed, cell)
      u, v, x = scenarios.sample(cell.family, cell.tau_scenario, n, seed)
      if x is None:
        raise SystemExit(
          f"E6 times the conditional pair; --tau-scenario {a.tau_scenario} "
          "has no covariate"
        )
      torch.manual_seed(derive_seed(seed, est.estimator_id))
      model = RosenblattBicop(
        spec.controls(
          FitControlsRosenblattBicop,
          device=a.device,
          projection_grid_size=grid_cfg.projection_grid_size,
          batch_size=grid_cfg.batch_size,
        )
      )
      times = time_pair(model, torch.column_stack([u, v]), x, eval_grid)
      device = str(model.v_given_ux_._device)
      del model
      release_gpu()
      if warm_up:
        continue
      rows.append(
        {
          "label": label,
          "backend": est.backend,
          "transform": est.transform,
          "family": cell.family,
          "tau_scenario": cell.tau_scenario,
          "n": n,
          "rep": rep,
          "cell_seed": seed,
          "device": device,
          "compute": "gpu"
          if est.backend in DEVICE_BACKENDS and device.startswith("cuda")
          else "cpu",
          "backend_threads": threads,
          "eval_points": int(eval_grid.uv.shape[0]),
          **times,
          **hw,
        }
      )
      print(f"{label} n={n} rep={rep}: {times}", flush=True)
  return pd.DataFrame(rows)


def vine(a: argparse.Namespace) -> pd.DataFrame:
  """The vine runner's per-arm times; the warm-up cell is dropped."""
  grid = load_vine_grid(a.config)
  hw = hardware()
  cells = [VineCell(d=min(a.d), n=min(a.n), rep=0)]
  cells += [
    VineCell(d=d, n=n, rep=rep)
    for d in a.d
    for n in a.n
    for rep in range(a.reps)
  ]
  rows = []
  for i, cell in enumerate(cells):
    _, runtime = summarize_one_cell(
      cell, grid, base_seed=a.base_seed, device=a.device
    )
    release_gpu()
    if i == 0:
      continue
    rows += [{**r, **hw} for r in runtime]
    print(f"done d={cell.d} n={cell.n} rep={cell.rep}", flush=True)
  return pd.DataFrame(rows)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  sub = p.add_subparsers(dest="what", required=True)
  b = sub.add_parser("bivariate", help="Per-direction pair-copula timings.")
  b.add_argument("--config", type=Path, default=Path("configs/study.toml"))
  b.add_argument("--labels", nargs="+", default=MAIN_LABELS)
  b.add_argument("--n", type=int, nargs="+", default=[200, 500, 1000, 2000])
  b.add_argument("--reps", type=int, default=3)
  b.add_argument("--family", default="clayton")
  b.add_argument("--tau-scenario", default="linear")
  b.add_argument(
    "--gb-threads",
    type=int,
    default=None,
    help="Thread count for the gradient boosters (default: all cores).",
  )
  v = sub.add_parser("vine", help="Per-arm vine timings.")
  v.add_argument("--config", type=Path, default=Path("configs/vine_study.toml"))
  v.add_argument("--d", type=int, nargs="+", default=[3, 5])
  v.add_argument("--n", type=int, nargs="+", default=[500, 1000])
  v.add_argument("--reps", type=int, default=3)
  for q in (b, v):
    q.add_argument("--base-seed", type=int, default=317)
    q.add_argument("--device", default=None)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--tag", default="", help="Suffix of the output file.")
  a = p.parse_args()

  table = bivariate(a) if a.what == "bivariate" else vine(a)
  a.out.mkdir(parents=True, exist_ok=True)
  name = "bivariate" if a.what == "bivariate" else "vines"
  name += f"_{a.tag}" if a.tag else ""
  table.to_csv(a.out / f"{name}.csv", index=False)
  print(f"{len(table)} rows -> {a.out / name}.csv")


if __name__ == "__main__":
  main()
