"""E4: numerical coherence diagnostics on E1's grid, refitted with E1's seeds.

For every (family, regime, n, rep) cell and estimator it refits the pair copula
exactly as the E1 runner does -- the same cell seed, ``scenarios.sample`` draw
and per-estimator torch seed, so the model equals E1's up to GPU
nondeterminism -- and writes one long-format CSV row per covariate value and
diagnostic into ``--out``:

- ``directional``: IAE between the two Rosenblatt directions' densities, and
  each direction's IAE and KL against the truth next to the average's;
- ``hfunc``: IAE (and max) between each direct h-function and the integral of
  the averaged density it should equal;
- ``projection``: pdf IAE and KL, the mass on the evaluation grid and the
  margin errors, unprojected (``sinkhorn_iters`` = none) and after each
  requested iteration count, with the wall time of the projected evaluation
  over the unprojected one.

``--part I/K`` takes every K-th cell, so the run can be split across
processes; every part writes ``coherence_<I>of<K>.csv``.

Usage (from the npcc repository root)::

    uv run --no-sync python scripts/paper/e4/coherence.py \
      --labels tabpfn-quantiles-v3-logit lightgbm-logit \
      --out results/e4_coherence
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from npcc.core import FitControlsRosenblattBicop, RosenblattBicop
from npcc.experiments import coherence, metrics, scenarios
from npcc.experiments._study import derive_seed
from npcc.experiments.config import Cell, load_grid
from npcc.experiments.runner import _cell_seed

PANELS = [(f, s) for f in ("clayton", "gaussian") for s in ("linear", "sin")]


def _part(text: str) -> tuple[int, int]:
  index, _, count = text.partition("/")
  return int(index), int(count)


def projection_rows(
  model: RosenblattBicop,
  grid: scenarios.EvalGrid,
  truth: torch.Tensor,
  iters: list[int],
) -> list[dict[str, object]]:
  """Accuracy, mass and margins with and without the projection, per x."""
  x = grid.x_flat
  t0 = perf_counter()
  model.pdf_by_projection(grid.uv, x=x, sinkhorn_iters=[None])
  base_time = perf_counter() - t0
  t0 = perf_counter()
  out = model.pdf_by_projection(grid.uv, x=x, sinkhorn_iters=[None, *iters])
  all_time = perf_counter() - t0

  shape = grid.shape
  true = truth.cpu().reshape(shape)
  nu, nv = grid.u_axis.shape[0], grid.v_axis.shape[0]
  rows = []
  for key, dens in out.items():
    d = dens.cpu().reshape(shape)
    for j, xv in enumerate(grid.x_axis.tolist()):
      m = metrics.grid_metrics(true[:, j], d[:, j], include_kl=True)
      margins = metrics.marginal_diagnostics(
        d[:, j].reshape(nu, nv), grid.u_axis, grid.v_axis
      )
      rows.append(
        {
          "x": xv,
          "sinkhorn_iters": "none" if key is None else int(key),
          "pdf_IAE": m["IAE"],
          "pdf_KL": m["KL"],
          "mass": float(d[:, j].mean()),
          **margins,
          # The extra cost of every projected count over the unprojected
          # evaluation, paid once for the whole sweep.
          "projection_overhead_s": all_time - base_time,
        }
      )
  return rows


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--config", type=Path, default=Path("configs/study.toml"))
  p.add_argument("--labels", nargs="+", required=True)
  p.add_argument("--n", type=int, nargs="+", default=[500, 2000])
  p.add_argument("--reps", type=int, default=5)
  p.add_argument("--sinkhorn", type=int, nargs="+", default=[20])
  p.add_argument("--n-int", type=int, default=400)
  p.add_argument("--base-seed", type=int, default=317)
  p.add_argument("--device", default=None)
  p.add_argument("--part", type=_part, default=(0, 1), metavar="I/K")
  p.add_argument("--out", type=Path, required=True)
  a = p.parse_args()

  grid_cfg = load_grid(a.config)
  by_label = {e.label: e for e in grid_cfg.estimators}
  missing = sorted(set(a.labels) - set(by_label))
  if missing:
    raise SystemExit(f"labels not in {a.config}: {missing}")
  cells = [
    Cell(family=f, tau_scenario=s, n=n, rep=r)
    for f, s in PANELS
    for n in a.n
    for r in range(a.reps)
  ]
  index, count = a.part
  cells = [c for i, c in enumerate(cells) if i % count == index]
  commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True,
    text=True,
    check=True,
  ).stdout.strip()

  rows: list[dict[str, object]] = []
  for cell in cells:
    seed = _cell_seed(a.base_seed, cell)
    u, v, x = scenarios.sample(cell.family, cell.tau_scenario, cell.n, seed)
    eval_grid = scenarios.eval_grid(
      cell.tau_scenario,
      conditional_uv_grid_n=grid_cfg.conditional_uv_grid_n,
      conditional_x_grid_n=grid_cfg.conditional_x_grid_n,
    )
    truth = scenarios.ground_truth(cell.family, cell.tau_scenario, eval_grid)
    for label in a.labels:
      est = by_label[label]
      torch.manual_seed(derive_seed(seed, est.estimator_id))
      model = RosenblattBicop(
        est.controls(
          FitControlsRosenblattBicop,
          device=a.device,
          projection_grid_size=grid_cfg.projection_grid_size,
          batch_size=grid_cfg.batch_size,
        )
      )
      t0 = perf_counter()
      model.fit(torch.column_stack([u, v]), x=x)
      fit_time = perf_counter() - t0
      base = {
        "family": cell.family,
        "tau_scenario": cell.tau_scenario,
        "n": cell.n,
        "rep": cell.rep,
        "cell_seed": seed,
        "label": label,
        "estimator_id": est.estimator_id,
        "fit_time_s": fit_time,
        "device": str(model._device),
        "npcc_commit": commit,
      }
      for diag, part_rows in (
        (
          "directional",
          coherence.directional_rows(model, eval_grid, truth["pdf"]),
        ),
        ("hfunc", coherence.hfunc_rows(model, eval_grid, n_int=a.n_int)),
        (
          "projection",
          projection_rows(model, eval_grid, truth["pdf"], a.sinkhorn),
        ),
      ):
        rows += [{**base, "diagnostic": diag, **r} for r in part_rows]
    print(
      f"done {cell.family}/{cell.tau_scenario} n={cell.n} rep={cell.rep}",
      flush=True,
    )

  a.out.mkdir(parents=True, exist_ok=True)
  pd.DataFrame(rows).to_csv(
    a.out / f"coherence_{index}of{count}.csv", index=False
  )


if __name__ == "__main__":
  main()
