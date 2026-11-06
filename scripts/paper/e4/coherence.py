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

``--projection-grid`` replaces all three with the projection alone, scored on
each given grid size from one fit per cell, into
``projection_grid_<I>of<K>.csv``; its ``proj_*`` columns are the projected
density's margin errors on the projection grid itself, which is how far the
Sinkhorn iteration got. ``--part I/K`` takes every K-th cell, so the
run can be split across processes; every part writes
``coherence_<I>of<K>.csv``.

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
from typing import cast

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


def projection_convergence(
  model: RosenblattBicop, x_axis: torch.Tensor, iters: list[int]
) -> dict[tuple[float, object], dict[str, float]]:
  """Each projection's margins on its own grid: how far Sinkhorn got.

  At the grid's midpoints the interpolated scalings are the grid's own, so
  the projected density there is the scaled grid whose margins the iteration
  drives to one. Keyed by x and iteration count, as :func:`projection_rows`.
  """
  axis = model._projection_axis().cpu()
  m = axis.shape[0]
  uu, vv = torch.meshgrid(axis, axis, indexing="ij")
  uv = torch.column_stack([uu.ravel(), vv.ravel()])
  xs = x_axis.cpu()
  out = model.pdf_by_projection(
    uv.repeat(xs.shape[0], 1),
    x=xs.repeat_interleave(m * m).reshape(-1, 1),
    sinkhorn_iters=[None, *iters],
  )
  result: dict[tuple[float, object], dict[str, float]] = {}
  for key, dens in out.items():
    d = dens.cpu().reshape(xs.shape[0], m, m)
    for j, xv in enumerate(xs.tolist()):
      margins = metrics.marginal_diagnostics(d[j], axis, axis)
      label = "none" if key is None else int(key)
      result[(xv, label)] = {f"proj_{k}": v for k, v in margins.items()}
  return result


def projection_grid_rows(
  model: RosenblattBicop,
  grid: scenarios.EvalGrid,
  truth: torch.Tensor,
  iters: list[int],
  sizes: list[int],
) -> list[dict[str, object]]:
  """:func:`projection_rows` on each projection grid size, from one fit.

  The grid size is read when the projection is laid out, not when the margins
  are fitted, so every size scores the same fitted model and the comparison
  is paired. Each row also carries :func:`projection_convergence`.
  """
  if grid.x_axis is None:
    raise ValueError("projection_grid_rows needs a conditional grid")
  rows: list[dict[str, object]] = []
  for size in sizes:
    model.projection_grid_size = size
    on_grid = projection_convergence(model, grid.x_axis, iters)
    rows += [
      {
        "projection_grid_size": size,
        **r,
        **on_grid[(cast("float", r["x"]), r["sinkhorn_iters"])],
      }
      for r in projection_rows(model, grid, truth, iters)
    ]
  return rows


def _timed(
  model: RosenblattBicop,
  uv: torch.Tensor,
  x: torch.Tensor,
  iters: list[int | None],
) -> float:
  """Wall time of one evaluation, moved to the host so a GPU finishes."""
  t0 = perf_counter()
  out = model.pdf_by_projection(uv, x=x, sinkhorn_iters=iters)
  for value in out.values():
    value.cpu()
  return perf_counter() - t0


def projection_cost_rows(
  model: RosenblattBicop, iters: list[int], ks: list[int], *, repeats: int
) -> list[dict[str, object]]:
  """Projection overhead against the number of distinct conditioning values.

  Every query has its own covariate, as every observation does in a vine, so
  each needs its own projection grid: ``k`` queries cost ``k`` grid
  evaluations of both directions plus ``k`` Sinkhorn solves. The median over
  ``repeats`` timings follows a warm-up call.
  """
  generator = torch.Generator().manual_seed(0)
  rows = []
  for k in ks:
    uv = 0.1 + 0.8 * torch.rand(
      (k, 2), generator=generator, dtype=torch.float64
    )
    x = torch.linspace(0.01, 0.99, k, dtype=torch.float64).reshape(-1, 1)
    for count in iters:
      settings: tuple[list[int | None], list[int | None]] = (
        [None],
        [None, count],
      )
      for iters_ in settings:
        _timed(model, uv, x, iters_)
      t_base, t_proj = (
        sorted(_timed(model, uv, x, iters_) for _ in range(repeats))[
          repeats // 2
        ]
        for iters_ in settings
      )
      rows.append(
        {
          "k_distinct_x": k,
          "sinkhorn_iters": count,
          "time_unprojected_s": t_base,
          "time_projected_s": t_proj,
          "overhead_s": t_proj - t_base,
          "overhead_per_x_s": (t_proj - t_base) / k,
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
  p.add_argument(
    "--cost-only",
    action="store_true",
    help=(
      "Only time the projection against the number of distinct x, on the "
      "first cell; run it on an otherwise idle device."
    ),
  )
  p.add_argument(
    "--projection-grid",
    type=int,
    nargs="+",
    default=None,
    metavar="SIZE",
    help=(
      "Only score the projection, on each of these grid sizes per axis, "
      "from one fit per cell."
    ),
  )
  p.add_argument("--cost-k", type=int, nargs="+", default=[1, 10, 100, 1000])
  p.add_argument("--cost-repeats", type=int, default=5)
  a = p.parse_args()
  if a.cost_only and a.projection_grid:
    p.error("--cost-only and --projection-grid are separate runs")

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
  if a.cost_only:
    cells = cells[:1]
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
      if a.cost_only:
        cost = projection_cost_rows(
          model, a.sinkhorn, a.cost_k, repeats=a.cost_repeats
        )
        rows += [{**base, "diagnostic": "projection_cost", **r} for r in cost]
        continue
      if a.projection_grid:
        sized = projection_grid_rows(
          model, eval_grid, truth["pdf"], a.sinkhorn, a.projection_grid
        )
        rows += [{**base, "diagnostic": "projection_grid", **r} for r in sized]
        continue
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
  stem = (
    "projection_cost"
    if a.cost_only
    else "projection_grid"
    if a.projection_grid
    else "coherence"
  )
  pd.DataFrame(rows).to_csv(a.out / f"{stem}_{index}of{count}.csv", index=False)


if __name__ == "__main__":
  main()
