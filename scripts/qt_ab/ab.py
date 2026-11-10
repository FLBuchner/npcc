"""Paired A/B of quantile-table read-outs on E1's cells.

Per data cell (E1's seeds), each estimator is fitted, and one evaluation pass
records every raw quantile row its two inner margins are asked for. Every
read-out arm is then scored by replaying those rows through E1's own scoring
code (:func:`_common.evaluate`), so arms differ only in the read-out and share
every fitted model.

- A backend that fits each level on its own (LightGBM), or reads every level
  off one predicted distribution (TabPFN, TabICLv2, Nori), is recorded once on
  81 levels, of which every grid is a subset.
- XGBoost and CatBoost fit all levels with one loss, so the level set changes
  the fit: each grid is its own fit.

Output: one parquet per cell under ``--out``. Rows carry ``kind`` (``metric``
/ ``diag`` / ``holdout`` / ``time`` / ``fit``), the read-out ``arm``, the
``grid``, and the ``commit`` they were produced from (``dirty`` if tracked
files differed from it).

Examples::

    # the tree backends on CPU: 4 single-threaded processes
    OMP_NUM_THREADS=1 uv run --no-sync python scripts/qt_ab/ab.py \
      --out results/qt_ab/gb --workers 4

    # TabPFN v3 on the GPU, n in {200, 2000}
    uv run --no-sync python scripts/qt_ab/ab.py --out results/qt_ab/tabpfn \
      --estimators tabpfn-quantiles-v3-logit --device cuda --n 200 2000
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
  BASE_SEED,
  GRIDS,
  REC_LEVELS,
  REC_LIMIT,
  UV_GRID_N,
  X_GRID_N,
  Store,
  cap_gpu_memory,
  cells,
  commit,
  evaluate,
  holdout,
)
from readouts import READOUTS, table_factory

import npcc.core.margin as margin_module
from npcc.core.bicop import RosenblattBicop
from npcc.core.controls import FitControlsRosenblattBicop
from npcc.core.margin import QuantileTableConfig
from npcc.experiments import runner, scenarios
from npcc.experiments._study import derive_seed
from npcc.experiments.config import Cell, EstimatorSpec

ESTIMATORS: dict[str, tuple[str, str, dict[str, object]]] = {
  **{
    f"{b}-{t}": (b, t, {})
    for b in ("lightgbm", "xgboost", "catboost", "tabicl", "nori")
    for t in ("logit", "probit")
  },
  **{
    f"tabpfn-quantiles-v3-{t}": (
      "tabpfn-quantiles",
      t,
      {"model_version": "v3"},
    )
    for t in ("logit", "probit")
  },
  # TabPFN v3 at softmax temperature 1.0 rather than its checkpoint's 0.9,
  # with E1's temperature ablation's settings and so its estimator id.
  "tabpfn-quantiles-v3-temp1.0-probit": (
    "tabpfn-quantiles",
    "probit",
    {"model_version": "v3", "model_kwargs": {"softmax_temperature": 1.0}},
  ),
}
TREES = [
  label
  for label, (backend, _, _) in ESTIMATORS.items()
  if backend in ("lightgbm", "xgboost", "catboost")
]
#: Backends whose fitted model depends on the level set.
REFIT_PER_GRID = frozenset({"xgboost", "catboost"})
#: On CPU every process is single-threaded, so ``--workers`` is the budget.
ONE_THREAD = {
  "lightgbm": {"n_jobs": 1},
  "xgboost": {"n_jobs": 1},
  "catboost": {"thread_count": 1},
}


def run_cell(
  cell: Cell,
  out: Path,
  labels: list[str],
  device: str,
  grids: list[str],
  arms: list[str] | None,
) -> None:
  """Fit, record and replay every estimator on one cell; write its parquet."""
  name = f"{cell.family}__{cell.tau_scenario}__n{cell.n}__rep{cell.rep}"
  path = out / f"{name}.parquet"
  if path.exists():
    return
  seed = runner._cell_seed(BASE_SEED, cell)
  u, v, x = scenarios.sample(cell.family, cell.tau_scenario, cell.n, seed)
  grid = scenarios.eval_grid(
    cell.tau_scenario,
    conditional_uv_grid_n=UV_GRID_N,
    conditional_x_grid_n=X_GRID_N,
  )
  truth = scenarios.ground_truth(cell.family, cell.tau_scenario, grid)
  ho = holdout(cell, seed)
  readouts = [rd for rd in READOUTS if arms is None or rd.label in arms]
  library_table = margin_module.QuantileTable
  rows: list[dict[str, object]] = []

  for label in labels:
    backend, transform, kwargs = ESTIMATORS[label]
    est = EstimatorSpec(
      label=label, backend=backend, transform=transform, backend_kwargs=kwargs
    )
    threads = ONE_THREAD.get(backend, {}) if device == "cpu" else {}
    if backend in REFIT_PER_GRID:
      recordings = {g: GRIDS[g] for g in grids}
    else:
      recordings = {f"m{REC_LEVELS}": (REC_LEVELS, REC_LIMIT)}

    for rec_name, (m_rec, l_rec) in recordings.items():
      controls = FitControlsRosenblattBicop(
        backend=backend,
        transform=transform,
        quantile_table_config=QuantileTableConfig(
          n_quantiles=m_rec, normal_limit=l_rec
        ),
        device=device,
        projection_grid_size=30,
        backend_kwargs={**kwargs, **threads},
      )
      torch.manual_seed(derive_seed(seed, est.estimator_id))
      t0 = perf_counter()
      model = RosenblattBicop(controls)
      model.fit(torch.column_stack([u, v]), x=x)
      fit_s = perf_counter() - t0

      margins = (model.v_given_ux_, model.u_given_vx_)
      stores = [
        Store(torch.linspace(-l_rec, l_rec, m_rec, dtype=torch.float64))
        for _ in margins
      ]
      for mg, store in zip(margins, stores, strict=True):
        mg._predict_quantiles = store.record(mg._predict_quantiles)
      t0 = perf_counter()
      evaluate(model, cell, est, seed, grid, truth, ho, {})
      record_s = perf_counter() - t0
      for mg, store in zip(margins, stores, strict=True):
        store.finalize()
        mg._predict_quantiles = store.replay
      assert stores[0].rows is not None
      rows.append(
        {
          **runner._base_row(cell, est, seed),
          "kind": "fit",
          "rec_grid": rec_name,
          "fit_s": fit_s,
          "record_s": record_s,
          "rows_recorded": int(stores[0].rows.shape[0]),
        }
      )

      served = [rec_name] if backend in REFIT_PER_GRID else grids
      try:
        for grid_name in served:
          m, lim = GRIDS[grid_name]
          for mg in margins:
            mg.quantile_table_config = QuantileTableConfig(
              n_quantiles=m, normal_limit=lim
            )
          for rd in readouts:
            margin_module.QuantileTable = table_factory(rd)
            tags = {
              "arm": rd.label,
              "smoothing": rd.smoothing,
              "width": rd.width,
              "tails": rd.tails,
              "grid": grid_name,
              "m": m,
              "L": lim,
            }
            rows += evaluate(model, cell, est, seed, grid, truth, ho, tags)
      finally:
        margin_module.QuantileTable = library_table
      del model, margins, stores
      if device != "cpu":
        torch.cuda.empty_cache()

  head, dirty = commit()
  tmp = path.with_suffix(".tmp")
  pd.DataFrame(rows).assign(commit=head, dirty=dirty).to_parquet(tmp)
  tmp.rename(path)


def _run(
  job: tuple[Cell, Path, list[str], str, list[str], list[str] | None],
) -> None:
  torch.set_num_threads(1)
  run_cell(*job)


def main() -> None:
  """Run the A/B over E1's cells."""
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument(
    "--estimators", nargs="+", default=TREES, choices=list(ESTIMATORS)
  )
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--n", type=int, nargs="+", default=[200, 500, 2000])
  parser.add_argument("--reps", type=int, nargs="+", default=list(range(10)))
  parser.add_argument(
    "--grids", nargs="+", default=list(GRIDS), choices=list(GRIDS)
  )
  parser.add_argument(
    "--arms",
    nargs="+",
    default=None,
    choices=[rd.label for rd in READOUTS],
    help="default all",
  )
  parser.add_argument(
    "--workers", type=int, default=1, help="CPU processes, at most 4"
  )
  parser.add_argument("--gpu-mem-gb", type=float, default=3.0)
  args = parser.parse_args()
  if args.workers > 4:
    raise SystemExit("at most 4 workers: the machine is shared")
  if args.device != "cpu" and args.workers != 1:
    raise SystemExit("a GPU run is one process")
  args.out.mkdir(parents=True, exist_ok=True)

  todo = sorted(cells(args.n, args.reps), key=lambda c: -c.n)
  jobs = [
    (c, args.out, args.estimators, args.device, args.grids, args.arms)
    for c in todo
  ]
  t0 = perf_counter()
  if args.device != "cpu":
    cap_gpu_memory(args.gpu_mem_gb)
    torch.set_num_threads(4)
    for i, job in enumerate(jobs):
      run_cell(*job)
      peak = torch.cuda.max_memory_reserved() / 2**30
      print(
        f"[{i + 1}/{len(jobs)}] {job[0]} {perf_counter() - t0:.0f}s "
        f"peak {peak:.2f} GiB",
        flush=True,
      )
    return
  with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=4) as pool:
    for i, _ in enumerate(pool.imap_unordered(_run, jobs)):
      print(f"[{i + 1}/{len(jobs)}] {perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
  main()
