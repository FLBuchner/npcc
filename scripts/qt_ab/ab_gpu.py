"""The read-out A/B for an in-context backend on the GPU (TabPFN v3, logit).

Same cells, arms, scoring and output as :mod:`ab_gb`. TabPFN computes every
level from one predicted histogram, so one recording on 81 levels serves every
grid. The recording wrapper predicts each distinct conditioning row once,
which is what keeps the GPU pass short; the backend is row-independent, so the
rows it returns are the ones a full pass would.

Run with the CUDA environment and the worktree's package, e.g.::

    PYTHONPATH=../npcc-qt-rework/src ~/npcc/.venv/bin/python \
      scripts/qt_ab/ab_gpu.py --out results/qt_ab/tabpfn --n 200 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ab_gb
from ab_gb import GRIDS, READOUTS, Store, evaluate, holdout
from readouts import table_factory

import npcc.core.margin as margin_module
from npcc.core.bicop import RosenblattBicop
from npcc.core.controls import FitControlsRosenblattBicop
from npcc.core.margin import QuantileTableConfig
from npcc.experiments import runner, scenarios
from npcc.experiments._study import derive_seed
from npcc.experiments.config import Cell, EstimatorSpec

ESTIMATORS = {
  "tabpfn-quantiles-v3-logit": (
    "tabpfn-quantiles",
    "logit",
    {"model_version": "v3"},
  ),
  "tabicl-logit": ("tabicl", "logit", {}),
  "nori-logit": ("nori", "logit", {}),
}


def cap_gpu_memory(gb: float) -> None:
  """Make this process fail on its own allocations past ``gb`` GiB.

  The GPU is shared with paper runs; an over-sized allocation here must raise
  here rather than push another process out of memory.
  """
  if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb / total))


def run_cell(
  cell: Cell,
  out: Path,
  device: str,
  labels: list[str],
  grids: list[str],
  arms: list[str] | None,
) -> None:
  path = (
    out
    / f"{cell.family}__{cell.tau_scenario}__n{cell.n}__rep{cell.rep}.parquet"
  )
  if path.exists():
    return
  seed = runner._cell_seed(ab_gb.BASE_SEED, cell)
  u, v, x = scenarios.sample(cell.family, cell.tau_scenario, cell.n, seed)
  grid = scenarios.eval_grid(
    cell.tau_scenario,
    conditional_uv_grid_n=ab_gb.UV_GRID_N,
    conditional_x_grid_n=ab_gb.X_GRID_N,
  )
  truth = scenarios.ground_truth(cell.family, cell.tau_scenario, grid)
  ho = holdout(cell, seed)
  rows: list[dict[str, object]] = []
  library_table = margin_module.QuantileTable

  readouts = [rd for rd in READOUTS if arms is None or rd.label in arms]
  for label in labels:
    backend, transform, kwargs = ESTIMATORS[label]
    est = EstimatorSpec(
      label=label, backend=backend, transform=transform, backend_kwargs=kwargs
    )
    controls = FitControlsRosenblattBicop(
      backend=backend,
      transform=transform,
      quantile_table_config=QuantileTableConfig(
        n_quantiles=81, normal_limit=ab_gb.REC_LIMIT
      ),
      device=device,
      projection_grid_size=30,
      batch_size=None,
      backend_kwargs=kwargs,
    )
    torch.manual_seed(derive_seed(seed, est.estimator_id))
    t0 = perf_counter()
    model = RosenblattBicop(controls)
    model.fit(torch.column_stack([u, v]), x=x)
    fit_s = perf_counter() - t0

    margins = (model.v_given_ux_, model.u_given_vx_)
    stores = []
    for mg in margins:
      store = Store(
        torch.linspace(
          -ab_gb.REC_LIMIT, ab_gb.REC_LIMIT, 81, dtype=torch.float64
        )
      )
      original = mg._predict_quantiles

      def recording(xc, alphas, _orig=original, _store=store):  # noqa: ANN001, ANN202
        distinct, inverse = torch.unique(xc, dim=0, return_inverse=True)
        out_q = _orig(distinct, alphas)
        _store.add(distinct.cpu(), out_q.cpu())
        return out_q[inverse]

      mg._predict_quantiles = recording
      stores.append(store)
    t0 = perf_counter()
    evaluate(model, cell, est, seed, grid, truth, ho, {})
    record_s = perf_counter() - t0
    for mg, store in zip(margins, stores, strict=True):
      store.finalize()

      # The store stays on the host; the margin's table goes back to its device.
      def replay(xc, alphas, _store=store):  # noqa: ANN001, ANN202
        return _store.replay(xc.cpu(), alphas.cpu()).to(xc.device)

      mg._predict_quantiles = replay
    rows.append(
      {
        **runner._base_row(cell, est, seed),
        "kind": "fit",
        "rec_grid": "m81",
        "fit_s": fit_s,
        "record_s": record_s,
        "rows_recorded": int(stores[0].rows.shape[0]),
      }
    )
    try:
      for grid_name in grids:
        m, lim = GRIDS[grid_name]
        cfg = QuantileTableConfig(n_quantiles=m, normal_limit=lim)
        for mg in margins:
          mg.quantile_table_config = cfg
        for rd in readouts:
          margin_module.QuantileTable = table_factory(rd)
          rows += evaluate(
            model,
            cell,
            est,
            seed,
            grid,
            truth,
            ho,
            {
              "arm": rd.label,
              "smoothing": rd.smoothing,
              "width": rd.width,
              "tails": rd.tails,
              "grid": grid_name,
              "m": m,
              "L": lim,
            },
          )
    finally:
      margin_module.QuantileTable = library_table
    del model, margins, stores
    torch.cuda.empty_cache()

  frame = pd.DataFrame(rows)
  tmp = path.with_suffix(".tmp")
  frame.to_parquet(tmp)
  tmp.rename(path)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--n", type=int, nargs="+", default=[200, 2000])
  parser.add_argument("--reps", type=int, nargs="+", default=list(range(10)))
  parser.add_argument("--device", default="cuda")
  parser.add_argument(
    "--labels",
    nargs="+",
    default=["tabpfn-quantiles-v3-logit"],
    choices=list(ESTIMATORS),
  )
  parser.add_argument(
    "--grids", nargs="+", default=list(GRIDS), choices=list(GRIDS)
  )
  parser.add_argument(
    "--arms",
    nargs="+",
    default=None,
    help="read-out labels to replay, e.g. 'ma1|current'; default all",
  )
  parser.add_argument(
    "--gpu-mem-gb",
    type=float,
    default=3.0,
    help="hard cap on this process's CUDA allocations",
  )
  args = parser.parse_args()
  torch.set_num_threads(4)
  cap_gpu_memory(args.gpu_mem_gb)
  args.out.mkdir(parents=True, exist_ok=True)
  todo = ab_gb.cells(args.n, args.reps)
  t0 = perf_counter()
  for i, c in enumerate(todo):
    run_cell(c, args.out, args.device, args.labels, args.grids, args.arms)
    peak = (
      torch.cuda.max_memory_reserved() / 2**30
      if torch.cuda.is_available()
      else 0.0
    )
    print(
      f"[{i + 1}/{len(todo)}] {c} {perf_counter() - t0:.0f}s "
      f"peak reserved {peak:.2f} GiB",
      flush=True,
    )


if __name__ == "__main__":
  main()
