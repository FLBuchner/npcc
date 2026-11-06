"""Paired A/B of quantile-table read-outs on E1's cells, gradient boosting.

Per data cell (E1's seeds, ``base_seed = 317``), each gradient-boosting
estimator is fitted once per recording grid. One evaluation pass records every
raw quantile row its two inner margins are asked for. Every read-out arm is
then scored by replaying those rows through E1's own scoring code, so the arms
differ only in the read-out and share every fitted model.

- LightGBM fits one booster per level, so it is recorded once on 81 levels and
  every grid is a subset of them.
- XGBoost and CatBoost fit all levels with one loss, so each grid is its own
  fit.

Output: one parquet per cell under ``--out``. Rows carry ``kind``
(``metric`` / ``diag`` / ``holdout`` / ``time``), the read-out ``arm``, and
the grid ``m`` / ``L``. Run with at most 4 processes; each is single-threaded.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
import torch

import npcc.core.margin as margin_module
from npcc.core.bicop import RosenblattBicop
from npcc.core.controls import FitControlsRosenblattBicop
from npcc.core.margin import QuantileTableConfig
from npcc.experiments import runner, scenarios
from npcc.experiments._study import derive_seed
from npcc.experiments.config import Cell, EstimatorSpec

sys.path.insert(0, str(Path(__file__).parent))
from readouts import Readout, table_factory

BASE_SEED = 317
UV_GRID_N, X_GRID_N, TAU_N = 20, 10, 1000
HOLDOUT_N = 2000
REC_LIMIT = 3.25
GRIDS: dict[str, tuple[int, float]] = {
  "m21": (21, 3.25),
  "m19": (19, 2.925),
  "m41": (41, 3.25),
}
THREADS = {
  "lightgbm": {"n_jobs": 1},
  "xgboost": {"n_jobs": 1},
  "catboost": {"thread_count": 1},
}
READOUTS = [
  Readout(smoothing=s, width=w, tails=t)
  for (s, w) in (("ma", 0.0), ("ma", 1.0), ("ma", 2.0), ("whittaker", 0.65))
  for t in ("current", "modeled")
]


class Store:
  """Raw quantile rows a margin was asked for, at the recording levels."""

  def __init__(self, scores: torch.Tensor) -> None:
    self.scores = scores
    self._rows: list[torch.Tensor] = []
    self._tables: list[torch.Tensor] = []
    self.rows: torch.Tensor | None = None
    self.table: torch.Tensor | None = None

  def add(self, x: torch.Tensor, table: torch.Tensor) -> None:
    self._rows.append(x.detach().clone())
    self._tables.append(table.detach().clone())

  def finalize(self) -> None:
    rows = torch.cat(self._rows)
    table = torch.cat(self._tables)
    uniq, inverse = torch.unique(rows, dim=0, return_inverse=True)
    first = torch.full((uniq.shape[0],), -1, dtype=torch.long)
    order = torch.arange(rows.shape[0])
    first.scatter_reduce_(0, inverse, order, reduce="amin", include_self=False)
    self.rows, self.table = uniq, table[first]
    self._rows, self._tables = [], []

  def columns(self, alphas: torch.Tensor) -> torch.Tensor:
    """Recorded columns holding the levels ``alphas``."""
    t = torch.special.ndtri(alphas)
    step = float(self.scores[1] - self.scores[0])
    idx = torch.round((t - self.scores[0]) / step).long()
    if (
      (idx < 0).any()
      or (idx >= self.scores.shape[0]).any()
      or (
        (self.scores[idx.clamp(0, self.scores.shape[0] - 1)] - t).abs().max()
        > 1e-6
      )
    ):
      raise ValueError("requested levels are not on the recorded grid")
    return idx

  def replay(self, x: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    assert self.rows is not None and self.table is not None
    n_store = self.rows.shape[0]
    both = torch.cat([self.rows, x])
    _, inverse = torch.unique(both, dim=0, return_inverse=True)
    group_to_store = torch.full((int(inverse.max()) + 1,), -1, dtype=torch.long)
    group_to_store[inverse[:n_store]] = torch.arange(n_store)
    idx = group_to_store[inverse[n_store:]]
    if (idx < 0).any():
      raise KeyError(f"{int((idx < 0).sum())} rows were not recorded")
    return self.table[idx][:, self.columns(alphas)]


def holdout(
  cell: Cell, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """A fresh sample from the cell's true copula and its true log density."""
  u, v, x = scenarios.sample(
    cell.family, cell.tau_scenario, HOLDOUT_N, derive_seed(seed, "holdout")
  )
  assert x is not None
  fam = scenarios.FAMILIES[cell.family]
  tau = scenarios.TAU_SCENARIOS[cell.tau_scenario].tau_of_x(x)
  uv = torch.column_stack([u, v])
  log_true = torch.tensor(
    [
      math.log(scenarios.bicop(fam, float(t)).pdf(uv[i : i + 1].numpy())[0])
      for i, t in enumerate(tau)
    ],
    dtype=torch.float64,
  )
  return uv, x.reshape(-1, 1), log_true


def evaluate(
  model: RosenblattBicop,
  cell: Cell,
  est: EstimatorSpec,
  seed: int,
  grid: scenarios.EvalGrid,
  truth: dict[str, torch.Tensor],
  ho: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
  tags: dict[str, object],
) -> list[dict[str, object]]:
  """E1's metric and diagnostic rows, plus held-out log-likelihood."""
  rows: list[dict[str, object]] = []
  t0 = perf_counter()
  with torch.inference_mode():
    hfuncs, pdfs, _ = runner._predict(model, grid, [None])
    for q, pred in hfuncs.items():
      rows += runner._metric_rows_for_quantity(
        cell,
        est,
        seed,
        q,
        truth[q],
        pred,
        grid,
        normalize="none",
        include_kl=False,
      )
    rows += runner._metric_rows_for_quantity(
      cell,
      est,
      seed,
      "pdf",
      truth["pdf"],
      pdfs["none"],
      grid,
      normalize="none",
      include_kl=True,
    )
    for r in rows:
      r["kind"] = "metric"
    diag, _ = runner._diagnostic_rows(
      cell,
      est,
      seed,
      model,
      pdfs,
      grid,
      enable_tau_diagnostics=True,
      tau_diagnostic_n=TAU_N,
    )
    for r in diag:
      r["kind"] = "diag"
    rows += diag
    uv, x, log_true = ho
    log_hat = torch.log(model.pdf(uv, x=x)).cpu()
    finite = torch.isfinite(log_hat)
    rows.append(
      {
        **runner._base_row(cell, est, seed),
        "kind": "holdout",
        "loglik": float(log_hat[finite].mean()),
        "holdout_kl": float((log_true - log_hat)[finite].mean()),
        "n_nonfinite": int((~finite).sum()),
      }
    )
  rows.append(
    {
      **runner._base_row(cell, est, seed),
      "kind": "time",
      "eval_s": perf_counter() - t0,
    }
  )
  for r in rows:
    r.update(tags)
  return rows


def run_cell(cell: Cell, out: Path, backends: list[str]) -> None:
  path = (
    out
    / f"{cell.family}__{cell.tau_scenario}__n{cell.n}__rep{cell.rep}.parquet"
  )
  if path.exists():
    return
  torch.set_num_threads(1)
  seed = runner._cell_seed(BASE_SEED, cell)
  u, v, x = scenarios.sample(cell.family, cell.tau_scenario, cell.n, seed)
  grid = scenarios.eval_grid(
    cell.tau_scenario,
    conditional_uv_grid_n=UV_GRID_N,
    conditional_x_grid_n=X_GRID_N,
  )
  truth = scenarios.ground_truth(cell.family, cell.tau_scenario, grid)
  ho = holdout(cell, seed)
  rows: list[dict[str, object]] = []
  library_table = margin_module.QuantileTable

  for backend in backends:
    for transform in ("logit", "probit"):
      est = EstimatorSpec(
        label=f"{backend}-{transform}", backend=backend, transform=transform
      )
      rec_grids = (
        {"m81": (81, REC_LIMIT)} if backend == "lightgbm" else dict(GRIDS)
      )
      for rec_name, (m_rec, l_rec) in rec_grids.items():
        controls = FitControlsRosenblattBicop(
          backend=backend,
          transform=transform,
          quantile_table_config=QuantileTableConfig(
            n_quantiles=m_rec, normal_limit=l_rec
          ),
          device="cpu",
          projection_grid_size=30,
          batch_size=1_000_000,
          backend_kwargs=THREADS[backend],
        )
        torch.manual_seed(derive_seed(seed, est.estimator_id))
        t0 = perf_counter()
        model = RosenblattBicop(controls)
        model.fit(torch.column_stack([u, v]), x=x.reshape(-1, 1))
        fit_s = perf_counter() - t0

        margins = (model.v_given_ux_, model.u_given_vx_)
        stores = []
        for mg in margins:
          store = Store(
            torch.linspace(-l_rec, l_rec, m_rec, dtype=torch.float64)
          )
          original = mg._predict_quantiles

          def recording(xc, alphas, _orig=original, _store=store):  # noqa: ANN001, ANN202
            out_q = _orig(xc, alphas)
            _store.add(xc, out_q)
            return out_q

          mg._predict_quantiles = recording
          stores.append(store)
        t0 = perf_counter()
        evaluate(model, cell, est, seed, grid, truth, ho, {})
        record_s = perf_counter() - t0
        for mg, store in zip(margins, stores, strict=True):
          store.finalize()
          mg._predict_quantiles = store.replay
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

        served = GRIDS if backend == "lightgbm" else {rec_name: GRIDS[rec_name]}
        try:
          for grid_name, (m, lim) in served.items():
            cfg = QuantileTableConfig(n_quantiles=m, normal_limit=lim)
            for mg in margins:
              mg.quantile_table_config = cfg
            for rd in READOUTS:
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

  frame = pd.DataFrame(rows)
  tmp = path.with_suffix(".tmp")
  frame.to_parquet(tmp)
  tmp.rename(path)


def cells(ns: list[int], reps: list[int]) -> list[Cell]:
  return [
    Cell(family=f, tau_scenario=s, n=n, rep=r)
    for f in ("clayton", "gaussian")
    for s in ("linear", "sin")
    for n in ns
    for r in reps
  ]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--n", type=int, nargs="+", default=[200, 500, 2000])
  parser.add_argument("--reps", type=int, nargs="+", default=list(range(10)))
  parser.add_argument(
    "--backends", nargs="+", default=["lightgbm", "xgboost", "catboost"]
  )
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument(
    "--only", nargs=4, metavar=("FAMILY", "SCENARIO", "N", "REP")
  )
  args = parser.parse_args()
  if args.workers > 4:
    raise SystemExit("at most 4 workers: the machine is shared with paper runs")
  args.out.mkdir(parents=True, exist_ok=True)
  todo = (
    [Cell(args.only[0], args.only[1], int(args.only[2]), int(args.only[3]))]
    if args.only
    else cells(args.n, args.reps)
  )
  # Largest n first, so the slow cells do not straggle at the end.
  todo.sort(key=lambda c: -c.n)
  if args.workers == 1:
    for c in todo:
      run_cell(c, args.out, args.backends)
    return
  import multiprocessing as mp

  ctx = mp.get_context("spawn")
  with ctx.Pool(args.workers, maxtasksperchild=4) as pool:
    for _ in pool.imap_unordered(
      _run, [(c, args.out, args.backends) for c in todo]
    ):
      pass


def _run(job: tuple[Cell, Path, list[str]]) -> None:
  run_cell(*job)


if __name__ == "__main__":
  main()
