"""The covariate study (E9): fit every estimator on a cell, score it per point.

A cell draws its training sample from
:func:`~npcc.experiments.covariate_scenarios.sample`. Every estimator is
scored at the scenario's fixed covariate points: at each, the IAE, ISE and
grid KL of the density and both h-functions on E1's ``uv`` grid, and the
density's Monte Carlo KL at ``holdout_n`` draws from the true copula there,
seeded per (cell, point) so every estimator of the cell scores the same
draws. A registry backend is fitted as E1 fits it, on the covariates its
:meth:`~npcc.experiments.covariate_config.CovariateGridConfig.covariates`
names; an external script is handed all of them, and the scenario.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import NamedTuple, cast

import pandas as pd
import torch

from npcc.core.bicop import RosenblattBicop
from npcc.core.controls import FitControlsRosenblattBicop
from npcc.experiments import _external, metrics
from npcc.experiments import covariate_scenarios as cs
from npcc.experiments._study import (
  Rows,
  derive_seed,
  memory_usage,
  release_gpu,
  reset_gpu_peaks,
  sorted_frame,
  summarize,
  sweep,
)
from npcc.experiments.config import EstimatorSpec, RunConfig
from npcc.experiments.covariate_config import CovariateCell, CovariateGridConfig
from npcc.experiments.scenarios import QUANTITIES, EvalGrid

logger = logging.getLogger(__name__)

TABLES: tuple[str, ...] = ("metrics", "runtime")
METRIC_NAMES: tuple[str, ...] = ("IAE", "ISE", "KL", "KL_mc")
_GROUP_AXES: tuple[str, ...] = (
  "family",
  "scenario",
  "n",
  "label",
  "backend",
  "transform",
  "covariates",
  "quantity",
)


def cell_seed(base_seed: int, cell: CovariateCell) -> int:
  """The seed of a cell's training sample and of its draws.

  Parameters
  ----------
  base_seed : int
      The run's base seed.
  cell : CovariateCell
      The cell.

  Returns
  -------
  int
      Its seed, hashed from the cell's axes by name.
  """
  return derive_seed(
    "E9", base_seed, cell.family, cell.scenario, cell.n, cell.rep
  )


def eval_points(
  grid: CovariateGridConfig, base_seed: int, scenario: str
) -> torch.Tensor:
  """The covariate points ``scenario`` is scored at.

  Parameters
  ----------
  grid : CovariateGridConfig
      The grid, for ``n_points``.
  base_seed : int
      The run's base seed.
  scenario : str
      The scenario.

  Returns
  -------
  torch.Tensor
      ``(n_points, 15)``, the same for every family, ``n``, repetition and
      estimator of the scenario.
  """
  return cs.points(grid.n_points, derive_seed("E9 points", base_seed, scenario))


def _regime_fingerprint() -> dict[str, list[float]]:
  """Every scenario's true tau at fixed probe covariates, per family."""
  probe = cs.points(8, 0)
  return {
    f"{family}:{scenario}": [
      round(v, 12) for v in cs.kendall_tau(family, scenario, probe).tolist()
    ]
    for family in cs.SCENARIO_FAMILIES
    for scenario in cs.SCENARIOS
  }


def grid_signature(grid: CovariateGridConfig, run: RunConfig) -> str:
  """Stable hash of everything that changes a cell's outputs (resume guard).

  Parameters
  ----------
  grid : CovariateGridConfig
      The grid.
  run : RunConfig
      The run-level options.

  Returns
  -------
  str
      A SHA-256 hex digest.
  """
  payload = {
    "study": "covariates",
    # The data-generating process, by value, so redefining a scenario
    # invalidates the cells drawn from the old one.
    "covariates": list(cs.COVARIATES),
    "regimes": _regime_fingerprint(),
    # The axes that choose which cells run are absent: a cell's data and
    # seeds derive from its own (family, scenario, n, rep).
    "n_points": grid.n_points,
    "uv_grid_n": grid.uv_grid_n,
    "holdout_n": grid.holdout_n,
    "projection_grid_size": grid.projection_grid_size,
    "base_seed": run.base_seed,
    "estimators": sorted(
      f"{e.estimator_id}:{grid.covariates(e)}" for e in grid.estimators
    ),
  }
  encoded = json.dumps(payload, sort_keys=True).encode()
  return hashlib.sha256(encoded).hexdigest()


class _CellData(NamedTuple):
  """One cell's sample, scoring points and truth, shared by its estimators."""

  seed: int
  u: torch.Tensor
  v: torch.Tensor
  x: torch.Tensor
  tau_points: torch.Tensor
  metric_grid: EvalGrid
  truth: dict[str, torch.Tensor]
  holdout: EvalGrid
  holdout_log_true: torch.Tensor


def _cell_data(
  cell: CovariateCell, grid: CovariateGridConfig, base_seed: int
) -> _CellData:
  seed = cell_seed(base_seed, cell)
  u, v, x = cs.sample(cell.family, cell.scenario, cell.n, seed)
  pts = eval_points(grid, base_seed, cell.scenario)
  metric_grid = cs.metric_grid(pts, grid.uv_grid_n)
  holdout = cs.holdout_grid(
    cell.family,
    cell.scenario,
    pts,
    grid.holdout_n,
    [derive_seed(seed, "holdout", j) for j in range(pts.shape[0])],
  )
  holdout_pdf = cs.ground_truth(cell.family, cell.scenario, holdout)["pdf"]
  return _CellData(
    seed=seed,
    u=u,
    v=v,
    x=x,
    tau_points=cs.kendall_tau(cell.family, cell.scenario, pts),
    metric_grid=metric_grid,
    truth=cs.ground_truth(cell.family, cell.scenario, metric_grid),
    holdout=holdout,
    holdout_log_true=metrics.log_density(holdout_pdf.reshape(-1)),
  )


def _base_row(
  cell: CovariateCell, est: EstimatorSpec, seed: int, covariates: str
) -> dict[str, object]:
  return {
    "family": cell.family,
    "scenario": cell.scenario,
    "n": cell.n,
    "rep": cell.rep,
    "seed": seed,
    "label": est.label,
    "estimator_id": est.estimator_id,
    "transform": est.transform,
    "backend": est.backend,
    "covariates": covariates,
    # Empty unless the fit failed; see `_external_rows`.
    "error": "",
  }


def _metric_rows(
  base: Mapping[str, object],
  data: _CellData,
  preds: Mapping[str, torch.Tensor],
  holdout: Sequence[Mapping[str, float]],
) -> Rows:
  """One row per (quantity, point): the grid metrics, and the density's KL_mc."""
  shape = data.metric_grid.shape
  rows: Rows = []
  for q in QUANTITIES:
    truth, pred = data.truth[q], preds[q].reshape(shape)
    for j in range(shape[1]):
      rows.append(
        {
          **base,
          "quantity": q,
          "point": j,
          "tau_true": float(data.tau_points[j]),
          **metrics.grid_metrics(
            truth[:, j], pred[:, j], include_kl=q == "pdf"
          ),
          **(holdout[j] if q == "pdf" else {}),
        }
      )
  return rows


def _select(
  x: torch.Tensor | None, scenario: str, covariates: str
) -> torch.Tensor | None:
  """The columns of ``x`` an estimator fitted on ``covariates`` sees."""
  if x is None or covariates == "none":
    return None
  if covariates == "all":
    return x
  return x[:, [cs.column(name) for name in cs.ACTIVE[scenario]]]


def _backend_rows(
  cell: CovariateCell,
  est: EstimatorSpec,
  data: _CellData,
  grid: CovariateGridConfig,
  *,
  device: str | None,
) -> tuple[Rows, Rows]:
  """Fit one registry-backend estimator and score it at every point."""
  covariates = grid.covariates(est)
  base = _base_row(cell, est, data.seed, covariates)
  # Fitted on every covariate, an estimator takes E1's seed for its id; on
  # a subset or none, the seed also names that, so the variants differ.
  salt = () if covariates == "all" else (covariates,)
  torch.manual_seed(derive_seed(data.seed, est.estimator_id, *salt))
  reset_gpu_peaks()

  t0 = perf_counter()
  model = RosenblattBicop(
    est.controls(
      FitControlsRosenblattBicop,
      device=device,
      projection_grid_size=grid.projection_grid_size,
      batch_size=grid.batch_size,
    )
  )
  model.fit(
    torch.column_stack([data.u, data.v]),
    x=_select(data.x, cell.scenario, covariates),
  )
  fit_time = perf_counter() - t0

  mg, hg = data.metric_grid, data.holdout
  x_metric = _select(mg.x_flat, cell.scenario, covariates)
  with torch.inference_mode():
    t0 = perf_counter()
    preds = {
      "pdf": model.pdf(mg.uv, x=x_metric).cpu(),
      "hfunc1": model.hfunc1(mg.uv, x=x_metric).cpu(),
      "hfunc2": model.hfunc2(mg.uv, x=x_metric).cpu(),
    }
    grid_time = perf_counter() - t0
    t0 = perf_counter()
    holdout_pdf = model.pdf(
      hg.uv, x=_select(hg.x_flat, cell.scenario, covariates)
    ).cpu()
    holdout_time = perf_counter() - t0
  holdout = metrics.holdout_kl_by_x(hg, data.holdout_log_true, holdout_pdf)

  memory = memory_usage()
  logger.info(
    "estimator %s (%s covariates) [%s]: fit %.1f s | GPU peak %.0f MiB | "
    "RSS %.0f MiB | MemAvailable %.0f MiB",
    est.label,
    covariates,
    cell.key,
    fit_time,
    memory["gpu_peak_reserved_mb"],
    memory["rss_mb"],
    memory["mem_available_mb"],
  )
  runtime = {
    **base,
    "fit_time": fit_time,
    "grid_time": grid_time,
    "holdout_time": holdout_time,
    "total_time": fit_time + grid_time + holdout_time,
    **memory,
  }
  del model
  release_gpu()
  return _metric_rows(base, data, preds, holdout), [runtime]


def _failed_rows(
  base: Mapping[str, object], data: _CellData, error: str
) -> tuple[Rows, Rows]:
  """A failed fit's rows: every score missing, and the error that says why.

  A failure is a result of the method, kept as rows so that a mean over the
  cells cannot drop it without saying so.
  """
  failed = {**base, "error": error}
  nan = float("nan")
  rows = [
    {
      **failed,
      "quantity": q,
      "point": j,
      "tau_true": float(data.tau_points[j]),
      "IAE": nan,
      "ISE": nan,
      "KL": nan,
      **({"KL_mc": nan} if q == "pdf" else {}),
    }
    for q in QUANTITIES
    for j in range(data.metric_grid.shape[1])
  ]
  runtime = {
    **failed,
    "fit_time": nan,
    "grid_time": nan,
    "holdout_time": nan,
    "total_time": nan,
  }
  return rows, [runtime]


def _external_rows(
  cell: CovariateCell, est: EstimatorSpec, data: _CellData
) -> tuple[Rows, Rows]:
  """Fit one external script on every covariate, and score it at every point.

  A script that fails, or whose predictions the exchange refuses, yields
  :func:`_failed_rows` rather than stopping the cell: a selection that finds
  nothing, or a family whose fit does not converge, is the method's outcome
  on that sample, and the analysis reports it as such.
  """
  assert est.script is not None
  base = _base_row(cell, est, data.seed, "all")
  try:
    fit = _external.fit_and_predict(
      est.script,
      (data.u, data.v, data.x),
      {"metric": data.metric_grid, "holdout": data.holdout},
      seed=derive_seed(data.seed, est.estimator_id),
      name=f"{est.label} on {cell.key}",
      covariates=cs.COVARIATES,
      extra_args=(cell.scenario, cell.family),
    )
  except (RuntimeError, ValueError) as exc:
    # A missing interpreter is this machine's fault, not the method's.
    if isinstance(exc.__cause__, FileNotFoundError):
      raise
    logger.warning("estimator %s [%s] failed: %s", est.label, cell.key, exc)
    return _failed_rows(base, data, str(exc))
  holdout = metrics.holdout_kl_by_x(
    data.holdout, data.holdout_log_true, fit.predictions["holdout"]["pdf"]
  )
  # The script runs on the host, so its GPU peaks are zero.
  memory = {
    **memory_usage(),
    "gpu_peak_reserved_mb": 0.0,
    "gpu_peak_alloc_mb": 0.0,
  }
  logger.info(
    "estimator %s [%s]: script run %.1f s (fit %.1f s) | RSS %.0f MiB | "
    "MemAvailable %.0f MiB",
    est.label,
    cell.key,
    fit.wall_time,
    fit.fit_time,
    memory["rss_mb"],
    memory["mem_available_mb"],
  )
  timings = fit.timings["metric"]
  grid_time = timings["pdf"] + timings["hfunc1"] + timings["hfunc2"]
  holdout_time = fit.timings["holdout"]["pdf"]
  runtime = {
    **base,
    "fit_time": fit.fit_time,
    "grid_time": grid_time,
    "holdout_time": holdout_time,
    "total_time": fit.fit_time + grid_time + holdout_time,
    **memory,
  }
  preds = {q: fit.predictions["metric"][q] for q in QUANTITIES}
  return _metric_rows(base, data, preds, holdout), [runtime]


def summarize_one_cell(
  cell: CovariateCell,
  grid: CovariateGridConfig,
  *,
  base_seed: int,
  device: str | None,
) -> tuple[Rows, Rows]:
  """Fit every estimator the cell's scenario takes, and return its rows.

  Parameters
  ----------
  cell : CovariateCell
      The cell.
  grid : CovariateGridConfig
      The grid.
  base_seed : int
      The run's base seed.
  device : str or None
      The registry backends' device.

  Returns
  -------
  tuple of list
      The metric rows and the runtime rows.
  """
  data = _cell_data(cell, grid, base_seed)
  metric_rows: Rows = []
  runtime_rows: Rows = []
  for est in grid.estimators:
    if not grid.fits(est, cell.scenario):
      continue
    if est.kind == "external_r":
      new = _external_rows(cell, est, data)
    else:
      new = _backend_rows(cell, est, data, grid, device=device)
    metric_rows += new[0]
    runtime_rows += new[1]
  release_gpu()
  return metric_rows, runtime_rows


def run_covariate_study(
  grid: CovariateGridConfig, run: RunConfig, *, resume: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
  """Sweep the grid; return the metric and runtime tables and the wall time.

  Each cell's rows are checkpointed to ``run.out/cells/<key>/`` as they
  finish. With ``resume=True``, cells already marked done are read back and
  skipped, after the grid signature is checked against the manifest.

  Parameters
  ----------
  grid : CovariateGridConfig
      The grid.
  run : RunConfig
      The run-level options.
  resume : bool, default False
      Whether to reuse completed cells.

  Returns
  -------
  tuple
      The metric table, the runtime table, and the seconds the sweep took.
  """
  logger.info("Covariate study: %d estimators per cell", len(grid.estimators))
  (metric_df, runtime_df), wall = sweep(
    grid.cells(),
    lambda cell: summarize_one_cell(
      cell, grid, base_seed=run.base_seed, device=run.device
    ),
    tables=TABLES,
    run=run,
    signature=grid_signature(grid, run),
    study="covariate",
    resume=resume,
  )
  axes = ["family", "scenario", "n", "label", "covariates", "rep"]
  return (
    sorted_frame(metric_df, [*axes, "quantity", "point"]),
    sorted_frame(runtime_df, axes),
    wall,
  )


def aggregate_covariate_outputs(
  metric_df: pd.DataFrame, runtime_df: pd.DataFrame
) -> dict[str, pd.DataFrame]:
  """Summaries over repetitions, per covariate point and averaged over them.

  Parameters
  ----------
  metric_df : pandas.DataFrame
      One row per (cell, estimator, quantity, point).
  runtime_df : pandas.DataFrame
      One row per (cell, estimator).

  Returns
  -------
  dict of str to pandas.DataFrame
      The tables to write, by file stem.
  """
  summary_by_point, summary_over_points = summarize(
    metric_df, METRIC_NAMES, _GROUP_AXES, x_axes=("point",)
  )
  axes = [a for a in _GROUP_AXES if a != "quantity"]
  runtime_summary = runtime_df.groupby(axes, as_index=False, dropna=False).agg(
    fit_time_mean=("fit_time", "mean"),
    fit_time_std=("fit_time", "std"),
    total_time_mean=("total_time", "mean"),
  )
  return {
    "metrics_by_point": metric_df,
    "summary_by_point": summary_by_point,
    "summary_over_points": summary_over_points,
    "runtime": runtime_df,
    "runtime_summary": cast("pd.DataFrame", runtime_summary),
  }


def points_table(grid: CovariateGridConfig, base_seed: int) -> pd.DataFrame:
  """Every scenario's covariate points, one row per (scenario, point).

  Parameters
  ----------
  grid : CovariateGridConfig
      The grid.
  base_seed : int
      The run's base seed.

  Returns
  -------
  pandas.DataFrame
      ``scenario``, ``point``, the 15 covariates, and the true tau per family.
  """
  frames = []
  for scenario in grid.scenarios:
    pts = eval_points(grid, base_seed, scenario)
    frame = pd.DataFrame(pts.numpy(), columns=list(cs.COVARIATES))
    frame.insert(0, "point", range(pts.shape[0]))
    frame.insert(0, "scenario", scenario)
    for family in grid.families:
      frame[f"tau_{family}"] = cs.kendall_tau(family, scenario, pts).numpy()
    frames.append(frame)
  return pd.concat(frames, ignore_index=True)
