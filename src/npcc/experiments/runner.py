"""Run the simulation study: sweep the grid, score, diagnose, aggregate.

Each data cell ``(family, tau_scenario, n, rep)`` samples its own training set
and ground truth once.  Estimator configs are fitted on that shared data.

Conditional scenarios are scored at fixed ``x`` values: errors are averaged over
the uv grid first, and only then summarized over Monte-Carlo repetitions or
conditioning values. This keeps KL as a fixed-``x`` copula-density metric.
"""

from __future__ import annotations

import hashlib
import json
import logging
from time import perf_counter
from typing import Any, NamedTuple, cast

import pandas as pd
import torch

from npcc.core.bicop import RosenblattBicop
from npcc.core.controls import FitControlsRosenblattBicop
from npcc.experiments import metrics, scenarios
from npcc.experiments._study import (
  SUMMARY_STATS,
  Rows,
  derive_seed,
  memory_usage,
  release_gpu,
  reset_gpu_peaks,
  summarize,
  summary_stats,
  sweep,
)
from npcc.experiments.config import Cell, EstimatorSpec, GridConfig, RunConfig
from npcc.experiments.scenarios import EvalGrid

logger = logging.getLogger(__name__)

TABLES: tuple[str, ...] = ("metrics", "quantities", "diagnostics", "runtime")


def _norm_label(norm: int | None) -> str:
  return "none" if norm is None else str(norm)


def _regime_fingerprint(grid: GridConfig) -> dict[str, list[float] | float]:
  """By-value fingerprint of each tau regime used in the grid.

  Evaluating each conditional regime on a fixed ``x`` grid (or reading a fixed
  tau) makes a change to a regime's bounds *or* shape invalidate a checkpoint;
  the band alone would miss a same-range redefinition, such as re-centering the
  sinusoid.
  """
  fp_x = torch.linspace(
    scenarios.X_MIN, scenarios.X_MAX, 11, dtype=torch.float64
  )
  fp: dict[str, list[float] | float] = {}
  for name in sorted(grid.tau_scenarios):
    spec = scenarios.TAU_SCENARIOS[name]
    if spec.conditional:
      assert spec.tau_of_x is not None
      fp[name] = [round(float(t), 8) for t in spec.tau_of_x(fp_x)]
    else:
      assert spec.tau is not None
      fp[name] = round(float(spec.tau), 8)
  return fp


def _grid_signature(grid: GridConfig, run: RunConfig) -> str:
  """Stable hash of everything that changes cell outputs (resume guard)."""
  payload = {
    "families": sorted(grid.families),
    "tau_scenarios": sorted(grid.tau_scenarios),
    # A by-value fingerprint of each regime, so changing a regime's bounds or
    # shape invalidates a checkpoint; a --resume must not mix definitions.
    "tau_regimes": _regime_fingerprint(grid),
    "n": sorted(grid.n),
    # n_rep is intentionally absent: each cell's seed hashes cell.rep, not
    # n_rep, so raising it adds reps and resumes rather than discarding cells.
    "normalize": sorted(_norm_label(x) for x in grid.normalize),
    "projection_grid_size": grid.projection_grid_size,
    # `batch_size` is absent because it chunks the query set and leaves each
    # prediction's context alone, so it changes throughput and not results.
    # Including it would make re-tuning the batch for a different GPU discard
    # every completed cell, which is the opposite of what `--resume` is for.
    "conditional_uv_grid_n": grid.conditional_uv_grid_n,
    "conditional_x_grid_n": grid.conditional_x_grid_n,
    "surface_tau_levels": sorted(grid.surface_tau_levels),
    "surface_families": sorted(grid.surface_families),
    "enable_tau_diagnostics": grid.enable_tau_diagnostics,
    "tau_diagnostic_n": grid.tau_diagnostic_n,
    "base_seed": run.base_seed,
    "estimators": sorted(s.estimator_id for s in grid.estimators),
  }
  encoded = json.dumps(payload, sort_keys=True).encode()
  return hashlib.sha256(encoded).hexdigest()


def _cell_seed(base_seed: int, cell: Cell) -> int:
  """Deterministic, axis-decorrelated seed for a data cell."""
  fam_idx = list(scenarios.FAMILIES).index(cell.family)
  scn_idx = list(scenarios.TAU_SCENARIOS).index(cell.tau_scenario)
  payload = f"{base_seed}:{fam_idx}:{scn_idx}:{cell.n}:{cell.rep}".encode()
  return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _base_row(cell: Cell, est: EstimatorSpec, seed: int) -> dict[str, object]:
  return {
    "family": cell.family,
    "tau_scenario": cell.tau_scenario,
    "n": cell.n,
    "rep": cell.rep,
    "seed": seed,
    "label": est.label,
    "estimator_id": est.estimator_id,
    "transform": est.transform,
    "backend": est.backend,
  }


def _tau_values(scenario: str, x_axis: torch.Tensor | None) -> torch.Tensor:
  spec = scenarios.TAU_SCENARIOS[scenario]
  if spec.conditional:
    assert spec.tau_of_x is not None
    assert x_axis is not None
    return spec.tau_of_x(x_axis)
  assert spec.tau is not None
  return torch.tensor([spec.tau], dtype=torch.float64)


def _metric_rows_for_quantity(
  cell: Cell,
  est: EstimatorSpec,
  seed: int,
  quantity: str,
  truth: torch.Tensor,
  pred: torch.Tensor,
  grid: EvalGrid,
  *,
  normalize: str,
  include_kl: bool,
) -> list[dict[str, object]]:
  """Metric rows after averaging pointwise errors over uv at fixed x."""
  base = {
    **_base_row(cell, est, seed),
    "normalize": normalize,
    "quantity": quantity,
  }

  if grid.conditional:
    assert grid.x_axis is not None
    tau_x = _tau_values(cell.tau_scenario, grid.x_axis)
    truth_grid = truth.reshape(grid.shape)
    pred_grid = pred.reshape(grid.shape)
    return [
      {
        **base,
        "x": float(x_val),
        "tau_true": float(tau_x[x_idx]),
        **metrics.grid_metrics(
          truth_grid[:, x_idx], pred_grid[:, x_idx], include_kl=include_kl
        ),
      }
      for x_idx, x_val in enumerate(grid.x_axis)
    ]

  stats = metrics.grid_metrics(truth, pred, include_kl=include_kl)
  tau = _tau_values(cell.tau_scenario, None)[0]
  return [{**base, "x": float("nan"), "tau_true": float(tau), **stats}]


def _surface_x_rows(
  scenario: str, tau_levels: list[float]
) -> list[dict[str, Any]]:
  """Map target Kendall-tau levels to deterministic x slice(s)."""
  spec = scenarios.TAU_SCENARIOS[scenario]
  if not spec.conditional:
    assert spec.tau is not None
    return [
      {
        "target_tau": float(spec.tau),
        "x": float("nan"),
        "tau_true": float(spec.tau),
      }
    ]

  assert spec.tau_of_x is not None
  x_dense = torch.linspace(
    scenarios.X_MIN, scenarios.X_MAX, 2001, dtype=torch.float64
  )
  tau_dense = spec.tau_of_x(x_dense)
  rows: list[dict[str, Any]] = []
  for target in tau_levels:
    roots: list[float] = []
    delta = tau_dense - target
    exact = torch.where(
      torch.isclose(delta, torch.zeros_like(delta), atol=1e-6)
    )[0]
    roots.extend(float(x_dense[i]) for i in exact)
    for i in range(x_dense.shape[0] - 1):
      if delta[i] == 0.0 or delta[i] * delta[i + 1] > 0.0:
        continue
      lo = float(x_dense[i])
      hi = float(x_dense[i + 1])
      for _ in range(40):
        mid = (lo + hi) / 2.0
        if (float(spec.tau_of_x(torch.tensor([lo]))[0]) - target) * (
          float(spec.tau_of_x(torch.tensor([mid]))[0]) - target
        ) <= 0.0:
          hi = mid
        else:
          lo = mid
      roots.append((lo + hi) / 2.0)
    if not roots:
      roots.append(float(x_dense[int(delta.abs().argmin())]))

    unique_roots: list[float] = []
    for root in sorted(roots):
      if not unique_roots or abs(root - unique_roots[-1]) > 1e-4:
        unique_roots.append(root)
    for root in unique_roots:
      tau_true = float(spec.tau_of_x(torch.tensor([root]))[0])
      rows.append(
        {"target_tau": float(target), "x": root, "tau_true": tau_true}
      )
  return rows


def _quantity_rows(
  cell: Cell,
  est: EstimatorSpec,
  seed: int,
  quantity: str,
  truth: torch.Tensor,
  pred: torch.Tensor,
  grid: EvalGrid,
  *,
  normalize: str,
  target_tau: torch.Tensor,
  tau_true: torch.Tensor,
) -> list[dict[str, object]]:
  base = _base_row(cell, est, seed)
  truth_grid = truth.reshape(grid.shape)
  pred_grid = pred.reshape(grid.shape)
  rows: list[dict[str, object]] = []
  assert grid.x_axis is not None
  for pair_idx in range(grid.shape[0]):
    u_val = float(grid.u_flat[pair_idx * grid.x_axis.shape[0]])
    v_val = float(grid.v_flat[pair_idx * grid.x_axis.shape[0]])
    for x_idx, x_val in enumerate(grid.x_axis):
      rows.append(
        {
          **base,
          "quantity": quantity,
          "normalize": normalize,
          "target_tau": float(target_tau[x_idx]),
          "x": float(x_val),
          "tau_true": float(tau_true[x_idx]),
          "u": u_val,
          "v": v_val,
          "truth": float(truth_grid[pair_idx, x_idx]),
          "pred": float(pred_grid[pair_idx, x_idx]),
          "error": float(
            pred_grid[pair_idx, x_idx] - truth_grid[pair_idx, x_idx]
          ),
        }
      )
  return rows


def _diagnostic_rows(
  cell: Cell,
  est: EstimatorSpec,
  seed: int,
  model: RosenblattBicop,
  pdf_by_norm: dict[str, torch.Tensor],
  grid: EvalGrid,
  *,
  enable_tau_diagnostics: bool,
  tau_diagnostic_n: int,
) -> tuple[list[dict[str, object]], float]:
  """Marginal-density and dependence diagnostics at fixed x."""
  base = _base_row(cell, est, seed)
  rows: list[dict[str, object]] = []
  tau_time = 0.0
  if not grid.conditional:
    for norm, pdf_hat in pdf_by_norm.items():
      c = pdf_hat.reshape((grid.u_axis.shape[0], grid.v_axis.shape[0]))
      diag = metrics.marginal_diagnostics(c, grid.u_axis, grid.v_axis)
      tau_true = float(_tau_values(cell.tau_scenario, None)[0])
      tau_hat = float("nan")
      if enable_tau_diagnostics:
        t0 = perf_counter()
        tau_hat = model.tau(n=tau_diagnostic_n)
        tau_time += perf_counter() - t0
      rows.append(
        {
          **base,
          "normalize": norm,
          "x": float("nan"),
          "tau_true": tau_true,
          "tau_hat": tau_hat,
          "tau_abs_err": abs(tau_hat - tau_true),
          **diag,
        }
      )
    return rows, tau_time

  assert grid.x_axis is not None
  tau_x = _tau_values(cell.tau_scenario, grid.x_axis)
  tau_hat_by_x: dict[int, float] = {}
  if enable_tau_diagnostics:
    for idx, x_val in enumerate(grid.x_axis):
      t0 = perf_counter()
      tau_hat_by_x[idx] = model.tau(
        x_row=torch.tensor([[float(x_val)]], dtype=grid.x_axis.dtype),
        n=tau_diagnostic_n,
      )
      tau_time += perf_counter() - t0
  for norm, pdf_hat in pdf_by_norm.items():
    pdf_grid = pdf_hat.reshape(grid.shape)
    for x_idx, x_val in enumerate(grid.x_axis):
      c = pdf_grid[:, x_idx].reshape(
        (grid.u_axis.shape[0], grid.v_axis.shape[0])
      )
      diag = metrics.marginal_diagnostics(c, grid.u_axis, grid.v_axis)
      tau_hat = tau_hat_by_x[x_idx] if enable_tau_diagnostics else float("nan")
      tau_true = float(tau_x[x_idx])
      rows.append(
        {
          **base,
          "normalize": norm,
          "x": float(x_val),
          "tau_true": tau_true,
          "tau_hat": tau_hat,
          "tau_abs_err": abs(tau_hat - tau_true),
          **diag,
        }
      )
  return rows, tau_time


class _Surface(NamedTuple):
  """The grid surfaces are stored on, with its truth and tau labels."""

  grid: EvalGrid
  truth: dict[str, torch.Tensor]
  target_tau: torch.Tensor
  tau_true: torch.Tensor


def _surface(cell: Cell, grid: GridConfig) -> _Surface | None:
  """The cell's surface grid, if its family stores surfaces and it has an x."""
  if (
    cell.family not in grid.surface_families
    or not scenarios.TAU_SCENARIOS[cell.tau_scenario].conditional
  ):
    return None

  rows = _surface_x_rows(cell.tau_scenario, grid.surface_tau_levels)
  surface_grid = scenarios.eval_grid_for_x(
    cell.tau_scenario,
    torch.tensor([row["x"] for row in rows], dtype=torch.float64),
    conditional_uv_grid_n=grid.conditional_uv_grid_n,
  )
  return _Surface(
    grid=surface_grid,
    truth=scenarios.ground_truth(cell.family, cell.tau_scenario, surface_grid),
    target_tau=torch.tensor(
      [row["target_tau"] for row in rows], dtype=torch.float64
    ),
    tau_true=torch.tensor(
      [row["tau_true"] for row in rows], dtype=torch.float64
    ),
  )


def _predict(
  model: RosenblattBicop, grid: EvalGrid, normalize: list[int | None]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
  """Every scored quantity on ``grid``, on the host, and what each took.

  The density is evaluated in one call for the whole ``normalize`` axis: the
  inner-backend passes a projection needs are the same for every iteration
  count, so sweeping ``sinkhorn_iters`` and calling ``pdf`` per value would
  re-pay for them.

  Returns
  -------
  hfuncs : dict
      ``hfunc1`` and ``hfunc2``.
  pdfs : dict
      The density under each ``normalize`` entry, keyed by its label.
  timings : dict
      Seconds per ``hfunc1``, ``hfunc2`` and ``pdf``.
  """
  hfuncs: dict[str, torch.Tensor] = {}
  timings: dict[str, float] = {}
  for name, fn in (("hfunc1", model.hfunc1), ("hfunc2", model.hfunc2)):
    t0 = perf_counter()
    hfuncs[name] = fn(grid.uv, x=grid.x_flat).cpu()
    timings[name] = perf_counter() - t0

  t0 = perf_counter()
  densities = model.pdf_by_projection(
    grid.uv, x=grid.x_flat, sinkhorn_iters=normalize
  )
  timings["pdf"] = perf_counter() - t0
  pdfs = {_norm_label(norm): densities[norm].cpu() for norm in normalize}
  return hfuncs, pdfs, timings


def summarize_one_cell(
  cell: Cell,
  grid: GridConfig,
  *,
  base_seed: int,
  device: str | None,
) -> tuple[Rows, Rows, Rows, Rows]:
  """Fit every estimator on one cell's data and return all output rows."""
  seed = _cell_seed(base_seed, cell)

  u, v, x = scenarios.sample(cell.family, cell.tau_scenario, cell.n, seed)
  metric_grid = scenarios.eval_grid(
    cell.tau_scenario,
    conditional_uv_grid_n=grid.conditional_uv_grid_n,
    conditional_x_grid_n=grid.conditional_x_grid_n,
  )
  truth = scenarios.ground_truth(cell.family, cell.tau_scenario, metric_grid)
  surface = _surface(cell, grid)

  metric_rows: Rows = []
  quantity_rows: Rows = []
  diagnostic_rows: Rows = []
  runtime_rows: Rows = []

  for est in grid.estimators:
    t0 = perf_counter()
    torch.manual_seed(derive_seed(seed, est.estimator_id))
    reset_gpu_peaks()
    model = RosenblattBicop(
      est.controls(
        FitControlsRosenblattBicop,
        device=device,
        projection_grid_size=grid.projection_grid_size,
        batch_size=grid.batch_size,
      )
    )
    model.fit(torch.column_stack([u, v]), x=x)
    fit_time = perf_counter() - t0

    with torch.inference_mode():
      hfuncs, pdf_by_norm, timings = _predict(
        model, metric_grid, grid.normalize
      )
      for q, pred in hfuncs.items():
        metric_rows += _metric_rows_for_quantity(
          cell,
          est,
          seed,
          q,
          truth[q],
          pred,
          metric_grid,
          normalize="none",
          include_kl=False,
        )
      for norm_label, pdf_hat in pdf_by_norm.items():
        metric_rows += _metric_rows_for_quantity(
          cell,
          est,
          seed,
          "pdf",
          truth["pdf"],
          pdf_hat,
          metric_grid,
          normalize=norm_label,
          include_kl=True,
        )

      diag_rows, tau_time = _diagnostic_rows(
        cell,
        est,
        seed,
        model,
        pdf_by_norm,
        metric_grid,
        enable_tau_diagnostics=grid.enable_tau_diagnostics,
        tau_diagnostic_n=grid.tau_diagnostic_n,
      )
      diagnostic_rows += diag_rows

      surface_time = 0.0
      if surface is not None:
        s_hfuncs, s_pdfs, s_timings = _predict(
          model, surface.grid, grid.normalize
        )
        surface_time = sum(s_timings.values())
        surface_preds = [
          *((q, "none", pred) for q, pred in s_hfuncs.items()),
          *(("pdf", label, pdf_hat) for label, pdf_hat in s_pdfs.items()),
        ]
        for q, norm_label, pred in surface_preds:
          quantity_rows += _quantity_rows(
            cell,
            est,
            seed,
            q,
            surface.truth[q],
            pred,
            surface.grid,
            normalize=norm_label,
            target_tau=surface.target_tau,
            tau_true=surface.tau_true,
          )

    memory = memory_usage()
    # Flushed per line, so this survives a hard machine crash and names the
    # last (culprit) estimator with its peak VRAM and the system-RAM headroom.
    logger.info(
      "estimator %s [%s/%s n=%d]: GPU peak %.0f MiB (alloc %.0f) | "
      "RSS %.0f MiB | MemAvailable %.0f MiB",
      est.label,
      cell.family,
      cell.tau_scenario,
      cell.n,
      memory["gpu_peak_reserved_mb"],
      memory["gpu_peak_alloc_mb"],
      memory["rss_mb"],
      memory["mem_available_mb"],
    )

    runtime_rows.append(
      {
        **_base_row(cell, est, seed),
        "fit_time": fit_time,
        "pdf_time": timings["pdf"],
        "h1_time": timings["hfunc1"],
        "h2_time": timings["hfunc2"],
        "tau_time": tau_time,
        "surface_time": surface_time,
        "total_estimator_time": (
          fit_time + tau_time + surface_time + sum(timings.values())
        ),
        **memory,
      }
    )

    # Drop the fitted model (and its GPU tensors) before the next estimator.
    del model
    release_gpu()

  release_gpu()
  return metric_rows, quantity_rows, diagnostic_rows, runtime_rows


def run_study(
  grid: GridConfig, run: RunConfig, *, resume: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
  """Sweep the full grid; return metric, quantity, diagnostic, runtime tables.

  Each cell's rows are checkpointed to ``run.out/cells/<key>/`` as they finish,
  so an interruption loses at most the cells still running. With
  ``resume=True``, cells already marked done are read back from their shards
  and skipped; the grid signature is checked first so a changed grid cannot
  silently reuse them.
  """
  logger.info(
    "Bivariate study: %d estimators x %d normalize per cell",
    len(grid.estimators),
    len(grid.normalize),
  )
  rows, wall = sweep(
    grid.cells(),
    lambda cell: summarize_one_cell(
      cell, grid, base_seed=run.base_seed, device=run.device
    ),
    tables=TABLES,
    run=run,
    signature=_grid_signature(grid, run),
    study="bivariate",
    resume=resume,
  )
  metric_rows, quantity_rows, diagnostic_rows, runtime_rows = rows

  sort_axes = [
    "family",
    "tau_scenario",
    "backend",
    "transform",
    "label",
    "n",
    "rep",
  ]
  metric_df = (
    pd.DataFrame(metric_rows)
    .sort_values([*sort_axes, "normalize", "quantity", "x"])
    .reset_index(drop=True)
  )
  quantity_df = pd.DataFrame(quantity_rows)
  if not quantity_df.empty:
    quantity_df = quantity_df.sort_values(
      [*sort_axes, "normalize", "quantity", "target_tau", "x", "u", "v"]
    ).reset_index(drop=True)
  diagnostic_df = (
    pd.DataFrame(diagnostic_rows)
    .sort_values([*sort_axes, "normalize", "x"])
    .reset_index(drop=True)
  )
  runtime_df = (
    pd.DataFrame(runtime_rows).sort_values(sort_axes).reset_index(drop=True)
  )
  return metric_df, quantity_df, diagnostic_df, runtime_df, wall


_METRIC_NAMES: tuple[str, ...] = ("IAE", "ISE", "KL")
_ESTIMATOR_AXES: tuple[str, ...] = (
  "family",
  "tau_scenario",
  "n",
  "transform",
  "backend",
  "label",
  "estimator_id",
)


def _runtime_summary(runtime_df: pd.DataFrame) -> pd.DataFrame:
  rt_group = list(_ESTIMATOR_AXES)
  summary = runtime_df.groupby(rt_group, as_index=False).agg(
    fit_time_mean=("fit_time", "mean"),
    fit_time_std=("fit_time", "std"),
    pdf_time_mean=("pdf_time", "mean"),
    h1_time_mean=("h1_time", "mean"),
    h2_time_mean=("h2_time", "mean"),
    tau_time_mean=("tau_time", "mean"),
    surface_time_mean=("surface_time", "mean"),
    total_estimator_time_mean=("total_estimator_time", "mean"),
  )
  return cast("pd.DataFrame", summary)


def _selection_summary(summary_over_x: pd.DataFrame) -> pd.DataFrame:
  ranked = summary_over_x[
    (summary_over_x["quantity"] == "pdf") & (summary_over_x["metric"] == "KL")
  ].copy()
  ranked = ranked.sort_values(
    ["rep_mean", "rep_median", "rep_p95", *_ESTIMATOR_AXES, "normalize"],
    na_position="last",
  ).reset_index(drop=True)
  ranked.insert(0, "rank", range(1, len(ranked) + 1))
  return cast("pd.DataFrame", ranked)


def _tau_summary(diagnostic_df: pd.DataFrame) -> pd.DataFrame:
  tau_df = diagnostic_df[diagnostic_df["normalize"] == "none"]
  group = [*_ESTIMATOR_AXES, "x", "tau_true"]
  return summary_stats(tau_df, group, "tau_abs_err").rename(
    columns={name: f"tau_abs_err_{name}" for name in SUMMARY_STATS}
  )


def _projection_summary(
  metric_df: pd.DataFrame, diagnostic_df: pd.DataFrame
) -> pd.DataFrame:
  pair_axes = [*_ESTIMATOR_AXES, "rep", "x", "tau_true"]
  pdf = metric_df[metric_df["quantity"] == "pdf"]
  metric_wide = pdf.pivot_table(
    index=pair_axes,
    columns="normalize",
    values=list(_METRIC_NAMES),
    aggfunc="mean",
    dropna=False,
  )
  if "none" not in metric_wide.columns.get_level_values(1):
    return pd.DataFrame()

  diag_metrics = [
    "row_mean_abs_err",
    "row_max_abs_err",
    "col_mean_abs_err",
    "col_max_abs_err",
  ]
  diag_wide = diagnostic_df.pivot_table(
    index=pair_axes,
    columns="normalize",
    values=diag_metrics,
    aggfunc="mean",
    dropna=False,
  )

  rows: list[dict[str, object]] = []
  projected = sorted(
    label
    for label in metric_wide.columns.get_level_values(1).unique()
    if label != "none"
  )
  for projected_label in projected:
    for idx in metric_wide.index:
      row = dict(zip(pair_axes, idx, strict=True))
      row["projected_normalize"] = projected_label
      for metric in _METRIC_NAMES:
        none_val = metric_wide.loc[idx, (metric, "none")]
        proj_val = metric_wide.loc[idx, (metric, projected_label)]
        row[f"{metric}_none"] = float(none_val)
        row[f"{metric}_projected"] = float(proj_val)
        row[f"{metric}_delta"] = float(proj_val - none_val)
      for metric in diag_metrics:
        none_val = diag_wide.loc[idx, (metric, "none")]
        proj_val = diag_wide.loc[idx, (metric, projected_label)]
        row[f"{metric}_none"] = float(none_val)
        row[f"{metric}_projected"] = float(proj_val)
        row[f"{metric}_delta"] = float(proj_val - none_val)
      rows.append(row)

  paired = pd.DataFrame(rows)
  if paired.empty:
    return paired

  group = [*_ESTIMATOR_AXES, "projected_normalize"]
  value_cols = [
    col
    for col in paired.columns
    if col.endswith(("_none", "_projected", "_delta"))
  ]
  means = paired.groupby(group, as_index=False, dropna=False)[value_cols].mean()
  return cast("pd.DataFrame", means)


def aggregate_study_outputs(
  metric_df: pd.DataFrame,
  diagnostic_df: pd.DataFrame,
  runtime_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
  """Build paper-facing study summaries from raw result tables."""
  summary_by_x, summary_over_x = summarize(
    metric_df,
    _METRIC_NAMES,
    [*_ESTIMATOR_AXES, "normalize", "quantity"],
    x_axes=("x", "tau_true"),
  )
  return {
    "metrics_by_x": metric_df,
    "summary_by_x": summary_by_x,
    "summary_over_x": summary_over_x,
    "selection_summary": _selection_summary(summary_over_x),
    "projection_summary": _projection_summary(metric_df, diagnostic_df),
    "tau_summary": _tau_summary(diagnostic_df),
    "runtime_summary": _runtime_summary(runtime_df),
  }
