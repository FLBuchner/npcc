"""Run the simulation study: sweep the grid, score, diagnose, aggregate.

Each data cell ``(family, tau_scenario, n, rep)`` samples its own training set
and ground truth once.  Estimator configs are fitted on that shared data.

Conditional scenarios are scored at fixed ``x`` values: errors are averaged over
the uv grid first, and only then summarized over Monte-Carlo repetitions or
conditioning values. This keeps KL as a fixed-``x`` copula-density metric.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import pandas as pd
import torch

from npcc.core.bicop import RosenblattBicop
from npcc.core.controls import FitControlsRosenblattBicop
from npcc.experiments import metrics, scenarios
from npcc.experiments.config import Cell, EstimatorSpec, GridConfig, RunConfig
from npcc.experiments.scenarios import EvalGrid

logger = logging.getLogger(__name__)

_EPS: float = 1e-12

# Avoid pandas/pyarrow Arrow-string inference crashes seen on Windows while
# building and sorting the large mixed string/numeric experiment tables.
pd.options.future.infer_string = False


def _norm_label(norm: int | None) -> str:
  return "none" if norm is None else str(norm)


def _data_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
  """Build a DataFrame without pandas' optional Arrow-backed string inference."""
  with pd.option_context("future.infer_string", False):
    return pd.DataFrame(rows)


def _system_mem_mb() -> tuple[float, float]:
  """(process RSS, system MemAvailable) in MiB from ``/proc``; (0, 0) if absent.

  A headless run that wedges the whole box over SSH is usually system-RAM
  exhaustion, not a GPU OOM — logging MemAvailable per estimator exposes a
  creeping leak before it kills the machine.
  """
  rss = avail = 0.0
  try:
    for line in Path("/proc/self/status").read_text().splitlines():
      if line.startswith("VmRSS:"):
        rss = float(line.split()[1]) / 1024.0
        break
    for line in Path("/proc/meminfo").read_text().splitlines():
      if line.startswith("MemAvailable:"):
        avail = float(line.split()[1]) / 1024.0
        break
  except OSError:
    pass
  return rss, avail


def _release_gpu() -> None:
  """Release cached CUDA memory between estimators/cells (no-op on CPU).

  The estimator loop reassigns ``model`` each iteration; without this, freed
  backend modules linger in the CUDA caching allocator and peak memory creeps
  up across the 20 estimators/cell (foundation-model and fine-tune backends
  are the heaviest). ``empty_cache`` only returns *unused* blocks to the
  driver, so it is safe under any worker count.
  """
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()


def _write_table(df: pd.DataFrame, base: Path, fmt: str) -> None:
  """Write ``df`` to ``base`` with the ``.parquet``/``.csv`` suffix for ``fmt``."""
  if fmt == "parquet":
    df.to_parquet(base.with_suffix(".parquet"), index=False)
  else:
    df.to_csv(base.with_suffix(".csv"), index=False)


def _read_table(base: Path, fmt: str) -> pd.DataFrame:
  """Read a table written by :func:`_write_table`; empty frame if absent."""
  path = base.with_suffix(".parquet" if fmt == "parquet" else ".csv")
  if not path.exists():
    return _data_frame([])
  with pd.option_context("future.infer_string", False):
    if fmt == "parquet":
      return pd.read_parquet(path)
    return pd.read_csv(path)


_SHARD_TABLES: tuple[str, ...] = (
  "metrics",
  "quantities",
  "diagnostics",
  "runtime",
)
_CellRows = tuple[
  list[dict[str, Any]],
  list[dict[str, Any]],
  list[dict[str, Any]],
  list[dict[str, Any]],
]


def _cell_key(cell: Cell) -> str:
  return f"{cell.family}__{cell.tau_scenario}__n{cell.n}__rep{cell.rep}"


def _cell_done(cells_root: Path, cell: Cell) -> bool:
  """True once a cell's shard is fully written (the ``DONE`` marker is last)."""
  return (cells_root / _cell_key(cell) / "DONE").exists()


def _write_cell_shard(
  cells_root: Path, cell: Cell, rows: _CellRows, fmt: str
) -> None:
  """Persist one cell's four row-lists, then a ``DONE`` marker written last.

  The marker-last ordering makes the shard atomic: a crash mid-write leaves no
  ``DONE``, so :func:`_cell_done` reports the cell as pending and it recomputes.
  """
  cell_dir = cells_root / _cell_key(cell)
  cell_dir.mkdir(parents=True, exist_ok=True)
  for name, cell_rows in zip(_SHARD_TABLES, rows, strict=True):
    if cell_rows:
      _write_table(_data_frame(cell_rows), cell_dir / name, fmt)
  (cell_dir / "DONE").write_text("")


def _read_cell_shard(cells_root: Path, cell: Cell, fmt: str) -> _CellRows:
  """Read back a cell's four row-lists written by :func:`_write_cell_shard`."""
  cell_dir = cells_root / _cell_key(cell)
  tables = [
    _read_table(cell_dir / name, fmt).to_dict("records")
    for name in _SHARD_TABLES
  ]
  return tables[0], tables[1], tables[2], tables[3]


def _grid_signature(grid: GridConfig, run: RunConfig) -> str:
  """Stable hash of everything that changes cell outputs (resume guard)."""
  payload = {
    "families": sorted(grid.families),
    "tau_scenarios": sorted(grid.tau_scenarios),
    "n": sorted(grid.n),
    "n_rep": grid.n_rep,
    "normalize": sorted(_norm_label(x) for x in grid.normalize),
    "projection_grid_size": grid.projection_grid_size,
    "conditional_uv_grid_n": grid.conditional_uv_grid_n,
    "conditional_x_grid_n": grid.conditional_x_grid_n,
    "surface_tau_levels": sorted(grid.surface_tau_levels),
    "surface_families": sorted(grid.surface_families),
    "enable_tau_diagnostics": grid.enable_tau_diagnostics,
    "tau_diagnostic_n": grid.tau_diagnostic_n,
    "base_seed": run.base_seed,
    "estimators": sorted(s.estimator_id for s in grid.estimator_specs()),
  }
  encoded = json.dumps(payload, sort_keys=True).encode()
  return hashlib.sha256(encoded).hexdigest()


def _guard_manifest(
  out: Path, grid: GridConfig, run: RunConfig, resume: bool
) -> None:
  """Write (or, on resume, verify) the grid-signature manifest."""
  path = out / "manifest.json"
  signature = _grid_signature(grid, run)
  if resume and path.exists():
    previous = json.loads(path.read_text()).get("signature")
    if previous != signature:
      raise ValueError(
        "Grid signature changed since the checkpoint in this output "
        "directory; use a fresh --out or drop --resume."
      )
  path.write_text(json.dumps({"signature": signature}, indent=2))


def _nan_quantile(s: pd.Series, q: float) -> float:
  values = torch.tensor(s.to_numpy(), dtype=torch.float64)
  if torch.isnan(values).all():
    return float("nan")
  return float(torch.nanquantile(values, q))


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


def _estimator_seed(cell_seed: int, estimator_id: str) -> int:
  """Deterministic per-estimator seed mixing the cell seed and estimator id.

  Decorrelates stochastic fits across estimators without feeding back into
  ``estimator_id`` (which must stay a pure function of the config).
  """
  digest = hashlib.sha256(f"{cell_seed}:{estimator_id}".encode()).digest()
  return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


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
  base = _base_row(cell, est, seed)
  base.update({"normalize": normalize, "quantity": quantity})

  if grid.conditional:
    assert grid.x_axis is not None
    tau_x = _tau_values(cell.tau_scenario, grid.x_axis)
    truth_grid = truth.reshape(grid.shape)
    pred_grid = pred.reshape(grid.shape)
    rows: list[dict[str, object]] = []
    for x_idx, x_val in enumerate(grid.x_axis):
      y_true = truth_grid[:, x_idx]
      y_hat = pred_grid[:, x_idx]
      err = y_hat - y_true
      kl = float("nan")
      if include_kl:
        y_true_pos = y_true.clamp_min(_EPS)
        y_hat_pos = y_hat.clamp_min(_EPS)
        kl = float((y_true_pos * torch.log(y_true_pos / y_hat_pos)).mean())
      rows.append(
        {
          **base,
          "x": float(x_val),
          "tau_true": float(tau_x[x_idx]),
          "IAE": float(err.abs().mean()),
          "ISE": float(err.square().mean()),
          "KL": kl,
        }
      )
    return rows

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


def summarize_one_cell(
  cell: Cell,
  estimator_specs: list[EstimatorSpec],
  normalize: list[int | None],
  *,
  base_seed: int,
  device: str | None,
  projection_grid_size: int,
  conditional_uv_grid_n: int,
  conditional_x_grid_n: int,
  surface_tau_levels: list[float],
  surface_families: list[str],
  enable_tau_diagnostics: bool,
  tau_diagnostic_n: int,
) -> tuple[
  list[dict[str, Any]],
  list[dict[str, Any]],
  list[dict[str, Any]],
  list[dict[str, Any]],
]:
  """Fit every estimator on one cell's data and return all output rows."""
  seed = _cell_seed(base_seed, cell)

  u, v, x = scenarios.sample(cell.family, cell.tau_scenario, cell.n, seed)
  metric_grid = scenarios.eval_grid(
    cell.tau_scenario,
    conditional_uv_grid_n=conditional_uv_grid_n,
    conditional_x_grid_n=conditional_x_grid_n,
  )
  truth = scenarios.ground_truth(cell.family, cell.tau_scenario, metric_grid)

  surface_rows = _surface_x_rows(cell.tau_scenario, surface_tau_levels)
  surface_grid: EvalGrid | None = None
  surface_truth: dict[str, torch.Tensor] | None = None
  surface_target_tau: torch.Tensor | None = None
  surface_tau_true: torch.Tensor | None = None
  if cell.family in surface_families and metric_grid.conditional:
    surface_x = torch.tensor(
      [row["x"] for row in surface_rows], dtype=torch.float64
    )
    surface_grid = scenarios.eval_grid_for_x(
      cell.tau_scenario, surface_x, conditional_uv_grid_n=conditional_uv_grid_n
    )
    surface_truth = scenarios.ground_truth(
      cell.family, cell.tau_scenario, surface_grid
    )
    surface_target_tau = torch.tensor(
      [row["target_tau"] for row in surface_rows], dtype=torch.float64
    )
    surface_tau_true = torch.tensor(
      [row["tau_true"] for row in surface_rows], dtype=torch.float64
    )

  metric_rows: list[dict[str, Any]] = []
  quantity_rows: list[dict[str, Any]] = []
  diagnostic_rows: list[dict[str, Any]] = []
  runtime_rows: list[dict[str, Any]] = []

  for est in estimator_specs:
    t0 = perf_counter()
    est_seed = _estimator_seed(seed, est.estimator_id)
    torch.manual_seed(est_seed)
    if torch.cuda.is_available():
      torch.cuda.reset_peak_memory_stats()
    model = RosenblattBicop(
      FitControlsRosenblattBicop(
        backend=est.backend,
        transform=cast("Literal['identity', 'logit', 'probit']", est.transform),
        device=device,
        projection_grid_size=projection_grid_size,
        backend_kwargs=dict(est.backend_kwargs),
      )
    )
    model.fit(torch.column_stack([u, v]), x=x)
    fit_time = perf_counter() - t0

    with torch.inference_mode():
      timings: dict[str, float] = {}
      single_preds: dict[str, torch.Tensor] = {}
      for q, fn in (
        (
          "hfunc1",
          lambda m=model: m.hfunc1(
            torch.column_stack([metric_grid.u_flat, metric_grid.v_flat]),
            x=metric_grid.x_flat,
          ),
        ),
        (
          "hfunc2",
          lambda m=model: m.hfunc2(
            torch.column_stack([metric_grid.u_flat, metric_grid.v_flat]),
            x=metric_grid.x_flat,
          ),
        ),
      ):
        t0 = perf_counter()
        single_preds[q] = fn().cpu()
        timings[q] = perf_counter() - t0

      for q in ("hfunc1", "hfunc2"):
        metric_rows += _metric_rows_for_quantity(
          cell,
          est,
          seed,
          q,
          truth[q],
          single_preds[q],
          metric_grid,
          normalize="none",
          include_kl=False,
        )

      pdf_time = 0.0
      pdf_by_norm: dict[str, torch.Tensor] = {}
      for norm in normalize:
        norm_label = _norm_label(norm)
        t0 = perf_counter()
        # Written onto the fitted model rather than passed per call: the
        # projection count is a control, and the sweep reads one fit under
        # several of them rather than refitting per value.
        model.sinkhorn_iters = norm
        pdf_hat = model.pdf(
          torch.column_stack([metric_grid.u_flat, metric_grid.v_flat]),
          x=metric_grid.x_flat,
        ).cpu()
        pdf_time += perf_counter() - t0
        pdf_by_norm[norm_label] = pdf_hat
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
        enable_tau_diagnostics=enable_tau_diagnostics,
        tau_diagnostic_n=tau_diagnostic_n,
      )
      diagnostic_rows += diag_rows

      surface_time = 0.0
      if (
        surface_grid is not None
        and surface_truth is not None
        and surface_target_tau is not None
        and surface_tau_true is not None
      ):
        surface_preds: dict[str, torch.Tensor] = {}
        for q, fn in (
          (
            "hfunc1",
            lambda m=model: m.hfunc1(
              torch.column_stack([surface_grid.u_flat, surface_grid.v_flat]),
              x=surface_grid.x_flat,
            ),
          ),
          (
            "hfunc2",
            lambda m=model: m.hfunc2(
              torch.column_stack([surface_grid.u_flat, surface_grid.v_flat]),
              x=surface_grid.x_flat,
            ),
          ),
        ):
          t0 = perf_counter()
          surface_preds[q] = fn().cpu()
          surface_time += perf_counter() - t0
          quantity_rows += _quantity_rows(
            cell,
            est,
            seed,
            q,
            surface_truth[q],
            surface_preds[q],
            surface_grid,
            normalize="none",
            target_tau=surface_target_tau,
            tau_true=surface_tau_true,
          )

        for norm in normalize:
          norm_label = _norm_label(norm)
          t0 = perf_counter()
          # As above: the projection count is a control, swept over one fit.
          model.sinkhorn_iters = norm
          pdf_hat = model.pdf(
            torch.column_stack([surface_grid.u_flat, surface_grid.v_flat]),
            x=surface_grid.x_flat,
          ).cpu()
          surface_time += perf_counter() - t0
          quantity_rows += _quantity_rows(
            cell,
            est,
            seed,
            "pdf",
            surface_truth["pdf"],
            pdf_hat,
            surface_grid,
            normalize=norm_label,
            target_tau=surface_target_tau,
            tau_true=surface_tau_true,
          )

    gpu_peak_reserved_mb = 0.0
    gpu_peak_alloc_mb = 0.0
    if torch.cuda.is_available():
      gpu_peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
      gpu_peak_alloc_mb = torch.cuda.max_memory_allocated() / 1024**2
    rss_mb, mem_available_mb = _system_mem_mb()
    # Flushed per line, so this survives a hard machine crash and names the
    # last (culprit) estimator with its peak VRAM and the system-RAM headroom.
    logger.info(
      "estimator %s [%s/%s n=%d]: GPU peak %.0f MiB (alloc %.0f) | "
      "RSS %.0f MiB | MemAvailable %.0f MiB",
      est.label,
      cell.family,
      cell.tau_scenario,
      cell.n,
      gpu_peak_reserved_mb,
      gpu_peak_alloc_mb,
      rss_mb,
      mem_available_mb,
    )

    runtime_rows.append(
      {
        "family": cell.family,
        "tau_scenario": cell.tau_scenario,
        "n": cell.n,
        "rep": cell.rep,
        "seed": seed,
        "label": est.label,
        "estimator_id": est.estimator_id,
        "transform": est.transform,
        "backend": est.backend,
        "fit_time": fit_time,
        "pdf_time": pdf_time,
        "h1_time": timings["hfunc1"],
        "h2_time": timings["hfunc2"],
        "tau_time": tau_time,
        "surface_time": surface_time,
        "total_estimator_time": (
          fit_time + pdf_time + tau_time + surface_time + sum(timings.values())
        ),
        "gpu_peak_reserved_mb": gpu_peak_reserved_mb,
        "gpu_peak_alloc_mb": gpu_peak_alloc_mb,
        "rss_mb": rss_mb,
        "mem_available_mb": mem_available_mb,
      }
    )

    # Drop the fitted model (and its GPU tensors) before the next estimator.
    del model
    _release_gpu()

  _release_gpu()
  return metric_rows, quantity_rows, diagnostic_rows, runtime_rows


def run_study(
  grid: GridConfig, run: RunConfig, *, resume: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
  """Sweep the full grid; return metric, quantity, diagnostic, runtime tables.

  Each cell's rows are checkpointed to ``run.out/cells/<key>/`` as they finish,
  so an interruption loses at most the in-flight cell. With ``resume=True``,
  cells already marked done are read back from their shards and skipped; the
  grid signature is checked first so a changed grid cannot silently reuse them.
  """
  cells = grid.cells()
  specs = grid.estimator_specs()
  cells_root = run.out / "cells"
  cells_root.mkdir(parents=True, exist_ok=True)
  _guard_manifest(run.out, grid, run, resume)

  if run.gpu_mem_fraction is not None and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(run.gpu_mem_fraction)
    logger.info(
      "Capped CUDA memory to %.2f of total (headroom for the display server)",
      run.gpu_mem_fraction,
    )

  done_set = {c for c in cells if resume and _cell_done(cells_root, c)}
  pending = [c for c in cells if c not in done_set]
  logger.info(
    "Study: %d cells (%d pending, %d resumed) x %d estimators x %d normalize",
    len(cells),
    len(pending),
    len(done_set),
    len(specs),
    len(grid.normalize),
  )

  metric_rows: list[dict[str, Any]] = []
  quantity_rows: list[dict[str, Any]] = []
  diagnostic_rows: list[dict[str, Any]] = []
  runtime_rows: list[dict[str, Any]] = []

  def _accumulate(rows: _CellRows) -> None:
    metric_rows.extend(rows[0])
    quantity_rows.extend(rows[1])
    diagnostic_rows.extend(rows[2])
    runtime_rows.extend(rows[3])

  for cell in cells:
    if cell in done_set:
      _accumulate(_read_cell_shard(cells_root, cell, run.fmt))

  t0_wall = perf_counter()

  def _do(cell: Cell) -> _CellRows:
    logger.debug("cell start: %s", cell)
    out = summarize_one_cell(
      cell,
      specs,
      grid.normalize,
      base_seed=run.base_seed,
      device=run.device,
      projection_grid_size=grid.projection_grid_size,
      conditional_uv_grid_n=grid.conditional_uv_grid_n,
      conditional_x_grid_n=grid.conditional_x_grid_n,
      surface_tau_levels=grid.surface_tau_levels,
      surface_families=grid.surface_families,
      enable_tau_diagnostics=grid.enable_tau_diagnostics,
      tau_diagnostic_n=grid.tau_diagnostic_n,
    )
    _write_cell_shard(cells_root, cell, out, run.fmt)
    logger.info("cell done: %s", cell)
    return out

  if run.workers <= 1:
    for cell in pending:
      _accumulate(_do(cell))
  elif pending:
    max_workers = min(run.workers, len(pending), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
      futures = [pool.submit(_do, cell) for cell in pending]
      for fut in as_completed(futures):
        _accumulate(fut.result())

  wall = perf_counter() - t0_wall
  logger.info("Study finished in %.1fs", wall)

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
    _data_frame(metric_rows)
    .sort_values([*sort_axes, "normalize", "quantity", "x"])
    .reset_index(drop=True)
  )
  quantity_df = _data_frame(quantity_rows)
  if not quantity_df.empty:
    quantity_df = quantity_df.sort_values(
      [*sort_axes, "normalize", "quantity", "target_tau", "x", "u", "v"]
    ).reset_index(drop=True)
  diagnostic_df = (
    _data_frame(diagnostic_rows)
    .sort_values([*sort_axes, "normalize", "x"])
    .reset_index(drop=True)
  )
  runtime_df = (
    _data_frame(runtime_rows).sort_values(sort_axes).reset_index(drop=True)
  )
  return metric_df, quantity_df, diagnostic_df, runtime_df, wall


_SUMMARY_STATS: tuple[str, ...] = (
  "rep_mean",
  "rep_std",
  "rep_median",
  "rep_p05",
  "rep_p95",
  "rep_max",
)
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


def _summary_stats(
  df: pd.DataFrame, group: list[str], value_col: str
) -> pd.DataFrame:
  stats = df.groupby(group, as_index=False, dropna=False).agg(
    rep_mean=(value_col, "mean"),
    rep_std=(value_col, "std"),
    rep_median=(value_col, "median"),
    rep_p05=(value_col, lambda s: _nan_quantile(s, 0.05)),
    rep_p95=(value_col, lambda s: _nan_quantile(s, 0.95)),
    rep_max=(value_col, "max"),
  )
  return cast("pd.DataFrame", stats)


def _metric_summary(metric_df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
  chunks: list[pd.DataFrame] = []
  for metric in _METRIC_NAMES:
    g = _summary_stats(metric_df, group, metric)
    g["metric"] = metric
    chunks.append(g)
  return pd.concat(chunks, ignore_index=True)


def _summary_over_x(metric_df: pd.DataFrame) -> pd.DataFrame:
  group = [
    *_ESTIMATOR_AXES,
    "normalize",
    "quantity",
  ]
  rep_group = [*group, "rep"]
  chunks: list[pd.DataFrame] = []
  for metric in _METRIC_NAMES:
    averaged = metric_df.groupby(rep_group, as_index=False, dropna=False).agg(
      value=(metric, "mean")
    )
    g = _summary_stats(averaged, group, "value")
    g["metric"] = metric
    chunks.append(g)
  return pd.concat(chunks, ignore_index=True)


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
  return _summary_stats(tau_df, group, "tau_abs_err").rename(
    columns={name: f"tau_abs_err_{name}" for name in _SUMMARY_STATS}
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
    return _data_frame([])

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

  paired = _data_frame(rows)
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


def aggregate_results(
  metric_df: pd.DataFrame, runtime_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Aggregate fixed-x metrics over repetitions and summarize runtimes."""
  group = [
    *_ESTIMATOR_AXES,
    "normalize",
    "quantity",
    "x",
    "tau_true",
  ]
  return _metric_summary(metric_df, group), _runtime_summary(runtime_df)


def aggregate_study_outputs(
  metric_df: pd.DataFrame,
  diagnostic_df: pd.DataFrame,
  runtime_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
  """Build paper-facing study summaries from raw result tables."""
  summary_by_x, runtime_summary = aggregate_results(metric_df, runtime_df)
  summary_over_x = _summary_over_x(metric_df)
  return {
    "metrics_by_x": metric_df,
    "summary_by_x": summary_by_x,
    "summary_over_x": summary_over_x,
    "selection_summary": _selection_summary(summary_over_x),
    "projection_summary": _projection_summary(metric_df, diagnostic_df),
    "tau_summary": _tau_summary(diagnostic_df),
    "runtime_summary": runtime_summary,
  }
