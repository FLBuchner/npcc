"""Run the vine study: draw truths, fit each arm, score, aggregate.

A truth, its random-structure competitor and its test set are keyed on
``(d, rep)`` alone, so every sample size ``n`` in a repetition faces the same
copula and the same test points and the curve over ``n`` is comparable.

Every arm estimates the joint distribution on the data scale, margins and
copula together: the NPCC arms as a :class:`~npcc.core.vinedist.
RosenblattVinedist` whose margins and pairs all see ``x``, and ``tll`` as a
:class:`pyvinecopulib.Vinedist` with kernel-density margins and TLL pairs on
the true structure, neither of which sees ``x``.

**Scoring.** At each evaluation ``x``, ``m`` points are drawn from the true
distribution there, and the estimate is scored by Monte-Carlo integrals under
the truth: ``KL = E[log f - log f_hat]``, ``L1 = E[|f_hat / f - 1|]`` (the
integrated absolute error) and ``ISE = E[(f_hat - f)^2 / f]`` (the integrated
squared error). Each is a fixed-``x`` metric, summarized over repetitions
afterward.

``KL_copula`` and ``L1_copula`` score the fitted vine copula alone, at the
true probability integral transforms, which separates a copula error from a
margin error. There is no copula-scale ISE: its Monte-Carlo estimate divides
by ``c``, and a copula density's corner peaks make it too heavy-tailed to
read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pyvinecopulib as pv
import torch
from pyvinecopulib.core import (
  ConditioningContext,
  NonSimplifiedContext,
  SimplifiedContext,
)

from npcc.core.controls import FitControlsRosenblattVinecop
from npcc.core.margin_quantile_table import QuantileTableConfig
from npcc.core.registry import create_backend
from npcc.core.vinecop import RosenblattVinecop
from npcc.core.vinedist import RosenblattVinedist
from npcc.experiments import scenarios, vine_scenarios
from npcc.experiments.config import EstimatorSpec, RunConfig
from npcc.experiments.runner import (
  _data_frame,
  _estimator_seed,
  _read_table,
  _release_gpu,
  _summary_stats,
  _system_mem_mb,
  _write_table,
)
from npcc.experiments.vine_config import VineCell, VineGridConfig
from npcc.experiments.vine_scenarios import EvalPoints, VineTruth

logger = logging.getLogger(__name__)

_EPS: float = 1e-300

METRIC_NAMES: tuple[str, ...] = ("KL", "L1", "ISE", "KL_copula", "L1_copula")
_SHARD_TABLES: tuple[str, ...] = ("metrics", "runtime")
_CellRows = tuple[list[dict[str, Any]], list[dict[str, Any]]]
_GROUP_AXES: tuple[str, ...] = (
  "d",
  "n",
  "arm",
  "label",
  "estimator_id",
  "backend",
  "transform",
)


def _seed(base_seed: int, *parts: object) -> int:
  """Deterministic 31-bit seed from ``base_seed`` and a salt."""
  payload = ":".join(str(p) for p in (base_seed, *parts)).encode()
  return (
    int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF
  )


def truth_for(base_seed: int, d: int, rep: int) -> VineTruth:
  """The true vine of repetition ``rep`` in dimension ``d``."""
  return vine_scenarios.draw_truth(d, _seed(base_seed, "truth", d, rep))


def random_structure_for(base_seed: int, d: int, rep: int) -> pv.RVineStructure:
  """The ``random`` arm's structure, drawn independently of the truth's."""
  return pv.RVineStructure.sample(d, seeds=[_seed(base_seed, "random", d, rep)])


def _cell_key(cell: VineCell) -> str:
  return f"d{cell.d}__n{cell.n}__rep{cell.rep}"


def _cell_done(cells_root: Path, cell: VineCell) -> bool:
  """True once a cell's shard is fully written (the ``DONE`` marker is last)."""
  return (cells_root / _cell_key(cell) / "DONE").exists()


def _write_cell_shard(
  cells_root: Path, cell: VineCell, rows: _CellRows, fmt: str
) -> None:
  """Persist one cell's row-lists, then a ``DONE`` marker written last."""
  cell_dir = cells_root / _cell_key(cell)
  cell_dir.mkdir(parents=True, exist_ok=True)
  for name, cell_rows in zip(_SHARD_TABLES, rows, strict=True):
    if cell_rows:
      _write_table(_data_frame(cell_rows), cell_dir / name, fmt)
  (cell_dir / "DONE").write_text("")


def _read_cell_shard(cells_root: Path, cell: VineCell, fmt: str) -> _CellRows:
  cell_dir = cells_root / _cell_key(cell)
  metrics = _read_table(cell_dir / "metrics", fmt).to_dict("records")
  runtime = _read_table(cell_dir / "runtime", fmt).to_dict("records")
  return metrics, runtime


def grid_signature(grid: VineGridConfig, run: RunConfig) -> str:
  """Stable hash of everything that changes cell outputs (resume guard)."""
  payload = {
    "study": "vine",
    # The truth's constants: a resumed cell must have been drawn from the same
    # data-generating process, not merely the same grid.
    "vine_tau_hi": vine_scenarios.VINE_TAU_HI,
    "tau_lo": scenarios.TAU_LO,
    "mu_slope": vine_scenarios.MU_SLOPE,
    "dims": sorted(grid.dims),
    "n": sorted(grid.n),
    "n_rep": grid.n_rep,
    "arms": sorted(grid.arms),
    "eval_x_n": grid.eval_x_n,
    "eval_m": grid.eval_m,
    "projection_grid_size": grid.projection_grid_size,
    # `batch_size` chunks the query set without changing any prediction's
    # context, so it changes throughput and not results.
    "base_seed": run.base_seed,
    "estimators": sorted(s.estimator_id for s in grid.estimators),
  }
  encoded = json.dumps(payload, sort_keys=True).encode()
  return hashlib.sha256(encoded).hexdigest()


def _guard_manifest(
  out: Path, grid: VineGridConfig, run: RunConfig, resume: bool
) -> None:
  """Write (or, on resume, verify) the grid-signature manifest."""
  path = out / "manifest.json"
  signature = grid_signature(grid, run)
  if resume and path.exists():
    previous = json.loads(path.read_text()).get("signature")
    if previous != signature:
      raise ValueError(
        "the vine grid must match the checkpoint in this output directory to "
        "resume; use a fresh --out or drop --resume"
      )
  path.write_text(json.dumps({"signature": signature}, indent=2))


def score(pdf_true: torch.Tensor, pdf_hat: torch.Tensor) -> dict[str, float]:
  """Monte-Carlo ``KL`` / ``L1`` / ``ISE`` from points drawn under the truth."""
  p = pdf_true.to(torch.float64).clamp_min(_EPS)
  q = pdf_hat.to(torch.float64).clamp_min(_EPS)
  return {
    "KL": float((p.log() - q.log()).mean()),
    "L1": float((q / p - 1.0).abs().mean()),
    "ISE": float(((q - p).square() / p).mean()),
    "loglik_true": float(p.log().mean()),
    "loglik_hat": float(q.log().mean()),
  }


def _score_both(
  points: EvalPoints,
  k: int,
  pdf_hat: torch.Tensor,
  copula_pdf_hat: torch.Tensor,
) -> dict[str, float]:
  joint = score(points.pdf_true[k], pdf_hat)
  copula = score(points.copula_pdf_true[k], copula_pdf_hat)
  return {**joint, "KL_copula": copula["KL"], "L1_copula": copula["L1"]}


def _gpu_peaks() -> tuple[float, float]:
  if not torch.cuda.is_available():
    return 0.0, 0.0
  return (
    torch.cuda.max_memory_reserved() / 1024**2,
    torch.cuda.max_memory_allocated() / 1024**2,
  )


def _fit_npcc(
  arm: str,
  est: EstimatorSpec,
  structure: pv.RVineStructure,
  y: torch.Tensor,
  x: torch.Tensor,
  *,
  device: str | None,
  projection_grid_size: int,
  batch_size: int | None,
) -> tuple[RosenblattVinedist, RosenblattVinecop]:
  """Fit one NPCC arm; return the distribution and its (fitted) copula."""
  context: ConditioningContext[torch.Tensor] = (
    SimplifiedContext() if arm == "oracle" else NonSimplifiedContext()
  )
  controls = FitControlsRosenblattVinecop(
    backend=est.backend,
    transform=cast("Literal['identity', 'logit', 'probit']", est.transform),
    device=device,
    projection_grid_size=projection_grid_size,
    batch_size=batch_size,
    backend_kwargs=dict(est.backend_kwargs),
  )
  # The margins run the pairs' backend on the data scale, hence "identity";
  # the estimator's own transform is for copula-scale responses.
  margins = [
    create_backend(
      controls.backend,
      transform="identity",
      quantile_table_config=controls.quantile_table_config
      or QuantileTableConfig(),
      eps=controls.eps,
      device=controls.device,
      batch_size=controls.batch_size,
      backend_kwargs=controls.backend_kwargs,
    )
    for _ in range(structure.dim)
  ]
  vine = RosenblattVinecop(None, structure, device=device, context=context)
  return RosenblattVinedist(vine, margins).fit(y, controls, x=x), vine


def _metric_rows(
  base: Mapping[str, object],
  points: EvalPoints,
  preds: list[torch.Tensor],
  copula_preds: list[torch.Tensor],
) -> list[dict[str, Any]]:
  return [
    {
      **base,
      "x": float(x_val),
      **_score_both(points, k, preds[k], copula_preds[k]),
    }
    for k, x_val in enumerate(points.x_axis)
  ]


def summarize_one_cell(
  cell: VineCell,
  grid: VineGridConfig,
  *,
  base_seed: int,
  device: str | None,
) -> _CellRows:
  """Fit every arm on one cell's data and return its metric and runtime rows."""
  truth = truth_for(base_seed, cell.d, cell.rep)
  y, _, x = vine_scenarios.sample(
    truth, cell.n, _seed(base_seed, "train", cell.d, cell.n, cell.rep)
  )
  points = vine_scenarios.eval_points(
    truth,
    scenarios.conditional_x_axis(grid.eval_x_n),
    grid.eval_m,
    _seed(base_seed, "eval", cell.d, cell.rep),
  )
  structures = {
    "oracle": truth.structure,
    "random": random_structure_for(base_seed, cell.d, cell.rep),
  }
  cell_seed = _seed(base_seed, "cell", cell.d, cell.n, cell.rep)
  cell_base = {"d": cell.d, "n": cell.n, "rep": cell.rep, "seed": cell_seed}

  metric_rows: list[dict[str, Any]] = []
  runtime_rows: list[dict[str, Any]] = []

  def _record(
    base: Mapping[str, object], fit_time: float, pdf_time: float
  ) -> None:
    reserved, alloc = _gpu_peaks()
    rss_mb, mem_available_mb = _system_mem_mb()
    logger.info(
      "arm %s / %s [d=%d n=%d rep=%d]: fit %.1fs | GPU peak %.0f MiB | "
      "RSS %.0f MiB | MemAvailable %.0f MiB",
      base["arm"],
      base["label"],
      cell.d,
      cell.n,
      cell.rep,
      fit_time,
      reserved,
      rss_mb,
      mem_available_mb,
    )
    runtime_rows.append(
      {
        **base,
        "fit_time": fit_time,
        "pdf_time": pdf_time,
        "total_time": fit_time + pdf_time,
        "gpu_peak_reserved_mb": reserved,
        "gpu_peak_alloc_mb": alloc,
        "rss_mb": rss_mb,
        "mem_available_mb": mem_available_mb,
      }
    )

  for arm in (a for a in grid.arms if a != "tll"):
    for est in grid.estimators:
      base = {
        **cell_base,
        "arm": arm,
        "label": est.label,
        "estimator_id": est.estimator_id,
        "backend": est.backend,
        "transform": est.transform,
      }
      torch.manual_seed(_estimator_seed(cell_seed, f"{arm}:{est.estimator_id}"))
      if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
      t0 = perf_counter()
      dist, vine = _fit_npcc(
        arm,
        est,
        structures[arm],
        y,
        x,
        device=device,
        projection_grid_size=grid.projection_grid_size,
        batch_size=grid.batch_size,
      )
      fit_time = perf_counter() - t0
      t0 = perf_counter()
      # One call per scale over every evaluation x at once: each call pays a
      # fixed inference cost per backend pass, so a call per x repays it.
      x_all = points.x_axis.repeat_interleave(grid.eval_m).reshape(-1, 1)
      with torch.inference_mode():
        preds = list(
          dist.pdf(torch.cat(points.y), x=x_all).cpu().split(grid.eval_m)
        )
        copula_preds = list(
          vine.pdf(torch.cat(points.u), x=x_all).cpu().split(grid.eval_m)
        )
      pdf_time = perf_counter() - t0
      metric_rows += _metric_rows(base, points, preds, copula_preds)
      _record(base, fit_time, pdf_time)
      del dist, vine
      _release_gpu()

  if "tll" in grid.arms:
    base = {
      **cell_base,
      "arm": "tll",
      "label": "kde-tll",
      "estimator_id": "",
      "backend": "pyvinecopulib",
      "transform": "none",
    }
    t0 = perf_counter()
    tll = pv.Vinedist.from_data(
      y.numpy(),
      pv.FitControlsVinecop(family_set=[pv.BicopFamily.tll]),
      structure=truth.structure,
    )
    fit_time = perf_counter() - t0
    t0 = perf_counter()
    preds = [
      torch.from_numpy(np.asarray(tll.pdf(pts.numpy()))) for pts in points.y
    ]
    copula_preds = [
      torch.from_numpy(np.asarray(tll.vinecop.pdf(pts.numpy())))
      for pts in points.u
    ]
    pdf_time = perf_counter() - t0
    metric_rows += _metric_rows(base, points, preds, copula_preds)
    _record(base, fit_time, pdf_time)

  return metric_rows, runtime_rows


def run_vine_study(
  grid: VineGridConfig, run: RunConfig, *, resume: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
  """Sweep the vine grid; return the metric and runtime tables and wall time.

  Each cell's rows are checkpointed to ``run.out/cells/<key>/`` as they
  finish. With ``resume=True``, cells already marked done are read back and
  skipped, after the grid signature is checked against the manifest.
  """
  cells = grid.cells()
  cells_root = run.out / "cells"
  cells_root.mkdir(parents=True, exist_ok=True)
  _guard_manifest(run.out, grid, run, resume)

  if run.gpu_mem_fraction is not None and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(run.gpu_mem_fraction)

  done = {c for c in cells if resume and _cell_done(cells_root, c)}
  pending = [c for c in cells if c not in done]
  logger.info(
    "Vine study: %d cells (%d pending, %d resumed) x %d arms x %d estimators",
    len(cells),
    len(pending),
    len(done),
    len(grid.arms),
    len(grid.estimators),
  )

  metric_rows: list[dict[str, Any]] = []
  runtime_rows: list[dict[str, Any]] = []

  def _accumulate(rows: _CellRows) -> None:
    metric_rows.extend(rows[0])
    runtime_rows.extend(rows[1])

  for cell in cells:
    if cell in done:
      _accumulate(_read_cell_shard(cells_root, cell, run.fmt))

  t0_wall = perf_counter()

  def _do(cell: VineCell) -> _CellRows:
    out = summarize_one_cell(
      cell, grid, base_seed=run.base_seed, device=run.device
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
  logger.info("Vine study finished in %.1fs", wall)

  sort_axes = ["d", "n", "arm", "label", "rep"]
  metric_df = (
    _data_frame(metric_rows)
    .sort_values([*sort_axes, "x"])
    .reset_index(drop=True)
  )
  runtime_df = (
    _data_frame(runtime_rows).sort_values(sort_axes).reset_index(drop=True)
  )
  return metric_df, runtime_df, wall


def truths(grid: VineGridConfig, base_seed: int) -> list[dict[str, object]]:
  """Every ``(d, rep)`` truth, with the ``random`` arm's structure beside it."""
  out: list[dict[str, object]] = []
  for d in grid.dims:
    for rep in range(grid.n_rep):
      record = truth_for(base_seed, d, rep).to_dict()
      random_matrix = random_structure_for(base_seed, d, rep).matrix
      record.update(
        {"rep": rep, "random_matrix": np.asarray(random_matrix).tolist()}
      )
      out.append(record)
  return out


def aggregate_vine_outputs(
  metric_df: pd.DataFrame, runtime_df: pd.DataFrame
) -> dict[str, pd.DataFrame]:
  """Summaries over repetitions, per ``x`` and averaged over ``x``."""
  by_x_group = [*_GROUP_AXES, "x"]
  by_x: list[pd.DataFrame] = []
  over_x: list[pd.DataFrame] = []
  for metric in METRIC_NAMES:
    g = _summary_stats(metric_df, by_x_group, metric)
    g["metric"] = metric
    by_x.append(g)
    averaged = metric_df.groupby(
      [*_GROUP_AXES, "rep"], as_index=False, dropna=False
    ).agg(value=(metric, "mean"))
    g = _summary_stats(averaged, list(_GROUP_AXES), "value")
    g["metric"] = metric
    over_x.append(g)
  runtime_summary = runtime_df.groupby(
    list(_GROUP_AXES), as_index=False, dropna=False
  ).agg(
    fit_time_mean=("fit_time", "mean"),
    fit_time_std=("fit_time", "std"),
    pdf_time_mean=("pdf_time", "mean"),
    total_time_mean=("total_time", "mean"),
  )
  return {
    "metrics_by_x": metric_df,
    "summary_by_x": pd.concat(by_x, ignore_index=True),
    "summary_over_x": pd.concat(over_x, ignore_index=True),
    "runtime": runtime_df,
    "runtime_summary": cast("pd.DataFrame", runtime_summary),
  }
