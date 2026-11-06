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
from collections.abc import Mapping
from time import perf_counter
from typing import Any, cast

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
from npcc.core.margin import QuantileTableConfig
from npcc.core.registry import create_backend
from npcc.core.vinecop import RosenblattVinecop
from npcc.core.vinedist import RosenblattVinedist
from npcc.experiments import scenarios, vine_scenarios
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
from npcc.experiments.vine_config import NPCC_ARMS, VineCell, VineGridConfig
from npcc.experiments.vine_scenarios import EvalPoints, VineTruth

logger = logging.getLogger(__name__)

_EPS: float = 1e-300

METRIC_NAMES: tuple[str, ...] = ("KL", "L1", "ISE", "KL_copula", "L1_copula")
TABLES: tuple[str, ...] = ("metrics", "runtime")
_GROUP_AXES: tuple[str, ...] = (
  "d",
  "n",
  "arm",
  "label",
  "estimator_id",
  "backend",
  "transform",
)


def truth_for(base_seed: int, d: int, rep: int) -> VineTruth:
  """The true vine of repetition ``rep`` in dimension ``d``."""
  return vine_scenarios.draw_truth(d, derive_seed(base_seed, "truth", d, rep))


def random_structure_for(base_seed: int, d: int, rep: int) -> pv.RVineStructure:
  """The ``random`` arm's structure, drawn independently of the truth's."""
  return pv.RVineStructure.sample(
    d, seeds=[derive_seed(base_seed, "random", d, rep)]
  )


def regime_fingerprint() -> dict[str, list[float]]:
  """Each vine regime's tau on a fixed ``x`` grid, rounded for stable hashing."""
  x = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
  out: dict[str, list[float]] = {}
  for name in vine_scenarios.REGIMES:
    tau_of_x = scenarios.TAU_SCENARIOS[name].tau_of_x
    assert tau_of_x is not None
    out[name] = [round(float(t), 12) for t in tau_of_x(x)]
  return out


def grid_signature(grid: VineGridConfig, run: RunConfig) -> str:
  """Stable hash of everything that changes cell outputs (resume guard)."""
  payload = {
    "study": "vine",
    # The data-generating process, not merely the grid: a resumed cell must
    # have been drawn from the same truth. The regimes are fingerprinted by
    # value, so redefining one -- not only moving its bounds -- invalidates.
    "tau_bounds": [scenarios.TAU_LO, scenarios.TAU_HI],
    "regimes": regime_fingerprint(),
    "mu_slope": vine_scenarios.MU_SLOPE,
    # `dims`, `n` and `n_rep` are absent because they choose which cells run,
    # not what any cell computes: a cell is keyed by its own (d, n, rep), and
    # its truth and seeds derive from those alone. Leaving them out is what
    # lets a grid grow -- more repetitions, a new dimension -- and resume on
    # top of the cells it already has.
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
  controls = est.controls(
    FitControlsRosenblattVinecop,
    device=device,
    projection_grid_size=projection_grid_size,
    batch_size=batch_size,
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
) -> tuple[Rows, Rows]:
  """Fit every arm on one cell's data and return its metric and runtime rows."""
  truth = truth_for(base_seed, cell.d, cell.rep)
  y, _, x = vine_scenarios.sample(
    truth, cell.n, derive_seed(base_seed, "train", cell.d, cell.n, cell.rep)
  )
  points = vine_scenarios.eval_points(
    truth,
    scenarios.conditional_x_axis(grid.eval_x_n),
    grid.eval_m,
    derive_seed(base_seed, "eval", cell.d, cell.rep),
  )
  structures = {
    "oracle": truth.structure,
    "random": random_structure_for(base_seed, cell.d, cell.rep),
  }
  cell_seed = derive_seed(base_seed, "cell", cell.d, cell.n, cell.rep)
  cell_base = {"d": cell.d, "n": cell.n, "rep": cell.rep, "seed": cell_seed}

  metric_rows: Rows = []
  runtime_rows: Rows = []

  def _record(
    base: Mapping[str, object], fit_time: float, pdf_time: float
  ) -> None:
    memory = memory_usage()
    logger.info(
      "arm %s / %s [d=%d n=%d rep=%d]: fit %.1fs | GPU peak %.0f MiB | "
      "RSS %.0f MiB | MemAvailable %.0f MiB",
      base["arm"],
      base["label"],
      cell.d,
      cell.n,
      cell.rep,
      fit_time,
      memory["gpu_peak_reserved_mb"],
      memory["rss_mb"],
      memory["mem_available_mb"],
    )
    runtime_rows.append(
      {
        **base,
        "fit_time": fit_time,
        "pdf_time": pdf_time,
        "total_time": fit_time + pdf_time,
        **memory,
      }
    )

  for arm in (a for a in grid.arms if a in NPCC_ARMS):
    for est in grid.estimators:
      base = {
        **cell_base,
        "arm": arm,
        "label": est.label,
        "estimator_id": est.estimator_id,
        "backend": est.backend,
        "transform": est.transform,
      }
      torch.manual_seed(derive_seed(cell_seed, arm, est.estimator_id))
      reset_gpu_peaks()
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
      release_gpu()

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
  logger.info(
    "Vine study: %d arms x %d estimators per cell",
    len(grid.arms),
    len(grid.estimators),
  )
  (metric_rows, runtime_rows), wall = sweep(
    grid.cells(),
    lambda cell: summarize_one_cell(
      cell, grid, base_seed=run.base_seed, device=run.device
    ),
    tables=TABLES,
    run=run,
    signature=grid_signature(grid, run),
    study="vine",
    resume=resume,
  )

  sort_axes = ["d", "n", "arm", "label", "rep"]
  metric_df = sorted_frame(metric_rows, [*sort_axes, "x"])
  runtime_df = sorted_frame(runtime_rows, sort_axes)
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
  summary_by_x, summary_over_x = summarize(
    metric_df, METRIC_NAMES, _GROUP_AXES, x_axes=("x",)
  )
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
    "summary_by_x": summary_by_x,
    "summary_over_x": summary_over_x,
    "runtime": runtime_df,
    "runtime_summary": cast("pd.DataFrame", runtime_summary),
  }
