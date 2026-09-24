"""End-to-end runner tests using the fake TabPFN regressor (hermetic)."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import pytest
import torch

from npcc.core.margin_quantile_table import QuantileTableConfig
from npcc.experiments import runner as runner_mod
from npcc.experiments.config import EstimatorSpec, GridConfig, RunConfig
from npcc.experiments.runner import (
  aggregate_results,
  aggregate_study_outputs,
  run_study,
)
from tests.conftest import (
  _TABPFN_REGRESSOR_TARGETS,
  _UniformQuantileBackend,
  _UniformQuantileRegressor,
)


def _run(
  grid: GridConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
  mp = pytest.MonkeyPatch()
  for target in _TABPFN_REGRESSOR_TARGETS:
    mp.setattr(target, _UniformQuantileRegressor)
  out = Path(tempfile.mkdtemp(prefix="npcc_run_"))
  try:
    return run_study(grid, RunConfig(out=out, workers=1))
  finally:
    mp.undo()


@pytest.fixture(scope="module")
def coverage_study() -> tuple[
  pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float
]:
  """Conditional coverage for estimator axes except Sinkhorn."""
  grid = GridConfig(
    families=["clayton"],
    tau_scenarios=["linear", "quadratic"],
    estimators=[
      EstimatorSpec(
        label="crit-logit", backend="tabpfn-criterion", transform="logit"
      ),
      EstimatorSpec(
        label="crit-identity",
        backend="tabpfn-criterion",
        transform="identity",
      ),
      EstimatorSpec(
        label="quant-logit", backend="tabpfn-quantiles", transform="logit"
      ),
      EstimatorSpec(
        label="quant-identity",
        backend="tabpfn-quantiles",
        transform="identity",
      ),
    ],
    normalize=[None],
    n=[20],
    n_rep=1,
    projection_grid_size=8,
    conditional_uv_grid_n=3,
    conditional_x_grid_n=2,
    surface_tau_levels=[0.5],
  )
  return _run(grid)


@pytest.fixture(scope="module")
def backend_study() -> tuple[
  pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float
]:
  """Two backends on the same conditional data cell."""
  grid = GridConfig(
    families=["clayton"],
    tau_scenarios=["linear"],
    estimators=[
      EstimatorSpec(
        label="crit", backend="tabpfn-criterion", transform="logit"
      ),
      EstimatorSpec(
        label="quant", backend="tabpfn-quantiles", transform="logit"
      ),
    ],
    normalize=[None],
    n=[20],
    n_rep=1,
    projection_grid_size=8,
    conditional_uv_grid_n=3,
    conditional_x_grid_n=2,
  )
  return _run(grid)


@pytest.fixture(scope="module")
def projection_study() -> tuple[
  pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float
]:
  """Small conditional projected-vs-unprojected study."""
  grid = GridConfig(
    families=["clayton"],
    tau_scenarios=["linear"],
    estimators=[
      EstimatorSpec(label="crit", backend="tabpfn-criterion", transform="logit")
    ],
    normalize=[None, 2],
    n=[20],
    n_rep=1,
    projection_grid_size=8,
    conditional_uv_grid_n=3,
    conditional_x_grid_n=2,
    surface_families=[],
  )
  return _run(grid)


def test_run_study_covers_conditional_axes(
  coverage_study: tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float
  ],
) -> None:
  metric_df, quantity_df, diagnostic_df, runtime_df, wall = coverage_study
  assert wall >= 0.0
  expected_cols = {
    "family",
    "tau_scenario",
    "n",
    "rep",
    "seed",
    "label",
    "estimator_id",
    "transform",
    "backend",
    "normalize",
    "quantity",
    "x",
    "tau_true",
    "IAE",
    "ISE",
    "KL",
  }
  assert expected_cols.issubset(metric_df.columns)
  assert {"target_tau", "truth", "pred"}.issubset(quantity_df.columns)
  assert {"tau_hat", "row_mean_abs_err", "col_max_abs_err"}.issubset(
    diagnostic_df.columns
  )
  assert "tau_time" in runtime_df.columns
  assert {
    "gpu_peak_reserved_mb",
    "gpu_peak_alloc_mb",
    "rss_mb",
    "mem_available_mb",
  }.issubset(runtime_df.columns)
  assert set(metric_df["backend"].unique()) == {
    "tabpfn-criterion",
    "tabpfn-quantiles",
  }
  assert set(metric_df["transform"].unique()) == {"logit", "identity"}
  assert set(metric_df["quantity"].unique()) == {
    "pdf",
    "hfunc1",
    "hfunc2",
  }
  assert set(metric_df["tau_scenario"].unique()) == {"linear", "quadratic"}
  assert metric_df["x"].notna().all()


def test_conditional_metrics_are_fixed_x_uv_summaries(
  coverage_study: tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float
  ],
) -> None:
  metric_df = coverage_study[0]
  grouped = metric_df.groupby(
    ["tau_scenario", "backend", "transform", "quantity", "normalize"],
    dropna=False,
  )
  assert grouped.size().min() == 2
  assert grouped.size().max() == 2
  non_pdf = metric_df[metric_df["quantity"] != "pdf"]
  assert non_pdf["KL"].isna().all()
  assert metric_df[metric_df["quantity"] == "pdf"]["KL"].notna().all()


def test_aggregate_results_groups_over_reps(
  coverage_study: tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float
  ],
) -> None:
  metric_df, _, _, runtime_df, _ = coverage_study
  mc_summary, runtime_summary = aggregate_results(metric_df, runtime_df)
  assert set(mc_summary["metric"].unique()) == {"IAE", "ISE", "KL"}
  assert {"x", "tau_true"}.issubset(mc_summary.columns)
  assert not runtime_summary.empty
  assert {"fit_time_mean", "pdf_time_mean", "tau_time_mean"}.issubset(
    runtime_summary.columns
  )


def test_aggregate_study_outputs_have_paper_tables(
  projection_study: tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float
  ],
) -> None:
  metric_df, _, diagnostic_df, runtime_df, _ = projection_study
  outputs = aggregate_study_outputs(metric_df, diagnostic_df, runtime_df)
  assert {
    "metrics_by_x",
    "summary_by_x",
    "summary_over_x",
    "selection_summary",
    "projection_summary",
    "tau_summary",
    "runtime_summary",
  } <= set(outputs)
  assert outputs["summary_by_x"]["x"].notna().all()
  assert "x" not in outputs["summary_over_x"].columns
  assert outputs["selection_summary"].iloc[0]["rank"] == 1
  assert set(outputs["selection_summary"]["quantity"].unique()) == {"pdf"}
  assert set(outputs["selection_summary"]["metric"].unique()) == {"KL"}


def test_backend_axis_labels_rows_and_aggregates(
  backend_study: tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float
  ],
) -> None:
  metric_df, _, _, runtime_df, _ = backend_study
  for col in ("backend", "label", "estimator_id"):
    assert col in metric_df.columns
    assert col in runtime_df.columns
  assert set(metric_df["backend"].unique()) == {
    "tabpfn-criterion",
    "tabpfn-quantiles",
  }
  assert set(metric_df["label"].unique()) == {"crit", "quant"}
  mc_summary, runtime_summary = aggregate_results(metric_df, runtime_df)
  assert "backend" in mc_summary.columns
  assert "label" in mc_summary.columns
  assert set(runtime_summary["backend"].unique()) == {
    "tabpfn-criterion",
    "tabpfn-quantiles",
  }


def test_normalize_axis_applies_only_to_pdf(
  projection_study: tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float
  ],
) -> None:
  metric_df = projection_study[0]
  pdf = metric_df[metric_df["quantity"] == "pdf"]
  non_pdf = metric_df[metric_df["quantity"] != "pdf"]
  assert set(pdf["normalize"].unique()) == {"none", "2"}
  assert set(non_pdf["normalize"].unique()) == {"none"}


def test_projection_summary_pairs_pdf_accuracy_and_margin_deltas(
  projection_study: tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float
  ],
) -> None:
  metric_df, _, diagnostic_df, runtime_df, _ = projection_study
  projection = aggregate_study_outputs(metric_df, diagnostic_df, runtime_df)[
    "projection_summary"
  ]
  assert set(projection["projected_normalize"].unique()) == {"2"}
  assert {
    "KL_delta",
    "IAE_delta",
    "row_mean_abs_err_delta",
    "col_mean_abs_err_delta",
  }.issubset(projection.columns)


def test_tau_diagnostics_can_be_disabled() -> None:
  grid = GridConfig(
    families=["clayton"],
    tau_scenarios=["linear"],
    estimators=[
      EstimatorSpec(label="crit", backend="tabpfn-criterion", transform="logit")
    ],
    normalize=[None],
    n=[20],
    n_rep=1,
    conditional_uv_grid_n=3,
    conditional_x_grid_n=2,
    surface_families=[],
    enable_tau_diagnostics=False,
  )
  _, _, diagnostic_df, runtime_df, _ = _run(grid)
  assert diagnostic_df["tau_hat"].isna().all()
  assert diagnostic_df["tau_abs_err"].isna().all()
  assert diagnostic_df["row_mean_abs_err"].notna().all()
  assert runtime_df["tau_time"].eq(0.0).all()


def _small_grid(n_rep: int = 2, batch_size: int | None = None) -> GridConfig:
  return GridConfig(
    families=["clayton"],
    tau_scenarios=["linear"],
    estimators=[
      EstimatorSpec(label="crit", backend="tabpfn-criterion", transform="logit")
    ],
    normalize=[None],
    n=[20],
    n_rep=n_rep,
    projection_grid_size=8,
    conditional_uv_grid_n=3,
    conditional_x_grid_n=2,
    surface_families=[],
    batch_size=batch_size,
  )


def _fake_mp() -> pytest.MonkeyPatch:
  mp = pytest.MonkeyPatch()
  for target in _TABPFN_REGRESSOR_TARGETS:
    mp.setattr(target, _UniformQuantileRegressor)
  return mp


def test_release_gpu_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
  # Must be a harmless no-op when CUDA is unavailable.
  runner_mod._release_gpu()


def test_release_gpu_calls_empty_cache(monkeypatch: pytest.MonkeyPatch) -> None:
  calls: list[int] = []
  monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
  monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
  runner_mod._release_gpu()
  assert calls == [1]


def test_resume_skips_completed_cells(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  grid = _small_grid(n_rep=2)
  run = RunConfig(out=tmp_path, workers=1)
  mp = _fake_mp()
  try:
    metric1 = run_study(grid, run)[0]
    cells = grid.cells()
    assert len(cells) == 2
    for cell in cells:
      done = tmp_path / "cells" / runner_mod._cell_key(cell) / "DONE"
      assert done.exists()

    # Force the first cell to recompute by removing only its marker.
    (tmp_path / "cells" / runner_mod._cell_key(cells[0]) / "DONE").unlink()

    seen: list[object] = []
    orig = cast("Callable[..., object]", runner_mod.summarize_one_cell)

    def _spy(*args: object, **kwargs: object) -> object:
      seen.append(args[0])
      return orig(*args, **kwargs)

    monkeypatch.setattr(runner_mod, "summarize_one_cell", _spy)
    metric2 = run_study(grid, run, resume=True)[0]
  finally:
    mp.undo()

  assert seen == [cells[0]]
  assert len(metric2) == len(metric1)


def test_resume_grid_signature_mismatch_raises(tmp_path: Path) -> None:
  run = RunConfig(out=tmp_path, workers=1)
  mp = _fake_mp()
  try:
    run_study(_small_grid(n_rep=1), run)
    with pytest.raises(ValueError, match="signature"):
      run_study(_small_grid(n_rep=3), run, resume=True)
  finally:
    mp.undo()


def test_resume_survives_a_batch_size_change(tmp_path: Path) -> None:
  """Retuning the batch must not discard completed cells.

  `batch_size` chunks the query set and leaves each prediction's context
  alone, so it changes throughput and not results. It is a knob the study
  expects to have re-tuned for whatever GPU the run lands on -- if it entered
  the resume signature, doing so on a new machine would silently re-run every
  cell already paid for.
  """
  run = RunConfig(out=tmp_path, workers=1)
  mp = _fake_mp()
  try:
    run_study(_small_grid(n_rep=1), run)
    metric = run_study(_small_grid(n_rep=1, batch_size=4), run, resume=True)[0]
  finally:
    mp.undo()

  assert not metric.empty


def test_run_study_forwards_backend_kwargs(tmp_path: Path) -> None:
  from npcc.core import registry

  captured: list[dict[str, Any]] = []

  def _factory(
    *,
    transform: Literal["identity", "logit", "probit"],
    quantile_table_config: QuantileTableConfig,
    eps: float,
    device: str | torch.device | None,
    batch_size: int | None,
    **kw: object,
  ) -> _UniformQuantileBackend:
    captured.append(dict(kw))
    return _UniformQuantileBackend(
      transform=transform,
      quantile_table_config=quantile_table_config,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )

  registry.register_backend(
    "capture-be", _factory, allowed_kwargs=frozenset({"foo"})
  )
  try:
    grid = GridConfig(
      families=["clayton"],
      tau_scenarios=["linear"],
      estimators=[
        EstimatorSpec(
          label="cap",
          backend="capture-be",
          transform="logit",
          backend_kwargs={"foo": 7},
        )
      ],
      normalize=[None],
      n=[20],
      n_rep=1,
      projection_grid_size=8,
      conditional_uv_grid_n=3,
      conditional_x_grid_n=2,
      surface_families=[],
    )
    run_study(grid, RunConfig(out=tmp_path, workers=1))
  finally:
    registry._REGISTRY.pop("capture-be", None)

  # Both Rosenblatt directions build the backend with the configured kwargs.
  assert captured
  assert all(kw == {"foo": 7} for kw in captured)
