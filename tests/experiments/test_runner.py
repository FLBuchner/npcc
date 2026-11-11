"""End-to-end runner tests on the hermetic ``normal-quantile`` backend."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import torch

from npcc.core import Transform, registry
from npcc.core.margin import MarginOptions
from npcc.experiments import _study, scenarios
from npcc.experiments import runner as runner_mod
from npcc.experiments.config import (
  Cell,
  EstimatorSpec,
  GridConfig,
  RunConfig,
  load_grid,
)
from npcc.experiments.runner import aggregate_study_outputs, run_study
from tests.conftest import (
  TABPFN_TARGET,
  FakeTabPFNRegressor,
  NormalQuantileBackend,
  hermetic_backend,
  normal_quantile_factory,
)

Tables = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]


def _normal(
  label: str = "normal", transform: Transform = "logit"
) -> EstimatorSpec:
  return EstimatorSpec(
    label=label, backend="normal-quantile", transform=transform
  )


def _small_grid(
  n_rep: int = 2,
  batch_size: int | None = None,
  *,
  estimators: list[EstimatorSpec] | None = None,
  tau_scenarios: tuple[str, ...] = ("linear",),
  normalize: tuple[int | None, ...] = (None,),
  surface_families: tuple[str, ...] = (),
  enable_tau_diagnostics: bool = True,
) -> GridConfig:
  return GridConfig(
    families=["clayton"],
    tau_scenarios=list(tau_scenarios),
    estimators=[_normal()] if estimators is None else estimators,
    normalize=list(normalize),
    n=[20],
    n_rep=n_rep,
    projection_grid_size=8,
    conditional_uv_grid_n=3,
    conditional_x_grid_n=2,
    surface_tau_levels=[0.5],
    surface_families=list(surface_families),
    batch_size=batch_size,
    enable_tau_diagnostics=enable_tau_diagnostics,
  )


@pytest.fixture(scope="module")
def coverage_study(tmp_path_factory: pytest.TempPathFactory) -> Tables:
  """Conditional coverage for estimator axes except Sinkhorn."""
  with hermetic_backend():
    grid = _small_grid(
      n_rep=1,
      estimators=[
        _normal("normal-logit"),
        _normal("normal-identity", transform="identity"),
      ],
      tau_scenarios=("linear", "quadratic"),
      surface_families=("clayton",),
    )
    run = RunConfig(out=tmp_path_factory.mktemp("coverage"), workers=1)
    return run_study(grid, run)


@pytest.fixture(scope="module")
def backend_study(tmp_path_factory: pytest.TempPathFactory) -> Tables:
  """Two backends on the same conditional data cell.

  TabPFN stands in through its fake, and is the one backend here that needs a
  patch rather than a registration.
  """
  with hermetic_backend(), pytest.MonkeyPatch.context() as mp:
    mp.setattr(TABPFN_TARGET, FakeTabPFNRegressor)
    grid = _small_grid(
      n_rep=1,
      estimators=[
        EstimatorSpec(
          label="quant", backend="tabpfn-quantiles", transform="logit"
        ),
        _normal(),
      ],
    )
    run = RunConfig(out=tmp_path_factory.mktemp("backend"), workers=1)
    return run_study(grid, run)


@pytest.fixture(scope="module")
def projection_study(tmp_path_factory: pytest.TempPathFactory) -> Tables:
  """Small conditional projected-vs-unprojected study."""
  with hermetic_backend():
    grid = _small_grid(n_rep=1, normalize=(None, 2))
    run = RunConfig(out=tmp_path_factory.mktemp("projection"), workers=1)
    return run_study(grid, run)


def test_run_study_covers_conditional_axes(coverage_study: Tables) -> None:
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
  assert set(metric_df["backend"].unique()) == {"normal-quantile"}
  assert set(metric_df["transform"].unique()) == {"logit", "identity"}
  assert set(metric_df["quantity"].unique()) == {
    "pdf",
    "hfunc1",
    "hfunc2",
  }
  assert set(metric_df["tau_scenario"].unique()) == {"linear", "quadratic"}
  assert metric_df["x"].notna().all()


def test_conditional_metrics_are_fixed_x_uv_summaries(
  coverage_study: Tables,
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


def test_surfaces_cover_every_quantity(coverage_study: Tables) -> None:
  """A surface family must store every scored quantity at each target tau.

  The surface predictions are the one output no metric is computed from, so
  a quantity dropped from them would go unnoticed until a plot came up empty.
  """
  quantity_df = coverage_study[1]
  assert set(quantity_df["quantity"].unique()) == {"pdf", "hfunc1", "hfunc2"}
  assert set(quantity_df["target_tau"].unique()) == {0.5}
  assert quantity_df["pred"].notna().all()


def test_summaries_group_over_reps(coverage_study: Tables) -> None:
  metric_df, _, diagnostic_df, runtime_df, _ = coverage_study
  outputs = aggregate_study_outputs(metric_df, diagnostic_df, runtime_df)
  mc_summary = outputs["summary_by_x"]
  runtime_summary = outputs["runtime_summary"]
  assert set(mc_summary["metric"].unique()) == {"IAE", "ISE", "KL", "KL_mc"}
  assert {"x", "tau_true"}.issubset(mc_summary.columns)
  assert not runtime_summary.empty
  assert {"fit_time_mean", "pdf_time_mean", "tau_time_mean"}.issubset(
    runtime_summary.columns
  )


def test_aggregate_study_outputs_have_paper_tables(
  projection_study: Tables,
) -> None:
  """Every summary table is written, and the selection ranks by KL_mc.

  The grid KL is not reported, so a selection ranked on it would order the
  estimators by a number the paper does not show.
  """
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
  assert set(outputs["selection_summary"]["metric"].unique()) == {"KL_mc"}


def test_backend_axis_labels_rows_and_aggregates(backend_study: Tables) -> None:
  metric_df, _, diagnostic_df, runtime_df, _ = backend_study
  for col in ("backend", "label", "estimator_id"):
    assert col in metric_df.columns
    assert col in runtime_df.columns
  assert set(metric_df["backend"].unique()) == {
    "tabpfn-quantiles",
    "normal-quantile",
  }
  assert set(metric_df["label"].unique()) == {"normal", "quant"}
  outputs = aggregate_study_outputs(metric_df, diagnostic_df, runtime_df)
  assert {"backend", "label"} <= set(outputs["summary_by_x"].columns)
  assert set(outputs["runtime_summary"]["backend"].unique()) == {
    "tabpfn-quantiles",
    "normal-quantile",
  }


def test_normalize_axis_applies_only_to_pdf(projection_study: Tables) -> None:
  metric_df = projection_study[0]
  pdf = metric_df[metric_df["quantity"] == "pdf"]
  non_pdf = metric_df[metric_df["quantity"] != "pdf"]
  assert set(pdf["normalize"].unique()) == {"none", "2"}
  assert set(non_pdf["normalize"].unique()) == {"none"}


def test_projection_summary_pairs_pdf_accuracy_and_margin_deltas(
  projection_study: Tables,
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


def test_projection_summary_holds_only_the_estimators_that_ran(
  register_hermetic_backends: None, tmp_path: Path
) -> None:
  """Each estimator's projection deltas, and no row for one that never ran.

  A pivot table keeping its empty rows reindexes on the product of the
  estimator axes, so two estimators that differ in label, transform and id
  come back as eight, six of them all-``NaN``. The product grows with every
  axis: for E1 it was some 80 million rows and over 20 GB, built even where
  nothing was projected.
  """
  grid = _small_grid(
    n_rep=1,
    normalize=(None, 2),
    estimators=[
      _normal("normal-logit"),
      _normal("normal-identity", transform="identity"),
    ],
  )
  metric_df, _, diagnostic_df, runtime_df, _ = run_study(
    grid, RunConfig(out=tmp_path, workers=1)
  )

  projection = aggregate_study_outputs(metric_df, diagnostic_df, runtime_df)[
    "projection_summary"
  ]

  ran = metric_df[["label", "transform", "estimator_id"]].drop_duplicates()
  assert len(projection) == len(ran) == 2
  assert projection["KL_delta"].notna().all()


def test_tau_diagnostics_can_be_disabled(
  register_hermetic_backends: None, tmp_path: Path
) -> None:
  grid = _small_grid(n_rep=1, enable_tau_diagnostics=False)
  _, _, diagnostic_df, runtime_df, _ = run_study(
    grid, RunConfig(out=tmp_path, workers=1)
  )
  assert diagnostic_df["tau_hat"].isna().all()
  assert diagnostic_df["tau_abs_err"].isna().all()
  assert diagnostic_df["row_mean_abs_err"].notna().all()
  assert runtime_df["tau_time"].eq(0.0).all()


def test_release_gpu_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
  # Must be a harmless no-op when CUDA is unavailable.
  _study.release_gpu()


def test_release_gpu_calls_empty_cache(monkeypatch: pytest.MonkeyPatch) -> None:
  calls: list[int] = []
  monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
  monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
  _study.release_gpu()
  assert calls == [1]


def _spy_cells(monkeypatch: pytest.MonkeyPatch) -> list[Cell]:
  """Record every cell the runner computes rather than reads back."""
  seen: list[Cell] = []
  orig = cast("Callable[..., object]", runner_mod.summarize_one_cell)

  def spy(cell: Cell, *args: object, **kwargs: object) -> object:
    seen.append(cell)
    return orig(cell, *args, **kwargs)

  monkeypatch.setattr(runner_mod, "summarize_one_cell", spy)
  return seen


def test_resume_skips_completed_cells(
  register_hermetic_backends: None,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  grid = _small_grid(n_rep=2)
  run = RunConfig(out=tmp_path, workers=1)
  metric1 = run_study(grid, run)[0]
  cells = grid.cells()
  assert len(cells) == 2
  for cell in cells:
    assert (tmp_path / "cells" / cell.key / "DONE").exists()

  # Force the first cell to recompute by removing only its marker.
  (tmp_path / "cells" / cells[0].key / "DONE").unlink()

  seen = _spy_cells(monkeypatch)
  metric2 = run_study(grid, run, resume=True)[0]

  assert seen == [cells[0]]
  assert len(metric2) == len(metric1)


def test_a_resumed_run_returns_the_fresh_run_tables(
  register_hermetic_backends: None, tmp_path: Path
) -> None:
  """Tables read back from the shards must equal the ones a fresh run returns.

  Every sharded run writes its tables from a pass that computes nothing and
  reads every cell back, so a difference in rows, column order or dtype
  between the two routes would reach every published table unseen.
  """
  grid = _small_grid(n_rep=2, surface_families=("clayton",))
  run = RunConfig(out=tmp_path, workers=1)
  fresh = run_study(grid, run)
  resumed = run_study(grid, run, resume=True)

  for got, want in zip(resumed[:4], fresh[:4], strict=True):
    pd.testing.assert_frame_equal(got, want)


def test_stacked_cells_keep_the_type_of_one_table() -> None:
  """A column missing in all of one cell's rows must stay numeric.

  That cell's frame holds the column as ``object``, all ``None``, and
  ``pd.concat`` keeps ``object`` for the whole column, where one frame built
  from every row reads ``float64`` with ``NaN``. An ``object`` column
  aggregates differently and writes another Parquet type.
  """
  rows: list[list[dict[str, object]]] = [
    [{"label": "a", "KL": None}],
    [{"label": "b", "KL": 0.5}, {"label": "c", "KL": 1.5}],
  ]

  stacked = _study._stack([pd.DataFrame(cell) for cell in rows])

  one_table = pd.DataFrame([row for cell in rows for row in cell])
  pd.testing.assert_frame_equal(stacked, one_table)
  assert stacked["KL"].dtype == "float64"


def test_kl_mc_is_scored_at_the_cells_draws(
  register_hermetic_backends: None, tmp_path: Path
) -> None:
  """The density at the cell's draws must be scored against the truth there.

  The normal-quantile backend under probit reads out the independence copula
  exactly, so its Monte Carlo KL is the mean true log density over the cell's
  draws. Draws other than :func:`holdout_points`', a truth taken at other
  points, or rows misaligned by ``x`` would each show up as a wrong
  ``KL_mc``; and a second estimator in the cell must not move it, since every
  estimator is scored on the same draws.
  """
  grid = _small_grid(
    n_rep=1,
    estimators=[
      _normal("normal-probit", transform="probit"),
      _normal("normal-identity", transform="identity"),
    ],
  )
  run = RunConfig(out=tmp_path, workers=1)
  metric_df = run_study(grid, run)[0]
  (cell,) = grid.cells()
  x_axis = scenarios.conditional_x_axis(grid.conditional_x_grid_n)
  points, log_true = runner_mod.holdout_points(
    cell, x_axis, grid.holdout_n, runner_mod._cell_seed(run.base_seed, cell)
  )
  expected = log_true.reshape(points.shape).mean(dim=0)

  pdf = metric_df[metric_df["quantity"] == "pdf"]
  probit = pdf[pdf["label"] == "normal-probit"].sort_values("x")
  assert probit["KL_mc"].to_numpy() == pytest.approx(expected.numpy(), rel=1e-9)
  assert (pdf["KL_mc"] > 0.0).all()
  assert (pdf["KL_mc_se"] > 0.0).all()
  assert metric_df.loc[metric_df["quantity"] != "pdf", "KL_mc"].isna().all()


def test_resume_grid_signature_mismatch_raises(
  register_hermetic_backends: None, tmp_path: Path
) -> None:
  """A change to a signature input (here the base seed, which reseeds every
  cell) must refuse a --resume rather than mixing incompatible checkpoints."""
  run = RunConfig(out=tmp_path, workers=1)
  run_study(_small_grid(), run)
  reseeded = RunConfig(out=tmp_path, workers=1, base_seed=run.base_seed + 1)
  with pytest.raises(ValueError, match="signature"):
    run_study(_small_grid(), reseeded, resume=True)


def test_resume_survives_an_n_rep_increase(
  register_hermetic_backends: None,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Raising n_rep must not discard completed cells: each cell's seed hashes
  cell.rep, not n_rep, so the existing reps stay valid and only new reps run."""
  run = RunConfig(out=tmp_path, workers=1)
  first = run_study(_small_grid(n_rep=1), run)[0]

  seen = _spy_cells(monkeypatch)
  again = run_study(_small_grid(n_rep=2), run, resume=True)[0]

  assert [c.rep for c in seen] == [1]
  kept = again[again["rep"] == 0]
  assert kept["KL"].tolist() == pytest.approx(first["KL"].tolist(), nan_ok=True)


def test_resume_survives_a_batch_size_change(
  register_hermetic_backends: None, tmp_path: Path
) -> None:
  """Retuning the batch must not discard completed cells.

  `batch_size` chunks the query set and leaves each prediction's context
  alone, so it changes throughput and not results. It is a knob the study
  expects to have re-tuned for whatever GPU the run lands on -- if it entered
  the resume signature, doing so on a new machine would silently re-run every
  cell already paid for.
  """
  run = RunConfig(out=tmp_path, workers=1)
  run_study(_small_grid(n_rep=1), run)
  metric = run_study(_small_grid(n_rep=1, batch_size=4), run, resume=True)[0]

  assert not metric.empty


def test_run_study_forwards_backend_kwargs(tmp_path: Path) -> None:
  """A study's per-estimator settings must reach the backend that fits."""
  captured: list[dict[str, Any]] = []

  def factory(**kwargs: object) -> NormalQuantileBackend:
    captured.append(
      {
        k: v
        for k, v in kwargs.items()
        if k not in MarginOptions.__annotations__
      }
    )
    return normal_quantile_factory(**kwargs)

  registry.register_backend("capture-be", factory, kwarg_types={"foo": int})
  try:
    grid = _small_grid(
      n_rep=1,
      estimators=[
        EstimatorSpec(
          label="cap",
          backend="capture-be",
          transform="logit",
          backend_kwargs={"foo": 7},
        )
      ],
    )
    run_study(grid, RunConfig(out=tmp_path, workers=1))
  finally:
    registry._REGISTRY.pop("capture-be", None)

  # Both Rosenblatt directions build the backend with the configured kwargs.
  assert captured
  assert all(kw == {"foo": 7} for kw in captured)


_SHAPE = ["clayton-gumbel-mix", "student-varying-nu"]


def _shape_grid(surface_x: list[float]) -> GridConfig:
  return GridConfig(
    families=list(_SHAPE),
    tau_scenarios=[scenarios.IMPLIED],
    estimators=[_normal()],
    normalize=[None],
    n=[20],
    n_rep=1,
    projection_grid_size=8,
    conditional_uv_grid_n=3,
    conditional_x_grid_n=2,
    surface_families=list(_SHAPE),
    surface_x=surface_x,
  )


def test_a_shape_family_study_scores_against_its_own_tau(
  register_hermetic_backends: None, tmp_path: Path
) -> None:
  """Every row of a shape family's cell must carry the family's own tau.

  ``implied`` names no tau of its own, so a runner that looked tau up by
  regime would fail on it -- or, given a fallback, label the Student family's
  rows with some regime's tau rather than 0.5. Its surfaces must sit at the
  grid's ``surface_x``, with no target tau, and against the true quantities
  there.
  """
  metric_df, quantity_df, diagnostic_df, _, _ = run_study(
    _shape_grid([0.1, 0.9]), RunConfig(out=tmp_path, workers=1)
  )
  for family in _SHAPE:
    tau_of_x = scenarios.SHAPE_FAMILIES[family].tau_of_x
    for df in (metric_df, quantity_df, diagnostic_df):
      rows = df[df["family"] == family]
      assert not rows.empty
      expected = tau_of_x(torch.tensor(rows["x"].to_numpy()))
      assert rows["tau_true"].tolist() == pytest.approx(expected.tolist())

  assert sorted(quantity_df["x"].unique()) == [0.1, 0.9]
  assert quantity_df["target_tau"].isna().all()
  surface = scenarios.eval_grid_for_x(
    scenarios.IMPLIED,
    torch.tensor([0.1, 0.9], dtype=torch.float64),
    conditional_uv_grid_n=3,
  )
  truth = scenarios.ground_truth(
    "clayton-gumbel-mix", scenarios.IMPLIED, surface
  )
  pdf = quantity_df[
    (quantity_df["family"] == "clayton-gumbel-mix")
    & (quantity_df["quantity"] == "pdf")
  ].sort_values(["x", "u", "v"])
  assert pdf["truth"].tolist() == pytest.approx(
    truth["pdf"].T.reshape(-1).tolist()
  )


def test_cell_seeds_hold_as_the_registries_grow(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A cell's seed must not move when a family or a regime is appended.

  A one-parameter cell is seeded by the positions of its family and regime,
  which appending leaves alone; the bivariate study's cells keep the seeds
  pinned here. A shape cell is seeded by its names: its family and
  ``implied`` sit outside both registries, so any position given them would
  shift as either grew.
  """
  one_par = {
    Cell("clayton", "linear", 200, 0): 1393628405,
    Cell("gaussian", "sin", 2000, 9): 2949676040,
  }
  shape = [
    Cell(f, scenarios.IMPLIED, n, rep)
    for f in _SHAPE
    for n, rep in ((200, 0), (2000, 9))
  ]
  before = {c: runner_mod._cell_seed(317, c) for c in shape}
  assert {c: runner_mod._cell_seed(317, c) for c in one_par} == one_par

  monkeypatch.setitem(
    scenarios.FAMILIES, "frank-again", scenarios.FAMILIES["frank"]
  )
  monkeypatch.setitem(
    scenarios.TAU_SCENARIOS,
    "step",
    scenarios.ScenarioSpec("step", True, lambda x: (x > 0.5).double()),
  )
  assert {c: runner_mod._cell_seed(317, c) for c in one_par} == one_par
  assert {c: runner_mod._cell_seed(317, c) for c in shape} == before
  seeds = [*before.values(), *one_par.values()]
  assert len(set(seeds)) == len(seeds)


def test_the_signature_hashes_shape_settings_only_where_a_grid_has_them(
  register_hermetic_backends: None,
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """A grid without shape families or ``surface_x`` must keep its signature.

  The bivariate study's checkpoints resume only under the signature they were
  written with, which is pinned here: a key added to every grid's payload
  would refuse that resume. Where a grid does have them, both must count --
  including a change to a shape family that leaves its tau alone, as moving
  the Student family's degrees of freedom does.
  """
  run = RunConfig(out=tmp_path)
  e1 = load_grid(Path(__file__).resolve().parents[2] / "configs" / "study.toml")
  assert (
    runner_mod._grid_signature(e1, run)
    == "d82c7bd74d5fe1bd2bad2d546daa44ca49f226ef5d0bfb0f77785a360514c02e"
  )

  base = runner_mod._grid_signature(_shape_grid([0.1, 0.9]), run)
  assert runner_mod._grid_signature(_shape_grid([0.1, 0.5]), run) != base
  monkeypatch.setattr(scenarios, "STUDENT_NU_RANGE", (3.0, 30.0))
  assert runner_mod._grid_signature(_shape_grid([0.1, 0.9]), run) != base


@pytest.mark.parametrize("label", ["null", "NA", "None", "nan"])
def test_a_csv_checkpoint_keeps_a_label_pandas_would_read_as_missing(
  tmp_path: Path, label: str
) -> None:
  """A label read back as NaN would drop its rows from every groupby.

  A resumed or aggregated CSV run reads its checkpoints back; only the empty
  field that ``to_csv`` writes for a missing value may become one.
  """
  frame = pd.DataFrame(
    {"scenario": [label, "linear"], "IAE": [float("nan"), 0.5]}
  )
  _study.write_table(frame, tmp_path / "t", "csv")
  back = _study.read_table(tmp_path / "t", "csv")
  assert back["scenario"].tolist() == [label, "linear"]
  assert back["IAE"].isna().tolist() == [True, False]
