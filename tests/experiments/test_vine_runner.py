"""Vine runner: arms, context wiring, scoring and the resume guard (hermetic)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pyvinecopulib as pv
import torch
from pyvinecopulib.core import NonSimplifiedContext, SimplifiedContext

from npcc.core.vinecop import RosenblattVinecop
from npcc.experiments import vine_runner
from npcc.experiments.config import EstimatorSpec, RunConfig
from npcc.experiments.vine_config import VineGridConfig


def _grid(*, arms: list[str] | None = None, eval_m: int = 20) -> VineGridConfig:
  return VineGridConfig(
    dims=[3],
    n=[40],
    n_rep=1,
    estimators=[
      EstimatorSpec(label="uni", backend="uniform-native", transform="logit")
    ],
    arms=["oracle", "random", "tll"] if arms is None else arms,
    eval_x_n=2,
    eval_m=eval_m,
  )


def test_score_is_zero_for_the_truth() -> None:
  """A perfect estimate must score zero, or every metric is biased."""
  p = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
  s = vine_runner.score(p, p)
  assert s["KL"] == pytest.approx(0.0)
  assert s["L1"] == pytest.approx(0.0)
  assert s["ISE"] == pytest.approx(0.0)


def test_random_structure_is_independent_of_the_truth() -> None:
  """A random arm reusing the truth's seed would draw the oracle's structure."""
  matrices = {
    str(vine_runner.random_structure_for(317, 5, rep).matrix.tolist())
    == str(vine_runner.truth_for(317, 5, rep).structure.matrix.tolist())
    for rep in range(5)
  }
  assert matrices == {False}


def test_one_cell_runs_every_arm(register_uniform_backends: None) -> None:
  """Every configured arm must yield finite scores at every evaluation x."""
  grid = _grid()
  metrics, runtime = vine_runner.summarize_one_cell(
    grid.cells()[0], grid, base_seed=317, device="cpu"
  )
  arms = {row["arm"] for row in metrics}
  assert arms == {"oracle", "random", "tll"}
  assert len(metrics) == 3 * grid.eval_x_n
  assert len(runtime) == 3
  for row in metrics:
    for name in vine_runner.METRIC_NAMES:
      assert math.isfinite(row[name]), name
    assert math.isfinite(row["loglik_true"])


def test_arms_use_their_contexts(
  register_uniform_backends: None, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The oracle is simplified and the random arm conditions on u_D.

  Swapping them silently would hand the oracle extra information and strip
  the random arm of the conditioning its wrong structure needs.
  """
  seen: list[type] = []
  original = RosenblattVinecop.__init__

  def spy(
    self: RosenblattVinecop,
    *args: Any,
    **kwargs: Any,
  ) -> None:
    seen.append(type(kwargs["context"]))
    original(self, *args, **kwargs)

  monkeypatch.setattr(RosenblattVinecop, "__init__", spy)
  grid = _grid(arms=["oracle", "random"])
  vine_runner.summarize_one_cell(
    grid.cells()[0], grid, base_seed=317, device="cpu"
  )
  assert seen == [SimplifiedContext, NonSimplifiedContext]


def test_resume_refuses_a_changed_grid(
  register_uniform_backends: None, tmp_path: Path
) -> None:
  """Resuming a different grid would silently mix two studies' cells."""
  run = RunConfig(out=tmp_path, device="cpu", fmt="csv")
  vine_runner.run_vine_study(_grid(arms=["tll"]), run)
  with pytest.raises(ValueError, match="resume"):
    vine_runner.run_vine_study(_grid(arms=["tll"], eval_m=30), run, resume=True)


def test_resume_reads_back_done_cells(
  register_uniform_backends: None, tmp_path: Path
) -> None:
  """A resumed run must return the checkpointed rows, not an empty table."""
  run = RunConfig(out=tmp_path, device="cpu", fmt="csv")
  first, _, _ = vine_runner.run_vine_study(_grid(arms=["tll"]), run)
  again, _, _ = vine_runner.run_vine_study(
    _grid(arms=["tll"]), run, resume=True
  )
  assert len(again) == len(first)
  assert again["KL"].tolist() == pytest.approx(first["KL"].tolist())


def test_tll_baseline_keeps_the_oracle_structure(
  register_uniform_backends: None, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A baseline that re-selects its structure would confound the x effect.

  ``kde-tll`` is there to price ignoring the covariate, so it must share the
  oracle's structure rather than pick its own from the data.
  """
  fitted: list[pv.Vinedist] = []
  original = pv.Vinedist.from_data

  def spy(*args: Any, **kwargs: Any) -> pv.Vinedist:
    model = original(*args, **kwargs)
    fitted.append(model)
    return model

  monkeypatch.setattr(pv.Vinedist, "from_data", spy)
  grid = _grid(arms=["tll"])
  vine_runner.summarize_one_cell(
    grid.cells()[0], grid, base_seed=317, device="cpu"
  )
  truth = vine_runner.truth_for(317, 3, 0)
  assert len(fitted) == 1
  copula = fitted[0].vinecop
  assert isinstance(copula, pv.Vinecop)
  np.testing.assert_array_equal(
    np.asarray(copula.matrix), np.asarray(truth.structure.matrix)
  )
