"""Tests for ``npcc.experiments.metrics``."""

from __future__ import annotations

import math

import pytest
import torch

from npcc.experiments import metrics


def test_grid_metrics_means() -> None:
  y_true = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
  y_hat = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64)
  out = metrics.grid_metrics(y_true, y_hat)
  assert out["IAE"] == pytest.approx(2.0 / 3.0)
  assert out["ISE"] == pytest.approx(4.0 / 3.0)


def test_marginal_diagnostics_uniform_density_is_exact() -> None:
  # c(u,v) == 1 has both margins integrating to 1 exactly.
  u = (torch.arange(20, dtype=torch.float64) + 0.5) / 20
  v = (torch.arange(20, dtype=torch.float64) + 0.5) / 20
  c = torch.ones((u.numel(), v.numel()), dtype=torch.float64)
  out = metrics.marginal_diagnostics(c, u, v)
  assert out == {
    "row_mean_abs_err": 0.0,
    "row_max_abs_err": 0.0,
    "col_mean_abs_err": 0.0,
    "col_max_abs_err": 0.0,
  }


def test_holdout_kl_is_zero_at_the_truth_and_positive_elsewhere() -> None:
  """The Monte Carlo KL must vanish for the true density and only there.

  At draws from ``c``, ``mean(log c - log c_hat)`` estimates ``KL(c || c_hat)``,
  which is nonnegative; the grid's ``mean p log(p / q)`` can be negative and
  rewards mass moved off the boundary, which is why it is not reported.
  """
  pdf = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float64)
  log_true = metrics.log_density(pdf)
  exact = metrics.holdout_kl(log_true, pdf)
  assert exact["KL_mc"] == 0.0
  assert exact["KL_mc_se"] == 0.0
  assert exact["KL_mc_floored"] == 0.0
  flat = metrics.holdout_kl(log_true, torch.ones_like(pdf))
  assert flat["KL_mc"] == pytest.approx(float(log_true.mean()))
  assert flat["KL_mc_se"] == pytest.approx(float(log_true.std() / 2.0))


def test_holdout_kl_floors_a_zero_density_and_counts_it() -> None:
  """A fitted density of zero at a draw must cost a finite, counted penalty.

  ``log 0`` would make the KL infinite for one draw in millions; the floor is
  the one E3 and VDC apply, and the count says how often it was reached. A
  ``float32`` density is floored too, though the floor is under its range.
  """
  log_true = torch.zeros(3, dtype=torch.float64)
  expected = -2.0 * math.log(metrics.LOG_FLOOR) / 3.0
  for dtype in (torch.float64, torch.float32):
    pdf = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    got = metrics.holdout_kl(log_true, pdf)
    assert got["KL_mc_floored"] == 2.0
    assert got["KL_mc"] == pytest.approx(expected)


def test_holdout_kl_counts_its_draws_and_any_non_finite_density() -> None:
  """A NaN KL must say why, and every KL must say on how many draws.

  A fitted density that is NaN at one draw makes ``KL_mc`` NaN; the count
  lets the analysis say so instead of calling the run outdated. The draw
  count lets it refuse to pair runs scored on different draws.
  """
  log_true = torch.zeros(4, dtype=torch.float64)
  pdf = torch.tensor([1.0, float("nan"), float("inf"), 1.0])

  got = metrics.holdout_kl(log_true, pdf)

  assert got["KL_mc_n"] == 4.0
  assert got["KL_mc_nonfinite"] == 2.0
  assert math.isnan(got["KL_mc"])
