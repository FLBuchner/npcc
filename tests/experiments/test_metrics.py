"""Tests for ``npcc.experiments.metrics``."""

from __future__ import annotations

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
