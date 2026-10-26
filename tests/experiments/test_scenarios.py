"""Tests for ``npcc.experiments.scenarios`` (sampling + ground truth)."""

from __future__ import annotations

import pytest
import pyvinecopulib as pv
import torch

from npcc.experiments import scenarios


def test_eval_grid_conditional_shapes() -> None:
  grid = scenarios.eval_grid("linear")
  n_pairs = scenarios.CONDITIONAL_UV_GRID_N**2
  assert grid.conditional is True
  assert grid.shape == (n_pairs, scenarios.CONDITIONAL_X_GRID_N)
  assert grid.u_flat.shape == (n_pairs * scenarios.CONDITIONAL_X_GRID_N,)
  assert grid.x_flat is not None
  assert grid.x_axis is not None
  # `x_flat` is one covariate per row, so `(n_points, 1)`; `x_axis` stays the
  # one-dimensional axis it tiles.
  assert grid.x_flat.shape == (
    n_pairs * scenarios.CONDITIONAL_X_GRID_N,
    1,
  )
  assert grid.x_axis.shape == (scenarios.CONDITIONAL_X_GRID_N,)
  # `x_flat` TILES `x_axis` while `u_flat`/`v_flat` repeat-interleave over it.
  # Swapping either verb keeps every shape and every row count, raises
  # nothing, and mis-pairs every (u, v, x) triple -- and no estimator can see
  # it, because the hermetic backends read only the row count. This assertion
  # is the only place that check exists.
  torch.testing.assert_close(
    grid.x_flat[: scenarios.CONDITIONAL_X_GRID_N, 0], grid.x_axis
  )
  assert grid.u_axis.shape == (scenarios.CONDITIONAL_UV_GRID_N,)
  expected_axis = (
    torch.arange(scenarios.CONDITIONAL_UV_GRID_N, dtype=torch.float64) + 0.5
  ) / scenarios.CONDITIONAL_UV_GRID_N
  torch.testing.assert_close(grid.u_axis, expected_axis)


def test_eval_grid_unconditional_has_no_x() -> None:
  grid = scenarios.eval_grid("uncond70")
  assert grid.conditional is False
  assert grid.x_flat is None
  assert grid.shape == (grid.u_flat.shape[0],)


def test_ground_truth_conditional_matches_pyvinecopulib() -> None:
  family, scenario = "clayton", "linear"
  truth = scenarios.ground_truth(family, scenario)
  grid = scenarios.eval_grid(scenario)
  assert grid.x_axis is not None

  spec = scenarios.TAU_SCENARIOS[scenario]
  assert spec.tau_of_x is not None
  tau_x = spec.tau_of_x(grid.x_axis)
  uu, vv = torch.meshgrid(grid.u_axis, grid.v_axis, indexing="ij")
  uv = torch.column_stack([uu.reshape(-1), vv.reshape(-1)])

  assert torch.unique(uv, dim=0).shape[0] == grid.shape[0]
  for x_idx, tau in enumerate(tau_x):
    cop = scenarios._bicop(scenarios.FAMILIES[family], float(tau))
    expected_pdf = torch.from_numpy(cop.pdf(uv.numpy()))
    torch.testing.assert_close(
      truth["pdf"][:, x_idx], expected_pdf, atol=1e-10, rtol=1e-7
    )
  assert truth["pdf"].shape == grid.shape


def test_ground_truth_unconditional_matches_pyvinecopulib() -> None:
  family, scenario = "frank", "uncond70"
  truth = scenarios.ground_truth(family, scenario)
  grid = scenarios.eval_grid(scenario)
  cop = scenarios._bicop(scenarios.FAMILIES[family], 0.70)
  uv = torch.column_stack([grid.u_flat, grid.v_flat])
  expected = torch.from_numpy(cop.cdf(uv.numpy()))
  torch.testing.assert_close(truth["cdf"], expected, atol=1e-10, rtol=1e-7)
  assert truth["pdf"].shape == grid.shape


def test_sample_conditional_returns_x_linspace_in_unit_square() -> None:
  u, v, x = scenarios.sample("gumbel", "linear", n=200, seed=0)
  assert x is not None
  assert u.shape == v.shape == (200,)
  assert x.shape == (200, 1)
  torch.testing.assert_close(
    x,
    torch.linspace(
      scenarios.X_MIN, scenarios.X_MAX, 200, dtype=torch.float64
    ).reshape(-1, 1),
  )
  assert (u > 0).all() and (u < 1).all()
  assert (v > 0).all() and (v < 1).all()


@pytest.mark.parametrize("family", ["clayton", "gumbel", "frank", "gaussian"])
def test_sample_unconditional_recovers_target_tau(family: str) -> None:
  u, v, x = scenarios.sample(family, "uncond70", n=4000, seed=1)
  assert x is None
  tau = float(pv.utils.wdm(u.numpy(), v.numpy(), "tau"))
  assert abs(tau - 0.70) < 0.06


def test_is_conditional_flags() -> None:
  assert scenarios.is_conditional("linear") is True
  assert scenarios.is_conditional("uncond70") is False


def test_eval_grid_for_x_shapes_and_tiling() -> None:
  """The second grid producer, which had no direct coverage at all.

  ``eval_grid_for_x`` builds its covariate column the same way ``eval_grid``
  does, so a change applied to one and not the other passes every other test
  in this file.
  """
  x_axis = torch.tensor([0.2, 0.8], dtype=torch.float64)
  grid = scenarios.eval_grid_for_x("linear", x_axis, conditional_uv_grid_n=3)
  n_pairs = 9

  assert grid.shape == (n_pairs, 2)
  assert grid.x_flat is not None
  assert grid.x_flat.shape == (n_pairs * 2, 1)
  assert grid.x_axis is not None
  assert grid.x_axis.shape == (2,)
  torch.testing.assert_close(grid.x_flat[:2, 0], x_axis)
