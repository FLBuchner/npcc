"""Tests for the coherence diagnostics, against closed forms on the hermetic pair.

Under the hermetic normal backend each direction's density is the logit-normal
density of its own response, ``g(v)`` for ``V | U`` and ``g(u)`` for ``U | V``,
whatever it conditions on. Every diagnostic then has a closed form.
"""

from __future__ import annotations

import math
from typing import cast

import pytest
import torch

from npcc.core import FitControlsRosenblattBicop, RosenblattBicop
from npcc.experiments import coherence, scenarios


def _g(y: torch.Tensor) -> torch.Tensor:
  z = torch.logit(y)
  return torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) / (y * (1 - y))


def _big_g(y: torch.Tensor) -> torch.Tensor:
  return cast("torch.Tensor", torch.special.ndtr(torch.logit(y)))


@pytest.fixture
def pair(register_hermetic_backends: None) -> RosenblattBicop:
  del register_hermetic_backends
  generator = torch.Generator().manual_seed(0)
  uv = 0.1 + 0.8 * torch.rand((40, 2), generator=generator, dtype=torch.float64)
  x = torch.rand((40, 1), generator=generator, dtype=torch.float64)
  model = RosenblattBicop(
    FitControlsRosenblattBicop(backend="normal-quantile", device="cpu")
  )
  return model.fit(uv, x=x)


@pytest.fixture
def grid() -> scenarios.EvalGrid:
  x_axis = torch.tensor([0.3, 0.7], dtype=torch.float64)
  return scenarios.eval_grid_for_x("linear", x_axis, conditional_uv_grid_n=6)


def test_directions_are_the_ones_the_pair_averages(
  pair: RosenblattBicop, grid: scenarios.EvalGrid
) -> None:
  """The diagnostic's two densities average to exactly what ``pdf`` returns.

  A diagnostic built on any other pair of densities would measure an
  incoherence the estimator does not have.
  """
  v_given_u, u_given_v = pair.pdf_by_direction(grid.uv, x=grid.x_flat)

  torch.testing.assert_close(
    0.5 * (v_given_u + u_given_v), pair.pdf(grid.uv, x=grid.x_flat)
  )
  torch.testing.assert_close(v_given_u, _g(grid.v_flat))
  torch.testing.assert_close(u_given_v, _g(grid.u_flat))


def test_directional_rows_match_the_closed_form(
  pair: RosenblattBicop, grid: scenarios.EvalGrid
) -> None:
  """Disagreement and accuracy are measured per x, on the right densities.

  With the average as the truth, the average's error is zero and each
  direction's error is half the gap between the two.
  """
  average = 0.5 * (_g(grid.v_flat) + _g(grid.u_flat))

  rows = coherence.directional_rows(pair, grid, average)

  gap = (_g(grid.v_flat) - _g(grid.u_flat)).abs().reshape(grid.shape)
  assert [r["x"] for r in rows] == pytest.approx([0.3, 0.7])
  for j, row in enumerate(rows):
    assert row["iae_between_directions"] == pytest.approx(
      float(gap[:, j].mean())
    )
    assert row["average_IAE"] == pytest.approx(0.0, abs=1e-12)
    assert row["v_given_u_IAE"] == pytest.approx(0.5 * float(gap[:, j].mean()))


def test_hfunc_rows_integrate_the_average_along_the_right_argument(
  pair: RosenblattBicop, grid: scenarios.EvalGrid
) -> None:
  """Each direct h is compared with the integral it should equal.

  ``h1 = G(v)`` directly, while integrating the average over ``(0, v)`` gives
  ``(G(v) + g(u) v) / 2``; ``h2`` is the same with the arguments swapped. An
  integral along the wrong argument, or of one direction alone, would give
  other numbers.
  """
  rows = coherence.hfunc_rows(pair, grid, n_int=800)

  u, v = (
    grid.u_flat.reshape(grid.shape)[:, 0],
    grid.v_flat.reshape(grid.shape)[:, 0],
  )
  gap1 = (0.5 * _big_g(v) - 0.5 * _g(u) * v).abs()
  gap2 = (0.5 * _big_g(u) - 0.5 * _g(v) * u).abs()
  for row in rows:
    assert row["h1_IAE"] == pytest.approx(float(gap1.mean()), abs=2e-3)
    assert row["h2_IAE"] == pytest.approx(float(gap2.mean()), abs=2e-3)


def test_hfunc_rows_refuse_a_projected_pair(
  pair: RosenblattBicop, grid: scenarios.EvalGrid
) -> None:
  """The integrals are of the averaged density, which a projection changes."""
  pair.sinkhorn_iters = 5

  with pytest.raises(ValueError, match="unprojected"):
    coherence.hfunc_rows(pair, grid)
