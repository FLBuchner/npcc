"""Tests for the coherence diagnostics, against closed forms on the hermetic pair.

Under the hermetic normal backend each direction's density is the logit-normal
density of its own response, ``g(v)`` for ``V | U`` and ``g(u)`` for ``U | V``,
whatever it conditions on. Every diagnostic then has a closed form.
"""

from __future__ import annotations

import math
from typing import cast

import pytest
import pyvinecopulib as pv
import torch

from npcc.core import FitControlsRosenblattBicop, RosenblattBicop
from npcc.experiments import coherence, metrics, scenarios


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


def test_directional_rows_score_each_density_at_the_draws(
  pair: RosenblattBicop,
) -> None:
  """Each direction's Monte Carlo KL must be scored at the cell's draws.

  Each KL is the mean of the true log density minus that direction's at the
  draws: a direction scored at the grid's points, or paired with the other
  direction's density, shows up as a wrong ``KL_mc``. The hermetic
  directions differ, ``g(v)`` against ``g(u)``, so a swap is visible.
  """
  x_axis = torch.tensor([0.3, 0.7], dtype=torch.float64)
  grid = scenarios.eval_grid_for_x("linear", x_axis, conditional_uv_grid_n=6)
  points = scenarios.holdout_grid("clayton", "linear", x_axis, 200, [1, 2])
  log_true = metrics.log_density(
    scenarios.ground_truth("clayton", "linear", points)["pdf"].reshape(-1)
  )

  rows = coherence.directional_rows(
    pair, grid, torch.ones_like(grid.u_flat), (points, log_true)
  )

  v_given_u, u_given_v = pair.pdf_by_direction(points.uv, x=points.x_flat)
  at_draws = {"v_given_u": v_given_u, "u_given_v": u_given_v}
  at_draws["average"] = 0.5 * (v_given_u + u_given_v)
  kl = {
    name: (log_true - d.log()).reshape(points.shape).mean(dim=0)
    for name, d in at_draws.items()
  }
  assert (kl["v_given_u"] - kl["u_given_v"]).abs().min() > 1e-3
  for name, by_x in kl.items():
    got = [row[f"{name}_KL_mc"] for row in rows]
    assert got == pytest.approx(by_x.tolist(), rel=1e-12)


class _TrueClayton:
  """A stand-in pair whose density is a Clayton copula's at tau 0.7."""

  def pdf_by_projection(
    self,
    uv: torch.Tensor,
    x: torch.Tensor | None = None,
    sinkhorn_iters: list[int | None] | None = None,
  ) -> dict[int | None, torch.Tensor]:
    del x
    family = pv.BicopFamily.clayton
    cop = pv.Bicop(
      family=family,
      parameters=pv.Bicop(family=family).tau_to_parameters(0.7),
    )
    pdf = torch.from_numpy(cop.pdf(uv.numpy()))
    return {key: pdf for key in sinkhorn_iters or [None]}


def test_normal_score_mass_is_exact_where_the_density_blows_up() -> None:
  """A density unbounded at a corner must still integrate to one.

  E4 divides each projected density by this mass before its Monte Carlo KL.
  Clayton at tau 0.7 is unbounded at the origin, where 200 x 200 midpoints
  in ``u`` overstate the mass by 8e-3 -- the size of the KLs it would shift.
  """
  # A stand-in with the one method normal_score_mass calls, not a fitted
  # pair: the true density makes the exact mass known.
  pair = cast("RosenblattBicop", _TrueClayton())
  x_axis = torch.tensor([0.2, 0.8], dtype=torch.float64)

  mass = coherence.normal_score_mass(pair, x_axis, [None, 20])

  for value in mass.values():
    torch.testing.assert_close(
      value, torch.ones(2, dtype=torch.float64), rtol=0.0, atol=1e-7
    )


def test_normal_score_mass_reads_every_x_and_iteration_count(
  register_hermetic_backends: None,
) -> None:
  """Each ``x`` and iteration count must get its own mass, from its density.

  The normal backend under probit is the independence copula, whose mass is
  one at every ``x`` and under every projection; a weight or an ``x`` read
  out of order would move it.
  """
  del register_hermetic_backends
  generator = torch.Generator().manual_seed(0)
  uv = 0.1 + 0.8 * torch.rand((40, 2), generator=generator, dtype=torch.float64)
  x = torch.rand((40, 1), generator=generator, dtype=torch.float64)
  pair = RosenblattBicop(
    FitControlsRosenblattBicop(
      backend="normal-quantile", device="cpu", transform="probit"
    )
  ).fit(uv, x=x)

  mass = coherence.normal_score_mass(
    pair, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64), [None, 5, 20]
  )

  assert sorted(mass, key=str) == sorted([None, 5, 20], key=str)
  for value in mass.values():
    torch.testing.assert_close(
      value, torch.ones(3, dtype=torch.float64), rtol=0.0, atol=1e-7
    )


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
