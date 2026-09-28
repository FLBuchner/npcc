"""Vine truth: tau-band decay, margins, seeding, and the exact densities."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from npcc.experiments import scenarios, vine_scenarios
from npcc.experiments.scenarios import TAU_HI, TAU_LO


@pytest.mark.parametrize("regime", vine_scenarios.REGIMES)
@pytest.mark.parametrize("tree", [1, 2, 3, 4])
def test_edge_tau_stays_in_the_decaying_band(regime: str, tree: int) -> None:
  """A deep edge must not be as strong as tree 1; the band is TAU_HI ** tree."""
  x = torch.linspace(0.0, 1.0, 201, dtype=torch.float64)
  tau = vine_scenarios.edge_tau(regime, tree, x)
  assert float(tau.min()) >= TAU_LO - 1e-12
  assert float(tau.max()) <= TAU_HI**tree + 1e-12


@pytest.mark.parametrize("regime", vine_scenarios.REGIMES)
def test_tree_one_is_the_bivariate_regime(regime: str) -> None:
  """Tree 1 must be exactly the bivariate study's copula, so the two compare."""
  x = torch.linspace(0.0, 1.0, 51, dtype=torch.float64)
  spec = scenarios.TAU_SCENARIOS[regime]
  assert spec.tau_of_x is not None
  torch.testing.assert_close(
    vine_scenarios.edge_tau(regime, 1, x), spec.tau_of_x(x)
  )


def test_the_truth_density_never_underflows_at_its_own_samples() -> None:
  """A test point with f ~ 0 under the truth makes L1 and ISE meaningless.

  Pins the shared ``TAU_HI`` cap against the saturation a band near 0.9
  exposed on dimension-5 truths.
  """
  x_axis = scenarios.conditional_x_axis(5)
  for rep in range(5):
    truth = vine_scenarios.draw_truth(5, 1000 + rep)
    points = vine_scenarios.eval_points(truth, x_axis, 400, rep)
    for c in points.copula_pdf_true:
      assert float(c.min()) > 1e-10


def test_regimes_are_the_conditional_scenarios() -> None:
  """An unconditional regime has no tau(x) and would crash a truth draw."""
  assert vine_scenarios.REGIMES
  for name in vine_scenarios.REGIMES:
    assert scenarios.TAU_SCENARIOS[name].conditional


@pytest.mark.parametrize("d", [3, 5])
def test_draw_truth_is_seeded_and_well_formed(d: int) -> None:
  """A truth that varies across runs of one seed would make reps irreproducible."""
  a = vine_scenarios.draw_truth(d, 11)
  b = vine_scenarios.draw_truth(d, 11)
  assert a.to_dict() == b.to_dict()
  assert a.d == d
  assert [len(row) for row in a.edges] == list(range(d - 1, 0, -1))
  assert all(e.tree == t + 1 for t, row in enumerate(a.edges) for e in row)
  assert a.to_dict() != vine_scenarios.draw_truth(d, 12).to_dict()


def test_sample_is_seeded_and_interior() -> None:
  """An exact 0 or 1 in u is refused by every NPCC pair; seeds must repeat."""
  truth = vine_scenarios.draw_truth(3, 5)
  y1, u1, x1 = vine_scenarios.sample(truth, 40, 7)
  y2, _, _ = vine_scenarios.sample(truth, 40, 7)
  assert y1.shape == (40, 3)
  assert u1.shape == (40, 3)
  assert x1.shape == (40, 1)
  torch.testing.assert_close(y1, y2)
  assert bool(((u1 > 0.0) & (u1 < 1.0)).all())
  torch.testing.assert_close(
    y1, vine_scenarios.to_data_scale(truth, u1, x1.reshape(-1))
  )


def test_margins_shift_with_x() -> None:
  """Margins constant in x would let a covariate-blind baseline match them."""
  truth = vine_scenarios.draw_truth(5, 3)
  assert len(truth.slopes) == 5
  assert all(
    -vine_scenarios.MU_SLOPE <= a <= vine_scenarios.MU_SLOPE
    for a in truth.slopes
  )
  mu = vine_scenarios.margin_mean(
    truth, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
  )
  torch.testing.assert_close(mu[1], torch.zeros(5, dtype=torch.float64))
  torch.testing.assert_close(
    mu[2] - mu[0], torch.tensor(truth.slopes, dtype=torch.float64)
  )


def test_eval_points_carry_the_true_densities() -> None:
  """Both KL baselines must be the truth's densities at the x drawn under.

  The joint one is checked against Sklar's factorization computed
  independently, through the Gaussian margins' closed form.
  """
  truth = vine_scenarios.draw_truth(3, 5)
  x_axis = torch.tensor([0.2, 0.8], dtype=torch.float64)
  points = vine_scenarios.eval_points(truth, x_axis, 30, 9)
  for k, x_val in enumerate(x_axis):
    vine = vine_scenarios.true_vine(truth, float(x_val))
    copula = np.asarray(vine.pdf(points.u[k].numpy()))
    np.testing.assert_allclose(points.copula_pdf_true[k].numpy(), copula)
    mu = torch.tensor(truth.slopes, dtype=torch.float64) * (float(x_val) - 0.5)
    z = points.y[k] - mu
    torch.testing.assert_close(torch.special.ndtr(z), points.u[k])
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    expected = torch.from_numpy(copula) * phi.prod(dim=1)
    torch.testing.assert_close(points.pdf_true[k], expected)
    assert points.y[k].shape == (30, 3)
