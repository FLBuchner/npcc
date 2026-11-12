"""E11's chain-rule copula density: its factors, features and refusals."""

from __future__ import annotations

import math
from typing import cast

import pytest
import torch

from npcc.core import (
  FitControlsRosenblattBicop,
  QuantileTableConfig,
  RosenblattBicop,
  Transform,
  create_backend,
)
from npcc.experiments.chain import ChainCopula, draw_order


def _controls(
  feature_transform: Transform = "identity",
) -> FitControlsRosenblattBicop:
  return FitControlsRosenblattBicop(
    backend="linear-gaussian",
    transform="probit",
    feature_transform=feature_transform,
    device="cpu",
  )


def _sample(n: int, d: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
  """Dependent copula data with ``x`` shifting every score, and ``x``."""
  gen = torch.Generator().manual_seed(seed)
  x = torch.rand(n, 1, generator=gen, dtype=torch.float64)
  z = torch.randn(n, d, generator=gen, dtype=torch.float64)
  z = z + 0.8 * z[:, [0]] + (x - 0.5)
  return cast("torch.Tensor", torch.special.ndtr(z / z.std(dim=0))), x


@pytest.mark.parametrize("feature_transform", ["identity", "probit"])
@pytest.mark.parametrize(("order", "direction"), [((1, 2), 0), ((2, 1), 1)])
def test_an_order_of_two_is_a_rosenblatt_direction(
  register_linear_gaussian_backend: None,
  order: tuple[int, int],
  direction: int,
  feature_transform: Transform,
) -> None:
  """At d = 2 the chain along an order is the pair copula's direction.

  A factor is meant to be the estimator a Rosenblatt direction is, with more
  features. A different clamp, feature transform, response or Jacobian would
  make E11 compare the vine against something other than its own building
  block. The test points include two inside ``eps`` of the boundary, where
  only the clamp decides the answer.

  To float tolerance rather than bit for bit: the linear-Gaussian backend's
  least-squares fit rounds with the alignment of its buffers, so one
  ``RosenblattBicop`` fitted twice on the same data already differs in the
  last bits (about 1e-14 here).
  """
  u, x = _sample(200, 2)
  test, x_test = _sample(50, 2, seed=1)
  test[0] = torch.tensor([1e-9, 0.5], dtype=torch.float64)
  test[1] = torch.tensor([0.3, 1.0 - 1e-9], dtype=torch.float64)
  controls = _controls(feature_transform)
  torch.manual_seed(3)
  pair = RosenblattBicop(controls).fit(u, x=x)
  torch.manual_seed(3)
  chain = ChainCopula(order, controls).fit(u, x=x, seeds=[3])
  expected = pair.pdf_by_direction(test, x=x_test)[direction]
  with torch.inference_mode():
    got = chain.pdf(test, x=x_test)
  torch.testing.assert_close(got, expected, rtol=1e-10, atol=0)


def test_each_factor_conditions_on_the_earlier_scores_and_x(
  register_linear_gaussian_backend: None,
) -> None:
  """Factor k is the response u_pi(k) given (u_pi(1), ..., u_pi(k-1), x).

  Built by hand from the registry for the order (3, 1, 2): a factor read off
  the wrong columns, or in another column order, would condition on other
  information than the chain rule asks for, and the density it multiplies
  into would not be ``c``'s.
  """
  u, x = _sample(150, 3)
  test, x_test = _sample(40, 3, seed=2)
  chain = ChainCopula([3, 1, 2], _controls()).fit(u, x=x)
  expected = []
  for response, given in ((0, [2]), (1, [2, 0])):
    factor = create_backend(
      "linear-gaussian",
      transform="probit",
      quantile_table_config=QuantileTableConfig(),
      eps=1e-6,
      device="cpu",
      batch_size=None,
      backend_kwargs={},
    )
    factor.fit(u[:, response], x=torch.cat([u[:, given], x], dim=1))
    expected.append(
      factor.pdf(test[:, response], x=torch.cat([test[:, given], x_test], 1))
    )
  factors = chain.factor_pdfs(test, x=x_test)
  assert factors.shape == (2, 40)
  torch.testing.assert_close(factors, torch.stack(expected), rtol=1e-10, atol=0)
  torch.testing.assert_close(
    chain.logpdf(test, x=x_test), factors.log().sum(dim=0), rtol=0, atol=0
  )


def test_the_first_variable_is_left_uniform(
  register_linear_gaussian_backend: None,
) -> None:
  """There is no factor for pi(1), so its margin stays uniform.

  On the copula scale U_pi(1) is uniform and its density 1. A chain that
  fitted one to the data -- here a first column far from uniform -- would
  carry that misfit into every point. Integrating the density over
  u_pi(2), on the probit scale, must give 1 at every u_pi(1).
  """
  gen = torch.Generator().manual_seed(4)
  z = torch.randn(300, 2, generator=gen, dtype=torch.float64)
  first = 0.4 * z[:, 0] - 0.8
  u = torch.special.ndtr(torch.stack([first, 0.6 * first + 0.8 * z[:, 1]], 1))
  chain = ChainCopula([1, 2], _controls()).fit(u)
  assert len(chain.factors_) == 1
  t = torch.linspace(-4.5, 4.5, 4001, dtype=torch.float64)
  phi = torch.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
  for a in (0.05, 0.3, 0.8):
    grid = torch.stack([torch.full_like(t, a), torch.special.ndtr(t)], dim=1)
    total = torch.trapezoid(chain.pdf(grid) * phi, t)
    assert float(total) == pytest.approx(1.0, abs=1e-4)


def test_an_order_is_a_uniform_permutation_of_its_seed() -> None:
  """An order must be a function of its seed alone, and cover every variable.

  A resumed or rerun cell would otherwise mix other orders than it scored;
  an order missing a variable would leave a margin of the copula unmodeled.
  """
  orders = [draw_order(5, s) for s in (11, 12, 11)]
  assert orders[0] == orders[2]
  assert orders[0] != orders[1]
  assert all(sorted(o) == [1, 2, 3, 4, 5] for o in orders)
  firsts = {draw_order(3, s)[0] for s in range(60)}
  assert firsts == {1, 2, 3}


@pytest.mark.parametrize("order", [[1], [1, 1], [0, 1], [1, 3]])
def test_an_order_must_be_a_permutation(
  register_linear_gaussian_backend: None, order: list[int]
) -> None:
  """A repeated or missing label would drop or double a factor silently."""
  with pytest.raises(ValueError, match="order must be a permutation"):
    ChainCopula(order, _controls())


def test_a_projection_is_refused_rather_than_dropped(
  register_hermetic_backends: None,
) -> None:
  """A chain factor has no second direction for a projection to balance.

  Accepting ``sinkhorn_iters`` and ignoring it would report a projected
  configuration for an unprojected fit.
  """
  controls = FitControlsRosenblattBicop(
    backend="normal-quantile", sinkhorn_iters=5, device="cpu"
  )
  with pytest.raises(ValueError, match="cannot honor sinkhorn_iters"):
    ChainCopula([1, 2], controls)


def test_a_fit_checks_its_seeds_and_its_data(
  register_linear_gaussian_backend: None,
) -> None:
  """A seed per factor, a column per variable, scores strictly inside (0, 1).

  A short seed list would seed some factors and not others; a column too
  few would read one variable as another; and an exact 0 or 1 is a defect
  upstream that a clamp would hide, as a pair copula refuses it.
  """
  u, x = _sample(60, 3)
  chain = ChainCopula([1, 2, 3], _controls())
  with pytest.raises(RuntimeError, match="must be fitted"):
    chain.pdf(u)
  with pytest.raises(ValueError, match="one seed per factor"):
    chain.fit(u, x=x, seeds=[1])
  with pytest.raises(ValueError, match=r"u must have shape \(n, 3\)"):
    chain.fit(u[:, :2], x=x)
  bad = u.clone()
  bad[0, 1] = 1.0
  with pytest.raises(ValueError, match="strictly inside"):
    chain.fit(bad, x=x)
