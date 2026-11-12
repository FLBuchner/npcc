"""Tests for ``npcc.experiments.scenarios`` (sampling + ground truth)."""

from __future__ import annotations

import functools
import itertools
import math

import numpy as np
import pytest
import pyvinecopulib as pv
import torch

from npcc.experiments import scenarios, vine_scenarios


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
    cop = scenarios.bicop(scenarios.FAMILIES[family], float(tau))
    expected_pdf = torch.from_numpy(cop.pdf(uv.numpy()))
    torch.testing.assert_close(
      truth["pdf"][:, x_idx], expected_pdf, atol=1e-10, rtol=1e-7
    )
  assert truth["pdf"].shape == grid.shape


def test_ground_truth_unconditional_matches_pyvinecopulib() -> None:
  family, scenario = "frank", "uncond70"
  truth = scenarios.ground_truth(family, scenario)
  grid = scenarios.eval_grid(scenario)
  cop = scenarios.bicop(scenarios.FAMILIES[family], 0.70)
  expected = torch.from_numpy(cop.hfunc1(grid.uv.numpy()))
  torch.testing.assert_close(truth["hfunc1"], expected, atol=1e-10, rtol=1e-7)
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


def test_one_parameter_draws_and_truths_are_pinned() -> None:
  """A one-parameter cell's data and truth must not move as scenarios are added.

  The bivariate study's checkpoints are resumed and aggregated against them,
  and nothing else fails when a new scenario reorders a draw or a lookup: the
  tables quietly mix two data-generating processes. The torch draws are pinned
  exactly; pyvinecopulib's, to a relative 1e-12, which a re-seeded stream
  misses by far while a last-bit difference between libm builds does not.
  """
  # The seed of the bivariate study's clayton/linear/n=200/rep-0 cell.
  u, v, x = scenarios.sample("clayton", "linear", 200, 1393628405)
  assert x is not None
  assert u[[0, 99, 199]].tolist() == [
    0.36903783062016426,
    0.04958093012268827,
    0.05907954207648691,
  ]
  assert v[[0, 99, 199]].tolist() == pytest.approx(
    [0.890936073538436, 0.1309041919978885, 0.10062644694104396], rel=1e-12
  )
  u, v, x = scenarios.sample("gaussian", "uncond35", 50, 11)
  assert x is None
  assert u[[0, 49]].tolist() == pytest.approx(
    [0.3594531842973083, 0.8447920146863908], rel=1e-12
  )
  assert v[[0, 49]].tolist() == pytest.approx(
    [0.22702375684931786, 0.703554002758022], rel=1e-12
  )

  # Off the diagonal, so hfunc1 and hfunc2 differ and a swap would show.
  truth = scenarios.ground_truth("gaussian", "sin")
  expected = {
    "pdf": [0.007300112844623042, 0.8896458240587466],
    "hfunc1": [0.9997297322788317, 0.10552937214472219],
    "hfunc2": [0.00022975930128815714, 0.8944706278552779],
  }
  for q, values in expected.items():
    got = [float(truth[q][75, 2]), float(truth[q][342, 7])]
    assert got == pytest.approx(values, rel=1e-12), q
  truth = scenarios.ground_truth("clayton", "uncond35")
  expected = {
    "pdf": [0.47000070069529176, 0.27830711065756336],
    "hfunc1": [0.9129749354584407, 0.01702591098566543],
    "hfunc2": [0.04151633028206299, 0.9689781411645959],
  }
  for q, values in expected.items():
    got = [float(truth[q][75]), float(truth[q][342])]
    assert got == pytest.approx(values, rel=1e-12), q


# ---------------------------------------------------------------------------
# Shape families
# ---------------------------------------------------------------------------

_MIX = "clayton-gumbel-mix"
_STUDENT = "student-varying-nu"


def _stated(family: str, x: float) -> list[tuple[float, pv.Bicop]]:
  """A shape family at ``x`` as the study design states it.

  Written out here rather than read from the module, so a slip in a weight,
  a parameter or ``nu(x)`` there has something to disagree with: Clayton and
  Gumbel at theta = 2 (tau = 0.5 for both) mixed with weight ``x``, and a
  Student t with ``rho = sin(pi / 4)`` and
  ``nu = exp(log 2 + x (log 50 - log 2))``.
  """
  if family == _MIX:
    theta = np.array([[2.0]])
    return [
      (1.0 - x, pv.Bicop(family=pv.BicopFamily.clayton, parameters=theta)),
      (x, pv.Bicop(family=pv.BicopFamily.gumbel, parameters=theta)),
    ]
  nu = math.exp(math.log(2.0) + x * (math.log(50.0) - math.log(2.0)))
  params = np.array([[math.sin(math.pi / 4.0)], [nu]])
  return [(1.0, pv.Bicop(family=pv.BicopFamily.student, parameters=params))]


def _stated_value(family: str, x: float, method: str, uv: np.ndarray) -> float:
  """The stated mixture's ``cdf``, ``hfunc1`` or ``pdf`` at one point."""
  return sum(
    weight * float(getattr(cop, method)(uv)[0])
    for weight, cop in _stated(family, x)
  )


def _truth_at(
  family: str, x: float, uv: torch.Tensor
) -> dict[str, torch.Tensor]:
  """``ground_truth`` of a shape family at one ``x``, at the points ``uv``."""
  x_axis = torch.tensor([x], dtype=torch.float64)
  grid = scenarios.EvalGrid(
    u_flat=uv[:, 0],
    v_flat=uv[:, 1],
    x_flat=x_axis.expand(uv.shape[0]).reshape(-1, 1),
    shape=(uv.shape[0], 1),
    conditional=True,
    x_axis=x_axis,
    u_axis=uv[:, 0],
    v_axis=uv[:, 1],
  )
  truth = scenarios.ground_truth(family, scenarios.IMPLIED, grid)
  return {q: t[:, 0] for q, t in truth.items()}


def _tanh_sinh(h: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
  """Tanh-sinh nodes and weights on ``(0, 1)``, with step ``h``.

  The rule converges double-exponentially for an integrand analytic inside
  the interval, whatever it does at the ends -- which is where each copula
  here is singular or has a kink.
  """
  t = torch.arange(-4.0, 4.0 + h / 2.0, h, dtype=torch.float64)
  a = torch.pi / 2.0 * torch.sinh(t)
  nodes = 0.5 * (1.0 + torch.tanh(a))
  weights = h * torch.pi / 4.0 * torch.cosh(t) / torch.cosh(a) ** 2
  keep = (nodes > 0.0) & (nodes < 1.0)
  return nodes[keep], weights[keep]


@pytest.mark.parametrize("family", sorted(scenarios.SHAPE_FAMILIES))
def test_shape_family_truth_is_the_stated_mixture(family: str) -> None:
  """The truth must be the mixture the study design states, at each scored x.

  Every estimator is scored against it, so a weight of ``1 - x`` on the wrong
  component, or ``nu`` read off the wrong end of its range, would rank the
  methods on a scenario the paper does not describe -- and still produce a
  smooth, plausible surface.
  """
  grid = scenarios.eval_grid(scenarios.IMPLIED)
  assert grid.x_axis is not None
  truth = scenarios.ground_truth(family, scenarios.IMPLIED, grid)
  pairs = grid.uv.reshape(*grid.shape, 2)[:, 0].numpy()
  for x_idx, x in enumerate(grid.x_axis.tolist()):
    for q in scenarios.QUANTITIES:
      stated = np.sum(
        [w * getattr(cop, q)(pairs) for w, cop in _stated(family, x)], axis=0
      )
      torch.testing.assert_close(
        truth[q][:, x_idx], torch.from_numpy(stated), rtol=1e-10, atol=1e-12
      )


@pytest.mark.parametrize("family", sorted(scenarios.SHAPE_FAMILIES))
def test_shape_family_density_integrates_to_one(family: str) -> None:
  """Each slice of the true density must have mass one on the unit square.

  Mixture weights that do not sum to one still give a smooth, positive
  surface, which only its mass gives away. Tanh-sinh quadrature on 127 nodes
  a side integrates the corner singularities of the Clayton, Gumbel and t
  densities to about 1e-9.
  """
  nodes, weights = _tanh_sinh()
  uu, vv = torch.meshgrid(nodes, nodes, indexing="ij")
  uv = torch.column_stack([uu.reshape(-1), vv.reshape(-1)])
  area = torch.outer(weights, weights).reshape(-1)
  for x in (0.1, 0.5, 0.9):
    mass = float((_truth_at(family, x, uv)["pdf"] * area).sum())
    assert mass == pytest.approx(1.0, abs=1e-8), x


@pytest.mark.parametrize("family", sorted(scenarios.SHAPE_FAMILIES))
def test_shape_family_h_functions_integrate_the_density(family: str) -> None:
  """``hfunc1`` and ``hfunc2`` must be the partial integrals of the density.

  ``hfunc1(a, b) = int_0^b c(a, s) ds`` and ``hfunc2(a, b) = int_0^a c(s, b)
  ds``. A truth that mixed its h-functions with other weights than its
  density, or swapped the two, would score every estimator's h-functions
  against the wrong target while the density looked right. The points sit
  off the diagonal, where the two differ.
  """
  nodes, weights = _tanh_sinh()
  for x in (0.1, 0.5, 0.9):
    for a, b in ((0.1, 0.2), (0.5, 0.7), (0.9, 0.8), (0.2, 0.9), (0.8, 0.1)):
      along_v = torch.column_stack([torch.full_like(nodes, a), b * nodes])
      along_u = torch.column_stack([a * nodes, torch.full_like(nodes, b)])
      h1 = b * float((_truth_at(family, x, along_v)["pdf"] * weights).sum())
      h2 = a * float((_truth_at(family, x, along_u)["pdf"] * weights).sum())
      at = _truth_at(family, x, torch.tensor([[a, b]], dtype=torch.float64))
      assert float(at["hfunc1"][0]) == pytest.approx(h1, abs=1e-9)
      assert float(at["hfunc2"][0]) == pytest.approx(h2, abs=1e-9)


def _ks(z: np.ndarray) -> float:
  """``sqrt(n)`` times the Kolmogorov-Smirnov distance of ``z`` to U(0, 1)."""
  z = np.sort(z)
  n = z.shape[0]
  rank = np.arange(1, n + 1)
  gap = max(float(np.max(rank / n - z)), float(np.max(z - (rank - 1) / n)))
  return gap * math.sqrt(n)


def _corner_probability(family: str, x: float, q: float, corner: str) -> float:
  """A corner's probability under the stated copula at ``x``.

  ``lower`` is ``u, v < q``; ``upper`` is ``u, v > 1 - q``; ``discordant``
  is ``u < q, v > 1 - q`` together with its mirror image.
  """
  cdf = functools.partial(_stated_value, family, x, "cdf")
  if corner == "lower":
    return cdf(np.array([[q, q]]))
  if corner == "upper":
    return 2.0 * q - 1.0 + cdf(np.array([[1.0 - q, 1.0 - q]]))
  return 2.0 * q - cdf(np.array([[q, 1.0 - q]])) - cdf(np.array([[1.0 - q, q]]))


def _in_corner(
  u: np.ndarray, v: np.ndarray, q: float, corner: str
) -> np.ndarray:
  """Which points ``(u, v)`` fall in the corner :func:`_corner_probability` names."""
  if corner == "lower":
    return (u < q) & (v < q)
  if corner == "upper":
    return (u > 1.0 - q) & (v > 1.0 - q)
  return ((u < q) & (v > 1.0 - q)) | ((u > 1.0 - q) & (v < q))


@pytest.mark.parametrize(
  ("family", "q", "corners"),
  [
    # Clayton and Gumbel at one tau differ in which concordant corner they
    # load.
    (_MIX, 0.05, ("lower", "upper")),
    # At a fixed rho, a t copula loads its discordant corners as nu falls.
    (_STUDENT, 0.2, ("discordant",)),
  ],
)
def test_shape_family_sample_follows_the_truth_in_its_corners(
  family: str, q: float, corners: tuple[str, ...]
) -> None:
  """A draw must come from the stated copula at its own ``x``.

  The families differ from their neighbors mainly in the corners, so the
  check counts the draws in each corner that tells them apart: below ``q``
  or above ``1 - q`` in both coordinates for the mixture, and below in one
  and above in the other for the Student family. In each half of the
  covariate range, each count must match its expectation under the stated
  copula within four standard deviations. At 20,000 draws, swapping the
  mixture weights, drawing every row from one component, or picking the
  component with the uniform that also gives ``u`` each moves a count by six
  or more; reversing ``nu(x)``, or holding ``nu`` at 2 or at 50, by five or
  more. The probability integral transform ``hfunc1(u, v | x)`` must also
  pass a Kolmogorov-Smirnov test of uniformity at the 0.001 level.
  """
  n = 20_000
  u, v, x = scenarios.sample(family, scenarios.IMPLIED, n, 2026)
  assert x is not None
  u_np, v_np, x_np = u.numpy(), v.numpy(), x[:, 0].numpy()
  pit = np.array(
    [
      _stated_value(family, x_i, "hfunc1", np.array([[u_i, v_i]]))
      for u_i, v_i, x_i in zip(u_np, v_np, x_np, strict=True)
    ]
  )
  assert _ks(pit) < 1.95

  for corner in corners:
    p = np.array([_corner_probability(family, x_i, q, corner) for x_i in x_np])
    hits = _in_corner(u_np, v_np, q, corner)
    for half in (x_np < 0.5, x_np >= 0.5):
      expected = float(p[half].sum())
      sd = math.sqrt(float((p[half] * (1.0 - p[half])).sum()))
      assert abs(int(hits[half].sum()) - expected) < 4.0 * sd, corner


def test_mixture_cross_integrals_match_quadrature() -> None:
  """The constants in the mixture's closed-form tau must be its integrals.

  ``int int C_A dC_B = 1/2 - int int h1_A h2_B du dv`` has a bounded
  integrand, which tanh-sinh quadrature integrates to about 1e-14: the same
  nodes give each component's known ``int int C dC = (tau + 1) / 4``.
  """
  nodes, weights = _tanh_sinh()
  uu, vv = torch.meshgrid(nodes, nodes, indexing="ij")
  uv = torch.column_stack([uu.reshape(-1), vv.reshape(-1)]).numpy()
  area = torch.outer(weights, weights).reshape(-1).numpy()
  (_, clayton), (_, gumbel) = _stated(_MIX, 0.5)

  def cross(a: pv.Bicop, b: pv.Bicop) -> float:
    return 0.5 - float(np.sum(a.hfunc1(uv) * b.hfunc2(uv) * area))

  assert cross(clayton, clayton) == pytest.approx(0.375, abs=1e-12)
  assert cross(gumbel, gumbel) == pytest.approx(0.375, abs=1e-12)
  i_cg, i_gc = scenarios.MIX_CROSS_INTEGRALS
  assert cross(clayton, gumbel) == pytest.approx(i_cg, abs=1e-12)
  assert cross(gumbel, clayton) == pytest.approx(i_gc, abs=1e-12)


@pytest.mark.parametrize("x", [0.2, 0.5, 0.8])
def test_mixture_tau_matches_monte_carlo(x: float) -> None:
  """The mixture's closed-form tau must match draws from the mixture itself.

  The quadrature test pins the cross-integrals; this pins the formula they
  enter, on 400,000 draws at a fixed ``x``. Pure Clayton and Gumbel draws on
  the same uniforms, each with tau exactly 0.5, serve as control variates and
  bring the standard error to about 2e-4. The tolerance of 8e-4 therefore
  resolves the dip below 0.5 -- 1.2e-3 at ``x = 0.2``, 1.8e-3 at ``x = 1/2``
  -- that a tau linear in the weight would miss.
  """
  u, w, r = np.random.default_rng(round(100 * x)).random((3, 400_000))
  (_, clayton), (_, gumbel) = _stated(_MIX, x)
  uw = np.column_stack([u, w])
  v_c, v_g = clayton.hinv1(uw), gumbel.hinv1(uw)
  v = np.where(r < 1.0 - x, v_c, v_g)
  tau_c = float(pv.utils.wdm(u, v_c, "tau"))
  tau_g = float(pv.utils.wdm(u, v_g, "tau"))
  estimate = float(pv.utils.wdm(u, v, "tau"))
  estimate -= (1.0 - x) * (tau_c - 0.5) + x * (tau_g - 0.5)

  tau_of_x = scenarios.SHAPE_FAMILIES[_MIX].tau_of_x
  closed = float(tau_of_x(torch.tensor([x], dtype=torch.float64))[0])
  assert estimate == pytest.approx(closed, abs=8e-4)


def test_student_family_moves_its_tails_at_tau_one_half() -> None:
  """The Student family must vary its degrees of freedom and nothing else.

  ``nu`` runs from exactly 2 at ``x = 0`` to exactly 50 at ``x = 1``,
  pyvinecopulib's upper bound, rising all the way; the tau the study scores
  against stays 0.5, and so does pyvinecopulib's own tau of every component.
  """
  assert scenarios.student_nu(0.0) == 2.0
  assert scenarios.student_nu(1.0) == 50.0
  family = scenarios.SHAPE_FAMILIES[_STUDENT]
  x = torch.linspace(0.0, 1.0, 21, dtype=torch.float64)
  torch.testing.assert_close(family.tau_of_x(x), torch.full_like(x, 0.5))
  nus = []
  for x_val in x.tolist():
    [(weight, cop)] = family.components(x_val)
    assert weight == 1.0
    assert cop.family == pv.BicopFamily.student
    assert cop.parameters_to_tau(cop.parameters) == pytest.approx(0.5)
    nus.append(float(cop.parameters[1, 0]))
  assert nus[0] == 2.0
  assert nus[-1] == 50.0
  assert all(a < b for a, b in itertools.pairwise(nus))


@pytest.mark.parametrize(
  ("family", "scenario"), [(_MIX, "linear"), ("clayton", scenarios.IMPLIED)]
)
def test_a_shape_family_takes_the_implied_regime_and_nothing_else(
  family: str, scenario: str
) -> None:
  """A shape family under a tau regime, or a one-parameter family under
  ``implied``, defines no copula: each is refused rather than half-built."""
  with pytest.raises(ValueError, match="tau_scenario must be 'implied'"):
    scenarios.sample(family, scenario, 10, 0)
  with pytest.raises(ValueError, match="tau_scenario must be 'implied'"):
    scenarios.ground_truth(family, scenario)


def test_shape_families_stay_out_of_the_registries_the_vine_draws_from() -> (
  None
):
  """The vine study draws every edge's family and regime by position from
  ``FAMILIES`` and the conditional ``TAU_SCENARIOS``. A shape family or
  ``implied`` in either would redraw every vine truth, with no signature
  change to refuse a resume over the old ones."""
  assert not set(scenarios.SHAPE_FAMILIES) & set(scenarios.FAMILIES)
  assert scenarios.IMPLIED not in scenarios.TAU_SCENARIOS
  assert scenarios.IMPLIED not in vine_scenarios.REGIMES


def _kendall_tau(u: torch.Tensor, v: torch.Tensor) -> float:
  """Kendall's tau-a of a sample, by its pairwise concordance."""
  du = torch.sign(u[:, None] - u[None, :])
  dv = torch.sign(v[:, None] - v[None, :])
  n = u.shape[0]
  return float((du * dv).sum() / (n * (n - 1)))


@pytest.mark.parametrize(
  ("family", "scenario"),
  [("clayton", "linear"), ("gumbel", "sin"), ("clayton-gumbel-mix", "implied")],
)
def test_holdout_draws_follow_the_copula_at_each_x(
  family: str, scenario: str
) -> None:
  """The draws at each ``x`` must come from the true copula at that ``x``.

  The Monte Carlo KL averages ``log c - log c_hat`` over them; draws from
  another ``x``'s copula, or with ``u`` and ``v`` swapped on an asymmetric
  family, would score every estimator against the wrong distribution, and
  the KL would no longer be one.
  """
  x_axis = torch.tensor([0.1, 0.9], dtype=torch.float64)
  grid = scenarios.holdout_grid(family, scenario, x_axis, 2000, [11, 12])
  assert grid.shape == (2000, 2)
  assert grid.x_flat is not None
  u = grid.u_flat.reshape(grid.shape)
  v = grid.v_flat.reshape(grid.shape)
  x = grid.x_flat.reshape(grid.shape)
  if family in scenarios.SHAPE_FAMILIES:
    target = scenarios.SHAPE_FAMILIES[family].tau_of_x(x_axis)
  else:
    tau_of_x = scenarios.TAU_SCENARIOS[scenario].tau_of_x
    assert tau_of_x is not None
    target = tau_of_x(x_axis)
  for j in range(2):
    assert torch.equal(x[:, j], x_axis[j].expand(2000))
    assert _kendall_tau(u[:, j], v[:, j]) == pytest.approx(
      float(target[j]), abs=0.05
    )
  # The same seed at either x: the copula, not the stream, separates them.
  same = scenarios.holdout_grid("clayton", "linear", x_axis, 2000, [11, 11])
  tau_lo, tau_hi = (
    _kendall_tau(*(t.reshape(same.shape)[:, j] for t in same.uv.T))
    for j in range(2)
  )
  assert tau_hi - tau_lo > 0.3


def test_holdout_draws_are_seeded_per_x() -> None:
  """Each ``x``'s draws must depend on its own seed alone.

  Every estimator of a cell is scored on the same draws, in E1 and E4 alike,
  only if they are a function of the seeds; and an ``x`` whose draws moved
  with another ``x``'s seed would move whenever the grid of ``x`` did.
  """
  x_axis = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
  a = scenarios.holdout_grid("frank", "sin", x_axis, 50, [1, 2, 3])
  b = scenarios.holdout_grid("frank", "sin", x_axis, 50, [1, 2, 3])
  c = scenarios.holdout_grid("frank", "sin", x_axis, 50, [1, 9, 3])
  assert torch.equal(a.uv, b.uv)
  a_uv, c_uv = (g.uv.reshape(*g.shape, 2) for g in (a, c))
  assert torch.equal(a_uv[:, [0, 2]], c_uv[:, [0, 2]])
  assert not torch.equal(a_uv[:, 1], c_uv[:, 1])
  assert ((a.uv > 0.0) & (a.uv < 1.0)).all()


def test_ground_truth_reads_each_x_at_its_own_points() -> None:
  """The truth at draws must be each ``x``'s copula at that ``x``'s pairs.

  A product grid holds the same pairs at every ``x``, so evaluating every
  column at the first one's pairs went unnoticed there; at draws it would
  pair one ``x``'s density with another's points.
  """
  x_axis = torch.tensor([0.1, 0.6], dtype=torch.float64)
  for family, scenario in (
    ("joe", "quadratic"),
    ("student-varying-nu", "implied"),
  ):
    grid = scenarios.holdout_grid(family, scenario, x_axis, 64, [5, 6])
    truth = scenarios.ground_truth(family, scenario, grid)
    uv = grid.uv.reshape(*grid.shape, 2)
    for j, x in enumerate(x_axis.tolist()):
      if family in scenarios.SHAPE_FAMILIES:
        ((_, cop),) = scenarios.SHAPE_FAMILIES[family].components(x)
      else:
        tau_of_x = scenarios.TAU_SCENARIOS[scenario].tau_of_x
        assert tau_of_x is not None
        tau = float(tau_of_x(torch.tensor([x], dtype=torch.float64))[0])
        cop = scenarios.bicop(scenarios.FAMILIES[family], tau)
      expected = torch.from_numpy(cop.pdf(uv[:, j].contiguous().numpy()))
      torch.testing.assert_close(truth["pdf"][:, j], expected)


def test_an_unconditional_holdout_draws_from_its_one_copula() -> None:
  """An unconditional scenario's draws need no covariate and one seed.

  It is scored like the conditional ones, on one column of draws; a grid with
  a covariate would hand every estimator an ``x`` it was not fitted on.
  """
  grid = scenarios.holdout_grid("gaussian", "uncond70", None, 2000, [3])
  assert grid.shape == (2000,)
  assert grid.x_flat is None
  assert not grid.conditional
  assert _kendall_tau(grid.u_flat, grid.v_flat) == pytest.approx(0.7, abs=0.05)


@pytest.mark.parametrize(
  ("x_axis", "n", "seeds", "match"),
  [
    (torch.tensor([0.5]), 1, [1], "n must be"),
    (None, 10, [1], "x_axis must be given"),
    (torch.tensor([0.2, 0.8]), 10, [1], "one seed per covariate value"),
  ],
)
def test_holdout_grid_refuses_what_it_cannot_draw(
  x_axis: torch.Tensor | None, n: int, seeds: list[int], match: str
) -> None:
  """A malformed request must fail before any draw, naming what is wrong.

  One draw has no standard error, a conditional scenario without its
  covariate would be drawn at no ``x``, and a missing seed would leave an
  ``x`` without a stream of its own.
  """
  with pytest.raises(ValueError, match=match):
    scenarios.holdout_grid("clayton", "linear", x_axis, n, seeds)
