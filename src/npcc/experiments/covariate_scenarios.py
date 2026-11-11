"""Copula scenarios with fifteen covariates of mixed type (E9).

The covariates are those of Vatter and Nagler (2018, JCGS), §4.1: ``b_1,
..., b_5`` iid Bernoulli(1/2), ``z_1, ..., z_5`` iid standard normal and
``t_1, ..., t_5`` iid uniform on ``(0, 1)``, all independent. The covariate
matrix has their :data:`COVARIATES` columns in that order, the ``b`` as
``0.0`` / ``1.0``, and every estimator receives all of them unless it is
fitted on a subset.

Each scenario (:data:`SCENARIOS`) is a copula of one of E1's families whose
parameter depends on some of the covariates:

- ``gampcc``: Vatter and Nagler's partially linear predictor,
  ``eta = beta_b'b + beta_z'z + s_1(t_1) + s_2(t_2) + s_3(t_3)``, which
  calibrates Kendall's tau, ``tau = tanh(eta / 2)``, for either family: the
  link of gamCopula's ``tau = TRUE``, the default its study's true model was
  built with. The Gaussian then has ``rho = sin(pi tau / 2)``; the Clayton is
  gamCopula's double Clayton type I, the Clayton at ``theta = 2 tau / (1 -
  tau)`` where ``tau > 0`` and the Clayton rotated by 90 degrees at that of
  ``|tau|`` below. Kendall's tau therefore changes sign, which it does
  nowhere in E1. Nine covariates have an effect, six none.
- ``interaction``: Kendall's tau is E1's ``linear`` regime in ``t_1`` where
  ``b_1 = 0`` and E1's ``sin`` regime where ``b_1 = 1``, so an additive
  model without a ``t_1 x b_1`` term is misspecified. Two covariates have an
  effect.
- ``constant``: Kendall's tau is 0.5, and no covariate has an effect.

Estimates are scored at a fixed set of covariate points (:func:`points`), the
same for every family, ``n``, repetition and estimator of a scenario: at each,
on E1's ``uv`` grid (:func:`metric_grid`) and at draws from the true copula
there (:func:`holdout_grid`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pyvinecopulib as pv
import torch

from npcc.experiments import scenarios
from npcc.experiments.scenarios import (
  FAMILIES,
  TAU_SCENARIOS,
  EvalGrid,
  bicop,
  interior_axis,
)

#: The scenarios, in the order their cells sweep.
SCENARIOS: tuple[str, ...] = ("gampcc", "interaction", "constant")

#: The families a scenario can take: the ``gampcc`` link is gamCopula's for
#: these two.
SCENARIO_FAMILIES: tuple[str, ...] = ("clayton", "gaussian")

#: The covariates per type, and the covariate matrix's columns in order.
PER_TYPE: int = 5
COVARIATES: tuple[str, ...] = tuple(
  f"{kind}{j + 1}" for kind in ("b", "z", "t") for j in range(PER_TYPE)
)

#: The ``gampcc`` predictor's linear coefficients, Vatter and Nagler's beta
#: without its zero intercept.
BETA_B: tuple[float, ...] = (0.25, 0.25, -0.25, 0.0, 0.0)
BETA_Z: tuple[float, ...] = (0.25, -0.25, -0.25, 0.0, 0.0)

#: The covariates with an effect in each scenario, in :data:`COVARIATES`
#: order.
ACTIVE: dict[str, tuple[str, ...]] = {
  "gampcc": ("b1", "b2", "b3", "z1", "z2", "z3", "t1", "t2", "t3"),
  "interaction": ("b1", "t1"),
  "constant": (),
}

#: The ``interaction`` scenario's regime in ``t_1`` where ``b_1`` is 0 and 1.
B1_REGIMES: tuple[str, str] = ("linear", "sin")

#: The tau of the ``constant`` scenario.
CONSTANT_TAU: float = 0.5

_EPS: float = 1e-9

_CLAYTON = pv.Bicop(family=pv.BicopFamily.clayton)
#: pyvinecopulib's bounds on the Clayton parameter. ``theta`` is floored at
#: the lower one, ``1e-10``, where the copula is independence to every digit a
#: density carries; ``eta`` is exactly zero with probability zero.
CLAYTON_BOUNDS: tuple[float, float] = (
  float(_CLAYTON.parameters_lower_bounds.item()),
  float(_CLAYTON.parameters_upper_bounds.item()),
)


def _check_x(x: torch.Tensor) -> torch.Tensor:
  if x.ndim != 2 or x.shape[1] != len(COVARIATES):
    raise ValueError(
      f"x must be an (m, {len(COVARIATES)}) matrix of the covariates "
      f"{list(COVARIATES)}; got shape {tuple(x.shape)}"
    )
  return x.to(torch.float64)


def _check_scenario(scenario: str) -> None:
  if scenario not in SCENARIOS:
    raise ValueError(f"scenario must be one of {SCENARIOS}; got {scenario!r}")


def _check_family(family: str) -> None:
  if family not in SCENARIO_FAMILIES:
    raise ValueError(
      f"family must be one of {SCENARIO_FAMILIES}; got {family!r}"
    )


def column(name: str) -> int:
  """The position of the covariate ``name`` in the covariate matrix.

  Parameters
  ----------
  name : str
      One of :data:`COVARIATES`.

  Returns
  -------
  int
      Its column.
  """
  return COVARIATES.index(name)


def eta(x: torch.Tensor) -> torch.Tensor:
  """The ``gampcc`` scenario's predictor at each row of ``x``.

  Parameters
  ----------
  x : torch.Tensor
      ``(m, 15)`` covariates, in :data:`COVARIATES` order.

  Returns
  -------
  torch.Tensor
      ``(m,)``, in ``float64``.
  """
  x = _check_x(x)
  b = x[:, :PER_TYPE]
  z = x[:, PER_TYPE : 2 * PER_TYPE]
  t = x[:, 2 * PER_TYPE :]
  beta_b = torch.tensor(BETA_B, dtype=torch.float64)
  beta_z = torch.tensor(BETA_Z, dtype=torch.float64)
  smooth = (
    (-0.25 + t[:, 0] / 2.0)
    + torch.sin(2.0 * math.pi * t[:, 1]) / 4.0
    + torch.sin(6.0 * math.pi * t[:, 2]) / 4.0
  )
  return b @ beta_b + z @ beta_z + smooth


def gampcc_tau(x: torch.Tensor) -> torch.Tensor:
  """The ``gampcc`` scenario's Kendall's tau at each row of ``x``.

  ``tanh(eta / 2)``, for either family: gamCopula's calibration of tau.

  Parameters
  ----------
  x : torch.Tensor
      ``(m, 15)`` covariates, in :data:`COVARIATES` order.

  Returns
  -------
  torch.Tensor
      ``(m,)``, in ``float64``, in ``(-1, 1)``.
  """
  return torch.tanh(eta(x) / 2.0)


def _interaction_tau(x: torch.Tensor) -> torch.Tensor:
  t1 = x[:, column("t1")]
  b1 = x[:, column("b1")].round().long()
  out = torch.empty_like(t1)
  for level, regime in enumerate(B1_REGIMES):
    rows = b1 == level
    tau_of_x = TAU_SCENARIOS[regime].tau_of_x
    assert tau_of_x is not None
    out[rows] = tau_of_x(t1[rows])
  return out


def copulas(family: str, scenario: str, x: torch.Tensor) -> list[pv.Bicop]:
  """The true copula at each row of the covariate matrix ``x``.

  Parameters
  ----------
  family : str
      One of :data:`SCENARIO_FAMILIES`.
  scenario : str
      One of :data:`SCENARIOS`.
  x : torch.Tensor
      ``(m, 15)`` covariates, in :data:`COVARIATES` order.

  Returns
  -------
  list of pyvinecopulib.Bicop
      One per row.
  """
  _check_family(family)
  _check_scenario(scenario)
  x = _check_x(x)
  if scenario == "gampcc":
    out = []
    for tau in gampcc_tau(x).tolist():
      if family == "gaussian":
        out.append(
          pv.Bicop(
            family=pv.BicopFamily.gaussian,
            parameters=np.array([[math.sin(math.pi * tau / 2.0)]]),
          )
        )
        continue
      theta = 2.0 * abs(tau) / (1.0 - abs(tau))
      theta = min(max(theta, CLAYTON_BOUNDS[0]), CLAYTON_BOUNDS[1])
      out.append(
        pv.Bicop(
          family=pv.BicopFamily.clayton,
          rotation=0 if tau >= 0.0 else 90,
          parameters=np.array([[theta]]),
        )
      )
    return out
  taus = (
    _interaction_tau(x)
    if scenario == "interaction"
    else torch.full((x.shape[0],), CONSTANT_TAU, dtype=torch.float64)
  )
  return [bicop(FAMILIES[family], value) for value in taus.tolist()]


def kendall_tau(family: str, scenario: str, x: torch.Tensor) -> torch.Tensor:
  """The true Kendall's tau at each row of ``x``.

  Parameters
  ----------
  family : str
      One of :data:`SCENARIO_FAMILIES`.
  scenario : str
      One of :data:`SCENARIOS`.
  x : torch.Tensor
      ``(m, 15)`` covariates, in :data:`COVARIATES` order.

  Returns
  -------
  torch.Tensor
      ``(m,)``, in ``float64``; negative where a ``gampcc`` row's ``eta`` is.
  """
  return torch.tensor(
    [cop.tau for cop in copulas(family, scenario, x)], dtype=torch.float64
  )


def covariates(n: int, generator: torch.Generator) -> torch.Tensor:
  """``n`` rows of the covariate design, in :data:`COVARIATES` order.

  Drawn as three blocks, in turn: ``b`` as uniforms under 1/2, ``z`` as
  standard normals, and ``t`` as uniforms.

  Parameters
  ----------
  n : int
      The number of rows.
  generator : torch.Generator
      The stream the rows are drawn from.

  Returns
  -------
  torch.Tensor
      ``(n, 15)``, in ``float64``.
  """
  b = (
    torch.rand(n, PER_TYPE, generator=generator, dtype=torch.float64) < 0.5
  ).to(torch.float64)
  z = torch.randn(n, PER_TYPE, generator=generator, dtype=torch.float64)
  t = torch.rand(n, PER_TYPE, generator=generator, dtype=torch.float64)
  return torch.column_stack([b, z, t])


def sample(
  family: str, scenario: str, n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Draw ``n`` training rows ``(u, v, x)`` from the true conditional copula.

  One generator, seeded with ``seed``, draws the covariates
  (:func:`covariates`), then ``u``, then the inverse-Rosenblatt noise ``w``;
  ``v = hinv1([u, w])`` under each row's own copula.

  Parameters
  ----------
  family : str
      One of :data:`SCENARIO_FAMILIES`.
  scenario : str
      One of :data:`SCENARIOS`.
  n : int
      The number of rows.
  seed : int
      The cell's seed.

  Returns
  -------
  tuple of torch.Tensor
      ``u`` and ``v``, ``(n,)``, and ``x``, ``(n, 15)``, all ``float64``.
  """
  generator = torch.Generator().manual_seed(seed)
  x = covariates(n, generator)
  u = torch.rand(n, generator=generator, dtype=torch.float64).clamp(
    _EPS, 1.0 - _EPS
  )
  w = torch.rand(n, generator=generator, dtype=torch.float64).clamp(
    _EPS, 1.0 - _EPS
  )
  uw = torch.column_stack([u, w]).numpy()
  v = torch.tensor(
    [
      float(cop.hinv1(uw[i : i + 1]).item())
      for i, cop in enumerate(copulas(family, scenario, x))
    ],
    dtype=torch.float64,
  )
  return u, v.clamp(_EPS, 1.0 - _EPS), x


def points(m: int, seed: int) -> torch.Tensor:
  """The ``m`` covariate points a scenario is scored at.

  Drawn from the covariate design itself, from a generator seeded with
  ``seed``, so every family, ``n``, repetition and estimator of the scenario
  is scored at the same points.

  Parameters
  ----------
  m : int
      The number of points.
  seed : int
      The points' seed.

  Returns
  -------
  torch.Tensor
      ``(m, 15)``, as :func:`covariates`.
  """
  if m < 1:
    raise ValueError(f"m must be >= 1 point; got {m}")
  return covariates(m, torch.Generator().manual_seed(seed))


def metric_grid(pts: torch.Tensor, uv_grid_n: int) -> EvalGrid:
  """E1's ``uv`` grid at each covariate point, as one :class:`EvalGrid`.

  Point ``i * m + j`` is pair ``i`` of the ``uv_grid_n x uv_grid_n`` interior
  grid at ``pts[j]``, the layout of E1's conditional grids; ``x_axis`` holds
  the points themselves, ``(m, 15)``.

  Parameters
  ----------
  pts : torch.Tensor
      ``(m, 15)`` covariate points, from :func:`points`.
  uv_grid_n : int
      The ``uv`` grid's points per axis.

  Returns
  -------
  EvalGrid
      ``shape`` ``(uv_grid_n ** 2, m)``.
  """
  pts = _check_x(pts)
  axis = interior_axis(uv_grid_n)
  uu, vv = torch.meshgrid(axis, axis, indexing="ij")
  u_pairs, v_pairs = uu.reshape(-1), vv.reshape(-1)
  n_pairs, m = u_pairs.shape[0], pts.shape[0]
  return EvalGrid(
    u_flat=u_pairs.repeat_interleave(m),
    v_flat=v_pairs.repeat_interleave(m),
    x_flat=pts.repeat(n_pairs, 1),
    shape=(n_pairs, m),
    conditional=True,
    x_axis=pts,
    u_axis=axis,
    v_axis=axis,
  )


def holdout_grid(
  family: str, scenario: str, pts: torch.Tensor, n: int, seeds: Sequence[int]
) -> EvalGrid:
  """``n`` draws from the true copula at each covariate point.

  At ``pts[j]``, ``n`` pairs from the true copula there, from a generator
  seeded with ``seeds[j]``, drawn and laid out as
  :func:`~npcc.experiments.scenarios.holdout_grid` draws and lays out E1's:
  point ``i * m + j`` is draw ``i`` at ``pts[j]``.

  Parameters
  ----------
  family : str
      One of :data:`SCENARIO_FAMILIES`.
  scenario : str
      One of :data:`SCENARIOS`.
  pts : torch.Tensor
      ``(m, 15)`` covariate points.
  n : int
      The draws per point.
  seeds : sequence of int
      One seed per point.

  Returns
  -------
  EvalGrid
      ``shape`` ``(n, m)``, with empty ``uv`` axes.
  """
  pts = _check_x(pts)
  if n < 2:
    raise ValueError(f"n must be >= 2 draws per point; got {n}")
  if len(seeds) != pts.shape[0]:
    raise ValueError(
      f"seeds must hold one seed per point; got {len(seeds)} for {pts.shape[0]}"
    )
  # E1's own sampler and truth (below), so both studies draw and score alike.
  draws = [
    scenarios._copula_draws(
      ((1.0, cop),), n, torch.Generator().manual_seed(seed)
    )
    for cop, seed in zip(copulas(family, scenario, pts), seeds, strict=True)
  ]
  empty = torch.empty(0, dtype=torch.float64)
  return EvalGrid(
    u_flat=torch.stack([d[0] for d in draws], dim=1).reshape(-1),
    v_flat=torch.stack([d[1] for d in draws], dim=1).reshape(-1),
    x_flat=pts.repeat(n, 1),
    shape=(n, pts.shape[0]),
    conditional=True,
    x_axis=pts,
    u_axis=empty,
    v_axis=empty,
  )


def ground_truth(
  family: str, scenario: str, grid: EvalGrid
) -> dict[str, torch.Tensor]:
  """Exact ``pdf``, ``hfunc1`` and ``hfunc2`` at every point of ``grid``.

  Parameters
  ----------
  family : str
      One of :data:`SCENARIO_FAMILIES`.
  scenario : str
      One of :data:`SCENARIOS`.
  grid : EvalGrid
      From :func:`metric_grid` or :func:`holdout_grid`.

  Returns
  -------
  dict of str to torch.Tensor
      Each quantity, ``grid.shape``: column ``j`` is the copula at
      ``grid.x_axis[j]``, at that column's own pairs.
  """
  assert grid.x_axis is not None
  pairs = grid.uv.reshape(*grid.shape, 2)
  cols = [
    scenarios._quantities(cop, pairs[:, j])
    for j, cop in enumerate(copulas(family, scenario, grid.x_axis))
  ]
  return {
    q: torch.stack([c[q] for c in cols], dim=1) for q in scenarios.QUANTITIES
  }
