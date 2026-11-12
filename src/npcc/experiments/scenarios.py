"""Ground-truth copula scenarios for the simulation study.

A *scenario* pairs a copula family with a Kendall's-tau regime and defines:

- :func:`sample` — draw a training set ``(u, v[, x])`` from the known copula;
- :func:`eval_grid` — the fixed points at which estimates are scored;
- :func:`holdout_grid` — draws from the true copula at each covariate value,
  at which the Monte Carlo KL is scored;
- :func:`ground_truth` — the exact ``pdf`` / ``hfunc1`` / ``hfunc2`` at the
  points of either.

Everything is backed by :mod:`pyvinecopulib`: ``Bicop.tau_to_parameters`` maps a
target Kendall's tau to the family's parameter, and ``Bicop.hinv1`` provides a
generic inverse-Rosenblatt sampler, so the study generalizes to any
one-parameter family without per-family closed forms.

Conditional scenarios vary ``tau`` with a scalar covariate ``x`` (``x`` is a
deterministic ``linspace`` over the unit interval, matching the original
notebook study); unconditional scenarios use a single fixed ``tau`` and no
covariate (``x is None``).

A *shape family* (:data:`SHAPE_FAMILIES`) moves the copula's shape with ``x``,
not only its strength, so no single parametric family with a smooth
``tau(x)`` can represent it. It takes the one regime :data:`IMPLIED`, since
its ``tau(x)`` is implied by that shape; each family states it in closed
form. The shape families sit outside :data:`FAMILIES` and
:data:`TAU_SCENARIOS`, which the vine study draws its edges from by position.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pyvinecopulib as pv
import torch

# Families restricted to the single-parameter set, for which
# ``tau_to_parameters`` is an unambiguous scalar map. The insertion order is
# part of the simulation seed: `runner.py` hashes each family's index into
# `_cell_seed`, so reordering these re-seeds every cell. That is also why this
# is not `pyvinecopulib.families.one_par`, which holds the same five in a
# different order.
FAMILIES: dict[str, pv.BicopFamily] = {
  "clayton": pv.BicopFamily.clayton,
  "gumbel": pv.BicopFamily.gumbel,
  "frank": pv.BicopFamily.frank,
  "gaussian": pv.BicopFamily.gaussian,
  "joe": pv.BicopFamily.joe,
}

QUANTITIES: tuple[str, ...] = ("pdf", "hfunc1", "hfunc2")

# Conditional evaluation grid defaults.
CONDITIONAL_UV_GRID_N: int = 20
CONDITIONAL_X_GRID_N: int = 10
X_MIN: float = 0.01
X_MAX: float = 0.99

# Unconditional evaluation: an interior uv grid (boundaries excluded).
UV_GRID_N: int = 20

# Keep tau in a band where every family's parameter stays well-conditioned. The
# upper end is 0.70, shared with the vine study: its h-function cascade saturates
# to 0/1 in double precision near tau = 0.9 and underflows the exact densities
# that score higher-dimensional fits, so tree 1 reproduces these regimes exactly.
TAU_LO: float = 0.05
TAU_HI: float = 0.70
_EPS: float = 1e-9


@dataclass(frozen=True)
class ScenarioSpec:
  """A Kendall's-tau regime: conditional ``tau(x)`` or a fixed ``tau``."""

  name: str
  conditional: bool
  tau_of_x: Callable[[torch.Tensor], torch.Tensor] | None = None
  tau: float | None = None


def _clip_tau(tau: torch.Tensor) -> torch.Tensor:
  return tau.clamp(TAU_LO, TAU_HI)


TAU_SCENARIOS: dict[str, ScenarioSpec] = {
  "linear": ScenarioSpec(
    "linear", True, lambda x: _clip_tau(TAU_LO + (TAU_HI - TAU_LO) * x)
  ),
  "constant": ScenarioSpec("constant", True, lambda x: torch.full_like(x, 0.5)),
  "sin": ScenarioSpec(
    "sin",
    True,
    lambda x: _clip_tau(
      (TAU_LO + TAU_HI) / 2.0
      + (TAU_HI - TAU_LO) / 2.0 * torch.sin(2.0 * torch.pi * x)
    ),
  ),
  "quadratic": ScenarioSpec(
    "quadratic",
    True,
    lambda x: _clip_tau(TAU_LO + (TAU_HI - TAU_LO) * (2.0 * x - 1.0) ** 2),
  ),
  "uncond35": ScenarioSpec("uncond35", False, tau=0.35),
  "uncond70": ScenarioSpec("uncond70", False, tau=0.70),
}


@dataclass(frozen=True)
class EvalGrid:
  """Flattened evaluation points and the shape to reshape predictions to.

  ``u_flat`` / ``v_flat`` are length ``n_points``; ``x_flat`` is the same
  length for conditional scenarios and ``None`` for unconditional ones.
  ``shape`` is ``(n_pairs, n_x)`` (conditional) or ``(n_points,)``
  (unconditional) so the runner can reshape per-quantity ground truth and
  predictions consistently: point ``i * n_x + j`` is pair ``i`` at
  ``x_axis[j]``. On a grid from :func:`eval_grid` every ``x`` holds the same
  pairs, the product of ``u_axis`` and ``v_axis``; on one from
  :func:`holdout_grid` each ``x`` holds its own draws, and both axes are
  empty.
  """

  u_flat: torch.Tensor
  v_flat: torch.Tensor
  x_flat: torch.Tensor | None
  shape: tuple[int, ...]
  conditional: bool
  x_axis: torch.Tensor | None
  u_axis: torch.Tensor
  v_axis: torch.Tensor

  @property
  def uv(self) -> torch.Tensor:
    """The evaluation points as one ``(n_points, 2)`` copula argument."""
    return torch.column_stack([self.u_flat, self.v_flat])


def interior_axis(n: int) -> torch.Tensor:
  """Midpoint grid on ``(0, 1)`` with boundaries excluded."""
  if n < 2:
    raise ValueError("n must be >= 2.")
  return (torch.arange(n, dtype=torch.float64) + 0.5) / n


def conditional_x_axis(n: int) -> torch.Tensor:
  """Conditional evaluation grid over the configured covariate support."""
  if n < 1:
    raise ValueError("n must be >= 1.")
  return torch.linspace(X_MIN, X_MAX, n, dtype=torch.float64)


def bicop(family: pv.BicopFamily, tau: float) -> pv.Bicop:
  """Return the ``family`` copula whose Kendall's tau equals ``tau``.

  Parameters
  ----------
  family : pyvinecopulib.BicopFamily
      A one-parameter family.
  tau : float
      The target Kendall's tau.

  Returns
  -------
  pyvinecopulib.Bicop
      The copula.
  """
  proto = pv.Bicop(family=family)
  params = proto.tau_to_parameters(float(tau))
  return pv.Bicop(family=family, parameters=params)


# ---------------------------------------------------------------------------
# Shape families
# ---------------------------------------------------------------------------

#: The tau regime of every shape family, and of nothing else.
IMPLIED: str = "implied"


@dataclass(frozen=True)
class ShapeFamily:
  """A conditional copula whose shape, not only its strength, moves with x.

  At each ``x`` it is the finite mixture ``sum_k w_k(x) c_k(u, v)`` that
  ``components(x)`` lists as ``(w_k, c_k)`` pairs, with weights summing to
  one. The density and both h-functions are linear in the mixture, so the
  ground truth is exact. ``tau_of_x`` is its Kendall's tau, in closed form.
  """

  name: str
  components: Callable[[float], tuple[tuple[float, pv.Bicop], ...]]
  tau_of_x: Callable[[torch.Tensor], torch.Tensor]


#: Kendall's tau of both components of ``clayton-gumbel-mix``.
MIX_TAU: float = 0.5

#: ``(I_CG, I_GC)`` for ``clayton-gumbel-mix``, where ``I_AB`` is
#: ``int int C_A dC_B`` and C and G are its Clayton and Gumbel components.
#: Each is computed as Nelsen's ``1/2 - int int h1_A h2_B du dv``, whose
#: integrand is bounded, by tensor tanh-sinh quadrature at 127, 254, 507 and
#: 1015 nodes a side, which agree to 5e-15; tensor Gauss-Legendre agrees to
#: 6e-14. The same nodes give ``I_CC = I_GG = (tau + 1) / 4 = 0.375`` to
#: 1e-14, so each value is correct to about 1e-13. The two are equal, since
#: Nelsen's concordance function is symmetric in its arguments.
MIX_CROSS_INTEGRALS: tuple[float, float] = (
  0.374081988542034,
  0.374081988542034,
)

#: Kendall's tau of ``student-varying-nu`` at every ``x``.
STUDENT_TAU: float = 0.5

#: The degrees of freedom of ``student-varying-nu`` at ``x = 0`` and ``x = 1``.
STUDENT_NU_RANGE: tuple[float, float] = (2.0, 50.0)


def student_nu(x: float) -> float:
  """Degrees of freedom of ``student-varying-nu`` at ``x``.

  ``exp{log 2 + x (log 50 - log 2)}``, written as ``2 (50 / 2) ** x`` so that
  it is exactly 2 at ``x = 0`` and exactly 50 at ``x = 1``, pyvinecopulib's
  upper bound.

  Parameters
  ----------
  x : float
      The covariate.

  Returns
  -------
  float
      The degrees of freedom.
  """
  lo, hi = STUDENT_NU_RANGE
  return lo * math.pow(hi / lo, x)


def _mix_components(x: float) -> tuple[tuple[float, pv.Bicop], ...]:
  return (
    (1.0 - x, bicop(pv.BicopFamily.clayton, MIX_TAU)),
    (x, bicop(pv.BicopFamily.gumbel, MIX_TAU)),
  )


def _mix_tau(x: torch.Tensor) -> torch.Tensor:
  """Kendall's tau of ``(1 - x) C_C + x C_G``, quadratic in the weight ``x``.

  ``tau = 4 int int C dC - 1`` and ``int int C dC`` is bilinear in the
  mixture, so with ``I_AB = int int C_A dC_B``::

    tau(x) = (1 - x)^2 tau_C + x^2 tau_G + x (1 - x) [4 (I_CG + I_GC) - 2]

  With both components at tau 0.5 it dips to about 0.4982 at ``x = 1/2``.
  """
  i_cg, i_gc = MIX_CROSS_INTEGRALS
  cross = 4.0 * (i_cg + i_gc) - 2.0
  return (1.0 - x) ** 2 * MIX_TAU + x**2 * MIX_TAU + x * (1.0 - x) * cross


def _student_components(x: float) -> tuple[tuple[float, pv.Bicop], ...]:
  # An elliptical copula's tau is (2 / pi) arcsin(rho).
  rho = math.sin(math.pi * STUDENT_TAU / 2.0)
  student = pv.Bicop(
    family=pv.BicopFamily.student,
    parameters=np.array([[rho], [student_nu(x)]], dtype=np.float64),
  )
  return ((1.0, student),)


SHAPE_FAMILIES: dict[str, ShapeFamily] = {
  # M1, a tail transition: lower-tail Clayton at x = 0 to upper-tail Gumbel
  # at x = 1, each at tau 0.5, mixed with weight x.
  "clayton-gumbel-mix": ShapeFamily(
    "clayton-gumbel-mix", _mix_components, _mix_tau
  ),
  # M2, varying tails at a fixed tau: a Student t copula at tau 0.5 whose
  # degrees of freedom rise log-linearly from 2 to 50 over the unit interval.
  "student-varying-nu": ShapeFamily(
    "student-varying-nu",
    _student_components,
    lambda x: torch.full_like(x, STUDENT_TAU),
  ),
}


def is_conditional(scenario: str) -> bool:
  """Whether ``scenario`` varies with a covariate.

  Parameters
  ----------
  scenario : str
      A key of :data:`TAU_SCENARIOS`, or :data:`IMPLIED`.

  Returns
  -------
  bool
      ``True`` for a conditional regime and for :data:`IMPLIED`.
  """
  return scenario == IMPLIED or TAU_SCENARIOS[scenario].conditional


def _check_pair(family: str, scenario: str) -> None:
  """Refuse a shape family off :data:`IMPLIED`, and the reverse."""
  if (family in SHAPE_FAMILIES) != (scenario == IMPLIED):
    raise ValueError(
      f"tau_scenario must be {IMPLIED!r} exactly when the family is one of "
      f"the shape families {sorted(SHAPE_FAMILIES)}; got family {family!r} "
      f"with tau_scenario {scenario!r}"
    )


def eval_grid(
  scenario: str,
  *,
  conditional_uv_grid_n: int = CONDITIONAL_UV_GRID_N,
  conditional_x_grid_n: int = CONDITIONAL_X_GRID_N,
) -> EvalGrid:
  """Fixed evaluation grid for ``scenario`` (conditional or unconditional)."""
  if is_conditional(scenario):
    return eval_grid_for_x(
      scenario,
      conditional_x_axis(conditional_x_grid_n),
      conditional_uv_grid_n=conditional_uv_grid_n,
    )
  axis = interior_axis(UV_GRID_N)
  uu, vv = torch.meshgrid(axis, axis, indexing="ij")
  u_flat = uu.reshape(-1)
  v_flat = vv.reshape(-1)
  return EvalGrid(
    u_flat=u_flat,
    v_flat=v_flat,
    x_flat=None,
    shape=(u_flat.shape[0],),
    conditional=False,
    x_axis=None,
    u_axis=axis,
    v_axis=axis,
  )


def eval_grid_for_x(
  scenario: str, x_axis: torch.Tensor, *, conditional_uv_grid_n: int
) -> EvalGrid:
  """Conditional evaluation grid at caller-selected ``x`` values."""
  if not is_conditional(scenario):
    raise ValueError("eval_grid_for_x is only valid for conditional scenarios.")
  x_axis = x_axis.to(dtype=torch.float64)
  u_axis = interior_axis(conditional_uv_grid_n)
  v_axis = interior_axis(conditional_uv_grid_n)
  uu, vv = torch.meshgrid(u_axis, v_axis, indexing="ij")
  u_pairs = uu.reshape(-1)
  v_pairs = vv.reshape(-1)
  n_pairs = u_pairs.shape[0]
  return EvalGrid(
    u_flat=u_pairs.repeat_interleave(x_axis.shape[0]),
    v_flat=v_pairs.repeat_interleave(x_axis.shape[0]),
    x_flat=x_axis.repeat(n_pairs).reshape(-1, 1),
    shape=(n_pairs, x_axis.shape[0]),
    conditional=True,
    x_axis=x_axis,
    u_axis=u_axis,
    v_axis=v_axis,
  )


def _copula_draws(
  components: tuple[tuple[float, pv.Bicop], ...],
  n: int,
  generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
  """``n`` pairs from the mixture ``components``, by inverse Rosenblatt.

  The stream is laid out as :func:`_sample_shape` lays out its own: ``u``,
  the inverse-Rosenblatt noise ``w``, then ``r``, which picks each pair's
  component, drawn for a one-component copula too. ``v`` is kept inside the
  open unit interval, where every estimator is defined.
  """
  u = torch.rand(n, generator=generator, dtype=torch.float64).clamp(
    _EPS, 1.0 - _EPS
  )
  w = torch.rand(n, generator=generator, dtype=torch.float64).clamp(
    _EPS, 1.0 - _EPS
  )
  r = torch.rand(n, generator=generator, dtype=torch.float64)
  bounds = torch.tensor(
    [weight for weight, _ in components], dtype=torch.float64
  ).cumsum(0)
  # Rounding can leave the weights summing to just under one.
  pick = torch.searchsorted(bounds, r, right=True).clamp(
    max=len(components) - 1
  )
  v = torch.empty(n, dtype=torch.float64)
  for k, (_, cop) in enumerate(components):
    rows = pick == k
    if rows.any():
      uw = torch.column_stack([u[rows], w[rows]]).numpy()
      v[rows] = torch.from_numpy(cop.hinv1(uw))
  return u, v.clamp(_EPS, 1.0 - _EPS)


def holdout_grid(
  family: str,
  scenario: str,
  x_axis: torch.Tensor | None,
  n: int,
  seeds: Sequence[int],
) -> EvalGrid:
  """``n`` draws from the true copula at each ``x``, as an :class:`EvalGrid`.

  The points at which the Monte Carlo KL is scored: at ``x_axis[j]``, ``n``
  pairs from the true conditional copula there, from a generator seeded with
  ``seeds[j]``, so every estimator of a cell is scored on the same draws and
  each ``x``'s draws do not depend on the others'. An unconditional scenario
  takes ``x_axis=None`` and one seed.

  Parameters
  ----------
  family : str
      A key of :data:`FAMILIES` or :data:`SHAPE_FAMILIES`.
  scenario : str
      A key of :data:`TAU_SCENARIOS`, or :data:`IMPLIED`.
  x_axis : torch.Tensor or None
      The covariate values; ``None`` for an unconditional scenario.
  n : int
      The draws per covariate value.
  seeds : sequence of int
      One seed per covariate value, or one for an unconditional scenario.

  Returns
  -------
  EvalGrid
      ``shape`` ``(n, n_x)``, or ``(n,)`` unconditionally, with empty axes.
  """
  _check_pair(family, scenario)
  if n < 2:
    raise ValueError(f"n must be >= 2 draws per covariate value; got {n}")
  conditional = is_conditional(scenario)
  if conditional != (x_axis is not None):
    raise ValueError(
      f"x_axis must be given exactly when the scenario is conditional; got "
      f"{'none' if x_axis is None else 'one'} for {scenario!r}"
    )
  xs = [float("nan")] if x_axis is None else x_axis.tolist()
  if len(seeds) != len(xs):
    raise ValueError(
      f"seeds must hold one seed per covariate value; got {len(seeds)} for "
      f"{len(xs)}"
    )

  def components(x: float) -> tuple[tuple[float, pv.Bicop], ...]:
    if family in SHAPE_FAMILIES:
      return SHAPE_FAMILIES[family].components(x)
    spec = TAU_SCENARIOS[scenario]
    if spec.conditional:
      assert spec.tau_of_x is not None
      tau = float(spec.tau_of_x(torch.tensor([x], dtype=torch.float64))[0])
    else:
      assert spec.tau is not None
      tau = spec.tau
    return ((1.0, bicop(FAMILIES[family], tau)),)

  draws = [
    _copula_draws(components(x), n, torch.Generator().manual_seed(seed))
    for x, seed in zip(xs, seeds, strict=True)
  ]
  u = torch.stack([d[0] for d in draws], dim=1)
  v = torch.stack([d[1] for d in draws], dim=1)
  empty = torch.empty(0, dtype=torch.float64)
  if x_axis is None:
    return EvalGrid(
      u_flat=u.reshape(-1),
      v_flat=v.reshape(-1),
      x_flat=None,
      shape=(n,),
      conditional=False,
      x_axis=None,
      u_axis=empty,
      v_axis=empty,
    )
  x_axis = x_axis.to(dtype=torch.float64)
  return EvalGrid(
    u_flat=u.reshape(-1),
    v_flat=v.reshape(-1),
    x_flat=x_axis.repeat(n).reshape(-1, 1),
    shape=(n, x_axis.shape[0]),
    conditional=True,
    x_axis=x_axis,
    u_axis=empty,
    v_axis=empty,
  )


def _quantities(cop: pv.Bicop, uv: torch.Tensor) -> dict[str, torch.Tensor]:
  """Every scored quantity of ``cop`` at the points ``uv``."""
  uv_host = uv.contiguous().numpy()
  return {
    "pdf": torch.from_numpy(cop.pdf(uv_host)),
    "hfunc1": torch.from_numpy(cop.hfunc1(uv_host)),
    "hfunc2": torch.from_numpy(cop.hfunc2(uv_host)),
  }


def _mixture_quantities(
  family: ShapeFamily, x: float, uv: torch.Tensor
) -> dict[str, torch.Tensor]:
  """Every scored quantity of ``family`` at ``x``, at the points ``uv``."""
  out = {q: torch.zeros(uv.shape[0], dtype=torch.float64) for q in QUANTITIES}
  for weight, cop in family.components(x):
    for q, values in _quantities(cop, uv).items():
      out[q] += weight * values
  return out


def ground_truth(
  family: str, scenario: str, grid: EvalGrid | None = None
) -> dict[str, torch.Tensor]:
  """Exact pdf/hfunc1/hfunc2 on ``grid`` for ``(family, scenario)``.

  Conditional: one ``Bicop`` per evaluation ``x``, evaluated at that ``x``'s
  own pairs, stacked into ``(n_pairs, n_x)`` grids.  Unconditional: a single
  ``Bicop`` evaluated on the flattened points. A shape family is conditional,
  and at each ``x`` mixes its components' quantities with their weights.
  ``grid`` defaults to the scenario's :func:`eval_grid`; it may also come from
  :func:`holdout_grid`, whose pairs differ from one ``x`` to the next.
  """
  _check_pair(family, scenario)
  grid = eval_grid(scenario) if grid is None else grid
  if family in SHAPE_FAMILIES:
    assert grid.x_axis is not None
    pairs = grid.uv.reshape(*grid.shape, 2)
    shape_cols = [
      _mixture_quantities(SHAPE_FAMILIES[family], float(x), pairs[:, j])
      for j, x in enumerate(grid.x_axis)
    ]
    return {
      q: torch.stack([c[q] for c in shape_cols], dim=1) for q in QUANTITIES
    }

  fam = FAMILIES[family]
  spec = TAU_SCENARIOS[scenario]

  if spec.conditional:
    assert spec.tau_of_x is not None and grid.x_axis is not None
    pairs = grid.uv.reshape(*grid.shape, 2)
    cols = [
      _quantities(bicop(fam, float(tau)), pairs[:, j])
      for j, tau in enumerate(spec.tau_of_x(grid.x_axis))
    ]
    return {q: torch.stack([c[q] for c in cols], dim=1) for q in QUANTITIES}

  assert spec.tau is not None
  return _quantities(bicop(fam, spec.tau), grid.uv)


def sample(
  family: str, scenario: str, n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
  """Draw ``n`` training points ``(u, v[, x])`` from the true copula.

  Conditional: ``x`` is a deterministic ``linspace`` and each row is sampled by
  inverse Rosenblatt, ``v = hinv1([u, w])`` under the row's own ``tau(x_i)``.
  Unconditional: ``Bicop.sample`` with no covariate (``x is None``). A shape
  family is conditional and is sampled by :func:`_sample_shape`.
  """
  _check_pair(family, scenario)
  if family in SHAPE_FAMILIES:
    return _sample_shape(SHAPE_FAMILIES[family], n, seed)

  fam = FAMILIES[family]
  spec = TAU_SCENARIOS[scenario]
  generator = torch.Generator().manual_seed(seed)

  if not spec.conditional:
    assert spec.tau is not None
    cop = bicop(fam, spec.tau)
    seeds = torch.randint(1, 2**31 - 1, (3,), generator=generator).tolist()
    uv = torch.from_numpy(
      cop.sample(n, seeds=[int(s) for s in seeds]),
    )
    return uv[:, 0], uv[:, 1], None

  assert spec.tau_of_x is not None
  x = torch.linspace(X_MIN, X_MAX, n, dtype=torch.float64)
  tau_x = spec.tau_of_x(x)
  u = torch.rand(n, generator=generator, dtype=torch.float64).clamp(
    _EPS, 1.0 - _EPS
  )
  w = torch.rand(n, generator=generator, dtype=torch.float64).clamp(
    _EPS, 1.0 - _EPS
  )
  v = torch.empty(n, dtype=torch.float64)
  # ponytail: one Bicop per row because tau(x) is continuous and pyvinecopulib
  # does not vectorize hinv1 over row-specific parameters. O(n) Bicop builds is
  # negligible next to an estimator's fit; if it ever bites, group by rounded
  # tau.
  for i in range(n):
    cop = bicop(fam, float(tau_x[i]))
    uv_host = torch.stack([u[i], w[i]]).reshape(1, 2).numpy()
    v[i] = float(cop.hinv1(uv_host).item())
  return u, v, x.reshape(-1, 1)


def _pick(components: tuple[tuple[float, pv.Bicop], ...], r: float) -> pv.Bicop:
  """The component whose interval of cumulative weight holds ``r``."""
  total = 0.0
  for weight, cop in components:
    total += weight
    if r < total:
      return cop
  # Rounding can leave the weights summing to just under one.
  return components[-1][1]


def _sample_shape(
  family: ShapeFamily, n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Draw ``n`` training points from a shape family, exactly.

  ``x`` is the conditional ``linspace``. One generator, seeded with ``seed``,
  draws three blocks of ``n`` uniforms in turn: ``u``; the inverse-Rosenblatt
  noise ``w``; and ``r``, which picks row ``i``'s component as the first whose
  cumulative weight at ``x_i`` exceeds ``r_i``, so component ``k`` with
  probability ``w_k(x_i)``. Each component has uniform margins and ``r`` is
  independent of ``u``, so ``v = hinv1([u, w])`` under the picked component
  is a draw from the mixture. ``r`` is drawn for a one-component family too,
  so every shape family lays its stream out the same way.
  """
  generator = torch.Generator().manual_seed(seed)
  x = torch.linspace(X_MIN, X_MAX, n, dtype=torch.float64)
  u = torch.rand(n, generator=generator, dtype=torch.float64).clamp(
    _EPS, 1.0 - _EPS
  )
  w = torch.rand(n, generator=generator, dtype=torch.float64).clamp(
    _EPS, 1.0 - _EPS
  )
  r = torch.rand(n, generator=generator, dtype=torch.float64)
  v = torch.empty(n, dtype=torch.float64)
  for i in range(n):
    cop = _pick(family.components(float(x[i])), float(r[i]))
    uv_host = torch.stack([u[i], w[i]]).reshape(1, 2).numpy()
    v[i] = float(cop.hinv1(uv_host).item())
  return u, v, x.reshape(-1, 1)
