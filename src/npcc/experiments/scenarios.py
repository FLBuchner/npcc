"""Ground-truth copula scenarios for the simulation study.

A *scenario* pairs a copula family with a Kendall's-tau regime and defines:

- :func:`sample` — draw a training set ``(u, v[, x])`` from the known copula;
- :func:`eval_grid` — the fixed points at which estimates are scored;
- :func:`ground_truth` — the exact ``pdf`` / ``cdf`` / ``hfunc1`` / ``hfunc2``
  on that evaluation grid.

Everything is backed by :mod:`pyvinecopulib`: ``Bicop.tau_to_parameters`` maps a
target Kendall's tau to the family's parameter, and ``Bicop.hinv1`` provides a
generic inverse-Rosenblatt sampler, so the study generalizes to any
one-parameter family without per-family closed forms.

Conditional scenarios vary ``tau`` with a scalar covariate ``x`` (``x`` is a
deterministic ``linspace`` over the unit interval, matching the original
notebook study); unconditional scenarios use a single fixed ``tau`` and no
covariate (``x is None``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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

QUANTITIES: tuple[str, ...] = ("pdf", "cdf", "hfunc1", "hfunc2")

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
  predictions consistently.
  """

  u_flat: torch.Tensor
  v_flat: torch.Tensor
  x_flat: torch.Tensor | None
  shape: tuple[int, ...]
  conditional: bool
  x_axis: torch.Tensor | None
  u_axis: torch.Tensor
  v_axis: torch.Tensor


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


def _bicop(family: pv.BicopFamily, tau: float) -> pv.Bicop:
  """Bicop of ``family`` whose Kendall's tau equals ``tau``."""
  proto = pv.Bicop(family=family)
  params = proto.tau_to_parameters(float(tau))
  return pv.Bicop(family=family, parameters=params)


def is_conditional(scenario: str) -> bool:
  return TAU_SCENARIOS[scenario].conditional


def eval_grid(
  scenario: str,
  *,
  conditional_uv_grid_n: int = CONDITIONAL_UV_GRID_N,
  conditional_x_grid_n: int = CONDITIONAL_X_GRID_N,
) -> EvalGrid:
  """Fixed evaluation grid for ``scenario`` (conditional or unconditional)."""
  spec = TAU_SCENARIOS[scenario]
  if spec.conditional:
    u_axis = interior_axis(conditional_uv_grid_n)
    v_axis = interior_axis(conditional_uv_grid_n)
    uu, vv = torch.meshgrid(u_axis, v_axis, indexing="ij")
    u_pairs = uu.reshape(-1)
    v_pairs = vv.reshape(-1)
    x_axis = conditional_x_axis(conditional_x_grid_n)
    n_pairs = u_pairs.shape[0]
    return EvalGrid(
      u_flat=u_pairs.repeat_interleave(conditional_x_grid_n),
      v_flat=v_pairs.repeat_interleave(conditional_x_grid_n),
      x_flat=x_axis.repeat(n_pairs).reshape(-1, 1),
      shape=(n_pairs, conditional_x_grid_n),
      conditional=True,
      x_axis=x_axis,
      u_axis=u_axis,
      v_axis=v_axis,
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
  if not TAU_SCENARIOS[scenario].conditional:
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


def ground_truth(
  family: str, scenario: str, grid: EvalGrid | None = None
) -> dict[str, torch.Tensor]:
  """Exact pdf/cdf/hfunc1/hfunc2 on :func:`eval_grid` for ``(family, scenario)``.

  Conditional: one ``Bicop`` per evaluation ``x`` (50), evaluated at the 25 uv
  pairs, stacked into ``(n_pairs, n_x)`` grids.  Unconditional: a single
  ``Bicop`` evaluated on the flattened uv grid.
  """
  fam = FAMILIES[family]
  spec = TAU_SCENARIOS[scenario]
  grid = eval_grid(scenario) if grid is None else grid

  if spec.conditional:
    assert spec.tau_of_x is not None and grid.x_axis is not None
    tau_x = spec.tau_of_x(grid.x_axis)
    u_grid = grid.u_flat.reshape(grid.shape)
    v_grid = grid.v_flat.reshape(grid.shape)
    uv = torch.column_stack([u_grid[:, 0], v_grid[:, 0]])
    uv_host = uv.numpy()
    cols: dict[str, list[torch.Tensor]] = {q: [] for q in QUANTITIES}
    for tau in tau_x:
      cop = _bicop(fam, float(tau))
      cols["pdf"].append(torch.from_numpy(cop.pdf(uv_host)))
      cols["cdf"].append(torch.from_numpy(cop.cdf(uv_host)))
      cols["hfunc1"].append(torch.from_numpy(cop.hfunc1(uv_host)))
      cols["hfunc2"].append(torch.from_numpy(cop.hfunc2(uv_host)))
    return {q: torch.stack(cols[q], dim=1) for q in QUANTITIES}

  assert spec.tau is not None
  cop = _bicop(fam, spec.tau)
  uv = torch.column_stack([grid.u_flat, grid.v_flat]).numpy()
  return {
    "pdf": torch.from_numpy(cop.pdf(uv)),
    "cdf": torch.from_numpy(cop.cdf(uv)),
    "hfunc1": torch.from_numpy(cop.hfunc1(uv)),
    "hfunc2": torch.from_numpy(cop.hfunc2(uv)),
  }


def sample(
  family: str, scenario: str, n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
  """Draw ``n`` training points ``(u, v[, x])`` from the true copula.

  Conditional: ``x`` is a deterministic ``linspace`` and each row is sampled by
  inverse Rosenblatt, ``v = hinv1([u, w])`` under the row's own ``tau(x_i)``.
  Unconditional: ``Bicop.sample`` with no covariate (``x is None``).
  """
  fam = FAMILIES[family]
  spec = TAU_SCENARIOS[scenario]
  generator = torch.Generator().manual_seed(seed)

  if not spec.conditional:
    assert spec.tau is not None
    cop = _bicop(fam, spec.tau)
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
  # negligible next to the TabPFN fit; if it ever bites, group by rounded tau.
  for i in range(n):
    cop = _bicop(fam, float(tau_x[i]))
    uv_host = torch.stack([u[i], w[i]]).reshape(1, 2).numpy()
    v[i] = float(cop.hinv1(uv_host).item())
  return u, v, x.reshape(-1, 1)
