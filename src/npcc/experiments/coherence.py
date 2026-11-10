"""Coherence diagnostics: how far a pair copula's own parts disagree.

A Rosenblatt pair copula averages two fitted conditional densities, and reads
its h-functions from those fits directly rather than integrating the average.
Under consistency all of these target the same population quantities; in a
finite sample they need not agree. Two diagnostics measure by how much, on a
study's evaluation grid, one row per covariate value:

- :func:`directional_rows` -- the IAE between the two directions' densities,
  and each direction's IAE and KL against the truth next to the average's,
  the KL on the grid and, given the cell's draws, by Monte Carlo;
- :func:`hfunc_rows` -- the IAE between each direct h-function and the
  integral of the averaged density it should equal:
  ``h1(u, v) = int_0^v c(u, s) ds`` and ``h2(u, v) = int_0^u c(s, v) ds``.

Both evaluate the unprojected density. :func:`normal_score_mass` integrates a
density, projected or not, over the unit square at each covariate value.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from npcc.core import RosenblattBicop
from npcc.experiments import metrics
from npcc.experiments.scenarios import EvalGrid

__all__ = ["directional_rows", "hfunc_rows", "normal_score_mass"]


def _covariates(grid: EvalGrid) -> torch.Tensor | None:
  return None if grid.x_flat is None else grid.x_flat.reshape(-1, 1)


def _columns(grid: EvalGrid) -> list[tuple[float, slice | int]]:
  """``(x, index)`` for each covariate value; one ``nan`` column if none."""
  if grid.x_axis is None:
    return [(float("nan"), slice(None))]
  return [(float(x), j) for j, x in enumerate(grid.x_axis)]


def directional_rows(
  model: RosenblattBicop,
  grid: EvalGrid,
  truth: torch.Tensor,
  holdout: tuple[EvalGrid, torch.Tensor] | None = None,
) -> list[dict[str, float]]:
  """The two directions against each other, and each against the truth.

  Parameters
  ----------
  model : RosenblattBicop
      A fitted pair copula.
  grid : EvalGrid
      The evaluation grid.
  truth : torch.Tensor
      The true density at the grid's points, in its flattened order.
  holdout : tuple of EvalGrid and torch.Tensor, optional
      The cell's draws from the true copula and the true log density at each,
      from :func:`npcc.experiments.runner.holdout_points`, with the same
      covariate values as ``grid``. Given, each density also gets its Monte
      Carlo KL.

  Returns
  -------
  list of dict
      Per covariate value: ``iae_between_directions``, and ``IAE`` and ``KL``
      against the truth for ``v_given_u``, ``u_given_v`` and ``average``, with
      ``KL_mc``, ``KL_mc_se`` and ``KL_mc_floored`` for each when ``holdout``
      is given.
  """
  v_given_u, u_given_v = model.pdf_by_direction(grid.uv, x=_covariates(grid))
  shape = grid.shape if grid.conditional else (grid.shape[0], 1)
  dens = {
    "v_given_u": v_given_u.cpu().reshape(shape),
    "u_given_v": u_given_v.cpu().reshape(shape),
  }
  dens["average"] = 0.5 * (dens["v_given_u"] + dens["u_given_v"])
  true = truth.cpu().reshape(shape)
  by_x: dict[str, list[dict[str, float]]] = {}
  if holdout is not None:
    points, log_true = holdout
    at_draws = dict(
      zip(
        ("v_given_u", "u_given_v"),
        model.pdf_by_direction(points.uv, x=_covariates(points)),
        strict=True,
      )
    )
    at_draws["average"] = 0.5 * (at_draws["v_given_u"] + at_draws["u_given_v"])
    by_x = {
      name: metrics.holdout_kl_by_x(points, log_true, d)
      for name, d in at_draws.items()
    }

  rows = []
  for x, j in _columns(grid):
    col = j if isinstance(j, int) else 0
    row: dict[str, float] = {
      "x": x,
      "iae_between_directions": float(
        (dens["v_given_u"][:, col] - dens["u_given_v"][:, col]).abs().mean()
      ),
    }
    for name, d in dens.items():
      m = metrics.grid_metrics(true[:, col], d[:, col], include_kl=True)
      row[f"{name}_IAE"] = m["IAE"]
      row[f"{name}_KL"] = m["KL"]
      for key, value in by_x.get(name, [{}] * (col + 1))[col].items():
        row[f"{name}_{key}"] = value
    rows.append(row)
  return rows


def normal_score_mass(
  model: RosenblattBicop,
  x_axis: torch.Tensor,
  sinkhorn_iters: Sequence[int | None],
  *,
  m: int = 200,
  limit: float = 6.0,
) -> dict[int | None, torch.Tensor]:
  """The density's mass over the unit square at each ``x``, per iteration count.

  The midpoint rule in normal scores: ``m`` midpoints a side on
  ``[-limit, limit]``, at ``u = Phi(s)`` with weight ``phi(s) ds``. A copula
  density that grows without bound at a corner, as Clayton's and Joe's do,
  is smooth in normal scores, so this is accurate where a uniform grid of the
  same size is not: for a Clayton copula at tau 0.7, ``m = 200`` gives the
  mass to 2e-9, against 8e-3 for 200 x 200 midpoints in ``u``.

  Parameters
  ----------
  model : RosenblattBicop
      A fitted pair copula.
  x_axis : torch.Tensor
      The covariate values.
  sinkhorn_iters : sequence of int or None
      The iteration counts, as :meth:`RosenblattBicop.pdf_by_projection`
      takes them; ``None`` is the unprojected density.
  m : int, default 200
      Midpoints a side.
  limit : float, default 6.0
      The normal-score range; ``Phi(-6)`` is about 1e-9.

  Returns
  -------
  dict
      Per iteration count, the mass at each ``x``, in ``x_axis`` order.
  """
  step = 2.0 * limit / m
  s = -limit + (torch.arange(m, dtype=torch.float64) + 0.5) * step
  normal = torch.distributions.Normal(0.0, 1.0)
  axis = normal.cdf(s)
  weight = normal.log_prob(s).exp() * step
  uu, vv = torch.meshgrid(axis, axis, indexing="ij")
  uv = torch.column_stack([uu.reshape(-1), vv.reshape(-1)])
  cell = torch.outer(weight, weight).reshape(-1)
  xs = x_axis.to(dtype=torch.float64).cpu()
  out = model.pdf_by_projection(
    uv.repeat(xs.shape[0], 1),
    x=xs.repeat_interleave(m * m).reshape(-1, 1),
    sinkhorn_iters=sinkhorn_iters,
  )
  return {
    key: (dens.cpu().reshape(xs.shape[0], m * m) * cell).sum(dim=1)
    for key, dens in out.items()
  }


def _cumulative(density: torch.Tensor, dim: int) -> torch.Tensor:
  """``int_0^{e_k}`` of a density on midpoint cells, at every cell edge.

  The result has one more entry than ``density`` along ``dim``: the integral
  at ``0``, which is zero, and then at the right edge of each cell.
  """
  m = density.shape[dim]
  cum = torch.cumsum(density, dim=dim) / m
  zero = torch.zeros_like(cum.narrow(dim, 0, 1))
  return torch.cat([zero, cum], dim=dim)


def _interp_rows(
  edges: torch.Tensor, values: torch.Tensor, at: torch.Tensor
) -> torch.Tensor:
  """Linear interpolation of each row of ``values`` (on ``edges``) at ``at``."""
  idx = torch.searchsorted(edges, at).clamp(1, edges.shape[0] - 1)
  lo, hi = edges[idx - 1], edges[idx]
  t = ((at - lo) / (hi - lo)).clamp(0.0, 1.0)
  return values[:, idx - 1] + t * (values[:, idx] - values[:, idx - 1])


def hfunc_rows(
  model: RosenblattBicop, grid: EvalGrid, *, n_int: int = 400
) -> list[dict[str, float]]:
  """Direct h-functions against integrals of the averaged density.

  Each integral is a midpoint sum over ``n_int`` equal cells of ``(0, 1)``,
  read off at the grid's points by linear interpolation between cell edges.

  Parameters
  ----------
  model : RosenblattBicop
      A fitted pair copula, unprojected (``sinkhorn_iters`` is ``None``).
  grid : EvalGrid
      The evaluation grid; its pairs are the product of its two axes.
  n_int : int, default=400
      Cells per integral.

  Returns
  -------
  list of dict
      Per covariate value: ``h1_IAE`` and ``h2_IAE``, the mean absolute gap
      between direct and integrated, and their maxima ``h1_max``, ``h2_max``.
  """
  if model.sinkhorn_iters is not None:
    raise ValueError(
      "hfunc_rows must compare against the unprojected density; got "
      f"sinkhorn_iters={model.sinkhorn_iters}"
    )
  u_axis, v_axis = grid.u_axis, grid.v_axis
  nu, nv = u_axis.shape[0], v_axis.shape[0]
  s = (torch.arange(n_int, dtype=torch.float64) + 0.5) / n_int
  edges = torch.linspace(0.0, 1.0, n_int + 1, dtype=torch.float64)

  x_all = _covariates(grid)
  h1 = model.hfunc1(grid.uv, x=x_all).cpu()
  h2 = model.hfunc2(grid.uv, x=x_all).cpu()
  shape = grid.shape if grid.conditional else (grid.shape[0], 1)
  h1, h2 = h1.reshape(shape), h2.reshape(shape)

  rows = []
  for x, j in _columns(grid):
    col = j if isinstance(j, int) else 0
    x_row = (
      None if grid.x_axis is None else torch.tensor([[x]], dtype=torch.float64)
    )
    # c(u_i, s): integrate along the second argument for h1.
    along_v = model.pdf_grid(u_axis, s, x_row=x_row).cpu()
    int_h1 = _interp_rows(edges, _cumulative(along_v, dim=1), v_axis)
    # c(s, v_k): integrate along the first argument for h2.
    along_u = model.pdf_grid(s, v_axis, x_row=x_row).cpu()
    int_h2 = _interp_rows(edges, _cumulative(along_u, dim=0).T, u_axis).T
    gap1 = (h1[:, col].reshape(nu, nv) - int_h1).abs()
    gap2 = (h2[:, col].reshape(nu, nv) - int_h2).abs()
    rows.append(
      {
        "x": x,
        "h1_IAE": float(gap1.mean()),
        "h1_max": float(gap1.max()),
        "h2_IAE": float(gap2.mean()),
        "h2_max": float(gap2.max()),
      }
    )
  return rows
