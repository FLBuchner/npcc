"""Error metrics for the simulation study.

- :func:`grid_metrics` — mean IAE / ISE / KL over a uv grid, at one ``x`` for
  a conditional scenario.
- :func:`marginal_diagnostics` — how far a density grid's margins are from the
  uniform-copula constraint ``int c du = int c dv = 1`` (quantifies the
  normalization axis).
"""

from __future__ import annotations

import torch

_EPS: float = 1e-12


def grid_metrics(
  y_true: torch.Tensor,
  y_hat: torch.Tensor,
  *,
  include_kl: bool = False,
) -> dict[str, float]:
  """Mean absolute / squared error (and optional KL) over a flattened grid."""
  err = y_hat - y_true

  iae = float(err.abs().mean())
  ise = float(err.square().mean())

  kl = float("nan")
  if include_kl:
    p = y_true.clamp_min(_EPS)
    q = y_hat.clamp_min(_EPS)
    kl = float((p * (p.log() - q.log())).mean())

  return {"IAE": iae, "ISE": ise, "KL": kl}


def marginal_diagnostics(
  c: torch.Tensor, u_grid: torch.Tensor, v_grid: torch.Tensor
) -> dict[str, float]:
  """Absolute deviation of a density grid's margins from the constraint = 1.

  ``u_grid`` and ``v_grid`` are midpoint grids covering the unit interval, and
  ``c`` has shape ``(len(u_grid), len(v_grid))``. Midpoint quadrature avoids
  evaluating potentially unstable copula densities at zero or one while still
  integrating over the full unit interval.
  """
  if c.shape != (u_grid.numel(), v_grid.numel()):
    raise ValueError(
      "c must have shape (len(u_grid), len(v_grid)); "
      f"got {c.shape} for ({u_grid.numel()}, {v_grid.numel()})."
    )
  int_over_v = c.mean(dim=1)
  int_over_u = c.mean(dim=0)
  err_rows = (int_over_v - 1.0).abs()
  err_cols = (int_over_u - 1.0).abs()
  return {
    "row_mean_abs_err": float(err_rows.mean()),
    "row_max_abs_err": float(err_rows.max()),
    "col_mean_abs_err": float(err_cols.mean()),
    "col_max_abs_err": float(err_cols.max()),
  }
