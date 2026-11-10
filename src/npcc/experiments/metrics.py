"""Error metrics for the simulation study.

- :func:`grid_metrics` — mean IAE / ISE / KL over a uv grid, at one ``x`` for
  a conditional scenario.
- :func:`holdout_kl` — the Monte Carlo KL, ``E[log c - log c_hat]`` over
  draws from the true copula ``c``, at one ``x`` for a conditional scenario;
  :func:`holdout_kl_by_x` at each ``x`` of a holdout grid.
- :func:`marginal_diagnostics` — how far a density grid's margins are from the
  uniform-copula constraint ``int c du = int c dv = 1`` (quantifies the
  normalization axis).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from npcc.experiments.scenarios import EvalGrid

_EPS: float = 1e-12

#: The density floor under every log of :func:`holdout_kl`, true and fitted
#: alike, as in E3 and VDC's Table 1 runner: ``log(max(pdf, 1e-300))``.
LOG_FLOOR: float = 1e-300


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


def log_density(pdf: torch.Tensor) -> torch.Tensor:
  """``log(max(pdf, LOG_FLOOR))``, the log density :func:`holdout_kl` reads.

  The floor is under ``float32``'s smallest value, so the density is read in
  ``float64`` first; in its own ``float32`` it would floor at zero.

  Parameters
  ----------
  pdf : torch.Tensor
      Density values.

  Returns
  -------
  torch.Tensor
      Their floored logs, in ``float64``.
  """
  return pdf.to(torch.float64).clamp_min(LOG_FLOOR).log()


def holdout_kl(
  log_true: torch.Tensor, pdf_hat: torch.Tensor
) -> dict[str, float]:
  """The Monte Carlo KL of a fitted density at draws from the true one.

  ``mean(log c - log c_hat)`` over the draws estimates ``KL(c || c_hat)``,
  which is nonnegative and scores the whole unit square, the band at the
  boundary included; a grid's ``sum p log(p / q)`` does neither.

  Parameters
  ----------
  log_true : torch.Tensor
      The true log density at the draws, from :func:`log_density`.
  pdf_hat : torch.Tensor
      The fitted density at the same draws.

  Returns
  -------
  dict
      ``KL_mc``; its Monte Carlo standard error ``KL_mc_se``; ``KL_mc_n``,
      the number of draws; ``KL_mc_floored``, the draws whose fitted density
      is at or under :data:`LOG_FLOOR`; and ``KL_mc_nonfinite``, those whose
      fitted density is not finite, which make ``KL_mc`` NaN.
  """
  ratio = log_true - log_density(pdf_hat)
  n = ratio.numel()
  fitted = pdf_hat.to(torch.float64)
  return {
    "KL_mc": float(ratio.mean()),
    "KL_mc_se": float(ratio.std() / n**0.5),
    "KL_mc_n": float(n),
    "KL_mc_floored": float((fitted <= LOG_FLOOR).sum()),
    "KL_mc_nonfinite": float((~fitted.isfinite()).sum()),
  }


def holdout_kl_by_x(
  grid: EvalGrid, log_true: torch.Tensor, pdf_hat: torch.Tensor
) -> list[dict[str, float]]:
  """:func:`holdout_kl` at each ``x`` of a holdout grid.

  Parameters
  ----------
  grid : EvalGrid
      The draws, from :func:`npcc.experiments.scenarios.holdout_grid`.
  log_true : torch.Tensor
      The true log density at the draws, flattened.
  pdf_hat : torch.Tensor
      The fitted density at the draws, flattened.

  Returns
  -------
  list of dict
      One entry per ``x`` in ``x_axis`` order, one for an unconditional grid.
  """
  if not grid.conditional:
    return [holdout_kl(log_true, pdf_hat.cpu())]
  truth = log_true.reshape(grid.shape)
  fitted = pdf_hat.cpu().reshape(grid.shape)
  return [holdout_kl(truth[:, j], fitted[:, j]) for j in range(grid.shape[1])]


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
