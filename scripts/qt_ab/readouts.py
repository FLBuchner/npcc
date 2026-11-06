"""Read-out variants for the quantile-table A/B.

Each variant is a drop-in replacement for
:class:`npcc.core._quantile_table.QuantileTable`: same constructor, same
``knots`` / ``cdf`` / ``pdf`` / ``icdf``. Every variant sorts each row, as the
library does; a variant is the pair of choices applied after that:

- **smoothing**: a moving average of the slopes ``dQ/dt`` over ``width``
  normal-score units (the library's; ``0`` is none), or ``"whittaker"``, a
  weighted second-difference smoother on the knots whose weights are the
  asymptotic precision of a sample quantile in normal-score coordinates, so it
  smooths hard at the extreme levels and lightly in the middle;
- **tails**: ``"current"`` (the library's: uniform on the response scale toward
  a finite end, exponential with a 3-cell secant rate toward an infinite one)
  or ``"modeled"``: on the modeled scale, an exponential tail in ``ln(alpha)``
  (TabICLv2 appendix I.2) for every transform but probit, which continues
  linearly in the normal score. Its one parameter per side is the anchored
  least-squares fit over the outermost ``TAIL_KNOTS`` knots.

The slope floor and the median anchoring are the library's in every variant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch.nn import functional

from npcc.core._quantile_table import (
  TAIL_CELLS,
  TAIL_MIN_SLOPE,
  QuantileTable,
  normal_scores,
  smoothing_window,
)
from npcc.core._transform import ResponseTransform, normal_cdf, normal_pdf

#: Knots, the anchor included, a modeled-scale tail is fitted on.
TAIL_KNOTS: int = 4
#: Upper clamp on a modeled-scale tail's boundary slope, as a multiple of the
#: row's mean slope; the lower clamp is the library's ``TAIL_MIN_SLOPE``.
TAIL_MAX_SLOPE: float = 4.0


@dataclass(frozen=True)
class Readout:
  """One read-out variant.

  Attributes
  ----------
  smoothing : str
      ``"ma"`` or ``"whittaker"``.
  width : float
      Moving-average width, or the Whittaker smoother's bandwidth at the
      central level, in normal-score units.
  tails : str
      ``"current"`` or ``"modeled"``.
  """

  smoothing: str = "ma"
  width: float = 2.0
  tails: str = "current"

  @property
  def label(self) -> str:
    """A short, unique label."""
    smooth = (
      f"ma{self.width:g}" if self.smoothing == "ma" else f"wh{self.width:g}"
    )
    return f"{smooth}|{self.tails}"

  def is_library(self) -> bool:
    """Whether this variant is the library's read-out, bit for bit."""
    return self.smoothing == "ma" and self.tails == "current"


@lru_cache(maxsize=32)
def _whittaker_matrix(
  m: int, limit: float, width: float, device: str
) -> torch.Tensor:
  """``S`` with ``smoothed = q @ S.T``; preserves straight lines in ``t``."""
  t = normal_scores(m, limit, dtype=torch.float64)
  alpha = normal_cdf(t)
  weight = normal_pdf(t) ** 2 / (alpha * (1.0 - alpha))
  weight = weight / weight[m // 2]
  step = 2.0 * limit / (m - 1)
  lam = (width / step) ** 4
  second = torch.diff(torch.eye(m, dtype=torch.float64), n=2, dim=0)
  w = torch.diag(weight)
  smoother = torch.linalg.solve(w + lam * second.T @ second, w)
  return smoother.to(device)


class ReadoutTable:
  """A :class:`QuantileTable` stand-in built by one :class:`Readout`.

  Parameters
  ----------
  quantiles : torch.Tensor
      Predicted quantiles of the modeled response, ``(n, m)``.
  limit : float
      The outermost normal score of the levels.
  transform : ResponseTransform
      The response transform.
  eps : float
      Distance the transform keeps from a finite end of the support.
  readout : Readout
      The variant.
  min_slope : float
      Floor on every slope, as a fraction of the row's mean slope.
  """

  def __init__(
    self,
    quantiles: torch.Tensor,
    *,
    limit: float,
    transform: ResponseTransform,
    eps: float,
    readout: Readout,
    min_slope: float,
  ) -> None:
    q = torch.sort(quantiles, dim=1).values
    n_rows, m = q.shape
    step = 2.0 * limit / (m - 1)
    mid = m // 2

    if readout.smoothing == "whittaker" and readout.width > 0:
      smoother = _whittaker_matrix(m, limit, readout.width, str(q.device))
      q_smooth = q @ smoother.T.to(q.dtype)
    else:
      q_smooth = q

    slope = (q_smooth[:, 1:] - q_smooth[:, :-1]) / step
    mean_slope = (q[:, -1:] - q[:, :1]) / (2.0 * limit)
    resolution = torch.finfo(q.dtype).eps * (1.0 + q[:, mid : mid + 1].abs())

    if readout.smoothing == "ma":
      window = smoothing_window(readout.width, step)
      if window > 1:
        pad = window // 2
        padded = torch.cat(
          [slope[:, :1].expand(-1, pad), slope, slope[:, -1:].expand(-1, pad)],
          dim=1,
        )
        slope = functional.avg_pool1d(
          padded.unsqueeze(1), window, stride=1
        ).squeeze(1)
    slope = torch.maximum(
      slope, torch.maximum(min_slope * mean_slope, resolution)
    )

    knots = torch.cat(
      [torch.zeros_like(q[:, :1]), (slope * step).cumsum(dim=1)], dim=1
    )
    knots = knots - knots[:, mid : mid + 1] + q_smooth[:, mid : mid + 1]

    self._transform = transform
    self._eps = eps
    self._limit = limit
    self._step = step
    self._scores = normal_scores(m, limit, device=q.device, dtype=q.dtype)
    self._knots = knots.contiguous()
    self._slope = slope
    self._n_rows = n_rows
    self._tail_mass = 0.5 * math.erfc(limit / math.sqrt(2.0))
    self._tails = readout.tails
    self._linear_tail = transform.name == "probit"

    tail_floor = torch.maximum(TAIL_MIN_SLOPE * mean_slope, resolution)
    edge_density = math.exp(-0.5 * limit * limit) / math.sqrt(2.0 * math.pi)
    tiny = torch.finfo(q.dtype).tiny
    lo, hi = transform.support

    if readout.tails == "modeled":
      # Boundary slope dQ/dt from the anchored least-squares fit; the tail's
      # one parameter follows from it.
      k = min(TAIL_KNOTS, m)
      t_out = self._scores[:k] - self._scores[0]
      lower_rise = knots[:, :k] - knots[:, :1]
      upper_rise = knots[:, -1:] - knots[:, m - k :].flip(1)
      if self._linear_tail:
        coord = t_out
        to_slope = 1.0
      else:
        log_alpha = torch.log(normal_cdf(self._scores[:k]))
        coord = log_alpha - log_alpha[0]
        # dQ/dt = beta * phi(t) / Phi(t) at t = -limit.
        to_slope = edge_density / self._tail_mass
      denom = (coord * coord).sum()
      lower_par = (lower_rise * coord).sum(dim=1, keepdim=True) / denom
      upper_par = (upper_rise * coord).sum(dim=1, keepdim=True) / denom
      cap = TAIL_MAX_SLOPE * mean_slope
      self._lower_par = (lower_par * to_slope).clamp(tail_floor, cap) / to_slope
      self._upper_par = (upper_par * to_slope).clamp(tail_floor, cap) / to_slope
      return

    cells = min(TAIL_CELLS, m - 1)
    self._lower_finite = math.isfinite(lo)
    if self._lower_finite:
      y_first = transform.inverse(knots[:, :1])
      self._lower_width = (y_first - lo).clamp_min(tiny)
    else:
      lower_slope = (knots[:, cells : cells + 1] - knots[:, :1]) / (
        cells * step
      )
      lower_slope = torch.maximum(lower_slope, tail_floor)
      self._lower_rate = edge_density / (lower_slope * self._tail_mass)
    self._upper_finite = math.isfinite(hi)
    if self._upper_finite:
      y_last = transform.inverse(knots[:, -1:])
      self._upper_width = (hi - y_last).clamp_min(tiny)
    else:
      upper_slope = (knots[:, -1:] - knots[:, -1 - cells : m - cells]) / (
        cells * step
      )
      upper_slope = torch.maximum(upper_slope, tail_floor)
      self._upper_rate = edge_density / (upper_slope * self._tail_mass)

  @property
  def knots(self) -> torch.Tensor:
    """The regularized table."""
    return self._knots

  def _modeled(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y_clip = self._transform.clip(y, self._eps)
    return y_clip, self._transform.forward(y_clip)

  def _interior(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = self._knots.shape[1]
    index = torch.searchsorted(self._knots, z.contiguous())
    cell = (index - 1).clamp(0, m - 2)
    slope = self._slope.gather(1, cell)
    score = self._scores[cell] + (z - self._knots.gather(1, cell)) / slope
    return score, slope

  # Modeled-scale tails: closed forms of F, f (on the modeled scale) and Q.

  def _modeled_lower(
    self, z: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    d = z - self._knots[:, :1]
    par = self._lower_par
    if self._linear_tail:
      s = -self._limit + d / par
      return normal_cdf(s), normal_pdf(s) / par
    cdf = self._tail_mass * torch.exp(d / par)
    return cdf, cdf / par

  def _modeled_upper(
    self, z: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    d = z - self._knots[:, -1:]
    par = self._upper_par
    if self._linear_tail:
      s = self._limit + d / par
      return normal_cdf(s), normal_pdf(s) / par
    survival = self._tail_mass * torch.exp(-d / par)
    return 1.0 - survival, survival / par

  def cdf(self, y: torch.Tensor) -> torch.Tensor:
    """Distribution function at responses ``y``, shape ``(n, k)``."""
    _, z = self._modeled(y)
    score, _ = self._interior(z)
    out = normal_cdf(score)
    first, last = self._knots[:, :1], self._knots[:, -1:]
    if self._tails == "modeled":
      lower, _ = self._modeled_lower(z)
      upper, _ = self._modeled_upper(z)
    else:
      lo, hi = self._transform.support
      y_in = y.clamp(lo, hi)
      mass = self._tail_mass
      if self._lower_finite:
        lower = mass * (y_in - lo) / self._lower_width
      else:
        lower = mass * torch.exp(self._lower_rate * (z - first))
      if self._upper_finite:
        upper = 1.0 - mass * (hi - y_in) / self._upper_width
      else:
        upper = 1.0 - mass * torch.exp(-self._upper_rate * (z - last))
    out = torch.where(z < first, lower, out)
    return torch.where(z > last, upper, out)

  def pdf(self, y: torch.Tensor) -> torch.Tensor:
    """Density at responses ``y``, on the response scale, shape ``(n, k)``."""
    y_clip, z = self._modeled(y)
    score, slope = self._interior(z)
    jacobian = self._transform.jacobian(y_clip)
    out = normal_pdf(score) / slope * jacobian
    first, last = self._knots[:, :1], self._knots[:, -1:]
    if self._tails == "modeled":
      _, lower = self._modeled_lower(z)
      _, upper = self._modeled_upper(z)
      lower = lower * jacobian
      upper = upper * jacobian
    else:
      mass = self._tail_mass
      if self._lower_finite:
        lower = (mass / self._lower_width).expand_as(out)
      else:
        rate = self._lower_rate
        lower = mass * rate * torch.exp(rate * (z - first)) * jacobian
      if self._upper_finite:
        upper = (mass / self._upper_width).expand_as(out)
      else:
        rate = self._upper_rate
        upper = mass * rate * torch.exp(-rate * (z - last)) * jacobian
    out = torch.where(z < first, lower, out)
    return torch.where(z > last, upper, out)

  def icdf(self, p: torch.Tensor) -> torch.Tensor:
    """Quantile function at levels ``p``, on the response scale."""
    m = self._knots.shape[1]
    score = torch.special.ndtri(p)
    cell = torch.floor((score + self._limit) / self._step).long()
    cell = cell.clamp(0, m - 2)
    z = self._knots.gather(1, cell) + self._slope.gather(1, cell) * (
      score - self._scores[cell]
    )
    out = self._transform.inverse(z)
    mass = self._tail_mass
    first, last = self._knots[:, :1], self._knots[:, -1:]
    if self._tails == "modeled":
      if self._linear_tail:
        lower_z = first + self._lower_par * (score + self._limit)
        upper_z = last + self._upper_par * (score - self._limit)
      else:
        lower_z = first + self._lower_par * torch.log(p / mass)
        upper_z = last - self._upper_par * torch.log((1.0 - p) / mass)
      lower = self._transform.inverse(lower_z)
      upper = self._transform.inverse(upper_z)
    else:
      if self._lower_finite:
        lower = self._transform.support[0] + p / mass * self._lower_width
      else:
        lower = self._transform.inverse(
          first + torch.log(p / mass) / self._lower_rate
        )
      if self._upper_finite:
        upper = (
          self._transform.support[1] - (1.0 - p) / mass * self._upper_width
        )
      else:
        upper = self._transform.inverse(
          last - torch.log((1.0 - p) / mass) / self._upper_rate
        )
    out = torch.where(score < -self._limit, lower, out)
    return torch.where(score > self._limit, upper, out)


def table_factory(readout: Readout, min_slope: float = 0.2):  # noqa: ANN201 - a QuantileTable-shaped callable
  """A callable with :class:`QuantileTable`'s signature, building ``readout``.

  The library's own read-out is returned as the library class itself, so the
  baseline arm is the code the paper ran, not a copy of it.
  """

  def build(
    quantiles: torch.Tensor,
    *,
    limit: float,
    transform: ResponseTransform,
    eps: float,
    smoothing: float,
    min_slope: float = min_slope,
  ) -> QuantileTable | ReadoutTable:
    del smoothing
    if readout.is_library():
      return QuantileTable(
        quantiles,
        limit=limit,
        transform=transform,
        eps=eps,
        smoothing=readout.width,
        min_slope=min_slope,
      )
    return ReadoutTable(
      quantiles,
      limit=limit,
      transform=transform,
      eps=eps,
      readout=readout,
      min_slope=min_slope,
    )

  return build
