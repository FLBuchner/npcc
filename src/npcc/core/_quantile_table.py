"""A conditional distribution read off a table of predicted quantiles.

A quantile backend predicts, for every conditioning row, the quantiles
``q_k = Q(alpha_k)`` of the modeled response ``z`` at fixed levels. This is
where such a table becomes a distribution -- a density, a distribution function
and a quantile function that agree with one another -- in three steps.

**The levels are a normal grid.** ``alpha_k = Phi(t_k)`` with ``t_k`` equally
spaced on ``[-L, L]``. This is pyvinecopulib's grid for nonparametric pair
copulas, laid on the probability axis instead of the unit square: the levels
crowd into the tails, where a quantile function bends, and thin out in the
middle, where it does not. ``Q`` is interpolated linearly in ``t`` rather than
in ``alpha``, so between knots

    F(z) = Phi(t(z)),   f(z) = phi(t(z)) / Q'(t),   F^{-1}(p) = Q(Phi^{-1}(p))

hold exactly and in closed form: the density is the derivative of the
distribution function, the quantile function is its inverse, and a normal
distribution on the modeled scale -- a straight line in ``t`` -- is reproduced
exactly.

**The slopes are regularized.** A fitted table is noisy across levels, the
more so when each level is a separate model, and sorting a row whose quantiles
crossed leaves cells of near-zero width and near-infinite density. The slopes
``dQ/dt`` are therefore averaged over a window ``smoothing`` normal-score units
wide, floored at ``min_slope`` times the row's mean slope, and integrated back
into a table anchored at its median.

**The tails carry exactly the mass the table leaves out.** The table spans the
levels ``[Phi(-L), Phi(L)]``, so ``Phi(-L)`` of mass lies beyond it on each
side, and it is placed according to the response's support:

- toward a *finite* end, uniformly on the response scale between that end and
  the outermost knot. This is the device pyvinecopulib's normal grid uses: the
  boundary takes the value at the last knot, so nothing is extrapolated;
- toward an *infinite* end, exponentially on the modeled scale. Its rate
  matches the density the outer few cells imply at the outermost knot, and is
  floored: those cells are the least reliable part of a fitted table, so the
  rate is read from several of them rather than from the last alone, and the
  density is continuous there only when they agree.

Every query lands in a cell or a tail, so the density is positive across the
whole support and no log-likelihood is ``-inf``.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional

from npcc.core._transform import ResponseTransform, normal_cdf, normal_pdf

__all__ = ["QuantileTable", "normal_scores", "smoothing_window"]

#: How many outer cells an exponential tail reads its rate from.
TAIL_CELLS: int = 3
#: The floor on that tail's slope, as a fraction of the row's mean slope.
TAIL_MIN_SLOPE: float = 0.25


def normal_scores(
  n: int,
  limit: float,
  *,
  device: str | torch.device | None = None,
  dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
  """Return ``n`` normal scores equally spaced on ``[-limit, limit]``.

  Parameters
  ----------
  n : int
      Number of scores.
  limit : float
      The outermost score.
  device : str or torch.device, optional
      Device of the result.
  dtype : torch.dtype, default=torch.float64
      Dtype of the result.

  Returns
  -------
  torch.Tensor
      The scores, shape ``(n,)``.
  """
  return torch.linspace(-limit, limit, n, dtype=dtype, device=device)


def smoothing_window(smoothing: float, step: float) -> int:
  """Return the odd number of cells closest to ``smoothing`` score units.

  Parameters
  ----------
  smoothing : float
      Window width on the normal-score axis.
  step : float
      Width of one cell on that axis.

  Returns
  -------
  int
      An odd window of at least one cell; one cell is no smoothing.
  """
  return max(1, 2 * round((smoothing / step - 1.0) / 2.0) + 1)


class QuantileTable:
  """A distribution per row, from a table of quantiles at normal-grid levels.

  Parameters
  ----------
  quantiles : torch.Tensor
      Predicted quantiles of the modeled response, shape ``(n, m)``, at the
      levels ``Phi(normal_scores(m, limit))``. Rows need not be sorted.
  limit : float
      The outermost normal score of the levels.
  transform : ResponseTransform
      The map from the response to the modeled scale.
  eps : float
      Distance the transform keeps from a finite end of the support.
  smoothing : float
      Width of the slope-averaging window, in normal-score units.
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
    smoothing: float,
    min_slope: float,
  ) -> None:
    q = torch.sort(quantiles, dim=1).values
    n_rows, m = q.shape
    step = 2.0 * limit / (m - 1)
    mid = m // 2

    slope = (q[:, 1:] - q[:, :-1]) / step
    mean_slope = (q[:, -1:] - q[:, :1]) / (2.0 * limit)
    # The last line of defense against a row with no spread at all, which a
    # relative floor cannot lift: below this a slope is rounding noise.
    resolution = torch.finfo(q.dtype).eps * (1.0 + q[:, mid : mid + 1].abs())

    window = smoothing_window(smoothing, step)
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
    knots = knots - knots[:, mid : mid + 1] + q[:, mid : mid + 1]

    self._transform = transform
    self._eps = eps
    self._limit = limit
    self._step = step
    self._scores = normal_scores(m, limit, device=q.device, dtype=q.dtype)
    self._knots = knots.contiguous()
    self._slope = slope
    self._n_rows = n_rows
    # Mass beyond each end of the table; the grid is symmetric.
    self._tail_mass = 0.5 * math.erfc(limit / math.sqrt(2.0))

    lo, hi = transform.support
    cells = min(TAIL_CELLS, m - 1)
    tail_floor = torch.maximum(TAIL_MIN_SLOPE * mean_slope, resolution)
    edge_density = math.exp(-0.5 * limit * limit) / math.sqrt(2.0 * math.pi)
    tiny = torch.finfo(q.dtype).tiny

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
    """The regularized, strictly increasing table, shape ``(n, m)``."""
    return self._knots

  def _check_rows(self, values: torch.Tensor) -> None:
    if values.ndim != 2 or values.shape[0] != self._n_rows:
      raise ValueError(
        f"values must have shape ({self._n_rows}, k), one row per table row; "
        f"got {tuple(values.shape)}"
      )

  def _modeled(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip responses and map them to the modeled scale."""
    y_clip = self._transform.clip(y, self._eps)
    return y_clip, self._transform.forward(y_clip)

  def _interior(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normal score and slope of the cell holding each ``z``."""
    m = self._knots.shape[1]
    index = torch.searchsorted(self._knots, z.contiguous())
    cell = (index - 1).clamp(0, m - 2)
    slope = self._slope.gather(1, cell)
    score = self._scores[cell] + (z - self._knots.gather(1, cell)) / slope
    return score, slope

  def cdf(self, y: torch.Tensor) -> torch.Tensor:
    """Evaluate the distribution function at responses ``y``.

    Parameters
    ----------
    y : torch.Tensor
        Responses, shape ``(n, k)``: row ``i`` is evaluated under table row
        ``i``.

    Returns
    -------
    torch.Tensor
        Probabilities, shape ``(n, k)``.
    """
    self._check_rows(y)
    _, z = self._modeled(y)
    score, _ = self._interior(z)
    out = normal_cdf(score)

    # A tail toward a finite end is linear on the response scale, so it reads
    # the response itself rather than its clipped stand-in: the clip exists
    # only to keep the map to the modeled scale finite.
    lo, hi = self._transform.support
    y_in = y.clamp(lo, hi)
    first, last = self._knots[:, :1], self._knots[:, -1:]
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
    """Evaluate the density at responses ``y``, on the response scale.

    Parameters
    ----------
    y : torch.Tensor
        Responses, shape ``(n, k)``: row ``i`` is evaluated under table row
        ``i``.

    Returns
    -------
    torch.Tensor
        Densities, shape ``(n, k)``.
    """
    self._check_rows(y)
    y_clip, z = self._modeled(y)
    score, slope = self._interior(z)
    jacobian = self._transform.jacobian(y_clip)
    out = normal_pdf(score) / slope * jacobian

    first, last = self._knots[:, :1], self._knots[:, -1:]
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
    """Evaluate the quantile function at levels ``p``, on the response scale.

    Parameters
    ----------
    p : torch.Tensor
        Levels strictly inside ``(0, 1)``, shape ``(n, k)``: row ``i`` is
        evaluated under table row ``i``.

    Returns
    -------
    torch.Tensor
        Responses, shape ``(n, k)``.
    """
    self._check_rows(p)
    m = self._knots.shape[1]
    score = torch.special.ndtri(p)
    cell = torch.floor((score + self._limit) / self._step).long()
    cell = cell.clamp(0, m - 2)
    z = self._knots.gather(1, cell) + self._slope.gather(1, cell) * (
      score - self._scores[cell]
    )
    out = self._transform.inverse(z)

    mass = self._tail_mass
    if self._lower_finite:
      lo = self._transform.support[0]
      lower = lo + p / mass * self._lower_width
    else:
      lower = self._transform.inverse(
        self._knots[:, :1] + torch.log(p / mass) / self._lower_rate
      )
    if self._upper_finite:
      hi = self._transform.support[1]
      upper = hi - (1.0 - p) / mass * self._upper_width
    else:
      upper = self._transform.inverse(
        self._knots[:, -1:] - torch.log((1.0 - p) / mass) / self._upper_rate
      )

    out = torch.where(score < -self._limit, lower, out)
    return torch.where(score > self._limit, upper, out)
