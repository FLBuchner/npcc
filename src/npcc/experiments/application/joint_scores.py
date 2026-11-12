"""Scores of a joint forecast: its log-likelihood and two scores of its draws.

Every function takes one row per test case. Draws are ``(n, m, d)``, ``m``
per case, and observations ``(n, d)``, both on the scale the score is read
on: the application divides each variable by its training standard deviation
first, so that no variable's units dominate a distance.
"""

from __future__ import annotations

import numpy as np


def joint_nll(logcop: np.ndarray, logpdf: np.ndarray) -> np.ndarray:
  """The joint negative log-likelihood of each case.

  The joint density is the copula density at the margins' probability
  integral transforms times the margins' densities, so its negative log is
  ``-(logcop + sum_j logpdf[:, j])``.

  Parameters
  ----------
  logcop : numpy.ndarray, shape (n,)
      The copula log-density of each case.
  logpdf : numpy.ndarray, shape (n, d)
      Each margin's log-density of each case.

  Returns
  -------
  numpy.ndarray, shape (n,)
      The joint negative log-likelihood.
  """
  return -(logcop + logpdf.sum(axis=1))


def inverse_margins(
  uniforms: np.ndarray, levels: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
  """Copula draws mapped to responses through each case's quantile table.

  Draw ``s`` of case ``i`` in variable ``j`` is the linear interpolation of
  ``(levels, quantiles[i, :, j])`` at ``uniforms[i, s, j]``; a level below
  the first or above the last takes the end quantile.

  Parameters
  ----------
  uniforms : numpy.ndarray, shape (n, m, d)
      Draws from the copula, in ``[0, 1]``.
  levels : numpy.ndarray, shape (L,)
      Increasing probability levels.
  quantiles : numpy.ndarray, shape (n, L, d)
      Each case's marginal quantiles at ``levels``, nondecreasing along the
      level axis.

  Returns
  -------
  numpy.ndarray, shape (n, m, d)
      The draws on the response scale.
  """
  draws = np.empty_like(uniforms, dtype=np.float64)
  for j in range(uniforms.shape[2]):
    for i in range(uniforms.shape[0]):
      draws[i, :, j] = np.interp(uniforms[i, :, j], levels, quantiles[i, :, j])
  return draws


def energy_score(draws: np.ndarray, observed: np.ndarray) -> np.ndarray:
  """The energy score of each case, estimated from its draws.

  ``ES = E||X - y|| - E||X - X'|| / 2``. The first term is the mean over the
  ``m`` draws. The second pairs the first half of the draws with the second,
  ``h = m // 2`` independent pairs, which estimates it without bias in
  ``O(m)`` memory where every pair would take ``O(m^2)``; with an odd ``m``
  the last draw enters the first term only.

  Parameters
  ----------
  draws : numpy.ndarray, shape (n, m, d)
      Draws from each case's forecast, ``m >= 2``.
  observed : numpy.ndarray, shape (n, d)
      Each case's observation.

  Returns
  -------
  numpy.ndarray, shape (n,)
      The energy score; lower is better.

  Raises
  ------
  ValueError
      If there are fewer than two draws per case.
  """
  m = draws.shape[1]
  if m < 2:
    raise ValueError(f"draws must hold at least two per case; got {m}")
  half = m // 2
  energy = np.linalg.norm(draws - observed[:, None], axis=2).mean(axis=1)
  spread = np.linalg.norm(draws[:, :half] - draws[:, half : 2 * half], axis=2)
  return energy - 0.5 * spread.mean(axis=1)


def variogram_score(
  draws: np.ndarray, observed: np.ndarray, *, p: float = 0.5
) -> np.ndarray:
  """The variogram score of order ``p`` of each case, with unit weights.

  ``VS_p = sum_{i < j} (|y_i - y_j|^p - E|X_i - X_j|^p)^2``, the expectation
  the mean over the draws.

  Parameters
  ----------
  draws : numpy.ndarray, shape (n, m, d)
      Draws from each case's forecast.
  observed : numpy.ndarray, shape (n, d)
      Each case's observation.
  p : float, default 0.5
      The order.

  Returns
  -------
  numpy.ndarray, shape (n,)
      The variogram score; lower is better.
  """
  score = np.zeros(observed.shape[0])
  for i in range(observed.shape[1]):
    for j in range(i):
      seen = np.abs(observed[:, i] - observed[:, j]) ** p
      drawn = (np.abs(draws[:, :, i] - draws[:, :, j]) ** p).mean(axis=1)
      score += (seen - drawn) ** 2
  return score
