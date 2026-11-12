"""The joint scores of the weather application's copula arms.

Each estimator is checked against its formula written out case by case.
"""

from __future__ import annotations

import numpy as np
import pytest

from npcc.experiments.application import joint_scores


def test_the_energy_score_is_its_formula() -> None:
  """The energy estimator must be the mean distance less half the halves'.

  A second term over the wrong pairs, or not halved, still gives a plausible
  number; only the formula, written out, tells them apart.
  """
  rng = np.random.default_rng(0)
  draws = rng.normal(size=(4, 7, 3))
  observed = rng.normal(size=(4, 3))
  want = []
  for i in range(4):
    first = np.mean([np.linalg.norm(x - observed[i]) for x in draws[i]])
    pairs = [np.linalg.norm(draws[i, s] - draws[i, s + 3]) for s in range(3)]
    want.append(first - 0.5 * np.mean(pairs))
  got = joint_scores.energy_score(draws, observed)
  np.testing.assert_allclose(got, want, rtol=1e-12)


def test_the_variogram_score_is_its_formula() -> None:
  """The variogram score must sum every pair once, at order 1/2.

  A pair counted twice, or the order applied to the mean rather than to each
  draw, changes the score's scale and its ranking of forecasts.
  """
  rng = np.random.default_rng(1)
  draws = rng.normal(size=(5, 6, 4))
  observed = rng.normal(size=(5, 4))
  want = np.zeros(5)
  for n in range(5):
    for i in range(4):
      for j in range(i + 1, 4):
        seen = abs(observed[n, i] - observed[n, j]) ** 0.5
        drawn = np.mean(
          [abs(draws[n, s, i] - draws[n, s, j]) ** 0.5 for s in range(6)]
        )
        want[n] += (seen - drawn) ** 2
  got = joint_scores.variogram_score(draws, observed)
  np.testing.assert_allclose(got, want, rtol=1e-12)


def test_the_energy_score_needs_two_draws() -> None:
  """One draw has no second half, and its spread term would be empty."""
  with pytest.raises(ValueError, match="at least two"):
    joint_scores.energy_score(np.zeros((2, 1, 3)), np.zeros((2, 3)))


def test_a_draw_at_a_level_is_that_levels_quantile() -> None:
  """The inverse margins must read each case's own quantile table.

  A draw mapped through another case's table, or another variable's, lands
  on the wrong quantile, which the synthetic tables below tell apart; and a
  draw between two levels must interpolate linearly between their quantiles.
  """
  rng = np.random.default_rng(2)
  levels = np.array([0.1, 0.5, 0.9])
  quantiles = np.sort(rng.normal(size=(4, 3, 2)), axis=1)
  at_levels = np.broadcast_to(levels[None, :, None], (4, 3, 2))
  np.testing.assert_array_equal(
    joint_scores.inverse_margins(at_levels, levels, quantiles), quantiles
  )
  midway = np.full((4, 1, 2), 0.3)
  np.testing.assert_allclose(
    joint_scores.inverse_margins(midway, levels, quantiles)[:, 0],
    0.5 * (quantiles[:, 0] + quantiles[:, 1]),
    rtol=1e-12,
  )
