"""Tests for the distribution a quantile table is read as."""

from __future__ import annotations

import math

import pytest
import torch

from npcc.core import QuantileTableConfig
from npcc.core._quantile_table import (
  QuantileTable,
  Smoother,
  smoothing_window,
  weighted_smoother,
)
from npcc.core._transform import normal_cdf, resolve_transform
from tests.conftest import TAIL_MASS, NormalQuantileBackend

LIMIT = QuantileTableConfig().normal_limit
SCORES = QuantileTableConfig().normal_scores()


def noisy_table(
  transform: str, *, seed: int = 0, rows: int = 3
) -> torch.Tensor:
  """A plausible fitted table: a skewed location-scale row plus level noise."""
  generator = torch.Generator().manual_seed(seed)
  # Probit with a scale below one keeps the response density bounded at the
  # ends, so a uniform grid can integrate it.
  rows_by_transform = {
    "identity": 1.0 + 2.0 * SCORES + 0.2 * SCORES.square(),
    "logit": 0.3 + 1.4 * SCORES,
    "probit": 0.2 + 0.8 * SCORES,
  }
  row = rows_by_transform[transform]
  noise = 0.1 * torch.randn(
    (rows, SCORES.shape[0]), generator=generator, dtype=torch.float64
  )
  return row + noise


def make_table(
  quantiles: torch.Tensor,
  transform: str,
  *,
  smoothing: float = 1.0,
  min_slope: float = 0.2,
  smoother: Smoother = "moving-average",
) -> QuantileTable:
  return QuantileTable(
    quantiles,
    limit=LIMIT,
    transform=resolve_transform(transform),
    eps=1e-12,
    smoothing=smoothing,
    min_slope=min_slope,
    smoother=smoother,
  )


def response_grid(transform: str, rows: int) -> torch.Tensor:
  """A fine grid over the support, wide enough to hold all but ~1e-9 of it."""
  if transform == "identity":
    y = torch.linspace(-60.0, 80.0, 400_001, dtype=torch.float64)
  else:
    y = torch.linspace(1e-12, 1.0 - 1e-12, 400_001, dtype=torch.float64)
  return y.unsqueeze(0).expand(rows, -1).contiguous()


@pytest.mark.parametrize("transform", ["logit", "probit", "identity"])
def test_tails_hold_exactly_the_mass_beyond_the_table(transform: str) -> None:
  """The distribution function is ``Phi(-L)`` at the first knot, and so on.

  Anything else means the tails and the table disagree about how much mass
  each holds, and the density no longer integrates to one.
  """
  table = make_table(noisy_table(transform), transform)
  g = resolve_transform(transform)
  ends = g.inverse(table.knots[:, [0, -1]])

  expected = torch.tensor([TAIL_MASS, 1.0 - TAIL_MASS], dtype=torch.float64)
  torch.testing.assert_close(table.cdf(ends), expected.expand_as(ends))


@pytest.mark.parametrize("transform", ["logit", "probit", "identity"])
def test_density_integrates_to_one(transform: str) -> None:
  """The tails make the table a proper distribution, for every tail shape.

  With density zero beyond the table, ``2 * Phi(-L)`` of mass was missing; a
  tail of the wrong height would add or remove mass instead.
  """
  table = make_table(noisy_table(transform), transform)
  y = response_grid(transform, 3)

  mass = torch.trapezoid(table.pdf(y), y, dim=1)

  torch.testing.assert_close(
    mass, torch.ones(3, dtype=torch.float64), atol=2e-4, rtol=0.0
  )


@pytest.mark.parametrize("transform", ["logit", "probit", "identity"])
def test_cdf_inverts_icdf_everywhere(transform: str) -> None:
  """``cdf(icdf(p)) == p`` in the table and in both tails.

  The inverse h-function is the quantile function, so a mismatch would make a
  simulated vine reproduce the wrong conditional distribution.
  """
  table = make_table(noisy_table(transform), transform)
  p = torch.tensor(
    [
      1e-10,
      1e-5,
      TAIL_MASS / 2,
      0.01,
      0.3,
      0.5,
      0.9,
      1 - TAIL_MASS / 2,
      1 - 1e-7,
    ],
    dtype=torch.float64,
  ).expand(3, -1)

  torch.testing.assert_close(table.cdf(table.icdf(p)), p, rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("transform", ["logit", "probit", "identity"])
def test_pdf_is_the_derivative_of_cdf(transform: str) -> None:
  """The density and the h-function describe one distribution.

  The previous reconstruction interpolated the density and the distribution
  function separately, so the pair copula's density was not the derivative of
  its h-function. Checked by central differences at random levels, which land
  in the table and in both tails.
  """
  table = make_table(noisy_table(transform), transform)
  generator = torch.Generator().manual_seed(2)
  levels = 1e-4 + (1 - 2e-4) * torch.rand(
    (3, 400), generator=generator, dtype=torch.float64
  )
  y = table.icdf(levels)
  h = 1e-6 * (1.0 + y.abs())

  numeric = (table.cdf(y + h) - table.cdf(y - h)) / (2 * h)

  relative = (numeric - table.pdf(y)).abs() / table.pdf(y)
  assert relative.median() < 1e-6


def test_density_is_positive_with_tied_extreme_quantiles() -> None:
  """Tied outer quantiles still give a positive, finite density everywhere.

  A tree backend's extreme quantiles collapse onto a leaf's minimum. The old
  reconstruction read that as zero width and returned density zero beyond it,
  hence ``-inf`` log-likelihoods; it is the case that motivated the tails.
  """
  q = noisy_table("logit").clone()
  q[:, :5] = q[:, 4:5]
  q[:, -4:] = q[:, -4:-3]
  table = make_table(q, "logit", smoothing=0.0)
  y = torch.logspace(-11, 0, 500, dtype=torch.float64).clamp_max(1 - 1e-11)
  y = y.unsqueeze(0).expand(3, -1).contiguous()

  density = table.pdf(y)

  assert torch.isfinite(density).all()
  assert (density > 0).all()


def test_min_slope_bounds_the_density() -> None:
  """A near-tie between two quantiles cannot produce an unbounded density.

  Sorting crossed quantiles leaves such ties; the old floor allowed densities
  up to ``1e6``, which is what made a quantile-table reconstruction spike.
  """
  q = (2.0 * SCORES).unsqueeze(0).clone()
  q[0, 10] = q[0, 11] - 1e-9
  table = make_table(q, "identity", smoothing=0.0, min_slope=0.2)
  y = torch.linspace(-8.0, 8.0, 20_001, dtype=torch.float64).unsqueeze(0)

  mean_slope = 2.0
  bound = 1.0 / math.sqrt(2.0 * math.pi) / (0.2 * mean_slope)
  assert table.pdf(y).max() <= bound * (1 + 1e-12)


def test_crossed_rows_are_sorted() -> None:
  """Rows need not arrive monotone; a crossed row reads as its sorted self."""
  q = noisy_table("logit", rows=1)
  shuffled = q[
    :, torch.randperm(q.shape[1], generator=torch.Generator().manual_seed(1))
  ]
  y = torch.linspace(0.01, 0.99, 50, dtype=torch.float64).unsqueeze(0)

  torch.testing.assert_close(
    make_table(shuffled, "logit").pdf(y), make_table(q, "logit").pdf(y)
  )


@pytest.mark.parametrize("smoother", ["moving-average", "weighted"])
@pytest.mark.parametrize("smoothing", [0.0, 0.65, 1.0, 2.0, 4.0])
def test_smoothing_leaves_a_normal_table_unchanged(
  smoothing: float, smoother: Smoother
) -> None:
  """A straight line in normal scores has nothing to smooth.

  So the hermetic normal backends are exact whatever a backend's default
  smoothing is, and a smoother that moved such a table would be biased.
  """
  q = (0.5 + 1.7 * SCORES).unsqueeze(0)

  table = make_table(q, "identity", smoothing=smoothing, smoother=smoother)

  torch.testing.assert_close(table.knots, q)


def test_weighted_smoother_keeps_the_middle_and_smooths_the_ends() -> None:
  """A level keeps more of its own estimate the more precise that estimate is.

  The weights are the precision of a sample quantile, highest at the median.
  Inverted, they would smooth the peak of the density hardest and leave the
  noisiest levels -- the outermost, which a per-level model estimates from a
  handful of observations -- almost as fitted.
  """
  own = weighted_smoother(21, LIMIT, 0.65).diagonal()

  assert bool((torch.diff(own[:11]) > 0).all())
  assert own[10] > 3 * own[0]
  torch.testing.assert_close(own, own.flip(0))


def clayton_conditional(
  theta: float, u: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Exact ``V | U = u`` under Clayton: its quantiles and its density.

  Returns the logit-scale quantiles at the default levels, an evaluation grid
  on ``v`` uniform in its normal score, the quadrature weights of that grid,
  and the true conditional density on it.
  """
  alpha = normal_cdf(SCORES)
  v_q = (
    (alpha * u ** (theta + 1)) ** (-theta / (1 + theta)) + 1 - u**-theta
  ) ** (-1 / theta)
  s = torch.linspace(-8.0, 8.0, 40001, dtype=torch.float64)
  v = normal_cdf(s)
  keep = (v > 0) & (v < 1)
  s, v = s[keep], v[keep]
  weight = torch.exp(-0.5 * s * s) / math.sqrt(2 * math.pi) * (16.0 / 40000)
  # A float to a float power is typed Any; the result is a tensor.
  density: torch.Tensor = (
    (1 + theta)
    * (u * v) ** (-theta - 1)
    * (u**-theta + v**-theta - 1) ** (-2 - 1 / theta)
  )
  return torch.logit(v_q).unsqueeze(0), v.unsqueeze(0), weight, density


@pytest.mark.parametrize(
  ("smoother", "smoothing", "iae"),
  [
    ("moving-average", 0.0, 0.0283),
    ("moving-average", 2.0, 0.0887),
    ("weighted", 0.65, 0.1162),
  ],
)
def test_bias_on_an_exact_table_is_the_one_the_ab_measured(
  smoother: Smoother, smoothing: float, iae: float
) -> None:
  """The smoothers keep the bias they had when the A/B chose between them.

  On the exact conditional quantiles of a Clayton copula at tau = 0.7, the
  weighted smoother at 0.65 is *more* biased than the 7-cell average; it was
  adopted for the per-level tree backends because it removes more of their
  noise than it adds bias (``scripts/qt_ab``). A change to either smoother
  that moved these numbers would void that comparison.
  """
  q, v, weight, truth = clayton_conditional(theta=2 * 0.7 / 0.3, u=0.5)
  table = make_table(q, "logit", smoothing=smoothing, smoother=smoother)

  error = ((table.pdf(v)[0] - truth).abs() * weight).sum()

  assert float(error) == pytest.approx(iae, abs=5e-4)


def test_an_unknown_smoother_is_refused() -> None:
  """A misspelled smoother is refused, not silently read as no smoothing."""
  with pytest.raises(ValueError, match="smoother must be"):
    QuantileTableConfig(smoother="lowess")  # ty: ignore[invalid-argument-type] - the wrong value is the point
  with pytest.raises(ValueError, match="smoother must be"):
    make_table(noisy_table("logit"), "logit", smoother="lowess")  # ty: ignore[invalid-argument-type] - the wrong value is the point


def test_exponential_tail_meets_a_straight_table_continuously() -> None:
  """Where the outer cells agree, the tail continues the table without a jump.

  The rate is read off the outer few cells rather than the outermost one, so a
  noisy table may jump at its edge; a normal one, whose cells all agree, must
  not, or the rate is wrong rather than merely robust.
  """
  q = (0.5 + 1.7 * SCORES).unsqueeze(0)
  table = make_table(q, "identity", smoothing=2.0)
  first, last = table.knots[:, :1], table.knots[:, -1:]
  h = 1e-7

  torch.testing.assert_close(
    table.pdf(first + h), table.pdf(first - h), rtol=1e-4, atol=0.0
  )
  torch.testing.assert_close(
    table.pdf(last - h), table.pdf(last + h), rtol=1e-4, atol=0.0
  )


@pytest.mark.parametrize(
  ("smoothing", "cells"), [(0.0, 1), (0.3, 1), (1.0, 3), (2.0, 7)]
)
def test_smoothing_window_is_measured_in_normal_scores(
  smoothing: float, cells: int
) -> None:
  """The window is a width on the score axis, not a number of levels.

  So a given ``smoothing`` means the same thing at any ``n_quantiles``.
  """
  step = 2 * LIMIT / 20

  assert smoothing_window(smoothing, step) == cells


def test_values_must_have_one_row_per_table_row() -> None:
  """A query with the wrong row count is refused rather than broadcast."""
  table = make_table(noisy_table("logit"), "logit")

  with pytest.raises(ValueError, match=r"must have shape \(3, k\)"):
    table.pdf(torch.full((2, 4), 0.5, dtype=torch.float64))


def test_margin_uses_its_backends_smoothing_unless_configured() -> None:
  """``smoothing=None`` defers to the backend; a number overrides it.

  The per-backend default is what lets a per-level tree backend smooth more
  than a joint model without every caller having to know which is which.
  """
  default = NormalQuantileBackend(transform="identity")
  configured = NormalQuantileBackend(
    transform="identity",
    quantile_table_config=QuantileTableConfig(smoothing=0.0),
  )

  assert default.smoothing == NormalQuantileBackend.default_smoothing
  assert configured.smoothing == 0.0


def test_margin_uses_its_backends_smoother_unless_configured() -> None:
  """``smoother=None`` defers to the backend; a name overrides it."""
  default = NormalQuantileBackend(transform="identity")
  configured = NormalQuantileBackend(
    transform="identity",
    quantile_table_config=QuantileTableConfig(smoother="weighted"),
  )

  assert default.smoother == "moving-average"
  assert configured.smoother == "weighted"


@pytest.mark.parametrize(
  ("package", "module", "name"),
  [
    ("lightgbm", "lightgbm_quantile", "LightGBMQuantileBackend"),
    ("xgboost", "xgboost_quantile", "XGBoostQuantileBackend"),
    ("catboost", "catboost_quantile", "CatBoostQuantileBackend"),
  ],
)
def test_per_level_tree_backends_default_to_the_weighted_smoother(
  package: str, module: str, name: str
) -> None:
  """The tree backends read their tables with what the A/B chose for them.

  On E1's cells the weighted smoother at 0.65 beat the 7-cell average for
  every tree backend at every sample size; a backend left on the old default
  would quietly report the read-out the A/B rejected.
  """
  pytest.importorskip(package)
  backend = getattr(
    __import__(f"npcc.core.backends.{module}", fromlist=[name]), name
  )

  assert backend.default_smoother == "weighted"
  assert backend.default_smoothing == 0.65


def test_levels_are_the_normal_grid() -> None:
  """The levels are ``Phi`` of equally spaced normal scores.

  The table interpolates linearly in those scores, so levels that drifted
  from them -- linearly spaced ones, say -- would be read at the wrong
  abscissae and bias every density.
  """
  config = QuantileTableConfig(n_quantiles=11, normal_limit=2.0)
  scores = torch.linspace(-2.0, 2.0, 11, dtype=torch.float64)

  torch.testing.assert_close(config.normal_scores(), scores)
  torch.testing.assert_close(
    config.alphas(), torch.special.ndtr(scores), rtol=1e-12, atol=0.0
  )
  assert QuantileTableConfig().alphas().shape == (21,)


def test_configuration_is_immutable() -> None:
  """A shared configuration cannot be changed under the margins holding it."""
  config = QuantileTableConfig()

  with pytest.raises(AttributeError):
    # Assigning to a frozen dataclass on purpose: the guard under test.
    config.n_quantiles = 11  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
  "settings",
  [
    {"n_quantiles": 4},
    {"normal_limit": 0.0},
    {"normal_limit": 7.0},
    {"smoothing": -1.0},
    {"min_slope": 1.0},
    {"min_slope": -0.1},
  ],
)
def test_rejects_invalid_configuration(settings: dict[str, float]) -> None:
  """A setting the reconstruction cannot use fails at construction.

  Beyond its limits each one fails later and less legibly: too few levels
  leave no interior cells, a limit past about 8 rounds the outer level to 1,
  and a slope floor of 1 flattens every table to its mean slope.
  """
  with pytest.raises(ValueError, match="must"):
    QuantileTableConfig(**settings)  # ty: ignore[invalid-argument-type]
