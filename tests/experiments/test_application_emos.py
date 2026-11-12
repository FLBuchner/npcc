"""The EMOS baselines: their closed forms, their fits and their predictions.

The pinned values were computed by FLBuchner's ``application/emos_*.py``
(npcc-paper PR #4, head aa3d5d4) on :func:`_sample`, so a port that fits a
different objective, starts elsewhere or stops early fails here.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy import integrate, stats
from scipy.special import expit, logit

from npcc.experiments.application.emos import (
  BetaEMOS,
  GaussianEMOS,
  SeasonalGaussianEMOS,
  TruncatedNormalEMOS,
  beta_crps,
  emos_for,
  gaussian_crps,
  truncated_normal_crps,
)

Sample = tuple[np.ndarray, np.ndarray]


def _sample(
  n: int = 300, seed: int = 20261112
) -> tuple[Sample, Sample, Sample]:
  """A Gaussian, a positive and a unit-interval response, with covariates.

  Each design is the response's ensemble mean, then the annual sine and
  cosine; the humidity's mean is in percent, as in the data.
  """
  rng = np.random.default_rng(seed)
  day = rng.uniform(0.0, 2.0 * np.pi, n)
  annual = np.column_stack([np.sin(day), np.cos(day)])
  m = rng.normal(10.0, 5.0, n)
  gauss = 1.0 + 0.9 * m + 0.8 * annual[:, 0] + rng.normal(0.0, 1.5, n)
  w = rng.gamma(4.0, 1.5, n)
  mixing = w * rng.lognormal(0.0, 0.25, n) + 0.05
  h = rng.uniform(0.2, 0.95, n)
  mean = expit(logit(h) + 0.3 * annual[:, 1])
  frac = rng.beta(mean * 30.0, (1.0 - mean) * 30.0)
  return (
    (np.column_stack([m, annual]), gauss),
    (np.column_stack([w, annual]), mixing),
    (np.column_stack([100.0 * h, annual]), frac),
  )


def _crps_by_quadrature(
  cdf: Callable[[float], float], y: float, lo: float, hi: float
) -> float:
  """``int (F(z) - 1{z >= y})^2 dz`` over ``[lo, hi]``, split at ``y``."""
  below = integrate.quad(lambda z: cdf(z) ** 2, lo, y, limit=200)[0]
  above = integrate.quad(lambda z: (1.0 - cdf(z)) ** 2, y, hi, limit=200)[0]
  return float(below) + float(above)


@pytest.mark.parametrize("y", [-2.0, 0.3, 4.5])
def test_the_gaussian_crps_is_the_integral(y: float) -> None:
  """A wrong closed form would fit every Gaussian EMOS to the wrong target."""
  dist = stats.norm(loc=0.7, scale=1.3)
  expected = _crps_by_quadrature(dist.cdf, y, -20.0, 20.0)

  assert float(gaussian_crps(np.array(y), 0.7, 1.3)) == pytest.approx(
    expected, rel=1e-7
  )


@pytest.mark.parametrize(("y", "mu"), [(0.4, 1.0), (3.0, 1.0), (2.0, -1.5)])
def test_the_truncated_normal_crps_is_the_integral(y: float, mu: float) -> None:
  """The truncated CRPS has a mass term near zero that is easy to drop."""
  sigma = 1.2
  dist = stats.truncnorm(-mu / sigma, np.inf, loc=mu, scale=sigma)
  expected = _crps_by_quadrature(dist.cdf, y, 0.0, 30.0)

  got = float(truncated_normal_crps(np.array(y), np.array(mu), sigma))
  assert got == pytest.approx(expected, rel=1e-6)


def test_an_observation_below_zero_scores_its_distance_to_it() -> None:
  """The support ends at zero; an observation below adds its distance."""
  at_zero = truncated_normal_crps(np.array(0.0), np.array(1.0), 1.0)
  below = truncated_normal_crps(np.array(-0.5), np.array(1.0), 1.0)

  assert float(below) == pytest.approx(float(at_zero) + 0.5)


@pytest.mark.parametrize("y", [0.05, 0.4, 0.93])
def test_the_beta_crps_is_the_integral(y: float) -> None:
  """A beta CRPS that misplaces a shape would favor the wrong precision."""
  a, b = 3.0, 5.5
  expected = _crps_by_quadrature(stats.beta(a, b).cdf, y, 0.0, 1.0)

  assert float(beta_crps(np.array(y), np.array(a), np.array(b))) == (
    pytest.approx(expected, rel=1e-7)
  )


def test_the_fits_reproduce_his_on_a_fixed_sample() -> None:
  """The port must be his estimator, not one that looks like it.

  The training CRPS and the parameters are his to rounding: the same
  objective, start, optimizer and bounds.
  """
  (xg, yg), (xm, ym), (xh, yh) = _sample()

  gauss = GaussianEMOS().fit(yg, x=xg[:, :1])
  assert gauss.parameters["training_crps"] == pytest.approx(
    0.8685042830215227, rel=1e-10
  )
  np.testing.assert_allclose(
    [gauss.center, *gauss.coefficients, gauss.sigma],
    [1.1136801350101444, 0.8944588671821357, 1.5248228445855403],
    rtol=1e-8,
  )

  seasonal = SeasonalGaussianEMOS().fit(yg, x=xg)
  assert seasonal.parameters["training_crps"] == pytest.approx(
    0.8277521131590121, rel=1e-10
  )
  np.testing.assert_allclose(
    [seasonal.center, *seasonal.coefficients, seasonal.sigma],
    [
      0.970607139172555,
      0.9051637688863122,
      0.6792212219069828,
      -0.029358142106530136,
      1.455242227114993,
    ],
    rtol=1e-8,
  )

  truncated = TruncatedNormalEMOS().fit(ym, x=xm)
  assert truncated.parameters["training_crps"] == pytest.approx(
    0.9287927005124663, rel=1e-10
  )
  np.testing.assert_allclose(
    [*truncated.theta, truncated.sigma],
    [
      0.32140704232984524,
      0.9459867896058585,
      0.2542663533854612,
      -0.0048383153535694065,
      1.4100112377871126,
    ],
    rtol=1e-6,
  )

  # The beta CRPS is flat at its minimum and its gradient a finite
  # difference, so the stopping point moves by about 1e-5 with BLAS's thread
  # count while the objective agrees to 1e-9.
  beta = BetaEMOS().fit(yh, x=xh)
  assert beta.parameters["training_crps"] == pytest.approx(
    0.046093573905725126, rel=1e-8
  )
  np.testing.assert_allclose(
    [*beta.theta, beta.precision],
    [
      -2.2950302752704594,
      0.04647558353358569,
      -0.03514238281889235,
      0.23168369117644602,
      27.332886211172678,
    ],
    rtol=1e-3,
  )


def test_the_predictions_reproduce_his() -> None:
  """CDF, log-density and quantiles must be his model's, case by case."""
  (xg, yg), (xm, ym), (xh, yh) = _sample()
  p = np.array([0.1, 0.5, 0.9])
  expected = {
    "seasonal": (
      [0.8115080990146898, 0.617328508169288, 0.8921380395842003],
      [-1.684368200807995, -1.3386536507771756, -2.0604066566977446],
      [9.26354659832766, 12.481196713923936, 7.626247252759723],
    ),
    "truncated": (
      [0.9022991242074034, 0.07907686653953636, 0.4914903858438489],
      [-2.0937417459746395, -2.258431742583606, -1.2285747167554903],
      [1.1058903360469856, 9.10840091303935, 4.414856146242398],
    ),
    "beta": (
      [0.4465419164518627, 0.4621145248377408, 0.46670839597421093],
      [1.5240145399424954, 1.452526077761629, 1.5880333451285154],
      [0.18834760033849127, 0.6463733492551534, 0.868111349854193],
    ),
  }
  models = {
    "seasonal": (SeasonalGaussianEMOS().fit(yg, x=xg), xg, yg),
    "truncated": (TruncatedNormalEMOS().fit(ym, x=xm), xm, ym),
    "beta": (BetaEMOS().fit(yh, x=xh), xh, yh),
  }
  for name, (model, x, y) in models.items():
    cdf, logpdf, icdf = expected[name]
    # The beta's fit moves with BLAS's thread count; see the test above.
    rtol = 1e-3 if name == "beta" else 1e-6
    np.testing.assert_allclose(model.cdf(y[:3], x=x[:3]), cdf, rtol=rtol)
    np.testing.assert_allclose(model.logpdf(y[:3], x=x[:3]), logpdf, rtol=rtol)
    np.testing.assert_allclose(model.icdf(p, x=x[:3]), icdf, rtol=rtol)


def test_each_family_is_the_scipy_distribution_it_names() -> None:
  """CDF, density, quantiles and CRPS must describe one distribution."""
  (xg, yg), (xm, ym), (xh, yh) = _sample()
  seasonal = SeasonalGaussianEMOS().fit(yg, x=xg)
  truncated = TruncatedNormalEMOS().fit(ym, x=xm)
  beta = BetaEMOS().fit(yh, x=xh)
  frozen = {
    "seasonal": (
      seasonal,
      xg,
      yg,
      stats.norm(loc=seasonal.mean(xg), scale=seasonal.sigma),
    ),
    "truncated": (
      truncated,
      xm,
      ym,
      stats.truncnorm(
        -truncated.location(xm) / truncated.sigma,
        np.inf,
        loc=truncated.location(xm),
        scale=truncated.sigma,
      ),
    ),
    "beta": (beta, xh, yh, stats.beta(*beta.shapes(xh))),
  }
  levels = (np.arange(19) + 0.5) / 19
  for model, x, y, dist in frozen.values():
    np.testing.assert_allclose(model.cdf(y, x=x), dist.cdf(y), rtol=1e-12)
    np.testing.assert_allclose(model.logpdf(y, x=x), dist.logpdf(y), rtol=1e-12)
    q = model.quantiles(levels, x=x)
    assert q.shape == (len(y), len(levels))
    np.testing.assert_allclose(q[:, 4], dist.ppf(levels[4]), rtol=1e-10)
    assert (np.diff(q, axis=1) > 0).all()
    np.testing.assert_allclose(
      model.cdf(q[:, 7], x=x), np.full(len(y), levels[7]), rtol=1e-8
    )


def test_the_closed_form_crps_matches_the_fitted_objective() -> None:
  """The objective minimized is the mean of the CRPS the model reports."""
  (xg, yg), (xm, ym), (xh, yh) = _sample()
  for model, x, y in (
    (SeasonalGaussianEMOS().fit(yg, x=xg), xg, yg),
    (TruncatedNormalEMOS().fit(ym, x=xm), xm, ym),
    (BetaEMOS().fit(yh, x=xh), xh, yh),
  ):
    assert float(model.crps(y, x=x).mean()) == pytest.approx(
      model.parameters["training_crps"], rel=1e-12
    )


def test_each_support_has_its_family() -> None:
  """His choice: beta for humidity, truncated normal for mixing ratio."""
  assert emos_for((0.0, 1.0)) is BetaEMOS
  assert emos_for((0.0, math.inf)) is TruncatedNormalEMOS
  assert emos_for((-math.inf, math.inf)) is SeasonalGaussianEMOS
  with pytest.raises(ValueError, match="support"):
    emos_for((1.0, 2.0))


@pytest.mark.parametrize(
  ("model", "x", "y", "match"),
  [
    (SeasonalGaussianEMOS(), np.ones((30, 1)), np.ones(30), "p in"),
    (SeasonalGaussianEMOS(), np.ones((30, 3)), np.ones(29), "shape"),
    (
      TruncatedNormalEMOS(),
      np.ones((30, 3)),
      np.r_[np.ones(29), 0.0],
      "positive",
    ),
    (BetaEMOS(), np.ones((30, 3)), np.r_[np.full(29, 0.5), 1.0], r"\(0, 1\)"),
    (BetaEMOS(), np.ones((3, 3)), np.full(3, 0.5), "at least"),
    (
      GaussianEMOS(),
      np.r_[np.ones(29), np.nan][:, None],
      np.ones(30),
      "finite",
    ),
  ],
)
def test_a_fit_on_data_it_cannot_model_is_refused(
  model: GaussianEMOS | TruncatedNormalEMOS | BetaEMOS,
  x: np.ndarray,
  y: np.ndarray,
  match: str,
) -> None:
  """A fit on a response off its family's support has no CRPS to minimize."""
  with pytest.raises(ValueError, match=match):
    model.fit(y, x=x)


def test_a_prediction_on_another_design_is_refused() -> None:
  """Covariates in another order or number would shift every coefficient."""
  (xg, yg), _, _ = _sample()
  model = SeasonalGaussianEMOS().fit(yg, x=xg)

  with pytest.raises(ValueError, match="3 columns"):
    model.cdf(yg, x=xg[:, :2])
