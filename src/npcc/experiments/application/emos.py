"""The EMOS baselines, each fitted by minimizing its closed-form mean CRPS.

Ensemble model output statistics: a parametric predictive distribution whose
location is linear in the covariates (the response's ensemble mean, then any
calendar terms) and whose scale is constant. One family per support:

- :class:`SeasonalGaussianEMOS` on the real line, the Gaussian of
  :class:`GaussianEMOS` with annual (and optionally diurnal) mean terms;
- :class:`TruncatedNormalEMOS` on ``(0, inf)``, a normal truncated at zero
  whose underlying location is linear;
- :class:`BetaEMOS` on ``(0, 1)``, with a logit-linear mean and constant
  precision.

Every model takes and returns NumPy arrays on its model scale: ``fit(y,
x=x)``, then ``cdf``, ``logpdf``, ``icdf`` and ``quantiles``, and ``crps``
in closed form. ``parameters`` records the fit.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self, cast

import numpy as np
from scipy.optimize import minimize
from scipy.special import betainc, betaln, expit, log_ndtr, logit, ndtr
from scipy.stats import beta, norm, truncnorm

_SQRT_PI = math.sqrt(math.pi)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
#: L-BFGS-B's options for the two Gaussian fits, which have a gradient.
_GAUSSIAN_OPTIONS = {"maxiter": 1000, "maxls": 100, "ftol": 1e-12, "gtol": 1e-8}
#: L-BFGS-B's options for the truncated-normal and beta fits.
_OPTIONS = {"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-7, "maxls": 50}
#: The floor on a Gaussian ``sigma``.
_SIGMA_FLOOR = 1e-6
#: The lowest ``mu / sigma`` a truncated normal may have: below it the mass
#: left above zero underflows.
_TRUNCATION_GUARD = -20.0


def gaussian_crps(
  y: np.ndarray, mu: np.ndarray | float, sigma: float
) -> np.ndarray:
  """The CRPS of ``Normal(mu, sigma**2)`` at ``y``, in closed form.

  Parameters
  ----------
  y : numpy.ndarray
      Observations.
  mu : numpy.ndarray or float
      Means, broadcast against ``y``.
  sigma : float
      The standard deviation.

  Returns
  -------
  numpy.ndarray
      The CRPS, in the units of ``y``.
  """
  z = (np.asarray(y) - np.asarray(mu)) / sigma
  phi = np.exp(-0.5 * z * z) / _SQRT_2PI
  return sigma * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * phi - 1.0 / _SQRT_PI)


def truncated_normal_crps(
  y: np.ndarray, mu: np.ndarray, sigma: float
) -> np.ndarray:
  """The CRPS of a normal truncated to ``(0, inf)``, in closed form.

  An observation below zero, outside the support, scores its distance to
  zero on top of the CRPS at zero.

  Parameters
  ----------
  y : numpy.ndarray
      Observations.
  mu : numpy.ndarray
      Locations of the underlying normal, broadcast against ``y``.
  sigma : float
      Scale of the underlying normal.

  Returns
  -------
  numpy.ndarray
      The CRPS, in the units of ``y``.
  """
  y, mu = np.asarray(y), np.asarray(mu)
  v = mu / sigma
  if np.any(v < _TRUNCATION_GUARD):
    raise ValueError(
      f"mu / sigma must be at least {_TRUNCATION_GUARD} for the truncated "
      f"normal's CRPS; got {float(v.min())}"
    )
  z = (np.maximum(y, 0.0) - mu) / sigma
  log_mass = log_ndtr(v)
  # F(y) = 1 - Phi(-z) / Phi(v); expm1 keeps it accurate near the bound.
  cdf = -np.expm1(np.minimum(log_ndtr(-z) - log_mass, 0.0))
  density = np.exp(-0.5 * z * z - 0.5 * math.log(2.0 * math.pi) - log_mass)
  pair = np.exp(log_ndtr(math.sqrt(2.0) * v) - 2.0 * log_mass) / _SQRT_PI
  return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * density - pair) + np.maximum(
    -y, 0.0
  )


def beta_crps(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """The CRPS of ``Beta(a, b)`` at ``y`` in ``[0, 1]``, in closed form.

  Parameters
  ----------
  y : numpy.ndarray
      Observations.
  a, b : numpy.ndarray
      The shapes, broadcast against ``y``.

  Returns
  -------
  numpy.ndarray
      The CRPS.
  """
  y = np.asarray(y)
  return (
    y * (2.0 * betainc(a, b, y) - 1.0)
    + a / (a + b) * (1.0 - 2.0 * betainc(a + 1.0, b, y))
    - 2.0 * np.exp(betaln(2.0 * a, 2.0 * b) - 2.0 * betaln(a, b)) / (a + b)
  )


def _check(
  y: np.ndarray, x: np.ndarray, widths: Sequence[int], owner: str
) -> tuple[np.ndarray, np.ndarray]:
  """Training arrays as float64, refusing a shape or value a fit cannot use."""
  y = np.asarray(y, dtype=np.float64)
  x = np.asarray(x, dtype=np.float64)
  if x.ndim != 2 or x.shape[1] not in widths:
    raise ValueError(
      f"{owner} must be fitted on x of shape (n, p) with p in {list(widths)}; "
      f"got shape {x.shape}"
    )
  if y.ndim != 1 or len(y) != len(x):
    raise ValueError(
      f"{owner} must be fitted on y of shape ({len(x)},); got {y.shape}"
    )
  if len(y) < x.shape[1] + 3:
    raise ValueError(
      f"{owner} must be fitted on at least {x.shape[1] + 3} cases; got {len(y)}"
    )
  if not (np.isfinite(x).all() and np.isfinite(y).all()):
    raise ValueError(f"{owner} must be fitted on finite x and y")
  return y, x


class EMOS(ABC):
  """A fitted parametric predictive distribution, on NumPy arrays.

  Subclasses implement :meth:`fit` and :meth:`_scipy`, which hands back the
  scipy distribution and its parameters for given covariates; everything
  else derives from those.

  Parameters
  ----------
  feature_names : sequence of str, optional
      The covariates' names, recorded in :attr:`parameters`.
  """

  def __init__(self, feature_names: Sequence[str] | None = None) -> None:
    self.feature_names = None if feature_names is None else list(feature_names)
    self.parameters: dict[str, object] = {}

  @abstractmethod
  def fit(self, y: np.ndarray, *, x: np.ndarray) -> Self:
    """Fit to responses ``y`` on covariates ``x``, by mean CRPS.

    Parameters
    ----------
    y : numpy.ndarray, shape (n,)
        Responses on the model scale.
    x : numpy.ndarray, shape (n, p)
        Covariates, the response's ensemble mean first.

    Returns
    -------
    EMOS
        ``self``, fitted.
    """

  @abstractmethod
  def _scipy(
    self, x: np.ndarray
  ) -> tuple[object, dict[str, np.ndarray | float]]:
    """The scipy distribution and its parameters, one per row of ``x``."""

  def _call(self, method: str, values: np.ndarray, x: np.ndarray) -> np.ndarray:
    distribution, params = self._scipy(np.asarray(x, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2:
      params = {
        k: np.asarray(v)[:, None] if np.ndim(v) else v
        for k, v in params.items()
      }
    return np.asarray(getattr(distribution, method)(values, **params))

  def cdf(self, y: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    """The distribution function at ``y``.

    Parameters
    ----------
    y : numpy.ndarray, shape (n,)
        Responses on the model scale.
    x : numpy.ndarray, shape (n, p)
        Covariates.

    Returns
    -------
    numpy.ndarray, shape (n,)
        ``F(y_i | x_i)``.
    """
    return self._call("cdf", y, x)

  def logpdf(self, y: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    """The log-density at ``y``.

    Parameters
    ----------
    y : numpy.ndarray, shape (n,)
        Responses on the model scale.
    x : numpy.ndarray, shape (n, p)
        Covariates.

    Returns
    -------
    numpy.ndarray, shape (n,)
        ``log f(y_i | x_i)``.
    """
    return self._call("logpdf", y, x)

  def icdf(self, p: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    """The quantile function at ``p``.

    Parameters
    ----------
    p : numpy.ndarray, shape (n,) or (n, L)
        Probabilities, one row per row of ``x``.
    x : numpy.ndarray, shape (n, p)
        Covariates.

    Returns
    -------
    numpy.ndarray
        ``F^-1(p | x_i)``, shaped like ``p``.
    """
    return self._call("ppf", p, x)

  def quantiles(self, levels: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    """Every row's quantiles at the same levels.

    Parameters
    ----------
    levels : numpy.ndarray, shape (L,)
        Probabilities in ``(0, 1)``.
    x : numpy.ndarray, shape (n, p)
        Covariates.

    Returns
    -------
    numpy.ndarray, shape (n, L)
        The quantiles on the model scale.
    """
    levels = np.asarray(levels, dtype=np.float64)
    grid = np.broadcast_to(levels, (len(x), len(levels)))
    return self.icdf(grid, x=x)

  @abstractmethod
  def crps(self, y: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    """The CRPS at ``y``, in closed form.

    Parameters
    ----------
    y : numpy.ndarray, shape (n,)
        Responses on the model scale.
    x : numpy.ndarray, shape (n, p)
        Covariates.

    Returns
    -------
    numpy.ndarray, shape (n,)
        The CRPS of each row's predictive distribution at its response.
    """


def _gaussian_objective(
  params: np.ndarray, design: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray]:
  """Mean Gaussian CRPS of a linear mean, and its gradient.

  ``params`` is the intercept, one coefficient per column of ``design``, then
  ``sigma``.
  """
  intercept, coefficients, sigma = params[0], params[1:-1], params[-1]
  mean = intercept + design @ coefficients
  z = (target - mean) / sigma
  phi = np.exp(-0.5 * z * z) / _SQRT_2PI
  dmean = 1.0 - 2.0 * ndtr(z)
  gradient = np.r_[
    dmean.mean(),
    design.T @ dmean / len(target),
    (2.0 * phi - 1.0 / _SQRT_PI).mean(),
  ]
  return float(gaussian_crps(target, mean, sigma).mean()), gradient


class GaussianEMOS(EMOS):
  """``Y | x ~ Normal(a + x @ b, sigma**2)``, on the ensemble mean alone.

  ``sigma`` is constant. The fit starts from least squares and minimizes the
  mean closed-form CRPS with its analytic gradient.
  """

  widths: tuple[int, ...] = (1,)
  center: float
  coefficients: np.ndarray
  sigma: float

  def fit(self, y: np.ndarray, *, x: np.ndarray) -> Self:
    """Fit by mean CRPS.

    Parameters
    ----------
    y : numpy.ndarray, shape (n,)
        Responses.
    x : numpy.ndarray, shape (n, p)
        The ensemble mean, then any calendar terms the class takes.

    Returns
    -------
    GaussianEMOS
        ``self``, fitted.
    """
    owner = type(self).__name__
    target, design = _check(y, x, self.widths, owner)
    start = np.linalg.lstsq(
      np.column_stack([np.ones(len(design)), design]), target, rcond=None
    )[0]
    residual = target - start[0] - design @ start[1:]
    initial = np.r_[start, max(float(np.sqrt(np.mean(residual**2))), 1e-3)]
    result = minimize(
      _gaussian_objective,
      initial,
      args=(design, target),
      jac=True,
      method="L-BFGS-B",
      bounds=[(None, None)] * (len(initial) - 1) + [(_SIGMA_FLOOR, None)],
      options=_GAUSSIAN_OPTIONS,
    )
    if not result.success or not np.isfinite(result.x).all():
      raise RuntimeError(
        f"{owner}'s CRPS minimization must converge; got {result.message}"
      )
    self.center = float(result.x[0])
    self.coefficients = np.asarray(result.x[1:-1], dtype=np.float64)
    self.sigma = float(result.x[-1])
    self.parameters = {
      "distribution": "normal",
      "feature_names": self.feature_names,
      "intercept": self.center,
      "coefficients": self.coefficients.tolist(),
      "sigma": self.sigma,
      "n_train": len(target),
      "training_crps": float(result.fun),
      "initial_crps": _gaussian_objective(initial, design, target)[0],
      "optimizer_iterations": int(result.nit),
      "fit_objective": "mean closed-form Gaussian CRPS",
    }
    return self

  def mean(self, x: np.ndarray) -> np.ndarray:
    """The predictive mean of each row.

    Parameters
    ----------
    x : numpy.ndarray, shape (n, p)
        Covariates, as fitted.

    Returns
    -------
    numpy.ndarray, shape (n,)
        ``a + x @ b``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(self.coefficients):
      raise ValueError(
        f"x must have {len(self.coefficients)} columns, as fitted; got shape "
        f"{x.shape}"
      )
    return self.center + x @ self.coefficients

  def _scipy(
    self, x: np.ndarray
  ) -> tuple[object, dict[str, np.ndarray | float]]:
    return norm, {"loc": self.mean(x), "scale": self.sigma}

  def crps(self, y: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    """The closed-form Gaussian CRPS at ``y``.

    Parameters
    ----------
    y : numpy.ndarray, shape (n,)
        Responses.
    x : numpy.ndarray, shape (n, p)
        Covariates.

    Returns
    -------
    numpy.ndarray, shape (n,)
        The CRPS.
    """
    return gaussian_crps(
      np.asarray(y, dtype=np.float64), self.mean(x), self.sigma
    )


class SeasonalGaussianEMOS(GaussianEMOS):
  """:class:`GaussianEMOS` with additive calendar terms in the mean.

  The covariates are the ensemble mean, the annual sine and cosine, and
  optionally the diurnal ones; there are no interactions, and ``sigma``
  stays constant.
  """

  widths = (3, 5)


class TruncatedNormalEMOS(EMOS):
  """A normal truncated at zero, with a linear underlying location.

  The underlying scale is constant; it is not the predictive standard
  deviation. Fitted from least squares by mean closed-form CRPS, on
  log-``sigma``, with ``mu / sigma`` kept above the truncation guard.
  """

  theta: np.ndarray
  sigma: float

  def fit(self, y: np.ndarray, *, x: np.ndarray) -> Self:
    """Fit by mean CRPS.

    Parameters
    ----------
    y : numpy.ndarray, shape (n,)
        Positive responses.
    x : numpy.ndarray, shape (n, p)
        The ensemble mean, then any calendar terms.

    Returns
    -------
    TruncatedNormalEMOS
        ``self``, fitted.
    """
    owner = type(self).__name__
    target, matrix = _check(y, x, (1, 3, 5), owner)
    if np.any(target <= 0.0):
      raise ValueError(
        f"{owner} must be fitted on positive responses; got "
        f"{int((target <= 0.0).sum())} at or below 0"
      )
    design = np.column_stack([np.ones(len(matrix)), matrix])
    coef = np.linalg.lstsq(design, target, rcond=None)[0]
    sigma = max(float(np.std(target - design @ coef)), 1e-3)
    # A feasible start, even where least squares predicts far below zero.
    coef[0] += max(0.0, float(-19.0 * sigma - (design @ coef).min()))
    initial = np.r_[coef, np.log(sigma)]

    def objective(theta: np.ndarray) -> float:
      mu = design @ theta[:-1]
      scale = np.exp(theta[-1])
      violation = np.maximum(_TRUNCATION_GUARD - mu / scale, 0.0)
      if np.any(violation):
        return 1e6 + float(np.mean(violation**2))
      return float(truncated_normal_crps(target, mu, scale).mean())

    result = minimize(
      objective,
      initial,
      method="L-BFGS-B",
      bounds=[(None, None)] * len(coef) + [(math.log(1e-6), math.log(1e3))],
      options=_OPTIONS,
    )
    if not result.success or not np.isfinite(result.fun) or result.fun >= 1e6:
      raise RuntimeError(
        f"{owner}'s CRPS minimization must converge; got {result.message}"
      )
    self.theta = np.asarray(result.x[:-1], dtype=np.float64)
    self.sigma = float(np.exp(result.x[-1]))
    self.parameters = {
      "distribution": "zero-truncated normal",
      "feature_names": self.feature_names,
      "intercept": float(self.theta[0]),
      "coefficients": self.theta[1:].tolist(),
      "sigma": self.sigma,
      "n_train": len(target),
      "training_crps": float(result.fun),
      "initial_crps": objective(initial),
      "optimizer_iterations": int(result.nit),
      "fit_objective": "mean closed-form zero-truncated Gaussian CRPS",
    }
    return self

  def location(self, x: np.ndarray) -> np.ndarray:
    """The underlying normal's location for each row.

    Parameters
    ----------
    x : numpy.ndarray, shape (n, p)
        Covariates, as fitted.

    Returns
    -------
    numpy.ndarray, shape (n,)
        The linear location.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(self.theta) - 1:
      raise ValueError(
        f"x must have {len(self.theta) - 1} columns, as fitted; got shape "
        f"{x.shape}"
      )
    return self.theta[0] + x @ self.theta[1:]

  def _scipy(
    self, x: np.ndarray
  ) -> tuple[object, dict[str, np.ndarray | float]]:
    mu = self.location(x)
    return truncnorm, {
      "a": -mu / self.sigma,
      "b": np.full_like(mu, np.inf),
      "loc": mu,
      "scale": self.sigma,
    }

  def crps(self, y: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    """The closed-form truncated-normal CRPS at ``y``.

    Parameters
    ----------
    y : numpy.ndarray, shape (n,)
        Responses.
    x : numpy.ndarray, shape (n, p)
        Covariates.

    Returns
    -------
    numpy.ndarray, shape (n,)
        The CRPS.
    """
    return truncated_normal_crps(
      np.asarray(y, dtype=np.float64), self.location(x), self.sigma
    )


class BetaEMOS(EMOS):
  """A beta on ``(0, 1)`` with a logit-linear mean and constant precision.

  With mean ``mu`` and precision ``phi``, the shapes are ``mu * phi`` and
  ``(1 - mu) * phi``, and the variance ``mu (1 - mu) / (1 + phi)``. Fitted
  from least squares on the logit by mean closed-form CRPS, on
  log-precision, within box bounds.
  """

  theta: np.ndarray
  precision: float

  def fit(self, y: np.ndarray, *, x: np.ndarray) -> Self:
    """Fit by mean CRPS.

    Parameters
    ----------
    y : numpy.ndarray, shape (n,)
        Responses strictly inside ``(0, 1)``.
    x : numpy.ndarray, shape (n, p)
        The ensemble mean, then any calendar terms.

    Returns
    -------
    BetaEMOS
        ``self``, fitted.
    """
    owner = type(self).__name__
    target, matrix = _check(y, x, (1, 3, 5), owner)
    if np.any((target <= 0.0) | (target >= 1.0)):
      raise ValueError(
        f"{owner} must be fitted on responses strictly inside (0, 1); got "
        f"{int(((target <= 0.0) | (target >= 1.0)).sum())} outside"
      )
    design = np.column_stack([np.ones(len(matrix)), matrix])
    coefficients = np.linalg.lstsq(design, logit(target), rcond=None)[0]
    initial = np.r_[coefficients, np.log(20.0)]

    def objective(theta: np.ndarray) -> float:
      mu = expit(design @ theta[:-1])
      a = np.maximum(mu * np.exp(theta[-1]), 1e-12)
      b = np.maximum((1.0 - mu) * np.exp(theta[-1]), 1e-12)
      return float(beta_crps(target, a, b).mean())

    result = minimize(
      objective,
      initial,
      method="L-BFGS-B",
      bounds=[(-30.0, 30.0)] * len(coefficients) + [(-5.0, 12.0)],
      options=_OPTIONS,
    )
    if not result.success or not np.isfinite(result.fun):
      raise RuntimeError(
        f"{owner}'s CRPS minimization must converge; got {result.message}"
      )
    self.theta = np.asarray(result.x[:-1], dtype=np.float64)
    self.precision = float(np.exp(result.x[-1]))
    self.parameters = {
      "distribution": "beta",
      "feature_names": self.feature_names,
      "intercept": float(self.theta[0]),
      "coefficients": self.theta[1:].tolist(),
      "precision": self.precision,
      "n_train": len(target),
      "training_crps": float(result.fun),
      "initial_crps": objective(initial),
      "optimizer_iterations": int(result.nit),
      "fit_objective": "mean closed-form beta CRPS",
    }
    return self

  def shapes(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The beta shapes ``(a, b)`` of each row.

    Parameters
    ----------
    x : numpy.ndarray, shape (n, p)
        Covariates, as fitted.

    Returns
    -------
    a, b : numpy.ndarray, shape (n,)
        The shapes, floored at ``1e-12``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(self.theta) - 1:
      raise ValueError(
        f"x must have {len(self.theta) - 1} columns, as fitted; got shape "
        f"{x.shape}"
      )
    mu = cast("np.ndarray", expit(self.theta[0] + x @ self.theta[1:]))
    return (
      np.maximum(mu * self.precision, 1e-12),
      np.maximum((1.0 - mu) * self.precision, 1e-12),
    )

  def _scipy(
    self, x: np.ndarray
  ) -> tuple[object, dict[str, np.ndarray | float]]:
    a, b = self.shapes(x)
    return beta, {"a": a, "b": b}

  def crps(self, y: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    """The closed-form beta CRPS at ``y``.

    Parameters
    ----------
    y : numpy.ndarray, shape (n,)
        Responses in ``[0, 1]``.
    x : numpy.ndarray, shape (n, p)
        Covariates.

    Returns
    -------
    numpy.ndarray, shape (n,)
        The CRPS.
    """
    a, b = self.shapes(x)
    return beta_crps(np.asarray(y, dtype=np.float64), a, b)


def emos_for(support: tuple[float, float]) -> type[EMOS]:
  """The EMOS family for a response's support.

  Parameters
  ----------
  support : tuple of float
      ``(0, 1)``, ``(0, inf)`` or the real line.

  Returns
  -------
  type
      :class:`BetaEMOS`, :class:`TruncatedNormalEMOS` or
      :class:`SeasonalGaussianEMOS`.
  """
  families: dict[tuple[float, float], type[EMOS]] = {
    (0.0, 1.0): BetaEMOS,
    (0.0, math.inf): TruncatedNormalEMOS,
    (-math.inf, math.inf): SeasonalGaussianEMOS,
  }
  if tuple(support) not in families:
    raise ValueError(
      f"support must be (0, 1), (0, inf) or the real line; got {support}"
    )
  return families[(float(support[0]), float(support[1]))]
