"""What a copula arm reads: one station's fitted margins, all variables."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MarginalSet:
  """One station's saved predictions from one marginal model.

  The copula stage reads nothing else: every array is aligned on the
  station's training or test cases, in time order, and on ``variables``.
  Responses are on their original scale (relative humidity in percent).

  Attributes
  ----------
  station : int
      The station id.
  model : str
      The marginal model's id, such as ``"emos-seasonal"`` or
      ``"tabpfn-quantiles-all"``.
  variables : tuple of str
      The ``d`` responses, as named in ``y.nc``.
  feature_names : tuple of str
      The ``p`` covariates, as named in ``x.nc``.
  x_train, x_test : numpy.ndarray, shape (n_train, p) and (n_test, p)
      Every covariate; a copula arm selects its own columns by scope.
  y_train, y_test : numpy.ndarray, shape (n_train, d) and (n_test, d)
      The observed responses.
  time_train, time_test : numpy.ndarray of datetime64[ns]
      The valid time of each case.
  pit_train, pit_test : numpy.ndarray, shape (n_train, d) and (n_test, d)
      The marginal CDF at the observation: in-sample on the training cases,
      held out on the test cases. Not clipped.
  logpdf_test : numpy.ndarray, shape (n_test, d)
      The marginal log-density at each test observation, on the response's
      original scale.
  levels : numpy.ndarray, shape (L,)
      The probability levels of ``quantiles_test``, increasing in (0, 1).
  quantiles_test : numpy.ndarray, shape (n_test, L, d)
      The marginal quantiles at ``levels`` for each test case, on the
      original scale, nondecreasing along the level axis.
  derived : tuple of str
      The ``k`` responses a copula's draws derive from the modeled ones
      (:mod:`~npcc.experiments.application.derived`); empty when every
      response is modeled.
  y_derived_train, y_derived_test : numpy.ndarray, shape (n_train, k) and (n_test, k)
      Their observations, which the joint forecast is scored against.
  """

  station: int
  model: str
  variables: tuple[str, ...]
  feature_names: tuple[str, ...]
  x_train: np.ndarray
  x_test: np.ndarray
  y_train: np.ndarray
  y_test: np.ndarray
  time_train: np.ndarray
  time_test: np.ndarray
  pit_train: np.ndarray
  pit_test: np.ndarray
  logpdf_test: np.ndarray
  levels: np.ndarray
  quantiles_test: np.ndarray
  derived: tuple[str, ...] = ()
  y_derived_train: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
  y_derived_test: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

  def __post_init__(self) -> None:
    """Refuse arrays that do not line up."""
    d, p = len(self.variables), len(self.feature_names)
    n_train, n_test = len(self.y_train), len(self.y_test)
    shapes = {
      "x_train": (self.x_train.shape, (n_train, p)),
      "x_test": (self.x_test.shape, (n_test, p)),
      "y_train": (self.y_train.shape, (n_train, d)),
      "y_test": (self.y_test.shape, (n_test, d)),
      "time_train": (self.time_train.shape, (n_train,)),
      "time_test": (self.time_test.shape, (n_test,)),
      "pit_train": (self.pit_train.shape, (n_train, d)),
      "pit_test": (self.pit_test.shape, (n_test, d)),
      "logpdf_test": (self.logpdf_test.shape, (n_test, d)),
      "quantiles_test": (
        self.quantiles_test.shape,
        (n_test, len(self.levels), d),
      ),
    }
    if self.derived:
      k = len(self.derived)
      shapes["y_derived_train"] = (self.y_derived_train.shape, (n_train, k))
      shapes["y_derived_test"] = (self.y_derived_test.shape, (n_test, k))
    for name, (got, want) in shapes.items():
      if got != want:
        raise ValueError(
          f"MarginalSet.{name} must have shape {want}; got {got}"
        )
