"""Fit one marginal model per station and response, and predict its cases.

A marginal model is seasonal EMOS, whose family follows the response's
support (:func:`~npcc.experiments.application.emos.emos_for`), or an npcc
backend: :func:`~npcc.core.registry.create_backend` with the estimator's
backend and ``backend_kwargs``, the response's transform, the default
quantile table and the default ``eps``. Either is fitted on the training
cases, on the response's model scale, and returns:

- the PIT of every training case (in sample) and test case, through ``cdf``;
- the test log-density on the original scale, which for a rescaled response
  loses ``log(scale)``;
- the test quantiles at the configured levels, through the public ``icdf``,
  on the original scale. An npcc margin predicts its quantile table once
  for the test rows and every level reuses it.

Every output must be finite, every PIT in ``[0, 1]`` and every case's
quantiles nondecreasing; anything else is refused rather than stored.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol, TypeVar

import numpy as np
import pandas as pd
import torch
from pyvinecopulib.core.extend import to_numpy

from npcc.core.controls import FitControlsRosenblattBicop
from npcc.core.margin import ConditionalMargin, QuantileTableConfig
from npcc.core.registry import create_backend
from npcc.experiments._study import derive_seed
from npcc.experiments.application.config import (
  ApplicationConfig,
  MarginalModel,
  ResponseSpec,
)
from npcc.experiments.application.data import StationData, feature_columns
from npcc.experiments.application.derived import (
  DEW_POINT,
  TEMPERATURE,
  nudge_saturated,
)
from npcc.experiments.application.emos import EMOS, emos_for
from npcc.experiments.application.scores import marginal_case_scores
from npcc.experiments.application.store import quantile_columns

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Predictor(Protocol):
  """A marginal model, on NumPy arrays on its model scale."""

  def fit(self, y: np.ndarray, *, x: np.ndarray) -> object:
    """Fit to responses ``y`` on covariates ``x``."""
    ...

  def cdf(self, y: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    """The distribution function at ``y``."""
    ...

  def logpdf(self, y: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    """The log-density at ``y``."""
    ...

  def quantiles(self, levels: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    """Each row's quantiles at ``levels``, shape ``(n, L)``."""
    ...


class NpccPredictor:
  """An npcc conditional margin, read through its public members.

  Parameters
  ----------
  margin : ConditionalMargin
      The margin, unfitted.
  """

  def __init__(self, margin: ConditionalMargin) -> None:
    self.margin = margin

  @staticmethod
  def _tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.float64)

  def fit(self, y: np.ndarray, *, x: np.ndarray) -> NpccPredictor:
    """Fit the margin.

    Parameters
    ----------
    y : numpy.ndarray, shape (n,)
        Responses on the model scale.
    x : numpy.ndarray, shape (n, p)
        Covariates.

    Returns
    -------
    NpccPredictor
        ``self``, fitted.
    """
    self.margin.fit(self._tensor(y), x=self._tensor(x))
    return self

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
    with torch.inference_mode():
      return to_numpy(self.margin.cdf(self._tensor(y), x=self._tensor(x)))

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
    with torch.inference_mode():
      return to_numpy(self.margin.logpdf(self._tensor(y), x=self._tensor(x)))

  def quantiles(self, levels: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    """Every row's quantiles at the same levels, through the public ``icdf``.

    The margin keeps the last quantile table it predicted, keyed on the
    rows, so the first level predicts it -- or reuses the one a ``cdf`` or
    ``logpdf`` on the same rows left -- and every later level reads it.

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
    rows = self._tensor(x)
    n = rows.shape[0]
    with torch.inference_mode():
      columns = [
        to_numpy(
          self.margin.icdf(
            torch.full((n,), float(p), dtype=torch.float64), x=rows
          )
        )
        for p in np.asarray(levels, dtype=np.float64)
      ]
    return np.column_stack(columns)


@dataclass(frozen=True)
class MarginFit:
  """One response's predictions from one marginal model at one station.

  Attributes
  ----------
  variable : str
      The response.
  pit_train, pit_test : numpy.ndarray, shape (n_train,) and (n_test,)
      The CDF at each case's observation.
  logpdf_test : numpy.ndarray, shape (n_test,)
      The log-density at each test observation, on the original scale.
  quantiles_test : numpy.ndarray, shape (n_test, L)
      The quantiles at the configured levels, on the original scale.
  timings : dict of str to float
      Seconds per stage.
  info : dict of str to object
      JSON-able: the seed, the estimator or the EMOS fit, the covariates.
  """

  variable: str
  pit_train: np.ndarray
  pit_test: np.ndarray
  logpdf_test: np.ndarray
  quantiles_test: np.ndarray
  timings: dict[str, float] = field(default_factory=dict)
  info: dict[str, object] = field(default_factory=dict)


def _synchronize() -> None:
  """Wait for queued CUDA work, where a CUDA context is already open."""
  if torch.cuda.is_available() and torch.cuda.is_initialized():
    torch.cuda.synchronize()


def _timed(timings: dict[str, float], stage: str, fn: Callable[[], T]) -> T:
  """Run ``fn``, recording its seconds under ``stage``."""
  _synchronize()
  start = perf_counter()
  result = fn()
  _synchronize()
  timings[stage] = perf_counter() - start
  return result


def build_predictor(
  model: MarginalModel,
  response: ResponseSpec,
  *,
  config: ApplicationConfig,
  feature_names: list[str],
  device: str | None,
) -> Predictor:
  """An unfitted marginal model for one response.

  Parameters
  ----------
  model : MarginalModel
      The model.
  response : ResponseSpec
      The response's scale.
  config : ApplicationConfig
      For the estimator and the inference batch size.
  feature_names : list of str
      The covariates it is fitted on, recorded by EMOS.
  device : str or None
      The npcc backend's device.

  Returns
  -------
  Predictor
      An EMOS model or an :class:`NpccPredictor`, with ``fit(y, x=x)``.
  """
  if model.backend is None:
    return emos_for(response.support)(feature_names=feature_names)
  spec = config.estimators_by_backend[model.backend]
  margin = create_backend(
    spec.backend,
    transform=response.transform,
    quantile_table_config=QuantileTableConfig(),
    eps=FitControlsRosenblattBicop().eps,
    device=device,
    batch_size=config.batch_size,
    backend_kwargs=dict(spec.backend_kwargs),
  )
  return NpccPredictor(margin)


def _check_outputs(fit: MarginFit, context: str) -> None:
  """Refuse predictions no score or copula can use."""
  for name in ("pit_train", "pit_test", "logpdf_test", "quantiles_test"):
    values = getattr(fit, name)
    if not np.isfinite(values).all():
      raise ValueError(
        f"{context}'s {name} must be finite; got "
        f"{int((~np.isfinite(values)).sum())} non-finite values"
      )
  for name in ("pit_train", "pit_test"):
    values = getattr(fit, name)
    if np.any((values < 0.0) | (values > 1.0)):
      raise ValueError(f"{context}'s {name} must lie in [0, 1]")
  decreasing = np.diff(fit.quantiles_test, axis=1) < 0.0
  if decreasing.any():
    raise ValueError(
      f"{context}'s quantiles must be nondecreasing along the levels; "
      f"{int(decreasing.any(axis=1).sum())} test cases are not"
    )


def fit_margin(
  data: StationData,
  model: MarginalModel,
  variable: str,
  *,
  config: ApplicationConfig,
  device: str | None,
) -> MarginFit:
  """Fit one marginal model to one response at one station, and predict.

  Its torch seed is derived from the base seed, the station, the model and
  the response, so a fit does not depend on what ran before it.

  Parameters
  ----------
  data : StationData
      The station's cases.
  model : MarginalModel
      The model.
  variable : str
      The response.
  config : ApplicationConfig
      The response's scale, the levels, the estimators.
  device : str or None
      The npcc backend's device; EMOS runs on the host.

  Returns
  -------
  MarginFit
      Its predictions, timings and details.
  """
  if variable not in config.responses:
    raise ValueError(
      f"responses must give the scale of every response in the data; got "
      f"none for {variable!r}"
    )
  response = config.responses[variable]
  j = data.column(variable)
  columns = feature_columns(data.feature_names, model.scope, variable)
  names = [data.feature_names[c] for c in columns]
  x_train, x_test = data.x_train[:, columns], data.x_test[:, columns]
  observed_train, observed_test = data.y_train[:, j], data.y_test[:, j]
  saturated: dict[str, int] = {}
  if variable == DEW_POINT and config.saturation_nudge > 0.0:
    t = data.column(TEMPERATURE)
    observed_train, saturated["saturated_train"] = nudge_saturated(
      data.y_train[:, t], observed_train, config.saturation_nudge
    )
    observed_test, saturated["saturated_test"] = nudge_saturated(
      data.y_test[:, t], observed_test, config.saturation_nudge
    )
  y_train = response.to_model(observed_train, variable)
  y_test = response.to_model(observed_test, variable)
  seed = derive_seed(
    config.base_seed, "margin", data.station, model.name, variable
  )
  torch.manual_seed(seed)

  timings: dict[str, float] = {}
  predictor = _timed(
    timings,
    "construction",
    lambda: build_predictor(
      model, response, config=config, feature_names=names, device=device
    ),
  )
  _timed(timings, "fit", lambda: predictor.fit(y_train, x=x_train))
  pit_train = _timed(
    timings, "pit_train", lambda: predictor.cdf(y_train, x=x_train)
  )
  pit_test = _timed(
    timings, "pit_test", lambda: predictor.cdf(y_test, x=x_test)
  )
  logpdf = _timed(
    timings, "logpdf_test", lambda: predictor.logpdf(y_test, x=x_test)
  )
  quantiles = _timed(
    timings,
    "quantiles_test",
    lambda: predictor.quantiles(config.levels, x=x_test),
  )
  info: dict[str, object] = {
    "seed": seed,
    "features": names,
    "response": {
      "transform": response.transform,
      "scale": response.scale,
      "boundary_nudge": response.boundary_nudge,
    },
    **saturated,
  }
  if isinstance(predictor, EMOS):
    info["emos"] = {"class": type(predictor).__name__, **predictor.parameters}
  elif model.backend is not None:
    spec = config.estimators_by_backend[model.backend]
    info["estimator"] = {
      "label": spec.label,
      "estimator_id": spec.estimator_id,
      "backend": spec.backend,
      "backend_kwargs": dict(spec.backend_kwargs),
      "device": device,
    }
  fit = MarginFit(
    variable=variable,
    pit_train=np.asarray(pit_train, dtype=np.float64),
    pit_test=np.asarray(pit_test, dtype=np.float64),
    logpdf_test=np.asarray(logpdf, dtype=np.float64) - response.log_scale,
    quantiles_test=np.asarray(quantiles, dtype=np.float64) * response.scale,
    timings=timings,
    info=info,
  )
  _check_outputs(fit, f"station {data.station}'s {model.name} {variable}")
  return fit


@dataclass(frozen=True)
class StationMargins:
  """One station's predictions from one marginal model, as stored.

  Attributes
  ----------
  predictions : pandas.DataFrame
      One row per ``(variable, split, row)``.
  scores : pandas.DataFrame
      One row per ``(variable, row)`` of the test split.
  timings : pandas.DataFrame
      One row per ``(variable, stage)``.
  info : dict of str to object
      The levels and each response's details.
  """

  predictions: pd.DataFrame
  scores: pd.DataFrame
  timings: pd.DataFrame
  info: dict[str, object]


def fit_station(
  data: StationData,
  model: MarginalModel,
  *,
  config: ApplicationConfig,
  device: str | None,
) -> StationMargins:
  """Fit one marginal model to every response of a station.

  Parameters
  ----------
  data : StationData
      The station's cases.
  model : MarginalModel
      The model.
  config : ApplicationConfig
      The configuration.
  device : str or None
      The npcc backend's device.

  Returns
  -------
  StationMargins
      The tables :func:`~npcc.experiments.application.store.write_margin`
      stores.
  """
  levels = config.levels
  q_columns = quantile_columns(len(levels))
  predictions, scores, timings = [], [], []
  details: dict[str, object] = {}
  modeled = data.variables if config.is_default_design else config.modeled
  for variable in modeled:
    logger.info(
      "station %d: fitting %s to %s", data.station, model.name, variable
    )
    fit = fit_margin(data, model, variable, config=config, device=device)
    observed = data.y_test[:, data.column(variable)]
    common = {
      "station": data.station,
      "model": model.name,
      "variable": variable,
    }
    predictions.append(
      pd.DataFrame(
        {
          **common,
          "split": "train",
          "row": data.row_train,
          "time": data.time_train,
          "pit": fit.pit_train,
          "logpdf": np.nan,
          **dict.fromkeys(q_columns, np.nan),
        }
      )
    )
    predictions.append(
      pd.DataFrame(
        {
          **common,
          "split": "test",
          "row": data.row_test,
          "time": data.time_test,
          "pit": fit.pit_test,
          "logpdf": fit.logpdf_test,
          **dict(zip(q_columns, fit.quantiles_test.T, strict=True)),
        }
      )
    )
    scores.append(
      pd.DataFrame(
        {
          **common,
          "row": data.row_test,
          "time": data.time_test,
          "observed": observed,
          **marginal_case_scores(
            observed,
            logpdf=fit.logpdf_test,
            pit=fit.pit_test,
            quantiles=fit.quantiles_test,
            levels=levels,
          ),
        }
      )
    )
    timings.extend(
      {**common, "stage": stage, "seconds": seconds}
      for stage, seconds in fit.timings.items()
    )
    details[variable] = fit.info
  return StationMargins(
    predictions=pd.concat(predictions, ignore_index=True),
    scores=pd.concat(scores, ignore_index=True),
    timings=pd.DataFrame(timings),
    info={
      "levels": levels.tolist(),
      "variables": details,
      **(
        {}
        if config.is_default_design
        else {"modeled": list(config.modeled), "derived": list(config.derived)}
      ),
    },
  )
