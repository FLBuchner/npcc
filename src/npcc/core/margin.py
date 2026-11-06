"""Conditional margins read off predicted quantile tables.

A conditional margin represents the distribution of a univariate response
``Y`` given an optional feature matrix ``X``. It implements pyvinecopulib's
:class:`MarginBase` contract and adds the Cartesian-grid evaluation methods
NPCC pair copulas use.

Every backend says what it knows about ``Y | X`` the same way: as the
conditional quantiles

    Q(alpha_k | x),   alpha_k = Phi(t_k),   t_k equally spaced on [-L, L],

of the modeled response, and the margin -- density, distribution function and
quantile function -- is read off that table. How it is read, and how its tails
are closed, is described in :mod:`npcc.core._quantile_table`.

Concrete backends implement two hooks: ``_fit_model``, to train on the
transformed targets, and ``_predict_quantiles``, to predict the table for a
chunk of conditioning rows. Table prediction is chunked over conditioning rows,
and grid evaluation predicts at most once per conditioning row.

All public numerical inputs and outputs are torch tensors.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NoReturn, Self, TypedDict

import numpy as np
import torch
from pyvinecopulib.core import MarginBase
from pyvinecopulib.core.extend import prepare_covariates

from npcc.core._placement import TensorPlacement, default_batch_size
from npcc.core._quantile_table import QuantileTable, normal_scores
from npcc.core._transform import Transform, normal_cdf, resolve_transform


@dataclass(frozen=True)
class QuantileTableConfig:
  """Configuration for the quantile-table reconstruction.

  Attributes
  ----------
  n_quantiles
    Number of probability levels. They are ``Phi(t)`` for ``t`` equally spaced
    on ``[-normal_limit, normal_limit]``, so they crowd into the tails.
  normal_limit
    Outermost normal score of the levels; ``Phi(-normal_limit)`` of mass lies
    beyond the table on each side. The default is pyvinecopulib's normal-grid
    limit.
  smoothing
    Bandwidth, in normal-score units at the median, of the smoother the
    predicted table is fitted with (see :mod:`npcc.core._quantile_table`);
    ``0`` turns smoothing off. The default suits every backend: it removes
    most of the level-to-level noise of a per-level model and barely changes
    a joint model's table.
  min_slope
    Floor on every quantile slope, as a fraction of the row's mean slope. It
    bounds the density a crossed or tied pair of quantiles can produce.
  """

  n_quantiles: int = 21
  normal_limit: float = 3.25
  smoothing: float = 0.65
  min_slope: float = 0.2

  def __post_init__(self) -> None:
    """Validate the quantile-table configuration."""
    if self.n_quantiles < 5:
      raise ValueError(
        f"n_quantiles must be at least 5; got {self.n_quantiles}"
      )

    # Beyond about 8, 1 - Phi(normal_limit) rounds to 0 in float64 and the
    # outermost level becomes 1.
    if not 0.0 < self.normal_limit <= 6.0:
      raise ValueError(
        f"normal_limit must lie in (0, 6]; got {self.normal_limit}"
      )

    if not self.smoothing >= 0.0:
      raise ValueError(f"smoothing must be nonnegative; got {self.smoothing}")

    if not 0.0 <= self.min_slope < 1.0:
      raise ValueError(f"min_slope must lie in [0, 1); got {self.min_slope}")

  def normal_scores(
    self,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
  ) -> torch.Tensor:
    """Return the normal scores ``t`` of the probability levels.

    Parameters
    ----------
    device : str or torch.device, optional
        Device of the result.
    dtype : torch.dtype, default=torch.float64
        Dtype of the result.

    Returns
    -------
    torch.Tensor
        Equally spaced scores on ``[-normal_limit, normal_limit]``, shape
        ``(n_quantiles,)``.
    """
    return normal_scores(
      self.n_quantiles, self.normal_limit, device=device, dtype=dtype
    )

  def alphas(
    self,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
  ) -> torch.Tensor:
    """Return the probability levels a backend predicts quantiles at.

    Parameters
    ----------
    device : str or torch.device, optional
        Device of the result.
    dtype : torch.dtype, default=torch.float64
        Dtype of the result.

    Returns
    -------
    torch.Tensor
        ``Phi`` of :meth:`normal_scores`, ascending, shape ``(n_quantiles,)``.
    """
    return normal_cdf(self.normal_scores(device=device, dtype=dtype))


class MarginOptions(TypedDict, total=False):
  """The construction arguments a backend forwards to its base class.

  A backend whose model settings are named parameters takes these as
  ``**margin: Unpack[MarginOptions]`` and passes them on untouched, so the
  defaults are :class:`ConditionalMargin`'s alone. One that forwards its model
  settings as ``**kwargs`` cannot take both, and names these instead.
  """

  transform: Transform
  quantile_table_config: QuantileTableConfig | None
  eps: float
  device: str | torch.device | None
  batch_size: int | None


class ConditionalMargin(TensorPlacement, MarginBase[torch.Tensor], ABC):
  """Abstract backend-powered conditional continuous margin.

  Concrete backends implement :meth:`_fit_model` and
  :meth:`_predict_quantiles`; everything else is read off the predicted table.

  Parameters
  ----------
  transform
    Transformation applied to the response before fitting. ``"identity"``
    is appropriate for original-scale margins on the real line, and ``"log"``
    for nonnegative ones. ``"logit"`` and ``"probit"`` map values from
    ``(0, 1)`` to the real line and are appropriate for copula-scale data. The
    transform fixes the margin's :attr:`support`. An unknown name is refused
    here, at construction.
  quantile_table_config
    Quantile-table reconstruction configuration. If omitted, the defaults of
    :class:`QuantileTableConfig`.
  eps
    Distance used when clipping values away from the boundaries of ``(0, 1)``
    before applying the logit or probit transform.
  device
    Device used for fitting, inference and sampling. If omitted, CUDA is
    selected when available and CPU otherwise.
  batch_size
      Default inference chunk size. If omitted, 400 is used on CPU and 2000 on
      CUDA.
  """

  supports_covariates: bool = True
  supports_weights: bool = False
  supports_controls: bool = False

  transform: Transform
  quantile_table_config: QuantileTableConfig
  eps: float
  batch_size: int
  _device: torch.device
  _fitted: bool

  def __init__(
    self,
    *,
    transform: Transform = "logit",
    quantile_table_config: QuantileTableConfig | None = None,
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
  ) -> None:
    if not 0.0 < eps < 0.5:
      raise ValueError("eps must lie strictly between 0 and 0.5.")

    self._response_transform = resolve_transform(transform)
    self.transform = transform
    self.quantile_table_config = quantile_table_config or QuantileTableConfig()
    self.eps = eps
    self._set_placement(device)

    if batch_size is not None and batch_size <= 0:
      raise ValueError("batch_size must be positive.")
    self.batch_size = batch_size or default_batch_size(self._device)

    self._fitted = False

  @property
  def is_fitted(self) -> bool:
    """Whether this margin has been fitted."""
    return self._fitted

  @property
  def support(self) -> tuple[float, float]:
    """Closed bounds of the response's support, as ``(lo, hi)``.

    The transform's: ``(0, 1)`` for logit and probit, ``(0, inf)`` for log and
    the real line for identity.

    Returns
    -------
    tuple of float
        The bounds, infinite where the support is unbounded.
    """
    return self._response_transform.support

  @property
  def family_name(self) -> str:
    """Name shown in margin summaries."""
    return type(self).__name__

  @property
  def n_parameters(self) -> float:
    """Zero -- these backends estimate no free parameters in this sense.

    A distributional-regression backend has no well-defined count of freely
    estimated parameters: a gradient-boosted ensemble's is a function of its
    tree structure, and a pretrained foundation model does not estimate any at
    fit time at all. ``MarginBase`` derives its information criteria from this
    number, so rather than let it silently stand for one, :meth:`aic`,
    :meth:`bic` and :meth:`aicc` are refused below.

    Returns
    -------
    float
        Always ``0.0``. Read as "not a parameter count", not as "unpenalized".
    """
    return 0.0

  def _refuse_criterion(self, name: str) -> NoReturn:
    """Raise, naming the criterion and why this margin has none."""
    raise NotImplementedError(
      f"{type(self).__name__} has no well-defined free-parameter count, so "
      f"{name} would penalize the fit by zero and rank every backend by "
      "log-likelihood alone. Compare backends on held-out log-likelihood "
      "instead."
    )

  def aic(self, y: torch.Tensor | None = None, /) -> float:
    """Refuse: see :attr:`n_parameters`."""
    del y
    self._refuse_criterion("aic")

  def bic(self, y: torch.Tensor | None = None, /) -> float:
    """Refuse: see :attr:`n_parameters`."""
    del y
    self._refuse_criterion("bic")

  def aicc(self, y: torch.Tensor | None = None, /) -> float:
    """Refuse: see :attr:`n_parameters`."""
    del y
    self._refuse_criterion("aicc")

  def _resolve_batch_size(self, batch_size: int | None) -> int:
    """Resolve and validate a method-level batch-size override."""
    effective = self.batch_size if batch_size is None else batch_size

    if effective <= 0:
      raise ValueError("batch_size must be positive.")

    return effective

  def _check_fitted(self) -> None:
    """Raise if the margin has not been fitted."""
    if not self._fitted:
      raise RuntimeError("The model is not fitted.")

  def _conditioning(
    self,
    values: torch.Tensor,
    x: torch.Tensor | None,
  ) -> torch.Tensor:
    """Place conditioning features and check they are row-aligned.

    An absent ``x`` becomes a single constant column rather than a zero-width
    one: the inner regressors are third-party estimators that require at least
    one feature, so an unconditional margin is fitted as a conditional one on a
    covariate that carries no information.

    The layout is upstream's:
    :func:`pyvinecopulib.core.extend.prepare_covariates` performs the check
    and ``(n,)`` is refused, so every entry point on this margin agrees with
    the ones it inherits.
    """
    n = values.shape[0]

    if x is None:
      return torch.zeros(
        (n, 1),
        dtype=torch.float64,
        device=self._device,
      )

    x_t = self._prep(x)

    prepare_covariates(self, x_t, n)

    return x_t

  def _grid_covariates(self, x: torch.Tensor) -> torch.Tensor:
    """Place the conditioning rows of a Cartesian-grid query, and check them.

    The grid methods take ``(n, p)`` like every other entry point, so this
    refuses ``(n,)`` for the reason
    :func:`pyvinecopulib.core.extend.prepare_covariates` does -- it says
    nothing about which axis is which. These methods accept ``p > 1``, so the
    check here is the general one rather than anything single-covariate.

    Placement is the other half. These methods reach the backend without
    passing through ``_prep``, so before this a NumPy ``x`` raised from inside
    the alpha grid rather than at the boundary, and a float32 one drove that
    grid at float32.

    Parameters
    ----------
    x : torch.Tensor
        Conditioning rows, shape ``(n, p)``.

    Returns
    -------
    torch.Tensor
        The same rows, ``float64`` on this margin's device.

    Raises
    ------
    ValueError
        If ``x`` is not two-dimensional.
    """
    x_t = self._prep(x)

    if x_t.ndim != 2:
      raise ValueError(
        f"x must have shape (n, p), with one row per observation; "
        f"got {tuple(x_t.shape)}"
      )

    return x_t

  def _transform_y(self, y: torch.Tensor) -> torch.Tensor:
    """Transform responses to the backend's modeled scale."""
    g = self._response_transform
    return g.forward(g.clip(y, self.eps))

  def fit(
    self,
    y: torch.Tensor,
    /,
    controls: object | None = None,
    *,
    var_type: str | None = None,
    support: tuple[float | None, float | None] | None = None,
    x: torch.Tensor | None = None,
  ) -> Self:
    """Fit the backend to responses and optional conditioning features.

    ``controls`` is not accepted -- this margin is configured entirely at
    construction, which is what :attr:`supports_controls` ``False`` declares.
    ``var_type`` and ``support`` are *declarations* a caller makes about the
    variable rather than configuration, and this margin honors only the
    continuous case :attr:`var_type` reports, on the support :attr:`support`
    reports -- which its transform fixes. Each is refused rather than dropped,
    so a caller who declares an atom or another bound is told instead of being
    handed a fit that quietly ignored it.

    Observation weights no longer reach a margin through this signature; they
    ride in ``controls``, and upstream refuses them against
    :attr:`supports_weights` before the call.
    """
    if controls is not None:
      raise ValueError(
        f"{type(self).__name__} declares `supports_controls = False`, so it "
        "is configured at construction and takes no `controls`; pass the "
        "backend settings to the constructor instead."
      )

    if var_type is not None and var_type != "c":
      raise ValueError(
        f"{type(self).__name__} must model continuous variables; got "
        f"var_type={var_type!r}. Its backends estimate a conditional density, "
        "not an atom's probability; use a Kde1d margin for a discrete or "
        "zero-inflated variable."
      )

    if support is not None and not self._honors(support):
      raise ValueError(
        f"{type(self).__name__} with transform={self.transform!r} must model "
        f"variables on {self.support}; got support={support!r}. The transform "
        "fixes the support: 'logit' or 'probit' for (0, 1), 'log' for "
        "(0, inf), 'identity' for the real line. Use a Kde1d margin for any "
        "other bound."
      )

    y_t = self._prep(y).reshape(-1)
    self._response_transform.check_fit(y_t, type(self).__name__)

    x_t = self._conditioning(y_t, x=x)
    z_t = self._transform_y(y_t)

    self._fitted = False
    self._fit_model(x_t, z_t)
    self._nobs = y_t.shape[0]
    self._fitted = True

    return self

  def _honors(self, support: tuple[float | None, float | None]) -> bool:
    """Whether a declared support is the one this margin's transform fixes."""
    declared = (
      -math.inf if support[0] is None else float(support[0]),
      math.inf if support[1] is None else float(support[1]),
    )
    return declared == self.support

  @abstractmethod
  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    """Train the backend using tensors on the configured device.

    A third-party backend that requires host-side inputs is responsible for
    detaching and moving these tensors to CPU within its own adapter.
    """

  @abstractmethod
  def _predict_quantiles(
    self,
    x: torch.Tensor,
    alphas: torch.Tensor,
  ) -> torch.Tensor:
    """Predict conditional quantiles for one chunk of conditioning rows.

    Parameters
    ----------
    x
      Conditioning matrix with shape ``(n_chunk, n_features)`` on the
      configured device.
    alphas
      Ascending probability levels with shape ``(n_alphas,)``.

    Returns
    -------
    torch.Tensor
      Predicted quantiles with shape ``(n_chunk, n_alphas)`` on the configured
      device. Rows need not be monotone because the table sorts them.
    """

  @staticmethod
  def _quantile_rows(
    predicted: object,
    x: torch.Tensor,
    *,
    levels_first: bool,
  ) -> torch.Tensor:
    """Read a third-party model's quantiles as one row per conditioning row.

    The orientation is the library's to state, not this method's to infer: a
    chunk with as many rows as there are levels gives a square table, whose
    shape cannot say which axis is which. A one-dimensional result is a single
    row whichever way the library orders its axes.

    Parameters
    ----------
    predicted : array_like
        The model's output, ``(n_chunk, n_alphas)``, or ``(n_alphas,
        n_chunk)`` -- which includes a list of one array per level -- when
        ``levels_first``.
    x : torch.Tensor
        The chunk of conditioning rows, whose dtype and device the result
        takes.
    levels_first : bool
        Whether the library puts the levels on the leading axis.

    Returns
    -------
    torch.Tensor
        The quantiles, ``(n_chunk, n_alphas)`` when ``predicted`` had the
        shape the library documents.
    """
    table = torch.as_tensor(
      np.asarray(predicted), dtype=x.dtype, device=x.device
    )
    if table.ndim == 1:
      return table.reshape(1, -1)
    return table.T if levels_first and table.ndim == 2 else table

  def _predict_quantile_table(
    self,
    x: torch.Tensor,
    *,
    batch_size: int | None = None,
  ) -> QuantileTable:
    """Predict the quantile table of every conditioning row.

    Each distinct row reaches the backend once. A pair copula's grid repeats
    its conditioning rows -- one per point of the other axis -- and a backend
    predicts every row independently of the others in its chunk, so a repeat
    would only recompute the same quantiles.
    """
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    cfg = self.quantile_table_config
    alphas = cfg.alphas(device=x.device, dtype=x.dtype)
    n_alphas = alphas.shape[0]
    distinct, inverse = torch.unique(x, dim=0, return_inverse=True)
    n_obs = distinct.shape[0]

    parts: list[torch.Tensor] = []

    for start in range(0, n_obs, effective_batch_size):
      end = min(start + effective_batch_size, n_obs)

      quantiles = self._predict_quantiles(distinct[start:end], alphas)

      expected_shape = (end - start, n_alphas)

      if quantiles.shape != expected_shape:
        raise RuntimeError(
          "_predict_quantiles must return shape (n_chunk, n_alphas); "
          f"got {tuple(quantiles.shape)}, expected {expected_shape}."
        )

      parts.append(quantiles)

    table = (
      torch.cat(parts, dim=0)[inverse]
      if parts
      else torch.empty((0, n_alphas), dtype=torch.float64, device=self._device)
    )

    return QuantileTable(
      table,
      limit=cfg.normal_limit,
      transform=self._response_transform,
      eps=self.eps,
      smoothing=cfg.smoothing,
      min_slope=cfg.min_slope,
    )

  def _rows(
    self, values: torch.Tensor, x: torch.Tensor | None, name: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Place and flatten pointwise values and the rows they pair with.

    The values are placed as well as the rows: the table is predicted on this
    margin's device, so a response left on another one would meet it there.
    """
    values_t = self._prep(values).reshape(-1)
    x_t = self._conditioning(values_t, x)

    if values_t.shape[0] != x_t.shape[0]:
      raise ValueError(f"x and {name} have incompatible lengths.")

    return values_t, x_t

  def pdf(
    self,
    y: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate ``f(y_i | x_i)`` for every observation."""
    y_t, x_t = self._rows(y, x, "y")
    table = self._predict_quantile_table(x_t, batch_size=batch_size)
    return table.pdf(y_t.unsqueeze(1)).squeeze(1)

  def cdf(
    self,
    y: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate ``F(y_i | x_i)`` for every observation."""
    y_t, x_t = self._rows(y, x, "y")
    table = self._predict_quantile_table(x_t, batch_size=batch_size)
    return table.cdf(y_t.unsqueeze(1)).squeeze(1)

  def icdf(
    self,
    p: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate ``F^-1(p_i | x_i)`` on the response scale."""
    p_t, x_t = self._rows(p, x, "p")

    if torch.any((p_t <= 0.0) | (p_t >= 1.0)):
      raise ValueError("p must lie strictly inside (0, 1).")

    table = self._predict_quantile_table(x_t, batch_size=batch_size)
    return table.icdf(p_t.unsqueeze(1)).squeeze(1)

  def _on_grid(
    self,
    y_grid: torch.Tensor,
    x: torch.Tensor,
    batch_size: int | None,
    *,
    density: bool,
  ) -> torch.Tensor:
    """Evaluate on the Cartesian product, one table per chunk of rows."""
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    y_grid_t = self._prep(y_grid).reshape(-1)

    if y_grid_t.numel() == 0:
      raise ValueError("y_grid must contain at least one value.")

    x_t = self._grid_covariates(x)
    n_y = y_grid_t.shape[0]

    chunks: list[torch.Tensor] = []

    for start in range(0, x_t.shape[0], effective_batch_size):
      end = min(start + effective_batch_size, x_t.shape[0])
      table = self._predict_quantile_table(
        x_t[start:end], batch_size=effective_batch_size
      )
      values = y_grid_t.unsqueeze(0).expand(end - start, n_y)
      chunks.append(table.pdf(values) if density else table.cdf(values))

    return (
      torch.cat(chunks, dim=0)
      if chunks
      else torch.empty((0, n_y), dtype=x_t.dtype, device=x_t.device)
    )

  def pdf_grid(
    self,
    y_grid: torch.Tensor,
    /,
    *,
    x: torch.Tensor,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate density on the Cartesian product of ``x`` and ``y_grid``."""
    return self._on_grid(y_grid, x, batch_size, density=True)

  def cdf_grid(
    self,
    y_grid: torch.Tensor,
    /,
    *,
    x: torch.Tensor,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate CDFs on the Cartesian product of ``x`` and ``y_grid``."""
    return self._on_grid(y_grid, x, batch_size, density=False)

  def _sample_uniform(
    self,
    n: int,
    seeds: list[int],
  ) -> torch.Tensor:
    """Draw base uniforms on the configured device."""
    return self._draw_uniform((n,), seeds=seeds)
