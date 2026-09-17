"""Backend-neutral conditional margins.

A concrete conditional margin represents the distribution of a univariate
response ``Y`` given an optional feature matrix ``X``. It implements
pyvinecopulib's :class:`MarginBase` contract and adds efficient Cartesian-grid
evaluation methods used by NPCC pair copulas.

Concrete backends implement:

- ``_fit_model`` to train on transformed targets;
- ``pdf`` and ``cdf`` as required by ``MarginBase``;
- ``icdf`` when a native or table-based inverse is available;
- ``pdf_grid`` and ``cdf_grid`` using at most one prediction per conditioning
  row.

All public numerical inputs and outputs are torch tensors.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal, NoReturn, Self

import torch
from pyvinecopulib.core import MarginBase
from pyvinecopulib.core.extend import prepare_covariates

from npcc.core._placement import TensorPlacement
from npcc.core._trim import logit


class ConditionalMargin(TensorPlacement, MarginBase[torch.Tensor], ABC):
  """Abstract backend-powered conditional continuous margin.

  Parameters
  ----------
  transform
    Transformation applied to the response before fitting. ``"identity"``
    is appropriate for original-scale margins. ``"logit"`` and ``"probit"``
    map values from ``(0, 1)`` to the real line and are appropriate for
    copula-scale data.
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

  transform: Literal["identity", "logit", "probit"]
  eps: float
  batch_size: int
  _device: torch.device
  _fitted: bool

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
  ) -> None:
    if not 0.0 < eps < 0.5:
      raise ValueError("eps must lie strictly between 0 and 0.5.")

    self.transform = transform
    self.eps = eps
    self._set_placement(device)

    if batch_size is None:
      self.batch_size = 2000 if self._device.type == "cuda" else 400
    else:
      if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
      self.batch_size = batch_size

    self._fitted = False

  @property
  def is_fitted(self) -> bool:
    """Whether this margin has been fitted."""
    return self._fitted

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
    if self.transform == "identity":
      return y

    if self.transform == "logit":
      y_clip = torch.clamp(y, self.eps, 1.0 - self.eps)
      return logit(y_clip)

    if self.transform == "probit":
      y_clip = torch.clamp(y, self.eps, 1.0 - self.eps)
      return math.sqrt(2.0) * torch.erfinv(2.0 * y_clip - 1.0)

    raise ValueError(f"Unknown transform: {self.transform}")

  def _jacobian_inverse(self, y: torch.Tensor) -> torch.Tensor:
    """Return the inverse-transform Jacobian evaluation on the response scale."""
    if self.transform == "identity":
      return torch.ones_like(y)

    if self.transform == "logit":
      y_clip = torch.clamp(y, self.eps, 1.0 - self.eps)
      return 1.0 / (y_clip * (1.0 - y_clip))

    if self.transform == "probit":
      y_clip = torch.clamp(y, self.eps, 1.0 - self.eps)
      z = math.sqrt(2.0) * torch.erfinv(2.0 * y_clip - 1.0)
      phi_z = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
      return 1.0 / phi_z

    raise ValueError(f"Unknown transform: {self.transform}")

  def _inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
    """Map transformed values back to the response scale."""
    if self.transform == "identity":
      return z

    if self.transform == "logit":
      return torch.sigmoid(z)

    if self.transform == "probit":
      return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

    raise ValueError(f"Unknown transform: {self.transform}")

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
    continuous unbounded case its inherited :attr:`var_type` and
    :attr:`support` already report. Each is refused rather than dropped, so a
    caller who declares an atom or a bound is told instead of being handed a
    fit that quietly ignored it.

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

    if support is not None and any(
      b is not None and math.isfinite(b) for b in support
    ):
      raise ValueError(
        f"{type(self).__name__} must model unbounded variables; got "
        f"support={support!r}. Its backends are fitted on a transform of the "
        "whole real line and impose no bound; use a Kde1d margin for a "
        "bounded variable."
      )

    y_t = self._prep(y).reshape(-1)

    x_t = self._conditioning(y_t, x=x)
    z_t = self._transform_y(y_t)

    self._fitted = False
    self._fit_model(x_t, z_t)
    self._nobs = y_t.shape[0]
    self._fitted = True

    return self

  @abstractmethod
  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    """Train the backend using tensors on the configured device.

    A third-party backend that requires host-side inputs is responsible for
    detaching and moving these tensors to CPU within its own adapter.
    """

  @abstractmethod
  def pdf(
    self,
    y: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate the conditional density ``f(y_i | x_i)``."""

  @abstractmethod
  def cdf(
    self,
    y: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate the conditional CDF ``F(y_i | x_i)``."""

  @abstractmethod
  def icdf(
    self,
    p: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate the conditional quantile ``F^-1(y_i | x_i)``."""

  @abstractmethod
  def pdf_grid(
    self,
    y_grid: torch.Tensor,
    /,
    *,
    x: torch.Tensor,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate the conditional density on the Cartesian product of ``x`` and ``y_grid``.

    Returns
    -------
    torch.Tensor
      Matrix with shape ``(n_x, n_y)`` where element ``(i, j)`` is
      ``f(y_grid[j] | x[i])``.

    Notes
    -----
    Implementations must predict at most once per conditioning row. They must
    not tile each row of ``x`` across the response grid before prediction.
    """

  @abstractmethod
  def cdf_grid(
    self,
    y_grid: torch.Tensor,
    /,
    *,
    x: torch.Tensor,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate the conditional CDF on the Cartesian product of ``x`` and ``y_grid``.

    Returns
    -------
    torch.Tensor
      Matrix with shape ``(n_x, n_y)`` where element ``(i, j)`` is
      ``F(y_grid[j] | x[i])``.

    Notes
    -----
    Implementations follow the same predict-once-per-conditioning-row
    requirement as :meth:`pdf_grid`.
    """

  def _sample_uniform(
    self,
    n: int,
    seeds: list[int],
  ) -> torch.Tensor:
    """Draw base uniforms on the configured device."""
    generator = torch.Generator(device=self._device)

    if seeds:
      generator.manual_seed(int(seeds[0]))
    else:
      generator.seed()

    return torch.rand(
      n,
      generator=generator,
      dtype=torch.float64,
      device=self._device,
    )
