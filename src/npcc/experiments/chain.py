"""The chain-rule copula density: E11's baseline, a copula without a vine.

**Approach.** The conditional backend behind one Rosenblatt direction of a
pair copula can condition on several scores at once, so a copula density can
be written down by the chain rule along any variable order ``pi``::

    c(u | x) = prod_{k=2..d} f(u_pi(k) | u_pi(1), ..., u_pi(k-1), x).

There is no factor for ``pi(1)``: on the copula scale ``U_pi(1) | x`` is
uniform, so its density is 1. An order needs ``d - 1`` regressions and no
structure, which is the question it answers -- what a vine adds over writing
the density down directly.

**One factor is one Rosenblatt direction.** Factor ``k`` is the estimator
:class:`~npcc.core.bicop.RosenblattBicop` builds for a direction: a
:class:`~npcc.core.margin.ConditionalMargin` from
:func:`~npcc.core.registry.create_backend` on the controls' backend, response
transform, quantile-table read-out, ``eps``, device, batch size and backend
settings. Only its features differ, ``[g(u_pi(1)), ..., g(u_pi(k-1)), x]``
rather than ``[g(u), x]``. Every copula argument is prepared as the pair
copula prepares its own: refused at or outside ``{0, 1}`` and clamped ``eps``
inside it, after which the conditioning scores pass through the controls'
``feature_transform`` ``g``. A factor's density is the margin's ``pdf`` on the
copula scale, the response transform's Jacobian included, which is how a
direction is evaluated.

**Sampling.** An order's density is exact for its own inverse Rosenblatt
transform: :meth:`ChainCopula.inverse_rosenblatt` keeps ``u_pi(1)`` at its
uniform level and reads every later score from its factor's quantile table,
the table its density comes from, given the scores already drawn and ``x``.
:meth:`ChainCopula.rosenblatt` is its inverse, each factor's CDF at the
point. A mixture over orders is sampled by drawing from each order in turn,
so its draws and its density, the orders' mean, describe one distribution.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
import torch
from pyvinecopulib.core.extend import prepare_covariates

from npcc.core._placement import TensorPlacement
from npcc.core._transform import TRANSFORMS
from npcc.core._trim import check_open_unit
from npcc.core.controls import FitControlsRosenblattBicop
from npcc.core.margin import ConditionalMargin, QuantileTableConfig
from npcc.core.registry import create_backend

__all__ = ["ChainCopula", "draw_order"]


def draw_order(d: int, seed: int) -> list[int]:
  """A variable order drawn uniformly from the ``d!`` permutations.

  Parameters
  ----------
  d : int
      The number of variables.
  seed : int
      Seed of the NumPy generator the permutation is drawn from; the same
      seed gives the same order.

  Returns
  -------
  list of int
      A permutation of ``1, ..., d``, as 1-based variable labels.
  """
  rng = np.random.default_rng(seed)
  return [int(v) + 1 for v in rng.permutation(d)]


def _controls(
  controls: FitControlsRosenblattBicop | None,
) -> FitControlsRosenblattBicop:
  """Narrow the caller's controls, refusing the one setting a chain lacks."""
  if controls is None:
    return FitControlsRosenblattBicop()
  if not isinstance(controls, FitControlsRosenblattBicop):
    raise TypeError(
      "controls must be a FitControlsRosenblattBicop, the controls of the "
      f"pair copula whose direction each factor is; got "
      f"{type(controls).__name__}. Covariates are passed as `x=`."
    )
  if controls.sinkhorn_iters is not None:
    raise ValueError(
      "ChainCopula cannot honor sinkhorn_iters, and will not drop it "
      f"silently; got {controls.sinkhorn_iters}. The Sinkhorn projection "
      "corrects a pair copula's two averaged directions, and a chain factor "
      "is one direction: pass sinkhorn_iters=None."
    )
  return controls


class ChainCopula(TensorPlacement):
  """The chain-rule copula density along one variable order.

  Parameters
  ----------
  order : sequence of int
      The order ``pi``, as 1-based variable labels, the convention of
      pyvinecopulib's orders: ``(1, 2)`` models ``U_2`` given ``U_1``.
  controls : FitControlsRosenblattBicop, or None, optional
      The pair copulas' controls, which configure every factor as they
      configure one Rosenblatt direction. ``None`` uses their defaults.

  Raises
  ------
  ValueError
      If ``order`` is not a permutation of ``1, ..., d`` with ``d >= 2``, or
      the controls ask for a Sinkhorn projection.
  TypeError
      If ``controls`` is not a :class:`FitControlsRosenblattBicop`.
  """

  def __init__(
    self,
    order: Sequence[int],
    controls: FitControlsRosenblattBicop | None = None,
  ) -> None:
    labels = [int(v) for v in order]
    if len(labels) < 2 or sorted(labels) != list(range(1, len(labels) + 1)):
      raise ValueError(
        f"order must be a permutation of 1, ..., d with d >= 2; got {labels}"
      )
    self.order: list[int] = labels
    self._apply_controls(_controls(controls))
    self.factors_: list[ConditionalMargin] = []

  def _apply_controls(self, controls: FitControlsRosenblattBicop) -> None:
    """Record the controls and the device every factor evaluates on."""
    self.controls = controls
    self._set_placement(controls.device)

  @property
  def d(self) -> int:
    """The number of variables."""
    return len(self.order)

  def _new_factor(self) -> ConditionalMargin:
    """One unfitted factor, built as a pair copula builds a direction."""
    c = self.controls
    return create_backend(
      c.backend,
      transform=c.transform,
      quantile_table_config=c.quantile_table_config or QuantileTableConfig(),
      eps=c.eps,
      device=self._device,
      batch_size=c.batch_size,
      backend_kwargs=dict(c.backend_kwargs or {}),
    )

  def _inputs(
    self, u: torch.Tensor, x: torch.Tensor | None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Place ``u`` and ``x``, check their layouts, and bring ``u`` inside.

    ``u`` takes a pair copula's domain step, :func:`check_open_unit` at the
    controls' ``eps``; ``x`` is placed and checked but never clamped, being
    arbitrary reals. An absent ``x`` becomes the ``(n, 0)`` block the
    features concatenate against.
    """
    u_t = self._prep(u)
    if u_t.ndim != 2 or u_t.shape[1] != self.d:
      raise ValueError(
        f"u must have shape (n, {self.d}), one column per variable of the "
        f"order {self.order}; got {tuple(u_t.shape)}"
      )
    u_t = check_open_unit(u_t, self.controls.eps, "u")
    n = u_t.shape[0]
    if x is None:
      return u_t, torch.empty((n, 0), dtype=torch.float64, device=self._device)
    x_t = self._prep(x)
    # For the layout and row-alignment check and its message; it returns the
    # placed `x_t` itself.
    prepare_covariates(self, x_t, n)
    return u_t, x_t

  def _column(self, k: int) -> int:
    """The column of ``u`` holding ``u_pi(k + 1)``, ``k`` 0-based."""
    return self.order[k] - 1

  def _features(self, u: torch.Tensor, k: int, x: torch.Tensor) -> torch.Tensor:
    """Factor ``k + 1``'s features, ``[g(u_pi(1)), ..., g(u_pi(k)), x]``.

    ``u`` has already been clamped at ``eps``, so the identity hands the
    regressor the clamped scores themselves, as a pair copula does.
    """
    given = u[:, [self._column(j) for j in range(k)]]
    if self.controls.feature_transform != "identity":
      given = TRANSFORMS[self.controls.feature_transform].forward(given)
    return torch.cat([given, x], dim=1)

  def fit(
    self,
    u: torch.Tensor,
    /,
    controls: FitControlsRosenblattBicop | None = None,
    *,
    x: torch.Tensor | None = None,
    seeds: Sequence[int] | None = None,
  ) -> Self:
    """Fit the ``d - 1`` factors, one regression each.

    Parameters
    ----------
    u : torch.Tensor, shape (n, d)
        Copula-scale observations, column ``j`` holding variable ``j + 1``.
    controls : FitControlsRosenblattBicop, or None, optional
        Replaces the controls given at construction; ``None`` keeps them.
    x : torch.Tensor, shape (n, p), or None, optional
        Covariates every factor conditions on.
    seeds : sequence of int, or None, optional
        One torch seed per factor, in the order's sequence, set before that
        factor's fit; ``None`` leaves the global generator alone.

    Returns
    -------
    ChainCopula
        This estimator, fitted.

    Raises
    ------
    ValueError
        If ``seeds`` does not hold one seed per factor, or ``u`` or ``x``
        has the wrong layout.
    """
    if controls is not None:
      self._apply_controls(_controls(controls))
    if seeds is not None and len(seeds) != self.d - 1:
      raise ValueError(
        f"seeds must hold one seed per factor, {self.d - 1} for the order "
        f"{self.order}; got {len(seeds)}"
      )
    u_t, x_t = self._inputs(u, x)
    factors: list[ConditionalMargin] = []
    for k in range(1, self.d):
      if seeds is not None:
        torch.manual_seed(int(seeds[k - 1]))
      factor = self._new_factor()
      factor.fit(u_t[:, self._column(k)], x=self._features(u_t, k, x_t))
      factors.append(factor)
    self.factors_ = factors
    return self

  def factor_pdfs(
    self, u: torch.Tensor, /, *, x: torch.Tensor | None = None
  ) -> torch.Tensor:
    """Each factor's density at ``u``, for ``k = 2, ..., d``.

    Parameters
    ----------
    u : torch.Tensor, shape (n, d)
        Copula-scale points, column ``j`` holding variable ``j + 1``.
    x : torch.Tensor, shape (n, p), or None, optional
        Covariates, one row per point.

    Returns
    -------
    torch.Tensor, shape (d - 1, n)
        Row ``k - 2`` is ``f(u_pi(k) | u_pi(1), ..., u_pi(k - 1), x)`` on the
        copula scale.

    Raises
    ------
    RuntimeError
        If the estimator has not been fitted.
    """
    self._check_fitted()
    u_t, x_t = self._inputs(u, x)
    return torch.stack(
      [
        factor.pdf(u_t[:, self._column(k)], x=self._features(u_t, k, x_t))
        for k, factor in enumerate(self.factors_, start=1)
      ]
    )

  def pdf(
    self, u: torch.Tensor, /, *, x: torch.Tensor | None = None
  ) -> torch.Tensor:
    """The copula density, the product of the factors' densities.

    Parameters
    ----------
    u : torch.Tensor, shape (n, d)
        Copula-scale points.
    x : torch.Tensor, shape (n, p), or None, optional
        Covariates, one row per point.

    Returns
    -------
    torch.Tensor, shape (n,)
        Density values.
    """
    return self.factor_pdfs(u, x=x).prod(dim=0)

  def _check_fitted(self) -> None:
    if len(self.factors_) != self.d - 1:
      raise RuntimeError("ChainCopula must be fitted before it is evaluated.")

  def rosenblatt(
    self, u: torch.Tensor, /, *, x: torch.Tensor | None = None
  ) -> torch.Tensor:
    """The order's Rosenblatt transform of ``u``: uniform levels, in order.

    Parameters
    ----------
    u : torch.Tensor, shape (n, d)
        Copula-scale points, column ``j`` holding variable ``j + 1``.
    x : torch.Tensor, shape (n, p), or None, optional
        Covariates, one row per point.

    Returns
    -------
    torch.Tensor, shape (n, d)
        Column 0 is ``u_pi(1)``; column ``k - 1`` is
        ``F(u_pi(k) | u_pi(1), ..., u_pi(k - 1), x)``, factor ``k``'s CDF.

    Raises
    ------
    RuntimeError
        If the estimator has not been fitted.
    """
    self._check_fitted()
    u_t, x_t = self._inputs(u, x)
    levels = [u_t[:, self._column(0)]]
    levels += [
      factor.cdf(u_t[:, self._column(k)], x=self._features(u_t, k, x_t))
      for k, factor in enumerate(self.factors_, start=1)
    ]
    return torch.stack(levels, dim=1)

  def inverse_rosenblatt(
    self, w: torch.Tensor, /, *, x: torch.Tensor | None = None
  ) -> torch.Tensor:
    """Map uniform levels to the copula scale along the order.

    ``u_pi(1)`` is the first level itself; each later score is its factor's
    conditional quantile at its level, given the scores before it and ``x``,
    and is kept ``eps`` inside the unit interval, as a pair copula keeps an
    inverse h-function's.

    Parameters
    ----------
    w : torch.Tensor, shape (n, d)
        Levels in ``(0, 1)``, column ``k - 1`` for ``u_pi(k)``; prepared as a
        copula argument is.
    x : torch.Tensor, shape (n, p), or None, optional
        Covariates, one row per point.

    Returns
    -------
    torch.Tensor, shape (n, d)
        Copula-scale points, column ``j`` holding variable ``j + 1``.

    Raises
    ------
    RuntimeError
        If the estimator has not been fitted.
    ValueError
        If a factor's quantile is not finite or not in ``[0, 1]``.
    """
    self._check_fitted()
    w_t, x_t = self._inputs(w, x)
    eps = self.controls.eps
    u = torch.empty_like(w_t)
    u[:, self._column(0)] = w_t[:, 0]
    for k, factor in enumerate(self.factors_, start=1):
      out = factor.icdf(w_t[:, k], x=self._features(u, k, x_t))
      if not torch.isfinite(out).all():
        raise ValueError(
          f"factor {k + 1} of the order {self.order} must return finite "
          f"quantiles; got non-finite values"
        )
      if torch.any((out < 0) | (out > 1)):
        raise ValueError(
          f"factor {k + 1} of the order {self.order} must return quantiles "
          f"in [0, 1]; got others. Its transform must map the unit interval."
        )
      u[:, self._column(k)] = torch.clamp(out, eps, 1.0 - eps)
    return u

  def sample(
    self,
    n: int,
    /,
    *,
    x: torch.Tensor | None = None,
    seeds: Sequence[int] = (),
  ) -> torch.Tensor:
    """Draw ``n`` points from the order's copula.

    Parameters
    ----------
    n : int
        The number of draws.
    x : torch.Tensor, shape (n, p), or None, optional
        Covariates, one row per draw.
    seeds : sequence of int, default=()
        Seeds of the uniform levels, the first used, as a vine's ``sample``
        takes them; empty for an unseeded draw.

    Returns
    -------
    torch.Tensor, shape (n, d)
        Copula-scale draws, column ``j`` holding variable ``j + 1``.
    """
    eps = self.controls.eps
    w = self._draw_uniform((n, self.d), seeds=seeds)
    return self.inverse_rosenblatt(torch.clamp(w, eps, 1.0 - eps), x=x)

  def logpdf(
    self, u: torch.Tensor, /, *, x: torch.Tensor | None = None
  ) -> torch.Tensor:
    """The copula log-density, the sum of the factors' log-densities.

    Summed in log space, as a vine's is, so a density whose product
    underflows still has its logarithm; a factor density of zero gives
    ``-inf``.

    Parameters
    ----------
    u : torch.Tensor, shape (n, d)
        Copula-scale points.
    x : torch.Tensor, shape (n, p), or None, optional
        Covariates, one row per point.

    Returns
    -------
    torch.Tensor, shape (n,)
        Log-density values.
    """
    return torch.log(self.factor_pdfs(u, x=x)).sum(dim=0)
