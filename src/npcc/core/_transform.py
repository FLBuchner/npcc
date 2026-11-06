"""The response scales a conditional margin can fit its backend on.

A margin fits its backend on ``z = g(y)`` rather than on ``y`` itself. Each
transform here is the one place that says four things about ``g``: the map,
its inverse, its derivative ``dz/dy`` -- the Jacobian a density picks up on
the way back to ``y`` -- and the support of ``y``. Which ends of that support
are finite is what decides how a quantile table closes its tails, so it is
stated here rather than inferred from the map.

Logit and probit are infinite at ``0`` and ``1``, and log at ``0``.
:meth:`ResponseTransform.clip` keeps ``y`` inside each such end before the map
is applied -- ``eps`` inside it on the copula scale, the smallest positive
double inside it for log, whose scale is the caller's -- and the Jacobian is
evaluated at the same clipped value, so the two always agree about where a
boundary observation sits.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal

import torch

__all__ = [
  "TRANSFORMS",
  "ResponseTransform",
  "Transform",
  "normal_cdf",
  "normal_pdf",
  "resolve_transform",
]

Transform = Literal["identity", "logit", "probit", "log"]
"""Name of a response transform, as the controls and the backends spell it."""

_SQRT_2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def normal_cdf(t: torch.Tensor) -> torch.Tensor:
  """Standard normal distribution function, accurate in the lower tail.

  Parameters
  ----------
  t : torch.Tensor
      Normal scores.

  Returns
  -------
  torch.Tensor
      ``Phi(t)``.
  """
  return 0.5 * torch.erfc(-t / _SQRT_2)


def normal_pdf(t: torch.Tensor) -> torch.Tensor:
  """Standard normal density.

  Parameters
  ----------
  t : torch.Tensor
      Normal scores.

  Returns
  -------
  torch.Tensor
      ``phi(t)``.
  """
  return torch.exp(-0.5 * t * t - _LOG_SQRT_2PI)


class ResponseTransform(ABC):
  """A monotone increasing map from the response scale to the modeled one.

  Attributes
  ----------
  name : str
      The name the controls use for it.
  support : tuple of float
      ``(lo, hi)``, the support of the response. An infinite end is one the
      modeled scale reaches only in the limit; a finite one is a bound the
      response cannot cross.
  """

  name: str
  support: tuple[float, float]

  def check_fit(self, y: torch.Tensor, owner: str) -> None:
    """Refuse fit responses the map cannot take.

    Parameters
    ----------
    y : torch.Tensor
        Responses a margin is about to be fitted on.
    owner : str
        Name of that margin's class, for the message.
    """
    del y, owner

  def clip(self, y: torch.Tensor, eps: float) -> torch.Tensor:
    """Keep ``y`` inside each end of the support the map cannot reach.

    Parameters
    ----------
    y : torch.Tensor
        Responses.
    eps : float
        Distance kept from a finite end of the support.

    Returns
    -------
    torch.Tensor
        ``y``, unchanged for a transform that is finite on all of its support.
    """
    del eps
    return y

  @abstractmethod
  def forward(self, y: torch.Tensor) -> torch.Tensor:
    """Map clipped responses to the modeled scale."""

  @abstractmethod
  def inverse(self, z: torch.Tensor) -> torch.Tensor:
    """Map the modeled scale back to responses."""

  @abstractmethod
  def jacobian(self, y: torch.Tensor) -> torch.Tensor:
    """Return ``dz/dy`` at clipped responses."""


class _Identity(ResponseTransform):
  name = "identity"
  support = (-math.inf, math.inf)

  def forward(self, y: torch.Tensor) -> torch.Tensor:
    return y

  def inverse(self, z: torch.Tensor) -> torch.Tensor:
    return z

  def jacobian(self, y: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(y)


class _Logit(ResponseTransform):
  name = "logit"
  support = (0.0, 1.0)

  def clip(self, y: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.clamp(y, eps, 1.0 - eps)

  def forward(self, y: torch.Tensor) -> torch.Tensor:
    return torch.log(y) - torch.log1p(-y)

  def inverse(self, z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z)

  def jacobian(self, y: torch.Tensor) -> torch.Tensor:
    return 1.0 / (y * (1.0 - y))


class _Probit(ResponseTransform):
  name = "probit"
  support = (0.0, 1.0)

  def clip(self, y: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.clamp(y, eps, 1.0 - eps)

  def forward(self, y: torch.Tensor) -> torch.Tensor:
    return _SQRT_2 * torch.erfinv(2.0 * y - 1.0)

  def inverse(self, z: torch.Tensor) -> torch.Tensor:
    return normal_cdf(z)

  def jacobian(self, y: torch.Tensor) -> torch.Tensor:
    return 1.0 / normal_pdf(self.forward(y))


class _Log(ResponseTransform):
  name = "log"
  support = (0.0, math.inf)

  def check_fit(self, y: torch.Tensor, owner: str) -> None:
    n_bad = int((y <= 0.0).sum())
    if n_bad:
      raise ValueError(
        f"{owner} with transform='log' must be fitted on positive responses; "
        f"got {n_bad} of {y.numel()} at or below 0. It models a continuous "
        "variable on (0, inf): shift the variable, or use a Kde1d margin for "
        "one with an atom at 0."
      )

  def clip(self, y: torch.Tensor, eps: float) -> torch.Tensor:
    # The scale of y is the caller's, so no fixed eps is small enough to be a
    # rounding of 0. A response at or below 0 reads as the bound itself, as one
    # outside (0, 1) reads as its nearest end under logit.
    del eps
    return y.clamp_min(torch.finfo(y.dtype).tiny)

  def forward(self, y: torch.Tensor) -> torch.Tensor:
    return torch.log(y)

  def inverse(self, z: torch.Tensor) -> torch.Tensor:
    return torch.exp(z)

  def jacobian(self, y: torch.Tensor) -> torch.Tensor:
    return 1.0 / y


TRANSFORMS: dict[str, ResponseTransform] = {
  t.name: t for t in (_Identity(), _Logit(), _Probit(), _Log())
}
"""Every transform, by name, in the order :data:`Transform` lists them."""


def resolve_transform(name: str) -> ResponseTransform:
  """Return the transform named ``name``.

  Parameters
  ----------
  name : str
      A member of :data:`Transform`.

  Returns
  -------
  ResponseTransform
      The transform.

  Raises
  ------
  ValueError
      If ``name`` is not a transform.
  """
  try:
    return TRANSFORMS[name]
  except KeyError:
    raise ValueError(
      f"Unknown transform: transform must be one of {list(TRANSFORMS)}; "
      f"got {name!r}"
    ) from None
