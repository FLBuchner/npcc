"""The domain step: copula arguments into the open unit interval.

Copula scores live in ``(0, 1)`` open. :func:`check_open_unit` is this
package's *domain* step, the counterpart of the ``trim`` upstream applies
inside ``_prep_args``, and it differs from it in both halves. ``trim`` clamps
silently at the working precision, about ``1e-10`` in ``float64``; this
**rejects** a value at or outside ``{0, 1}`` first, and then clamps to a
caller-chosen ``eps``.

Each difference has its own reason. A copula argument of exactly ``0`` or ``1`` is
a caller error here rather than a rounding artifact, since every score reaching
an estimator comes from a probability integral transform that cannot produce
one. And the clamp width is not free to change: the inner regressors are fitted
on the logit of these values, where the two widths are some ten units of
feature space apart. ``RosenblattBicop._prep_args`` and its grid methods are
where both apply.
"""

from __future__ import annotations

import torch

__all__ = ["check_open_unit"]


def check_open_unit(
  values: torch.Tensor,
  eps: float,
  name: str,
) -> torch.Tensor:
  """Refuse values outside ``(0, 1)``, then clamp them ``eps`` inside it.

  Parameters
  ----------
  values : torch.Tensor
      Copula-scale values, of any shape.
  eps : float
      Distance kept from ``0`` and ``1``.
  name : str
      What ``values`` are, for the message.

  Returns
  -------
  torch.Tensor
      ``values``, clamped to ``[eps, 1 - eps]``.

  Raises
  ------
  ValueError
      If any value lies at or outside ``{0, 1}``.
  """
  if torch.any((values <= 0.0) | (values >= 1.0)):
    raise ValueError(f"{name} must lie strictly inside (0, 1).")

  return torch.clamp(values, eps, 1.0 - eps)
