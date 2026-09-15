"""Declare the placement this package's estimators evaluate on.

pyvinecopulib draws three separable steps over every input -- *placement*,
onto the dtype and device the object's own tensors live on; *layout*, the
shapes it admits; and *domain*, clamping copula arguments into the open unit
square. The first two are overridable hooks per level, ``_prep`` and
``_layout``; the domain step is the module-level ``trim`` that ``_prep_args``
applies after them, which :class:`~npcc.core.bicop.RosenblattBicop` overrides
to reject rather than clamp and to use its own ``eps``.

The ``_prep`` hook itself is upstream's
:class:`~pyvinecopulib.torch.TensorPlacementMixin`, which resolves a placement
in three steps: a registered floating-point tensor, then a ``device`` and
``dtype`` the host *declares*, then an empty CPU ``float64`` tensor. None of
these estimators is a ``torch.nn.Module``, so the first step finds nothing and
the second is the one that applies -- which is all
:class:`TensorPlacement` supplies.

Two things about that are worth keeping in view.

**The mixin goes ahead of the canonical base.** All four bases already inherit
``PlacementMixin``, so a mixin placed *after* one never wins the lookup and
``_prep`` silently becomes the array-API inference again.

**``as_tensor``, not ``asarray``, is what keeps a gradient.** The array-API
route delegates to ``torch.asarray``, whose default for ``requires_grad``
changed -- silently ``False`` on torch 2.11, ``obj.requires_grad`` from 2.13.
The hook's ``as_tensor`` has always carried it, so the answer here does not
depend on which torch is installed.
"""

from __future__ import annotations

import torch
from pyvinecopulib.torch import TensorPlacementMixin

__all__ = ["TensorPlacement", "resolve_device"]


def resolve_device(device: str | torch.device | None) -> torch.device:
  """Resolve ``None`` to ``cuda`` if available, else ``cpu``.

  A bare ``cuda`` device (no index) is normalized to ``cuda:<current index>``
  so it compares equal to the device tensors actually materialize on (e.g.
  ``cuda:0``); ``torch.device("cuda") != torch.device("cuda:0")`` otherwise.

  Parameters
  ----------
  device : str, torch.device, or None
      The requested device, or ``None`` to pick the best available one.

  Returns
  -------
  torch.device
      The resolved device, with a CUDA index filled in.
  """
  if device is None:
    resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
    resolved = torch.device(device)

  if resolved.type == "cuda" and resolved.index is None:
    resolved = torch.device("cuda", torch.cuda.current_device())

  return resolved


class TensorPlacement(TensorPlacementMixin):
  """Declare ``float64`` on ``_device``, and let the mixin do the rest.

  The declaration is by ``_ref_tensor`` rather than by exposing ``device`` and
  ``dtype`` attributes for the mixin's second step to read: these estimators
  keep the device private, and an override is what upstream's docstring names
  for a host none of the three steps fits.
  """

  _device: torch.device

  def _set_placement(self, device: str | torch.device | None) -> None:
    """Record the device this estimator evaluates on.

    Parameters
    ----------
    device : str, torch.device, or None
        The requested device; ``None`` picks the best available one.

    Returns
    -------
    None
    """
    self._device = resolve_device(device)

  def _ref_tensor(self) -> torch.Tensor:
    """An empty tensor naming this estimator's dtype and device.

    Returns
    -------
    torch.Tensor
        Empty, ``float64``, on ``_device``.
    """
    return torch.empty(0, dtype=torch.float64, device=self._device)
