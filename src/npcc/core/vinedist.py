"""Original-scale distributions built from Rosenblatt vine copulas."""

from __future__ import annotations

import torch
from pyvinecopulib.core import ControlsLike, VinecopLike, VinedistBase

from npcc.core._placement import TensorPlacement, resolve_device


class RosenblattVinedist(TensorPlacement, VinedistBase[torch.Tensor]):
  """Original-scale distribution backed by a Rosenblatt vine.

  The distribution is constructed from an unfitted or fitted vine copula and
  one margin per variable. The inherited :meth:`fit` method first fits the
  existing margins, transforms the observations to the copula scale, and then
  fits the existing vine copula along its fixed structure.

  The fit controls name the placement, and the observations are brought onto
  it -- so a NumPy array or a CPU tensor handed to a CUDA-configured
  distribution is converted rather than refused.

  Notes
  -----
  This class intentionally does not declare ``vinecop_class`` or
  ``margin_class``. The generic :meth:`VinedistBase.from_data` construction
  path cannot create a non-simplified Rosenblatt vine correctly. Construct
  the required parts first and then call :meth:`fit`.
  """

  supports_fit_covariates: bool = True

  def __init__(
    self,
    vinecop: VinecopLike[torch.Tensor],
    # Upstream's own type, and not narrowed to a sequence: a single margin
    # standing for every variable is a documented call, and a foreign
    # distribution object is coerced with `as_margin` rather than refused.
    margins: object,
  ) -> None:
    super().__init__(vinecop, margins)
    # The copula names the placement, since it was constructed with it and
    # holds the pair copulas that evaluate on it. Recording it here is what
    # lets the inherited data-scale surface place a foreign array: the base's
    # `_prep` infers a placement from arrays this object holds, and a vine
    # distribution holds only its two parts.
    self._set_placement(getattr(vinecop, "_device", None))

  @classmethod
  def _coerce_fit_data(
    cls,
    # Anything `torch.as_tensor` accepts, which is this hook's whole job: a
    # caller may reasonably hand a NumPy array to a torch distribution.
    y: object,
    controls: ControlsLike | None,
  ) -> torch.Tensor:
    """Put the fit inputs on the placement the controls name.

    The **controls** decide the placement and the data follow, not the other
    way round. Every other part of the lane is built from the controls -- the
    conditional margins through
    :func:`~npcc.core.registry.create_backend`, the vine through its own
    ``device`` argument -- so reading the placement off ``y`` instead would
    leave the margins wherever the caller's data happened to be while the pair
    copulas went to the controls' device, and the first concatenation of the
    two would fail.

    ``float64`` throughout, which is this package's working precision: an
    integer ``y`` would otherwise give the margins an integer grid.

    Parameters
    ----------
    y : object
        The caller's observations, in any form ``torch.as_tensor`` accepts.
    controls : ControlsLike, or None
        Fit configuration. Its ``device``, when it has one, is the placement.
        Observation weights ride here too, and are refused against each part's
        ``supports_weights`` one level up rather than placed here.

    Returns
    -------
    torch.Tensor
        The observations, on the placement the controls name.
    """
    del cls

    device = resolve_device(getattr(controls, "device", None))
    return torch.as_tensor(y).to(device=device, dtype=torch.float64)
