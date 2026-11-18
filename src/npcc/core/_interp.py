"""One-dimensional linear interpolation on the device a grid lives on.

NumPy has :func:`numpy.interp`; this is its Torch equivalent, so the Sinkhorn
projection and the joint-CDF integration in
:mod:`npcc.core.bicop` interpolate on the device their grids were evaluated on
rather than round-tripping through the host.

pyvinecopulib's own interpolation helpers do not cover this: its
``torch._bicop_interp`` is bilinear on the unit square for pair-copula density
grids, and ``torch._margin_kde1d_interp`` is cubic over kernel-density cells.
Both are private, and neither is a general one-dimensional linear
interpolator.
"""

from __future__ import annotations

import torch

__all__ = ["interp"]


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
  """1-D linear interpolation, analog of :func:`numpy.interp`.

  ``xp`` must be sorted ascending.  Values of ``x`` outside
  ``[xp[0], xp[-1]]`` are clamped to the endpoints (flat extrapolation),
  matching NumPy's default behavior.  Inputs are 1-D; output has the
  shape of ``x``.
  """
  n = xp.shape[0]
  idx = torch.searchsorted(xp, x.contiguous()).clamp(1, n - 1)
  x0 = xp[idx - 1]
  x1 = xp[idx]
  y0 = fp[idx - 1]
  y1 = fp[idx]
  denom = (x1 - x0).clamp_min(torch.finfo(xp.dtype).tiny)
  t = ((x - x0) / denom).clamp(0.0, 1.0)
  return y0 + t * (y1 - y0)
