"""
NGBoost parametric conditional margin.

NGBoost fits a parametric conditional distribution using natural-gradient
boosting, and ``pred_dist(x).dist`` is a vectorized SciPy distribution with
one parameter set per conditioning row. Its quantile function, read at the
table's levels, is the table.

The default ``Normal`` distribution is a straight line in the table's normal
scores, so the reconstruction reproduces it exactly inside the table and only
its tails beyond ``Phi(+-normal_limit)`` are the table's own.

NumPy and SciPy conversions are isolated inside this adapter. All public inputs
and outputs remain torch tensors.

This backend requires the ``ngboost`` optional dependency.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from ngboost import NGBRegressor
from ngboost.distns import Normal
from pyvinecopulib.core.extend import to_numpy

from npcc.core._transform import Transform
from npcc.core.margin import ConditionalMargin, QuantileTableConfig


class NGBoostBackend(ConditionalMargin):
  """Conditional margin using an NGBoost parametric distribution.

  Parameters
  ----------
  transform
    Response transformation inherited from
    :class:`~npcc.core.margin.ConditionalMargin`.
  quantile_table_config
    Quantile-table reconstruction configuration.
  eps
    Boundary clipping distance for logit and probit transformations.
  device
    Device used for returned tensors.
  batch_size
    Maximum number of conditioning rows evaluated in one prediction call.
  dist
    NGBoost distribution class. The default is ``Normal``.
  **ngb_kwargs
    Additional arguments passed to ``NGBRegressor``.
  """

  #: A parametric quantile function is already smooth across levels.
  default_smoothing: ClassVar[float] = 0.0

  model_: NGBRegressor | None

  def __init__(
    self,
    *,
    transform: Transform = "logit",
    quantile_table_config: QuantileTableConfig | None = None,
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    dist: type | None = None,
    **ngb_kwargs: object,
  ) -> None:
    super().__init__(
      transform=transform,
      quantile_table_config=quantile_table_config,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )
    self._dist = dist if dist is not None else Normal
    self.ngb_kwargs = dict(ngb_kwargs)
    self.model_ = None

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    """Fit the NGBoost model on transformed responses."""
    model = NGBRegressor(Dist=self._dist, verbose=False, **self.ngb_kwargs)
    model.fit(to_numpy(x), to_numpy(z))
    self.model_ = model

  def _predict_quantiles(
    self,
    x: torch.Tensor,
    alphas: torch.Tensor,
  ) -> torch.Tensor:
    """Predict one quantile table for a chunk of conditioning rows."""
    assert self.model_ is not None

    frozen = self.model_.pred_dist(to_numpy(x)).dist
    # The frozen distribution holds one parameter set per row, so levels go on
    # a leading axis and broadcast against them.
    predicted = frozen.ppf(to_numpy(alphas)[:, None])
    return self._quantile_rows(predicted, x, levels_first=True)
