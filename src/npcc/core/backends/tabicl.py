"""
TabICL quantile conditional margin.

TabICL is an in-context tabular foundation model whose regressor exposes
conditional quantiles directly. It predicts every requested probability level
in a single call.

PDF, CDF, inverse CDF, and grid evaluation are inherited from
:class:`~npcc.core.margin.ConditionalMargin`.

TabICL uses a NumPy-facing estimator interface. Input conversion is isolated
inside this adapter, and predictions are immediately converted back to torch
tensors on the input device and with the input dtype.

This backend requires the ``tabicl`` optional dependency.
"""

from __future__ import annotations

from typing import Any, Unpack

import torch
from pyvinecopulib.core.extend import to_numpy
from tabicl import TabICLRegressor

from npcc.core.margin import ConditionalMargin, MarginOptions


class TabICLBackend(ConditionalMargin):
  """Conditional margin using TabICL's quantile output.

  Parameters
  ----------
  model_kwargs : dict, optional
      Additional arguments passed to ``TabICLRegressor``.
  **margin : MarginOptions
      Forwarded to :class:`~npcc.core.margin.ConditionalMargin`.
  """

  model_: TabICLRegressor | None

  def __init__(
    self,
    *,
    model_kwargs: dict[str, Any] | None = None,
    **margin: Unpack[MarginOptions],
  ) -> None:
    super().__init__(**margin)
    self.model_kwargs = dict(model_kwargs or {})
    self.model_kwargs.setdefault("device", str(self._device))
    self.model_ = None

  def _fit_model(
    self,
    x: torch.Tensor,
    z: torch.Tensor,
  ) -> None:
    """Prepare TabICL for conditional prediction."""
    self.model_ = TabICLRegressor(**self.model_kwargs)

    self.model_.fit(
      to_numpy(x),
      to_numpy(z),
    )

  def _predict_quantiles(
    self,
    x: torch.Tensor,
    alphas: torch.Tensor,
  ) -> torch.Tensor:
    """Predict one quantile table for a chunk of conditioning rows.

    TabICL documents ``(n_samples, n_quantiles)``, so the rows lead.
    """
    assert self.model_ is not None

    predicted = self.model_.predict(
      to_numpy(x),
      output_type="quantiles",
      alphas=alphas.tolist(),
    )
    return self._quantile_rows(predicted, x, levels_first=False)
