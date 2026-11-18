"""
Synthefy Nori quantile conditional margin.

Nori is an in-context tabular foundation model whose regressor exposes
conditional quantiles directly. It predicts every requested probability level
in a single call.

PDF, CDF, inverse CDF, and grid evaluation are inherited from
:class:`~npcc.core.margin.ConditionalMargin`.

Nori uses a NumPy-facing estimator interface. Input conversion is isolated
inside this adapter, and predictions are immediately converted back to torch
tensors on the input device and with the input dtype.

This backend requires the ``synthefy-nori`` optional dependency.
"""

from __future__ import annotations

from typing import Any

import torch
from synthefy_nori import NoriRegressor

from npcc.core._transform import Transform
from npcc.core.margin import ConditionalMargin, QuantileTableConfig


class NoriBackend(ConditionalMargin):
  """Conditional margin using Synthefy Nori's quantile output.

  Parameters
  ----------
  transform
    Response transformation inherited from
    :class:`~npcc.core.margin.ConditionalMargin`.
  quantile_table_config
    Quantile-table reconstruction configuration.
  eps
    Distance used when clipping values away from the boundaries of
    ``(0, 1)`` before the logit or probit transform.
  device
    Device used by Nori and returned tensors.
  batch_size
    Maximum number of conditioning rows evaluated in one prediction call.
  **nori_kwargs
    Additional arguments passed to ``NoriRegressor``.
  """

  model_: NoriRegressor | None

  def __init__(
    self,
    *,
    transform: Transform = "logit",
    quantile_table_config: QuantileTableConfig | None = None,
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    **nori_kwargs: Any,  # noqa: ANN401 - forwarded to NoriRegressor
  ) -> None:
    super().__init__(
      transform=transform,
      quantile_table_config=quantile_table_config,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )
    self.nori_kwargs = dict(nori_kwargs)
    self.nori_kwargs.setdefault("device", str(self._device))
    self.model_ = None

  def _fit_model(
    self,
    x: torch.Tensor,
    z: torch.Tensor,
  ) -> None:
    """Prepare Nori for conditional prediction."""
    model = NoriRegressor(**self.nori_kwargs)

    x_host = (
      x.detach()
      .to(
        device="cpu",
        dtype=torch.float32,
      )
      .numpy()
    )
    z_host = (
      z.detach()
      .to(
        device="cpu",
        dtype=torch.float64,
      )
      .numpy()
    )

    model.fit(x_host, z_host)

    self.model_ = model

  def _predict_quantiles(
    self,
    x: torch.Tensor,
    alphas: torch.Tensor,
  ) -> torch.Tensor:
    """Predict one quantile table for a chunk of conditioning rows.

    Nori documents ``(len(quantiles), n_samples)``, so the levels lead.
    """
    assert self.model_ is not None

    x_host = (
      x.detach()
      .to(
        device="cpu",
        dtype=torch.float64,
      )
      .numpy()
    )

    predicted = self.model_.predict(
      x_host,
      output_type="quantiles",
      quantiles=alphas.tolist(),
    )

    return self._quantile_rows(predicted, x, levels_first=True)
