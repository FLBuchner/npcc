"""
Quantile-based TabPFN conditional margin.

This backend reads TabPFN's ``output_type="quantiles"`` result into the
quantile-table reconstruction machinery. It implements model fitting and
quantile prediction; PDF, CDF, inverse CDF, and grid evaluation are inherited
from :class:`~npcc.core.margin.ConditionalMargin`.

This module is the one place the concrete ``TabPFNRegressor`` is imported, and
it is looked up as a module global when a model is built -- which makes this
module the single monkeypatch target in the tests.

TabPFN receives host-side input tensors. These conversions are isolated in this
adapter, while all outputs are returned as torch tensors on the configured
model device.
"""

from __future__ import annotations

from typing import Any, Unpack

import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from npcc.core.margin import ConditionalMargin, MarginOptions

DEFAULT_MODEL_VERSION = ModelVersion.V3
"""Default TabPFN model version, shared by both TabPFN backends."""


class TabPFNQuantileBackend(ConditionalMargin):
  """Conditional margin using TabPFN's predicted quantiles.

  Parameters
  ----------
  model_kwargs : dict, optional
      Additional arguments passed to ``TabPFNRegressor``.
  model_version : ModelVersion, str, or None, default=ModelVersion.V3
      TabPFN model version, as the enum or its value (``"v2.5"``, ``"v3"``).
      ``None`` builds a bare ``TabPFNRegressor`` rather than a versioned
      default.
  **margin : MarginOptions
      Forwarded to :class:`~npcc.core.margin.ConditionalMargin`.
  """

  model_: Any | None

  def __init__(
    self,
    *,
    model_kwargs: dict[str, Any] | None = None,
    model_version: ModelVersion | str | None = DEFAULT_MODEL_VERSION,
    **margin: Unpack[MarginOptions],
  ) -> None:
    super().__init__(**margin)
    self.model_kwargs = dict(model_kwargs or {})
    self.model_kwargs.setdefault("device", str(self._device))
    self.model_version = (
      None if model_version is None else ModelVersion(model_version)
    )
    self.model_ = None

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    """Fit TabPFN."""
    if self.model_version is None:
      self.model_ = TabPFNRegressor(**self.model_kwargs)
    else:
      self.model_ = TabPFNRegressor.create_default_for_version(
        self.model_version, **self.model_kwargs
      )

    # TabPFN converts its inputs to NumPy itself, which a CUDA tensor refuses.
    self.model_.fit(x.detach().cpu(), z.detach().cpu())

  def _predict_quantiles(
    self,
    x: torch.Tensor,
    alphas: torch.Tensor,
  ) -> torch.Tensor:
    """Predict one quantile table for a chunk of conditioning rows.

    TabPFN returns one array per level, so the levels lead.
    """
    assert self.model_ is not None

    predicted = self.model_.predict(
      x.detach().cpu(),
      output_type="quantiles",
      quantiles=alphas.tolist(),
    )
    return self._quantile_rows(predicted, x, levels_first=True)
