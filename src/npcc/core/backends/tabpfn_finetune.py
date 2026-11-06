"""Fine-tuned TabPFN backend.

Fine-tunes TabPFN (`FinetunedTabPFNRegressor`, which fine-tunes the v2.5
checkpoint) with a proper-scoring-rule loss, then reads the predictive
distribution off its quantiles exactly as
:class:`~npcc.core.backends.tabpfn_quantile.TabPFNQuantileBackend` does -- so it
subclasses that backend and overrides only ``_fit_model``.

**Loss for our problem.** The downstream metric is the copula KL / log
predictive density, and by the Rosenblatt factorization the inner conditional
NLL equals the copula log-density (per direction).  So the density-optimal
fine-tuning objective is the bar-distribution **negative log-likelihood**
(`ce_loss_weight`), i.e. maximum likelihood = minimize KL.  We default to
``ce_loss_weight=1`` with the other terms off (TabPFN's own default is
``crps + mse``, which optimizes calibration/point, not density).  All loss
weights stay tunable.

Requires the core ``tabpfn`` package (fine-tuning is native, GPU-recommended).
Reference: ScoringBench, "Distributional Regression with Tabular Foundation
Models ... Proper Scoring Rules", arXiv:2603.08206.
"""

from __future__ import annotations

from typing import Any, Unpack

import torch
from pyvinecopulib.core.extend import to_numpy

from npcc.core.backends.tabpfn_quantile import TabPFNQuantileBackend
from npcc.core.margin import MarginOptions


class FinetunedTabPFNBackend(TabPFNQuantileBackend):
  """TabPFN fine-tuned with a scoring-rule loss, read through its quantiles.

  Parameters
  ----------
  epochs, learning_rate, early_stopping
      Fine-tuning schedule for ``FinetunedTabPFNRegressor``.
  ce_loss_weight, crps_loss_weight, crls_loss_weight, mse_loss_weight, mae_loss_weight
      Proper-scoring-rule loss weights.  Default = pure bar-distribution NLL
      (`ce_loss_weight=1`, rest 0) — the density-optimal objective for copula
      KL (see module docstring).
  finetune_kwargs : dict, optional
      Extra keyword arguments for ``FinetunedTabPFNRegressor`` (e.g.
      ``n_estimators_final_inference``).
  **margin : MarginOptions
      Forwarded to :class:`~npcc.core.margin.ConditionalMargin`.
  """

  def __init__(
    self,
    *,
    epochs: int = 30,
    learning_rate: float = 1e-5,
    early_stopping: bool = True,
    ce_loss_weight: float = 1.0,
    crps_loss_weight: float = 0.0,
    crls_loss_weight: float = 0.0,
    mse_loss_weight: float = 0.0,
    mae_loss_weight: float = 0.0,
    finetune_kwargs: dict[str, Any] | None = None,
    **margin: Unpack[MarginOptions],
  ) -> None:
    super().__init__(**margin)
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.early_stopping = early_stopping
    self.ce_loss_weight = ce_loss_weight
    self.crps_loss_weight = crps_loss_weight
    self.crls_loss_weight = crls_loss_weight
    self.mse_loss_weight = mse_loss_weight
    self.mae_loss_weight = mae_loss_weight
    self.finetune_kwargs = dict(finetune_kwargs or {})

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor

    model = FinetunedTabPFNRegressor(
      device=str(self._device),
      epochs=self.epochs,
      learning_rate=self.learning_rate,
      early_stopping=self.early_stopping,
      ce_loss_weight=self.ce_loss_weight,
      crps_loss_weight=self.crps_loss_weight,
      crls_loss_weight=self.crls_loss_weight,
      mse_loss_weight=self.mse_loss_weight,
      mae_loss_weight=self.mae_loss_weight,
      **self.finetune_kwargs,
    )
    model.fit(to_numpy(x), to_numpy(z))
    self.model_ = model
