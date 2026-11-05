"""Fine-tuned TabICL backend, read through its quantile head.

Fine-tunes TabICL (`FinetunedTabICLRegressor`) and reads the predictive
distribution off the same quantile head as
:class:`~npcc.core.backends.tabicl.TabICLBackend` — so it subclasses that
backend and overrides only ``_fit_model``.

Note: TabICL fine-tunes on a **pinball (quantile) loss** over its raw
quantile outputs (not configurable to a density/NLL loss), so — unlike the
TabPFN fine-tune backend — its objective is calibration-oriented rather than
the density-optimal bar-NLL.

Requires the ``tabicl`` extra (``pip install npcc[tabicl]``; GPU-recommended).
Reference: ScoringBench, arXiv:2603.08206 (fine-tuning tabular foundation
models with proper scoring rules); TabICL, arXiv:2502.05564.
"""

from __future__ import annotations

from typing import Any, Unpack

import torch
from pyvinecopulib.core.extend import to_numpy

from npcc.core.backends.tabicl import TabICLBackend
from npcc.core.margin import MarginOptions


class FinetunedTabICLBackend(TabICLBackend):
  """TabICL fine-tuned on pinball loss; quantile read-out.

  Parameters
  ----------
  epochs, learning_rate, early_stopping
      Fine-tuning schedule for ``FinetunedTabICLRegressor``.
  finetune_kwargs : dict, optional
      Extra keyword arguments for ``FinetunedTabICLRegressor`` (e.g.
      ``patience``).
  **margin : MarginOptions
      Forwarded to :class:`~npcc.core.margin.ConditionalMargin`.
  """

  def __init__(
    self,
    *,
    epochs: int = 30,
    learning_rate: float = 1e-5,
    early_stopping: bool = True,
    finetune_kwargs: dict[str, Any] | None = None,
    **margin: Unpack[MarginOptions],
  ) -> None:
    super().__init__(**margin)
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.early_stopping = early_stopping
    self.finetune_kwargs = dict(finetune_kwargs or {})

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    from tabicl import FinetunedTabICLRegressor

    model = FinetunedTabICLRegressor(
      device=str(self._device),
      epochs=self.epochs,
      learning_rate=self.learning_rate,
      early_stopping=self.early_stopping,
      **self.finetune_kwargs,
    )
    model.fit(to_numpy(x), to_numpy(z))
    self.model_ = model
