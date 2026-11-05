"""
CatBoost quantile conditional margin.

This backend fits one CatBoost model with the ``MultiQuantile`` loss over every
probability level in ``config.alphas()``. Unlike the per-level tree backends,
the levels share one ensemble of symmetric trees, and each leaf holds one value
per level.

CatBoost receives NumPy-backed host arrays and runs on the host. Predictions
are converted immediately to torch tensors on the input device and with the
input dtype.

PDF, CDF, inverse CDF, and grid evaluation are inherited from
:class:`~npcc.core.margin.ConditionalMargin`.

This backend requires the ``catboost`` optional dependency.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from catboost import CatBoostRegressor
from pyvinecopulib.core.extend import to_numpy

from npcc.core._transform import Transform
from npcc.core.margin import ConditionalMargin, QuantileTableConfig

#: The schedule the LightGBM backend uses, so the two are compared at the same
#: capacity: depth-three trees and a short schedule. CatBoost's own defaults
#: (depth 6, 1000 iterations) fitted a far noisier table at the study's sizes.
_DEFAULT_PARAMS: dict[str, object] = {
  "learning_rate": 0.1,
  "depth": 3,
}

#: Set by this backend, so a caller cannot pass them -- under CatBoost's own
#: names or any alias it would read in their place.
_RESERVED = frozenset(
  {
    "loss_function",
    "objective",
    "iterations",
    "n_estimators",
    "num_boost_round",
    "num_trees",
    "thread_count",
    "logging_level",
    "verbose",
    "silent",
    "allow_writing_files",
  }
)


class CatBoostQuantileBackend(ConditionalMargin):
  """Conditional margin using CatBoost multi-quantile regression.

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
    Device used for returned tensors. CatBoost itself runs on the host.
  batch_size
    Maximum number of conditioning rows evaluated in one prediction call.
  iterations
    Number of boosting iterations.
  thread_count
    Number of threads CatBoost fits and predicts with; ``None`` uses every
    available core.
  **catboost_params
    CatBoost parameters, overriding the defaults ``learning_rate=0.1`` and
    ``depth=3``.

  Raises
  ------
  ValueError
      If ``catboost_params`` sets a parameter this backend controls.
  """

  #: A fitted tree table, noisy across levels, is smoothed as much as the other
  #: tree backends' are.
  default_smoothing: ClassVar[float] = 2.0

  model_: CatBoostRegressor | None

  def __init__(
    self,
    *,
    transform: Transform = "logit",
    quantile_table_config: QuantileTableConfig | None = None,
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    iterations: int = 100,
    thread_count: int | None = None,
    **catboost_params: Any,  # noqa: ANN401 - forwarded to CatBoostRegressor
  ) -> None:
    super().__init__(
      transform=transform,
      quantile_table_config=quantile_table_config,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )

    reserved = sorted(_RESERVED & set(catboost_params))
    if reserved:
      raise ValueError(
        f"CatBoostQuantileBackend must control {reserved} itself; got them "
        "in its parameters. Drop them: the loss and levels come from the "
        "quantile table, the schedule from `iterations`, and threading from "
        "`thread_count`."
      )

    self.iterations = iterations
    self.thread_count = thread_count
    self.catboost_params = {**_DEFAULT_PARAMS, **catboost_params}
    self.model_ = None

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    """Fit one model predicting every probability level."""
    levels = self.quantile_table_config.alphas(device="cpu").tolist()
    model = CatBoostRegressor(
      iterations=self.iterations,
      loss_function="MultiQuantile:alpha="
      + ",".join(f"{alpha:.12g}" for alpha in levels),
      thread_count=-1 if self.thread_count is None else self.thread_count,
      logging_level="Silent",
      # CatBoost otherwise writes a `catboost_info/` directory into the
      # working directory on every fit.
      allow_writing_files=False,
      **self.catboost_params,
    )
    model.fit(to_numpy(x), to_numpy(z))
    self.model_ = model

  def _predict_quantiles(
    self,
    x: torch.Tensor,
    alphas: torch.Tensor,
  ) -> torch.Tensor:
    """Predict one quantile table for a chunk of conditioning rows.

    One output column per level, so the rows lead.
    """
    assert self.model_ is not None

    del alphas
    predicted = self.model_.predict(to_numpy(x))
    return self._quantile_rows(predicted, x, levels_first=False)
