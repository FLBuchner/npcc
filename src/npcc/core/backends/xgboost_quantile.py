"""
XGBoost quantile conditional margin.

This backend fits one XGBoost model with the ``reg:quantileerror`` objective,
taking every probability level in ``config.alphas()`` as ``quantile_alpha``.
Under XGBoost's default ``one_output_per_tree`` strategy each level grows trees
of its own, so the levels are separate models that share one fit call.

XGBoost runs on the host whatever the margin's device: its CUDA path is not
supported on every device this package runs on -- it fails on a virtualized
GPU -- and these models are small. Predictions are converted immediately to
torch tensors on the input device and with the input dtype.

PDF, CDF, inverse CDF, and grid evaluation are inherited from
:class:`~npcc.core.margin.ConditionalMargin`.

This backend requires the ``xgboost`` optional dependency.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from pyvinecopulib.core.extend import to_numpy
from xgboost.sklearn import XGBRegressor

from npcc.core._quantile_table import Smoother
from npcc.core._transform import Transform
from npcc.core.margin import ConditionalMargin, QuantileTableConfig

#: The schedule the LightGBM backend uses, so the two are compared at the same
#: capacity: shallow trees, a short schedule, and a leaf floor of 20 rows.
_DEFAULT_PARAMS: dict[str, object] = {
  "learning_rate": 0.1,
  "max_depth": 3,
  "min_child_weight": 20.0,
}

#: Set by this backend, so a caller cannot pass them. ``device`` is not among
#: them because it never reaches here: it is the margin's own argument.
_RESERVED = frozenset(
  {
    "objective",
    "quantile_alpha",
    "tree_method",
    "n_estimators",
    "n_jobs",
    "nthread",
    "verbosity",
  }
)


class XGBoostQuantileBackend(ConditionalMargin):
  """Conditional margin using XGBoost multi-level quantile regression.

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
    Device used for returned tensors. XGBoost itself runs on the host.
  batch_size
    Maximum number of conditioning rows evaluated in one prediction call.
  n_estimators
    Number of boosting rounds.
  n_jobs
    Number of threads XGBoost fits and predicts with; ``None`` uses every
    available core.
  **xgb_params
    XGBoost parameters, overriding the defaults ``learning_rate=0.1``,
    ``max_depth=3`` and ``min_child_weight=20``.

  Raises
  ------
  ValueError
      If ``xgb_params`` sets a parameter this backend controls.
  """

  #: Every level grows its own trees, so the table is smoothed as the
  #: LightGBM backend's is.
  default_smoother: ClassVar[Smoother] = "weighted"
  default_smoothing: ClassVar[float] = 0.65

  model_: XGBRegressor | None

  def __init__(
    self,
    *,
    transform: Transform = "logit",
    quantile_table_config: QuantileTableConfig | None = None,
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    n_estimators: int = 100,
    n_jobs: int | None = None,
    **xgb_params: Any,  # noqa: ANN401 - forwarded to XGBRegressor
  ) -> None:
    super().__init__(
      transform=transform,
      quantile_table_config=quantile_table_config,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )

    reserved = sorted(_RESERVED & set(xgb_params))
    if reserved:
      raise ValueError(
        f"XGBoostQuantileBackend must control {reserved} itself; got them in "
        "its parameters. Drop them: the objective and levels come from the "
        "quantile table, the schedule from `n_estimators`, and threading "
        "from `n_jobs`."
      )

    self.n_estimators = n_estimators
    self.n_jobs = n_jobs
    self.xgb_params = {**_DEFAULT_PARAMS, **xgb_params}
    self.model_ = None

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    """Fit one model predicting every probability level."""
    model = XGBRegressor(
      objective="reg:quantileerror",
      quantile_alpha=self.quantile_table_config.alphas(device="cpu").tolist(),
      tree_method="hist",
      device="cpu",
      n_estimators=self.n_estimators,
      n_jobs=self.n_jobs,
      **self.xgb_params,
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
