"""
LightGBM quantile conditional margin.

This backend fits one LightGBM booster with the ``quantile`` objective for
every probability level in ``config.alphas()``. LightGBM's quantile objective
renews each leaf to the matching percentile of the residuals it holds, so the
extreme levels reach the tails of the sample rather than stalling near the
initial score, and boosters trained in separate threads release the GIL, so the
levels are fitted concurrently.

PDF, CDF, inverse CDF, and grid evaluation are inherited from
:class:`~npcc.core.margin.ConditionalMargin`.

LightGBM receives NumPy-backed host arrays. Predictions are converted
immediately to torch tensors on the input device and with the input dtype.

This backend requires the ``lightgbm`` optional dependency.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, TypeVar

import lightgbm
import numpy as np
import torch
from pyvinecopulib.core.extend import to_numpy

from npcc.core._quantile_table import Smoother
from npcc.core._transform import Transform
from npcc.core.margin import ConditionalMargin, QuantileTableConfig

_S = TypeVar("_S")
_T = TypeVar("_T")

#: The defaults are shallow trees and a short, fast schedule. Depth-three trees
#: matched scikit-learn's quantile GBM on the bivariate study at a fiftieth of
#: its fit time; deeper, longer-trained trees fitted noisier tables.
_DEFAULT_PARAMS: dict[str, object] = {
  "learning_rate": 0.1,
  "max_depth": 3,
  "num_leaves": 8,
  "min_child_samples": 20,
}

#: Set by this backend, so a caller cannot pass them -- under LightGBM's own
#: names or any alias it would read in their place.
_RESERVED = frozenset(
  {
    "objective",
    "objective_type",
    "app",
    "application",
    "loss",
    "alpha",
    "num_threads",
    "num_thread",
    "nthread",
    "nthreads",
    "verbosity",
    "verbose",
  }
)


class LightGBMQuantileBackend(ConditionalMargin):
  """Conditional margin using per-level LightGBM quantile regression.

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
    Device used for returned tensors. LightGBM itself runs on the host.
  batch_size
    Maximum number of conditioning rows evaluated in one prediction call.
  n_estimators
    Number of boosting rounds for every booster.
  n_jobs
    Number of levels fitted, and predicted, at once; each booster runs on one
    thread. ``None`` uses every available core.
  **lgbm_params
    LightGBM parameters for every booster, under LightGBM's own names or the
    scikit-learn aliases it accepts (``min_child_samples``, ``subsample``,
    ``random_state``, ...). They override the defaults:
    ``learning_rate=0.1``, ``max_depth=3``, ``num_leaves=8``,
    ``min_child_samples=20``.

  Raises
  ------
  ValueError
      If ``lgbm_params`` sets a parameter this backend controls.
  """

  #: Every level is a separate model, so the predicted table is noisy across
  #: levels, most at the outermost ones, which few observations reach: the
  #: weighted smoother smooths those hard and the middle lightly.
  default_smoother: ClassVar[Smoother] = "weighted"
  default_smoothing: ClassVar[float] = 0.65

  boosters_: list[lightgbm.Booster] | None

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
    **lgbm_params: Any,  # noqa: ANN401 - forwarded to lightgbm.train
  ) -> None:
    super().__init__(
      transform=transform,
      quantile_table_config=quantile_table_config,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )

    reserved = sorted(_RESERVED & set(lgbm_params))
    if reserved:
      raise ValueError(
        f"LightGBMQuantileBackend must control {reserved} itself; got them "
        "in its parameters. Drop them: the objective and level come from the "
        "quantile table, and threading from `n_jobs`."
      )

    if n_jobs is not None and n_jobs <= 0:
      raise ValueError(f"n_jobs must be None or positive; got {n_jobs}")

    self.n_estimators = n_estimators
    self.n_jobs = n_jobs or os.cpu_count() or 1
    self.lgbm_params = {**_DEFAULT_PARAMS, **lgbm_params}
    self.boosters_ = None

  def _per_level(
    self, task: Callable[[_S], _T], items: Sequence[_S]
  ) -> list[_T]:
    """Run ``task`` once per level, ``n_jobs`` levels at a time."""
    with ThreadPoolExecutor(max_workers=min(self.n_jobs, len(items))) as pool:
      return list(pool.map(task, items))

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    """Fit one quantile booster per probability level."""
    features = to_numpy(x)
    target = to_numpy(z)
    alphas = self.quantile_table_config.alphas(device="cpu").tolist()

    def fit_level(alpha: float) -> lightgbm.Booster:
      level_params = {
        **self.lgbm_params,
        "objective": "quantile",
        "alpha": alpha,
        "num_threads": 1,
        "verbosity": -1,
      }
      return lightgbm.train(
        level_params,
        lightgbm.Dataset(features, label=target),
        num_boost_round=self.n_estimators,
      )

    self.boosters_ = self._per_level(fit_level, alphas)

  def _predict_quantiles(
    self,
    x: torch.Tensor,
    alphas: torch.Tensor,
  ) -> torch.Tensor:
    """Predict one quantile table for a chunk of conditioning rows."""
    assert self.boosters_ is not None

    del alphas

    features = to_numpy(x)
    per_level = self._per_level(
      lambda booster: np.asarray(booster.predict(features, num_threads=1)),
      self.boosters_,
    )
    return self._quantile_rows(per_level, x, levels_first=True)
