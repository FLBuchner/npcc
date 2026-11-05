"""NPCC-side helpers for experiment E3: the estimators and one timed fit.

Imported by ``e3a_npcc_fit.py`` and ``e3b_npcc_fit.py``, in the NPCC
environment only (``uv run --no-sync python ...`` from ``/home/thibault/npcc``).

**Estimators.** The logit arms of E1 (``configs/study.toml``), with E1's
labels, so E3 rows join E1 on ``estimator``. Each is an
:class:`npcc.experiments.config.EstimatorSpec`, the object E1 builds its
controls from, so the backend, transform, ``backend_kwargs`` and per-fit
torch seed (``derive_seed(<salt>, estimator_id)``) are built the same way.
No backend hyperparameter is changed from E1's defaults.

**Timing.** ``fit_s`` is the construction of :class:`RosenblattBicop` and its
``fit`` (both Rosenblatt directions); for TabPFN that includes building the
regressor from its checkpoint, which TabPFN reads per instance. A warm-up
fit is made first and is not timed, as VDC's protocol excludes loading and
warm-up. ``inference_s`` is ``pdf`` at every test point. CUDA is
synchronized before each clock is read.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from dotenv import load_dotenv

from npcc import FitControlsRosenblattBicop, RosenblattBicop
from npcc.experiments._study import derive_seed
from npcc.experiments.config import EstimatorSpec

# TabPFN reads TABPFN_TOKEN from the environment; an exported token wins.
load_dotenv("/home/thibault/npcc/.env", override=False)

#: label -> (backend, model_version or None, device class)
ESTIMATORS: dict[str, tuple[str, str | None, str]] = {
  "tabpfn-quantiles-v2.5-logit": ("tabpfn-quantiles", "v2.5", "gpu"),
  "tabpfn-quantiles-v3-logit": ("tabpfn-quantiles", "v3", "gpu"),
  "tabpfn-quantiles-v3.5-logit": ("tabpfn-quantiles", "v3.5", "gpu"),
  "tabicl-logit": ("tabicl", None, "gpu"),
  "nori-logit": ("nori", None, "gpu"),
  "lightgbm-logit": ("lightgbm", None, "cpu"),
  "xgboost-logit": ("xgboost", None, "cpu"),
  "catboost-logit": ("catboost", None, "cpu"),
}
CPU_ESTIMATORS = [k for k, v in ESTIMATORS.items() if v[2] == "cpu"]
GPU_ESTIMATORS = [k for k, v in ESTIMATORS.items() if v[2] == "gpu"]

#: E1's projection grid size; unused, since the Sinkhorn projection is off.
PROJECTION_GRID_SIZE = 30


def spec(label: str) -> EstimatorSpec:
  """The E1 estimator called ``label``."""
  backend, version, _ = ESTIMATORS[label]
  kwargs = {} if version is None else {"model_version": version}
  return EstimatorSpec(
    label=label, backend=backend, transform="logit", backend_kwargs=kwargs
  )


@dataclass
class FitResult:
  """What one timed fit and evaluation returns."""

  log_pdf: np.ndarray
  fit_s: float
  inference_s: float
  n_nan: int


def _sync(device: str) -> None:
  if device.startswith("cuda"):
    torch.cuda.synchronize()


def fit_and_score(
  est: EstimatorSpec,
  train: np.ndarray,
  test: np.ndarray,
  *,
  device: str,
  salt: tuple,
  x_train: np.ndarray | None = None,
  x_test: np.ndarray | None = None,
  log_floor: float = 1e-300,
) -> FitResult:
  """Fit ``est`` on ``train`` and return its floored log-density at ``test``.

  Parameters
  ----------
  est : EstimatorSpec
      The estimator.
  train, test : ndarray, shape (n, 2)
      Pseudo-observations.
  device : str
      ``"cpu"`` or ``"cuda"``.
  salt : tuple
      Hashed with the estimator id into the torch seed, as E1 does with its
      cell seed.
  x_train, x_test : ndarray, shape (n, p), optional
      Covariates; ``None`` fits the pair unconditionally.
  log_floor : float
      Density floor before the log, VDC's runner's 1e-300 by default.

  Returns
  -------
  FitResult
      Log-density at the test points, fit and inference seconds, and the
      number of non-finite densities (counted before flooring).
  """
  torch.manual_seed(derive_seed(*salt, est.estimator_id))
  as_t = lambda a: None if a is None else torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
  _sync(device)
  t0 = time.perf_counter()
  model = RosenblattBicop(
    est.controls(
      FitControlsRosenblattBicop,
      device=device,
      projection_grid_size=PROJECTION_GRID_SIZE,
      batch_size=None,
    )
  )
  model.fit(as_t(train), x=as_t(x_train))
  _sync(device)
  fit_s = time.perf_counter() - t0
  with torch.inference_mode():
    t0 = time.perf_counter()
    density = model.pdf(as_t(test), x=as_t(x_test))
    _sync(device)
    inference_s = time.perf_counter() - t0
  c = density.detach().cpu().numpy().astype(np.float64)
  n_nan = int(np.sum(~np.isfinite(c)))
  del model
  if device.startswith("cuda"):
    torch.cuda.empty_cache()
  return FitResult(
    log_pdf=np.log(np.maximum(c, log_floor)),
    fit_s=fit_s,
    inference_s=inference_s,
    n_nan=n_nan,
  )
