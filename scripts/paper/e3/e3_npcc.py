"""NPCC-side helpers for experiment E3: the estimators and one timed fit.

Imported by ``e3a_npcc_fit.py`` and ``e3b_npcc_fit.py``, in the NPCC
environment only (``uv run --no-sync python ...`` from ``/home/thibault/npcc``).

**Estimators.** E1's estimators (``configs/study.toml``), all at the probit
transform, with E1's labels, so E3 rows join E1 on ``estimator``. Each is an
:class:`npcc.experiments.config.EstimatorSpec`, the object E1 builds its
controls from, so the backend, transform, ``backend_kwargs`` and per-fit
torch seed (``derive_seed(<salt>, estimator_id)``) are built the same way:
:func:`spec` reads them from E1's config, which is where TabPFN's release and
softmax temperature are set. No backend hyperparameter is changed from E1's.

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
from functools import cache
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

from npcc import FitControlsRosenblattBicop, RosenblattBicop
from npcc.experiments._study import derive_seed
from npcc.experiments.config import EstimatorSpec, load_grid

# TabPFN reads TABPFN_TOKEN from the environment; an exported token wins.
load_dotenv("/home/thibault/npcc/.env", override=False)

#: The transform of every estimator, as in E1: the paper presents probit only.
TRANSFORM = "probit"

#: label -> (backend, model_version or None, device class)
ESTIMATORS: dict[str, tuple[str, str | None, str]] = {
  "tabpfn-quantiles-v2.5-probit": ("tabpfn-quantiles", "v2.5", "gpu"),
  "tabpfn-quantiles-v3-probit": ("tabpfn-quantiles", "v3", "gpu"),
  "tabpfn-quantiles-v3.5-probit": ("tabpfn-quantiles", "v3.5", "gpu"),
  "tabicl-probit": ("tabicl", None, "gpu"),
  "nori-probit": ("nori", None, "gpu"),
  "lightgbm-probit": ("lightgbm", None, "cpu"),
  "xgboost-probit": ("xgboost", None, "cpu"),
  "catboost-probit": ("catboost", None, "cpu"),
}
CPU_ESTIMATORS = [k for k, v in ESTIMATORS.items() if v[2] == "cpu"]
GPU_ESTIMATORS = [k for k, v in ESTIMATORS.items() if v[2] == "gpu"]

#: The estimator of E3c's ``rosenblatt-vinecop-check``: fitted in
#: ``npcc.RosenblattVinecop``, it must equal the shared vine's
#: ``npcc-<estimator>`` arm.
VINECOP_CHECK = "catboost-probit"

#: E1's projection grid size; unused, since the Sinkhorn projection is off.
PROJECTION_GRID_SIZE = 30


#: E1's config, the source of every estimator's settings here.
E1_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "study.toml"


@cache
def _e1_estimators() -> dict[str, EstimatorSpec]:
  return {e.label: e for e in load_grid(E1_CONFIG).estimators}


def spec(label: str) -> EstimatorSpec:
  """The E1 estimator called ``label``, as E1's config builds it."""
  backend, version, _ = ESTIMATORS[label]
  est = _e1_estimators().get(label)
  if est is None:
    raise SystemExit(f"{label} must be an estimator of {E1_CONFIG}; it is not")
  stated = (backend, TRANSFORM, version)
  found = (est.backend, est.transform, est.backend_kwargs.get("model_version"))
  if found != stated:
    raise SystemExit(
      f"{label} must be (backend, transform, model_version) {stated} in "
      f"{E1_CONFIG}, as ESTIMATORS states; got {found}"
    )
  return est


@dataclass
class FitResult:
  """What one timed fit and evaluation returns."""

  log_pdf: np.ndarray
  fit_s: float
  inference_s: float
  n_nan: int
  #: ``pdf_grid`` on the requested midpoints, ``(m, m)``, ``u`` major.
  grid_pdf: np.ndarray | None = None
  grid_s: float | None = None
  #: Pointwise ``pdf`` at the spot-check points.
  spot_pdf: np.ndarray | None = None


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
  grid: np.ndarray | None = None,
  spot: np.ndarray | None = None,
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
  grid : ndarray, shape (m,), optional
      Midpoints at which to evaluate ``pdf_grid`` on ``grid x grid``
      (unconditional fits only), timed separately as ``grid_s``.
  spot : ndarray, shape (k, 2), optional
      Points at which to evaluate pointwise ``pdf`` too, to check the grid
      path against it.

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
    grid_pdf = spot_pdf = grid_s = None
    if grid is not None:
      if x_train is not None:
        raise ValueError("grid must be None for a conditional fit; pdf_grid takes one x row")
      t0 = time.perf_counter()
      g_t = as_t(grid)
      grid_pdf = model.pdf_grid(g_t, g_t).detach().cpu().numpy().astype(np.float64)
      _sync(device)
      grid_s = time.perf_counter() - t0
    if spot is not None:
      spot_pdf = model.pdf(as_t(spot)).detach().cpu().numpy().astype(np.float64)
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
    grid_pdf=grid_pdf,
    grid_s=grid_s,
    spot_pdf=spot_pdf,
  )
