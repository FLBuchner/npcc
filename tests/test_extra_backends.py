"""Smoke tests for the backends that need an extra, a GPU or a download.

Each test ``importorskip``s its backend dependency, so the suite still
passes in environments where the extra is not installed.  LightGBM and NGBoost
fit locally and quickly; TabICL and Nori are foundation models whose first fit
downloads weights, so they are additionally guarded against download/runtime
failures. The TabPFN fine-tune smoke is opt-in: it is a GPU training loop.
"""

from __future__ import annotations

import os
from typing import cast

import pytest
import torch

from npcc.core import (
  FitControlsRosenblattBicop,
  QuantileTableConfig,
  RosenblattBicop,
)


def _gaussian_copula_sample(
  n: int, rho: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
  """Draw ``n`` (u, v) pairs from a Gaussian copula with correlation rho."""
  generator = torch.Generator().manual_seed(seed)
  z1 = torch.randn(n, generator=generator)
  z2 = rho * z1 + (1.0 - rho**2) ** 0.5 * torch.randn(n, generator=generator)
  standard_normal = torch.distributions.Normal(0.0, 1.0)
  u = standard_normal.cdf(z1)
  v = standard_normal.cdf(z2)
  return (
    cast("torch.Tensor", u.clamp(1e-3, 1 - 1e-3)),
    cast("torch.Tensor", v.clamp(1e-3, 1 - 1e-3)),
  )


def _check_fitted_model(m: RosenblattBicop) -> None:
  pts = torch.tensor([0.3, 0.5, 0.7])
  pdf = m.pdf(torch.column_stack([pts, pts]))
  assert torch.all(torch.isfinite(pdf)) and torch.all(pdf >= 0.0)

  h = m.hfunc1(torch.column_stack([pts, pts]))
  assert torch.all((h >= 0.0) & (h <= 1.0))

  grid = m.pdf_grid(torch.linspace(0.2, 0.8, 4), torch.linspace(0.2, 0.8, 5))
  assert grid.shape == (4, 5)
  assert torch.all(grid >= 0.0)

  tau = m.tau(n=200)
  assert isinstance(tau, float)
  assert -1.0 <= tau <= 1.0


def test_lightgbm_backend_smoke() -> None:
  pytest.importorskip("lightgbm")
  u, v = _gaussian_copula_sample(200, rho=0.6, seed=0)
  m = RosenblattBicop(
    FitControlsRosenblattBicop(
      backend="lightgbm",
      device="cpu",
      backend_kwargs={"n_estimators": 20, "random_state": 0},
    )
  ).fit(torch.column_stack([u, v]))
  _check_fitted_model(m)


def test_lightgbm_fits_one_booster_per_level() -> None:
  """Every probability level the table reads has its own booster.

  The table is read at ``config.alphas()``; a booster count that drifted from
  it would hand the reconstruction a table of the wrong width.
  """
  pytest.importorskip("lightgbm")
  from npcc.core.backends.lightgbm_quantile import LightGBMQuantileBackend

  config = QuantileTableConfig(n_quantiles=9)
  margin = LightGBMQuantileBackend(
    quantile_table_config=config, device="cpu", n_estimators=5
  )
  u, v = _gaussian_copula_sample(120, rho=0.5, seed=1)
  margin.fit(v.double(), x=u.double().reshape(-1, 1))

  assert margin.boosters_ is not None
  assert len(margin.boosters_) == config.n_quantiles
  table = margin.cdf(
    torch.tensor([0.2, 0.8], dtype=torch.float64),
    x=torch.tensor([[0.5], [0.5]], dtype=torch.float64),
  )
  assert table[0] < table[1]


def test_lightgbm_refuses_a_parameter_it_controls() -> None:
  """A caller's ``objective`` or ``alpha`` would silently replace the level's.

  Every booster's objective and level come from the quantile table, so letting
  either through would fit every level to the same quantile.
  """
  pytest.importorskip("lightgbm")
  from npcc.core.backends.lightgbm_quantile import LightGBMQuantileBackend

  with pytest.raises(ValueError, match=r"must control \['alpha'\]"):
    LightGBMQuantileBackend(alpha=0.5)


def test_xgboost_backend_smoke() -> None:
  pytest.importorskip("xgboost")
  u, v = _gaussian_copula_sample(200, rho=0.6, seed=0)
  m = RosenblattBicop(
    FitControlsRosenblattBicop(
      backend="xgboost",
      device="cpu",
      backend_kwargs={"n_estimators": 20, "random_state": 0},
    )
  ).fit(torch.column_stack([u, v]))
  _check_fitted_model(m)


def test_xgboost_predicts_one_column_per_level() -> None:
  """The model predicts the table at every level the reconstruction reads.

  XGBoost takes the levels as ``quantile_alpha`` and answers one column per
  level; a width that drifted from ``config.alphas()`` would be read at the
  wrong normal scores.
  """
  pytest.importorskip("xgboost")
  from npcc.core.backends.xgboost_quantile import XGBoostQuantileBackend

  config = QuantileTableConfig(n_quantiles=9)
  margin = XGBoostQuantileBackend(
    quantile_table_config=config, device="cpu", n_estimators=5
  )
  u, v = _gaussian_copula_sample(120, rho=0.5, seed=1)
  x = u.double().reshape(-1, 1)
  margin.fit(v.double(), x=x)

  table = margin._predict_quantiles(x[:3], config.alphas())
  assert table.shape == (3, config.n_quantiles)
  cdf = margin.cdf(
    torch.tensor([0.2, 0.8], dtype=torch.float64),
    x=torch.tensor([[0.5], [0.5]], dtype=torch.float64),
  )
  assert cdf[0] < cdf[1]


def test_xgboost_refuses_a_parameter_it_controls() -> None:
  """A caller's ``quantile_alpha`` would silently replace the table's levels.

  The reconstruction reads the predictions at ``config.alphas()``, so a model
  fitted at other levels would be read at the wrong normal scores.
  """
  pytest.importorskip("xgboost")
  from npcc.core.backends.xgboost_quantile import XGBoostQuantileBackend

  with pytest.raises(ValueError, match=r"must control \['quantile_alpha'\]"):
    XGBoostQuantileBackend(quantile_alpha=[0.5])


def test_ngboost_backend_smoke() -> None:
  pytest.importorskip("ngboost")
  u, v = _gaussian_copula_sample(80, rho=0.6, seed=1)
  m = RosenblattBicop(
    FitControlsRosenblattBicop(
      backend="ngboost",
      backend_kwargs={"n_estimators": 40},
    )
  ).fit(torch.column_stack([u, v]))
  _check_fitted_model(m)


def test_pytabkit_realmlp_backend_smoke() -> None:
  pytest.importorskip("pytabkit")
  u, v = _gaussian_copula_sample(80, rho=0.6, seed=5)
  m = RosenblattBicop(
    FitControlsRosenblattBicop(
      backend="pytabkit-realmlp",
      backend_kwargs={"n_epochs": 8},
      quantile_table_config=QuantileTableConfig(n_quantiles=21),
    )
  ).fit(torch.column_stack([u, v]))
  _check_fitted_model(m)


def test_nori_backend_smoke() -> None:
  pytest.importorskip("synthefy_nori")
  u, v = _gaussian_copula_sample(80, rho=0.6, seed=7)
  try:
    m = RosenblattBicop(
      FitControlsRosenblattBicop(
        backend="nori",
        quantile_table_config=QuantileTableConfig(n_quantiles=41),
      )
    ).fit(torch.column_stack([u, v]))
  except Exception as exc:  # noqa: BLE001 - weight download / runtime issues
    pytest.skip(f"Nori unavailable at runtime: {exc}")
  _check_fitted_model(m)


def test_tabpfn_finetune_backend_smoke() -> None:
  # Opt-in: fine-tuning is a GPU training loop (minutes), too slow for the
  # default suite. Enable with NPCC_RUN_FINETUNE=1.
  if not os.environ.get("NPCC_RUN_FINETUNE"):
    pytest.skip("set NPCC_RUN_FINETUNE=1 to run the fine-tune smoke")
  torch = pytest.importorskip("torch")
  if not torch.cuda.is_available():
    pytest.skip("fine-tuning needs a GPU")
  if not os.environ.get("TABPFN_TOKEN"):
    pytest.skip("no TABPFN_TOKEN")
  u, v = _gaussian_copula_sample(120, rho=0.6, seed=6)
  try:
    m = RosenblattBicop(
      FitControlsRosenblattBicop(
        backend="tabpfn-finetune",
        backend_kwargs={"epochs": 1, "early_stopping": False},
      )
    ).fit(torch.column_stack([u, v]))
    pdf = m.pdf(torch.tensor([[0.5, 0.5]]))
  except Exception as exc:  # noqa: BLE001 - license/download/runtime issues
    pytest.skip(f"TabPFN fine-tune unavailable at runtime: {exc}")
  assert torch.all(torch.isfinite(pdf)) and torch.all(pdf >= 0.0)


def test_tabicl_backend_smoke() -> None:
  pytest.importorskip("tabicl")
  u, v = _gaussian_copula_sample(80, rho=0.6, seed=2)
  try:
    m = RosenblattBicop(
      FitControlsRosenblattBicop(
        backend="tabicl",
        backend_kwargs={"model_kwargs": {"n_estimators": 1}},
      )
    ).fit(torch.column_stack([u, v]))
    pdf = m.pdf(torch.tensor([[0.3, 0.4], [0.5, 0.6]]))
  except Exception as exc:  # noqa: BLE001 - weight download / runtime issues
    pytest.skip(f"TabICL unavailable at runtime: {exc}")
  assert torch.all(torch.isfinite(pdf)) and torch.all(pdf >= 0.0)
