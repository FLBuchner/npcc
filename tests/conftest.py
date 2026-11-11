"""
Shared test fixtures and fakes for npcc tests.

Skips the entire suite when ``tabpfn`` cannot be imported (the package
is a hard runtime dependency, but importing it needs a working auth
setup, so the test layer absorbs its absence).

Every fake has one ground truth, ``Z ~ Normal(0, 1)`` on the modeled scale,
whose quantile function ``Q(alpha) = Phi^{-1}(alpha)`` is what it predicts. The
quantile table interpolates linearly in ``Phi^{-1}(alpha)``, so it reproduces
that distribution exactly, whatever its smoothing:

- :class:`FakeTabPFNRegressor` mimics ``TabPFNRegressor``'s
  ``output_type="quantiles"``, one array per level;
- :class:`NormalQuantileBackend` is a TabPFN-free backend registered as
  ``"normal-quantile"``, which proves the registry and the vine cascade
  without any third-party model.

A run that does not select the ``cuda`` tests sees no CUDA device
(:func:`pytest_configure`), so every other test runs on the CPU wherever it
runs.
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Iterator
from typing import cast

import numpy as np
import pytest
import torch

# pytest's own `-m` parser: a run selects a test exactly when it does.
from _pytest.mark.expression import Expression

pytest.importorskip("tabpfn")

from npcc.core import ConditionalMargin, QuantileTableConfig, registry
from npcc.core.margin import MarginOptions

#: The single symbol the TabPFN backends build their regressor through.
TABPFN_TARGET = "npcc.core.backends.tabpfn_quantile.TabPFNRegressor"

#: Mass beyond each end of the default quantile table.
TAIL_MASS = 0.5 * math.erfc(QuantileTableConfig().normal_limit / math.sqrt(2.0))


def selects_cuda(markexpr: str) -> bool:
  """Whether ``-m markexpr`` selects a test whose one mark is ``cuda``.

  An empty expression selects every test, that one included.
  """
  if not markexpr.strip():
    return True
  return Expression.compile(markexpr).evaluate(lambda name, **_: name == "cuda")


def pytest_configure(config: pytest.Config) -> None:
  """Hide every CUDA device from a run that does not select the ``cuda`` tests.

  No other test names a device, and ``resolve_device(None)`` picks ``cuda``
  wherever one exists, so each xdist worker of ``make test`` would otherwise
  open a CUDA context of its own and place its tensors on the GPU. That is
  not the CPU suite CI runs, and a GPU that cannot hold several contexts at
  once fails those tests at random. ``make test-cuda``'s ``-m cuda`` keeps
  the devices. Nothing imported above queries CUDA, so hiding it here, before
  collection, still takes effect.
  """
  if not selects_cuda(config.getoption("markexpr")):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def normal_quantiles(levels: torch.Tensor) -> torch.Tensor:
  """``Phi^{-1}`` of ``levels``, the ground truth every fake predicts."""
  return cast("torch.Tensor", torch.special.ndtri(levels))


class FakeTabPFNRegressor:
  """Mimics ``TabPFNRegressor``'s quantile output for ``Z ~ Normal(0, 1)``.

  Like the real regressor, it returns one array of ``n_samples`` per level.
  Unlike it, it names no checkpoint unless given one, so the adapter has none
  to load.
  """

  def __init__(self, *, model_path: object = None, **_: object) -> None:
    self.fitted_: bool = False
    self.model_path = model_path

  @classmethod
  def create_default_for_version(
    cls, version: object, **overrides: object
  ) -> FakeTabPFNRegressor:
    return cls(**overrides)

  def fit(self, X: np.ndarray, y: np.ndarray) -> FakeTabPFNRegressor:
    self.fitted_ = True
    return self

  def predict(
    self,
    X: np.ndarray,
    *,
    output_type: str = "mean",
    quantiles: list[float] | None = None,
  ) -> object:
    if output_type != "quantiles":
      raise ValueError(f"Unsupported output_type: {output_type}")
    if quantiles is None:
      raise ValueError("quantiles is required for output_type='quantiles'.")
    levels = normal_quantiles(torch.as_tensor(quantiles, dtype=torch.float64))
    return [np.full(X.shape[0], q) for q in levels.tolist()]


class BadShapeQuantileRegressor(FakeTabPFNRegressor):
  """Returns a 3-D quantile array to trigger the unexpected-shape error."""

  def predict(
    self,
    X: np.ndarray,
    *,
    output_type: str = "mean",
    quantiles: list[float] | None = None,
  ) -> object:
    del X, output_type, quantiles
    return np.zeros((2, 3, 4))


@pytest.fixture
def patch_tabpfn(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(TABPFN_TARGET, FakeTabPFNRegressor)


@pytest.fixture
def patch_bad_shape(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(TABPFN_TARGET, BadShapeQuantileRegressor)


def normal_density_y(y: torch.Tensor) -> torch.Tensor:
  """Analytic ``f_Y(y)`` under logit transform with Z ~ Normal(0, 1)."""
  z = torch.logit(y)
  return torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) / (y * (1.0 - y))


class NormalQuantileBackend(ConditionalMargin):
  """Closed-form quantile backend for Z ~ Normal(0, 1); no external model."""

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    return None

  def _predict_quantiles(
    self, x: torch.Tensor, alphas: torch.Tensor
  ) -> torch.Tensor:
    row = normal_quantiles(alphas)
    return row.unsqueeze(0).expand(x.shape[0], -1).clone()


class LinearGaussianBackend(ConditionalMargin):
  """Gaussian linear regression of the modeled response on every feature.

  Unlike :class:`NormalQuantileBackend`, whose pair copulas are all the
  independence copula, its answers follow the data and the conditioning rows,
  so vines on different structures differ, and a value read from the wrong
  rows shows.
  """

  _beta: torch.Tensor
  _sd: torch.Tensor

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    design = torch.cat([torch.ones_like(z).unsqueeze(1), x], dim=1)
    beta = torch.linalg.lstsq(design, z.unsqueeze(1)).solution.squeeze(1)
    self._beta = beta
    self._sd = (z - design @ beta).std()

  def _predict_quantiles(
    self, x: torch.Tensor, alphas: torch.Tensor
  ) -> torch.Tensor:
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
    mean = torch.cat([ones, x], dim=1) @ self._beta
    return mean.unsqueeze(1) + self._sd * normal_quantiles(alphas)


def normal_quantile_factory(**kwargs: object) -> NormalQuantileBackend:
  """A registry factory for :class:`NormalQuantileBackend`.

  It takes the construction arguments every factory is handed and ignores any
  backend-specific ones, so it stands in for a backend of any configuration.
  """
  margin = {
    k: v for k, v in kwargs.items() if k in MarginOptions.__annotations__
  }
  return NormalQuantileBackend(**cast("MarginOptions", margin))


@contextlib.contextmanager
def hermetic_backend(name: str = "normal-quantile") -> Iterator[None]:
  """Register :func:`normal_quantile_factory` under ``name`` for a block.

  A context manager rather than only a fixture, so a module-scoped fixture can
  hold the registration for its whole lifetime.
  """
  registry.register_backend(name, normal_quantile_factory)
  try:
    yield
  finally:
    registry._REGISTRY.pop(name, None)


@pytest.fixture
def register_hermetic_backends() -> Iterator[None]:
  """Register the hermetic ``normal-quantile`` backend."""
  with hermetic_backend():
    yield


@pytest.fixture
def register_linear_gaussian_backend() -> Iterator[None]:
  """Register :class:`LinearGaussianBackend` as ``"linear-gaussian"``."""

  def factory(**kwargs: object) -> LinearGaussianBackend:
    margin = {
      k: v for k, v in kwargs.items() if k in MarginOptions.__annotations__
    }
    return LinearGaussianBackend(**cast("MarginOptions", margin))

  registry.register_backend("linear-gaussian", factory)
  try:
    yield
  finally:
    registry._REGISTRY.pop("linear-gaussian", None)


@pytest.fixture
def pair_fit_widths(monkeypatch: pytest.MonkeyPatch) -> list[int | None]:
  """The covariate width every pair-copula fit receives, in fit order.

  ``None`` for a fit handed no covariates. The vine's edges are fitted through
  ``RosenblattBicop.fit``, so recording there sees what each edge conditions
  on, whichever level -- vine or distribution -- started the fit.
  """
  from pyvinecopulib.core import ControlsLike

  from npcc.core import RosenblattBicop

  widths: list[int | None] = []
  original_fit = RosenblattBicop.fit

  def record_fit(
    self: RosenblattBicop,
    uv: torch.Tensor,
    /,
    controls: ControlsLike | None = None,
    *,
    var_types: list[str] | None = None,
    x: torch.Tensor | None = None,
  ) -> RosenblattBicop:
    widths.append(None if x is None else int(x.shape[1]))
    return original_fit(self, uv, controls, var_types=var_types, x=x)

  monkeypatch.setattr(RosenblattBicop, "fit", record_fit)
  return widths
