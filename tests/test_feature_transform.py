"""The conditioning score a pair copula hands its backend, and its transform."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Iterator
from typing import Unpack, cast

import pytest
import torch

from npcc import FitControlsRosenblattBicop, RosenblattBicop
from npcc.core import registry
from npcc.core.margin import MarginOptions
from tests.conftest import NormalQuantileBackend

SPY = "feature-spy"
EPS = 1e-6


class FeatureSpy(NormalQuantileBackend):
  """The hermetic Normal backend, recording every feature matrix it meets."""

  def __init__(self, **margin: Unpack[MarginOptions]) -> None:
    super().__init__(**margin)
    self.fitted_x: torch.Tensor | None = None
    self.predicted_x: list[torch.Tensor] = []

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    self.fitted_x = x.clone()

  def _predict_quantiles(
    self, x: torch.Tensor, alphas: torch.Tensor
  ) -> torch.Tensor:
    self.predicted_x.append(x.clone())
    return super()._predict_quantiles(x, alphas)


def _spy_factory(**kwargs: object) -> FeatureSpy:
  margin = {
    k: v for k, v in kwargs.items() if k in MarginOptions.__annotations__
  }
  return FeatureSpy(**cast("MarginOptions", margin))


@pytest.fixture
def spy_backend() -> Iterator[None]:
  registry.register_backend(SPY, _spy_factory)
  try:
    yield
  finally:
    registry._REGISTRY.pop(SPY, None)


def _controls(
  feature_transform: str = "identity", backend: str = SPY
) -> FitControlsRosenblattBicop:
  return FitControlsRosenblattBicop(
    backend=backend,
    transform="probit",
    feature_transform=feature_transform,  # ty: ignore[invalid-argument-type] - a str from the parametrization
    eps=EPS,
    device="cpu",
    projection_grid_size=11,
  )


def _uv(n: int = 40, *, seed: int = 3) -> torch.Tensor:
  generator = torch.Generator().manual_seed(seed)
  uv = torch.rand((n, 2), generator=generator, dtype=torch.float64)
  # One value inside the eps band at each end, where the clamp acts.
  uv[0, 0], uv[1, 1] = 1e-9, 1.0 - 1e-9
  return uv


def _x(n: int = 40) -> torch.Tensor:
  return torch.linspace(-1.0, 2.0, n, dtype=torch.float64).reshape(-1, 1)


def _spies(model: RosenblattBicop) -> tuple[FeatureSpy, FeatureSpy]:
  v_given_u, u_given_v = model.v_given_ux_, model.u_given_vx_
  assert isinstance(v_given_u, FeatureSpy)
  assert isinstance(u_given_v, FeatureSpy)
  return v_given_u, u_given_v


def test_default_is_the_identity() -> None:
  """A default that transformed the score would move every fit already run."""
  assert FitControlsRosenblattBicop().feature_transform == "identity"
  assert FitControlsRosenblattBicop().to_dict()["feature_transform"] == (
    "identity"
  )


def test_unknown_feature_transform_is_refused() -> None:
  """A misspelled transform must not fall back to the identity unnoticed."""
  with pytest.raises(ValueError, match="feature_transform must be one of"):
    FitControlsRosenblattBicop(feature_transform="qnorm")  # ty: ignore[invalid-argument-type] - an invalid name on purpose


def test_identity_hands_the_backend_the_score_itself(
  spy_backend: None,
) -> None:
  """Off, the features are ``[u | x]`` bit for bit: nothing already run moves.

  Every fit and evaluation before the setting existed handed the backend the
  clamped score unchanged; the identity must still do exactly that, at the
  fit and at an evaluation, in both directions.
  """
  del spy_backend
  uv, x = _uv(), _x()
  model = RosenblattBicop(_controls()).fit(uv, x=x)
  clamped = uv.clamp(EPS, 1.0 - EPS)
  v_given_u, u_given_v = _spies(model)

  assert v_given_u.fitted_x is not None and u_given_v.fitted_x is not None
  assert torch.equal(v_given_u.fitted_x, torch.column_stack([clamped[:, 0], x]))
  assert torch.equal(u_given_v.fitted_x, torch.column_stack([clamped[:, 1], x]))

  model.pdf(uv, x=x)
  seen = torch.cat(v_given_u.predicted_x)
  assert torch.equal(
    torch.unique(seen, dim=0),
    torch.unique(torch.column_stack([clamped[:, 0], x]), dim=0),
  )


def test_probit_hands_the_backend_qnorm_of_the_clamped_score(
  spy_backend: None,
) -> None:
  """On, the backend sees ``Phi^{-1}(u)`` and ``x`` untouched, everywhere.

  The fit and every evaluation must agree on the feature scale -- a model
  fitted on ``qnorm(u)`` and queried at ``u`` would answer for the wrong
  point -- and the covariates must never pass through the map.
  """
  del spy_backend
  uv, x = _uv(), _x()
  model = RosenblattBicop(_controls("probit")).fit(uv, x=x)
  qnorm = torch.special.ndtri(uv.clamp(EPS, 1.0 - EPS))
  v_given_u, u_given_v = _spies(model)

  assert v_given_u.fitted_x is not None and u_given_v.fitted_x is not None
  torch.testing.assert_close(
    v_given_u.fitted_x, torch.column_stack([qnorm[:, 0], x])
  )
  torch.testing.assert_close(
    u_given_v.fitted_x, torch.column_stack([qnorm[:, 1], x])
  )
  # The clamp held the boundary values to a finite score.
  assert torch.isfinite(v_given_u.fitted_x).all()

  for evaluate in (model.pdf, model.hfunc1, model.hinv1):
    v_given_u.predicted_x.clear()
    evaluate(uv, x=x)
    seen = torch.cat(v_given_u.predicted_x)
    torch.testing.assert_close(
      torch.unique(seen, dim=0),
      torch.unique(torch.column_stack([qnorm[:, 0], x]), dim=0),
    )


def test_probit_reaches_the_grid_and_integration_paths(
  spy_backend: None,
) -> None:
  """The paths that build their own score grids transform them too.

  ``pdf_grid``, the per-row ``cdf`` and ``tau`` each lay out conditioning
  scores of their own; a path left on ``u`` would query a backend fitted on
  ``qnorm(u)`` far outside its features.
  """
  del spy_backend
  uv = _uv()
  model = RosenblattBicop(_controls("probit")).fit(uv)
  v_given_u, _ = _spies(model)
  grid = torch.linspace(0.05, 0.95, 7, dtype=torch.float64)
  for evaluate in (
    lambda: model.pdf_grid(grid, grid),
    lambda: model.cdf(uv),
    lambda: model.tau(n=50),
  ):
    v_given_u.predicted_x.clear()
    evaluate()
    seen = torch.cat(v_given_u.predicted_x)
    # Every score the backend saw is a normal score of a point in (0, 1):
    # outside [ndtri(eps), ndtri(1 - eps)] only if it skipped the map.
    bound = float(
      torch.special.ndtri(torch.tensor(1.0 - EPS, dtype=torch.float64))
    )
    assert seen.abs().max() <= bound + 1e-9
    assert (seen < 0.0).any()


def test_off_and_on_agree_where_the_backend_ignores_its_features(
  register_hermetic_backends: None,
) -> None:
  """The map changes what the backend sees, not the copula's change of scale.

  The Normal backend ignores its features, so both settings must give the
  same density: a map that leaked into the response or its Jacobian would
  change it.
  """
  del register_hermetic_backends
  uv, x = _uv(), _x()
  off = RosenblattBicop(_controls(backend="normal-quantile")).fit(uv, x=x)
  on = RosenblattBicop(_controls("probit", "normal-quantile")).fit(uv, x=x)

  torch.testing.assert_close(on.pdf(uv, x=x), off.pdf(uv, x=x))


def test_setting_survives_a_copy_and_a_pickle(
  register_hermetic_backends: None,
) -> None:
  """A copied or unpickled pair must query on the scale it was fitted on."""
  del register_hermetic_backends
  uv = _uv()
  model = RosenblattBicop(_controls("probit", "normal-quantile")).fit(uv)
  for twin in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
    assert twin.feature_transform == "probit"
    torch.testing.assert_close(twin.pdf(uv), model.pdf(uv))


def test_foreign_controls_carrying_the_setting_are_honored(
  spy_backend: None,
) -> None:
  """A ``ControlsLike`` from elsewhere must not lose the setting on the way.

  ``_bicop_controls`` rebuilds such controls from ``to_dict``; a key it did
  not know would be refused, and one it dropped would fit on ``u``.
  """
  del spy_backend

  class Foreign:
    def to_dict(self) -> dict[str, object]:
      return _controls("probit").to_dict()

  uv = _uv()
  model = RosenblattBicop(Foreign()).fit(uv)

  assert model.feature_transform == "probit"
  v_given_u, _ = _spies(model)
  assert v_given_u.fitted_x is not None
  torch.testing.assert_close(
    v_given_u.fitted_x[:, 0],
    torch.special.ndtri(uv[:, 0].clamp(EPS, 1.0 - EPS)),
  )
