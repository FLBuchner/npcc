"""Tests for Rosenblatt fit controls."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from pyvinecopulib.core import ControlsLike

from npcc.core.bicop import RosenblattBicop
from npcc.core.controls import (
  FitControlsRosenblattBicop,
  FitControlsRosenblattVinecop,
  Transform,
)
from npcc.core.errors import (
  InvalidBackendKwargsError,
  UnknownBackendError,
)
from npcc.core.margin_quantile_table import QuantileTableConfig


def test_bicop_controls_defaults() -> None:
  controls = FitControlsRosenblattBicop()

  assert controls.backend == "tabpfn-criterion"
  assert controls.eps == 1e-6
  assert controls.transform == "logit"
  assert controls.device is None
  assert controls.batch_size is None
  assert controls.backend_kwargs == {}
  assert controls.sinkhorn_iters is None
  assert controls.projection_grid_size == 101
  assert controls.cdf_n_int == 12


def test_controls_satisfy_pyvinecopulib_protocol() -> None:
  controls = FitControlsRosenblattBicop()

  assert isinstance(controls, ControlsLike)


def test_vinecop_controls_are_bicop_controls() -> None:
  controls = FitControlsRosenblattVinecop()

  assert isinstance(controls, FitControlsRosenblattBicop)
  assert isinstance(controls, ControlsLike)


def test_device_is_normalized() -> None:
  controls = FitControlsRosenblattBicop(device="cpu")

  assert controls.device == torch.device("cpu")


def test_backend_kwargs_are_copied_on_construction() -> None:
  backend_kwargs: dict[str, object] = {"model_version": "v2.5"}

  controls = FitControlsRosenblattBicop(
    backend_kwargs=backend_kwargs,
  )
  backend_kwargs["model_version"] = "v3"

  assert (controls.backend_kwargs or {})["model_version"] == "v2.5"


def test_to_dict_returns_backend_kwargs_copy() -> None:
  controls = FitControlsRosenblattBicop(
    backend_kwargs={"model_version": "v2.5"},
  )

  settings = controls.to_dict()
  serialized_kwargs = cast(
    "dict[str, object]",
    settings["backend_kwargs"],
  )

  assert isinstance(serialized_kwargs, dict)
  assert serialized_kwargs is not controls.backend_kwargs

  serialized_kwargs["model_version"] = "v3"

  assert (controls.backend_kwargs or {})["model_version"] == "v2.5"


def test_to_dict_contains_all_settings() -> None:
  controls = FitControlsRosenblattBicop(
    backend="tabpfn-criterion",
    transform="probit",
    device="cpu",
    batch_size=64,
    backend_kwargs={"model_version": "v2.5"},
    sinkhorn_iters=5,
    projection_grid_size=51,
    cdf_n_int=16,
  )

  settings = controls.to_dict()

  assert settings == {
    "backend": "tabpfn-criterion",
    "quantile_table_config": controls.quantile_table_config,
    "eps": 1e-6,
    "transform": "probit",
    "device": torch.device("cpu"),
    "batch_size": 64,
    "backend_kwargs": {"model_version": "v2.5"},
    "sinkhorn_iters": 5,
    "projection_grid_size": 51,
    "cdf_n_int": 16,
  }


def test_invalid_transform_is_rejected() -> None:
  with pytest.raises(ValueError, match="transform must be"):
    FitControlsRosenblattBicop(
      transform=cast("Transform", "invalid"),
    )


@pytest.mark.parametrize("eps", [0.0, 0.5, -1e-6])
def test_invalid_eps_is_rejected(eps: float) -> None:
  with pytest.raises(ValueError, match="eps must lie"):
    FitControlsRosenblattBicop(eps=eps)


def test_nonpositive_batch_size_is_rejected() -> None:
  with pytest.raises(ValueError, match="batch_size must be positive"):
    FitControlsRosenblattBicop(batch_size=0)


def test_nonpositive_sinkhorn_iterations_are_rejected() -> None:
  with pytest.raises(ValueError, match="sinkhorn_iters"):
    FitControlsRosenblattBicop(sinkhorn_iters=0)


def test_small_projection_grid_is_rejected() -> None:
  with pytest.raises(ValueError, match="projection_grid_size"):
    FitControlsRosenblattBicop(projection_grid_size=1)


def test_small_cdf_integration_grid_is_rejected() -> None:
  """A trapezoid rule needs two points; one integrates nothing at all.

  The guard used to sit on a per-call ``n_int`` keyword on ``cdf``. That
  keyword is gone -- the base's ``cdf`` is a dispatcher forwarding nothing but
  ``x`` -- so without this the setting would reach the integration unchecked.
  """
  with pytest.raises(ValueError, match="cdf_n_int"):
    FitControlsRosenblattBicop(cdf_n_int=1)


def test_unknown_backend_is_rejected() -> None:
  with pytest.raises(UnknownBackendError, match="Unknown backend"):
    FitControlsRosenblattBicop(backend="unknown")


def test_invalid_backend_kwargs_are_rejected() -> None:
  with pytest.raises(
    InvalidBackendKwargsError,
    match="unknown backend_kwargs",
  ):
    FitControlsRosenblattBicop(
      backend_kwargs={"not_a_real_kwarg": 1},
    )


def test_none_means_the_default_for_the_composite_settings() -> None:
  """A caller assembling controls from a table passes ``None``, not a default.

  ``BACKEND_KWARGS.get(name)`` is ``None`` for a backend with nothing to
  configure, which is the natural spelling and not an error. The constructor
  this replaced accepted it (`backend_kwargs or {}`); dropping that tolerance
  broke a notebook with ``'NoneType' object is not iterable``.
  """
  controls = FitControlsRosenblattBicop(
    backend_kwargs=None,
    quantile_table_config=None,
  )

  assert controls.backend_kwargs == {}
  assert controls.quantile_table_config == QuantileTableConfig()
  assert controls.to_dict()["backend_kwargs"] == {}

  # And it survives the round trip a foreign `ControlsLike` takes.
  assert RosenblattBicop(controls).backend_kwargs == {}
