"""Tests for the backend registry: names, construction and kwarg validation."""

from __future__ import annotations

import sys

import pytest

from npcc.core import (
  ConditionalMargin,
  InvalidBackendKwargsError,
  MissingBackendDependencyError,
  QuantileTableConfig,
  UnknownBackendError,
  available_backends,
  create_backend,
  documented_n_range,
  validate_backend_kwargs,
)
from tests.conftest import NormalQuantileBackend, hermetic_backend

_BUILTINS = (
  "catboost",
  "lightgbm",
  "ngboost",
  "nori",
  "pytabkit-realmlp",
  "pytabkit-tabm",
  "tabicl",
  "tabicl-finetune",
  "tabpfn-finetune",
  "tabpfn-quantiles",
  "xgboost",
)


def create(name: str, **backend_kwargs: object) -> ConditionalMargin:
  """``create_backend`` with the common arguments at their defaults."""
  return create_backend(
    name,
    transform="logit",
    quantile_table_config=QuantileTableConfig(),
    eps=1e-6,
    device="cpu",
    batch_size=None,
    backend_kwargs=backend_kwargs,
  )


def test_every_builtin_is_registered() -> None:
  """The built-ins are registered at import, whether or not their extra is.

  A backend whose name only appears once its extra is installed could not be
  named in a study config that is checked on a machine without it.
  """
  assert set(_BUILTINS) <= set(available_backends())


def test_an_unknown_name_is_refused_with_the_known_ones() -> None:
  """A typo names what is available rather than failing with a ``KeyError``."""
  with pytest.raises(UnknownBackendError, match="Unknown backend.*Available"):
    create("does-not-exist")


def test_a_missing_extra_names_its_install_command(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """An absent optional dependency surfaces as the extra to install.

  The adapter's import is blocked to stand in for the missing package; the
  factory must turn the ``ImportError`` into a ``pip install`` hint.
  """
  monkeypatch.setitem(sys.modules, "npcc.core.backends.ngboost", None)

  with pytest.raises(
    MissingBackendDependencyError, match=r"pip install npcc\[ngboost\]"
  ):
    create("ngboost")


def test_a_registered_factory_receives_the_common_arguments() -> None:
  """A registered factory is what ``create_backend`` builds through."""
  with hermetic_backend("normal-quantile-registry"):
    margin = create("normal-quantile-registry")

  assert isinstance(margin, NormalQuantileBackend)
  assert margin.transform == "logit"


def test_an_unknown_kwarg_fails_before_the_constructor() -> None:
  """A typo fails at config-load time, not deep inside a backend's init."""
  with pytest.raises(InvalidBackendKwargsError, match="unknown backend"):
    create("tabpfn-quantiles", not_a_real_kwarg=123)


@pytest.mark.parametrize(
  ("name", "kwargs", "match"),
  [
    ("lightgbm", {"max_dpeth": 4}, "unknown backend"),
    ("lightgbm", {"max_depth": 3.5}, "max_depth must be int"),
    ("lightgbm", {"max_depth": True}, "max_depth must be int"),
    ("tabpfn-quantiles", {"model_kwargs": 5}, "model_kwargs must be dict"),
    ("nori", {"bad": object()}, "JSON"),
  ],
)
def test_backend_kwargs_are_checked_against_the_schema(
  name: str, kwargs: dict[str, object], match: str
) -> None:
  """Keys outside a backend's schema, mistyped values and non-JSON are refused.

  A ``bool`` is an ``int`` to ``isinstance``, and is refused where a number
  is wanted: ``max_depth = true`` in a TOML config is a mistake, not a depth.
  """
  with pytest.raises(InvalidBackendKwargsError, match=match):
    validate_backend_kwargs(name, kwargs)


@pytest.mark.parametrize(
  ("name", "kwargs"),
  [
    ("lightgbm", {"max_depth": 4, "subsample": 0.8, "reg_alpha": 1}),
    ("tabpfn-quantiles", {"model_version": "v2.5", "model_kwargs": {}}),
    ("nori", {"anything": [1, 2, 3]}),
  ],
)
def test_backend_kwargs_within_the_schema_are_accepted(
  name: str, kwargs: dict[str, object]
) -> None:
  """Within the schema a call goes through, so the checks are not too tight.

  An integer passes where a float is wanted, and a backend with no schema
  takes any JSON-shaped kwarg.
  """
  validate_backend_kwargs(name, kwargs)


def test_documented_n_range() -> None:
  """The documented sample-size range is reported where one is declared."""
  assert documented_n_range("tabicl") == (300, 48000)
  assert documented_n_range("lightgbm") is None
