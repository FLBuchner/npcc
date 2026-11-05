"""Name to backend-factory registry for the inner conditional distribution.

The registry :class:`~npcc.core.bicop.RosenblattBicop` selects its inner
conditional-density model from.

A *factory* is any callable that accepts the common construction kwargs
(``transform``, ``quantile_table_config``, ``eps``, ``device``,
``batch_size``) plus arbitrary backend-specific keyword arguments, and returns a
:class:`~npcc.core.margin.ConditionalMargin`.

Built-in backends are registered lazily: the factory imports its backend
module (and, for the optional-extra backends, its third-party
dependency) only when the backend is actually requested, so ``import
npcc`` and :func:`available_backends` never import NGBoost / TabICL /
LightGBM.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

import torch

from npcc.core._transform import Transform
from npcc.core.errors import (
  InvalidBackendKwargsError,
  MissingBackendDependencyError,
  UnknownBackendError,
)
from npcc.core.margin import ConditionalMargin, QuantileTableConfig

BackendFactory = Callable[..., ConditionalMargin]

_JSON_SCALARS = (bool, int, float, str)


@dataclass(frozen=True)
class BackendSpec:
  """A registered backend: its factory plus config-validation metadata.

  ``kwarg_types=None`` means the backend accepts arbitrary ``backend_kwargs``
  (only their JSON shape is checked) -- used by hermetic test fakes and any
  opt-out backend. A mapping restricts the keys a study config may pass to its
  own, and checks each value against the type it names, so typos fail at
  config-load time instead of hours into a run. ``n_range`` is the backend's
  documented sample-size range.
  """

  factory: BackendFactory
  kwarg_types: Mapping[str, type] | None = None
  n_range: tuple[int | None, int | None] | None = None


_REGISTRY: dict[str, BackendSpec] = {}


def register_backend(
  name: str,
  factory: BackendFactory,
  *,
  kwarg_types: Mapping[str, type] | None = None,
  n_range: tuple[int | None, int | None] | None = None,
) -> None:
  """Register ``factory`` under ``name`` (overwrites any existing entry)."""
  _REGISTRY[name] = BackendSpec(
    factory=factory,
    kwarg_types=None if kwarg_types is None else dict(kwarg_types),
    n_range=n_range,
  )


def available_backends() -> list[str]:
  """Return the sorted list of registered backend names."""
  return sorted(_REGISTRY)


def _spec(name: str) -> BackendSpec:
  try:
    return _REGISTRY[name]
  except KeyError:
    raise UnknownBackendError(
      f"Unknown backend {name!r}. Available: {available_backends()}."
    ) from None


def _is_json(value: object) -> bool:
  if value is None or isinstance(value, _JSON_SCALARS):
    return True
  if isinstance(value, (list, tuple)):
    return all(_is_json(v) for v in value)
  if isinstance(value, dict):
    return all(isinstance(k, str) and _is_json(v) for k, v in value.items())
  return False


def _matches_type(value: object, expected: type) -> bool:
  # bool is an int subclass; never accept it where a number is wanted.
  if expected in (int, float) and isinstance(value, bool):
    return False
  if expected is float:
    return isinstance(value, (int, float))
  return isinstance(value, expected)


def validate_backend_kwargs(name: str, kwargs: Mapping[str, object]) -> None:
  """Validate ``backend_kwargs`` for ``name`` against its :class:`BackendSpec`.

  Always checks JSON shape; when the spec declares ``kwarg_types`` it also
  rejects unknown keys and type-checks the rest.
  """
  spec = _spec(name)
  for key, value in kwargs.items():
    if not _is_json(value):
      raise InvalidBackendKwargsError(
        f"{name}.{key} must be JSON-serializable "
        f"(scalars, lists, dicts); got {type(value).__name__}."
      )
  if spec.kwarg_types is None:
    return
  unknown = sorted(set(kwargs) - set(spec.kwarg_types))
  if unknown:
    raise InvalidBackendKwargsError(
      f"{name}: unknown backend_kwargs {unknown}; "
      f"allowed: {sorted(spec.kwarg_types)}."
    )
  for key, value in kwargs.items():
    expected = spec.kwarg_types[key]
    if not _matches_type(value, expected):
      raise InvalidBackendKwargsError(
        f"{name}.{key} must be {expected.__name__}; got {type(value).__name__}."
      )


def documented_n_range(name: str) -> tuple[int | None, int | None] | None:
  """Return the backend's documented ``(min_n, max_n)`` range, if declared."""
  return _spec(name).n_range


def create_backend(
  name: str,
  *,
  transform: Transform,
  quantile_table_config: QuantileTableConfig,
  eps: float,
  device: str | torch.device | None,
  batch_size: int | None,
  backend_kwargs: Mapping[str, Any] | None = None,
) -> ConditionalMargin:
  """Instantiate the backend registered under ``name``.

  ``backend_kwargs`` are validated against the backend's :class:`BackendSpec`
  (unknown or mis-typed keys raise :class:`InvalidBackendKwargsError`) and then
  forwarded to the factory alongside the common construction kwargs.
  """
  spec = _spec(name)
  kwargs = dict(backend_kwargs or {})
  validate_backend_kwargs(name, kwargs)
  return spec.factory(
    transform=transform,
    quantile_table_config=quantile_table_config,
    eps=eps,
    device=device,
    batch_size=batch_size,
    **kwargs,
  )


# ------------------------------------------------------------------
# Built-in backends (lazy factories).
# ------------------------------------------------------------------


def _register_builtin(
  name: str,
  target: str,
  *,
  extra: str | None = None,
  install_hint: str | None = None,
  kwarg_types: Mapping[str, type] | None = None,
  n_range: tuple[int | None, int | None] | None = None,
) -> None:
  """Register a factory that imports its adapter when called, not now.

  That ``import npcc`` needs no optional extra rests on this deferral alone,
  so the adapter is named by its import path rather than imported here.

  Parameters
  ----------
  name : str
      The registry name.
  target : str
      ``"module:Class"`` of the adapter.
  extra : str, optional
      The optional extra the adapter needs; ``None`` for a core one.
  install_hint : str, optional
      A message to raise instead of the generic missing-extra one.
  kwarg_types : mapping of str to type, optional
      Forwarded to :func:`register_backend`.
  n_range : tuple, optional
      Forwarded to :func:`register_backend`.
  """
  module_name, class_name = target.split(":")

  if install_hint is None and extra is not None:
    install_hint = (
      f"The {name!r} backend requires an optional dependency. "
      f"Install it with `pip install npcc[{extra}]` (or `uv sync "
      f"--extra {extra}`)."
    )

  def factory(**kwargs: Any) -> ConditionalMargin:  # noqa: ANN401 - heterogeneous backend kwargs
    try:
      backend_class = cast(
        "type[ConditionalMargin]",
        getattr(import_module(module_name), class_name),
      )
    except ImportError as exc:
      if install_hint is None:
        raise
      raise MissingBackendDependencyError(install_hint) from exc
    return backend_class(**kwargs)

  register_backend(name, factory, kwarg_types=kwarg_types, n_range=n_range)


_FINETUNE_KWARGS: dict[str, type] = {
  "epochs": int,
  "learning_rate": float,
  "early_stopping": bool,
}
_PYTABKIT_KWARGS: dict[str, type] = {
  "hpo": bool,
  "n_cv": int,
  "n_refit": int,
  "random_state": int,
  "n_epochs": int,
}

# TabPFN model version (v2.5 / v3) is a hyperparameter, not a distinct
# backend: pass `backend_kwargs = { model_version = "v2.5" }` (default v3).
_register_builtin(
  "tabpfn-quantiles",
  "npcc.core.backends.tabpfn_quantile:TabPFNQuantileBackend",
  kwarg_types={"model_kwargs": dict, "model_version": str},
  n_range=(None, 100_000),
)
# Fine-tuning is native to the core `tabpfn` package (no extra).
_register_builtin(
  "tabpfn-finetune",
  "npcc.core.backends.tabpfn_finetune:FinetunedTabPFNBackend",
  kwarg_types={
    **_FINETUNE_KWARGS,
    "ce_loss_weight": float,
    "crps_loss_weight": float,
    "crls_loss_weight": float,
    "mse_loss_weight": float,
    "mae_loss_weight": float,
  },
  n_range=(None, 50_000),
)
_register_builtin(
  "ngboost",
  "npcc.core.backends.ngboost:NGBoostBackend",
  extra="ngboost",
  kwarg_types={
    "n_estimators": int,
    "learning_rate": float,
    "minibatch_frac": float,
    "random_state": int,
  },
)
_register_builtin(
  "lightgbm",
  "npcc.core.backends.lightgbm_quantile:LightGBMQuantileBackend",
  extra="lightgbm",
  kwarg_types={
    "n_estimators": int,
    "learning_rate": float,
    "max_depth": int,
    "num_leaves": int,
    "min_child_samples": int,
    "min_child_weight": float,
    "subsample": float,
    "subsample_freq": int,
    "colsample_bytree": float,
    "reg_alpha": float,
    "reg_lambda": float,
    "n_jobs": int,
    "random_state": int,
  },
)
_register_builtin(
  "pytabkit-realmlp",
  "npcc.core.backends.pytabkit:PyTabKitRealMLPBackend",
  extra="pytabkit",
  kwarg_types=_PYTABKIT_KWARGS,
)
_register_builtin(
  "pytabkit-tabm",
  "npcc.core.backends.pytabkit:PyTabKitTabMBackend",
  extra="pytabkit",
  kwarg_types=_PYTABKIT_KWARGS,
)
# Nori's NoriRegressor kwargs are model-release-specific; left unrestricted.
_register_builtin(
  "nori",
  "npcc.core.backends.nori:NoriBackend",
  install_hint=(
    "The 'nori' backend requires synthefy-nori. Until the torch-uncapping fix "
    "reaches PyPI, install the fork into a cu128 env:\n"
    '  uv pip install "synthefy-nori @ '
    'git+https://github.com/tvatter/synthefy-nori.git@allow-newer-torch-cuda"'
  ),
)
_register_builtin(
  "tabicl",
  "npcc.core.backends.tabicl:TabICLBackend",
  extra="tabicl",
  kwarg_types={"model_kwargs": dict},
  n_range=(300, 48_000),
)
_register_builtin(
  "tabicl-finetune",
  "npcc.core.backends.tabicl_finetune:FinetunedTabICLBackend",
  extra="tabicl",
  kwarg_types=_FINETUNE_KWARGS,
  n_range=(300, 48_000),
)
