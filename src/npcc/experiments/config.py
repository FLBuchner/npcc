"""Experiment configuration: the sweep grid and run-level options.

The grid is declared in a TOML file (loaded with the stdlib :mod:`tomllib`);
run-level options come from the CLI.  TOML has no ``null`` literal, so the
``normalize`` axis accepts the string ``"none"`` (or ``"off"``/``0``) to mean
"no Sinkhorn projection".
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from npcc.core.errors import (
  EstimatorConfigError,
  InvalidBackendKwargsError,
  UnknownBackendError,
)
from npcc.core.registry import available_backends, validate_backend_kwargs
from npcc.experiments.scenarios import FAMILIES, TAU_SCENARIOS

TRANSFORMS: tuple[str, ...] = ("identity", "logit", "probit")


@dataclass(frozen=True)
class EstimatorSpec:
  """One estimator: a registry backend + transform + its own hyperparameters.

  ``backend_kwargs`` are validated against the backend's allow-list at
  construction; ``estimator_id`` is a stable content hash of the *effective*
  config (``backend``/``transform``/``backend_kwargs``, not the display
  ``label``), used to seed per-estimator RNG and join result rows.
  """

  label: str
  backend: str
  transform: str
  backend_kwargs: Mapping[str, object] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.label:
      raise EstimatorConfigError("Estimator label must be non-empty.")
    if self.backend not in available_backends():
      raise UnknownBackendError(
        f"Unknown backend {self.backend!r}. Available: {available_backends()}."
      )
    if self.transform not in TRANSFORMS:
      raise EstimatorConfigError(
        f"Unknown transform {self.transform!r}. Allowed: {list(TRANSFORMS)}."
      )
    frozen = MappingProxyType(deepcopy(dict(self.backend_kwargs)))
    validate_backend_kwargs(self.backend, frozen)
    object.__setattr__(self, "backend_kwargs", frozen)

  def canonical_dict(self) -> dict[str, object]:
    """The effective config, exactly as forwarded at runtime (for hashing)."""
    return {
      "backend": self.backend,
      "transform": self.transform,
      "backend_kwargs": dict(self.backend_kwargs),
    }

  @property
  def estimator_id(self) -> str:
    encoded = json.dumps(
      self.canonical_dict(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Cell:
  """One data-generating cell of the sweep (its own sampled data + truth)."""

  family: str
  tau_scenario: str
  n: int
  rep: int


def _check_subset(name: str, values: list[Any], allowed: Iterable[str]) -> None:
  allowed_set = set(allowed)
  if not values:
    raise ValueError(f"{name} must be non-empty.")
  unknown = [v for v in values if v not in allowed_set]
  if unknown:
    raise ValueError(
      f"Unknown {name}: {unknown}. Allowed: {sorted(allowed_set)}."
    )


def _coerce_normalize(raw: list[Any]) -> list[int | None]:
  out: list[int | None] = []
  for x in raw:
    if x is None or (isinstance(x, str) and x.lower() in {"none", "off"}):
      out.append(None)
    elif isinstance(x, bool):  # bool is an int subclass; reject explicitly.
      raise ValueError(f"normalize entries must be int or 'none', got {x!r}.")
    elif isinstance(x, int) and x == 0:
      out.append(None)
    elif isinstance(x, int) and x > 0:
      out.append(x)
    else:
      raise ValueError(
        f"normalize entries must be a positive int or 'none', got {x!r}."
      )
  return out


@dataclass
class GridConfig:
  """The cartesian grid of axes to sweep."""

  families: list[str]
  tau_scenarios: list[str]
  estimators: list[EstimatorSpec]
  normalize: list[int | None]
  n: list[int]
  n_rep: int
  projection_grid_size: int = 30
  batch_size: int | None = None
  conditional_uv_grid_n: int = 20
  conditional_x_grid_n: int = 10
  surface_tau_levels: list[float] = field(
    default_factory=lambda: [0.1, 0.5, 0.9]
  )
  surface_families: list[str] = field(default_factory=lambda: ["clayton"])
  enable_tau_diagnostics: bool = True
  tau_diagnostic_n: int = 1000

  def __post_init__(self) -> None:
    _check_subset("families", self.families, FAMILIES)
    _check_subset("tau_scenarios", self.tau_scenarios, TAU_SCENARIOS)
    if not self.estimators:
      raise ValueError("estimators must be non-empty.")
    labels = [e.label for e in self.estimators]
    if len(labels) != len(set(labels)):
      raise EstimatorConfigError("Estimator labels must be unique.")
    if not self.normalize:
      raise ValueError("normalize must be non-empty (e.g. [None]).")
    for entry in self.normalize:
      if entry is not None and (
        isinstance(entry, bool) or not isinstance(entry, int) or entry <= 0
      ):
        raise ValueError(
          "normalize entries must be None or a positive int "
          f"(got {entry!r}); the string 'none' is only for TOML input."
        )
    if not self.n or any(v <= 0 for v in self.n):
      raise ValueError("n must be a non-empty list of positive ints.")
    if self.n_rep <= 0:
      raise ValueError("n_rep must be a positive int.")
    if self.projection_grid_size < 2:
      raise ValueError("projection_grid_size must be >= 2.")

    if self.batch_size is not None and self.batch_size <= 0:
      raise ValueError("batch_size must be a positive int or omitted.")
    if self.conditional_uv_grid_n < 2:
      raise ValueError("conditional_uv_grid_n must be >= 2.")
    if self.conditional_x_grid_n < 1:
      raise ValueError("conditional_x_grid_n must be >= 1.")
    if not self.surface_tau_levels:
      raise ValueError("surface_tau_levels must be non-empty.")
    for tau in self.surface_tau_levels:
      if tau <= 0.0 or tau >= 1.0:
        raise ValueError("surface_tau_levels entries must be in (0, 1).")
    if self.surface_families:
      _check_subset("surface_families", self.surface_families, FAMILIES)
    if self.tau_diagnostic_n < 10:
      raise ValueError("tau_diagnostic_n must be >= 10.")

  def estimator_specs(self) -> list[EstimatorSpec]:
    return list(self.estimators)

  def cells(self) -> list[Cell]:
    return [
      Cell(family=f, tau_scenario=s, n=n, rep=rep)
      for f in self.families
      for s in self.tau_scenarios
      for n in self.n
      for rep in range(self.n_rep)
    ]


@dataclass
class RunConfig:
  """Run-level options (not part of the swept grid)."""

  out: Path
  device: str | None = None
  workers: int = 1
  base_seed: int = 317
  log_level: str = "INFO"
  fmt: str = "parquet"
  gpu_mem_fraction: float | None = None

  def __post_init__(self) -> None:
    if self.workers < 1:
      raise ValueError("workers must be >= 1.")
    if self.fmt not in {"csv", "parquet"}:
      raise ValueError("fmt must be 'csv' or 'parquet'.")
    if self.gpu_mem_fraction is not None and not (
      0.0 < self.gpu_mem_fraction <= 1.0
    ):
      raise ValueError("gpu_mem_fraction must be in (0, 1].")


def _parse_estimator(entry: Mapping[str, object]) -> EstimatorSpec:
  """Build an :class:`EstimatorSpec` from one ``[[grid.estimators]]`` table."""
  try:
    label = str(entry["label"])
    backend = str(entry["backend"])
    transform = str(entry["transform"])
  except KeyError as exc:
    raise EstimatorConfigError(
      f"Estimator entry is missing required key: {exc}."
    ) from exc
  raw = entry.get("backend_kwargs", {})
  if not isinstance(raw, dict):
    raise InvalidBackendKwargsError("backend_kwargs must be a TOML table.")
  return EstimatorSpec(
    label=label,
    backend=backend,
    transform=transform,
    backend_kwargs={str(k): v for k, v in raw.items()},
  )


def load_grid(path: str | Path) -> GridConfig:
  """Load a :class:`GridConfig` from the ``[grid]`` table of a TOML file."""
  with Path(path).open("rb") as fh:
    data = tomllib.load(fh)
  grid = data.get("grid", data)
  raw_estimators = grid.get("estimators")
  if not isinstance(raw_estimators, list) or not raw_estimators:
    raise EstimatorConfigError(
      "Config must define a non-empty [[grid.estimators]] array."
    )
  try:
    return GridConfig(
      families=list(grid["families"]),
      tau_scenarios=list(grid["tau_scenarios"]),
      estimators=[_parse_estimator(e) for e in raw_estimators],
      normalize=_coerce_normalize(list(grid["normalize"])),
      n=[int(v) for v in grid["n"]],
      n_rep=int(grid["n_rep"]),
      projection_grid_size=int(grid.get("projection_grid_size", 30)),
      batch_size=(
        None if grid.get("batch_size") is None else int(grid["batch_size"])
      ),
      conditional_uv_grid_n=int(grid.get("conditional_uv_grid_n", 20)),
      conditional_x_grid_n=int(grid.get("conditional_x_grid_n", 10)),
      surface_tau_levels=[
        float(v) for v in grid.get("surface_tau_levels", [0.1, 0.5, 0.9])
      ],
      surface_families=[
        str(v) for v in grid.get("surface_families", ["clayton"])
      ],
      enable_tau_diagnostics=bool(grid.get("enable_tau_diagnostics", True)),
      tau_diagnostic_n=int(grid.get("tau_diagnostic_n", 1000)),
    )
  except KeyError as exc:
    raise ValueError(f"Missing required [grid] key: {exc}.") from exc
