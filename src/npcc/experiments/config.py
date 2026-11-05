"""Experiment configuration: the sweep grid and run-level options.

The grid is declared in a TOML file (loaded with the stdlib :mod:`tomllib`);
run-level options come from the CLI.  TOML has no ``null`` literal, so the
``normalize`` axis accepts the string ``"none"`` (or ``"off"``/``0``) to mean
"no Sinkhorn projection".

A key the TOML leaves out takes the dataclass default, so each default is
written once, on the field.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeVar, cast, get_args

from npcc.core.controls import FitControlsRosenblattBicop, Transform
from npcc.core.errors import (
  EstimatorConfigError,
  InvalidBackendKwargsError,
  UnknownBackendError,
)
from npcc.core.registry import available_backends, validate_backend_kwargs
from npcc.experiments.scenarios import (
  CONDITIONAL_UV_GRID_N,
  CONDITIONAL_X_GRID_N,
  FAMILIES,
  IMPLIED,
  SHAPE_FAMILIES,
  TAU_SCENARIOS,
  X_MAX,
  X_MIN,
  is_conditional,
)

ControlsT = TypeVar("ControlsT", bound=FitControlsRosenblattBicop)

#: How an estimator is fitted: ``"backend"`` is a ``RosenblattBicop`` on a
#: registry backend, ``"external_r"`` a script run by
#: :mod:`npcc.experiments._external`.
EstimatorKind = Literal["backend", "external_r"]


@dataclass(frozen=True)
class EstimatorSpec:
  """One estimator: a registry backend + transform + its own hyperparameters.

  ``backend_kwargs`` are validated against the backend's allow-list at
  construction; ``estimator_id`` is a stable content hash of the *effective*
  config (``backend``/``transform``/``backend_kwargs``, not the display
  ``label``), used to seed per-estimator RNG and join result rows.

  An ``external_r`` estimator is instead a ``script`` that fits and predicts
  on its own; ``backend`` names the method in the result tables, and its
  ``transform`` is ``"identity"`` because the script fits on the copula
  scale. Its ``estimator_id`` also hashes the script's content, so an edited
  script is a new estimator, and a ``--resume`` refuses to mix the two.
  """

  label: str
  backend: str
  transform: Transform
  backend_kwargs: Mapping[str, object] = field(default_factory=dict)
  kind: EstimatorKind = "backend"
  #: The method's script, relative to the working directory unless absolute.
  script: str | None = None
  #: SHA-256 of the script's content, read at construction.
  script_sha256: str | None = field(default=None, init=False)

  def __post_init__(self) -> None:
    if not self.label:
      raise EstimatorConfigError("Estimator label must be non-empty.")
    if self.kind not in get_args(EstimatorKind):
      raise EstimatorConfigError(
        f"kind must be one of {list(get_args(EstimatorKind))}; "
        f"got {self.kind!r} for {self.label!r}"
      )
    if self.kind == "external_r":
      self._check_external()
      return
    if self.script is not None:
      raise EstimatorConfigError(
        f"script must be omitted for kind='backend'; got {self.script!r} for "
        f"{self.label!r}. Set kind = 'external_r' to run a script."
      )
    if self.backend not in available_backends():
      raise UnknownBackendError(
        f"Unknown backend {self.backend!r}. Available: {available_backends()}."
      )
    if self.transform not in get_args(Transform):
      raise EstimatorConfigError(
        f"Unknown transform {self.transform!r}. "
        f"Allowed: {list(get_args(Transform))}."
      )
    frozen = MappingProxyType(deepcopy(dict(self.backend_kwargs)))
    validate_backend_kwargs(self.backend, frozen)
    object.__setattr__(self, "backend_kwargs", frozen)

  def _check_external(self) -> None:
    """Refuse what a script cannot honor, then fingerprint the script."""
    if not self.backend:
      raise EstimatorConfigError(
        f"backend must name the method of the external_r estimator "
        f"{self.label!r}; got an empty string"
      )
    if self.transform != "identity":
      raise EstimatorConfigError(
        f"transform must be 'identity' for the external_r estimator "
        f"{self.label!r}, whose script fits on the copula scale; "
        f"got {self.transform!r}"
      )
    if self.backend_kwargs:
      raise EstimatorConfigError(
        f"backend_kwargs must be empty for the external_r estimator "
        f"{self.label!r}, since nothing forwards them to its script; got "
        f"{dict(self.backend_kwargs)}. Set the method's options in the script."
      )
    if self.script is None or not Path(self.script).is_file():
      raise EstimatorConfigError(
        f"script must name an existing file for the external_r estimator "
        f"{self.label!r}; got {self.script!r}, resolved against {Path.cwd()}"
      )
    digest = hashlib.sha256(Path(self.script).read_bytes()).hexdigest()
    object.__setattr__(self, "backend_kwargs", MappingProxyType({}))
    object.__setattr__(self, "script_sha256", digest)

  def canonical_dict(self) -> dict[str, object]:
    """The effective config, exactly as forwarded at runtime (for hashing)."""
    if self.kind == "external_r":
      return {
        "kind": self.kind,
        "backend": self.backend,
        "transform": self.transform,
        "script": self.script,
        "script_sha256": self.script_sha256,
      }
    return {
      "backend": self.backend,
      "transform": self.transform,
      "backend_kwargs": dict(self.backend_kwargs),
    }

  @property
  def estimator_id(self) -> str:
    """Stable content hash of :meth:`canonical_dict`."""
    encoded = json.dumps(
      self.canonical_dict(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()

  def controls(
    self,
    cls: type[ControlsT],
    *,
    device: str | None,
    projection_grid_size: int,
    batch_size: int | None,
  ) -> ControlsT:
    """Fit controls for this estimator, under a study's run settings.

    Parameters
    ----------
    cls : type
        :class:`~npcc.core.controls.FitControlsRosenblattBicop` or a subclass.
    device : str or None
        The run's device.
    projection_grid_size : int
        The grid's Sinkhorn projection size.
    batch_size : int or None
        The grid's inference batch size.

    Returns
    -------
    FitControlsRosenblattBicop
        An instance of ``cls``.

    Raises
    ------
    EstimatorConfigError
        If the estimator is not a registry backend.
    """
    if self.kind != "backend":
      raise EstimatorConfigError(
        f"controls must be built for a kind='backend' estimator; got "
        f"kind={self.kind!r} for {self.label!r}, which runs its own script"
      )
    return cls(
      backend=self.backend,
      transform=self.transform,
      device=device,
      projection_grid_size=projection_grid_size,
      batch_size=batch_size,
      backend_kwargs=dict(self.backend_kwargs),
    )


@dataclass(frozen=True)
class Cell:
  """One data-generating cell of the sweep (its own sampled data + truth)."""

  family: str
  tau_scenario: str
  n: int
  rep: int

  @property
  def key(self) -> str:
    """The cell's checkpoint directory name."""
    return f"{self.family}__{self.tau_scenario}__n{self.n}__rep{self.rep}"


def _check_subset(name: str, values: list[Any], allowed: Iterable[str]) -> None:
  allowed_set = set(allowed)
  if not values:
    raise ValueError(f"{name} must be non-empty.")
  unknown = [v for v in values if v not in allowed_set]
  if unknown:
    raise ValueError(
      f"Unknown {name}: {unknown}. Allowed: {sorted(allowed_set)}."
    )


def _check_common(
  *,
  n: list[int],
  n_rep: int,
  estimators: Sequence[EstimatorSpec],
  projection_grid_size: int,
  batch_size: int | None,
) -> None:
  """Validate the fields both studies' grids share."""
  if not n or any(v <= 0 for v in n):
    raise ValueError(f"n must be a non-empty list of positive ints; got {n}")
  if n_rep <= 0:
    raise ValueError(f"n_rep must be a positive int; got {n_rep}")
  labels = [e.label for e in estimators]
  if len(labels) != len(set(labels)):
    raise EstimatorConfigError(f"estimator labels must be unique; got {labels}")
  if projection_grid_size < 2:
    raise ValueError(
      f"projection_grid_size must be >= 2; got {projection_grid_size}"
    )
  if batch_size is not None and batch_size <= 0:
    raise ValueError(
      f"batch_size must be a positive int or omitted; got {batch_size}"
    )


def _coerce_normalize(raw: list[Any]) -> list[Any]:
  """Map TOML's spellings of "no projection" to ``None``.

  Everything else passes through to :class:`GridConfig`, which validates it.
  """
  return [
    None
    if (isinstance(x, str) and x.lower() in {"none", "off"})
    or (type(x) is int and x == 0)
    else x
    for x in raw
  ]


@dataclass
class GridConfig:
  """The cartesian grid of axes to sweep.

  ``families`` are either one-parameter families, each crossed with every
  regime in ``tau_scenarios``, or shape families, whose one regime is
  ``"implied"``; a grid holds one kind or the other.
  """

  families: list[str]
  tau_scenarios: list[str]
  estimators: list[EstimatorSpec]
  normalize: list[int | None]
  n: list[int]
  n_rep: int
  projection_grid_size: int = 30
  batch_size: int | None = None
  conditional_uv_grid_n: int = CONDITIONAL_UV_GRID_N
  conditional_x_grid_n: int = CONDITIONAL_X_GRID_N
  surface_tau_levels: list[float] = field(
    default_factory=lambda: [0.1, 0.4, 0.7]
  )
  surface_families: list[str] = field(default_factory=lambda: ["clayton"])
  enable_tau_diagnostics: bool = True
  tau_diagnostic_n: int = 1000
  #: The ``x`` values to store surfaces at, in place of the ones
  #: ``surface_tau_levels`` picks; empty keeps that choice by tau. A shape
  #: family's tau need not pick out an ``x`` -- the Student family's is
  #: constant -- so its surfaces need these.
  surface_x: list[float] = field(default_factory=list)

  def __post_init__(self) -> None:
    _check_subset("families", self.families, [*FAMILIES, *SHAPE_FAMILIES])
    _check_subset(
      "tau_scenarios", self.tau_scenarios, [*TAU_SCENARIOS, IMPLIED]
    )
    self._check_shape()
    if not self.estimators:
      raise ValueError("estimators must be non-empty.")
    _check_common(
      n=self.n,
      n_rep=self.n_rep,
      estimators=self.estimators,
      projection_grid_size=self.projection_grid_size,
      batch_size=self.batch_size,
    )
    if not self.normalize:
      raise ValueError("normalize must be non-empty (e.g. [None]).")
    for entry in self.normalize:
      if entry is not None and (
        isinstance(entry, bool) or not isinstance(entry, int) or entry <= 0
      ):
        raise ValueError(
          "normalize entries must be None, 'none' or a positive int; "
          f"got {entry!r}"
        )
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
      _check_subset(
        "surface_families",
        self.surface_families,
        [*FAMILIES, *SHAPE_FAMILIES],
      )
    self._check_surface_x()
    if self.tau_diagnostic_n < 10:
      raise ValueError("tau_diagnostic_n must be >= 10.")
    self._check_external()

  def _check_shape(self) -> None:
    """Refuse a cell that would pair a shape family with another regime."""
    shape = [f for f in self.families if f in SHAPE_FAMILIES]
    if shape and set(self.tau_scenarios) != {IMPLIED}:
      raise ValueError(
        f"tau_scenarios must be [{IMPLIED!r}] with the shape families "
        f"{shape}, each of which implies its own tau(x); got "
        f"{self.tau_scenarios}"
      )
    one_par = [f for f in self.families if f not in SHAPE_FAMILIES]
    if IMPLIED in self.tau_scenarios and one_par:
      raise ValueError(
        f"families must all be shape families with tau_scenario "
        f"{IMPLIED!r}, which only a shape family defines; got the "
        f"one-parameter {one_par}. Sweep them in a grid of their own."
      )

  def _check_surface_x(self) -> None:
    """Refuse a surface ``x`` off the covariate's support, or a missing one."""
    outside = [x for x in self.surface_x if not X_MIN <= x <= X_MAX]
    if outside:
      raise ValueError(
        f"surface_x entries must lie in [{X_MIN}, {X_MAX}], the covariate's "
        f"support; got {outside}"
      )
    shape = [f for f in self.surface_families if f in SHAPE_FAMILIES]
    if shape and not self.surface_x:
      raise ValueError(
        f"surface_x must be set to store surfaces of the shape families "
        f"{shape}, whose tau(x) need not pick out an x; got none"
      )

  def _check_external(self) -> None:
    """Refuse the grid axes an external estimator's script cannot honor."""
    external = [e.label for e in self.estimators if e.kind == "external_r"]
    if not external:
      return
    if any(norm is not None for norm in self.normalize):
      raise EstimatorConfigError(
        f"normalize must be ['none'] with the external_r estimators "
        f"{external}: a Sinkhorn projection needs the density on its own "
        f"grid, which a script does not return; got {self.normalize}"
      )
    unconditional = [s for s in self.tau_scenarios if not is_conditional(s)]
    if unconditional:
      raise EstimatorConfigError(
        f"tau_scenarios must all be conditional with the external_r "
        f"estimators {external}, whose scripts are fitted on a covariate; "
        f"got the unconditional {unconditional}"
      )

  def cells(self) -> list[Cell]:
    """Every data cell of the grid, in sweep order."""
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
  fmt: str = "parquet"
  gpu_mem_fraction: float | None = None
  #: ``(index, count)``: compute only the cells whose position in the study's
  #: fixed order is ``index`` modulo ``count``, and stop before aggregating.
  shard: tuple[int, int] | None = None

  def __post_init__(self) -> None:
    if self.workers < 1:
      raise ValueError("workers must be >= 1.")
    if self.shard is not None:
      index, count = self.shard
      if not 0 <= index < count:
        raise ValueError(
          f"shard must be I/K with 0 <= I < K; got {index}/{count}"
        )
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
  script = entry.get("script")
  return EstimatorSpec(
    label=label,
    backend=backend,
    # `__post_init__` refuses any string that is not a transform or a kind.
    transform=cast("Transform", transform),
    backend_kwargs={str(k): v for k, v in raw.items()},
    kind=cast("EstimatorKind", str(entry.get("kind", "backend"))),
    script=None if script is None else str(script),
  )


def _read_grid(path: str | Path) -> dict[str, Any]:
  """The ``[grid]`` table of a TOML file (or the file, if it has none)."""
  with Path(path).open("rb") as fh:
    data = tomllib.load(fh)
  return data.get("grid", data)


def _present(
  grid: Mapping[str, Any], converters: Mapping[str, Callable[[Any], object]]
) -> dict[str, Any]:
  """The optional keys ``grid`` sets, converted; the rest keep their default."""
  return {k: conv(grid[k]) for k, conv in converters.items() if k in grid}


_GRID_OPTIONAL: dict[str, Callable[[Any], object]] = {
  "projection_grid_size": int,
  "batch_size": int,
  "conditional_uv_grid_n": int,
  "conditional_x_grid_n": int,
  "surface_tau_levels": lambda v: [float(t) for t in v],
  "surface_families": lambda v: [str(f) for f in v],
  "enable_tau_diagnostics": bool,
  "tau_diagnostic_n": int,
  "surface_x": lambda v: [float(x) for x in v],
}


def load_grid(path: str | Path) -> GridConfig:
  """Load a :class:`GridConfig` from the ``[grid]`` table of a TOML file."""
  grid = _read_grid(path)
  if "surface_x" in grid and "surface_tau_levels" in grid:
    raise ValueError(
      "surface_tau_levels must be omitted when surface_x is set, since the "
      "surfaces are then stored at those x and no tau level is read; got both"
    )
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
      **_present(grid, _GRID_OPTIONAL),
    )
  except KeyError as exc:
    raise ValueError(f"Missing required [grid] key: {exc}.") from exc
