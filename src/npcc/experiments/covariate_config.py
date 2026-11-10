"""Covariate-study configuration (E9): the grid, read from a TOML ``[grid]``.

Run-level options are the bivariate study's :class:`~npcc.experiments.config.
RunConfig`, unchanged; estimators are its
:class:`~npcc.experiments.config.EstimatorSpec`, so a label means the same
backend configuration as in E1.

An estimator is fitted on every covariate unless its label is listed in
``active_covariates``, which fits it on the scenario's active covariates only
(:data:`~npcc.experiments.covariate_scenarios.ACTIVE`), or in
``without_covariates``, which fits it on ``(U, V)`` alone. Either may name a
label whose backend configuration another estimator shares: the rows carry
the ``covariates`` an estimator was fitted on, and its seed derives from it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from npcc.core.errors import EstimatorConfigError
from npcc.experiments.config import (
  EstimatorSpec,
  _check_common,
  _check_subset,
  _parse_estimator,
  _present,
  _read_grid,
)
from npcc.experiments.covariate_scenarios import (
  ACTIVE,
  SCENARIO_FAMILIES,
  SCENARIOS,
)

#: What an estimator is fitted on: every covariate, the scenario's active
#: ones, or none.
COVARIATE_MODES: tuple[str, ...] = ("all", "active", "none")


@dataclass(frozen=True)
class CovariateCell:
  """One data cell: a family, a scenario, a sample size and a repetition."""

  family: str
  scenario: str
  n: int
  rep: int

  @property
  def key(self) -> str:
    """The cell's checkpoint directory name."""
    return f"{self.family}__{self.scenario}__n{self.n}__rep{self.rep}"


@dataclass
class CovariateGridConfig:
  """The cartesian grid of the covariate study."""

  families: list[str]
  scenarios: list[str]
  n: list[int]
  n_rep: int
  estimators: list[EstimatorSpec]
  #: Labels fitted on the scenario's active covariates only.
  active_covariates: list[str] = field(default_factory=list)
  #: Labels fitted on ``(U, V)`` alone.
  without_covariates: list[str] = field(default_factory=list)
  #: The covariate points each scenario is scored at.
  n_points: int = 20
  #: The ``uv`` grid's points per axis, at each covariate point.
  uv_grid_n: int = 20
  #: The Monte Carlo KL's draws per covariate point.
  holdout_n: int = 2000
  projection_grid_size: int = 30
  batch_size: int | None = None

  def __post_init__(self) -> None:
    _check_subset("families", self.families, SCENARIO_FAMILIES)
    _check_subset("scenarios", self.scenarios, SCENARIOS)
    _check_common(
      n=self.n,
      n_rep=self.n_rep,
      estimators=self.estimators,
      projection_grid_size=self.projection_grid_size,
      batch_size=self.batch_size,
    )
    kinds = {e.label: e.kind for e in self.estimators}
    for name in ("active_covariates", "without_covariates"):
      listed = getattr(self, name)
      unknown = [label for label in listed if label not in kinds]
      if unknown:
        raise EstimatorConfigError(
          f"{name} must name estimators of the grid; got {unknown}, not in "
          f"{list(kinds)}"
        )
      external = [label for label in listed if kinds[label] != "backend"]
      if external:
        raise EstimatorConfigError(
          f"{name} must name kind='backend' estimators, since a script is "
          f"handed every covariate; got the external {external}"
        )
    both = sorted(set(self.active_covariates) & set(self.without_covariates))
    if both:
      raise EstimatorConfigError(
        "active_covariates and without_covariates must not share a label; "
        f"got {both} in both"
      )
    for name, low in (("n_points", 1), ("uv_grid_n", 2), ("holdout_n", 2)):
      if getattr(self, name) < low:
        raise ValueError(f"{name} must be >= {low}; got {getattr(self, name)}")

  def covariates(self, est: EstimatorSpec) -> str:
    """What ``est`` is fitted on, one of :data:`COVARIATE_MODES`.

    Parameters
    ----------
    est : EstimatorSpec
        An estimator of the grid.

    Returns
    -------
    str
        ``"active"``, ``"none"`` or ``"all"``.
    """
    if est.label in self.active_covariates:
      return "active"
    if est.label in self.without_covariates:
      return "none"
    return "all"

  def fits(self, est: EstimatorSpec, scenario: str) -> bool:
    """Whether ``est`` is fitted in ``scenario``'s cells.

    An estimator on the active covariates is not fitted where there are
    none: there it would be the estimator without covariates.

    Parameters
    ----------
    est : EstimatorSpec
        An estimator of the grid.
    scenario : str
        One of the grid's scenarios.

    Returns
    -------
    bool
        ``False`` only for an ``active`` estimator in a scenario without an
        active covariate.
    """
    return self.covariates(est) != "active" or bool(ACTIVE[scenario])

  def cells(self) -> list[CovariateCell]:
    """Every data cell of the grid, in sweep order."""
    return [
      CovariateCell(family=f, scenario=s, n=n, rep=rep)
      for f in self.families
      for s in self.scenarios
      for n in self.n
      for rep in range(self.n_rep)
    ]


_COVARIATE_OPTIONAL: dict[str, Callable[[Any], object]] = {
  "active_covariates": lambda v: [str(a) for a in v],
  "without_covariates": lambda v: [str(a) for a in v],
  "n_points": int,
  "uv_grid_n": int,
  "holdout_n": int,
  "projection_grid_size": int,
  "batch_size": int,
}


def load_covariate_grid(path: str | Path) -> CovariateGridConfig:
  """Load a :class:`CovariateGridConfig` from a TOML file's ``[grid]`` table.

  Parameters
  ----------
  path : str or pathlib.Path
      The TOML file.

  Returns
  -------
  CovariateGridConfig
      The grid.
  """
  grid = _read_grid(path)
  raw_estimators = grid.get("estimators")
  if not isinstance(raw_estimators, list) or not raw_estimators:
    raise EstimatorConfigError(
      "estimators must be a non-empty [[grid.estimators]] array of tables"
    )
  try:
    return CovariateGridConfig(
      families=[str(v) for v in grid["families"]],
      scenarios=[str(v) for v in grid["scenarios"]],
      n=[int(v) for v in grid["n"]],
      n_rep=int(grid["n_rep"]),
      estimators=[_parse_estimator(e) for e in raw_estimators],
      **_present(grid, _COVARIATE_OPTIONAL),
    )
  except KeyError as exc:
    raise ValueError(f"[grid] must define {exc}") from exc
