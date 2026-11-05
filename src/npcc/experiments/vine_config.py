"""Vine-study configuration: the sweep grid, read from a TOML ``[grid]`` table.

Run-level options are the bivariate study's :class:`~npcc.experiments.config.
RunConfig`, unchanged; estimators are its
:class:`~npcc.experiments.config.EstimatorSpec`, so a label means the same
backend configuration in both studies.
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
  _parse_estimator,
  _present,
  _read_grid,
)

# `oracle`: the true structure, a simplified context (pairs see `x` only).
# `random`: a uniformly drawn structure, a non-simplified context, because a
#   wrong structure makes the truth depend on each edge's conditioning set.
# `tll`: pyvinecopulib with kernel-density margins and TLL pairs on the true
#   structure, ignoring `x` -- what leaving the covariate out costs.
ARMS: tuple[str, ...] = ("oracle", "random", "tll")
NPCC_ARMS: tuple[str, ...] = ("oracle", "random")


@dataclass(frozen=True)
class VineCell:
  """One data-generating cell: its own truth, training sample and test set."""

  d: int
  n: int
  rep: int

  @property
  def key(self) -> str:
    """The cell's checkpoint directory name."""
    return f"d{self.d}__n{self.n}__rep{self.rep}"


@dataclass
class VineGridConfig:
  """The cartesian grid of the vine study."""

  dims: list[int]
  n: list[int]
  n_rep: int
  estimators: list[EstimatorSpec]
  arms: list[str] = field(default_factory=lambda: list(ARMS))
  eval_x_n: int = 5
  eval_m: int = 1000
  projection_grid_size: int = 30
  batch_size: int | None = None

  def __post_init__(self) -> None:
    if not self.dims or any(d < 2 for d in self.dims):
      raise ValueError(
        f"dims must be a non-empty list of ints >= 2; got {self.dims}"
      )
    _check_common(
      n=self.n,
      n_rep=self.n_rep,
      estimators=self.estimators,
      projection_grid_size=self.projection_grid_size,
      batch_size=self.batch_size,
    )
    if not self.arms:
      raise ValueError("arms must be non-empty")
    unknown = [a for a in self.arms if a not in ARMS]
    if unknown:
      raise ValueError(f"arms must be drawn from {list(ARMS)}; got {unknown}")
    if len(set(self.arms)) != len(self.arms):
      raise ValueError(f"arms must be unique; got {self.arms}")
    if any(a in NPCC_ARMS for a in self.arms) and not self.estimators:
      raise EstimatorConfigError(
        "estimators must be non-empty when an NPCC arm (oracle, random) runs"
      )
    external = [e.label for e in self.estimators if e.kind != "backend"]
    if external:
      raise EstimatorConfigError(
        f"vine-study estimators must be kind='backend', since every vine pair "
        f"is a RosenblattBicop; got the external {external}"
      )
    if self.eval_x_n < 1:
      raise ValueError(f"eval_x_n must be >= 1; got {self.eval_x_n}")
    if self.eval_m < 10:
      raise ValueError(f"eval_m must be >= 10; got {self.eval_m}")

  def cells(self) -> list[VineCell]:
    """Every data cell of the grid, in sweep order."""
    return [
      VineCell(d=d, n=n, rep=rep)
      for d in self.dims
      for n in self.n
      for rep in range(self.n_rep)
    ]


_VINE_OPTIONAL: dict[str, Callable[[Any], object]] = {
  "arms": lambda v: [str(a) for a in v],
  "eval_x_n": int,
  "eval_m": int,
  "projection_grid_size": int,
  "batch_size": int,
}


def load_vine_grid(path: str | Path) -> VineGridConfig:
  """Load a :class:`VineGridConfig` from the ``[grid]`` table of a TOML file."""
  grid = _read_grid(path)
  raw_estimators = grid.get("estimators", [])
  if not isinstance(raw_estimators, list):
    raise EstimatorConfigError(
      "estimators must be a [[grid.estimators]] array of tables"
    )
  try:
    return VineGridConfig(
      dims=[int(v) for v in grid["dims"]],
      n=[int(v) for v in grid["n"]],
      n_rep=int(grid["n_rep"]),
      estimators=[_parse_estimator(e) for e in raw_estimators],
      **_present(grid, _VINE_OPTIONAL),
    )
  except KeyError as exc:
    raise ValueError(f"[grid] must define {exc}") from exc
