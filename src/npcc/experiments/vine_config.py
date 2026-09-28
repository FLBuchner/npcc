"""Vine-study configuration: the sweep grid, read from a TOML ``[grid]`` table.

Run-level options are the bivariate study's :class:`~npcc.experiments.config.
RunConfig`, unchanged; estimators are its
:class:`~npcc.experiments.config.EstimatorSpec`, so a label means the same
backend configuration in both studies.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from npcc.core.errors import EstimatorConfigError
from npcc.experiments.config import EstimatorSpec, _parse_estimator

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


@dataclass
class VineGridConfig:
  """The cartesian grid of the vine study."""

  dims: list[int]
  n: list[int]
  n_rep: int
  estimators: list[EstimatorSpec]
  arms: list[str]
  eval_x_n: int = 5
  eval_m: int = 1000
  projection_grid_size: int = 30
  batch_size: int | None = None

  def __post_init__(self) -> None:
    if not self.dims or any(d < 2 for d in self.dims):
      raise ValueError(
        f"dims must be a non-empty list of ints >= 2; got {self.dims}"
      )
    if not self.n or any(v <= 0 for v in self.n):
      raise ValueError(
        f"n must be a non-empty list of positive ints; got {self.n}"
      )
    if self.n_rep <= 0:
      raise ValueError(f"n_rep must be a positive int; got {self.n_rep}")
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
    labels = [e.label for e in self.estimators]
    if len(labels) != len(set(labels)):
      raise EstimatorConfigError(
        f"estimator labels must be unique; got {labels}"
      )
    if self.eval_x_n < 1:
      raise ValueError(f"eval_x_n must be >= 1; got {self.eval_x_n}")
    if self.eval_m < 10:
      raise ValueError(f"eval_m must be >= 10; got {self.eval_m}")
    if self.projection_grid_size < 2:
      raise ValueError(
        f"projection_grid_size must be >= 2; got {self.projection_grid_size}"
      )
    if self.batch_size is not None and self.batch_size <= 0:
      raise ValueError(
        f"batch_size must be a positive int or omitted; got {self.batch_size}"
      )

  def cells(self) -> list[VineCell]:
    return [
      VineCell(d=d, n=n, rep=rep)
      for d in self.dims
      for n in self.n
      for rep in range(self.n_rep)
    ]


def load_vine_grid(path: str | Path) -> VineGridConfig:
  """Load a :class:`VineGridConfig` from the ``[grid]`` table of a TOML file."""
  with Path(path).open("rb") as fh:
    data = tomllib.load(fh)
  grid = data.get("grid", data)
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
      arms=[str(a) for a in grid.get("arms", list(ARMS))],
      eval_x_n=int(grid.get("eval_x_n", 5)),
      eval_m=int(grid.get("eval_m", 1000)),
      projection_grid_size=int(grid.get("projection_grid_size", 30)),
      batch_size=(
        None if grid.get("batch_size") is None else int(grid["batch_size"])
      ),
    )
  except KeyError as exc:
    raise ValueError(f"[grid] must define {exc}") from exc
