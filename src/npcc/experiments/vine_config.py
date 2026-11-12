"""Vine-study configuration: the sweep grid, read from a TOML ``[grid]`` table.

Run-level options are the bivariate study's :class:`~npcc.experiments.config.
RunConfig`, unchanged; estimators are its
:class:`~npcc.experiments.config.EstimatorSpec`, so a label means the same
backend configuration in both studies.
"""

from __future__ import annotations

import hashlib
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
# `selected`: the structure that Dissmann's algorithm with TLL pairs selects
#   on the probability integral transforms of the arm's own fitted margins,
#   ignoring `x`, with a non-simplified context as for `random`. This is what
#   a user who does not know the structure would fit.
# `tll`: pyvinecopulib with kernel-density margins and TLL pairs on the true
#   structure, ignoring `x` -- what leaving the covariate out costs.
# `mixture`: the equal-weight mixture of `mixture_k` non-simplified vines
#   sharing one set of fitted margins. The first member's structure is the
#   `selected` arm's; each other is a weighted random spanning-tree draw on
#   the same transforms, diverse but favoring strong pairs. A non-simplified
#   vine is consistent on any structure, so averaging over structures hedges
#   selecting one; by Jensen's inequality the mixture's KL is at most its
#   members' mean.
# `chain`: E11's baseline without a vine, the equal-weight mixture of
#   `chain_k` chain-rule copula densities (`npcc.experiments.chain`), each
#   along a uniformly random variable order and sharing the margins as the
#   mixture's members do. Factor k of an order is one Rosenblatt direction
#   conditioning on the earlier scores and `x` together.
# `gamvine-oracle`, `gamvine-selected`, `gamvine-ns`: E2b's GAM vines, fitted
#   in R by `gamvine_script` on mgcv margins with `x` a smooth term in every
#   pair -- on the true structure, on a selected one, and on a selected one
#   whose pairs also see their conditioning variables.
ARMS: tuple[str, ...] = (
  "oracle",
  "random",
  "selected",
  "tll",
  "mixture",
  "chain",
  "gamvine-oracle",
  "gamvine-selected",
  "gamvine-ns",
)
NPCC_ARMS: tuple[str, ...] = (
  "oracle",
  "random",
  "selected",
  "mixture",
  "chain",
)
GAMVINE_ARMS: tuple[str, ...] = (
  "gamvine-oracle",
  "gamvine-selected",
  "gamvine-ns",
)


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
  mixture_k: int = 5
  chain_k: int = 5
  #: The GAM vines' script, relative to the working directory unless
  #: absolute; read only when a GAM-vine arm runs.
  gamvine_script: str = "r_methods/fit_gamvine.R"
  #: SHA-256 of the script's content, read at construction when a GAM-vine
  #: arm runs.
  gamvine_script_sha256: str | None = field(default=None, init=False)

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
        f"estimators must be non-empty when an NPCC arm "
        f"({', '.join(NPCC_ARMS)}) runs"
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
    if self.mixture_k < 1:
      raise ValueError(f"mixture_k must be >= 1; got {self.mixture_k}")
    if self.chain_k < 1:
      raise ValueError(f"chain_k must be >= 1; got {self.chain_k}")
    if any(a in GAMVINE_ARMS for a in self.arms):
      script = Path(self.gamvine_script)
      if not script.is_file():
        raise ValueError(
          f"gamvine_script must name an existing file when a GAM-vine arm "
          f"({', '.join(GAMVINE_ARMS)}) runs; got {self.gamvine_script!r}, "
          f"resolved against {Path.cwd()}"
        )
      digest = hashlib.sha256(script.read_bytes()).hexdigest()
      self.gamvine_script_sha256 = digest

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
  "mixture_k": int,
  "chain_k": int,
  "gamvine_script": str,
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
