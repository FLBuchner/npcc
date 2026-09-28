"""Simulation-study tooling for npcc.

Requires the ``experiments`` extra (``pandas``, ``matplotlib``). The core
package (``import npcc``) does not import this subpackage, so it stays light.

Typical use::

    from npcc.experiments import GridConfig, RunConfig, run_study
    metric, quantities, diagnostics, runtime, wall = run_study(grid, run)

or via the CLI: ``npcc-simstudy --config study.toml --out results/``.
"""

from __future__ import annotations

from npcc.experiments.config import GridConfig, RunConfig, load_grid
from npcc.experiments.runner import (
  aggregate_results,
  aggregate_study_outputs,
  run_study,
)
from npcc.experiments.vine_config import VineGridConfig, load_vine_grid
from npcc.experiments.vine_runner import aggregate_vine_outputs, run_vine_study

__all__ = [
  "GridConfig",
  "RunConfig",
  "VineGridConfig",
  "aggregate_results",
  "aggregate_study_outputs",
  "aggregate_vine_outputs",
  "load_grid",
  "load_vine_grid",
  "run_study",
  "run_vine_study",
]
