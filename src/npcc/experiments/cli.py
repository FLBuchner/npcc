"""Command-line entry point: ``npcc-simstudy --config study.toml --out results/``.

Loads the experiment grid from a TOML file, runs the study, and writes the
raw result tables, paper-facing summaries, and an echo of the resolved
config to the output directory.
"""

from __future__ import annotations

import argparse
import json
import logging

from npcc.experiments._study import (
  config_echo,
  start,
  study_parser,
  write_tables,
)
from npcc.experiments.config import load_grid
from npcc.experiments.runner import aggregate_study_outputs, run_study

logger = logging.getLogger("npcc.experiments")


def _build_parser() -> argparse.ArgumentParser:
  return study_parser(
    "npcc-simstudy",
    "Run the npcc Rosenblatt pair-copula simulation study.",
  )


def main(argv: list[str] | None = None) -> int:
  """Run the bivariate study from the command line.

  Parameters
  ----------
  argv : list of str, optional
      The arguments; ``None`` reads them from ``sys.argv``.

  Returns
  -------
  int
      The exit status.
  """
  args = _build_parser().parse_args(argv)
  run = start(args)
  grid = load_grid(args.config)

  metric_df, quantity_df, diagnostic_df, runtime_df, wall = run_study(
    grid, run, resume=args.resume
  )
  outputs = aggregate_study_outputs(metric_df, diagnostic_df, runtime_df)
  write_tables(
    {
      # The headline table, under the name the run handoff documents.
      "summary": outputs["summary_by_x"],
      **outputs,
      "diagnostics": diagnostic_df,
      "quantities": quantity_df,
      "runtime": runtime_df,
    },
    run.out,
    run.fmt,
  )
  (run.out / "config.json").write_text(
    json.dumps(config_echo(grid, run, wall), indent=2)
  )

  logger.info("Wrote simulation-study result tables to %s", run.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
