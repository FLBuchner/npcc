"""Command-line entry point: ``npcc-covstudy --config e9.toml --out out/``.

Loads the covariate grid (E9) from a TOML file, runs the study, and writes the
raw tables, their summaries, every scenario's covariate points, and an echo of
the resolved config to the output directory. Run-level flags match
``npcc-simstudy``.
"""

from __future__ import annotations

import json
import logging

from npcc.experiments._study import (
  config_echo,
  start,
  study_parser,
  write_table,
  write_tables,
)
from npcc.experiments.covariate_config import load_covariate_grid
from npcc.experiments.covariate_runner import (
  aggregate_covariate_outputs,
  points_table,
  run_covariate_study,
)

logger = logging.getLogger("npcc.experiments")


def main(argv: list[str] | None = None) -> int:
  """Run the covariate study from the command line.

  Parameters
  ----------
  argv : list of str, optional
      The arguments; ``None`` reads them from ``sys.argv``.

  Returns
  -------
  int
      The exit status.
  """
  parser = study_parser(
    "npcc-covstudy",
    "Run the npcc covariate study (E9): fifteen covariates of mixed type.",
  )
  args = parser.parse_args(argv)
  run = start(args)
  grid = load_covariate_grid(args.config)

  metric_df, runtime_df, wall = run_covariate_study(
    grid, run, resume=args.resume
  )
  if run.shard is not None:
    logger.info(
      "Shard %d/%d done; run again without --shard to write the tables",
      *run.shard,
    )
    return 0
  write_tables(
    aggregate_covariate_outputs(metric_df, runtime_df), run.out, run.fmt
  )
  write_table(points_table(grid, run.base_seed), run.out / "points", run.fmt)
  (run.out / "config.json").write_text(
    json.dumps(config_echo(grid, run, wall), indent=2)
  )
  logger.info("Wrote covariate-study result tables to %s", run.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
