"""Command-line entry point: ``npcc-vinestudy --config vine.toml --out out/``.

Loads the vine grid from a TOML file, runs the study, and writes the raw
tables, their summaries, every ``(d, rep)`` truth, and an echo of the resolved
config to the output directory. Run-level flags match ``npcc-simstudy``.
"""

from __future__ import annotations

import json
import logging

from npcc.experiments._study import (
  config_echo,
  start,
  study_parser,
  write_tables,
)
from npcc.experiments.vine_config import load_vine_grid
from npcc.experiments.vine_runner import (
  aggregate_vine_outputs,
  run_vine_study,
  truths,
)

logger = logging.getLogger("npcc.experiments")


def main(argv: list[str] | None = None) -> int:
  """Run the vine study from the command line.

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
    "npcc-vinestudy", "Run the npcc Rosenblatt-vine simulation study."
  )
  args = parser.parse_args(argv)
  run = start(args)
  grid = load_vine_grid(args.config)

  metric_df, runtime_df, wall = run_vine_study(grid, run, resume=args.resume)
  write_tables(aggregate_vine_outputs(metric_df, runtime_df), run.out, run.fmt)
  (run.out / "truths.json").write_text(
    json.dumps(truths(grid, run.base_seed), indent=2)
  )
  (run.out / "config.json").write_text(
    json.dumps(config_echo(grid, run, wall), indent=2)
  )

  logger.info("Wrote vine-study result tables to %s", run.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
