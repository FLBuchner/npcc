"""Command-line entry point: ``npcc-application <stage> --config app.toml``.

The stages, in order: ``fetch`` downloads and verifies the data;
``preflight`` checks what a run needs and fits nothing; ``margins`` and
``copulas`` fit and store; ``evaluate`` writes the aggregate tables. See
:mod:`npcc.experiments.application.runner`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from npcc.experiments._study import _shard
from npcc.experiments.application.config import load_config
from npcc.experiments.application.data import fetch_data
from npcc.experiments.application.runner import (
  RunOptions,
  evaluate,
  preflight,
  run_copulas,
  run_margins,
)

logger = logging.getLogger("npcc.experiments")

_STAGES = {
  "fetch": "download the data into the configured folder and verify it",
  "preflight": "check the data, the estimators and the copula stage",
  "margins": "fit and store the marginal models",
  "copulas": "fit and store the copula arms on the stored margins",
  "evaluate": "write the aggregate tables of a run folder",
}


def _stations(values: list[str] | None) -> tuple[int, ...] | None:
  """``--stations``: ids, or ``all`` for the configuration's choice."""
  if values is None or values == ["all"]:
    return None
  try:
    stations = tuple(int(v) for v in values)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(
      f"stations must be integer ids or 'all'; got {values}"
    ) from exc
  if any(s < 0 for s in stations):
    raise argparse.ArgumentTypeError(
      f"stations must be ids >= 0; got {list(stations)}"
    )
  return stations


def build_parser() -> argparse.ArgumentParser:
  """The ``npcc-application`` argument parser.

  Returns
  -------
  argparse.ArgumentParser
      One subcommand per stage.
  """
  parser = argparse.ArgumentParser(
    prog="npcc-application",
    description="Run the weather application: margins, then copulas.",
  )
  stages = parser.add_subparsers(dest="stage", required=True)
  for stage, help_text in _STAGES.items():
    sub = stages.add_parser(stage, help=help_text, description=help_text)
    sub.add_argument(
      "--config", required=True, type=Path, help="The application TOML."
    )
    if stage == "fetch":
      sub.add_argument(
        "--verify-only",
        action="store_true",
        help="Check the files already in the data folder; download nothing.",
      )
      sub.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
      )
      continue
    sub.add_argument(
      "--out",
      required=stage != "preflight",
      type=Path,
      default=None,
      help="The run folder.",
    )
    sub.add_argument(
      "--stations",
      nargs="+",
      default=None,
      help="Station ids, or 'all'; the configuration's by default.",
    )
    sub.add_argument(
      "--models",
      nargs="+",
      default=None,
      help="Marginal models, from the configuration's; all by default.",
    )
    sub.add_argument(
      "--copulas",
      nargs="+",
      default=None,
      help="Copula arms, from the configuration's; all by default.",
    )
    sub.add_argument(
      "--log-level",
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    if stage in ("margins", "copulas"):
      sub.add_argument(
        "--shard",
        type=_shard,
        default=None,
        metavar="I/K",
        help=(
          "Run every K-th station from the I-th; K such processes split the "
          "stations between them."
        ),
      )
      sub.add_argument(
        "--resume",
        action="store_true",
        help="Skip what an earlier run of this folder stored.",
      )
      sub.add_argument(
        "--device",
        default=None,
        help="The device of every backend, over the configuration's.",
      )
  return parser


def main(argv: list[str] | None = None) -> int:
  """Run one stage from the command line.

  Parameters
  ----------
  argv : list of str, optional
      The arguments; ``None`` reads them from ``sys.argv``.

  Returns
  -------
  int
      The exit status.
  """
  parser = build_parser()
  args = parser.parse_args(argv)
  logging.basicConfig(
    level=args.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )
  try:
    stations = _stations(getattr(args, "stations", None))
  except argparse.ArgumentTypeError as exc:
    parser.error(str(exc))
  config = load_config(args.config)
  if args.stage == "fetch":
    try:
      digests = fetch_data(config.data_dir, verify_only=args.verify_only)
    except (OSError, RuntimeError, ValueError) as exc:
      print(f"npcc-application fetch: {exc}", file=sys.stderr)
      return 1
    for name, digest in digests.items():
      print(f"{config.data_dir / name}: sha256 {digest} (verified)")
    return 0
  options = RunOptions(
    out=args.out,
    stations=stations,
    models=None if args.models is None else tuple(args.models),
    arms=None if args.copulas is None else tuple(args.copulas),
    shard=getattr(args, "shard", None),
    resume=getattr(args, "resume", False),
    device=getattr(args, "device", None),
  )
  match args.stage:
    case "preflight":
      for line in preflight(config, options):
        print(line)
      print("preflight passed")
    case "margins":
      run_margins(config, options)
    case "copulas":
      run_copulas(config, options)
    case "evaluate":
      for name in evaluate(config, options):
        print(args.out / "tables" / f"{name}.csv")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
