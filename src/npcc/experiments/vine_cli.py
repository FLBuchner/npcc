"""Command-line entry point: ``npcc-vinestudy --config vine.toml --out out/``.

Loads the vine grid from a TOML file, runs the study, and writes the raw
tables, their summaries, every ``(d, rep)`` truth, and an echo of the resolved
config to the output directory. Run-level flags match ``npcc-simstudy``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from npcc.experiments.config import RunConfig
from npcc.experiments.runner import _write_table
from npcc.experiments.vine_config import VineGridConfig, load_vine_grid
from npcc.experiments.vine_runner import (
  aggregate_vine_outputs,
  run_vine_study,
  truths,
)

logger = logging.getLogger("npcc.experiments")


def _build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    prog="npcc-vinestudy",
    description="Run the npcc Rosenblatt-vine simulation study.",
  )
  p.add_argument("--config", required=True, type=Path)
  p.add_argument("--out", required=True, type=Path)
  p.add_argument(
    "--device",
    default=None,
    help="torch device (default: cuda if available, else cpu).",
  )
  p.add_argument(
    "--workers",
    type=int,
    default=1,
    help="Number of data cells to run concurrently (threads).",
  )
  p.add_argument("--base-seed", type=int, default=317)
  p.add_argument(
    "--resume",
    action="store_true",
    help="Reuse completed per-cell checkpoints in --out and run only the rest.",
  )
  p.add_argument(
    "--gpu-mem-fraction",
    type=float,
    default=None,
    help="Cap this process's CUDA memory to a fraction of total VRAM (0-1].",
  )
  p.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
  )
  p.add_argument(
    "--format",
    dest="fmt",
    default="parquet",
    choices=["csv", "parquet"],
    help="Output table format (parquet needs pyarrow).",
  )
  return p


def _config_echo(
  grid: VineGridConfig, run: RunConfig, wall: float
) -> dict[str, Any]:
  return {
    "grid": {
      "dims": grid.dims,
      "n": grid.n,
      "n_rep": grid.n_rep,
      "arms": grid.arms,
      "estimators": [
        {"label": e.label, "estimator_id": e.estimator_id, **e.canonical_dict()}
        for e in grid.estimators
      ],
      "eval_x_n": grid.eval_x_n,
      "eval_m": grid.eval_m,
      "projection_grid_size": grid.projection_grid_size,
    },
    "run": {
      "device": run.device,
      "workers": run.workers,
      "base_seed": run.base_seed,
      "fmt": run.fmt,
      "gpu_mem_fraction": run.gpu_mem_fraction,
    },
    "wall_seconds": wall,
  }


def main(argv: list[str] | None = None) -> int:
  os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

  args = _build_parser().parse_args(argv)
  logging.basicConfig(
    level=args.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )

  grid = load_vine_grid(args.config)
  run = RunConfig(
    out=args.out,
    device=args.device,
    workers=args.workers,
    base_seed=args.base_seed,
    log_level=args.log_level,
    fmt=args.fmt,
    gpu_mem_fraction=args.gpu_mem_fraction,
  )
  run.out.mkdir(parents=True, exist_ok=True)

  metric_df, runtime_df, wall = run_vine_study(grid, run, resume=args.resume)
  for name, df in aggregate_vine_outputs(metric_df, runtime_df).items():
    _write_table(df, run.out / name, run.fmt)
  (run.out / "truths.json").write_text(
    json.dumps(truths(grid, run.base_seed), indent=2)
  )
  (run.out / "config.json").write_text(
    json.dumps(_config_echo(grid, run, wall), indent=2)
  )

  logger.info("Wrote vine-study result tables to %s", run.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
