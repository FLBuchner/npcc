"""The harness both studies run on: checkpointed sweeps, tables, the CLI.

A study is a list of cells, each of which computes a fixed set of row tables
from its own seeds. :func:`sweep` runs the pending ones, checkpoints each cell's
rows as a shard the moment it finishes, and reads finished shards back on a
resume, after the study's signature has been checked against the manifest in
the output directory. What a cell computes, and how its signature is hashed,
is each study's own.

Importing this module sets pandas' ``future.infer_string`` off for the
process. The experiment tables mix strings and numbers, and Arrow-backed string
inference has crashed building and sorting them on Windows; object columns are
what every table here is written and read back with.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, TypeVar, cast

import pandas as pd
import torch

from npcc.experiments.config import GridConfig, RunConfig
from npcc.experiments.vine_config import VineGridConfig

logger = logging.getLogger(__name__)

pd.options.future.infer_string = False

Rows = list[dict[str, Any]]

SUMMARY_STATS: tuple[str, ...] = (
  "rep_mean",
  "rep_std",
  "rep_median",
  "rep_p05",
  "rep_p95",
  "rep_max",
)


class StudyCell(Protocol):
  """A cell of a study: hashable, and named by the directory it shards to."""

  @property
  def key(self) -> str:
    """The cell's shard directory name, unique within its study."""
    ...


CellT = TypeVar("CellT", bound=StudyCell)


def derive_seed(*parts: object) -> int:
  """Deterministic 31-bit seed hashed from ``parts``.

  Parameters
  ----------
  *parts : object
      Salt, joined with ``":"`` in order.

  Returns
  -------
  int
      A seed in ``[0, 2**31)``.
  """
  payload = ":".join(str(p) for p in parts).encode()
  return (
    int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF
  )


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def release_gpu() -> None:
  """Release cached CUDA memory between estimators and cells.

  A loop that reassigns its fitted model each iteration otherwise leaves the
  freed backend modules in the CUDA caching allocator, and peak memory creeps
  up across a cell's estimators. ``empty_cache`` only returns *unused* blocks
  to the driver, so it is safe under any worker count, and a no-op on CPU.
  """
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()


def reset_gpu_peaks() -> None:
  """Start a fresh peak-memory window, when there is a GPU to measure."""
  if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()


def memory_usage() -> dict[str, float]:
  """The four memory columns of a runtime row.

  GPU peaks since the last :func:`reset_gpu_peaks` (``0`` without CUDA), and
  the process RSS and system ``MemAvailable`` from ``/proc`` (``0`` where it is
  absent). A headless run that wedges the whole box over SSH is usually
  system-RAM exhaustion, not a GPU OOM, which is why the system figure is
  logged with every estimator: a creeping leak shows up before it kills the
  machine.

  Returns
  -------
  dict of str to float
      ``gpu_peak_reserved_mb``, ``gpu_peak_alloc_mb``, ``rss_mb`` and
      ``mem_available_mb``, all in MiB.
  """
  reserved = alloc = 0.0
  if torch.cuda.is_available():
    reserved = torch.cuda.max_memory_reserved() / 1024**2
    alloc = torch.cuda.max_memory_allocated() / 1024**2

  rss = avail = 0.0
  try:
    for line in Path("/proc/self/status").read_text().splitlines():
      if line.startswith("VmRSS:"):
        rss = float(line.split()[1]) / 1024.0
        break
    for line in Path("/proc/meminfo").read_text().splitlines():
      if line.startswith("MemAvailable:"):
        avail = float(line.split()[1]) / 1024.0
        break
  except OSError:
    pass

  return {
    "gpu_peak_reserved_mb": reserved,
    "gpu_peak_alloc_mb": alloc,
    "rss_mb": rss,
    "mem_available_mb": avail,
  }


# ---------------------------------------------------------------------------
# Tables and shards
# ---------------------------------------------------------------------------


def write_table(df: pd.DataFrame, base: Path, fmt: str) -> None:
  """Write ``df`` to ``base`` with the ``.parquet``/``.csv`` suffix for ``fmt``.

  Parameters
  ----------
  df : pandas.DataFrame
      The table.
  base : pathlib.Path
      Its path, without a suffix.
  fmt : str
      ``"parquet"`` or ``"csv"``.
  """
  if fmt == "parquet":
    df.to_parquet(base.with_suffix(".parquet"), index=False)
  else:
    df.to_csv(base.with_suffix(".csv"), index=False)


def read_table(base: Path, fmt: str) -> pd.DataFrame:
  """Read a table written by :func:`write_table`; an empty frame if absent.

  Parameters
  ----------
  base : pathlib.Path
      Its path, without a suffix.
  fmt : str
      ``"parquet"`` or ``"csv"``.

  Returns
  -------
  pandas.DataFrame
      The table.
  """
  path = base.with_suffix(".parquet" if fmt == "parquet" else ".csv")
  if not path.exists():
    return pd.DataFrame()
  if fmt == "parquet":
    return pd.read_parquet(path)
  # The default parser is not exact for float64, so a resumed run would
  # differ from a fresh one in the last digits of every metric.
  return pd.read_csv(path, float_precision="round_trip")


def write_tables(
  tables: Mapping[str, pd.DataFrame], out: Path, fmt: str
) -> None:
  """Write every table of ``tables`` under ``out``, named by its key.

  Parameters
  ----------
  tables : mapping of str to pandas.DataFrame
      The tables, by file stem.
  out : pathlib.Path
      The output directory.
  fmt : str
      ``"parquet"`` or ``"csv"``.
  """
  for name, df in tables.items():
    write_table(df, out / name, fmt)


def _write_shard(
  cell_dir: Path, tables: Sequence[str], rows: Sequence[Rows], fmt: str
) -> None:
  """Persist one cell's row tables, then a ``DONE`` marker written last.

  The marker-last ordering makes the shard atomic: a crash mid-write leaves no
  ``DONE``, so the cell reads as pending and recomputes.
  """
  cell_dir.mkdir(parents=True, exist_ok=True)
  for name, cell_rows in zip(tables, rows, strict=True):
    if cell_rows:
      write_table(pd.DataFrame(cell_rows), cell_dir / name, fmt)
  (cell_dir / "DONE").write_text("")


def _read_shard(cell_dir: Path, tables: Sequence[str], fmt: str) -> list[Rows]:
  return [
    cast("Rows", read_table(cell_dir / name, fmt).to_dict("records"))
    for name in tables
  ]


def guard_manifest(
  out: Path, signature: str, *, study: str, resume: bool
) -> None:
  """Write the study signature, or on a resume, check it first.

  Parameters
  ----------
  out : pathlib.Path
      The output directory holding ``manifest.json``.
  signature : str
      The study's hash of everything that changes a cell's output.
  study : str
      The study's name, for the message.
  resume : bool
      Whether completed cells are about to be reused.

  Raises
  ------
  ValueError
      If resuming over a checkpoint written under another signature.
  """
  path = out / "manifest.json"
  if resume and path.exists():
    previous = json.loads(path.read_text()).get("signature")
    if previous != signature:
      raise ValueError(
        f"the {study} study's grid signature must match the checkpoint in "
        f"{out} to resume; use a fresh --out or drop --resume"
      )
  path.write_text(json.dumps({"signature": signature}, indent=2))


def sweep(
  cells: Sequence[CellT],
  compute: Callable[[CellT], Sequence[Rows]],
  *,
  tables: Sequence[str],
  run: RunConfig,
  signature: str,
  study: str,
  resume: bool,
) -> tuple[list[Rows], float]:
  """Run every cell of a study, checkpointing each as it finishes.

  Each cell's rows go to ``run.out/cells/<cell.key>/`` the moment it finishes,
  so an interruption loses at most the cells still running. With ``resume``,
  cells already marked done are read back from their shards and skipped, once
  the signature has been checked against the manifest so a changed grid cannot
  silently reuse them.

  Parameters
  ----------
  cells : sequence of StudyCell
      The study's cells, in a fixed order.
  compute : callable
      Computes one cell's rows: one list per entry of ``tables``.
  tables : sequence of str
      The names of the row tables every cell produces.
  run : RunConfig
      Output directory, format, worker count and GPU memory cap.
  signature : str
      The study's resume signature.
  study : str
      The study's name, for logs and messages.
  resume : bool
      Whether to reuse completed cells.

  Returns
  -------
  rows : list of list of dict
      Every cell's rows, one list per table.
  wall : float
      Seconds spent computing the pending cells.
  """
  cells_root = run.out / "cells"
  cells_root.mkdir(parents=True, exist_ok=True)
  guard_manifest(run.out, signature, study=study, resume=resume)

  if run.gpu_mem_fraction is not None and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(run.gpu_mem_fraction)
    logger.info(
      "Capped CUDA memory to %.2f of total (headroom for the display server)",
      run.gpu_mem_fraction,
    )

  if run.shard is not None:
    index, count = run.shard
    cells = [c for position, c in enumerate(cells) if position % count == index]

  done = {c for c in cells if resume and (cells_root / c.key / "DONE").exists()}
  pending = [c for c in cells if c not in done]
  logger.info(
    "%s study: %d cells (%d pending, %d resumed)",
    study,
    len(cells),
    len(pending),
    len(done),
  )

  out: list[Rows] = [[] for _ in tables]

  def accumulate(cell_rows: Sequence[Rows]) -> None:
    for acc, rows in zip(out, cell_rows, strict=True):
      acc.extend(rows)

  for cell in cells:
    if cell in done:
      accumulate(_read_shard(cells_root / cell.key, tables, run.fmt))

  t0 = perf_counter()

  def do(cell: CellT) -> Sequence[Rows]:
    logger.debug("cell start: %s", cell)
    cell_rows = compute(cell)
    _write_shard(cells_root / cell.key, tables, cell_rows, run.fmt)
    logger.info("cell done: %s", cell)
    return cell_rows

  if run.workers <= 1:
    for cell in pending:
      accumulate(do(cell))
  elif pending:
    max_workers = min(run.workers, len(pending), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
      futures = [pool.submit(do, cell) for cell in pending]
      for fut in as_completed(futures):
        accumulate(fut.result())

  wall = perf_counter() - t0
  logger.info("%s study finished in %.1fs", study, wall)
  return out, wall


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summary_stats(
  df: pd.DataFrame, group: list[str], value_col: str
) -> pd.DataFrame:
  """Summarize ``value_col`` over repetitions within each ``group``.

  Parameters
  ----------
  df : pandas.DataFrame
      One row per repetition (and whatever else ``group`` does not name).
  group : list of str
      The columns that identify a summary row.
  value_col : str
      The column summarized.

  Returns
  -------
  pandas.DataFrame
      ``group`` plus one column per :data:`SUMMARY_STATS`; the quantiles skip
      missing values.
  """
  stats = df.groupby(group, as_index=False, dropna=False).agg(
    rep_mean=(value_col, "mean"),
    rep_std=(value_col, "std"),
    rep_median=(value_col, "median"),
    rep_p05=(value_col, lambda s: s.quantile(0.05)),
    rep_p95=(value_col, lambda s: s.quantile(0.95)),
    rep_max=(value_col, "max"),
  )
  return cast("pd.DataFrame", stats)


def summarize(
  df: pd.DataFrame,
  metrics: Sequence[str],
  group: Sequence[str],
  *,
  x_axes: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Fixed-``x`` metrics summarized over repetitions, per ``x`` and over it.

  Parameters
  ----------
  df : pandas.DataFrame
      One row per ``(group, rep, x)``, one column per metric.
  metrics : sequence of str
      The metric columns.
  group : sequence of str
      The columns that identify an estimate.
  x_axes : sequence of str
      The columns that identify an evaluation ``x``.

  Returns
  -------
  by_x : pandas.DataFrame
      Per ``(group, x_axes, metric)``: the metric summarized over repetitions.
  over_x : pandas.DataFrame
      Per ``(group, metric)``: the metric averaged over ``x`` within each
      repetition, then summarized over repetitions.
  """
  by_x: list[pd.DataFrame] = []
  over_x: list[pd.DataFrame] = []
  for metric in metrics:
    g = summary_stats(df, [*group, *x_axes], metric)
    g["metric"] = metric
    by_x.append(g)
    averaged = df.groupby([*group, "rep"], as_index=False, dropna=False).agg(
      value=(metric, "mean")
    )
    g = summary_stats(averaged, list(group), "value")
    g["metric"] = metric
    over_x.append(g)
  return pd.concat(by_x, ignore_index=True), pd.concat(
    over_x, ignore_index=True
  )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _shard(text: str) -> tuple[int, int]:
  """Parse ``--shard I/K`` into ``(I, K)``."""
  index, sep, count = text.partition("/")
  if not sep or not index.isdigit() or not count.isdigit():
    raise argparse.ArgumentTypeError(
      f"shard must be I/K with whole numbers; got {text!r}"
    )
  return int(index), int(count)


def study_parser(prog: str, description: str) -> argparse.ArgumentParser:
  """The argument parser both study commands share.

  Parameters
  ----------
  prog : str
      The command's name.
  description : str
      What it runs.

  Returns
  -------
  argparse.ArgumentParser
      ``--config`` and ``--out``, plus the run-level flags.
  """
  p = argparse.ArgumentParser(prog=prog, description=description)
  p.add_argument(
    "--config",
    required=True,
    type=Path,
    help="TOML file with a [grid] table defining the sweep.",
  )
  p.add_argument(
    "--out",
    required=True,
    type=Path,
    help="Output directory for the result tables.",
  )
  p.add_argument(
    "--device",
    default=None,
    help="torch device (default: auto-select cuda if available, else cpu).",
  )
  p.add_argument(
    "--workers",
    type=int,
    default=1,
    help="Number of data cells to run concurrently (threads).",
  )
  p.add_argument(
    "--shard",
    type=_shard,
    default=None,
    metavar="I/K",
    help=(
      "Compute only every K-th cell, starting at the I-th, into --out, and "
      "stop before aggregating. Run K of these as separate processes (with "
      "--resume), then the same command once more without --shard to write "
      "the tables. Processes run in parallel where threads cannot: the cells "
      "hold the interpreter lock for much of their work."
    ),
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
    help=(
      "Cap this process's CUDA memory to a fraction of total VRAM (0-1]. "
      "Leaves headroom for a display server sharing the GPU: a peak that "
      "would starve it raises a clean OOM instead of hanging the machine."
    ),
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


def start(args: argparse.Namespace) -> RunConfig:
  """Configure the process for a parsed study command line.

  The CUDA allocator is left at torch's default. Expandable segments would
  reduce fragmentation across many per-estimator fits, but they are built on
  CUDA's virtual-memory API, which a virtualized GPU may not offer -- and there
  every allocation fails. A study peaks at a few GB, so on a GPU that has the
  API, ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` is the caller's to
  export.

  Every process appends one line to ``provenance.jsonl`` in the output
  directory: when it started, its parsed options and :func:`code_provenance`.
  The result tables carry no commit, and the shards of one run are separate
  processes, any of which may have started from another checkout.

  Parameters
  ----------
  args : argparse.Namespace
      Parsed by a :func:`study_parser`.

  Returns
  -------
  RunConfig
      The run-level options, with the output directory created.
  """
  logging.basicConfig(
    level=args.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )
  run = RunConfig(
    out=args.out,
    device=args.device,
    workers=args.workers,
    base_seed=args.base_seed,
    fmt=args.fmt,
    gpu_mem_fraction=args.gpu_mem_fraction,
    shard=args.shard,
  )
  run.out.mkdir(parents=True, exist_ok=True)
  record = {
    "started": datetime.now().astimezone().isoformat(timespec="seconds"),
    "options": vars(args),
    **code_provenance(),
  }
  with (run.out / "provenance.jsonl").open("a") as f:
    f.write(json.dumps(record, default=str) + "\n")
  return run


def code_provenance(
  module: Path = Path(__file__),
) -> dict[str, str | list[str] | None]:
  """The npcc checkout a process runs: its commit and its uncommitted files.

  Both are ``None`` unless ``module`` sits at ``src/npcc/experiments`` of a git
  checkout's work tree -- an installed wheel, say, records nothing rather than
  the commit of whatever repository happens to enclose it.

  Parameters
  ----------
  module : pathlib.Path, optional
      A file of the package; this module by default.

  Returns
  -------
  dict
      ``npcc_commit``, the checkout's ``HEAD``, and ``uncommitted``, the
      tracked files under ``src``, ``configs`` and ``r_methods`` that differ
      from it.
  """
  root = module.resolve().parents[3]

  def git(*command: str) -> str | None:
    try:
      done = subprocess.run(
        ["git", "-C", str(root), *command],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
      )
    except (OSError, subprocess.SubprocessError):
      return None
    return done.stdout.strip()

  top = git("rev-parse", "--show-toplevel")
  if top is None or Path(top).resolve() != root:
    return {"npcc_commit": None, "uncommitted": None}
  status = git(
    "status",
    "--porcelain",
    "--untracked-files=no",
    "--",
    "src",
    "configs",
    "r_methods",
  )
  return {
    "npcc_commit": git("rev-parse", "HEAD"),
    "uncommitted": None
    if status is None
    else [line[3:] for line in status.splitlines()],
  }


def sorted_frame(
  rows: Sequence[Mapping[str, object]], by: list[str]
) -> pd.DataFrame:
  """Rows as a table sorted on ``by``, or an empty table when there are none.

  A shard can own no cells -- a run with more shards than cells -- and an
  empty table has none of the columns there are to sort on.

  Parameters
  ----------
  rows : sequence of mapping
      One mapping per row.
  by : list of str
      The columns to sort on, in order.

  Returns
  -------
  pandas.DataFrame
      The rows, sorted and renumbered from 0.
  """
  frame = pd.DataFrame(rows)
  if frame.empty:
    return frame
  return frame.sort_values(by).reset_index(drop=True)


def config_echo(
  grid: GridConfig | VineGridConfig, run: RunConfig, wall: float
) -> dict[str, Any]:
  """The resolved configuration of a finished run, as written to JSON.

  Parameters
  ----------
  grid : GridConfig or VineGridConfig
      The swept grid.
  run : RunConfig
      The run-level options.
  wall : float
      Seconds the sweep took.

  Returns
  -------
  dict
      ``grid`` (every field, each estimator with its ``estimator_id``),
      ``run`` (every option but the output directory) and ``wall_seconds``.
  """
  grid_echo: dict[str, Any] = {
    f.name: getattr(grid, f.name) for f in fields(grid)
  }
  grid_echo["estimators"] = [
    {"label": e.label, "estimator_id": e.estimator_id, **e.canonical_dict()}
    for e in grid.estimators
  ]
  return {
    "grid": grid_echo,
    "run": {
      f.name: getattr(run, f.name) for f in fields(run) if f.name != "out"
    },
    "wall_seconds": wall,
  }
