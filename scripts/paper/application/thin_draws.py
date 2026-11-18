"""Thin a run's stored copula draws to the first ``k`` per test case, in place.

The fallback for a run whose disk fills: the copula stage stores every arm's
draws (``draws.parquet`` in each copula unit, ``[copulas] save_draws``), and
this keeps only the draws numbered ``0`` to ``k - 1`` of each test case for
the arms named. Every score was computed from all the draws before they were
stored, so no score changes; the thinned file no longer reproduces the
sample scores, and says so.

For each unit thinned, the file is rewritten beside itself and renamed over
it, ``draws_thinned.json`` in the unit records what was kept, and a line is
appended to ``<run>/logs/draws_thinned.jsonl``. A unit already holding at
most ``k`` draws per case is left as it is and logged as such.

Usage::

    python scripts/paper/application/thin_draws.py --run <run folder> \\
        --arms gaussian boostcopula-all --keep 50 \\
        [--models lightgbm-all ...] [--stations 0 1 ...]
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

DRAWS = "draws.parquet"
RECORD = "draws_thinned.json"
LOG = Path("logs") / "draws_thinned.jsonl"


def thin_unit(unit: Path, keep: int) -> dict[str, object] | None:
  """Keep a copula unit's first ``keep`` draws per test case.

  Parameters
  ----------
  unit : pathlib.Path
      ``<run>/stations/<NNN>/copulas/<model>/<arm>``.
  keep : int
      The draws kept per test case, at least 2.

  Returns
  -------
  dict or None
      What was done, as written to the logs; ``None`` for a unit that
      stores no draws.
  """
  if keep < 2:
    raise ValueError(f"keep must be at least 2, as a score needs; got {keep}")
  path = unit / DRAWS
  if not path.is_file():
    return None
  draws = pd.read_parquet(path)
  before_rows, before_bytes = len(draws), path.stat().st_size
  record: dict[str, object] = {
    "unit": str(unit),
    "keep": keep,
    "rows_before": before_rows,
    "bytes_before": before_bytes,
    "when": datetime.now(UTC).isoformat(timespec="seconds"),
  }
  if draws.empty or int(draws["draw"].max()) < keep:
    return {**record, "action": "left as it is: at most keep draws per case"}
  kept = draws.loc[draws["draw"] < keep].reset_index(drop=True)
  temporary = unit / f".{DRAWS}.partial"
  kept.to_parquet(temporary, index=False, compression="zstd")
  os.replace(temporary, path)
  record.update(
    action="thinned",
    rows_after=len(kept),
    bytes_after=path.stat().st_size,
    note=(
      "the unit's scores were computed from every draw before thinning; "
      "the stored draws no longer reproduce its sample scores"
    ),
  )
  (unit / RECORD).write_text(json.dumps(record, indent=2) + "\n")
  return record


def thin(
  run: Path,
  arms: Sequence[str],
  keep: int,
  *,
  models: Sequence[str] | None = None,
  stations: Sequence[int] | None = None,
) -> list[dict[str, object]]:
  """Thin every stored unit of the named arms, and log each.

  Parameters
  ----------
  run : pathlib.Path
      The run folder.
  arms : sequence of str
      The arms as stored, such as ``gaussian`` or ``gaussian@5d``.
  keep : int
      The draws kept per test case.
  models : sequence of str, optional
      Only these marginal models; every one when omitted.
  stations : sequence of int, optional
      Only these stations; every one when omitted.

  Returns
  -------
  list of dict
      One record per unit that stores draws.
  """
  root = Path(run) / "stations"
  if not root.is_dir():
    raise FileNotFoundError(f"{run} must hold stations/; it has none")
  wanted = None if stations is None else {f"{int(s):03d}" for s in stations}
  records = []
  for station in sorted(p for p in root.iterdir() if p.name.isdigit()):
    if wanted is not None and station.name not in wanted:
      continue
    copulas = station / "copulas"
    if not copulas.is_dir():
      continue
    for model in sorted(p for p in copulas.iterdir() if p.is_dir()):
      if models is not None and model.name not in models:
        continue
      for arm in arms:
        record = thin_unit(model / arm, keep)
        if record is not None:
          records.append(record)
  log = Path(run) / LOG
  log.parent.mkdir(parents=True, exist_ok=True)
  with log.open("a") as fh:
    for record in records:
      fh.write(json.dumps(record) + "\n")
  return records


def main(argv: list[str] | None = None) -> int:
  """Run :func:`thin` from the command line.

  Parameters
  ----------
  argv : list of str, optional
      The arguments; ``None`` reads them from ``sys.argv``.

  Returns
  -------
  int
      The exit status.
  """
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--run", required=True, type=Path, help="The run folder.")
  p.add_argument("--arms", required=True, nargs="+", help="The arms to thin.")
  p.add_argument(
    "--keep", required=True, type=int, help="Draws kept per test case."
  )
  p.add_argument("--models", nargs="+", default=None, help="Only these.")
  p.add_argument(
    "--stations", type=int, nargs="+", default=None, help="Only these."
  )
  a = p.parse_args(argv)
  records = thin(a.run, a.arms, a.keep, models=a.models, stations=a.stations)
  thinned = [r for r in records if r["action"] == "thinned"]
  saved = sum(
    int(str(r["bytes_before"])) - int(str(r["bytes_after"])) for r in thinned
  )
  print(
    f"{len(thinned)} units thinned to {a.keep} draws per case, "
    f"{len(records) - len(thinned)} left as they were; {saved / 1e6:.1f} MB "
    f"freed; logged in {a.run / LOG}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
