"""Each station's surface-pressure error, and the largest step in it.

The error is the observed surface pressure less the forecast's ensemble
mean, per case, in hPa. A station whose barometer is moved or recalibrated
shows a step in it that no covariate explains. Per station this writes:

- the error's mean and standard deviation on the training and test cases,
  and ``shift``, the test mean less the training mean;
- the largest single step in the error over all the station's cases in time
  order: the split maximizing the two-sample statistic
  ``|mean_after - mean_before| sqrt(n_before n_after / n)`` with at least
  ``--min-side`` cases on each side, the time of the first case after it,
  the means on either side, ``step`` (after less before), and how many
  training cases follow it.

Reads the cases a run stored (``stations/<id>/data/cases.parquet``) and fits
nothing. Writes ``pressure_shift.csv`` into ``--out`` (``<run>/checks`` by
default) and prints the stations with the largest ``|shift|``.

Usage::

    python scripts/paper/application/pressure_shift.py --run <run folder>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from npcc.experiments.application.store import read_station_data

OBSERVED = "obs:surface_air_pressure"
FORECAST = "coe:surface_air_pressure_ensavg"


def largest_step(errors: np.ndarray, min_side: int) -> int:
  """Where the largest step in ``errors`` falls.

  Parameters
  ----------
  errors : numpy.ndarray, shape (n,)
      The errors, in time order.
  min_side : int
      The fewest cases allowed on either side.

  Returns
  -------
  int
      The index of the first case after the step.

  Raises
  ------
  ValueError
      If ``errors`` has fewer than ``2 * min_side`` cases.
  """
  n = len(errors)
  if n < 2 * min_side:
    raise ValueError(
      f"errors must hold at least 2 * min_side = {2 * min_side} cases to "
      f"place a step; got {n}"
    )
  before = np.arange(min_side, n - min_side + 1)
  sums = np.cumsum(errors)
  total = sums[-1]
  head = sums[before - 1] / before
  tail = (total - sums[before - 1]) / (n - before)
  stat = np.abs(tail - head) * np.sqrt(before * (n - before) / n)
  return int(before[np.argmax(stat)])


def station_row(run: Path, station: int, min_side: int) -> dict[str, object]:
  """One station's errors by split, and its largest step."""
  data = read_station_data(run, station)
  observed = data.column(OBSERVED)
  forecast = data.feature_names.index(FORECAST)
  train = data.y_train[:, observed] - data.x_train[:, forecast]
  test = data.y_test[:, observed] - data.x_test[:, forecast]
  times = np.concatenate([data.time_train, data.time_test])
  errors = np.concatenate([train, test])
  order = np.argsort(times, kind="stable")
  times, errors = times[order], errors[order]
  cut = largest_step(errors, min_side)
  is_train = np.concatenate(
    [np.ones(len(train), bool), np.zeros(len(test), bool)]
  )
  return {
    "station": station,
    "n_train": len(train),
    "n_test": len(test),
    "mean_train": float(train.mean()),
    "mean_test": float(test.mean()),
    "shift": float(test.mean() - train.mean()),
    "sd_train": float(train.std()),
    "sd_test": float(test.std()),
    "step_time": pd.Timestamp(times[cut]),
    "mean_before": float(errors[:cut].mean()),
    "mean_after": float(errors[cut:].mean()),
    "step": float(errors[cut:].mean() - errors[:cut].mean()),
    "n_before": cut,
    "n_train_after": int(is_train[order][cut:].sum()),
  }


def build(run: Path, out: Path, *, min_side: int = 30) -> pd.DataFrame:
  """Write ``pressure_shift.csv`` for every station a run stored.

  Parameters
  ----------
  run : pathlib.Path
      The run folder.
  out : pathlib.Path
      Where to write.
  min_side : int, default=30
      The fewest cases on either side of a step.

  Returns
  -------
  pandas.DataFrame
      One row per station.

  Raises
  ------
  FileNotFoundError
      If the run stored no station's cases.
  """
  stations = sorted(
    int(p.parent.parent.name)
    for p in (run / "stations").glob("*/data/cases.parquet")
  )
  if not stations:
    raise FileNotFoundError(
      f"{run} must hold stored station cases (stations/<id>/data); it has none"
    )
  table = pd.DataFrame([station_row(run, s, min_side) for s in stations])
  out.mkdir(parents=True, exist_ok=True)
  table.to_csv(out / "pressure_shift.csv", index=False)
  return table


def main(argv: list[str] | None = None) -> int:
  """Run :func:`build` from the command line.

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
  p.add_argument(
    "--out",
    type=Path,
    default=None,
    help="Where to write; <run>/checks by default.",
  )
  p.add_argument(
    "--min-side",
    type=int,
    default=30,
    help="The fewest cases on either side of a step.",
  )
  a = p.parse_args(argv)
  out = a.out if a.out is not None else a.run / "checks"
  table = build(a.run, out, min_side=a.min_side)
  top = table.reindex(table["shift"].abs().sort_values(ascending=False).index)
  with pd.option_context("display.width", 200):
    print(top.head(5).to_string(index=False, float_format="{:.3f}".format))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
