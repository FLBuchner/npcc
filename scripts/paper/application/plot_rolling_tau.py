"""Rolling Kendall's tau of a station's training cases, observed and as PITs.

Ports FLBuchner's ``application/plot_rolling_tau.py`` (npcc-paper PR #4)
onto a run folder: for one station, every pair of responses, and trailing
windows of ``--window`` consecutive training cases, Kendall's tau-b of the
observations and of each stored marginal model's training PITs (in sample:
the margins are not refitted per window). Reads the stored cases and
margins, fits nothing. Writes, under
``<run>/reports/rolling_tau_<NNN>_<window>`` (or ``--out``):

- ``rolling_kendall_tau.csv``: one row per series, pair and window;
- ``rolling_kendall_tau.pdf`` and ``<series>/<pair>.png``: one plot each;
- ``rolling_kendall_tau_comparison.pdf`` and ``comparison/<pair>.png``:
  every series of a pair on one plot.

Usage (from the repository root)::

    python scripts/paper/application/plot_rolling_tau.py --run results/application --station 0
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_common as common

from npcc.experiments.application.store import (
  read_marginal_set,
  read_station_data,
  station_dir,
)

OBSERVATIONS = "observations"


def _short(name: str) -> str:
  return name.removeprefix("obs:").replace("_", " ")


def rolling_tau(
  series: dict[str, np.ndarray],
  variables: tuple[str, ...],
  times: pd.DatetimeIndex,
  window: int,
) -> pd.DataFrame:
  """Kendall's tau-b of every pair, in every full trailing window."""
  records = []
  for name, values in series.items():
    for i, j in combinations(range(len(variables)), 2):
      for end in range(window - 1, len(times)):
        start = end - window + 1
        tau = kendalltau(
          values[start : end + 1, i], values[start : end + 1, j], variant="b"
        ).statistic
        records.append(
          {
            "series": name,
            "variable_a": variables[i],
            "variable_b": variables[j],
            "window_start": times[start],
            "time": times[end],
            "n_pairs": window,
            "tau": tau,
          }
        )
  return pd.DataFrame(records)


def _plot(frame: pd.DataFrame, title: str, path: Path, pdf: PdfPages) -> None:
  figure, axis = plt.subplots(figsize=(11.7, 4.5), layout="constrained")
  for name, curve in frame.groupby("series", sort=False):
    observed = name == OBSERVATIONS
    axis.plot(
      curve["time"],
      curve["tau"],
      label=name,
      color="black" if observed else None,
      linestyle="--" if observed else "-",
      linewidth=1.6 if observed else 0.9,
    )
  axis.axhline(0.0, color="0.5", linewidth=0.7)
  axis.set(
    title=title,
    xlabel="Window end (valid time)",
    ylabel="Kendall's tau-b",
    ylim=(-1, 1),
  )
  axis.grid(alpha=0.2)
  if frame["series"].nunique() > 1:
    axis.legend(
      loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8
    )
  figure.savefig(path, dpi=150)
  pdf.savefig(figure)
  plt.close(figure)


def build(
  run: Path,
  station: int,
  window: int,
  models: list[str] | None,
  out: Path,
  variables: list[str] | None = None,
) -> Path:
  """Write the rolling-tau plots and table of one station into ``out``."""
  data = read_station_data(run, station)
  times = pd.DatetimeIndex(data.time_train)
  if (
    len(times) < window
    or not times.is_monotonic_increasing
    or not times.is_unique
  ):
    raise ValueError(
      f"station {station} must have at least {window} distinct, ordered "
      f"training cases; got {len(times)}"
    )
  if models is None:
    root = station_dir(run, station) / "margins"
    models = sorted(
      p.name
      for p in root.iterdir()
      if p.is_dir() and not p.name.startswith(".")
    )
  sets = {model: read_marginal_set(run, station, model) for model in models}
  # The modeled responses: every one, unless the run derives some.
  modeled = next(iter(sets.values())).variables if sets else data.variables
  names = list(modeled) if variables is None else variables
  series = {OBSERVATIONS: data.y_train[:, [data.column(v) for v in names]]}
  for model, ms in sets.items():
    pit = ms.pit_train[:, [ms.variables.index(v) for v in names]]
    if not (np.isfinite(pit) & (pit >= 0) & (pit <= 1)).all():
      raise ValueError(f"{model}'s training PITs must lie in [0, 1]")
    series[model] = pit
  frame = rolling_tau(series, tuple(names), times, window)
  out.mkdir(parents=True, exist_ok=True)
  frame.to_csv(out / "rolling_kendall_tau.csv", index=False)
  label = f"trailing {window} training cases; PITs in sample"
  with PdfPages(out / "rolling_kendall_tau.pdf") as pdf:
    for name, part in frame.groupby("series", sort=False):
      folder = out / str(name)
      folder.mkdir(exist_ok=True)
      for (a, b), pair in part.groupby(
        ["variable_a", "variable_b"], sort=False
      ):
        _plot(
          pair,
          f"Station {station} | {_short(a)} / {_short(b)}\n{name}; {label}",
          folder / f"{a}__{b}.png".replace(":", "_"),
          pdf,
        )
  comparison = out / "comparison"
  comparison.mkdir(exist_ok=True)
  with PdfPages(out / "rolling_kendall_tau_comparison.pdf") as pdf:
    for (a, b), pair in frame.groupby(["variable_a", "variable_b"], sort=False):
      _plot(
        pair,
        f"Station {station} | {_short(a)} / {_short(b)}\n{label}",
        comparison / f"{a}__{b}.png".replace(":", "_"),
        pdf,
      )
  return out


def main(argv: list[str] | None = None) -> int:
  """Plot one station's rolling Kendall's tau."""
  p = common.parser(__doc__.splitlines()[0])
  p.add_argument("--station", type=int, default=0)
  p.add_argument("--window", type=int, default=30)
  p.add_argument(
    "--models", nargs="+", default=None, help="Marginal models; all stored."
  )
  p.add_argument(
    "--variables", nargs="+", default=None, help="Responses; all by default."
  )
  args = p.parse_args(argv)
  if args.window < 2:
    p.error(f"window must be at least 2; got {args.window}")
  out = args.out or (
    args.run / "reports" / f"rolling_tau_{args.station:03d}_{args.window}"
  )
  build(args.run, args.station, args.window, args.models, out, args.variables)
  print(f"wrote {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
