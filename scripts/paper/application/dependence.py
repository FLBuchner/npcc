"""Kendall's tau of the out-of-fold training PITs: binned, and rolling.

Reads a run's stored cases and margins, and fits nothing. Both subcommands
read the out-of-fold training PITs (``pit_out_of_fold``) of
``tabpfn-quantiles-all`` and ``emos-seasonal`` (``--models``) on the run's
modeled responses (``--variables``), and write a CSV and, per pair of
responses, a page of one PDF and a ``<pair>.png``.

``binned``, FLBuchner's ``binned_tau`` (npcc-paper PR #6): per station,
model and pair, tau-b within ``--bins`` bins of two covariates. The day of
year of the valid time, in bins of equal width over days 1 to 366, one
calendar for every station; and the ensemble mean of the pair's first
response (``coe:<response>_ensavg``; ``--covariate`` names one for every
pair), in quantile bins of the station's training cases. A bin of fewer
than ``--min-pairs`` cases keeps its count and no tau. The observations are
not binned: within a bin the margins still move with the other covariates.
Strata show where the dependence changes; they do not test conditional
independence. Writes ``binned_kendall_tau.csv`` and per pair the median
over stations by bin, quartiles shaded, to ``<run>/reports/binned_tau``.

``rolling``, FLBuchner's ``plot_rolling_tau.py`` (npcc-paper PR #4): at one
station (``--station``), tau-b of every pair in each trailing window of
``--window`` training cases, of the observations and of each model's PITs,
on one plot per pair. Writes ``rolling_kendall_tau.csv`` to
``<run>/reports/rolling_tau_<NNN>_<window>``.

Usage (from the repository root)::

    python scripts/paper/application/dependence.py binned --run <run>
    python scripts/paper/application/dependence.py rolling --run <run> --station 0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_common import Report

from npcc.experiments.application.marginal_set import MarginalSet
from npcc.experiments.application.store import (
  read_marginal_set,
  stored_stations,
)

#: The margins read by default: the paper's and the reference pipeline's.
MODELS: tuple[str, ...] = ("tabpfn-quantiles-all", "emos-seasonal")
#: FLBuchner's defaults.
BINS = 5
MIN_PAIRS = 30
WINDOW = 30
DAY_OF_YEAR = "day_of_year"
FORECAST = "forecast"
OBSERVATIONS = "observations"
#: The days of year the calendar bins cover, a leap day included.
DAYS = (1.0, 367.0)


def tau_record(
  a: np.ndarray, b: np.ndarray, min_pairs: int
) -> dict[str, float]:
  """Kendall's tau-b of the finite pairs, or nan with fewer than ``min_pairs``.

  Returns ``n_pairs`` and ``tau``; tau-b is nan for a constant sample too.
  """
  finite = np.isfinite(a) & np.isfinite(b)
  n = int(finite.sum())
  tau = float("nan")
  if n >= min_pairs:
    tau = float(kendalltau(a[finite], b[finite], variant="b").statistic)
  return {"n_pairs": n, "tau": tau}


def quantile_bins(
  values: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
  """Each value's quantile bin and the distinct edges, as FLBuchner bins.

  Tied edges merge, and a value at an inner edge is in the bin above it.
  """
  finite = values[np.isfinite(values)]
  edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)))
  return np.searchsorted(edges[1:-1], values, side="right"), edges


def day_of_year_bins(
  times: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
  """Each case's bin of equal width in the day of year, and the edges.

  The ``bins + 1`` edges span days 1 to 367, and a bin holds the days from
  its lower edge up to, not including, its upper.
  """
  dates = np.asarray(times).astype("datetime64[D]")
  day = (dates - dates.astype("datetime64[Y]")).astype(np.float64) + 1.0
  edges = np.linspace(*DAYS, bins + 1)
  return np.searchsorted(edges[1:-1], day, side="right"), edges


def forecast_covariate(variable: str) -> str:
  """The ensemble-mean forecast of ``obs:<name>``, ``coe:<name>_ensavg``."""
  return "coe:" + variable.removeprefix("obs:") + "_ensavg"


def training_pits(ms: MarginalSet) -> np.ndarray:
  """The out-of-fold training PITs, refused unless each is in [0, 1]."""
  pit = ms.pit_train
  bad = int((~(np.isfinite(pit) & (pit >= 0.0) & (pit <= 1.0))).sum())
  if bad:
    raise ValueError(
      f"station {ms.station}'s {ms.model} training PITs must lie in [0, 1]; "
      f"got {bad} outside"
    )
  return pit


def binned_tau(
  ms: MarginalSet,
  *,
  covariate: str | None = None,
  bins: int = BINS,
  min_pairs: int = MIN_PAIRS,
) -> pd.DataFrame:
  """Kendall's tau of every pair of training PITs within covariate bins.

  ``covariate`` is every pair's ``forecast`` binning's; by default, each
  pair's first response's ensemble mean (:func:`forecast_covariate`).
  Returns one row per pair, binning and bin: ``variable_a``,
  ``variable_b``, ``binning``, ``covariate``, ``bin``, ``lower``,
  ``upper``, ``n_cases``, ``n_pairs`` and ``tau``.
  """
  pit = training_pits(ms)
  by_day = day_of_year_bins(ms.time_train, bins)
  records = []
  for i, j in combinations(range(len(ms.variables)), 2):
    name = covariate or forecast_covariate(ms.variables[i])
    if name not in ms.feature_names:
      raise ValueError(
        f"covariate must be one of station {ms.station}'s "
        f"{list(ms.feature_names)}; got {name!r}"
      )
    x = ms.x_train[:, ms.feature_names.index(name)]
    for binning, label, (labels, edges) in (
      (DAY_OF_YEAR, DAY_OF_YEAR, by_day),
      (FORECAST, name, quantile_bins(x, bins)),
    ):
      for k in range(len(edges) - 1):
        selected = labels == k
        records.append(
          {
            "variable_a": ms.variables[i],
            "variable_b": ms.variables[j],
            "binning": binning,
            "covariate": label,
            "bin": k,
            "lower": float(edges[k]),
            "upper": float(edges[k + 1]),
            "n_cases": int(selected.sum()),
            **tau_record(pit[selected, i], pit[selected, j], min_pairs),
          }
        )
  return pd.DataFrame(records)


def rolling_tau(
  series: dict[str, np.ndarray],
  variables: Sequence[str],
  times: pd.DatetimeIndex,
  window: int,
) -> pd.DataFrame:
  """Tau-b of every pair of each ``(n, d)`` series, per trailing window."""
  records = []
  for name, values in series.items():
    for i, j in combinations(range(len(variables)), 2):
      for end in range(window, len(times) + 1):
        start = end - window
        records.append(
          {
            "series": name,
            "variable_a": variables[i],
            "variable_b": variables[j],
            "window_start": times[start],
            "time": times[end - 1],
            **tau_record(values[start:end, i], values[start:end, j], 2),
          }
        )
  return pd.DataFrame(records)


def modeled_responses(run: Path) -> list[str] | None:
  """The main design's modeled responses, from the run's ``config.json``."""
  path = Path(run) / "config.json"
  if not path.exists():
    return None
  modeled = json.loads(path.read_text()).get("modeled")
  return list(modeled) if modeled else None


def _short(name: str) -> str:
  return name.removeprefix("obs:").replace("_", " ")


def _day_label(lower: float, upper: float) -> str:
  """A bin's first and last day, as dates of a year with 29 February."""
  start = pd.Timestamp("2020-01-01")
  first = start + pd.Timedelta(days=int(np.ceil(lower)) - 1)
  last = start + pd.Timedelta(days=int(np.ceil(upper)) - 2)
  return f"{first.day} {first:%b} -\n{last.day} {last:%b}"


def _panel(axis: Axes, part: pd.DataFrame, title: str) -> None:
  """Tau by bin, per model: the median over stations, quartiles shaded."""
  for model, curve in part.groupby("model", sort=False):
    table = curve.pivot(index="station", columns="bin", values="tau")
    median = table.median(axis=0)
    line = axis.plot(median.index, median.to_numpy(), marker="o", label=model)
    if len(table) > 1:
      axis.fill_between(
        table.columns,
        table.quantile(0.25, axis=0).to_numpy(),
        table.quantile(0.75, axis=0).to_numpy(),
        color=line[0].get_color(),
        alpha=0.2,
        linewidth=0,
      )
  axis.axhline(0.0, color="0.5", linewidth=0.7)
  axis.set(title=title, ylabel="Kendall's tau-b", ylim=(-1, 1))
  axis.grid(alpha=0.2)


def _binned_figure(pair: pd.DataFrame, title: str) -> Figure:
  """Tau by bin of each binning, one line per model."""
  n = pair["station"].nunique()
  where = f"station {pair['station'].iloc[0]}"
  where = where if n == 1 else f"median over {n} stations, quartiles shaded"
  figure, axes = plt.subplots(1, 2, figsize=(11.7, 4.8), sharey=True)
  days = pair[pair["binning"] == DAY_OF_YEAR]
  _panel(axes[0], days, "Day of year (valid time)")
  edges = days.drop_duplicates("bin").sort_values("bin")
  axes[0].set_xticks(
    edges["bin"].tolist(),
    [_day_label(lo, hi) for lo, hi in zip(edges["lower"], edges["upper"])],
    fontsize=8,
  )
  forecast = pair[pair["binning"] == FORECAST]
  names = sorted(set(forecast["covariate"]))
  _panel(axes[1], forecast, f"Quantile bins of {', '.join(names)}")
  ticks = sorted(set(forecast["bin"]))
  axes[1].set_xticks(ticks, [str(k + 1) for k in ticks])
  axes[1].set_xlabel("bin, lowest to highest, per station")
  axes[1].legend(fontsize=8, loc="lower right")
  figure.suptitle(f"{title}: out-of-fold training PITs; {where}", fontsize=11)
  figure.tight_layout()
  return figure


def _rolling_figure(pair: pd.DataFrame, title: str) -> Figure:
  """Tau by window end, one line per series, the observations dashed."""
  figure, axis = plt.subplots(figsize=(11.7, 4.5), layout="constrained")
  for name, curve in pair.groupby("series", sort=False):
    observed = name == OBSERVATIONS
    axis.plot(
      pd.to_datetime(curve["time"]),
      curve["tau"],
      label=name,
      color="black" if observed else None,
      linestyle="--" if observed else "-",
      linewidth=1.6 if observed else 0.9,
    )
  axis.axhline(0.0, color="0.5", linewidth=0.7)
  axis.set(title=title, xlabel="Window end (valid time)", ylim=(-1, 1))
  axis.set_ylabel("Kendall's tau-b")
  axis.grid(alpha=0.2)
  axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
  return figure


def write(
  frame: pd.DataFrame,
  out: Path,
  name: str,
  figure: Callable[[pd.DataFrame, str], Figure],
) -> None:
  """``<name>.csv``, and per pair a page of ``<name>.pdf`` and a ``.png``."""
  out.mkdir(parents=True, exist_ok=True)
  frame.to_csv(out / f"{name}.csv", index=False)
  with Report(out / f"{name}.pdf", name.replace("_", " ")) as report:
    for (a, b), pair in frame.groupby(["variable_a", "variable_b"], sort=False):
      page = figure(pair, f"{_short(a)} / {_short(b)}")
      page.savefig(out / f"{a}__{b}.png".replace(":", "_"), dpi=150)
      report.save(page)


def margins(
  run: Path,
  station: int,
  models: Sequence[str],
  variables: Sequence[str] | None,
) -> dict[str, MarginalSet]:
  """Each model's margins at a station, on ``variables`` or the run's."""
  modeled = list(variables) if variables else modeled_responses(run)
  return {
    m: read_marginal_set(run, station, m, modeled=modeled) for m in models
  }


def binned(
  run: Path,
  out: Path,
  *,
  stations: Sequence[int] | None = None,
  models: Sequence[str] = MODELS,
  variables: Sequence[str] | None = None,
  covariate: str | None = None,
  bins: int = BINS,
  min_pairs: int = MIN_PAIRS,
) -> pd.DataFrame:
  """Write the binned taus of ``stations`` (all stored) into ``out``.

  ``variables`` are the run's modeled responses by default, and
  ``covariate`` is :func:`binned_tau`'s. Returns ``binned_kendall_tau.csv``.
  """
  if bins < 2 or min_pairs < 2:
    raise ValueError(
      f"bins and min_pairs must be at least 2; got {bins} and {min_pairs}"
    )
  stations = list(stored_stations(run) if stations is None else stations)
  if not stations:
    raise FileNotFoundError(f"{run} must hold stored stations; got none")
  frames = [
    binned_tau(ms, covariate=covariate, bins=bins, min_pairs=min_pairs).assign(
      station=station, model=model
    )
    for station in stations
    for model, ms in margins(run, station, models, variables).items()
  ]
  frame = pd.concat(frames, ignore_index=True)
  leading = ["station", "model"]
  frame = frame[[*leading, *(c for c in frame.columns if c not in leading)]]
  write(frame, out, "binned_kendall_tau", _binned_figure)
  return frame


def rolling(
  run: Path,
  out: Path,
  *,
  station: int = 0,
  window: int = WINDOW,
  models: Sequence[str] = MODELS,
  variables: Sequence[str] | None = None,
) -> pd.DataFrame:
  """Write one station's rolling taus into ``out``; return them."""
  sets = margins(run, station, models, variables)
  if window < 2 or not sets:
    raise ValueError(
      f"window must be at least 2 and models not empty; got {window} and "
      f"{list(models)}"
    )
  first = next(iter(sets.values()))
  times = pd.DatetimeIndex(first.time_train)
  if len(times) < window or not times.is_monotonic_increasing:
    raise ValueError(
      f"station {station} must have at least {window} training cases in "
      f"time order; got {len(times)}"
    )
  series = {OBSERVATIONS: first.y_train}
  series |= {model: training_pits(ms) for model, ms in sets.items()}
  frame = rolling_tau(series, first.variables, times, window)
  label = f"trailing {window} training cases; PITs out of fold"

  def figure(pair: pd.DataFrame, title: str) -> Figure:
    return _rolling_figure(pair, f"Station {station} | {title}\n{label}")

  write(frame, out, "rolling_kendall_tau", figure)
  return frame


def main(argv: list[str] | None = None) -> int:
  """Write the binned or the rolling taus of a run."""
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  commands = p.add_subparsers(dest="command", required=True)
  by_bin = commands.add_parser("binned", help="Tau within covariate bins.")
  by_window = commands.add_parser("rolling", help="Tau in trailing windows.")
  for sub in (by_bin, by_window):
    sub.add_argument("--run", required=True, type=Path, help="The run folder.")
    sub.add_argument("--out", type=Path, help="Where to write.")
    sub.add_argument("--models", nargs="+", default=list(MODELS))
    sub.add_argument("--variables", nargs="+", help="The run's modeled ones.")
  by_bin.add_argument("--stations", type=int, nargs="+", help="All stored.")
  by_bin.add_argument(
    "--covariate",
    help="One covariate for every pair; each first response's ensemble "
    "mean by default.",
  )
  by_bin.add_argument("--bins", type=int, default=BINS)
  by_bin.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
  by_window.add_argument("--station", type=int, default=0)
  by_window.add_argument("--window", type=int, default=WINDOW)
  a = p.parse_args(argv)
  if a.command == "binned":
    out = a.out or a.run / "reports" / "binned_tau"
    binned(
      a.run,
      out,
      stations=a.stations,
      models=a.models,
      variables=a.variables,
      covariate=a.covariate,
      bins=a.bins,
      min_pairs=a.min_pairs,
    )
  else:
    out = a.out or a.run / "reports" / f"rolling_tau_{a.station:03d}_{a.window}"
    rolling(
      a.run,
      out,
      station=a.station,
      window=a.window,
      models=a.models,
      variables=a.variables,
    )
  print(f"wrote {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
