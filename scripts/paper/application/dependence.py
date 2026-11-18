"""Kendall's tau of the out-of-fold training PITs within covariate bins.

Ports FLBuchner's ``binned_tau`` (``application/analysis/dependence.py``,
npcc-paper PR #6) onto a run folder. Per station, marginal model and pair of
responses, Kendall's tau-b of the pair's out-of-fold training PITs
(``pit_out_of_fold``, *Training PITs* in the README) within bins of a
covariate, two binnings per pair:

- ``day_of_year``: the day of year of each case's valid time, in ``--bins``
  bins of equal width over days 1 to 366, the same calendar at every
  station;
- ``forecast``: the ensemble-mean forecast of the pair's first response
  (``coe:<response>_ensavg``, the responses in the data's order), in
  ``--bins`` quantile bins of the station's training cases, as FLBuchner
  bins every covariate; ``--covariate`` names one covariate for every pair
  instead.

The observations themselves are not binned: within a bin the margins still
move with the other covariates, so their tau would mix the margins' shifts
with the dependence. A bin with fewer than ``--min-pairs`` cases keeps its
count and a missing tau. Strata show where the dependence changes; they do
not test conditional independence. Reads the stored cases and margins, fits
nothing. Writes, under ``<run>/reports/binned_tau`` (or ``--out``):

- ``binned_kendall_tau.csv``: one row per station, model, pair, binning and
  bin, with its edges, its cases and the pairs tau was computed on;
- ``binned_kendall_tau.pdf`` and ``<pair>.png``: per pair, tau by bin of
  each binning, one line per model, the median over stations with their
  interquartile range.

Usage (from the repository root)::

    python scripts/paper/application/dependence.py --run <run folder>
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_common as common

from npcc.experiments.application.marginal_set import MarginalSet
from npcc.experiments.application.store import (
  read_marginal_set,
  stored_stations,
)

#: The margins binned by default: the paper's and the reference pipeline's.
MODELS: tuple[str, ...] = ("tabpfn-quantiles-all", "emos-seasonal")
#: FLBuchner's defaults.
BINS = 5
MIN_PAIRS = 30
DAY_OF_YEAR = "day_of_year"
FORECAST = "forecast"
#: The days of year the calendar bins cover, a leap day included.
DAYS = (1.0, 367.0)


def tau_record(
  a: np.ndarray, b: np.ndarray, min_pairs: int
) -> dict[str, float]:
  """Kendall's tau-b of the finite pairs, or nan with fewer than ``min_pairs``.

  Parameters
  ----------
  a, b : numpy.ndarray
      Paired values.
  min_pairs : int
      The fewest finite pairs tau is computed on.

  Returns
  -------
  dict
      ``n_pairs`` and ``tau``; tau-b is ``nan`` for a constant sample too.
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
  """Each value's quantile bin, as FLBuchner bins: tied edges merge.

  Parameters
  ----------
  values : numpy.ndarray
      A covariate's values.
  bins : int
      The number of quantile bins.

  Returns
  -------
  labels, edges : numpy.ndarray
      Each value's bin, ``0`` to ``len(edges) - 2``, and the distinct edges;
      a value at an inner edge is in the bin above it.
  """
  finite = values[np.isfinite(values)]
  edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)))
  return np.searchsorted(edges[1:-1], values, side="right"), edges


def day_of_year_bins(
  times: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
  """Each case's bin of equal width in the day of year.

  Parameters
  ----------
  times : numpy.ndarray of datetime64
      The cases' valid times.
  bins : int
      The number of bins over days 1 to 366.

  Returns
  -------
  labels, edges : numpy.ndarray
      Each case's bin and the ``bins + 1`` edges, in days of year; a bin
      holds the days from its lower edge up to, not including, its upper.
  """
  dates = np.asarray(times).astype("datetime64[D]")
  day = (dates - dates.astype("datetime64[Y]")).astype(np.float64) + 1.0
  edges = np.linspace(*DAYS, bins + 1)
  return np.searchsorted(edges[1:-1], day, side="right"), edges


def forecast_covariate(variable: str) -> str:
  """The ensemble-mean forecast of a response, ``coe:<name>_ensavg``.

  Parameters
  ----------
  variable : str
      The response, as ``obs:<name>``.

  Returns
  -------
  str
      Its covariate's name.
  """
  return "coe:" + variable.removeprefix("obs:") + "_ensavg"


def binned_tau(
  ms: MarginalSet,
  *,
  covariate: str | None = None,
  bins: int = BINS,
  min_pairs: int = MIN_PAIRS,
) -> pd.DataFrame:
  """Kendall's tau of every pair of training PITs within covariate bins.

  Parameters
  ----------
  ms : MarginalSet
      One station's margins, its training PITs out of fold.
  covariate : str, optional
      The covariate of every pair's ``forecast`` binning; by default each
      pair's first response's ensemble mean (:func:`forecast_covariate`).
  bins : int, default=5
      Bins per binning.
  min_pairs : int, default=30
      The fewest cases a bin's tau is computed on.

  Returns
  -------
  pandas.DataFrame
      One row per pair, binning and bin: ``variable_a``, ``variable_b``,
      ``binning``, ``covariate``, ``bin``, ``lower``, ``upper``,
      ``n_cases``, ``n_pairs`` and ``tau``.
  """
  pit = ms.pit_train
  if not (np.isfinite(pit) & (pit >= 0.0) & (pit <= 1.0)).all():
    raise ValueError(
      f"station {ms.station}'s {ms.model} training PITs must lie in [0, 1]"
    )
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


def build(
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
  """Write the binned taus of a run's stations into ``out``.

  Parameters
  ----------
  run : pathlib.Path
      The run folder.
  out : pathlib.Path
      Where to write.
  stations : sequence of int, optional
      The stations; every stored one by default.
  models : sequence of str, default=MODELS
      The marginal models.
  variables : sequence of str, optional
      The responses; the run's modeled ones by default.
  covariate : str, optional
      One covariate for every pair's ``forecast`` binning.
  bins : int, default=5
      Bins per binning.
  min_pairs : int, default=30
      The fewest cases a bin's tau is computed on.

  Returns
  -------
  pandas.DataFrame
      ``binned_kendall_tau.csv``.
  """
  if bins < 2 or min_pairs < 2:
    raise ValueError(
      f"bins and min_pairs must be at least 2; got {bins} and {min_pairs}"
    )
  stations = list(stored_stations(run) if stations is None else stations)
  if not stations:
    raise FileNotFoundError(f"{run} must hold stored stations; got none")
  modeled = list(variables) if variables else modeled_responses(run)
  frames = []
  for station in stations:
    for model in models:
      ms = read_marginal_set(
        run, station, model, train_pits="out_of_fold", modeled=modeled
      )
      frames.append(
        binned_tau(
          ms, covariate=covariate, bins=bins, min_pairs=min_pairs
        ).assign(station=station, model=model, pits=ms.train_pits)
      )
  frame = pd.concat(frames, ignore_index=True)
  leading = ["station", "model", "pits"]
  frame = frame[[*leading, *(c for c in frame.columns if c not in leading)]]
  out.mkdir(parents=True, exist_ok=True)
  frame.to_csv(out / "binned_kendall_tau.csv", index=False)
  n = frame["station"].nunique()
  where = f"station {stations[0]}" if n == 1 else f"median over {n} stations"
  with common.Report(
    out / "binned_kendall_tau.pdf", "Kendall's tau within covariate bins"
  ) as report:
    for (a, b), pair in frame.groupby(["variable_a", "variable_b"], sort=False):
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
      axes[1].set_xticks(
        sorted(set(forecast["bin"])),
        [str(k + 1) for k in sorted(set(forecast["bin"]))],
      )
      axes[1].set_xlabel("bin, lowest to highest, per station")
      axes[1].legend(fontsize=8, loc="lower right")
      figure.suptitle(
        f"{_short(a)} / {_short(b)}: out-of-fold training PITs; {where}, "
        "quartiles shaded",
        fontsize=11,
      )
      figure.tight_layout()
      figure.savefig(out / f"{a}__{b}.png".replace(":", "_"), dpi=150)
      report.save(figure)
  return frame


def main(argv: list[str] | None = None) -> int:
  """Write the binned taus of a run from the command line.

  Parameters
  ----------
  argv : list of str, optional
      The arguments; ``None`` reads them from ``sys.argv``.

  Returns
  -------
  int
      The exit status.
  """
  p = common.parser(__doc__.splitlines()[0])
  p.add_argument(
    "--stations", type=int, nargs="+", default=None, help="All stored."
  )
  p.add_argument("--models", nargs="+", default=list(MODELS))
  p.add_argument(
    "--variables", nargs="+", default=None, help="The run's modeled ones."
  )
  p.add_argument(
    "--covariate",
    default=None,
    help="One covariate for every pair; each first response's ensemble "
    "mean by default.",
  )
  p.add_argument("--bins", type=int, default=BINS)
  p.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
  a = p.parse_args(argv)
  out = a.out or a.run / "reports" / "binned_tau"
  build(
    a.run,
    out,
    stations=a.stations,
    models=a.models,
    variables=a.variables,
    covariate=a.covariate,
    bins=a.bins,
    min_pairs=a.min_pairs,
  )
  print(f"wrote {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
