"""Physical consistency of the raw data: Td and w against T, RH and p.

Ports FLBuchner's ``application/eda_physical_consistency`` (npcc-paper
PR #6) onto this package's formulas
(:mod:`npcc.experiments.application.derived`). Reads ``x.nc`` and ``y.nc``
directly, not a run folder: every station and every date, at a 24-hour lead
from 00:00 UTC, with no train/test split. One case is a station and an
initialization; each is read twice, as observed (``obs:*``) and as the
ensemble means of the forecast (``coe:*_ensavg``).

**Formulas.** With MeteoSwiss's piecewise Magnus formula, both constant sets
picked by the sign of the air temperature ``T`` (the convention the stored
dew point was computed with), the vapor pressure is ``e = e_s(Td)`` and

- ``RH* = 100 e / e_s(T)``, in percent;
- ``w* = 1000 * 0.622 e / (p - e)``, in g/kg.

A residual is the recorded value less the derived one. A case enters the
residuals where all five values are finite and the formulas' domain holds:
``p > e``, ``p > 0``, and ``T`` and ``Td`` above ``-238.83`` C. The
coefficient-sensitivity table repeats the residuals with each argument's
own constants (``e_s(Td)`` with ``Td``'s sign), which differ only where
``T`` and ``Td`` straddle 0. No value is clipped.

**Bounds.** Each over the cases whose inputs are finite: ``Td > T``
exactly and by more than 0.001 C, ``RH`` outside [0, 100], ``w < 0``,
``p <= e``, and a non-finite value among the five.

Writes into ``--out``:

- ``coverage.csv``: per source, the cases, stations, first and last
  initialization, the complete cases and those in the formulas' domain;
- ``bounds_checks.csv``: per source and check, the eligible cases, the
  cases breaking it and their percentage, and the largest excess
  ``Td - T``;
- ``residual_summary.csv``: per source and response, the residuals' ``n``,
  mean, MAE, RMSE, median, 5 % and 95 % quantiles and largest magnitude,
  pooled over cases and over station means;
- ``station_summary.csv``: per source and station, each residual's count,
  mean and largest magnitude;
- ``tolerance_summary.csv``: the percentage of residuals within each
  descriptive tolerance;
- ``coefficient_sensitivity.csv``: the pooled residuals under both
  conventions;
- ``case_residuals.csv.gz``: every case, with its source row;
- ``metadata.json``: the selection, the formula and each file's SHA-256;
- ``physical_consistency.pdf``: all of the above, with box plots.

Usage (from the repository root)::

    python scripts/paper/application/eda_physical_consistency.py \\
      --data ~/npcc-data/meteoswiss_zanetta2023 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import PAGE, Report

from npcc.experiments.application.data import (
  DATA_SHA256,
  check_aligned,
  data_sha256,
  lead_in_hours,
  verify_data,
)
from npcc.experiments.application.derived import (
  DEW_POINT,
  METEOSWISS,
  MIXING_FACTOR,
  MIXING_RATIO,
  PRESSURE,
  RELATIVE_HUMIDITY,
  TEMPERATURE,
)

#: The five responses, by their names without a prefix.
VARIABLES: tuple[str, ...] = tuple(
  v.removeprefix("obs:")
  for v in (TEMPERATURE, DEW_POINT, PRESSURE, RELATIVE_HUMIDITY, MIXING_RATIO)
)
#: Each source's label, and the prefix and suffix of its variables' names.
SOURCES: dict[str, tuple[str, str, str]] = {
  "Observations": ("y.nc", "obs:", ""),
  "Ensemble means": ("x.nc", "coe:", "_ensavg"),
}
#: The derived responses' labels, units, and descriptive tolerances.
RESPONSES: dict[str, tuple[str, tuple[float, ...]]] = {
  "RH": ("percentage points", (0.01, 0.1, 1.0)),
  "w": ("g/kg", (0.001, 0.01, 0.1)),
}
#: The bounds checks, in the order the tables list them.
CHECKS: tuple[str, ...] = (
  "Td_above_T",
  "Td_above_T_0.001C",
  "RH_outside_0_100",
  "w_negative",
  "p_not_above_e",
  "nonfinite",
)
#: The smallest temperature the domain admits, ``-b`` of the warm constants.
DOMAIN_FLOOR = -METEOSWISS.b
#: A dew point above the air temperature by more than this is a violation
#: beyond float32 rounding, in degrees Celsius.
TD_TOLERANCE = 0.001
LEAD_HOURS = 24
INIT_HOUR = 0

ENGINE = "h5netcdf"


def load_cases(
  directory: Path, *, lead_hours: int = LEAD_HOURS, init_hour: int = INIT_HOUR
) -> pd.DataFrame:
  """Every station's cases at one lead and initialization hour, both sources.

  Parameters
  ----------
  directory : pathlib.Path
      The folder holding ``x.nc`` and ``y.nc``.
  lead_hours : int, default=24
      The forecast lead.
  init_hour : int, default=0
      The initialization hour, UTC.

  Returns
  -------
  pandas.DataFrame
      One row per source and case: ``source_row`` (the row in both files),
      ``station``, ``initialization``, ``source``, and the five responses
      as floats, named as in :data:`VARIABLES`.
  """
  directory = Path(directory).expanduser()
  with (
    xr.open_dataset(directory / "x.nc", engine=ENGINE) as x,
    xr.open_dataset(directory / "y.nc", engine=ENGINE) as y,
  ):
    check_aligned(x, y)
    ref = np.asarray(x["forecast_reference_time"].values).astype(
      "datetime64[ns]"
    )
    hour = ref - ref.astype("datetime64[D]")
    rows = np.flatnonzero(
      (lead_in_hours(x["t"].values) == lead_hours)
      & (hour == np.timedelta64(init_hour, "h"))
    )
    if not len(rows):
      raise ValueError(
        f"the files must hold forecasts at a {lead_hours} h lead from "
        f"{init_hour:02d}:00 UTC; got none"
      )
    base = pd.DataFrame(
      {
        "source_row": rows.astype(np.int64),
        "station": np.asarray(x["station_id"].values)[rows].astype(np.int64),
        "initialization": ref[rows],
      }
    )
    repeated = int(base.duplicated(["station", "initialization"]).sum())
    if repeated:
      raise ValueError(
        f"cases must be one per station and initialization; got {repeated} "
        "repeated"
      )
    files = {"x.nc": x, "y.nc": y}
    frames = []
    for source, (name, prefix, suffix) in SOURCES.items():
      frame = base.assign(source=source)
      for v in VARIABLES:
        column = np.asarray(files[name][prefix + v + suffix].values)
        frame[v] = column[rows].astype(np.float64)
      frames.append(frame)
  return pd.concat(frames, ignore_index=True)


def from_dew_point(
  t: np.ndarray, td: np.ndarray, p: np.ndarray, *, own_sign: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Relative humidity, mixing ratio and vapor pressure from ``T, Td, p``.

  Parameters
  ----------
  t, td, p : numpy.ndarray
      Air temperature and dew point (C), and pressure (hPa).
  own_sign : bool, default=False
      Pick the constants of ``e_s(Td)`` by ``Td``'s sign instead of ``T``'s.

  Returns
  -------
  rh, w, e : numpy.ndarray
      ``100 e / e_s(T)`` (percent), ``1000 * 0.622 e / (p - e)`` (g/kg),
      and ``e = e_s(Td)`` (hPa); not finite where the formulas break down.
  """
  with np.errstate(all="ignore"):
    e = METEOSWISS.saturation_vapor_pressure(td, None if own_sign else t)
    rh = 100.0 * e / METEOSWISS.saturation_vapor_pressure(t)
    w = MIXING_FACTOR * e / (p - e)
  return rh, w, e


def _flag(eligible: np.ndarray, broken: np.ndarray) -> np.ndarray:
  """1 where broken, 0 where not, nan where the check cannot be made."""
  return np.where(eligible, broken.astype(np.float64), np.nan)


def analyze(frame: pd.DataFrame) -> pd.DataFrame:
  """Each case's residuals and bounds checks.

  Parameters
  ----------
  frame : pandas.DataFrame
      :func:`load_cases`'s output.

  Returns
  -------
  pandas.DataFrame
      ``frame`` with ``complete``, ``domain``, ``RH_derived``,
      ``w_derived``, ``<RH|w>_residual`` and ``<RH|w>_alternative_residual``
      (nan outside the domain), ``Td_minus_T``, and one column per check of
      :data:`CHECKS` (1 broken, 0 not, nan not checkable).
  """
  t, td, p, rh, w = (frame[v].to_numpy(dtype=np.float64) for v in VARIABLES)
  derived_rh, derived_w, e = from_dew_point(t, td, p)
  alt_rh, alt_w, _ = from_dew_point(t, td, p, own_sign=True)
  out = frame.copy()
  complete = np.isfinite(frame[list(VARIABLES)].to_numpy()).all(axis=1)
  with np.errstate(invalid="ignore"):
    domain = (
      complete & (p > e) & (p > 0.0) & (t > DOMAIN_FLOOR) & (td > DOMAIN_FLOOR)
    )
  out["complete"] = complete
  out["domain"] = domain
  for name, recorded, derived, alternative in (
    ("RH", rh, derived_rh, alt_rh),
    ("w", w, derived_w, alt_w),
  ):
    out[f"{name}_derived"] = derived
    out[f"{name}_residual"] = np.where(
      domain & np.isfinite(derived), recorded - derived, np.nan
    )
    out[f"{name}_alternative_residual"] = np.where(
      domain & np.isfinite(alternative), recorded - alternative, np.nan
    )
  temperatures = np.isfinite(t) & np.isfinite(td)
  with np.errstate(invalid="ignore"):
    out["Td_minus_T"] = np.where(temperatures, td - t, np.nan)
    out["Td_above_T"] = _flag(temperatures, td > t)
    out["Td_above_T_0.001C"] = _flag(temperatures, td > t + TD_TOLERANCE)
    out["RH_outside_0_100"] = _flag(np.isfinite(rh), (rh < 0.0) | (rh > 100.0))
    out["w_negative"] = _flag(np.isfinite(w), w < 0.0)
    out["p_not_above_e"] = _flag(temperatures & np.isfinite(p), p <= e)
  out["nonfinite"] = (~complete).astype(np.float64)
  return out


def residual_stats(values: pd.Series | np.ndarray) -> dict[str, float]:
  """``n``, mean, MAE, RMSE, median, 5 % and 95 % quantiles, ``max_abs``.

  Parameters
  ----------
  values : pandas.Series or numpy.ndarray
      Residuals; non-finite ones are left out.

  Returns
  -------
  dict of str to float
      The statistics, ``nan`` without a finite residual.
  """
  a = np.asarray(values, dtype=np.float64)
  a = a[np.isfinite(a)]
  if not len(a):
    nan = float("nan")
    return {
      "n": 0,
      "mean": nan,
      "MAE": nan,
      "RMSE": nan,
      "median": nan,
      "p05": nan,
      "p95": nan,
      "max_abs": nan,
    }
  return {
    "n": len(a),
    "mean": float(a.mean()),
    "MAE": float(np.abs(a).mean()),
    "RMSE": float(np.sqrt(np.mean(a * a))),
    "median": float(np.median(a)),
    "p05": float(np.quantile(a, 0.05)),
    "p95": float(np.quantile(a, 0.95)),
    "max_abs": float(np.abs(a).max()),
  }


def max_abs(values: pd.Series) -> float:
  """The largest magnitude of a series, ``nan`` where it holds none.

  Parameters
  ----------
  values : pandas.Series
      Residuals.

  Returns
  -------
  float
      ``max |values|`` over its finite entries.
  """
  return float(values.abs().max())


def summarize(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
  """The EDA's tables, from :func:`analyze`'s output.

  Parameters
  ----------
  frame : pandas.DataFrame
      Each case's residuals and checks.

  Returns
  -------
  dict of str to pandas.DataFrame
      ``coverage``, ``bounds_checks``, ``residual_summary``,
      ``station_summary``, ``tolerance_summary`` and
      ``coefficient_sensitivity``, each named for its CSV.
  """
  by_source = list(frame.groupby("source", sort=False))
  coverage = (
    frame.groupby("source", sort=False)
    .agg(
      cases=("source_row", "size"),
      stations=("station", "nunique"),
      first=("initialization", "min"),
      last=("initialization", "max"),
      complete=("complete", "sum"),
      domain=("domain", "sum"),
    )
    .reset_index()
  )
  checks = []
  for source, group in by_source:
    for check in CHECKS:
      flags = group[check]
      checks.append(
        {
          "source": source,
          "check": check,
          "eligible": int(flags.count()),
          "violations": int(flags.sum()),
          "percent": 100.0 * flags.mean(),
          "max_Td_minus_T": group["Td_minus_T"].max()
          if check.startswith("Td_above_T")
          else np.nan,
        }
      )
  summaries = []
  for source, group in by_source:
    for name in RESPONSES:
      column = f"{name}_residual"
      for aggregation, values in (
        ("Pooled cases", group[column]),
        ("Station means", group.groupby("station")[column].mean()),
      ):
        summaries.append(
          {
            "source": source,
            "variable": name,
            "aggregation": aggregation,
            **residual_stats(values),
          }
        )
  station = frame.groupby(["source", "station"], sort=False)[
    [f"{name}_residual" for name in RESPONSES]
  ].agg(["count", "mean", max_abs])
  station.columns = [f"{c}_{s}" for c, s in station.columns]
  tolerances = []
  for source, group in by_source:
    for name, (_, thresholds) in RESPONSES.items():
      a = group[f"{name}_residual"].dropna().abs()
      for threshold in thresholds:
        tolerances.append(
          {
            "source": source,
            "variable": name,
            "tolerance": threshold,
            "eligible": len(a),
            "within_percent": 100.0 * float((a <= threshold).mean()),
          }
        )
  sensitivity = []
  for source, group in by_source:
    t, td = group[VARIABLES[0]], group[VARIABLES[1]]
    straddle = int((group["domain"] & ((t >= 0) != (td >= 0))).sum())
    for name in RESPONSES:
      for convention, column in (
        ("T's sign", f"{name}_residual"),
        ("each argument's own sign", f"{name}_alternative_residual"),
      ):
        stats = residual_stats(group[column])
        sensitivity.append(
          {
            "source": source,
            "variable": name,
            "convention": convention,
            "straddling_cases": straddle,
            **{k: stats[k] for k in ("n", "mean", "MAE", "RMSE", "max_abs")},
          }
        )
  return {
    "coverage": coverage,
    "bounds_checks": pd.DataFrame(checks),
    "residual_summary": pd.DataFrame(summaries),
    "station_summary": station.reset_index(),
    "tolerance_summary": pd.DataFrame(tolerances),
    "coefficient_sensitivity": pd.DataFrame(sensitivity),
  }


def _findings(tables: dict[str, pd.DataFrame]) -> list[str]:
  """The headline numbers, one line each."""
  summary, checks = tables["residual_summary"], tables["bounds_checks"]
  lines = []
  for source in SOURCES:
    for name, (unit, _) in RESPONSES.items():
      row = summary[
        (summary["source"] == source)
        & (summary["variable"] == name)
        & (summary["aggregation"] == "Pooled cases")
      ].iloc[0]
      lines.append(
        f"{source}, {name} residuals: MAE {row['MAE']:.3g}, RMSE "
        f"{row['RMSE']:.3g}, largest {row['max_abs']:.3g} {unit}."
      )
    c = checks[checks["source"] == source].set_index("check")["violations"]
    excess = checks[
      (checks["source"] == source) & (checks["check"] == "Td_above_T")
    ]["max_Td_minus_T"].iloc[0]
    lines.append(
      f"{source}: Td > T in {int(c['Td_above_T']):,} cases (largest Td - T "
      f"{excess:.3g} C), by more than {TD_TOLERANCE} C in "
      f"{int(c['Td_above_T_0.001C']):,}."
    )
    lines.append(
      f"{source}: RH outside [0, 100] in {int(c['RH_outside_0_100']):,}; "
      f"w < 0 in {int(c['w_negative']):,}; p <= e in "
      f"{int(c['p_not_above_e']):,}; a value not finite in "
      f"{int(c['nonfinite']):,}."
    )
  return lines


def _exact_counts(frame: pd.DataFrame) -> pd.DataFrame:
  """``frame`` with its integer columns as text, so a page prints them whole."""
  out = frame.copy()
  for column in out:
    if pd.api.types.is_integer_dtype(out[column]):
      out[column] = out[column].map(lambda v: f"{v:,}")
  return out


def _box_page(report: Report, frame: pd.DataFrame, aggregation: str) -> None:
  """RH and w residuals of both sources, as boxes, on one page."""
  figure, axes = plt.subplots(1, 2, figsize=PAGE)
  for axis, (name, (unit, _)) in zip(axes, RESPONSES.items(), strict=True):
    samples = []
    for source in SOURCES:
      group = frame[frame["source"] == source]
      values = (
        group.groupby("station")[f"{name}_residual"].mean()
        if aggregation == "Station means"
        else group[f"{name}_residual"]
      )
      samples.append(values.dropna().to_numpy())
    boxes = axis.boxplot(
      samples,
      showfliers=True,
      flierprops={"markersize": 1, "alpha": 0.2},
    )
    for flier in boxes["fliers"]:
      flier.set_rasterized(True)
    axis.set_xticks([1, 2], list(SOURCES))
    axis.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    axis.set(ylabel=f"{name} residual ({unit})", title=name)
    axis.grid(axis="y", alpha=0.2)
  figure.suptitle(f"{aggregation}: recorded less derived", fontsize=14)
  figure.text(
    0.06,
    0.02,
    "Boxes: quartiles and median; whiskers: 1.5 IQR; points: outliers. "
    "The spread is descriptive, not a confidence interval.",
    fontsize=9,
  )
  figure.tight_layout(rect=(0, 0.05, 1, 0.95))
  report.save(figure)


def write_pdf(
  path: Path, frame: pd.DataFrame, tables: dict[str, pd.DataFrame]
) -> None:
  """The EDA's report.

  Parameters
  ----------
  path : pathlib.Path
      The PDF to write.
  frame : pandas.DataFrame
      Each case's residuals and checks.
  tables : dict of str to pandas.DataFrame
      :func:`summarize`'s output.
  """
  a, b, c = METEOSWISS.a, METEOSWISS.b, METEOSWISS.c
  below = METEOSWISS.below or (a, b, c)
  with Report(path, "Physical consistency of the raw data") as report:
    report.text(
      f"Physical consistency: {LEAD_HOURS}-hour forecasts from "
      f"{INIT_HOUR:02d}:00 UTC",
      [
        "Every station and date; no train/test split. One case is a "
        "station and an initialization, observed and as ensemble means.",
        "Data: Zanetta, Nerini, Beucler and Liniger (2023), arXiv:2212.04487.",
        "e = e_s(Td), RH* = 100 e / e_s(T) [%], w* = 1000 * 0.622 e / "
        "(p - e) [g/kg], with e_s(t) = c exp(a t / (b + t)) hPa.",
        f"MeteoSwiss's constants (a, b, c): ({a}, {b}, {c}) at and above "
        f"0 C, {below} below, both picked by the sign of T.",
        "Residual = recorded less derived; zero is exact agreement.",
        "The sensitivity page picks e_s(Td)'s constants by Td's own sign; "
        "the two differ where T and Td straddle 0.",
        "Residuals use the cases with five finite values, p > e, p > 0 "
        f"and T, Td > {DOMAIN_FLOOR} C; nothing is clipped.",
        "The tolerances are descriptive, not measurement accuracy.",
        "Ensemble means need not obey a nonlinear relation their members "
        "obey; the files hold no members.",
      ],
    )
    report.text("Findings", _findings(tables))
    report.table(
      _exact_counts(tables["coverage"]), "Coverage and the formulas' domain"
    )
    report.table(
      _exact_counts(tables["bounds_checks"]),
      "Physical bounds",
      "Eligible: cases whose inputs to the check are finite.",
    )
    summary = tables["residual_summary"]
    for aggregation in ("Pooled cases", "Station means"):
      report.table(
        _exact_counts(
          summary[summary["aggregation"] == aggregation].drop(
            columns="aggregation"
          )
        ),
        f"Residuals: {aggregation.lower()}",
        "RH in percentage points, w in g/kg. Station means weigh stations "
        "equally; a signed mean can hide day-to-day deviations.",
      )
    report.table(
      _exact_counts(tables["tolerance_summary"]),
      "Residuals within descriptive tolerances",
      "Of the finite residuals in the formulas' domain.",
    )
    for aggregation in ("Station means", "Pooled cases"):
      _box_page(report, frame, aggregation)
    report.table(
      _exact_counts(tables["coefficient_sensitivity"]),
      "Coefficient sensitivity: pooled cases",
      "Straddling: cases in the domain with T and Td on either side of 0.",
    )


def build(data: Path, out: Path, *, verify: bool = True) -> list[Path]:
  """Run the EDA on a data folder and write its outputs.

  Parameters
  ----------
  data : pathlib.Path
      The folder holding ``x.nc`` and ``y.nc``.
  out : pathlib.Path
      Where to write; created if absent.
  verify : bool, default=True
      Refuse files whose SHA-256 is not the published one.

  Returns
  -------
  list of pathlib.Path
      The files written.
  """
  data = Path(data).expanduser()
  if verify:
    verify_data(data)
  frame = analyze(load_cases(data))
  tables = summarize(frame)
  out.mkdir(parents=True, exist_ok=True)
  written = []
  for name, table in tables.items():
    path = out / f"{name}.csv"
    table.to_csv(path, index=False)
    written.append(path)
  path = out / "case_residuals.csv.gz"
  frame.to_csv(path, index=False)
  written.append(path)
  digests = data_sha256(data)
  metadata = {
    "selection": {
      "lead_hours": LEAD_HOURS,
      "initialization_utc": f"{INIT_HOUR:02d}:00",
      "dates": "all available",
    },
    "formula": {
      "magnus": METEOSWISS.name,
      "constants_picked_by": "the sign of T, for e_s(T) and e_s(Td)",
    },
    "input_files": {
      name: {
        "path": str((data / name).resolve()),
        "sha256": digest,
        "published": digest == DATA_SHA256[name],
      }
      for name, digest in digests.items()
    },
  }
  path = out / "metadata.json"
  path.write_text(json.dumps(metadata, indent=2) + "\n")
  written.append(path)
  path = out / "physical_consistency.pdf"
  write_pdf(path, frame, tables)
  written.append(path)
  return written


def main(argv: Sequence[str] | None = None) -> int:
  """Run :func:`build` from the command line.

  Parameters
  ----------
  argv : sequence of str, optional
      The arguments; ``None`` reads them from ``sys.argv``.

  Returns
  -------
  int
      The exit status.
  """
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument(
    "--data", required=True, type=Path, help="The folder of x.nc and y.nc."
  )
  p.add_argument("--out", required=True, type=Path, help="Where to write.")
  p.add_argument(
    "--no-verify",
    action="store_true",
    help="Read files whose SHA-256 is not the published one.",
  )
  a = p.parse_args(argv)
  written = {f.name: f for f in build(a.data, a.out, verify=not a.no_verify)}
  for name in ("coverage", "bounds_checks", "residual_summary"):
    print(pd.read_csv(written[f"{name}.csv"]).to_string(index=False))
  print(written["physical_consistency.pdf"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
