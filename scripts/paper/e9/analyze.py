"""E9's tables: each estimator's KL and IAE, paired by repetition against TabPFN.

Reads the finished cells of one or more covariate-study runs (the npcc run and
its R competitors, on the same grid, so their cells hold the same data) and
writes into ``<first run>/analysis/`` (or ``--out``):

- ``kl.csv``: per scenario, ``n`` and estimator, the Monte Carlo KL averaged
  over the covariate points and the families within each repetition, its mean
  and SE over repetitions, and the same for its difference from TabPFN v3 on
  every covariate, with the share of repetitions in which it is lower; the
  main text's table;
- ``kl_by_family.csv``: the same per family;
- ``iae.csv``: the IAE of the density and both h-functions, as ``kl.csv``;
- ``per_point.csv``: per family, scenario, ``n``, estimator and covariate
  point, the KL and IAE averaged over repetitions, with the point's true tau;
- ``failures.csv``: every fit that failed, with its error. A failed fit is
  missing from every mean, and every table counts it in ``failed``;
- ``fits.csv``: per R fit, from the line its script logs, the family it
  chose, boostCopula's cross-validated stop out of its cap, the covariates
  it kept, counted as active and inert for the scenario, and whether the
  oracle's Fisher scoring converged within its iterations;
- ``fits_summary.csv``: the same per estimator, scenario and ``n``: how
  often the chosen family is the true one, how many stops landed within 5%
  of the cap (where the cap, not the cross-validation, may have stopped),
  the mean number of active and inert covariates kept, and how many oracle
  fits stopped at the iteration cap unconverged, which gamCopula warns
  should not be trusted.

Usage (from the npcc repository root)::

    uv run --no-sync python scripts/paper/e9/analyze.py \\
      results/study_covariates_<TAG> --with results/study_covariates_<TAG>_r
"""

from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path

import pandas as pd

from npcc.experiments.covariate_scenarios import ACTIVE, COVARIATES

#: The estimator every other is paired against.
BASELINE: str = "tabpfn-quantiles-v3-probit"

#: Each label as the tables show it.
DISPLAY: dict[str, str] = {
  "tabpfn-quantiles-v3-probit": "TabPFN v3",
  "tabpfn-quantiles-v3-probit-active": "TabPFN v3, active covariates",
  "tabpfn-quantiles-v3-probit-none": "no covariates",
  "tabicl-probit": "TabICLv2",
  "lightgbm-probit": "LightGBM",
  "xgboost-probit": "XGBoost",
  "catboost-probit": "CatBoost",
  "gamcopula-select": "gamCopula, selection",
  "gamcopula-oracle": "gamCopula, oracle",
  "boostcopula-smooth-cv": "boostCopula-CV",
}

CELL: list[str] = ["family", "scenario", "n", "rep"]

#: Each family's code in gamCopula and boostCopula, for the chosen family.
TRUE_FAMILY: dict[str, int] = {"gaussian": 1, "clayton": 301}

#: A log record's start, which a script's multi-line report continues.
_RECORD = re.compile(r"^\d{4}-\d{2}-\d{2} [\d:,]+ ", re.MULTILINE)
#: The line ``npcc.experiments._external`` logs for a script's report.
_REPORT = re.compile(
  r"INFO npcc\.experiments\._external: (?P<label>[\w-]+) on "
  r"(?P<family>[a-z]+)__(?P<scenario>[a-z]+)__n(?P<n>\d+)__rep(?P<rep>\d+): "
  r"(?P<report>.*)",
  re.DOTALL,
)
_KEPT = re.compile(
  r"covariates (?:selected|chosen) (?P<kept>.*?), (?:EDF|par2)"
)


def load(runs: list[Path]) -> pd.DataFrame:
  """Every finished cell's metric rows, over every run.

  Parameters
  ----------
  runs : list of pathlib.Path
      The run folders.

  Returns
  -------
  pandas.DataFrame
      One row per (cell, estimator, quantity, point).
  """
  frames = []
  for run in runs:
    done = sorted(glob.glob(str(run / "cells" / "*" / "DONE")))
    if not done:
      raise SystemExit(f"{run} must hold finished cells; got none")
    for marker in done:
      cell = Path(marker).parent
      parquet, csv = cell / "metrics.parquet", cell / "metrics.csv"
      frames.append(
        pd.read_parquet(parquet) if parquet.is_file() else pd.read_csv(csv)
      )
  metrics = pd.concat(frames, ignore_index=True)
  keys = [*CELL, "label", "quantity", "point"]
  duplicated = metrics.duplicated(keys)
  if duplicated.any():
    labels = sorted(metrics.loc[duplicated, "label"].unique())
    raise SystemExit(
      f"each (cell, estimator, quantity, point) must come from one run; got "
      f"{int(duplicated.sum())} repeated rows for {labels}"
    )
  if "error" not in metrics.columns:
    raise SystemExit("the runs must record each fit's error; got no column")
  metrics["error"] = metrics["error"].fillna("").astype(str)
  pdf = metrics[(metrics["quantity"] == "pdf") & (metrics["error"] == "")]
  bad = pdf[pdf["KL_mc"].isna()]
  if not bad.empty:
    raise SystemExit(
      f"every density row must carry a finite KL_mc; got {len(bad)} without, "
      f"for {sorted(bad['label'].unique())}"
    )
  draws = sorted(pdf["KL_mc_n"].unique())
  if len(draws) != 1:
    raise SystemExit(
      f"the runs must score the same number of draws per point; got {draws}"
    )
  metrics["estimator"] = metrics["label"].map(DISPLAY).fillna(metrics["label"])
  return metrics


def fit_logs(runs: list[Path]) -> pd.DataFrame:
  """Each R fit's report, from the runs' logs.

  Parameters
  ----------
  runs : list of pathlib.Path
      The run folders; each one's ``logs/*.log``.

  Returns
  -------
  pandas.DataFrame
      One row per (cell, estimator) whose script reported, with the family
      it chose, its cross-validated stop and cap (boostCopula only), and the
      covariates it kept (gamCopula's selection and boostCopula only).
  """
  rows = []
  for run in runs:
    for log in sorted((run / "logs").glob("*.log")):
      for record in _RECORD.split(log.read_text()):
        match = _REPORT.match(record.strip())
        if match is None:
          continue
        report = match["report"]
        chosen = re.search(r"\bfamily (\d+)", report)
        stop = re.search(r"cv-stop (\d+)/(\d+)", report)
        kept = _KEPT.search(report)
        converged = re.search(r"converged (yes|no)", report)
        row: dict[str, object] = {
          "label": match["label"],
          "family": match["family"],
          "scenario": match["scenario"],
          "n": int(match["n"]),
          "rep": int(match["rep"]),
          "chosen_family": int(chosen[1]) if chosen else math.nan,
          "cv_stop": int(stop[1]) if stop else math.nan,
          "maxit": int(stop[2]) if stop else math.nan,
          "kept": "",
          "active_kept": math.nan,
          "inert_kept": math.nan,
          "converged": converged[1] if converged else "",
        }
        if kept is not None:
          names = re.findall(r"\b[bzt][1-5]\b", kept["kept"])
          active = set(ACTIVE[match["scenario"]])
          row["kept"] = " ".join(names)
          row["active_kept"] = sum(name in active for name in names)
          row["inert_kept"] = sum(name not in active for name in names)
        rows.append(row)
  if not rows:
    return pd.DataFrame()
  fits = pd.DataFrame(rows)
  # A cell resumed after an interruption reports twice; the last is its fit.
  fits = fits.drop_duplicates([*CELL, "label"], keep="last")
  fits["estimator"] = fits["label"].map(DISPLAY).fillna(fits["label"])
  fits["true_family"] = fits["chosen_family"] == fits["family"].map(TRUE_FAMILY)
  fits["near_cap"] = fits["cv_stop"] >= 0.95 * fits["maxit"]
  return fits.sort_values([*CELL, "label"], ignore_index=True)


def fits_summary(fits: pd.DataFrame) -> pd.DataFrame:
  """:func:`fit_logs` per estimator, scenario and ``n``.

  Parameters
  ----------
  fits : pandas.DataFrame
      From :func:`fit_logs`.

  Returns
  -------
  pandas.DataFrame
      One row per (estimator, scenario, n).
  """
  out = (
    fits.groupby(["label", "estimator", "scenario", "n"], as_index=False)
    .agg(
      fits=("rep", "size"),
      true_family=("true_family", "mean"),
      near_cap=("near_cap", "sum"),
      median_stop=("cv_stop", "median"),
      max_stop=("cv_stop", "max"),
      active_kept=("active_kept", "mean"),
      inert_kept=("inert_kept", "mean"),
      not_converged=("converged", lambda c: int((c == "no").sum())),
    )
    .sort_values(["label", "scenario", "n"], ignore_index=True)
  )
  out["active"] = out["scenario"].map(lambda s: len(ACTIVE[s]))
  out["inert"] = len(COVARIATES) - out["active"]
  return out


def per_rep(metrics: pd.DataFrame, quantity: str, value: str) -> pd.DataFrame:
  """``value`` of ``quantity``, averaged over the points of each repetition.

  Parameters
  ----------
  metrics : pandas.DataFrame
      From :func:`load`.
  quantity : str
      ``pdf``, ``hfunc1`` or ``hfunc2``.
  value : str
      The metric column.

  Returns
  -------
  pandas.DataFrame
      One row per (cell, estimator), with ``value``.
  """
  rows = metrics[metrics["quantity"] == quantity].assign(
    failed=lambda d: d["error"] != ""
  )
  out = rows.groupby([*CELL, "label", "estimator"], as_index=False).agg(
    value=(value, "mean"), failed=("failed", "any")
  )
  # A failed fit has no score: NaN, rather than a mean over what it lacks.
  out.loc[out["failed"], "value"] = math.nan
  return out.rename(columns={"value": value})


def _mean_se(values: pd.Series) -> tuple[float, float]:
  n = int(values.notna().sum())
  sd = float(values.std(ddof=1)) if n > 1 else math.nan
  return float(values.mean()), sd / math.sqrt(n) if n else math.nan


def paired(reps: pd.DataFrame, value: str, by: list[str]) -> pd.DataFrame:
  """Each estimator's mean and its paired difference from :data:`BASELINE`.

  Within each repetition, ``value`` is first averaged over whatever of
  ``family`` is not in ``by``; the means and SEs are then over repetitions,
  and each difference pairs an estimator with the baseline on the same data.

  Parameters
  ----------
  reps : pandas.DataFrame
      From :func:`per_rep`.
  value : str
      The metric column.
  by : list of str
      The breakdown, from ``family``, ``scenario`` and ``n``.

  Returns
  -------
  pandas.DataFrame
      One row per breakdown level and estimator.
  """
  if BASELINE not in set(reps["label"]):
    raise SystemExit(f"the runs must hold {BASELINE!r}, the baseline; got none")
  unit = [*by, "rep"]
  # A repetition with a failed fit in any family it averages is missing:
  # the mean skips nothing.
  per = reps.groupby([*unit, "label", "estimator"], as_index=False).agg(
    value=(value, lambda v: v.mean(skipna=False)), failed=("failed", "any")
  )
  per = per.rename(columns={"value": value})
  base = per[per["label"] == BASELINE][[*unit, value]].rename(
    columns={value: "baseline"}
  )
  joined = per.merge(base, on=unit, how="inner")
  joined["diff"] = joined[value] - joined["baseline"]
  rows = []
  for keys, group in joined.groupby([*by, "label", "estimator"]):
    scored = group.dropna(subset=[value, "baseline"])
    mean, se = _mean_se(scored[value])
    diff, diff_se = _mean_se(scored["diff"])
    rows.append(
      {
        **dict(zip([*by, "label", "estimator"], keys, strict=True)),
        "mean": mean,
        "se": se,
        "diff_vs_tabpfn": diff,
        "diff_se": diff_se,
        "share_lower": float((scored["diff"] < 0).mean()),
        "reps": len(scored),
        "failed": int(group["failed"].sum()),
      }
    )
  return pd.DataFrame(rows)


def per_point(metrics: pd.DataFrame) -> pd.DataFrame:
  """The density's KL and IAE at each point, averaged over repetitions.

  Parameters
  ----------
  metrics : pandas.DataFrame
      From :func:`load`.

  Returns
  -------
  pandas.DataFrame
      One row per (family, scenario, n, estimator, point).
  """
  pdf = metrics[(metrics["quantity"] == "pdf") & (metrics["error"] == "")]
  return pdf.groupby(
    ["family", "scenario", "n", "label", "estimator", "point"], as_index=False
  ).agg(
    tau_true=("tau_true", "first"),
    KL=("KL_mc", "mean"),
    IAE=("IAE", "mean"),
    reps=("rep", "nunique"),
  )


def main() -> None:
  """Write E9's tables and print the main one."""
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("run", type=Path)
  parser.add_argument("--with", dest="others", type=Path, nargs="*", default=[])
  parser.add_argument("--out", type=Path, default=None)
  args = parser.parse_args()
  out = args.out or args.run / "analysis"
  out.mkdir(parents=True, exist_ok=True)

  metrics = load([args.run, *args.others])
  kl_reps = per_rep(metrics, "pdf", "KL_mc")
  kl = paired(kl_reps, "KL_mc", ["scenario", "n"])
  kl.to_csv(out / "kl.csv", index=False)
  paired(kl_reps, "KL_mc", ["family", "scenario", "n"]).to_csv(
    out / "kl_by_family.csv", index=False
  )
  iae = pd.concat(
    [
      paired(per_rep(metrics, q, "IAE"), "IAE", ["scenario", "n"]).assign(
        quantity=q
      )
      for q in ("pdf", "hfunc1", "hfunc2")
    ],
    ignore_index=True,
  )
  iae.to_csv(out / "iae.csv", index=False)
  per_point(metrics).to_csv(out / "per_point.csv", index=False)
  failures = (
    metrics[metrics["error"] != ""]
    .drop_duplicates([*CELL, "label"])[[*CELL, "label", "estimator", "error"]]
    .sort_values([*CELL, "label"])
  )
  failures.to_csv(out / "failures.csv", index=False)
  external = sorted(
    label
    for label in metrics["label"].unique()
    if label.startswith(("gamcopula", "boostcopula"))
  )
  fits = fit_logs([args.run, *args.others])
  reported = set(fits["label"]) if not fits.empty else set()
  if missing := [label for label in external if label not in reported]:
    raise SystemExit(
      f"every R estimator's fits must be reported in its run's logs/*.log; "
      f"got none for {missing}"
    )
  if not fits.empty:
    fits.to_csv(out / "fits.csv", index=False)
    summary = fits_summary(fits)
    summary.to_csv(out / "fits_summary.csv", index=False)

  table = kl.assign(
    cell=lambda d: (
      d["mean"].map("{:.4f}".format)
      + " ("
      + d["diff_vs_tabpfn"].map("{:+.4f}".format)
      + " ± "
      + d["diff_se"].map("{:.4f}".format)
      + ")"
    )
  ).pivot_table(
    index=["scenario", "n"], columns="estimator", values="cell", aggfunc="first"
  )
  pd.set_option("display.width", 250, "display.max_columns", 20)
  print(
    "KL_mc: mean over points, families and reps (difference from TabPFN v3 ± SE)"
  )
  print(table.to_string())
  if failures.empty:
    print("\nno fit failed")
  else:
    print(f"\n{len(failures)} fits failed; each is missing from every mean:")
    print(failures.groupby(["label", "scenario", "n"]).size().to_string())
  if not fits.empty:
    stopped = fits.dropna(subset=["cv_stop"])
    if not stopped.empty:
      print(
        f"\nboostCopula: {int(stopped['near_cap'].sum())} of {len(stopped)} "
        f"cross-validated stops within 5% of the cap (largest "
        f"{int(stopped['cv_stop'].max())} of {int(stopped['maxit'].max())})"
      )
    if (unconverged := int((fits["converged"] == "no").sum())) > 0:
      print(
        f"\n{unconverged} oracle fits stopped at n.iters unconverged "
        "(gamCopula: not to be trusted); they are scored as returned"
      )
    print("\nR fits: chosen family and covariates kept")
    print(summary.drop(columns="label").to_string(index=False))
  print(f"\nwrote {out}")


if __name__ == "__main__":
  main()
