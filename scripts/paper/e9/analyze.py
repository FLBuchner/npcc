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
  point, the KL and IAE averaged over repetitions, with the point's true tau.

Usage (from the npcc repository root)::

    uv run --no-sync python scripts/paper/e9/analyze.py \\
      results/study_covariates_<TAG> --with results/study_covariates_<TAG>_r
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import pandas as pd

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
  "gamcopula-true": "gamCopula, true formula",
  "boostcopula-smooth-cv": "boostCopula-CV",
}

CELL: list[str] = ["family", "scenario", "n", "rep"]


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
  pdf = metrics[metrics["quantity"] == "pdf"]
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
  rows = metrics[metrics["quantity"] == quantity]
  return rows.groupby([*CELL, "label", "estimator"], as_index=False)[
    value
  ].mean()


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
  per = reps.groupby([*unit, "label", "estimator"], as_index=False)[
    value
  ].mean()
  base = per[per["label"] == BASELINE][[*unit, value]].rename(
    columns={value: "baseline"}
  )
  joined = per.merge(base, on=unit, how="inner")
  joined["diff"] = joined[value] - joined["baseline"]
  rows = []
  for keys, group in joined.groupby([*by, "label", "estimator"]):
    mean, se = _mean_se(group[value])
    diff, diff_se = _mean_se(group["diff"])
    rows.append(
      {
        **dict(zip([*by, "label", "estimator"], keys, strict=True)),
        "mean": mean,
        "se": se,
        "diff_vs_tabpfn": diff,
        "diff_se": diff_se,
        "share_lower": float((group["diff"] < 0).mean()),
        "reps": len(group),
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
  pdf = metrics[metrics["quantity"] == "pdf"]
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
  print(f"\nwrote {out}")


if __name__ == "__main__":
  main()
