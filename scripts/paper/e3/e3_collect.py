#!/usr/bin/env python3
"""E3: merge the per-method CSVs into long tables and summarize them.

Run in the NPCC environment (it needs pandas), from the repository root:

    cd /home/thibault/npcc
    uv run --no-sync python scripts/paper/e3/e3_collect.py

Rerun it whenever an arm finishes (the GPU arms arrive later); it reads
whatever per-method CSVs exist and lists the E1 estimators still missing.

**Inputs.** ``results/e3_vdc/e3a/{vdc_side,npcc_*}.csv`` and
``results/e3_vdc/e3b/{vdc_side,npcc_*}.csv``.

**Outputs** (in ``results/e3_vdc/``):

- ``e3a_long.csv`` / ``e3b_long.csv``: every row, one per case x seed x
  method, with its provenance columns;
- ``e3a_summary.csv``: per method, the mean KL over the 45 fits and its
  standard error, *stratified by case* (seeds are the replicates:
  ``sqrt(sum_c s_c^2 / n_seeds) / n_cases``), the paired difference to VDC
  with its stratified standard error, and the median and mean fit,
  projection and inference seconds;
- ``e3a_kl_by_case.csv``: mean KL per case and method (seeds averaged);
- ``e3b_summary.csv``: per case and method, the mean test NLL and excess NLL
  over the ten seeds with their standard errors (``sd / sqrt(n_seeds)``),
  the paired difference to VDC, and timings;
- ``e3_summary.json``: both summaries with the shared provenance.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_common as common  # noqa: E402
import e3_npcc  # noqa: E402

E1_COMMIT = "2f8fa94"
PROVENANCE_COLUMNS = [
  "vdc_tag", "vdc_commit", "vdc_main_merge_commit", "checkpoint_sha256",
  "pyvinecopulib_vdc_env", "pyvinecopulib_npcc_env", "npcc_commit",
  "npcc_src_dirty", "npcc_core_dirty", "tabpfn_version", "device", "gpu", "env",
]


def read_long(folder: Path) -> pd.DataFrame:
  """Every per-method CSV under ``folder``, stacked."""
  frames = [pd.read_csv(folder / "vdc_side.csv")]
  frames += [pd.read_csv(path) for path in sorted(folder.glob("npcc_*.csv"))]
  return pd.concat(frames, ignore_index=True, sort=False)


def provenance_of(group: pd.DataFrame) -> dict:
  """The provenance values of a group of rows, joined when they differ."""
  out = {}
  for col in PROVENANCE_COLUMNS:
    if col in group:
      values = sorted({str(v) for v in group[col].dropna()})
      out[col] = ";".join(values)
  return out


def timing(group: pd.DataFrame) -> dict:
  """Median and mean seconds, and the mean load average while they ran."""
  out = {}
  for col in ("fit_s", "projection_s", "inference_s"):
    values = pd.to_numeric(group[col], errors="coerce").dropna()
    out[f"{col}_median"] = float(values.median()) if len(values) else None
    out[f"{col}_mean"] = float(values.mean()) if len(values) else None
  load = pd.to_numeric(group.get("load_avg_1m"), errors="coerce").dropna()
  out["load_avg_1m_mean"] = float(load.mean()) if len(load) else None
  return out


def stratified(values: pd.DataFrame, column: str) -> tuple[float, float, int, int]:
  """Mean over cases of the per-case means, and its standard error."""
  by_case = values.groupby("case")[column]
  n_cases = by_case.ngroups
  n_seeds = int(by_case.count().min())
  mean = float(by_case.mean().mean())
  se = float(np.sqrt((by_case.var(ddof=1) / by_case.count()).sum()) / n_cases)
  return mean, se, n_cases, n_seeds


def summarize_e3a(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  vdc = long[long["method"] == "vdc"].set_index(["case", "seed"])["kl"]
  rows = []
  for method, group in long.groupby("method", sort=False):
    mean, se, n_cases, n_seeds = stratified(group, "kl")
    paired = group.set_index(["case", "seed"])["kl"] - vdc
    paired = paired.dropna().reset_index(name="d")
    d_mean, d_se, _, _ = stratified(paired, "d")
    rows.append({
      "method": method,
      "method_group": group["method_group"].iloc[0],
      "n_fits": len(group), "n_cases": n_cases, "n_seeds": n_seeds,
      "seeds": ";".join(str(s) for s in sorted(group["seed"].unique())),
      "n_train": int(group["n_train"].iloc[0]), "n_test": int(group["n_test"].iloc[0]),
      "mean_kl": mean, "se_kl": se,
      "mean_kl_minus_vdc": d_mean, "se_kl_minus_vdc": d_se,
      "max_kl": float(group["kl"].max()),
      "mean_kl_mc_se": float(group["kl_mc_se"].mean()),
      "n_floored_total": int(group["n_floored"].sum()),
      **timing(group),
      **provenance_of(group),
    })
  summary = pd.DataFrame(rows).sort_values("mean_kl", kind="stable")
  by_case = long.pivot_table(index="case", columns="method", values="kl", aggfunc="mean")
  order = long.drop_duplicates("case").sort_values("case_index")["case"]
  return summary, by_case.loc[order, summary["method"]]


def summarize_e3b(long: pd.DataFrame) -> pd.DataFrame:
  vdc = long[long["method"] == "vdc"].set_index(["case", "seed"])["excess_nll"]
  rows = []
  for (case, method), group in long.groupby(["case", "method"], sort=False):
    n = len(group)
    paired = (group.set_index(["case", "seed"])["excess_nll"] - vdc).dropna()
    rows.append({
      "case": case, "method": method,
      "method_group": group["method_group"].iloc[0],
      "conditions_on_u1": group["conditions_on_u1"].iloc[0],
      "n_seeds": n, "seeds": ";".join(str(s) for s in sorted(group["seed"].unique())),
      "n_train": int(group["n_train"].iloc[0]), "n_test": int(group["n_test"].iloc[0]),
      "mean_test_nll": float(group["test_nll"].mean()),
      "se_test_nll": float(group["test_nll"].std(ddof=1) / np.sqrt(n)),
      "mean_excess_nll": float(group["excess_nll"].mean()),
      "se_excess_nll": float(group["excess_nll"].std(ddof=1) / np.sqrt(n)),
      "mean_excess_minus_vdc": float(paired.mean()),
      "se_excess_minus_vdc": float(paired.std(ddof=1) / np.sqrt(len(paired))),
      "mean_oracle_nll": float(group["oracle_nll"].mean()),
      "n_floored_total": int(group["n_floored"].sum()),
      **timing(group),
      **provenance_of(group),
    })
  return pd.DataFrame(rows).sort_values(["case", "mean_excess_nll"], kind="stable")


def missing(long: pd.DataFrame) -> list[str]:
  have = set(long["method"])
  return [label for label in e3_npcc.ESTIMATORS if f"npcc-{label}" not in have]


def main() -> None:
  out = common.RESULTS
  report: dict = {"provenance": common.provenance(__file__, "n/a (collection only)")}
  # E1 ran at 2f8fa94. Rows written before `npcc_core_dirty` was recorded per
  # row rely on this: the estimator code is the same in HEAD as in E1.
  core_diff = subprocess.run(
    ["git", "-C", str(common.NPCC_REPO), "diff", "--stat", E1_COMMIT, "HEAD", "--",
     "src/npcc/core"], check=True, capture_output=True, text=True).stdout.strip()
  report["npcc_core_unchanged_since_e1_commit"] = {
    "e1_commit": E1_COMMIT, "unchanged": core_diff == "",
    "core_dirty_now": report["provenance"]["npcc_core_dirty"]}

  a_dir = out / "e3a"
  if (a_dir / "vdc_side.csv").exists():
    a_long = read_long(a_dir)
    a_long.to_csv(out / "e3a_long.csv", index=False)
    a_sum, a_case = summarize_e3a(a_long)
    a_sum.to_csv(out / "e3a_summary.csv", index=False)
    a_case.to_csv(out / "e3a_kl_by_case.csv")
    report["e3a"] = {"summary": a_sum.to_dict("records"), "pending": missing(a_long),
                     "published_mean_kl": {"vdc": 0.00611124, "tll": 0.01348007, "parametric": 0.00036485}}
    with pd.option_context("display.width", 200, "display.max_columns", 20):
      print(a_sum[["method", "n_fits", "mean_kl", "se_kl", "mean_kl_minus_vdc", "se_kl_minus_vdc",
                   "fit_s_median", "projection_s_median", "inference_s_median", "device"]].to_string(index=False))
      print("E3a pending:", missing(a_long))

  b_dir = out / "e3b"
  if (b_dir / "vdc_side.csv").exists():
    b_long = read_long(b_dir)
    b_long.to_csv(out / "e3b_long.csv", index=False)
    b_sum = summarize_e3b(b_long)
    b_sum.to_csv(out / "e3b_summary.csv", index=False)
    report["e3b"] = {"summary": b_sum.to_dict("records"), "pending": missing(b_long)}
    with pd.option_context("display.width", 200, "display.max_columns", 20):
      print(b_sum[["case", "method", "n_seeds", "mean_excess_nll", "se_excess_nll",
                   "mean_excess_minus_vdc", "se_excess_minus_vdc", "fit_s_median",
                   "inference_s_median", "device"]].to_string(index=False))
      print("E3b pending:", missing(b_long))

  common.write_json(json.loads(pd.Series(report).to_json()), out / "e3_summary.json")


if __name__ == "__main__":
  main()
