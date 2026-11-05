#!/usr/bin/env python3
"""E3: merge the per-method CSVs into long tables and summarize them.

Run in the NPCC environment (it needs pandas), from the repository root:

    cd /home/thibault/npcc
    uv run --no-sync python scripts/paper/e3/e3_collect.py

Rerun it whenever an arm finishes (the GPU arms arrive later); it reads
whatever per-method CSVs exist and lists the E1 estimators still missing.

**Roles.** Every summary row carries a ``role`` and a readable
``method_label``, assigned here from the method name (one place):

- ``arm``: a compared estimator -- VDC, TLL itself (no family selection),
  parametric family selection (BIC), and NPCC with each backend;
- ``sensitivity``: a variant of an arm (TLL with the quadratic method);
- ``oracle`` / ``reference``: the true conditional density, the true pooled
  density, and in E3b the separate copulas on each side of the known split,
  which no arm is told;
- ``control``: NPCC without ``U1`` in E3b;
- ``reproduction_check``: the TLL of VDC's Table 1 runner, a BIC choice
  between independence and TLL. It appears only in
  ``e3a_table1_reproduction.csv``, never in a summary;
- ``excluded``: the same BIC-selected TLL in E3b, kept in ``e3b_long.csv``
  for the record and in no summary.

**Inputs.** ``results/e3_vdc/e3a/{vdc_side,npcc_*}.csv`` and
``results/e3_vdc/e3b/{vdc_side,npcc_*}.csv``.

**Outputs** (in ``results/e3_vdc/``):

- ``e3a_long.csv`` / ``e3b_long.csv``: every row, one per case x seed x
  method (x n_train in E3a), with its provenance columns;
- ``e3a_summary.csv``: per n_train and method (arms and sensitivity rows),
  the mean KL, IAE and ISE over the 45 fits with standard errors
  *stratified by case* (seeds are the replicates: ``sqrt(sum_c s_c^2 /
  n_seeds) / n_cases``), the paired KL difference to VDC, and timings;
- ``e3a_by_case.csv``: the same metrics per n_train, case and method;
- ``e3a_table1_reproduction.csv``: VDC's Table 1 means, published, from the
  unmodified runner, and from ``e3a_vdc_side.py``;
- ``e3a_independence_probe.csv``: on the independence case, the largest
  fitted density on the 200 x 200 midpoint grid and where it occurs, per
  n_train, method and seed (when the grid metrics exist);
- ``e3a_vdc_crosscheck.csv``: per fit, VDC through its own lookup against
  VDC as a pyvinecopulib 1.0 pair copula (``vdc_adapted``): KL, IAE, ISE and
  the largest grid density of each, their gaps, and whether the two grids are
  bitwise equal;
- ``e3b_summary.csv``: per case and method, the mean test NLL and excess
  NLL over the ten seeds with standard errors (``sd / sqrt(n_seeds)``), the
  paired difference to VDC, and timings;
- ``e3c_long.csv`` / ``e3c_summary.csv``: per d and method, the mean held-out
  log-likelihood and excess NLL over the five seeds with standard errors,
  the paired difference to VDC (in the same vine machinery), timings, and
  VDC's server and inter-process seconds;
- ``e3c_crosscheck.csv``: the shared vine's TLL against pyvinecopulib's
  compiled TLL vine, and its NPCC against ``npcc.RosenblattVinecop``;
- ``e3_summary.json``: all of the summaries with the shared provenance.
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
SPLIT_KNOWN = "separate copulas on each side of the known split U1 = 0.5 (reference, not an arm)"
ROLES = {
  "E3a": {
    "vdc": ("arm", "VDC (vdc-denoiser-m64-v1, its grid lookup)"),
    "tll": ("arm", "TLL (pyvinecopulib, constant, mult 1; no family selection)"),
    "tll_quadratic": ("sensitivity", "TLL (pyvinecopulib, quadratic, mult 1; no family selection)"),
    "parametric": ("arm", "parametric family selection (BIC)"),
    "tll_runner_bic": ("reproduction_check",
                       "TLL as VDC's Table 1 runner fits it: BIC choice between independence and TLL"),
    "vdc_adapted": ("cross_check", "VDC as a pyvinecopulib 1.0 pair copula (padded grid, "
                                   "TorchTllBicop norm_maxiter=0; PR #217 adapted); cross-check of VDC's lookup"),
  },
  "E3c": {
    "vdc": ("arm", "VDC pair copula (density server, padded grid) in the shared pyvinecopulib 1.0 vine"),
    "tll": ("arm", "TLL (pyvinecopulib defaults: constant, mult 1, 30-point grid) in the shared vine"),
    "parametric": ("arm", "parametric family selection (BIC) per edge, pyvinecopulib 1.0 compiled vine"),
    "tll_cpp": ("cross_check", "pyvinecopulib 1.0 compiled TLL vine (checks the shared machinery's TLL)"),
    "rosenblatt-vinecop-check": ("cross_check", "npcc.RosenblattVinecop, SimplifiedContext, catboost-logit "
                                                "(checks the shared machinery's NPCC)"),
    "vdc_own": ("reference", "VDC's own vine implementation (D-vine on the true order, UCI-runner settings)"),
  },
  "E3b": {
    "oracle": ("oracle", "oracle: true conditional density c(u2, u3 | u1)"),
    "pooled_oracle": ("reference", "true pooled density (C_low + C_high) / 2: the floor for any "
                                   "estimator that ignores U1 (reference, not an arm)"),
    "vdc": ("arm", "VDC, pooled over U1"),
    "tll": ("arm", "TLL (constant, mult 1; no family selection), pooled over U1"),
    "tll_quadratic": ("sensitivity", "TLL (quadratic, mult 1; no family selection), pooled over U1"),
    "parametric": ("arm", "parametric family selection (BIC), pooled over U1"),
    "regime_gaussian": ("reference", "Gaussian copula, " + SPLIT_KNOWN + "; the appendix's 'Regime Gaussian'"),
    "vdc_regime": ("reference", "VDC, " + SPLIT_KNOWN),
    "tll_regime": ("reference", "TLL (no family selection), " + SPLIT_KNOWN),
    "parametric_regime": ("reference", "parametric family selection (BIC), " + SPLIT_KNOWN),
    "tll_runner_bic": ("excluded", "TLL as a BIC choice between independence and TLL (not an arm)"),
  },
}


def role_of(experiment: str, method: str) -> tuple[str, str]:
  """The ``(role, method_label)`` of ``method`` in ``experiment``."""
  if method.startswith("npcc-"):
    rest = method[len("npcc-"):]
    if rest.endswith("-pooled"):
      return "control", f"NPCC ({rest[:-len('-pooled')]}) without U1, pooled (control)"
    if experiment == "E3b":
      return "arm", f"NPCC ({rest}), conditional on U1"
    if experiment == "E3c":
      return "arm", f"NPCC ({rest}) in the shared pyvinecopulib 1.0 vine"
    return "arm", f"NPCC ({rest})"
  try:
    return ROLES[experiment][method]
  except KeyError:
    raise KeyError(f"{experiment} method {method!r} must have a role; add it to ROLES") from None


def read_long(folder: Path, experiment: str) -> pd.DataFrame:
  """Every per-method CSV under ``folder``, stacked, with roles assigned."""
  frames = [pd.read_csv(folder / "vdc_side.csv")]
  frames += [pd.read_csv(path) for path in sorted(folder.glob("npcc_*.csv"))]
  long = pd.concat(frames, ignore_index=True, sort=False)
  if "n_train" in long:
    long["n_train"] = long["n_train"].astype(int)
  roles = long["method"].map(lambda m: role_of(experiment, m))
  if "role" in long:
    written = long["role"].notna()
    assigned = roles.map(lambda r: r[0])
    if (long.loc[written, "role"] != assigned[written]).any():
      bad = long.loc[written & (long["role"] != assigned), ["method", "role"]].drop_duplicates()
      raise ValueError(f"role column must agree with ROLES; got {bad.to_dict('records')}")
  long["role"] = roles.map(lambda r: r[0])
  long["method_label"] = roles.map(lambda r: r[1])
  return long


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
  for col in ("fit_s", "projection_s", "inference_s", "grid_s"):
    if col not in group:
      continue
    values = pd.to_numeric(group[col], errors="coerce").dropna()
    out[f"{col}_median"] = float(values.median()) if len(values) else None
    out[f"{col}_mean"] = float(values.mean()) if len(values) else None
  load = pd.to_numeric(group.get("load_avg_1m"), errors="coerce").dropna()
  out["load_avg_1m_mean"] = float(load.mean()) if len(load) else None
  return out


def stratified(values: pd.DataFrame, column: str) -> tuple[float, float, int, int]:
  """Mean over cases of the per-case means, and its standard error."""
  by_case = pd.to_numeric(values[column], errors="coerce").groupby(values["case"])
  n_cases = by_case.ngroups
  n_seeds = int(by_case.count().min())
  mean = float(by_case.mean().mean())
  se = float(np.sqrt((by_case.var(ddof=1) / by_case.count()).sum()) / n_cases)
  return mean, se, n_cases, n_seeds


def summarize_e3a(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  shown = long[long["role"].isin(["arm", "sensitivity"])]
  metrics = [m for m in ("kl", "iae", "ise") if m in shown and shown[m].notna().any()]
  rows, case_rows = [], []
  for (n_train, method), group in shown.groupby(["n_train", "method"], sort=False):
    vdc = shown[(shown["method"] == "vdc") & (shown["n_train"] == n_train)]
    vdc = vdc.set_index(["case", "seed"])["kl"]
    row = {
      "n_train": n_train, "method": method, "role": group["role"].iloc[0],
      "method_label": group["method_label"].iloc[0],
      "n_fits": len(group),
      "seeds": ";".join(str(s) for s in sorted(group["seed"].unique())),
      "n_test": int(group["n_test"].iloc[0]),
    }
    for metric in metrics:
      mean, se, n_cases, n_seeds = stratified(group, metric)
      row.update({f"mean_{metric}": mean, f"se_{metric}": se})
    row.update({"n_cases": n_cases, "n_seeds": n_seeds})
    paired = (group.set_index(["case", "seed"])["kl"] - vdc).dropna().reset_index(name="d")
    if len(paired):
      d_mean, d_se, _, _ = stratified(paired, "d")
      row.update({"mean_kl_minus_vdc": d_mean, "se_kl_minus_vdc": d_se})
    row.update({
      "max_kl": float(group["kl"].max()),
      "mean_kl_mc_se": float(group["kl_mc_se"].mean()),
      "n_floored_total": int(group["n_floored"].sum()),
      **timing(group), **provenance_of(group),
    })
    rows.append(row)
    for case, by_case in group.groupby("case", sort=False):
      case_row = {"n_train": n_train, "case_index": int(by_case["case_index"].iloc[0]),
                  "case": case, "method": method, "role": row["role"],
                  "method_label": row["method_label"], "n_seeds": len(by_case)}
      for metric in metrics:
        values = pd.to_numeric(by_case[metric], errors="coerce")
        case_row[f"mean_{metric}"] = float(values.mean())
        case_row[f"se_{metric}"] = float(values.std(ddof=1) / np.sqrt(len(values)))
      case_rows.append(case_row)
  summary = pd.DataFrame(rows).sort_values(["n_train", "mean_kl"], kind="stable")
  by_case = pd.DataFrame(case_rows).sort_values(["n_train", "case_index", "mean_kl"], kind="stable")
  return summary, by_case


def table1_reproduction(folder: Path) -> pd.DataFrame:
  """VDC's Table 1 means: published, the unmodified runner, and our rerun."""
  info = json.loads((folder / "vdc_side.json").read_text())
  runner = json.loads((common.RESULTS / "table1_rerun_cpu" / "results.json").read_text())
  rows = []
  for method, values in info["table1_reproduction"].items():
    rows.append({
      "method": method, "role": "reproduction_check" if method == "tll_runner_bic" else "arm",
      "method_label": role_of("E3a", method)[1],
      "published_mean_kl": values["published"],
      "runner_rerun_mean_kl": values["runner_rerun"],
      "e3a_vdc_side_mean_kl": values["this_rerun"],
      "runner_minus_published": values["runner_rerun"] - values["published"],
      "max_abs_kl_gap_e3a_vs_runner": info["max_abs_kl_gap_vs_runner"][method],
      "n_fits": len(runner["rows"]), "n_train": 2000, "n_test": 20000,
      "seeds": "101;202;303;404;505",
      "runner_source_sha256": runner["source_sha256"],
      "checkpoint_sha256": runner["checkpoint_sha256"],
      "runner_device": runner["device"],
      "runner_versions": json.dumps(runner["versions"], sort_keys=True),
      "runner_wall_seconds": runner["wall_seconds"],
      "vdc_commit": info["provenance"]["vdc_commit"],
      "vdc_main_merge_commit": info["provenance"]["vdc_main_merge_commit"],
      "pyvinecopulib_vdc_env": info["provenance"]["pyvinecopulib_vdc_env"],
      "pyvinecopulib_npcc_env": info["provenance"]["pyvinecopulib_npcc_env"],
      "npcc_commit": info["provenance"]["npcc_commit"],
      "tabpfn_version": info["provenance"]["tabpfn_version"],
    })
  return pd.DataFrame(rows)


def independence_probe(long: pd.DataFrame) -> pd.DataFrame | None:
  """The largest fitted density on the grid for the independence case."""
  cols = ["grid_max_pdf", "grid_argmax_u", "grid_argmax_v"]
  if not all(c in long for c in cols):
    return None
  probe = long[(long["family"] == "indep") & long["grid_max_pdf"].notna()
               & long["role"].isin(["arm", "sensitivity", "cross_check"])]
  return probe[["n_train", "method", "role", "method_label", "seed", *cols,
                "grid_max_ratio", "iae", "ise", "kl"]].sort_values(["n_train", "method", "seed"])


def vdc_crosscheck(long: pd.DataFrame) -> pd.DataFrame | None:
  """VDC through its own lookup against the adapted pyvinecopulib 1.0 pair."""
  own = long[long["method"] == "vdc"]
  adapted = long[long["method"] == "vdc_adapted"]
  if adapted.empty or "iae" not in own:
    return None
  keys = ["n_train", "case", "seed"]
  both = own.merge(adapted, on=keys, suffixes=("_own", "_adapted"))
  out = both[keys + ["kl_own", "kl_adapted", "iae_own", "iae_adapted", "ise_own", "ise_adapted",
                     "grid_max_pdf_own", "grid_max_pdf_adapted", "grid_equal_vdc_side_adapted",
                     "grid_max_abs_diff_vdc_side_adapted"]].copy()
  for m in ("kl", "iae", "ise", "grid_max_pdf"):
    out[f"{m}_gap"] = out[f"{m}_adapted"] - out[f"{m}_own"]
  return out


def summarize_e3c(long: pd.DataFrame) -> pd.DataFrame:
  shown = long[long["role"].isin(["arm", "reference"])]
  vdc = shown[shown["method"] == "vdc"].set_index(["d", "seed"])["excess_nll"]
  rows = []
  for (d, method), group in shown.groupby(["d", "method"], sort=False):
    n = len(group)
    paired = (group.set_index(["d", "seed"])["excess_nll"] - vdc).dropna()
    row = {
      "d": d, "method": method, "role": group["role"].iloc[0],
      "method_label": group["method_label"].iloc[0],
      "vine_machinery": group["vine_machinery"].iloc[0],
      "n_seeds": n, "seeds": ";".join(str(s) for s in sorted(group["seed"].unique())),
      "n_train": int(group["n_train"].iloc[0]), "n_test": int(group["n_test"].iloc[0]),
      "n_edges": int(group["n_edges"].iloc[0]),
      "mean_test_loglik": float(group["test_loglik"].mean()),
      "se_test_loglik": float(group["test_loglik"].std(ddof=1) / np.sqrt(n)),
      "mean_true_loglik": float(group["true_loglik"].mean()),
      "mean_excess_nll": float(group["excess_nll"].mean()),
      "se_excess_nll": float(group["excess_nll"].std(ddof=1) / np.sqrt(n)),
      "mean_excess_minus_vdc": float(paired.mean()) if len(paired) else None,
      "se_excess_minus_vdc": float(paired.std(ddof=1) / np.sqrt(len(paired))) if len(paired) > 1 else None,
      **timing(group),
    }
    for col in ("vdc_server_s", "vdc_projection_s", "vdc_ipc_s"):
      if col in group and group[col].notna().any():
        row[f"{col}_mean"] = float(group[col].mean())
    row.update({
      "timing_note": "true structure for every arm, no selection; one pinned CPU core per arm. "
                     "VDC's published vine timings compared TLL with R-vine selection on a CPU "
                     "against VDC on a greedy D-vine on an H100.",
      **provenance_of(group),
    })
    rows.append(row)
  order = {"arm": 0, "reference": 1}
  out = pd.DataFrame(rows)
  out["_order"] = out["role"].map(order)
  return out.sort_values(["d", "_order", "mean_excess_nll"], kind="stable").drop(columns="_order")


def e3c_crosscheck(long: pd.DataFrame) -> pd.DataFrame | None:
  """The shared machinery against pyvinecopulib's compiled vine and npcc's own."""
  pairs = [("tll", "tll_cpp"), ("npcc-catboost-logit", "rosenblatt-vinecop-check")]
  rows = []
  for a_m, b_m in pairs:
    a = long[long["method"] == a_m].set_index(["d", "seed"])["test_loglik"]
    b = long[long["method"] == b_m].set_index(["d", "seed"])["test_loglik"]
    both = pd.concat([a.rename("shared"), b.rename("reference")], axis=1).dropna()
    for (d, seed), r in both.iterrows():
      rows.append({"shared_arm": a_m, "check_arm": b_m, "d": d, "seed": seed,
                   "loglik_shared": r["shared"], "loglik_check": r["reference"],
                   "abs_gap": abs(r["shared"] - r["reference"])})
  return pd.DataFrame(rows) if rows else None


def summarize_e3b(long: pd.DataFrame) -> pd.DataFrame:
  shown = long[long["role"] != "excluded"]
  vdc = shown[shown["method"] == "vdc"].set_index(["case", "seed"])["excess_nll"]
  rows = []
  for (case, method), group in shown.groupby(["case", "method"], sort=False):
    n = len(group)
    paired = (group.set_index(["case", "seed"])["excess_nll"] - vdc).dropna()
    rows.append({
      "case": case, "method": method, "role": group["role"].iloc[0],
      "method_label": group["method_label"].iloc[0],
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
  order = {"oracle": 0, "reference": 1, "arm": 2, "sensitivity": 3, "control": 4}
  out = pd.DataFrame(rows)
  out["_order"] = out["role"].map(order)
  return out.sort_values(["case", "_order", "mean_excess_nll"], kind="stable").drop(columns="_order")


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
  def show():  # noqa: ANN202 - a pandas option context, entered once per use
    return pd.option_context("display.width", 250, "display.max_columns", 30)

  a_dir = out / "e3a"
  if (a_dir / "vdc_side.csv").exists():
    a_long = read_long(a_dir, "E3a")
    a_long.to_csv(out / "e3a_long.csv", index=False)
    a_sum, a_case = summarize_e3a(a_long)
    a_sum.to_csv(out / "e3a_summary.csv", index=False)
    a_case.to_csv(out / "e3a_by_case.csv", index=False)
    repro = table1_reproduction(a_dir)
    repro.to_csv(out / "e3a_table1_reproduction.csv", index=False)
    probe = independence_probe(a_long)
    report["e3a"] = {"summary": a_sum.to_dict("records"), "pending": missing(a_long),
                     "table1_reproduction": repro.to_dict("records")}
    if probe is not None:
      probe.to_csv(out / "e3a_independence_probe.csv", index=False)
      report["e3a"]["independence_probe"] = probe.to_dict("records")
    xcheck = vdc_crosscheck(a_long)
    if xcheck is not None:
      xcheck.to_csv(out / "e3a_vdc_crosscheck.csv", index=False)
      report["e3a"]["vdc_crosscheck"] = {
        "n_fits": len(xcheck), "all_grids_equal": bool(xcheck["grid_equal_vdc_side_adapted"].all()),
        **{f"max_abs_{m}_gap": float(xcheck[f"{m}_gap"].abs().max()) for m in ("kl", "iae", "ise", "grid_max_pdf")}}
      print("VDC cross-check:", report["e3a"]["vdc_crosscheck"])
    with show():
      cols = [c for c in ["n_train", "method", "role", "n_fits", "mean_kl", "se_kl", "mean_iae",
                          "mean_ise", "mean_kl_minus_vdc", "fit_s_median", "inference_s_median",
                          "device"] if c in a_sum]
      print(a_sum[cols].to_string(index=False))
      print(repro[["method", "published_mean_kl", "runner_rerun_mean_kl",
                   "e3a_vdc_side_mean_kl"]].to_string(index=False))
      print("E3a pending:", missing(a_long))

  b_dir = out / "e3b"
  if (b_dir / "vdc_side.csv").exists():
    b_long = read_long(b_dir, "E3b")
    b_long.to_csv(out / "e3b_long.csv", index=False)
    b_sum = summarize_e3b(b_long)
    b_sum.to_csv(out / "e3b_summary.csv", index=False)
    report["e3b"] = {"summary": b_sum.to_dict("records"), "pending": missing(b_long)}
    with show():
      print(b_sum[["case", "method", "role", "n_seeds", "mean_excess_nll", "se_excess_nll",
                   "mean_excess_minus_vdc", "fit_s_median", "inference_s_median",
                   "device"]].to_string(index=False))
      print("E3b pending:", missing(b_long))

  c_dir = out / "e3c"
  if (c_dir / "vdc_side.csv").exists():
    c_long = read_long(c_dir, "E3c")
    c_long.to_csv(out / "e3c_long.csv", index=False)
    c_sum = summarize_e3c(c_long)
    c_sum.to_csv(out / "e3c_summary.csv", index=False)
    report["e3c"] = {"summary": c_sum.to_dict("records"), "pending": missing(c_long)}
    c_check = e3c_crosscheck(c_long)
    if c_check is not None:
      c_check.to_csv(out / "e3c_crosscheck.csv", index=False)
      report["e3c"]["crosscheck_max_abs_gap"] = c_check.groupby("check_arm")["abs_gap"].max().to_dict()
    with show():
      print(c_sum[["d", "method", "role", "n_seeds", "mean_test_loglik", "mean_true_loglik",
                   "mean_excess_nll", "se_excess_nll", "mean_excess_minus_vdc",
                   "fit_s_median", "inference_s_median"]].to_string(index=False))
      if c_check is not None:
        print(c_check.groupby("check_arm")["abs_gap"].max())

  common.write_json(json.loads(pd.Series(report).to_json()), out / "e3_summary.json")


if __name__ == "__main__":
  main()
