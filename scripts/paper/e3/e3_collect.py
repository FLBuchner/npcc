#!/usr/bin/env python3
"""E3: merge the per-method CSVs into long tables and summarize them.

Run in the NPCC environment (it needs pandas), from the repository root:

    cd /home/thibault/npcc
    uv run --no-sync python scripts/paper/e3/e3_collect.py

Rerun it whenever an arm finishes (the GPU arms arrive later); it reads
whatever has been written, E3c's and E3d's cells as they land, and lists the
E1 estimators still missing.

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

**Inputs.** Each sub-experiment's ``vdc_side.csv`` and its npcc side: the
per-method CSVs ``npcc_*.csv`` in ``results/e3_vdc/e3a/`` and ``e3b/``, and
the per-cell JSON files of E3c (``npcc_cells/<arm>/d*.json``) and E3d
(``cells/<arm>/*.json``), since a fit script writes its CSV only when its run
ends (``read_long``).

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
- ``e3c_long.csv`` / ``e3c_summary.csv``: per design (``decay``, the primary
  one, from ``results/e3_vdc/e3c_decay/``; ``strong``, the stress variant,
  from ``results/e3_vdc/e3c/``), d and method, the mean held-out
  log-likelihood and excess NLL over the five seeds with standard errors,
  the paired difference to VDC (in the same vine machinery), timings, and
  VDC's server and inter-process seconds;
- E3c and E3d are scored on the *common set*: the test points where every
  arm of a cell and the truth are finite (``common_set``; the paper lead's
  rule for the few NaN densities pyvinecopulib's compiled vine returns), with
  ``n_excluded`` per row;
- ``e3c_crosscheck.csv``: per design, the shared vine's TLL against
  pyvinecopulib's compiled TLL vine, and its NPCC against
  ``npcc.RosenblattVinecop``;
- in ``results/e3d_unknown_structure/``: ``e3d_long.csv`` and
  ``e3d_summary.csv`` (per design, d and arm: excess NLL with its standard
  error over the seeds, the paired gap to each competitor that selects a
  structure -- VDC, TLL, parametric -- the structure recovery and timings);
- ``e3d_paired.csv``, beside them, the numbers to cite: per design and d,
  each NPCC arm's paired difference to VDC, parametric and TLL with its
  standard error and wins, (a) minus (b) and (c) minus (a) per backend, (b)
  minus VDC on the true structure, the structure recovery of every arm and
  competitor, and the excluded test points (``paired_e3d``);
- ``e3_code_check.csv``: every code hash the rows recorded (``code_sha256``,
  and ``vdc_server_code_sha256`` where VDC's server ran) against committed
  code and the file on disk, with the commit holding each version, plus the
  Table 1 runner's ``source_sha256``. The collector exits with an error if
  any row came from code that is neither, unless ``--allow-stale``;
- ``e3_summary.json``: all of the summaries with the shared provenance and
  the code check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import e3_common as common  # noqa: E402
import e3_npcc  # noqa: E402
import readout  # noqa: E402

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
  "E3d": {
    "vdc": ("arm", "VDC, its own pipeline and structure choice (greedy D-vine)"),
    "parametric": ("arm", "parametric family selection (BIC), Dissmann structure selection"),
    "tll": ("arm", "TLL (pyvinecopulib defaults), Dissmann structure selection"),
    "parametric_true": ("reference", "parametric family selection (BIC) on the true structure"),
    "tll_true": ("reference", "TLL on the true structure"),
    "vdc_true": ("reference", "VDC pair copulas in the shared vine on the true structure"),
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


E3D_VARIANTS = {
  "random-ns": ("arm", "random structure, non-simplified"),
  "tll-ns": ("arm", "TLL's selected structure, non-simplified"),
  "random-s": ("ablation", "random structure, simplified (ablation)"),
  "true-s": ("reference", "true structure, simplified (reference)"),
}
#: The experiment plan's letters for E3d's NPCC arms; ``true-s`` is a reference.
E3D_LETTERS = {"random-ns": "a", "tll-ns": "b", "random-s": "c"}
#: The competitors, each with its own structure selection.
E3D_COMPETITORS = ("vdc", "parametric", "tll")
E3D_RESULTS = common.NPCC_REPO / "results" / "e3d_unknown_structure"


def e3d_npcc(method: str) -> tuple[str, str] | None:
  """``(backend, variant)`` of an E3d NPCC method, or None for any other."""
  if method.startswith("npcc-"):
    for variant in E3D_VARIANTS:
      if method.endswith(f"-{variant}"):
        return method[len("npcc-"):-len(variant) - 1], variant
  return None


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
    if experiment == "E3d" and (parsed := e3d_npcc(method)):
      backend, variant = parsed
      role, text = E3D_VARIANTS[variant]
      return role, f"NPCC ({backend}), {text}"
    return "arm", f"NPCC ({rest})"
  try:
    return ROLES[experiment][method]
  except KeyError:
    raise KeyError(f"{experiment} method {method!r} must have a role; add it to ROLES") from None


def readout_check(long: pd.DataFrame, experiment: str) -> pd.DataFrame:
  """``long``, once every NPCC row in it is known to use the current read-out.

  The cross-check arm fits CatBoost through the read-out too. VDC, TLL and the
  parametric fits do not touch it and keep the commits they ran at.
  """
  method = long["method"].astype(str)
  npcc = method.str.startswith("npcc-") | (method == "rosenblatt-vinecop-check")
  readout.check(long.loc[npcc, "npcc_commit"], f"{experiment}'s NPCC rows")
  return long


def read_long(folder: Path, experiment: str, cells: str | None = None) -> pd.DataFrame:
  """Every row under ``folder``, stacked, with roles assigned.

  The VDC side's rows come from ``vdc_side.csv``. The npcc side's come from
  the per-method CSVs, or, given ``cells`` (a glob under ``folder``), from
  the per-cell JSON files those CSVs are made of. A fit script writes its
  CSV only when its run ends, but each cell as soon as it is scored, so
  reading the cells sees a stopped or still-running stage up to its last
  finished cell. E3c and E3d read the cells: they pair by seed and state
  ``n_seeds``. E3a and E3b read the CSVs, which hold every case of a
  finished run; E3a's summary averages over cases, and a partial run would
  leave some of them out.
  """
  frames = [pd.read_csv(folder / "vdc_side.csv")]
  if cells is None:
    frames += [pd.read_csv(path) for path in sorted(folder.glob("npcc_*.csv"))]
  else:
    rows = [json.loads(path.read_text()) for path in sorted(folder.glob(cells))]
    if rows:
      frames.append(pd.DataFrame(rows))
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


#: The primary E3c design first, then the stress variant.
DESIGN_ORDER = {"decay": 0, "strong": 1}
#: E3c's folder for each design. The folder is what names the design: the
#: decay design's ``vdc_side.json`` has ``"design": "decay"``, but in the
#: strong design's the ``design`` key describes the sampling (dims, seeds,
#: cases), and its rows have no ``design`` column.
E3C_FOLDERS = {"decay": common.RESULTS / "e3c_decay", "strong": common.RESULTS / "e3c"}


FLOOR = float(np.log(common.LOG_FLOOR))


def common_set(long: pd.DataFrame, experiment: str, folders: dict[str, Path]) -> pd.DataFrame:
  """Score every arm on the test points where all arms and the truth are finite.

  The paper lead's rule for densities an arm cannot evaluate: no floor, and
  one common set of test points per design, d and seed, shared by every arm
  of that cell. Each arm's per-point log-densities are read from its
  ``.npy``, stored as float32, so a common-set mean agrees with a row's own
  float64 mean to about 1e-8. E3c's npcc-side arrays hold the floor where
  the density was NaN, so there a point at the floor counts as non-finite,
  and ``mask_exact`` says whether those points are exactly the row's
  ``n_nonfinite``. E3d's arrays and VDC's own-vine arrays are raw, NaN kept.

  Returns
  -------
  DataFrame
      ``long`` with ``excess_nll``, ``test_loglik`` and ``true_loglik``
      recomputed on the common set (the row's own values kept as
      ``*_row``), and ``n_excluded``, ``n_common``, ``n_bad_arm`` and
      ``mask_exact`` per row.
  """
  # A row without a design would fall out of every groupby below, silently.
  if long["design"].isna().any():
    raise ValueError(f"{experiment} rows must each have a design; got "
                     f"{int(long['design'].isna().sum())} without")
  out = long.copy()
  for col in ("excess_nll", "test_loglik", "true_loglik"):
    out[f"{col}_row"] = out[col]
  for col in ("n_excluded", "n_common", "n_bad_arm"):
    out[col] = pd.Series(pd.NA, index=out.index, dtype="Int64")
  out["mask_exact"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
  for (design, d, seed), group in out.groupby(["design", "d", "seed"], sort=False):
    folder = folders[design]
    if experiment == "E3c":
      key, logtrue_key = f"d{d}_s{seed}", f"d{d}_s{seed}_logtrue"
    else:
      key = f"{design}_d{d}_s{seed}"
      logtrue_key = f"{key}_logtrue"
    with np.load(folder / "samples.npz") as f:
      logtrue = f[logtrue_key].astype(np.float64)
    arrays, bad = {}, {}
    for idx, row in group.iterrows():
      method = row["method"]
      if experiment == "E3c":
        path = (folder / "vdc_own_cells" / f"{key}.npy" if method == "vdc_own"
                else folder / "npcc_cells" / method / f"{key}.npy")
        floored = method != "vdc_own"
      else:
        path = folder / "cells" / method / f"{key}.npy"
        floored = False
      if not path.exists():
        raise FileNotFoundError(f"{path} must exist to score {method} on the common set")
      lp = np.load(path).astype(np.float64)
      b = ~np.isfinite(lp)
      if floored:
        b |= lp <= FLOOR + 1e-3
      arrays[idx], bad[idx] = lp, b
      out.loc[idx, "n_bad_arm"] = int(b.sum())
      nonfinite = row.get("n_nonfinite")
      out.loc[idx, "mask_exact"] = bool(int(b.sum()) == int(0 if pd.isna(nonfinite) else nonfinite))
    keep = np.isfinite(logtrue)
    for b in bad.values():
      keep &= ~b
    for idx, lp in arrays.items():
      test = np.maximum(lp[keep], FLOOR)
      out.loc[idx, "excess_nll"] = float(np.mean(logtrue[keep] - test))
      out.loc[idx, "test_loglik"] = float(np.mean(test))
      out.loc[idx, "true_loglik"] = float(np.mean(logtrue[keep]))
      out.loc[idx, "n_excluded"] = int((~keep).sum())
      out.loc[idx, "n_common"] = int(keep.sum())
  return out


def summarize_e3c(long: pd.DataFrame) -> pd.DataFrame:
  shown = long[long["role"].isin(["arm", "reference"])]
  vdc = shown[shown["method"] == "vdc"].set_index(["design", "d", "seed"])["excess_nll"]
  rows = []
  for (design, d, method), group in shown.groupby(["design", "d", "method"], sort=False):
    n = len(group)
    paired = (group.set_index(["design", "d", "seed"])["excess_nll"] - vdc).dropna()
    row = {
      "design": design, "d": d, "method": method, "role": group["role"].iloc[0],
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
      "n_nonfinite_total": int(pd.to_numeric(group.get("n_nonfinite"), errors="coerce").fillna(0).sum()),
      "n_excluded_total": int(pd.to_numeric(group.get("n_excluded"), errors="coerce").fillna(0).sum()),
      "share_excluded": float(pd.to_numeric(group.get("n_excluded"), errors="coerce").fillna(0).sum()
                              / (n * int(group["n_test"].iloc[0]))),
      **timing(group),
    }
    if "excess_nll_finite" in group and group["excess_nll_finite"].notna().any():
      row["mean_excess_nll_finite"] = float(pd.to_numeric(group["excess_nll_finite"]).mean())
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
  out["_design"] = out["design"].map(DESIGN_ORDER)
  return out.sort_values(["_design", "d", "_order", "mean_excess_nll"],
                         kind="stable").drop(columns=["_order", "_design"])


def e3c_crosscheck(long: pd.DataFrame) -> pd.DataFrame | None:
  """The shared machinery against pyvinecopulib's compiled vine and npcc's own.

  It compares each row's own log-likelihood, computed in float64 by the
  process that fitted it, not the common-set one, which is computed from the
  stored float32 log-densities and would hide a gap below about 1e-8.
  """
  pairs = [("tll", "tll_cpp"), ("npcc-catboost-logit", "rosenblatt-vinecop-check")]
  col = "test_loglik_row" if "test_loglik_row" in long else "test_loglik"
  rows = []
  for a_m, b_m in pairs:
    a = long[long["method"] == a_m].set_index(["design", "d", "seed"])[col]
    b = long[long["method"] == b_m].set_index(["design", "d", "seed"])[col]
    both = pd.concat([a.rename("shared"), b.rename("reference")], axis=1).dropna()
    for (design, d, seed), r in both.iterrows():
      rows.append({"design": design, "shared_arm": a_m, "check_arm": b_m, "d": d, "seed": seed,
                   "loglik_shared": r["shared"], "loglik_check": r["reference"],
                   "abs_gap": abs(r["shared"] - r["reference"])})
  return pd.DataFrame(rows) if rows else None


def summarize_e3d(long: pd.DataFrame) -> pd.DataFrame:
  """Per design, d and arm: excess NLL, paired gaps to each selecting competitor, recovery."""
  rows = []
  competitors = ("vdc", "tll", "parametric")
  base = {c: long[long["method"] == c].set_index(["design", "d", "seed"])["excess_nll"] for c in competitors}
  for (design, d, method), group in long.groupby(["design", "d", "method"], sort=False):
    n = len(group)
    row = {
      "design": design, "d": d, "method": method, "role": group["role"].iloc[0],
      "method_label": group["method_label"].iloc[0], "structure": group["structure"].iloc[0],
      "n_seeds": n, "seeds": ";".join(str(s) for s in sorted(group["seed"].unique())),
      "n_train": int(group["n_train"].iloc[0]), "n_test": int(group["n_test"].iloc[0]),
      "mean_true_loglik": float(group["true_loglik"].mean()),
      "mean_test_loglik": float(group["test_loglik"].mean()),
      "mean_excess_nll": float(group["excess_nll"].mean()),
      "se_excess_nll": float(group["excess_nll"].std(ddof=1) / np.sqrt(n)),
    }
    indexed = group.set_index(["design", "d", "seed"])["excess_nll"]
    for c in competitors:
      paired = (indexed - base[c]).dropna()
      if len(paired) > 1 and c != method:
        row[f"mean_excess_minus_{c}"] = float(paired.mean())
        row[f"se_excess_minus_{c}"] = float(paired.std(ddof=1) / np.sqrt(len(paired)))
    row.update({
      "mean_recovery_tree1": float(group["recovery_tree1"].mean()),
      "mean_recovery_overall": float(group["recovery_overall"].mean()),
      "share_structure_equals_truth": float(group["structure_equals_truth"].astype(str).eq("True").mean()),
      "n_nonfinite_total": int(pd.to_numeric(group.get("n_nonfinite"), errors="coerce").fillna(0).sum()),
      "n_excluded_total": int(pd.to_numeric(group.get("n_excluded"), errors="coerce").fillna(0).sum()),
      "share_excluded": float(pd.to_numeric(group.get("n_excluded"), errors="coerce").fillna(0).sum()
                              / (n * int(group["n_test"].iloc[0]))),
      **timing(group), **provenance_of(group),
    })
    rows.append(row)
  order = {"arm": 0, "ablation": 1, "reference": 2}
  out = pd.DataFrame(rows)
  out["_o"] = out["role"].map(order)
  out["_d"] = out["design"].map(DESIGN_ORDER)
  return out.sort_values(["_d", "d", "_o", "mean_excess_nll"], kind="stable").drop(columns=["_o", "_d"])


#: ``block``, ``method`` variant, ``versus`` variant: E3d's within-backend contrasts.
E3D_CONTRASTS = (
  ("random_minus_selected", "random-ns", "tll-ns"),
  ("simplified_minus_nonsimplified", "random-s", "random-ns"),
)
E3D_PAIRED_COLUMNS = [
  "design", "d", "block", "method", "versus", "backend", "variant", "n_seeds", "seeds",
  "mean_method", "mean_versus", "mean_diff", "se_diff", "wins",
  "recovery_tree1", "se_recovery_tree1", "recovery_overall", "se_recovery_overall",
  "share_structure_equals_truth", "structure",
  "n_excluded", "n_test_points", "share_excluded", "n_excluded_by_seed", "nonfinite_by_method",
  "method_label", "versus_label",
]


def paired_e3d(long: pd.DataFrame) -> pd.DataFrame:
  """E3d's paired differences, structure recovery and excluded points.

  One row per design, d and quantity, in blocks named by ``block``:

  - ``arm_minus_competitor``: each NPCC arm minus VDC, parametric and TLL,
    each competitor selecting its own structure. The arms are (a) a random
    structure and (b) TLL's selected one, both non-simplified, and (c) the
    random structure simplified, an ablation;
  - ``random_minus_selected``: (a) minus (b), per backend;
  - ``simplified_minus_nonsimplified``: (c) minus (a), per backend, on the
    same random structure;
  - ``selected_minus_vdc_true``: (b) minus VDC's pair copulas on the true
    structure;
  - ``recovery``: the tree-1 and overall edge recovery of every arm and
    competitor;
  - ``excluded``: the test points outside the common set, in total and per
    seed (``n_excluded_by_seed``), and the arms whose densities were not
    finite there.

  A difference pairs the seeds both methods have, each scored on its cell's
  common set. ``n_seeds`` counts them, ``se_diff`` is ``sd / sqrt(n_seeds)``,
  and ``wins`` counts the seeds where ``method`` has the lower excess NLL.
  A stage that is still running contributes the seeds it has finished.

  Returns
  -------
  DataFrame
      The rows, in ``E3D_PAIRED_COLUMNS`` order; a column a block does not
      use is empty.
  """
  excess = {key: group.set_index("seed")["excess_nll"]
            for key, group in long.groupby(["design", "d", "method"])}
  label = dict(zip(long["method"], long["method_label"], strict=True))
  rows: list[dict] = []

  def described(method: str) -> dict:
    backend, variant = e3d_npcc(method) or ("", "")
    return {"method": method, "backend": backend, "variant": E3D_LETTERS.get(variant, ""),
            "method_label": label[method]}

  def difference(design: str, d: int, block: str, method: str, versus: str) -> None:
    if (design, d, method) not in excess or (design, d, versus) not in excess:
      return
    both = pd.concat([excess[design, d, method].rename("m"), excess[design, d, versus].rename("v")],
                     axis=1, join="inner").dropna()
    n = len(both)
    if n == 0:
      return
    delta = both["m"] - both["v"]
    rows.append({
      "design": design, "d": d, "block": block, **described(method), "versus": versus,
      "versus_label": label[versus], "n_seeds": n,
      "seeds": ";".join(str(s) for s in sorted(both.index)),
      "mean_method": float(both["m"].mean()), "mean_versus": float(both["v"].mean()),
      "mean_diff": float(delta.mean()),
      "se_diff": float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
      "wins": int((delta < 0).sum()),
    })

  def rank(method: str) -> tuple:
    if method in E3D_COMPETITORS:
      return (0, E3D_COMPETITORS.index(method))
    backend, variant = e3d_npcc(method)
    return (1, backend, E3D_LETTERS[variant])

  for (design, d), cell in long.groupby(["design", "d"]):
    arms = {e3d_npcc(m): m for m in cell["method"].unique() if e3d_npcc(m)}
    backends = sorted({backend for backend, _ in arms})
    for backend in backends:
      for variant in E3D_LETTERS:
        if (backend, variant) in arms:
          for competitor in E3D_COMPETITORS:
            difference(design, d, "arm_minus_competitor", arms[backend, variant], competitor)
    for block, x, y in E3D_CONTRASTS:
      for backend in backends:
        if (backend, x) in arms and (backend, y) in arms:
          difference(design, d, block, arms[backend, x], arms[backend, y])
    for backend in backends:
      if (backend, "tll-ns") in arms:
        difference(design, d, "selected_minus_vdc_true", arms[backend, "tll-ns"], "vdc_true")

    shown = cell[cell["role"].isin(["arm", "ablation"])]
    for method in sorted(shown["method"].unique(), key=rank):
      group = shown[shown["method"] == method]
      n = len(group)
      tree1 = pd.to_numeric(group["recovery_tree1"])
      overall = pd.to_numeric(group["recovery_overall"])
      rows.append({
        "design": design, "d": d, "block": "recovery", **described(method), "n_seeds": n,
        "seeds": ";".join(str(s) for s in sorted(group["seed"].unique())),
        "recovery_tree1": float(tree1.mean()),
        "se_recovery_tree1": float(tree1.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
        "recovery_overall": float(overall.mean()),
        "se_recovery_overall": float(overall.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
        "share_structure_equals_truth": float(group["structure_equals_truth"].astype(str).eq("True").mean()),
        "structure": group["structure"].iloc[0],
      })

    by_seed, n_test, nonfinite = {}, 0, {}
    for seed, group in cell.groupby("seed"):
      counts = set(pd.to_numeric(group["n_excluded"]).astype(int))
      if len(counts) != 1:
        raise ValueError(f"{design} d={d} seed {seed}: every arm must share one common set; "
                         f"got n_excluded {sorted(counts)}")
      by_seed[seed] = counts.pop()
      n_test += int(group["n_test"].iloc[0])
      for method, bad in zip(group["method"], pd.to_numeric(group["n_bad_arm"]), strict=True):
        if bad > 0:
          nonfinite[method] = nonfinite.get(method, 0) + int(bad)
    rows.append({
      "design": design, "d": d, "block": "excluded", "n_seeds": len(by_seed),
      "seeds": ";".join(str(s) for s in by_seed),
      "n_excluded": sum(by_seed.values()), "n_test_points": n_test,
      "share_excluded": sum(by_seed.values()) / n_test,
      "n_excluded_by_seed": ";".join(f"{s}:{k}" for s, k in by_seed.items()),
      "nonfinite_by_method": ";".join(f"{m}:{k}" for m, k in sorted(nonfinite.items())),
    })

  out = pd.DataFrame(rows, columns=E3D_PAIRED_COLUMNS)
  for col in ("n_seeds", "wins", "n_excluded", "n_test_points"):
    out[col] = out[col].astype("Int64")
  out["_d"] = out["design"].map(DESIGN_ORDER)
  return out.sort_values(["_d", "d"], kind="stable").drop(columns="_d")


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


def committed_versions(name: str) -> dict[str, str]:
  """sha256 -> the first commit holding that version of an E3 module."""
  rel = f"scripts/paper/e3/{name}"
  repo = str(common.NPCC_REPO)
  commits = subprocess.run(["git", "-C", repo, "rev-list", "--all", "--reverse", "--", rel],
                           check=True, capture_output=True, text=True).stdout.split()
  versions: dict[str, str] = {}
  for commit in commits:
    blob = subprocess.run(["git", "-C", repo, "show", f"{commit}:{rel}"],
                          capture_output=True, check=False).stdout
    if blob:
      versions.setdefault(hashlib.sha256(blob).hexdigest(), commit[:7])
  return versions


def code_check(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
  """Every recorded code hash against committed code and the file on disk.

  One row per experiment, hash column (the writing process's ``code_sha256``,
  or the VDC server's ``vdc_server_code_sha256``) and module. ``match`` says
  whether the version that ran is committed or is the file on disk, and
  ``version`` names it: the first commit holding it (a later comment-only
  edit, such as a spelling fix, leaves rows made with the earlier commit
  valid and regenerable), or ``working tree`` while not yet committed. A row
  with no recorded hashes at all is reported under ``<none recorded>``.
  """
  rows = []
  on_disk: dict[str, str] = {}
  history: dict[str, dict[str, str]] = {}

  def current(name: str) -> str | None:
    if name not in on_disk:
      path = common.E3_DIR / name
      on_disk[name] = common.sha256_file(path) if path.exists() else None
    return on_disk[name]

  def version(name: str, sha: str) -> str | None:
    if name not in history:
      history[name] = committed_versions(name)
    if sha in history[name]:
      return history[name][sha]
    return "working tree" if sha == current(name) else None

  for experiment, long in frames.items():
    missing_hash = long["code_sha256"].isna() if "code_sha256" in long else pd.Series(True, index=long.index)
    if missing_hash.any():
      rows.append({"experiment": experiment, "source": "code_sha256", "module": "<none recorded>",
                   "recorded": None, "on_disk": None, "match": False,
                   "n_rows": int(missing_hash.sum()),
                   "methods": ";".join(sorted(long.loc[missing_hash, "method"].unique()))})
    for col in ("code_sha256", "vdc_server_code_sha256"):
      if col not in long:
        continue
      for value, group in long.dropna(subset=[col]).groupby(col):
        for name, sha in json.loads(value).items():
          found = version(name, sha)
          rows.append({"experiment": experiment, "source": col, "module": name, "recorded": sha,
                       "on_disk": current(name), "version": found, "match": found is not None,
                       "n_rows": len(group), "methods": ";".join(sorted(group["method"].unique()))})
  runner = json.loads((common.RESULTS / "table1_rerun_cpu" / "results.json").read_text())
  now = common.sha256_file(common.TABLE1_RUNNER)
  rows.append({"experiment": "table1_rerun_cpu", "source": "runner source_sha256",
               "module": common.TABLE1_RUNNER.name, "recorded": runner["source_sha256"],
               "on_disk": now, "version": f"VDC {common.VDC_TAG}" if runner["source_sha256"] == now else None,
               "match": runner["source_sha256"] == now, "n_rows": len(runner["rows"]),
               "methods": "vdc;tll;parametric"})
  return pd.DataFrame(rows)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  p.add_argument("--allow-stale", action="store_true",
                 help="write the summaries even if a row was produced by code that is not on disk")
  args = p.parse_args()
  out = common.RESULTS
  frames: dict[str, pd.DataFrame] = {}
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
    frames["E3a"] = readout_check(a_long, "E3a")
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
    frames["E3b"] = readout_check(b_long, "E3b")
    b_long.to_csv(out / "e3b_long.csv", index=False)
    b_sum = summarize_e3b(b_long)
    b_sum.to_csv(out / "e3b_summary.csv", index=False)
    report["e3b"] = {"summary": b_sum.to_dict("records"), "pending": missing(b_long)}
    with show():
      print(b_sum[["case", "method", "role", "n_seeds", "mean_excess_nll", "se_excess_nll",
                   "mean_excess_minus_vdc", "fit_s_median", "inference_s_median",
                   "device"]].to_string(index=False))
      print("E3b pending:", missing(b_long))

  c_frames = []
  for design, c_dir in E3C_FOLDERS.items():
    if not (c_dir / "vdc_side.csv").exists():
      continue
    named = json.loads((c_dir / "vdc_side.json").read_text()).get("design")
    if isinstance(named, str) and named != design:
      raise ValueError(f"{c_dir} must hold the {design!r} design; its vdc_side.json names {named!r}")
    frame = read_long(c_dir, "E3c", cells="npcc_cells/*/d*.json")
    if "design" in frame and frame["design"].notna().any() and (frame["design"].dropna() != design).any():
      raise ValueError(f"{c_dir} rows must all be design {design!r}")
    frame["design"] = design
    c_frames.append(frame)
  if c_frames:
    c_long = pd.concat(c_frames, ignore_index=True, sort=False)
    c_long = common_set(c_long, "E3c", E3C_FOLDERS)
    frames["E3c"] = readout_check(c_long, "E3c")
    c_long.to_csv(out / "e3c_long.csv", index=False)
    c_sum = summarize_e3c(c_long)
    c_sum.to_csv(out / "e3c_summary.csv", index=False)
    report["e3c"] = {"summary": c_sum.to_dict("records"), "pending": missing(c_long)}
    c_check = e3c_crosscheck(c_long)
    if c_check is not None:
      c_check.to_csv(out / "e3c_crosscheck.csv", index=False)
      report["e3c"]["crosscheck_max_abs_gap"] = {
        f"{design}/{arm}": gap for (design, arm), gap in
        c_check.groupby(["design", "check_arm"])["abs_gap"].max().items()}
    with show():
      print(c_sum[["design", "d", "method", "role", "n_seeds", "mean_test_loglik", "mean_true_loglik",
                   "mean_excess_nll", "se_excess_nll", "mean_excess_minus_vdc",
                   "fit_s_median", "inference_s_median"]].to_string(index=False))
      if c_check is not None:
        print(c_check.groupby(["design", "check_arm"])["abs_gap"].max())

  if (E3D_RESULTS / "vdc_side.csv").exists():
    # VDC's own rows are in vdc_side.csv; cells/vdc/ holds only its densities.
    d_long = common_set(read_long(E3D_RESULTS, "E3d", cells="cells/*/*.json"), "E3d",
                        {"decay": E3D_RESULTS, "strong": E3D_RESULTS})
    frames["E3d"] = readout_check(d_long, "E3d")
    d_long.to_csv(E3D_RESULTS / "e3d_long.csv", index=False)
    d_sum = summarize_e3d(d_long)
    d_sum.to_csv(E3D_RESULTS / "e3d_summary.csv", index=False)
    d_paired = paired_e3d(d_long)
    d_paired.to_csv(E3D_RESULTS / "e3d_paired.csv", index=False)
    report["e3d"] = {"summary": d_sum.to_dict("records"), "paired": d_paired.to_dict("records")}
    with show():
      print(d_sum[["design", "d", "method", "role", "n_seeds", "mean_excess_nll", "se_excess_nll",
                   "mean_recovery_tree1", "mean_recovery_overall", "fit_s_median",
                   "inference_s_median"]].to_string(index=False))
      diffs = d_paired[d_paired["mean_diff"].notna()]
      print(diffs[["design", "d", "block", "method", "versus", "n_seeds", "mean_diff", "se_diff",
                   "wins"]].to_string(index=False))

  check = code_check(frames)
  check.to_csv(out / "e3_code_check.csv", index=False)
  stale = check[~check["match"]]
  report["code_check"] = {
    "all_rows_from_code_on_disk": bool(stale.empty),
    "modules_checked": sorted(set(check["module"])),
    "mismatches": stale.to_dict("records"),
  }
  print("code check:", "every row was produced by committed code or the code on disk" if stale.empty
        else f"{len(stale)} stale or unfingerprinted entries (see e3_code_check.csv)")
  common.write_json(json.loads(pd.Series(report).to_json()), out / "e3_summary.json")
  if not stale.empty and not args.allow_stale:
    sys.exit("e3_collect: rows must come from committed code or the code on disk; rerun the stale arms "
             "or pass --allow-stale")


if __name__ == "__main__":
  main()
