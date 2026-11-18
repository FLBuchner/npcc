"""Consolidate E6's timings into one long table and its summary.

Reads, from the E6 folder, whichever of these exist:

- ``bivariate.csv`` and ``bivariate_gb1.csv``: ``timing.py bivariate``, at
  the boosters' default thread count and at one thread;
- ``r_gamcopula/`` and ``r_boostcopula_cv/``: the R competitors' study runs,
  whose ``runtime.parquet`` holds each script's own step timers;
- ``vines.csv``: ``timing.py vine``;
- ``projection_cost/**/projection_cost_*.csv``: ``coherence.py --cost-only``,
  each row already a median over its repeats;
- ``e3_timing.csv``: the unconditional comparison's own timing script.

Writes ``timings.csv``, one row per timed call -- section, method, setting,
n, d, k, grid, stage, seconds, rep, device, compute, cpu, threads,
npcc_commit -- and ``timings_summary.csv``, the median seconds and
``n_reps`` per (section, method, setting, n, d, k, grid, stage).

Usage::

    uv run --no-sync python scripts/paper/e6/consolidate.py results/e6_timing
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

COLUMNS = [
  "section",
  "method",
  "setting",
  "n",
  "d",
  "k",
  "grid",
  "stage",
  "seconds",
  "rep",
  "device",
  "compute",
  "cpu",
  "threads",
  "npcc_commit",
]
KEYS = ["section", "method", "setting", "n", "d", "k", "grid", "stage"]


def _long(
  df: pd.DataFrame, stages: dict[str, str], fixed: dict[str, object]
) -> pd.DataFrame:
  """One row per (input row, stage): ``stages`` maps a column to its stage."""
  parts = []
  for column, stage in stages.items():
    part = df.drop(columns=list(stages)).assign(stage=stage, seconds=df[column])
    parts.append(part.assign(**fixed))
  return pd.concat(parts, ignore_index=True)


def bivariate(run: Path) -> list[pd.DataFrame]:
  stages = {
    "fit_v_given_u_s": "fit V|U",
    "fit_u_given_v_s": "fit U|V",
    "pdf_v_given_u_s": "density V|U",
    "pdf_u_given_v_s": "density U|V",
    "hfunc1_s": "h-function 1",
    "hfunc2_s": "h-function 2",
    "pdf_pair_s": "pair density",
  }
  out = []
  for path in sorted(run.glob("bivariate*.csv")):
    df = pd.read_csv(path)
    if "backend_threads" not in df:
      df["backend_threads"] = ""
    df = df.assign(
      method=df["label"],
      setting=[
        "" if pd.isna(t) or t == "" else f"threads={t}"
        for t in df["backend_threads"]
      ],
      d=2,
      threads=df["torch_threads"],
    )
    keep = [*stages, "method", "setting", "n", "d", "rep", "device"]
    keep += ["compute", "cpu", "threads", "npcc_commit"]
    out.append(_long(df[keep], stages, {"section": "bivariate"}))
  return out


def _commit(run: Path) -> str:
  """The commit the first process of a study run recorded."""
  lines = (run / "provenance.jsonl").read_text().splitlines()
  return str(json.loads(lines[0])["npcc_commit"])[:7]


def r_competitors(run: Path, cpu: str) -> list[pd.DataFrame]:
  stages = {
    "fit_time": "fit",
    "pdf_time": "pair density",
    "h1_time": "h-function 1",
    "h2_time": "h-function 2",
  }
  out = []
  for folder in sorted(run.glob("r_*")):
    df = pd.read_parquet(folder / "runtime.parquet")
    df = df.assign(
      method=df["label"],
      setting="threads=1",
      d=2,
      device="cpu",
      compute="cpu",
      cpu=cpu,
      threads=1,
      npcc_commit=_commit(folder),
    )
    keep = [*stages, "method", "setting", "n", "d", "rep", "device"]
    keep += ["compute", "cpu", "threads", "npcc_commit"]
    out.append(_long(df[keep], stages, {"section": "bivariate"}))
  return out


def vines(run: Path) -> list[pd.DataFrame]:
  path = run / "vines.csv"
  if not path.exists():
    return []
  df = pd.read_csv(path)
  df = df.assign(
    method=df["label"],
    setting=df["arm"],
    compute=["gpu" if a != "tll" else "cpu" for a in df["arm"]],
    device="",
    threads=df["torch_threads"],
  )
  stages = {"fit_time": "fit", "pdf_time": "evaluation"}
  keep = [*stages, "method", "setting", "n", "d", "rep", "device"]
  keep += ["compute", "cpu", "threads", "npcc_commit"]
  return [_long(df[keep], stages, {"section": "vine"})]


def projection(run: Path, cpu: str) -> list[pd.DataFrame]:
  paths = sorted((run / "projection_cost").rglob("projection_cost_*.csv"))
  out = []
  for path in paths:
    df = pd.read_csv(path)
    df = df.assign(
      method=df["label"],
      setting=[f"sinkhorn={s}" for s in df["sinkhorn_iters"]],
      d=2,
      k=df["k_distinct_x"],
      grid=df["projection_grid_size"],
      compute=[
        "gpu" if str(v).startswith("cuda") else "cpu" for v in df["device"]
      ],
      cpu=cpu,
      threads="",
      rep="median",
    )
    stages = {
      "time_unprojected_s": "unprojected",
      "time_projected_s": "projected",
      "overhead_per_x_s": "overhead per distinct x",
    }
    keep = [*stages, "method", "setting", "n", "d", "k", "grid", "rep"]
    keep += ["device", "compute", "cpu", "threads", "npcc_commit"]
    out.append(_long(df[keep], stages, {"section": "projection"}))
  return out


def unconditional(run: Path) -> list[pd.DataFrame]:
  path = run / "e3_timing.csv"
  if not path.exists():
    return []
  df = pd.read_csv(path).rename(columns={"seed": "rep"})
  return [df]


def cpu_model() -> str:
  for line in Path("/proc/cpuinfo").read_text().splitlines():
    if line.startswith("model name"):
      return line.split(":", 1)[1].strip()
  return ""


def main() -> None:
  run = Path(sys.argv[1])
  cpu = cpu_model()
  parts = [
    *bivariate(run),
    *r_competitors(run, cpu),
    *vines(run),
    *projection(run, cpu),
    *unconditional(run),
  ]
  if not parts:
    raise SystemExit(f"{run} must hold E6 outputs; found none")
  table = pd.concat(parts, ignore_index=True).reindex(columns=COLUMNS)
  table[KEYS] = table[KEYS].fillna("")
  group = table.groupby(KEYS, sort=False)["seconds"]
  table["n_reps"] = group.transform("size")
  table.to_csv(run / "timings.csv", index=False)
  summary = group.agg(seconds_median="median", n_reps="size").reset_index()
  summary.to_csv(run / "timings_summary.csv", index=False)
  print(f"{len(table)} timed calls, {len(summary)} groups -> {run}")


if __name__ == "__main__":
  main()
