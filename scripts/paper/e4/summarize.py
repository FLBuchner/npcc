"""Summarize E4's coherence CSVs: mean and SE over reps, by estimator and n.

Within a repetition each quantity is averaged over families, regimes and x;
the spread is then taken across repetitions, as in the E1 tables. It reads
every part under the run folder, subfolders included, so estimators run
separately (on another device, say) each keep their own ``--out``. Writes
``directional.csv``, ``hfunc.csv``, ``projection.csv`` and ``mass_tail.csv``
(the share of (cell, x) whose mass on the evaluation grid exceeds 1.1 and 2,
with its median and maximum) into the run folder; from a ``--cost-only``
run ``projection_cost.csv`` with the projected-over-unprojected multiplier;
and from a ``--projection-grid`` run ``projection_grid.csv``, each grid size's
accuracy and margins, with ``projection_grid_change.csv``, their change from
the same fit's unprojected density, paired per (cell, x).

Usage::

    uv run --no-sync python scripts/paper/e4/summarize.py results/e4_coherence
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import readout


def mean_se(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
  per_rep = df.groupby([*by, "rep"])[cols].mean()
  grouped = per_rep.groupby(level=by)
  mean = grouped.mean().add_suffix("_mean")
  se = (grouped.std(ddof=1) / grouped.count().map(math.sqrt)).add_suffix("_se")
  out = pd.concat([mean, se], axis=1)
  out["n_rep"] = grouped.size()
  return out[sorted(out.columns)].reset_index()


#: Accuracy, mass and margin errors of a projected (or unprojected) density.
PROJECTION_COLS = [
  "pdf_IAE",
  "pdf_KL",
  "mass",
  "row_max_abs_err",
  "col_max_abs_err",
  "row_mean_abs_err",
  "col_mean_abs_err",
]


def read_parts(run: Path, parts: list[Path], extra: list[str]) -> pd.DataFrame:
  """Concatenate the parts, refusing a row that two of them both hold.

  A part copied or rerun into a second folder would be averaged in twice.
  """
  df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
  readout.check(df["npcc_commit"], f"the parts under {run}")
  key = ["label", "family", "tau_scenario", "n", "rep", "diagnostic", "x"]
  dup = df.duplicated([*key, *extra]).sum()
  if dup:
    raise SystemExit(f"{dup} rows of {run} appear in more than one part")
  return df


def projection_grid(run: Path, parts: list[Path]) -> None:
  """Each grid size's projected density, and its change from none.

  Parts that record it also give the margin errors on the projection grid
  itself (``proj_*``), and ``projection_grid_cost.csv`` the overhead per
  cell and per distinct x.
  """
  df = read_parts(run, parts, ["projection_grid_size", "sinkhorn_iters"])
  df["sinkhorn_iters"] = df["sinkhorn_iters"].astype(str)
  by = ["label", "n", "projection_grid_size", "sinkhorn_iters"]
  on_grid = [c for c in df.columns if c.startswith("proj_")]
  mean_se(df, by, PROJECTION_COLS + on_grid).to_csv(
    run / "projection_grid.csv", index=False
  )
  # One overhead per cell and grid size, covering every iteration count.
  cell_keys = ["label", "family", "tau_scenario", "n", "rep"]
  cells = df.drop_duplicates([*cell_keys, "projection_grid_size"])
  n_x = df.groupby(cell_keys)["x"].nunique().max()
  cost = (
    cells.groupby(["label", "n", "projection_grid_size"])[
      "projection_overhead_s"
    ]
    .agg(["mean", "median"])
    .add_prefix("overhead_per_cell_s_")
    .reset_index()
  )
  cost["overhead_per_x_s_median"] = cost["overhead_per_cell_s_median"] / n_x
  cost.to_csv(run / "projection_grid_cost.csv", index=False)
  # Every grid size scores one fit, so each projected (cell, x) is compared
  # with that fit's own unprojected density before anything is averaged.
  cell = ["label", "family", "tau_scenario", "n", "rep", "x"]
  cell.append("projection_grid_size")
  none = df[df["sinkhorn_iters"] == "none"][[*cell, *PROJECTION_COLS]]
  change = df[df["sinkhorn_iters"] != "none"].merge(
    none, on=cell, suffixes=("", "_none"), validate="many_to_one"
  )
  for col in PROJECTION_COLS:
    change[col] = change[col] - change[f"{col}_none"]
  mean_se(change, by, PROJECTION_COLS).to_csv(
    run / "projection_grid_change.csv", index=False
  )


def main() -> None:
  run = Path(sys.argv[1])
  parts = sorted(run.rglob("coherence_*.csv"))
  grids = sorted(run.rglob("projection_grid_*.csv"))
  costs = sorted(run.rglob("projection_cost_*.csv"))
  if not (parts or grids or costs):
    raise SystemExit(f"no E4 parts under {run}")
  if grids:
    projection_grid(run, grids)
  if costs:
    cost = pd.concat((pd.read_csv(p) for p in costs), ignore_index=True)
    readout.check(cost["npcc_commit"], f"the cost parts under {run}")
    cost["multiplier"] = cost["time_projected_s"] / cost["time_unprojected_s"]
    cost.to_csv(run / "projection_cost.csv", index=False)
  if parts:
    coherence(run, parts)
  print(f"{len(parts) + len(grids) + len(costs)} part(s) -> {run}")


def coherence(run: Path, parts: list[Path]) -> None:
  """The directional, h-function, projection and mass tables."""
  df = read_parts(run, parts, ["sinkhorn_iters"])
  dirn = df[df["diagnostic"] == "directional"]
  cols = [
    "iae_between_directions",
    "v_given_u_IAE",
    "u_given_v_IAE",
    "average_IAE",
    "v_given_u_KL",
    "u_given_v_KL",
    "average_KL",
  ]
  mean_se(dirn, ["label", "n"], cols).to_csv(
    run / "directional.csv", index=False
  )
  h = df[df["diagnostic"] == "hfunc"]
  mean_se(h, ["label", "n"], ["h1_IAE", "h2_IAE", "h1_max", "h2_max"]).to_csv(
    run / "hfunc.csv", index=False
  )
  proj = df[df["diagnostic"] == "projection"].copy()
  proj["sinkhorn_iters"] = proj["sinkhorn_iters"].astype(str)
  table = mean_se(proj, ["label", "n", "sinkhorn_iters"], PROJECTION_COLS)
  # One overhead per cell (it covers the whole sweep), so average per cell.
  cells = proj.drop_duplicates(["family", "tau_scenario", "n", "rep", "label"])
  overhead = cells.groupby(["label", "n"])["projection_overhead_s"].mean()
  table = table.merge(overhead.reset_index(), on=["label", "n"], how="left")
  table.to_csv(run / "projection.csv", index=False)

  # Each (cell, x) separately: a mean over them hides the rare blow-ups.
  g = proj.groupby(["label", "n", "sinkhorn_iters"])["mass"]
  tail = pd.DataFrame(
    {
      "share_mass_gt_1.1": g.apply(lambda s: float((s > 1.1).mean())),
      "share_mass_gt_2": g.apply(lambda s: float((s > 2.0).mean())),
      "median_mass": g.median(),
      "max_mass": g.max(),
      "n_cell_x": g.size(),
    }
  )
  tail.reset_index().to_csv(run / "mass_tail.csv", index=False)


if __name__ == "__main__":
  main()
