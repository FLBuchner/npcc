"""Analyze the E2 vine study: the paper's table, figure and in-text numbers.

Reads a finished ``npcc-vinestudy`` run -- its aggregated
``metrics_by_x.parquet``, one row per (d, n, repetition, arm, evaluation x),
and ``truths.json`` -- and writes into ``<run>/analysis/``:

- ``vine_table.csv`` -- one row per (d, n), in the columns of the paper's
  ``tab:vine-results``: the joint and copula KL of oracle, random and kde-tll
  as mean and SE, the random - oracle joint KL paired by repetition, and
  ``oracle_wins``, the repetitions in which the oracle's joint KL is lower.
- ``vine_kl_by_x.pdf`` -- joint KL at each evaluation x by d, n and arm, as
  mean +-2 SE on a log scale; a PNG beside it, and the plotted values in
  ``vine_kl_by_x.csv``.
- ``vine_text_numbers.json`` -- the numbers the text quotes: kde-tll's joint
  KL over each NPCC arm's; the share of repetitions in which both NPCC arms
  beat kde-tll; joint KL at x = 0.01, 0.99 and 0.5 for d = 3, n = 500; the
  change from the smallest to the largest n; the joint - copula KL gap; joint
  ISE at two (d, n); and how often the random structure is the true one,
  with the random - oracle difference split on it.

Each KL or ISE is a fixed-x divergence. An x-averaged one is its mean over
the evaluation x within a repetition -- nothing is integrated over x -- and
every mean and standard error is then taken over repetitions. Every arm and
sample size in a repetition faces the same truth and test points, so
differences between arms, or between sample sizes, are paired by repetition.
Timings are left to E6, which runs on an otherwise idle machine.

Usage (from the npcc repository root)::

    uv run --no-sync python scripts/paper/e2/analyze.py results/vine_2026-11-06
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvinecopulib as pv

# This file shares E1's file name, so E1's folder goes first on the path:
# behind this script's own folder, ``import analyze`` would find this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e1"))

from analyze import _log_ticks, _mean_se, _style, save

#: Display name and plot style of each arm, in the paper's order. The colors
#: are the Okabe-Ito palette of the E1 figures; the covariate-blind baseline
#: is gray and dashed, and each arm has its own marker, so identity never
#: rests on color.
ARMS: dict[str, dict[str, str]] = {
  "oracle": {
    "name": "oracle",
    "color": "#0072B2",
    "marker": "o",
    "linestyle": "-",
  },
  "random": {
    "name": "random",
    "color": "#E69F00",
    "marker": "s",
    "linestyle": "-",
  },
  "tll": {
    "name": "kde-tll",
    "color": "#999999",
    "marker": "^",
    "linestyle": "--",
  },
}
NPCC_ARMS = ("oracle", "random")
#: The fixed-x metrics read: joint KL and ISE, and the copula KL.
METRICS = ("KL", "KL_copula", "ISE")
#: Each arm's offset from x_k in the figure, in units of x, so that the error
#: bars of the two NPCC arms, which lie close together, do not overlap.
DODGE = {"oracle": -0.015, "random": 0.015, "tll": 0.0}

#: Where the text quotes joint KL at single x: the (d, n), the x and the arms.
KL_AT_X_CELL = (3, 500)
KL_AT_X = (0.01, 0.99, 0.5)
KL_AT_X_ARMS = ("tll", "oracle")
#: The (d, n) at which the text quotes joint ISE.
ISE_CELLS = ((3, 500), (5, 1000))


def load(run: Path) -> pd.DataFrame:
  """The run's fixed-x metrics, refused unless the grid is complete.

  Every (d, n, arm, rep, x) must appear once, so that each mean over
  repetitions and each x-average compares like with like, and arms and
  sample sizes pair by repetition.
  """
  path = run / "metrics_by_x.parquet"
  if not path.is_file():
    raise SystemExit(f"{path} must exist; run the aggregation pass first")
  df = pd.read_parquet(path)
  unknown = sorted(set(df["arm"]) - set(ARMS))
  if unknown:
    raise SystemExit(f"arms must be drawn from {list(ARMS)}; got {unknown}")
  labels = df.groupby("arm")["label"].unique()
  if (labels.map(len) > 1).any():
    raise SystemExit(
      f"each arm must hold one estimator; got {labels.map(list).to_dict()}"
    )
  key = ["d", "n", "arm", "rep", "x"]
  full = pd.MultiIndex.from_product(
    [sorted(df[k].unique()) for k in key], names=key
  )
  index = pd.MultiIndex.from_frame(df[key])
  if index.has_duplicates or len(full.difference(index)):
    raise SystemExit(
      f"{path} must hold every (d, n, arm, rep, x) once; it has {len(df)} "
      f"rows for {len(full)} combinations"
    )
  return df


def x_averaged(metrics: pd.DataFrame) -> pd.DataFrame:
  """Each metric's mean over the evaluation x, per (d, n, arm, rep)."""
  grouped = metrics.groupby(["d", "n", "arm", "rep"], as_index=False)
  return grouped[list(METRICS)].mean()


def by_arm(xavg: pd.DataFrame, metric: str) -> pd.DataFrame:
  """One column per arm and one row per (d, n, rep), the unit arms pair on."""
  return xavg.pivot(index=["d", "n", "rep"], columns="arm", values=metric)


def kl_by_x(metrics: pd.DataFrame) -> pd.DataFrame:
  """Joint KL at each evaluation x: mean and SE over repetitions."""
  stats = (
    metrics.groupby(["d", "n", "arm", "x"])["KL"]
    .apply(_mean_se)
    .unstack()
    .reset_index()
  )
  stats["n_rep"] = stats["n_rep"].astype(int)
  stats.insert(3, "arm_name", stats["arm"].map(lambda a: ARMS[a]["name"]))
  return stats


def check_against_runner(
  run: Path, xavg: pd.DataFrame, by_x: pd.DataFrame
) -> None:
  """Refuse numbers that disagree with the aggregation pass's summaries.

  ``summary_over_x`` and ``summary_by_x`` hold the runner's mean and standard
  deviation over repetitions. Recomputing them here from the rows catches a
  stale table, or an average taken over the wrong axis.
  """
  keys = ["d", "n", "arm"]
  summary = pd.read_parquet(run / "summary_over_x.parquet")
  for metric in METRICS:
    ours = xavg.groupby(keys)[metric].agg(["mean", "std"])
    what = f"x-averaged {metric}"
    _agree(summary[summary["metric"] == metric], ours, keys, what)
  summary = pd.read_parquet(run / "summary_by_x.parquet")
  per_x = by_x.set_index([*keys, "x"])
  ours = pd.DataFrame(
    {"mean": per_x["mean"], "std": per_x["se"] * np.sqrt(per_x["n_rep"])}
  )
  _agree(summary[summary["metric"] == "KL"], ours, [*keys, "x"], "KL by x")


def _agree(
  theirs: pd.DataFrame, ours: pd.DataFrame, keys: list[str], what: str
) -> None:
  """Exit unless ``ours`` holds the runner's mean and SD, row for row."""
  theirs = theirs.set_index(keys)[["rep_mean", "rep_std"]].sort_index()
  ours = ours.sort_index()
  same = theirs.index.equals(ours.index) and np.allclose(
    theirs.to_numpy(), ours.to_numpy(), rtol=1e-9, atol=0.0
  )
  if not same:
    raise SystemExit(
      f"{what} must match the run's summary tables; rerun the aggregation"
    )


def _column(arm: str) -> str:
  """An arm's display name as a column-name token: kde-tll -> kde_tll."""
  return ARMS[arm]["name"].replace("-", "_")


def results_table(xavg: pd.DataFrame) -> pd.DataFrame:
  """The paper's table: x-averaged KL by arm, one row per (d, n).

  ``random_minus_oracle_joint_KL`` is the mean and SE of the per-repetition
  difference, and ``oracle_wins`` counts the repetitions (of ``n_rep``) in
  which the oracle's joint KL is strictly lower.
  """
  wide = xavg.pivot(
    index=["d", "n", "rep"], columns="arm", values=["KL", "KL_copula"]
  )
  rows = []
  for (d, n), cell in wide.groupby(level=["d", "n"]):
    row: dict[str, float | int] = {"d": int(d), "n": int(n)}
    for metric, prefix in (("KL", "joint_KL"), ("KL_copula", "copula_KL")):
      for arm in ARMS:
        s = _mean_se(cell[(metric, arm)])
        row[f"{prefix}_{_column(arm)}_mean"] = float(s["mean"])
        row[f"{prefix}_{_column(arm)}_se"] = float(s["se"])
    diff = cell[("KL", "random")] - cell[("KL", "oracle")]
    s = _mean_se(diff)
    row["random_minus_oracle_joint_KL_mean"] = float(s["mean"])
    row["random_minus_oracle_joint_KL_se"] = float(s["se"])
    row["oracle_wins"] = int((diff > 0).sum())
    row["n_rep"] = int(diff.size)
    rows.append(row)
  return pd.DataFrame(rows)


def fig_kl_by_x(by_x: pd.DataFrame, path: Path) -> None:
  """Joint KL against x, one panel per (d, n), +-2 SE over repetitions.

  Rows are d and columns n, and each row shares its log y axis, so the two
  sample sizes of one dimension read against each other.
  """
  _style()
  dims, ns = sorted(by_x["d"].unique()), sorted(by_x["n"].unique())
  fig, axes = plt.subplots(
    len(dims),
    len(ns),
    figsize=(6.3, 4.4),
    sharex=True,
    sharey="row",
    constrained_layout=True,
    squeeze=False,
  )
  for (i, d), (j, n) in itertools.product(enumerate(dims), enumerate(ns)):
    ax = axes[i, j]
    cell = by_x[(by_x["d"] == d) & (by_x["n"] == n)]
    for arm, spec in ARMS.items():
      rows = cell[cell["arm"] == arm].sort_values("x")
      ax.errorbar(
        rows["x"] + DODGE[arm],
        rows["mean"],
        yerr=2 * rows["se"],
        capsize=2,
        elinewidth=0.8,
        color=spec["color"],
        marker=spec["marker"],
        linestyle=spec["linestyle"],
        label=spec["name"],
      )
    ax.set_title(f"$d = {d}$, $n = {n}$", fontsize=9)
    ax.set_yscale("log")
  for row in axes:
    # After both panels of the row are drawn, so the ticks span both.
    _log_ticks(row[0])
    row[0].set_ylabel("joint KL")
  xs = sorted(by_x["x"].unique())
  for ax in axes[-1]:
    ax.set_xticks(xs, [f"{x:g}" for x in xs])
    ax.set_xlabel("covariate $x$")
  handles, names = axes[0, 0].get_legend_handles_labels()
  fig.legend(handles, names, loc="outside upper center", ncol=len(names))
  save(fig, path)


def _stat(values: pd.Series) -> dict[str, float | int]:
  """Mean and standard error over repetitions, as plain JSON numbers."""
  s = _mean_se(values)
  return {
    "mean": float(s["mean"]),
    "se": float(s["se"]),
    "n_rep": int(s["n_rep"]),
  }


def baseline_ratio(xavg: pd.DataFrame) -> dict[str, object]:
  """kde-tll's x-averaged joint KL over each NPCC arm's, per (d, n).

  The ratio of the means over repetitions -- the table's entries -- with its
  range over the (d, n).
  """
  means = xavg.groupby(["d", "n", "arm"])["KL"].mean().unstack("arm")
  out: dict[str, object] = {}
  for arm in NPCC_ARMS:
    ratio = means["tll"] / means[arm]
    out[arm] = {
      "by_cell": [
        {"d": int(d), "n": int(n), "ratio": float(r)}
        for (d, n), r in ratio.items()
      ],
      "min": float(ratio.min()),
      "max": float(ratio.max()),
    }
  return out


def npcc_beat_baseline(xavg: pd.DataFrame) -> dict[str, float | int]:
  """The (d, n, rep) where both NPCC arms' joint KL is below kde-tll's."""
  kl = by_arm(xavg, "KL")
  both = (kl["oracle"] < kl["tll"]) & (kl["random"] < kl["tll"])
  return {
    "count": int(both.sum()),
    "of": int(both.size),
    "share": float(both.mean()),
  }


def kl_at_x(metrics: pd.DataFrame) -> list[dict[str, object]]:
  """Joint KL at the x the text quotes, refused unless the grid holds them.

  The grid is a ``linspace``, so a quoted x matches a grid value only up to
  rounding; each record carries the grid's own value.
  """
  d, n = KL_AT_X_CELL
  grid = np.sort(metrics["x"].unique())
  cell = metrics[(metrics["d"] == d) & (metrics["n"] == n)]
  rows: list[dict[str, object]] = []
  for target in KL_AT_X:
    hit = grid[np.isclose(grid, target, rtol=0.0, atol=1e-9)]
    if hit.size != 1:
      raise SystemExit(
        f"the evaluation grid must hold x = {target}; got {grid.tolist()}"
      )
    for arm in KL_AT_X_ARMS:
      values = cell.loc[(cell["arm"] == arm) & (cell["x"] == hit[0]), "KL"]
      rows.append(
        {
          "d": d,
          "n": n,
          "arm": ARMS[arm]["name"],
          "x": float(hit[0]),
          **_stat(values),
        }
      )
  return rows


def n_effect(xavg: pd.DataFrame) -> list[dict[str, object]]:
  """The change in x-averaged joint KL from the smallest to the largest n.

  ``reduction_pct`` compares the means over repetitions. ``reps_lower``
  counts the repetitions in which the larger sample has the lower KL, a
  pairing the design allows: a repetition's truth and test points do not
  depend on n.
  """
  kl = xavg.pivot(index=["arm", "d", "rep"], columns="n", values="KL")
  small, large = min(kl.columns), max(kl.columns)
  rows: list[dict[str, object]] = []
  for arm in ARMS:
    for d, g in kl.loc[arm].groupby(level="d"):
      m_small, m_large = g[small].mean(), g[large].mean()
      rows.append(
        {
          "arm": ARMS[arm]["name"],
          "d": int(d),
          "n_from": int(small),
          "n_to": int(large),
          "mean_from": float(m_small),
          "mean_to": float(m_large),
          "reduction_pct": float(100.0 * (1.0 - m_large / m_small)),
          "reps_lower": int((g[large] < g[small]).sum()),
          "n_rep": len(g),
        }
      )
  return rows


def copula_gap(xavg: pd.DataFrame) -> dict[str, object]:
  """Joint minus copula KL per arm and (d, n), paired within a repetition.

  The copula KL is scored at the true probability integral transforms, so
  the gap is only a rough indication of the margins' share.
  """
  gap = xavg.assign(gap=xavg["KL"] - xavg["KL_copula"])
  rows: list[dict[str, object]] = []
  ranges: dict[str, list[float]] = {}
  for arm in ARMS:
    stats = [
      {"arm": ARMS[arm]["name"], "d": int(d), "n": int(n), **_stat(g["gap"])}
      for (d, n), g in gap[gap["arm"] == arm].groupby(["d", "n"])
    ]
    means = [float(s["mean"]) for s in stats]
    ranges[ARMS[arm]["name"]] = [min(means), max(means)]
    rows += stats
  return {"by_cell": rows, "range_of_means": ranges}


def joint_ise(xavg: pd.DataFrame) -> list[dict[str, object]]:
  """x-averaged joint ISE of every arm at the (d, n) the text quotes."""
  rows: list[dict[str, object]] = []
  for d, n in ISE_CELLS:
    cell = xavg[(xavg["d"] == d) & (xavg["n"] == n)]
    for arm in ARMS:
      values = cell.loc[cell["arm"] == arm, "ISE"]
      rows.append({"d": d, "n": n, "arm": ARMS[arm]["name"], **_stat(values)})
  return rows


def _edges(matrix: list[list[int]]) -> set[tuple[frozenset[int], ...]]:
  """A vine's edges as (conditioned pair, conditioning set), every tree."""
  array = np.asfortranarray(np.asarray(matrix, dtype=np.uint64))
  trees = pv.RVineStructure.from_matrix(array).get_trees()
  return {
    (frozenset((a, b)), frozenset(conditioning))
    for edges in trees
    for a, b, conditioning in edges
  }


def random_is_true(run: Path) -> pd.DataFrame:
  """Per (d, rep): whether the random arm's structure is the true one.

  Two R-vine matrices can encode one vine, so the structures are compared as
  edge sets, the way pyvinecopulib reads each matrix.
  """
  truths = json.loads((run / "truths.json").read_text())
  return pd.DataFrame(
    [
      {
        "d": t["d"],
        "rep": t["rep"],
        "random_is_true": _edges(t["matrix"]) == _edges(t["random_matrix"]),
      }
      for t in truths
    ]
  )


def structure_split(
  xavg: pd.DataFrame, matches: pd.DataFrame
) -> list[dict[str, object]]:
  """random - oracle joint KL, split by whether the random structure is true.

  Where it is, the two arms fit one structure and differ in the random arm's
  non-simplified context, so the split separates the cost of a wrong
  structure from the cost of that context.
  """
  kl = by_arm(xavg, "KL").reset_index()
  kl = kl.merge(matches, on=["d", "rep"], validate="many_to_one")
  kl["diff"] = kl["random"] - kl["oracle"]
  return [
    {
      "d": int(d),
      "n": int(n),
      "random_is_true": bool(same),
      "oracle_wins": int((g["diff"] > 0).sum()),
      **_stat(g["diff"]),
    }
    for (d, n, same), g in kl.groupby(["d", "n", "random_is_true"])
  ]


def text_numbers(
  run: Path, metrics: pd.DataFrame, xavg: pd.DataFrame
) -> dict[str, object]:
  """Every number the text quotes, keyed by what it measures."""
  matches = random_is_true(run)
  same = matches.groupby("d")["random_is_true"]
  labels = metrics.groupby("arm")["label"].first()
  return {
    "run": str(run),
    "estimators": {ARMS[a]["name"]: str(labels[a]) for a in ARMS},
    "x_grid": [float(x) for x in np.sort(metrics["x"].unique())],
    "kde_tll_over_npcc_joint_KL": baseline_ratio(xavg),
    "both_npcc_beat_kde_tll": npcc_beat_baseline(xavg),
    "joint_KL_at_x": kl_at_x(metrics),
    "joint_KL_change_with_n": n_effect(xavg),
    "joint_minus_copula_KL": copula_gap(xavg),
    "joint_ISE": joint_ise(xavg),
    "random_structure_is_true": [
      {"d": int(d), "count": int(s.sum()), "of": int(s.size)} for d, s in same
    ],
    "random_minus_oracle_joint_KL_by_structure": structure_split(xavg, matches),
  }


def _entry(row: pd.Series, prefix: str) -> str:
  """One table entry as the paper sets it: ``mean \\se{SE}``."""
  return f"{row[prefix + '_mean']:.3f} \\se{{{row[prefix + '_se']:.3f}}}"


def latex_rows(table: pd.DataFrame) -> str:
  """The table's body in the layout of ``tab:vine-results``."""
  pad = " " * 13
  lines = []
  for _, row in table.iterrows():
    joint = " & ".join(_entry(row, f"joint_KL_{_column(a)}") for a in ARMS)
    copula = " & ".join(_entry(row, f"copula_KL_{_column(a)}") for a in ARMS)
    diff = _entry(row, "random_minus_oracle_joint_KL")
    wins = f"{int(row['oracle_wins'])}/{int(row['n_rep'])}"
    lines += [
      f"    {int(row['d'])} & {int(row['n']):<4} & {joint}",
      f"{pad}& {copula}",
      f"{pad}& {diff} & {wins} \\\\",
    ]
  return "\n".join(lines)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("run", type=Path, help="The vine study's output folder.")
  p.add_argument(
    "--out", type=Path, default=None, help="Default: <run>/analysis."
  )
  a = p.parse_args()
  out = a.out or a.run / "analysis"
  out.mkdir(parents=True, exist_ok=True)

  metrics = load(a.run)
  xavg = x_averaged(metrics)
  by_x = kl_by_x(metrics)
  check_against_runner(a.run, xavg, by_x)

  table = results_table(xavg)
  table.to_csv(out / "vine_table.csv", index=False)
  by_x.to_csv(out / "vine_kl_by_x.csv", index=False)
  fig_kl_by_x(by_x, out / "vine_kl_by_x.pdf")
  numbers = text_numbers(a.run, metrics, xavg)
  (out / "vine_text_numbers.json").write_text(
    json.dumps(numbers, indent=2) + "\n"
  )

  cells = metrics.groupby(["d", "n", "rep"]).ngroups
  print(f"{cells} cells -> {out}")
  print("estimators:", numbers["estimators"])
  print(latex_rows(table))


if __name__ == "__main__":
  main()
