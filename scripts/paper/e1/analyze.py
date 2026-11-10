"""Analyze the E1 bivariate backend study: paper tables, checks and figures.

Reads the per-cell checkpoints of a ``npcc-simstudy`` run (``<run>/cells/*``,
finished cells only, so a partial run can be previewed) and writes into
``<run>/analysis/``:

- ``overall.csv`` -- pdf IAE and KL, h1 and h2 IAE by estimator. Each entry is
  the mean and standard error over repetitions of the per-repetition average
  over families, tau regimes, n and x (IAE already integrates the grid).

Every KL here is the Monte Carlo KL (E8), the mean of ``log c - log c_hat``
over draws from the true copula at each x, which a run records as ``KL_mc``;
IAE stays on the grid. The grid KL is kept as ``KL_grid`` and not reported,
and a run without ``KL_mc`` is refused rather than reported on the grid.
- ``by_n.csv`` -- the same by estimator and n.
- ``ranking.csv`` -- each estimator's rank by pdf KL within every family x
  regime panel next to its overall rank, and ``ranking_agreement.csv`` with
  Kendall's tau between each panel's ranking and the overall one.
- ``margins.csv`` -- the margin checks (gap between each integrated margin of
  the fitted density and 1, mean and max over grid rows and columns) by
  estimator and n.
- ``accuracy_by_n.pdf`` -- pdf IAE and KL against n for the six backends at the
  main-text TabPFN version and transform, with +-2 standard errors.
- ``conditional_tau.pdf`` -- fitted tau(x) against the target at n = 1000 for
  TabPFN, the best other in-context learner and the best gradient booster;
  ``conditional_tau_full.pdf`` shows all six backends.
- ``accuracy_by_n_full.pdf`` -- every estimator, for the supplement.

Usage (from the npcc repository root)::

    uv run --no-sync python scripts/paper/e1/analyze.py \
      results/study_2026-11-05 --tabpfn v3 --transform probit
"""

from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from collections.abc import Sequence

mpl.use("Agg")

#: Display name, group and plot style of each backend. The colors are the
#: Okabe-Ito palette the existing paper figures use; in-context learners are
#: solid lines and gradient boosters dashed, so identity never rests on color.
BACKENDS: dict[str, dict[str, str]] = {
  "tabpfn": {
    "name": "TabPFN",
    "group": "ICL",
    "color": "#0072B2",
    "marker": "o",
  },
  "tabicl": {
    "name": "TabICLv2",
    "group": "ICL",
    "color": "#56B4E9",
    "marker": "s",
  },
  "nori": {"name": "Nori", "group": "ICL", "color": "#CC79A7", "marker": "^"},
  "lightgbm": {
    "name": "LightGBM",
    "group": "GB",
    "color": "#E69F00",
    "marker": "D",
  },
  "xgboost": {
    "name": "XGBoost",
    "group": "GB",
    "color": "#009E73",
    "marker": "v",
  },
  "catboost": {
    "name": "CatBoost",
    "group": "GB",
    "color": "#D55E00",
    "marker": "P",
  },
  # The established competitors (E1b), run as R scripts on the same cells:
  # grays and dotted, so the Okabe-Ito colors stay with the six backends.
  "gamcopula": {
    "name": "gamCopula",
    "group": "R",
    "color": "#333333",
    "marker": "X",
  },
  "boostcopula-smooth-cv": {
    "name": "boostCopula",
    "group": "R",
    "color": "#777777",
    "marker": "*",
  },
  "boostcopula-smooth": {
    "name": "boostCopula (AIC stop)",
    "group": "R",
    "color": "#999999",
    "marker": "p",
  },
  "boostcopula": {
    "name": "boostCopula (linear in x)",
    "group": "R",
    "color": "#AAAAAA",
    "marker": "h",
  },
}
LINESTYLE = {"ICL": "-", "GB": "--", "R": ":"}

#: The R competitors the figures show; the tables keep every R arm run.
FIGURE_COMPETITORS = ("gamcopula", "boostcopula-smooth-cv")

#: Panel titles of the shape families (E1c), whose tau(x) is implied by the
#: family itself rather than set by a regime.
FAMILY_NAMES = {
  "clayton-gumbel-mix": "Clayton-Gumbel mixture",
  "student-varying-nu": r"Student $t$, varying $\nu$",
}

_LABEL = re.compile(
  r"^(?:(?P<tabpfn>tabpfn)-quantiles-(?P<version>v[\d.]+)"
  r"(?:-temp(?P<temp>[\d.]+))?|(?P<other>[a-z]+))"
  r"-(?P<transform>logit|probit|identity)$"
)
#: An R competitor's label is its key alone: its script fits on the copula
#: scale, so it has no transform to name.
_R_LABELS = {k for k, v in BACKENDS.items() if v["group"] == "R"}

QUANTITIES = (
  ("pdf", "IAE"),
  ("pdf", "KL"),
  ("hfunc1", "IAE"),
  ("hfunc2", "IAE"),
)


#: The per-cell tables a run checkpoints; each row carries its ``label``.
TABLES = ("metrics", "diagnostics", "quantities", "runtime")


def load(
  runs: list[Path], replacing: Sequence[Path] = ()
) -> dict[str, pd.DataFrame]:
  """Concatenate one table over every finished cell of one or more runs.

  Runs on the same grid share cell keys and data seeds, so their rows join on
  the cell: E1b's R competitors fit exactly E1's data. A run in
  ``replacing`` refits some of the estimators of ``runs`` -- ``npcc-simstudy
  --labels`` after a code change -- and its rows take the place of theirs. It
  must cover the same cells, or the replaced estimators would be scored on
  fewer cells than the rest.
  """
  tables = _finished(runs)
  for run in replacing:
    new = _finished([run])
    cells = set(tables["cells"]["cell"])
    if set(new["cells"]["cell"]) != cells:
      odd = sorted(cells.symmetric_difference(new["cells"]["cell"]))
      raise SystemExit(
        f"{run} must cover the cells of the runs it replaces; got "
        f"{len(odd)} in only one, e.g. {odd[0]}"
      )
    labels = set(new["metrics"]["label"])
    for name in TABLES:
      old = tables[name]
      if not old.empty:
        old = old[~old["label"].isin(labels)]
      parts = [df for df in (old, new[name]) if not df.empty]
      tables[name] = (
        pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
      )
  same_draws(tables["metrics"])
  return tables


def _finished(runs: list[Path]) -> dict[str, pd.DataFrame]:
  """Every table over the finished cells of ``runs``, rows concatenated."""
  done = sorted(
    Path(p).parent
    for run in runs
    for p in glob.glob(str(run / "cells" / "*" / "DONE"))
  )
  if not done:
    raise SystemExit(f"{runs} have no finished cells yet")
  tables = {}
  for name in TABLES:
    # A run with no surface families writes no quantities table.
    files = [d / f"{name}.parquet" for d in done]
    # A metric row names its run, which is cells/<cell>/ two levels down.
    parts = [
      pd.read_parquet(f).assign(run=str(f.parents[2]))
      if name == "metrics"
      else pd.read_parquet(f)
      for f in files
      if f.exists()
    ]
    tables[name] = (
      annotate(pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame()
    )
  tables["metrics"] = monte_carlo_kl(tables["metrics"])
  tables["cells"] = pd.DataFrame({"cell": sorted({d.name for d in done})})
  return tables


def monte_carlo_kl(metrics: pd.DataFrame) -> pd.DataFrame:
  """``KL`` as the Monte Carlo KL, the grid's kept as ``KL_grid``.

  Every pdf row must carry a finite ``KL_mc``. A run from before it has no
  ``KL_mc_n``, and its grid KL must not be reported under the name; a row
  that was scored but whose fitted density was not finite at some draw has
  a NaN ``KL_mc``, which no mean may skip. Either refusal names the labels.
  """
  pdf = metrics["quantity"] == "pdf"
  if "KL_mc_n" not in metrics:
    metrics = metrics.assign(KL_mc=np.nan, KL_mc_n=np.nan, KL_mc_nonfinite=0.0)
  unscored = pdf & metrics["KL_mc_n"].isna()
  if unscored.any():
    labels = sorted(metrics.loc[unscored, "label"].unique())
    raise SystemExit(
      "every pdf row must carry the Monte Carlo KL (KL_mc); the runs of "
      f"{labels} predate it, so rerun them with this npcc"
    )
  nan = pdf & metrics["KL_mc"].isna()
  if nan.any():
    bad = metrics[nan]
    raise SystemExit(
      f"KL_mc must be finite; got NaN on {len(bad)} pdf rows of "
      f"{sorted(bad['label'].unique())}, whose fitted density is not finite "
      f"at {int(bad['KL_mc_nonfinite'].sum())} draws in all"
    )
  return metrics.rename(columns={"KL": "KL_grid", "KL_mc": "KL"})


def same_draws(metrics: pd.DataFrame) -> None:
  """Refuse runs scored on different numbers of draws per x.

  The draws are a function of the cell's seed and of ``holdout_n``, so two
  runs with different ``holdout_n`` score their estimators on different
  draws, and pairing them would compare more than the estimators.
  """
  pdf = metrics[metrics["quantity"] == "pdf"]
  per_run = pdf.groupby("run")["KL_mc_n"].unique()
  if pdf["KL_mc_n"].nunique() > 1:
    shown = {run: sorted(int(n) for n in ns) for run, ns in per_run.items()}
    raise SystemExit(
      f"every run must be scored on the same holdout_n; got {shown}"
    )


def annotate(df: pd.DataFrame) -> pd.DataFrame:
  """Split each label into backend, TabPFN version and transform."""
  parsed = df["label"].str.extract(_LABEL)
  r_method = df["label"].isin(_R_LABELS)
  if (parsed["transform"].isna() & ~r_method).any():
    bad = sorted(
      df.loc[parsed["transform"].isna() & ~r_method, "label"].unique()
    )
    raise SystemExit(f"labels the analysis cannot parse: {bad}")
  df = df.copy()
  df["key"] = np.where(
    r_method,
    df["label"],
    np.where(parsed["tabpfn"].notna(), "tabpfn", parsed["other"]),
  )
  # A TabPFN temperature other than E1's 1.0 is part of the version's name,
  # so the ablation's fits stay apart from E1's.
  df["version"] = parsed["version"].fillna("") + np.where(
    parsed["temp"].notna(), " (T=" + parsed["temp"].fillna("") + ")", ""
  )
  df["transform_"] = parsed["transform"].fillna("identity")
  df["backend_name"] = df["key"].map(lambda k: BACKENDS[k]["name"])
  df["group"] = df["key"].map(lambda k: BACKENDS[k]["group"])
  df["display"] = np.where(
    df["version"] != "",
    df["backend_name"] + " " + df["version"],
    df["backend_name"],
  )
  return df


def _mean_se(values: pd.Series) -> pd.Series:
  n = values.notna().sum()
  sd = values.std(ddof=1) if n > 1 else math.nan
  return pd.Series({"mean": values.mean(), "se": sd / math.sqrt(n), "n_rep": n})


def summarize(metrics: pd.DataFrame, by: list[str]) -> pd.DataFrame:
  """Mean and SE over repetitions of the per-repetition average.

  Within a repetition the metric is averaged over everything not in ``by``
  (families, regimes, n, x); the spread is then taken across repetitions.
  """
  keys = ["label", "display", "key", "version", "transform_", "group"]
  rows = []
  for quantity, metric in QUANTITIES:
    part = metrics[metrics["quantity"] == quantity]
    per_rep = part.groupby([*keys, *by, "rep"])[metric].mean()
    stats = per_rep.groupby(level=[*keys, *by]).apply(_mean_se).unstack()
    stats = stats.reset_index()
    stats["quantity"], stats["metric"] = quantity, metric
    rows.append(stats)
  return pd.concat(rows, ignore_index=True)


def ranking(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Rank estimators by pdf KL within each family x regime and overall."""
  kl = metrics[metrics["quantity"] == "pdf"]
  panel = kl.groupby(["family", "tau_scenario", "label"])["KL"].mean()
  ranks = (
    panel.groupby(level=["family", "tau_scenario"])
    .rank()
    .unstack(["family", "tau_scenario"])
  )
  ranks.columns = [f"rank_{f}_{s}" for f, s in ranks.columns]
  overall = kl.groupby("label")["KL"].mean()
  ranks.insert(0, "rank_overall", overall.rank())
  ranks.insert(0, "pdf_KL_overall", overall)
  ranks = ranks.sort_values("rank_overall")

  agreement = []
  for col in [
    c for c in ranks.columns if c.startswith("rank_") and c != "rank_overall"
  ]:
    tau = _kendall_tau(ranks["rank_overall"].to_numpy(), ranks[col].to_numpy())
    agreement.append(
      {"panel": col.removeprefix("rank_"), "kendall_tau_vs_overall": tau}
    )
  return ranks.reset_index(), pd.DataFrame(agreement)


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
  """Kendall's tau-a between two rankings (no ties expected)."""
  n = len(a)
  concordant = 0
  for i in range(n):
    for j in range(i + 1, n):
      concordant += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
  return float(concordant / (n * (n - 1) / 2)) if n > 1 else math.nan


def margins(diagnostics: pd.DataFrame) -> pd.DataFrame:
  """Margin checks by estimator and n: mean of each gap, and the worst max."""
  cols = [
    "row_mean_abs_err",
    "row_max_abs_err",
    "col_mean_abs_err",
    "col_max_abs_err",
  ]
  grouped = diagnostics.groupby(["label", "display", "transform_", "n"])
  out = grouped[cols].mean().add_prefix("mean_")
  out["worst_row_max_abs_err"] = grouped["row_max_abs_err"].max()
  out["worst_col_max_abs_err"] = grouped["col_max_abs_err"].max()
  return out.reset_index()


def main_text_labels(
  overall: pd.DataFrame, tabpfn: str, transform: str
) -> list[str]:
  """The six backends at the main-text TabPFN version and transform."""
  pick = overall[
    (overall["transform_"] == transform) & (overall["quantity"] == "pdf")
  ]
  pick = pick[(pick["key"] != "tabpfn") | (pick["version"] == tabpfn)]
  order = list(BACKENDS)
  labels = sorted(
    pick["label"].unique(),
    key=lambda lab: order.index(pick.loc[pick["label"] == lab, "key"].iloc[0]),
  )
  return labels


def best_of(
  overall: pd.DataFrame, labels: list[str], group: str, exclude: str
) -> str | None:
  """The label with the lowest overall pdf KL in a group, excluding a backend."""
  kl = overall[(overall["quantity"] == "pdf") & (overall["metric"] == "KL")]
  kl = kl[
    kl["label"].isin(labels) & (kl["group"] == group) & (kl["key"] != exclude)
  ]
  return None if kl.empty else str(kl.sort_values("mean").iloc[0]["label"])


def _style() -> None:
  mpl.rcParams.update(
    {
      "font.family": "serif",
      "mathtext.fontset": "dejavuserif",
      "font.size": 9,
      "axes.spines.top": False,
      "axes.spines.right": False,
      "axes.grid": True,
      "axes.grid.axis": "y",
      "grid.color": "#dddddd",
      "grid.linewidth": 0.6,
      "lines.linewidth": 1.5,
      "lines.markersize": 4.5,
      "legend.frameon": False,
      "pdf.fonttype": 42,
    }
  )


def _plot_style(meta: pd.Series) -> dict[str, object]:
  spec = BACKENDS[str(meta["key"])]
  return {
    "color": spec["color"],
    "marker": spec["marker"],
    "linestyle": LINESTYLE[spec["group"]],
    "label": str(meta["display"]),
  }


def _log_ticks(ax: plt.Axes, steps: tuple[int, ...] = (1, 2, 5)) -> None:
  """Label a log y axis at ``steps`` times each power of ten in its range."""
  lo, hi = ax.get_ylim()
  ticks = [
    m * 10.0**e
    for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
    for m in steps
    if lo <= m * 10.0**e <= hi
  ]
  ax.set_yticks(ticks, [f"{v:g}" for v in ticks])
  ax.minorticks_off()


def save(fig: plt.Figure, path: Path) -> None:
  """Write the PDF and, next to it, a 150-dpi PNG for review."""
  fig.savefig(path)
  fig.savefig(path.with_suffix(".png"), dpi=150)
  plt.close(fig)


def fig_accuracy_by_n(
  by_n: pd.DataFrame, labels: list[str], path: Path
) -> None:
  """pdf IAE and KL against n, one line per estimator, +-2 SE."""
  _style()
  fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)
  # Each estimator sits a few percent off its n, so the error bars of
  # neighbors do not overlap; on a log axis a factor is the same shift at
  # every n.
  dodge = dict(zip(labels, np.linspace(-0.03, 0.03, len(labels)), strict=True))
  for ax, metric in zip(axes, ("IAE", "KL"), strict=True):
    part = by_n[(by_n["quantity"] == "pdf") & (by_n["metric"] == metric)]
    for label in labels:
      rows = part[part["label"] == label].sort_values("n")
      if rows.empty:
        continue
      style = _plot_style(rows.iloc[0])
      ax.errorbar(
        rows["n"] * (1.0 + dodge[label]),
        rows["mean"],
        yerr=2 * rows["se"],
        capsize=2,
        elinewidth=0.8,
        **style,
      )
    ax.set_xscale("log")
    ns = sorted(part["n"].unique())
    ax.set_xticks(ns, [f"{n}" for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("$n$")
    ax.set_ylabel(f"pdf {metric}")
    # Both axes on log scales, so a slope reads as a convergence rate.
    ax.set_yscale("log")
    if metric == "IAE":
      # Always span the labeled 0.1-0.5, so the range between labels reads.
      lo, hi = ax.get_ylim()
      ax.set_ylim(min(lo, 0.095), max(hi, 0.52))
    _log_ticks(ax, (1, 2, 3, 5) if metric == "IAE" else (1, 2, 5))
  handles, names = axes[0].get_legend_handles_labels()
  fig.legend(
    handles,
    names,
    loc="outside upper center",
    ncol=len(names) if len(names) <= 6 else math.ceil(len(names) / 2),
  )
  save(fig, path)


def fig_conditional_tau(
  diagnostics: pd.DataFrame, labels: list[str], path: Path, *, n: int = 1000
) -> None:
  """tau_hat(x) averaged over repetitions against the target, per panel."""
  _style()
  part = diagnostics[
    (diagnostics["n"] == n) & diagnostics["label"].isin(labels)
  ]
  # One row per family and one column per regime, as the study swept them.
  families = sorted(diagnostics["family"].unique())
  scenarios = sorted(diagnostics["tau_scenario"].unique())
  fig, axes = plt.subplots(
    len(families),
    len(scenarios),
    figsize=(3.5 * len(scenarios), 2.5 * len(families)),
    sharex=True,
    sharey=True,
    constrained_layout=True,
    squeeze=False,
  )
  panels = [(f, s) for f in families for s in scenarios]
  for ax, (family, scenario) in zip(axes.ravel(), panels, strict=True):
    sub = part[(part["family"] == family) & (part["tau_scenario"] == scenario)]
    name = FAMILY_NAMES.get(family, family.capitalize())
    title = name if scenario == "implied" else f"{name}, {scenario}"
    ax.set_title(title, fontsize=9)
    if sub.empty:
      continue
    truth = sub.groupby("x")["tau_true"].first()
    ax.plot(
      truth.index,
      truth.to_numpy(),
      color="black",
      linewidth=2.2,
      marker="o",
      label="Truth",
    )
    for label in labels:
      rows = sub[sub["label"] == label]
      if rows.empty:
        continue
      by_x = rows.groupby("x")["tau_hat"]
      mean = by_x.mean()
      se = by_x.std(ddof=1) / np.sqrt(by_x.count())
      style = _plot_style(rows.iloc[0])
      ax.fill_between(
        mean.index,
        (mean - 2 * se).to_numpy(),
        (mean + 2 * se).to_numpy(),
        color=style["color"],
        alpha=0.15,
        linewidth=0,
      )
      ax.plot(mean.index, mean.to_numpy(), **style)
    # The target stays within [0.05, 0.70].
    ax.set_ylim(0.0, 0.8)
  for ax in axes[-1]:
    ax.set_xlabel("$x$")
  for ax in axes[:, 0]:
    ax.set_ylabel(r"Kendall's $\tau$")
  handles, names = axes[0, 0].get_legend_handles_labels()
  if not handles:
    plt.close(fig)
    print(f"no estimator has n = {n} yet; skipped {path.name}")
    return
  fig.legend(
    handles,
    names,
    loc="outside upper center",
    ncol=len(names) if len(names) <= 7 else math.ceil(len(names) / 2),
  )
  save(fig, path)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("run", type=Path, help="The study's output folder.")
  p.add_argument(
    "--with",
    dest="extra",
    type=Path,
    nargs="*",
    default=[],
    help="Further runs on the same grid, e.g. E1b's R competitors.",
  )
  p.add_argument(
    "--replacing",
    type=Path,
    nargs="*",
    default=[],
    help=(
      "Runs that refit some estimators of the others (npcc-simstudy "
      "--labels); their rows replace those estimators' rows. Pass --out "
      "too, so the base run's own analysis is not overwritten."
    ),
  )
  p.add_argument("--tabpfn", default="v3", help="Main-text TabPFN version.")
  p.add_argument("--transform", default="probit", help="Main-text transform.")
  p.add_argument(
    "--out", type=Path, default=None, help="Default: <run>/analysis."
  )
  p.add_argument(
    "--tau-n", type=int, default=1000, help="n of the tau(x) figures."
  )
  a = p.parse_args()
  out = a.out or a.run / "analysis"
  out.mkdir(parents=True, exist_ok=True)

  t = load([a.run, *a.extra], replacing=a.replacing)
  overall = summarize(t["metrics"], [])
  by_n = summarize(t["metrics"], ["n"])
  ranks, agreement = ranking(t["metrics"])
  overall.to_csv(out / "overall.csv", index=False)
  by_n.to_csv(out / "by_n.csv", index=False)
  ranks.to_csv(out / "ranking.csv", index=False)
  agreement.to_csv(out / "ranking_agreement.csv", index=False)
  margins(t["diagnostics"]).to_csv(out / "margins.csv", index=False)

  main_labels = main_text_labels(overall, a.tabpfn, a.transform)
  fig_accuracy_by_n(by_n, main_labels, out / "accuracy_by_n.pdf")
  fig_accuracy_by_n(
    by_n, sorted(by_n["label"].unique()), out / "accuracy_by_n_full.pdf"
  )
  tabpfn_label = next(lab for lab in main_labels if lab.startswith("tabpfn"))
  subset = [tabpfn_label]
  for group in ("ICL", "GB"):
    best = best_of(overall, main_labels, group, exclude="tabpfn")
    if best is not None:
      subset.append(best)
  fig_conditional_tau(
    t["diagnostics"], subset, out / "conditional_tau.pdf", n=a.tau_n
  )
  fig_conditional_tau(
    t["diagnostics"], main_labels, out / "conditional_tau_full.pdf", n=a.tau_n
  )
  competitors = [k for k in FIGURE_COMPETITORS if k in set(by_n["label"])]
  if competitors:
    fig_accuracy_by_n(
      by_n,
      main_labels + competitors,
      out / "accuracy_by_n_with_competitors.pdf",
    )

  print(f"{len(t['cells'])} cells -> {out}")
  print("main-text estimators:", main_labels)
  print("tau subset:", subset)


if __name__ == "__main__":
  main()
