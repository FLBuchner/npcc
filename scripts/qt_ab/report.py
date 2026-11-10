"""Report a read-out A/B: arms against a baseline arm, paired by fitted model.

Every arm is compared with the baseline on the same fitted model, so
differences are paired by (estimator, cell, rep). Metrics are E1's, averaged
over the uv grid at each x and then over x: pdf KL and IAE, hfunc1 / hfunc2
IAE and the tau error; plus the held-out KL from true to estimated density.

- ``--arms``: each listed arm minus the baseline, mean ± SE over the pairs and
  the share of pairs it improves, on the held-out reps 5-9 by estimator and n.
- ``--select``: for each backend, every arm ranked by its mean paired
  difference in ``--rank-by`` (pdf KL by default) on reps 0-4, and the best
  one reported on reps 5-9, so the reported gain is not the largest of many
  noisy comparisons. With ``--pooled``, one arm is also chosen for every
  backend, on the mean over backends of each backend's mean difference.
- ``--relative-selection``: E7a′'s rule, one arm for every backend; see
  :func:`relative_selection`.
- ``--compare-labels BASE OTHER``: two estimators under one read-out arm,
  ``--baseline``; see :func:`relative_labels`.
- ``--tau``: tau-hat at the strongest-dependence x of clayton / linear.

Examples::

    # the tree backends; their E1 read-out was the 7-cell average
    python scripts/qt_ab/report.py --out results/qt_ab/gb \
      --baseline 'm21:ma2|current' --select --arms 'm21:wh0.65|current'

    # E7a', probit: ma0, ma2 and wh0.65, simplest first
    python scripts/qt_ab/report.py --out results/qt_ab_probit/gb \
      results/qt_ab_probit/tabpfn results/qt_ab_probit/icl \
      --baseline 'm21:ma0|current' \
      --arms 'm21:ma2|current' 'm21:wh0.65|current' --relative-selection

    # TabPFN; its E1 read-out was the 3-cell average
    python scripts/qt_ab/report.py --out results/qt_ab/tabpfn \
      --baseline 'm21:ma1|current' --arms 'm21:wh0.65|current' \
      'm19:wh0.65|current'
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["label", "backend", "transform", "family", "tau_scenario", "n", "rep"]
ARM = ["grid", "arm"]
METRICS = [
  "pdf_KL",
  "pdf_IAE",
  "hfunc1_IAE",
  "hfunc2_IAE",
  "tau_err",
  "holdout_kl",
]


def parse_arm(text: str) -> tuple[str, str]:
  """``"m21:wh0.65|current"`` -> ``("m21", "wh0.65|current")``."""
  grid, arm = text.split(":", 1)
  return grid, arm


def per_rep(d: pd.DataFrame) -> pd.DataFrame:
  """One row per (estimator, cell, rep, arm), every metric averaged over x."""
  met = d[d.kind == "metric"]
  wide = (
    met.groupby([*KEYS, *ARM, "quantity"])[["IAE", "KL"]]
    .mean()
    .unstack("quantity")
  )
  wide.columns = [f"{q}_{m}" for m, q in wide.columns]
  wide = wide[["pdf_KL", "pdf_IAE", "hfunc1_IAE", "hfunc2_IAE"]]
  diag = d[d.kind == "diag"]
  wide["tau_err"] = diag.groupby([*KEYS, *ARM]).tau_abs_err.mean()
  strongest = diag.tau_true >= (
    diag.groupby([*KEYS, *ARM]).tau_true.transform("max") - 1e-9
  )
  wide["tau_hat_max"] = diag[strongest].groupby([*KEYS, *ARM]).tau_hat.mean()
  wide["holdout_kl"] = (
    d[d.kind == "holdout"].groupby([*KEYS, *ARM]).holdout_kl.mean()
  )
  return wide.reset_index()


def paired(w: pd.DataFrame, baseline: tuple[str, str]) -> pd.DataFrame:
  """Each arm minus the baseline arm of the same estimator, cell and rep."""
  base = w[(w.grid == baseline[0]) & (w.arm == baseline[1])].set_index(KEYS)
  diff = w.set_index(KEYS)[[*ARM, *METRICS]].copy()
  diff[METRICS] = diff[METRICS] - base[METRICS].reindex(diff.index)
  return diff.reset_index()


def summary(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
  """``mean ± SE (share of pairs improved)`` for every metric."""
  g = frame.groupby(by)[METRICS]
  mean, se = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
  wins = g.agg(lambda c: float((c < 0).mean()))
  return (
    mean.round(4).astype(str)
    + " ± "
    + se.round(4).astype(str)
    + " ("
    + (100 * wins).round(0).astype(int).astype(str)
    + "%)"
  )


#: How far apart two scores must be, in standard errors, to count as apart.
TIE_SE = 2.0
#: The relative worsening against a backend's own best arm that is flagged.
FLAG_REL = 0.10


def pooled_mean_se(pairs: pd.DataFrame, value: str) -> tuple[float, float]:
  """The equal-weight mean over rows of each row's mean over its pairs.

  A row is a ``(label, n)``; its pairs are its cells x reps. The rows are
  independent samples, so the SE of their mean is
  ``sqrt(sum_r s_r**2 / n_r) / R``, each row's variance taken over its pairs.
  """
  g = pairs.groupby(["label", "n"])[value]
  means, var, count = g.mean(), g.var(ddof=1), g.count()
  return float(means.mean()), float(np.sqrt((var / count).sum()) / len(means))


def relative_selection(
  w: pd.DataFrame, baseline: tuple[str, str], arms: list[tuple[str, str]]
) -> None:
  """E7a′'s rule: one arm for every backend, by relative held-out KL.

  Fixed before any probit rep 0-4 was read (experiments.md, E8):

  - **Score.** For each pair (estimator, cell, rep) on reps 0-4, probit only,
    the arm's relative change in held-out KL against ``baseline``,
    ``(KL_arm - KL_base) / KL_base``; averaged over the pairs of each
    ``(label, n)`` row, then over the rows with equal weight. The SE is
    :func:`pooled_mean_se`'s.
  - **Pick.** The arm with the lowest score.
  - **Tie.** The candidates in simplicity order are ``baseline`` and then
    ``arms`` as given. The first whose paired difference from the best is
    within :data:`TIE_SE` of its SE is taken. IAE does not enter.
  - **Flags.** Each backend for which the pick's held-out KL exceeds that of
    the backend's own best arm by more than :data:`TIE_SE` SE *and* by more
    than :data:`FLAG_REL` relative is flagged: the case a per-backend
    exception would have to make.

  Every row's numbers are printed, and the pick's again on reps 5-9.
  """
  probit = w[w["transform"] == "probit"]
  if probit.empty:
    raise SystemExit("the relative selection needs probit rows; got none")
  candidates = [baseline, *arms]
  kl = probit.set_index([*KEYS, *ARM]).holdout_kl.unstack(ARM)
  missing = [c for c in candidates if c not in kl.columns]
  if missing:
    raise SystemExit(f"every candidate must have rows; got none for {missing}")
  names = [f"{g}:{a}" for g, a in candidates]
  kl = kl[candidates].dropna()
  kl.columns = names
  rel = kl.sub(kl[names[0]], axis=0).div(kl[names[0]], axis=0).reset_index()
  kl = kl.reset_index()
  select, held = rel[rel.rep <= 4], rel[rel.rep >= 5]

  print("\n== relative change in held-out KL against the baseline, ==")
  print("== mean over pairs per row (SE), probit, reps 0-4 ==")
  g = select.groupby(["label", "n"])[names]
  table = (
    g.mean().round(4).astype(str)
    + " ("
    + (g.std(ddof=1) / np.sqrt(g.count())).round(4).astype(str)
    + ")"
  )
  print(table.to_string())

  scores = {c: pooled_mean_se(select, c) for c in names}
  print("\n== pooled score: mean over rows (SE) ==")
  for c, (m, se) in scores.items():
    print(f"  {c:>22}  {m:+.4f}  ({se:.4f})")
  best = min(names, key=lambda c: scores[c][0])
  pick = best
  for c in names[: names.index(best)]:
    gap = select.assign(d=select[c] - select[best])
    m, se = pooled_mean_se(gap, "d")
    print(f"  {c} minus {best}: {m:+.4f} ({se:.4f})")
    if m <= TIE_SE * se:
      pick = c
      break
  print(f"\n-- lowest score {best}; picked {pick} --")

  print("\n== flags: the pick against each backend's own best, reps 0-4 ==")
  flagged = []
  kl_sel = kl[kl.rep <= 4]
  for backend, rows in select.groupby("backend"):
    own = min(names, key=lambda c: pooled_mean_se(rows, c)[0])
    k = kl_sel[kl_sel.backend == backend]
    worse = k.assign(d=(k[pick] - k[own]) / k[own])
    m, se = pooled_mean_se(worse, "d")
    flag = own != pick and m > TIE_SE * se and m > FLAG_REL
    flagged += [backend] if flag else []
    print(
      f"  {backend:>17}: own best {own}; the pick is {m:+.4f} ({se:.4f}) "
      f"relative{'  FLAG' if flag else ''}"
    )
  print(f"  flagged: {flagged or 'none'}")

  print(f"\n== {pick} against the baseline, probit, reps 5-9 ==")
  g = held.groupby(["label", "n"])[pick]
  print(
    pd.DataFrame(
      {
        "mean": g.mean(),
        "se": g.std(ddof=1) / np.sqrt(g.count()),
        "pairs": g.count(),
      }
    )
    .round(4)
    .to_string()
  )


def ratio_of_means(
  pairs: pd.DataFrame, base: str, other: str
) -> tuple[pd.DataFrame, float, float]:
  """``other`` against ``base`` per ``n``, as a ratio of means, and pooled.

  ``pairs`` holds one row per pair (cell, rep), with ``n`` and a column per
  label. For each ``n``, the score is the mean over its pairs of
  ``other - base`` divided by the mean of ``base``, and its SE the SE of that
  mean difference over the same mean ``base``. The pooled score is the
  equal-weight mean of the per-n scores; the ``n`` are independent samples,
  so its SE is ``sqrt(sum_n se_n**2) / N``.
  """
  rows = {}
  for n, part in pairs.groupby("n"):
    d = part[other] - part[base]
    scale = part[base].mean()
    rows[n] = {
      "score": d.mean() / scale,
      "se": d.std(ddof=1) / np.sqrt(len(d)) / scale,
      "mean_diff": d.mean(),
      "mean_base": scale,
      "pairs": len(d),
    }
  table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("n")
  pooled = float(table.score.mean())
  se = float(np.sqrt((table.se**2).sum()) / len(table))
  return table, pooled, se


def relative_labels(
  w: pd.DataFrame, base: str, other: str, arm: tuple[str, str]
) -> None:
  """Two estimators under one read-out: adopt ``other`` unless clearly worse.

  The rule, fixed before reps 0-4 are read (npcc-paper-27, E8): under the
  read-out ``arm``, on reps 0-4, :func:`ratio_of_means` of the held-out KL,
  ``other`` against ``base``. A ratio of means rather than a mean of per-pair
  ratios, so that a pair whose ``base`` KL is near zero cannot dominate the
  score. Adopt ``other`` unless the pooled score is above ``+TIE_SE`` SE.

  The same comparison is printed for pdf IAE, and both again on reps 5-9;
  neither decides.
  """
  at = w[(w.grid == arm[0]) & (w.arm == arm[1])]
  cell = ["family", "tau_scenario", "n", "rep"]
  for metric in ("holdout_kl", "pdf_IAE"):
    wide = at[at.label.isin([base, other])].pivot_table(
      index=cell, columns="label", values=metric
    )
    missing = [lab for lab in (base, other) if lab not in wide.columns]
    if missing:
      raise SystemExit(
        f"both labels must have rows under {arm}; got none for {missing}"
      )
    wide = wide.dropna().reset_index()
    for name, part in (
      ("reps 0-4", wide[wide.rep <= 4]),
      ("reps 5-9", wide[wide.rep >= 5]),
    ):
      table, m, se = ratio_of_means(part, base, other)
      print(
        f"\n== {metric}: {other} against {base} under {arm[0]}:{arm[1]}, "
        f"{name}, ratio of means =="
      )
      print(table.round(4).to_string())
      print(f"  pooled over n: {m:+.4f} ({se:.4f})")
      if metric == "holdout_kl" and name == "reps 0-4":
        keep = m > TIE_SE * se
        print(f"-- {'keep ' + base if keep else 'adopt ' + other} --")


def main() -> None:
  """Print the report."""
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--out", type=Path, nargs="+", required=True)
  parser.add_argument("--baseline", type=parse_arm, required=True)
  parser.add_argument("--arms", type=parse_arm, nargs="*", default=[])
  parser.add_argument("--select", action="store_true")
  parser.add_argument(
    "--rank-by", choices=METRICS, default="pdf_KL", help="--select's metric"
  )
  parser.add_argument(
    "--pooled",
    action="store_true",
    help="--select one arm for every backend, each backend weighted equally",
  )
  parser.add_argument(
    "--relative-selection",
    action="store_true",
    help="E7a′'s rule: --baseline, then --arms, simplest first",
  )
  parser.add_argument(
    "--compare-labels",
    nargs=2,
    metavar=("BASE", "OTHER"),
    default=None,
    help="two estimators under the --baseline arm; adopt OTHER unless worse",
  )
  parser.add_argument("--tau", action="store_true")
  args = parser.parse_args()
  pd.set_option("display.width", 250, "display.max_rows", 500)
  pd.set_option("display.max_columns", 30)

  frames = [
    pd.read_parquet(p)
    for out in args.out
    for p in sorted(out.glob("*.parquet"))
  ]
  w = per_rep(pd.concat(frames, ignore_index=True))
  diff = paired(w, args.baseline)
  held_out = diff[diff.rep >= 5]
  base = w[(w.grid == args.baseline[0]) & (w.arm == args.baseline[1])]

  print(f"== Baseline {args.baseline}: absolute metrics, all reps ==")
  print(base.groupby(["label", "n"])[METRICS].mean().round(4).to_string())

  for grid, arm in args.arms:
    r = held_out[(held_out.grid == grid) & (held_out.arm == arm)]
    print(f"\n== {grid} {arm} minus baseline, held-out reps 5-9 ==")
    print(summary(r, ["label", "n"]).to_string())

  if args.select:
    ranked = diff[diff.rep <= 4].groupby(["backend", *ARM])[METRICS].mean()
    for backend, frame in ranked.groupby(level="backend"):
      print(f"\n== {backend}: every arm minus baseline, reps 0-4 ==")
      print(frame.droplevel("backend").sort_values(args.rank_by).round(4))
      grid, arm = frame[args.rank_by].idxmin()[1:]
      r = held_out[
        (held_out.backend == backend)
        & (held_out.grid == grid)
        & (held_out.arm == arm)
      ]
      print(f"\n-- selected {grid} {arm}, reported on reps 5-9 --")
      print(summary(r, ["transform", "n"]).to_string())
    if args.pooled:
      pooled = ranked.groupby(level=ARM).mean().sort_values(args.rank_by)
      print("\n== every arm minus baseline, mean over backends, reps 0-4 ==")
      print(pooled.round(4))
      grid, arm = pooled[args.rank_by].idxmin()
      r = held_out[(held_out.grid == grid) & (held_out.arm == arm)]
      print(f"\n-- selected {grid} {arm} for every backend, reps 5-9 --")
      print(summary(r, ["label", "n"]).to_string())

  if args.relative_selection:
    relative_selection(w, args.baseline, args.arms)

  if args.compare_labels:
    relative_labels(w, *args.compare_labels, args.baseline)

  if args.tau:
    a = w[(w.family == "clayton") & (w.tau_scenario == "linear")]
    table = a.groupby(["label", "n", *ARM]).tau_hat_max.mean().unstack(ARM)
    shown = [args.baseline, *args.arms]
    print("\n== tau-hat at the strongest-dependence x, clayton / linear ==")
    print(table[[c for c in shown if c in table.columns]].round(3).to_string())


if __name__ == "__main__":
  main()
