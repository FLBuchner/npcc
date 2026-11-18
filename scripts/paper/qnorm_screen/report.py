"""The qnorm screen's verdict: TabPFN v3 handed qnorm(u) against u.

Reads the screen's two runs, written by ``run.sh`` beside this file:

- ``--e1``: ``npcc-simstudy`` on ``configs/qnorm_screen.toml``, E1's cells at
  reps 5-9, the reps E7a held out;
- ``--e3a``: ``scripts/paper/e3/e3a_npcc_fit.py --out`` on E3a's nine
  copulas, five seeds each, at ``n_train`` 2000 and 10000.

Both hold the same two estimators: ``BASE``, E1's TabPFN v3 at the probit
transform, and ``ARM``, the same handed the conditioning copula score as
``qnorm(u) = Phi^{-1}(u)``. They are paired on identical data.

**The metric** is each study's pdf KL: E1's Monte Carlo KL averaged over the
10 x values of a cell, and E3a's KL at its 20,000 test points. A pair is an
E1 cell (family, regime, n, rep) or an E3a fit (copula, seed, n), and its
difference is ``d = KL(ARM) - KL(BASE)``, negative where qnorm fits better.

**The rule**, fixed before any result of the screen was read:

- a *stratum* is one study at one n: E1 at n = 200, 500, 1000 and 2000, E3a at
  2000 and 10000;
- a group's *score* is the mean of ``d`` over its pairs in each stratum,
  averaged over its strata with equal weight; its SE is ``sqrt(sum se_s**2) /
  S``, ``se_s`` the SE of the stratum's mean over its pairs;
- qnorm is **adopted** only if the score over every pair is below ``-2`` SE (a
  gain beyond 2 SE overall), and no n and no family has a score above ``+2``
  SE (no loss beyond 2 SE). The n are 200, 500, 1000, 2000 (E1 and E3a) and
  10000; the families are E1's two and E3a's seven, a family of both studies
  counting once;
- otherwise u is kept. pdf IAE is reported the same way and does not decide.

The differences are absolute, not relative to the baseline's KL: E3a's
independence copula has a KL near zero, below its own Monte Carlo error, so
a ratio to it would be noise.

Writes ``pairs.csv`` (every pair's KL and IAE for both arms) and
``groups.csv`` (every group's score, SE and z) into ``--out``, and prints the
tables and the verdict. A partial run is refused unless ``--allow-partial``,
which prints the tables marked as a preview and no verdict.

Usage (from the npcc repository root)::

    python scripts/paper/qnorm_screen/report.py \
      --e1 results/qnorm_screen_<TAG>/e1 --e3a results/qnorm_screen_<TAG>/e3a \
      --out results/qnorm_screen_<TAG>/analysis
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# E1's analysis: its loader refuses a run without the Monte Carlo KL, a
# non-finite one, and runs scored on different draws.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e1"))
from analyze import load

BASE = "tabpfn-quantiles-v3-probit"
ARM = "tabpfn-quantiles-v3-qnorm-probit"
#: How far a score must sit from zero, in SE, to count as a gain or a loss.
TIE_SE = 2.0
METRICS = ("KL", "IAE")
#: The design: E1's reps held out in E7a, and E3a's sizes and fits per size.
E1_REPS = frozenset(range(5, 10))
E1_PAIRS = 80
E3A_N = (2000, 10000)
E3A_FITS_PER_N = 45


def _paired(long: pd.DataFrame, keys: list[str], what: str) -> pd.DataFrame:
  """One row per pair, ``BASE`` and ``ARM`` side by side.

  Refuses a pair that only one arm has: the screen compares the two on
  identical data, and a missing fit is a failure to report, not to drop.
  """
  labels = set(long["label"])
  missing = [lab for lab in (BASE, ARM) if lab not in labels]
  if missing:
    raise SystemExit(f"{what} must hold both arms; got no rows for {missing}")
  long = long[long["label"].isin([BASE, ARM])]
  if long.duplicated([*keys, "label"]).any():
    raise SystemExit(f"{what} must hold one row per pair and arm; got repeats")
  wide = long.pivot(index=keys, columns="label", values=list(METRICS))
  lone = wide.isna().any(axis=1)
  if lone.any():
    raise SystemExit(
      f"every pair of {what} must have both arms; got {int(lone.sum())} with "
      f"one, e.g. {wide.index[lone][0]}"
    )
  out = pd.DataFrame(index=wide.index)
  for metric in METRICS:
    out[f"base_{metric}"] = wide[(metric, BASE)]
    out[f"arm_{metric}"] = wide[(metric, ARM)]
  return out.reset_index()


def e1_pairs(run: Path) -> pd.DataFrame:
  """E1's pairs: each cell's pdf KL and IAE, averaged over its x values.

  Parameters
  ----------
  run : Path
      The ``npcc-simstudy`` run on ``configs/qnorm_screen.toml``.

  Returns
  -------
  pandas.DataFrame
      One row per cell, with ``study``, ``n``, ``family`` and ``unit``.
  """
  metrics = load([run])["metrics"]
  pdf = metrics[metrics["quantity"] == "pdf"]
  cell = ["family", "tau_scenario", "n", "rep"]
  per = pdf.groupby(["label", *cell])[list(METRICS)].mean().reset_index()
  reps = sorted(set(per["rep"]) - E1_REPS)
  if reps:
    raise SystemExit(
      f"E1's pairs must be reps 5-9, the reps E7a held out; got reps {reps}"
    )
  pairs = _paired(per, cell, f"E1 run {run}")
  pairs["unit"] = pairs["tau_scenario"] + "/rep" + pairs["rep"].astype(str)
  pairs["study"] = "E1"
  return pairs.drop(columns=["tau_scenario", "rep"])


def e3a_pairs(folder: Path) -> pd.DataFrame:
  """E3a's pairs: each fit's KL at the test points and IAE on the grid.

  Parameters
  ----------
  folder : Path
      The ``--out`` of ``e3a_npcc_fit.py``, holding
      ``npcc_cells/<label>/n<n_train>/k*.json``.

  Returns
  -------
  pandas.DataFrame
      One row per fit, with ``study``, ``n``, ``family`` and ``unit``.
  """
  rows = [
    json.loads(path.read_text())
    for label in (BASE, ARM)
    for path in sorted((folder / "npcc_cells" / label).glob("n*/k*.json"))
  ]
  if not rows:
    raise SystemExit(f"E3a's folder {folder} must hold fits; got none")
  long = pd.DataFrame(rows).rename(
    columns={"estimator": "label", "kl": "KL", "iae": "IAE", "n_train": "n"}
  )
  bad = long[~np.isfinite(long["KL"].astype(float))]
  if not bad.empty:
    raise SystemExit(
      f"E3a's KL must be finite; got {len(bad)} fits without one, e.g. "
      f"{bad.iloc[0]['label']} {bad.iloc[0]['case']} s{bad.iloc[0]['seed']}"
    )
  long["unit"] = long["case"] + "/s" + long["seed"].astype(str)
  pairs = _paired(long, ["n", "family", "unit"], f"E3a folder {folder}")
  pairs["study"] = "E3a"
  return pairs


def check_design(pairs: pd.DataFrame) -> list[str]:
  """What the pairs lack of the full design, as one line per gap.

  Parameters
  ----------
  pairs : pandas.DataFrame
      Both studies' pairs.

  Returns
  -------
  list of str
      Empty when E1 has its 80 cells and E3a its 45 fits at each size.
  """
  gaps = []
  e1 = int((pairs["study"] == "E1").sum())
  if e1 != E1_PAIRS:
    gaps.append(f"E1 has {e1} pairs of {E1_PAIRS}")
  e3a = pairs[pairs["study"] == "E3a"]
  for n in E3A_N:
    got = int((e3a["n"] == n).sum())
    if got != E3A_FITS_PER_N:
      gaps.append(f"E3a at n = {n} has {got} pairs of {E3A_FITS_PER_N}")
  extra = sorted(set(e3a["n"]) - set(E3A_N))
  if extra:
    gaps.append(f"E3a has pairs at n = {extra}, outside the design")
  return gaps


def strata(pairs: pd.DataFrame, metric: str = "KL") -> pd.DataFrame:
  """Each stratum's paired difference, ``ARM`` minus ``BASE``.

  Parameters
  ----------
  pairs : pandas.DataFrame
      The pairs, or a group's share of them.
  metric : str
      ``"KL"`` or ``"IAE"``.

  Returns
  -------
  pandas.DataFrame
      Per (study, n): the pairs, both arms' means, the mean difference and
      its SE, the relative change, and the share of pairs ``ARM`` wins.
  """
  out = []
  for (study, n), part in pairs.groupby(["study", "n"]):
    d = part[f"arm_{metric}"] - part[f"base_{metric}"]
    base = float(part[f"base_{metric}"].mean())
    out.append(
      {
        "study": study,
        "n": n,
        "pairs": len(d),
        "base": base,
        "arm": float(part[f"arm_{metric}"].mean()),
        "diff": float(d.mean()),
        "se": float(d.std(ddof=1) / math.sqrt(len(d))),
        "relative": float(d.mean()) / base if base > 0 else math.nan,
        "wins": float((d < 0).mean()),
      }
    )
  return pd.DataFrame(out)


def score(pairs: pd.DataFrame, metric: str = "KL") -> tuple[float, float]:
  """A group's score and SE: its strata's mean differences, equal weight.

  Parameters
  ----------
  pairs : pandas.DataFrame
      The group's pairs.
  metric : str
      ``"KL"`` or ``"IAE"``.

  Returns
  -------
  tuple of float
      The equal-weight mean over strata of each stratum's mean difference,
      and its SE, ``sqrt(sum se_s**2) / S``: the strata are independent.
  """
  table = strata(pairs, metric)
  return (
    float(table["diff"].mean()),
    # A stratum of one pair has no SE, and neither has a score built on it.
    float(np.sqrt((table["se"] ** 2).sum(skipna=False)) / len(table)),
  )


def groups(pairs: pd.DataFrame, metric: str = "KL") -> pd.DataFrame:
  """Every group the rule reads: all pairs, each n and each family.

  Parameters
  ----------
  pairs : pandas.DataFrame
      Both studies' pairs.
  metric : str
      ``"KL"`` or ``"IAE"``.

  Returns
  -------
  pandas.DataFrame
      One row per group, with its score, SE, z and number of strata.
  """
  members = [("overall", "all", pairs)]
  members += [("n", str(n), part) for n, part in pairs.groupby("n")]
  members += [("family", f, part) for f, part in pairs.groupby("family")]
  out = []
  for kind, level, part in members:
    m, se = score(part, metric)
    out.append(
      {
        "group": kind,
        "level": level,
        "strata": int(part.groupby(["study", "n"]).ngroups),
        "pairs": len(part),
        "score": m,
        "se": se,
        "z": m / se if se > 0 else math.nan,
      }
    )
  return pd.DataFrame(out)


def verdict(table: pd.DataFrame) -> tuple[bool, str]:
  """Apply the rule to :func:`groups`' KL table.

  Parameters
  ----------
  table : pandas.DataFrame
      The KL groups.

  Returns
  -------
  tuple
      Whether qnorm is adopted, and the reason in one line.

  Raises
  ------
  SystemExit
      If a group has no SE, which a comparison would read as neither a gain
      nor a loss.
  """
  undefined = table[table["se"].isna()]
  if not undefined.empty:
    raise SystemExit(
      "every group must have an SE for the rule to read; got none for "
      + ", ".join(
        f"{g} {lev}"
        for g, lev in zip(undefined["group"], undefined["level"], strict=True)
      )
    )
  overall = table[table["group"] == "overall"].iloc[0]
  gain = bool(overall["score"] < -TIE_SE * overall["se"])
  rest = table[table["group"] != "overall"]
  losses = rest[rest["score"] > TIE_SE * rest["se"]]
  named = ", ".join(
    f"{g} {lev}"
    for g, lev in zip(losses["group"], losses["level"], strict=True)
  )
  if not gain:
    return False, (
      f"no gain beyond {TIE_SE:g} SE overall (z = {overall['z']:+.2f})"
      + (f"; loss beyond {TIE_SE:g} SE in {named}" if named else "")
    )
  if named:
    return False, (
      f"gain overall (z = {overall['z']:+.2f}), but loss beyond {TIE_SE:g} "
      f"SE in {named}"
    )
  return True, (
    f"gain overall (z = {overall['z']:+.2f}) and no loss beyond {TIE_SE:g} "
    "SE in any n or family"
  )


def _commits(e1: Path, e3a: Path) -> str:
  """The commits both runs record, for the log."""
  lines = (e1 / "provenance.jsonl").read_text().splitlines()
  e1_commits = sorted({json.loads(line)["npcc_commit"][:7] for line in lines})
  e3a_commits = sorted(
    {
      json.loads(p.read_text())["npcc_commit"]
      for p in (e3a / "npcc_cells").glob("*/n*/k*.json")
    }
  )
  return f"E1 run from {e1_commits}; E3a fits from {e3a_commits}"


def main() -> None:
  """Print the screen's tables and verdict."""
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--e1", type=Path, required=True)
  p.add_argument("--e3a", type=Path, required=True)
  p.add_argument("--out", type=Path, required=True)
  p.add_argument(
    "--allow-partial",
    action="store_true",
    help="preview a run that lacks pairs; prints no verdict",
  )
  a = p.parse_args()
  pd.set_option("display.width", 200, "display.max_columns", 20)

  pairs = pd.concat([e1_pairs(a.e1), e3a_pairs(a.e3a)], ignore_index=True)
  gaps = check_design(pairs)
  if gaps and not a.allow_partial:
    raise SystemExit(
      "the screen must have every pair of its design; got: " + "; ".join(gaps)
    )
  a.out.mkdir(parents=True, exist_ok=True)
  pairs.to_csv(a.out / "pairs.csv", index=False)
  tables = []
  for metric in METRICS:
    strata_table = strata(pairs, metric)
    group_table = groups(pairs, metric)
    tables.append(group_table.assign(metric=metric))
    decides = "decides" if metric == "KL" else "reported, does not decide"
    print(f"\n== pdf {metric}, {ARM} minus {BASE}, by stratum ({decides}) ==")
    print(strata_table.to_string(index=False, float_format="{:.5f}".format))
    print(f"\n== pdf {metric}: groups, equal weight over strata ==")
    print(group_table.to_string(index=False, float_format="{:+.5f}".format))
  pd.concat(tables, ignore_index=True).to_csv(a.out / "groups.csv", index=False)
  print(f"\n{_commits(a.e1, a.e3a)}")
  print(f"{len(pairs)} pairs -> {a.out / 'pairs.csv'}, {a.out / 'groups.csv'}")
  if gaps:
    print("\nPREVIEW, no verdict: " + "; ".join(gaps))
    return
  adopt, why = verdict(tables[0])
  print(f"\n-- verdict: {'ADOPT qnorm(u)' if adopt else 'KEEP u'}: {why} --")


if __name__ == "__main__":
  main()
