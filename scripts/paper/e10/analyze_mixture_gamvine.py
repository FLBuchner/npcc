"""Compare E10's conditional structure mixtures with E2b's GAM vines.

Reads a finished vine run holding the ``mixture`` arm
(``configs/vine_study_mixture*.toml``) and, through ``--with``, E2b's runs on
the same cells (``configs/vine_study_gamvine*.toml``), whose arms
``gamvine-oracle``, ``gamvine-selected`` and ``gamvine-ns`` the mixture is
read against. Writes ``vine_mixture_gamvine.csv`` into ``--out``, one row per
(d, n, GAM arm). For the joint and the copula KL, each x-averaged within a
repetition before any difference is taken:

- the mixture's and the GAM arm's means over the repetitions both hold;
- the paired difference mixture - GAM arm, as mean and SE over those
  repetitions;
- ``*_mixture_lower``, the repetitions in which the mixture's KL is lower.

``n_rep`` counts the repetitions paired, and ``n_failed`` those in which the
GAM fit failed, which every mean leaves out.

The runs must pair, as in ``analyze_mixture.py``, whose loading and pairing
this reuses: every (d, n, rep, x) both hold must have the same cell seed and
the same mean true log-density, which pins the cell's truth and test points.
A ``--with`` run that shares no cell with the mixture run is refused, so pass
the GAM runs at the mixture run's d and n only.

``--out`` has no default: the mixture runs' own ``analysis/`` folders are
cited, and this writes beside nothing there.

Usage (from the npcc repository root)::

    uv run --no-sync python scripts/paper/e10/analyze_mixture_gamvine.py \\
        results/vine_study_mixture_<TAG> \\
        --with results/vine_study_gamvine_<TAG> \\
        results/vine_study_gamvine_n2000_<TAG> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from analyze_mixture import METRICS, load, others, stat, wide

GAMVINE = ("gamvine-oracle", "gamvine-selected", "gamvine-ns")


def table(metrics: pd.DataFrame, gam: pd.DataFrame) -> pd.DataFrame:
  """One row per (d, n, GAM arm): the mixture against that arm, paired."""
  if "mixture" not in set(metrics["arm"]):
    raise SystemExit("the run must hold the mixture arm")
  ours = {name: wide(metrics, m)["mixture"] for name, m in METRICS.items()}
  theirs = {name: wide(gam, m) for name, m in METRICS.items()}
  cells = (
    ours["joint"]
    .index.droplevel("rep")
    .unique()
    .intersection(theirs["joint"].index.droplevel("rep").unique())
    .sort_values()
  )
  rows: list[dict[str, object]] = []
  for d, n in cells:
    for arm in (a for a in GAMVINE if a in theirs["joint"].columns):
      columns = {
        (name, side): frame
        for name in METRICS
        for side, frame in (
          ("mixture", ours[name].loc[(d, n)]),
          ("gamvine", theirs[name].loc[(d, n), arm]),
        )
      }
      both = pd.concat(columns, axis=1, join="inner")
      failed = both[("joint", "gamvine")].isna()
      both = both.dropna()
      row: dict[str, object] = {
        "d": int(d),
        "n": int(n),
        "arm": arm,
        "n_rep": len(both),
        "n_failed": int(failed.sum()),
      }
      for name in METRICS:
        mixture, other = both[(name, "mixture")], both[(name, "gamvine")]
        diff = mixture - other
        row[f"{name}_KL_mixture_mean"], _ = stat(mixture)
        row[f"{name}_KL_gamvine_mean"], _ = stat(other)
        (
          row[f"{name}_mixture_minus_gamvine_mean"],
          row[f"{name}_mixture_minus_gamvine_se"],
        ) = stat(diff)
        row[f"{name}_mixture_lower"] = int((diff < 0).sum())
      rows.append(row)
  if not rows:
    raise SystemExit(
      "the --with runs must hold a GAM arm on the mixture's cells"
    )
  return pd.DataFrame(rows)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("run", type=Path, help="A vine run holding the mixture arm.")
  p.add_argument(
    "--with",
    dest="others",
    type=Path,
    nargs="+",
    required=True,
    help="E2b's GAM vine runs on the mixture run's cells.",
  )
  p.add_argument(
    "--out",
    type=Path,
    required=True,
    help="Where to write vine_mixture_gamvine.csv.",
  )
  a = p.parse_args()
  metrics = load(a.run)
  gam = others(a.others, metrics, arms=GAMVINE)
  result = table(metrics, gam)
  a.out.mkdir(parents=True, exist_ok=True)
  result.to_csv(a.out / "vine_mixture_gamvine.csv", index=False)
  with pd.option_context("display.width", 200, "display.max_columns", 40):
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
  main()
