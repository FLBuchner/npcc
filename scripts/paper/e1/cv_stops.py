"""How close each cross-validated boostCopula fit stopped to its iteration cap.

``fit_boostcopula_smooth_cv.R`` logs, per fit, the family it selected and
``cv-stop K/M``: the iteration the cross-validation chose out of the cap. A
stop within 5% of the cap is one the cap, not the cross-validation, may have
set. This writes ``<run>/analysis/cv_stops.csv``, one row per fit, and
``cv_stops_by_n.csv``: per n and overall, the fits, how many stopped within
5% of the cap, and the median and largest stop.

Usage::

    uv run --no-sync python scripts/paper/e1/cv_stops.py \
      results/study_2026-11-05_r_smooth_cv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

_LINE = re.compile(
  r"(?P<label>[\w-]+) on (?P<family>[\w-]+?)__(?P<tau_scenario>[\w-]+?)"
  r"__n(?P<n>\d+)__rep(?P<rep>\d+): .*?family (?P<chosen>\d+) .*?"
  r"cv-stop (?P<stop>\d+)/(?P<maxit>\d+)"
)
KEYS = ["label", "family", "tau_scenario", "n", "rep"]


def main() -> None:
  run = Path(sys.argv[1])
  rows = [
    m.groupdict()
    for log in sorted((run / "logs").glob("shard*.log"))
    for line in log.read_text().splitlines()
    if (m := _LINE.search(line))
  ]
  if not rows:
    raise SystemExit(f"{run}/logs must hold cv-stop lines; found none")
  fits = pd.DataFrame(rows).astype(
    {"n": int, "rep": int, "chosen": int, "stop": int, "maxit": int}
  )
  # A cell rerun after an interruption logs twice; the last line is its fit.
  fits = fits.drop_duplicates(KEYS, keep="last").sort_values(KEYS)
  fits["near_cap"] = fits["stop"] >= 0.95 * fits["maxit"]

  def summary(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
      {
        "fits": len(group),
        "near_cap": int(group["near_cap"].sum()),
        "median_stop": float(group["stop"].median()),
        "max_stop": int(group["stop"].max()),
        "maxit": int(group["maxit"].max()),
      }
    )

  by_n = fits.groupby("n").apply(summary, include_groups=False).reset_index()
  overall = summary(fits).to_frame().T.assign(n="all")
  table = pd.concat([by_n, overall], ignore_index=True).astype(
    {"fits": int, "near_cap": int, "max_stop": int, "maxit": int}
  )
  out = run / "analysis"
  out.mkdir(exist_ok=True)
  fits.to_csv(out / "cv_stops.csv", index=False)
  table.to_csv(out / "cv_stops_by_n.csv", index=False)
  print(table.to_string(index=False))


if __name__ == "__main__":
  main()
