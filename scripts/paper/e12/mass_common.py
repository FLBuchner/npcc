"""E12's shared pieces: the truth-only mass estimate, and where it fails.

**The estimate.** A cell's test points ``y_i ~ f`` are drawn from the true
vine ``f``. With a fitted vine's stored log-densities ``log f_hat(y_i)`` and
the true ``log f(y_i)``, the ratios ``r_i = f_hat(y_i) / f(y_i)`` give

    m_hat = mean_i r_i,

an importance-sampling estimate of the total mass ``int f_hat``, which is 1
for a proper density. It is unbiased when ``f > 0`` wherever ``f_hat > 0``,
and its Monte Carlo SE is ``sd(r) / sqrt(n)``. :func:`stats` adds robust
companions and the diagnostics that say when the estimate is not to be
trusted.

**Where it fails.** The truth decides where the test points fall, so mass
that ``f_hat`` puts where ``f`` is small is reached only through a few huge
ratios, and a low ESS share flags it. On E3's ``strong`` design at d = 10
the parametric vine (Dissmann with BIC), a proper density, is estimated at
0.04 on average over E3d's ten cells, with a median ESS share of 5e-4;
``refit_mass.py``'s defensive-mixture proposal puts the same fits at 0.93
to 0.96. At ``strong`` d = 5 it reads 0.85, with cells from 0.54 to 1.14.

**The rule.** A cell's mass is reported only if its ESS share is at least
:data:`MIN_ESS_SHARE`; a row (an arm on one design at one d) only if at
least :data:`MIN_PASSING` of every :data:`OF_CELLS` of its cells pass, and
its mass is then the mean over the passing cells. Everything else is not
estimable by importance sampling under the truth: :func:`mark_cells` and
:func:`blank_rows` blank it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: The npcc checkout these scripts live in; inputs default to its results.
REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results"

#: A cell's mass is reported only if its ESS share is at least this.
MIN_ESS_SHARE = 0.1

#: A row is reported only if at least ``MIN_PASSING`` of every ``OF_CELLS``
#: of its cells pass: 8 of E3d's 10 seeds, 4 of E3c's 5.
MIN_PASSING, OF_CELLS = 8, 10

#: The per-cell columns that derive from the mass estimate.
CELL_IS_COLUMNS = [
  "mass",
  "mass_se",
  "neglogmass",
  "mass_mom20",
  "mass_trim_top0.1pct",
]

#: What a table prints in place of an estimate the rule blanks.
NOT_ESTIMABLE = "not estimable"

#: The NPCC estimators' arm-name prefixes, shortened in the tables.
TABPFN = "npcc-tabpfn-quantiles-v3-probit"
CATBOOST = "npcc-catboost-probit"


def shorten(
  arm: str, tabpfn: str = "tabpfn", catboost: str = "catboost"
) -> str:
  """``arm`` with the NPCC estimators' prefixes replaced by short names."""
  return arm.replace(TABPFN, tabpfn).replace(CATBOOST, catboost)


def stats(lr: np.ndarray, n_blocks: int = 20) -> dict[str, float | int]:
  """The mass estimate and its diagnostics, from finite log-ratios.

  Parameters
  ----------
  lr : np.ndarray
      ``log f_hat - log f`` at the test points, all finite.
  n_blocks : int
      The number of blocks of the median of means.

  Returns
  -------
  dict
      ``n_finite``; ``mass`` (``m_hat``), ``mass_se`` and ``neglogmass``
      (``-log m_hat``); ``mass_mom20``, the median of the means of
      ``n_blocks`` consecutive blocks; ``mass_trim_top0.1pct``, the mean
      without the largest 0.1% of the ratios; ``ess_share``, the effective
      sample size over n, ``(sum r)^2 / (n sum r^2)``; ``top1_share`` and
      ``top10_share``, the largest and the ten largest ratios' share of
      ``sum r``; ``max_ratio``; and ``excess_nll``, ``-mean(lr)``.
  """
  n = lr.size
  r = np.exp(lr)
  m = r.mean()
  se = r.std(ddof=1) / np.sqrt(n)
  blocks = np.array_split(r, n_blocks)
  mom = float(np.median([b.mean() for b in blocks]))
  k = max(1, round(0.001 * n))
  srt = np.sort(r)
  trim = float(srt[:-k].mean())
  ess = float(r.sum() ** 2 / (n * np.square(r).sum()))
  top = float(srt[-1] / r.sum())
  top10 = float(srt[-10:].sum() / r.sum())
  return {
    "n_finite": n,
    "mass": float(m),
    "mass_se": float(se),
    "neglogmass": float(-np.log(m)),
    "mass_mom20": mom,
    "mass_trim_top0.1pct": trim,
    "ess_share": ess,
    "top1_share": top,
    "top10_share": top10,
    "max_ratio": float(srt[-1]),
    "excess_nll": float(-lr.mean()),
  }


def mark_cells(cells: pd.DataFrame) -> pd.DataFrame:
  """``cells`` with ``estimable`` added and the estimate blanked where it fails.

  Parameters
  ----------
  cells : pd.DataFrame
      One row per cell, with :func:`stats`'s columns.

  Returns
  -------
  pd.DataFrame
      A copy, with NaN in :data:`CELL_IS_COLUMNS` on the cells whose ESS
      share is below :data:`MIN_ESS_SHARE`.
  """
  out = cells.copy()
  out["estimable"] = out["ess_share"] >= MIN_ESS_SHARE
  out.loc[~out["estimable"], CELL_IS_COLUMNS] = np.nan
  return out


def rows_pass(n_pass: pd.Series, n: pd.Series) -> pd.Series:
  """Whether rows of ``n`` cells, ``n_pass`` of them passing, are reported."""
  return n_pass * OF_CELLS >= MIN_PASSING * n


def arm_rows_pass(cells: pd.DataFrame) -> dict[tuple[str, int], bool]:
  """Whether each of an arm's rows passes, from its marked cells.

  Parameters
  ----------
  cells : pd.DataFrame
      One arm's cells, as :func:`mark_cells` returns them, indexed by
      design, d and seed.

  Returns
  -------
  dict
      ``(design, d)`` to whether that row is reported.
  """
  rows = cells.groupby(level=[0, 1])["estimable"].agg(["sum", "size"])
  ok = rows_pass(rows["sum"], rows["size"])
  return {(des, int(d)): bool(v) for (des, d), v in ok.items()}


def blank_rows(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
  """``table`` with ``estimable`` added and ``columns`` blanked where not.

  Parameters
  ----------
  table : pd.DataFrame
      Rows with ``n`` (cells) and ``n_pass`` (cells that pass) columns.
  columns : list of str
      The columns that derive from the mass estimate.

  Returns
  -------
  pd.DataFrame
      A copy, with NaN in ``columns`` on the rows :func:`rows_pass` refuses.
  """
  out = table.copy()
  out["estimable"] = rows_pass(out["n_pass"], out["n"])
  out.loc[~out["estimable"], columns] = np.nan
  return out


def printable(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
  """``table`` as text, ``columns`` reading ``not estimable`` where blanked.

  Parameters
  ----------
  table : pd.DataFrame
      Rows with an ``estimable`` column, as :func:`blank_rows` adds.
  columns : list of str
      The columns that derive from the mass estimate.

  Returns
  -------
  pd.DataFrame
      The same index and columns, every float written to four decimals.
  """
  ok = table["estimable"].to_numpy(dtype=bool)
  out = table.astype(object)
  for c in table.columns:
    if pd.api.types.is_float_dtype(table[c]):
      out[c] = [f"{x:.4f}" if np.isfinite(x) else "NaN" for x in table[c]]
  out.loc[~ok, columns] = NOT_ESTIMABLE
  return out


def refuse_duplicates(table: pd.DataFrame, keys: list[str], what: str) -> None:
  """Refuse a table that holds one cell twice, which a mean would count twice.

  Parameters
  ----------
  table : pd.DataFrame
      The rows, before they are paired or averaged.
  keys : list of str
      The columns (or index levels) that name a cell.
  what : str
      What the rows are, for the message.
  """
  frame = table.reset_index()
  dup = frame[frame.duplicated(keys, keep=False)]
  if len(dup):
    raise SystemExit(
      f"{what} must hold each cell once; got duplicates of "
      f"{dup[keys].drop_duplicates().to_dict('records')}. Two input folders "
      "hold the same cells: pass each once."
    )
