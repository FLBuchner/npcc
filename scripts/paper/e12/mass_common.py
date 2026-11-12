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
ratios. On E3's ``strong`` design at d = 10 the true vine is concentrated
enough that a handful of test points decide the estimate. There, the
parametric vine (Dissmann with BIC), a proper density, is estimated at
0.04 on average over E3d's ten cells and at most 0.12, with an effective
sample size of a few points out of 20,000. On the two such cells that
``refit_mass.py`` reached, its defensive-mixture proposal puts the same
parametric fits at 0.95 and 0.93. Every E12 table therefore reports those
cells as not estimable; :data:`NOT_ESTIMABLE` lists them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: The npcc checkout these scripts live in; inputs default to its results.
REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results"

#: (design, d) where the truth-only estimate fails. Every table blanks the
#: estimate's columns there and marks the row ``estimable = False``.
NOT_ESTIMABLE = frozenset({("strong", 10)})

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


def estimable(design: str, d: int) -> bool:
  """Whether the truth-only estimate holds on ``design`` at dimension ``d``."""
  return (design, int(d)) not in NOT_ESTIMABLE


def blank_not_estimable(
  table: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
  """``table`` with ``estimable`` added and ``columns`` blanked where not.

  Parameters
  ----------
  table : pd.DataFrame
      Rows with ``design`` and ``d`` columns.
  columns : list of str
      The columns that derive from the truth-only estimate.

  Returns
  -------
  pd.DataFrame
      A copy, with NaN in ``columns`` on the rows of :data:`NOT_ESTIMABLE`.
  """
  out = table.copy()
  out["estimable"] = [
    estimable(des, d) for des, d in zip(out["design"], out["d"], strict=True)
  ]
  out.loc[~out["estimable"], columns] = np.nan
  return out


def printable(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
  """``table`` as text, ``columns`` reading ``not estimable`` where it fails.

  Parameters
  ----------
  table : pd.DataFrame
      Rows with ``design`` and ``d`` as columns or index levels.
  columns : list of str
      The columns that derive from the truth-only estimate.

  Returns
  -------
  pd.DataFrame
      The same index and columns, every float written to four decimals.
  """
  flat = table.reset_index()
  ok = np.array(
    [
      estimable(des, d)
      for des, d in zip(flat["design"], flat["d"], strict=True)
    ]
  )
  out = table.astype(object)
  for c in table.columns:
    if pd.api.types.is_float_dtype(table[c]):
      out[c] = [f"{x:.4f}" if np.isfinite(x) else "NaN" for x in table[c]]
  out.loc[~ok, columns] = "not estimable"
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
