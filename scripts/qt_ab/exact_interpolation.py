"""Interior error on exact copula quantiles: level grid and interpolation scale.

No backend is involved and nothing is smoothed: the true quantiles of
``V | U = u`` (Clayton and Gaussian copulas, Kendall's tau 0.5 / 0.7,
u = 0.05 / 0.5) are interpolated, and the density's integrated absolute error
over the span of the table is reported for

- ``lin_t_m21_L3.25``: the library's interpolation, linear in the normal
  score ``t``, on 21 levels with ``normal_limit = 3.25``;
- ``lin_t_m19_L2.925`` and ``lin_t_m41_L3.25``: the same on the other grids
  of the A/B;
- ``lin_alpha_m21_L3.25``: TabICLv2's interpolation (appendix I.2), linear in
  the level ``alpha``, which makes the density constant in each cell on the
  modeled scale, at the library's 21 levels.

Output: ``results/qt_ab/exact_interpolation.csv``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from _common import RESULTS, commit
from exact_readout_bias import DV, FAMILIES, V
from scipy.stats import norm

from npcc.core._quantile_table import QuantileTable
from npcc.core._transform import resolve_transform

OUT = RESULTS / "exact_interpolation.csv"

Hinv = Callable[[np.ndarray, float], np.ndarray]
Dens = Callable[[float, np.ndarray], np.ndarray]


def iae_linear_t(
  hinv: Hinv, dens: Dens, u: float, tr: str, m: int, lim: float
) -> float:
  """IAE of the library's linear-in-t interpolation, inside the table."""
  t = np.linspace(-lim, lim, m)
  a = norm.cdf(t)
  g = resolve_transform(tr)
  z = g.forward(torch.as_tensor(hinv(a, u)))[None]
  table = QuantileTable(
    z, limit=lim, transform=g, eps=1e-12, smoothing=0.0, min_slope=0.0
  )
  est = table.pdf(torch.as_tensor(V)[None]).numpy()[0]
  inside = (V > hinv(a[0], u)) & (V < hinv(a[-1], u))
  return float(np.sum((np.abs(est - dens(u, V)) * DV)[inside]))


def iae_linear_alpha(
  hinv: Hinv, dens: Dens, u: float, tr: str, m: int, lim: float
) -> float:
  """IAE of linear-in-alpha interpolation at the same levels."""
  t = np.linspace(-lim, lim, m)
  a = norm.cdf(t)
  g = resolve_transform(tr)
  vq = hinv(a, u)
  z = g.forward(torch.as_tensor(vq)).numpy()
  zv = g.forward(torch.as_tensor(V)).numpy()
  jac = g.jacobian(torch.as_tensor(V)).numpy()
  k = np.clip(np.searchsorted(z, zv) - 1, 0, m - 2)
  est = (a[k + 1] - a[k]) / (z[k + 1] - z[k]) * jac
  inside = (V > vq[0]) & (V < vq[-1])
  return float(np.sum((np.abs(est - dens(u, V)) * DV)[inside]))


def main() -> None:
  """Write the CSV, with the commit it came from."""
  torch.set_num_threads(1)
  head, dirty = commit()
  rows = []
  for family, (maker, par) in FAMILIES.items():
    for tau in (0.5, 0.7):
      hinv, dens = maker(par(tau))
      for u in (0.05, 0.5):
        for tr in ("logit", "probit"):
          rows.append(
            {
              "family": family,
              "tau": tau,
              "u": u,
              "transform": tr,
              "lin_t_m21_L3.25": iae_linear_t(hinv, dens, u, tr, 21, 3.25),
              "lin_t_m19_L2.925": iae_linear_t(hinv, dens, u, tr, 19, 2.925),
              "lin_t_m41_L3.25": iae_linear_t(hinv, dens, u, tr, 41, 3.25),
              "lin_alpha_m21_L3.25": iae_linear_alpha(
                hinv, dens, u, tr, 21, 3.25
              ),
            }
          )
  frame = pd.DataFrame(rows).assign(commit=head, dirty=dirty)
  OUT.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(OUT, index=False)
  print(frame.drop(columns=["commit", "dirty"]).round(4).to_string())
  print(f"wrote {OUT}")


if __name__ == "__main__":
  main()
