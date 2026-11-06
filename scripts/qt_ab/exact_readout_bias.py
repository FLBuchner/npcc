"""The read-out's own error, on exact copula conditional quantiles.

No backend is involved: the true quantiles of ``V | U = u`` (Clayton and
Gaussian copulas, Kendall's tau 0.3 / 0.5 / 0.7, u = 0.05 / 0.5 / 0.95) at the
21 normal-grid levels go through the library's ``QuantileTable`` under logit
and probit, and its density is compared with the true conditional density.

Columns of ``results/qt_ab/exact_readout_bias.csv``:

- ``iae_s{w}_f{f}``: integrated absolute error of the density over (0, 1) with
  slope smoothing ``w`` normal-score units (0, 1, 2 = 1, 3, 7 cells) and slope
  floor ``f`` (0 or 0.2);
- ``iae_weighted0.65_f0.2``: the same with the weighted smoother at bandwidth
  0.65, the gradient-boosting backends' default;
- ``tail_kl_current`` / ``tail_kl_modeled``: the part of KL(true || read-out)
  beyond the outermost knots, with the library's tails (uniform on the copula
  scale) and with modeled-scale tails (exponential in logit z, linear in the
  probit score) fitted by anchored least squares on the 4 outer knots;
- ``floor_binds``: the share of cells whose exact slope is below the floor.

Integrals are on a grid uniform in the normal score of ``v``, so peaked
conditionals and both tails are resolved.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from npcc.core._quantile_table import QuantileTable
from npcc.core._transform import resolve_transform

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "qt_ab" / "exact_readout_bias.csv"

L, M = 3.25, 21
T = np.linspace(-L, L, M)
ALPHA = norm.cdf(T)
A0 = ALPHA[0]
TAIL_KNOTS = 4

Fn = Callable[[np.ndarray, float], np.ndarray]


def clayton(
  theta: float,
) -> tuple[Fn, Callable[[float, np.ndarray], np.ndarray]]:
  """Clayton's inverse h-function ``hinv(p, u)`` and density ``dens(u, v)``."""

  def hinv(p: np.ndarray, u: float) -> np.ndarray:
    return (
      (p * u ** (theta + 1)) ** (-theta / (1 + theta)) + 1 - u**-theta
    ) ** (-1 / theta)

  def dens(u: float, v: np.ndarray) -> np.ndarray:
    return (
      (1 + theta)
      * (u * v) ** (-theta - 1)
      * (u**-theta + v**-theta - 1) ** (-2 - 1 / theta)
    )

  return hinv, dens


def gauss(rho: float) -> tuple[Fn, Callable[[float, np.ndarray], np.ndarray]]:
  """The Gaussian copula's ``hinv(p, u)`` and density ``dens(u, v)``."""
  r = math.sqrt(1 - rho * rho)

  def hinv(p: np.ndarray, u: float) -> np.ndarray:
    return norm.cdf(norm.ppf(p) * r + rho * norm.ppf(u))

  def dens(u: float, v: np.ndarray) -> np.ndarray:
    x, y = norm.ppf(u), norm.ppf(v)
    return (
      np.exp(-(rho**2 * (x * x + y * y) - 2 * rho * x * y) / (2 * r * r)) / r
    )

  return hinv, dens


FAMILIES = {
  "clayton": (clayton, lambda tau: 2 * tau / (1 - tau)),
  "gaussian": (gauss, lambda tau: math.sin(math.pi * tau / 2)),
}

_S = np.linspace(-8.0, 8.0, 40001)
_V = norm.cdf(_S)
_DV = norm.pdf(_S) * (_S[1] - _S[0])
_KEEP = (_V > 0) & (_V < 1)
#: Evaluation grid on v, uniform in its normal score, and its weights.
V, DV = _V[_KEEP], _DV[_KEEP]


def modeled_tail_density(z: np.ndarray, transform: str) -> np.ndarray:
  """Modeled-scale tails beyond the table, on v; NaN inside the table."""
  g = resolve_transform(transform)
  vt = torch.as_tensor(V)
  zv = g.forward(g.clip(vt, 1e-300)).numpy()
  jac = g.jacobian(g.clip(vt, 1e-300)).numpy()
  out = np.full_like(V, np.nan)
  k = TAIL_KNOTS
  for lower in (True, False):
    zz = z[:k] if lower else -z[::-1][:k]
    mask = zv < z[0] if lower else zv > z[-1]
    w = (zv if lower else -zv) - zz[0]
    y = zz - zz[0]
    if transform == "logit":
      x = np.log(ALPHA[:k]) - np.log(ALPHA[0])
      beta = (x @ y) / (x @ x)
      f = A0 * np.exp(w / beta) / beta
    else:
      x = T[:k] - T[0]
      s = (x @ y) / (x @ x)
      f = norm.pdf(-L + w / s) / s
    out[mask] = (f * jac)[mask]
  return out


def rows() -> list[dict[str, object]]:
  """One row per (family, tau, u, transform)."""
  out = []
  for family, (maker, par) in FAMILIES.items():
    for tau in (0.3, 0.5, 0.7):
      hinv, dens = maker(par(tau))
      for u in (0.05, 0.5, 0.95):
        truth = dens(u, V)
        vq = hinv(ALPHA, u)
        for tr in ("logit", "probit"):
          g = resolve_transform(tr)
          z = g.forward(torch.as_tensor(vq)).numpy()
          slopes = np.diff(z) / (T[1] - T[0])
          mean_slope = (z[-1] - z[0]) / (2 * L)
          row: dict[str, object] = {
            "family": family,
            "tau": tau,
            "u": u,
            "transform": tr,
            "floor_binds": float((slopes < 0.2 * mean_slope).mean()),
          }
          unsmoothed = np.empty(0)
          for width, floor in ((0.0, 0.0), (0.0, 0.2), (1.0, 0.2), (2.0, 0.2)):
            table = QuantileTable(
              torch.as_tensor(z)[None],
              limit=L,
              transform=g,
              eps=1e-12,
              smoothing=width,
              min_slope=floor,
            )
            est = table.pdf(torch.as_tensor(V)[None]).numpy()[0]
            row[f"iae_s{width:g}_f{floor:g}"] = float(
              np.sum(np.abs(est - truth) * DV)
            )
            if (width, floor) == (0.0, 0.0):
              unsmoothed = est
          weighted = QuantileTable(
            torch.as_tensor(z)[None],
            limit=L,
            transform=g,
            eps=1e-12,
            smoothing=0.65,
            min_slope=0.2,
            smoother="weighted",
          )
          est = weighted.pdf(torch.as_tensor(V)[None]).numpy()[0]
          row["iae_weighted0.65_f0.2"] = float(np.sum(np.abs(est - truth) * DV))
          tail = modeled_tail_density(z, tr)
          beyond = ~np.isnan(tail)
          row["tail_kl_current"] = float(
            np.sum((truth * np.log(truth / unsmoothed) * DV)[beyond])
          )
          row["tail_kl_modeled"] = float(
            np.sum((truth * np.log(truth / tail) * DV)[beyond])
          )
          out.append(row)
  return out


def main() -> None:
  """Write the CSV, with the commit it came from."""
  from nori_probe import commit

  torch.set_num_threads(1)
  head, dirty = commit()
  frame = pd.DataFrame(rows()).assign(commit=head, dirty=dirty)
  OUT.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(OUT, index=False)
  print(frame.drop(columns=["commit", "dirty"]).round(5).to_string())
  print(f"wrote {OUT}")


if __name__ == "__main__":
  main()
