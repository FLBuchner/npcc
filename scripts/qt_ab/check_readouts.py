"""Self-checks of the read-out variants before they are used in the A/B."""

from __future__ import annotations

import itertools

import numpy as np
import torch
from readouts import Readout, ReadoutTable

from npcc.core._quantile_table import QuantileTable, normal_scores
from npcc.core._transform import resolve_transform

torch.set_num_threads(1)
gen = torch.Generator().manual_seed(0)


def noisy_table(n: int, m: int, limit: float, noise: float) -> torch.Tensor:
  t = normal_scores(m, limit)
  base = 0.3 + 1.1 * t + 0.15 * t**2 * torch.sign(t)
  return base + noise * torch.randn(n, m, generator=gen, dtype=torch.float64)


# 1. ma|current reproduces the library bit for bit.
for tr, width in itertools.product(
  ("logit", "probit", "identity", "log"), (0.0, 1.0, 2.0)
):
  g = resolve_transform(tr)
  q = noisy_table(500, 21, 3.25, 0.3)
  lib = QuantileTable(
    q, limit=3.25, transform=g, eps=1e-6, smoothing=width, min_slope=0.2
  )
  var = ReadoutTable(
    q,
    limit=3.25,
    transform=g,
    eps=1e-6,
    readout=Readout(width=width),
    min_slope=0.2,
  )
  lo, hi = g.support
  if np.isfinite(lo) and np.isfinite(hi):
    y = torch.rand(500, 50, generator=gen, dtype=torch.float64)
  elif np.isfinite(lo):
    y = torch.exp(3 * torch.randn(500, 50, generator=gen, dtype=torch.float64))
  else:
    y = 4 * torch.randn(500, 50, generator=gen, dtype=torch.float64)
  p = torch.rand(500, 50, generator=gen, dtype=torch.float64).clamp(
    1e-9, 1 - 1e-9
  )
  d = max(
    (lib.pdf(y) - var.pdf(y)).abs().max().item(),
    (lib.cdf(y) - var.cdf(y)).abs().max().item(),
    (lib.icdf(p) - var.icdf(p)).abs().max().item(),
  )
  assert d == 0.0, (tr, width, d)
print("library reproduction: exact")

# 2. Every variant is a distribution: cdf(icdf(p)) = p, pdf = d cdf, cdf in
#    [0, 1] and increasing, including in the tails.
for tr, rd in itertools.product(
  ("logit", "probit", "identity", "log"),
  [
    Readout(s, w, t)
    for (s, w) in (("ma", 0.0), ("ma", 2.0), ("whittaker", 0.65))
    for t in ("current", "modeled")
  ],
):
  g = resolve_transform(tr)
  q = noisy_table(64, 21, 3.25, 0.2)
  tab = ReadoutTable(
    q, limit=3.25, transform=g, eps=1e-12, readout=rd, min_slope=0.2
  )
  p = torch.logspace(-9, -1e-9, 200, dtype=torch.float64)
  p = torch.cat([p, 1 - p]).sort().values.expand(64, -1).contiguous()
  y = tab.icdf(p)
  # Beyond what float64 resolves on the response scale, y saturates at an end
  # of the support and cannot round-trip; that is representability, not the
  # read-out.
  lo, hi = g.support
  ok = (y > max(lo, 0.0) + 1e-12 if np.isfinite(lo) else torch.isfinite(y)) & (
    y < hi - 1e-12 if np.isfinite(hi) else torch.isfinite(y)
  )
  err_round = (tab.cdf(y) - p).abs()[ok].max().item()
  # Density vs a central difference of the cdf, away from knots.
  yy = tab.icdf(
    torch.linspace(1e-6, 1 - 1e-6, 400, dtype=torch.float64)
    .expand(64, -1)
    .contiguous()
  )
  h = 1e-6 * (1 + yy.abs())
  num = (tab.cdf(yy + h) - tab.cdf(yy - h)) / (2 * h)
  rel = ((tab.pdf(yy) - num).abs() / num.clamp_min(1e-300)).median().item()
  assert err_round < 1e-8 and rel < 1e-4, (tr, rd.label, err_round, rel)
print("variants are consistent distributions")

# 3. Modeled tails on an exact N(0,1) probit table: uniform, tails included.
g = resolve_transform("probit")
q = normal_scores(21, 3.25).expand(3, -1).contiguous()
tab = ReadoutTable(
  q,
  limit=3.25,
  transform=g,
  eps=1e-12,
  readout=Readout(width=0.0, tails="modeled"),
  min_slope=0.2,
)
y = torch.tensor(
  [1e-9, 1e-6, 1e-4, 0.3, 1 - 1e-4, 1 - 1e-9], dtype=torch.float64
).expand(3, -1)
print("probit exact-normal pdf (want 1):", tab.pdf(y.contiguous())[0].numpy())
