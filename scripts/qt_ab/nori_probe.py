"""Nori's precision on the GPU: the dtype of its quantile bank, ties, clamping.

On one E1 cell (clayton / linear, rep 0, E1's ``base_seed = 317``) at
n = 200 and 2000, under logit and probit, with E1's settings (21 levels,
``normal_limit = 3.25``, Nori's defaults), it measures on the 21-level tables
of E1's 200 distinct conditioning rows of ``V | U, X``:

- ``output_dtypes``: the dtype of the raw model output Nori turns into its
  999-level bank, captured at ``NoriPredictor._unwrap_model_output``;
- ``tie_share`` / ``crossing_share``: the share of adjacent levels that are
  tied exactly, or cross, before the read-out sorts them;
- ``distinct_per_row``: distinct values per 21-level row;
- ``outer_cell_ratio_lo`` / ``_hi``: the width of the outermost cell over its
  neighbor's, before the read-out. Nori clamps our outermost level
  (``Phi(-3.25)``) to its own 0.001 quantile, which halves that cell;
- ``outer_slope_ratio_after_lo`` / ``_hi``: the same ratio on the regularized
  slopes the read-out uses, and ``outer_density_excess_lo`` / ``_hi``, the
  density in the outermost cell relative to one with its neighbor's slope,
  minus one;
- ``floored_share``: the share of cells whose slope sits at the ``min_slope``
  floor after the read-out;
- ``outer_cell_mass``: the probability mass of the outermost cell.

Each statistic is a median (ratios) or a mean (shares) over the rows. Output:
``results/qt_ab/nori_probe.csv``, with the commit it was produced from.

Run with the CUDA environment and the worktree's package::

    PYTHONPATH=src ~/npcc/.venv/bin/python scripts/qt_ab/nori_probe.py
"""

from __future__ import annotations

import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ab_gb
from ab_gpu import cap_gpu_memory

from npcc.core._quantile_table import QuantileTable
from npcc.core.bicop import RosenblattBicop
from npcc.core.controls import FitControlsRosenblattBicop
from npcc.experiments import runner, scenarios
from npcc.experiments.config import Cell

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "qt_ab" / "nori_probe.csv"


def commit() -> tuple[str, bool]:
  """The checked-out commit, and whether the tree differs from it."""
  head = subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
    check=True,
  ).stdout.strip()
  dirty = bool(
    subprocess.run(
      ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
      capture_output=True,
      text=True,
      check=True,
    ).stdout.strip()
  )
  return head, dirty


def probe(n: int, transform: str) -> dict[str, object]:
  cell = Cell("clayton", "linear", n, 0)
  seed = runner._cell_seed(ab_gb.BASE_SEED, cell)
  u, v, x = scenarios.sample(cell.family, cell.tau_scenario, n, seed)
  grid = scenarios.eval_grid(
    cell.tau_scenario, conditional_uv_grid_n=20, conditional_x_grid_n=10
  )
  model = RosenblattBicop(
    FitControlsRosenblattBicop(
      backend="nori", transform=transform, device="cuda"
    )
  )
  model.fit(torch.column_stack([u, v]), x=x)
  margin = model.v_given_ux_
  predictor = margin.model_._get_predictor()
  dtypes: Counter[str] = Counter()
  original = predictor._unwrap_model_output

  def spy(output, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
    found = (
      [output]
      if torch.is_tensor(output)
      else [
        t
        for t in (output.values() if isinstance(output, dict) else output)
        if torch.is_tensor(t)
      ]
    )
    for t in found:
      dtypes[str(t.dtype)] += 1
    return original(output, *args, **kwargs)

  predictor._unwrap_model_output = spy
  features = torch.unique(
    model._features(grid.u_flat, grid.x_flat.reshape(-1, 1)), dim=0
  )
  cfg = margin.quantile_table_config
  raw = margin._predict_quantiles(
    features.to(margin._device), cfg.alphas(device=margin._device)
  ).double()
  diff = torch.diff(raw, dim=1)
  srt = torch.sort(raw, dim=1).values
  width = torch.diff(srt, dim=1).clamp_min(1e-300)
  table = QuantileTable(
    raw,
    limit=cfg.normal_limit,
    transform=margin._response_transform,
    eps=margin.eps,
    smoothing=cfg.smoothing,
    min_slope=cfg.min_slope,
  )
  slope = table._slope
  mean_slope = (srt[:, -1:] - srt[:, :1]) / (2 * cfg.normal_limit)
  floor = cfg.min_slope * mean_slope
  t = torch.linspace(-cfg.normal_limit, cfg.normal_limit, cfg.n_quantiles)
  phi = 0.5 * torch.erfc(-t / math.sqrt(2.0))
  row = {
    "family": cell.family,
    "tau_scenario": cell.tau_scenario,
    "rep": cell.rep,
    "seed": seed,
    "n": n,
    "transform": transform,
    "n_quantiles": cfg.n_quantiles,
    "normal_limit": cfg.normal_limit,
    "smoothing": cfg.smoothing,
    "min_slope": cfg.min_slope,
    "rows": raw.shape[0],
    "output_dtypes": ";".join(f"{k}:{c}" for k, c in sorted(dtypes.items())),
    "tie_share": float((diff == 0).double().mean()),
    "crossing_share": float((diff < 0).double().mean()),
    "distinct_per_row": float(
      torch.tensor(
        [r.unique().numel() for r in raw], dtype=torch.float64
      ).mean()
    ),
    "outer_cell_ratio_lo": float((width[:, 0] / width[:, 1]).median()),
    "outer_cell_ratio_hi": float((width[:, -1] / width[:, -2]).median()),
    "outer_slope_ratio_after_lo": float((slope[:, 0] / slope[:, 1]).median()),
    "outer_slope_ratio_after_hi": float((slope[:, -1] / slope[:, -2]).median()),
    "outer_density_excess_lo": float((slope[:, 1] / slope[:, 0]).median() - 1),
    "outer_density_excess_hi": float(
      (slope[:, -2] / slope[:, -1]).median() - 1
    ),
    "floored_share": float((slope <= floor * (1 + 1e-9)).double().mean()),
    "outer_cell_mass": float(phi[1] - phi[0]),
  }
  del model
  torch.cuda.empty_cache()
  return row


def main() -> None:
  cap_gpu_memory(1.5)
  head, dirty = commit()
  rows = [probe(n, tr) for n in (200, 2000) for tr in ("logit", "probit")]
  frame = pd.DataFrame(rows).assign(
    commit=head,
    dirty=dirty,
    torch=torch.__version__,
    device=torch.cuda.get_device_name(0),
    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
  )
  OUT.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(OUT, index=False)
  pd.set_option("display.width", 250, "display.max_columns", 40)
  print(
    frame.drop(columns=["family", "tau_scenario", "rep", "seed"])
    .round(4)
    .to_string()
  )
  print(f"wrote {OUT}")


if __name__ == "__main__":
  main()
