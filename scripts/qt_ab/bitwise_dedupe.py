"""Is predicting each distinct conditioning row once (56d9983) bitwise neutral?

For each estimator, one model is fitted on an E1 cell (E1's settings and
seeds), and E1's scored quantities -- hfunc1, hfunc2 and pdf on the metric
grid, and the tau diagnostic's inverse h-function at every grid x -- are
computed twice on that same model: with the library's
``_predict_quantile_table``, which deduplicates rows, and with the version
before 56d9983, vendored below. Output: one row per estimator and quantity
with ``bitwise`` and the largest absolute difference, in
``results/qt_ab/bitwise_dedupe_<device>.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import BASE_SEED, RESULTS, commit

from npcc.core._quantile_table import QuantileTable
from npcc.core.bicop import RosenblattBicop
from npcc.core.controls import FitControlsRosenblattBicop
from npcc.core.margin import ConditionalMargin
from npcc.experiments import runner, scenarios
from npcc.experiments._study import derive_seed
from npcc.experiments.config import Cell, EstimatorSpec

ESTIMATORS = {
  "lightgbm-logit": ("lightgbm", "logit", {}),
  "lightgbm-probit": ("lightgbm", "probit", {}),
  "xgboost-logit": ("xgboost", "logit", {}),
  "xgboost-probit": ("xgboost", "probit", {}),
  "catboost-logit": ("catboost", "logit", {}),
  "catboost-probit": ("catboost", "probit", {}),
  "tabpfn-quantiles-v3-logit": (
    "tabpfn-quantiles",
    "logit",
    {"model_version": "v3"},
  ),
}


def table_before_56d9983(
  self: ConditionalMargin, x: torch.Tensor, *, batch_size: int | None = None
) -> QuantileTable:
  """``_predict_quantile_table`` as of d18480b: every row, in order."""
  self._check_fitted()
  effective_batch_size = self._resolve_batch_size(batch_size)
  cfg = self.quantile_table_config
  alphas = cfg.alphas(device=x.device, dtype=x.dtype)
  n_alphas = alphas.shape[0]
  n_obs = x.shape[0]
  parts: list[torch.Tensor] = []
  for start in range(0, n_obs, effective_batch_size):
    end = min(start + effective_batch_size, n_obs)
    quantiles = self._predict_quantiles(x[start:end], alphas)
    if quantiles.shape != (end - start, n_alphas):
      raise RuntimeError("bad shape")
    parts.append(quantiles)
  table = (
    torch.cat(parts, dim=0)
    if parts
    else torch.empty((0, n_alphas), dtype=torch.float64, device=self._device)
  )
  return QuantileTable(
    table,
    limit=cfg.normal_limit,
    transform=self._response_transform,
    eps=self.eps,
    smoothing=cfg.smoothing,
    min_slope=cfg.min_slope,
  )


def quantities(
  model: RosenblattBicop, grid: scenarios.EvalGrid
) -> dict[str, torch.Tensor]:
  """E1's scored quantities and the tau diagnostic's samples."""
  with torch.inference_mode():
    hfuncs, pdfs, _ = runner._predict(model, grid, [None])
    out = {**hfuncs, "pdf": pdfs["none"]}
    assert grid.x_axis is not None
    out["tau"] = torch.tensor(
      [
        model.tau(
          x_row=torch.tensor([[float(xv)]], dtype=torch.float64), n=1000
        )
        for xv in grid.x_axis
      ]
    )
  return {k: v.detach().cpu() for k, v in out.items()}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--labels", nargs="+", default=list(ESTIMATORS)[:6])
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--n", type=int, default=500)
  args = parser.parse_args()
  torch.set_num_threads(4)
  head, dirty = commit()
  cell = Cell("clayton", "linear", args.n, 0)
  seed = runner._cell_seed(BASE_SEED, cell)
  u, v, x = scenarios.sample(cell.family, cell.tau_scenario, cell.n, seed)
  grid = scenarios.eval_grid(
    cell.tau_scenario, conditional_uv_grid_n=20, conditional_x_grid_n=10
  )
  deduplicating = ConditionalMargin._predict_quantile_table
  rows = []
  for label in args.labels:
    backend, transform, kwargs = ESTIMATORS[label]
    est = EstimatorSpec(
      label=label, backend=backend, transform=transform, backend_kwargs=kwargs
    )
    torch.manual_seed(derive_seed(seed, est.estimator_id))
    model = RosenblattBicop(
      FitControlsRosenblattBicop(
        backend=backend,
        transform=transform,
        device=args.device,
        projection_grid_size=30,
        backend_kwargs=kwargs,
      )
    )
    model.fit(torch.column_stack([u, v]), x=x)
    try:
      ConditionalMargin._predict_quantile_table = table_before_56d9983
      before = quantities(model, grid)
      # The same code twice: what the backend itself varies from run to run,
      # against which a difference from deduplication is judged.
      repeat = quantities(model, grid)
    finally:
      ConditionalMargin._predict_quantile_table = deduplicating
    after = quantities(model, grid)

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
      return float(((a - b).abs() / b.abs().clamp_min(1e-12)).max())

    for q in before:
      rows.append(
        {
          "label": label,
          "quantity": q,
          "n": cell.n,
          "bitwise": bool(torch.equal(before[q], after[q])),
          "max_abs_diff": float((before[q] - after[q]).abs().max()),
          "max_rel_diff": rel(after[q], before[q]),
          "bitwise_repeat": bool(torch.equal(before[q], repeat[q])),
          "max_abs_diff_repeat": float((before[q] - repeat[q]).abs().max()),
          "max_rel_diff_repeat": rel(repeat[q], before[q]),
        }
      )
    print(pd.DataFrame(rows[-len(before) :]).to_string(index=False), flush=True)
    del model
  frame = pd.DataFrame(rows).assign(
    commit=head, dirty=dirty, device=args.device
  )
  out = RESULTS / f"bitwise_dedupe_{args.device}.csv"
  out.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(out, index=False)
  print(f"wrote {out}")


if __name__ == "__main__":
  main()
