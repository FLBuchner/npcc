"""How much TabPFN's chunk composition moves E1's metrics.

TabPFN v3's output for a conditioning row depends slightly on which other rows
share its forward pass (``bitwise_dedupe.py``: deterministic run to run, not
bitwise under a different chunking). This measures what that does to E1's
scores on two E1 cells, for one fitted model each, under three chunkings of
the same queries:

- ``e1_chunks2000``: every row, in E1's order, 2000 per pass (E1 on CUDA);
- ``dedup``: each distinct row once (commit 56d9983);
- ``e1_chunks500``: every row, in E1's order, 500 per pass.

Output: ``results/qt_ab/tabpfn_chunking.csv``, one row per cell and chunking
with pdf IAE and KL and hfunc1 / hfunc2 IAE, averaged as E1 averages them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import RESULTS, commit
from bitwise_dedupe import ESTIMATORS, table_before_56d9983

from npcc.core.bicop import RosenblattBicop
from npcc.core.controls import FitControlsRosenblattBicop
from npcc.core.margin import ConditionalMargin
from npcc.experiments import runner, scenarios
from npcc.experiments._study import derive_seed
from npcc.experiments.config import Cell, EstimatorSpec

CELLS = [Cell("clayton", "linear", 500, 0), Cell("gaussian", "sin", 2000, 1)]
LABEL = "tabpfn-quantiles-v3-logit"


def main() -> None:
  """Write the CSV, with the commit it came from."""
  torch.set_num_threads(2)
  head, dirty = commit()
  deduplicating = ConditionalMargin._predict_quantile_table
  backend, transform, kwargs = ESTIMATORS[LABEL]
  est = EstimatorSpec(
    label=LABEL, backend=backend, transform=transform, backend_kwargs=kwargs
  )
  rows = []
  for cell in CELLS:
    seed = runner._cell_seed(317, cell)
    u, v, x = scenarios.sample(cell.family, cell.tau_scenario, cell.n, seed)
    grid = scenarios.eval_grid(
      cell.tau_scenario, conditional_uv_grid_n=20, conditional_x_grid_n=10
    )
    truth = scenarios.ground_truth(cell.family, cell.tau_scenario, grid)
    torch.manual_seed(derive_seed(seed, est.estimator_id))
    model = RosenblattBicop(
      FitControlsRosenblattBicop(
        backend=backend,
        transform=transform,
        device="cuda",
        backend_kwargs=kwargs,
      )
    )
    model.fit(torch.column_stack([u, v]), x=x)
    margins = (model.v_given_ux_, model.u_given_vx_)
    for name, method, batch in (
      ("e1_chunks2000", table_before_56d9983, 2000),
      ("dedup", deduplicating, 2000),
      ("e1_chunks500", table_before_56d9983, 500),
    ):
      for mg in margins:
        mg.batch_size = batch
      try:
        ConditionalMargin._predict_quantile_table = method
        with torch.inference_mode():
          hfuncs, pdfs, _ = runner._predict(model, grid, [None])
      finally:
        ConditionalMargin._predict_quantile_table = deduplicating
      scored = {"pdf": pdfs["none"], **hfuncs}
      row: dict[str, object] = {
        "family": cell.family,
        "tau_scenario": cell.tau_scenario,
        "n": cell.n,
        "rep": cell.rep,
        "chunking": name,
      }
      for q, pred in scored.items():
        per_x = runner._metric_rows_for_quantity(
          cell,
          est,
          seed,
          q,
          truth[q],
          pred,
          grid,
          normalize="none",
          include_kl=q == "pdf",
        )
        frame = pd.DataFrame(per_x)
        row[f"{q}_IAE"] = float(frame.IAE.mean())
        if q == "pdf":
          row["pdf_KL"] = float(frame.KL.mean())
      rows.append(row)
    del model
    torch.cuda.empty_cache()
  out = RESULTS / "tabpfn_chunking.csv"
  frame = pd.DataFrame(rows).assign(commit=head, dirty=dirty)
  frame.to_csv(out, index=False)
  print(
    frame.drop(columns=["commit"]).to_string(index=False, float_format="%.6f")
  )
  print(f"wrote {out}")


if __name__ == "__main__":
  main()
