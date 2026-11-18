"""Check that reusing quantile tables changes no TabPFN vine density.

Each :class:`~npcc.core.ConditionalMargin` keeps its last quantile table and
returns it for a repeat read at the same conditioning rows, which a vine's
pair copulas make once per direction: the density, then the h-function. This
fits two vines with TabPFN v3 once each and evaluates each three times, with
reuse off, on and off again, timing every pass:

- E2's ``random`` arm on cell ``d = 5, n = 1000, rep 0`` (non-simplified, with
  the covariate), at the study's 5 x 1000 test points, joint and copula;
- E3d's ``npcc-tabpfn-quantiles-v3-probit-random-ns`` arm on one sample,
  ``decay_d5_s101`` by default (unconditional), at its 20,000 test points or
  the first ``--n-test``.

``--skip-e2`` runs the E3d part alone, as on another machine, where only a
same-machine comparison counts: two machines agree to about 1e-13, not
bitwise. ``--e3d-folder`` then names where E3d's ``samples.npz`` was copied.

The two passes without reuse show whether the backend is deterministic at
all; reuse is adopted only if the pass with it is bitwise equal to both.
Writes ``check.json`` into ``--out``.

Usage (from the npcc repository root, on a free GPU)::

    uv run --no-sync python scripts/paper/e10/check_table_reuse.py \\
        --out results/e10_table_reuse_<TAG> --device cuda
    uv run --no-sync python scripts/paper/e10/check_table_reuse.py \\
        --out <folder> --device cuda --skip-e2 --e3d-folder <copy> \\
        --e3d-key decay_d10_s101 --n-test 2000
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pyvinecopulib as pv
import torch
from pyvinecopulib.core import NonSimplifiedContext

from npcc.core.controls import FitControlsRosenblattVinecop
from npcc.core.margin import ConditionalMargin
from npcc.core.vinecop import RosenblattVinecop
from npcc.experiments import scenarios, vine_runner, vine_scenarios
from npcc.experiments._study import derive_seed
from npcc.experiments.vine_config import VineCell, load_vine_grid

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "paper" / "e3"))
import e3_common
import e3_npcc

#: E3d's samples, read only: results live in the main checkout.
E3D = e3_common.NPCC_REPO / "results" / "e3d_unknown_structure"
BASE_SEED = 317
LABEL = "tabpfn-quantiles-v3-probit"


@contextmanager
def reuse(on: bool) -> Iterator[None]:
  """Run a block with table reuse on, or with every table predicted afresh."""
  if on:
    yield
    return
  original = ConditionalMargin._predict_quantile_table

  def fresh(
    self: ConditionalMargin, x: torch.Tensor, **kwargs: object
  ) -> object:
    self._last_table = None
    return original(self, x, **kwargs)  # ty: ignore[invalid-argument-type] - forwards the caller's own keywords

  ConditionalMargin._predict_quantile_table = fresh  # ty: ignore[invalid-assignment] - a temporary wrapper with the same call shape
  try:
    yield
  finally:
    ConditionalMargin._predict_quantile_table = original


def forget_tables() -> None:
  """Drop every margin's kept table, so a pass starts as a fresh call does.

  A pass evaluates the same points as the one before it, so without this the
  tables the previous pass left behind would answer its first reads, and the
  timing would credit reuse with work it never saved within one evaluation.
  """
  for obj in gc.get_objects():
    if isinstance(obj, ConditionalMargin):
      obj._last_table = None


def three_passes(
  evaluate: Callable[[], dict[str, torch.Tensor]], device: str
) -> dict[str, object]:
  """Evaluate off, on, off; report bitwise equality and the seconds."""
  outs: list[dict[str, torch.Tensor]] = []
  seconds: list[float] = []
  for on in (False, True, False):
    forget_tables()
    with reuse(on), torch.inference_mode():
      if device == "cuda":
        torch.cuda.synchronize()
      t0 = perf_counter()
      out = evaluate()
      if device == "cuda":
        torch.cuda.synchronize()
      seconds.append(perf_counter() - t0)
    outs.append({k: v.detach().cpu() for k, v in out.items()})
  report: dict[str, object] = {
    "seconds_off_on_off": seconds,
    "speedup": (seconds[0] + seconds[2]) / 2 / seconds[1],
  }
  for key in outs[0]:
    off1, on, off2 = (o[key] for o in outs)
    report[key] = {
      "off_off_bitwise": bool(torch.equal(off1, off2)),
      "on_off_bitwise": bool(torch.equal(on, off1) and torch.equal(on, off2)),
      "max_abs_on_off": float((on - off1).abs().max()),
      "n": int(on.numel()),
    }
  return report


def e2_cell(device: str) -> dict[str, object]:
  grid = load_vine_grid(REPO / "configs" / "vine_study.toml")
  grid = replace(grid, arms=["random"])
  est = next(e for e in grid.estimators if e.label == LABEL)
  cell = VineCell(d=5, n=1000, rep=0)
  truth = vine_runner.truth_for(BASE_SEED, cell.d, cell.rep)
  y, _, x = vine_runner.training_sample(BASE_SEED, cell)
  points = vine_scenarios.eval_points(
    truth,
    scenarios.conditional_x_axis(grid.eval_x_n),
    grid.eval_m,
    derive_seed(BASE_SEED, "eval", cell.d, cell.rep),
  )
  cell_seed = derive_seed(BASE_SEED, "cell", cell.d, cell.n, cell.rep)
  torch.manual_seed(derive_seed(cell_seed, "random", est.estimator_id))
  dist, vine, _ = vine_runner._fit_npcc(
    "random",
    est,
    vine_runner.random_structure_for(BASE_SEED, cell.d, cell.rep),
    y,
    x,
    device=device,
    projection_grid_size=grid.projection_grid_size,
    batch_size=grid.batch_size,
  )
  x_all = points.x_axis.repeat_interleave(grid.eval_m).reshape(-1, 1)
  y_all, u_all = torch.cat(points.y), torch.cat(points.u)
  return three_passes(
    lambda: {
      "joint_pdf": dist.pdf(y_all, x=x_all),
      "copula_pdf": vine.pdf(u_all, x=x_all),
    },
    device,
  )


def e3d_cell(
  device: str, folder: Path, key: str, n_test: int | None
) -> dict[str, object]:
  design, dim, seed = key.split("_")
  d, sample = int(dim.removeprefix("d")), int(seed.removeprefix("s"))
  with np.load(folder / "samples.npz") as f:
    train, test = f[f"{key}_train"], f[f"{key}_test"]
    structure = pv.RVineStructure.from_matrix(f[f"{key}_random_matrix"])
  test = test[:n_test]
  est = e3_npcc.spec(LABEL)
  controls = est.controls(
    FitControlsRosenblattVinecop,
    device=device,
    projection_grid_size=e3_npcc.PROJECTION_GRID_SIZE,
    batch_size=None,
  )
  torch.manual_seed(
    derive_seed("E3d", design, d, sample, "random-ns", est.estimator_id)
  )
  vine = RosenblattVinecop(
    None, structure, device=device, context=NonSimplifiedContext()
  )
  vine.fit(torch.as_tensor(train, dtype=torch.float64), controls)
  test_t = torch.as_tensor(test, dtype=torch.float64)
  return three_passes(lambda: {"logpdf": vine.logpdf(test_t)}, device)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
  p.add_argument("--skip-e2", action="store_true")
  p.add_argument("--e3d-folder", type=Path, default=E3D)
  p.add_argument("--e3d-key", default="decay_d5_s101")
  p.add_argument("--n-test", type=int, default=None)
  a = p.parse_args()
  a.out.mkdir(parents=True, exist_ok=False)
  report: dict[str, object] = {}
  if not a.skip_e2:
    report["e2_d5_n1000_rep0_random"] = e2_cell(a.device)
    print(json.dumps(report, indent=2), flush=True)
  report[f"e3d_{a.e3d_key}_random_ns"] = {
    "n_test": a.n_test,
    **e3d_cell(a.device, a.e3d_folder, a.e3d_key, a.n_test),
  }
  (a.out / "check.json").write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
