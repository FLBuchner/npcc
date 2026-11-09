"""What the read-out A/B scripts share: E1's cells, scoring and provenance.

- the cells and settings of E1 (``base_seed = 317``, a 20 x 20 uv grid at 10
  x slices, a 1000-draw tau diagnostic), and the three level grids compared;
- :class:`Store`, which records the raw quantile rows a margin is asked for
  and replays them, so that every read-out is scored on one fitted model;
- :func:`evaluate`, E1's own metric and diagnostic rows plus a held-out
  log-likelihood;
- :func:`commit`, the commit an output was produced from, and
  :func:`cap_gpu_memory`, which keeps a run on a shared GPU from pushing
  another process out of memory.
"""

from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import torch

import npcc
from npcc.core.bicop import RosenblattBicop
from npcc.experiments import runner, scenarios
from npcc.experiments._study import derive_seed
from npcc.experiments.config import Cell, EstimatorSpec

ROOT = Path(__file__).resolve().parents[2]
# Outputs are stamped with this checkout's commit, so the package that ran
# must be this checkout's: an interpreter from another checkout imports its
# own unless PYTHONPATH puts this one's src first.
if not Path(npcc.__file__).resolve().is_relative_to(ROOT):
  raise SystemExit(
    f"npcc must be imported from {ROOT / 'src'}; got {npcc.__file__}. "
    f"Run with PYTHONPATH={ROOT / 'src'}."
  )
#: Where the single checks write their CSV; ``QT_AB_RESULTS`` moves it, so that
#: a stamped rerun goes to a fresh folder.
RESULTS = Path(os.environ.get("QT_AB_RESULTS", ROOT / "results" / "qt_ab"))

BASE_SEED = 317
UV_GRID_N, X_GRID_N, TAU_N = 20, 10, 1000
HOLDOUT_N = 2000
#: The levels every recording is made on, unless the fit itself depends on
#: the level set: 81 equally spaced scores on [-3.25, 3.25] hold each grid.
REC_LEVELS, REC_LIMIT = 81, 3.25
GRIDS: dict[str, tuple[int, float]] = {
  "m21": (21, 3.25),
  "m19": (19, 2.925),
  "m41": (41, 3.25),
}


def cells(ns: list[int], reps: list[int]) -> list[Cell]:
  """E1's conditional cells at the given sample sizes and reps."""
  return [
    Cell(family=f, tau_scenario=s, n=n, rep=r)
    for f in ("clayton", "gaussian")
    for s in ("linear", "sin")
    for n in ns
    for r in reps
  ]


def commit() -> tuple[str, bool]:
  """The checked-out commit, and whether tracked files differ from it."""
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


def cap_gpu_memory(gb: float) -> None:
  """Make this process fail on its own allocations past ``gb`` GiB."""
  if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb / total))


Predict = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""A margin's ``_predict_quantiles``: conditioning rows, levels -> table."""


class Store:
  """Raw quantile rows a margin was asked for, at the recording levels."""

  def __init__(self, scores: torch.Tensor) -> None:
    self.scores = scores
    self._rows: list[torch.Tensor] = []
    self._tables: list[torch.Tensor] = []
    self.rows: torch.Tensor | None = None
    self.table: torch.Tensor | None = None

  def record(self, predict: Predict) -> Predict:
    """Wrap a margin's ``_predict_quantiles`` to keep every row it predicts."""

    def recording(x: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
      distinct, inverse = torch.unique(x, dim=0, return_inverse=True)
      table = predict(distinct, alphas)
      self._rows.append(distinct.detach().cpu())
      self._tables.append(table.detach().cpu())
      return table[inverse]

    return recording

  def finalize(self) -> None:
    """Collapse the recordings to one row per distinct conditioning row."""
    rows = torch.cat(self._rows)
    table = torch.cat(self._tables)
    uniq, inverse = torch.unique(rows, dim=0, return_inverse=True)
    first = torch.full((uniq.shape[0],), -1, dtype=torch.long)
    order = torch.arange(rows.shape[0])
    first.scatter_reduce_(0, inverse, order, reduce="amin", include_self=False)
    self.rows, self.table = uniq, table[first]
    self._rows, self._tables = [], []

  def columns(self, alphas: torch.Tensor) -> torch.Tensor:
    """Recorded columns holding the levels ``alphas``."""
    t = torch.special.ndtri(alphas)
    step = float(self.scores[1] - self.scores[0])
    idx = torch.round((t - self.scores[0]) / step).long()
    inside = (idx >= 0) & (idx < self.scores.shape[0])
    clamped = idx.clamp(0, self.scores.shape[0] - 1)
    if not bool(inside.all()) or (self.scores[clamped] - t).abs().max() > 1e-6:
      raise ValueError("requested levels are not on the recorded grid")
    return idx

  def replay(self, x: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    """The recorded quantiles of the rows ``x`` at the levels ``alphas``."""
    assert self.rows is not None and self.table is not None
    host_x, host_alphas = x.cpu(), alphas.cpu()
    n_store = self.rows.shape[0]
    _, inverse = torch.unique(
      torch.cat([self.rows, host_x]), dim=0, return_inverse=True
    )
    group_to_store = torch.full((int(inverse.max()) + 1,), -1, dtype=torch.long)
    group_to_store[inverse[:n_store]] = torch.arange(n_store)
    idx = group_to_store[inverse[n_store:]]
    if (idx < 0).any():
      raise KeyError(f"{int((idx < 0).sum())} rows were not recorded")
    return self.table[idx][:, self.columns(host_alphas)].to(x.device)


def holdout(
  cell: Cell, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """A fresh sample from the cell's true copula and its true log density."""
  u, v, x = scenarios.sample(
    cell.family, cell.tau_scenario, HOLDOUT_N, derive_seed(seed, "holdout")
  )
  assert x is not None
  fam = scenarios.FAMILIES[cell.family]
  tau_of_x = scenarios.TAU_SCENARIOS[cell.tau_scenario].tau_of_x
  assert tau_of_x is not None
  uv = torch.column_stack([u, v])
  log_true = torch.tensor(
    [
      math.log(scenarios.bicop(fam, float(t)).pdf(uv[i : i + 1].numpy())[0])
      for i, t in enumerate(tau_of_x(x))
    ],
    dtype=torch.float64,
  )
  return uv, x.reshape(-1, 1), log_true


def evaluate(
  model: RosenblattBicop,
  cell: Cell,
  est: EstimatorSpec,
  seed: int,
  grid: scenarios.EvalGrid,
  truth: dict[str, torch.Tensor],
  ho: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
  tags: dict[str, object],
) -> list[dict[str, object]]:
  """E1's metric and diagnostic rows, plus held-out log-likelihood."""
  rows: list[dict[str, object]] = []
  t0 = perf_counter()
  with torch.inference_mode():
    hfuncs, pdfs, _ = runner._predict(model, grid, [None])
    for q, pred in hfuncs.items():
      rows += runner._metric_rows_for_quantity(
        cell, est, seed, q, truth[q], pred, grid,
        normalize="none", include_kl=False,
      )  # fmt: skip
    rows += runner._metric_rows_for_quantity(
      cell, est, seed, "pdf", truth["pdf"], pdfs["none"], grid,
      normalize="none", include_kl=True,
    )  # fmt: skip
    for r in rows:
      r["kind"] = "metric"
    diag, _ = runner._diagnostic_rows(
      cell, est, seed, model, pdfs, grid,
      enable_tau_diagnostics=True, tau_diagnostic_n=TAU_N,
    )  # fmt: skip
    for r in diag:
      r["kind"] = "diag"
    rows += diag
    uv, x, log_true = ho
    log_hat = torch.log(model.pdf(uv, x=x)).cpu()
    finite = torch.isfinite(log_hat)
    rows.append(
      {
        **runner._base_row(cell, est, seed),
        "kind": "holdout",
        "loglik": float(log_hat[finite].mean()),
        "holdout_kl": float((log_true - log_hat)[finite].mean()),
        "n_nonfinite": int((~finite).sum()),
      }
    )
  rows.append(
    {
      **runner._base_row(cell, est, seed),
      "kind": "time",
      "eval_s": perf_counter() - t0,
    }
  )
  for r in rows:
    r.update(tags)
  return rows
