"""Compare fixed-structure conditional vines and chains on copula scores."""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from importlib.metadata import version
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Protocol, TypedDict

import numpy as np
import pandas as pd
import pyvinecopulib as pv
import torch
from pyvinecopulib.core import NonSimplifiedContext
from scripts.paper.e12.mass_common import (
  CELL_IS_COLUMNS,
  mark_cells,
  rows_pass,
  stats,
)

from npcc import FitControlsRosenblattVinecop, RosenblattVinecop
from npcc.experiments._study import (
  Rows,
  _shard,
  derive_seed,
  read_table,
  release_gpu,
  sweep,
)
from npcc.experiments.chain import ChainCopula, draw_orders
from npcc.experiments.config import RunConfig
from npcc.experiments.vine_runner import mixture_log_density

DIM = 10
ROOTS = 3
ORDER = list(range(DIM, 0, -1))
DESIGNS = ("stable", "reversal")
SIZES = (500, 1000, 2000)
TEST_SIZE = 5000
BACKEND_KWARGS = {
  "model_version": "v3",
  "model_kwargs": {"softmax_temperature": 1.0},
}
FAMILIES = (
  pv.BicopFamily.gaussian,
  pv.BicopFamily.clayton,
  pv.BicopFamily.gumbel,
)
COMPARISONS = (
  ("truncated_vine", "full_chain"),
  ("truncated_vine", "sparse_chain"),
  ("full_vine_mixture", "full_chain"),
  ("full_true_order", "truncated_vine"),
  ("full_true_order", "full_vine_mixture"),
)
GROUP = ["design", "n_train", "arm", "regime"]
TABLES = ("scores", "mass", "runtime")
Scalar = str | int | float | bool
Row = dict[str, Scalar]


class CellRecord(TypedDict):
  """Saved numerical results and provenance for one replicate."""

  design: str
  n_train: int
  seed: int
  train_seed: int
  test_seed: int
  truth_structure: list[list[int]]
  vine_structures: list[list[list[int]]]
  chain_orders: list[list[int]]
  settings: dict[str, object]
  versions: dict[str, str]
  hardware: dict[str, Scalar | None]
  scores: list[Row]
  mass: list[Row]
  runtime: list[Row]


@dataclass(frozen=True)
class Cell:
  """One design, sample size, and replicate seed in the sweep."""

  design: str
  n: int
  seed: int

  @property
  def key(self) -> str:
    """Return the directory name of the cell's checkpoint."""
    return f"{self.design}_n{self.n}_seed{self.seed}"


class SparseRootsChain(ChainCopula):
  """Restrict every factor's predecessors to the available roots."""

  def __init__(self, controls: FitControlsRosenblattVinecop) -> None:
    super().__init__(range(1, DIM + 1), controls)

  def _features(self, u: torch.Tensor, k: int, x: torch.Tensor) -> torch.Tensor:
    return super()._features(u, min(k, ROOTS), x)


@cache
def _truth(design: str, regime: int) -> pv.Vinecop:
  structure = pv.CVineStructure(order=ORDER, trunc_lvl=ROOTS)
  taus = (0.50, 0.35, 0.25) if regime == 1 else (0.40, 0.28, 0.20)
  if design == "reversal" and regime == -1:
    taus = (-taus[0], *taus[1:])
  pairs: list[list[pv.Bicop]] = []
  for tree, tau in enumerate(taus):
    root = tree + 1
    labels = structure.order[: DIM - root]
    if sorted(labels) != list(range(root + 1, DIM + 1)) or not np.all(
      structure.matrix[tree, : len(labels)] == root
    ):
      raise RuntimeError(f"Expected U{root} as root in tree {root}")
    level = []
    for label in labels:
      family = FAMILIES[(label - root - 1) % len(FAMILIES)]
      rotation = 90 if tau < 0 and family != pv.BicopFamily.gaussian else 0
      template = pv.Bicop(family=family, rotation=rotation)
      pair = pv.Bicop(
        family=family,
        rotation=rotation,
        parameters=template.tau_to_parameters(tau),
      )
      if not math.isclose(pair.tau, tau, abs_tol=1e-12):
        raise RuntimeError(f"Expected tau {tau} on edge ({root}, {label})")
      level.append(pair)
    pairs.append(level)
  return pv.Vinecop.from_structure(structure=structure, pair_copulas=pairs)


def _sample(
  truths: dict[int, pv.Vinecop], n: int, seed: int, *, balanced: bool
) -> tuple[torch.Tensor, torch.Tensor]:
  rng = np.random.default_rng(seed)
  if balanced:
    if n % 2:
      raise ValueError(f"Balanced test size must be even, got {n}")
    regimes = np.repeat([-1, 1], n // 2)
    rng.shuffle(regimes)
  else:
    regimes = rng.choice([-1, 1], size=n)
  u = np.empty((n, DIM))
  for regime, truth in truths.items():
    mask = regimes == regime
    if mask.any():
      u[mask] = truth.sample(
        int(mask.sum()), seeds=[derive_seed(seed, "regime", regime)]
      )
  # Repair exact floating-point endpoints while preserving interior draws.
  np.clip(u, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0), out=u)
  return torch.from_numpy(u), torch.from_numpy(regimes[:, None])


def _draws(seed: int) -> tuple[list[pv.RVineStructure], list[list[int]]]:
  structures = [pv.CVineStructure(order=ORDER, trunc_lvl=DIM - 1)]
  matrices = {structures[0].matrix.tobytes()}
  attempt = 0
  while len(structures) < 5:
    draw = pv.RVineStructure.sample(
      DIM, seeds=[derive_seed(seed, "structure", attempt)]
    )
    attempt += 1
    key = draw.matrix.tobytes()
    if key not in matrices:
      matrices.add(key)
      structures.append(draw)
  orders = draw_orders(DIM, [derive_seed(seed, "order", j) for j in range(5)])
  return structures, orders


@dataclass
class Prediction:
  logs: torch.Tensor
  fit_time: float
  prediction_time: float
  regressors: int
  pairs: int = 0


class LogDensity(Protocol):
  def logpdf(self, u: torch.Tensor, /, *, x: torch.Tensor) -> torch.Tensor: ...


def _predict(
  model: LogDensity,
  test: tuple[torch.Tensor, torch.Tensor],
  start: float,
  device: torch.device | str | None,
  regressors: int,
  pairs: int = 0,
) -> Prediction:
  u, x = test
  if device is not None and torch.device(device).type == "cuda":
    torch.cuda.synchronize(device)
  start_prediction = perf_counter()
  with torch.inference_mode():
    logs = model.logpdf(u, x=x).detach().cpu()
  return Prediction(
    logs,
    start_prediction - start,
    perf_counter() - start_prediction,
    regressors,
    pairs,
  )


def _vine(
  structure: pv.RVineStructure,
  controls: FitControlsRosenblattVinecop,
  train: tuple[torch.Tensor, torch.Tensor],
  test: tuple[torch.Tensor, torch.Tensor],
  seed: int,
  expected_pairs: list[int],
) -> Prediction:
  torch.manual_seed(seed)
  u, x = train
  start = perf_counter()
  model = RosenblattVinecop(
    None, structure, device=controls.device, context=NonSimplifiedContext()
  ).fit(u, controls, x=x)
  counts = [len(level) for level in model.pair_copulas]
  if counts != expected_pairs:
    raise RuntimeError(f"Expected {expected_pairs} pairs by tree, got {counts}")
  return _predict(
    model, test, start, controls.device, 2 * sum(counts), sum(counts)
  )


def _chain(
  model: ChainCopula,
  train: tuple[torch.Tensor, torch.Tensor],
  test: tuple[torch.Tensor, torch.Tensor],
  seed: int,
) -> Prediction:
  u, x = train
  start = perf_counter()
  model.fit(u, x=x, seeds=[derive_seed(seed, k) for k in range(1, DIM)])
  count = len(model.factors_)
  if count != DIM - 1:
    raise RuntimeError(f"Expected {DIM - 1} chain factors, got {count}")
  return _predict(model, test, start, model.controls.device, count)


def _mixture(members: list[Prediction]) -> Prediction:
  start = perf_counter()
  logs = mixture_log_density(torch.stack([member.logs for member in members]))
  return Prediction(
    logs,
    sum(member.fit_time for member in members),
    sum(member.prediction_time for member in members) + perf_counter() - start,
    sum(member.regressors for member in members),
    sum(member.pairs for member in members),
  )


def _regimes(x: torch.Tensor) -> tuple[tuple[str, torch.Tensor], ...]:
  return (
    ("overall", torch.ones_like(x, dtype=torch.bool)),
    ("-1", x == -1),
    ("+1", x == 1),
  )


def _scores(
  logs: torch.Tensor, truth: torch.Tensor, x: torch.Tensor
) -> list[Row]:
  rows: list[Row] = []
  for label, mask in _regimes(x):
    fitted, exact = logs[mask], truth[mask]
    rows.append(
      {
        "regime": label,
        "n_test": int(mask.sum()),
        "negative_log_score": float(-fitted.mean()),
        "truth_negative_log_score": float(-exact.mean()),
        "excess_nll": float((exact - fitted).mean()),
        "n_nonfinite": int((~torch.isfinite(fitted)).sum()),
      }
    )
  return rows


def _mass_rows(
  logs: torch.Tensor, truth: torch.Tensor, x: torch.Tensor
) -> list[Row]:
  rows: list[Row] = []
  for label, mask in _regimes(x):
    ratios = (logs[mask] - truth[mask]).numpy()
    valid = ~np.isnan(ratios) & ~np.isposinf(ratios)
    row: Row = {
      "regime": label,
      "n_test": ratios.size,
      "n_invalid": int((~valid).sum()),
      "ess_share": math.nan,
      **dict.fromkeys(CELL_IS_COLUMNS, math.nan),
    }
    # Zero fitted density is a valid zero importance weight.
    if valid.all() and np.isfinite(ratios).any():
      # Shift before E12's computation to keep the relative diagnostics stable.
      shift = float(ratios.max())
      diagnostics = stats(ratios - shift, n_blocks=min(20, ratios.size))
      diagnostics.pop("excess_nll")
      diagnostics.pop("n_finite")
      diagnostics["neglogmass"] -= shift
      with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for key in (
          "mass",
          "mass_se",
          "mass_mom20",
          "mass_trim_top0.1pct",
          "max_ratio",
        ):
          diagnostics[key] = (
            float(np.exp(np.log(diagnostics[key]) + shift))
            if diagnostics[key]
            else 0.0
          )
      row.update(diagnostics)
    rows.append(row)
  # Fixed allocation to the two regimes makes the pooled mean stratified.
  rows[0]["mass_se"] = 0.5 * math.hypot(
    float(rows[1]["mass_se"]), float(rows[2]["mass_se"])
  )
  return rows


def _hardware(device: torch.device) -> dict[str, Scalar | None]:
  processor = platform.processor()
  if not processor:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
      processor = next(
        (
          line.partition(":")[2].strip()
          for line in cpuinfo.read_text().splitlines()
          if line.startswith("model name")
        ),
        platform.machine(),
      )
  return {
    "device": str(device),
    "accelerator": torch.cuda.get_device_name(device)
    if device.type == "cuda"
    else None,
    "processor": processor or platform.machine(),
    "platform": platform.platform(),
    "torch_threads": torch.get_num_threads(),
  }


def _run_cell(
  design: str,
  n: int,
  seed: int,
  device: torch.device,
  test_size: int = TEST_SIZE,
) -> tuple[CellRecord, dict[str, np.ndarray]]:
  train_seed = derive_seed(seed, "train", design, n)
  test_seed = derive_seed(seed, "test", design)
  truths = {regime: _truth(design, regime) for regime in (-1, 1)}
  structures, orders = _draws(seed)
  train = _sample(truths, n, train_seed, balanced=False)
  test = _sample(truths, test_size, test_seed, balanced=True)
  u_test, x_test = test
  x = x_test[:, 0]
  exact = torch.empty(test_size, dtype=torch.float64)
  for regime, truth in truths.items():
    mask = x == regime
    exact[mask] = torch.from_numpy(np.log(truth.pdf(u_test[mask].numpy())))
  if not torch.isfinite(exact).all():
    raise RuntimeError("The exact test log-density must be finite")

  controls = FitControlsRosenblattVinecop(
    device=device,
    backend="tabpfn-quantiles",
    transform="probit",
    projection_grid_size=30,
    backend_kwargs=BACKEND_KWARGS,
  )
  predictions = {
    "truncated_vine": _vine(
      truths[1].structure,
      controls,
      train,
      test,
      derive_seed(seed, "true-fit"),
      [9, 8, 7],
    )
  }
  release_gpu()
  vines = []
  for j, structure in enumerate(structures):
    fit_seed = (
      derive_seed(seed, "true-fit")
      if j == 0
      else derive_seed(seed, "vine-fit", j)
    )
    vines.append(
      _vine(structure, controls, train, test, fit_seed, list(range(9, 0, -1)))
    )
    release_gpu()
  chains = []
  for j, order in enumerate(orders):
    chains.append(
      _chain(
        ChainCopula(order, controls),
        train,
        test,
        derive_seed(seed, "chain-fit", j),
      )
    )
    release_gpu()
  predictions.update(
    {
      "full_vine_mixture": _mixture(vines),
      "full_true_order": vines[0],
      "full_chain": _mixture(chains),
      "sparse_chain": _chain(
        SparseRootsChain(controls), train, test, derive_seed(seed, "sparse-fit")
      ),
    }
  )
  release_gpu()

  cell = {"design": design, "n_train": n, "seed": seed}
  score_rows: list[Row] = []
  mass_rows: list[Row] = []
  runtime: list[Row] = []
  models = predictions | {
    **{
      f"vine_member_{j}": member for j, member in enumerate(vines[1:], start=2)
    },
    **{f"chain_member_{j}": member for j, member in enumerate(chains, start=1)},
  }
  for name, result in models.items():
    key = cell | {"arm": name}
    if name in predictions:
      score_rows.extend(key | row for row in _scores(result.logs, exact, x))
    if result.pairs:
      mass_rows.extend(key | row for row in _mass_rows(result.logs, exact, x))
    runtime.append(
      key
      | {
        "fit_time": result.fit_time,
        "prediction_time": result.prediction_time,
        "regressors": result.regressors,
        "pairs": result.pairs,
      }
    )

  return (
    {
      "design": design,
      "n_train": n,
      "seed": seed,
      "train_seed": train_seed,
      "test_seed": test_seed,
      "truth_structure": truths[1].structure.matrix.tolist(),
      "vine_structures": [
        structure.matrix.tolist() for structure in structures
      ],
      "chain_orders": orders,
      "settings": {
        "backend": controls.backend,
        "transform": controls.transform,
        "feature_transform": controls.feature_transform,
        "projection_grid_size": controls.projection_grid_size,
        "backend_kwargs": controls.backend_kwargs,
      },
      "versions": {
        name: version(name)
        for name in ("numpy", "torch", "tabpfn", "pyvinecopulib")
      },
      "hardware": _hardware(device),
      "scores": score_rows,
      "mass": mass_rows,
      "runtime": runtime,
    },
    {"truth": exact.numpy(), "x": x.numpy()}
    | {name: result.logs.numpy() for name, result in models.items()},
  )


def _checkpoint_cell(cell: Cell, config: RunConfig) -> list[Rows]:
  record, logs = _run_cell(
    cell.design, cell.n, cell.seed, torch.device(config.device or "cpu")
  )
  path = config.out / "cells" / cell.key
  path.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(path / "log_densities.npz", allow_pickle=False, **logs)
  metadata = {key: value for key, value in record.items() if key not in TABLES}
  (path / "result.json").write_text(json.dumps(metadata, indent=2))
  return [record["scores"], record["mass"], record["runtime"]]


def _summarize(out: Path) -> None:
  cells = [mark.parent for mark in sorted((out / "cells").glob("*/DONE"))]
  if not cells:
    raise ValueError(f"No completed cells in {out}")
  scores, mass, runtime = (
    pd.concat(
      [read_table(cell / table, "csv") for cell in cells], ignore_index=True
    )
    for table in TABLES
  )
  scores.to_csv(out / "scores.csv", index=False)
  runtime.to_csv(out / "runtime.csv", index=False)
  mass = mark_cells(mass)
  mass.to_csv(out / "mass_cells.csv", index=False)
  mass_summary = (
    mass.groupby(GROUP)
    .agg(
      n_rep=("seed", "size"),
      n_pass=("estimable", "sum"),
      mass=("mass", "mean"),
      median_cell_se=("mass_se", "median"),
      min_ess_share=("ess_share", "min"),
    )
    .reset_index()
  )
  mass_summary["estimable"] = rows_pass(mass_summary.n_pass, mass_summary.n_rep)
  mass_summary.loc[~mass_summary.estimable, ["mass", "median_cell_se"]] = np.nan
  mass_summary.to_csv(out / "mass_summary.csv", index=False)

  summary = scores.groupby(GROUP).agg(
    n_rep=("seed", "size"),
    n_nonfinite=("n_nonfinite", "sum"),
    negative_log_score=("negative_log_score", _score_mean),
    truth_negative_log_score=("truth_negative_log_score", _score_mean),
    excess_nll=("excess_nll", _score_mean),
  )
  summary.to_csv(out / "summary.csv")

  index = ["design", "n_train", "seed", "regime"]
  wide = scores.pivot(index=index, columns="arm", values="negative_log_score")
  differences = pd.concat(
    [
      (wide[left] - wide[right])
      .rename("difference")
      .reset_index()
      .assign(left=left, right=right)
      for left, right in COMPARISONS
    ],
    ignore_index=True,
  )
  differences.to_csv(out / "paired_cells.csv", index=False)
  differences["finite_difference"] = differences.difference.where(
    np.isfinite(differences.difference)
  )
  paired = differences.groupby(
    ["design", "n_train", "regime", "left", "right"]
  ).agg(
    n_rep=("difference", "size"),
    n_finite=("finite_difference", "count"),
    mean_difference=("difference", "mean"),
    se=("finite_difference", "sem"),
  )
  paired.loc[paired.n_rep != paired.n_finite, ["mean_difference", "se"]] = (
    np.nan
  )
  paired.to_csv(out / "paired.csv")


def _score_mean(scores: pd.Series) -> float:
  """Keep infinite scores and propagate undefined scores across replicates."""
  return float(np.mean(scores.to_numpy()))


def main() -> None:
  """Run the full grid, one selected cell, or aggregate completed cells."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--design", choices=DESIGNS)
  parser.add_argument("--n", type=int, choices=SIZES)
  parser.add_argument("--seed", type=int)
  parser.add_argument(
    "--device", default="cuda" if torch.cuda.is_available() else "cpu"
  )
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument("--fit-only", action="store_true")
  mode.add_argument("--summarize-only", action="store_true")
  parser.add_argument(
    "--shard",
    type=_shard,
    metavar="I/K",
    help="Fit only shard I of K, with zero-based indexing.",
  )
  args = parser.parse_args()
  if args.shard is not None and (
    args.summarize_only
    or any(v is not None for v in (args.design, args.n, args.seed))
  ):
    parser.error(
      "--shard must select from the full grid. Use it without cell selectors or --summarize-only."
    )
  if not args.summarize_only:
    logging.basicConfig(level=logging.INFO)
    try:
      config = RunConfig(
        out=args.out, device=args.device, fmt="csv", shard=args.shard
      )
    except ValueError as exc:
      parser.error(str(exc))
    cells = [
      Cell(design, n, seed)
      for design, n, seed in product(
        (args.design,) if args.design else DESIGNS,
        (args.n,) if args.n else SIZES,
        (args.seed,) if args.seed is not None else range(20),
      )
    ]
    sweep(
      cells,
      lambda cell: _checkpoint_cell(cell, config),
      tables=TABLES,
      run=config,
      signature=sha256(Path(__file__).read_bytes()).hexdigest(),
      study="E14",
      resume=True,
    )
  if not args.fit_only and args.shard is None:
    _summarize(args.out)


if __name__ == "__main__":
  main()
