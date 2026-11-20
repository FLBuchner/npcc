"""Regression tests for the truncated conditional vine experiment."""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pandas as pd
import pytest
import pyvinecopulib as pv
import torch
from scripts.paper.e14 import run

from npcc import FitControlsRosenblattVinecop, TabPFNQuantileBackend
from npcc.experiments._study import guard_manifest


@pytest.mark.parametrize(
  ("design", "regime", "taus"),
  [
    ("stable", 1, (0.50, 0.35, 0.25)),
    ("stable", -1, (0.40, 0.28, 0.20)),
    ("reversal", 1, (0.50, 0.35, 0.25)),
    ("reversal", -1, (-0.40, 0.28, 0.20)),
  ],
)
def test_truth_edges(design: str, regime: int, taus: tuple[float, ...]) -> None:
  """Catch reversed edge labels, unsigned rotations, and extra truth trees."""
  truth = run._truth(design, regime)
  assert truth.order == list(range(10, 0, -1))
  assert truth.trunc_lvl == 3
  assert [len(tree) for tree in truth.pair_copulas] == [9, 8, 7]
  families = [
    pv.BicopFamily.gaussian,
    pv.BicopFamily.clayton,
    pv.BicopFamily.gumbel,
  ]
  for t, (tau, level) in enumerate(zip(taus, truth.pair_copulas, strict=True)):
    assert (truth.matrix[t, : len(level)] == t + 1).all()
    for j, pair in enumerate(reversed(level)):
      assert pair.family == families[j % 3]
      assert pair.tau == pytest.approx(tau)
      assert pair.rotation == (90 if tau < 0 and j % 3 else 0)


def test_sparse_features() -> None:
  """A leaf must never inherit an earlier leaf from the full chain."""
  model = run.SparseRootsChain(FitControlsRosenblattVinecop())
  u = torch.arange(30, dtype=torch.float64).reshape(3, 10) / 32
  x = torch.tensor([[-1.0], [1.0], [-1.0]], dtype=torch.float64)
  for k, width in enumerate([1, 2, 3, 3, 3, 3, 3, 3, 3], start=1):
    torch.testing.assert_close(
      model._features(u, k, x), torch.cat([u[:, :width], x], dim=1)
    )


def test_run_cell(
  patch_tabpfn: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  """Catch dropped fits, conditioning columns, or member-level predictions."""
  fit_model = TabPFNQuantileBackend._fit_model
  fit_inputs: list[torch.Tensor] = []

  def capture(
    self: TabPFNQuantileBackend, x: torch.Tensor, z: torch.Tensor
  ) -> None:
    fit_inputs.append(x.clone())
    fit_model(self, x, z)

  monkeypatch.setattr(TabPFNQuantileBackend, "_fit_model", capture)
  record, logs = run._run_cell("reversal", 32, 2, torch.device("cpu"), 40)
  assert len(fit_inputs) == 552
  assert all(features.shape[0] == 32 for features in fit_inputs)
  assert all(
    torch.equal(features[:, -1], fit_inputs[0][:, -1])
    for features in fit_inputs
  )
  assert [f.shape[1] for f in fit_inputs[:48]] == [2] * 18 + [3] * 16 + [4] * 14
  assert [f.shape[1] for f in fit_inputs[-9:]] == [2, 3] + [4] * 7
  assert (logs["x"] == 1).sum() == (logs["x"] == -1).sum() == 20

  runtime = {row["arm"]: row for row in record["runtime"]}
  for arm, count in {
    "truncated_vine": 48,
    "full_vine_mixture": 450,
    "full_chain": 45,
    "sparse_chain": 9,
  }.items():
    assert runtime[arm]["regressors"] == count
  assert runtime["truncated_vine"]["pairs"] == 24
  assert runtime["full_true_order"]["pairs"] == 45
  for j in range(2, 6):
    assert runtime[f"vine_member_{j}"]["pairs"] == 45

  for mixture, members in (
    (
      "full_vine_mixture",
      ["full_true_order"] + [f"vine_member_{j}" for j in range(2, 6)],
    ),
    ("full_chain", [f"chain_member_{j}" for j in range(1, 6)]),
  ):
    np.testing.assert_allclose(
      np.exp(logs[mixture]),
      np.exp([logs[name] for name in members]).mean(axis=0),
    )

  computed: list[int] = []

  def compute(
    design: str, n: int, seed: int, device: torch.device
  ) -> tuple[run.CellRecord, dict[str, np.ndarray]]:
    computed.append(seed)
    assert (design, n, device.type) == ("stable", 500, "cpu")
    cell = record.copy()
    cell["scores"] = [
      row
      | {
        "seed": seed,
        "negative_log_score": float(row["negative_log_score"])
        + (2 * seed if row["arm"] == "truncated_vine" else 0),
      }
      for row in record["scores"]
    ]
    return cell, logs

  def invoke(*options: str) -> None:
    monkeypatch.setattr(
      sys, "argv", ["e14", "--out", str(tmp_path), "--device", "cpu", *options]
    )
    run.main()

  monkeypatch.setattr(run, "_run_cell", compute)
  invoke("--shard", "0/120")
  invoke("--design", "stable", "--n", "500", "--seed", "1", "--fit-only")
  invoke("--shard", "1/120")
  invoke("--shard", "0/120")
  assert computed == [0, 1]
  assert len(list((tmp_path / "cells").glob("*/DONE"))) == 2
  assert not list(tmp_path.glob("*.csv"))
  incomplete = tmp_path / "cells" / "incomplete"
  incomplete.mkdir()
  (incomplete / "scores.csv").write_text("unfinished\n")
  invoke("--summarize-only")
  paired = pd.read_csv(tmp_path / "paired.csv")
  assert len(paired) == 15
  primary = paired[
    (paired.left == "truncated_vine") & (paired.right == "full_chain")
  ]
  assert primary.se.to_list() == pytest.approx([1.0] * 3)
  assert len(pd.read_csv(tmp_path / "mass_cells.csv")) == 42


def test_first_shards_share_manifest(tmp_path: Path) -> None:
  """Concurrent first shards must see one complete, consistent manifest."""
  ready = Barrier(2)

  def publish(signature: str) -> str | None:
    ready.wait(timeout=5)
    try:
      guard_manifest(tmp_path, signature, study="E14", resume=True)
    except ValueError:
      return None
    return signature

  with ThreadPoolExecutor(max_workers=2) as pool:
    accepted = [s for s in pool.map(publish, ["first", "second"]) if s]
  assert len(accepted) == 1
  guard_manifest(tmp_path, accepted[0], study="E14", resume=True)


def test_draws_reproducible() -> None:
  """Catch duplicate members or accidental dependence on unseeded RNGs."""
  structures, orders = run._draws(7)
  structures_again, orders_again = run._draws(7)
  assert len({structure.matrix.tobytes() for structure in structures}) == 5
  assert all(structure.trunc_lvl == 9 for structure in structures)
  for structure, again in zip(structures, structures_again, strict=True):
    np.testing.assert_array_equal(structure.matrix, again.matrix)
  assert orders == orders_again
  assert len({tuple(order) for order in orders}) == 5


def test_mass_by_regime() -> None:
  """Opposite conditional mass errors can cancel in the pooled diagnostic."""
  x = torch.tensor([-1, 1] * 20)
  truth = torch.zeros(40, dtype=torch.float64)
  logs = torch.where(x == 1, 1.5, 0.5).to(truth).log()
  rows = run._mass_rows(logs, truth, x)
  assert [row["mass"] for row in rows] == pytest.approx([1.0, 0.5, 1.5])
  assert [row["ess_share"] for row in rows] == pytest.approx([0.8, 1.0, 1.0])
