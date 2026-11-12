"""E11 on E3d: one cell of chain mixtures, and its comparison with the vines."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

from npcc.experiments._study import derive_seed
from npcc.experiments.chain import ChainCopula, draw_order
from npcc.experiments.config import EstimatorSpec

_PAPER = Path(__file__).resolve().parents[2] / "scripts" / "paper"


def _load_script(name: str, path: Path) -> ModuleType:
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


chain_script = _load_script("e3d_chain", _PAPER / "e3" / "e3d_chain.py")
compare_script = _load_script(
  "compare_e3d_chain", _PAPER / "e11" / "compare_e3d_chain.py"
)

KEY = "decay_d3_s101"


def _cell() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """A small Gaussian-copula cell: training and test scores, true log c."""
  rng = np.random.default_rng(5)
  corr = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]])
  chol = np.linalg.cholesky(corr)
  z_tr = rng.standard_normal((120, 3)) @ chol.T
  z_te = rng.standard_normal((80, 3)) @ chol.T
  inv = np.linalg.inv(corr)
  quad = np.einsum("ni,ij,nj->n", z_te, inv - np.eye(3), z_te)
  logtrue = -0.5 * np.log(np.linalg.det(corr)) - 0.5 * quad
  ndtr = torch.special.ndtr
  tr = ndtr(torch.as_tensor(z_tr)).numpy()
  te = ndtr(torch.as_tensor(z_te)).numpy()
  return tr, te, logtrue


def _est() -> EstimatorSpec:
  return EstimatorSpec(
    label="lin", backend="linear-gaussian", transform="probit"
  )


def test_a_cell_mixes_its_seeded_orders(
  register_linear_gaussian_backend: None, tmp_path: Path
) -> None:
  """Every order and factor seed of a cell must come from its key.

  The orders are a function of (design, d, seed) alone, so every estimator
  and machine mixes the same ones; each factor's torch seed adds the
  estimator. The orders' saved log-densities must be the ones a chain
  fitted with those seeds gives, and the mixture must score at most their
  mean, as Jensen's inequality bounds it.
  """
  tr, te, logtrue = _cell()
  est = _est()
  cell_dir = tmp_path / "cells" / "chain-lin"
  row = chain_script.run_cell(
    est,
    KEY,
    tr,
    te,
    logtrue,
    k=2,
    device="cpu",
    cell_dir=cell_dir,
    extra={"note": "test"},
  )
  orders = [
    draw_order(3, derive_seed("E3d", "chain", "decay", 3, 101, j))
    for j in (1, 2)
  ]
  assert json.loads(row["orders"]) == orders
  seeds = [
    [
      derive_seed("E3d", "chain", "decay", 3, 101, j, f, est.estimator_id)
      for f in (2, 3)
    ]
    for j in (1, 2)
  ]
  assert json.loads(row["member_factor_seeds"]) == seeds
  assert row["note"] == "test"
  assert json.loads((cell_dir / f"{KEY}.json").read_text()) == row
  members = np.load(cell_dir / f"{KEY}_members.npy")
  assert members.shape == (2, te.shape[0])
  for j, (order, these) in enumerate(zip(orders, seeds, strict=True)):
    controls = chain_script.controls(est, "cpu")
    direct = ChainCopula(order, controls).fit(torch.as_tensor(tr), seeds=these)
    expected = direct.logpdf(torch.as_tensor(te)).numpy()
    np.testing.assert_allclose(members[j], expected, rtol=1e-5)
  assert row["excess_nll"] <= row["members_mean_excess_nll_common"] + 1e-12
  assert math.isfinite(row["excess_nll"])
  assert row["n_factors"] == 4


def test_a_cell_resumes_at_its_next_order(
  register_linear_gaussian_backend: None,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A finished order is read back, not refitted, and must be the same order.

  An interrupted d = 10 cell on the GPU would otherwise refit every order;
  and an order saved under another draw would mix a vine of other orders
  than the row records.
  """
  tr, te, logtrue = _cell()
  cell_dir = tmp_path / "cells" / "chain-lin"
  first = chain_script.run_cell(
    _est(), KEY, tr, te, logtrue, k=2, device="cpu", cell_dir=cell_dir, extra={}
  )

  def refuse(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("a finished order was refitted")

  monkeypatch.setattr(chain_script, "fit_order", refuse)
  again = chain_script.run_cell(
    _est(), KEY, tr, te, logtrue, k=2, device="cpu", cell_dir=cell_dir, extra={}
  )
  assert again["excess_nll"] == first["excess_nll"]

  saved = cell_dir / "members" / f"{KEY}_m2.json"
  member = json.loads(saved.read_text())
  member["order"] = member["order"][::-1]
  saved.write_text(json.dumps(member))
  with pytest.raises(SystemExit, match="must hold the order"):
    chain_script.run_cell(
      _est(),
      KEY,
      tr,
      te,
      logtrue,
      k=2,
      device="cpu",
      cell_dir=cell_dir,
      extra={},
    )


def test_the_comparison_reads_the_chain_against_its_orders_and_the_vines(
  tmp_path: Path,
) -> None:
  """Each reference must be scored on the points it shares with the chain.

  A reference with a point the chain lacks, or the other way round, would
  otherwise mix a point-set difference into the comparison; and the best
  order must be read on the points the mixture and every order share.
  """
  rng = np.random.default_rng(1)
  logtrue = rng.normal(size=200)
  orders = np.stack([logtrue - 0.3, logtrue - 0.1])
  orders[0, 3] = np.nan
  mixture = np.log(np.mean(np.exp(orders), axis=0))
  folder = tmp_path / "chain"
  folder.mkdir()
  np.save(folder / f"{KEY}.npy", mixture.astype(np.float32))
  np.save(folder / f"{KEY}_members.npy", orders.astype(np.float32))
  label = "x"
  vine = tmp_path / "e3d" / "cells" / f"npcc-{label}-tll-ns"
  vine.mkdir(parents=True)
  ref = logtrue - 0.2
  ref[7] = np.nan
  np.save(vine / f"{KEY}.npy", ref.astype(np.float32))
  e10 = tmp_path / "e10" / "cells" / f"npcc-{label}"
  e10.mkdir(parents=True)
  np.save(e10 / f"{KEY}_members.npy", np.stack([logtrue - 0.4]))

  refs = compare_script.references(
    KEY, label, e3d=tmp_path / "e3d", e10=tmp_path / "e10", e3e=tmp_path
  )
  assert set(refs) == {f"e3d:npcc-{label}-tll-ns", f"e10:npcc-{label}:m1"}
  rows = {
    r["reference"]: r
    for r in compare_script.cell_rows(KEY, folder, logtrue, refs)
  }
  assert rows["orders_mean"]["n_common"] == 199
  assert rows["best_order"]["best_order"] == 2
  assert rows["best_order"]["ref"] == pytest.approx(0.1, abs=1e-6)
  assert rows[f"e3d:npcc-{label}-tll-ns"]["n_common"] == 198
  assert rows[f"e3d:npcc-{label}-tll-ns"]["ref"] == pytest.approx(0.2, abs=1e-6)
  assert rows[f"e10:npcc-{label}:m1"]["ref"] == pytest.approx(0.4, abs=1e-6)
  assert rows["best_order"]["mixture"] <= rows["orders_mean"]["ref"]
