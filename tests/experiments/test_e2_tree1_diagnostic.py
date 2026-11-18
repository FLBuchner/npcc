"""E2's first-tree diagnostic: the edges it sums, its bound, the data it redraws."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import pyvinecopulib as pv

from npcc.experiments import vine_cli

_SCRIPT = (
  Path(__file__).resolve().parents[2]
  / "scripts"
  / "paper"
  / "e2"
  / "tree1_diagnostic.py"
)


def _load_script() -> ModuleType:
  spec = importlib.util.spec_from_file_location("e2_tree1", _SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


diagnostic = _load_script()


def _path_sample() -> tuple[pv.RVineStructure, np.ndarray]:
  """A D-vine on the path 1 - 3 - 2 with strong first-tree edges."""
  structure = pv.RVineStructure.from_order([1, 3, 2])
  strong = pv.Bicop(
    family=pv.BicopFamily.gaussian, parameters=np.array([[0.8]])
  )
  vine = pv.Vinecop.from_structure(
    structure=structure, pair_copulas=[[strong, strong], [pv.Bicop()]]
  )
  return structure, vine.sample(400, seeds=[3])


def test_edges_are_summed_on_their_own_columns() -> None:
  """pyvinecopulib labels variables from 1 and the data's columns from 0.

  An off-by-one between the two would sum the taus of pairs the truth does
  not join, and the comparison would say nothing about the selection.
  """
  structure, u = _path_sample()
  edges = diagnostic.tree1(np.asarray(structure.matrix).tolist())
  assert edges == {frozenset((1, 3)), frozenset((3, 2))}
  taus = diagnostic.abs_tau(u)
  expected = abs(pv.utils.wdm(u[:, 0], u[:, 2], "tau")) + abs(
    pv.utils.wdm(u[:, 2], u[:, 1], "tau")
  )
  assert sum(taus[p] for p in edges) == pytest.approx(expected)
  assert diagnostic.max_spanning_sum(taus, 3) == pytest.approx(expected)


def test_the_spanning_sum_bounds_every_first_tree() -> None:
  """The bound must be the largest first tree, or it bounds nothing.

  Dissmann's first tree maximizes the sum of ``|tau|``, so no structure's
  first tree may sum to more on the same data.
  """
  rng = np.random.default_rng(5)
  u = rng.random((300, 5))
  u[:, 1] = (u[:, 0] + 0.2 * u[:, 1]) / 1.2
  taus = diagnostic.abs_tau(u)
  bound = diagnostic.max_spanning_sum(taus, 5)
  for seed in range(30):
    other = pv.RVineStructure.sample(5, seeds=[seed])
    edges = diagnostic.tree1(np.asarray(other.matrix).tolist())
    assert sum(taus[p] for p in edges) <= bound + 1e-12
  selected = pv.Vinecop.from_data(
    u, controls=pv.FitControlsVinecop(family_set=[pv.BicopFamily.indep])
  ).structure
  edges = diagnostic.tree1(np.asarray(selected.matrix).tolist())
  assert sum(taus[p] for p in edges) == pytest.approx(bound)


_SELECTED = """
[grid]
dims = [4]
n = [40]
n_rep = 2
arms = ["selected"]
eval_x_n = 2
eval_m = 20

[[grid.estimators]]
label = "uni"
backend = "normal-quantile"
transform = "logit"
"""


def _selected_run(tmp_path: Path) -> Path:
  config = tmp_path / "vine.toml"
  config.write_text(_SELECTED)
  run = tmp_path / "run"
  argv = ["--config", str(config), "--out", str(run), "--device", "cpu"]
  assert vine_cli.main(argv) == 0
  return run


def test_every_selected_cell_is_diagnosed(
  register_hermetic_backends: None,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Each cell gets its sums, and neither may exceed the first-tree bound."""
  run = _selected_run(tmp_path)
  monkeypatch.setattr(sys, "argv", ["tree1_diagnostic.py", str(run)])
  diagnostic.main()
  cells = pd.read_csv(run / "analysis" / "tree1_diagnostic_cells.csv")
  assert sorted(cells["rep"]) == [0, 1]
  assert (cells["selected_sum"] <= cells["mst_sum"] + 1e-12).all()
  assert (cells["true_sum"] <= cells["mst_sum"] + 1e-12).all()
  [row] = pd.read_csv(run / "analysis" / "tree1_diagnostic.csv").to_dict(
    "records"
  )
  assert row["cells"] == 2
  assert row["tree1_edges"] == 3


def test_a_run_with_another_truth_is_refused(
  register_hermetic_backends: None, tmp_path: Path
) -> None:
  """Redrawn data must be the run's: another truth means other training data."""
  run = _selected_run(tmp_path)
  truths = json.loads((run / "truths.json").read_text())
  truths[0]["matrix"] = truths[1]["matrix"]
  (run / "truths.json").write_text(json.dumps(truths))
  with pytest.raises(SystemExit, match="truths.json"):
    diagnostic.cell_rows(run)
