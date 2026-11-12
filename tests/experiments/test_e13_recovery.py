"""E13: structure recovery, and the covariate-aware first tree."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pyvinecopulib as pv

_SCRIPT = (
  Path(__file__).resolve().parents[2]
  / "scripts"
  / "paper"
  / "e13"
  / "structure_recovery.py"
)


def _load_script() -> ModuleType:
  spec = importlib.util.spec_from_file_location("structure_recovery", _SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


e13 = _load_script()


def test_the_spanning_tree_is_the_maximum_one() -> None:
  """Prim's tree must maximize the weights, as Dissmann's does.

  A minimum spanning tree, or a tree that stops short of ``d - 1`` edges,
  would read as poor recovery for a reason that has nothing to do with the
  criterion being compared.
  """
  w = np.array(
    [
      [0.0, 0.9, 0.1, 0.2],
      [0.9, 0.0, 0.8, 0.1],
      [0.1, 0.8, 0.0, 0.7],
      [0.2, 0.1, 0.7, 0.0],
    ]
  )
  edges = e13.max_spanning_tree(w)
  pairs = {tuple(sorted(p)) for p, cond in edges}
  assert pairs == {(1, 2), (2, 3), (3, 4)}
  assert all(cond == frozenset() for _, cond in edges)


def test_binned_tau_sees_an_edge_whose_sign_flips_with_x() -> None:
  """A dependence that reverses halfway along ``x`` must not read as none.

  Pooled over ``x``, ``u2 = u1`` on one half and ``1 - u1`` on the other has
  a Kendall's tau near 0; within each bin it is +-1. Averaging the signed
  tau over bins, rather than its absolute value, would also return about 0
  and miss the very edge the binned criterion exists to find.
  """
  rng = np.random.default_rng(3)
  n = 1000
  x = np.linspace(0.0, 1.0, n)
  u1 = rng.uniform(size=n)
  u2 = np.where(x < 0.5, u1, 1.0 - u1)
  u3 = (0.3 * u1 + rng.uniform(size=n)) / 1.3
  u = np.column_stack([u1, u2, u3])
  pooled, binned = e13.abs_tau(u), e13.binned_abs_tau(u, x, 2)
  assert pooled[0, 1] < 0.1
  assert binned[0, 1] > 0.99
  tree = e13.max_spanning_tree(binned)
  assert (frozenset((1, 2)), frozenset()) in tree
  assert (frozenset((1, 2)), frozenset()) not in e13.max_spanning_tree(pooled)


def test_recovery_reads_every_tree_and_pads_the_rest() -> None:
  """A vine recovers itself in every tree; other vines do not.

  Reading the matrix's rows the wrong way round, or matching only conditioned
  pairs and not conditioning sets, would score a different D-vine on the same
  first tree as the truth. Trees beyond ``d - 1`` must stay ``nan``, not 0,
  so a mean over cells of different ``d`` never counts them as misses.
  """
  truth_matrix = np.asarray(pv.DVineStructure(order=[1, 2, 3, 4, 5]).matrix)
  truth = e13.tree_edges(truth_matrix)
  same = e13.recovery_row(truth_matrix, truth)
  assert [same[f"tree{t}"] for t in range(1, 5)] == [1.0] * 4
  assert same["overall"] == 1.0
  assert same["equal"]
  assert all(math.isnan(same[f"tree{t}"]) for t in range(5, 10))

  other = np.asarray(pv.DVineStructure(order=[2, 1, 3, 4, 5]).matrix)
  row = e13.recovery_row(other, truth)
  assert row["tree1"] == 0.75
  assert row["overall"] < 1.0
  assert not row["equal"]
