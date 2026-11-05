"""Vine-structure edges and their distance to a true structure, for E3d.

Imported in both environments (it needs only NumPy). An edge of a regular vine
is its conditioned pair and its conditioning set, both unordered, so two
structures agree on an edge whatever their matrix encodings or the
orientation of the pair.

**Reading a matrix.** pyvinecopulib's structure matrix ``M`` (``d x d``,
1-based labels, the order on the anti-diagonal): the edge in tree ``t``
(0-based) at position ``e`` has the conditioned pair ``{M[d-1-e, e],
M[t, e]}`` and the conditioning set ``{M[0, e], ..., M[t-1, e]}``.
:func:`dvine_edges` builds the same from a D-vine order, which is how VDC
reports its structure; :func:`self_check` confirms the two agree on
pyvinecopulib's own D-vines.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Edge = tuple[frozenset, frozenset]


def edges_from_matrix(matrix: np.ndarray) -> list[set[Edge]]:
  """The edges of each tree of a vine given by its pyvinecopulib matrix."""
  m = np.asarray(matrix, dtype=int)
  d = m.shape[0]
  trees: list[set[Edge]] = []
  for t in range(d - 1):
    tree: set[Edge] = set()
    for e in range(d - 1 - t):
      if m[t, e] == 0:  # a truncated vine stops here
        continue
      pair = frozenset((int(m[d - 1 - e, e]), int(m[t, e])))
      cond = frozenset(int(m[k, e]) for k in range(t))
      tree.add((pair, cond))
    trees.append(tree)
  return trees


def dvine_edges(order: Sequence[int]) -> list[set[Edge]]:
  """The edges of each tree of the D-vine on ``order`` (1-based labels)."""
  o = [int(v) for v in order]
  d = len(o)
  return [{(frozenset((o[i], o[i + t + 1])), frozenset(o[i + 1:i + t + 1]))
           for i in range(d - 1 - t)} for t in range(d - 1)]


def recovery(selected: list[set[Edge]], truth: list[set[Edge]]) -> dict:
  """Share of the true edges each tree of ``selected`` recovers.

  Returns
  -------
  dict
      ``per_tree``: the share for each tree (true edges recovered over true
      edges); ``overall``: over all trees; ``equal``: whether every tree's
      edge set is the truth's.
  """
  per_tree = []
  hit = total = 0
  for t, true_tree in enumerate(truth):
    sel = selected[t] if t < len(selected) else set()
    k = len(sel & true_tree)
    per_tree.append(k / len(true_tree))
    hit += k
    total += len(true_tree)
  equal = all((selected[t] if t < len(selected) else set()) == truth[t] for t in range(len(truth)))
  return {"per_tree": per_tree, "overall": hit / total, "equal": equal}


def self_check(pv) -> None:  # noqa: ANN001 - the pyvinecopulib module of the caller
  """Matrix reading against D-vine edges, on pyvinecopulib's own D-vines."""
  for order in ([3, 1, 4, 2, 5], [2, 5, 1, 4, 3, 6], list(range(1, 11))):
    structure = pv.DVineStructure(order=order)
    if edges_from_matrix(np.asarray(structure.matrix)) != dvine_edges(order):
      raise RuntimeError(f"structure edges must agree with the D-vine on {order}")
