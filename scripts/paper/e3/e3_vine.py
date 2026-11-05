"""One simplified vine for every pair-copula estimator of E3c.

Imported in the NPCC environment (pyvinecopulib 1.0) by ``e3c_npcc_fit.py``.

:class:`SharedVinecop` is pyvinecopulib 1.0's ``VinecopBase`` with nothing of
its own but storage: a fixed structure, a simplified context (every pair
copula unconditional), and any pair copula that answers in torch on the CPU
-- ``e3_vdc_pair.VdcBicop``, ``pyvinecopulib.torch.TorchTllBicop`` and
``npcc.RosenblattBicop`` alike. The cascades that fit the pairs tree by tree
(``VinecopBase.fit`` with a ``fit_edge`` callback) and that evaluate the
density (``VinecopBase.logpdf``) are the base's, so two arms differ only in
the pair copula each edge holds.

It mirrors :class:`npcc.RosenblattVinecop` -- the same placement of inputs
before the inherited fit, the same storage -- minus that class's check that
every pair is a ``RosenblattBicop``. Under a ``SimplifiedContext`` a
``RosenblattVinecop`` of ``RosenblattBicop`` pairs is the same model, which
``e3c_npcc_fit.py`` checks on its smallest cells.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Self

import numpy as np
import torch
from pyvinecopulib import RVineStructure
from pyvinecopulib.core import BicopLike, ControlsLike, SimplifiedContext, VinecopBase

from npcc.core._placement import TensorPlacement


class SharedVinecop(TensorPlacement, VinecopBase[torch.Tensor]):
  """A fixed-structure, simplified vine of any torch pair copulas.

  Parameters
  ----------
  structure : RVineStructure
      The true structure; nothing is selected.
  device : str
      Where inputs are placed; the pairs must answer there.
  """

  #: Every pair here is unconditional.
  supports_covariates: bool = False
  #: None of the pair estimators takes weights.
  supports_weights: bool = False
  #: No default pair class: each arm passes its own ``fit_edge``.
  bicop_class: ClassVar[None] = None

  def __init__(self, structure: RVineStructure, *, device: str = "cpu") -> None:
    self.pair_copulas: list[list[BicopLike[torch.Tensor]]] = []
    self._bind_vine(structure, SimplifiedContext(), var_types=None)
    self._set_placement(device)

  def fit(
    self,
    u: np.ndarray | torch.Tensor,
    /,
    controls: ControlsLike | None = None,
    *,
    var_types: list[str] | None = None,
    x: torch.Tensor | None = None,
    fit_edge: Callable[..., Any] | None = None,
    fit_level: Callable[..., Any] | None = None,
  ) -> Self:
    """Fit every pair along the structure, inputs placed as ``RosenblattVinecop`` does."""
    return super().fit(self._prep(u), controls, var_types=var_types,
                       x=None if x is None else self._prep(x),
                       fit_edge=fit_edge, fit_level=fit_level)

  def get_pair_copula(self, tree: int, edge: int) -> BicopLike[torch.Tensor]:
    """The pair copula at a tree and edge position."""
    return self.pair_copulas[tree][edge]

  def set_pair_copulas(self, pair_copulas: Sequence[Sequence[BicopLike[torch.Tensor]]]) -> None:
    """Install the pairs the inherited fit engine produced."""
    rows = [list(row) for row in pair_copulas]
    if len(rows) != self.trunc_lvl or any(len(r) != self.d - t - 1 for t, r in enumerate(rows)):
      raise ValueError(f"pair_copulas must have {self.trunc_lvl} trees of d - t - 1 edges")
    self.pair_copulas = rows
    self._invalidate_batched()

  def _sample_uniform(self, n: int, qrng: bool, seeds: list[int]) -> torch.Tensor:
    """Base uniforms for inherited sampling."""
    return self._draw_uniform((n, self.d), qrng=qrng, seeds=seeds)
