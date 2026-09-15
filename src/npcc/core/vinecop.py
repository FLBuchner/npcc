"""Fixed-structure non-simplified vines built from Rosenblatt pair copulas."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Self

import numpy as np
import torch
from pyvinecopulib import RVineStructure
from pyvinecopulib.core import (
  BicopLike,
  ControlsLike,
  NonSimplifiedContext,
  VinecopBase,
)

from npcc.core._placement import TensorPlacement, resolve_device
from npcc.core.bicop import RosenblattBicop, _reject_discrete


class RosenblattVinecop(TensorPlacement, VinecopBase[torch.Tensor]):
  """Fixed-structure non-simplified vine of Rosenblatt pair copulas.

  Each edge contains a fitted :class:`RosenblattBicop`. Higher-tree edges
  receive their vine conditioning-set values followed by any external
  covariates.

  Parameters
  ----------
  pair_copulas
    Fitted pair copulas arranged as ``[tree][edge]``. ``None`` creates an unfitted
    vine whose pairs can subsequently be estimated with :meth:`fit`.
  structure
    R-vine structure defining the edge layout and order.
  var_types
    Variable types in variable order. Only continuous variables are currently
    supported by :class:`RosenblattBicop`.
  device
    Device this vine evaluates on: every input is placed onto it, and it is
    where sampling draws. For a fitted vine, omitting it adopts the pair
    copulas' common device and supplying a different one is refused. For an
    unfitted vine there are no pairs to read, so it resolves the same way the
    fit controls do -- CUDA when available, CPU otherwise.

  Notes
  -----
  Public numerical methods accept and return torch tensors. Quasi-random
  sampling uses pyvinecopulib's NumPy-based sampling utility internally and
  converts its result to torch at that boundary.
  """

  supports_covariates: bool = True
  bicop_class: ClassVar[type[RosenblattBicop]] = RosenblattBicop

  def __init__(
    self,
    pair_copulas: list[list[RosenblattBicop]] | None,
    structure: RVineStructure,
    *,
    var_types: list[str] | None = None,
    device: str | torch.device | None = None,
  ) -> None:
    # At construction rather than per edge: every pair in this vine is a
    # `RosenblattBicop`, which models no atoms, so the vine can say so once
    # instead of letting each edge discover it partway through a fit.
    _reject_discrete(var_types)

    self.pair_copulas: list[list[RosenblattBicop]] = []

    self._bind_vine(
      structure,
      NonSimplifiedContext(),
      var_types=var_types,
    )

    if pair_copulas is None:
      self._set_placement(device)
      return

    checked_pairs = self._check_pair_copulas(pair_copulas)
    self._validate_pair_copulas(checked_pairs)

    self.pair_copulas = checked_pairs
    self._set_placement(self._resolve_vine_device(checked_pairs, device=device))

  @staticmethod
  def _check_pair_copulas(
    pair_copulas: Sequence[Sequence[BicopLike[torch.Tensor]]],
  ) -> list[list[RosenblattBicop]]:
    """Check every pair's implementation and rebuild the nested containers.

    The pairs themselves are shared, not copied: a caller handing in fitted
    pair copulas keeps them, and the fit engine's output has no other owner.
    Only the lists are this vine's own, so a later mutation of the caller's
    sequence cannot reshape the vine.
    """
    checked: list[list[RosenblattBicop]] = []

    for row in pair_copulas:
      checked_row: list[RosenblattBicop] = []

      for pair in row:
        if not isinstance(pair, RosenblattBicop):
          raise TypeError(
            f"RosenblattVinecop hosts RosenblattBicop pairs; got "
            f"{type(pair).__name__}."
          )

        checked_row.append(pair)

      checked.append(checked_row)

    return checked

  def _validate_pair_copulas(
    self,
    pair_copulas: list[list[RosenblattBicop]],
  ) -> None:
    """Validate the tree and edge dimensions of a pair-copula matrix."""
    if len(pair_copulas) != self.trunc_lvl:
      raise ValueError(
        f"pair_copulas has {len(pair_copulas)} trees, "
        f"expected {self.trunc_lvl}."
      )

    for tree, row in enumerate(pair_copulas):
      expected_edges = self.d - tree - 1

      if len(row) != expected_edges:
        raise ValueError(
          f"pair_copulas tree {tree} has {len(row)} edges, "
          f"expected {expected_edges}."
        )

  def _resolve_vine_device(
    self,
    pair_copulas: list[list[RosenblattBicop]],
    *,
    device: str | torch.device | None = None,
  ) -> torch.device:
    """Infer and validate the common pair-copula device."""
    pair_devices = {pair._device for row in pair_copulas for pair in row}

    if len(pair_devices) > 1:
      devices = ", ".join(
        sorted(str(pair_device) for pair_device in pair_devices)
      )
      raise ValueError(
        f"All pair copulas must use the same device; found: {devices}."
      )

    requested_device = None if device is None else resolve_device(device)

    if pair_devices:
      pair_device = next(iter(pair_devices))

      if requested_device is not None and requested_device != pair_device:
        raise ValueError(
          f"Requested vine device {requested_device} does not match "
          f"pair-copula device {pair_device}."
        )

      return pair_device

    return (
      requested_device if requested_device is not None else resolve_device(None)
    )

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
    """Fit every pair along the fixed structure, on this vine's placement.

    Overridden only to place the inputs. The inherited implementation hands
    them to the fit engine untouched -- which is right for a vine whose pairs
    answer in whatever namespace they were given, and wrong here: the engine
    allocates its per-tree scratch in the namespace of the ``u`` it received,
    while a :class:`~npcc.core.bicop.RosenblattBicop` always answers in torch
    on its own device. Handed a NumPy ``u``, the cascade would try to assign a
    CUDA tensor into a NumPy row.

    Parameters
    ----------
    u : array, shape (n, d), dtype float
        Pseudo-observations, in any form ``torch.as_tensor`` accepts.
    controls : ControlsLike, or None, optional
        Backend and numerical configuration, passed to every pair.
    var_types : list of str, or None, optional
        One ``"c"`` per variable; only continuous pairs are supported.
    x : array, shape (n, p), or None, optional
        External covariates, threaded to every pair alongside each edge's
        conditioning values.
    fit_edge : callable, or None, optional
        Per-edge pair fitter; defaults to fitting ``bicop_class``.
    fit_level : callable, or None, optional
        Whole-tree fitter.

    Returns
    -------
    RosenblattVinecop
        ``self``, so the call chains.
    """
    return super().fit(
      self._prep(u),
      controls,
      var_types=var_types,
      x=None if x is None else self._prep(x),
      fit_edge=fit_edge,
      fit_level=fit_level,
    )

  def get_pair_copula(
    self,
    tree: int,
    edge: int,
  ) -> RosenblattBicop:
    """Return the pair copula at a tree and edge position."""
    return self.pair_copulas[tree][edge]

  def set_pair_copulas(
    self,
    pair_copulas: list[list[BicopLike[torch.Tensor]]],
  ) -> None:
    """Install pair copulas produced by the inherited fit engine."""
    checked_pairs = self._check_pair_copulas(pair_copulas)
    self._validate_pair_copulas(checked_pairs)

    self.pair_copulas = checked_pairs
    self._set_placement(self._resolve_vine_device(checked_pairs))
    # The pairs change here without the structure changing, which is the case
    # `_bind_vine` does not cover, so the base asks an implementation to drop
    # anything it memoized from them -- and names the hook rather than the
    # attribute behind it. Nothing here memoizes the pairs: this vine's
    # context assembles conditioning, so no batched state is ever built. The
    # call is the contract.
    self._invalidate_batched()

  def _sample_uniform(
    self,
    n: int,
    qrng: bool,
    seeds: list[int],
  ) -> torch.Tensor:
    """Draw base uniforms for inherited vine sampling."""
    if qrng:
      from pyvinecopulib.utils import sample_uniform

      draws = sample_uniform(n, self.d, qrng=True, seeds=list(seeds))
      return self._prep(draws)

    generator = torch.Generator(device=self._device)
    if seeds:
      generator.manual_seed(int(seeds[0]))
    else:
      generator.seed()

    return torch.rand(
      (n, self.d),
      generator=generator,
      dtype=torch.float64,
      device=self._device,
    )
