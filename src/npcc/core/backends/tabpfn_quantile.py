"""
Quantile-based TabPFN conditional margin.

This backend reads TabPFN's ``output_type="quantiles"`` result into the
quantile-table reconstruction machinery. It implements model fitting and
quantile prediction; PDF, CDF, inverse CDF, and grid evaluation are inherited
from :class:`~npcc.core.margin.ConditionalMargin`.

This module is the one place the concrete ``TabPFNRegressor`` is imported, and
it is looked up as a module global when a model is built -- which makes this
module the single monkeypatch target in the tests.

TabPFN receives host-side input tensors. These conversions are isolated in this
adapter, while all outputs are returned as torch tensors on the configured
model device.

**One network per checkpoint.** A ``TabPFNRegressor`` builds its network from
the checkpoint at every ``fit`` and keeps it on its device for as long as it
lives: about 220 MiB of weights for TabPFN v3, held once per fitted margin. A
vine fits a margin per variable and two per pair, all from one checkpoint, so
it would hold as many copies of the same weights. Each regressor built here is
instead handed the network loaded for its checkpoint, device and inference
precision, as TabPFN's ``ModelSpecs``, which TabPFN uses by reference. Those
three are the key because they are what TabPFN applies to the network in
place; nothing else at fit or predict time writes to it, so a regressor reads
the same weights from the shared network as from a copy of its own. The
network is released once no fitted regressor holds it.

Every fit and prediction through a shared network holds its lock. Two passes
through one TabPFN module side by side were never shown to be safe, and with
one thread the lock is never contended.
"""

from __future__ import annotations

import threading
import weakref
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Unpack

import torch
from tabpfn import TabPFNRegressor
from tabpfn.base import ModelSpecs, initialize_tabpfn_model
from tabpfn.constants import ModelVersion
from tabpfn.utils import DevicesSpecification, infer_devices

from npcc.core.margin import ConditionalMargin, MarginOptions

if TYPE_CHECKING:
  from contextlib import AbstractContextManager

DEFAULT_MODEL_VERSION = ModelVersion.V3
"""Default TabPFN model version, shared by both TabPFN backends."""


class _SharedNetwork:
  """A loaded network, and the lock every pass through it holds.

  A lock cannot be copied or pickled, so a deep copy or a pickle carries the
  network alone and gets a lock of its own. Inside a deep copy of a margin
  that is the network its copied regressor holds, which no other margin
  shares.
  """

  spec: ModelSpecs
  lock: threading.Lock

  def __init__(self, spec: ModelSpecs) -> None:
    self.spec = spec
    self.lock = threading.Lock()

  def __getstate__(self) -> dict[str, ModelSpecs]:
    return {"spec": self.spec}

  def __setstate__(self, state: dict[str, ModelSpecs]) -> None:
    self.spec = state["spec"]
    self.lock = threading.Lock()


_networks: weakref.WeakValueDictionary[tuple[str, str, str], _SharedNetwork] = (
  weakref.WeakValueDictionary()
)
_networks_lock = threading.Lock()


def _shared_network(
  checkpoint: str | Path,
  device: DevicesSpecification,
  precision: torch.dtype | str,
) -> _SharedNetwork:
  """Return the network loaded for ``checkpoint``, loading it on first use.

  It is loaded by the function ``TabPFNRegressor.fit`` loads a checkpoint
  with, so its weights, target distribution and inference config are the ones
  a regressor would load for itself. It is moved to the device a regressor's
  inference engine would move it to before any regressor is handed it, so one
  fitting on another thread never meets it half moved.
  """
  key = (str(checkpoint), str(device), str(precision))
  with _networks_lock:
    network = _networks.get(key)
    if network is None:
      models, configs, criterion, config = initialize_tabpfn_model(
        checkpoint, "regressor"
      )
      models[0].to(infer_devices(device)[0])
      network = _SharedNetwork(
        ModelSpecs(
          model=models[0],
          architecture_config=configs[0],
          inference_config=config,
          norm_criterion=criterion,
        )
      )
      _networks[key] = network
  return network


class TabPFNQuantileBackend(ConditionalMargin):
  """Conditional margin using TabPFN's predicted quantiles.

  Parameters
  ----------
  model_kwargs : dict, optional
      Additional arguments passed to ``TabPFNRegressor``.
  model_version : ModelVersion, str, or None, default=ModelVersion.V3
      TabPFN model version, as the enum or its value (``"v2.5"``, ``"v3"``).
      ``None`` builds a bare ``TabPFNRegressor`` rather than a versioned
      default.
  **margin : MarginOptions
      Forwarded to :class:`~npcc.core.margin.ConditionalMargin`.
  """

  model_: Any | None

  def __init__(
    self,
    *,
    model_kwargs: dict[str, Any] | None = None,
    model_version: ModelVersion | str | None = DEFAULT_MODEL_VERSION,
    **margin: Unpack[MarginOptions],
  ) -> None:
    super().__init__(**margin)
    self.model_kwargs = dict(model_kwargs or {})
    self.model_kwargs.setdefault("device", str(self._device))
    self.model_version = (
      None if model_version is None else ModelVersion(model_version)
    )
    self.model_ = None
    self._network: _SharedNetwork | None = None

  def _passes(self) -> AbstractContextManager[object]:
    """The lock of the shared network, or none for a model of its own."""
    return nullcontext() if self._network is None else self._network.lock

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    """Fit TabPFN."""
    if self.model_version is None:
      self.model_ = TabPFNRegressor(**self.model_kwargs)
    else:
      self.model_ = TabPFNRegressor.create_default_for_version(
        self.model_version, **self.model_kwargs
      )

    # A checkpoint named by its path is swapped for the network loaded for it;
    # an in-memory model, or a list of checkpoints, is the caller's to keep.
    self._network = None
    if isinstance(self.model_.model_path, (str, Path)):
      self._network = _shared_network(
        self.model_.model_path,
        self.model_.device,
        self.model_.inference_precision,
      )
      self.model_.model_path = self._network.spec

    # TabPFN converts its inputs to NumPy itself, which a CUDA tensor refuses.
    with self._passes():
      self.model_.fit(x.detach().cpu(), z.detach().cpu())

  def _predict_quantiles(
    self,
    x: torch.Tensor,
    alphas: torch.Tensor,
  ) -> torch.Tensor:
    """Predict one quantile table for a chunk of conditioning rows.

    TabPFN returns one array per level, so the levels lead.
    """
    assert self.model_ is not None

    with self._passes():
      predicted = self.model_.predict(
        x.detach().cpu(),
        output_type="quantiles",
        quantiles=alphas.tolist(),
      )
    return self._quantile_rows(predicted, x, levels_first=True)
