"""Tests for the TabPFN adapter: what it hands TabPFN and how it reads it back.

Everything the adapter inherits from ``ConditionalMargin`` is tested once, in
``test_margin.py``; what is left here is the adapter itself, driven through a
fake regressor that answers the way ``TabPFNRegressor`` does.
"""

from __future__ import annotations

import copy
import gc
import threading
import time
import weakref

import numpy as np
import pytest
import torch
from tabpfn.base import ModelSpecs
from tabpfn.constants import ModelVersion

from npcc import TabPFNQuantileBackend, create_backend
from npcc.core import QuantileTableConfig
from tests.conftest import TABPFN_TARGET, FakeTabPFNRegressor, normal_density_y

#: The adapter module, whose checkpoint loader and table of loaded networks
#: the sharing tests replace.
ADAPTER = "npcc.core.backends.tabpfn_quantile"

#: The checkpoint every :class:`CheckpointRegressor` names by default.
CHECKPOINT = "tabpfn-v3-regressor.ckpt"


def fitted_backend(
  *,
  x: torch.Tensor | None = None,
  batch_size: int | None = None,
  model_kwargs: dict[str, object] | None = None,
) -> TabPFNQuantileBackend:
  """A TabPFN margin fitted on ten responses, through whichever fake is set."""
  backend = TabPFNQuantileBackend(
    device="cpu", batch_size=batch_size, model_kwargs=model_kwargs
  )
  backend.fit(
    torch.full((10,), 0.5, dtype=torch.float64),
    x=torch.zeros((10, 1), dtype=torch.float64) if x is None else x,
  )
  return backend


class CheckpointRegressor(FakeTabPFNRegressor):
  """A fake that names a checkpoint, as the real regressor does.

  It records the device and inference precision it was built with, which is
  what the adapter reads beside the checkpoint.
  """

  def __init__(
    self,
    *,
    model_path: object = CHECKPOINT,
    device: object = "auto",
    inference_precision: object = "auto",
    **_: object,
  ) -> None:
    super().__init__(model_path=model_path)
    self.device = device
    self.inference_precision = inference_precision


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> list[object]:
  """Every checkpoint the adapter loads, each as an empty stand-in network.

  The table of loaded networks starts empty, so no test is handed a network
  another one loaded.
  """
  loaded: list[object] = []

  def load(
    checkpoint: object, which: str
  ) -> tuple[list[torch.nn.Module], list[None], None, None]:
    del which
    loaded.append(checkpoint)
    return [torch.nn.Module()], [None], None, None

  monkeypatch.setattr(TABPFN_TARGET, CheckpointRegressor)
  monkeypatch.setattr(f"{ADAPTER}.initialize_tabpfn_model", load)
  monkeypatch.setattr(f"{ADAPTER}._networks", weakref.WeakValueDictionary())
  return loaded


def checkpoint_of(backend: TabPFNQuantileBackend) -> object:
  """What a fitted margin's regressor was handed as its ``model_path``."""
  assert isinstance(backend.model_, CheckpointRegressor)
  return backend.model_.model_path


@pytest.mark.parametrize("feature_dim", [1, 3])
def test_fit_builds_a_regressor_on_the_margins_device(
  patch_tabpfn: None, feature_dim: int
) -> None:
  """The regressor is built at fit time, on the device the margin names.

  TabPFN picks its own device unless told; one left to choose could put the
  model on CUDA while the margin evaluates on CPU.
  """
  del patch_tabpfn
  x = torch.rand((10, feature_dim), dtype=torch.float64)

  backend = fitted_backend(x=x)

  assert isinstance(backend.model_, FakeTabPFNRegressor)
  assert backend.model_.fitted_
  assert backend.model_kwargs["device"] == "cpu"


def test_model_version_accepts_its_string_value(patch_tabpfn: None) -> None:
  """A study config spells the version as text; the adapter reads it.

  TOML has no enum, so ``backend_kwargs = { model_version = "v2.5" }`` is the
  only way a config can select one, and it must arrive as the enum.
  """
  del patch_tabpfn
  backend = create_backend(
    "tabpfn-quantiles",
    transform="logit",
    quantile_table_config=QuantileTableConfig(),
    eps=1e-6,
    device="cpu",
    batch_size=None,
    backend_kwargs={"model_version": "v2.5"},
  )

  assert isinstance(backend, TabPFNQuantileBackend)
  assert backend.model_version is ModelVersion.V2_5
  assert TabPFNQuantileBackend().model_version is ModelVersion.V3


def test_quantiles_are_read_back_as_the_normal(patch_tabpfn: None) -> None:
  """TabPFN's per-level arrays become the table that reproduces the truth."""
  del patch_tabpfn
  backend = fitted_backend()
  y = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

  result = backend.pdf(y, x=torch.zeros((3, 1), dtype=torch.float64))

  torch.testing.assert_close(result, normal_density_y(y))


def test_a_chunk_as_long_as_the_table_is_not_transposed(
  patch_tabpfn: None,
) -> None:
  """TabPFN puts the levels first, which the adapter states, not infers.

  Inferred from the shape, the orientation of a chunk with exactly
  ``n_quantiles`` rows is a square that reads either way. Read the wrong way,
  every row holds one level's quantile, repeated: a table with no spread.
  """
  del patch_tabpfn
  n = QuantileTableConfig().n_quantiles
  backend = fitted_backend(batch_size=n)
  y = torch.linspace(0.2, 0.8, n, dtype=torch.float64)

  result = backend.pdf(y, x=torch.zeros((n, 1), dtype=torch.float64))

  torch.testing.assert_close(result, normal_density_y(y))


def test_an_output_of_another_shape_is_refused(patch_bad_shape: None) -> None:
  """A result of the wrong shape fails at the read-out, naming the contract."""
  del patch_bad_shape
  backend = fitted_backend()

  with pytest.raises(RuntimeError, match="must return shape"):
    backend.pdf(
      torch.tensor([0.5], dtype=torch.float64),
      x=torch.zeros((1, 1), dtype=torch.float64),
    )


def test_margins_share_the_network_of_their_checkpoint(
  loads: list[object],
) -> None:
  """Every margin fitted from one checkpoint is handed the same network.

  A regressor left to load it builds its own copy at ``fit`` and keeps it on
  its device: about 220 MiB per margin for TabPFN v3, where a vine of ten
  variables fits a hundred margins.
  """
  first, second = fitted_backend(), fitted_backend()

  assert isinstance(checkpoint_of(first), ModelSpecs)
  assert checkpoint_of(first) is checkpoint_of(second)
  assert loads == [CHECKPOINT]


@pytest.mark.parametrize(
  "model_kwargs",
  [{"device": "cuda:1"}, {"inference_precision": torch.float16}],
  ids=["device", "precision"],
)
def test_a_network_is_shared_on_one_device_at_one_precision(
  loads: list[object], model_kwargs: dict[str, object]
) -> None:
  """A margin on another device, or at a forced precision, loads its own.

  TabPFN moves the network to its regressor's device, and casts it to a forced
  precision, in place: one shared across either would move away from the
  margins already holding it, or hand them weights rounded to half precision.
  """
  shared = fitted_backend()
  other = fitted_backend(model_kwargs=model_kwargs)

  assert checkpoint_of(other) is not checkpoint_of(shared)
  assert loads == [CHECKPOINT, CHECKPOINT]


@pytest.mark.parametrize(
  "model_path",
  [["v2.5.ckpt", "v3.ckpt"], object()],
  ids=["checkpoint-list", "in-memory-model"],
)
def test_a_model_the_caller_names_reaches_tabpfn_as_given(
  loads: list[object], model_path: object
) -> None:
  """Only a single checkpoint file is swapped for its shared network.

  A list of checkpoints is an ensemble of several networks, and an in-memory
  model (TabPFN's ``ModelSpecs``, stood in for here by a bare object) is one
  the caller already holds; loading a checkpoint in place of either would
  replace the model they asked for.
  """
  backend = fitted_backend(model_kwargs={"model_path": model_path})

  assert checkpoint_of(backend) is model_path
  assert loads == []


def test_a_network_no_margin_holds_is_released(loads: list[object]) -> None:
  """Once the last margin holding a network is gone, the network goes too.

  A study drops each fitted model and empties the CUDA cache before the next
  one; a table that kept every network it loaded would hold each checkpoint's
  weights on the device for the rest of the process.
  """
  backend = fitted_backend()
  del backend
  gc.collect()

  fitted_backend()

  assert loads == [CHECKPOINT, CHECKPOINT]


def test_passes_through_a_shared_network_take_turns(
  loads: list[object], monkeypatch: pytest.MonkeyPatch
) -> None:
  """Margins on two threads never run TabPFN through one network at once.

  The network's weights are only read, but TabPFN's fit and predict were never
  shown to be safe side by side on one module, and every margin of a vine
  holds the same one; without its lock, two threads' passes would interleave.
  """
  del loads
  running, most = [0], [0]
  guard = threading.Lock()

  class Overlapping(CheckpointRegressor):
    """Records how many of its passes run at once."""

    def _pass(self) -> None:
      with guard:
        running[0] += 1
        most[0] = max(most[0], running[0])
      time.sleep(0.005)
      with guard:
        running[0] -= 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> Overlapping:
      self._pass()
      super().fit(X, y)
      return self

    def predict(
      self,
      X: np.ndarray,
      *,
      output_type: str = "mean",
      quantiles: list[float] | None = None,
    ) -> object:
      self._pass()
      return super().predict(X, output_type=output_type, quantiles=quantiles)

  monkeypatch.setattr(TABPFN_TARGET, Overlapping)

  def work() -> None:
    margin = fitted_backend()
    for _ in range(5):
      margin.cdf(
        torch.full((4,), 0.5, dtype=torch.float64),
        x=torch.zeros((4, 1), dtype=torch.float64),
      )

  threads = [threading.Thread(target=work) for _ in range(4)]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()

  assert most[0] == 1


def test_a_fitted_margin_survives_a_deep_copy(loads: list[object]) -> None:
  """A fitted margin can be deep-copied, and the copy reads as the original.

  pyvinecopulib deep-copies a margin that stands at several positions of a
  vine. A lock copied the ordinary way raises, so a margin holding its shared
  network's lock would make that copy fail.
  """
  del loads
  original = fitted_backend()
  y = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
  x = torch.zeros((3, 1), dtype=torch.float64)

  duplicate = copy.deepcopy(original)

  torch.testing.assert_close(duplicate.cdf(y, x=x), original.cdf(y, x=x))
