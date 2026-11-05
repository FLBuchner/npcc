"""Tests for the TabPFN adapter: what it hands TabPFN and how it reads it back.

Everything the adapter inherits from ``ConditionalMargin`` is tested once, in
``test_margin.py``; what is left here is the adapter itself, driven through a
fake regressor that answers the way ``TabPFNRegressor`` does.
"""

from __future__ import annotations

import pytest
import torch
from tabpfn.constants import ModelVersion

from npcc import TabPFNQuantileBackend, create_backend
from npcc.core import QuantileTableConfig
from tests.conftest import FakeTabPFNRegressor, normal_density_y


def fitted_backend(
  *, x: torch.Tensor | None = None, batch_size: int | None = None
) -> TabPFNQuantileBackend:
  """A TabPFN margin fitted on ten responses, through whichever fake is set."""
  backend = TabPFNQuantileBackend(device="cpu", batch_size=batch_size)
  backend.fit(
    torch.full((10,), 0.5, dtype=torch.float64),
    x=torch.zeros((10, 1), dtype=torch.float64) if x is None else x,
  )
  return backend


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
