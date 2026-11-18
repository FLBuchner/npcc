"""The suite's device: the CPU, unless a run selects the ``cuda`` tests."""

from __future__ import annotations

import pytest
import torch

from npcc.core import RosenblattBicop
from tests.conftest import selects_cuda


@pytest.mark.parametrize(
  ("markexpr", "selected"),
  [
    ("not cuda", False),
    ("cuda", True),
    ("cuda or real_model", True),
    ("real_model and not cuda", False),
    ("real_model", False),
    ("", True),
  ],
)
def test_only_a_run_selecting_cuda_keeps_the_device(
  markexpr: str, selected: bool
) -> None:
  """``make test-cuda`` must see the device, and the default run must not.

  Read the wrong way round, every ``cuda`` test would skip under ``-m cuda``,
  or every xdist worker of ``make test`` would open its own CUDA context.
  """
  assert selects_cuda(markexpr) is selected


def test_an_estimator_left_without_a_device_is_on_the_cpu(
  request: pytest.FixtureRequest,
) -> None:
  """No test outside the ``cuda`` ones may place its tensors on a GPU.

  An estimator given no device resolves to ``cuda`` wherever one exists, so on
  a GPU machine the CPU suite CI runs would otherwise run on the GPU, from
  every worker at once.
  """
  if selects_cuda(request.config.getoption("markexpr")):
    pytest.skip("this run selects the cuda tests, which keep the device")
  assert not torch.cuda.is_available()
  assert RosenblattBicop()._device == torch.device("cpu")
