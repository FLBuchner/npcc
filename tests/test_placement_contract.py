"""Every estimator places what the base hands it, on both devices.

pyvinecopulib's bases prepare a covariate matrix through
``prepare_covariates``, which routes through the object's ``_prep`` hook --
so a class whose placement is *declared* rather than held as a tensor is
honored on that path as well as on the argument path. It was not always: the
function reached placement through the module-level ``place``, which answers
by looking for a float array the object holds, and these estimators hold none.
A covariate matrix then came back untouched and met placed copula arguments
inside a concatenation several frames later, which is invisible on CPU and
fatal on CUDA.

Reachable from ``BicopBase.loglik``, ``sample``, ``hinv1``, ``hinv2`` and
every edge of the vine cascade. These tests are what keep it honored.
"""

from __future__ import annotations

import numpy
import pytest
import torch
from pyvinecopulib import RVineStructure
from pyvinecopulib.core.extend import prepare_covariates

from npcc.core import (
  FitControlsRosenblattBicop,
  RosenblattBicop,
  RosenblattVinecop,
  RosenblattVinedist,
)
from npcc.core._placement import TensorPlacement, resolve_device
from tests.conftest import NormalQuantileBackend


def _cuda_available() -> bool:
  """Whether a CUDA device can actually be used.

  Defensive twice over: ``torch.cuda`` raises rather than returning ``False``
  on a build without CUDA support, and a half-installed driver can raise from
  somewhere else again. Neither may break collection.

  Returns
  -------
  bool
      ``True`` only if a device is present and usable.
  """
  try:
    return torch.cuda.is_available()
  except Exception:  # noqa: BLE001 - collection must not fail for any reason
    return False


# Parameterized over both devices, with the marks on the *param*: `-m "not
# cuda"` (the declared default) deselects the GPU half, and `skipif` makes an
# explicit `-m cuda` safe on a machine without one. A mark applied from inside
# the test body lands after collection and would do neither.
_DEVICES = [
  "cpu",
  pytest.param(
    "cuda",
    marks=[
      pytest.mark.cuda,
      pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA device"),
    ],
  ),
]


def _estimators(device: str) -> dict[str, TensorPlacement]:
  """One of each estimator, on ``device``.

  Parameters
  ----------
  device : str
      The device to construct them on.

  Returns
  -------
  dict
      The four estimators, keyed by a short name. Typed by the mixin they
      share, which is what carries ``_prep`` and ``_placement_ref``.
  """
  controls = FitControlsRosenblattBicop(
    backend="normal-quantile", device=device
  )
  structure = RVineStructure.from_order([1, 2, 3])
  vinecop = RosenblattVinecop(None, structure, device=device)
  margins = [
    NormalQuantileBackend(transform="identity", device=device) for _ in range(3)
  ]
  return {
    "bicop": RosenblattBicop(controls),
    "margin": NormalQuantileBackend(transform="identity", device=device),
    "vinecop": vinecop,
    "vinedist": RosenblattVinedist(vinecop, margins),
  }


@pytest.mark.parametrize("name", ["bicop", "margin", "vinecop", "vinedist"])
@pytest.mark.parametrize("device", _DEVICES)
def test_place_finds_every_estimators_placement(
  register_hermetic_backends: None,
  device: str,
  name: str,
) -> None:
  """A host NumPy covariate block arrives placed, through the hook."""
  del register_hermetic_backends

  expected = resolve_device(device)
  estimator = _estimators(device)[name]
  host = numpy.arange(6, dtype=float).reshape(3, 2)

  covariates = prepare_covariates(estimator, host, 3)

  assert isinstance(covariates, torch.Tensor)
  assert covariates.dtype is torch.float64
  assert covariates.device == expected
  torch.testing.assert_close(covariates, estimator._prep(host))


@pytest.mark.parametrize("name", ["bicop", "margin", "vinecop", "vinedist"])
def test_every_estimator_declares_its_placement(
  register_hermetic_backends: None,
  name: str,
) -> None:
  """The declaration is ``_ref_tensor``, not an array the object happens to hold.

  Upstream's mixin resolves a placement in three steps -- a registered tensor,
  then a declared ``device``/``dtype``, then an empty CPU ``float64``. None of
  these estimators is an ``nn.Module``, so the first finds nothing and the
  last would silently put a CUDA estimator's inputs on the host. The override
  is what makes the second step the one that answers.
  """
  del register_hermetic_backends

  estimator = _estimators("cpu")[name]
  reference = estimator._ref_tensor()

  assert reference.dtype is torch.float64
  assert reference.device == estimator._device
  assert reference.numel() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_prep_keeps_a_gradient_whatever_torch_is_installed(
  dtype: torch.dtype,
) -> None:
  """``_prep`` preserves autograd across a dtype change, by construction.

  This is why ``_prep`` cannot delegate to ``place``, and the reason is about
  version independence rather than about ``place`` being wrong.
  ``torch.as_tensor`` has always carried a gradient. ``place`` reaches
  placement through ``array_api_compat``'s ``xp.asarray``, which delegates to
  ``torch.asarray``, whose default for ``requires_grad`` changed: silently
  ``False`` on torch 2.11, ``obj.requires_grad`` from 2.13. So asserting what
  ``place`` does would pin the installed torch rather than a property of this
  package -- what is asserted here is only that ``_prep`` does not care.

  The float32 case is the one that converts; float64 already matches the
  reference and would survive any implementation.
  """
  model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))
  tracked = torch.tensor([[0.3, 0.4]], dtype=dtype, requires_grad=True)

  placed = model._prep(tracked)

  assert placed.dtype is torch.float64
  assert placed.requires_grad
  assert placed.grad_fn is not None or placed is tracked


def test_resolve_device_keeps_an_explicit_choice() -> None:
  """A named device, as text or as an object, is the device used."""
  assert resolve_device("cpu") == torch.device("cpu")
  assert resolve_device(torch.device("cpu")) == torch.device("cpu")


@pytest.mark.parametrize("name", ["bicop", "margin", "vinecop"])
@pytest.mark.parametrize("device", _DEVICES)
def test_base_uniforms_are_drawn_seeded_on_the_estimators_device(
  register_hermetic_backends: None,
  device: str,
  name: str,
) -> None:
  """Every estimator's sampling hook draws on its own device, reproducibly.

  The hook is upstream's to call and this package's to answer, once for all
  three levels; a draw on the host would meet a CUDA cascade in its first
  concatenation, and an unseeded one would make ``sample(seeds=...)`` a lie.
  """
  del register_hermetic_backends
  estimator = _estimators(device)[name]
  draw = estimator._draw_uniform

  first = draw((5, 2), seeds=[7])

  assert first.device == resolve_device(device)
  assert first.dtype is torch.float64
  torch.testing.assert_close(first, draw((5, 2), seeds=[7]))
  assert draw((5, 2), qrng=True, seeds=[7]).device == resolve_device(device)
