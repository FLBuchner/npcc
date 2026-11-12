"""Tests for the response transforms a conditional margin fits on."""

from __future__ import annotations

import math
from typing import get_args

import pytest
import torch

from npcc.core import Transform
from npcc.core._transform import TRANSFORMS, resolve_transform


def test_every_named_transform_has_an_implementation() -> None:
  """The public ``Transform`` names and the implementations stay one list.

  A name added to the alias without an implementation would pass the controls'
  validation and then fail inside a fit; one added only to the table would be
  reachable but untyped.
  """
  assert tuple(TRANSFORMS) == get_args(Transform)


@pytest.mark.parametrize("name", get_args(Transform))
def test_inverse_undoes_forward(name: str) -> None:
  """``inverse(forward(y)) == y`` inside the support, for every transform.

  A mismatched pair would make ``icdf`` return a response that ``cdf`` does not
  map back to the level it was asked for.
  """
  g = resolve_transform(name)
  lo, hi = g.support
  y = torch.linspace(
    max(lo, -5.0) + 0.05, min(hi, 5.0) - 0.05, 9, dtype=torch.float64
  )
  torch.testing.assert_close(g.inverse(g.forward(g.clip(y, 1e-6))), y)


@pytest.mark.parametrize("name", get_args(Transform))
def test_jacobian_is_the_derivative_of_forward(name: str) -> None:
  """``jacobian`` is ``dz/dy``, checked against a central difference.

  The density on the response scale is the modeled density times this factor,
  so a wrong one is a density that does not integrate to one.
  """
  g = resolve_transform(name)
  lo, hi = g.support
  y = torch.linspace(
    max(lo, -5.0) + 0.1, min(hi, 5.0) - 0.1, 7, dtype=torch.float64
  )
  h = 1e-6
  numeric = (g.forward(y + h) - g.forward(y - h)) / (2 * h)
  torch.testing.assert_close(g.jacobian(y), numeric, rtol=1e-6, atol=0.0)


def test_unknown_name_is_refused() -> None:
  """An unknown name raises, naming what is allowed and what was given."""
  with pytest.raises(ValueError, match=r"must be one of .*; got 'exp'"):
    resolve_transform("exp")


def test_support_marks_finite_ends() -> None:
  """Copula-scale transforms are bounded on both sides, log below, identity not.

  The quantile table closes a tail differently at a finite end than at an
  infinite one, and reads which is which from here.
  """
  assert resolve_transform("logit").support == (0.0, 1.0)
  assert resolve_transform("probit").support == (0.0, 1.0)
  assert resolve_transform("log").support == (0.0, math.inf)
  assert resolve_transform("identity").support == (-math.inf, math.inf)


def test_logit_is_antisymmetric_about_one_half() -> None:
  """``logit(1 - p) == -logit(p)``, to the last bit near either end.

  Written as ``log(p) - log1p(-p)`` rather than ``log(p / (1 - p))`` so the
  two ends lose the same precision; the pair copula fits both directions on
  this scale and averages them, so an asymmetric rounding biases one.
  """
  g = resolve_transform("logit")
  p = torch.tensor([1e-6, 0.1, 0.4, 0.5], dtype=torch.float64)

  torch.testing.assert_close(g.forward(1.0 - p), -g.forward(p))
  assert g.forward(torch.tensor([0.5], dtype=torch.float64)).item() == 0.0


def test_probit_inverse_keeps_its_precision_in_the_lower_tail() -> None:
  """``Phi`` is computed through ``erfc``, which keeps the lower tail.

  Through ``1 + erf(z / sqrt 2)`` it cancels to zero below about ``z = -8``,
  and a zero there is a zero density for every response that maps beyond it.
  """
  z = torch.tensor([-10.0, -8.5, -3.0], dtype=torch.float64)

  torch.testing.assert_close(
    resolve_transform("probit").inverse(z), torch.special.ndtr(z)
  )
