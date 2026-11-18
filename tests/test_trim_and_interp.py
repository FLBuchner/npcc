"""Tests for the domain step and the device-side interpolator."""

from __future__ import annotations

import pytest
import torch

from npcc.core._interp import interp
from npcc.core._trim import check_open_unit


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_a_value_at_or_outside_the_boundary_is_refused(bad: float) -> None:
  """An exact ``0`` or ``1`` is a defect upstream, not a rounding artifact.

  Every score reaching an estimator comes from a probability integral
  transform, which cannot produce one; clamping it would turn the defect into
  a plausible number.
  """
  values = torch.tensor([[0.5, bad]], dtype=torch.float64)

  with pytest.raises(ValueError, match="u must lie strictly inside"):
    check_open_unit(values, 1e-6, "u")


def test_interior_values_are_clamped_eps_inside_and_keep_their_shape() -> None:
  """The clamp is at the caller's ``eps``, elementwise, on any shape.

  The pair's ``_prep_args`` hands it the whole ``(n, 2)`` block, so a flatten
  here would break every pair evaluation.
  """
  values = torch.tensor([[1e-9, 0.5], [0.5, 1.0 - 1e-9]], dtype=torch.float64)

  result = check_open_unit(values, 1e-6, "u")

  expected = torch.tensor([[1e-6, 0.5], [0.5, 1.0 - 1e-6]], dtype=torch.float64)
  torch.testing.assert_close(result, expected)


def test_interp_is_linear_inside_and_flat_outside() -> None:
  """``interp`` matches ``numpy.interp``, including its flat extrapolation.

  The Sinkhorn scalings are interpolated back to query points with it, and a
  query outside the projection grid must read the boundary scaling rather
  than an extrapolated one.
  """
  xp = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
  fp = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64)
  x = torch.tensor([-0.5, 0.5, 2.5], dtype=torch.float64)

  result = interp(x, xp, fp)

  torch.testing.assert_close(
    result, torch.tensor([0.0, 5.0, 20.0], dtype=torch.float64)
  )
