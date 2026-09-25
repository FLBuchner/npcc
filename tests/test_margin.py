"""Tests for backend-powered conditional margins."""

from __future__ import annotations

import copy
import math

import numpy
import pytest
import torch
from pyvinecopulib.core import MarginBase

from npcc.core.margin import ConditionalMargin
from npcc.core.margin_quantile_table import QuantileTableConfig
from npcc.core.registry import create_backend


def make_margin(
  register_uniform_backends: None,
) -> ConditionalMargin:
  """Construct the hermetic native backend on the original scale."""
  return create_backend(
    "uniform-native",
    transform="identity",
    quantile_table_config=QuantileTableConfig(),
    eps=1e-6,
    device="cpu",
    batch_size=None,
  )


def test_conditional_margin_subclasses_margin_base(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)

  assert isinstance(margin, MarginBase)
  assert margin.supports_covariates is True
  assert margin.supports_controls is False
  assert margin.supports_weights is False
  assert margin.var_type == "c"
  assert margin.support == (float("-inf"), float("inf"))
  assert margin.is_fitted is False
  assert margin.nobs is None
  assert margin.n_parameters == 0.0


@pytest.mark.parametrize("eps", [0.0, 0.5, -1e-6])
def test_invalid_eps_is_rejected(
  register_uniform_backends: None,
  eps: float,
) -> None:
  with pytest.raises(ValueError, match="eps must lie"):
    create_backend(
      "uniform-native",
      transform="identity",
      quantile_table_config=QuantileTableConfig(),
      eps=eps,
      device="cpu",
      batch_size=None,
    )


@pytest.mark.parametrize(
  ("x", "expected_width"),
  [
    (None, 1),
    (torch.ones((8, 1)), 1),
    (torch.ones((8, 3)), 3),
  ],
)
def test_fit_constructs_expected_feature_matrix(
  register_uniform_backends: None,
  monkeypatch: pytest.MonkeyPatch,
  x: torch.Tensor | None,
  expected_width: int,
) -> None:
  margin = make_margin(register_uniform_backends)
  observed_shapes: list[tuple[int, ...]] = []
  original = margin._fit_model

  def record_fit(features: torch.Tensor, z: torch.Tensor) -> None:
    observed_shapes.append(tuple(features.shape))
    original(features, z)

  monkeypatch.setattr(margin, "_fit_model", record_fit)

  y = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64)
  result = margin.fit(y, x=x)

  assert result is margin
  assert margin.is_fitted is True
  assert margin.nobs == 8
  assert margin.n_parameters == 0.0
  assert observed_shapes == [(8, expected_width)]


def test_identity_scale_distribution_operations(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)
  margin.fit(torch.linspace(-1.0, 1.0, 20, dtype=torch.float64))

  y = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
  p = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)

  torch.testing.assert_close(margin.pdf(y), torch.full_like(y, 0.25))
  torch.testing.assert_close(margin.cdf(y), p)
  torch.testing.assert_close(margin.icdf(p), y)


def test_evaluation_before_fit_raises(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)

  with pytest.raises(RuntimeError, match="model is not fitted"):
    margin.pdf(torch.tensor([0.0], dtype=torch.float64))


def test_rejects_mismatched_rows(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)

  with pytest.raises(ValueError, match="one row per observation"):
    margin.fit(
      torch.linspace(-1.0, 1.0, 8, dtype=torch.float64),
      x=torch.ones((7, 2), dtype=torch.float64),
    )


def test_inherited_margin_operations(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)
  margin.fit(torch.linspace(-1.0, 1.0, 20, dtype=torch.float64))

  y = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
  expected_logpdf = torch.full_like(y, math.log(0.25))

  torch.testing.assert_close(margin.logpdf(y), expected_logpdf)
  torch.testing.assert_close(margin.cdf_left(y), margin.cdf(y))
  torch.testing.assert_close(margin.loglik(y), expected_logpdf.sum())


def test_cdf_icdf_round_trip_with_covariates(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)
  train_y = torch.linspace(-1.0, 1.0, 20, dtype=torch.float64)
  train_x = torch.column_stack((train_y, train_y.square()))
  margin.fit(train_y, x=train_x)

  p = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
  x = torch.tensor(
    [
      [-1.0, 1.0],
      [0.0, 0.0],
      [1.0, 1.0],
    ],
    dtype=torch.float64,
  )

  quantiles = margin.icdf(p, x=x)

  torch.testing.assert_close(margin.cdf(quantiles, x=x), p)


def test_grid_evaluation_shapes(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)
  margin.fit(torch.linspace(-1.0, 1.0, 20, dtype=torch.float64))
  x = torch.ones((4, 2), dtype=torch.float64)
  y_grid = torch.linspace(-1.0, 1.0, 7, dtype=torch.float64)

  assert margin.pdf_grid(y_grid, x=x).shape == (4, 7)
  assert margin.cdf_grid(y_grid, x=x).shape == (4, 7)


def test_sample_is_reproducible(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)
  margin.fit(torch.linspace(-1.0, 1.0, 20, dtype=torch.float64))

  first = margin.sample(10, seeds=[42])
  second = margin.sample(10, seeds=[42])

  torch.testing.assert_close(first, second)
  assert torch.all((first >= -2.0) & (first <= 2.0))


def test_conditional_sample_returns_tensor(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)
  train_y = torch.linspace(-1.0, 1.0, 20, dtype=torch.float64)
  train_x = torch.ones((20, 2), dtype=torch.float64)
  margin.fit(train_y, x=train_x)

  result = margin.sample(
    5,
    x=torch.ones((5, 2), dtype=torch.float64),
    seeds=[42],
  )

  assert result.shape == (5,)


def test_copied_margins_are_independent(
  register_uniform_backends: None,
) -> None:
  first = make_margin(register_uniform_backends)
  second = copy.deepcopy(first)

  assert first is not second

  first.fit(torch.linspace(-1.0, 1.0, 20, dtype=torch.float64))

  assert first.is_fitted is True
  assert second.is_fitted is False


def test_information_criteria_are_refused(
  register_uniform_backends: None,
) -> None:
  """A distributional-regression backend has no free-parameter count.

  ``MarginBase`` derives its criteria from ``_n_free``, which defaults to
  ``0.0`` -- so inheriting them would have ranked every backend by
  log-likelihood alone under a name that claims to penalize complexity.
  """
  margin = make_margin(register_uniform_backends)
  margin.fit(torch.linspace(-1.0, 1.0, 20, dtype=torch.float64))

  assert margin.n_parameters == 0.0

  for name in ("aic", "bic", "aicc"):
    with pytest.raises(NotImplementedError, match="free-parameter count"):
      getattr(margin, name)()


def test_fit_refuses_controls_it_cannot_honor(
  register_uniform_backends: None,
) -> None:
  """``supports_controls = False`` means refuse, not drop."""
  margin = make_margin(register_uniform_backends)

  with pytest.raises(ValueError, match="takes no `controls`"):
    margin.fit(
      torch.rand(20, dtype=torch.float64),
      object(),
    )


def test_fit_refuses_a_declared_atom(
  register_uniform_backends: None,
) -> None:
  """A declaration this margin cannot honor is refused, not dropped.

  ``MarginBase.fit`` gained ``var_type`` when weights moved into the
  controls, and a vine distribution hands each margin the type it resolved
  for that column. These backends estimate a conditional density and no
  atom, so accepting ``"d"`` would fit a continuous model under a discrete
  declaration and report nothing.
  """
  margin = make_margin(register_uniform_backends)

  with pytest.raises(ValueError, match="must model continuous variables"):
    margin.fit(torch.rand(20, dtype=torch.float64), var_type="d")


def test_fit_refuses_a_declared_bound(
  register_uniform_backends: None,
) -> None:
  """A declared support is refused for the same reason as a declared atom.

  The backends are fitted on a transform of the whole real line and impose
  no bound, so a bounded declaration would be silently unenforced.
  """
  margin = make_margin(register_uniform_backends)

  with pytest.raises(ValueError, match="must model unbounded variables"):
    margin.fit(torch.rand(20, dtype=torch.float64), support=(0.0, None))


def test_fit_rejects_a_one_dimensional_covariate(
  register_uniform_backends: None,
) -> None:
  """``(n,)`` is refused here, as it already was on the inherited methods.

  Before this contract was narrowed, a one-dimensional ``x`` was reshaped on
  this margin's own methods and refused on ``logpdf`` / ``cdf_left`` /
  ``loglik`` / ``sample``, which call ``prepare_covariates`` themselves. Two
  contracts on one object; this pins the one that is left.
  """
  margin = make_margin(register_uniform_backends)

  with pytest.raises(ValueError, match=r"must have shape \(n, p\)"):
    margin.fit(
      torch.linspace(-1.0, 1.0, 8, dtype=torch.float64),
      x=torch.ones(8, dtype=torch.float64),
    )


def test_grid_methods_refuse_a_one_dimensional_covariate(
  register_uniform_backends: None,
) -> None:
  """The Cartesian-grid methods take ``(n, p)`` like every other entry point.

  These six sites reshaped ``(n,)`` and, unlike the pointwise path, performed
  no layout check at all -- so this is the only test that can see the choice.
  The hermetic backends read a covariate's row count and nothing else, which
  is why no existing test constrained it.
  """
  margin = create_backend(
    "uniform-quantile",
    transform="identity",
    quantile_table_config=QuantileTableConfig(),
    eps=1e-6,
    device="cpu",
    batch_size=None,
  )
  margin.fit(torch.linspace(-1.0, 1.0, 12, dtype=torch.float64))
  y_grid = torch.linspace(-0.5, 0.5, 4, dtype=torch.float64)

  with pytest.raises(ValueError, match=r"must have shape \(n, p\)"):
    margin.pdf_grid(y_grid, x=torch.zeros(3, dtype=torch.float64))

  with pytest.raises(ValueError, match=r"must have shape \(n, p\)"):
    margin.cdf_grid(y_grid, x=torch.zeros(3, dtype=torch.float64))

  # And the wider contract survives: `p > 1` is still accepted, so this path
  # needs the general check rather than a single-covariate one.
  assert margin.pdf_grid(
    y_grid, x=torch.zeros((3, 2), dtype=torch.float64)
  ).shape == (
    3,
    4,
  )


def test_grid_methods_place_both_arguments(
  register_uniform_backends: None,
) -> None:
  """A NumPy grid and a NumPy covariate both arrive placed.

  These methods reach the backend without passing through the pointwise
  boundary, so before ``_grid_covariates`` a foreign array raised from inside
  the alpha grid rather than at the edge -- and placing one argument without
  the other split them across devices.
  """
  margin = create_backend(
    "uniform-quantile",
    transform="identity",
    quantile_table_config=QuantileTableConfig(),
    eps=1e-6,
    device="cpu",
    batch_size=None,
  )
  margin.fit(torch.linspace(-1.0, 1.0, 12, dtype=torch.float64))

  # NumPy on purpose. The declared contract is torch-only, so `ty` is right
  # to object; what is pinned here is that a foreign array is *placed* at the
  # boundary rather than raising from inside the alpha grid several frames on.
  out = margin.pdf_grid(
    numpy.linspace(-0.5, 0.5, 4),  # ty: ignore[invalid-argument-type]
    x=numpy.zeros((3, 1)),  # ty: ignore[invalid-argument-type]
  )

  assert out.dtype is torch.float64
  assert out.shape == (3, 4)
