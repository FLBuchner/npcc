"""Tests for :class:`~npcc.core.ConditionalMargin`, on the hermetic backend.

Everything a backend inherits -- the contract with ``MarginBase``, the
read-out of its quantile table on every transform, the tails, chunking and
placement -- is tested here once, on the closed-form normal backend, so a
test of a concrete backend only has its adapter left to cover.
"""

from __future__ import annotations

import copy
import math
from typing import cast

import numpy
import pytest
import torch
from pyvinecopulib.core import MarginBase

from npcc.core import QuantileTableConfig, Transform
from tests.conftest import TAIL_MASS, NormalQuantileBackend, normal_density_y

#: The logit-scale response at the table's outermost knot, ``z = -L``.
FIRST_KNOT_Y = 1.0 / (1.0 + math.exp(QuantileTableConfig().normal_limit))


def make_margin(
  transform: Transform = "identity",
  *,
  batch_size: int | None = None,
  quantile_table_config: QuantileTableConfig | None = None,
) -> NormalQuantileBackend:
  """The hermetic normal backend, unfitted, on CPU."""
  return NormalQuantileBackend(
    transform=transform,
    device="cpu",
    batch_size=batch_size,
    quantile_table_config=quantile_table_config,
  )


def fitted_margin(
  transform: Transform = "identity", *, batch_size: int | None = None
) -> NormalQuantileBackend:
  """The hermetic normal backend, fitted; what it is fitted on is ignored."""
  margin = make_margin(transform, batch_size=batch_size)
  margin.fit(torch.full((10,), 0.5, dtype=torch.float64))
  return margin


def rows(n: int, p: int = 1) -> torch.Tensor:
  """``n`` constant conditioning rows of width ``p``."""
  return torch.zeros((n, p), dtype=torch.float64)


# ---------------------------------------------------------------------------
# The MarginBase contract
# ---------------------------------------------------------------------------


def test_conditional_margin_subclasses_margin_base() -> None:
  """The declared capabilities are the ones consumers read before a call.

  ``supports_controls`` and ``supports_weights`` default the other way on
  ``MarginBase``; inheriting them would accept a weighted fit and ignore the
  weights.
  """
  margin = make_margin()

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
def test_invalid_eps_is_rejected(eps: float) -> None:
  """An ``eps`` outside ``(0, 0.5)`` clips nothing or everything."""
  with pytest.raises(ValueError, match="eps must lie"):
    NormalQuantileBackend(eps=eps)


def test_unknown_transform_is_refused_at_construction() -> None:
  """An unknown transform is refused at construction, not at first use.

  Refusing it only when a response is first transformed let a margin be
  built, handed to a bicop and fitted for minutes before the name was read.
  """
  # Not a Transform on purpose: the refusal is what is under test.
  invalid = cast("Transform", "exp")

  with pytest.raises(ValueError, match="Unknown transform"):
    NormalQuantileBackend(transform=invalid)


@pytest.mark.parametrize(
  ("x", "expected_width"),
  [
    (None, 1),
    (torch.ones((8, 1)), 1),
    (torch.ones((8, 3)), 3),
  ],
)
def test_fit_constructs_expected_feature_matrix(
  monkeypatch: pytest.MonkeyPatch,
  x: torch.Tensor | None,
  expected_width: int,
) -> None:
  """An absent ``x`` becomes one constant column; a given one passes as is.

  The third-party regressors need at least one feature, so an unconditional
  margin must not hand them a zero-width matrix.
  """
  margin = make_margin()
  observed_shapes: list[tuple[int, ...]] = []
  original = margin._fit_model

  def record_fit(features: torch.Tensor, z: torch.Tensor) -> None:
    observed_shapes.append(tuple(features.shape))
    original(features, z)

  monkeypatch.setattr(margin, "_fit_model", record_fit)

  result = margin.fit(torch.linspace(-1.0, 1.0, 8, dtype=torch.float64), x=x)

  assert result is margin
  assert margin.is_fitted is True
  assert margin.nobs == 8
  assert observed_shapes == [(8, expected_width)]


def test_inherited_margin_operations() -> None:
  """``logpdf``, ``cdf_left`` and ``loglik`` come from the base and agree."""
  margin = fitted_margin()
  y = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
  expected_logpdf = torch.distributions.Normal(0.0, 1.0).log_prob(y)

  torch.testing.assert_close(margin.logpdf(y), expected_logpdf)
  torch.testing.assert_close(margin.cdf_left(y), margin.cdf(y))
  torch.testing.assert_close(margin.loglik(y), expected_logpdf.sum())


def test_sample_is_reproducible() -> None:
  """A seeded draw repeats, unconditionally and conditionally."""
  margin = fitted_margin()

  first = margin.sample(10, seeds=[42])
  torch.testing.assert_close(first, margin.sample(10, seeds=[42]))
  assert margin.sample(5, x=rows(5, 2), seeds=[42]).shape == (5,)


def test_copied_margins_are_independent() -> None:
  """A deep copy shares no fitted state with its original."""
  first = make_margin()
  second = copy.deepcopy(first)

  first.fit(torch.linspace(-1.0, 1.0, 20, dtype=torch.float64))

  assert first.is_fitted is True
  assert second.is_fitted is False


def test_information_criteria_are_refused() -> None:
  """A distributional-regression backend has no free-parameter count.

  ``MarginBase`` derives its criteria from ``_n_free``, which defaults to
  ``0.0`` -- so inheriting them would have ranked every backend by
  log-likelihood alone under a name that claims to penalize complexity.
  """
  margin = fitted_margin()

  for name in ("aic", "bic", "aicc"):
    with pytest.raises(NotImplementedError, match="free-parameter count"):
      getattr(margin, name)()


def test_fit_refuses_controls_it_cannot_honor() -> None:
  """``supports_controls = False`` means refuse, not drop."""
  with pytest.raises(ValueError, match="takes no `controls`"):
    make_margin().fit(torch.rand(20, dtype=torch.float64), object())


def test_fit_refuses_a_declared_atom() -> None:
  """A declaration this margin cannot honor is refused, not dropped.

  A vine distribution hands each margin the type it resolved for that column.
  These backends estimate a conditional density and no atom, so accepting
  ``"d"`` would fit a continuous model under a discrete declaration and report
  nothing.
  """
  with pytest.raises(ValueError, match="must model continuous variables"):
    make_margin().fit(torch.rand(20, dtype=torch.float64), var_type="d")


def test_fit_refuses_a_declared_bound() -> None:
  """A support other than the transform's is refused, like a declared atom.

  The identity transform maps onto the whole real line and imposes no bound,
  so a bounded declaration would be silently unenforced.
  """
  with pytest.raises(
    ValueError, match=r"must model variables on \(-inf, inf\)"
  ):
    make_margin().fit(torch.rand(20, dtype=torch.float64), support=(0.0, None))


@pytest.mark.parametrize(
  ("transform", "support"),
  [("log", (0.0, None)), ("log", (0.0, math.inf)), ("logit", (0.0, 1.0))],
)
def test_fit_accepts_the_support_its_transform_fixes(
  transform: Transform,
  support: tuple[float, float | None],
) -> None:
  """A bound the transform enforces is a declaration the margin can honor.

  A vine distribution hands each margin the support it resolved for that
  column, so refusing a matching one would make a nonnegative column
  unfittable even with the transform that models it.
  """
  margin = make_margin(transform)

  margin.fit(0.1 + 0.8 * torch.rand(20, dtype=torch.float64), support=support)

  assert margin.is_fitted


# ---------------------------------------------------------------------------
# Layout of the arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["pdf", "cdf", "icdf", "pdf_grid"])
def test_evaluation_before_fit_raises(method: str) -> None:
  """Every evaluation path checks for a fit, the grid ones included."""
  margin = make_margin()
  values = torch.tensor([0.5], dtype=torch.float64)

  with pytest.raises(RuntimeError, match="not fitted"):
    getattr(margin, method)(values, x=rows(1))


def test_fit_rejects_mismatched_rows() -> None:
  """A covariate matrix with another row count than ``y`` is refused."""
  with pytest.raises(ValueError, match="one row per observation"):
    make_margin().fit(
      torch.linspace(-1.0, 1.0, 8, dtype=torch.float64), x=rows(7, 2)
    )


@pytest.mark.parametrize("method", ["pdf", "cdf", "icdf"])
def test_pointwise_methods_reject_mismatched_rows(method: str) -> None:
  """Each pointwise method checks its values against its rows."""
  with pytest.raises(ValueError, match="one row per observation"):
    getattr(fitted_margin(), method)(
      torch.full((6,), 0.5, dtype=torch.float64), x=rows(5)
    )


def test_fit_rejects_a_one_dimensional_covariate() -> None:
  """``(n,)`` is refused here, as it already was on the inherited methods.

  Before this contract was narrowed, a one-dimensional ``x`` was reshaped on
  this margin's own methods and refused on ``logpdf`` / ``cdf_left`` /
  ``loglik`` / ``sample``, which call ``prepare_covariates`` themselves. Two
  contracts on one object; this pins the one that is left. Matching on the
  shape clause rather than on "one row per observation": the row-count
  refusal contains that phrase too.
  """
  with pytest.raises(ValueError, match=r"must have shape \(n, p\)"):
    make_margin().fit(
      torch.linspace(-1.0, 1.0, 8, dtype=torch.float64),
      x=torch.ones(8, dtype=torch.float64),
    )


def test_grid_methods_refuse_a_one_dimensional_covariate() -> None:
  """The Cartesian-grid methods take ``(n, p)`` like every other entry point.

  These sites reshaped ``(n,)`` and, unlike the pointwise path, performed no
  layout check at all. The hermetic backend reads a covariate's row count and
  nothing else, which is why no other test constrains it.
  """
  margin = fitted_margin()
  y_grid = torch.linspace(-0.5, 0.5, 4, dtype=torch.float64)

  for method in (margin.pdf_grid, margin.cdf_grid):
    with pytest.raises(ValueError, match=r"must have shape \(n, p\)"):
      method(y_grid, x=torch.zeros(3, dtype=torch.float64))

  # And the wider contract survives: `p > 1` is still accepted, so this path
  # needs the general check rather than a single-covariate one.
  assert margin.pdf_grid(y_grid, x=rows(3, 2)).shape == (3, 4)
  assert margin.cdf_grid(y_grid, x=rows(3, 2)).shape == (3, 4)


def test_grid_refuses_an_empty_grid() -> None:
  """An empty grid is a caller error, not an empty answer."""
  with pytest.raises(ValueError, match="at least one value"):
    fitted_margin().pdf_grid(torch.empty(0, dtype=torch.float64), x=rows(1))


def test_icdf_rejects_a_level_outside_the_unit_interval() -> None:
  """A level of ``0`` or ``1`` has no finite quantile on an open support."""
  with pytest.raises(ValueError, match="strictly inside"):
    fitted_margin("logit").icdf(
      torch.tensor([0.5, 1.0], dtype=torch.float64), x=rows(2)
    )


# ---------------------------------------------------------------------------
# The read-out, per transform
# ---------------------------------------------------------------------------


def test_identity_margin_is_the_normal() -> None:
  """A normal table on the identity scale is read back exactly."""
  margin = fitted_margin()
  y = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
  normal = torch.distributions.Normal(0.0, 1.0)
  p = normal.cdf(y)

  torch.testing.assert_close(margin.pdf(y), normal.log_prob(y).exp())
  torch.testing.assert_close(margin.cdf(y), p)
  torch.testing.assert_close(margin.icdf(p), y)


def test_logit_margin_is_the_logit_normal() -> None:
  """Under logit the margin is ``sigmoid(Normal(0, 1))``, Jacobian included."""
  margin = fitted_margin("logit")
  y = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
  p = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)

  torch.testing.assert_close(margin.pdf(y, x=rows(3)), normal_density_y(y))
  torch.testing.assert_close(
    margin.cdf(y, x=rows(3)), torch.special.ndtr(torch.logit(y))
  )
  torch.testing.assert_close(
    margin.icdf(p, x=rows(3)), torch.sigmoid(torch.special.ndtri(p))
  )


def test_probit_margin_is_uniform_tails_included() -> None:
  """A normal table under probit is ``Uniform(0, 1)``, tails included.

  Inside the table the reconstruction is exact for a normal; beyond it the
  uniform tail toward a finite end has the same density as the truth, so the
  whole distribution is pinned here, not only its middle.
  """
  margin = fitted_margin("probit")
  y = torch.tensor([1e-4, 0.2, 0.5, 0.8, 1.0 - 1e-4], dtype=torch.float64)

  torch.testing.assert_close(margin.pdf(y, x=rows(5)), torch.ones_like(y))
  torch.testing.assert_close(margin.cdf(y, x=rows(5)), y)
  torch.testing.assert_close(margin.icdf(y, x=rows(5)), y)


def make_log_margin() -> NormalQuantileBackend:
  """The hermetic normal quantile table on the log scale: a lognormal."""
  margin = make_margin("log")
  margin.fit(torch.ones(10, dtype=torch.float64))
  return margin


def test_log_margin_refuses_nonpositive_fit_responses() -> None:
  """An exact zero would be fitted as ``log(0) = -inf``.

  That is an atom the margin cannot model, so it is refused with what to do
  instead rather than handed to a backend as an infinite target.
  """
  y = torch.tensor([0.5, 0.0, 2.0, -1.0], dtype=torch.float64)

  with pytest.raises(ValueError, match="2 of 4 at or below 0"):
    make_margin("log").fit(y)


def test_log_margin_lives_on_the_positive_half_line() -> None:
  """A log margin is a distribution on ``[0, inf)``, boundary included.

  Its support is reported as such, ``y = 0`` evaluates to ``F = 0`` with a
  finite density rather than ``log(0)``, the interior is the exact lognormal,
  and the whole density integrates to one.
  """
  margin = make_log_margin()

  assert margin.support == (0.0, math.inf)

  at_zero = torch.zeros(1, dtype=torch.float64)
  assert margin.cdf(at_zero, x=rows(1)).item() == 0.0
  assert 0.0 < margin.pdf(at_zero, x=rows(1)).item() < math.inf

  y = torch.tensor([0.5, 1.0, 3.0], dtype=torch.float64)
  lognormal = torch.exp(-0.5 * y.log().square()) / (y * math.sqrt(2 * math.pi))
  torch.testing.assert_close(margin.pdf(y, x=rows(3)), lognormal)

  grid = torch.cat(
    [
      torch.linspace(0.0, 0.05, 20_001, dtype=torch.float64),
      torch.logspace(math.log10(0.05), 6, 400_000, dtype=torch.float64)[1:],
    ]
  )
  density = margin.pdf_grid(grid, x=rows(1))[0]
  mass = torch.trapezoid(density, grid)
  torch.testing.assert_close(
    mass, torch.tensor(1.0, dtype=torch.float64), atol=1e-4, rtol=0.0
  )


def test_cdf_icdf_round_trip_with_covariates() -> None:
  """``cdf(icdf(p | x) | x) == p`` row by row, with several covariates."""
  margin = make_margin()
  train_y = torch.linspace(-1.0, 1.0, 20, dtype=torch.float64)
  margin.fit(train_y, x=torch.column_stack((train_y, train_y.square())))

  p = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
  x = torch.tensor([[-1.0, 1.0], [0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

  torch.testing.assert_close(margin.cdf(margin.icdf(p, x=x), x=x), p)


# ---------------------------------------------------------------------------
# Beyond the table
# ---------------------------------------------------------------------------


def test_pdf_beyond_the_table_is_its_tail_mass() -> None:
  """Beyond the table the density is the tail mass spread to the bound.

  The table used to give density zero there, so any observation in the outer
  ``2 * alpha_min`` of the distribution had log-likelihood ``-inf``. The
  logit scale is bounded, so the mass beyond the outermost knot is spread
  uniformly between it and ``0`` (or ``1``) -- on the grid path as on the
  pointwise one.
  """
  margin = fitted_margin("logit")
  y = torch.tensor([1e-5, 0.02, 0.98, 1.0 - 1e-5], dtype=torch.float64)
  expected = TAIL_MASS / FIRST_KNOT_Y

  torch.testing.assert_close(
    margin.pdf(y, x=rows(4)), torch.full_like(y, expected)
  )
  torch.testing.assert_close(
    margin.pdf_grid(y, x=rows(3)), torch.full((3, 4), expected).double()
  )


def test_cdf_beyond_the_table_runs_linearly_to_the_bound() -> None:
  """Beyond the table the distribution function keeps moving.

  It used to stop at ``alpha_min`` and ``alpha_max``, so the h-function was
  flat over the outer tails and its inverse put an atom on the table's edge.
  """
  margin = fitted_margin("logit")
  y = torch.tensor([0.01, 0.02, 0.98, 0.99], dtype=torch.float64)

  lower = TAIL_MASS * y[:2] / FIRST_KNOT_Y
  upper = 1.0 - TAIL_MASS * (1.0 - y[2:]) / FIRST_KNOT_Y

  torch.testing.assert_close(
    margin.cdf(y, x=rows(4)), torch.cat([lower, upper])
  )


def test_icdf_beyond_the_table_inverts_the_tail() -> None:
  """A level beyond the table maps inside the tail, not onto its edge.

  Clamping it to the outermost knot put every such draw on one value, an
  atom of mass ``alpha_min`` at each end of every simulated margin.
  """
  margin = fitted_margin("logit")
  p = torch.tensor([1e-6, 1e-4, 1.0 - 1e-4], dtype=torch.float64)

  lower = p[:2] / TAIL_MASS * FIRST_KNOT_Y
  upper = 1.0 - (1.0 - p[2:]) / TAIL_MASS * FIRST_KNOT_Y

  torch.testing.assert_close(
    margin.icdf(p, x=rows(3)), torch.cat([lower, upper])
  )


# ---------------------------------------------------------------------------
# The Cartesian grid and chunking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["pdf", "cdf"])
def test_grid_matches_pointwise(method: str) -> None:
  """``*_grid`` is the pointwise method on the Cartesian product."""
  margin = fitted_margin("logit")
  x = rows(3)
  y_grid = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

  grid = getattr(margin, f"{method}_grid")(y_grid, x=x)
  tiled = getattr(margin, method)(
    y_grid.repeat(3), x=x.repeat_interleave(3, dim=0)
  ).reshape(3, 3)

  torch.testing.assert_close(grid, tiled)


def test_chunked_grid_matches_unchunked() -> None:
  """The batch size changes how many tables are predicted, not the answer."""
  margin = fitted_margin("logit")
  y_grid = torch.linspace(0.2, 0.8, 9, dtype=torch.float64)

  torch.testing.assert_close(
    margin.pdf_grid(y_grid, x=rows(7)),
    margin.pdf_grid(y_grid, x=rows(7), batch_size=3),
  )


@pytest.mark.parametrize(
  ("method", "n_rows", "n_values", "batch_size"),
  [
    ("pdf", 17, 17, 4),
    ("cdf", 17, 17, 4),
    ("pdf_grid", 6, 25, 1000),
    ("cdf_grid", 5, 30, 2),
  ],
)
def test_one_prediction_per_chunk_of_rows(
  monkeypatch: pytest.MonkeyPatch,
  method: str,
  n_rows: int,
  n_values: int,
  batch_size: int,
) -> None:
  """A table is predicted once per chunk of conditioning rows, never per cell.

  The grid methods are the speed invariant every pair copula leans on: a
  per-cell path would call the backend ``n_rows * n_values`` times.
  """
  margin = fitted_margin("logit", batch_size=batch_size)
  calls = 0
  original = margin._predict_quantiles

  def spy(x: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    nonlocal calls
    calls += 1
    return original(x, alphas)

  monkeypatch.setattr(margin, "_predict_quantiles", spy)
  values = torch.linspace(0.15, 0.85, n_values, dtype=torch.float64)
  # Distinct rows, so the count is not shortened by deduplication.
  x = torch.arange(n_rows, dtype=torch.float64).reshape(-1, 1)

  getattr(margin, method)(values, x=x)

  assert calls == math.ceil(n_rows / batch_size)


@pytest.mark.parametrize("method", ["pdf", "cdf", "icdf"])
def test_a_repeated_row_reaches_the_backend_once(
  monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
  """A conditioning row repeated across a query is predicted once.

  A pair copula's grid repeats each conditioning row once per point of the
  other axis; predicting every repeat would multiply the backend's work by
  the grid size. The answer must be the one the repeats would have given.
  """
  margin = fitted_margin("logit")
  seen: list[int] = []
  original = margin._predict_quantiles

  def shifted(x: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    # Quantiles that depend on the row, so a row handed back to the wrong
    # observation changes the answer.
    seen.append(x.shape[0])
    return original(x, alphas) + 3.0 * x[:, :1]

  monkeypatch.setattr(margin, "_predict_quantiles", shifted)
  distinct = torch.tensor([[0.9], [0.1], [0.5]], dtype=torch.float64)
  x = distinct.repeat(4, 1)
  values = torch.linspace(0.15, 0.85, 12, dtype=torch.float64)
  one_at_a_time = getattr(margin, method)
  expected = torch.cat(
    [one_at_a_time(values[i : i + 1], x=x[i : i + 1]) for i in range(12)]
  )

  seen.clear()
  got = getattr(margin, method)(values, x=x)

  assert sum(seen) == distinct.shape[0]
  # A misrouted row is off by order one; the tolerance only absorbs the
  # smoother's matrix product rounding differently for one row than for
  # twelve, which some BLAS builds do.
  torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def test_grid_methods_place_both_arguments() -> None:
  """A NumPy grid and a NumPy covariate both arrive placed.

  These methods reach the backend without passing through the pointwise
  boundary, so before ``_grid_covariates`` a foreign array raised from inside
  the alpha grid rather than at the edge -- and placing one argument without
  the other split them across devices.
  """
  # NumPy on purpose. The declared contract is torch-only, so `ty` is right
  # to object; what is pinned here is that a foreign array is *placed* at the
  # boundary rather than raising from inside the alpha grid several frames on.
  out = fitted_margin().pdf_grid(
    numpy.linspace(-0.5, 0.5, 4),  # ty: ignore[invalid-argument-type]
    x=numpy.zeros((3, 1)),  # ty: ignore[invalid-argument-type]
  )

  assert out.dtype is torch.float64
  assert out.shape == (3, 4)


@pytest.mark.parametrize("method", ["pdf", "cdf", "icdf"])
def test_pointwise_methods_place_the_values(method: str) -> None:
  """The values a pointwise method evaluates are placed, not only its rows.

  The table is predicted on the margin's device from the placed conditioning
  rows, so unplaced values met it on another device -- a CPU response against
  a CUDA margin failed inside ``searchsorted`` -- or, as a NumPy array, failed
  before reaching it at all.
  """
  values = numpy.array([0.25, 0.5, 0.75])

  # NumPy on purpose, as above: what is pinned is that it is *placed*. Looked
  # up by name, so no type check reads the argument.
  out = getattr(fitted_margin(), method)(values)

  assert out.dtype is torch.float64
  assert out.shape == (3,)
