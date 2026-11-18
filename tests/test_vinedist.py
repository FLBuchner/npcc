"""Tests for original-scale Rosenblatt vine distributions."""

from __future__ import annotations

import pytest
import torch
from pyvinecopulib import RVineStructure
from pyvinecopulib.core import VinedistBase

from npcc.core import (
  ConditionalMargin,
  FitControlsRosenblattVinecop,
  QuantileTableConfig,
  RosenblattVinecop,
  RosenblattVinedist,
  create_backend,
)
from npcc.core._placement import resolve_device
from tests.conftest import NormalQuantileBackend, hermetic_backend


def make_structure() -> RVineStructure:
  return RVineStructure.from_order([1, 2, 3])


def make_data() -> torch.Tensor:
  generator = torch.Generator(device="cpu")
  generator.manual_seed(42)
  return (
    2.0
    * torch.rand(
      (40, 3),
      generator=generator,
      dtype=torch.float64,
    )
    - 1.0
  )


def make_controls(
  backend: str = "normal-quantile",
) -> FitControlsRosenblattVinecop:
  return FitControlsRosenblattVinecop(
    backend=backend,
    device="cpu",
  )


def make_distribution(
  controls: FitControlsRosenblattVinecop,
) -> RosenblattVinedist:
  structure = make_structure()
  margins = [
    create_backend(
      controls.backend,
      transform="identity",
      quantile_table_config=controls.quantile_table_config
      or QuantileTableConfig(),
      eps=controls.eps,
      device=controls.device,
      batch_size=controls.batch_size,
      backend_kwargs=controls.backend_kwargs,
    )
    for _ in range(structure.dim)
  ]
  vinecop = RosenblattVinecop(
    None,
    structure,
    device=controls.device,
  )
  return RosenblattVinedist(vinecop, margins)


def fit_distribution(
  register_hermetic_backends: None,
  *,
  backend: str = "normal-quantile",
  x: torch.Tensor | None = None,
) -> RosenblattVinedist:
  controls = make_controls(backend)
  distribution = make_distribution(controls)
  return distribution.fit(make_data(), controls, x=x)


def test_rosenblatt_vinedist_subclasses_vinedist_base() -> None:
  assert issubclass(RosenblattVinedist, VinedistBase)


def test_constructor_binds_fixed_parts(
  register_hermetic_backends: None,
) -> None:
  controls = make_controls()
  distribution = make_distribution(controls)

  assert distribution.dim == 3
  assert len(distribution.margins) == 3
  assert isinstance(distribution.vinecop, RosenblattVinecop)
  assert distribution.vinecop.structure == make_structure()


def test_fit_fits_one_independent_margin_per_column(
  register_hermetic_backends: None,
) -> None:
  distribution = fit_distribution(register_hermetic_backends)

  assert all(
    isinstance(margin, ConditionalMargin) and margin.is_fitted
    for margin in distribution.margins
  )
  assert len({id(margin) for margin in distribution.margins}) == 3


def test_fit_uses_one_backend_for_margins_and_pairs(
  register_hermetic_backends: None,
) -> None:
  """The configured backend reaches the margins and every pair, not the default.

  Registered under a second name, so a pair that fell back to the controls'
  default backend would be told apart from one that read the configured one.
  """
  with hermetic_backend("normal-quantile-configured"):
    distribution = fit_distribution(
      register_hermetic_backends, backend="normal-quantile-configured"
    )

  for margin in distribution.margins:
    assert isinstance(margin, NormalQuantileBackend)

  vinecop = distribution.vinecop
  assert isinstance(vinecop, RosenblattVinecop)
  assert all(
    pair.backend == "normal-quantile-configured"
    for row in vinecop.pair_copulas
    for pair in row
  )


def test_marginal_cdf_is_probability_integral_transform(
  register_hermetic_backends: None,
) -> None:
  distribution = fit_distribution(register_hermetic_backends)
  y = torch.tensor(
    [
      [-1.0, 0.0, 1.0],
      [0.5, -0.5, 0.0],
    ],
    dtype=torch.float64,
  )

  result = distribution.marginal_cdf(y)
  expected = torch.distributions.Normal(0.0, 1.0).cdf(y)

  torch.testing.assert_close(result, expected)


def test_joint_pdf_matches_sklar_factorization(
  register_hermetic_backends: None,
) -> None:
  distribution = fit_distribution(register_hermetic_backends)
  y = torch.tensor(
    [
      [-0.5, 0.0, 0.5],
      [0.5, -0.5, 0.0],
    ],
    dtype=torch.float64,
  )

  u = distribution.marginal_cdf(y)
  copula_density = distribution.vinecop.pdf(u)
  margin_density = torch.distributions.Normal(0.0, 1.0).log_prob(y).exp()
  expected = copula_density * margin_density.prod(dim=1)

  torch.testing.assert_close(distribution.pdf(y), expected)


def test_conditional_fit_and_pdf_return_tensors(
  register_hermetic_backends: None,
) -> None:
  generator = torch.Generator(device="cpu")
  generator.manual_seed(43)
  x = torch.randn(
    (40, 2),
    generator=generator,
    dtype=torch.float64,
  )
  distribution = fit_distribution(register_hermetic_backends, x=x)

  result = distribution.pdf(make_data()[:5], x=x[:5])

  assert result.shape == (5,)
  assert result.dtype == torch.float64
  assert result.device.type == "cpu"


def test_rosenblatt_round_trip(
  register_hermetic_backends: None,
) -> None:
  distribution = fit_distribution(register_hermetic_backends)
  y = make_data()[:5]

  transformed = distribution.rosenblatt(y)
  recovered = distribution.inverse_rosenblatt(transformed)

  torch.testing.assert_close(recovered, y, atol=1e-8, rtol=1e-8)


def test_unconditional_sample_returns_original_scale_tensor(
  register_hermetic_backends: None,
) -> None:
  distribution = fit_distribution(register_hermetic_backends)

  first = distribution.sample(8, seeds=[42])
  second = distribution.sample(8, seeds=[42])

  assert first.shape == (8, 3)
  assert first.dtype == torch.float64
  assert first.device.type == "cpu"
  torch.testing.assert_close(first, second)
  assert torch.all((first >= -2.0) & (first <= 2.0))


def test_fit_places_foreign_data_on_the_controls_device(
  register_hermetic_backends: None,
) -> None:
  """Anything ``torch.as_tensor`` accepts is accepted, and placed.

  The placement comes from the controls, not from the data. Every other part
  of the lane is built from the controls -- the margins through
  ``create_backend``, the vine through its own ``device`` -- so reading it off
  the observations would leave the margins on the caller's device while the
  pair copulas went to the controls', and the first concatenation of the two
  would fail.
  """
  controls = make_controls()
  distribution = make_distribution(controls)

  # NumPy in, and integer-valued, so the dtype has to be imposed too.
  fitted = distribution.fit(
    make_data().numpy(),
    controls,
  )

  expected = resolve_device(controls.device)
  sample = fitted.sample(4, seeds=[1])

  assert sample.dtype is torch.float64
  assert sample.device == expected


def test_fit_rejects_dimension_mismatch(
  register_hermetic_backends: None,
) -> None:
  controls = make_controls()
  distribution = make_distribution(controls)
  y = torch.ones((10, 2), dtype=torch.float64)

  with pytest.raises(ValueError, match="has 3 variables"):
    distribution.fit(y, controls)


def test_fit_rejects_covariate_row_mismatch(
  register_hermetic_backends: None,
) -> None:
  controls = make_controls()
  distribution = make_distribution(controls)
  y = torch.ones((10, 3), dtype=torch.float64)
  x = torch.ones((9, 2), dtype=torch.float64)

  with pytest.raises(ValueError, match="x must have one row per observation"):
    distribution.fit(y, controls, x=x)


def test_fit_rejects_weights_carried_in_the_controls(
  register_hermetic_backends: None,
) -> None:
  """A vine that cannot weight must say so, or it returns the unweighted fit.

  Weights stopped being a ``fit`` argument and now ride in the controls, so
  the refusal moved onto ``RosenblattVinecop.supports_weights``. That flag
  defaults to ``True`` on ``VinecopBase``; without the override this call
  fitted and silently discarded the weights.
  """
  distribution = make_distribution(make_controls())

  class WeightedControls:
    weights = torch.ones(30, dtype=torch.float64)

    def to_dict(self) -> dict[str, object]:
      return {"backend": "normal-quantile", "device": "cpu"}

  with pytest.raises(TypeError, match="honors no observation weights"):
    distribution.fit(make_data(), WeightedControls())


def test_conditional_sample_returns_tensor(
  register_hermetic_backends: None,
) -> None:
  generator = torch.Generator(device="cpu")
  generator.manual_seed(43)
  x = torch.randn(
    (40, 2),
    generator=generator,
    dtype=torch.float64,
  )
  distribution = fit_distribution(register_hermetic_backends, x=x)

  result = distribution.sample(5, x=x[:5], seeds=[42])

  assert result.shape == (5, 3)
  assert result.dtype == torch.float64
  assert result.device.type == "cpu"


class _RecordingMargin(ConditionalMargin):
  """A margin that records the column it was fitted on, and nothing else.

  The hermetic normal backend ignores its data, so no fitted quantity of its
  varies by column -- which is exactly what a sharing bug would look like. This
  records the responses instead, so "each margin saw its own column" is
  checkable directly.
  """

  def __init__(self) -> None:
    super().__init__(transform="identity", device="cpu")
    self.seen: torch.Tensor | None = None

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    self.seen = z.clone()

  def _predict_quantiles(
    self, x: torch.Tensor, alphas: torch.Tensor
  ) -> torch.Tensor:
    return (2.0 * alphas - 1.0).unsqueeze(0).expand(x.shape[0], -1).clone()


def test_one_unfitted_margin_is_not_shared_across_variables(
  register_hermetic_backends: None,
) -> None:
  """One margin standing for every variable is copied, not aliased.

  Estimating a margin mutates it, so a single unfitted margin passed for a
  3-dimensional copula used to be fitted once per column with each fit
  overwriting the last, leaving every variable on the last column's fit.
  ``VinedistBase._bind_dist`` now ``unshare``s an unfitted one, and
  ``_reestimate`` does the same before refitting per column.
  """
  del register_hermetic_backends

  controls = make_controls()
  structure = make_structure()
  one_margin = _RecordingMargin()
  assert not one_margin.is_fitted

  distribution = RosenblattVinedist(
    RosenblattVinecop(None, structure, device=controls.device),
    one_margin,
  )

  margins = distribution.margins
  assert len(margins) == structure.dim
  assert len({id(margin) for margin in margins}) == structure.dim

  data = make_data()
  distribution.fit(data, controls)

  for j, margin in enumerate(distribution.margins):
    assert isinstance(margin, _RecordingMargin)
    assert margin.seen is not None
    torch.testing.assert_close(margin.seen, data[:, j])


def test_fit_threads_covariates_through_to_every_pair(
  register_hermetic_backends: None,
  pair_fit_widths: list[int | None],
) -> None:
  """A conditional distribution fit must reach the copula half, not just the margins.

  ``VinedistBase`` forwards ``x`` to the copula only when the copula declares
  ``supports_covariates``, and re-estimates in place through
  ``_reestimate_copula``. That is two hops that a downstream class can get
  wrong silently -- the fit would succeed, the margins would read ``x``, and
  every pair copula would be estimated unconditionally under a conditional
  name.

  The widths pin the non-simplified assembly as well: two tree-zero edges see
  the two external covariates, and the tree-one edge sees its one
  conditioning value first and the covariates after.
  """
  controls = make_controls()
  distribution = make_distribution(controls)
  data = make_data()
  x = torch.randn(
    (data.shape[0], 2),
    generator=torch.Generator(device="cpu").manual_seed(7),
    dtype=torch.float64,
  )

  distribution.fit(data, controls, x=x)

  assert pair_fit_widths == [2, 2, 3]


def test_fit_without_covariates_hands_the_pairs_none(
  register_hermetic_backends: None,
  pair_fit_widths: list[int | None],
) -> None:
  """The counterpart: an unconditional fit must not fabricate a covariate block."""
  controls = make_controls()
  distribution = make_distribution(controls)

  distribution.fit(make_data(), controls)

  assert pair_fit_widths == [None, None, 1]
