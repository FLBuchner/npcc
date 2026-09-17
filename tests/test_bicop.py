"""Tests for the Torch-native Rosenblatt pair-copula estimator."""

from __future__ import annotations

import numpy
import pytest
import torch
from pyvinecopulib.core import BicopBase, BicopLike

from npcc.core.backends.tabpfn_criterion import TabPFNCriterionBackend
from npcc.core.backends.tabpfn_quantile import TabPFNQuantileBackend
from npcc.core.bicop import RosenblattBicop, _sinkhorn_project
from npcc.core.controls import (
  FitControlsRosenblattBicop,
  FitControlsRosenblattVinecop,
)


def random_uv(n: int = 30, *, seed: int = 0) -> torch.Tensor:
  generator = torch.Generator(device="cpu").manual_seed(seed)
  return 0.15 + 0.7 * torch.rand(
    (n, 2),
    generator=generator,
    dtype=torch.float64,
  )


def make_controls(
  method: str = "criterion",
  *,
  transform: str = "logit",
  batch_size: int | None = None,
  sinkhorn_iters: int | None = None,
  cdf_n_int: int = 12,
) -> FitControlsRosenblattBicop:
  return FitControlsRosenblattBicop(
    backend=f"tabpfn-{method}",
    transform=transform,  # ty: ignore[invalid-argument-type]
    device="cpu",
    batch_size=batch_size,
    sinkhorn_iters=sinkhorn_iters,
    projection_grid_size=21,
    cdf_n_int=cdf_n_int,
  )


def fit_bicop(
  patch_uniform: None,
  method: str = "criterion",
  *,
  x: torch.Tensor | None = None,
  transform: str = "logit",
  sinkhorn_iters: int | None = None,
  cdf_n_int: int = 12,
) -> RosenblattBicop:
  del patch_uniform
  model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))
  return model.fit(
    random_uv(),
    make_controls(
      method,
      transform=transform,
      sinkhorn_iters=sinkhorn_iters,
      cdf_n_int=cdf_n_int,
    ),
    x=x,
  )


class TestRosenblattBicopConstruction:
  def test_implements_bicop_contract(self) -> None:
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    assert isinstance(model, BicopBase)
    assert isinstance(model, BicopLike)

  def test_default_backend_is_criterion(self) -> None:
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    assert model.backend == "tabpfn-criterion"
    assert model.transform == "logit"
    assert isinstance(model.v_given_ux_, TabPFNCriterionBackend)
    assert isinstance(model.u_given_vx_, TabPFNCriterionBackend)

  def test_fit_controls_replace_backend_configuration(
    self,
    patch_uniform: None,
  ) -> None:
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))
    controls = make_controls(
      "quantiles",
      transform="identity",
      batch_size=17,
    )

    model.fit(random_uv(), controls)

    assert model.backend == "tabpfn-quantiles"
    assert model.transform == "identity"
    assert model.batch_size == 17
    assert isinstance(model.v_given_ux_, TabPFNQuantileBackend)
    assert isinstance(model.u_given_vx_, TabPFNQuantileBackend)

  def test_default_batch_size_on_cpu_is_400(self) -> None:
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    assert model.batch_size == 400
    assert model.v_given_ux_.batch_size == 400


@pytest.mark.parametrize("method", ["criterion", "quantiles"])
class TestRosenblattBicopValidation:
  @pytest.mark.parametrize(
    "uv",
    [
      torch.tensor([0.2, 0.3], dtype=torch.float64),
      torch.ones((2, 1), dtype=torch.float64),
      torch.ones((2, 3), dtype=torch.float64),
    ],
  )
  def test_fit_rejects_invalid_shape(
    self,
    patch_uniform: None,
    method: str,
    uv: torch.Tensor,
  ) -> None:
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
      model.fit(uv, make_controls(method))

  def test_fit_rejects_covariate_length_mismatch(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    with pytest.raises(ValueError, match="one row per observation"):
      model.fit(
        random_uv(10),
        make_controls(method),
        x=torch.zeros((5, 2), dtype=torch.float64),
      )

  def test_pdf_rejects_covariate_length_mismatch(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)

    with pytest.raises(ValueError, match="one row per observation"):
      model.pdf(
        random_uv(10),
        x=torch.zeros((5, 2), dtype=torch.float64),
      )

  def test_fit_rejects_boundary_values(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    uv = torch.tensor(
      [[0.5, 0.3], [0.0, 0.4]],
      dtype=torch.float64,
    )
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    with pytest.raises(ValueError, match="strictly inside"):
      model.fit(uv, make_controls(method))


def test_sinkhorn_project_rejects_nonpositive_iterations() -> None:
  density = torch.ones((3, 3), dtype=torch.float64)
  weights = torch.ones(3, dtype=torch.float64)

  with pytest.raises(ValueError, match="n_iters must be positive"):
    _sinkhorn_project(density, weights, weights, 0)


def test_trapezoidal_weights_singleton_grid() -> None:
  grid = torch.tensor([0.25], dtype=torch.float64)

  result = RosenblattBicop._trapezoidal_weights(grid)

  torch.testing.assert_close(
    result,
    torch.tensor([1.0], dtype=torch.float64),
  )


@pytest.mark.parametrize("method", ["criterion", "quantiles"])
class TestRosenblattBicopEvaluation:
  def test_pdf_returns_positive_tensor(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)
    uv = random_uv(12, seed=1)

    result = model.pdf(uv)

    assert result.shape == (12,)
    assert result.dtype == torch.float64
    assert result.device.type == "cpu"
    assert torch.isfinite(result).all()
    assert torch.all(result > 0.0)

  def test_pdf_accepts_covariates(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    train_x = torch.zeros((30, 2), dtype=torch.float64)
    model = fit_bicop(patch_uniform, method, x=train_x)

    result = model.pdf(
      random_uv(8, seed=2),
      x=torch.zeros((8, 2), dtype=torch.float64),
    )

    assert result.shape == (8,)

  def test_log_pdf_matches_pdf(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)
    uv = random_uv(8, seed=3)

    torch.testing.assert_close(model.log_pdf(uv), torch.log(model.pdf(uv)))

  def test_loglik_matches_summed_log_pdf(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)
    uv = random_uv(8, seed=4)

    torch.testing.assert_close(model.loglik(uv), model.log_pdf(uv).sum())

  def test_pdf_grid_matches_pointwise_pdf(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)
    u_grid = torch.linspace(0.25, 0.75, 4, dtype=torch.float64)
    v_grid = torch.linspace(0.3, 0.7, 5, dtype=torch.float64)

    result = model.pdf_grid(u_grid, v_grid)
    uv = torch.column_stack(
      (
        u_grid.repeat_interleave(len(v_grid)),
        v_grid.repeat(len(u_grid)),
      )
    )
    expected = model.pdf(uv).reshape(len(u_grid), len(v_grid))

    assert result.shape == (4, 5)
    torch.testing.assert_close(result, expected)

  def test_pdf_grid_rejects_multiple_covariate_rows(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)
    grid = torch.linspace(0.2, 0.8, 4, dtype=torch.float64)

    with pytest.raises(ValueError, match=r"shape \(p,\) or \(1, p\)"):
      model.pdf_grid(
        grid,
        grid,
        x_row=torch.zeros((2, 1), dtype=torch.float64),
      )

  def test_hfuncs_return_probabilities(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)
    uv = random_uv(10, seed=5)

    first = model.hfunc1(uv)
    second = model.hfunc2(uv)

    assert first.shape == (10,)
    assert second.shape == (10,)
    assert torch.all((first > 0.0) & (first < 1.0))
    assert torch.all((second > 0.0) & (second < 1.0))

  def test_inverse_hfuncs_invert_forward_hfuncs(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)
    uv = random_uv(10, seed=6)

    first_input = torch.column_stack((uv[:, 0], model.hfunc1(uv)))
    second_input = torch.column_stack((model.hfunc2(uv), uv[:, 1]))

    torch.testing.assert_close(
      model.hinv1(first_input),
      uv[:, 1],
      atol=2e-2,
      rtol=2e-2,
    )
    torch.testing.assert_close(
      model.hinv2(second_input),
      uv[:, 0],
      atol=2e-2,
      rtol=2e-2,
    )

  def test_cdf_returns_probabilities(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method, cdf_n_int=16)

    result = model.cdf(random_uv(6, seed=7))

    assert result.shape == (6,)
    assert torch.all((result >= 0.0) & (result <= 1.0))

  def test_cdf_grid_matches_pointwise_cdf(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method, cdf_n_int=32)
    grid = torch.linspace(0.25, 0.75, 4, dtype=torch.float64)

    result = model.cdf_grid(grid, grid, n_int=32)
    uv = torch.column_stack(
      (
        grid.repeat_interleave(len(grid)),
        grid.repeat(len(grid)),
      )
    )
    expected = model.cdf(uv).reshape(len(grid), len(grid))

    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("method", ["criterion", "quantiles"])
class TestRosenblattBicopSampling:
  def test_sample_is_seeded(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)

    first = model.sample(10, seeds=[42])
    second = model.sample(10, seeds=[42])

    assert first.shape == (10, 2)
    assert first.dtype == torch.float64
    assert first.device.type == "cpu"
    torch.testing.assert_close(first, second)

  def test_qrng_sample_is_seeded(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)

    first = model.sample(10, qrng=True, seeds=[42])
    second = model.sample(10, qrng=True, seeds=[42])

    torch.testing.assert_close(first, second)

  def test_sample_accepts_covariates(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    train_x = torch.zeros((30, 2), dtype=torch.float64)
    model = fit_bicop(patch_uniform, method, x=train_x)

    result = model.sample(
      8,
      x=torch.zeros((8, 2), dtype=torch.float64),
      seeds=[42],
    )

    assert result.shape == (8, 2)

  def test_flip_remains_unsupported(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)

    with pytest.raises(NotImplementedError):
      model.flip()


class TestRosenblattBicopTau:
  def test_tau_is_deterministic_and_bounded(
    self,
    patch_uniform: None,
  ) -> None:
    model = fit_bicop(patch_uniform)

    first = model.tau(n=100, seeds=[42])
    second = model.tau(n=100, seeds=[42])

    assert -1.0 <= first <= 1.0
    assert first == pytest.approx(second)

  def test_tau_rejects_small_sample(self, patch_uniform: None) -> None:
    model = fit_bicop(patch_uniform)

    with pytest.raises(ValueError, match="at least 10"):
      model.tau(n=9)

  def test_tau_rejects_multiple_covariate_rows(
    self,
    patch_uniform: None,
  ) -> None:
    model = fit_bicop(patch_uniform)

    with pytest.raises(ValueError, match=r"shape \(p,\) or \(1, p\)"):
      model.tau(torch.zeros((2, 1), dtype=torch.float64))


class TestSinkhornProjection:
  def test_projection_preserves_shape_and_nonnegativity(
    self,
    patch_uniform: None,
  ) -> None:
    model = fit_bicop(patch_uniform, sinkhorn_iters=3)
    uv = random_uv(10, seed=8)

    result = model.pdf(uv)

    assert result.shape == (10,)
    assert torch.all(result >= 0.0)

  def test_projection_grid_is_cached(self, patch_uniform: None) -> None:
    model = fit_bicop(patch_uniform, sinkhorn_iters=3)

    assert model._u_grid_borders_ is not None
    assert model._v_grid_borders_ is not None

    first_u = model._u_grid_borders_
    first_v = model._v_grid_borders_
    model._get_grid_borders()

    assert model._u_grid_borders_ is not first_u
    assert model._v_grid_borders_ is not first_v

  def test_projection_grid_pdf_is_finite(self, patch_uniform: None) -> None:
    model = fit_bicop(patch_uniform, sinkhorn_iters=3)
    grid = torch.linspace(0.2, 0.8, 8, dtype=torch.float64)

    result = model.pdf_grid(grid, grid)

    assert result.shape == (8, 8)
    assert torch.isfinite(result).all()
    assert torch.all(result >= 0.0)


class TestPlacement:
  """Inputs are brought onto the estimator's dtype and device.

  ``PlacementMixin._prep`` infers a placement from arrays the object holds,
  and finds none on these estimators -- so it returned its argument untouched,
  and the half-placement written in its stead placed the copula arguments and
  left the covariates alone. On CUDA that met a host ``x`` with a device ``u``
  inside a ``column_stack``; on every device it let a float32 input stay
  float32 under a float64 contract.
  """

  def test_prep_normalizes_dtype_and_device(self, patch_uniform: None) -> None:
    model = fit_bicop(patch_uniform)

    placed = model._prep(numpy.asarray([[0.3, 0.4]], dtype=numpy.float32))

    assert placed.dtype is torch.float64
    assert placed.device == model._device

  @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
  def test_pdf_returns_float64_whatever_it_is_given(
    self, patch_uniform: None, dtype: torch.dtype
  ) -> None:
    model = fit_bicop(patch_uniform)
    uv = torch.tensor([[0.3, 0.4], [0.6, 0.7]], dtype=dtype)

    assert model.pdf(uv).dtype is torch.float64

  def test_covariates_are_placed_alongside_the_observations(
    self, patch_uniform: None
  ) -> None:
    """A float32 covariate matrix must not poison the feature matrix.

    `u` and `x` are concatenated, so placing only one of them is what a
    device mismatch -- and, on one device, a silent downcast -- comes from.
    """
    x = torch.zeros((30, 2), dtype=torch.float64)
    model = fit_bicop(patch_uniform, x=x)

    out = model.pdf(
      torch.tensor([[0.3, 0.4]], dtype=torch.float32),
      x=torch.zeros((1, 2), dtype=torch.float32),
    )

    assert out.dtype is torch.float64

  def test_the_inherited_plot_runs(self, patch_uniform: None) -> None:
    """``BicopBase.plot`` manufactures a NumPy grid and places it via ``_prep``.

    The one path that reaches these estimators with a foreign array type, and
    the reason ``_prep`` returning its argument untouched was survivable
    before: the density evaluation re-placed it. Nothing re-places the grid.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    model = fit_bicop(patch_uniform)
    model.plot(grid_size=8)

    matplotlib.pyplot.close("all")

  def test_cdf_grid_accumulates_in_float64(self, patch_uniform: None) -> None:
    """The cumulative integral used to land in a dtype-less ``torch.zeros``."""
    model = fit_bicop(patch_uniform)
    grid = torch.linspace(0.1, 0.9, 5, dtype=torch.float64)

    assert model.cdf_grid(grid, grid, n_int=8).dtype is torch.float64


class TestControlsContract:
  def test_tau_accepts_a_one_dimensional_covariate_row(
    self, patch_uniform: None
  ) -> None:
    """``(p,)`` is documented, and was rejected.

    The reshape tested ``x_row.dim == 1``, comparing a bound method to an
    int, so a 1-D row was never reshaped and then tripped the shape check.
    """
    x = torch.zeros((30, 1), dtype=torch.float64)
    model = fit_bicop(patch_uniform, x=x)

    flat = model.tau(torch.zeros(1, dtype=torch.float64), n=50)
    column = model.tau(torch.zeros((1, 1), dtype=torch.float64), n=50)

    assert flat == pytest.approx(column)

  def test_an_array_in_the_controls_slot_names_the_argument_order(
    self, patch_uniform: None
  ) -> None:
    """``fit(uv, x)`` binds covariates to ``controls``; say so."""
    del patch_uniform
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    with pytest.raises(TypeError, match="array where `controls` goes"):
      model.fit(
        random_uv(),
        # A tensor where controls go, on purpose: the guard under test.
        torch.zeros((30, 2), dtype=torch.float64),  # ty: ignore[invalid-argument-type]
      )

  def test_a_setting_that_cannot_be_honored_is_refused(self) -> None:
    """``ControlsLike`` asks a consumer to refuse, not to drop silently."""

    class ForeignControls:
      def to_dict(self) -> dict[str, object]:
        return {"backend": "tabpfn-criterion", "tree_criterion": "tau"}

    with pytest.raises(ValueError, match="cannot honor tree_criterion"):
      RosenblattBicop(ForeignControls())

  def test_a_foreign_controls_object_carrying_only_known_settings_works(
    self,
  ) -> None:
    class ForeignControls:
      def to_dict(self) -> dict[str, object]:
        return {"backend": "tabpfn-criterion", "device": "cpu", "eps": 1e-5}

    model = RosenblattBicop(ForeignControls())

    assert model.eps == pytest.approx(1e-5)
    assert model._device == torch.device("cpu")

  def test_vine_controls_are_valid_pair_controls(self) -> None:
    model = RosenblattBicop(
      FitControlsRosenblattVinecop(device="cpu", eps=1e-4)
    )

    assert model.eps == pytest.approx(1e-4)


class TestTheBicopBaseContract:
  """The leaves this estimator writes, and the dispatchers it does not."""

  def test_every_public_evaluation_member_comes_from_the_base(self) -> None:
    """A leftover override would bypass the base's var_types dispatch.

    The migration's whole premise is that this class writes continuous leaves
    and overrides no public member. An override that survived would still
    work today -- the pair is continuous -- and would silently skip the
    dispatch the moment anything declared a type.
    """
    for name in ("pdf", "cdf", "hfunc1", "hfunc2", "hinv1", "hinv2"):
      assert getattr(RosenblattBicop, name) is getattr(BicopBase, name), name

  def test_the_pair_declares_that_it_reads_covariates(self) -> None:
    """`pair_eval` reads `supports_covariates` before it calls the leaf.

    `BicopBase` defaults the flag to `False`, and this pair is conditional by
    construction. Without the declaration every covariate evaluation raises a
    `TypeError` naming the class instead of reaching leaves that all declare
    `x` -- the leaf signature was never what marked a pair conditional.
    """
    assert RosenblattBicop.supports_covariates is True

  @pytest.mark.parametrize("var_types", [["d", "c"], ["c", "d"], ["d", "d"]])
  def test_fit_refuses_a_discrete_declaration(
    self,
    patch_uniform: None,
    var_types: list[str],
  ) -> None:
    """`fit` used to `del var_types`, accepting an atom it cannot model.

    The vine's engines declare an edge's types on the pair only *after* it is
    fitted, so `fit` is the first place an atom is visible. Dropping it there
    meant a discrete edge was accepted and then answered with a difference
    quotient of a trapezoidal integration -- a plausible number from a model
    that estimates no atom.
    """
    del patch_uniform
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    with pytest.raises(ValueError, match="must model continuous pairs"):
      model.fit(random_uv(), make_controls(), var_types=var_types)

  def test_with_var_types_refuses_a_discrete_declaration(self) -> None:
    """`with_var_types` is the other door, and the one the engines use.

    Refusing only in `fit` would leave a pre-fitted pair declarable through
    `_declared`, which the fit engines call on every edge carrying an atom.
    """
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    with pytest.raises(ValueError, match="must model continuous pairs"):
      model.with_var_types(["d", "c"])

  def test_declaring_a_pair_continuous_returns_it_unchanged(self) -> None:
    """A refusal that did not test for "d" would break the cascade.

    `continuous_of` calls `with_var_types()` with the all-continuous default
    on every pair in the inverse-Rosenblatt cascade and in the density plot,
    so an unconditional raise would make both unreachable.
    """
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    assert model.with_var_types() is model
    assert model.with_var_types(["c", "c"]) is model

  def test_the_domain_clamp_is_eps_and_not_the_library_width(self) -> None:
    """Inheriting `_prep_args` would clamp at ~1e-10 instead of at `eps`.

    Asserted on the hook rather than through a density, because a density
    would only notice where it varies: the inherited clamp does not raise and
    does not change a shape, it moves the value the inner regressors see. They
    are fitted on `logit(u)`, where the two widths are some ten units of
    feature space apart, so the defect this prevents is silent by
    construction.
    """
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu", eps=1e-3))

    prepped = model._prep_args(
      torch.tensor([[1e-6, 1.0 - 1e-6]], dtype=torch.float64)
    )

    torch.testing.assert_close(
      prepped, torch.tensor([[1e-3, 1.0 - 1e-3]], dtype=torch.float64)
    )

  @pytest.mark.parametrize(
    "name", ["pdf", "cdf", "hfunc1", "hfunc2", "hinv1", "hinv2", "loglik"]
  )
  def test_boundary_values_are_rejected_on_every_inherited_member(
    self,
    patch_uniform: None,
    name: str,
  ) -> None:
    """A rejection written inside a leaf could never fire.

    The base clamps in `_prep_args` before a leaf sees anything, so the check
    has to sit in that hook. Each public member reaches it through `_atoms`,
    and `loglik` reaches it one level further out through `pdf`.
    """
    model = fit_bicop(patch_uniform)

    with pytest.raises(ValueError, match="strictly inside"):
      getattr(model, name)(torch.tensor([[0.0, 0.5]], dtype=torch.float64))

  def test_hinv_reads_the_backend_quantile_rather_than_bisecting(
    self,
    patch_uniform: None,
  ) -> None:
    """`_hinv1_raw` is optional upstream, and the default is far worse.

    Without the override the base bisects `_hfunc1_raw` through
    `solve_increasing`, turning one inner `icdf` into tens of batched forward
    passes and returning an approximation of a quantile the backend knows
    exactly.
    """
    model = fit_bicop(patch_uniform)
    uv = random_uv(5, seed=11)

    result = model.hinv1(uv)
    expected = model.v_given_ux_.icdf(
      uv[:, 1], x=model._features(uv[:, 0], model._default_x(5))
    )

    torch.testing.assert_close(result, expected)

  def test_the_projection_runs_when_the_base_dispatcher_calls(
    self,
    patch_uniform: None,
  ) -> None:
    """The Sinkhorn projection used to live in a public `pdf` override.

    Moved into `_pdf_raw`, it has to still run on every path that reaches the
    pair through the base -- `loglik`, `plot`, and each vine cascade -- which
    a projection left in an override would have skipped.
    """
    projected = fit_bicop(patch_uniform, sinkhorn_iters=3)
    plain = fit_bicop(patch_uniform)
    uv = random_uv(6, seed=3)

    assert not torch.allclose(projected.pdf(uv), plain.pdf(uv))
    assert projected.loglik(uv) != plain.loglik(uv)

  def test_the_projection_count_can_be_changed_after_the_fit(
    self,
    patch_uniform: None,
  ) -> None:
    """The study sweeps the projection over one fit and must not refit.

    `sinkhorn_iters` moved from a per-call keyword onto the controls, so the
    sweep in `npcc.experiments.runner` now writes the setting between
    evaluations. That only works because the projection grid is built lazily:
    a model fitted without projection caches no borders, and a sweep that
    silently returned the unprojected density would compare a setting against
    itself.
    """
    model = fit_bicop(patch_uniform)
    uv = random_uv(5, seed=13)

    assert model._u_grid_borders_ is None
    unprojected = model.pdf(uv).clone()

    model.sinkhorn_iters = 3
    projected = model.pdf(uv)

    assert model._u_grid_borders_ is not None
    assert not torch.allclose(unprojected, projected)
