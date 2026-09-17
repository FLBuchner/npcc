import pytest
import torch
from pyvinecopulib import RVineStructure
from pyvinecopulib.core import BicopLike, NonSimplifiedContext, VinecopBase

from npcc import RosenblattBicop, RosenblattVinecop
from npcc.core.controls import (
  FitControlsRosenblattBicop,
  FitControlsRosenblattVinecop,
)


def random_tensor(
  shape: tuple[int, ...],
  *,
  low: float,
  high: float,
  seed: int,
) -> torch.Tensor:
  """Return reproducible float64 data without changing global RNG state."""
  generator = torch.Generator(device="cpu")
  generator.manual_seed(seed)
  values = torch.rand(shape, generator=generator, dtype=torch.float64)
  return low + (high - low) * values


def make_structure() -> RVineStructure:
  return RVineStructure.from_order([1, 2, 3])


def make_pairs() -> list[list[RosenblattBicop]]:
  return [
    [
      RosenblattBicop(FitControlsRosenblattBicop(device="cpu")),
      RosenblattBicop(FitControlsRosenblattBicop(device="cpu")),
    ],
    [RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))],
  ]


def make_controls() -> FitControlsRosenblattVinecop:
  return FitControlsRosenblattVinecop(
    backend="uniform-native",
    device="cpu",
  )


def fit_vine(
  u: torch.Tensor,
  *,
  x: torch.Tensor | None = None,
) -> RosenblattVinecop:
  vine = RosenblattVinecop(None, make_structure(), device="cpu")
  return vine.fit(u, make_controls(), x=x)


@pytest.fixture
def fitted_vine(
  register_uniform_backends: None,
) -> RosenblattVinecop:
  u = random_tensor((40, 3), low=0.1, high=0.9, seed=42)
  return fit_vine(u)


def test_the_vine_declares_that_it_honors_no_weights() -> None:
  """`VinecopBase` defaults `supports_weights` to True, and this vine is not.

  Weights ride in the controls now rather than in a `fit` argument, and
  `reject_weights` reads this flag to decide. Left at the default, a weighted
  controls object would be accepted and the unweighted fit returned.
  """
  assert RosenblattVinecop.supports_weights is False


def test_rosenblatt_vinecop_subclass_vinecop_base() -> None:
  assert issubclass(RosenblattVinecop, VinecopBase)


def test_constructor_binds_structure_and_pairs() -> None:
  structure = make_structure()
  pairs = make_pairs()

  vine = RosenblattVinecop(pairs, structure)

  assert vine.dim == 3
  assert vine.trunc_lvl == 2
  assert vine.order == (1, 2, 3)
  assert vine.supports_covariates is True
  assert isinstance(vine._context, NonSimplifiedContext)
  assert vine.get_pair_copula(0, 1) is pairs[0][1]
  assert vine.get_pair_copula(1, 0) is pairs[1][0]


def test_constructor_creates_unfitted_vine() -> None:
  vine = RosenblattVinecop(None, make_structure(), device="cpu")

  assert vine.pair_copulas == []
  assert vine._device == torch.device("cpu")
  assert isinstance(vine._context, NonSimplifiedContext)


def test_constructor_rejects_wrong_tree_count() -> None:
  with pytest.raises(ValueError, match="has 1 trees, expected 2"):
    RosenblattVinecop([[RosenblattBicop()]], make_structure())


def test_constructor_rejects_wrong_edge_count() -> None:
  pairs = [[RosenblattBicop()], [RosenblattBicop()]]

  with pytest.raises(ValueError, match="tree 0 has 1 edges, expected 2"):
    RosenblattVinecop(pairs, make_structure())


def test_constructor_copies_pair_rows() -> None:
  pairs = make_pairs()
  vine = RosenblattVinecop(pairs, make_structure())

  pairs[0].clear()

  assert len(vine.pair_copulas[0]) == 2


def test_constructor_infers_pair_device() -> None:
  vine = RosenblattVinecop(make_pairs(), make_structure())

  assert vine._device == torch.device("cpu")


def test_constructor_rejects_mixed_pair_devices() -> None:
  pairs = make_pairs()
  pairs[0][1]._device = torch.device("cuda:0")

  with pytest.raises(ValueError, match="must use the same device"):
    RosenblattVinecop(pairs, make_structure())


def test_constructor_rejects_explicit_device_mismatch() -> None:
  with pytest.raises(ValueError, match="does not match"):
    RosenblattVinecop(
      make_pairs(),
      make_structure(),
      device="cuda:0",
    )


def test_fit_installs_pair_copulas(
  register_uniform_backends: None,
) -> None:
  u = random_tensor((40, 3), low=0.1, high=0.9, seed=42)

  vine = fit_vine(u)

  assert [len(row) for row in vine.pair_copulas] == [2, 1]
  assert vine._device == torch.device("cpu")

  for row in vine.pair_copulas:
    for pair in row:
      assert pair.v_given_ux_._fitted is True
      assert pair.u_given_vx_._fitted is True


def test_fit_assembles_non_simplified_context(
  register_uniform_backends: None,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  u = random_tensor((40, 3), low=0.1, high=0.9, seed=42)
  x = random_tensor((40, 2), low=-1.0, high=1.0, seed=43)
  context_widths: list[int | None] = []

  original_fit = RosenblattBicop.fit

  def record_fit(
    self: RosenblattBicop,
    uv: torch.Tensor,
    /,
    controls: FitControlsRosenblattBicop | None = None,
    *,
    var_types: list[str] | None = None,
    x: torch.Tensor | None = None,
  ) -> RosenblattBicop:
    width = None if x is None else int(x.shape[1])
    context_widths.append(width)
    return original_fit(
      self,
      uv,
      controls,
      var_types=var_types,
      x=x,
    )

  monkeypatch.setattr(RosenblattBicop, "fit", record_fit)

  fit_vine(u, x=x)

  assert context_widths == [2, 2, 3]


def test_sample_is_seeded_and_on_configured_device(
  fitted_vine: RosenblattVinecop,
) -> None:
  first = fitted_vine.sample(8, seeds=[42])
  second = fitted_vine.sample(8, seeds=[42])

  assert first.shape == (8, 3)
  assert first.dtype == torch.float64
  assert first.device.type == "cpu"
  assert torch.equal(first, second)
  assert torch.all((first > 0.0) & (first < 1.0))


def test_sample_qrng_is_reproducible(
  fitted_vine: RosenblattVinecop,
) -> None:
  first = fitted_vine.sample(8, qrng=True, seeds=[42])
  second = fitted_vine.sample(8, qrng=True, seeds=[42])

  assert first.shape == (8, 3)
  assert first.dtype == torch.float64
  assert first.device.type == "cpu"
  assert torch.equal(first, second)


def test_pdf_returns_torch_tensor(
  fitted_vine: RosenblattVinecop,
) -> None:
  u = random_tensor((5, 3), low=0.2, high=0.8, seed=42)

  result = fitted_vine.pdf(u)

  assert result.shape == (5,)
  assert result.dtype == torch.float64
  assert result.device.type == "cpu"
  assert torch.isfinite(result).all()
  assert torch.all(result > 0.0)


def test_rosenblatt_inverse_round_trip(
  fitted_vine: RosenblattVinecop,
) -> None:
  u = random_tensor((5, 3), low=0.2, high=0.8, seed=42)

  transformed = fitted_vine.rosenblatt(u)
  recovered = fitted_vine.inverse_rosenblatt(transformed)

  torch.testing.assert_close(recovered, u, atol=1e-8, rtol=1e-8)


def test_loglik_matches_summed_log_pdf(
  fitted_vine: RosenblattVinecop,
) -> None:
  u = random_tensor((5, 3), low=0.2, high=0.8, seed=42)

  density = fitted_vine.pdf(u)
  result = fitted_vine.loglik(u)
  expected = torch.log(density).sum()

  torch.testing.assert_close(result, expected)


def test_pdf_with_covariates(
  register_uniform_backends: None,
) -> None:
  train_u = random_tensor((40, 3), low=0.1, high=0.9, seed=42)
  train_x = random_tensor((40, 2), low=-1.0, high=1.0, seed=43)
  query_u = random_tensor((5, 3), low=0.2, high=0.8, seed=44)
  query_x = random_tensor((5, 2), low=-1.0, high=1.0, seed=45)
  vine = fit_vine(train_u, x=train_x)

  result = vine.pdf(query_u, x=query_x)

  assert result.shape == (5,)
  assert result.dtype == torch.float64
  assert result.device.type == "cpu"


def test_cdf_returns_torch_tensor(
  fitted_vine: RosenblattVinecop,
) -> None:
  query = random_tensor((3, 3), low=0.2, high=0.8, seed=42)

  result = fitted_vine.cdf(query, N=64, seeds=[42])

  assert result.shape == (3,)
  assert result.dtype == torch.float64
  assert result.device.type == "cpu"
  assert torch.all((result >= 0.0) & (result <= 1.0))


def test_sample_conditional_returns_torch_tensor(
  fitted_vine: RosenblattVinecop,
) -> None:
  u_cond = torch.tensor([[0.3], [0.5], [0.7]], dtype=torch.float64)

  result = fitted_vine.sample_conditional(u_cond, seeds=[42])

  assert result.shape == (3, 3)
  assert result.dtype == torch.float64
  assert result.device.type == "cpu"
  torch.testing.assert_close(result[:, 2], u_cond[:, 0])


def test_fit_rejects_structure_dimension_mismatch(
  register_uniform_backends: None,
) -> None:
  u = torch.full((10, 2), 0.5, dtype=torch.float64)
  vine = RosenblattVinecop(None, make_structure(), device="cpu")

  with pytest.raises(ValueError, match=r"u must have shape \(n, 3\)"):
    vine.fit(u, make_controls())


def test_fit_rejects_covariate_row_mismatch(
  register_uniform_backends: None,
) -> None:
  u = torch.full((10, 3), 0.5, dtype=torch.float64)
  x = torch.full((9, 1), 0.5, dtype=torch.float64)
  vine = RosenblattVinecop(None, make_structure(), device="cpu")

  with pytest.raises(
    ValueError,
    match="x must have one row per observation",
  ):
    vine.fit(u, make_controls(), x=x)


def test_cdf_rejects_external_covariates(
  fitted_vine: RosenblattVinecop,
) -> None:
  u = torch.full((10, 3), 0.5, dtype=torch.float64)
  x = torch.full((10, 1), 0.5, dtype=torch.float64)

  with pytest.raises(NotImplementedError, match="Conditional cdf"):
    fitted_vine.cdf(u, x=x, N=32)


def test_sample_conditional_rejects_reorientation(
  fitted_vine: RosenblattVinecop,
) -> None:
  u_cond = torch.tensor([[0.4], [0.6]], dtype=torch.float64)

  with pytest.raises(NotImplementedError, match="non-simplified"):
    fitted_vine.sample_conditional(
      u_cond,
      conditioning_set=[1],
      seeds=[42],
    )


def test_sample_with_covariates_returns_torch_tensor(
  register_uniform_backends: None,
) -> None:
  train_u = random_tensor((40, 3), low=0.1, high=0.9, seed=42)
  train_x = random_tensor((40, 2), low=-1.0, high=1.0, seed=43)
  query_x = random_tensor((5, 2), low=-1.0, high=1.0, seed=44)
  vine = fit_vine(train_u, x=train_x)

  result = vine.sample(5, x=query_x, seeds=[42])

  assert result.shape == (5, 3)
  assert result.dtype == torch.float64
  assert result.device.type == "cpu"


def test_set_pair_copulas_calls_the_invalidation_hook() -> None:
  """Installing pairs drops anything memoized from the previous ones.

  ``set_pair_copulas`` is the one path where the pairs change while the
  structure stays, so ``_bind_vine``'s own reset does not cover it and the base
  asks the implementation to invalidate. Nothing in the suite calls
  ``set_pair_copulas`` otherwise, so a regression that dropped the call would
  be invisible -- and unobservable behaviorally, since this vine's context
  assembles conditioning and therefore never builds batched state to drop.
  """
  calls: list[int] = []

  class RecordingVinecop(RosenblattVinecop):
    def _invalidate_batched(self) -> None:
      calls.append(1)
      super()._invalidate_batched()

  vine = RecordingVinecop(None, make_structure(), device="cpu")
  before = len(calls)

  pairs: list[list[BicopLike[torch.Tensor]]] = [
    list(row) for row in make_pairs()
  ]
  vine.set_pair_copulas(pairs)

  assert len(calls) == before + 1
  assert vine._batched is None
