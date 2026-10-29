"""Regression coverage for paired TabPFN quantiles and compatibility."""

from __future__ import annotations

import math

import pytest
import torch
from tabpfn.architectures.base.bar_distribution import (
  BarDistribution,
  FullSupportBarDistribution,
)

from npcc.core.backends._tabpfn_icdf import inversion_kind, rowwise_icdf
from npcc.core.backends.tabpfn_criterion import TabPFNCriterionBackend


class TailCriterion(FullSupportBarDistribution):
  """Native upstream 89cc3a35 inversion, retained as an independent oracle."""

  @staticmethod
  def halfnormal_with_p_weight_before(
    range_max: float | torch.Tensor, p: float = 0.5
  ) -> torch.distributions.HalfNormal:
    """Build a half-normal placing ``p`` of its mass below ``range_max``.

    The scale is computed in ``range_max``'s floating dtype, or in the
    default dtype when ``range_max`` is not a floating tensor.
    """
    as_tensor = torch.as_tensor(range_max)
    dtype = (
      as_tensor.dtype
      if as_tensor.dtype.is_floating_point
      else torch.get_default_dtype()
    )
    s = range_max / torch.distributions.HalfNormal(
      torch.tensor(1.0, dtype=dtype)
    ).icdf(torch.tensor(p, dtype=dtype))
    return torch.distributions.HalfNormal(s)

  borders: torch.Tensor

  def icdf(self, logits: torch.Tensor, left_prob: float) -> torch.Tensor:
    """Calculate quantiles using half-normal tails in the outer buckets."""
    if left_prob == 0.0:
      return logits.new_full(logits.shape[:-1], float("-inf"))
    if left_prob == 1.0:
      return logits.new_full(logits.shape[:-1], float("inf"))
    probs = logits.softmax(-1)
    cumprobs = torch.cumsum(probs, -1)
    left_prob_tensor = torch.full(
      (*cumprobs.shape[:-1], 1),
      left_prob,
      dtype=logits.dtype,
      device=logits.device,
    )
    idx = torch.searchsorted(cumprobs, left_prob_tensor).squeeze(-1)
    idx = idx.clamp(0, self.num_bars - 1)
    cumprobs_before = torch.cat(
      (torch.zeros_like(cumprobs[..., :1]), cumprobs[..., :-1]), dim=-1
    )
    selected_probs = probs.gather(-1, idx[..., None]).squeeze(-1)
    conditional_prob = (
      left_prob - cumprobs_before.gather(-1, idx[..., None]).squeeze(-1)
    ) / selected_probs
    conditional_prob = conditional_prob.clamp(0.0, 1.0)
    values = self.borders[idx] + self.bucket_widths[idx] * conditional_prob
    side_normals = (
      self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
      self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
    )
    left_tail_values = self.borders[1] - side_normals[0].icdf(
      1.0 - conditional_prob
    )
    right_tail_values = self.borders[-2] + side_normals[1].icdf(
      conditional_prob
    )
    values = torch.where(idx == 0, left_tail_values, values)
    return torch.where(idx == self.num_bars - 1, right_tail_values, values)


DEVICES = [
  "cpu",
  pytest.param(
    "cuda",
    marks=[
      pytest.mark.cuda,
      pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA device required"
      ),
    ],
  ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("p_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
  "criterion_type", [BarDistribution, FullSupportBarDistribution, TailCriterion]
)
def test_native_agreement(
  device: str,
  dtype: torch.dtype,
  p_dtype: torch.dtype,
  criterion_type: type[BarDistribution],
) -> None:
  """Paired probabilities must preserve bin selection, rounding, and tails."""
  criterion = criterion_type(
    torch.tensor([-3.0, -1.0, 0.0, 2.0, 5.0], dtype=dtype, device=device)
  )
  generator = torch.Generator().manual_seed(104)
  logits = torch.randn(37, 4, generator=generator, dtype=dtype).to(device)
  p = torch.linspace(1e-7, 1 - 1e-7, 37, dtype=p_dtype, device=device)
  p[0] = 1e-12
  p[-4:] = 1 - 1e-12
  # Force exact cumulative boundaries and adjacent representable values.
  boundary = logits.softmax(-1).cumsum(-1)[:, 1].to(p_dtype)
  p[3:6] = boundary[3:6]
  p[6:9] = torch.nextafter(boundary[6:9], torch.zeros_like(p[6:9]))
  expected = torch.cat(
    [criterion.icdf(row[None], float(prob)) for row, prob in zip(logits, p)]
  ).to(p_dtype)
  result = rowwise_icdf(criterion, logits, p)
  assert result.shape == (37,)
  assert result.dtype == p_dtype
  torch.testing.assert_close(result, expected, rtol=0, atol=0)
  assert inversion_kind(type(criterion).icdf) is not None


def test_unknown_criterion_fallback() -> None:
  """An overridden inverse must never be mistaken for ordinary bins."""

  class CustomCriterion(BarDistribution):
    calls = 0

    def icdf(self, logits: torch.Tensor, left_prob: float) -> torch.Tensor:
      self.calls += 1
      return logits[:, 0] + left_prob**2

  criterion = CustomCriterion(torch.tensor([-1.0, 0.0, 1.0]))
  logits = torch.arange(14, dtype=torch.float32).reshape(7, 2)
  p = torch.linspace(0.01, 0.99, 7)
  result = rowwise_icdf(criterion, logits, p)
  assert criterion.calls == 7
  assert result.shape == (7,)
  torch.testing.assert_close(result, logits[:, 0] + p.square())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_size", [1, 4, 17, 32])
def test_prediction_and_inversion_share_batches(
  monkeypatch: pytest.MonkeyPatch, batch_size: int, device: str
) -> None:
  """Partial batches must predict once and pair each row with its own p."""
  backend = TabPFNCriterionBackend(transform="identity", device=device)
  # Isolate evaluation from model fitting and downloads.
  backend._fitted = True
  criterion = BarDistribution(
    torch.tensor([-3.0, -1.0, 1.0, 4.0], device=device)
  )
  calls: list[int] = []
  logits = torch.randn(17, 3, generator=torch.Generator().manual_seed(12)).to(
    device
  )
  p = torch.linspace(0.00001, 0.99999, 17, dtype=torch.float64, device=device)
  x = torch.arange(17, dtype=torch.float64)[:, None]

  class Model:
    def predict(
      self, features: torch.Tensor, *, output_type: str
    ) -> dict[str, object]:
      assert output_type == "full"
      calls.append(len(features))
      return {
        "logits": logits[features[:, 0].long()],
        "criterion": criterion,
      }

  backend.model_ = Model()
  original = torch.Tensor.item

  def no_item(self: torch.Tensor, *args: object) -> object:
    raise AssertionError("optimized inversion must not call item")

  with monkeypatch.context() as patch:
    patch.setattr(torch.Tensor, "item", no_item)
    result = backend.icdf(p, x=x, batch_size=batch_size)
  assert torch.Tensor.item is original
  expected = rowwise_icdf(criterion, logits, p)
  assert result.shape == (17,)
  assert len(calls) == math.ceil(17 / batch_size)
  assert calls == [min(batch_size, 17 - i) for i in range(0, 17, batch_size)]
  torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("criterion_type", [BarDistribution, TailCriterion])
def test_endpoint_conventions(criterion_type: type[BarDistribution]) -> None:
  """Full-support endpoints are infinite only in implementations with tails."""
  criterion = criterion_type(torch.tensor([-2.0, 0.0, 3.0]))
  logits = torch.tensor([[0.1, -0.2], [0.3, 0.6]])
  p = torch.tensor([0.0, 1.0])
  expected = torch.cat(
    [criterion.icdf(logits[:1], 0), criterion.icdf(logits[1:], 1)]
  )
  torch.testing.assert_close(rowwise_icdf(criterion, logits, p), expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_many_bins_batch_consistency(device: str, dtype: torch.dtype) -> None:
  """CUDA reduction kernels can change with batch size at realistic bin counts."""
  criterion = FullSupportBarDistribution(
    torch.linspace(-4, 4, 1001, dtype=dtype, device=device)
  )
  logits = torch.randn(
    129, 1000, dtype=dtype, generator=torch.Generator().manual_seed(813)
  ).to(device)
  p = torch.linspace(1e-7, 1 - 1e-7, 129, dtype=torch.float64, device=device)
  expected = torch.cat(
    [criterion.icdf(row[None], float(prob)) for row, prob in zip(logits, p)]
  ).to(p.dtype)
  # Different CUDA softmax/cumsum reductions need a precision-scaled tolerance.
  tolerance = 5e-5 if dtype == torch.float32 else 1e-12
  for batch_size in (1, 64, 129):
    result = torch.cat(
      [
        rowwise_icdf(
          criterion, logits[i : i + batch_size], p[i : i + batch_size]
        )
        for i in range(0, len(p), batch_size)
      ]
    )
    assert result.shape == (129,)
    torch.testing.assert_close(result, expected, atol=tolerance, rtol=0)
