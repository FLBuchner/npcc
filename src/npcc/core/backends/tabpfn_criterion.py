"""
Native-head TabPFN conditional margin.

TabPFN's regressor represents its predictive distribution using a binned
criterion. ``predict(x, output_type="full")`` returns logits together with a
criterion exposing PDF, CDF, and inverse-CDF operations.

Reading this criterion directly avoids reconstructing the distribution from
predicted quantiles. Grid evaluation predicts the logits once per conditioning
row and then evaluates those logits at every requested response value.

TabPFN receives host-side input tensors. These conversions are isolated in this
adapterm while all public inputs and outputs remain torch tensors.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

import torch

from npcc.core.backends._tabpfn_icdf import rowwise_icdf
from npcc.core.backends.tabpfn_common import (
  _DEFAULT_MODEL_VERSION,
  ModelVersion,
  make_tabpfn_regressor,
)
from npcc.core.margin import ConditionalMargin


class _CriterionLike(Protocol):
  """Operations used from TabPFN's predictive criterion."""

  def pdf(self, logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor: ...
  def cdf(self, logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor: ...
  def icdf(self, logits: torch.Tensor, left_prob: float) -> torch.Tensor: ...


def _coerce_logits_tensor(
  logits: object,
  *,
  device: torch.device | str,
) -> torch.Tensor:
  """Convert TabPFN logits to a float32 tensor on the model device."""
  if isinstance(logits, torch.Tensor):
    return logits.to(
      device=device,
      dtype=torch.float32,
    )

  return torch.as_tensor(logits, dtype=torch.float32, device=device)


class TabPFNCriterionBackend(ConditionalMargin):
  """Conditional margin using TabPFN's native predictive criterion.

  Parameters
  ----------
  transform
    Response transformation inherited from
    class:`~npcc.core.margin.ConditionalMargin`.
  eps
    Boundary clipping distance for logit and probit transformations.
  device
    Device used by the TabPFN model and returned tensors.
  batch_size
    Maximum number of conditioning rows predicted and inverted per batch.
  model_kwargs
    Additional arguments passed to ``TabPFNRegressor``.
  model_version
    TabPFN model version. The default is the shared current version.
  """

  model_: Any | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
    model_version: ModelVersion | None = _DEFAULT_MODEL_VERSION,
  ) -> None:
    super().__init__(
      transform=transform,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )

    self.model_kwargs = dict(model_kwargs or {})
    self.model_kwargs.setdefault("device", str(self._device))
    self.model_version = model_version
    self.model_ = None

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    """Fit TabPFN."""
    self.model_ = make_tabpfn_regressor(
      self.model_version,
      self.model_kwargs,
    )

    self.model_.fit(
      x.detach().cpu(),
      z.detach().cpu(),
    )

  def _predict_full(
    self,
    x: torch.Tensor,
  ) -> tuple[torch.Tensor, _CriterionLike]:
    """Predict logits and the associated TabPFN criterion."""
    assert self.model_ is not None

    prediction = self.model_.predict(
      x.detach().cpu(),
      output_type="full",
    )

    logits = _coerce_logits_tensor(
      prediction["logits"],
      device=self._device,
    )

    criterion: _CriterionLike = prediction["criterion"]

    return logits, criterion

  def _criterion_pdf(
    self,
    logits: torch.Tensor,
    criterion: _CriterionLike,
    z: torch.Tensor,
  ) -> torch.Tensor:
    """Evaluate the criterion density on the transformed scale."""
    z_evaluation = z.to(
      dtype=logits.dtype,
      device=logits.device,
    ).reshape(-1, 1)

    density = criterion.pdf(logits, z_evaluation)

    return density.reshape(-1).to(dtype=z.dtype)

  def _criterion_cdf(
    self,
    logits: torch.Tensor,
    criterion: _CriterionLike,
    z: torch.Tensor,
  ) -> torch.Tensor:
    """Evaluate the criterion CDF on the transformed scale."""
    z_evaluation = z.to(
      dtype=logits.dtype,
      device=logits.device,
    ).reshape(-1, 1)

    cdf = criterion.cdf(logits, z_evaluation)
    return cdf.reshape(-1).to(dtype=z.dtype)

  def pdf(
    self,
    y: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate ``f(y_i | x_i)`` for each observation."""
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    y_t = y.reshape(-1)
    x_t = self._conditioning(y_t, x=x)

    if y_t.shape[0] != x_t.shape[0]:
      raise ValueError("x and y have incompatible lengths.")

    z = self._transform_y(y_t)
    parts: list[torch.Tensor] = []

    for start in range(0, y_t.shape[0], effective_batch_size):
      end = min(start + effective_batch_size, y_t.shape[0])

      logits, criterion = self._predict_full(x_t[start:end])

      parts.append(
        self._criterion_pdf(
          logits,
          criterion,
          z[start:end],
        )
      )

    density_z = (
      torch.cat(parts)
      if parts
      else torch.empty(
        0,
        device=y_t.device,
        dtype=y_t.dtype,
      )
    )

    return density_z * self._jacobian_inverse(y_t)

  def cdf(
    self,
    y: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate ``F(y_i | x_i)`` for every observation."""
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    y_t = y.reshape(-1)
    x_t = self._conditioning(y_t, x=x)

    if y_t.shape[0] != x_t.shape[0]:
      raise ValueError("x and y have incompatible lengths.")

    z = self._transform_y(y_t)
    parts: list[torch.Tensor] = []

    for start in range(0, y_t.shape[0], effective_batch_size):
      end = min(start + effective_batch_size, y_t.shape[0])

      logits, criterion = self._predict_full(x_t[start:end])

      parts.append(
        self._criterion_cdf(
          logits,
          criterion,
          z[start:end],
        )
      )

    return (
      torch.cat(parts)
      if parts
      else torch.empty(
        0,
        device=y_t.device,
        dtype=y_t.dtype,
      )
    )

  def icdf(
    self,
    p: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate ``F^{-1}(p_i | x_i)`` on the response scale."""
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    p_t = p.reshape(-1)
    x_t = self._conditioning(p_t, x=x)

    if p_t.shape[0] != x_t.shape[0]:
      raise ValueError("x and p have incompatible lengths.")

    if torch.any((p_t <= 0.0) | (p_t >= 1.0)):
      raise ValueError("p must lie strictly inside (0, 1).")

    transformed_quantiles = torch.empty(
      p_t.shape[0],
      dtype=p_t.dtype,
      device=p_t.device,
    )

    for start in range(0, p_t.shape[0], effective_batch_size):
      end = min(start + effective_batch_size, p_t.shape[0])

      logits, criterion = self._predict_full(x_t[start:end])

      transformed_quantiles[start:end] = rowwise_icdf(
        criterion, logits, p_t[start:end]
      )

    return self._inverse_transform(transformed_quantiles)

  def pdf_grid(
    self,
    y_grid: torch.Tensor,
    /,
    *,
    x: torch.Tensor,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate density on the Cartesian product of ``x`` and ``y_grid``."""
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    y_grid_t = self._prep(y_grid).reshape(-1)

    if y_grid_t.numel() == 0:
      raise ValueError("y_grid must contain at least one value.")

    x_t = self._grid_covariates(x)
    transformed_grid = self._transform_y(y_grid_t)
    jacobian = self._jacobian_inverse(y_grid_t)

    n_grid = y_grid_t.shape[0]
    chunks: list[torch.Tensor] = []

    for start in range(0, x_t.shape[0], effective_batch_size):
      end = min(start + effective_batch_size, x_t.shape[0])

      x_chunk = x_t[start:end]
      logits, criterion = self._predict_full(x_chunk)

      repeated_logits = logits.repeat_interleave(n_grid, dim=0)
      repeated_grid = transformed_grid.repeat(x_chunk.shape[0])

      density = self._criterion_pdf(
        repeated_logits,
        criterion,
        repeated_grid,
      ).reshape(x_chunk.shape[0], n_grid)

      chunks.append(density * jacobian.unsqueeze(0))

    return (
      torch.cat(chunks, dim=0)
      if chunks
      else torch.empty(
        (0, n_grid),
        dtype=y_grid_t.dtype,
        device=y_grid_t.device,
      )
    )

  def cdf_grid(
    self,
    y_grid: torch.Tensor,
    /,
    *,
    x: torch.Tensor,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate CDF on the Cartesian product of ``x`` and ``y_grid``."""
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    y_grid_t = self._prep(y_grid).reshape(-1)

    if y_grid_t.numel() == 0:
      raise ValueError("y_grid must contain at least one value.")

    x_t = self._grid_covariates(x)
    transformed_grid = self._transform_y(y_grid_t)

    n_grid = y_grid_t.shape[0]
    chunks: list[torch.Tensor] = []

    for start in range(0, x_t.shape[0], effective_batch_size):
      end = min(start + effective_batch_size, x_t.shape[0])

      x_chunk = x_t[start:end]
      logits, criterion = self._predict_full(x_chunk)

      repeated_logits = logits.repeat_interleave(n_grid, dim=0)
      repeated_grid = transformed_grid.repeat(x_chunk.shape[0])

      probabilities = self._criterion_cdf(
        repeated_logits,
        criterion,
        repeated_grid,
      ).reshape(x_chunk.shape[0], n_grid)

      chunks.append(probabilities)

    return (
      torch.cat(chunks, dim=0)
      if chunks
      else torch.empty(
        (0, n_grid),
        dtype=y_grid_t.dtype,
        device=y_grid_t.device,
      )
    )
