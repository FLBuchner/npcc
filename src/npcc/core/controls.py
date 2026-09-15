"""Fit controls for Rosenblatt pair copulas and vines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import torch

from npcc.core.margin_quantile_table import QuantileTableConfig
from npcc.core.registry import validate_backend_kwargs

Transform = Literal["identity", "logit", "probit"]


@dataclass
class FitControlsRosenblattBicop:
  """Configuration for fitting a Rosenblatt pair copula.

  These controls configure the conditional-distribution backend used for both
  Rosenblatt directions. They satisfy pyvinecopulib's ``ControlsLike`` contract
  through :meth:`to_dict``.

  Attributes
  ----------
  backend
    Registered conditional-margin backend.
  quantile_table_config
    Quantile-table reconstruction configuration.
  eps
    Boundary clipping distance used throughout copula and transformed-margin
    computations.
  transform
    Transformation applied to copula-scale responses before backend fitting.
  device
    Device used for fitting and evaluation. If omitted, CUDA is selected when
    available and CPU otherwise; input data is placed onto the resolved device
    rather than read for one.
  batch_size
    Default backend inference batch size. If omitted, 400 is used on CPU and
    2000 on CUDA.
  backend_kwargs
    Backend-specific constructor arguments.
  sinkhorn_iters
    Number of Sinkhorn projection iterations. ``None`` disables projection.
  projection_grid_size
    Number of points per axis in the projection grid.
  cdf_n_int
    Number of trapezoid steps the per-row ``cdf`` integrates over. The
    Cartesian-grid ``cdf_grid`` keeps its own, finer default: it shares one
    grid across the whole product and can afford it.
  """

  backend: str = "tabpfn-criterion"
  quantile_table_config: QuantileTableConfig | None = field(
    default_factory=QuantileTableConfig
  )
  eps: float = 1e-6
  transform: Transform = "logit"
  device: str | torch.device | None = None
  batch_size: int | None = None
  backend_kwargs: Mapping[str, object] | None = field(default_factory=dict)
  sinkhorn_iters: int | None = None
  projection_grid_size: int = 101
  cdf_n_int: int = 12

  def __post_init__(self) -> None:
    """Validate and normalize the controls.

    ``None`` is accepted for the two composite settings and means "the
    default": a caller assembling controls from a table reaches for
    ``table.get(name)``, which is ``None`` for an entry with nothing to
    configure, and that is not an error.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any setting is outside its admissible range, or if
        ``backend_kwargs`` has keys the named backend does not accept.
    """
    if self.quantile_table_config is None:
      self.quantile_table_config = QuantileTableConfig()

    if self.backend_kwargs is None:
      self.backend_kwargs = {}

    if self.transform not in ("identity", "logit", "probit"):
      raise ValueError("transform must be 'identity', 'logit', or 'probit'.")

    if not 0.0 < self.eps < 0.5:
      raise ValueError("eps must lie strictly between 0 and 0.5.")

    if self.device is not None:
      self.device = torch.device(self.device)

    if self.batch_size is not None and self.batch_size <= 0:
      raise ValueError("batch_size must be positive.")

    if self.sinkhorn_iters is not None and self.sinkhorn_iters <= 0:
      raise ValueError("sinkhorn_iters must be None or a positive integer.")

    if self.projection_grid_size < 2:
      raise ValueError("projection_grid_size must be at least 2.")

    if self.cdf_n_int < 2:
      raise ValueError("cdf_n_int must be at least 2.")

    self.backend_kwargs = dict(self.backend_kwargs)
    validate_backend_kwargs(self.backend, self.backend_kwargs)

  def to_dict(self) -> dict[str, object]:
    """Return the settings as a plain dictionary."""
    return {
      "backend": self.backend,
      "quantile_table_config": self.quantile_table_config,
      "eps": self.eps,
      "transform": self.transform,
      "device": self.device,
      "batch_size": self.batch_size,
      "backend_kwargs": dict(self.backend_kwargs or {}),
      "sinkhorn_iters": self.sinkhorn_iters,
      "projection_grid_size": self.projection_grid_size,
      "cdf_n_int": self.cdf_n_int,
    }


@dataclass
class FitControlsRosenblattVinecop(FitControlsRosenblattBicop):
  """Configuration for fitting a fixed-structure Rosenblatt vine.

  A vine's controls are also valid pair-copula controls. Consequently, the
  backend configuration is passed unchanged to every pair copula in the vine.

  The class currently adds no vine-specific fields because NPCC requires a
  fixed structure and does not perform selection.
  """
