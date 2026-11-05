"""Core estimators: the Rosenblatt pair copula, vine, distribution and margins.

A supported import path, the way :mod:`pyvinecopulib.core` is: every name here
is also re-exported from :mod:`npcc`, and either spelling is fine. What is not
supported is reaching past this module into the underscore-prefixed ones, which
are named for the level they serve rather than for anything to import.
"""

from npcc.core.bicop import RosenblattBicop
from npcc.core.controls import (
  FitControlsRosenblattBicop,
  FitControlsRosenblattVinecop,
  Transform,
)
from npcc.core.errors import (
  BackendError,
  EstimatorConfigError,
  InvalidBackendKwargsError,
  MissingBackendDependencyError,
  NpccError,
  UnknownBackendError,
)
from npcc.core.margin import ConditionalMargin, QuantileTableConfig
from npcc.core.registry import (
  BackendSpec,
  available_backends,
  create_backend,
  documented_n_range,
  register_backend,
  validate_backend_kwargs,
)
from npcc.core.vinecop import RosenblattVinecop
from npcc.core.vinedist import RosenblattVinedist

__all__ = [
  "BackendError",
  "BackendSpec",
  "ConditionalMargin",
  "EstimatorConfigError",
  "FitControlsRosenblattBicop",
  "FitControlsRosenblattVinecop",
  "InvalidBackendKwargsError",
  "MissingBackendDependencyError",
  "NpccError",
  "QuantileTableConfig",
  "RosenblattBicop",
  "RosenblattVinecop",
  "RosenblattVinedist",
  "Transform",
  "UnknownBackendError",
  "available_backends",
  "create_backend",
  "documented_n_range",
  "register_backend",
  "validate_backend_kwargs",
]
