"""The application tests' helpers: a config, small data, a fake copula stage.

The configs fit the hermetic ``linear-gaussian`` backend of
``tests/conftest.py``, which the ``register_linear_gaussian_backend`` fixture
registers, so every margin runs on the CPU in milliseconds. The fake copula
stage honors the contract of ``npcc.experiments.application.copulas`` and
nothing more, so the runner's copula path is tested whether or not the real
stage is installed.
"""

from __future__ import annotations

import json
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from npcc.experiments.application.data import write_synthetic
from npcc.experiments.application.marginal_set import MarginalSet
from npcc.experiments.config import EstimatorSpec

#: The responses' scales, as configs/application.toml gives them.
RESPONSES = """
[responses]
"obs:air_temperature" = { transform = "identity" }
"obs:dew_point_temperature" = { transform = "identity" }
"obs:relative_humidity" = { transform = "logit", scale = 100.0, boundary_nudge = 0.05 }
"obs:surface_air_pressure" = { transform = "identity" }
"obs:water_vapor_mixing_ratio" = { transform = "log" }
"""

MODELS = ("emos-seasonal", "linear-gaussian-all", "linear-gaussian-emos")
ARMS = ("independence", "gaussian")


def write_data(directory: Path) -> Path:
  """Three stations, weekly: about 209 training and 52 test cases each."""
  write_synthetic(directory, n_stations=3, step_days=7, seed=3)
  return directory


def config_text(
  data_dir: Path,
  *,
  models: Sequence[str] = MODELS,
  arms: Sequence[str] = ARMS,
  stations: str = '"all"',
  n_levels: int = 19,
  base_seed: int = 11,
  extra_data: str = "",
  copula_margins: Sequence[str] | None = None,
) -> str:
  """A config on the hermetic backend, as TOML text."""
  margins = (
    ""
    if copula_margins is None
    else f"margins = {json.dumps(list(copula_margins))}"
  )
  return f"""
base_seed = {base_seed}

[data]
dir = "{data_dir}"
verify = false
stations = {stations}
train = [2017-01-01, 2020-12-31]
test = [2021-01-01, 2021-12-31]
{extra_data}

[margins]
models = {json.dumps(list(models))}
n_levels = {n_levels}
{RESPONSES}
[copulas]
arms = {json.dumps(list(arms))}
{margins}
n_draws = 8
sample_batch_size = 50

[[estimators]]
label = "linear-gaussian-probit"
backend = "linear-gaussian"
transform = "probit"
"""


def write_config(
  path: Path,
  data_dir: Path,
  *,
  models: Sequence[str] = MODELS,
  arms: Sequence[str] = ARMS,
  stations: str = '"all"',
  extra_data: str = "",
  base_seed: int = 11,
  copula_margins: Sequence[str] | None = None,
) -> Path:
  """Write :func:`config_text` to ``path``."""
  path.write_text(
    config_text(
      data_dir,
      models=models,
      arms=arms,
      stations=stations,
      extra_data=extra_data,
      base_seed=base_seed,
      copula_margins=copula_margins,
    )
  )
  return path


@dataclass(frozen=True)
class FakeCopulaResult:
  """The contract's ``CopulaResult``."""

  arm: str
  logcop_test: np.ndarray
  joint_nll: np.ndarray
  energy: np.ndarray
  variogram: np.ndarray
  timings: dict[str, float] = field(default_factory=dict)
  info: dict[str, object] = field(default_factory=dict)
  sample_crps: dict[str, np.ndarray] = field(default_factory=dict)
  dew_point_above_share: np.ndarray | None = None


class FakeCopulaFailure(ValueError):
  """The contract's ``CopulaFailure``, naming the test cases at fault."""

  def __init__(self, message: str, indices: Sequence[int] = ()) -> None:
    super().__init__(message)
    self.indices = list(indices)


@dataclass
class FakeCopulas:
  """A copula stage that records its calls and fails where told to."""

  fail: set[tuple[int, str, str]] = field(default_factory=set)
  calls: list[dict[str, object]] = field(default_factory=list)
  arms: tuple[str, ...] = ARMS
  r_error: str | None = None
  r_checks: int = 0

  def check_boostcopula(self, script: str | None = None) -> str:
    """R's check: passes, or raises what it was told to."""
    del script
    self.r_checks += 1
    if self.r_error is not None:
      raise RuntimeError(self.r_error)
    return "R 4.4.0; boostCopula 0.1.0"

  def run_copula_arm(
    self,
    arm: str,
    ms: MarginalSet,
    *,
    estimators: Mapping[str, EstimatorSpec],
    n_draws: int,
    sample_batch_size: int,
    mixture_k: int,
    seed: int,
    device: str | None,
  ) -> list[FakeCopulaResult]:
    """Independence scores, or a little better for any other arm."""
    self.calls.append(
      {
        "arm": arm,
        "station": ms.station,
        "model": ms.model,
        "estimators": dict(estimators),
        "n_draws": n_draws,
        "sample_batch_size": sample_batch_size,
        "mixture_k": mixture_k,
        "seed": seed,
        "device": device,
      }
    )
    if (ms.station, ms.model, arm) in self.fail:
      raise FakeCopulaFailure("density evaluation failed", [0, 2])
    rng = np.random.default_rng(seed)
    n = len(ms.y_test)
    logcop = np.zeros(n) if arm == "independence" else rng.normal(0.2, 0.1, n)
    marginal = -ms.logpdf_test.sum(axis=1)
    return [
      FakeCopulaResult(
        arm=arm,
        logcop_test=logcop,
        joint_nll=marginal - logcop,
        energy=rng.gamma(4.0, 0.5, n),
        variogram=rng.gamma(4.0, 0.5, n),
        timings={"fit": 0.01, "sample": 0.02},
        info={"seed": seed},
      )
    ]

  def module(self) -> types.ModuleType:
    """This stage as an importable module."""
    module = types.ModuleType("npcc.experiments.application.copulas")
    for name, value in (
      ("CopulaResult", FakeCopulaResult),
      ("CopulaFailure", FakeCopulaFailure),
      ("COPULA_ARMS", self.arms),
      ("run_copula_arm", self.run_copula_arm),
      ("check_boostcopula", self.check_boostcopula),
    ):
      setattr(module, name, value)
    return module
