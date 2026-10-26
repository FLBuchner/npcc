"""Tests for ``npcc.experiments.config`` (grid parsing + validation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from npcc.core.errors import (
  EstimatorConfigError,
  InvalidBackendKwargsError,
  UnknownBackendError,
)
from npcc.experiments.config import (
  EstimatorSpec,
  GridConfig,
  RunConfig,
  load_grid,
)

_TOML = """
[grid]
families = ["clayton", "gumbel"]
tau_scenarios = ["linear", "uncond70"]
normalize = ["none", 5]
n = [50, 100]
n_rep = 3
conditional_uv_grid_n = 4
conditional_x_grid_n = 2
surface_tau_levels = [0.1, 0.5]
surface_families = ["clayton"]
enable_tau_diagnostics = false
tau_diagnostic_n = 100

[[grid.estimators]]
label = "crit-logit"
backend = "tabpfn-criterion"
transform = "logit"

[[grid.estimators]]
label = "quant-identity"
backend = "tabpfn-quantiles"
transform = "identity"
"""


def _write(tmp_path: Path, text: str) -> Path:
  p = tmp_path / "study.toml"
  p.write_text(text)
  return p


def test_load_grid_parses_and_coerces_normalize(tmp_path: Path) -> None:
  grid = load_grid(_write(tmp_path, _TOML))
  assert grid.families == ["clayton", "gumbel"]
  assert grid.normalize == [None, 5]
  assert grid.n_rep == 3
  assert grid.conditional_uv_grid_n == 4
  assert grid.conditional_x_grid_n == 2
  assert grid.surface_tau_levels == [0.1, 0.5]
  assert grid.surface_families == ["clayton"]
  assert grid.enable_tau_diagnostics is False
  assert grid.tau_diagnostic_n == 100
  assert [e.label for e in grid.estimators] == ["crit-logit", "quant-identity"]
  assert [e.backend for e in grid.estimators] == [
    "tabpfn-criterion",
    "tabpfn-quantiles",
  ]


def test_cells_and_estimator_specs_are_explicit(tmp_path: Path) -> None:
  grid = load_grid(_write(tmp_path, _TOML))
  # 2 families x 2 scenarios x 2 n x 3 rep
  assert len(grid.cells()) == 2 * 2 * 2 * 3
  # explicit list, not a product
  assert len(grid.estimator_specs()) == 2


def test_estimator_backend_kwargs_flow_through(tmp_path: Path) -> None:
  text = _TOML + (
    '\n[[grid.estimators]]\nlabel = "gbm-deep"\nbackend = "gbm"\n'
    'transform = "logit"\nbackend_kwargs = { max_depth = 8 }\n'
  )
  grid = load_grid(_write(tmp_path, text))
  est = grid.estimators[-1]
  assert est.backend == "gbm"
  assert dict(est.backend_kwargs) == {"max_depth": 8}


def test_same_backend_different_kwargs_distinct_ids(tmp_path: Path) -> None:
  text = _TOML + (
    '\n[[grid.estimators]]\nlabel = "gbm-shallow"\nbackend = "gbm"\n'
    'transform = "logit"\nbackend_kwargs = { max_depth = 3 }\n'
    '\n[[grid.estimators]]\nlabel = "gbm-deep"\nbackend = "gbm"\n'
    'transform = "logit"\nbackend_kwargs = { max_depth = 9 }\n'
  )
  grid = load_grid(_write(tmp_path, text))
  shallow, deep = grid.estimators[-2], grid.estimators[-1]
  assert shallow.backend == deep.backend == "gbm"
  assert shallow.estimator_id != deep.estimator_id


def test_unknown_backend_rejected(tmp_path: Path) -> None:
  text = _TOML.replace('backend = "tabpfn-criterion"', 'backend = "nope"')
  with pytest.raises(UnknownBackendError, match="Unknown backend"):
    load_grid(_write(tmp_path, text))


def test_unknown_transform_rejected(tmp_path: Path) -> None:
  text = _TOML.replace('transform = "logit"', 'transform = "nope"')
  with pytest.raises(EstimatorConfigError, match="transform"):
    load_grid(_write(tmp_path, text))


def test_bad_backend_kwargs_rejected(tmp_path: Path) -> None:
  text = _TOML + (
    '\n[[grid.estimators]]\nlabel = "gbm-typo"\nbackend = "gbm"\n'
    'transform = "logit"\nbackend_kwargs = { max_dpeth = 3 }\n'
  )
  with pytest.raises(InvalidBackendKwargsError, match="unknown backend_kwargs"):
    load_grid(_write(tmp_path, text))


def test_mistyped_backend_kwarg_rejected(tmp_path: Path) -> None:
  text = _TOML + (
    '\n[[grid.estimators]]\nlabel = "gbm-bad"\nbackend = "gbm"\n'
    'transform = "logit"\nbackend_kwargs = { max_depth = 3.5 }\n'
  )
  with pytest.raises(InvalidBackendKwargsError, match="max_depth"):
    load_grid(_write(tmp_path, text))


def test_duplicate_labels_rejected(tmp_path: Path) -> None:
  text = _TOML + (
    '\n[[grid.estimators]]\nlabel = "crit-logit"\nbackend = "tabpfn-quantiles"\n'
    'transform = "probit"\n'
  )
  with pytest.raises(EstimatorConfigError, match="unique"):
    load_grid(_write(tmp_path, text))


def test_missing_estimators_rejected(tmp_path: Path) -> None:
  text = "\n".join(
    line
    for line in _TOML.splitlines()
    if not line.startswith("[[grid.estimators]]")
    and not line.startswith("label =")
    and not line.startswith("backend =")
    and not line.startswith("transform =")
  )
  with pytest.raises(EstimatorConfigError, match="grid.estimators"):
    load_grid(_write(tmp_path, text))


def test_normalize_zero_and_off_become_none(tmp_path: Path) -> None:
  text = _TOML.replace('normalize = ["none", 5]', 'normalize = [0, "off", 5]')
  grid = load_grid(_write(tmp_path, text))
  assert grid.normalize == [None, None, 5]


@pytest.mark.parametrize(
  "field,bad",
  [
    ("families", '["clayton", "nope"]'),
    ("tau_scenarios", '["linear", "nope"]'),
  ],
)
def test_unknown_axis_value_rejected(
  tmp_path: Path, field: str, bad: str
) -> None:
  text = _TOML
  for line in _TOML.splitlines():
    if line.startswith(f"{field} ="):
      text = _TOML.replace(line, f"{field} = {bad}")
  with pytest.raises(ValueError, match="Unknown"):
    load_grid(_write(tmp_path, text))


def test_negative_normalize_rejected(tmp_path: Path) -> None:
  text = _TOML.replace('normalize = ["none", 5]', "normalize = [-1]")
  with pytest.raises(ValueError, match="positive int"):
    load_grid(_write(tmp_path, text))


def test_small_tau_diagnostic_n_rejected(tmp_path: Path) -> None:
  text = _TOML.replace("tau_diagnostic_n = 100", "tau_diagnostic_n = 9")
  with pytest.raises(ValueError, match="tau_diagnostic_n"):
    load_grid(_write(tmp_path, text))


def test_missing_key_raises(tmp_path: Path) -> None:
  text = "\n".join(
    line for line in _TOML.splitlines() if not line.startswith("n_rep")
  )
  with pytest.raises(ValueError, match="Missing required"):
    load_grid(_write(tmp_path, text))


def test_estimator_id_stable_and_label_independent() -> None:
  a = EstimatorSpec(label="a", backend="tabpfn-criterion", transform="logit")
  b = EstimatorSpec(label="b", backend="tabpfn-criterion", transform="logit")
  # Recomputing is stable and independent of the display label.
  assert a.estimator_id == a.estimator_id
  assert a.estimator_id == b.estimator_id
  # Golden hash locks the canonicalization (backend/transform/backend_kwargs).
  assert (
    a.estimator_id
    == "18948e8a94a2031010fbe3e37b32c07f4346bd7d446b89bda1418f0eceeef123"
  )


def test_shipped_study_config_loads() -> None:
  """The committed paper config must load and match the study design."""
  cfg_path = Path(__file__).resolve().parents[2] / "configs" / "study.toml"
  grid = load_grid(cfg_path)
  assert grid.families == ["clayton", "gaussian"]
  assert grid.tau_scenarios == ["linear", "sin"]
  assert grid.n == [200, 500, 1000, 2000]
  assert grid.n_rep == 10
  assert grid.normalize == [None]
  assert len(grid.estimators) == 18
  labels = [e.label for e in grid.estimators]
  assert len(labels) == len(set(labels))
  ids = [e.estimator_id for e in grid.estimators]
  assert len(ids) == len(set(ids))
  # xgb-quantile is intentionally excluded (tree tail collapse; see registry.py).
  assert {e.backend for e in grid.estimators} == {
    "tabpfn-criterion",
    "tabpfn-quantiles",
    "nori",
    "tabicl",
    "ngboost",
    "pytabkit-realmlp",
    "pytabkit-tabm",
  }
  # TabPFN entries pin the model version through backend_kwargs.
  versions = {
    dict(e.backend_kwargs).get("model_version")
    for e in grid.estimators
    if e.backend.startswith("tabpfn-") and e.backend_kwargs
  }
  assert versions == {"v2.5", "v3"}
  assert len(grid.cells()) == 2 * 2 * 4 * 10


def test_runconfig_validates_workers_and_fmt(tmp_path: Path) -> None:
  assert RunConfig(out=tmp_path).fmt == "parquet"
  assert RunConfig(out=tmp_path).gpu_mem_fraction is None
  with pytest.raises(ValueError, match="workers"):
    RunConfig(out=tmp_path, workers=0)
  with pytest.raises(ValueError, match="fmt"):
    RunConfig(out=tmp_path, fmt="xml")
  with pytest.raises(ValueError, match="gpu_mem_fraction"):
    RunConfig(out=tmp_path, gpu_mem_fraction=1.5)


def test_gridconfig_rejects_empty_axis() -> None:
  with pytest.raises(ValueError, match="non-empty"):
    GridConfig(
      families=[],
      tau_scenarios=["linear"],
      estimators=[
        EstimatorSpec(
          label="crit", backend="tabpfn-criterion", transform="logit"
        )
      ],
      normalize=[None],
      n=[100],
      n_rep=1,
    )
