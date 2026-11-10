"""Tests for ``npcc.experiments.config`` (grid parsing + validation)."""

from __future__ import annotations

from dataclasses import fields
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
  select_estimators,
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
label = "quant-logit"
backend = "tabpfn-quantiles"
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
  assert [e.label for e in grid.estimators] == ["quant-logit", "quant-identity"]
  assert [e.backend for e in grid.estimators] == [
    "tabpfn-quantiles",
    "tabpfn-quantiles",
  ]


def test_cells_and_estimators_are_explicit(tmp_path: Path) -> None:
  grid = load_grid(_write(tmp_path, _TOML))
  # 2 families x 2 scenarios x 2 n x 3 rep
  assert len(grid.cells()) == 2 * 2 * 2 * 3
  # explicit list, not a product
  assert len(grid.estimators) == 2


def test_estimator_backend_kwargs_flow_through(tmp_path: Path) -> None:
  text = _TOML + (
    '\n[[grid.estimators]]\nlabel = "lightgbm-deep"\nbackend = "lightgbm"\n'
    'transform = "logit"\nbackend_kwargs = { max_depth = 8 }\n'
  )
  grid = load_grid(_write(tmp_path, text))
  est = grid.estimators[-1]
  assert est.backend == "lightgbm"
  assert dict(est.backend_kwargs) == {"max_depth": 8}


def test_same_backend_different_kwargs_distinct_ids(tmp_path: Path) -> None:
  text = _TOML + (
    '\n[[grid.estimators]]\nlabel = "lightgbm-shallow"\nbackend = "lightgbm"\n'
    'transform = "logit"\nbackend_kwargs = { max_depth = 3 }\n'
    '\n[[grid.estimators]]\nlabel = "lightgbm-deep"\nbackend = "lightgbm"\n'
    'transform = "logit"\nbackend_kwargs = { max_depth = 9 }\n'
  )
  grid = load_grid(_write(tmp_path, text))
  shallow, deep = grid.estimators[-2], grid.estimators[-1]
  assert shallow.backend == deep.backend == "lightgbm"
  assert shallow.estimator_id != deep.estimator_id


def test_unknown_backend_rejected(tmp_path: Path) -> None:
  text = _TOML.replace('backend = "tabpfn-quantiles"', 'backend = "nope"')
  with pytest.raises(UnknownBackendError, match="Unknown backend"):
    load_grid(_write(tmp_path, text))


def test_unknown_transform_rejected(tmp_path: Path) -> None:
  text = _TOML.replace('transform = "logit"', 'transform = "nope"')
  with pytest.raises(EstimatorConfigError, match="transform"):
    load_grid(_write(tmp_path, text))


def test_bad_backend_kwargs_rejected(tmp_path: Path) -> None:
  text = _TOML + (
    '\n[[grid.estimators]]\nlabel = "lightgbm-typo"\nbackend = "lightgbm"\n'
    'transform = "logit"\nbackend_kwargs = { max_dpeth = 3 }\n'
  )
  with pytest.raises(InvalidBackendKwargsError, match="unknown backend_kwargs"):
    load_grid(_write(tmp_path, text))


def test_mistyped_backend_kwarg_rejected(tmp_path: Path) -> None:
  text = _TOML + (
    '\n[[grid.estimators]]\nlabel = "lightgbm-bad"\nbackend = "lightgbm"\n'
    'transform = "logit"\nbackend_kwargs = { max_depth = 3.5 }\n'
  )
  with pytest.raises(InvalidBackendKwargsError, match="max_depth"):
    load_grid(_write(tmp_path, text))


def test_duplicate_labels_rejected(tmp_path: Path) -> None:
  text = _TOML + (
    '\n[[grid.estimators]]\nlabel = "quant-logit"\nbackend = "tabpfn-quantiles"\n'
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
  a = EstimatorSpec(label="a", backend="tabpfn-quantiles", transform="logit")
  b = EstimatorSpec(label="b", backend="tabpfn-quantiles", transform="logit")
  # Recomputing is stable and independent of the display label.
  assert a.estimator_id == a.estimator_id
  assert a.estimator_id == b.estimator_id
  # Golden hash locks the canonicalization (backend/transform/backend_kwargs).
  assert (
    a.estimator_id
    == "eaf54a98cc1b68d623481297d830f5c1c4f8fe9d373dc9f73d4c1e10f5d33999"
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
  assert len(grid.estimators) == 16
  labels = [e.label for e in grid.estimators]
  assert len(labels) == len(set(labels))
  ids = [e.estimator_id for e in grid.estimators]
  assert len(ids) == len(set(ids))
  assert {e.backend for e in grid.estimators} == {
    "tabpfn-quantiles",
    "nori",
    "tabicl",
    "lightgbm",
    "xgboost",
    "catboost",
  }
  # TabPFN entries pin the model version through backend_kwargs.
  versions = {
    dict(e.backend_kwargs).get("model_version")
    for e in grid.estimators
    if e.backend.startswith("tabpfn-") and e.backend_kwargs
  }
  assert versions == {"v2.5", "v3", "v3.5"}
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
          label="quant", backend="tabpfn-quantiles", transform="logit"
        )
      ],
      normalize=[None],
      n=[100],
      n_rep=1,
    )


_ROOT = Path(__file__).resolve().parents[2]
_SHAPE = ["clayton-gumbel-mix", "student-varying-nu"]


def _shape_grid(
  *,
  families: list[str] | None = None,
  tau_scenarios: list[str] | None = None,
  surface_families: list[str] | None = None,
  surface_x: list[float] | None = None,
) -> GridConfig:
  return GridConfig(
    families=_SHAPE if families is None else families,
    tau_scenarios=["implied"] if tau_scenarios is None else tau_scenarios,
    estimators=[
      EstimatorSpec(
        label="quant", backend="tabpfn-quantiles", transform="logit"
      )
    ],
    normalize=[None],
    n=[100],
    n_rep=1,
    surface_families=[] if surface_families is None else surface_families,
    surface_x=[] if surface_x is None else surface_x,
  )


@pytest.mark.parametrize(
  ("families", "tau_scenarios", "match"),
  [
    (
      ["clayton-gumbel-mix"],
      ["linear"],
      r"tau_scenarios must be \['implied'\]",
    ),
    (_SHAPE, ["implied", "sin"], r"tau_scenarios must be \['implied'\]"),
    (["clayton", "student-varying-nu"], ["implied"], "families must all be"),
    (["gaussian"], ["implied"], "families must all be"),
  ],
)
def test_a_shape_family_takes_the_implied_regime_alone(
  families: list[str], tau_scenarios: list[str], match: str
) -> None:
  """A grid that mixes the two kinds of family must be refused at load.

  Its cells are the product of families and regimes, so it would build cells
  that define no copula -- a shape family under a tau regime, or a
  one-parameter family under ``implied`` -- and fail only once a sweep
  reached them.
  """
  with pytest.raises(ValueError, match=match):
    _shape_grid(families=families, tau_scenarios=tau_scenarios)


def test_a_shape_family_needs_surface_x_to_store_surfaces() -> None:
  """A shape family's surfaces must be placed by x, not by target tau.

  Placed by tau, the Student family's surfaces are wherever its tau of 0.5
  meets each level: a level of 0.5 matches all 2001 points of the search and
  stores a surface at each, and any other level falls back to the first.
  """
  with pytest.raises(ValueError, match="surface_x must be set"):
    _shape_grid(surface_families=["student-varying-nu"])
  grid = _shape_grid(surface_families=_SHAPE, surface_x=[0.1, 0.5, 0.9])
  assert grid.surface_x == [0.1, 0.5, 0.9]


@pytest.mark.parametrize("surface_x", [[0.0], [0.5, 1.2]])
def test_surface_x_must_lie_on_the_covariate_support(
  surface_x: list[float],
) -> None:
  """A surface outside ``[X_MIN, X_MAX]`` is an extrapolation no training
  covariate supports, and an R competitor's spline basis is bounded at the
  range of the training ``x``."""
  with pytest.raises(ValueError, match="surface_x entries must lie in"):
    _shape_grid(surface_x=surface_x)


def test_surface_x_and_surface_tau_levels_are_not_both_read(
  tmp_path: Path,
) -> None:
  """With ``surface_x`` set the tau levels are not read, so a config that
  gives both would have one of them dropped without a word."""
  text = _TOML.replace(
    "surface_tau_levels = [0.1, 0.5]",
    "surface_tau_levels = [0.1, 0.5]\nsurface_x = [0.5]",
  )
  with pytest.raises(ValueError, match="surface_tau_levels must be omitted"):
    load_grid(_write(tmp_path, text))


def test_the_misspecification_configs_share_one_grid(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The R competitors must be scored on the backends' exact cells.

  The two configs are compared estimator by estimator, so a data axis that
  drifted between them would compare the methods on different samples. The
  backends must also be E1's, at the logit transform, with E1's estimator
  ids, which is what lets the two studies' tables be joined.
  """
  monkeypatch.chdir(_ROOT)
  backends = load_grid(_ROOT / "configs" / "study_misspec.toml")
  external = load_grid(_ROOT / "configs" / "study_misspec_r.toml")

  for f in fields(GridConfig):
    if f.name != "estimators":
      assert getattr(external, f.name) == getattr(backends, f.name), f.name
  assert backends.families == _SHAPE
  assert backends.surface_x == [0.1, 0.5, 0.9]
  assert backends.n == [200, 500, 1000, 2000]
  assert len(backends.cells()) == 2 * 4 * 10

  e1 = load_grid(_ROOT / "configs" / "study.toml")
  assert {e.estimator_id for e in backends.estimators} <= {
    e.estimator_id for e in e1.estimators
  }
  assert {e.transform for e in backends.estimators} == {"logit"}
  assert {e.backend for e in backends.estimators} == {
    "tabpfn-quantiles",
    "tabicl",
    "nori",
    "lightgbm",
    "xgboost",
    "catboost",
  }
  assert [(e.label, e.backend, e.script) for e in external.estimators] == [
    ("gamcopula", "gamCopula", "r_methods/fit_gamcopula.R"),
    (
      "boostcopula-smooth-cv",
      "boostCopula",
      "r_methods/fit_boostcopula_smooth_cv.R",
    ),
  ]


def test_selecting_an_unknown_label_is_refused(tmp_path: Path) -> None:
  """A mistyped ``--labels`` entry must not run fewer estimators silently."""
  grid = load_grid(_write(tmp_path, _TOML))
  with pytest.raises(EstimatorConfigError, match="quant-logti"):
    select_estimators(grid, ["quant-identity", "quant-logti"])


def test_selecting_estimators_leaves_every_other_axis_alone(
  tmp_path: Path,
) -> None:
  """A restricted run must refit its estimators on the full run's grid.

  Dropping or resetting any other field -- the projection grid, the
  evaluation grids, the diagnostics -- would score the kept estimators on
  different terms than the rows they replace.
  """
  grid = load_grid(_write(tmp_path, _TOML))
  kept = select_estimators(grid, ["quant-identity"])
  assert [e.label for e in kept.estimators] == ["quant-identity"]
  assert kept.estimators[0] == grid.estimators[1]
  for f in fields(GridConfig):
    if f.name != "estimators":
      assert getattr(kept, f.name) == getattr(grid, f.name), f.name


def test_holdout_n_is_read_from_the_grid_and_refused_below_two(
  tmp_path: Path,
) -> None:
  """``holdout_n`` must reach the grid from TOML, and one draw must be refused.

  A key the loader dropped would score every study on the default count
  whatever its config says; a single draw per ``x`` has no standard error.
  """
  config = tmp_path / "study.toml"
  body = (
    '[grid]\nfamilies = ["clayton"]\ntau_scenarios = ["linear"]\n'
    'normalize = ["none"]\nn = [30]\nn_rep = 1\nholdout_n = {n}\n\n'
    '[[grid.estimators]]\nlabel = "lightgbm-probit"\nbackend = "lightgbm"\n'
    'transform = "probit"\n'
  )
  config.write_text(body.format(n=500))
  assert load_grid(config).holdout_n == 500
  config.write_text(body.format(n=1))
  with pytest.raises(ValueError, match="holdout_n must be >= 2"):
    load_grid(config)
