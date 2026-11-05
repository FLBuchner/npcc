"""External estimators: the ``external_r`` kind, its exchange, and a CLI run.

R is not needed: every test runs the exchange under the Python interpreter,
with a stand-in script that predicts the independence copula -- density 1,
``hfunc1 = v``, ``hfunc2 = u``, Kendall's tau 0 -- so each score it receives
has a closed form.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch

from npcc.core.controls import FitControlsRosenblattBicop
from npcc.core.errors import EstimatorConfigError
from npcc.experiments import _external, _study, cli, metrics, scenarios
from npcc.experiments.config import (
  EstimatorSpec,
  GridConfig,
  RunConfig,
  load_grid,
)
from npcc.experiments.runner import _grid_signature, run_study
from npcc.experiments.vine_config import VineGridConfig
from tests.conftest import hermetic_backend

_ROOT = Path(__file__).resolve().parents[2]

#: The timings the stand-in reports: fit, then pdf, hfunc1, hfunc2 and tau.
#: Dyadic, so every sum the runtime table takes of them is exact.
_TIMINGS = (0.5, 0.125, 0.25, 0.25, 0.0625)

#: A script speaking the exchange for the independence copula. ``MUTATION``
#: is replaced by a statement that breaks one rule of the exchange, or by
#: nothing. Each run appends its seed, training size and grids to a ``.calls``
#: file beside the script.
_STAND_IN = """\
import csv
import sys
from pathlib import Path

train_csv, eval_csv, pred_csv, seed = sys.argv[1:]
with open(train_csv, newline="") as fh:
  n_train = sum(1 for _ in csv.DictReader(fh))
with open(eval_csv, newline="") as fh:
  points = list(csv.DictReader(fh))
grids = list(dict.fromkeys(p["grid"] for p in points))
with open(Path(__file__).with_suffix(".calls"), "a") as fh:
  fh.write(" ".join([seed, str(n_train), *grids]) + "\\n")

columns = ["grid", "u1", "u2", "x", "pdf", "hfunc1", "hfunc2", "tau"]
timing_columns = ["fit_time", "pdf_time", "hfunc1_time", "hfunc2_time"]
timing_columns += ["tau_time"]
columns += timing_columns
timings = dict(zip(timing_columns, TIMINGS))
rows = [
  {**p, "pdf": 1.0, "hfunc1": p["u2"], "hfunc2": p["u1"], "tau": 0.0, **timings}
  for p in points
]
MUTATION
with open(pred_csv, "w", newline="") as fh:
  writer = csv.DictWriter(fh, columns, extrasaction="ignore")
  writer.writeheader()
  writer.writerows(rows)
print("fitted the independence copula")
"""


def _write_stand_in(directory: Path, mutation: str = "") -> Path:
  script = directory / "stand_in.py"
  script.write_text(
    _STAND_IN.replace("TIMINGS", repr(_TIMINGS)).replace("MUTATION", mutation)
  )
  return script


@pytest.fixture
def stand_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  """The stand-in script, with the exchange's interpreter pointed at Python."""
  monkeypatch.setattr(_external, "RSCRIPT", sys.executable)
  return _write_stand_in(tmp_path)


def _spec(script: Path, label: str = "independence") -> EstimatorSpec:
  return EstimatorSpec(
    label=label,
    backend="stand-in",
    transform="identity",
    kind="external_r",
    script=str(script),
  )


def _grid(
  *estimators: EstimatorSpec,
  normalize: tuple[int | None, ...] = (None,),
  tau_scenarios: tuple[str, ...] = ("linear",),
) -> GridConfig:
  return GridConfig(
    families=["clayton"],
    tau_scenarios=list(tau_scenarios),
    estimators=list(estimators),
    normalize=list(normalize),
    n=[30],
    n_rep=1,
    conditional_uv_grid_n=4,
    conditional_x_grid_n=3,
    surface_tau_levels=[0.4],
    tau_diagnostic_n=50,
  )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_an_external_entry_parses_with_its_script(stand_in: Path) -> None:
  """A ``[[grid.estimators]]`` table must reach the spec as ``external_r``.

  Read as a registry backend, the entry would be refused for naming no
  registered backend, or fitted as one if its name happened to match.
  """
  config = stand_in.parent / "study.toml"
  config.write_text(
    '[grid]\nfamilies = ["clayton"]\ntau_scenarios = ["linear"]\n'
    'normalize = ["none"]\nn = [30]\nn_rep = 1\n\n'
    '[[grid.estimators]]\nlabel = "independence"\nkind = "external_r"\n'
    f'backend = "stand-in"\ntransform = "identity"\nscript = \'{stand_in}\'\n'
  )

  (spec,) = load_grid(config).estimators

  assert spec.kind == "external_r"
  assert spec.script == str(stand_in)
  assert spec.script_sha256 == hashlib.sha256(stand_in.read_bytes()).hexdigest()
  assert spec.canonical_dict()["script_sha256"] == spec.script_sha256


def test_a_backend_estimator_keeps_its_id_and_signature(tmp_path: Path) -> None:
  """A backend's estimator id and resume signature must not see ``kind``.

  The id seeds every fit and joins result rows, and the signature guards each
  checkpoint, so a shift in either would re-seed a study that is already
  running and refuse its ``--resume``.
  """
  implicit = EstimatorSpec(
    label="a", backend="tabpfn-quantiles", transform="logit"
  )
  explicit = EstimatorSpec(
    label="b", backend="tabpfn-quantiles", transform="logit", kind="backend"
  )
  assert implicit.canonical_dict() == {
    "backend": "tabpfn-quantiles",
    "transform": "logit",
    "backend_kwargs": {},
  }
  # The hash `test_config.py` pins for this spec.
  golden = "eaf54a98cc1b68d623481297d830f5c1c4f8fe9d373dc9f73d4c1e10f5d33999"
  assert implicit.estimator_id == explicit.estimator_id == golden

  config = tmp_path / "study.toml"
  config.write_text(
    '[grid]\nfamilies = ["clayton"]\ntau_scenarios = ["linear"]\n'
    'normalize = ["none"]\nn = [30]\nn_rep = 1\n\n'
    '[[grid.estimators]]\nlabel = "implicit"\nbackend = "lightgbm"\n'
    'transform = "probit"\n\n'
    '[[grid.estimators]]\nlabel = "explicit"\nkind = "backend"\n'
    'backend = "lightgbm"\ntransform = "probit"\n'
  )
  ids = {e.estimator_id for e in load_grid(config).estimators}
  assert len(ids) == 1

  grid = GridConfig(
    families=["clayton"],
    tau_scenarios=["linear"],
    estimators=[
      EstimatorSpec(
        label="quant",
        backend="tabpfn-quantiles",
        transform="logit",
        backend_kwargs={"model_version": "v3"},
      )
    ],
    normalize=[None],
    n=[50],
    n_rep=1,
  )
  # Pinned: a backend-only grid hashes no field an external estimator adds.
  assert (
    _grid_signature(grid, RunConfig(out=tmp_path))
    == "193ed1911aa344dfe95686dbb77381ade99d46b96cef6fe41f5f795b0d2f48a5"
  )


@pytest.mark.parametrize(
  ("overrides", "match"),
  [
    ({"transform": "logit"}, "transform must be 'identity'"),
    ({"backend_kwargs": {"familyset": 3}}, "backend_kwargs must be empty"),
    ({"backend": ""}, "backend must name the method"),
    ({"script": None}, "script must name an existing file"),
    ({"script": "no/such/script.R"}, "script must name an existing file"),
    ({"kind": "external_julia"}, "kind must be one of"),
  ],
)
def test_an_external_entry_refuses_what_its_script_cannot_honor(
  stand_in: Path, overrides: dict[str, object], match: str
) -> None:
  """A setting an external script cannot honor must be refused, not dropped.

  The script never sees a transform or a backend kwarg, so accepting either
  would label rows with a configuration that never ran.
  """
  settings: dict[str, object] = {
    "label": "independence",
    "backend": "stand-in",
    "transform": "identity",
    "kind": "external_r",
    "script": str(stand_in),
  }
  settings.update(overrides)
  with pytest.raises(EstimatorConfigError, match=match):
    EstimatorSpec(**settings)  # ty: ignore[invalid-argument-type] - mistyped values exercise the guard


def test_a_backend_entry_refuses_a_script(stand_in: Path) -> None:
  """A ``script`` on a backend entry means a forgotten ``kind``.

  Accepted, it would fit the registry backend and ignore the script.
  """
  with pytest.raises(EstimatorConfigError, match="script must be omitted"):
    EstimatorSpec(
      label="quant",
      backend="tabpfn-quantiles",
      transform="logit",
      script=str(stand_in),
    )


@pytest.mark.parametrize(
  ("normalize", "tau_scenarios", "match"),
  [
    ((None, 5), ("linear",), "normalize must be"),
    ((None,), ("linear", "uncond35"), "must all be conditional"),
  ],
)
def test_a_grid_refuses_axes_an_external_script_cannot_honor(
  stand_in: Path,
  normalize: tuple[int | None, ...],
  tau_scenarios: tuple[str, ...],
  match: str,
) -> None:
  """A script returns no projected density and is fitted on a covariate.

  Without the check, a Sinkhorn entry or an unconditional scenario would only
  fail once the sweep reached it, hours into a run.
  """
  with pytest.raises(EstimatorConfigError, match=match):
    _grid(_spec(stand_in), normalize=normalize, tau_scenarios=tau_scenarios)


def test_an_external_estimator_has_no_fit_controls(stand_in: Path) -> None:
  """Every vine pair is a ``RosenblattBicop``, which a script is not.

  The vine grid has to refuse an external estimator at load; otherwise the run
  would reach ``controls()`` mid-sweep, which must refuse it in turn rather
  than build a pair on a backend named after an R package.
  """
  spec = _spec(stand_in)
  with pytest.raises(EstimatorConfigError, match="vine-study estimators"):
    VineGridConfig(dims=[3], n=[40], n_rep=1, estimators=[spec])
  with pytest.raises(EstimatorConfigError, match="controls must be built"):
    spec.controls(
      FitControlsRosenblattBicop,
      device=None,
      projection_grid_size=30,
      batch_size=None,
    )


def test_the_r_study_scores_the_bivariate_study_cells(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The R baselines must be scored on the bivariate study's exact cells.

  The two studies are compared estimator by estimator, so a data axis that
  drifted between their configs would compare them on different samples.
  """
  monkeypatch.chdir(_ROOT)
  bivariate = load_grid(_ROOT / "configs" / "study.toml")
  external = load_grid(_ROOT / "configs" / "study_r.toml")

  for f in fields(GridConfig):
    if f.name != "estimators":
      assert getattr(external, f.name) == getattr(bivariate, f.name), f.name
  assert external.cells() == bivariate.cells()
  assert [(e.label, e.kind, e.backend) for e in external.estimators] == [
    ("gamcopula", "external_r", "gamCopula"),
    ("boostcopula", "external_r", "boostCopula"),
  ]


# ---------------------------------------------------------------------------
# The exchange
# ---------------------------------------------------------------------------


def _probe(script: Path) -> _external.ExternalFit:
  grid = scenarios.eval_grid(
    "linear", conditional_uv_grid_n=2, conditional_x_grid_n=2
  )
  u = torch.tensor([0.2, 0.7], dtype=torch.float64)
  x = torch.tensor([[0.1], [0.9]], dtype=torch.float64)
  return _external.fit_and_predict(
    str(script), (u, u.flip(0), x), {"metric": grid}, seed=1, name="probe"
  )


def test_a_failing_script_reports_its_stderr(stand_in: Path) -> None:
  """A script that fails must stop the cell and say why.

  Its stderr is the only record of an R error, so a bare non-zero status would
  leave a shard's log naming no cause.
  """
  stand_in.write_text(
    "import sys\nsys.stderr.write('fit diverged\\n')\nsys.exit(3)\n"
  )
  with pytest.raises(RuntimeError, match=r"status 0 for probe; got 3") as info:
    _probe(stand_in)
  assert "fit diverged" in str(info.value)


def test_a_scripts_report_on_its_fit_is_logged(
  stand_in: Path, caplog: pytest.LogCaptureFixture
) -> None:
  """What a script prints about its fit must reach the run's log.

  It is the only record of the model each fit selected, such as its family,
  since the result tables have no column for it.
  """
  with caplog.at_level("INFO", logger=_external.__name__):
    _probe(stand_in)
  assert "probe: fitted the independence copula" in caplog.messages


def test_a_missing_interpreter_is_named(
  stand_in: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Without R on the path, the error must name the interpreter it needs."""
  monkeypatch.setattr(_external, "RSCRIPT", "no-such-rscript")
  with pytest.raises(RuntimeError, match="no-such-rscript must be on PATH"):
    _probe(stand_in)


@pytest.mark.parametrize(
  ("mutation", "match"),
  [
    ("rows = rows[:-1]", "one row per evaluation point"),
    ("rows = rows[::-1]", "echo the evaluation points in their order"),
    ("rows[1]['x'] = 0.5", "got .* at row 1"),
    ("rows[0]['pdf'] = 'nan'", "finite pdf"),
    ("rows[1]['hfunc1'] = 'inf'", "finite hfunc1"),
    ("columns.remove('tau_time')", r"got no \['tau_time'\]"),
    ("pred_csv += '.elsewhere'", "must write its predictions"),
  ],
)
def test_predictions_that_break_the_exchange_are_refused(
  stand_in: Path, mutation: str, match: str
) -> None:
  """Predictions must cover every point, in order, with finite values.

  A missing or reordered row would score a prediction against another point's
  truth, and a non-finite one would turn the metric into a NaN that the
  summaries then skip, so that repetition would drop out of the means unseen.
  Reversing one grid keeps every row's ``grid``, so only the echoed points
  can show it.
  """
  script = _write_stand_in(stand_in.parent, mutation)
  with pytest.raises(ValueError, match=match):
    _probe(script)


def test_an_external_study_leaves_cuda_uninitialized(
  stand_in: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A study of external estimators alone must not open a CUDA context.

  Its scripts run on the host, and a context opened by each of its processes
  would take device memory from any study already running on the GPU.
  Resetting the peak-memory counters is the call that would open one.
  """
  monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

  def opens_a_context(device: object = None) -> None:
    raise AssertionError("an external estimator initialized CUDA")

  monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", opens_a_context)
  runtime = run_study(_grid(_spec(stand_in)), RunConfig(out=tmp_path))[3]
  assert runtime["gpu_peak_reserved_mb"].to_numpy() == pytest.approx(0.0)
  assert runtime["gpu_peak_alloc_mb"].to_numpy() == pytest.approx(0.0)


def test_an_edited_script_refuses_a_resume(
  stand_in: Path, tmp_path: Path
) -> None:
  """An edited script is a new estimator, so a ``--resume`` must refuse it.

  Under the old id, its rows would sit beside the old script's, and every
  summary would average the two methods as one.
  """
  run = RunConfig(out=tmp_path / "out")
  run_study(_grid(_spec(stand_in)), run)
  stand_in.write_text(stand_in.read_text() + "# edited\n")
  with pytest.raises(ValueError, match="signature"):
    run_study(_grid(_spec(stand_in)), run, resume=True)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def external_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
  """A CLI run of a backend and the stand-in over two cells, as CSV tables."""
  root = tmp_path_factory.mktemp("external")
  script = _write_stand_in(root)
  config = root / "study.toml"
  config.write_text(
    '[grid]\nfamilies = ["clayton"]\ntau_scenarios = ["linear"]\n'
    'normalize = ["none"]\nn = [30]\nn_rep = 2\nconditional_uv_grid_n = 4\n'
    "conditional_x_grid_n = 3\nsurface_tau_levels = [0.4]\n"
    "tau_diagnostic_n = 50\n\n"
    '[[grid.estimators]]\nlabel = "normal-logit"\nbackend = "normal-quantile"\n'
    'transform = "logit"\n\n'
    '[[grid.estimators]]\nlabel = "independence"\nkind = "external_r"\n'
    f'backend = "stand-in"\ntransform = "identity"\nscript = \'{script}\'\n'
  )
  out = root / "out"
  with hermetic_backend(), pytest.MonkeyPatch.context() as mp:
    mp.setattr(_external, "RSCRIPT", sys.executable)
    args = ["--config", str(config), "--out", str(out), "--format", "csv"]
    assert cli.main(args) == 0
  return root


def _table(run: Path, name: str, label: str = "independence") -> pd.DataFrame:
  df = _study.read_table(run / "out" / name, "csv")
  return cast("pd.DataFrame", df[df["label"] == label].reset_index(drop=True))


def test_one_script_run_predicts_both_grids(external_run: Path) -> None:
  """Each cell must run the script once, with both grids in the one run.

  A second run would refit the method, so the stored surfaces would come from
  another model than the one the metrics score.
  """
  calls = (external_run / "stand_in.calls").read_text().splitlines()
  assert len(calls) == 2
  assert all(call.split()[1:] == ["30", "metric", "surface"] for call in calls)
  assert len({call.split()[0] for call in calls}) == 2


def test_external_predictions_are_scored_against_the_truth(
  external_run: Path,
) -> None:
  """A script's predictions must be scored at the points they were made at.

  The independence copula has a closed-form error against the cell's truth,
  so a prediction attached to the wrong point, grid or quantity shows up as a
  wrong metric.
  """
  grid = scenarios.eval_grid(
    "linear", conditional_uv_grid_n=4, conditional_x_grid_n=3
  )
  truth = scenarios.ground_truth("clayton", "linear", grid)
  independence = {
    "pdf": torch.ones_like(grid.u_flat),
    "hfunc1": grid.v_flat,
    "hfunc2": grid.u_flat,
  }
  assert grid.x_axis is not None
  scored = _table(external_run, "metrics_by_x")
  for q, pred in independence.items():
    for i, x_val in enumerate(grid.x_axis.tolist()):
      expected = metrics.grid_metrics(
        truth[q][:, i], pred.reshape(grid.shape)[:, i], include_kl=q == "pdf"
      )
      at = scored[(scored["quantity"] == q) & np.isclose(scored["x"], x_val)]
      # Both repetitions score the same prediction against the same truth.
      for name in ("IAE", "ISE", "KL"):
        got = at[name].tolist()
        assert got == pytest.approx([expected[name]] * 2, nan_ok=True), name

  surface = _table(external_run, "quantities")
  assert set(surface["quantity"]) == {"pdf", "hfunc1", "hfunc2"}
  pdf = surface[surface["quantity"] == "pdf"]
  assert pdf["pred"].to_numpy() == pytest.approx(1.0)
  for q, argument in (("hfunc1", "v"), ("hfunc2", "u")):
    at = surface[surface["quantity"] == q]
    assert at["pred"].to_numpy() == pytest.approx(at[argument].to_numpy())


def test_external_rows_carry_the_scripts_tau_and_timings(
  external_run: Path,
) -> None:
  """The tau and timing columns must be the script's own numbers.

  Its tau is the fitted model's, and its timings leave out the interpreter's
  start-up; replacing either with a harness measurement would compare R's
  baselines with the backends on a different footing.
  """
  diagnostics = _table(external_run, "diagnostics")
  assert diagnostics["tau_hat"].to_numpy() == pytest.approx(0.0)
  assert diagnostics["tau_abs_err"].to_numpy() == pytest.approx(
    diagnostics["tau_true"].to_numpy()
  )

  runtime = _table(external_run, "runtime")
  fit, pdf, h1, h2, tau = _TIMINGS
  expected = {
    "fit_time": fit,
    "pdf_time": pdf,
    "h1_time": h1,
    "h2_time": h2,
    "tau_time": tau,
    "surface_time": pdf + h1 + h2,
    "total_estimator_time": fit + tau + 2 * (pdf + h1 + h2),
  }
  assert len(runtime) == 2
  for column, seconds in expected.items():
    assert runtime[column].to_numpy() == pytest.approx(seconds), column
  assert set(runtime["backend"]) == {"stand-in"}
  assert set(runtime["transform"]) == {"identity"}


def test_the_config_echo_records_the_script(external_run: Path) -> None:
  """The echoed config must name the script and its content.

  It is the only record of which script version produced a results folder.
  """
  echo = json.loads((external_run / "out" / "config.json").read_text())
  script = external_run / "stand_in.py"
  (entry,) = [
    e for e in echo["grid"]["estimators"] if e["label"] == "independence"
  ]
  assert entry["kind"] == "external_r"
  assert entry["script"] == str(script)
  assert (
    entry["script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
  )
  backend = [
    e for e in echo["grid"]["estimators"] if e["label"] == "normal-logit"
  ]
  assert "kind" not in backend[0]
