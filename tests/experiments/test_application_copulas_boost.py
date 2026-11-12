"""The boostCopula arms: the R exchange, and R itself.

Most tests run the exchange under the Python interpreter with a stand-in
script that "fits" the independence copula, so every score it leads to has
a closed form. The tests marked with ``needs_r`` run
``r_methods/fit_boostcopula_vine.R`` itself, and skip without R or
boostCopula.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import pyvinecopulib as pv

from npcc.experiments import _external
from npcc.experiments._study import derive_seed
from npcc.experiments.application import copulas
from npcc.experiments.application.copulas import (
  CopulaFailure,
  CopulaResult,
  run_copula_arm,
)
from npcc.experiments.application.marginal_set import MarginalSet
from tests.experiments.test_application_copulas import FEATURES, gaussian_set

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "r_methods" / "fit_boostcopula_vine.R"

#: The seconds the stand-in reports for fit_time, pdf_time and sim_time.
#: Dyadic, so their sums are exact.
_TIMINGS = (0.5, 0.25, 0.125)

#: A script speaking the exchange for the independence copula on a D-vine.
#: Each run appends its arguments and inputs to a ``.calls`` file beside it.
#: ``MUTATION`` is replaced by statements that break one rule of the
#: exchange, or by nothing.
_STAND_IN = """\
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyvinecopulib as pv

root, seed, n_draws = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
train = list(csv.DictReader(open(root / "train.csv", newline="")))
test = list(csv.DictReader(open(root / "test.csv", newline="")))
u = [k for k in test[0] if k.startswith("u")]
with open(Path(__file__).with_suffix(".calls"), "a") as fh:
  call = {"seed": seed, "n_draws": n_draws, "train": train, "test": test}
  fh.write(json.dumps(call) + "\\n")

density = [{**{k: row[k] for k in u}, "pdf": 1.0} for row in test]
rng = np.random.default_rng(int(seed))
draws = [
  {"case": i + 1, **dict(zip(u, rng.random(len(u)).tolist()))}
  for i in range(len(test))
  for _ in range(n_draws)
]
order = pv.RVineStructure.from_order(list(range(1, len(u) + 1)))
fitted = np.asarray(order.matrix, dtype=np.int64)[::-1]
families = np.tril(np.ones_like(fitted), -1)
info = [
  {"key": "r_version", "value": "R stand-in"},
  {"key": "packages", "value": "boostCopula 0.0.0, VineCopula 0.0.0"},
  *({"key": k, "value": v} for k, v in zip(
    ["fit_time", "pdf_time", "sim_time"], TIMINGS
  )),
]
MUTATION
np.savetxt(root / "fitted.csv", fitted, fmt="%d", delimiter=",")
np.savetxt(root / "families.csv", families, fmt="%d", delimiter=",")
for name, table in (
  ("density.csv", density), ("draws.csv", draws), ("info.csv", info)
):
  with open(root / name, "w", newline="") as fh:
    writer = csv.DictWriter(fh, list(table[0]))
    writer.writeheader()
    writer.writerows(table)
print("fitted the independence copula")
"""


def _write_stand_in(directory: Path, mutation: str = "") -> Path:
  script = directory / "stand_in.py"
  script.write_text(
    _STAND_IN.replace("TIMINGS", repr(_TIMINGS)).replace("MUTATION", mutation)
  )
  return script


def _calls(script: Path) -> list[dict[str, object]]:
  lines = script.with_suffix(".calls").read_text().splitlines()
  return [json.loads(line) for line in lines]


@pytest.fixture
def python_rscript(monkeypatch: pytest.MonkeyPatch) -> None:
  """The exchange's interpreter pointed at Python, for the stand-in."""
  monkeypatch.setattr(_external, "RSCRIPT", sys.executable)


def _run(arm: str, ms: MarginalSet, *, n_draws: int = 4) -> list[CopulaResult]:
  return run_copula_arm(
    arm,
    ms,
    estimators={},
    n_draws=n_draws,
    sample_batch_size=100,
    mixture_k=1,
    seed=1,
    device=None,
  )


@pytest.fixture
def stand_in(
  python_rscript: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
  script = _write_stand_in(tmp_path)
  monkeypatch.setattr(copulas, "BOOSTCOPULA_SCRIPT", str(script))
  return script


def _mutated(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
  script = _write_stand_in(tmp_path, mutation)
  monkeypatch.setattr(copulas, "BOOSTCOPULA_SCRIPT", str(script))


# ---------------------------------------------------------------------------
# The exchange
# ---------------------------------------------------------------------------


def test_r_is_handed_the_clipped_transforms_and_the_scopes_covariates(
  stand_in: Path,
) -> None:
  """R must fit the clipped transforms on the scope's covariates, as given.

  A transform of exactly 0 is refused by the script, so it must arrive
  clipped; a covariate is a real number and must arrive unchanged; and the
  time scope is the two day-of-year columns, picked by name.
  """
  ms = gaussian_set(d=5)
  pit = ms.pit_train.copy()
  pit[0, 0] = 0.0
  x = ms.x_train.copy()
  x[0, 2] = 1e3
  _run("boostcopula-time", dataclasses.replace(ms, pit_train=pit, x_train=x))
  (call,) = _calls(stand_in)
  train = call["train"]
  assert isinstance(train, list)
  assert list(train[0]) == ["u1", "u2", "u3", "u4", "u5", "x1", "x2"]
  assert float(train[0]["u1"]) == 1e-6
  assert float(train[0]["x1"]) == 1e3
  time = [FEATURES.index(f) for f in copulas.TIME_FEATURES]
  got = np.array([[float(r["x1"]), float(r["x2"])] for r in train])
  np.testing.assert_array_equal(got, x[:, time])
  assert call["seed"] == str(
    derive_seed(1, 0, "emos-seasonal", "boostcopula-time")
  )
  assert call["n_draws"] == 4


def test_the_scores_are_read_from_what_r_wrote(stand_in: Path) -> None:
  """The density, the draws, the structure and R's seconds must be R's own.

  The stand-in's copula is independence, so the joint log-likelihood is the
  margins' alone; its seconds are dyadic, so they add up exactly.
  """
  ms = gaussian_set(d=5)
  (result,) = _run("boostcopula-all", ms)
  assert result.arm == "boostcopula-all"
  np.testing.assert_array_equal(result.logcop_test, np.zeros(6))
  np.testing.assert_array_equal(result.joint_nll, -ms.logpdf_test.sum(axis=1))
  assert np.isfinite(result.energy).all()
  assert result.timings["fit"] == 0.5
  assert result.timings["copula_prediction"] == 0.375
  assert result.timings["bridge_overhead"] >= 0.0
  order = pv.RVineStructure.from_order([1, 2, 3, 4, 5])
  assert result.info["structure"] == np.asarray(order.matrix).tolist()
  assert result.info["covariates"] == list(FEATURES)
  assert result.info["r_packages"] == "boostCopula 0.0.0, VineCopula 0.0.0"
  assert result.info["script_sha256"]


def test_a_failed_r_fit_is_the_arms_failure(
  python_rscript: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """An error in R on a station's data is that arm's outcome there.

  It must be recorded and the run continue, rather than stop every other
  station.
  """
  _mutated(tmp_path, monkeypatch, "sys.exit(1)")
  with pytest.raises(CopulaFailure, match="failed"):
    _run("boostcopula-all", gaussian_set(d=5))


def test_a_refused_input_stops_the_run(
  python_rscript: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A refusal by the script is a defect, not the arm's outcome.

  Recorded as a failure, a broken exchange would read as boostCopula
  failing at every station.
  """
  _mutated(tmp_path, monkeypatch, "sys.exit(3)")
  with pytest.raises(_external.ExchangeError) as info:
    _run("boostcopula-all", gaussian_set(d=5))
  assert not isinstance(info.value, CopulaFailure)


@pytest.mark.parametrize("pdf", ["NA", "0", "-1"])
def test_a_density_that_is_not_positive_fails_with_its_case(
  python_rscript: None,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  pdf: str,
) -> None:
  """A density R could not evaluate must fail the arm, naming the case.

  Its log would be ``nan`` or ``-inf``, and the joint log-likelihood with
  it, at one case only.
  """
  _mutated(tmp_path, monkeypatch, f'density[2]["pdf"] = "{pdf}"')
  with pytest.raises(CopulaFailure, match="density") as info:
    _run("boostcopula-all", gaussian_set(d=5))
  assert info.value.indices == (2,)


@pytest.mark.parametrize(
  "mutation",
  [
    "density.reverse()",
    "draws.pop()",
    'info = [r for r in info if r["key"] != "fit_time"]',
    "fitted[0, 0] = 9",
  ],
)
def test_output_that_breaks_the_exchange_stops_the_run(
  python_rscript: None,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  mutation: str,
) -> None:
  """Densities out of order, a missing draw, timing or a broken matrix stop.

  Each would otherwise score a case with another case's values, or record
  a structure or a time that R did not report.
  """
  _mutated(tmp_path, monkeypatch, mutation)
  with pytest.raises(_external.ExchangeError):
    _run("boostcopula-all", gaussian_set(d=5))


def test_a_dimension_other_than_five_is_refused(stand_in: Path) -> None:
  """The script's initial matrix fixes five variables; another d must stop.

  R would refuse it too, but only after starting; the check names the
  reason before anything runs.
  """
  with pytest.raises(ValueError, match="5 variables"):
    _run("boostcopula-all", gaussian_set(d=4))
  assert not stand_in.with_suffix(".calls").exists()


def test_a_missing_script_stops_the_run(
  python_rscript: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Without its script every boostCopula fit would fail as the method's.

  The interpreter would exit with an error status, which reads as the
  arm's failure at every station.
  """
  missing = tmp_path / "missing.R"
  monkeypatch.setattr(copulas, "BOOSTCOPULA_SCRIPT", str(missing))
  with pytest.raises(RuntimeError, match="must exist"):
    _run("boostcopula-all", gaussian_set(d=5))
  with pytest.raises(RuntimeError, match="must exist"):
    copulas.check_boostcopula(str(missing))


# ---------------------------------------------------------------------------
# R itself
# ---------------------------------------------------------------------------


def _r_available() -> bool:
  if shutil.which("Rscript") is None:
    return False
  try:
    copulas.check_boostcopula(str(_SCRIPT))
  except RuntimeError:
    return False
  return True


needs_r = pytest.mark.skipif(
  not _r_available(),
  reason="needs Rscript with boostCopula, VineCopula and Rfast",
)


@needs_r
def test_the_check_reports_rs_versions() -> None:
  """``--check`` must load the packages and print their versions.

  The preflight runs it before any fit, so a missing package stops the run
  there rather than failing every boostCopula fit.
  """
  done = subprocess.run(
    ["Rscript", str(_SCRIPT), "--check"],
    capture_output=True,
    text=True,
    check=False,
  )
  assert done.returncode == 0, done.stderr
  for package in ("R version", "boostCopula", "VineCopula", "Rfast"):
    assert package in done.stdout
  assert copulas.check_boostcopula(str(_SCRIPT)) == done.stdout.strip()


@needs_r
def test_boostcopula_fits_and_scores_a_small_sample(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """boostCopula's vine must fit, evaluate and draw through the exchange.

  On a small sample: every score finite, the structure a valid 5-variable
  R-vine with a family on every pair, and R's versions recorded.
  """
  monkeypatch.setattr(copulas, "BOOSTCOPULA_SCRIPT", str(_SCRIPT))
  ms = gaussian_set(d=5, n_train=40, n_test=3)
  (result,) = _run("boostcopula-time", ms, n_draws=4)
  for scores in (
    result.logcop_test,
    result.joint_nll,
    result.energy,
    result.variogram,
  ):
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
  structure = np.asarray(result.info["structure"], dtype=np.uint64)
  pv.RVineStructure.from_matrix(np.asfortranarray(structure))
  families = np.asarray(result.info["families_vinecopula"])
  assert (families[np.tril_indices(5, -1)] != 0).all()
  assert "boostCopula" in str(result.info["r_packages"])
  assert result.timings["fit"] > 0
