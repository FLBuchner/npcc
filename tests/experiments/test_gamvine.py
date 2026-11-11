"""E2b's GAM vines: the R exchange, the structure conversion, and the arms.

Most tests run the exchange under the Python interpreter with a stand-in
script that "fits" the independence copula on standard normal margins, so
every score it receives has a closed form. The tests marked with ``needs_r``
run ``r_methods/fit_gamvine.R`` itself, and skip without R or its packages.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pyvinecopulib as pv
import torch

from npcc.experiments import (
  _external,
  _external_vine,
  scenarios,
  vine_cli,
  vine_runner,
  vine_scenarios,
)
from npcc.experiments._study import derive_seed
from npcc.experiments.config import RunConfig
from npcc.experiments.vine_config import GAMVINE_ARMS, VineGridConfig
from tests.conftest import hermetic_backend
from tests.experiments.test_e2_analyze import analyze

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "r_methods" / "fit_gamvine.R"

#: The seconds the stand-in reports for margin_time, vine_time,
#: joint_pdf_time and copula_pdf_time. Dyadic, so their sums are exact.
_TIMINGS = (0.5, 0.25, 0.125, 0.0625)

#: A script speaking the exchange for standard normal margins and the
#: independence copula. Each run appends its arm, seed and inputs to a
#: ``.calls`` file beside it. ``MUTATION`` is replaced by statements that
#: break one rule of the exchange, or by nothing.
_STAND_IN = """\
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyvinecopulib as pv

root, seed, arm = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
train = list(csv.DictReader(open(root / "train.csv", newline="")))
points = list(csv.DictReader(open(root / "eval.csv", newline="")))
d = sum(1 for k in train[0] if k.startswith("y"))
given = root / "structure.csv"
structure = given.read_text() if given.is_file() else None
with open(Path(__file__).with_suffix(".calls"), "a") as fh:
  call = {"arm": arm, "seed": seed, "structure": structure}
  fh.write(json.dumps({**call, "train": train, "eval": points}) + "\\n")

timings = dict(zip(
  ["margin_time", "vine_time", "joint_pdf_time", "copula_pdf_time"], TIMINGS
))
rows = []
for p in points:
  log_pdf = sum(
    -0.5 * float(p[f"y{j + 1}"]) ** 2 - 0.5 * math.log(2 * math.pi)
    for j in range(d)
  )
  rows.append({**p, "pdf": math.exp(log_pdf), "copula_pdf": 1.0, **timings})
if structure is not None:
  fitted = structure
else:
  order = pv.RVineStructure.from_order(list(range(1, d + 1)))
  matrix = np.asarray(order.matrix)[::-1]
  fitted = "\\n".join(",".join(str(int(v)) for v in row) for row in matrix)
pairs = [
  {"tree": 1, "first": 1, "second": 2, "conditioning": "", "family": 0,
   "family_name": "independence", "kind": "independence", "formula": "",
   "covariates": "", "edf": 0, "par": 0, "par2": 0}
] * (d * (d - 1) // 2)
MUTATION
(root / "fitted.csv").write_text(fitted)
for name, table in (("pred.csv", rows), ("pairs.csv", pairs)):
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


def _grid(script: Path, *, arms: list[str] | None = None) -> VineGridConfig:
  return VineGridConfig(
    dims=[3],
    n=[40],
    n_rep=1,
    estimators=[],
    arms=list(GAMVINE_ARMS) if arms is None else arms,
    eval_x_n=2,
    eval_m=20,
    gamvine_script=str(script),
  )


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


def _vinecopula_edges(matrix: np.ndarray) -> set[_external_vine.Edge]:
  """A matrix's edges as VineCopula reads it: column i, row k > i."""
  d = matrix.shape[0]
  return {
    (
      frozenset((int(matrix[i, i]), int(matrix[k, i]))),
      frozenset(int(v) for v in matrix[k + 1 :, i]),
    )
    for i in range(d - 1)
    for k in range(i + 1, d)
  }


@pytest.mark.parametrize("d", [3, 4, 5, 10])
def test_a_structure_has_its_edges_in_both_conventions(d: int) -> None:
  """R must be handed the truth's vine, not another vine on its variables.

  Both packages encode a vine in a triangular matrix, but each reads its
  trees from the opposite end, so a matrix passed unconverted encodes a
  different vine, which the oracle arm would then fit as the truth.
  """
  for seed in range(5):
    structure = pv.RVineStructure.sample(d, seeds=[seed + 1])
    matrix = _external_vine.to_vinecopula(structure)
    assert np.array_equal(matrix, np.tril(matrix))
    assert _vinecopula_edges(matrix) == _external_vine.edges(structure)
    back = _external_vine.from_vinecopula(matrix)
    assert np.array_equal(np.asarray(back.matrix), np.asarray(structure.matrix))


# ---------------------------------------------------------------------------
# The exchange, under the stand-in
# ---------------------------------------------------------------------------


def test_r_is_handed_the_cells_sample_and_test_points(
  python_rscript: None, tmp_path: Path
) -> None:
  """Each arm must fit the cell's training sample and score its test points.

  It is paired with E2's arms by cell, so data drawn or ordered otherwise
  would compare the methods on different samples. Only the oracle may see
  the true structure.
  """
  script = _write_stand_in(tmp_path)
  grid = _grid(script)
  cell = grid.cells()[0]
  vine_runner.summarize_one_cell(cell, grid, base_seed=317, device=None)

  y, _, x = vine_runner.training_sample(317, cell)
  truth = vine_runner.truth_for(317, cell.d, cell.rep)
  points = vine_scenarios.eval_points(
    truth,
    scenarios.conditional_x_axis(grid.eval_x_n),
    grid.eval_m,
    derive_seed(317, "eval", cell.d, cell.rep),
  )
  calls = _calls(script)
  assert [c["arm"] for c in calls] == list(GAMVINE_ARMS)
  cell_seed = derive_seed(317, "cell", cell.d, cell.n, cell.rep)
  for call in calls:
    assert call["seed"] == str(derive_seed(cell_seed, call["arm"]))
    train = pd.DataFrame(call["train"]).astype(float)
    assert np.array_equal(train["x"], x[:, 0].numpy())
    assert np.array_equal(train[["y1", "y2", "y3"]], y.numpy())
    test = pd.DataFrame(call["eval"]).astype(float)
    x_axis = points.x_axis.repeat_interleave(grid.eval_m).numpy()
    assert np.array_equal(test["x"], x_axis)
    assert np.array_equal(test[["y1", "y2", "y3"]], torch.cat(points.y))
    assert np.array_equal(test[["u1", "u2", "u3"]], torch.cat(points.u))
    if call["arm"] == "gamvine-oracle":
      given = np.loadtxt(
        str(call["structure"]).splitlines(), delimiter=",", dtype=np.int64
      )
      assert np.array_equal(
        given, _external_vine.to_vinecopula(truth.structure)
      )
    else:
      assert call["structure"] is None


def test_the_densities_are_scored_where_r_evaluated_them(
  python_rscript: None, tmp_path: Path
) -> None:
  """The joint is scored on the data scale, the copula at the true PITs.

  The stand-in's joint density is a product of standard normals and its
  copula density 1, so each score has a closed form; a density matched to
  another point, or to the other scale, would miss it.
  """
  grid = _grid(_write_stand_in(tmp_path), arms=["gamvine-selected"])
  cell = grid.cells()[0]
  metrics, runtime = vine_runner.summarize_one_cell(
    cell, grid, base_seed=317, device=None
  )
  truth = vine_runner.truth_for(317, cell.d, cell.rep)
  points = vine_scenarios.eval_points(
    truth,
    scenarios.conditional_x_axis(grid.eval_x_n),
    grid.eval_m,
    derive_seed(317, "eval", cell.d, cell.rep),
  )
  for k, row in enumerate(metrics):
    normal = (-0.5 * points.y[k].square() - 0.5 * math.log(2 * math.pi)).sum(1)
    assert row["KL"] == pytest.approx(
      float((points.pdf_true[k].log() - normal).mean())
    )
    assert row["KL_copula"] == pytest.approx(
      float(points.copula_pdf_true[k].log().mean())
    )
  [record] = runtime
  assert record["fit_time"] == 0.75
  assert record["pdf_time"] == 0.1875
  assert record["total_time"] == 0.9375
  assert record["error"] == ""
  assert len(json.loads(record["pairs"])) == 3


def test_an_oracle_fit_on_another_structure_is_refused(
  python_rscript: None, tmp_path: Path
) -> None:
  """The oracle arm must fit the true structure, which its rows claim.

  A script that reordered or normalized the matrix into another vine would
  otherwise be scored as the oracle.
  """
  script = _write_stand_in(
    tmp_path,
    "order = pv.RVineStructure.from_order(list(range(d, 0, -1)))\n"
    "matrix = np.asarray(order.matrix)[::-1]\n"
    "fitted = '\\n'.join(','.join(str(int(v)) for v in r) for r in matrix)",
  )
  grid = _grid(script, arms=["gamvine-oracle"])
  for cell_seed in range(20):
    truth = vine_runner.truth_for(cell_seed, 3, 0)
    d_vine = pv.RVineStructure.from_order([3, 2, 1])
    if _external_vine.edges(truth.structure) != _external_vine.edges(d_vine):
      break
  with pytest.raises(_external.ExchangeError, match="true structure"):
    vine_runner.summarize_one_cell(
      grid.cells()[0], grid, base_seed=cell_seed, device=None
    )


@pytest.mark.parametrize(
  ("mutation", "match"),
  [
    ("rows.reverse()", "echo the test points"),
    ("rows = rows[1:]", "one row per test point"),
    ("rows[0]['pdf'] = float('nan')", "finite, nonnegative pdf"),
    ("pairs = pairs[1:]", "one row per pair copula"),
    ("fitted = '1,0\\n2,2'", "3 x 3 R-vine matrix"),
    ("sys.exit(3)", "refused with status 3"),
  ],
)
def test_output_that_breaks_the_exchange_stops_the_run(
  python_rscript: None, tmp_path: Path, mutation: str, match: str
) -> None:
  """A script's defect must stop the study rather than be scored.

  Densities matched to the wrong points, or a structure the runner cannot
  read, would otherwise become a plausible row of the method's results.
  """
  grid = _grid(_write_stand_in(tmp_path, mutation), arms=["gamvine-ns"])
  with pytest.raises(_external.ExchangeError, match=match):
    vine_runner.summarize_one_cell(
      grid.cells()[0], grid, base_seed=317, device=None
    )


def test_a_failed_fit_is_recorded_and_still_pairs(
  python_rscript: None, tmp_path: Path
) -> None:
  """A fit that fails on its sample is the method's outcome, not a crash.

  It must leave the arm's metrics nan and its message in the runtime row,
  and keep the cell's seed and true log-density, so the analysis can count
  the failure while the run still pairs with E2's.
  """
  script = _write_stand_in(
    tmp_path, "if arm == 'gamvine-ns':\n  sys.exit('no convergence')"
  )
  grid = _grid(script)
  metrics, runtime = vine_runner.summarize_one_cell(
    grid.cells()[0], grid, base_seed=317, device=None
  )
  failed = [r for r in metrics if r["arm"] == "gamvine-ns"]
  assert len(failed) == grid.eval_x_n
  assert all(math.isnan(r["KL"]) and math.isnan(r["KL_copula"]) for r in failed)
  fitted = [r for r in metrics if r["arm"] == "gamvine-oracle"]
  assert [r["loglik_true"] for r in failed] == [
    r["loglik_true"] for r in fitted
  ]
  [record] = [r for r in runtime if r["arm"] == "gamvine-ns"]
  assert "no convergence" in record["error"]
  assert math.isnan(record["fit_time"])


def test_an_edited_script_refuses_a_resume(
  python_rscript: None, tmp_path: Path
) -> None:
  """An edited script is a new method, so a resume must refuse to mix them.

  A grid without a GAM arm must keep its signature, so the runs E2 already
  has can still be resumed.
  """
  script = _write_stand_in(tmp_path)
  run = RunConfig(out=tmp_path / "out")
  vine_runner.run_vine_study(_grid(script, arms=["gamvine-ns"]), run)
  script.write_text(script.read_text() + "# edited\n")
  with pytest.raises(ValueError, match="signature"):
    vine_runner.run_vine_study(
      _grid(script, arms=["gamvine-ns"]), run, resume=True
    )

  npcc_only = [
    VineGridConfig(
      dims=[3],
      n=[40],
      n_rep=1,
      estimators=[],
      arms=["tll"],
      gamvine_script=name,
    )
    for name in (str(script), "elsewhere.R")
  ]
  assert len({vine_runner.grid_signature(g, run) for g in npcc_only}) == 1


_MAIN = """
[grid]
dims = [3]
n = [40, 60]
n_rep = 2
arms = ["oracle", "random", "tll"]
eval_x_n = 2
eval_m = 20

[[grid.estimators]]
label = "uni"
backend = "normal-quantile"
transform = "logit"
"""


def test_a_gamvine_run_pairs_with_e2s_in_the_analysis(
  python_rscript: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A GAM-vine run must pass E2's pairing checks and reach its own table.

  It is a separate run on E2's cells, so the analysis must take it through
  --with, refusing it only if it was scored on other data, and must count
  the repetitions in which a fit failed rather than average over them.
  """
  script = _write_stand_in(
    tmp_path,
    "if arm == 'gamvine-ns' and len(train) == 40:\n  sys.exit('failed')",
  )
  main_config = tmp_path / "main.toml"
  main_config.write_text(_MAIN)
  gam_config = tmp_path / "gamvine.toml"
  gam_config.write_text(
    _MAIN.split("[[grid.estimators]]")[0].replace(
      '["oracle", "random", "tll"]', json.dumps(list(GAMVINE_ARMS))
    )
    + f"gamvine_script = '{script}'\n"
  )
  main, gam = tmp_path / "main", tmp_path / "gam"
  with hermetic_backend():
    for config, out in ((main_config, main), (gam_config, gam)):
      args = ["--config", str(config), "--out", str(out)]
      assert vine_cli.main([*args, "--device", "cpu"]) == 0

  monkeypatch.setattr(
    sys, "argv", ["analyze.py", str(main), "--with", str(gam)]
  )
  analyze.main()
  out = main / "analysis"
  table = pd.read_csv(out / "vine_gamvine.csv")
  assert sorted(set(table["arm"])) == sorted(GAMVINE_ARMS)
  assert len(table) == 2 * len(GAMVINE_ARMS)
  ns = table[table["arm"] == "gamvine-ns"].set_index("n")
  assert ns.loc[40, "n_failed"] == 2
  assert ns.loc[40, "n_rep"] == 0
  assert ns.loc[60, "n_failed"] == 0
  assert ns.loc[60, "n_rep"] == 2
  families = pd.read_csv(out / "vine_gamvine_families.csv")
  assert set(families["family_name"]) == {"independence"}
  fits = families.groupby(["n", "arm"])["count"].sum()
  # Two repetitions of three pairs each, less the failed fits.
  assert fits[(40, "gamvine-oracle")] == 6
  assert (40, "gamvine-ns") not in fits.index
  numbers = json.loads((out / "vine_text_numbers.json").read_text())
  assert numbers["gamvine"]["run"] == str(gam)


# ---------------------------------------------------------------------------
# R itself
# ---------------------------------------------------------------------------


def _r_available() -> bool:
  if shutil.which("Rscript") is None:
    return False
  try:
    _external.check_packages(str(_SCRIPT))
  except RuntimeError:
    return False
  return True


needs_r = pytest.mark.skipif(
  not _r_available(),
  reason="needs Rscript with gamCopula, mgcv and VineCopula",
)

#: VineCopula's code of each truth family; each takes pyvinecopulib's
#: parameter as it is.
_VINECOPULA_FAMILY = {
  "gaussian": 1,
  "clayton": 3,
  "gumbel": 4,
  "frank": 5,
  "joe": 6,
}


@needs_r
def test_vinecopula_reads_the_converted_truth_as_pyvinecopulib_does(
  tmp_path: Path,
) -> None:
  """R's density of the converted truth must be pyvinecopulib's.

  This checks the conversion against VineCopula itself rather than against
  this file's reading of its convention: each pair copula sits in the matrix
  where its edge does, so a matrix that put an edge elsewhere would give
  another density.
  """
  truth = vine_scenarios.draw_truth(5, seed=11)
  vine = vine_scenarios.true_vine(truth, 0.3)
  u = np.asarray(vine.simulate(200, seeds=[1, 2]))
  d = truth.d
  family = np.zeros((d, d), dtype=np.int64)
  par = np.zeros((d, d))
  for t, row in enumerate(truth.edges):
    for e, edge in enumerate(row):
      family[d - 1 - t, e] = _VINECOPULA_FAMILY[edge.family]
      par[d - 1 - t, e] = float(
        np.asarray(vine.get_pair_copula(t, e).parameters).item()
      )
  for name, values in (
    ("matrix", _external_vine.to_vinecopula(truth.structure)),
    ("family", family),
    ("par", par),
    ("u", u),
  ):
    np.savetxt(tmp_path / f"{name}.csv", values, delimiter=",", fmt="%.17g")
  code = f"""
  suppressPackageStartupMessages(library(VineCopula))
  read <- function(name) unname(as.matrix(read.csv(
    file.path("{tmp_path}", paste0(name, ".csv")), header = FALSE)))
  rvm <- RVineMatrix(read("matrix"), read("family"), read("par"))
  cat(sprintf("%.17g", RVinePDF(read("u"), rvm)), sep = "\\n")
  """
  done = subprocess.run(
    ["Rscript", "-e", code], capture_output=True, text=True, check=True
  )
  r_pdf = np.array([float(v) for v in done.stdout.split()])
  np.testing.assert_allclose(r_pdf, np.asarray(vine.pdf(u)), rtol=1e-8)


def _small_cell(
  structure: pv.RVineStructure,
  pair_copulas: list[list[pv.Bicop]],
  n: int,
  m: int,
) -> tuple[
  tuple[torch.Tensor, torch.Tensor],
  tuple[torch.Tensor, torch.Tensor, torch.Tensor],
  torch.Tensor,
]:
  """A sample on standard normal margins, test points and their copula pdf."""
  vine = pv.Vinecop.from_structure(
    structure=structure, pair_copulas=pair_copulas
  )
  u_train = torch.from_numpy(np.asarray(vine.simulate(n, seeds=[1])))
  u_test = torch.from_numpy(np.asarray(vine.simulate(m, seeds=[2])))
  x_train = torch.linspace(0.0, 1.0, n, dtype=torch.float64).reshape(-1, 1)
  x_test = torch.linspace(0.0, 1.0, m, dtype=torch.float64).reshape(-1, 1)
  copula_pdf = torch.from_numpy(np.asarray(vine.pdf(u_test.numpy())))
  y_train: torch.Tensor = torch.special.ndtri(u_train)
  y_test: torch.Tensor = torch.special.ndtri(u_test)
  return (y_train, x_train), (y_test, u_test, x_test), copula_pdf


@needs_r
@pytest.mark.parametrize("arm", GAMVINE_ARMS)
def test_negatively_dependent_pairs_keep_their_rotation(arm: str) -> None:
  """A GAM vine must fit a negatively dependent pair as itself.

  gamCopula 0.0.8 stores such a pair's double family mirrored -- its 90- and
  270-degree members exchanged -- in gamVineCopSelect with its default
  rotations, and fits the next tree of gamVineStructureSelect on the
  mirror's h-functions; either costs a sample like this one far more than
  estimation does. The script's own check, that gamVinePDF is the product of
  the fitted pairs' densities, runs within each fit.
  """
  structure = pv.RVineStructure.from_order([1, 2, 3])
  pairs = [
    [
      pv.Bicop(
        pv.BicopFamily.clayton, rotation=90, parameters=np.array([[3.0]])
      ),
      pv.Bicop(
        pv.BicopFamily.gumbel, rotation=270, parameters=np.array([[2.0]])
      ),
    ],
    [
      pv.Bicop(
        pv.BicopFamily.clayton, rotation=90, parameters=np.array([[1.5]])
      )
    ],
  ]
  sample, points, copula_pdf = _small_cell(structure, pairs, 600, 400)
  fit = _external_vine.fit_and_evaluate(
    str(_SCRIPT),
    sample,
    points,
    arm=arm,
    seed=1,
    name=arm,
    structure=structure if arm == "gamvine-oracle" else None,
  )
  kl = float((copula_pdf.log() - fit.copula_pdf.log()).mean())
  assert kl < 0.05
  assert len(fit.pairs) == 3
  if arm == "gamvine-oracle":
    assert _external_vine.edges(fit.structure) == _external_vine.edges(
      structure
    )


def _nonsimplified_sample(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
  """A non-simplified D-vine on 1 - 2 - 3, and its copula density there.

  Both first-tree pairs are Gaussian with correlation 0.85; the pair
  (1, 3 | 2) is a Clayton whose Kendall's tau is 0.05 + 0.5 u2.
  """
  w = np.random.default_rng(seed).uniform(size=(n, 3))
  gaussian = pv.Bicop(pv.BicopFamily.gaussian, parameters=np.array([[0.85]]))
  clayton = pv.Bicop(pv.BicopFamily.clayton)
  u = np.empty_like(w)
  pdf = np.empty(n)
  u[:, 1] = w[:, 1]
  for i in range(n):
    theta = clayton.tau_to_parameters(0.05 + 0.5 * float(u[i, 1]))
    given = pv.Bicop(pv.BicopFamily.clayton, parameters=theta)
    a = w[i, 0]
    b = float(given.hinv1(np.array([[a, w[i, 2]]]))[0])
    u[i, 0] = float(gaussian.hinv2(np.array([[a, u[i, 1]]]))[0])
    u[i, 2] = float(gaussian.hinv2(np.array([[b, u[i, 1]]]))[0])
    first = np.array([[u[i, 0], u[i, 1]], [u[i, 2], u[i, 1]]])
    pdf[i] = float(
      np.prod(gaussian.pdf(first)) * given.pdf(np.array([[a, b]]))[0]
    )
  return u, pdf


@needs_r
def test_a_conditioning_variable_is_not_read_as_a_pair_argument() -> None:
  """The non-simplified arm must see a conditioning variable as itself.

  gamCopula names a pair's arguments u1 and u2 beside its covariates, so a
  conditioning variable named u1 or u2 would be read as the pair's own
  argument, and the fit would miss a dependence that changes with it, as
  this sample's second tree does. The script's own check, that gamVinePDF
  is the product of the pair densities, refuses such a fit too.
  """
  u_train, _ = _nonsimplified_sample(1000, 3)
  u_test, pdf = _nonsimplified_sample(500, 4)
  x_train = torch.linspace(0.0, 1.0, 1000, dtype=torch.float64).reshape(-1, 1)
  x_test = torch.linspace(0.0, 1.0, 500, dtype=torch.float64).reshape(-1, 1)
  y_train: torch.Tensor = torch.special.ndtri(torch.from_numpy(u_train))
  y_test: torch.Tensor = torch.special.ndtri(torch.from_numpy(u_test))
  fit = _external_vine.fit_and_evaluate(
    str(_SCRIPT),
    (y_train, x_train),
    (y_test, torch.from_numpy(u_test), x_test),
    arm="gamvine-ns",
    seed=1,
    name="ns",
  )
  # The true structure, whose second tree conditions on variable 2: the
  # name that collides with the pair's second argument.
  [deep] = [p for p in fit.pairs if p["tree"] == 2]
  assert deep["conditioning"] == "2"
  assert set(str(deep["covariates"]).split()) - {"x"}
  # A simplified fit of this sample, with that pair constant, scores about
  # 0.04; this one about 0.
  kl = float(np.mean(np.log(pdf) - np.log(fit.copula_pdf.numpy())))
  assert kl < 0.02
