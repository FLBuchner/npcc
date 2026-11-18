"""E9's scenarios: fifteen covariates, three regimes, and their exact truth."""

from __future__ import annotations

import io
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pyvinecopulib as pv
import torch
from scipy.stats import kendalltau

from npcc.experiments import covariate_scenarios as cs
from npcc.experiments.scenarios import TAU_SCENARIOS


def _x(n: int = 400, seed: int = 0) -> torch.Tensor:
  return cs.covariates(n, torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("family", cs.SCENARIO_FAMILIES)
@pytest.mark.parametrize("scenario", cs.SCENARIOS)
def test_only_the_active_covariates_move_the_copula(
  family: str, scenario: str
) -> None:
  """An inactive covariate with an effect would make it relevant.

  The study's questions -- what irrelevant covariates cost, whether npcc
  overfits under constant -- assume that only :data:`ACTIVE` moves the truth.
  """
  x = _x()
  base = cs.kendall_tau(family, scenario, x)
  redrawn = _x(seed=1)
  for name in cs.COVARIATES:
    k = cs.column(name)
    moved = x.clone()
    moved[:, k] = redrawn[:, k]
    changed = not torch.equal(cs.kendall_tau(family, scenario, moved), base)
    assert changed == (name in cs.ACTIVE[scenario]), name


def test_no_scenario_reads_back_from_a_csv_as_missing() -> None:
  """A scenario pandas reads as NaN drops its rows from every groupby.

  The tables are written as CSV, and read back by whoever analyzes them,
  with pandas' defaults.
  """
  frame = pd.DataFrame({"scenario": cs.SCENARIOS})
  back = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
  assert back["scenario"].tolist() == list(cs.SCENARIOS)


@pytest.mark.parametrize("family", cs.SCENARIO_FAMILIES)
def test_gampcc_calibrates_kendalls_tau(family: str) -> None:
  """The truth must be the model gamCopula fits, or its formula is not true.

  gamCopula's fits calibrate Kendall's tau, ``tau = tanh(eta / 2)``, for
  every family; the parameter follows from tau. A link on the parameter, as
  its ``tau = FALSE`` has, would leave the oracle's formula misspecified.
  The Clayton's tau is negative where ``eta`` is, by its rotation.
  """
  x = _x()
  eta = cs.eta(x)
  assert (eta < 0).any() and (eta > 0).any()
  torch.testing.assert_close(
    cs.kendall_tau(family, "gampcc", x),
    torch.tanh(eta / 2.0),
    rtol=0.0,
    atol=1e-12,
  )


def test_gampcc_eta_is_vatter_and_naglers() -> None:
  """A coefficient or a smooth off by a sign would score another model."""
  x = torch.zeros(1, len(cs.COVARIATES), dtype=torch.float64)
  for name, value in {
    "b1": 1,
    "b3": 1,
    "z1": 1.0,
    "z2": 2.0,
    "t1": 1.0,
    "t2": 0.25,
    "t3": 1.0 / 12.0,
  }.items():
    x[0, cs.column(name)] = value
  # b: (1 - 1) / 4; z: (1 - 2) / 4; s1(1) = 1/4; s2(1/4) = 1/4;
  # s3(1/12) = 1/4. Flipping any one sign moves eta by at least 1/2.
  eta = 0.0 - 0.25 + 0.25 + 0.25 + 0.25
  assert float(cs.eta(x)[0]) == pytest.approx(eta, abs=1e-15)
  for family in cs.SCENARIO_FAMILIES:
    tau = float(cs.kendall_tau(family, "gampcc", x)[0])
    assert tau == pytest.approx(math.tanh(eta / 2.0), abs=1e-15)


def _r_has(package: str) -> bool:
  rscript = shutil.which("Rscript")
  if rscript is None:
    return False
  done = subprocess.run(
    [rscript, "-e", f"library({package})"], capture_output=True, check=False
  )
  return done.returncode == 0


#: Fits ``family`` with the formula ``~ e``, sets its coefficients so that
#: the calibration function is ``e`` itself, and predicts tau and the
#: parameter where ``e`` is each value of ``eta.csv``.
_GAMCOPULA_AT_ETA = """\
suppressPackageStartupMessages({library(gamCopula); library(VineCopula)})
args <- commandArgs(trailingOnly = TRUE)
eta <- read.csv(args[[1]])$eta
set.seed(1)
n <- 200
fitted <- gamBiCopFit(
  data.frame(u1 = runif(n), u2 = runif(n), e = rnorm(n)), ~ e,
  family = as.integer(args[[2]]), tau = TRUE, n.iters = 1
)$res
fitted@model$coefficients[] <- c(0, 1)
at <- gamBiCopPredict(
  fitted, newdata = data.frame(e = eta), target = c("calib", "tau", "par")
)
out <- data.frame(calib = at$calib, tau = at$tau, par = at$par)
write.csv(
  data.frame(lapply(out, sprintf, fmt = "%.17g")), args[[3]],
  row.names = FALSE, quote = FALSE
)
"""


@pytest.mark.skipif(
  not _r_has("gamCopula"), reason="needs Rscript with gamCopula"
)
@pytest.mark.parametrize(
  ("family", "code"), [("gaussian", 1), ("clayton", 301)]
)
def test_the_truth_is_what_gamcopula_predicts_at_its_eta(
  tmp_path: Path, family: str, code: int
) -> None:
  """The oracle is correctly specified only if gamCopula's model is the truth.

  A gamCopula fit of the true family whose calibration function is the true
  ``eta`` must predict the true tau and parameter. gamCopula shrinks tau by
  a factor ``1 - 1e-8`` at prediction, hence the tolerance; the double
  Clayton's parameter is VineCopula's, negative for the 90-degree rotation.
  """
  x = _x(n=300, seed=5)
  eta = cs.eta(x)
  (tmp_path / "eta.csv").write_text(
    "eta\n" + "\n".join(f"{v:.17g}" for v in eta.tolist()) + "\n"
  )
  script = tmp_path / "at_eta.R"
  script.write_text(_GAMCOPULA_AT_ETA)
  out = tmp_path / "out.csv"
  subprocess.run(
    [
      shutil.which("Rscript") or "Rscript",
      str(script),
      str(tmp_path / "eta.csv"),
      str(code),
      str(out),
    ],
    check=True,
    capture_output=True,
  )
  r = np.loadtxt(out, delimiter=",", skiprows=1)
  np.testing.assert_allclose(r[:, 0], eta.numpy(), rtol=0.0, atol=1e-12)
  tau = cs.kendall_tau(family, "gampcc", x).numpy()
  np.testing.assert_allclose(r[:, 1], tau, rtol=0.0, atol=2e-8)
  cops = cs.copulas(family, "gampcc", x)
  par = np.array(
    [
      float(c.parameters.item()) * (-1.0 if c.rotation == 90 else 1.0)
      for c in cops
    ]
  )
  np.testing.assert_allclose(r[:, 2], par, rtol=1e-6, atol=1e-9)


def test_a_zero_eta_is_a_valid_clayton() -> None:
  """``theta = |eta|`` at zero is outside pyvinecopulib's Clayton bounds."""
  x = torch.zeros(1, len(cs.COVARIATES), dtype=torch.float64)
  x[0, cs.column("t1")] = 0.5
  assert float(cs.eta(x)[0]) == 0.0
  (cop,) = cs.copulas("clayton", "gampcc", x)
  assert float(cop.parameters.item()) == cs.CLAYTON_BOUNDS[0]


def test_interaction_switches_e1s_regimes_on_b1() -> None:
  """The interaction must be E1's regimes, so the two studies' bands agree."""
  x = _x()
  tau = cs.kendall_tau("gaussian", "interaction", x)
  t1, b1 = x[:, cs.column("t1")], x[:, cs.column("b1")]
  for level, regime in enumerate(cs.B1_REGIMES):
    tau_of_x = TAU_SCENARIOS[regime].tau_of_x
    assert tau_of_x is not None
    rows = b1 == level
    torch.testing.assert_close(
      tau[rows], tau_of_x(t1[rows]), rtol=0.0, atol=1e-12
    )


def test_the_design_is_vatter_and_naglers() -> None:
  """A covariate of the wrong type or range would change what is learned."""
  x = _x(n=20_000)
  b = x[:, :5]
  z = x[:, 5:10]
  t = x[:, 10:]
  assert set(b.unique().tolist()) == {0.0, 1.0}
  assert float(b.mean()) == pytest.approx(0.5, abs=0.02)
  assert float(z.mean()) == pytest.approx(0.0, abs=0.03)
  assert float(z.std()) == pytest.approx(1.0, abs=0.03)
  assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0
  # Independent columns: no correlation beyond sampling noise.
  corr = np.corrcoef(x.numpy(), rowvar=False)
  assert np.abs(corr - np.eye(15)).max() < 0.05


def test_a_sample_is_its_seed() -> None:
  """A cell's estimators must share its data, drawn from its seed alone."""
  a = cs.sample("clayton", "gampcc", 50, 3)
  b = cs.sample("clayton", "gampcc", 50, 3)
  c = cs.sample("clayton", "gampcc", 50, 4)
  for left, right in zip(a, b, strict=True):
    assert torch.equal(left, right)
  assert not torch.equal(a[0], c[0])


def test_the_sample_follows_each_rows_copula() -> None:
  """A rotation mixed up in the sampler would flip the dependence's sign."""
  u, v, x = cs.sample("clayton", "gampcc", 6000, 11)
  eta = cs.eta(x)
  low, high = eta < -0.4, eta > 0.4
  assert kendalltau(u[low], v[low]).statistic < -0.05
  assert kendalltau(u[high], v[high]).statistic > 0.05
  u, v, _ = cs.sample("gaussian", "constant", 3000, 12)
  assert kendalltau(u, v).statistic == pytest.approx(0.5, abs=0.03)


def test_the_metric_grid_is_e1s_layout() -> None:
  """Point ``i * m + j`` must be pair ``i`` at point ``j``, as E1 reads it."""
  pts = cs.points(4, 0)
  grid = cs.metric_grid(pts, 3)
  assert grid.shape == (9, 4)
  x = grid.x_flat
  assert x is not None
  for i in range(9):
    for j in range(4):
      assert torch.equal(x[i * 4 + j], pts[j])
  assert torch.equal(
    grid.u_flat.reshape(9, 4)[:, 0], grid.u_flat.reshape(9, 4)[:, 3]
  )


def test_each_points_draws_are_its_own_seeds() -> None:
  """A point's draws must not move when another point's seed does."""
  pts = cs.points(3, 0)
  a = cs.holdout_grid("clayton", "gampcc", pts, 20, [1, 2, 3])
  b = cs.holdout_grid("clayton", "gampcc", pts, 20, [1, 9, 3])
  ua, ub = a.u_flat.reshape(20, 3), b.u_flat.reshape(20, 3)
  assert torch.equal(ua[:, 0], ub[:, 0]) and torch.equal(ua[:, 2], ub[:, 2])
  assert not torch.equal(ua[:, 1], ub[:, 1])


@pytest.mark.parametrize("family", cs.SCENARIO_FAMILIES)
def test_the_truth_is_each_points_copula(family: str) -> None:
  """A column scored against another point's copula would be a wrong truth."""
  pts = cs.points(5, 2)
  grid = cs.holdout_grid(family, "gampcc", pts, 30, list(range(5)))
  truth = cs.ground_truth(family, "gampcc", grid)
  pairs = grid.uv.reshape(30, 5, 2)
  for j, cop in enumerate(cs.copulas(family, "gampcc", pts)):
    uv = pairs[:, j].numpy()
    assert isinstance(cop, pv.Bicop)
    np.testing.assert_array_equal(truth["pdf"][:, j].numpy(), cop.pdf(uv))
    np.testing.assert_array_equal(truth["hfunc1"][:, j].numpy(), cop.hfunc1(uv))
    np.testing.assert_array_equal(truth["hfunc2"][:, j].numpy(), cop.hfunc2(uv))
