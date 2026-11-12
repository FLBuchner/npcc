"""Vine runner: arms, context wiring, scoring and the resume guard (hermetic)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pyvinecopulib as pv
import torch
from pyvinecopulib.core import NonSimplifiedContext, SimplifiedContext

from npcc.core.vinecop import RosenblattVinecop
from npcc.experiments import scenarios, vine_runner, vine_scenarios
from npcc.experiments.config import EstimatorSpec, RunConfig
from npcc.experiments.vine_config import VineCell, VineGridConfig


def _grid(*, arms: list[str] | None = None, eval_m: int = 20) -> VineGridConfig:
  return VineGridConfig(
    dims=[3],
    n=[40],
    n_rep=1,
    estimators=[
      EstimatorSpec(label="uni", backend="normal-quantile", transform="logit")
    ],
    arms=["oracle", "random", "tll"] if arms is None else arms,
    eval_x_n=2,
    eval_m=eval_m,
  )


def test_score_is_zero_for_the_truth() -> None:
  """A perfect estimate must score zero, or every metric is biased."""
  p = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
  s = vine_runner.score(p, p)
  assert s["KL"] == pytest.approx(0.0)
  assert s["L1"] == pytest.approx(0.0)
  assert s["ISE"] == pytest.approx(0.0)


def test_random_structure_is_independent_of_the_truth() -> None:
  """A random arm reusing the truth's seed would draw the oracle's structure."""
  matrices = {
    str(vine_runner.random_structure_for(317, 5, rep).matrix.tolist())
    == str(vine_runner.truth_for(317, 5, rep).structure.matrix.tolist())
    for rep in range(5)
  }
  assert matrices == {False}


def test_one_cell_runs_every_arm(register_hermetic_backends: None) -> None:
  """Every configured arm must yield finite scores at every evaluation x."""
  grid = _grid()
  metrics, runtime = vine_runner.summarize_one_cell(
    grid.cells()[0], grid, base_seed=317, device="cpu"
  )
  arms = {row["arm"] for row in metrics}
  assert arms == {"oracle", "random", "tll"}
  assert len(metrics) == 3 * grid.eval_x_n
  assert len(runtime) == 3
  for row in metrics:
    for name in vine_runner.METRIC_NAMES:
      assert math.isfinite(row[name]), name
    assert math.isfinite(row["loglik_true"])


def test_arms_use_their_contexts(
  register_hermetic_backends: None, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The oracle is simplified; the random and selected arms condition on u_D.

  Swapping them silently would hand the oracle extra information and strip
  an arm on a possibly wrong structure of the conditioning it needs.
  """
  seen: list[type] = []
  original = RosenblattVinecop.__init__

  def spy(
    self: RosenblattVinecop,
    *args: Any,
    **kwargs: Any,
  ) -> None:
    seen.append(type(kwargs["context"]))
    original(self, *args, **kwargs)

  monkeypatch.setattr(RosenblattVinecop, "__init__", spy)
  grid = _grid(arms=["oracle", "random", "selected"])
  vine_runner.summarize_one_cell(
    grid.cells()[0], grid, base_seed=317, device="cpu"
  )
  assert seen == [SimplifiedContext, NonSimplifiedContext, NonSimplifiedContext]


def _fit(
  arm: str,
  structure: pv.RVineStructure | None,
  y: torch.Tensor,
  x: torch.Tensor,
) -> tuple[Any, RosenblattVinecop, float]:
  """Fit one NPCC arm with the hermetic estimator, on the CPU."""
  return vine_runner._fit_npcc(
    arm,
    _est(),
    structure,
    y,
    x,
    device="cpu",
    projection_grid_size=30,
    batch_size=None,
  )


def _est() -> EstimatorSpec:
  """The hermetic estimator, built once its backend is registered."""
  return EstimatorSpec(
    label="uni", backend="normal-quantile", transform="logit"
  )


def test_the_selected_arm_fits_as_the_random_arm_does(
  register_hermetic_backends: None, monkeypatch: pytest.MonkeyPatch
) -> None:
  """On one structure, the selected arm's fit must be the random arm's.

  The arms are compared paired, so a two-step fit that differed from
  ``RosenblattVinedist.fit`` in anything but the structure -- the margins'
  controls, the data's placement, the copula's context -- would show up in
  selected - random as if the structure caused it.
  """
  truth = vine_runner.truth_for(317, 3, 0)
  y, _, x = vine_scenarios.sample(truth, 60, 11)
  structure = vine_runner.random_structure_for(317, 3, 0)
  monkeypatch.setattr(vine_runner, "select_structure", lambda u: structure)
  torch.manual_seed(5)
  given, _, none = _fit("random", structure, y, x)
  torch.manual_seed(5)
  chosen, vine, seconds = _fit("selected", None, y, x)
  assert math.isnan(none)
  assert seconds >= 0.0
  np.testing.assert_array_equal(
    np.asarray(vine.structure.matrix), np.asarray(structure.matrix)
  )
  y_eval, _, x_eval = vine_scenarios.sample(truth, 20, 12)
  torch.testing.assert_close(
    chosen.pdf(y_eval, x=x_eval), given.pdf(y_eval, x=x_eval), rtol=0, atol=0
  )


def test_the_selection_sees_the_margins_transforms(
  register_hermetic_backends: None, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The structure is selected on the scale the pair copulas see, not on y.

  Ranks of ``y`` carry the dependence that the margins' shift in ``x``
  induces, which is not the copula's, so a selection on them would choose
  partly on the margins.
  """
  seen: list[torch.Tensor] = []
  original = vine_runner.select_structure

  def spy(u: torch.Tensor) -> pv.RVineStructure:
    seen.append(u.clone())
    return original(u)

  monkeypatch.setattr(vine_runner, "select_structure", spy)
  truth = vine_runner.truth_for(317, 3, 0)
  y, _, x = vine_scenarios.sample(truth, 60, 11)
  dist, vine, _ = _fit("selected", None, y, x)
  [u] = seen
  torch.testing.assert_close(u, dist.marginal_cdf(y, x=x), rtol=0, atol=0)
  np.testing.assert_array_equal(
    np.asarray(vine.structure.matrix), np.asarray(original(u).matrix)
  )


def test_the_selection_follows_the_data() -> None:
  """A selection that ignored its data would hand every cell one structure.

  Data from a D-vine on the path 1 - 3 - 2, not the natural order, with a
  strong Gaussian on both of its first-tree edges: the selection must find
  that tree.
  """
  structure = pv.RVineStructure.from_order([1, 3, 2])
  strong = pv.Bicop(
    family=pv.BicopFamily.gaussian, parameters=np.array([[0.8]])
  )
  truth = pv.Vinecop.from_structure(
    structure=structure, pair_copulas=[[strong, strong], [pv.Bicop()]]
  )
  u = torch.as_tensor(truth.sample(500, seeds=[1]))
  chosen = vine_runner.select_structure(u)
  pairs = {frozenset((a, b)) for a, b, _ in chosen.get_trees()[0]}
  assert pairs == {frozenset((1, 3)), frozenset((3, 2))}


def test_each_npcc_arm_records_its_structure(
  register_hermetic_backends: None,
) -> None:
  """Whether an arm's structure is the truth is read from its runtime row.

  The analysis compares each recorded matrix with the truth's, so a row that
  held another arm's structure, or none, would misreport the recovery; and
  only the selected arm spends time selecting.
  """
  grid = _grid(arms=["oracle", "random", "selected"])
  _, runtime = vine_runner.summarize_one_cell(
    grid.cells()[0], grid, base_seed=317, device="cpu"
  )
  rows = {row["arm"]: row for row in runtime}
  truth = vine_runner.truth_for(317, 3, 0).structure
  drawn = vine_runner.random_structure_for(317, 3, 0)
  assert json.loads(str(rows["oracle"]["structure"])) == (
    np.asarray(truth.matrix).tolist()
  )
  assert json.loads(str(rows["random"]["structure"])) == (
    np.asarray(drawn.matrix).tolist()
  )
  selected = np.asarray(json.loads(str(rows["selected"]["structure"])))
  assert (
    pv.RVineStructure.from_matrix(
      np.asfortranarray(selected.astype(np.uint64))
    ).dim
    == 3
  )
  assert math.isnan(float(rows["oracle"]["selection_time"]))
  assert math.isnan(float(rows["random"]["selection_time"]))
  assert float(rows["selected"]["selection_time"]) >= 0.0


def test_resume_refuses_a_changed_grid(
  register_hermetic_backends: None, tmp_path: Path
) -> None:
  """Resuming a different grid would silently mix two studies' cells."""
  run = RunConfig(out=tmp_path, device="cpu", fmt="csv")
  vine_runner.run_vine_study(_grid(arms=["tll"]), run)
  with pytest.raises(ValueError, match="resume"):
    vine_runner.run_vine_study(_grid(arms=["tll"], eval_m=30), run, resume=True)


def test_resume_reads_back_done_cells(
  register_hermetic_backends: None, tmp_path: Path
) -> None:
  """A resumed run must return the checkpointed rows, not an empty table."""
  run = RunConfig(out=tmp_path, device="cpu", fmt="csv")
  first, _, _ = vine_runner.run_vine_study(_grid(arms=["tll"]), run)
  again, _, _ = vine_runner.run_vine_study(
    _grid(arms=["tll"]), run, resume=True
  )
  assert len(again) == len(first)
  assert again["KL"].tolist() == pytest.approx(first["KL"].tolist())


def test_tll_baseline_keeps_the_oracle_structure(
  register_hermetic_backends: None, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A baseline that re-selects its structure would confound the x effect.

  ``kde-tll`` is there to price ignoring the covariate, so it must share the
  oracle's structure rather than pick its own from the data.
  """
  fitted: list[pv.Vinedist] = []
  original = pv.Vinedist.from_data

  def spy(*args: Any, **kwargs: Any) -> pv.Vinedist:
    model = original(*args, **kwargs)
    fitted.append(model)
    return model

  monkeypatch.setattr(pv.Vinedist, "from_data", spy)
  grid = _grid(arms=["tll"])
  vine_runner.summarize_one_cell(
    grid.cells()[0], grid, base_seed=317, device="cpu"
  )
  truth = vine_runner.truth_for(317, 3, 0)
  assert len(fitted) == 1
  copula = fitted[0].vinecop
  assert isinstance(copula, pv.Vinecop)
  np.testing.assert_array_equal(
    np.asarray(copula.matrix), np.asarray(truth.structure.matrix)
  )


def test_the_signature_fingerprints_the_regimes(
  register_hermetic_backends: None, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Redefining a regime changes every truth, so it must refuse a resume.

  Moving the tau bounds is not the only way to change the truth: a regime
  can be redefined inside unchanged bounds, as the sine regime once was.
  """
  run = RunConfig(out=Path("unused"), device="cpu")
  grid = _grid(arms=["tll"])
  before = vine_runner.grid_signature(grid, run)
  spec = scenarios.TAU_SCENARIOS["sin"]
  monkeypatch.setitem(
    scenarios.TAU_SCENARIOS,
    "sin",
    scenarios.ScenarioSpec("sin", True, lambda x: torch.full_like(x, 0.3)),
  )
  assert spec is not scenarios.TAU_SCENARIOS["sin"]
  assert vine_runner.grid_signature(grid, run) != before


def test_more_repetitions_resume_on_the_finished_cells(
  register_hermetic_backends: None,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Growing a grid must not refit, or throw away, the cells it already has.

  A cell's output depends on its own (d, n, rep) only, so a signature that
  hashed ``n_rep`` would refuse the resume and force a full rerun.
  """
  run = RunConfig(out=tmp_path, device="cpu", fmt="csv")
  small = _grid(arms=["tll"])
  first, _, _ = vine_runner.run_vine_study(small, run)

  fitted: list[VineCell] = []
  original = vine_runner.summarize_one_cell

  def spy(cell: VineCell, *args: Any, **kwargs: Any) -> Any:
    fitted.append(cell)
    return original(cell, *args, **kwargs)

  monkeypatch.setattr(vine_runner, "summarize_one_cell", spy)
  grown = VineGridConfig(
    dims=small.dims,
    n=small.n,
    n_rep=2,
    estimators=small.estimators,
    arms=small.arms,
    eval_x_n=small.eval_x_n,
    eval_m=small.eval_m,
  )
  again, _, _ = vine_runner.run_vine_study(grown, run, resume=True)
  assert [c.rep for c in fitted] == [1]
  kept = again[again["rep"] == 0]
  assert kept["KL"].tolist() == pytest.approx(first["KL"].tolist())


def _mixture_grid(
  *, arms: list[str], mixture_k: int, backend: str = "linear-gaussian"
) -> VineGridConfig:
  return VineGridConfig(
    dims=[4],
    n=[60],
    n_rep=1,
    estimators=[
      EstimatorSpec(label="lin", backend=backend, transform="probit")
    ],
    arms=arms,
    eval_x_n=2,
    eval_m=30,
    mixture_k=mixture_k,
  )


def _by_arm(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
  out: dict[str, list[dict[str, Any]]] = {}
  for row in rows:
    out.setdefault(str(row["arm"]), []).append(row)
  return out


def test_weighted_random_structures_follow_their_seed() -> None:
  """A member's structure must be a function of the data and its seed alone.

  Otherwise a resumed or rerun cell would mix other members than it scored,
  and members sharing a seed would all be one structure.
  """
  gen = torch.Generator().manual_seed(1)
  z = torch.randn(300, 5, generator=gen, dtype=torch.float64)
  z = z + 0.8 * z[:, [0]]
  u = torch.special.ndtr(z)
  draws = [vine_runner.weighted_random_structure(u, s) for s in (11, 12, 11)]
  matrices = [np.asarray(s.matrix).tolist() for s in draws]
  assert matrices[0] == matrices[2]
  assert matrices[0] != matrices[1]


def test_identical_members_reproduce_the_member() -> None:
  """A mixture of k copies of one density must be that density.

  Dropping the ``- log k`` would shift every mixture log-density by it.
  """
  member = torch.tensor([-3.0, -0.5, 1.25, -800.0], dtype=torch.float64)
  margins = [torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float64)]
  mixture, members = vine_runner.joint_log_densities(
    member.expand(3, -1), margins
  )
  torch.testing.assert_close(mixture, member + margins[0])
  torch.testing.assert_close(members, (member + margins[0]).expand(3, -1))


def test_the_mixtures_kl_is_at_most_its_members_mean() -> None:
  """Jensen's inequality must hold for the Monte-Carlo KL, point by point.

  The log of a mean density is at least the mean of the log-densities at
  every point, so a mixture scored above its members' mean is wired wrong.
  """
  gen = torch.Generator().manual_seed(0)
  z = torch.randn(2000, generator=gen, dtype=torch.float64)
  normal = torch.distributions.Normal(0.0, 1.0)
  log_true = normal.log_prob(z)
  logs = torch.stack(
    [torch.distributions.Normal(mu, 1.0).log_prob(z) for mu in (-0.6, 0.2, 1.0)]
  )
  mixture = vine_runner.mixture_log_density(logs)
  kl_mixture = vine_runner.score(log_true.exp(), mixture.exp())["KL"]
  kl_members = [
    vine_runner.score(log_true.exp(), member.exp())["KL"] for member in logs
  ]
  assert kl_mixture <= float(np.mean(kl_members))
  assert kl_mixture < float(np.mean(kl_members)) - 1e-3


def test_a_mixture_of_one_reproduces_the_selected_arm(
  register_linear_gaussian_backend: None,
) -> None:
  """With k = 1 the mixture is the selected arm, fitted and scored the same.

  Its joint density is assembled from parts rather than through
  ``RosenblattVinedist.pdf``, so any drift there would show here as a
  mixture that is not its own single member.
  """
  grid = _mixture_grid(arms=["selected", "mixture"], mixture_k=1)
  metrics, runtime = vine_runner.summarize_one_cell(
    grid.cells()[0], grid, base_seed=317, device="cpu"
  )
  rows = _by_arm(metrics)
  assert set(rows) == {"selected", "mixture", "mixture-member-1"}
  for name in (*vine_runner.METRIC_NAMES, "loglik_hat"):
    for arm in ("mixture", "mixture-member-1"):
      assert [r[name] for r in rows[arm]] == pytest.approx(
        [r[name] for r in rows["selected"]], rel=1e-12, abs=1e-12
      ), (arm, name)
  structures = {str(r["arm"]): r["structure"] for r in runtime}
  assert structures["mixture"] == structures["selected"]


def test_the_mixture_scores_itself_and_each_member(
  register_linear_gaussian_backend: None,
) -> None:
  """Every member must be scored, and the mixture by at most their mean KL.

  The paper reads the mixture against its average member; without the
  members' own rows, or with a mean over some other set, it could not.
  """
  k = 3
  grid = _mixture_grid(arms=["selected", "mixture"], mixture_k=k)
  metrics, runtime = vine_runner.summarize_one_cell(
    grid.cells()[0], grid, base_seed=317, device="cpu"
  )
  rows = _by_arm(metrics)
  members = [f"mixture-member-{j}" for j in range(1, k + 1)]
  assert set(rows) == {"selected", "mixture", *members}
  for i, row in enumerate(rows["mixture"]):
    for name in ("KL", "KL_copula"):
      mean = float(np.mean([rows[m][i][name] for m in members]))
      assert row[f"{name}_members_mean"] == pytest.approx(mean)
      assert math.isfinite(row[name])
    assert row["KL"] <= row["KL_members_mean"] + 1e-12
  kls = {m: [r["KL"] for r in rows[m]] for m in members}
  assert len({tuple(v) for v in kls.values()}) > 1
  assert rows["mixture-member-1"][0]["KL"] == pytest.approx(
    rows["selected"][0]["KL"], rel=1e-12
  )
  mixture = next(r for r in runtime if r["arm"] == "mixture")
  matrices = json.loads(str(mixture["mixture_structures"]))
  assert len(matrices) == k
  assert matrices[0] == json.loads(str(mixture["structure"]))
  assert len({str(m) for m in matrices}) > 1
  assert float(mixture["selection_time"]) >= 0.0


def test_the_signature_reads_k_only_for_a_mixture(
  register_linear_gaussian_backend: None,
) -> None:
  """k changes a mixture's cells, and nothing else's.

  Hashing it for every grid would refuse to resume a run made before the
  arm existed; not hashing it would resume a mixture of 5 as one of 3.
  """
  run = RunConfig(out=Path("unused"), device="cpu")
  plain = [_mixture_grid(arms=["random"], mixture_k=k) for k in (3, 5)]
  assert len({vine_runner.grid_signature(g, run) for g in plain}) == 1
  mixed = [_mixture_grid(arms=["mixture"], mixture_k=k) for k in (3, 5)]
  assert len({vine_runner.grid_signature(g, run) for g in mixed}) == 2
