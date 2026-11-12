"""The weather application's copula arms, their scores and their draws.

Every test reads a synthetic :class:`MarginalSet` whose margins are known
Gaussians and whose copula is a known Gaussian copula, so a score has a
closed form to compare with. The npcc arms run on the hermetic
``linear-gaussian`` backend of ``tests/conftest.py``, standing in for each
E1 estimator under the arm's backend name.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Protocol, cast

import numpy as np
import pytest
import pyvinecopulib as pv
import torch
from scipy import stats

from npcc.core import RosenblattVinecop
from npcc.experiments._study import derive_seed
from npcc.experiments.application import copulas
from npcc.experiments.application.copulas import (
  COPULA_ARMS,
  CopulaFailure,
  CopulaResult,
  run_copula_arm,
)
from npcc.experiments.application.marginal_set import MarginalSet
from npcc.experiments.config import EstimatorSpec
from npcc.experiments.vine_runner import select_structure

#: The application's quantile levels, the 199 midpoints.
LEVELS = (np.arange(199) + 0.5) / 199

FEATURES = (
  "coe:t2m_ensavg",
  "coe:wind_speed_ensavg",
  "time:sin_dayofyear",
  "time:cos_dayofyear",
)


def equicorrelation(d: int, rho: float) -> np.ndarray:
  return rho * np.ones((d, d)) + (1 - rho) * np.eye(d)


def gaussian_set(
  *,
  d: int = 3,
  n_train: int = 80,
  n_test: int = 6,
  correlation: np.ndarray | None = None,
  seed: int = 0,
  station: int = 0,
  model: str = "emos-seasonal",
) -> MarginalSet:
  """Gaussian margins ``N(mu_j, sigma_j)`` joined by a Gaussian copula.

  The transforms are the true CDFs at the observations, and the quantiles
  the true quantile functions at :data:`LEVELS`.
  """
  rng = np.random.default_rng(seed)
  corr = equicorrelation(d, 0.6) if correlation is None else correlation
  mu = rng.normal(scale=5.0, size=d)
  sigma = rng.uniform(0.5, 3.0, size=d)

  def split(n: int, start: int) -> tuple[np.ndarray, ...]:
    y = mu + sigma * rng.multivariate_normal(np.zeros(d), corr, size=n)
    days = np.arange(start, start + n)
    angle = 2 * np.pi * days / 365.25
    x = np.column_stack([rng.normal(size=(n, 2)), np.sin(angle), np.cos(angle)])
    time = np.datetime64("2017-01-01") + days.astype("timedelta64[D]")
    return x, y, time.astype("datetime64[ns]")

  x_train, y_train, time_train = split(n_train, 0)
  x_test, y_test, time_test = split(n_test, n_train)
  quantiles = mu + sigma * stats.norm.ppf(LEVELS)[:, None]
  return MarginalSet(
    station=station,
    model=model,
    variables=tuple(f"obs:v{j}" for j in range(d)),
    feature_names=FEATURES,
    x_train=x_train,
    x_test=x_test,
    y_train=y_train,
    y_test=y_test,
    time_train=time_train,
    time_test=time_test,
    pit_train=stats.norm.cdf(y_train, mu, sigma),
    pit_test=stats.norm.cdf(y_test, mu, sigma),
    logpdf_test=stats.norm.logpdf(y_test, mu, sigma),
    levels=LEVELS,
    quantiles_test=np.broadcast_to(quantiles, (n_test, *quantiles.shape)),
  )


@pytest.fixture
def estimators(
  register_linear_gaussian_backend: None,
) -> dict[str, EstimatorSpec]:
  """Every npcc backend name mapped to the hermetic linear-Gaussian one."""
  spec = EstimatorSpec(
    label="lin", backend="linear-gaussian", transform="probit"
  )
  return dict.fromkeys(copulas.BACKENDS, spec)


class Run(Protocol):
  def __call__(
    self,
    arm: str,
    ms: MarginalSet,
    *,
    estimators: Mapping[str, EstimatorSpec] | None = None,
    sample_batch_size: int = 20,
    mixture_k: int = 3,
    seed: int = 1,
  ) -> list[CopulaResult]: ...


@pytest.fixture
def run(estimators: dict[str, EstimatorSpec]) -> Run:
  """``run_copula_arm`` with small settings, some of them overridable."""
  default = estimators

  def call(
    arm: str,
    ms: MarginalSet,
    *,
    estimators: Mapping[str, EstimatorSpec] | None = None,
    sample_batch_size: int = 20,
    mixture_k: int = 3,
    seed: int = 1,
  ) -> list[CopulaResult]:
    return run_copula_arm(
      arm,
      ms,
      estimators=default if estimators is None else estimators,
      n_draws=8,
      sample_batch_size=sample_batch_size,
      mixture_k=mixture_k,
      seed=seed,
      device=None,
    )

  return call


def assert_same(a: CopulaResult, b: CopulaResult) -> None:
  for field in ("logcop_test", "joint_nll", "energy", "variogram"):
    np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


# ---------------------------------------------------------------------------
# Independence and the Gaussian copula
# ---------------------------------------------------------------------------


def test_independence_scores_the_margins_alone(run: Run) -> None:
  """Independence's copula log-density must be exactly 0 at every case.

  The joint log-likelihood is then the margins' sum, which is the reference
  every other arm's improvement is read against.
  """
  ms = gaussian_set()
  (result,) = run("independence", ms)
  np.testing.assert_array_equal(result.logcop_test, np.zeros(6))
  np.testing.assert_array_equal(result.joint_nll, -ms.logpdf_test.sum(axis=1))


def test_the_gaussian_arm_recovers_the_correlation(run: Run) -> None:
  """The Gaussian arm must fit the normal scores' correlation.

  The transforms here are a Gaussian copula's, so the fit must recover its
  correlation matrix; one fitted to the transforms themselves, Spearman's
  rather than the normal scores', would be biased toward 0.
  """
  truth = np.array([[1.0, 0.7, -0.3], [0.7, 1.0, 0.1], [-0.3, 0.1, 1.0]])
  ms = gaussian_set(n_train=4000, correlation=truth)
  (result,) = run("gaussian", ms)
  fitted = np.asarray(result.info["correlation"])
  np.testing.assert_allclose(fitted, truth, atol=0.04)


def test_the_gaussian_log_density_is_the_normal_ratio(run: Run) -> None:
  """The Gaussian copula density must be the joint normal over its margins.

  Checked against scipy's multivariate normal at the fitted correlation, so
  a sign or a factor of 1/2 in the quadratic form shows.
  """
  ms = gaussian_set(n_train=500)
  (result,) = run("gaussian", ms)
  correlation = np.asarray(result.info["correlation"])
  z = stats.norm.ppf(np.clip(ms.pit_test, 1e-6, 1 - 1e-6))
  want = stats.multivariate_normal(np.zeros(3), correlation).logpdf(z)
  want -= stats.norm.logpdf(z).sum(axis=1)
  np.testing.assert_allclose(result.logcop_test, want, rtol=1e-10)


def test_the_joint_nll_adds_the_copula_to_the_margins(run: Run) -> None:
  """The joint log-likelihood must be the copula's plus every margin's.

  Leaving a margin out, or the copula term, still gives finite numbers.
  """
  ms = gaussian_set()
  (result,) = run("gaussian", ms)
  want = -(result.logcop_test + ms.logpdf_test.sum(axis=1))
  np.testing.assert_allclose(result.joint_nll, want, rtol=1e-14)


def test_a_constant_transform_cannot_fit_a_correlation(run: Run) -> None:
  """A variable whose transforms never vary has no correlation to fit.

  ``corrcoef`` would return ``nan`` there, and the arm would be scored on it
  rather than recorded as having failed on that marginal set.
  """
  ms = gaussian_set()
  pit = ms.pit_train.copy()
  pit[:, 1] = 0.5
  with pytest.raises(CopulaFailure, match="varying"):
    run("gaussian", dataclasses.replace(ms, pit_train=pit))


# ---------------------------------------------------------------------------
# Failures and refusals
# ---------------------------------------------------------------------------


def test_a_non_finite_joint_nll_fails_with_its_cases(run: Run) -> None:
  """A non-finite joint log-likelihood must fail the arm, naming its cases.

  Averaged in, one infinite case would make the station's mean infinite;
  the run instead records the arm as failed and which case caused it.
  """
  ms = gaussian_set()
  logpdf = ms.logpdf_test.copy()
  logpdf[3, 1] = -np.inf
  with pytest.raises(CopulaFailure) as info:
    run("independence", dataclasses.replace(ms, logpdf_test=logpdf))
  assert info.value.indices == (3,)


@pytest.mark.parametrize("bad", [np.nan, 1.5, -0.1])
def test_invalid_draws_fail_the_arm_with_their_cases(
  run: Run, monkeypatch: pytest.MonkeyPatch, bad: float
) -> None:
  """A draw that is not finite or outside [0, 1] must fail the arm.

  ``np.interp`` would clamp a draw beyond the levels to an end quantile and
  return a finite score for it, so only the check catches it.
  """
  original = RosenblattVinecop.sample

  def corrupt(
    self: RosenblattVinecop,
    n: int,
    *,
    seeds: list[int] | None = None,
    x: torch.Tensor | None = None,
  ) -> torch.Tensor:
    draws = original(self, n, seeds=seeds, x=x).clone()
    # The third test case's eight draws, in the one chunk of the call.
    draws[2 * 8 : 3 * 8, 1] = bad
    return draws

  monkeypatch.setattr(RosenblattVinecop, "sample", corrupt)
  with pytest.raises(CopulaFailure, match="draws") as info:
    run("npcc-random-lightgbm-time", gaussian_set(), sample_batch_size=1000)
  assert info.value.indices == (2,)


def test_an_unknown_arm_or_backend_is_refused(run: Run) -> None:
  """An arm outside the list, or one whose backend has no estimator, stops.

  Neither is the arm's outcome on a station, so neither may be recorded as
  a failure and skipped.
  """
  ms = gaussian_set()
  with pytest.raises(ValueError, match="arm must be one of"):
    run("npcc-random-catboost-all", ms)
  with pytest.raises(ValueError, match="estimators must map"):
    run("npcc-random-tabicl-all", ms, estimators={})


def test_the_arm_list_holds_every_scope_and_backend() -> None:
  """The runner selects arms from this list; a missing one never runs."""
  assert len(COPULA_ARMS) == len(set(COPULA_ARMS)) == 16
  for kind in ("random", "mixture"):
    for backend in ("lightgbm", "tabpfn-quantiles", "tabicl"):
      for scope in ("all", "time"):
        assert f"npcc-{kind}-{backend}-{scope}" in COPULA_ARMS


# ---------------------------------------------------------------------------
# The npcc arms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  ("scope", "width"), [("all", len(FEATURES)), ("time", 2)]
)
def test_a_vine_conditions_on_its_scope_in_every_tree(
  run: Run, pair_fit_widths: list[int | None], scope: str, width: int
) -> None:
  """Every pair must see the scope's covariates, and its conditioning set.

  The time scope picks two columns by name; passing every column, or none,
  would fit another arm under its name. The context is non-simplified, so a
  pair in tree 2 also sees its one conditioning variable.
  """
  run(f"npcc-random-lightgbm-{scope}", gaussian_set())
  assert sorted(w or 0 for w in pair_fit_widths) == [width, width, width + 1]


def test_the_random_structure_is_the_stations_own(run: Run) -> None:
  """One random structure per station, shared by every marginal model.

  Drawn per marginal model or backend, a difference between two of them
  would mix the structure's luck into the comparison.
  """
  ms = gaussian_set(d=4, station=7)
  other = gaussian_set(d=4, station=7, model="lightgbm-all", seed=5)
  results = [
    run(arm, m)[0]
    for arm, m in (
      ("npcc-random-lightgbm-all", ms),
      ("npcc-random-tabicl-time", other),
    )
  ]
  seed = derive_seed(1, "random-structure", 7)
  want = pv.RVineStructure.sample(4, seeds=[seed])
  for result in results:
    assert result.info["structure_seed"] == seed
    assert result.info["structure"] == np.asarray(want.matrix).tolist()


def test_the_mixture_returns_itself_and_each_member(run: Run) -> None:
  """The mixture must come with every member, the first on the selected one.

  The paper reads the mixture against its members, and member 1 is the
  vine on the structure Dissmann's algorithm selects from the training
  transforms, which is that comparison's reference.
  """
  ms = gaussian_set(d=5, n_train=150)
  arm = "npcc-mixture-tabpfn-quantiles-all"
  results = run(arm, ms, mixture_k=4)
  assert [r.arm for r in results] == [
    arm,
    *(f"{arm}-m{k}" for k in range(1, 5)),
  ]
  for r in results:
    for scores in (r.logcop_test, r.joint_nll, r.energy, r.variogram):
      assert scores.shape == (6,)
      assert np.isfinite(scores).all()
  selected = select_structure(
    torch.from_numpy(np.clip(ms.pit_train, 1e-6, 1 - 1e-6))
  )
  members = cast("list[dict[str, object]]", results[0].info["members"])
  assert members[0]["structure"] == np.asarray(selected.matrix).tolist()
  assert members[0]["structure_seed"] is None
  assert [m["structure_seed"] for m in members[1:]] == [
    derive_seed(1, "mixture-structure", 0, k) for k in range(2, 5)
  ]
  distinct = results[0].info["n_distinct_structures"]
  assert isinstance(distinct, int)
  assert 1 <= distinct <= 4
  for stage in ("structure_selection", "fit"):
    assert results[0].timings[stage] == pytest.approx(
      sum(r.timings[stage] for r in results[1:])
    )


def test_the_mixture_density_is_at_least_its_members_mean_log(run: Run) -> None:
  """Jensen's inequality must hold at every test case.

  The log of the members' mean density is at least the mean of their logs,
  so a mixture log-density below that anywhere is combined wrongly, such as
  without the ``- log k``.
  """
  results = run("npcc-mixture-lightgbm-all", gaussian_set(d=5, n_train=150))
  member_logs = np.stack([r.logcop_test for r in results[1:]])
  assert (results[0].logcop_test >= member_logs.mean(axis=0) - 1e-12).all()


def test_a_mixture_of_one_is_its_member(run: Run) -> None:
  """With one member the mixture must be that member, bit-identical.

  Its density, its draws and so every score; a mixture that drew its member
  choice from the member's generator, or combined one log-density with any
  rounding, would differ.
  """
  results = run("npcc-mixture-lightgbm-time", gaussian_set(d=4), mixture_k=1)
  assert len(results) == 2
  assert_same(results[0], results[1])


def test_member_draws_are_reproducible_from_the_seed(
  run: Run, register_hermetic_backends: None
) -> None:
  """The same seed must give the same draws, and another seed others.

  Scores that change between two runs of one configuration could not be
  resumed or compared across marginal models. The closed-form
  ``normal-quantile`` backend fits nothing, so the draws depend on the seeds
  alone: the linear-Gaussian one's least squares differ between two calls in
  the last bits on some CPUs.
  """
  spec = EstimatorSpec(label="z", backend="normal-quantile", transform="probit")
  ms = gaussian_set(d=4)
  arm = "npcc-mixture-lightgbm-all"
  first, again = (run(arm, ms, estimators={"lightgbm": spec}) for _ in "ab")
  for a, b in zip(first, again, strict=True):
    assert_same(a, b)
  other = run(arm, ms, estimators={"lightgbm": spec}, seed=2)
  for a, b in zip(first, other, strict=True):
    assert not np.array_equal(a.energy, b.energy)


def test_the_member_choice_is_uniform() -> None:
  """Each draw must come from a member picked uniformly, slot by slot.

  Member ``k``'s draws all hold ``k`` here, so each slot shows which member
  it was taken from; a biased choice, or one shared by a case's draws, shows
  in the counts.
  """
  k, n, m = 4, 50, 200
  members = np.broadcast_to(
    np.arange(k, dtype=np.float64)[:, None, None, None], (k, n, m, 2)
  )
  picked = copulas.mixture_draws(members, np.random.default_rng(3))
  assert picked.shape == (n, m, 2)
  np.testing.assert_array_equal(picked[..., 0], picked[..., 1])
  counts = np.bincount(picked[..., 0].astype(int).ravel(), minlength=k)
  assert stats.chisquare(counts).pvalue > 1e-3
  per_case = np.array([len(np.unique(row)) for row in picked[..., 0]])
  assert (per_case > 1).all()


def test_a_member_draw_is_taken_at_its_own_slot() -> None:
  """The mixture's draw ``s`` of case ``i`` must be a member's at ``(i, s)``.

  Taken from another slot, it would pair a case with another case's draw.
  """
  rng = np.random.default_rng(4)
  members = rng.random((3, 5, 7, 2))
  picked = copulas.mixture_draws(members, np.random.default_rng(5))
  choice = np.random.default_rng(5).integers(3, size=(5, 7))
  for i in range(5):
    for s in range(7):
      np.testing.assert_array_equal(picked[i, s], members[choice[i, s], i, s])
