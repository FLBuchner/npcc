"""The copula arms, each fitted to one station's margins and scored jointly.

Every arm reads one :class:`~npcc.experiments.application.marginal_set.
MarginalSet`, one station's margins from one marginal model, and nothing
else. It is fitted to the training probability integral transforms, in
sample or out of fold as the set carries them, and evaluated at the test
ones, each clipped into ``[PIT_CLIP, 1 - PIT_CLIP]``:
a saved CDF can reach 0 or 1 exactly, which no copula argument may be. The
covariates are passed as they are.

**Arms.** :data:`COPULA_ARMS` lists every name :func:`run_copula_arm` takes:

- ``independence``, whose log-density is 0;
- ``gaussian``, the Gaussian copula with the normal scores' correlation,
  which ignores the covariates;
- ``boostcopula-<scope>``, boostCopula's boosted conditional R-vine, its
  structure, families and calibration selected and fitted in R by
  :data:`BOOSTCOPULA_SCRIPT` at the package's defaults, for d >= 2 variables;
- ``npcc-random-<backend>-<scope>``, a
  :class:`~npcc.core.vinecop.RosenblattVinecop` in the non-simplified context
  on one uniformly random R-vine structure per station, the same for every
  marginal model and backend;
- ``npcc-mixture-<backend>-<scope>``, E10's mixture
  (:mod:`~npcc.experiments.vine_runner`): ``mixture_k`` non-simplified vines
  on the same margins, the first on the structure
  :func:`~npcc.experiments.vine_runner.select_structure` selects from the
  training transforms, each other on weighted random spanning trees
  (:func:`~npcc.experiments.vine_runner.weighted_random_structure`). Its
  copula density is the members' mean. Each member is also returned and
  scored as ``<arm>-m<k>``, so the selected structure's vine is ``-m1``;
- ``npcc-chain-<backend>-<scope>``, E11's chain-rule mixture
  (:class:`~npcc.experiments.chain.ChainCopula`): ``min(mixture_k, d!)``
  copula densities written down by the chain rule along distinct variable
  orders drawn uniformly per station, as the vine study's ``chain`` arm draws
  them per repetition, by the same functions, on the same margins. Its copula density
  is the orders' mean, and its draws are exact for it: each order draws its
  share of every case's ``n_draws``. Each order is also returned as
  ``<arm>-m<k>``, scored by its log score alone (its energy, variogram and
  sample CRPS are ``nan``), since it draws only its share.

An npcc arm is fitted with its backend's E1 estimator whole -- backend,
transform and backend keyword arguments -- taken from the ``estimators``
mapping by the backend's name in the arm.

**Scopes.** ``all`` conditions on every covariate of the marginal set,
``time`` on :data:`TIME_FEATURES` alone.

**Scores.** Per test case (:mod:`~npcc.experiments.application.
joint_scores`): the joint negative log-likelihood, the copula's
log-density plus the margins'; and the energy score and the variogram score
of order 1/2 of ``n_draws`` draws from the joint forecast, each variable
divided by its training standard deviation. A draw is the copula's,
mapped to the response scale through the case's quantile table.

**Mixture draws.** Every member draws ``n_draws`` per case, which its own
scores need. The mixture's draw ``s`` of case ``i`` is draw ``s`` of case
``i`` of a member picked uniformly at random, independently for every
``(i, s)`` (:func:`mixture_draws`). A chain mixture's draws are split
instead (:func:`split_draws`): each of its ``k`` orders draws
``n_draws // k`` per case, the first ``n_draws % k`` orders one more, and
the mixture's draws of a case are theirs side by side, so every case has
each order's share exactly.

**Seeds.** Every seed derives from the run's ``seed`` with
:func:`~npcc.experiments._study.derive_seed`: the random structure from
``(seed, "random-structure", station)``, mixture member ``k``'s structure
from ``(seed, "mixture-structure", station, k)``, and the rest from
``(seed, station, model, result)``, with ``result`` the arm or member
name: the torch seed set before a vine is fitted (salted ``"fit"``),
each sampling chunk's (``"sample"`` and the chunk's index), the NumPy
generator of independence's and the Gaussian's draws (``"draws"``), the
mixture's member choice (``"choice"``), and R's seed (unsalted). A chain
mixture's orders are the vine study's
(:func:`~npcc.experiments.vine_runner.chain_orders`) with the station in
the repetition's place, ``derive_seed(seed, "chain", d, station, j)`` for
order ``j``, so they are shared across marginal models and backends; its
factors' torch seeds are the vine study's too
(:func:`~npcc.experiments.vine_runner.chain_factor_seeds`), from the cell
seed ``derive_seed(seed, station, model, arm)``. An npcc
vine or chain order samples its rows per case in chunks of at most
``sample_batch_size`` rows over the flattened cases and draws, so its
draws are reproducible for a fixed chunk size and case order.

**Timings.** Seconds per stage, each separate from the others, so a
result's total is their sum: ``fit``; ``structure_selection`` for a mixture
and its members; ``copula_prediction``, the copula's log-density and
draws; ``inverse_margins``; ``scoring``; and for boostCopula
``bridge_overhead``, the R process's start-up and the file exchange, while
its ``fit`` and ``copula_prediction`` are the seconds R reports for its own
steps. A mixture's ``fit``, ``structure_selection`` and
``copula_prediction`` sum its members', the last plus the mixing. Work on a
GPU is synchronized at every stage boundary.

**Failures.** An arm that cannot be scored on every test case raises
:class:`CopulaFailure`: invalid draws, a non-finite or nonpositive
boostCopula density, a non-finite joint log-likelihood or sample score, a
Gaussian copula without a correlation to fit, or boostCopula's script
exiting with an error. Anything else, a malformed input, a missing
interpreter or package, or output that breaks the R exchange, is a defect
and raises something else, which must stop the run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import pandas as pd
import pyvinecopulib as pv
import torch
from pyvinecopulib.core.extend import to_numpy

from npcc.core.controls import FitControlsRosenblattVinecop
from npcc.core.vinecop import RosenblattVinecop
from npcc.experiments import _external, _external_vine
from npcc.experiments._study import derive_seed, release_gpu
from npcc.experiments.application.derived import derive, physical_violations
from npcc.experiments.application.joint_scores import (
  energy_score,
  inverse_margins,
  joint_nll,
  variogram_score,
)
from npcc.experiments.application.marginal_set import MarginalSet
from npcc.experiments.chain import ChainCopula
from npcc.experiments.config import EstimatorSpec
from npcc.experiments.vine_runner import (
  chain_factor_seeds,
  chain_orders,
  mixture_log_density,
  select_structure,
  weighted_random_structure,
)

logger = logging.getLogger(__name__)

#: Where every probability integral transform is clipped before a copula
#: sees it, the npcc controls' ``eps``.
PIT_CLIP: float = 1e-6

#: The npcc arms' backends, as the arms name them.
BACKENDS: tuple[str, ...] = ("lightgbm", "tabpfn-quantiles", "tabicl")

#: The covariate scopes of a conditional arm.
SCOPES: tuple[str, ...] = ("all", "time")

#: The ``time`` scope's covariates, by name.
TIME_FEATURES: tuple[str, ...] = ("time:sin_dayofyear", "time:cos_dayofyear")

#: Every arm :func:`run_copula_arm` takes.
COPULA_ARMS: tuple[str, ...] = (
  "independence",
  "gaussian",
  *(f"boostcopula-{scope}" for scope in SCOPES),
  *(
    f"npcc-{kind}-{backend}-{scope}"
    for kind in ("random", "mixture", "chain")
    for backend in BACKENDS
    for scope in SCOPES
  ),
)

#: The boostCopula arms' script, relative to the working directory unless
#: absolute.
BOOSTCOPULA_SCRIPT: str = "r_methods/fit_boostcopula_vine.R"

#: The fewest variables the boostCopula script fits.
BOOSTCOPULA_MIN_DIMENSION: int = 2

#: E1's grid setting, which sizes the Sinkhorn projection that E1's controls
#: leave off; passed only because building the controls needs it.
PROJECTION_GRID_SIZE: int = 30

#: Below this standard deviation, a normal score holds no correlation.
_CONSTANT_SD: float = 1e-12

#: The floor on the normal scores' correlation's eigenvalues.
_EIGEN_FLOOR: float = 1e-8


@dataclass(frozen=True)
class CopulaResult:
  """One copula's scores on one marginal set's test cases.

  Attributes
  ----------
  arm : str
      The arm, or for a mixture member ``<arm>-m<k>``.
  logcop_test : numpy.ndarray, shape (n_test,)
      The copula log-density at each test case's clipped transforms.
  joint_nll : numpy.ndarray, shape (n_test,)
      The joint negative log-likelihood of each test case.
  energy : numpy.ndarray, shape (n_test,)
      The energy score of each test case.
  variogram : numpy.ndarray, shape (n_test,)
      The variogram score of order 1/2 of each test case.
  timings : dict of str to float
      Seconds per stage; the stages are listed in this module's docstring.
  info : dict of str to object
      What was fitted, as JSON-ready values: structures, seeds, settings and
      versions.
  sample_crps : dict of str to numpy.ndarray
      With derived responses, or where :func:`run_copula_arm` is asked for
      it: every response's CRPS of each test case, modeled and derived,
      estimated from the draws in its own units.
  sampled : bool
      Whether the sample scores come from draws of this copula; ``False``
      for a chain mixture's order, scored by its log score alone, whose
      ``energy`` and ``variogram`` are ``nan``.
  draws : numpy.ndarray or None, shape (n_test, n_draws, d + k)
      The draws the sample scores were computed from, on each response's
      own scale: the ``d`` modeled responses, then the ``k`` derived ones;
      ``None`` for a result that draws nothing of its own.
  physical : dict of str to numpy.ndarray
      Per physical check
      (:func:`~npcc.experiments.application.derived.physical_violations`),
      how many of each test case's draws break it; empty for a result that
      draws nothing of its own.
  extra : dict of str to numpy.ndarray
      Further per-case scores, by their column: each response subset's
      ``energy_<name>`` and ``variogram_<name>`` for a result with draws of
      its own, and an independence or Gaussian arm's projected log score
      where :func:`run_copula_arm` is asked for it.
  """

  arm: str
  logcop_test: np.ndarray
  joint_nll: np.ndarray
  energy: np.ndarray
  variogram: np.ndarray
  timings: dict[str, float]
  info: dict[str, object]
  sample_crps: dict[str, np.ndarray] = field(default_factory=dict)
  sampled: bool = True
  draws: np.ndarray | None = None
  physical: dict[str, np.ndarray] = field(default_factory=dict)
  extra: dict[str, np.ndarray] = field(default_factory=dict)


class CopulaFailure(ValueError):
  """An arm that cannot be scored on every test case of its marginal set.

  The arm's outcome on that marginal set, which a run records and continues
  past, rather than a defect of the run.

  Parameters
  ----------
  message : str
      What failed.
  indices : iterable of int, default ()
      The test cases concerned, as positions in the marginal set's test
      arrays, when they are known.

  Attributes
  ----------
  indices : tuple of int
      The test cases concerned, sorted, each once; empty when unknown.
  """

  def __init__(self, message: str, indices: Iterable[int] = ()) -> None:
    super().__init__(message)
    self.indices: tuple[int, ...] = tuple(sorted({int(i) for i in indices}))


@dataclass(frozen=True)
class _Arm:
  name: str
  kind: str
  backend: str | None
  scope: str | None


@dataclass(frozen=True)
class _Inputs:
  """What every arm reads, checked and clipped once."""

  ms: MarginalSet
  u_train: np.ndarray
  u_test: np.ndarray
  scale: np.ndarray
  derived_scale: np.ndarray
  n_draws: int
  seed: int
  sample_crps: bool = False
  subsets: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
  project_onto: tuple[str, ...] | None = None
  projected_column: str = "joint_nll_modeled"

  @property
  def draws_shape(self) -> tuple[int, int, int]:
    """``(n_test, n_draws, d)``, the shape of a copula's draws."""
    return (len(self.ms.y_test), self.n_draws, len(self.ms.variables))

  def salt(self, result: str) -> tuple[object, ...]:
    return (self.seed, self.ms.station, self.ms.model, result)


def _parse(arm: str) -> _Arm:
  if arm not in COPULA_ARMS:
    raise ValueError(f"arm must be one of {list(COPULA_ARMS)}; got {arm!r}")
  if arm in ("independence", "gaussian"):
    return _Arm(arm, arm, None, None)
  head, scope = arm.rsplit("-", 1)
  if head == "boostcopula":
    return _Arm(arm, head, None, scope)
  kind, backend = head.split("-", 2)[1:]
  return _Arm(arm, f"npcc-{kind}", backend, scope)


def scope_columns(feature_names: tuple[str, ...], scope: str) -> list[int]:
  """The columns of a marginal set's covariates that ``scope`` conditions on.

  Parameters
  ----------
  feature_names : tuple of str
      The marginal set's covariate names.
  scope : str
      One of :data:`SCOPES`.

  Returns
  -------
  list of int
      Every column for ``all``; those of :data:`TIME_FEATURES`, in that
      order, for ``time``.

  Raises
  ------
  ValueError
      If ``scope`` is unknown, or a covariate it names is missing.
  """
  if scope == "all":
    return list(range(len(feature_names)))
  if scope != "time":
    raise ValueError(f"scope must be one of {list(SCOPES)}; got {scope!r}")
  missing = [f for f in TIME_FEATURES if f not in feature_names]
  if missing:
    raise ValueError(
      f"feature_names must hold {list(TIME_FEATURES)} for the time scope; "
      f"got no {missing}"
    )
  return [feature_names.index(f) for f in TIME_FEATURES]


def n_rvine_structures(d: int) -> int:
  """The number of distinct R-vine structures on ``d`` labeled variables.

  ``d! / 2 * 2^((d - 2)(d - 3) / 2)`` for ``d >= 3`` (Morales-Napoles,
  2011): 3 at ``d = 3``, 24 at 4, 480 at 5; and 1 at ``d = 2``.

  Parameters
  ----------
  d : int
      The number of variables, at least 2.

  Returns
  -------
  int
      The count.
  """
  if d < 2:
    raise ValueError(f"d must be at least 2; got {d}")
  if d == 2:
    return 1
  return math.factorial(d) // 2 * int(math.pow(2, (d - 2) * (d - 3) // 2))


def all_structures(d: int) -> list[pv.RVineStructure]:
  """Every R-vine structure on ``d <= 3`` variables, in a fixed order.

  At ``d = 3`` a structure is fixed by the middle variable of its first
  tree, so there are three D-vines, ordered by that variable: ``2-1-3``,
  ``1-2-3``, ``1-3-2``.

  Parameters
  ----------
  d : int
      The number of variables, 2 or 3.

  Returns
  -------
  list of pyvinecopulib.RVineStructure
      Every structure, each once.
  """
  if d == 2:
    return [pv.RVineStructure.from_order([1, 2])]
  if d == 3:
    return [
      pv.RVineStructure.from_order(order)
      for order in ([2, 1, 3], [1, 2, 3], [1, 3, 2])
    ]
  raise ValueError(
    f"d must be 2 or 3 to enumerate every structure; got {d}, which has "
    f"{n_rvine_structures(d)}. Use a smaller mixture_k."
  )


def split_draws(n_draws: int, k: int) -> list[int]:
  """Each of ``k`` orders' share of a case's ``n_draws``.

  Parameters
  ----------
  n_draws : int
      Draws per case.
  k : int
      Orders, at most ``n_draws``.

  Returns
  -------
  list of int
      ``n_draws // k`` for each order, the first ``n_draws % k`` one more;
      they sum to ``n_draws``.

  Raises
  ------
  ValueError
      If ``k`` is not in ``1, ..., n_draws``, so an order would draw nothing.
  """
  if not 1 <= k <= n_draws:
    raise ValueError(
      f"k must be between 1 and n_draws = {n_draws}, so that every order "
      f"draws; got {k}"
    )
  share, extra = divmod(n_draws, k)
  return [share + (1 if j < extra else 0) for j in range(k)]


def mixture_draws(
  member_draws: np.ndarray, generator: np.random.Generator
) -> np.ndarray:
  """Draws from the members' equal-weight mixture, one member's per slot.

  Parameters
  ----------
  member_draws : numpy.ndarray, shape (k, n, m, d)
      Each member's ``m`` draws for each of ``n`` cases, independent across
      members.
  generator : numpy.random.Generator
      Picks the members.

  Returns
  -------
  numpy.ndarray, shape (n, m, d)
      Draw ``s`` of case ``i`` is ``member_draws[c, i, s]``, with ``c``
      uniform on the ``k`` members and drawn independently for every
      ``(i, s)``.
  """
  k, n, m = member_draws.shape[:3]
  choice = generator.integers(k, size=(n, m))
  return member_draws[choice, np.arange(n)[:, None], np.arange(m)[None, :]]


def check_boostcopula(script: str | None = None) -> str:
  """Stop unless R can run the boostCopula arms; return its versions.

  Runs the script's ``--check``, which loads its packages without fitting.

  Parameters
  ----------
  script : str, optional
      The script; :data:`BOOSTCOPULA_SCRIPT` when omitted.

  Returns
  -------
  str
      R's version and each package's, as the script prints them.

  Raises
  ------
  RuntimeError
      If the script, the interpreter or a package is missing.
  """
  path = BOOSTCOPULA_SCRIPT if script is None else script
  if not Path(path).is_file():
    raise RuntimeError(
      f"the boostCopula script must exist; got {path!r}, resolved against "
      f"{Path.cwd()}"
    )
  try:
    done = subprocess.run(
      [_external.RSCRIPT, path, "--check"],
      capture_output=True,
      text=True,
      check=False,
    )
  except FileNotFoundError as exc:
    raise RuntimeError(
      f"{_external.RSCRIPT} must be on PATH to run {path}; it was not found"
    ) from exc
  if done.returncode != 0:
    raise RuntimeError(
      f"{path} --check must exit with status 0; got {done.returncode}, with "
      f"stderr:\n{done.stderr.strip()}"
    )
  return done.stdout.strip()


def run_copula_arm(
  arm: str,
  ms: MarginalSet,
  *,
  estimators: Mapping[str, EstimatorSpec],
  n_draws: int,
  sample_batch_size: int,
  mixture_k: int,
  seed: int,
  device: str | None,
  sample_crps: bool = False,
  subsets: Mapping[str, Sequence[str]] | None = None,
  project_onto: Sequence[str] | None = None,
  projected_column: str = "joint_nll_modeled",
) -> list[CopulaResult]:
  """Fit one copula arm to a marginal set and score it on its test cases.

  Parameters
  ----------
  arm : str
      One of :data:`COPULA_ARMS`.
  ms : MarginalSet
      One station's margins from one marginal model.
  estimators : mapping of str to EstimatorSpec
      Each npcc backend's E1 estimator, by the backend's name in the arm;
      read only by an npcc arm, for its own backend.
  n_draws : int
      Draws per test case for the sample scores, at least 2.
  sample_batch_size : int
      The most rows an npcc vine samples in one call.
  mixture_k : int
      A mixture's number of members, and a chain mixture's number of orders.
  seed : int
      The run's seed, from which every seed of the arm derives.
  device : str or None
      Where an npcc vine fits and evaluates; ``None`` picks CUDA when it is
      available.
  sample_crps : bool, default=False
      Whether to estimate every response's CRPS from the draws even when no
      response is derived, as for an arm of another design that models every
      response is derived; with derived responses it is estimated regardless.
  subsets : mapping of str to sequence of str, optional
      Response subsets by name: every result with draws of its own is also
      scored on each, as ``energy_<name>`` and ``variogram_<name>``, the
      responses divided by the same scales as for ``energy``.
  project_onto : sequence of str, optional
      For an independence or Gaussian arm, responses of the marginal set
      whose joint log score under the arm's own model is also returned, in
      ``projected_column``: the projection of either copula onto a subset
      is exact. Ignored by every other arm, whose projection is not.
  projected_column : str, default="joint_nll_modeled"
      The column of that log score: the main design's, which it compares
      with.

  Returns
  -------
  list of CopulaResult
      The arm's result; for a mixture or a chain mixture, the mixture's
      followed by each member's, ``<arm>-m1`` to ``<arm>-m<mixture_k>``.

  Raises
  ------
  CopulaFailure
      If the arm cannot be scored on every test case.
  ValueError
      If the arm is unknown, its backend has no estimator, or the marginal
      set or a setting is invalid.
  """
  parsed = _parse(arm)
  if n_draws < 2:
    raise ValueError(f"n_draws must be at least 2; got {n_draws}")
  if sample_batch_size < 1:
    raise ValueError(
      f"sample_batch_size must be positive; got {sample_batch_size}"
    )
  if mixture_k < 1:
    raise ValueError(f"mixture_k must be positive; got {mixture_k}")
  inputs = _inputs(
    ms,
    n_draws=n_draws,
    seed=seed,
    sample_crps=sample_crps,
    subsets=subsets or {},
    project_onto=project_onto,
    projected_column=projected_column,
  )
  logger.info("station %d, %s: fitting %s", ms.station, ms.model, arm)
  if parsed.kind == "independence":
    results = [_independence(arm, inputs)]
  elif parsed.kind == "gaussian":
    results = [_gaussian(arm, inputs)]
  else:
    assert parsed.scope is not None
    columns = scope_columns(ms.feature_names, parsed.scope)
    if parsed.kind == "boostcopula":
      results = [_boostcopula(parsed, inputs, columns)]
    else:
      assert parsed.backend is not None
      spec = estimators.get(parsed.backend)
      if spec is None:
        raise ValueError(
          f"estimators must map the backend {parsed.backend!r} of {arm!r} to "
          f"its estimator; got {sorted(estimators)}"
        )
      controls = spec.controls(
        FitControlsRosenblattVinecop,
        device=device,
        projection_grid_size=PROJECTION_GRID_SIZE,
        batch_size=None,
      )
      vines = _Vines(
        inputs,
        columns,
        controls,
        device=device,
        sample_batch_size=sample_batch_size,
        info=_npcc_info(spec, parsed, ms, columns, sample_batch_size),
      )
      if parsed.kind == "npcc-random":
        results = [vines.random(arm)]
      elif parsed.kind == "npcc-chain":
        results = vines.chain(arm, mixture_k)
      else:
        results = vines.mixture(arm, mixture_k)
  for result in results:
    logger.info(
      "station %d, %s, %s: %s",
      ms.station,
      ms.model,
      result.arm,
      ", ".join(f"{k} {v:.2f}s" for k, v in result.timings.items()),
    )
  return results


def _inputs(
  ms: MarginalSet,
  *,
  n_draws: int,
  seed: int,
  sample_crps: bool = False,
  subsets: Mapping[str, Sequence[str]] | None = None,
  project_onto: Sequence[str] | None = None,
  projected_column: str = "joint_nll_modeled",
) -> _Inputs:
  """Refuse a marginal set no arm can be scored on; clip its transforms."""
  drawn = (*ms.variables, *ms.derived)
  for name, subset in (subsets or {}).items():
    missing = [v for v in subset if v not in drawn]
    if missing:
      raise ValueError(
        f"subsets.{name} must be drawn from the responses {list(drawn)} of "
        f"station {ms.station}, {ms.model}; got {missing}"
      )
  if project_onto is not None:
    missing = [v for v in project_onto if v not in ms.variables]
    if missing:
      raise ValueError(
        f"project_onto must name modeled responses {list(ms.variables)} of "
        f"station {ms.station}, {ms.model}; got {missing}"
      )
  for name in ("pit_train", "pit_test"):
    values = getattr(ms, name)
    if not (np.isfinite(values) & (values >= 0) & (values <= 1)).all():
      raise ValueError(
        f"MarginalSet.{name} must hold finite values in [0, 1]; got others "
        f"for station {ms.station}, {ms.model}"
      )
  scale = np.std(ms.y_train, axis=0)
  if not (np.isfinite(scale) & (scale > 0)).all():
    raise ValueError(
      f"MarginalSet.y_train must vary in every variable to standardize the "
      f"sample scores; got standard deviations {scale.tolist()} for station "
      f"{ms.station}, {ms.model}"
    )
  derived_scale = np.std(ms.y_derived_train, axis=0)
  if (
    ms.derived and not (np.isfinite(derived_scale) & (derived_scale > 0)).all()
  ):
    raise ValueError(
      f"MarginalSet.y_derived_train must vary in every variable to "
      f"standardize the sample scores; got standard deviations "
      f"{derived_scale.tolist()} for station {ms.station}, {ms.model}"
    )
  return _Inputs(
    ms=ms,
    u_train=np.clip(ms.pit_train, PIT_CLIP, 1 - PIT_CLIP),
    u_test=np.clip(ms.pit_test, PIT_CLIP, 1 - PIT_CLIP),
    scale=scale,
    derived_scale=derived_scale,
    n_draws=n_draws,
    seed=seed,
    sample_crps=sample_crps,
    subsets={k: tuple(v) for k, v in (subsets or {}).items()},
    project_onto=None if project_onto is None else tuple(project_onto),
    projected_column=projected_column,
  )


def _score(
  name: str,
  inputs: _Inputs,
  logcop: np.ndarray,
  uniforms: np.ndarray,
  timings: dict[str, float],
  info: dict[str, object],
) -> CopulaResult:
  """Check a copula's draws, map them to responses and score the cases."""
  ms = inputs.ms
  shape = inputs.draws_shape
  if uniforms.shape != shape:
    raise CopulaFailure(
      f"{name}'s draws must have shape {shape}; got {uniforms.shape}"
    )
  valid = np.isfinite(uniforms) & (uniforms >= 0) & (uniforms <= 1)
  bad = np.flatnonzero(~valid.all(axis=(1, 2)))
  if bad.size:
    raise CopulaFailure(
      f"{name}'s draws must be finite and in [0, 1]; got others at "
      f"{bad.size} test cases",
      bad,
    )
  t0 = perf_counter()
  draws = inverse_margins(uniforms, ms.levels, ms.quantiles_test)
  timings["inverse_margins"] = perf_counter() - t0

  sample_crps: dict[str, np.ndarray] = {}
  observed_raw, scale = ms.y_test, inputs.scale
  if ms.derived:
    t0 = perf_counter()
    extra = derive(draws, ms.variables, ms.derived, ms.derivation)
    draws = np.concatenate([draws, extra], axis=2)
    observed_raw = np.concatenate([ms.y_test, ms.y_derived_test], axis=1)
    scale = np.concatenate([inputs.scale, inputs.derived_scale])
    timings["derived_responses"] = perf_counter() - t0
    bad = np.flatnonzero(~np.isfinite(extra).all(axis=(1, 2)))
    if bad.size:
      raise CopulaFailure(
        f"{name}'s derived responses must be finite; got others at "
        f"{bad.size} test cases",
        bad,
      )

  t0 = perf_counter()
  standardized = draws / scale
  observed = observed_raw / scale
  energy = energy_score(standardized, observed)
  variogram = variogram_score(standardized, observed, p=0.5)
  drawn = (*ms.variables, *ms.derived)
  extra: dict[str, np.ndarray] = {}
  for subset_name, subset in inputs.subsets.items():
    columns = [drawn.index(v) for v in subset]
    extra[f"energy_{subset_name}"] = energy_score(
      standardized[:, :, columns], observed[:, columns]
    )
    extra[f"variogram_{subset_name}"] = variogram_score(
      standardized[:, :, columns], observed[:, columns], p=0.5
    )
  if ms.derived or inputs.sample_crps:
    for j, variable in enumerate((*ms.variables, *ms.derived)):
      sample_crps[variable] = energy_score(
        draws[:, :, j : j + 1], observed_raw[:, j : j + 1]
      )
  nll = joint_nll(logcop, ms.logpdf_test)
  bad = np.flatnonzero(~np.isfinite(nll))
  if bad.size:
    raise CopulaFailure(
      f"{name}'s joint log-likelihood must be finite; got non-finite values "
      f"at {bad.size} test cases",
      bad,
    )
  finite = np.isfinite(energy) & np.isfinite(variogram)
  for values in extra.values():
    finite &= np.isfinite(values)
  bad = np.flatnonzero(~finite)
  if bad.size:
    raise CopulaFailure(
      f"{name}'s energy and variogram scores must be finite; got non-finite "
      f"values at {bad.size} test cases",
      bad,
    )
  physical = physical_violations(
    draws, (*ms.variables, *ms.derived), ms.derivation
  )
  timings["scoring"] = perf_counter() - t0
  if ms.derived:
    info = {
      **info,
      "modeled": list(ms.variables),
      "derived": list(ms.derived),
      "magnus": ms.derivation.magnus,
    }
  return CopulaResult(
    arm=name,
    logcop_test=logcop,
    joint_nll=nll,
    energy=energy,
    variogram=variogram,
    timings=timings,
    info=info,
    sample_crps=sample_crps,
    draws=draws,
    physical=physical,
    extra=extra,
  )


def _projected(
  inputs: _Inputs, result: CopulaResult, logcop: np.ndarray | None
) -> CopulaResult:
  """Add the arm's exact log score of ``project_onto``, if asked for.

  ``logcop`` is the copula log-density of those responses at the test
  cases; ``None`` for independence, whose projection is independence.
  """
  if inputs.project_onto is None:
    return result
  ms = inputs.ms
  columns = [ms.variables.index(v) for v in inputs.project_onto]
  sub = np.zeros(len(ms.y_test)) if logcop is None else logcop
  nll = joint_nll(sub, ms.logpdf_test[:, columns])
  if not np.isfinite(nll).all():
    raise CopulaFailure(
      f"{result.arm}'s projected log-likelihood must be finite; got "
      f"non-finite values at {int((~np.isfinite(nll)).sum())} test cases",
      np.flatnonzero(~np.isfinite(nll)),
    )
  return replace(result, extra={**result.extra, inputs.projected_column: nll})


def _log_score_only(
  name: str,
  inputs: _Inputs,
  logcop: np.ndarray,
  timings: dict[str, float],
  info: dict[str, object],
) -> CopulaResult:
  """A copula scored by its log score alone; its sample scores are ``nan``."""
  ms = inputs.ms
  t0 = perf_counter()
  nll = joint_nll(logcop, ms.logpdf_test)
  bad = np.flatnonzero(~np.isfinite(nll))
  if bad.size:
    raise CopulaFailure(
      f"{name}'s joint log-likelihood must be finite; got non-finite values "
      f"at {bad.size} test cases",
      bad,
    )
  timings["scoring"] = perf_counter() - t0
  if ms.derived:
    info = {
      **info,
      "modeled": list(ms.variables),
      "derived": list(ms.derived),
      "magnus": ms.derivation.magnus,
    }
  missing = np.full(len(ms.y_test), np.nan)
  return CopulaResult(
    arm=name,
    logcop_test=logcop,
    joint_nll=nll,
    energy=missing,
    variogram=missing.copy(),
    timings=timings,
    info=info,
    sampled=False,
  )


def _independence(name: str, inputs: _Inputs) -> CopulaResult:
  t0 = perf_counter()
  draws_seed = derive_seed(*inputs.salt(name), "draws")
  logcop = np.zeros(len(inputs.ms.y_test))
  uniforms = np.random.default_rng(draws_seed).random(inputs.draws_shape)
  timings = {"fit": 0.0, "copula_prediction": perf_counter() - t0}
  info: dict[str, object] = {"draws_seed": draws_seed}
  return _projected(
    inputs, _score(name, inputs, logcop, uniforms, timings, info), None
  )


def _normal_scores(u: np.ndarray) -> np.ndarray:
  return to_numpy(torch.special.ndtri(torch.from_numpy(u)), dtype=np.float64)


def _gaussian(name: str, inputs: _Inputs) -> CopulaResult:
  """The Gaussian copula with the training normal scores' correlation.

  The sample correlation is projected onto the positive definite matrices,
  its eigenvalues floored at ``_EIGEN_FLOOR``, and rescaled to a unit
  diagonal.
  """
  t0 = perf_counter()
  z = _normal_scores(inputs.u_train)
  sd = np.std(z, axis=0)
  if (sd < _CONSTANT_SD).any():
    raise CopulaFailure(
      f"{name} must have varying training transforms to fit a correlation; "
      f"got normal-score standard deviations {sd.tolist()}"
    )
  sample = np.corrcoef(z, rowvar=False)
  values, vectors = np.linalg.eigh(sample)
  floored = (vectors * np.maximum(values, _EIGEN_FLOOR)) @ vectors.T
  unit = np.sqrt(np.diag(floored))
  correlation = floored / unit[:, None] / unit[None, :]
  timings = {"fit": perf_counter() - t0}

  t0 = perf_counter()
  z_test = _normal_scores(inputs.u_test)
  precision = np.linalg.inv(correlation) - np.eye(len(correlation))
  logcop = -0.5 * np.linalg.slogdet(correlation)[1] - 0.5 * np.einsum(
    "ni,ij,nj->n", z_test, precision, z_test
  )
  draws_seed = derive_seed(*inputs.salt(name), "draws")
  normal = np.random.default_rng(draws_seed).standard_normal(inputs.draws_shape)
  correlated = normal @ np.linalg.cholesky(correlation).T
  uniforms = to_numpy(
    torch.special.ndtr(torch.from_numpy(correlated)), dtype=np.float64
  )
  timings["copula_prediction"] = perf_counter() - t0
  info: dict[str, object] = {
    "correlation": correlation.tolist(),
    "draws_seed": draws_seed,
  }
  result = _score(name, inputs, logcop, uniforms, timings, info)
  projected = None
  if inputs.project_onto is not None:
    # The Gaussian copula's projection onto a subset is the Gaussian copula
    # of its correlation's submatrix.
    keep = [inputs.ms.variables.index(v) for v in inputs.project_onto]
    sub = correlation[np.ix_(keep, keep)]
    z_sub = z_test[:, keep]
    projected = -0.5 * np.linalg.slogdet(sub)[1] - 0.5 * np.einsum(
      "ni,ij,nj->n", z_sub, np.linalg.inv(sub) - np.eye(len(keep)), z_sub
    )
  return _projected(inputs, result, projected)


def _synchronize() -> None:
  """Wait for queued GPU work, so a stage's seconds include it."""
  if torch.cuda.is_available() and torch.cuda.is_initialized():
    torch.cuda.synchronize()


def _matrix(structure: pv.RVineStructure) -> list[list[int]]:
  # pyvinecopulib's compiled `RVineStructure` is untyped here.
  return cast("list[list[int]]", np.asarray(structure.matrix).tolist())


def _npcc_info(
  spec: EstimatorSpec,
  parsed: _Arm,
  ms: MarginalSet,
  columns: list[int],
  sample_batch_size: int,
) -> dict[str, object]:
  # Through JSON, so nested keyword arguments are plain values.
  kwargs = json.loads(json.dumps(dict(spec.backend_kwargs)))
  return {
    "estimator_label": spec.label,
    "estimator_id": spec.estimator_id,
    "backend": spec.backend,
    "transform": spec.transform,
    "feature_transform": spec.feature_transform,
    "backend_kwargs": kwargs,
    "scope": parsed.scope,
    "covariates": [ms.feature_names[c] for c in columns],
    "context": "non-simplified",
    "sample_batch_size": sample_batch_size,
  }


class _Vines:
  """The npcc arms' vines on one marginal set, fitted one at a time."""

  def __init__(
    self,
    inputs: _Inputs,
    columns: list[int],
    controls: FitControlsRosenblattVinecop,
    *,
    device: str | None,
    sample_batch_size: int,
    info: dict[str, object],
  ) -> None:
    ms = inputs.ms
    self.inputs = inputs
    self.controls = controls
    self.device = device
    self.sample_batch_size = sample_batch_size
    self.info = info
    self.u_train = torch.from_numpy(inputs.u_train)
    self.u_test = torch.from_numpy(inputs.u_test)
    self.x_train = torch.as_tensor(ms.x_train[:, columns], dtype=torch.float64)
    self.x_test = torch.as_tensor(ms.x_test[:, columns], dtype=torch.float64)

  def fit(
    self, name: str, structure: pv.RVineStructure
  ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit one vine; return its test log-density, its draws and timings.

    The vine is released before this returns, so only one vine's pair
    copulas occupy the device at a time. The torch seed is set before the
    fit, as E1's runner sets it, for a backend that draws from torch's
    generator.
    """
    salt = self.inputs.salt(name)
    torch.manual_seed(derive_seed(*salt, "fit"))
    _synchronize()
    t0 = perf_counter()
    vine = RosenblattVinecop(None, structure, device=self.device)
    vine.fit(self.u_train, self.controls, x=self.x_train)
    _synchronize()
    timings = {"fit": perf_counter() - t0}

    t0 = perf_counter()
    with torch.inference_mode():
      logcop = to_numpy(
        vine.logpdf(self.u_test, x=self.x_test), dtype=np.float64
      )
      uniforms = self._sample(vine, salt)
    _synchronize()
    timings["copula_prediction"] = perf_counter() - t0
    del vine
    release_gpu()
    return logcop, uniforms, timings

  def _sample(
    self,
    model: RosenblattVinecop | ChainCopula,
    salt: tuple[object, ...],
    n_draws: int | None = None,
  ) -> np.ndarray:
    """``n_draws`` draws per test case, sampled in bounded chunks.

    ``n_draws`` defaults to the run's; a chain mixture's order passes its
    share.
    """
    n, d = self.u_test.shape
    n_draws = self.inputs.n_draws if n_draws is None else n_draws
    total = n * n_draws
    out = np.empty((total, d), dtype=np.float64)
    for chunk, start in enumerate(range(0, total, self.sample_batch_size)):
      stop = min(start + self.sample_batch_size, total)
      rows = torch.arange(start, stop) // n_draws
      draws = model.sample(
        stop - start,
        seeds=[derive_seed(*salt, "sample", chunk)],
        x=self.x_test[rows],
      )
      out[start:stop] = to_numpy(draws, dtype=np.float64)
    return out.reshape(n, n_draws, d)

  def random(self, arm: str) -> CopulaResult:
    """The vine on the station's uniformly random structure."""
    ms = self.inputs.ms
    structure_seed = derive_seed(
      self.inputs.seed, "random-structure", ms.station
    )
    structure = pv.RVineStructure.sample(
      len(ms.variables), seeds=[structure_seed]
    )
    logcop, uniforms, timings = self.fit(arm, structure)
    info = {
      **self.info,
      "structure": _matrix(structure),
      "structure_seed": structure_seed,
      "fit_seed": derive_seed(*self.inputs.salt(arm), "fit"),
    }
    return _score(arm, self.inputs, logcop, uniforms, timings, info)

  def mixture(self, arm: str, k: int) -> list[CopulaResult]:
    """The mixture of ``k`` vines, then each member on its own.

    Where ``k`` reaches the number of distinct R-vine structures on the
    variables, as at ``d = 3``, the mixture is of every structure: the
    selected one first, then the others in :func:`all_structures`'s order.
    """
    ms = self.inputs.ms
    requested = k
    d = len(ms.variables)
    others: list[pv.RVineStructure] = []
    every = k >= n_rvine_structures(d)
    if every:
      k = n_rvine_structures(d)
    members: list[CopulaResult] = []
    members_info: list[dict[str, object]] = []
    logs: list[np.ndarray] = []
    draws: list[np.ndarray] = []
    structures: list[pv.RVineStructure] = []
    for j in range(1, k + 1):
      name = f"{arm}-m{j}"
      t0 = perf_counter()
      if j == 1:
        structure_seed = None
        method = "dissmann-tll"
        structure = select_structure(self.u_train)
        if every:
          chosen = _external_vine.edges(structure)
          others = [
            s for s in all_structures(d) if _external_vine.edges(s) != chosen
          ]
      elif every:
        structure_seed = None
        method = "enumerated"
        structure = others[j - 2]
      else:
        structure_seed = derive_seed(
          self.inputs.seed, "mixture-structure", ms.station, j
        )
        method = "weighted-random"
        structure = weighted_random_structure(self.u_train, structure_seed)
      selection = perf_counter() - t0
      logcop, uniforms, timings = self.fit(name, structure)
      member_info: dict[str, object] = {
        "member": j,
        "arm": name,
        "structure": _matrix(structure),
        "structure_seed": structure_seed,
        "structure_method": method,
        "fit_seed": derive_seed(*self.inputs.salt(name), "fit"),
      }
      members_info.append(member_info)
      members.append(
        _score(
          name,
          self.inputs,
          logcop,
          uniforms,
          {"structure_selection": selection, **timings},
          {**self.info, **member_info},
        )
      )
      logs.append(logcop)
      draws.append(uniforms)
      structures.append(structure)

    t0 = perf_counter()
    logcop = to_numpy(
      mixture_log_density(torch.from_numpy(np.stack(logs))), dtype=np.float64
    )
    choice_seed = derive_seed(*self.inputs.salt(arm), "choice")
    uniforms = mixture_draws(
      np.stack(draws), np.random.default_rng(choice_seed)
    )
    mixing = perf_counter() - t0
    timings = {
      stage: sum(m.timings[stage] for m in members)
      for stage in ("structure_selection", "fit", "copula_prediction")
    }
    timings["copula_prediction"] += mixing
    distinct = {frozenset(_external_vine.edges(s)) for s in structures}
    info = {
      **self.info,
      "mixture_k": k,
      "mixture_k_requested": requested,
      "every_structure": every,
      "members": members_info,
      "n_distinct_structures": len(distinct),
      "choice_seed": choice_seed,
    }
    mixture = _score(arm, self.inputs, logcop, uniforms, timings, info)
    return [mixture, *members]

  def chain(self, arm: str, k: int) -> list[CopulaResult]:
    """The chain-rule mixture over ``min(k, d!)`` orders, then each order.

    Each order is fitted, evaluated at the test cases, sampled for its share
    of the draws (:func:`split_draws`) and released before the next, so only
    one order's factors occupy the device at a time. An order is scored by
    its log score alone.
    """
    ms = self.inputs.ms
    d = len(ms.variables)
    orders = chain_orders(self.inputs.seed, d, ms.station, k)
    counts = split_draws(self.inputs.n_draws, len(orders))
    cell_seed = derive_seed(*self.inputs.salt(arm))
    estimator_id = str(self.info["estimator_id"])
    members: list[CopulaResult] = []
    members_info: list[dict[str, object]] = []
    logs: list[np.ndarray] = []
    draws: list[np.ndarray] = []
    for j, (order, count) in enumerate(zip(orders, counts, strict=True), 1):
      name = f"{arm}-m{j}"
      seeds = chain_factor_seeds(cell_seed, estimator_id, j, d)
      _synchronize()
      t0 = perf_counter()
      chain = ChainCopula(order, self.controls).fit(
        self.u_train, x=self.x_train, seeds=seeds
      )
      _synchronize()
      timings = {"fit": perf_counter() - t0}
      t0 = perf_counter()
      with torch.inference_mode():
        logcop = to_numpy(
          chain.logpdf(self.u_test, x=self.x_test), dtype=np.float64
        )
        uniforms = self._sample(chain, self.inputs.salt(name), count)
      _synchronize()
      timings["copula_prediction"] = perf_counter() - t0
      del chain
      release_gpu()
      member_info: dict[str, object] = {
        "member": j,
        "arm": name,
        "order": order,
        "factor_seeds": seeds,
        "draws_per_case": count,
      }
      members_info.append(member_info)
      members.append(
        _log_score_only(
          name, self.inputs, logcop, timings, {**self.info, **member_info}
        )
      )
      logs.append(logcop)
      draws.append(uniforms)

    t0 = perf_counter()
    logcop = to_numpy(
      mixture_log_density(torch.from_numpy(np.stack(logs))), dtype=np.float64
    )
    uniforms = np.concatenate(draws, axis=1)
    mixing = perf_counter() - t0
    timings = {
      stage: sum(m.timings[stage] for m in members)
      for stage in ("fit", "copula_prediction")
    }
    timings["copula_prediction"] += mixing
    info = {
      **self.info,
      "context": "chain-rule",
      "mixture_k": len(orders),
      "mixture_k_requested": k,
      "orders": orders,
      "n_distinct_orders": len({tuple(o) for o in orders}),
      "cell_seed": cell_seed,
      "draws_per_case": counts,
      "members": members_info,
    }
    mixture = _score(arm, self.inputs, logcop, uniforms, timings, info)
    return [mixture, *members]


def _cases(u: np.ndarray, x: np.ndarray) -> pd.DataFrame:
  return pd.DataFrame(
    {
      **{f"u{j + 1}": u[:, j] for j in range(u.shape[1])},
      **{f"x{k + 1}": x[:, k] for k in range(x.shape[1])},
    }
  )


def _written(path: Path, *, script: str) -> Path:
  if not path.is_file():
    raise _external.ExchangeError(
      f"{script} must write {path.name}; got no such file"
    )
  return path


def _read_numbers(path: Path, *, script: str) -> pd.DataFrame:
  # The default parser is not exact for float64.
  return pd.read_csv(
    _written(path, script=script), float_precision="round_trip"
  )


def _boostcopula(
  arm: _Arm, inputs: _Inputs, columns: list[int]
) -> CopulaResult:
  """Fit and evaluate boostCopula's vine with one run of its script."""
  ms = inputs.ms
  d = len(ms.variables)
  name = arm.name
  if d < BOOSTCOPULA_MIN_DIMENSION:
    raise ValueError(
      f"{name} must be fitted to at least {BOOSTCOPULA_MIN_DIMENSION} "
      f"variables; got {d}"
    )
  script = BOOSTCOPULA_SCRIPT
  if not Path(script).is_file():
    raise RuntimeError(
      f"the boostCopula script must exist; got {script!r}, resolved against "
      f"{Path.cwd()}"
    )
  r_seed = derive_seed(*inputs.salt(name))
  train = _cases(inputs.u_train, ms.x_train[:, columns])
  test = _cases(inputs.u_test, ms.x_test[:, columns])
  u_columns = [f"u{j + 1}" for j in range(d)]
  label = f"{name} on station {ms.station}, {ms.model}"
  with tempfile.TemporaryDirectory(prefix="npcc-boostcopula-") as tmp:
    root = Path(tmp)
    train.to_csv(root / "train.csv", index=False, float_format="%.17g")
    test.to_csv(root / "test.csv", index=False, float_format="%.17g")
    t0 = perf_counter()
    try:
      _external._run(
        [
          _external.RSCRIPT,
          script,
          str(root),
          str(r_seed),
          str(inputs.n_draws),
        ],
        name=label,
        refused_status=_external.REFUSED_STATUS,
      )
    except _external.ScriptFailedError as exc:
      raise CopulaFailure(f"{label} failed: {exc}") from exc
    wall = perf_counter() - t0
    density = _read_numbers(root / "density.csv", script=script)
    draws = _read_numbers(root / "draws.csv", script=script)
    fitted, families = (
      np.loadtxt(_written(root / f, script=script), delimiter=",", ndmin=2)
      for f in ("fitted.csv", "families.csv")
    )
    r_info = pd.read_csv(
      _written(root / "info.csv", script=script),
      dtype=str,
      keep_default_na=False,
    )

  if list(density.columns) != [*u_columns, "pdf"] or len(density) != len(test):
    raise _external.ExchangeError(
      f"{script} must write density.csv with the columns {u_columns} and pdf, "
      f"one row per test case; got {list(density.columns)} and "
      f"{len(density)} rows for {len(test)} cases"
    )
  echoed = np.isclose(
    density[u_columns].to_numpy(dtype=np.float64),
    test[u_columns].to_numpy(),
    rtol=0.0,
    atol=1e-12,
  ).all(axis=1)
  if not echoed.all():
    raise _external.ExchangeError(
      f"{script} must echo the test cases in their order in density.csv; got "
      f"a different case at row {int(np.argmin(echoed))}"
    )
  cases = np.repeat(np.arange(1, len(test) + 1), inputs.n_draws)
  if list(draws.columns) != ["case", *u_columns] or not np.array_equal(
    draws["case"].to_numpy(), cases
  ):
    raise _external.ExchangeError(
      f"{script} must write draws.csv with the columns case and {u_columns}, "
      f"{inputs.n_draws} rows per test case in their order; got "
      f"{list(draws.columns)} and {len(draws)} rows"
    )
  try:
    r_values = dict(zip(r_info["key"], r_info["value"], strict=True))
    r_seconds = {
      step: float(r_values[step])
      for step in ("fit_time", "pdf_time", "sim_time")
    }
    structure = _external_vine.from_vinecopula(fitted)
  except (KeyError, ValueError, RuntimeError) as exc:
    raise _external.ExchangeError(
      f"{script} must write its timings to info.csv and a valid R-vine "
      f"matrix to fitted.csv; got {r_info.to_dict('records')} and "
      f"{fitted.tolist()}"
    ) from exc

  pdf = density["pdf"].to_numpy(dtype=np.float64)
  bad = np.flatnonzero(~(np.isfinite(pdf) & (pdf > 0)))
  if bad.size:
    raise CopulaFailure(
      f"{label}'s density must be finite and positive; got others at "
      f"{bad.size} test cases",
      bad,
    )
  uniforms = draws[u_columns].to_numpy(dtype=np.float64)
  prediction = r_seconds["pdf_time"] + r_seconds["sim_time"]
  timings = {
    "fit": r_seconds["fit_time"],
    "copula_prediction": prediction,
    "bridge_overhead": max(0.0, wall - r_seconds["fit_time"] - prediction),
  }
  info: dict[str, object] = {
    "scope": arm.scope,
    "covariates": [ms.feature_names[c] for c in columns],
    "structure": _matrix(structure),
    "structure_vinecopula": fitted.astype(np.int64).tolist(),
    "families_vinecopula": families.astype(np.int64).tolist(),
    "r_seed": r_seed,
    "r_version": r_values.get("r_version", ""),
    "r_packages": r_values.get("packages", ""),
    "script": script,
    "script_sha256": hashlib.sha256(Path(script).read_bytes()).hexdigest(),
  }
  return _score(
    name,
    inputs,
    np.log(pdf),
    uniforms.reshape(len(test), inputs.n_draws, d),
    timings,
    info,
  )
