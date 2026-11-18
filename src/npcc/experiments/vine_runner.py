"""Run the vine study: draw truths, fit each arm, score, aggregate.

A truth, its random-structure competitor and its test set are keyed on
``(d, rep)`` alone, so every sample size ``n`` in a repetition faces the same
copula and the same test points and the curve over ``n`` is comparable.

Every arm estimates the joint distribution on the data scale, margins and
copula together: the NPCC arms as a :class:`~npcc.core.vinedist.
RosenblattVinedist` whose margins and pairs all see ``x``, and ``tll`` as a
:class:`pyvinecopulib.Vinedist` with kernel-density margins and TLL pairs on
the true structure, neither of which sees ``x``.

**Structures.** ``oracle`` fits the true structure and ``random`` one drawn
per ``(d, rep)``. ``selected`` fits the structure :func:`select_structure`
picks from the probability integral transforms of the arm's own fitted
margins, so it varies with ``n``. Each NPCC arm's runtime row records the
structure it fitted, and ``selected``'s the seconds its selection took.

**Mixture.** ``mixture`` fits ``mixture_k`` non-simplified vines on one set
of fitted margins, each on its own structure picked from the margins'
probability integral transforms: the first is the ``selected`` arm's, and
each other a draw of pyvinecopulib's weighted random spanning trees
(:func:`weighted_random_structure`), diverse but favoring strong pairs. It
scores their equal-weight mixture: its joint log-density is the margins' plus
``logsumexp`` of the members' copula log-densities less ``log K``, and its
copula density the members' mean. Each member is also scored on its own, as
the arm ``mixture-member-<k>``, so the mixture can be read against its
average member and against its first, the selected structure.

**Chain.** ``chain`` asks what a vine adds at all. On the same fitted margins
it fits ``min(chain_k, d!)`` chain-rule copula densities
(:class:`~npcc.experiments.chain.ChainCopula`), each along a variable order,
the orders distinct and drawn uniformly per ``(d, rep)``
(:func:`chain_orders`), and scores their
equal-weight mixture exactly as ``mixture`` scores its vines. Each order is
also scored on its own, as the arm ``chain-m<k>``. Factor ``k`` of an order
conditions on the earlier scores and ``x`` together and has a torch seed of
its own (:func:`chain_factor_seeds`).

**GAM vines.** E2b's arms are fitted in R by the grid's ``gamvine_script``,
one script run per arm and cell (:mod:`~npcc.experiments._external_vine`),
on the cell's training sample, and evaluated at its test points.
``gamvine-oracle`` is handed the true structure and refused unless its fit
has the truth's edges; the other two select their own. Each runtime row
records the structure fitted and, in ``pairs``, every pair copula's family
and calibration function. A script that fails on its sample leaves the
arm's metrics ``nan`` in that cell and its message in the runtime row's
``error``, as for E9's R methods.

**Scoring.** At each evaluation ``x``, ``m`` points are drawn from the true
distribution there, and the estimate is scored by Monte-Carlo integrals under
the truth: ``KL = E[log f - log f_hat]``, ``L1 = E[|f_hat / f - 1|]`` (the
integrated absolute error) and ``ISE = E[(f_hat - f)^2 / f]`` (the integrated
squared error). Each is a fixed-``x`` metric, summarized over repetitions
afterward.

``KL_copula`` and ``L1_copula`` score the fitted vine copula alone, at the
true probability integral transforms, which separates a copula error from a
margin error. There is no copula-scale ISE: its Monte-Carlo estimate divides
by ``c``, and a copula density's corner peaks make it too heavy-tailed to
read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping
from time import perf_counter
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
import pyvinecopulib as pv
import torch
from pyvinecopulib.core import (
  ConditioningContext,
  NonSimplifiedContext,
  SimplifiedContext,
)
from pyvinecopulib.core.extend import to_numpy

from npcc.core.controls import FitControlsRosenblattVinecop
from npcc.core.margin import ConditionalMargin, QuantileTableConfig
from npcc.core.registry import create_backend
from npcc.core.vinecop import RosenblattVinecop
from npcc.core.vinedist import RosenblattVinedist
from npcc.experiments import (
  _external,
  _external_vine,
  scenarios,
  vine_scenarios,
)
from npcc.experiments._study import (
  Rows,
  derive_seed,
  memory_usage,
  release_gpu,
  reset_gpu_peaks,
  sorted_frame,
  summarize,
  sweep,
)
from npcc.experiments.chain import ChainCopula, draw_orders
from npcc.experiments.config import EstimatorSpec, RunConfig
from npcc.experiments.vine_config import (
  GAMVINE_ARMS,
  VineCell,
  VineGridConfig,
)
from npcc.experiments.vine_scenarios import EvalPoints, VineTruth

logger = logging.getLogger(__name__)

_EPS: float = 1e-300

METRIC_NAMES: tuple[str, ...] = ("KL", "L1", "ISE", "KL_copula", "L1_copula")
TABLES: tuple[str, ...] = ("metrics", "runtime")
_GROUP_AXES: tuple[str, ...] = (
  "d",
  "n",
  "arm",
  "label",
  "estimator_id",
  "backend",
  "transform",
)


def truth_for(base_seed: int, d: int, rep: int) -> VineTruth:
  """The true vine of repetition ``rep`` in dimension ``d``."""
  return vine_scenarios.draw_truth(d, derive_seed(base_seed, "truth", d, rep))


def training_sample(
  base_seed: int, cell: VineCell
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """The cell's training sample: ``y`` and its true PITs ``u``, and ``x``.

  Drawn from the truth of the cell's ``(d, rep)`` with a seed of its own
  ``(d, n, rep)``, so a diagnostic that redraws it sees the data every arm
  was fitted on.

  Parameters
  ----------
  base_seed : int
      The run's base seed.
  cell : VineCell
      The cell.

  Returns
  -------
  tuple of torch.Tensor
      ``y`` and ``u`` of shape ``(n, d)``, and ``x`` of shape ``(n, 1)``.
  """
  return vine_scenarios.sample(
    truth_for(base_seed, cell.d, cell.rep),
    cell.n,
    derive_seed(base_seed, "train", cell.d, cell.n, cell.rep),
  )


def random_structure_for(base_seed: int, d: int, rep: int) -> pv.RVineStructure:
  """The ``random`` arm's structure, drawn independently of the truth's."""
  return pv.RVineStructure.sample(
    d, seeds=[derive_seed(base_seed, "random", d, rep)]
  )


def chain_orders(base_seed: int, d: int, rep: int, k: int) -> list[list[int]]:
  """The ``chain`` arm's ``min(k, d!)`` distinct variable orders in ``rep``.

  Keyed on ``(d, rep)`` alone, as the truth and the test points are, so every
  sample size in a repetition mixes the same orders. They are distinct, as
  the mixture's structures are: an order drawn twice would weigh twice in an
  equal-weight mixture, which at d = 3 is likely (five draws from six
  orders repeat one with probability 0.91).

  Parameters
  ----------
  base_seed : int
      The run's base seed.
  d : int
      The dimension.
  rep : int
      The repetition.
  k : int
      How many orders wanted.

  Returns
  -------
  list of list of int
      ``min(k, d!)`` distinct orders by
      :func:`~npcc.experiments.chain.draw_orders`, order ``j`` (1-based)
      from the seed ``derive_seed(base_seed, "chain", d, rep, j)``: where no
      draw repeats, exactly the orders each seed's first draw gives.
  """
  return draw_orders(
    d, [derive_seed(base_seed, "chain", d, rep, j) for j in range(1, k + 1)]
  )


def chain_factor_seeds(
  cell_seed: int, estimator_id: str, j: int, d: int
) -> list[int]:
  """The torch seeds of order ``j``'s factors, one per factor ``2, ..., d``.

  Parameters
  ----------
  cell_seed : int
      The cell's seed.
  estimator_id : str
      The estimator's id, so two estimators' factors draw differently.
  j : int
      The order, 1-based.
  d : int
      The dimension.

  Returns
  -------
  list of int
      ``derive_seed(cell_seed, "chain", estimator_id, j, f)`` for factor
      ``f = 2, ..., d``.
  """
  return [
    derive_seed(cell_seed, "chain", estimator_id, j, f) for f in range(2, d + 1)
  ]


def mixture_log_density(member_logs: torch.Tensor) -> torch.Tensor:
  """The log-density of the members' equal-weight mixture.

  Parameters
  ----------
  member_logs : torch.Tensor, shape (k, n)
      Each member's log-density at the same ``n`` points.

  Returns
  -------
  torch.Tensor, shape (n,)
      ``log((1 / k) * sum_j exp(member_logs[j]))``, computed without leaving
      log space, so members that underflow in density still count.
  """
  return torch.logsumexp(member_logs, dim=0) - math.log(member_logs.shape[0])


def joint_log_densities(
  copula_logs: torch.Tensor, margin_logs: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
  """The mixture's and each member's joint log-density, on shared margins.

  The members share their margins, so the mixture's joint density is the
  margins' times the mixture of the copula densities. Each margin term is
  added after the copula term, in variable order, as
  ``VinedistBase.logpdf`` adds them, so a member's log-density is the one
  :class:`~npcc.core.vinedist.RosenblattVinedist` gives for that member.

  Parameters
  ----------
  copula_logs : torch.Tensor, shape (k, n)
      Each member's copula log-density at the points' probability integral
      transforms.
  margin_logs : list of torch.Tensor, each of shape (n,)
      Each margin's log-density at the points, in variable order.

  Returns
  -------
  mixture : torch.Tensor, shape (n,)
      The mixture's joint log-density.
  members : torch.Tensor, shape (k, n)
      Each member's joint log-density.
  """
  mixture = mixture_log_density(copula_logs)
  members = copula_logs
  for term in margin_logs:
    mixture = mixture + term
    members = members + term
  return mixture, members


def select_structure(u: torch.Tensor) -> pv.RVineStructure:
  """The structure Dissmann's algorithm with TLL pairs selects on ``u``.

  E3d's ``tll-ns`` call (``scripts/paper/e3/e3d_npcc_fit.py``): pyvinecopulib's
  compiled selection with TLL as the only family, otherwise default
  controls, on one thread. It sees the copula-scale data only, not ``x``.

  Parameters
  ----------
  u : torch.Tensor, shape (n, d)
      Copula-scale observations.

  Returns
  -------
  pyvinecopulib.RVineStructure
      The selected structure.
  """
  controls = pv.FitControlsVinecop(
    family_set=[pv.BicopFamily.tll], num_threads=1
  )
  data = to_numpy(u, dtype=np.float64)
  # pyvinecopulib's compiled `Vinecop` is untyped here.
  selected = pv.Vinecop.from_data(data, controls=controls).structure
  return cast("pv.RVineStructure", selected)


def weighted_random_structure(u: torch.Tensor, seed: int) -> pv.RVineStructure:
  """A structure of weighted random spanning trees on ``u``.

  :func:`select_structure`'s call with pyvinecopulib's ``random_weighted``
  tree algorithm: each tree is a random spanning tree whose chance grows with
  its edges' ``|Kendall's tau|``, rather than the maximum one. TLL pairs and
  one thread, as there; ``x`` is not seen.

  Parameters
  ----------
  u : torch.Tensor, shape (n, d)
      Copula-scale observations.
  seed : int
      The draw's seed; the same ``u`` and seed give the same structure.

  Returns
  -------
  pyvinecopulib.RVineStructure
      The drawn structure.
  """
  controls = pv.FitControlsVinecop(
    family_set=[pv.BicopFamily.tll],
    tree_algorithm="random_weighted",
    seeds=[seed],
    num_threads=1,
  )
  data = to_numpy(u, dtype=np.float64)
  # pyvinecopulib's compiled `Vinecop` is untyped here.
  drawn = pv.Vinecop.from_data(data, controls=controls).structure
  return cast("pv.RVineStructure", drawn)


def regime_fingerprint() -> dict[str, list[float]]:
  """Each vine regime's tau on a fixed ``x`` grid, rounded for stable hashing."""
  x = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
  out: dict[str, list[float]] = {}
  for name in vine_scenarios.REGIMES:
    tau_of_x = scenarios.TAU_SCENARIOS[name].tau_of_x
    assert tau_of_x is not None
    out[name] = [round(float(t), 12) for t in tau_of_x(x)]
  return out


def grid_signature(grid: VineGridConfig, run: RunConfig) -> str:
  """Stable hash of everything that changes cell outputs (resume guard)."""
  payload = {
    "study": "vine",
    # The data-generating process, not merely the grid: a resumed cell must
    # have been drawn from the same truth. The regimes are fingerprinted by
    # value, so redefining one -- not only moving its bounds -- invalidates.
    "tau_bounds": [scenarios.TAU_LO, scenarios.TAU_HI],
    "regimes": regime_fingerprint(),
    "mu_slope": vine_scenarios.MU_SLOPE,
    # `dims`, `n` and `n_rep` are absent because they choose which cells run,
    # not what any cell computes: a cell is keyed by its own (d, n, rep), and
    # its truth and seeds derive from those alone. Leaving them out is what
    # lets a grid grow -- more repetitions, a new dimension -- and resume on
    # top of the cells it already has.
    "arms": sorted(grid.arms),
    "eval_x_n": grid.eval_x_n,
    "eval_m": grid.eval_m,
    "projection_grid_size": grid.projection_grid_size,
    # Only where it changes what a cell computes, so that a grid without the
    # mixture keeps the signature it had before the arm existed.
    **({"mixture_k": grid.mixture_k} if "mixture" in grid.arms else {}),
    # The chain's orders are distinct (`chain_orders`); a folder of chains
    # drawn before they were, where one may repeat, must not be resumed.
    **(
      {"chain_k": grid.chain_k, "chain_orders": "distinct"}
      if "chain" in grid.arms
      else {}
    ),
    # `batch_size` chunks the query set without changing any prediction's
    # context, so it changes throughput and not results.
    "base_seed": run.base_seed,
    "estimators": sorted(s.estimator_id for s in grid.estimators),
  }
  # Only where a GAM-vine arm runs, so every other grid keeps its signature.
  # An edited script is a new method, which a resume must refuse to mix in.
  if any(a in GAMVINE_ARMS for a in grid.arms):
    payload["gamvine_script_sha256"] = grid.gamvine_script_sha256
  encoded = json.dumps(payload, sort_keys=True).encode()
  return hashlib.sha256(encoded).hexdigest()


def score(pdf_true: torch.Tensor, pdf_hat: torch.Tensor) -> dict[str, float]:
  """Monte-Carlo ``KL`` / ``L1`` / ``ISE`` from points drawn under the truth."""
  p = pdf_true.to(torch.float64).clamp_min(_EPS)
  q = pdf_hat.to(torch.float64).clamp_min(_EPS)
  return {
    "KL": float((p.log() - q.log()).mean()),
    "L1": float((q / p - 1.0).abs().mean()),
    "ISE": float(((q - p).square() / p).mean()),
    "loglik_true": float(p.log().mean()),
    "loglik_hat": float(q.log().mean()),
  }


def _score_both(
  points: EvalPoints,
  k: int,
  pdf_hat: torch.Tensor,
  copula_pdf_hat: torch.Tensor,
) -> dict[str, float]:
  joint = score(points.pdf_true[k], pdf_hat)
  copula = score(points.copula_pdf_true[k], copula_pdf_hat)
  return {**joint, "KL_copula": copula["KL"], "L1_copula": copula["L1"]}


def _new_margins(
  controls: FitControlsRosenblattVinecop, d: int
) -> list[ConditionalMargin]:
  """One unfitted margin per variable, on the pairs' backend.

  The margins run the pairs' backend on the data scale, hence "identity";
  the estimator's own transform is for copula-scale responses.
  """
  return [
    create_backend(
      controls.backend,
      transform="identity",
      quantile_table_config=controls.quantile_table_config
      or QuantileTableConfig(),
      eps=controls.eps,
      device=controls.device,
      batch_size=controls.batch_size,
      backend_kwargs=controls.backend_kwargs,
    )
    for _ in range(d)
  ]


def _fit_npcc(
  arm: str,
  est: EstimatorSpec,
  structure: pv.RVineStructure | None,
  y: torch.Tensor,
  x: torch.Tensor,
  *,
  device: str | None,
  projection_grid_size: int,
  batch_size: int | None,
) -> tuple[RosenblattVinedist, RosenblattVinecop, float]:
  """Fit one NPCC arm; return the distribution, its copula, the selection time.

  With ``structure=None`` the margins are fitted first, as
  :meth:`RosenblattVinedist.fit` fits them, and :func:`select_structure`
  picks the structure from their probability integral transforms. The vine
  is then fitted on it, and the two parts are composed. The selection time
  is ``nan`` when a structure is given.
  """
  context: ConditioningContext[torch.Tensor] = (
    SimplifiedContext() if arm == "oracle" else NonSimplifiedContext()
  )
  controls = est.controls(
    FitControlsRosenblattVinecop,
    device=device,
    projection_grid_size=projection_grid_size,
    batch_size=batch_size,
  )
  margins = _new_margins(controls, y.shape[1])
  if structure is not None:
    vine = RosenblattVinecop(None, structure, device=device, context=context)
    dist = RosenblattVinedist(vine, margins).fit(y, controls, x=x)
    return dist, vine, math.nan
  # `RosenblattVinedist.fit` in two halves, with the selection between them:
  # each margin is fitted on its column with no margin controls, and the
  # copula on the transforms `copula_data` assembles from those margins.
  for j, margin in enumerate(margins):
    margin.fit(y[:, j], None, x=x)
  u = RosenblattVinedist.copula_data(margins, y, x=x)
  t0 = perf_counter()
  structure = select_structure(u)
  selection_time = perf_counter() - t0
  vine = RosenblattVinecop(None, structure, device=device, context=context)
  vine.fit(u, controls, x=x)
  return RosenblattVinedist(vine, margins), vine, selection_time


def _fit_margins(
  controls: FitControlsRosenblattVinecop, y: torch.Tensor, x: torch.Tensor
) -> tuple[list[ConditionalMargin], torch.Tensor]:
  """Fit the margins as :meth:`RosenblattVinedist.fit` does; return their PITs.

  Each margin is fitted on its column with no margin controls, and the copula
  data are the transforms ``copula_data`` assembles from them.
  """
  margins = _new_margins(controls, y.shape[1])
  for j, margin in enumerate(margins):
    margin.fit(y[:, j], None, x=x)
  return margins, RosenblattVinedist.copula_data(margins, y, x=x)


def _margin_terms(
  margins: list[ConditionalMargin], points: EvalPoints, x_eval: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
  """The test points' fitted transforms and each margin's log-density there.

  Returns
  -------
  layout : torch.Tensor, shape (m, d)
      The fitted margins' probability integral transforms of the test points.
  margin_logs : list of torch.Tensor
      Each margin's log-density at the test points, in variable order.
  """
  y_eval = torch.cat(points.y)
  with torch.inference_mode():
    layout = RosenblattVinedist.copula_data(margins, y_eval, x=x_eval)
    y_eval = y_eval.to(device=layout.device)
    margin_logs = [
      m.logpdf(y_eval[:, j], x=x_eval) for j, m in enumerate(margins)
    ]
  return layout, margin_logs


def _mixture_densities(
  est: EstimatorSpec,
  member_seeds: list[int],
  y: torch.Tensor,
  x: torch.Tensor,
  points: EvalPoints,
  x_eval: torch.Tensor,
  *,
  device: str | None,
  projection_grid_size: int,
  batch_size: int | None,
) -> tuple[
  torch.Tensor,
  list[torch.Tensor],
  torch.Tensor,
  list[pv.RVineStructure],
  dict[str, float],
]:
  """Fit the mixture's members and evaluate each at the test points.

  The margins are fitted once, as :meth:`RosenblattVinedist.fit` fits them.
  On the transforms they give, the first member's structure is selected as
  the ``selected`` arm's is, and each other member's is drawn by
  :func:`weighted_random_structure` from its seed in ``member_seeds``. Each
  member is evaluated as soon as it is fitted and then released, so only one
  member's pair copulas occupy the device at a time.

  Returns
  -------
  copula_logs : torch.Tensor, shape (k, m)
      Each member's copula log-density at the test points' fitted transforms.
  margin_logs : list of torch.Tensor
      Each margin's log-density at the test points, in variable order.
  copula_pdfs : torch.Tensor, shape (k, m)
      Each member's copula density at the test points' true transforms.
  structures : list of pyvinecopulib.RVineStructure
      Each member's structure.
  seconds : dict of str to float
      ``fit_time`` (margins and members, structures included),
      ``selection_time`` (the structures alone) and ``pdf_time``.
  """
  controls = est.controls(
    FitControlsRosenblattVinecop,
    device=device,
    projection_grid_size=projection_grid_size,
    batch_size=batch_size,
  )
  t0 = perf_counter()
  margins, u = _fit_margins(controls, y, x)
  t1 = perf_counter()
  structures = [select_structure(u)] + [
    weighted_random_structure(u, seed) for seed in member_seeds
  ]
  selection_time = perf_counter() - t1
  fit_time = perf_counter() - t0

  t0 = perf_counter()
  u_eval = torch.cat(points.u)
  layout, margin_logs = _margin_terms(margins, points, x_eval)
  pdf_time = perf_counter() - t0

  copula_logs: list[torch.Tensor] = []
  copula_pdfs: list[torch.Tensor] = []
  for structure in structures:
    t0 = perf_counter()
    vine = RosenblattVinecop(
      None, structure, device=device, context=NonSimplifiedContext()
    )
    vine.fit(u, controls, x=x)
    fit_time += perf_counter() - t0
    t0 = perf_counter()
    with torch.inference_mode():
      copula_logs.append(vine.logpdf(layout, x=x_eval))
      copula_pdfs.append(vine.pdf(u_eval, x=x_eval))
    pdf_time += perf_counter() - t0
    del vine
    release_gpu()
  return (
    torch.stack(copula_logs),
    margin_logs,
    torch.stack(copula_pdfs),
    structures,
    {
      "fit_time": fit_time,
      "selection_time": selection_time,
      "pdf_time": pdf_time,
    },
  )


def _chain_densities(
  est: EstimatorSpec,
  orders: list[list[int]],
  factor_seeds: list[list[int]],
  y: torch.Tensor,
  x: torch.Tensor,
  points: EvalPoints,
  x_eval: torch.Tensor,
  *,
  device: str | None,
  projection_grid_size: int,
  batch_size: int | None,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, dict[str, float]]:
  """Fit one chain-rule density per order and evaluate each at the test points.

  The margins are fitted once, as :func:`_mixture_densities` fits them, and
  every order's factors on the transforms they give, with ``x``. Each order
  is evaluated as soon as it is fitted and then released, so only one
  order's factors occupy the device at a time.

  Returns
  -------
  copula_logs : torch.Tensor, shape (k, m)
      Each order's copula log-density at the test points' fitted transforms.
  margin_logs : list of torch.Tensor
      Each margin's log-density at the test points, in variable order.
  copula_pdfs : torch.Tensor, shape (k, m)
      Each order's copula density at the test points' true transforms.
  seconds : dict of str to float
      ``fit_time`` (margins and every order) and ``pdf_time``.
  """
  controls = est.controls(
    FitControlsRosenblattVinecop,
    device=device,
    projection_grid_size=projection_grid_size,
    batch_size=batch_size,
  )
  t0 = perf_counter()
  margins, u = _fit_margins(controls, y, x)
  fit_time = perf_counter() - t0

  t0 = perf_counter()
  u_eval = torch.cat(points.u)
  layout, margin_logs = _margin_terms(margins, points, x_eval)
  pdf_time = perf_counter() - t0

  copula_logs: list[torch.Tensor] = []
  copula_pdfs: list[torch.Tensor] = []
  for order, seeds in zip(orders, factor_seeds, strict=True):
    t0 = perf_counter()
    density = ChainCopula(order, controls).fit(u, x=x, seeds=seeds)
    fit_time += perf_counter() - t0
    t0 = perf_counter()
    with torch.inference_mode():
      copula_logs.append(density.logpdf(layout, x=x_eval))
      copula_pdfs.append(density.pdf(u_eval, x=x_eval))
    pdf_time += perf_counter() - t0
    del density
    release_gpu()
  return (
    torch.stack(copula_logs),
    margin_logs,
    torch.stack(copula_pdfs),
    {"fit_time": fit_time, "pdf_time": pdf_time},
  )


def _metric_rows(
  base: Mapping[str, object],
  points: EvalPoints,
  preds: list[torch.Tensor],
  copula_preds: list[torch.Tensor],
) -> list[dict[str, Any]]:
  return [
    {
      **base,
      "x": float(x_val),
      **_score_both(points, k, preds[k], copula_preds[k]),
    }
    for k, x_val in enumerate(points.x_axis)
  ]


def _mixture_metric_rows(
  base: Mapping[str, object],
  points: EvalPoints,
  copula_logs: torch.Tensor,
  margin_logs: list[torch.Tensor],
  copula_pdfs: torch.Tensor,
  member_prefix: str,
) -> list[dict[str, Any]]:
  """A mixture's metric rows, then each member's, as arm ``<prefix><j>``.

  The mixture's joint density is the margins' times the members' mixed copula
  densities, and its copula density their mean. Beside each of its rows go
  its members' mean ``KL`` and ``KL_copula`` at the same ``x``, which
  Jensen's inequality bounds the mixture's by.
  """
  joint, joint_members = joint_log_densities(copula_logs, margin_logs)
  split = points.y[0].shape[0]
  rows = _metric_rows(
    base,
    points,
    list(joint.exp().cpu().split(split)),
    list(copula_pdfs.mean(dim=0).cpu().split(split)),
  )
  member_rows = [
    _metric_rows(
      {**base, "arm": f"{member_prefix}{j + 1}"},
      points,
      list(joint_members[j].exp().cpu().split(split)),
      list(copula_pdfs[j].cpu().split(split)),
    )
    for j in range(copula_logs.shape[0])
  ]
  for i, row in enumerate(rows):
    for name in ("KL", "KL_copula"):
      row[f"{name}_members_mean"] = float(
        np.mean([m[i][name] for m in member_rows])
      )
  return rows + [row for member in member_rows for row in member]


def summarize_one_cell(
  cell: VineCell,
  grid: VineGridConfig,
  *,
  base_seed: int,
  device: str | None,
) -> tuple[Rows, Rows]:
  """Fit every arm on one cell's data and return its metric and runtime rows."""
  truth = truth_for(base_seed, cell.d, cell.rep)
  y, _, x = training_sample(base_seed, cell)
  points = vine_scenarios.eval_points(
    truth,
    scenarios.conditional_x_axis(grid.eval_x_n),
    grid.eval_m,
    derive_seed(base_seed, "eval", cell.d, cell.rep),
  )
  structures: dict[str, pv.RVineStructure | None] = {
    "oracle": truth.structure,
    "random": random_structure_for(base_seed, cell.d, cell.rep),
    "selected": None,
  }
  cell_seed = derive_seed(base_seed, "cell", cell.d, cell.n, cell.rep)
  cell_base = {"d": cell.d, "n": cell.n, "rep": cell.rep, "seed": cell_seed}

  metric_rows: Rows = []
  runtime_rows: Rows = []

  def _record(
    base: Mapping[str, object],
    fit_time: float,
    pdf_time: float,
    **extra: object,
  ) -> None:
    memory = memory_usage()
    logger.info(
      "arm %s / %s [d=%d n=%d rep=%d]: fit %.1fs | GPU peak %.0f MiB | "
      "RSS %.0f MiB | MemAvailable %.0f MiB",
      base["arm"],
      base["label"],
      cell.d,
      cell.n,
      cell.rep,
      fit_time,
      memory["gpu_peak_reserved_mb"],
      memory["rss_mb"],
      memory["mem_available_mb"],
    )
    runtime_rows.append(
      {
        **base,
        "fit_time": fit_time,
        "pdf_time": pdf_time,
        "total_time": fit_time + pdf_time,
        **extra,
        **memory,
      }
    )

  for arm in (a for a in grid.arms if a in structures):
    for est in grid.estimators:
      base = {
        **cell_base,
        "arm": arm,
        "label": est.label,
        "estimator_id": est.estimator_id,
        "backend": est.backend,
        "transform": est.transform,
      }
      torch.manual_seed(derive_seed(cell_seed, arm, est.estimator_id))
      reset_gpu_peaks()
      t0 = perf_counter()
      dist, vine, selection_time = _fit_npcc(
        arm,
        est,
        structures[arm],
        y,
        x,
        device=device,
        projection_grid_size=grid.projection_grid_size,
        batch_size=grid.batch_size,
      )
      fit_time = perf_counter() - t0
      t0 = perf_counter()
      # One call per scale over every evaluation x at once: each call pays a
      # fixed inference cost per backend pass, so a call per x repays it.
      x_all = points.x_axis.repeat_interleave(grid.eval_m).reshape(-1, 1)
      with torch.inference_mode():
        preds = list(
          dist.pdf(torch.cat(points.y), x=x_all).cpu().split(grid.eval_m)
        )
        copula_preds = list(
          vine.pdf(torch.cat(points.u), x=x_all).cpu().split(grid.eval_m)
        )
      pdf_time = perf_counter() - t0
      metric_rows += _metric_rows(base, points, preds, copula_preds)
      _record(
        base,
        fit_time,
        pdf_time,
        structure=json.dumps(np.asarray(vine.structure.matrix).tolist()),
        selection_time=selection_time,
      )
      del dist, vine
      release_gpu()

  if "mixture" in grid.arms:
    member_seeds = [
      derive_seed(base_seed, "mixture", cell.d, cell.rep, j)
      for j in range(2, grid.mixture_k + 1)
    ]
    x_all = points.x_axis.repeat_interleave(grid.eval_m).reshape(-1, 1)
    for est in grid.estimators:
      base = {
        **cell_base,
        "arm": "mixture",
        "label": est.label,
        "estimator_id": est.estimator_id,
        "backend": est.backend,
        "transform": est.transform,
      }
      torch.manual_seed(derive_seed(cell_seed, "mixture", est.estimator_id))
      reset_gpu_peaks()
      copula_logs, margin_logs, copula_pdfs, members, seconds = (
        _mixture_densities(
          est,
          member_seeds,
          y,
          x,
          points,
          x_all,
          device=device,
          projection_grid_size=grid.projection_grid_size,
          batch_size=grid.batch_size,
        )
      )
      metric_rows += _mixture_metric_rows(
        base,
        points,
        copula_logs,
        margin_logs,
        copula_pdfs,
        "mixture-member-",
      )
      _record(
        base,
        seconds["fit_time"],
        seconds["pdf_time"],
        structure=json.dumps(np.asarray(members[0].matrix).tolist()),
        mixture_structures=json.dumps(
          [np.asarray(s.matrix).tolist() for s in members]
        ),
        selection_time=seconds["selection_time"],
      )
      del copula_logs, margin_logs, copula_pdfs
      release_gpu()

  if "chain" in grid.arms:
    orders = chain_orders(base_seed, cell.d, cell.rep, grid.chain_k)
    x_all = points.x_axis.repeat_interleave(grid.eval_m).reshape(-1, 1)
    for est in grid.estimators:
      base = {
        **cell_base,
        "arm": "chain",
        "label": est.label,
        "estimator_id": est.estimator_id,
        "backend": est.backend,
        "transform": est.transform,
      }
      factor_seeds = [
        chain_factor_seeds(cell_seed, est.estimator_id, j, cell.d)
        for j in range(1, len(orders) + 1)
      ]
      torch.manual_seed(derive_seed(cell_seed, "chain", est.estimator_id))
      reset_gpu_peaks()
      copula_logs, margin_logs, copula_pdfs, seconds = _chain_densities(
        est,
        orders,
        factor_seeds,
        y,
        x,
        points,
        x_all,
        device=device,
        projection_grid_size=grid.projection_grid_size,
        batch_size=grid.batch_size,
      )
      metric_rows += _mixture_metric_rows(
        base, points, copula_logs, margin_logs, copula_pdfs, "chain-m"
      )
      _record(
        base,
        seconds["fit_time"],
        seconds["pdf_time"],
        chain_orders=json.dumps(orders),
        chain_factor_seeds=json.dumps(factor_seeds),
      )
      del copula_logs, margin_logs, copula_pdfs
      release_gpu()

  if "tll" in grid.arms:
    base = {
      **cell_base,
      "arm": "tll",
      "label": "kde-tll",
      "estimator_id": "",
      "backend": "pyvinecopulib",
      "transform": "none",
    }
    t0 = perf_counter()
    tll = pv.Vinedist.from_data(
      y.numpy(),
      pv.FitControlsVinecop(family_set=[pv.BicopFamily.tll]),
      structure=truth.structure,
    )
    fit_time = perf_counter() - t0
    t0 = perf_counter()
    preds = [
      torch.from_numpy(np.asarray(tll.pdf(pts.numpy()))) for pts in points.y
    ]
    copula_preds = [
      torch.from_numpy(np.asarray(tll.vinecop.pdf(pts.numpy())))
      for pts in points.u
    ]
    pdf_time = perf_counter() - t0
    metric_rows += _metric_rows(base, points, preds, copula_preds)
    _record(base, fit_time, pdf_time)

  for arm in (a for a in grid.arms if a in GAMVINE_ARMS):
    base = {
      **cell_base,
      "arm": arm,
      "label": arm,
      "estimator_id": "",
      "backend": "gamCopula",
      "transform": "none",
    }
    metric_rows += _gamvine_rows(
      arm, base, cell, grid, truth, (y, x), points, _record
    )

  return metric_rows, runtime_rows


class _Record(Protocol):
  def __call__(
    self,
    base: Mapping[str, object],
    fit_time: float,
    pdf_time: float,
    **extra: object,
  ) -> None: ...


def _gamvine_rows(
  arm: str,
  base: Mapping[str, object],
  cell: VineCell,
  grid: VineGridConfig,
  truth: VineTruth,
  sample: tuple[torch.Tensor, torch.Tensor],
  points: EvalPoints,
  record: _Record,
) -> Rows:
  """Fit one GAM-vine arm with one script run; return its metric rows.

  The runtime row goes through ``record``. A script that fails on its sample
  leaves every metric ``nan`` and its message in the runtime row: the
  method's outcome, which the analysis counts, and not a reason to stop the
  other cells.
  """
  m = grid.eval_m
  x_all = points.x_axis.repeat_interleave(m).reshape(-1, 1)
  name = f"{arm} on {cell.key}"
  try:
    fit = _external_vine.fit_and_evaluate(
      grid.gamvine_script,
      sample,
      (torch.cat(points.y), torch.cat(points.u), x_all),
      arm=arm,
      seed=derive_seed(base["seed"], arm),
      name=name,
      structure=truth.structure if arm == "gamvine-oracle" else None,
    )
  except _external.ScriptFailedError as exc:
    logger.warning("arm %s [%s] failed: %s", arm, cell.key, exc)
    failed = [torch.full((m,), math.nan, dtype=torch.float64)] * len(
      points.x_axis
    )
    record(base, math.nan, math.nan, error=str(exc))
    return _metric_rows(base, points, failed, failed)
  if arm == "gamvine-oracle" and _external_vine.edges(
    fit.structure
  ) != _external_vine.edges(truth.structure):
    raise _external.ExchangeError(
      f"{name} must fit the true structure; got the matrix "
      f"{np.asarray(fit.structure.matrix).tolist()} for "
      f"{np.asarray(truth.structure.matrix).tolist()}"
    )
  timings = fit.timings
  record(
    base,
    timings["margin_time"] + timings["vine_time"],
    timings["joint_pdf_time"] + timings["copula_pdf_time"],
    structure=json.dumps(np.asarray(fit.structure.matrix).tolist()),
    selection_time=math.nan,
    pairs=json.dumps(fit.pairs),
    wall_time=fit.wall_time,
    error="",
    **timings,
  )
  return _metric_rows(
    base, points, list(fit.pdf.split(m)), list(fit.copula_pdf.split(m))
  )


def run_vine_study(
  grid: VineGridConfig, run: RunConfig, *, resume: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
  """Sweep the vine grid; return the metric and runtime tables and wall time.

  Each cell's rows are checkpointed to ``run.out/cells/<key>/`` as they
  finish. With ``resume=True``, cells already marked done are read back and
  skipped, after the grid signature is checked against the manifest.
  """
  logger.info(
    "Vine study: %d arms x %d estimators per cell",
    len(grid.arms),
    len(grid.estimators),
  )
  # A package missing from this machine would fail every GAM-vine fit, each
  # recorded as the method's failure; it must stop the run first.
  if any(a in GAMVINE_ARMS for a in grid.arms):
    _external.check_packages(grid.gamvine_script)
  (metrics, runtimes), wall = sweep(
    grid.cells(),
    lambda cell: summarize_one_cell(
      cell, grid, base_seed=run.base_seed, device=run.device
    ),
    tables=TABLES,
    run=run,
    signature=grid_signature(grid, run),
    study="vine",
    resume=resume,
  )

  sort_axes = ["d", "n", "arm", "label", "rep"]
  metric_df = sorted_frame(metrics, [*sort_axes, "x"])
  runtime_df = sorted_frame(runtimes, sort_axes)
  return metric_df, runtime_df, wall


def truths(grid: VineGridConfig, base_seed: int) -> list[dict[str, object]]:
  """Every ``(d, rep)`` truth, with the ``random`` arm's structure beside it."""
  out: list[dict[str, object]] = []
  for d in grid.dims:
    for rep in range(grid.n_rep):
      record = truth_for(base_seed, d, rep).to_dict()
      random_matrix = random_structure_for(base_seed, d, rep).matrix
      record.update(
        {"rep": rep, "random_matrix": np.asarray(random_matrix).tolist()}
      )
      out.append(record)
  return out


def aggregate_vine_outputs(
  metric_df: pd.DataFrame, runtime_df: pd.DataFrame
) -> dict[str, pd.DataFrame]:
  """Summaries over repetitions, per ``x`` and averaged over ``x``."""
  summary_by_x, summary_over_x = summarize(
    metric_df, METRIC_NAMES, _GROUP_AXES, x_axes=("x",)
  )
  runtime_summary = runtime_df.groupby(
    list(_GROUP_AXES), as_index=False, dropna=False
  ).agg(
    fit_time_mean=("fit_time", "mean"),
    fit_time_std=("fit_time", "std"),
    pdf_time_mean=("pdf_time", "mean"),
    total_time_mean=("total_time", "mean"),
  )
  return {
    "metrics_by_x": metric_df,
    "summary_by_x": summary_by_x,
    "summary_over_x": summary_over_x,
    "runtime": runtime_df,
    "runtime_summary": cast("pd.DataFrame", runtime_summary),
  }
