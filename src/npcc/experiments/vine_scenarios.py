"""Ground-truth vine distributions for the vine simulation study.

A truth is an R-vine structure drawn uniformly over all structures on ``d``
variables, with one ``(family, tau regime)`` per edge. Each edge's Kendall's
tau follows a bivariate-study regime ``tau(x)`` in a scalar covariate ``x``,
rescaled into a band whose upper end decays with the tree level: tree ``t``
(1-indexed) lives in ``[TAU_LO, TAU_HI ** t]``, so tree 1 reproduces the
bivariate regime exactly and deeper trees are progressively weaker.

The copula is simplified in the vine's own conditioning sets -- no pair
depends on ``u_D`` -- so at any fixed ``x`` it is an ordinary parametric
:class:`pyvinecopulib.Vinecop`, which is what supplies sampling and the exact
density.

**Margins.** Variable ``j`` is ``N(mu_j(x), 1)`` with
``mu_j(x) = a_j (x - 1/2)`` and a slope ``a_j`` drawn uniformly in
``[-MU_SLOPE, MU_SLOPE]``, so a covariate-blind estimator is wrong on the
margins as well as on the copula. The joint density is
``f(y | x) = c(u | x) prod_j phi(y_j - mu_j(x))`` with
``u_j = Phi(y_j - mu_j(x))``, and normal margins keep it bounded where the
copula density has corner peaks, which is what makes an integrated squared
error finite to estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyvinecopulib as pv
import torch

from npcc.experiments import scenarios
from npcc.experiments.scenarios import TAU_HI, TAU_LO, X_MAX, X_MIN

# The bivariate regimes an edge may draw, in `TAU_SCENARIOS` insertion order;
# that order is part of the truth's seed, as `FAMILIES` order is.
REGIMES: tuple[str, ...] = tuple(
  name for name, spec in scenarios.TAU_SCENARIOS.items() if spec.conditional
)

# Bound on each margin's location slope in x: at 2, a margin's mean moves by
# up to one standard deviation either side over the unit interval.
MU_SLOPE: float = 2.0

_EPS: float = 1e-9


@dataclass(frozen=True)
class EdgeSpec:
  """One true pair copula: its family, tau regime and 1-indexed tree."""

  family: str
  regime: str
  tree: int


@dataclass(frozen=True)
class VineTruth:
  """A true vine distribution: structure, edges and margin slopes.

  ``edges`` is laid out ``[tree][edge]``, matching ``pair_copulas`` in
  :meth:`pyvinecopulib.Vinecop.from_structure`; ``slopes`` holds one ``a_j``
  per variable, in variable order.
  """

  structure: pv.RVineStructure
  edges: tuple[tuple[EdgeSpec, ...], ...]
  slopes: tuple[float, ...]

  @property
  def d(self) -> int:
    return int(self.structure.dim)

  def to_dict(self) -> dict[str, object]:
    """A JSON-serializable record of the truth, for the run's echo."""
    return {
      "d": self.d,
      "matrix": np.asarray(self.structure.matrix).tolist(),
      "edges": [
        [{"family": e.family, "regime": e.regime, "tree": e.tree} for e in row]
        for row in self.edges
      ],
      "slopes": list(self.slopes),
    }


def tau_upper(tree: int) -> float:
  """Upper end of tree ``tree``'s tau band, ``TAU_HI ** tree`` (1-indexed)."""
  if tree < 1:
    raise ValueError(f"tree must be >= 1 (1-indexed); got {tree}")
  return float(TAU_HI**tree)


def edge_tau(regime: str, tree: int, x: torch.Tensor) -> torch.Tensor:
  """Kendall's tau of a tree-``tree`` edge following ``regime`` at ``x``.

  The bivariate ``tau(x)`` lives in ``[TAU_LO, TAU_HI]``; it is mapped
  affinely onto ``[TAU_LO, TAU_HI ** tree]``, which keeps its shape and is
  the identity on tree 1.
  """
  spec = scenarios.TAU_SCENARIOS[regime]
  if spec.tau_of_x is None:
    raise ValueError(f"regime must be a conditional scenario; got {regime!r}")
  base = spec.tau_of_x(x)
  scale = (tau_upper(tree) - TAU_LO) / (TAU_HI - TAU_LO)
  return TAU_LO + scale * (base - TAU_LO)


def draw_truth(d: int, seed: int) -> VineTruth:
  """Draw a structure, one ``(family, regime)`` per edge and the margin slopes."""
  if d < 2:
    raise ValueError(f"d must be >= 2; got {d}")
  generator = torch.Generator().manual_seed(seed)
  structure_seed = int(torch.randint(1, 2**31 - 1, (1,), generator=generator))
  structure = pv.RVineStructure.sample(d, seeds=[structure_seed])
  families = list(scenarios.FAMILIES)
  edges: list[tuple[EdgeSpec, ...]] = []
  for tree in range(d - 1):
    n_edges = d - tree - 1
    fam_idx = torch.randint(len(families), (n_edges,), generator=generator)
    reg_idx = torch.randint(len(REGIMES), (n_edges,), generator=generator)
    edges.append(
      tuple(
        EdgeSpec(families[int(f)], REGIMES[int(r)], tree + 1)
        for f, r in zip(fam_idx, reg_idx, strict=True)
      )
    )
  unit = torch.rand(d, generator=generator, dtype=torch.float64)
  slopes = tuple(float(a) for a in MU_SLOPE * (2.0 * unit - 1.0))
  return VineTruth(structure=structure, edges=tuple(edges), slopes=slopes)


def margin_mean(truth: VineTruth, x: torch.Tensor) -> torch.Tensor:
  """``mu(x)`` of shape ``(n, d)`` for covariate values ``x`` of shape ``(n,)``."""
  slopes = torch.tensor(truth.slopes, dtype=torch.float64)
  return (x.reshape(-1, 1).to(torch.float64) - 0.5) * slopes


def to_data_scale(
  truth: VineTruth, u: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
  """Map copula-scale ``u`` at covariates ``x`` to the data scale ``y``."""
  z: torch.Tensor = torch.special.ndtri(u)
  return margin_mean(truth, x) + z


def margin_log_pdf(
  truth: VineTruth, y: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
  """``sum_j log phi(y_j - mu_j(x))``, one value per row."""
  z = y - margin_mean(truth, x)
  return (-0.5 * z.square() - 0.5 * math.log(2.0 * math.pi)).sum(dim=1)


def true_vine(truth: VineTruth, x_val: float) -> pv.Vinecop:
  """The true vine at one covariate value, as a parametric ``Vinecop``."""
  x = torch.tensor([float(x_val)], dtype=torch.float64)
  # `scenarios._bicop` is the bivariate study's tau-to-parameter map; sharing
  # it keeps tree 1 of a vine and a bivariate cell the same copula.
  pair_copulas = [
    [
      scenarios._bicop(
        scenarios.FAMILIES[e.family], float(edge_tau(e.regime, e.tree, x)[0])
      )
      for e in row
    ]
    for row in truth.edges
  ]
  return pv.Vinecop.from_structure(
    structure=truth.structure, pair_copulas=pair_copulas
  )


def sample(
  truth: VineTruth, n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Draw ``n`` training rows from the true conditional distribution.

  ``x`` is a deterministic ``linspace`` over ``[X_MIN, X_MAX]``, as in the
  bivariate study; each row's ``u`` is the inverse Rosenblatt transform of
  uniform noise under the vine at that row's own ``x``, and ``y`` is ``u``
  on the data scale. Returns ``y`` and ``u`` of shape ``(n, d)`` and ``x`` of
  shape ``(n, 1)``.
  """
  generator = torch.Generator().manual_seed(seed)
  x = torch.linspace(X_MIN, X_MAX, n, dtype=torch.float64)
  w = torch.rand((n, truth.d), generator=generator, dtype=torch.float64)
  w = w.clamp(_EPS, 1.0 - _EPS).numpy()
  # One vine per row because every edge's tau is continuous in x and
  # pyvinecopulib does not vectorize over row-specific parameters. That is
  # O(n d^2) Bicop builds, negligible next to fitting d(d-1)/2 NPCC pairs.
  u = np.empty_like(w)
  for i in range(n):
    u[i] = true_vine(truth, float(x[i])).inverse_rosenblatt(w[i : i + 1])[0]
  u_t = torch.from_numpy(u).clamp(_EPS, 1.0 - _EPS)
  return to_data_scale(truth, u_t, x), u_t, x.reshape(-1, 1)


@dataclass(frozen=True)
class EvalPoints:
  """Test points drawn from the truth at each evaluation ``x``.

  ``y[k]`` has shape ``(m, d)`` and is drawn from the true distribution at
  ``x_axis[k]``, and ``u[k]`` holds the same points on the copula scale, the
  true probability integral transforms. ``pdf_true[k]`` is the exact joint
  density at ``y[k]`` and ``copula_pdf_true[k]`` the exact copula density at
  ``u[k]``.
  """

  x_axis: torch.Tensor
  y: tuple[torch.Tensor, ...]
  u: tuple[torch.Tensor, ...]
  pdf_true: tuple[torch.Tensor, ...]
  copula_pdf_true: tuple[torch.Tensor, ...]


def eval_points(
  truth: VineTruth, x_axis: torch.Tensor, m: int, seed: int
) -> EvalPoints:
  """Draw ``m`` points per ``x`` from the truth and record its density there."""
  generator = torch.Generator().manual_seed(seed)
  ys: list[torch.Tensor] = []
  us: list[torch.Tensor] = []
  pdfs: list[torch.Tensor] = []
  cop_pdfs: list[torch.Tensor] = []
  for x_val in x_axis:
    vine = true_vine(truth, float(x_val))
    draw_seeds = torch.randint(1, 2**31 - 1, (3,), generator=generator)
    u = np.clip(
      vine.sample(m, seeds=[int(s) for s in draw_seeds]), _EPS, 1.0 - _EPS
    )
    u_t = torch.from_numpy(u)
    x_rows = torch.full((m,), float(x_val), dtype=torch.float64)
    y = to_data_scale(truth, u_t, x_rows)
    cop_pdf = torch.from_numpy(np.asarray(vine.pdf(u)))
    ys.append(y)
    us.append(u_t)
    cop_pdfs.append(cop_pdf)
    pdfs.append(cop_pdf * margin_log_pdf(truth, y, x_rows).exp())
  return EvalPoints(
    x_axis=x_axis.to(dtype=torch.float64),
    y=tuple(ys),
    u=tuple(us),
    pdf_true=tuple(pdfs),
    copula_pdf_true=tuple(cop_pdfs),
  )
