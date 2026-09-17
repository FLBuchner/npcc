"""Rosenblatt conditional bivariate copula.

**Approach.** A bivariate copula density factorizes through the Rosenblatt
construction. Conditioning on covariates ``X`` and using the uniform-margin
property of the copula scale ``U``::

    c(u, v | x) = f_{V | U, X}(v | u, x).

So estimating a *conditional bivariate copula density* reduces to estimating a
*univariate conditional density*, which is what a
:class:`~npcc.core.margin.ConditionalMargin` backend provides. The features
fed to the inner regressor are::

    W = [u, x]    (when predicting V | U, X)

so the inner backend sees the conditioning copula score and the covariates
side by side.

**Pluggable backend.** The inner conditional-density estimator is any
registered backend (see :mod:`npcc.core.registry`), named in the fit controls
as :attr:`~npcc.core.controls.FitControlsRosenblattBicop.backend`:
``"tabpfn-criterion"`` (the default) reads TabPFN's native binned head, which
is the fastest read-out; ``"tabpfn-quantiles"`` inverts TabPFN's quantile
output; the rest are behind extras. :class:`RosenblattBicop` reads only the
backend's ``fit`` / ``pdf`` / ``cdf`` / ``icdf`` / ``pdf_grid`` / ``cdf_grid``
surface, so it is otherwise backend-agnostic.

**Symmetric averaging.** A single Rosenblatt direction is ordering-dependent:
it satisfies ``int c(u, v | x) dv = 1`` by construction but generally not
``int c(u, v | x) du = 1``. To reduce that directional bias the estimator
always fits both directions and averages them::

    c(u, v | x) =
        0.5 * f_{V | U, X}(v | u, x)
      + 0.5 * f_{U | V, X}(u | v, x).

Averaging still does not impose exact uniform copula margins. Where those are
required, the optional Sinkhorn projection does, on a uniform copula-scale grid
of ``projection_grid_size`` points per axis: :meth:`RosenblattBicop.pdf_grid`
applies it to the evaluated grid directly, while pointwise
:meth:`RosenblattBicop.pdf` computes the correction on the internal projection
grid and interpolates it back to the queried points.

**Plotting.** :class:`RosenblattBicop` is a ``BicopBase``, so it inherits
``plot`` and needs no adapter. The inherited implementation manufactures a
NumPy evaluation grid, which ``_prep`` brings onto this estimator's dtype and
device before the Cartesian-grid fast path evaluates it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from typing import Self

import torch
from pyvinecopulib.core import BicopBase, ControlsLike
from pyvinecopulib.core.extend import (
  covariate_row,
  prepare_covariates,
  to_numpy,
)

from npcc.core._interp import interp
from npcc.core._placement import TensorPlacement
from npcc.core._trim import check_uv
from npcc.core.controls import FitControlsRosenblattBicop
from npcc.core.margin import ConditionalMargin
from npcc.core.margin_quantile_table import QuantileTableConfig
from npcc.core.registry import create_backend


def _reject_discrete(var_types: Sequence[str] | None) -> None:
  """Refuse a variable-type declaration this estimator cannot honor.

  A pair declared discrete reads the four-column layout and answers with
  difference quotients of its distribution function. This estimator's ``cdf``
  is a trapezoidal integration of an inner conditional density, so differencing
  it would return a plausible number from a model that estimates no atom at
  all -- which is why the refusal is here rather than left to the
  ``_cdf_raw``-is-missing guard upstream applies to a pair with no ``cdf``.

  Parameters
  ----------
  var_types : sequence of str, or None
      The two variable types, ``"c"`` or ``"d"``; ``None`` means continuous.

  Returns
  -------
  None

  Raises
  ------
  ValueError
      If either type is discrete.
  """
  if var_types is None or all(t == "c" for t in var_types):
    return

  raise ValueError(
    "RosenblattBicop must model continuous pairs; got "
    f"var_types={list(var_types)}. Its inner backends estimate a conditional "
    "density rather than an atom's probability, so put a Bicop or a "
    "TorchTllBicop on a discrete edge."
  )


def _bicop_controls(
  controls: ControlsLike | None,
) -> FitControlsRosenblattBicop:
  """Coerce a ``ControlsLike`` into this estimator's own controls.

  What a consumer is handed is the ``ControlsLike`` contract -- an object with
  a ``to_dict`` -- so :meth:`RosenblattBicop.fit` may not declare less than
  that, and this is where it narrows. A
  :class:`~npcc.core.controls.FitControlsRosenblattVinecop` passes straight
  through, a vine's controls being valid pair controls.

  Anything else is read through ``to_dict`` and rebuilt, so a foreign controls
  object carrying exactly these settings works. One carrying a setting this
  estimator can neither honor nor delegate is **refused** rather than dropped,
  which is what ``ControlsLike`` asks of a consumer.

  Parameters
  ----------
  controls : ControlsLike, or None
      The caller's fit configuration, or ``None`` for the defaults.

  Returns
  -------
  FitControlsRosenblattBicop
      Controls this estimator can read field by field.

  Raises
  ------
  TypeError
      If ``controls`` is an array -- the argument-order mistake -- or carries
      no ``to_dict``.
  ValueError
      If it carries a setting this estimator cannot honor.
  """
  if controls is None:
    return FitControlsRosenblattBicop()

  if isinstance(controls, FitControlsRosenblattBicop):
    return controls

  if hasattr(controls, "shape"):
    raise TypeError(
      "RosenblattBicop received an array where `controls` goes. The order is "
      "(observations, controls) with everything else keyword-only, so "
      "covariates are passed as `x=`: fit(u, controls, x=covariates)."
    )

  to_dict = getattr(controls, "to_dict", None)
  if to_dict is None:
    raise TypeError(
      "controls must be a FitControlsRosenblattBicop, or a ControlsLike -- an "
      f"object with `to_dict`; got {type(controls).__name__}."
    )

  settings = dict(to_dict())
  unknown = sorted(
    set(settings) - {f.name for f in fields(FitControlsRosenblattBicop)}
  )

  if unknown:
    plural = "them" if len(unknown) > 1 else "it"
    raise ValueError(
      f"RosenblattBicop cannot honor {', '.join(unknown)} from "
      f"{type(controls).__name__}, and will not drop {plural} silently; pass "
      "a FitControlsRosenblattBicop instead."
    )

  return FitControlsRosenblattBicop(**settings)


def _sinkhorn_project(
  density: torch.Tensor,
  wu: torch.Tensor,
  wv: torch.Tensor,
  n_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute Sinkhorn row and column scaling factors in log domain.

  Performs iterative proportional fitting (IPF) to project the density
  matrix onto the space of densities with marginal constraints
  ``int c(u, v) dv = 1`` and ``int c(u, v) du = 1`` under the trapezoidal
  rule with weights ``wu`` and ``wv``.

  The output scalings ``r`` and ``s`` satisfy:
  - Row constraint: ``(density * s[j] * wv[j]).sum() ~ 1 / wu[i]`` for row i.
  - Col constraint: ``(density * r[i] * wu[i]).sum() ~ 1 / wv[j]`` for col j.

  Parameters
  ----------
  density
      Shape ``(m, n)`` — raw density matrix at grid points.
  wu
      Shape ``(m,)`` — trapezoidal weights for u-axis (row dimension).
  wv
      Shape ``(n,)`` — trapezoidal weights for v-axis (column dimension).
  n_iters
      Number of alternating normalization iterations.

  Returns
  -------
  r : torch.Tensor
      Shape ``(m,)`` — row scalings.
  s : torch.Tensor
      Shape ``(n,)`` — column scalings.
  """
  if n_iters <= 0:
    raise ValueError("n_iters must be positive.")

  m, n = density.shape
  tiny = torch.finfo(density.dtype).tiny

  # Work in log space to avoid underflow/overflow in repeated updates.
  log_density = torch.clamp(density, min=tiny).log()
  log_wu = torch.clamp(wu, min=tiny).log()
  log_wv = torch.clamp(wv, min=tiny).log()

  log_r = torch.zeros(m, dtype=density.dtype, device=density.device)
  log_s = torch.zeros(n, dtype=density.dtype, device=density.device)

  for _ in range(n_iters):
    # Row normalization: enforce sum_j density_ij * s_j * wv_j = 1.
    log_row_sums = torch.logsumexp(
      log_density + log_s[None, :] + log_wv[None, :],
      dim=1,
    )
    log_r = -log_row_sums

    # Column normalization: enforce sum_i density_ij * r_i * wu_i = 1.
    log_col_sums = torch.logsumexp(
      log_density + log_r[:, None] + log_wu[:, None],
      dim=0,
    )
    log_s = -log_col_sums

  r = torch.exp(log_r)
  s = torch.exp(log_s)

  return r, s


class RosenblattBicop(TensorPlacement, BicopBase[torch.Tensor]):
  """Rosenblatt conditional bivariate copula estimator.

  Parameters
  ----------
  controls : ControlsLike, or None, optional
      Backend and numerical configuration; see
      :class:`~npcc.core.controls.FitControlsRosenblattBicop`. ``None`` uses
      its defaults, which is what lets the inherited
      :meth:`~pyvinecopulib.core.BicopBase.from_data` construct one pair per
      vine edge with no arguments.

  Notes
  -----
  - The estimator fits both Rosenblatt directions and averages them to
    reduce directional bias (see the module docstring).
  - Public numerical methods accept and return torch tensors. Inputs are
    brought onto this estimator's dtype and device by ``_prep``, which is what
    lets the inherited plotting implementation hand it a NumPy grid.
  """

  # `BicopBase` defaults this to `False`, and `pair_eval` reads it *before*
  # the call -- so leaving the default makes every conditional evaluation
  # raise rather than reaching the leaves, which all declare `x`.
  supports_covariates: bool = True

  def __init__(self, controls: ControlsLike | None = None) -> None:
    self._apply_controls(_bicop_controls(controls))

  def _apply_controls(
    self,
    controls: FitControlsRosenblattBicop,
  ) -> None:
    """Apply fit controls and create fresh conditional estimators."""
    self.backend = controls.backend
    # `__post_init__` filled both in; the fallbacks narrow away the `None`
    # the declared types still carry, since that is what a caller may pass.
    self.quantile_table_config = (
      controls.quantile_table_config or QuantileTableConfig()
    )
    self.eps = controls.eps
    self.transform = controls.transform
    self._set_placement(controls.device)

    if controls.batch_size is None:
      self.batch_size = 2000 if self._device.type == "cuda" else 400
    else:
      self.batch_size = controls.batch_size

    self.backend_kwargs = dict(controls.backend_kwargs or {})
    self.sinkhorn_iters = controls.sinkhorn_iters
    self.projection_grid_size = controls.projection_grid_size
    self.cdf_n_int = controls.cdf_n_int

    self.v_given_ux_: ConditionalMargin = self._make_distribution()
    self.u_given_vx_: ConditionalMargin = self._make_distribution()

    # Grid borders (cached after fit)
    self._v_grid_borders_: torch.Tensor | None = None
    self._u_grid_borders_: torch.Tensor | None = None

  def _make_distribution(self) -> ConditionalMargin:
    """Construct one conditional estimator from the active controls."""
    return create_backend(
      self.backend,
      transform=self.transform,
      quantile_table_config=self.quantile_table_config,
      eps=self.eps,
      device=self._device,
      batch_size=self.batch_size,
      backend_kwargs=self.backend_kwargs,
    )

  def _get_grid_borders(self) -> None:
    """Cache the 1-D uniform projection grid used for Sinkhorn projection.

    The projection uses a uniform grid of ``projection_grid_size`` points
    on the copula scale ``(0, 1)`` for every backend; the inner backends'
    ``pdf_grid`` fast path evaluates the density there in a single
    forward pass per grid row.
    """
    eps = self.eps
    borders = torch.linspace(
      eps,
      1 - eps,
      steps=self.projection_grid_size,
      dtype=torch.float64,
      device=self._device,
    )
    self._v_grid_borders_ = borders
    self._u_grid_borders_ = borders

  def _features(
    self, first_coord: torch.Tensor, x: torch.Tensor
  ) -> torch.Tensor:
    """Build the inner regressor's feature matrix ``[first_coord | x]``."""
    return torch.column_stack([first_coord, x])

  def _default_x(self, n: int) -> torch.Tensor:
    """Empty covariate matrix used when ``x`` is omitted."""
    return torch.empty((n, 0), dtype=torch.float64, device=self._device)

  def _prep_args(self, u: torch.Tensor) -> torch.Tensor:
    """Place ``u``, check its layout, and bring it into the open unit square.

    Overrides the inherited hook, which every evaluation member reaches
    through ``BicopBase._atoms``, so this is the single site the domain step
    runs at. Only the *domain* step departs: placement and layout stay
    upstream's ``_prep`` and ``_layout``, so this estimator refuses a shape
    with the library's own message. The clamp is the half that forces the
    override: upstream clamps at the working
    precision, about ``1e-10`` in ``float64``, where this estimator clamps at
    the caller's :attr:`eps`. The inner backends are fitted on ``logit(u)``,
    where those two are some ten units of feature space apart, so inheriting
    the narrower clamp would quietly move every near-boundary answer.

    The rejection is the other half; :mod:`npcc.core._trim` says why a value
    at ``{0, 1}`` is a defect here rather than a rounding artifact. Note what
    it does *not* reach: a vine clamps at its own entry points before the
    cascade builds anything, so a pair inside one never sees a boundary value
    whatever this does.

    Parameters
    ----------
    u : torch.Tensor, shape (n, 2)
        Pair pseudo-observations.

    Returns
    -------
    torch.Tensor, shape (n, 2)
        Placed, checked and clamped strictly inside the unit square.
    """
    ua = self._layout(self._prep(u))

    return torch.column_stack(check_uv(ua[:, 0], ua[:, 1], self.eps))

  def _covariates(
    self,
    x: torch.Tensor | None,
    n: int,
  ) -> torch.Tensor:
    """Place covariates, check the row alignment, and name the empty case.

    Placement and layout only, never the domain step: covariates are arbitrary
    reals rather than copula arguments, so they are brought onto this
    estimator's dtype and device but never clamped. That is the split
    :func:`pyvinecopulib.core.extend.prepare_covariates` draws, and whose
    row-alignment check this delegates to.

    What it adds over that function is the empty default. Upstream answers
    ``None`` for an absent ``x``; the inner regressors take a feature matrix,
    so an unconditional pair needs the ``(n, 0)`` block :meth:`_features`
    concatenates against.

    The layout is upstream's and is not widened here: ``(n,)`` is refused,
    because it says nothing about which axis is which. Reshaping a single
    covariate to ``(n, 1)`` is the producer's to do -- it computed ``n`` and so
    can state which axis is which, where this method knows only that it
    received a covariate.
    """
    if x is None:
      return self._default_x(n)

    x_t = self._prep(x)

    # For the layout and row-alignment check, and its message. `place`
    # short-circuits on a tensor `_prep` already placed, so this returns
    # `x_t` itself and no gradient is severed.
    prepare_covariates(self, x_t, n)

    return x_t

  def _prepare_x_row(self, x_row: torch.Tensor | None) -> torch.Tensor:
    """Place the single covariate row a grid query shares, and shape it.

    ``(p,)`` and ``(1, p)`` both mean one row, which is why
    :func:`pyvinecopulib.core.extend.covariate_row` accepts either where
    ``prepare_covariates`` refuses a one-dimensional ``x``: a single row is
    unambiguous, a row-aligned block of them is not. That function shapes but
    does not place, so ``_prep`` runs first.
    """
    if x_row is None:
      return self._default_x(1)

    return covariate_row(self._prep(x_row), name="x_row")

  def _prepare_grid_inputs(
    self,
    u_grid: torch.Tensor,
    v_grid: torch.Tensor,
    x_row: torch.Tensor | None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place and validate copula grids and an optional shared covariate row."""
    u = self._prep(u_grid).reshape(-1)
    v = self._prep(v_grid).reshape(-1)

    if torch.any((u <= 0.0) | (u >= 1.0)) or torch.any((v <= 0.0) | (v >= 1.0)):
      raise ValueError("u_grid and v_grid must lie strictly inside (0, 1).")

    eps = self.eps
    u = torch.clamp(u, eps, 1.0 - eps)
    v = torch.clamp(v, eps, 1.0 - eps)

    return u, v, self._prepare_x_row(x_row)

  @staticmethod
  def _trapezoidal_weights(grid: torch.Tensor) -> torch.Tensor:
    """Compute trapezoidal rule weights for a sorted 1-D grid.

    For a grid ``g`` of length ``m``, the weight at position ``i`` is the
    average distance to neighbors:

        w[i] = (g[min(i+1, m-1)] - g[max(i-1, 0)]) / 2

    This is the standard weight in the composite trapezoidal rule.
    """
    m = grid.shape[0]
    if m == 1:
      return torch.ones(1, dtype=grid.dtype, device=grid.device)

    weights = torch.zeros(m, dtype=grid.dtype, device=grid.device)
    weights[0] = (grid[1] - grid[0]) / 2.0
    weights[-1] = (grid[-1] - grid[-2]) / 2.0
    if m > 2:
      weights[1:-1] = (grid[2:] - grid[:-2]) / 2.0

    return weights

  def fit(
    self,
    u: torch.Tensor,
    /,
    controls: ControlsLike | None = None,
    *,
    var_types: list[str] | None = None,
    x: torch.Tensor | None = None,
  ) -> Self:
    """Fit both Rosenblatt directions.

    Parameters
    ----------
    u
      Continuous bivariate pseudo-observations with shape ``(n, 2)``.
    controls
      Backend and numerical configuration. When omitted, the configuration
      currently stored on the object is retained.
    var_types
      Variable types supplied by pyvinecopulib. Refused unless both are
      continuous -- see :func:`_reject_discrete`.
    x
      Optional external covariates with shape ``(n, p)``.
    """
    # Before the fit rather than after it: the vine's engines declare an
    # edge's types on the pair only once it is fitted, so this is the first
    # place an atom is visible, and refusing here costs no backend call.
    _reject_discrete(var_types)

    if controls is not None:
      self._apply_controls(_bicop_controls(controls))

    # `_apply_controls` may just have changed `eps`, which `_prep_args` reads.
    uv_t = self._prep_args(u)
    u_t, v_t = uv_t[:, 0], uv_t[:, 1]
    x_t = self._covariates(x, uv_t.shape[0])

    self.v_given_ux_.fit(v_t, x=self._features(u_t, x_t))
    self.u_given_vx_.fit(u_t, x=self._features(v_t, x_t))

    # Cache grid borders for Sinkhorn projection (if enabled)
    if self.sinkhorn_iters is not None:
      self._get_grid_borders()

    return self

  def with_var_types(self, var_types: Sequence[str] = ("c", "c")) -> Self:
    """The same pair copula under different variable types.

    Overridden to refuse a discrete declaration, which is the door one arrives
    through: the vine's fit engines call this on a pair whose edge has an
    atom. It refuses only when a type *is* discrete -- the all-continuous call
    is the one ``continuous_of`` makes on every pair in the inverse-Rosenblatt
    cascade and in the density plot, and it must keep working.

    Parameters
    ----------
    var_types : sequence of str, default=("c", "c")
        The two types, ``"c"`` or ``"d"``.

    Returns
    -------
    RosenblattBicop
        This pair copula, the all-continuous declaration being a no-op.

    Raises
    ------
    ValueError
        If either type is discrete.
    """
    _reject_discrete(var_types)
    return super().with_var_types(var_types)

  def _pdf_raw(
    self,
    u: torch.Tensor,
    *,
    x: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Conditional copula density ``c(u_i, v_i | x_i)``.

    The continuous leaf ``BicopBase.pdf`` dispatches to, so ``u`` arrives
    placed, checked and clamped and this must not prepare it again. The
    optional Sinkhorn projection runs here rather than in a public override,
    which is what keeps it on the paths that reach the pair through the base
    -- ``loglik``, ``plot``, and every vine cascade.

    Parameters
    ----------
    u : torch.Tensor, shape (n, 2)
        Pair pseudo-observations in the open unit square.
    x : torch.Tensor, shape (n, p), or None, optional
        Exogenous covariates, one row per observation.

    Returns
    -------
    torch.Tensor, shape (n,)
        Density values.
    """
    u_t, v_t = u[:, 0], u[:, 1]
    x_t = self._covariates(x, u.shape[0])

    with torch.inference_mode():
      c_unprojected = self._unprojected_pdf(
        u_t, v_t, x_t, batch_size=self.batch_size
      )

      if self.sinkhorn_iters is None:
        return c_unprojected

      return self._project_points_by_x(
        c_unprojected,
        u_t,
        v_t,
        x_t,
        batch_size=self.batch_size,
        sinkhorn_iters=self.sinkhorn_iters,
      )

  def _project_points_by_x(
    self,
    c_unprojected: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    *,
    batch_size: int,
    sinkhorn_iters: int,
  ) -> torch.Tensor:
    if self._u_grid_borders_ is None or self._v_grid_borders_ is None:
      self._get_grid_borders()

    assert self._u_grid_borders_ is not None
    assert self._v_grid_borders_ is not None

    u_grid = self._u_grid_borders_
    v_grid = self._v_grid_borders_

    wu = self._trapezoidal_weights(u_grid)
    wv = self._trapezoidal_weights(v_grid)

    if x.shape[1] == 0:
      x_unique = x[:1]
      x_inverse = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    else:
      x_unique, x_inverse = torch.unique(x, dim=0, return_inverse=True)

    # All unique-x density grids in a few batched forward passes.
    density_all = self._unprojected_pdf_grids_by_x(
      u_grid, v_grid, x_unique, batch_size=batch_size
    )

    out = torch.empty_like(c_unprojected)
    for x_idx in range(x_unique.shape[0]):
      mask = x_inverse == x_idx
      if not torch.any(mask):
        continue

      # Sinkhorn IPF stays per-x (cheap, no forward pass).
      r, s = _sinkhorn_project(density_all[:, x_idx, :], wu, wv, sinkhorn_iters)

      r_interp = interp(u[mask], u_grid, r)
      s_interp = interp(v[mask], v_grid, s)

      out[mask] = c_unprojected[mask] * r_interp * s_interp

    return out

  def _project_grid(
    self,
    c_grid_unprojected: torch.Tensor,
    u_grid: torch.Tensor,
    v_grid: torch.Tensor,
    *,
    sinkhorn_iters: int,
  ) -> torch.Tensor:
    wu = self._trapezoidal_weights(u_grid)
    wv = self._trapezoidal_weights(v_grid)
    r, s = _sinkhorn_project(c_grid_unprojected, wu, wv, sinkhorn_iters)
    return r[:, None] * c_grid_unprojected * s[None, :]

  def _unprojected_pdf(
    self,
    u: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    *,
    batch_size: int,
  ) -> torch.Tensor:
    c_v_given_u = self.v_given_ux_.pdf(
      v, x=self._features(u, x), batch_size=batch_size
    )
    c_u_given_v = self.u_given_vx_.pdf(
      u, x=self._features(v, x), batch_size=batch_size
    )

    return 0.5 * (c_v_given_u + c_u_given_v)

  def _unprojected_pdf_grid(
    self,
    u_grid: torch.Tensor,
    v_grid: torch.Tensor,
    x_row: torch.Tensor,
    *,
    batch_size: int,
  ) -> torch.Tensor:
    """Symmetric raw density grid ``out[i, j] = c(u_grid[i], v_grid[j] | x_row)``.

    The single-covariate special case of :py:meth:`_unprojected_pdf_grids_by_x`;
    ``x_row`` is a single ``(1, p)`` row, so ``n_x = 1`` and we slice it
    back out.
    """
    return self._unprojected_pdf_grids_by_x(
      u_grid, v_grid, x_row, batch_size=batch_size
    )[:, 0, :]

  def _unprojected_pdf_grids_by_x(
    self,
    u_grid: torch.Tensor,
    v_grid: torch.Tensor,
    x_unique: torch.Tensor,
    *,
    batch_size: int,
  ) -> torch.Tensor:
    """Symmetric raw density grids for every unique covariate row at once.

    Returns shape ``(n_u, n_x, n_v)`` with ``out[i, k, j] = c(u_grid[i],
    v_grid[j] | x_unique[k])`` (Rosenblatt-symmetric, pre-Sinkhorn).

    Both Rosenblatt directions are evaluated with a single batched
    ``pdf_grid`` call each: the conditioning rows are the Cartesian
    product ``(grid, x_unique)``, so all ``n_x`` per-x grids share the
    forward passes instead of looping one inner call per unique x.  This
    is the main speed lever for the conditional Sinkhorn projection;
    :py:meth:`_unprojected_pdf_grid` is the ``n_x = 1`` special case.
    """
    n_u, n_v, n_x = u_grid.shape[0], v_grid.shape[0], x_unique.shape[0]

    # V|U: conditioning rows (u_i, x_k) in u-major / x-minor order, so the
    # flat result reshapes to [u, x, v].
    first_vu = u_grid.repeat_interleave(n_x)
    x_vu = x_unique.repeat(n_u, 1)
    grid_vu = self.v_given_ux_.pdf_grid(
      v_grid, x=self._features(first_vu, x_vu), batch_size=batch_size
    )
    grid_vu = grid_vu.reshape(n_u, n_x, n_v)

    # U|V: conditioning rows (v_j, x_k) -> [v, x, u]; permute to [u, x, v]
    # so both directions index [u, x, v] before averaging.
    first_uv = v_grid.repeat_interleave(n_x)
    x_uv = x_unique.repeat(n_v, 1)
    grid_uv = self.u_given_vx_.pdf_grid(
      u_grid, x=self._features(first_uv, x_uv), batch_size=batch_size
    )
    grid_uv = grid_uv.reshape(n_v, n_x, n_u).permute(2, 1, 0)

    return 0.5 * (grid_vu + grid_uv)

  def log_pdf(
    self,
    uv: torch.Tensor,
    *,
    x: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Log of the optionally projected ``pdf``, floored at tiny.

    Parameters
    ----------
    uv : torch.Tensor, shape (n, 2)
        Pair pseudo-observations.
    x : torch.Tensor, shape (n, p), or None, optional
        Exogenous covariates, one row per observation.

    Returns
    -------
    torch.Tensor, shape (n,)
        Log-density values.
    """
    density = self.pdf(uv, x=x)
    return torch.log(torch.clamp(density, min=torch.finfo(density.dtype).tiny))

  def pdf_grid(
    self,
    u_grid: torch.Tensor,
    v_grid: torch.Tensor,
    *,
    x_row: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Density on the Cartesian product ``out[i, j] = c(u_grid[i], v_grid[j] | x)``.

    Available for every backend (each backend's ``pdf_grid`` predicts once
    per conditioning row).  ``x_row`` is a single covariate row reused on
    both axes; when ``None`` an empty covariate row is used.  Both
    Rosenblatt directions are evaluated on the same Cartesian product
    (transposing the reverse one) and averaged.

    The batch size and the Sinkhorn iteration count come from the controls,
    as they do for ``pdf``.
    """
    u_t, v_t, x_row_t = self._prepare_grid_inputs(u_grid, v_grid, x_row)

    with torch.inference_mode():
      c_unprojected = self._unprojected_pdf_grid(
        u_t,
        v_t,
        x_row_t,
        batch_size=self.batch_size,
      )

      if self.sinkhorn_iters is None:
        return c_unprojected

      return self._project_grid(
        c_unprojected,
        u_t,
        v_t,
        sinkhorn_iters=self.sinkhorn_iters,
      )

  # -------------------------------------------------------------------
  # h-functions (conditional CDFs along one axis)
  #
  # We follow pyvinecopulib's numbering convention: ``hfunc_i``
  # conditions on the i-th argument:
  #
  #   hfunc1(u, v | x) = P(V <= v | U = u, X = x) = F_{V | U, X}(v|u,x)
  #   hfunc2(u, v | x) = P(U <= u | V = v, X = x) = F_{U | V, X}(u|v,x)
  #
  # Equivalently, hfunc1 = dC/du and hfunc2 = dC/dv.
  # -------------------------------------------------------------------

  def _hfunc1_raw(
    self,
    u: torch.Tensor,
    *,
    x: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """``h_1(u, v | x) = P(V <= v | U = u, X = x) = F_{V | U, X}(v | u, x)``.

    Always available (the V|U regressor is always fitted). A direct read of
    the inner regressor's conditional CDF -- no integration, one batched inner
    ``cdf`` call. Convention matches
    :py:meth:`pyvinecopulib.Bicop.hfunc1`: ``hfunc1`` conditions on the first
    argument.

    Parameters
    ----------
    u : torch.Tensor, shape (n, 2)
        Pair pseudo-observations, already prepared by ``_prep_args``.
    x : torch.Tensor, shape (n, p), or None, optional
        Exogenous covariates, one row per observation.

    Returns
    -------
    torch.Tensor, shape (n,)
        Conditional distribution values.
    """
    u_t, v_t = u[:, 0], u[:, 1]
    x_t = self._covariates(x, u.shape[0])

    out = self.v_given_ux_.cdf(v_t, x=self._features(u_t, x_t))

    return torch.clamp(
      out,
      self.eps,
      1.0 - self.eps,
    )

  def _hfunc2_raw(
    self,
    u: torch.Tensor,
    *,
    x: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """``h_2(u, v | x) = P(U <= u | V = v, X = x) = F_{U | V, X}(u | v, x)``.

    A direct read of the U|V regressor's conditional CDF. Convention matches
    :py:meth:`pyvinecopulib.Bicop.hfunc2`: ``hfunc2`` conditions on the second
    argument.

    Parameters
    ----------
    u : torch.Tensor, shape (n, 2)
        Pair pseudo-observations, already prepared by ``_prep_args``.
    x : torch.Tensor, shape (n, p), or None, optional
        Exogenous covariates, one row per observation.

    Returns
    -------
    torch.Tensor, shape (n,)
        Conditional distribution values.
    """
    u_t, v_t = u[:, 0], u[:, 1]
    x_t = self._covariates(x, u.shape[0])

    out = self.u_given_vx_.cdf(u_t, x=self._features(v_t, x_t))

    return torch.clamp(
      out,
      self.eps,
      1.0 - self.eps,
    )

  def _hinv1_raw(
    self,
    u: torch.Tensor,
    *,
    x: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Invert ``_hfunc1_raw`` using the V|U backend's native quantiles.

    Overriding this optional leaf is what keeps the inversion exact and O(1)
    in backend calls: the inherited default bisects ``_hfunc1_raw``, which
    would turn one inner ``icdf`` into tens of batched forward passes and
    return an approximation of a quantile the backend already knows.

    Parameters
    ----------
    u : torch.Tensor, shape (n, 2)
        Column 0 is the conditioning value; column 1 is the level to invert.
    x : torch.Tensor, shape (n, p), or None, optional
        Exogenous covariates, one row per observation.

    Returns
    -------
    torch.Tensor, shape (n,)
        The inverted values.
    """
    u_t, alpha_t = u[:, 0], u[:, 1]
    x_t = self._covariates(x, u.shape[0])
    return self.v_given_ux_.icdf(alpha_t, x=self._features(u_t, x_t))

  def _hinv2_raw(
    self,
    u: torch.Tensor,
    *,
    x: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Invert ``_hfunc2_raw`` using the U|V backend's native quantiles.

    The counterpart of :meth:`_hinv1_raw`, and overridden for the same reason.

    Parameters
    ----------
    u : torch.Tensor, shape (n, 2)
        Column 0 is the level to invert; column 1 is the conditioning value.
    x : torch.Tensor, shape (n, p), or None, optional
        Exogenous covariates, one row per observation.

    Returns
    -------
    torch.Tensor, shape (n,)
        The inverted values.
    """
    alpha_t, v_t = u[:, 0], u[:, 1]
    x_t = self._covariates(x, u.shape[0])
    return self.u_given_vx_.icdf(alpha_t, x=self._features(v_t, x_t))

  def _sample_uniform(
    self,
    n: int,
    qrng: bool,
    seeds: list[int],
  ) -> torch.Tensor:
    """Draw float64 base uniforms on the estimator's configured device."""
    if qrng:
      from pyvinecopulib.utils import sample_uniform

      draws = sample_uniform(n, 2, qrng=True, seeds=list(seeds))
      return self._prep(draws)

    generator = torch.Generator(device=self._device)
    if seeds:
      generator.manual_seed(int(seeds[0]))
    else:
      generator.seed()
    return torch.rand(
      (n, 2),
      generator=generator,
      dtype=torch.float64,
      device=self._device,
    )

  # -------------------------------------------------------------------
  # Joint CDF
  # -------------------------------------------------------------------

  def _cdf_raw(
    self,
    u: torch.Tensor,
    *,
    x: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Joint CDF ``C(u_i, v_i | x_i)`` evaluated row-by-row.

    Trapezoidal integration of the inner conditional CDF, averaged over both
    Rosenblatt directions::

        C(u, v | x) = 0.5 (int_0^u F_{V|U,X}(v|s,x) ds
                           + int_0^v F_{U|V,X}(u|t,x) dt)

    The number of trapezoid steps is the controls'
    :attr:`~npcc.core.controls.FitControlsRosenblattBicop.cdf_n_int`;
    :meth:`cdf_grid` keeps its own, finer default, sharing one grid across the
    whole Cartesian product.

    Two inherited members reach this leaf *without* going through
    ``_prep_args`` -- ``rect_prob`` and ``cond_interval_prob`` call it
    directly, since their own arguments are rectangle corners rather than a
    pair argument. Neither is reachable here: both exist to serve a discrete
    edge, which this estimator refuses.

    Parameters
    ----------
    u : torch.Tensor, shape (n, 2)
        Pair pseudo-observations, already prepared by ``_prep_args``.
    x : torch.Tensor, shape (n, p), or None, optional
        Exogenous covariates, one row per observation.

    Returns
    -------
    torch.Tensor, shape (n,)
        Distribution values.
    """
    n_int = self.cdf_n_int
    effective_batch_size = self.batch_size
    u_t, v_t = u[:, 0], u[:, 1]
    x_t = self._covariates(x, u.shape[0])

    cdf_v_dir = self._integrate_one_direction(
      upper=u_t,
      conditioned=v_t,
      x=x_t,
      module=self.v_given_ux_,
      n_int=n_int,
      batch_size=effective_batch_size,
    )
    cdf_u_dir = self._integrate_one_direction(
      upper=v_t,
      conditioned=u_t,
      x=x_t,
      module=self.u_given_vx_,
      n_int=n_int,
      batch_size=effective_batch_size,
    )
    return 0.5 * (cdf_v_dir + cdf_u_dir)

  def _integrate_one_direction(
    self,
    *,
    upper: torch.Tensor,
    conditioned: torch.Tensor,
    x: torch.Tensor,
    module: ConditionalMargin,
    n_int: int,
    batch_size: int,
  ) -> torch.Tensor:
    """Compute int_eps^{upper_i} F(conditioned_i | s, x_i) ds for each row."""
    eps = self.eps
    n = upper.shape[0]
    upper_safe = torch.clamp(upper, min=eps + 1e-12)

    # One integration grid per row, from eps up to that row's own upper
    # limit, so every row is integrated over its own interval.
    t = torch.linspace(
      0.0, 1.0, n_int + 1, dtype=torch.float64, device=self._device
    )
    s_grids = eps + (upper_safe.unsqueeze(1) - eps) * t.unsqueeze(0)

    s_flat = s_grids.reshape(-1)
    cond_flat = conditioned.repeat_interleave(n_int + 1)
    x_flat = x.repeat_interleave(n_int + 1, dim=0)

    features = self._features(s_flat, x_flat)
    cdf_flat = module.cdf(cond_flat, x=features, batch_size=batch_size)
    cdf_grid = cdf_flat.reshape(n, n_int + 1)

    # Per-row trapezoidal integral over the s axis.
    ds = torch.diff(s_grids, dim=1)
    avgs = 0.5 * (cdf_grid[:, :-1] + cdf_grid[:, 1:])
    return torch.sum(avgs * ds, dim=1)

  def cdf_grid(
    self,
    u_grid: torch.Tensor,
    v_grid: torch.Tensor,
    *,
    x_row: torch.Tensor | None = None,
    n_int: int = 64,
  ) -> torch.Tensor:
    """Cartesian-grid joint CDF ``out[i, j] = C(u_grid[i], v_grid[j] | x_row)``.

    Available for every backend (uses the inner ``cdf_grid`` fast path).
    Builds a single shared fine ``s``-grid covering ``[eps, max(u_grid)]``,
    evaluates the inner CDF on the Cartesian product ``(s_fine x v_grid)``
    (one forward pass per row of ``s_fine``), then for each ``u_grid[i]``
    reads off the cumulative trapezoidal integral up to ``u_grid[i]`` via
    interpolation, then averages the analogous ``v``-axis integral.
    """
    if n_int < 2:
      raise ValueError("n_int must be at least 2.")

    u_t, v_t, x_row_t = self._prepare_grid_inputs(u_grid, v_grid, x_row)

    cdf_v_dir = self._integrate_grid_one_direction(
      upper_grid=u_t,
      conditioned_grid=v_t,
      x_row=x_row_t,
      module=self.v_given_ux_,
      n_int=n_int,
    )
    cdf_u_dir = self._integrate_grid_one_direction(
      upper_grid=v_t,
      conditioned_grid=u_t,
      x_row=x_row_t,
      module=self.u_given_vx_,
      n_int=n_int,
    )
    return 0.5 * (cdf_v_dir + cdf_u_dir.T)

  def _integrate_grid_one_direction(
    self,
    *,
    upper_grid: torch.Tensor,
    conditioned_grid: torch.Tensor,
    x_row: torch.Tensor,
    module: ConditionalMargin,
    n_int: int,
  ) -> torch.Tensor:
    """Compute int_0^{upper_grid[i]} F(conditioned_grid[j] | s, x_row) ds.

    Returns shape ``(len(upper_grid), len(conditioned_grid))``.
    """
    eps = self.eps
    n_u, n_v = upper_grid.shape[0], conditioned_grid.shape[0]

    # Shared fine s-grid covering [eps, max(upper_grid)].
    s_max = torch.clamp(upper_grid.max(), min=eps + 1e-12)
    s_fine = torch.linspace(
      eps,
      float(s_max.item()),
      n_int + 1,
      dtype=torch.float64,
      device=self._device,
    )

    x_for_s = x_row.repeat_interleave(s_fine.shape[0], dim=0)
    features = self._features(s_fine, x_for_s)
    cdf_table = module.cdf_grid(conditioned_grid, x=features)

    # Cumulative trapezoid along axis=0.
    ds = torch.diff(s_fine)
    avgs = 0.5 * (cdf_table[:-1] + cdf_table[1:])
    cum = torch.zeros(
      (s_fine.shape[0], n_v), dtype=torch.float64, device=self._device
    )
    cum[1:] = torch.cumsum(avgs * ds.unsqueeze(1), dim=0)

    # For each upper_grid[i], interpolate cum at s = upper_grid[i].
    out = torch.empty((n_u, n_v), dtype=torch.float64, device=self._device)
    for j in range(n_v):
      out[:, j] = interp(upper_grid, s_fine, cum[:, j])
    return out

  # -------------------------------------------------------------------
  # Kendall's tau (sample-based, mirroring vinecopulib's KernelBicop, which
  # has no Python binding)
  # -------------------------------------------------------------------

  # Default seeds used by pyvinecopulib's
  # ``KernelBicop::parameters_to_tau``.  Reusing them gives bit-identical
  # reproducibility against vinecopulib.
  _GHALTON_DEFAULT_SEEDS: tuple[int, ...] = (
    204967043,
    733593603,
    184618802,
    399707801,
    290266245,
  )

  def tau(
    self,
    x_row: torch.Tensor | None = None,
    *,
    n: int = 1000,
    seeds: list[int] | None = None,
  ) -> float:
    """Kendall's tau via vinecopulib's ``KernelBicop::parameters_to_tau`` recipe.

    1. Draw a deterministic 2-D Generalised-Halton quasi-random sample
       ``(u_i, alpha_i)`` of size ``n`` via
       :func:`pyvinecopulib.utils.ghalton`.
    2. Apply the inverse Rosenblatt transform along the first axis:
       ``v_i = F_{V | U, X}^{-1}(alpha_i | u_i, x_row)``.  The resulting
       ``(u_i, v_i)`` pairs are distributed according to the fitted
       copula.
    3. Return the weighted (rank-)Kendall ``tau`` of the sample via
       :func:`pyvinecopulib.utils.wdm`.

    Available for every backend (uses the inner ``icdf``).
    """
    if n < 10:
      raise ValueError("n must be at least 10.")

    if seeds is None:
      seeds_list = list(self._GHALTON_DEFAULT_SEEDS)
    else:
      seeds_list = list(seeds)

    from pyvinecopulib.utils import ghalton, wdm

    quasi = self._prep(ghalton(n, 2, seeds_list))

    eps = self.eps
    u_t = torch.clamp(quasi[:, 0], eps, 1.0 - eps)
    alpha_t = torch.clamp(quasi[:, 1], eps, 1.0 - eps)

    if x_row is None:
      x_t = self._default_x(n)
    else:
      x_t = self._prepare_x_row(x_row).repeat_interleave(n, dim=0)

    # Inverse Rosenblatt: v = F_{V | U, X}^{-1}(alpha | u, x).
    v_t = self.v_given_ux_.icdf(alpha_t, x=self._features(u_t, x_t))

    return float(wdm(to_numpy(u_t), to_numpy(v_t), "tau"))
