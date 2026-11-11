"""Fit a vine outside this process by running its script; read its densities.

E2b's GAM vines (``r_methods/fit_gamvine.R``) are fitted in R, each on a
cell's data-scale training sample, and evaluated at the cell's test points.
A fit is one script run in a fresh directory::

    Rscript <script> <dir> <seed> <arm>

- ``train.csv`` has the columns ``x``, ``y1``, ..., ``yd``, one row per
  training observation.
- ``eval.csv`` has ``x``, ``y1``, ..., ``yd`` and ``u1``, ..., ``ud``, one
  row per test point: the point on the data scale, and its true probability
  integral transforms.
- ``structure.csv``, for an arm fitted on a given structure only, holds that
  structure's R-vine matrix in VineCopula's convention
  (:func:`to_vinecopula`), without a header.

The script writes into the same directory:

- ``pred.csv``, one row per evaluation row in the same order: the row's
  ``x``, ``y`` and ``u``, echoed; ``pdf``, the fitted joint density at the
  ``y``, and ``copula_pdf``, the fitted copula density at the ``u``; and the
  seconds the script's steps took, the same on every row -- ``margin_time``
  and ``vine_time`` fitting, ``joint_pdf_time`` and ``copula_pdf_time``
  evaluating;
- ``fitted.csv``, the fitted structure's R-vine matrix in VineCopula's
  convention, without a header;
- ``pairs.csv``, one row per pair copula of the fit: its ``tree``, its
  arguments ``first`` and ``second`` and ``conditioning`` set, its gamCopula
  ``family`` code and ``family_name``, its ``kind`` (``gam``, ``constant`` or
  ``independence``), its calibration ``formula`` and ``covariates``, and its
  ``edf``, ``par`` and ``par2``.

The echoed points are what lets a density be checked against the point it
is scored at. Failures are reported as in :mod:`~npcc.experiments._external`:
a nonzero exit is the method's failure, :class:`~npcc.experiments._external.
ScriptFailedError`; the exit status :data:`~npcc.experiments._external.
REFUSED_STATUS`, or output that breaks the exchange, is
:class:`~npcc.experiments._external.ExchangeError`, which must stop a study.

**Matrices.** pyvinecopulib writes an R-vine matrix with its natural order on
the antidiagonal, tree ``t`` in row ``t`` from the top; VineCopula writes it
lower triangular, its order on the diagonal, tree ``t`` in row ``t`` from the
bottom. Each is the other turned upside down.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from time import perf_counter
from typing import NamedTuple, cast

import numpy as np
import pandas as pd
import pyvinecopulib as pv
import torch
from pyvinecopulib.core.extend import to_numpy

from npcc.experiments import _external
from npcc.experiments._external import REFUSED_STATUS, ExchangeError

#: An edge of a vine: its conditioned pair and its conditioning set.
Edge = tuple[frozenset[int], frozenset[int]]

#: The steps whose seconds the script reports, in ``pred.csv``'s order.
TIMINGS: tuple[str, ...] = (
  "margin_time",
  "vine_time",
  "joint_pdf_time",
  "copula_pdf_time",
)

#: The columns of ``pairs.csv``.
PAIR_COLUMNS: tuple[str, ...] = (
  "tree",
  "first",
  "second",
  "conditioning",
  "family",
  "family_name",
  "kind",
  "formula",
  "covariates",
  "edf",
  "par",
  "par2",
)


class VineFit(NamedTuple):
  """One script run: the densities at every test point, and the fit."""

  #: The fitted joint density at each test point, in the points' order.
  pdf: torch.Tensor
  #: The fitted copula density at each test point's true transforms.
  copula_pdf: torch.Tensor
  #: The fitted structure.
  structure: pv.RVineStructure
  #: One record per pair copula, with :data:`PAIR_COLUMNS` as keys.
  pairs: list[dict[str, object]]
  #: Seconds the script spent on each of :data:`TIMINGS`.
  timings: dict[str, float]
  #: Seconds the whole run took, start-up and file exchange included.
  wall_time: float


def to_vinecopula(structure: pv.RVineStructure) -> np.ndarray:
  """``structure``'s R-vine matrix in VineCopula's convention.

  Parameters
  ----------
  structure : pyvinecopulib.RVineStructure
      The structure.

  Returns
  -------
  numpy.ndarray
      The ``(d, d)`` lower-triangular integer matrix.
  """
  return np.asarray(structure.matrix, dtype=np.int64)[::-1].copy()


def from_vinecopula(matrix: np.ndarray) -> pv.RVineStructure:
  """The structure an R-vine matrix in VineCopula's convention encodes.

  Parameters
  ----------
  matrix : numpy.ndarray
      A ``(d, d)`` lower-triangular R-vine matrix.

  Returns
  -------
  pyvinecopulib.RVineStructure
      The structure, which pyvinecopulib validates.
  """
  flipped = np.asarray(matrix, dtype=np.uint64)[::-1]
  return pv.RVineStructure.from_matrix(np.asfortranarray(flipped))


def edges(structure: pv.RVineStructure) -> set[Edge]:
  """A vine's edges, each as (conditioned pair, conditioning set).

  Two R-vine matrices can encode one vine, so structures are compared by
  their edges.

  Parameters
  ----------
  structure : pyvinecopulib.RVineStructure
      The structure.

  Returns
  -------
  set of tuple
      One ``(frozenset, frozenset)`` per edge, over every tree.
  """
  # pyvinecopulib's compiled `get_trees` is untyped here.
  trees = cast("list[list[tuple[int, int, list[int]]]]", structure.get_trees())
  return {
    (frozenset((a, b)), frozenset(conditioning))
    for tree in trees
    for a, b, conditioning in tree
  }


def _data_columns(prefix: str, values: torch.Tensor) -> dict[str, np.ndarray]:
  array = to_numpy(values, dtype=np.float64)
  return {f"{prefix}{j + 1}": array[:, j] for j in range(array.shape[1])}


def _read_matrix(path: Path, d: int, *, script: str) -> pv.RVineStructure:
  if not path.is_file():
    raise ExchangeError(
      f"{script} must write its structure; got no {path.name}"
    )
  matrix = np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=2)
  if matrix.shape != (d, d):
    raise ExchangeError(
      f"{script} must write a {d} x {d} R-vine matrix; got {matrix.shape}"
    )
  try:
    return from_vinecopula(matrix)
  except (RuntimeError, ValueError) as exc:
    raise ExchangeError(
      f"{script} must write a valid R-vine matrix; got {matrix.tolist()}"
    ) from exc


def _read_pairs(path: Path, d: int, *, script: str) -> list[dict[str, object]]:
  if not path.is_file():
    raise ExchangeError(f"{script} must write its pairs; got no {path.name}")
  # Text stays text: a conditioning set "2" is not the number 2.
  text_columns = (
    "conditioning",
    "family_name",
    "kind",
    "formula",
    "covariates",
  )
  pairs = pd.read_csv(
    path,
    keep_default_na=False,
    na_values=["NA"],
    dtype=dict.fromkeys(text_columns, str),
  )
  missing = [c for c in PAIR_COLUMNS if c not in pairs.columns]
  if missing or len(pairs) != d * (d - 1) // 2:
    raise ExchangeError(
      f"{script} must write one row per pair copula, {d * (d - 1) // 2}, "
      f"with the columns {list(PAIR_COLUMNS)}; got {len(pairs)} rows, "
      f"missing {missing}"
    )
  # Through JSON, so every value is a plain number, string or None.
  text = cast("str", pairs[list(PAIR_COLUMNS)].to_json(orient="records"))
  return cast("list[dict[str, object]]", json.loads(text))


def _read_predictions(
  path: Path, points: pd.DataFrame, *, script: str
) -> pd.DataFrame:
  """The script's ``pred.csv``, checked against the rows it was handed."""
  if not path.is_file():
    raise ExchangeError(
      f"{script} must write its predictions; got no {path.name}"
    )
  # The default parser is not exact for float64.
  pred = pd.read_csv(path, float_precision="round_trip")
  columns = [*points.columns, "pdf", "copula_pdf", *TIMINGS]
  missing = [c for c in columns if c not in pred.columns]
  if missing:
    raise ExchangeError(
      f"{script} must write the columns {columns}; got no {missing}"
    )
  if len(pred) != len(points):
    raise ExchangeError(
      f"{script} must write one row per test point; got {len(pred)} rows "
      f"for {len(points)} points"
    )
  coords = list(points.columns)
  same = np.isclose(
    pred[coords].to_numpy(dtype=np.float64),
    points.to_numpy(dtype=np.float64),
    rtol=0.0,
    atol=1e-12,
  ).all(axis=1)
  if not same.all():
    first = int(np.argmin(same))
    raise ExchangeError(
      f"{script} must echo the test points in their order; got "
      f"{pred.loc[first, coords].tolist()} at row {first}, where the point "
      f"is {points.loc[first].tolist()}"
    )
  for q in ("pdf", "copula_pdf"):
    values = pred[q].to_numpy(dtype=np.float64)
    if not (np.isfinite(values) & (values >= 0.0)).all():
      raise ExchangeError(
        f"{script} must write a finite, nonnegative {q} at every point"
      )
  return pred


def fit_and_evaluate(
  script: str,
  sample: tuple[torch.Tensor, torch.Tensor],
  points: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
  *,
  arm: str,
  seed: int,
  name: str,
  structure: pv.RVineStructure | None = None,
) -> VineFit:
  """Fit ``script``'s ``arm`` on ``sample`` and evaluate it at ``points``.

  Parameters
  ----------
  script : str
      The method's script, run under :data:`~npcc.experiments._external.
      RSCRIPT`.
  sample : tuple of torch.Tensor
      ``(y, x)``, the training sample: ``y`` of shape ``(n, d)`` on the data
      scale and ``x`` of shape ``(n, 1)``.
  points : tuple of torch.Tensor
      ``(y, u, x)``, the test points: ``y`` and its true transforms ``u``,
      each ``(m, d)``, and ``x`` of shape ``(m, 1)``.
  arm : str
      The arm the script fits.
  seed : int
      Seeds the script's random number generator.
  name : str
      The run, as messages and logs name it.
  structure : pyvinecopulib.RVineStructure, optional
      The structure to fit, for an arm that is given one.

  Returns
  -------
  VineFit
      The densities at every test point, in their order, and the fit.

  Raises
  ------
  ScriptFailedError
      If the script exits with a nonzero status other than
      :data:`~npcc.experiments._external.REFUSED_STATUS`.
  ExchangeError
      If it exits with that status, or its output breaks the exchange, or the
      inputs disagree in shape.
  RuntimeError
      If the interpreter is missing.
  """
  y, x = sample
  y_eval, u_eval, x_eval = points
  d = y.shape[1]
  if (
    y.ndim != 2
    or x.shape != (y.shape[0], 1)
    or y_eval.shape[1:] != (d,)
    or u_eval.shape != y_eval.shape
    or x_eval.shape != (y_eval.shape[0], 1)
  ):
    raise ExchangeError(
      f"{name} must have y (n, d) with x (n, 1), and test y and u (m, d) with "
      f"x (m, 1); got {tuple(y.shape)}, {tuple(x.shape)}, "
      f"{tuple(y_eval.shape)}, {tuple(u_eval.shape)}, {tuple(x_eval.shape)}"
    )
  train = pd.DataFrame(
    {"x": to_numpy(x, dtype=np.float64)[:, 0], **_data_columns("y", y)}
  )
  evaluation = pd.DataFrame(
    {
      "x": to_numpy(x_eval, dtype=np.float64)[:, 0],
      **_data_columns("y", y_eval),
      **_data_columns("u", u_eval),
    }
  )
  with tempfile.TemporaryDirectory(prefix="npcc-external-vine-") as tmp:
    root = Path(tmp)
    train.to_csv(root / "train.csv", index=False, float_format="%.17g")
    evaluation.to_csv(root / "eval.csv", index=False, float_format="%.17g")
    if structure is not None:
      np.savetxt(
        root / "structure.csv",
        to_vinecopula(structure),
        fmt="%d",
        delimiter=",",
      )
    t0 = perf_counter()
    # Read at call time, so a test can point the interpreter elsewhere.
    _external._run(
      [_external.RSCRIPT, script, str(root), str(seed), arm],
      name=name,
      refused_status=REFUSED_STATUS,
    )
    wall_time = perf_counter() - t0
    pred = _read_predictions(root / "pred.csv", evaluation, script=script)
    fitted = _read_matrix(root / "fitted.csv", d, script=script)
    pairs = _read_pairs(root / "pairs.csv", d, script=script)

  def column(q: str) -> torch.Tensor:
    return torch.from_numpy(pred[q].to_numpy(dtype=np.float64, copy=True))

  return VineFit(
    pdf=column("pdf"),
    copula_pdf=column("copula_pdf"),
    structure=fitted,
    pairs=pairs,
    timings={t: float(pred[t].iloc[0]) for t in TIMINGS},
    wall_time=wall_time,
  )
