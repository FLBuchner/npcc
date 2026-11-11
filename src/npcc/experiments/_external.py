"""Fit an external method by running its script, and read back its predictions.

An ``external_r`` estimator is a script that fits a conditional copula outside
this process. A cell runs it once, handing it the training sample and every
point it scores, so the metric grid and the surface grid are predicted from
the one fit. The exchange is three CSV files and a seed, in this order::

    Rscript <script> <train.csv> <eval.csv> <pred.csv> <seed>

- ``train.csv`` has the columns ``u1``, ``u2`` and the covariates, one row
  per observation.
- ``eval.csv`` has ``grid``, ``u1``, ``u2`` and the covariates, one row per
  evaluation point; ``grid`` names the grid the point belongs to, such as
  ``metric`` or ``surface``.
- ``pred.csv``, which the script writes, has one row per evaluation row, in
  the same order: the row's ``grid``, ``u1``, ``u2`` and covariates, echoed;
  the fitted ``pdf``, ``hfunc1``, ``hfunc2`` and ``tau`` at that point; and
  the seconds the script's steps took -- ``fit_time``, the same on every row,
  and ``pdf_time``, ``hfunc1_time``, ``hfunc2_time`` and ``tau_time``, each
  the same within a grid.

The covariates are named by the caller. E1's studies have one, ``x``, and
call the script with the four arguments above. E9's
(:mod:`~npcc.experiments.covariate_scenarios`) has fifteen, ``b1``, ...,
``b5`` (each 0 or 1), ``z1``, ..., ``z5`` and ``t1``, ..., ``t5``, and passes
two more arguments, the cell's scenario (``gampcc``, ``interaction`` or
``constant``) and family (``gaussian`` or ``clayton``), which the oracle, fitting
the true model, reads::

    Rscript <script> <train.csv> <eval.csv> <pred.csv> <seed> <scenario> \
      <family>

``hfunc1`` is ``P(V <= v | U = u)``, the copula's derivative in ``u``, and
``hfunc2`` is ``P(U <= u | V = v)``, as everywhere in this package. The echoed
points are what lets a prediction be checked against the point it is scored
at, since a script that reorders its rows within a grid would otherwise go
unnoticed. The script times its own steps, so the interpreter's start-up and
the file exchange, which belong to this harness rather than to the method,
are not counted.

What a script prints is logged: its stdout at ``INFO``, the place for a line
on the model it fitted, such as the family it selected; its stderr, where R
reports its warnings, at ``WARNING``.

A script that exits with a nonzero status raises :class:`ScriptFailedError`:
the method's own failure on that sample, which a study may record as a
result. Predictions that break the exchange raise :class:`ExchangeError`, a
defect of the script or of this harness, and a missing interpreter or
package a plain :class:`RuntimeError`, a defect of the machine; neither is
the method's outcome, and a study must stop on them. A caller may also name
a status, :data:`REFUSED_STATUS` for E9's scripts, with which a script
reports that it refused its input or failed a check of its own; that exit
raises :class:`ExchangeError` too.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from pyvinecopulib.core.extend import to_numpy

from npcc.experiments.scenarios import EvalGrid

logger = logging.getLogger(__name__)

#: The interpreter every script runs under. A module constant, so a test can
#: point it at a Python stand-in and run the exchange without R.
RSCRIPT: str = "Rscript"

#: What a script predicts at each evaluation point.
QUANTITIES: tuple[str, ...] = ("pdf", "hfunc1", "hfunc2", "tau")

#: The stderr lines quoted in a failure message or a warning.
_TAIL_LINES = 30

#: The exit status with which E9's scripts refuse their input or report a
#: failed check of their own, as opposed to the method failing.
REFUSED_STATUS: int = 3

#: A call that loads a package, with its arguments.
_LOAD = re.compile(r"\b(?:library|require|requireNamespace)\((?P<args>[^)]*)\)")
#: A package's namespace prefix, ``name::`` or ``name:::``.
_PREFIX = re.compile(r"\b(?P<name>[A-Za-z][A-Za-z0-9.]*):::?(?=[A-Za-z.`])")
#: The package a load call names first, bare or quoted.
_NAME = re.compile(
  r"^\s*(?:package\s*=\s*)?(?P<q>[\"']?)(?P<name>[A-Za-z][A-Za-z0-9.]*)(?P=q)\s*$"
)


class ScriptFailedError(RuntimeError):
  """A script exited with a nonzero status: the method failed on its input."""


class ExchangeError(ValueError):
  """A script's predictions, or the harness's input, break the exchange."""


class ExternalFit(NamedTuple):
  """One script run: the fit, and every grid's predictions and timings."""

  #: Seconds the script spent fitting.
  fit_time: float
  #: Per grid name, each of :data:`QUANTITIES` in the grid's point order.
  predictions: dict[str, dict[str, torch.Tensor]]
  #: Per grid name, the seconds each of :data:`QUANTITIES` took on it.
  timings: dict[str, dict[str, float]]
  #: Seconds the whole run took, start-up and file exchange included.
  wall_time: float


def _columns(
  x: torch.Tensor, covariates: Sequence[str], *, what: str
) -> dict[str, np.ndarray]:
  """The covariate matrix ``x`` as named columns."""
  values = to_numpy(x, dtype=np.float64).reshape(x.shape[0], -1)
  if values.shape[1] != len(covariates):
    raise ExchangeError(
      f"{what} must have one covariate column per name in {list(covariates)}; "
      f"got {values.shape[1]}"
    )
  return {name: values[:, k] for k, name in enumerate(covariates)}


def _points(
  name: str, grid: EvalGrid, covariates: Sequence[str]
) -> pd.DataFrame:
  """The evaluation rows of one grid, labeled with its name."""
  if grid.x_flat is None:
    raise ValueError(
      f"grid {name!r} must be conditional, since an external method is "
      "fitted on a covariate; got an unconditional grid"
    )
  return pd.DataFrame(
    {
      "grid": name,
      "u1": to_numpy(grid.u_flat),
      "u2": to_numpy(grid.v_flat),
      **_columns(grid.x_flat, covariates, what=f"grid {name!r}"),
    }
  )


def _tail(text: str) -> str:
  return "\n".join(text.strip().splitlines()[-_TAIL_LINES:])


def _run(
  command: list[str], *, name: str, refused_status: int | None = None
) -> None:
  """Run one script to completion; raise with its stderr if it fails."""
  try:
    done = subprocess.run(command, capture_output=True, text=True, check=False)
  except FileNotFoundError as exc:
    raise RuntimeError(
      f"{command[0]} must be on PATH to run {name}; it was not found"
    ) from exc
  if refused_status is not None and done.returncode == refused_status:
    raise ExchangeError(
      f"{command[1]} must accept its input and pass its own checks for "
      f"{name}; it refused with status {refused_status} and stderr:\n"
      f"{_tail(done.stderr)}"
    )
  if done.returncode != 0:
    raise ScriptFailedError(
      f"{command[1]} must exit with status 0 for {name}; got "
      f"{done.returncode}, with stdout:\n{_tail(done.stdout)}\n"
      f"and stderr:\n{_tail(done.stderr)}"
    )
  if done.stdout.strip():
    logger.info("%s: %s", name, _tail(done.stdout))
  if done.stderr.strip():
    logger.warning("%s wrote to stderr:\n%s", name, _tail(done.stderr))


def script_packages(text: str) -> list[str]:
  """The packages an R script's code loads or calls into.

  Every ``library``, ``require`` or ``requireNamespace`` call's package,
  bare or quoted, whatever arguments follow it, and every ``name::`` or
  ``name:::`` prefix, outside comments. A call with ``character.only`` names
  a variable rather than a package, and is skipped.

  Parameters
  ----------
  text : str
      The script.

  Returns
  -------
  list of str
      The package names, sorted, each once.
  """
  # A `#` inside a string would cut its line short here; none of the
  # scripts this reads has one before a package name.
  code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
  names = set(_PREFIX.findall(code))
  for call in _LOAD.finditer(code):
    args = call["args"]
    if "character.only" in args:
      continue
    first = _NAME.match(args.split(",", 1)[0])
    if first is not None:
      names.add(first["name"])
  return sorted(names)


def check_packages(script: str) -> None:
  """Stop unless every package ``script`` loads or calls into loads.

  A package missing from this machine would otherwise make every fit of the
  script fail, and each failure would read as the method's own.

  Parameters
  ----------
  script : str
      Path to the script; :func:`script_packages` reads its packages.
  """
  packages = script_packages(Path(script).read_text())
  if not packages:
    return
  load = "; ".join(
    f"suppressPackageStartupMessages(library({p}))" for p in packages
  )
  try:
    done = subprocess.run(
      [RSCRIPT, "-e", load], capture_output=True, text=True, check=False
    )
  except FileNotFoundError as exc:
    raise RuntimeError(
      f"{RSCRIPT} must be on PATH to run {script}; it was not found"
    ) from exc
  if done.returncode != 0:
    raise RuntimeError(
      f"{script} must find its packages {packages} installed; loading them "
      f"failed with:\n{_tail(done.stderr)}"
    )


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
  columns = [
    *points.columns,
    *QUANTITIES,
    "fit_time",
    *(f"{q}_time" for q in QUANTITIES),
  ]
  missing = [c for c in columns if c not in pred.columns]
  if missing:
    raise ExchangeError(
      f"{script} must write the columns {columns}; got no {missing}"
    )
  if len(pred) != len(points):
    raise ExchangeError(
      f"{script} must write one row per evaluation point; got {len(pred)} "
      f"rows for {len(points)} points"
    )
  # A point read and written back by another parser may move by an ulp.
  coords = [c for c in points.columns if c != "grid"]
  same = (pred["grid"].astype(str).to_numpy() == points["grid"].to_numpy()) & (
    np.isclose(
      pred[coords].to_numpy(dtype=np.float64),
      points[coords].to_numpy(dtype=np.float64),
      rtol=0.0,
      atol=1e-12,
    ).all(axis=1)
  )
  if not same.all():
    first = int(np.argmin(same))
    raise ExchangeError(
      f"{script} must echo the evaluation points in their order; got "
      f"{pred.loc[first, ['grid', *coords]].tolist()} at row {first}, where "
      f"the point is {points.loc[first].tolist()}"
    )
  for q in QUANTITIES:
    bad = int((~np.isfinite(pred[q].to_numpy(dtype=np.float64))).sum())
    if bad:
      raise ExchangeError(
        f"{script} must predict a finite {q} at every point; got {bad} "
        "non-finite values"
      )
  return pred


def fit_and_predict(
  script: str,
  sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
  grids: Mapping[str, EvalGrid],
  *,
  seed: int,
  name: str,
  covariates: Sequence[str] = ("x",),
  extra_args: Sequence[str] = (),
  refused_status: int | None = None,
) -> ExternalFit:
  """Fit ``script`` on ``sample`` once and predict at every point of ``grids``.

  Parameters
  ----------
  script : str
      The method's script, run under :data:`RSCRIPT`.
  sample : tuple of torch.Tensor
      ``(u, v, x)``, the training sample; ``x`` has one column per entry of
      ``covariates``.
  grids : mapping of str to EvalGrid
      The conditional grids to predict on, by name.
  seed : int
      Seeds the script's random number generator.
  name : str
      The run, as messages and logs name it.
  covariates : sequence of str, default ("x",)
      The covariate columns' names, in the column order of ``x`` and of every
      grid's ``x_flat``.
  extra_args : sequence of str, default ()
      Arguments passed to the script after the seed.
  refused_status : int, optional
      The exit status with which the script refuses its input or reports a
      failed check of its own, if it has one: :data:`REFUSED_STATUS` for
      E9's scripts. Otherwise every nonzero exit is the method's failure.

  Returns
  -------
  ExternalFit
      The fit time, and each grid's predictions, in its point order, with
      their timings.

  Raises
  ------
  ScriptFailedError
      If the script exits with a nonzero status other than
      ``refused_status``.
  ExchangeError
      If the script exits with ``refused_status``, or the predictions do not
      match the points, or ``x`` the covariates.
  RuntimeError
      If the interpreter is missing.
  ValueError
      If a grid is unconditional.
  """
  u, v = (to_numpy(t, dtype=np.float64).reshape(-1) for t in sample[:2])
  train = {
    "u1": u,
    "u2": v,
    **_columns(sample[2], covariates, what="the training covariates"),
  }
  points = pd.concat(
    [_points(grid_name, grid, covariates) for grid_name, grid in grids.items()],
    ignore_index=True,
  )
  with tempfile.TemporaryDirectory(prefix="npcc-external-") as tmp:
    train_csv, eval_csv, pred_csv = (
      Path(tmp) / f"{stem}.csv" for stem in ("train", "eval", "pred")
    )
    pd.DataFrame(train).to_csv(train_csv, index=False)
    points.to_csv(eval_csv, index=False)
    t0 = perf_counter()
    _run(
      [
        RSCRIPT,
        script,
        str(train_csv),
        str(eval_csv),
        str(pred_csv),
        str(seed),
        *extra_args,
      ],
      name=name,
      refused_status=refused_status,
    )
    wall_time = perf_counter() - t0
    pred = _read_predictions(pred_csv, points, script=script)

  predictions: dict[str, dict[str, torch.Tensor]] = {}
  timings: dict[str, dict[str, float]] = {}
  for grid_name in grids:
    rows = pred[pred["grid"].astype(str) == grid_name]
    predictions[grid_name] = {
      q: torch.from_numpy(rows[q].to_numpy(dtype=np.float64, copy=True))
      for q in QUANTITIES
    }
    timings[grid_name] = {
      q: float(rows[f"{q}_time"].iloc[0]) for q in QUANTITIES
    }
  return ExternalFit(
    fit_time=float(pred["fit_time"].iloc[0]),
    predictions=predictions,
    timings=timings,
    wall_time=wall_time,
  )
