"""The run folder: what each stage writes, and how the next one reads it.

One folder per run::

    <out>/
      COMMIT, env.txt          the first process's checkout and environment
      provenance.jsonl         one line per process: when, what, which commit
      manifest.json            the configuration's signature
      config.json              the resolved configuration
      stations/<NNN>/
        data/                  cases.parquet: the station's cases
        margins/<model>/       predictions, scores, timings, info.json
        copulas/<model>/<arm>/ scores, timings, info.json; or failure.json
      tables/                  what ``evaluate`` writes

Each directory under ``stations/`` is a unit: written whole into a hidden
temporary directory beside it, then renamed into place, so a unit either
exists complete or not at all, and a process that stops mid-way leaves
nothing a later one could mistake for a result. Units are per station, so
processes running different stations never write the same path.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from npcc.experiments._study import code_provenance
from npcc.experiments.application.config import ApplicationConfig
from npcc.experiments.application.data import StationData, data_sha256
from npcc.experiments.application.derived import Derivation
from npcc.experiments.application.marginal_set import MarginalSet
from npcc.experiments.application.scores import sample_crps_column

_SPLITS = ("train", "test")


class CopulaResultLike(Protocol):
  """What the store reads from a copula arm's result."""

  @property
  def arm(self) -> str:
    """The arm's name."""
    ...

  @property
  def logcop_test(self) -> np.ndarray:
    """The copula log-density at each test case."""
    ...

  @property
  def joint_nll(self) -> np.ndarray:
    """The joint negative log-density at each test case."""
    ...

  @property
  def energy(self) -> np.ndarray:
    """The energy score at each test case."""
    ...

  @property
  def variogram(self) -> np.ndarray:
    """The variogram score at each test case."""
    ...

  @property
  def timings(self) -> Mapping[str, float]:
    """Seconds per stage."""
    ...

  @property
  def info(self) -> Mapping[str, object]:
    """JSON-able details of the fit."""
    ...

  @property
  def sample_crps(self) -> Mapping[str, np.ndarray]:
    """Each response's CRPS from the draws, when responses are derived."""
    ...

  @property
  def dew_point_above_share(self) -> np.ndarray | None:
    """Each test case's share of draws whose dew point exceeds T."""
    ...


# ---------------------------------------------------------------------------
# Paths and atomic writes
# ---------------------------------------------------------------------------


def station_dir(out: Path, station: int) -> Path:
  """The folder of one station's units.

  Parameters
  ----------
  out : pathlib.Path
      The run folder.
  station : int
      The station id.

  Returns
  -------
  pathlib.Path
      ``out/stations/<NNN>``.
  """
  return Path(out) / "stations" / f"{int(station):03d}"


def margin_dir(out: Path, station: int, model: str) -> Path:
  """The unit of one station's predictions from one marginal model.

  Parameters
  ----------
  out : pathlib.Path
      The run folder.
  station : int
      The station id.
  model : str
      The marginal model.

  Returns
  -------
  pathlib.Path
      ``out/stations/<NNN>/margins/<model>``.
  """
  return station_dir(out, station) / "margins" / model


def copula_dir(out: Path, station: int, model: str, arm: str) -> Path:
  """The unit of one copula arm on one station's marginal model.

  Parameters
  ----------
  out : pathlib.Path
      The run folder.
  station : int
      The station id.
  model : str
      The marginal model.
  arm : str
      The copula arm, as run.

  Returns
  -------
  pathlib.Path
      ``out/stations/<NNN>/copulas/<model>/<arm>``.
  """
  return station_dir(out, station) / "copulas" / model / arm


def _json(value: object) -> str:
  def default(obj: object) -> object:
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, np.generic):
      return obj.item()
    return str(obj)

  return json.dumps(value, indent=2, default=default) + "\n"


def _write_text(path: Path, text: str) -> None:
  """Write a file whole, by a rename from a temporary file beside it."""
  path.parent.mkdir(parents=True, exist_ok=True)
  handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
  try:
    with os.fdopen(handle, "w") as fh:
      fh.write(text)
    Path(name).replace(path)
  except BaseException:
    Path(name).unlink(missing_ok=True)
    raise


@contextmanager
def _unit(path: Path) -> Iterator[Path]:
  """A temporary directory, renamed to ``path`` once its block succeeds."""
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
  try:
    yield temporary
    temporary.chmod(0o755)
    temporary.rename(path)
  except OSError as exc:
    shutil.rmtree(temporary, ignore_errors=True)
    if path.exists():
      raise FileExistsError(
        f"{path} must not exist when a unit is written there; another "
        "process wrote it first"
      ) from exc
    raise
  except BaseException:
    shutil.rmtree(temporary, ignore_errors=True)
    raise


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _environment() -> str:
  """Python, platform, devices and every installed distribution."""
  import torch

  lines = [
    f"python {sys.version.split()[0]}",
    f"platform {platform.platform()}",
    f"torch {torch.__version__}, cuda {torch.version.cuda}",
  ]
  if torch.cuda.is_available():
    lines.append(f"gpu {torch.cuda.get_device_name(0)}")
  try:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
      if line.startswith("model name"):
        lines.append(f"cpu {line.split(':', 1)[1].strip()}")
        break
  except OSError:
    pass
  lines.append("")
  lines.extend(
    sorted(
      f"{d.metadata['Name']}=={d.version}"
      for d in metadata.distributions()
      if d.metadata["Name"]
    )
  )
  return "\n".join(lines) + "\n"


def start_run(
  out: Path, config: ApplicationConfig, options: Mapping[str, object]
) -> None:
  """Open a run folder for a process, or refuse one another config started.

  Writes ``manifest.json`` and ``config.json``, ``COMMIT`` and ``env.txt``
  when absent, and appends this process's line to ``provenance.jsonl``,
  which records the SHA-256 of each input file it finds.

  Parameters
  ----------
  out : pathlib.Path
      The run folder; created if absent.
  config : ApplicationConfig
      The configuration this process runs.
  options : mapping
      The command line, for the provenance line.
  """
  out = Path(out)
  out.mkdir(parents=True, exist_ok=True)
  signature = config.signature()
  manifest = out / "manifest.json"
  if manifest.exists():
    previous = json.loads(manifest.read_text()).get("signature")
    if previous != signature:
      raise ValueError(
        f"the configuration's signature must match the one {out} was "
        f"started with, since its results were computed under it; got "
        f"{signature[:12]}, not {str(previous)[:12]}. Use a fresh --out."
      )
  else:
    _write_text(manifest, _json({"signature": signature}))
  _write_text(out / "config.json", _json(config.echo()))
  provenance = code_provenance()
  if not (out / "COMMIT").exists():
    changed = provenance.get("uncommitted") or []
    _write_text(
      out / "COMMIT",
      f"HEAD {provenance.get('npcc_commit')}\n"
      "uncommitted (empty = clean):\n"
      + "".join(f"  {name}\n" for name in changed),
    )
  if not (out / "env.txt").exists():
    _write_text(out / "env.txt", _environment())
  record = {
    "started": datetime.now().astimezone().isoformat(timespec="seconds"),
    "pid": os.getpid(),
    "options": dict(options),
    "data_dir": str(config.data_dir),
    "data_sha256": data_sha256(config.data_dir),
    **provenance,
  }
  with (out / "provenance.jsonl").open("a") as fh:
    fh.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Station data
# ---------------------------------------------------------------------------


def _cases_frame(data: StationData) -> pd.DataFrame:
  parts = []
  for split in _SPLITS:
    x = getattr(data, f"x_{split}")
    y = getattr(data, f"y_{split}")
    frame = pd.DataFrame(
      {
        "split": split,
        "row": getattr(data, f"row_{split}"),
        "time": getattr(data, f"time_{split}"),
        **{f"x:{n}": x[:, j] for j, n in enumerate(data.feature_names)},
        **{f"y:{n}": y[:, j] for j, n in enumerate(data.variables)},
      }
    )
    parts.append(frame)
  return pd.concat(parts, ignore_index=True)


def read_station_data(out: Path, station: int) -> StationData:
  """A station's cases, as stored by :func:`ensure_station_data`.

  Parameters
  ----------
  out : pathlib.Path
      The run folder.
  station : int
      The station id.

  Returns
  -------
  StationData
      The cases the station's margins were fitted and scored on.
  """
  path = station_dir(out, station) / "data" / "cases.parquet"
  if not path.exists():
    raise FileNotFoundError(
      f"station {station}'s cases must be stored in {path.parent} before "
      "they are read; run `margins` for it first"
    )
  frame = pd.read_parquet(path)
  features = tuple(c[2:] for c in frame.columns if c.startswith("x:"))
  variables = tuple(c[2:] for c in frame.columns if c.startswith("y:"))
  arrays: dict[str, np.ndarray] = {}
  for split in _SPLITS:
    part = frame.loc[frame["split"] == split]
    arrays[f"x_{split}"] = part[[f"x:{n}" for n in features]].to_numpy(
      dtype=np.float64
    )
    arrays[f"y_{split}"] = part[[f"y:{n}" for n in variables]].to_numpy(
      dtype=np.float64
    )
    arrays[f"time_{split}"] = part["time"].to_numpy(dtype="datetime64[ns]")
    arrays[f"row_{split}"] = part["row"].to_numpy(dtype=np.int64)
  return StationData(
    station=int(station),
    feature_names=features,
    variables=variables,
    **arrays,
  )


def _same_data(a: StationData, b: StationData) -> list[str]:
  """The fields on which two stations' cases differ."""
  differ = [
    name
    for name in ("feature_names", "variables")
    if getattr(a, name) != getattr(b, name)
  ]
  if differ:
    return differ
  return [
    f"{kind}_{split}"
    for kind in ("x", "y", "time", "row")
    for split in _SPLITS
    if not np.array_equal(
      getattr(a, f"{kind}_{split}"), getattr(b, f"{kind}_{split}")
    )
  ]


def ensure_station_data(
  out: Path, data: StationData, *, seconds: float
) -> None:
  """Store a station's cases, or check they are the ones stored.

  Parameters
  ----------
  out : pathlib.Path
      The run folder.
  data : StationData
      The cases, as just loaded.
  seconds : float
      How long loading them took, recorded beside them.

  Raises
  ------
  ValueError
      If the stored cases differ from ``data``: the input files changed
      under results computed from them.
  """
  unit = station_dir(out, data.station) / "data"
  if unit.exists():
    differ = _same_data(read_station_data(out, data.station), data)
    if differ:
      raise ValueError(
        f"station {data.station}'s cases must be the ones stored in {unit}, "
        f"which its results were computed on; {differ} differ. The input "
        "data changed: use a fresh --out."
      )
    return
  with _unit(unit) as temporary:
    _cases_frame(data).to_parquet(temporary / "cases.parquet", index=False)
    (temporary / "info.json").write_text(
      _json({"station": data.station, "load_seconds": seconds})
    )


# ---------------------------------------------------------------------------
# Margins
# ---------------------------------------------------------------------------


def quantile_columns(n_levels: int) -> list[str]:
  """The column names of ``n_levels`` quantiles: ``q_000``, ``q_001``, ...

  Parameters
  ----------
  n_levels : int
      The number of levels.

  Returns
  -------
  list of str
      Zero-padded to at least three digits.
  """
  width = max(3, len(str(n_levels - 1)))
  return [f"q_{k:0{width}d}" for k in range(n_levels)]


def write_margin(
  out: Path,
  station: int,
  model: str,
  *,
  predictions: pd.DataFrame,
  scores: pd.DataFrame,
  timings: pd.DataFrame,
  info: Mapping[str, object],
) -> None:
  """Write one station's predictions from one marginal model, as a unit.

  Parameters
  ----------
  out : pathlib.Path
      The run folder.
  station : int
      The station id.
  model : str
      The marginal model.
  predictions : pandas.DataFrame
      One row per ``(variable, split, row)``, in case order within each:
      ``pit``, and on test rows ``logpdf`` and the quantile columns.
  scores : pandas.DataFrame
      One row per ``(variable, row)`` of the test split.
  timings : pandas.DataFrame
      One row per ``(variable, stage)``, with ``seconds``.
  info : mapping
      JSON-able: ``levels``, seeds, the estimator, fitted parameters.
  """
  with _unit(margin_dir(out, station, model)) as temporary:
    predictions.to_parquet(temporary / "predictions.parquet", index=False)
    scores.to_parquet(temporary / "scores.parquet", index=False)
    timings.to_parquet(temporary / "timings.parquet", index=False)
    (temporary / "info.json").write_text(_json(dict(info)))


def read_marginal_set(out: Path, station: int, model: str) -> MarginalSet:
  """One station's saved predictions from one marginal model.

  Parameters
  ----------
  out : pathlib.Path
      The run folder.
  station : int
      The station id.
  model : str
      The marginal model.

  Returns
  -------
  MarginalSet
      What a copula arm reads.
  """
  unit = margin_dir(out, station, model)
  if not unit.exists():
    raise FileNotFoundError(
      f"station {station}'s {model} predictions must be in {unit} before a "
      "copula reads them; run `margins` for them first"
    )
  data = read_station_data(out, station)
  info = json.loads((unit / "info.json").read_text())
  levels = np.asarray(info["levels"], dtype=np.float64)
  # A unit of the default design records neither list: it models every
  # response and derives none.
  modeled = tuple(info.get("modeled", data.variables))
  derived = tuple(info.get("derived", ()))
  derivation = Derivation(**info.get("derivation", {}))
  modeled_columns = [data.column(v) for v in modeled]
  derived_columns = [data.column(v) for v in derived]
  predictions = pd.read_parquet(unit / "predictions.parquet")
  columns = quantile_columns(len(levels))
  per_split: dict[str, list[np.ndarray]] = {"pit_train": [], "pit_test": []}
  logpdf: list[np.ndarray] = []
  quantiles: list[np.ndarray] = []
  for variable in modeled:
    rows = predictions.loc[predictions["variable"] == variable]
    for split in _SPLITS:
      part = rows.loc[rows["split"] == split]
      if not np.array_equal(
        part["row"].to_numpy(), getattr(data, f"row_{split}")
      ):
        raise ValueError(
          f"station {station}'s {model} {split} predictions of {variable} "
          f"must align with its stored cases; they do not"
        )
      per_split[f"pit_{split}"].append(part["pit"].to_numpy(np.float64))
      if split == "test":
        logpdf.append(part["logpdf"].to_numpy(np.float64))
        quantiles.append(part[columns].to_numpy(np.float64))
  return MarginalSet(
    station=int(station),
    model=model,
    variables=modeled,
    feature_names=data.feature_names,
    x_train=data.x_train,
    x_test=data.x_test,
    y_train=data.y_train[:, modeled_columns],
    y_test=data.y_test[:, modeled_columns],
    time_train=data.time_train,
    time_test=data.time_test,
    pit_train=np.column_stack(per_split["pit_train"]),
    pit_test=np.column_stack(per_split["pit_test"]),
    logpdf_test=np.column_stack(logpdf),
    levels=levels,
    quantiles_test=np.stack(quantiles, axis=2),
    derived=derived,
    derivation=derivation,
    y_derived_train=data.y_train[:, derived_columns],
    y_derived_test=data.y_test[:, derived_columns],
  )


def stored_stations(out: Path) -> tuple[int, ...]:
  """Every station with stored cases.

  Parameters
  ----------
  out : pathlib.Path
      The run folder.

  Returns
  -------
  tuple of int
      Increasing station ids.
  """
  root = Path(out) / "stations"
  if not root.is_dir():
    return ()
  return tuple(
    sorted(
      int(p.name)
      for p in root.iterdir()
      if p.name.isdigit() and (p / "data").is_dir()
    )
  )


def _read_units(paths: Iterable[Path], name: str) -> pd.DataFrame:
  frames = [pd.read_parquet(p / name) for p in paths]
  frames = [f for f in frames if not f.empty]
  return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_margin_table(
  out: Path, stations: Sequence[int], models: Sequence[str], table: str
) -> pd.DataFrame:
  """One table of every selected margin unit, stacked.

  Parameters
  ----------
  out : pathlib.Path
      The run folder.
  stations : sequence of int
      The stations.
  models : sequence of str
      The marginal models.
  table : str
      ``"scores"``, ``"timings"`` or ``"predictions"``.

  Returns
  -------
  pandas.DataFrame
      Their rows, each with ``station`` and ``model``.

  Raises
  ------
  FileNotFoundError
      If a selected station has no unit for a selected model, which would
      leave the models scored on different cases.
  """
  missing = [
    (s, m)
    for s in stations
    for m in models
    if not margin_dir(out, s, m).is_dir()
  ]
  if missing:
    raise FileNotFoundError(
      f"every selected station must have every selected marginal model, so "
      f"they are compared on the same cases; missing (station, model) "
      f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
    )
  return _read_units(
    (margin_dir(out, s, m) for s in stations for m in models),
    f"{table}.parquet",
  )


# ---------------------------------------------------------------------------
# Copulas
# ---------------------------------------------------------------------------


def copula_status(out: Path, station: int, model: str, arm: str) -> str | None:
  """Whether a copula unit finished: ``"done"``, ``"failed"`` or ``None``.

  Parameters
  ----------
  out : pathlib.Path
      The run folder.
  station : int
      The station id.
  model : str
      The marginal model.
  arm : str
      The copula arm.

  Returns
  -------
  str or None
      ``None`` where no unit exists.
  """
  unit = copula_dir(out, station, model, arm)
  if not unit.is_dir():
    return None
  return "failed" if (unit / "failure.json").exists() else "done"


def write_copula(
  out: Path,
  ms: MarginalSet,
  arm: str,
  results: Sequence[CopulaResultLike],
  *,
  rows: np.ndarray,
  seed: int,
  wall_seconds: float,
) -> None:
  """Write one copula arm's results on one marginal set, as a unit.

  Parameters
  ----------
  out : pathlib.Path
      The run folder.
  ms : MarginalSet
      The margins the arm was fitted on.
  arm : str
      The arm as run; its results may name more arms, such as a mixture's
      members.
  results : sequence of CopulaResult
      What the arm returned.
  rows : numpy.ndarray, shape (n_test,)
      The test cases' rows in the input files.
  seed : int
      The seed it was given.
  wall_seconds : float
      How long the call took.
  """
  n_test = len(ms.y_test)
  if np.shape(rows) != (n_test,):
    raise ValueError(
      f"rows must name each of the {n_test} test cases; got shape "
      f"{np.shape(rows)}"
    )
  frames, timings = [], []
  names = [r.arm for r in results]
  if not results or len(set(names)) != len(names):
    raise ValueError(
      f"{arm} must return results with distinct arm names; got {names}"
    )
  # With derived responses, the log score covers the modeled ones only and
  # is named so, apart from a log score of every response.
  nll = "joint_nll_modeled" if ms.derived else "joint_nll"
  for result in results:
    columns = {
      "logcop": result.logcop_test,
      nll: result.joint_nll,
      "energy": result.energy,
      "variogram": result.variogram,
      **{
        sample_crps_column(v): values
        for v, values in result.sample_crps.items()
      },
    }
    if result.dew_point_above_share is not None:
      columns["dew_point_above_share"] = result.dew_point_above_share
    for name, values in columns.items():
      values = np.asarray(values)
      if values.shape != (n_test,) or not np.isfinite(values).all():
        raise ValueError(
          f"{result.arm}'s {name} must be finite with shape ({n_test},); got "
          f"shape {values.shape}"
        )
    frames.append(
      pd.DataFrame(
        {
          "station": ms.station,
          "model": ms.model,
          "run_arm": arm,
          "arm": result.arm,
          "row": rows,
          "time": ms.time_test,
          **{k: np.asarray(v, dtype=np.float64) for k, v in columns.items()},
        }
      )
    )
    timings.extend(
      {
        "station": ms.station,
        "model": ms.model,
        "run_arm": arm,
        "arm": result.arm,
        "stage": stage,
        "seconds": float(seconds),
      }
      for stage, seconds in result.timings.items()
    )
  info = {
    "station": ms.station,
    "model": ms.model,
    "arm": arm,
    "seed": seed,
    "wall_seconds": wall_seconds,
    **(
      {"modeled": list(ms.variables), "derived": list(ms.derived)}
      if ms.derived
      else {}
    ),
    "results": {
      r.arm: {"timings": dict(r.timings), "info": dict(r.info)} for r in results
    },
  }
  with _unit(copula_dir(out, ms.station, ms.model, arm)) as temporary:
    pd.concat(frames, ignore_index=True).to_parquet(
      temporary / "scores.parquet", index=False
    )
    pd.DataFrame(timings).to_parquet(temporary / "timings.parquet", index=False)
    (temporary / "info.json").write_text(_json(info))


def write_copula_failure(
  out: Path,
  ms: MarginalSet,
  arm: str,
  error: Exception,
  *,
  rows: np.ndarray,
  seed: int,
) -> None:
  """Record that an arm could not be scored on a marginal set, as a unit.

  Parameters
  ----------
  out : pathlib.Path
      The run folder.
  ms : MarginalSet
      The margins the arm was fitted on.
  arm : str
      The arm.
  error : Exception
      What it raised. An ``indices`` attribute, if any, names the test cases
      at fault.
  rows : numpy.ndarray, shape (n_test,)
      The test cases' rows in the input files.
  seed : int
      The seed it was given.
  """
  indices = [int(i) for i in getattr(error, "indices", ())]
  record = {
    "station": ms.station,
    "model": ms.model,
    "arm": arm,
    "seed": seed,
    "error_type": type(error).__name__,
    "reason": str(error),
    "cases": [
      {"row": int(rows[i]), "time": str(ms.time_test[i])} for i in indices
    ],
  }
  with _unit(copula_dir(out, ms.station, ms.model, arm)) as temporary:
    (temporary / "failure.json").write_text(_json(record))


def read_copula_tables(
  out: Path,
  stations: Sequence[int],
  models: Sequence[str],
  arms: Sequence[str],
) -> dict[str, pd.DataFrame]:
  """Every selected copula unit's scores, timings and failures.

  Parameters
  ----------
  out : pathlib.Path
      The run folder.
  stations : sequence of int
      The stations.
  models : sequence of str
      The marginal models.
  arms : sequence of str
      The arms as run.

  Returns
  -------
  dict of str to pandas.DataFrame
      ``scores`` (one row per result arm and test case), ``timings`` and
      ``failures`` (one row per failed unit).

  Raises
  ------
  FileNotFoundError
      If a selected unit is missing, neither done nor failed.
  """
  done, failures, missing = [], [], []
  for station in stations:
    for model in models:
      for arm in arms:
        status = copula_status(out, station, model, arm)
        unit = copula_dir(out, station, model, arm)
        if status == "done":
          done.append(unit)
        elif status == "failed":
          failures.append(json.loads((unit / "failure.json").read_text()))
        else:
          missing.append((station, model, arm))
  if missing:
    raise FileNotFoundError(
      f"every selected copula unit must have run; missing (station, model, "
      f"arm) {missing[:10]}{' ...' if len(missing) > 10 else ''}"
    )
  return {
    "scores": _read_units(done, "scores.parquet"),
    "timings": _read_units(done, "timings.parquet"),
    "failures": pd.DataFrame(
      [
        {k: v for k, v in f.items() if k != "cases"}
        | {"n_cases": len(f["cases"])}
        for f in failures
      ],
      columns=[
        "station",
        "model",
        "arm",
        "seed",
        "error_type",
        "reason",
        "n_cases",
      ],
    ),
  }
