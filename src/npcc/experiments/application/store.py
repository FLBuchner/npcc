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
        copulas/<model>/<arm>/ scores, timings, info.json, and draws where
                               the run stores them; or failure.json
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


def read_marginal_set(
  out: Path,
  station: int,
  model: str,
  *,
  train_pits: str = "in_sample",
  modeled: Sequence[str] | None = None,
  derived: Sequence[str] | None = None,
  derivation: Derivation | None = None,
) -> MarginalSet:
  """One station's saved predictions from one marginal model.

  Parameters
  ----------
  out : pathlib.Path
      The run folder.
  station : int
      The station id.
  model : str
      The marginal model.
  train_pits : {"in_sample", "out_of_fold"}, default="in_sample"
      Which training PITs the set carries: the stored ``pit`` of the
      training rows, or their ``pit_out_of_fold``.
  modeled : sequence of str, optional
      The responses the set carries, a subset of those the unit fitted, as
      a copula design names them; the unit's own list when omitted.
  derived : sequence of str, optional
      The responses derived from them; the unit's own list when ``modeled``
      is omitted, and none otherwise.
  derivation : Derivation, optional
      The derivations' Magnus formula; the unit's own when omitted.

  Returns
  -------
  MarginalSet
      What a copula arm reads, its responses in the data's order.

  Raises
  ------
  ValueError
      If ``train_pits`` is ``"out_of_fold"`` and the unit stored none, or
      ``modeled`` names a response the unit did not fit.
  """
  if train_pits not in ("in_sample", "out_of_fold"):
    raise ValueError(
      f"train_pits must be 'in_sample' or 'out_of_fold'; got {train_pits!r}"
    )
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
  fitted = tuple(info.get("modeled", data.variables))
  if modeled is None:
    modeled = fitted
    derived = tuple(info.get("derived", ()))
  else:
    stray = [v for v in modeled if v not in fitted]
    if stray:
      raise ValueError(
        f"modeled must be drawn from the responses station {station}'s "
        f"{model} unit fitted, {list(fitted)}; got {stray}"
      )
    modeled = tuple(v for v in data.variables if v in set(modeled))
    derived = tuple(v for v in data.variables if v in set(derived or ()))
  if derivation is None:
    derivation = Derivation(**info.get("derivation", {}))
  modeled_columns = [data.column(v) for v in modeled]
  derived_columns = [data.column(v) for v in derived]
  predictions = pd.read_parquet(unit / "predictions.parquet")
  if train_pits == "out_of_fold" and "pit_out_of_fold" not in predictions:
    raise ValueError(
      f"station {station}'s {model} unit must store out-of-fold training "
      f"PITs for train_pits = 'out_of_fold'; {unit} has none. Fit its "
      "margins with train_pits = 'out_of_fold'"
    )
  train_column = "pit" if train_pits == "in_sample" else "pit_out_of_fold"
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
      pit = train_column if split == "train" else "pit"
      per_split[f"pit_{split}"].append(part[pit].to_numpy(np.float64))
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
    train_pits=train_pits,
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
  save_draws: bool = False,
  nll_column: str | None = None,
  stored_draws: int | None = None,
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
  save_draws : bool, default=False
      Whether to also write ``draws.parquet``, the arm's draws
      (:func:`draws_frame`); a mixture member's are not stored.
  nll_column : str, optional
      The log score's column: ``joint_nll_<design>`` for an arm of another
      design of the copula stage. When omitted, ``joint_nll_modeled`` with
      derived responses, whose log score covers the modeled ones only, and
      ``joint_nll`` otherwise.
  stored_draws : int, optional
      With ``save_draws``, how many of each test case's draws to store, the
      first ones; all of them when omitted. The scores, computed before,
      use every draw. The count is recorded in ``info.json``.
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
  # is named so, apart from a log score of every response; another design's
  # is named for the design, so no table puts the two in one column.
  if nll_column is not None:
    nll = nll_column
  else:
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
    # Each response subset's scores and a projected log score, by column;
    # none may take a column of the scores above.
    extra = dict(getattr(result, "extra", {}))
    clash = sorted(set(extra) & set(columns))
    if clash:
      raise ValueError(
        f"{result.arm}'s extra scores must not reuse the columns {clash} of "
        f"its other scores; got them in both"
      )
    columns.update(extra)
    # A result scored by its log score alone, a chain mixture's order, has
    # no draws of its own: its sample scores are nan throughout.
    unsampled = (
      () if getattr(result, "sampled", True) else ("energy", "variogram")
    )
    for name, values in columns.items():
      values = np.asarray(values)
      if name in unsampled:
        if values.shape != (n_test,) or not np.isnan(values).all():
          raise ValueError(
            f"{result.arm}'s {name} must be nan with shape ({n_test},), as it "
            f"draws nothing of its own; got shape {values.shape}"
          )
        continue
      if values.shape != (n_test,) or not np.isfinite(values).all():
        raise ValueError(
          f"{result.arm}'s {name} must be finite with shape ({n_test},); got "
          f"shape {values.shape}"
        )
    # How many of each case's draws break a physical bound, by check; none
    # for a result without draws of its own.
    physical = {
      physical_column(k): np.asarray(v, dtype=np.float64)
      for k, v in getattr(result, "physical", {}).items()
    }
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
          **physical,
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
    "train_pits": ms.train_pits,
    # What the energy and variogram scores divide each response by.
    "scales": {
      v: float(s)
      for v, s in zip(
        (*ms.variables, *ms.derived),
        np.concatenate(
          [np.std(ms.y_train, axis=0), np.std(ms.y_derived_train, axis=0)]
        )
        if ms.derived
        else np.std(ms.y_train, axis=0),
        strict=True,
      )
    },
    **(
      {"modeled": list(ms.variables), "derived": list(ms.derived)}
      if ms.derived
      else {}
    ),
    "results": {
      r.arm: {"timings": dict(r.timings), "info": dict(r.info)} for r in results
    },
  }
  draws = (
    draws_frame(
      ms,
      arm,
      [r for r in results if r.arm == arm],
      rows=rows,
      keep=stored_draws,
    )
    if save_draws
    else None
  )
  if draws is not None:
    info["stored_draws"] = int(draws["draw"].max()) + 1
  with _unit(copula_dir(out, ms.station, ms.model, arm)) as temporary:
    pd.concat(frames, ignore_index=True).to_parquet(
      temporary / "scores.parquet", index=False
    )
    pd.DataFrame(timings).to_parquet(temporary / "timings.parquet", index=False)
    if draws is not None:
      draws.to_parquet(
        temporary / "draws.parquet", index=False, compression="zstd"
      )
    (temporary / "info.json").write_text(_json(info))


def physical_column(check: str) -> str:
  """The ``scores.parquet`` column of a physical check, e.g. ``broken_any``.

  Parameters
  ----------
  check : str
      A key of
      :data:`~npcc.experiments.application.derived.PHYSICAL_CHECKS`, or
      ``"any"``.

  Returns
  -------
  str
      ``broken_`` and the check.
  """
  return f"broken_{check}"


def draws_frame(
  ms: MarginalSet,
  arm: str,
  results: Sequence[CopulaResultLike],
  *,
  rows: np.ndarray,
  keep: int | None = None,
) -> pd.DataFrame:
  """Every sampled result's draws, one row per test case and draw.

  The columns are ``station``, ``model``, ``run_arm``, ``arm``, ``row`` (the
  test case's row in the input files, as in ``scores.parquet``), ``draw``
  (``0`` to ``n_draws - 1``), then one ``float32`` column per response,
  named as in ``y.nc``: the modeled ones, then the derived ones. A result
  that draws nothing of its own, a chain mixture's order, has no rows.

  Parameters
  ----------
  ms : MarginalSet
      The margins the arm was fitted on.
  arm : str
      The arm as run.
  results : sequence of CopulaResult
      What the arm returned; each sampled one carries ``draws``.
  rows : numpy.ndarray, shape (n_test,)
      The test cases' rows in the input files.
  keep : int, optional
      Only each case's first ``keep`` draws, ``0`` to ``keep - 1``; all of
      them when omitted.

  Returns
  -------
  pandas.DataFrame
      The draws, in result, case and draw order.

  Raises
  ------
  ValueError
      If a sampled result carries no draws, or draws of another shape than
      ``(n_test, n_draws, d + k)``.
  """
  variables = (*ms.variables, *ms.derived)
  n_test = len(ms.y_test)
  frames = []
  for result in results:
    if not getattr(result, "sampled", True):
      continue
    draws = getattr(result, "draws", None)
    if draws is None or draws.ndim != 3 or draws.shape[0] != n_test:
      raise ValueError(
        f"{result.arm}'s draws must have shape (n_test={n_test}, n_draws, "
        f"{len(variables)}) to be stored; got "
        f"{None if draws is None else draws.shape}"
      )
    if draws.shape[2] != len(variables):
      raise ValueError(
        f"{result.arm}'s draws must hold the {len(variables)} responses "
        f"{list(variables)}; got {draws.shape[2]}"
      )
    if keep is not None:
      if not 1 <= keep <= draws.shape[1]:
        raise ValueError(
          f"keep must lie between 1 and the {draws.shape[1]} draws per case "
          f"of {result.arm}; got {keep}"
        )
      draws = draws[:, :keep]
    n_draws = draws.shape[1]
    flat = draws.reshape(n_test * n_draws, len(variables)).astype(np.float32)
    frames.append(
      pd.DataFrame(
        {
          "station": np.int32(ms.station),
          "model": ms.model,
          "run_arm": arm,
          "arm": result.arm,
          "row": np.repeat(np.asarray(rows, dtype=np.int64), n_draws),
          "draw": np.tile(np.arange(n_draws, dtype=np.int32), n_test),
          **{v: flat[:, j] for j, v in enumerate(variables)},
        }
      )
    )
  if not frames:
    raise ValueError(
      f"{arm} must return a sampled result to store draws of; got "
      f"{[r.arm for r in results]}"
    )
  return pd.concat(frames, ignore_index=True)


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
