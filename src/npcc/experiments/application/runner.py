"""The four stages of an application run.

- :func:`preflight` reads every selected station and checks the backends,
  the copula arms and, when an arm needs it, R and boostCopula, fitting
  nothing.
- :func:`run_margins` fits every selected marginal model to every response
  of every selected station and stores the predictions and scores.
- :func:`run_copulas` fits every selected copula arm to every stored
  marginal set. An arm that raises
  :class:`~npcc.experiments.application.copulas.CopulaFailure` is recorded
  as failed and the run goes on; any other error stops it.
- :func:`evaluate` writes the aggregate tables under ``<out>/tables``.

The two fitting stages are one process each, one station after another, with
no concurrent GPU work; ``shard = (i, k)`` keeps every ``k``-th station from
the ``i``-th, so ``k`` processes split the stations between them. Stations
are increasing, so every process agrees on the split.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from time import perf_counter

import pandas as pd

from npcc.core.errors import MissingBackendDependencyError
from npcc.experiments._study import derive_seed, release_gpu
from npcc.experiments.application.config import ApplicationConfig
from npcc.experiments.application.data import (
  feature_columns,
  load_station,
  station_ids,
)
from npcc.experiments.application.derived import residuals
from npcc.experiments.application.margins import build_predictor, fit_station
from npcc.experiments.application.scores import (
  copula_tables,
  marginal_tables,
  summarize_timings,
)
from npcc.experiments.application.store import (
  copula_dir,
  copula_status,
  ensure_station_data,
  margin_dir,
  read_copula_tables,
  read_margin_table,
  read_marginal_set,
  read_station_data,
  start_run,
  station_dir,
  stored_stations,
  write_copula,
  write_copula_failure,
  write_margin,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunOptions:
  """What the command line chooses on top of the configuration.

  Attributes
  ----------
  out : pathlib.Path or None
      The run folder; ``preflight`` needs none.
  stations : tuple of int or None
      The stations; ``None`` for the configuration's.
  models : tuple of str or None
      The marginal models, from the configuration's; ``None`` for all.
  arms : tuple of str or None
      The copula arms, from the configuration's; ``None`` for all.
  shard : tuple of int or None
      ``(i, k)``: only every ``k``-th station, from the ``i``-th.
  resume : bool
      Skip the units a run of the same folder already wrote.
  device : str or None
      The device of every backend, over the configuration's.
  """

  out: Path | None = None
  stations: tuple[int, ...] | None = None
  models: tuple[str, ...] | None = None
  arms: tuple[str, ...] | None = None
  shard: tuple[int, int] | None = None
  resume: bool = False
  device: str | None = None

  def __post_init__(self) -> None:
    """Refuse an impossible shard."""
    if self.shard is not None:
      index, count = self.shard
      if not 0 <= index < count:
        raise ValueError(
          f"shard must be I/K with 0 <= I < K; got {index}/{count}"
        )

  def echo(self) -> dict[str, object]:
    """The options, as JSON-able values.

    Returns
    -------
    dict
        Every field.
    """
    return {
      "out": None if self.out is None else str(self.out),
      "stations": None if self.stations is None else list(self.stations),
      "models": None if self.models is None else list(self.models),
      "arms": None if self.arms is None else list(self.arms),
      "shard": None if self.shard is None else list(self.shard),
      "resume": self.resume,
      "device": self.device,
    }


def _out(options: RunOptions) -> Path:
  if options.out is None:
    raise ValueError("out must name the run folder for this stage; got None")
  return options.out


def _subset(
  name: str, chosen: Sequence[str] | None, allowed: Sequence[str]
) -> tuple[str, ...]:
  """The chosen names, in the configuration's order; all of them if none."""
  if chosen is None:
    return tuple(allowed)
  unknown = sorted(set(chosen) - set(allowed))
  if unknown:
    raise ValueError(
      f"{name} must be drawn from the configuration's {list(allowed)}; got "
      f"{unknown}"
    )
  return tuple(a for a in allowed if a in set(chosen))


def _stations(
  config: ApplicationConfig, options: RunOptions, *, from_data: bool
) -> tuple[int, ...]:
  """The stations of this process: chosen, configured, or every one there is.

  Every one there is means every station in the data for ``from_data``, and
  every station with stored cases otherwise.
  """
  if options.stations is not None:
    stations = tuple(sorted(set(options.stations)))
  elif config.stations is not None:
    stations = tuple(sorted(config.stations))
  elif from_data:
    stations = station_ids(config.data_dir)
  else:
    stations = stored_stations(_out(options))
  if options.shard is not None:
    index, count = options.shard
    stations = stations[index::count]
  return stations


def _models(config: ApplicationConfig, options: RunOptions) -> tuple[str, ...]:
  return _subset("models", options.models, config.models)


def _copula_models(
  config: ApplicationConfig, options: RunOptions
) -> tuple[str, ...]:
  return _subset("models", options.models, config.copula_models)


def _arms(config: ApplicationConfig, options: RunOptions) -> tuple[str, ...]:
  return _subset("copulas", options.arms, config.copula_arms)


def _arms_by_model(
  config: ApplicationConfig, models: Sequence[str], arms: Sequence[str]
) -> dict[str, tuple[str, ...]]:
  """The selected arms each selected model runs, as the configuration allows.

  A model left with none is dropped, so its marginal sets are not read.
  """
  chosen = {
    m: tuple(a for a in arms if a in config.copula_arms_for(m)) for m in models
  }
  return {m: a for m, a in chosen.items() if a}


def _read_copula_units(
  out: Path, stations: Sequence[int], units: Mapping[str, Sequence[str]]
) -> dict[str, pd.DataFrame]:
  """:func:`read_copula_tables` over each model's own arms, stacked."""
  parts = [read_copula_tables(out, stations, [m], a) for m, a in units.items()]
  stacked: dict[str, pd.DataFrame] = {}
  for key in parts[0]:
    frames = [p[key] for p in parts if not p[key].empty]
    stacked[key] = (
      pd.concat(frames, ignore_index=True) if frames else parts[0][key]
    )
  return stacked


def _check_arms(arms: Sequence[str]) -> None:
  """Refuse an arm the copula stage does not know, before fitting any."""
  from npcc.experiments.application.copulas import COPULA_ARMS

  unknown = [a for a in arms if a not in COPULA_ARMS]
  if unknown:
    raise ValueError(
      f"copula arms must be drawn from {list(COPULA_ARMS)}; got {unknown}"
    )


def check_boostcopula() -> str:
  """Run the copula stage's boostCopula check, which fits nothing.

  Returns
  -------
  str
      R's and each package's version.
  """
  from npcc.experiments.application.copulas import (
    check_boostcopula as check,
  )

  return check()


def _device(
  config: ApplicationConfig, options: RunOptions, name: str
) -> str | None:
  return (
    options.device if options.device is not None else config.device_for(name)
  )


def preflight(config: ApplicationConfig, options: RunOptions) -> list[str]:
  """Check the data, the estimators and the copula stage; fit nothing.

  Parameters
  ----------
  config : ApplicationConfig
      The configuration.
  options : RunOptions
      The selection.

  Returns
  -------
  list of str
      One line per check, for the command to print.
  """
  lines: list[str] = []
  for name in ("x.nc", "y.nc"):
    if not (config.data_dir / name).is_file():
      raise FileNotFoundError(
        f"data.dir must hold {name}; got {str(config.data_dir)!r}, resolved "
        f"against {Path.cwd()}. Fetch the data with `npcc-application fetch`."
      )
  models = [config.marginal_model(m) for m in _models(config, options)]
  arms = _arms(config, options)
  for station in _stations(config, options, from_data=True):
    data = load_station(
      config.data_dir, station, config.selection, sha256=config.data_sha256
    )
    missing = sorted(set(data.variables) - set(config.responses))
    absent = sorted(set(config.responses) - set(data.variables))
    if missing or absent:
      raise ValueError(
        f"responses must be exactly station {station}'s "
        f"{list(data.variables)}; "
        f"missing {missing}, absent from the data {absent}"
      )
    for variable in data.variables:
      config.responses[variable].to_model(
        data.y_train[:, data.column(variable)], variable
      )
      config.responses[variable].to_model(
        data.y_test[:, data.column(variable)], variable
      )
      for model in models:
        feature_columns(data.feature_names, model.scope, variable)
    feature_columns(data.feature_names, "time")
    if config.derived:
      found = residuals(
        data.y_train[:, [data.column(v) for v in config.modeled]],
        config.modeled,
        config.derived,
        data.y_train[:, [data.column(v) for v in config.derived]],
        config.derivation,
      )
      lines.extend(
        f"station {station}: derived {name} against the observed, training "
        f"residual sd {stats['sd']:.3g}, max {stats['max_abs']:.3g}"
        for name, stats in found.items()
      )
    lines.append(
      f"station {station}: {len(data.row_train)} training and "
      f"{len(data.row_test)} test cases, {len(data.feature_names)} covariates, "
      f"{len(data.variables)} responses"
    )
  for model in models:
    if model.backend is None:
      continue
    try:
      build_predictor(
        model,
        next(iter(config.responses.values())),
        config=config,
        feature_names=[],
        device="cpu",
      )
    except MissingBackendDependencyError as exc:
      raise RuntimeError(f"{model.name} cannot be built: {exc}") from exc
    lines.append(
      f"{model.name}: {config.estimators_by_backend[model.backend].label} on "
      f"{_device(config, options, model.backend) or 'its default device'}"
    )
  if arms:
    try:
      _check_arms(arms)
    except ImportError as exc:
      raise RuntimeError(
        f"the copula stage must be importable to check the arms {list(arms)}; "
        f"got {exc}"
      ) from exc
    lines.append(f"copula arms: {', '.join(arms)}")
    lines.append(f"copula margins: {', '.join(config.copula_models)}")
    lines.extend(
      f"copula arms on {m}: {', '.join(config.copula_arms_for(m))}"
      for m in config.copula_models
      if m in config.copula_arms_by_margin
    )
    if any(a.startswith("boostcopula") for a in arms):
      lines.append(f"boostCopula: {check_boostcopula()}")
  return lines


def run_margins(config: ApplicationConfig, options: RunOptions) -> None:
  """Fit and store every selected marginal model at every selected station.

  Parameters
  ----------
  config : ApplicationConfig
      The configuration.
  options : RunOptions
      The run folder and selection.
  """
  out = _out(options)
  models = [config.marginal_model(m) for m in _models(config, options)]
  stations = _stations(config, options, from_data=True)
  if not options.resume:
    done = [
      str(margin_dir(out, s, m.name))
      for s in stations
      for m in models
      if margin_dir(out, s, m.name).exists()
    ]
    if done:
      raise FileExistsError(
        f"the margins must not be stored yet without --resume; found "
        f"{done[:5]}{' ...' if len(done) > 5 else ''}"
      )
  start_run(out, config, {"stage": "margins", **options.echo()})
  logger.info("margins: %d stations, %d models", len(stations), len(models))
  for station in stations:
    start = perf_counter()
    data = load_station(
      config.data_dir, station, config.selection, sha256=config.data_sha256
    )
    ensure_station_data(out, data, seconds=perf_counter() - start)
    for model in models:
      if margin_dir(out, station, model.name).exists():
        logger.info("station %d: %s already stored", station, model.name)
        continue
      device = (
        None
        if model.backend is None
        else _device(config, options, model.backend)
      )
      margins = fit_station(data, model, config=config, device=device)
      write_margin(
        out,
        station,
        model.name,
        predictions=margins.predictions,
        scores=margins.scores,
        timings=margins.timings,
        info=margins.info,
      )
      logger.info("station %d: %s stored", station, model.name)
      release_gpu()


def run_copulas(config: ApplicationConfig, options: RunOptions) -> None:
  """Fit and store every selected copula arm on every selected marginal set.

  Parameters
  ----------
  config : ApplicationConfig
      The configuration.
  options : RunOptions
      The run folder and selection.
  """
  from npcc.experiments.application.copulas import (
    CopulaFailure,
    run_copula_arm,
  )

  out = _out(options)
  arms = _arms(config, options)
  _check_arms(arms)
  units = _arms_by_model(config, _copula_models(config, options), arms)
  stations = _stations(config, options, from_data=False)
  if not options.resume:
    done = [
      str(copula_dir(out, s, m, a))
      for s in stations
      for m, model_arms in units.items()
      for a in model_arms
      if copula_status(out, s, m, a) is not None
    ]
    if done:
      raise FileExistsError(
        f"the copula arms must not be stored yet without --resume; found "
        f"{done[:5]}{' ...' if len(done) > 5 else ''}"
      )
  if any(a.startswith("boostcopula") for a in chain(*units.values())):
    check_boostcopula()
  # The copula stage derives each fit's seed from this one, the station, the
  # marginal model and the arm; its random structures from this one and the
  # station alone, so they are shared across marginal models and backends.
  seed = derive_seed(config.base_seed, "copulas")
  start_run(out, config, {"stage": "copulas", **options.echo()})
  logger.info(
    "copulas: %d stations, %d models, %d arms on them",
    len(stations),
    len(units),
    sum(len(a) for a in units.values()),
  )
  for station in stations:
    rows = read_station_data(out, station).row_test
    for model, model_arms in units.items():
      ms = read_marginal_set(out, station, model)
      for arm in model_arms:
        status = copula_status(out, station, model, arm)
        if status is not None:
          logger.info(
            "station %d: %s %s already %s", station, model, arm, status
          )
          continue
        start = perf_counter()
        try:
          results = run_copula_arm(
            arm,
            ms,
            estimators=config.estimators_by_backend,
            n_draws=config.n_draws,
            sample_batch_size=config.sample_batch_size,
            mixture_k=config.mixture_k,
            seed=seed,
            device=_device(config, options, arm),
          )
        except CopulaFailure as error:
          write_copula_failure(out, ms, arm, error, rows=rows, seed=seed)
          logger.warning(
            "station %d: %s %s failed and is recorded: %s",
            station,
            model,
            arm,
            error,
          )
        else:
          write_copula(
            out,
            ms,
            arm,
            results,
            rows=rows,
            seed=seed,
            wall_seconds=perf_counter() - start,
          )
          logger.info("station %d: %s %s stored", station, model, arm)
        release_gpu()


def _bounds(
  out: Path, stations: Sequence[int], models: Sequence[str]
) -> pd.DataFrame:
  """The cases at a bound of the unit interval, where a margin counted them.

  They are the cases the response's boundary nudge moves inside it, such as
  RH = 100 moved to 99.95, counted per station, split and marginal model.
  """
  keys = ("at_upper_train", "at_upper_test", "at_lower_train", "at_lower_test")
  rows = []
  for station in stations:
    for model in models:
      info = json.loads(
        (margin_dir(out, station, model) / "info.json").read_text()
      )
      for variable, detail in info["variables"].items():
        if "at_upper_train" in detail:
          rows.append(
            {
              "station": station,
              "model": model,
              "variable": variable,
              **{k: detail[k] for k in keys},
            }
          )
  return pd.DataFrame(rows)


def _residuals(
  config: ApplicationConfig, out: Path, stations: Sequence[int]
) -> pd.DataFrame:
  """The derivation against each station's observed training responses."""
  if not config.derived:
    return pd.DataFrame()
  rows = []
  for station in stations:
    data = read_station_data(out, station)
    modeled = [data.column(v) for v in config.modeled]
    derived = [data.column(v) for v in config.derived]
    found = residuals(
      data.y_train[:, modeled],
      config.modeled,
      config.derived,
      data.y_train[:, derived],
      config.derivation,
    )
    rows.extend(
      {"station": station, "variable": name, **stats}
      for name, stats in found.items()
    )
  return pd.DataFrame(rows)


def _write_csv(tables: Mapping[str, pd.DataFrame], directory: Path) -> None:
  directory.mkdir(parents=True, exist_ok=True)
  for name, frame in tables.items():
    path = directory / f"{name}.csv"
    temporary = path.with_name(f".{path.name}.partial")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def evaluate(config: ApplicationConfig, options: RunOptions) -> list[str]:
  """Write the aggregate tables of a run folder, fitting nothing.

  Margins: per station, pooled and over stations, each model with its CRPS
  skill and log-score gain on seasonal EMOS; and their timings. Copulas,
  where the copula stage has run: the same for each arm against
  independence on its own margins, with every failed unit, and the timings.
  An arm that failed at any selected station is left out of every copula
  table but the failures, so the arms compared cover the same cases.

  Parameters
  ----------
  config : ApplicationConfig
      The configuration.
  options : RunOptions
      The run folder and selection.

  Returns
  -------
  list of str
      The tables written.
  """
  out = _out(options)
  stations = _stations(config, options, from_data=False)
  if not stations:
    raise FileNotFoundError(f"{out} must hold stored stations to evaluate")
  models = _models(config, options)
  seed = config.base_seed
  scores = read_margin_table(out, stations, models, "scores")
  margin = marginal_tables(scores, seed=seed)
  timings = read_margin_table(out, stations, models, "timings")
  tables: dict[str, pd.DataFrame] = {
    "margins_by_station": margin["by_station"],
    "margins_pooled": margin["pooled"],
    "margins_station_mean": margin["station_mean"],
    "margins_timings": summarize_timings(timings, ["model"]),
    "margins_timings_by_station": timings.groupby(
      ["station", "model", "stage"], as_index=False
    )["seconds"].sum(),
  }
  bounds = _bounds(out, stations, models)
  if not bounds.empty:
    tables["bounds_by_station"] = bounds
  residual = _residuals(config, out, stations)
  if not residual.empty:
    tables["derivation_residuals"] = residual
  copula_models = [m for m in config.copula_models if m in models]
  ran = any(
    (station_dir(out, s) / "copulas" / m).is_dir()
    for s in stations
    for m in copula_models
  )
  units = _arms_by_model(config, copula_models, _arms(config, options))
  if ran and units:
    copula = _read_copula_units(out, stations, units)
    cases = copula["scores"]
    failures = copula["failures"]
    if not failures.empty:
      failed = set(zip(failures["model"], failures["arm"], strict=True))
      keep = [
        (m, a) not in failed
        for m, a in zip(cases["model"], cases["run_arm"], strict=True)
      ]
      cases = cases.loc[keep]
    repeated = cases.duplicated(["station", "model", "arm", "row"])
    if repeated.any():
      raise ValueError(
        "copula results must name each arm once per marginal set; an arm's "
        "result repeats another's name"
      )
    tables["copula_failures"] = failures
    if not cases.empty:
      summary = copula_tables(cases, seed=seed)
      tables["copulas_by_station"] = summary["by_station"]
      tables["copulas_pooled"] = summary["pooled"]
      tables["copulas_station_mean"] = summary["station_mean"]
    copula_timings = copula["timings"]
    if not copula_timings.empty:
      tables["copulas_timings"] = summarize_timings(
        copula_timings, ["model", "arm"]
      )
  _write_csv(tables, out / "tables")
  return sorted(tables)
