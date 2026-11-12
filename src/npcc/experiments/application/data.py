"""One station's cases: which rows of ``x.nc`` and ``y.nc`` enter, and how.

The inputs are two netCDF files sharing a row dimension ``s``: ``x.nc`` holds
the covariates, ``coe:*`` (ensemble statistics) and ``time:*`` (calendar
terms), and ``y.nc`` the responses, ``obs:*``. Both carry ``station_id``,
``forecast_reference_time`` and the lead ``t`` (hours) per row.

A station's cases are its forecasts initialized at ``init_hour`` with lead
``lead_hours``, split by initialization date into a training and a test
period, each sorted by time. Nothing is dropped silently: a non-finite value,
a repeated date or a split with too few cases is refused.

:func:`write_synthetic` writes a small pair of files in the same layout, for
the tests and for a dry run before the real data is at hand.

**Source.** The data is the anonymized MeteoSwiss station sample of Zanetta,
Nerini, Beucler and Liniger (2023), "Physics-constrained deep learning
postprocessing of temperature and humidity", arXiv:2212.04487, which the
paper cites. It is distributed through the notebook
https://colab.research.google.com/github/tbeucler/2023_MOOC_ECMWF/blob/main/Physically_constrained_postprocessing.ipynb
from the URLs, and with the SHA-256 digests, in :data:`DATA_FILES`.
:func:`fetch_data` downloads and verifies them into a folder outside the
repository, :data:`DEFAULT_DATA_DIR` by default, and :func:`load_station`
refuses a file whose digest is not the published one.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import xarray as xr

from npcc.experiments.application.derived import mixing_ratio, relative_humidity

#: The day-of-year terms: the ``time`` scope, and part of the ``emos`` one.
TIME_FEATURES: tuple[str, ...] = ("time:sin_dayofyear", "time:cos_dayofyear")
#: Covariates no model sees: the lead is constant within a station's cases,
#: and so is the hour of day at a fixed initialization and lead.
EXCLUDED_FEATURES: tuple[str, ...] = (
  "coe:leadtime",
  "time:sin_hourofday",
  "time:cos_hourofday",
)
#: The covariate scopes :func:`feature_columns` knows.
SCOPES: tuple[str, ...] = ("all", "emos", "time")
#: The responses :func:`write_synthetic` writes, in the order ``y.nc`` sorts.
SYNTHETIC_VARIABLES: tuple[str, ...] = (
  "obs:air_temperature",
  "obs:dew_point_temperature",
  "obs:relative_humidity",
  "obs:surface_air_pressure",
  "obs:water_vapor_mixing_ratio",
)

_SPLITS: tuple[str, ...] = ("train", "test")
_ENGINE = "h5netcdf"

_SOURCE = "https://unils-my.sharepoint.com/:u:/g/personal/tom_beucler_unil_ch/"


@dataclass(frozen=True)
class DataFile:
  """One published input file: where it is served, and its digest.

  Attributes
  ----------
  url : str
      Where the notebook that distributes it downloads it from.
  sha256 : str
      The SHA-256 hex digest the notebook pins.
  """

  url: str
  sha256: str


#: The two input files of the MeteoSwiss sample (Zanetta et al., 2023), as
#: the distributing notebook pins them with pooch.
DATA_FILES: Mapping[str, DataFile] = MappingProxyType(
  {
    "x.nc": DataFile(
      url=_SOURCE + "EdAG3RBBgk5Kmvo54RPgT2kBp-NJqqGF6Il-gTmh9DbdeA?download=1",
      sha256="c6acaf62051b81dfd3d5a4aa516d545615fd2597c8c38f4db4e571a621201878",
    ),
    "y.nc": DataFile(
      url=_SOURCE + "EdVQCVKqnb9Bh495opeuRCEBBZFPDdG0g3xSpIFgNGJeJA?download=1",
      sha256="6265a5f0272e5427c823b95725b8aabbc48a9a97d7554fd5732e6c4b480f3ab3",
    ),
  }
)
#: Where :func:`fetch_data` puts them unless told otherwise: outside git.
DEFAULT_DATA_DIR = Path("~/npcc-data/meteoswiss_zanetta2023")
#: Each file's published digest, the default of what :func:`load_station`
#: checks.
DATA_SHA256: Mapping[str, str] = MappingProxyType(
  {name: f.sha256 for name, f in DATA_FILES.items()}
)

_CHUNK = 1 << 20
_digests: dict[tuple[str, int, int], str] = {}


def file_sha256(path: Path) -> str:
  """The SHA-256 hex digest of a file, computed once per process per version.

  A digest is kept against the file's path, size and modification time, so a
  run that loads many stations reads each file once, and an edited file is
  read again.

  Parameters
  ----------
  path : pathlib.Path
      The file.

  Returns
  -------
  str
      Its digest.
  """
  path = Path(path).resolve()
  status = path.stat()
  key = (str(path), status.st_size, status.st_mtime_ns)
  if key not in _digests:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
      while chunk := fh.read(_CHUNK):
        digest.update(chunk)
    _digests[key] = digest.hexdigest()
  return _digests[key]


def data_sha256(directory: Path) -> dict[str, str | None]:
  """The digest of each input file present in ``directory``.

  Parameters
  ----------
  directory : pathlib.Path
      The data folder.

  Returns
  -------
  dict of str to str or None
      For ``x.nc`` and ``y.nc``, its digest, or ``None`` where it is absent.
  """
  directory = Path(directory).expanduser()
  return {
    name: file_sha256(directory / name)
    if (directory / name).is_file()
    else None
    for name in DATA_FILES
  }


def verify_data(
  directory: Path, expected: Mapping[str, str] = DATA_SHA256
) -> dict[str, str]:
  """Check that each input file is there and is the published one.

  Parameters
  ----------
  directory : pathlib.Path
      The data folder.
  expected : mapping of str to str, optional
      Each file's digest; the published ones by default.

  Returns
  -------
  dict of str to str
      Each file's digest.

  Raises
  ------
  FileNotFoundError
      If a file is missing.
  ValueError
      If a file's digest is not the expected one.
  """
  directory = Path(directory).expanduser()
  digests: dict[str, str] = {}
  for name, want in expected.items():
    path = directory / name
    if not path.is_file():
      raise FileNotFoundError(
        f"{path} must exist; fetch the data with `npcc-application fetch`"
      )
    got = file_sha256(path)
    if got != want:
      raise ValueError(
        f"{path} must have SHA-256 {want}, the published file's; got {got}. "
        "It is another file, or a partial or altered copy: replace it with "
        "the published one (`npcc-application fetch`)."
      )
    digests[name] = got
  return digests


def _browser_message(directory: Path, name: str, reason: object) -> str:
  urls = "\n".join(f"  {n}: {f.url}" for n, f in DATA_FILES.items())
  return (
    f"downloading {name} must succeed; got {reason}. The server refuses "
    f"scripted downloads, so download the two files in a browser instead, "
    f"save them as x.nc and y.nc in {directory}, and then run "
    f"`npcc-application fetch --config <config> --verify-only`:\n{urls}"
  )


def fetch_data(directory: Path, *, verify_only: bool = False) -> dict[str, str]:
  """Download the published input files into ``directory``, verified.

  Each file is streamed into a temporary file beside its destination, its
  digest checked, and only then renamed into place, so a failed or partial
  download never sits where :func:`load_station` reads. A file already in
  place with the published digest is kept.

  Parameters
  ----------
  directory : pathlib.Path
      The data folder; created if absent.
  verify_only : bool, default=False
      Check the files already there instead of downloading.

  Returns
  -------
  dict of str to str
      Each file's digest.

  Raises
  ------
  RuntimeError
      If a download fails, with how to fetch the files by hand.
  ValueError
      If a download, or a file already in place, is not the published file.
  """
  directory = Path(directory).expanduser()
  if verify_only:
    return verify_data(
      directory, {name: f.sha256 for name, f in DATA_FILES.items()}
    )
  directory.mkdir(parents=True, exist_ok=True)
  digests: dict[str, str] = {}
  for name, source in DATA_FILES.items():
    path = directory / name
    if path.exists():
      digests.update(verify_data(directory, {name: source.sha256}))
      continue
    handle, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=directory)
    try:
      digest = hashlib.sha256()
      with open(handle, "wb") as out:
        try:
          with urlopen(source.url, timeout=120) as response:
            while chunk := response.read(_CHUNK):
              digest.update(chunk)
              out.write(chunk)
        except (URLError, OSError) as exc:
          raise RuntimeError(_browser_message(directory, name, exc)) from exc
      if digest.hexdigest() != source.sha256:
        raise ValueError(
          f"the download of {name} must have SHA-256 {source.sha256}; got "
          f"{digest.hexdigest()}. Nothing was written to {path}."
        )
      Path(temporary).replace(path)
    finally:
      Path(temporary).unlink(missing_ok=True)
    digests[name] = source.sha256
  return digests


@dataclass(frozen=True)
class CaseSelection:
  """Which forecasts of a station are cases, and which split each is in.

  Attributes
  ----------
  train, test : tuple of datetime.date
      First and last initialization date of each split, both inclusive.
  lead_hours : int
      The forecast lead, in hours.
  init_hour : int
      The initialization hour, UTC.
  min_cases : int
      The fewest cases a split may hold.
  """

  train: tuple[dt.date, dt.date]
  test: tuple[dt.date, dt.date]
  lead_hours: int = 24
  init_hour: int = 0
  min_cases: int = 20

  def __post_init__(self) -> None:
    """Refuse empty, inverted or overlapping splits."""
    for name, (start, end) in (("train", self.train), ("test", self.test)):
      if start > end:
        raise ValueError(
          f"{name} must start on or before its last date; got "
          f"{start.isoformat()} to {end.isoformat()}"
        )
    if self.train[1] >= self.test[0]:
      raise ValueError(
        f"train must end before test starts, so no case is in both; got "
        f"train to {self.train[1].isoformat()} and test from "
        f"{self.test[0].isoformat()}"
      )
    if self.lead_hours <= 0:
      raise ValueError(f"lead_hours must be positive; got {self.lead_hours}")
    if not 0 <= self.init_hour < 24:
      raise ValueError(f"init_hour must lie in [0, 24); got {self.init_hour}")
    if self.min_cases < 2:
      raise ValueError(f"min_cases must be at least 2; got {self.min_cases}")


@dataclass(frozen=True)
class StationData:
  """One station's cases, aligned across covariates and responses.

  Attributes
  ----------
  station : int
      The station id.
  feature_names : tuple of str
      The ``p`` covariates of the ``all`` scope, sorted.
  variables : tuple of str
      The ``d`` responses, sorted.
  x_train, x_test : numpy.ndarray, shape (n_train, p) and (n_test, p)
      The covariates.
  y_train, y_test : numpy.ndarray, shape (n_train, d) and (n_test, d)
      The responses, on their original scale.
  time_train, time_test : numpy.ndarray of datetime64[ns]
      The valid time of each case, increasing.
  row_train, row_test : numpy.ndarray of int64
      Each case's row in the input files.
  """

  station: int
  feature_names: tuple[str, ...]
  variables: tuple[str, ...]
  x_train: np.ndarray
  x_test: np.ndarray
  y_train: np.ndarray
  y_test: np.ndarray
  time_train: np.ndarray
  time_test: np.ndarray
  row_train: np.ndarray
  row_test: np.ndarray

  def column(self, variable: str) -> int:
    """The column of ``variable`` in ``y_train`` and ``y_test``.

    Parameters
    ----------
    variable : str
        A response name.

    Returns
    -------
    int
        Its index in :attr:`variables`.
    """
    if variable not in self.variables:
      raise ValueError(
        f"variable must be one of station {self.station}'s responses "
        f"{list(self.variables)}; got {variable!r}"
      )
    return self.variables.index(variable)


def feature_columns(
  names: Sequence[str], scope: str, variable: str | None = None
) -> list[int]:
  """The columns of ``names`` a scope selects.

  Parameters
  ----------
  names : sequence of str
      Covariate names, as in :attr:`StationData.feature_names`.
  scope : str
      ``"all"``: every ``coe:`` and ``time:`` covariate but
      :data:`EXCLUDED_FEATURES`. ``"emos"``: the response's own ensemble mean,
      ``coe:<name>_ensavg``, then :data:`TIME_FEATURES`. ``"time"``:
      :data:`TIME_FEATURES` alone.
  variable : str, optional
      The response, as ``obs:<name>``; required by the ``emos`` scope.

  Returns
  -------
  list of int
      Indices into ``names``, in the scope's order.
  """
  names = list(names)
  match scope:
    case "all":
      selected = [
        n
        for n in names
        if n.startswith(("coe:", "time:")) and n not in EXCLUDED_FEATURES
      ]
    case "time":
      selected = list(TIME_FEATURES)
    case "emos":
      if variable is None:
        raise ValueError(
          "variable must name the response for the 'emos' scope, whose "
          "covariate is that response's ensemble mean; got None"
        )
      ensemble = "coe:" + variable.removeprefix("obs:") + "_ensavg"
      selected = [ensemble, *TIME_FEATURES]
    case _:
      raise ValueError(f"scope must be one of {list(SCOPES)}; got {scope!r}")
  missing = [n for n in selected if n not in names]
  if not selected or missing:
    raise ValueError(
      f"covariates must include the {scope!r} scope's {selected or 'any'}; "
      f"missing {missing}"
    )
  return [names.index(n) for n in selected]


def station_ids(directory: Path) -> tuple[int, ...]:
  """Every station in ``directory/x.nc``, increasing.

  Parameters
  ----------
  directory : pathlib.Path
      The folder holding ``x.nc``.

  Returns
  -------
  tuple of int
      The distinct ``station_id`` values.
  """
  directory = Path(directory).expanduser()
  with xr.open_dataset(directory / "x.nc", engine=_ENGINE) as x:
    ids = np.unique(np.asarray(x["station_id"].values))
  return tuple(int(i) for i in ids)


def _lead_hours(values: np.ndarray) -> np.ndarray:
  """The lead in hours, whether stored as a number or decoded as a duration."""
  values = np.asarray(values)
  if values.dtype.kind == "m":
    return values / np.timedelta64(1, "h")
  return values.astype(float)


def _check_aligned(x: xr.Dataset, y: xr.Dataset) -> None:
  """Refuse two files whose rows are not the same forecasts."""
  if x.sizes["s"] != y.sizes["s"]:
    raise ValueError(
      f"x.nc and y.nc must have the same rows; got {x.sizes['s']} and "
      f"{y.sizes['s']}"
    )
  for key in ("station_id", "forecast_reference_time", "t"):
    if key in y and not np.array_equal(x[key].values, y[key].values):
      raise ValueError(
        f"x.nc and y.nc must agree on {key} in every row; they differ"
      )


def load_station(
  directory: Path,
  station: int,
  selection: CaseSelection,
  *,
  sha256: Mapping[str, str] | None = DATA_SHA256,
) -> StationData:
  """One station's cases, in time order within each split.

  Parameters
  ----------
  directory : pathlib.Path
      The folder holding ``x.nc`` and ``y.nc``.
  station : int
      The station id.
  selection : CaseSelection
      The lead, initialization hour and split dates.
  sha256 : mapping of str to str or None, optional
      The digest each file must have; the published ones by default, and
      ``None`` for files of another source, such as synthetic ones.

  Returns
  -------
  StationData
      The covariates of the ``all`` scope and every ``obs:`` response.

  Raises
  ------
  ValueError
      If a file is not the expected one, the files disagree, a split has
      fewer than ``selection.min_cases`` cases or a repeated date, or a value
      is not finite.
  """
  directory = Path(directory).expanduser()
  if sha256 is not None:
    verify_data(directory, sha256)
  with (
    xr.open_dataset(directory / "x.nc", engine=_ENGINE) as x,
    xr.open_dataset(directory / "y.nc", engine=_ENGINE) as y,
  ):
    _check_aligned(x, y)
    ref = np.asarray(x["forecast_reference_time"].values).astype(
      "datetime64[ns]"
    )
    hour = ref - ref.astype("datetime64[D]")
    eligible = (
      (np.asarray(x["station_id"].values) == station)
      & (_lead_hours(x["t"].values) == selection.lead_hours)
      & (hour == np.timedelta64(selection.init_hour, "h"))
    )
    all_names = sorted(str(n) for n in x.data_vars)
    names = [all_names[i] for i in feature_columns(all_names, "all")]
    variables = sorted(str(n) for n in y.data_vars if str(n).startswith("obs:"))
    if not variables:
      raise ValueError(
        f"y.nc must hold at least one obs: response; got {list(y.data_vars)}"
      )
    lead = np.timedelta64(selection.lead_hours, "h")
    arrays: dict[str, np.ndarray] = {}
    for split, (start, end) in zip(
      _SPLITS, (selection.train, selection.test), strict=True
    ):
      first = np.datetime64(start, "D")
      after = np.datetime64(end, "D") + np.timedelta64(1, "D")
      rows = np.flatnonzero(eligible & (ref >= first) & (ref < after))
      rows = rows[np.argsort(ref[rows], kind="stable")]
      if len(rows) < selection.min_cases:
        raise ValueError(
          f"station {station}'s {split} split must hold at least "
          f"{selection.min_cases} cases; got {len(rows)}"
        )
      if len(np.unique(ref[rows])) != len(rows):
        raise ValueError(
          f"station {station}'s {split} split must hold one case per "
          f"initialization; got {len(rows) - len(np.unique(ref[rows]))} "
          "repeated"
        )
      xs = x[names].isel(s=rows)
      ys = y[variables].isel(s=rows)
      arrays[f"x_{split}"] = np.column_stack(
        [np.asarray(xs[n].values, dtype=np.float64) for n in names]
      )
      arrays[f"y_{split}"] = np.column_stack(
        [np.asarray(ys[n].values, dtype=np.float64) for n in variables]
      )
      arrays[f"time_{split}"] = ref[rows] + lead
      arrays[f"row_{split}"] = rows.astype(np.int64)
  for key in ("x_train", "x_test", "y_train", "y_test"):
    bad = ~np.isfinite(arrays[key])
    if bad.any():
      raise ValueError(
        f"station {station}'s {key} must be finite, since no case is dropped "
        f"silently; got {int(bad.sum())} non-finite values"
      )
  return StationData(
    station=int(station),
    feature_names=tuple(names),
    variables=tuple(variables),
    **arrays,
  )


def _humidity(temperature: np.ndarray, dew_point: np.ndarray) -> np.ndarray:
  """Relative humidity in percent, by the Magnus formula."""
  a, b = 17.625, 243.04
  return 100.0 * np.exp(
    a * dew_point / (b + dew_point) - a * temperature / (b + temperature)
  )


def _mixing_ratio(dew_point: np.ndarray, pressure: np.ndarray) -> np.ndarray:
  """Water vapor mixing ratio in g/kg, from dew point (C) and pressure (hPa)."""
  vapor = 6.112 * np.exp(17.67 * dew_point / (dew_point + 243.5))
  return 622.0 * vapor / (pressure - vapor)


def write_synthetic(
  directory: Path,
  *,
  n_stations: int = 3,
  start: dt.date = dt.date(2016, 12, 1),
  end: dt.date = dt.date(2022, 1, 31),
  step_days: int = 1,
  seed: int = 0,
  saturated_share: float = 0.0,
) -> None:
  """Write a synthetic ``x.nc`` and ``y.nc`` in the application's layout.

  Every station has a forecast per initialization date from ``start`` to
  ``end``, ``step_days`` apart, at midnight with a 24 h lead, plus two that
  :class:`CaseSelection`'s defaults leave out: a 12 h lead from midnight and
  a 24 h lead from noon. Rows are shuffled, so selection has to sort them.

  The responses are :data:`SYNTHETIC_VARIABLES`: temperatures and pressure
  around a seasonal climate, relative humidity in percent from the two
  temperatures, with some exactly 0 and some exactly 100, and a positive
  mixing ratio. Each has an ensemble mean, ``coe:<name>_ensavg``, and spread,
  ``coe:<name>_ensstd``, that forecast it with an error.

  Parameters
  ----------
  directory : pathlib.Path
      The folder to write into; created if absent.
  n_stations : int, default=3
      Stations ``0`` to ``n_stations - 1``.
  start, end : datetime.date
      First and last initialization date.
  step_days : int, default=1
      Days between initializations.
  seed : int, default=0
      Seed of the draws.
  saturated_share : float, default=0.0
      Above 0, the responses obey the identities the real data obeys
      (:mod:`~npcc.experiments.application.derived`): the dew point equals
      the temperature in about this share of rows, and relative humidity and
      the mixing ratio are computed from temperature, dew point and pressure.
      At 0, they are drawn with noise of their own.
  """
  if n_stations < 1 or step_days < 1:
    raise ValueError(
      f"n_stations and step_days must be positive; got {n_stations} and "
      f"{step_days}"
    )
  rng = np.random.default_rng(seed)
  days = np.arange(
    np.datetime64(start, "D"),
    np.datetime64(end, "D") + np.timedelta64(1, "D"),
    np.timedelta64(step_days, "D"),
  )
  # (init hour, lead): the cases, then two forecasts the selection excludes.
  kinds = ((0, 24), (0, 12), (12, 24))
  station = np.repeat(np.arange(n_stations), len(days) * len(kinds))
  day = np.tile(np.repeat(days, len(kinds)), n_stations)
  init = np.tile(np.array([k[0] for k in kinds]), n_stations * len(days))
  lead = np.tile(np.array([k[1] for k in kinds]), n_stations * len(days))
  ref = day.astype("datetime64[ns]") + init.astype("timedelta64[h]")
  valid = ref + lead.astype("timedelta64[h]")
  n = len(ref)

  year_start = valid.astype("datetime64[Y]").astype("datetime64[D]")
  doy = (valid.astype("datetime64[D]") - year_start).astype(float)
  annual = 2.0 * np.pi * doy / 365.25
  hours = (valid - valid.astype("datetime64[D]")) / np.timedelta64(1, "h")
  diurnal = 2.0 * np.pi * hours / 24.0

  temperature = (
    8.0 + 2.0 * station + 9.0 * np.sin(annual - 1.9) + rng.normal(0, 3, n)
  )
  dew_point = temperature - rng.gamma(2.0, 2.0, n)
  pressure = 960.0 - 6.0 * station + rng.normal(0.0, 8.0, n)
  truth = {
    "air_temperature": temperature,
    "dew_point_temperature": dew_point,
    "relative_humidity": _humidity(temperature, dew_point),
    "surface_air_pressure": pressure,
    "water_vapor_mixing_ratio": _mixing_ratio(dew_point, pressure),
  }
  noise = {
    "air_temperature": 1.2,
    "dew_point_temperature": 1.5,
    "relative_humidity": 6.0,
    "surface_air_pressure": 1.0,
    "water_vapor_mixing_ratio": 0.08,
  }
  covariates: dict[str, np.ndarray] = {}
  responses: dict[str, np.ndarray] = {}
  for name, value in truth.items():
    scale = noise[name]
    if name == "water_vapor_mixing_ratio":
      responses[name] = value * rng.lognormal(0.0, scale, n)
      ensemble = value * rng.lognormal(0.0, 2.0 * scale, n)
    else:
      responses[name] = value + rng.normal(0.0, scale, n)
      ensemble = value + rng.normal(0.0, 1.5 * scale, n)
    if name == "relative_humidity":
      responses[name] = np.clip(responses[name], 0.0, 100.0)
      ensemble = np.clip(ensemble, 1.0, 100.0)
    covariates[f"coe:{name}_ensavg"] = ensemble
    covariates[f"coe:{name}_ensstd"] = scale * rng.gamma(9.0, 1.0 / 9.0, n)
  # Exact bounds in every split of every station, as real observations have.
  humidity = responses["relative_humidity"]
  humidity[np.arange(n) % 11 == 0] = 100.0
  humidity[np.arange(n) % 17 == 1] = 0.0
  if saturated_share > 0.0:
    t = responses["air_temperature"]
    td = np.minimum(responses["dew_point_temperature"], t)
    saturated = rng.random(n) < saturated_share
    td[saturated] = t[saturated]
    responses["dew_point_temperature"] = td
    responses["relative_humidity"] = relative_humidity(t, td)
    responses["water_vapor_mixing_ratio"] = mixing_ratio(
      t, td, responses["surface_air_pressure"]
    )
  covariates["coe:leadtime"] = lead.astype(float)
  covariates["time:sin_dayofyear"] = np.sin(annual)
  covariates["time:cos_dayofyear"] = np.cos(annual)
  covariates["time:sin_hourofday"] = np.sin(diurnal)
  covariates["time:cos_hourofday"] = np.cos(diurnal)

  order = rng.permutation(n)
  coords = {
    "station_id": ("s", station[order].astype(np.int64)),
    "forecast_reference_time": ("s", ref[order]),
    "t": ("s", lead[order].astype(np.int64)),
  }
  x = xr.Dataset(
    {k: ("s", v[order]) for k, v in sorted(covariates.items())}, coords=coords
  )
  y = xr.Dataset(
    {f"obs:{k}": ("s", v[order]) for k, v in sorted(responses.items())},
    coords=coords,
  )
  directory = Path(directory)
  directory.mkdir(parents=True, exist_ok=True)
  x.to_netcdf(directory / "x.nc", engine=_ENGINE)
  y.to_netcdf(directory / "y.nc", engine=_ENGINE)
