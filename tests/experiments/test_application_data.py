"""Which rows of x.nc and y.nc become a station's cases, and what is refused."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
from collections.abc import Callable
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pytest
import xarray as xr

from npcc.experiments.application import data as data_module
from npcc.experiments.application.data import (
  DATA_FILES,
  DATA_SHA256,
  EXCLUDED_FEATURES,
  SYNTHETIC_VARIABLES,
  TIME_FEATURES,
  CaseSelection,
  StationData,
  data_sha256,
  feature_columns,
  fetch_data,
  load_station,
  station_ids,
  verify_data,
  write_synthetic,
)
from tests.experiments._application import write_data


def _load(
  directory: Path, station: int, selection: CaseSelection
) -> StationData:
  """A synthetic station: its files are not the published ones."""
  return load_station(directory, station, selection, sha256=None)


_SELECTION = CaseSelection(
  train=(dt.date(2017, 1, 1), dt.date(2020, 12, 31)),
  test=(dt.date(2021, 1, 1), dt.date(2021, 12, 31)),
)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
  return write_data(tmp_path_factory.mktemp("data"))


Edit = Callable[[xr.Dataset, xr.Dataset], tuple[xr.Dataset, xr.Dataset]]


def _rewrite(source: Path, target: Path, edit: Edit) -> Path:
  """Copy the two files, applying ``edit(x, y)`` to the loaded datasets."""
  target.mkdir()
  with (
    xr.open_dataset(source / "x.nc", engine="h5netcdf") as x,
    xr.open_dataset(source / "y.nc", engine="h5netcdf") as y,
  ):
    x, y = x.load(), y.load()
  x, y = edit(x, y)
  x.to_netcdf(target / "x.nc", engine="h5netcdf")
  y.to_netcdf(target / "y.nc", engine="h5netcdf")
  return target


def test_a_station_gets_its_midnight_day_ahead_cases_in_time_order(
  data_dir: Path,
) -> None:
  """A wrong lead, hour, station or split would fit on the wrong forecasts.

  The synthetic files hold three forecasts per date and station, shuffled:
  the case (midnight, 24 h) and two the selection must leave out.
  """
  data = _load(data_dir, 1, _SELECTION)
  with xr.open_dataset(data_dir / "x.nc", engine="h5netcdf") as x:
    ref = x["forecast_reference_time"].values
    station = x["station_id"].values
    lead = x["t"].values

  for split, (first, last) in (
    ("train", _SELECTION.train),
    ("test", _SELECTION.test),
  ):
    rows = getattr(data, f"row_{split}")
    times = getattr(data, f"time_{split}")
    assert (station[rows] == 1).all()
    assert (lead[rows] == 24).all()
    init = ref[rows]
    assert (init == init.astype("datetime64[D]")).all()
    assert init.min() >= np.datetime64(first)
    assert init.max() < np.datetime64(last) + np.timedelta64(1, "D")
    np.testing.assert_array_equal(times, init + np.timedelta64(24, "h"))
    assert (np.diff(times) > np.timedelta64(0)).all()
  assert len(data.row_train) >= 200
  assert len(data.row_test) >= 50
  assert data.variables == SYNTHETIC_VARIABLES
  assert not set(EXCLUDED_FEATURES) & set(data.feature_names)
  assert set(TIME_FEATURES) <= set(data.feature_names)
  assert list(data.feature_names) == sorted(data.feature_names)


def test_the_values_are_the_selected_rows(data_dir: Path) -> None:
  """A covariate read from another row than its response breaks the pairing."""
  data = _load(data_dir, 2, _SELECTION)
  with (
    xr.open_dataset(data_dir / "x.nc", engine="h5netcdf") as x,
    xr.open_dataset(data_dir / "y.nc", engine="h5netcdf") as y,
  ):
    for j, name in enumerate(data.feature_names):
      np.testing.assert_array_equal(
        data.x_test[:, j], x[name].values[data.row_test]
      )
    for j, name in enumerate(data.variables):
      np.testing.assert_array_equal(
        data.y_train[:, j], y[name].values[data.row_train]
      )


def test_the_humidity_holds_both_bounds_in_both_splits(data_dir: Path) -> None:
  """The bound nudges would go untested on data that never reaches a bound."""
  data = _load(data_dir, 0, _SELECTION)
  j = data.column("obs:relative_humidity")
  for y in (data.y_train[:, j], data.y_test[:, j]):
    assert (y == 0.0).any()
    assert (y == 100.0).any()
    assert ((y >= 0.0) & (y <= 100.0)).all()
  k = data.column("obs:water_vapor_mixing_ratio")
  assert (data.y_train[:, k] > 0).all()


def test_another_lead_or_hour_selects_other_forecasts(data_dir: Path) -> None:
  """The selection must read its lead and hour rather than assume 24 h at 0."""
  for lead, hour in ((12, 0), (24, 12)):
    selection = CaseSelection(
      train=_SELECTION.train,
      test=_SELECTION.test,
      lead_hours=lead,
      init_hour=hour,
    )
    data = _load(data_dir, 0, selection)
    with xr.open_dataset(data_dir / "x.nc", engine="h5netcdf") as x:
      ref = x["forecast_reference_time"].values[data.row_train]
      assert (x["t"].values[data.row_train] == lead).all()
    hours = (ref - ref.astype("datetime64[D]")) / np.timedelta64(1, "h")
    assert (hours == hour).all()


def test_a_lead_stored_as_a_duration_is_read_in_hours(
  data_dir: Path, tmp_path: Path
) -> None:
  """xarray decodes a lead with time units to a duration, which == 24 misses."""

  def as_duration(
    x: xr.Dataset, y: xr.Dataset
  ) -> tuple[xr.Dataset, xr.Dataset]:
    hours = x["t"].values.astype("timedelta64[h]").astype("timedelta64[ns]")
    return x.assign_coords(t=("s", hours)), y.assign_coords(t=("s", hours))

  edited = _rewrite(data_dir, tmp_path / "duration", as_duration)
  with xr.open_dataset(edited / "x.nc", engine="h5netcdf") as x:
    assert x["t"].values.dtype.kind == "m"
  data = _load(edited, 0, _SELECTION)
  np.testing.assert_array_equal(
    data.row_train, _load(data_dir, 0, _SELECTION).row_train
  )


def test_every_station_is_listed(data_dir: Path) -> None:
  """``--stations all`` must reach every station in the data."""
  assert station_ids(data_dir) == (0, 1, 2)


def test_a_split_with_too_few_cases_is_refused(data_dir: Path) -> None:
  """A model fitted on a handful of cases is not one the paper reports."""
  selection = CaseSelection(
    train=_SELECTION.train,
    test=(dt.date(2021, 1, 1), dt.date(2021, 1, 31)),
  )
  with pytest.raises(ValueError, match="at least 20 cases"):
    _load(data_dir, 0, selection)


def test_a_non_finite_value_is_refused_rather_than_dropped(
  data_dir: Path, tmp_path: Path
) -> None:
  """Dropping a case would misalign it across models and copulas."""
  data = _load(data_dir, 0, _SELECTION)
  row = int(data.row_test[3])

  def poison(x: xr.Dataset, y: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    values = y["obs:air_temperature"].values.copy()
    values[row] = np.nan
    return x, y.assign({"obs:air_temperature": ("s", values)})

  edited = _rewrite(data_dir, tmp_path / "nan", poison)
  with pytest.raises(ValueError, match="y_test must be finite"):
    _load(edited, 0, _SELECTION)
  _load(edited, 1, _SELECTION)


def test_a_repeated_initialization_is_refused(
  data_dir: Path, tmp_path: Path
) -> None:
  """Two cases on one date would be two answers to one forecast."""
  row = int(_load(data_dir, 0, _SELECTION).row_train[5])

  def repeat(x: xr.Dataset, y: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    return (
      xr.concat([x, x.isel(s=[row])], dim="s"),
      xr.concat([y, y.isel(s=[row])], dim="s"),
    )

  edited = _rewrite(data_dir, tmp_path / "repeat", repeat)
  with pytest.raises(ValueError, match="one case per initialization"):
    _load(edited, 0, _SELECTION)


def test_files_whose_rows_disagree_are_refused(
  data_dir: Path, tmp_path: Path
) -> None:
  """A response paired with another row's covariates is a silent error."""

  def shift(x: xr.Dataset, y: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    ref = y["forecast_reference_time"].values.copy()
    ref[0] = ref[1]
    return x, y.assign_coords(forecast_reference_time=("s", ref))

  edited = _rewrite(data_dir, tmp_path / "shift", shift)
  with pytest.raises(ValueError, match="agree on forecast_reference_time"):
    _load(edited, 0, _SELECTION)


@pytest.mark.parametrize(
  ("scope", "variable", "expected"),
  [
    ("time", None, list(TIME_FEATURES)),
    (
      "emos",
      "obs:relative_humidity",
      ["coe:relative_humidity_ensavg", *TIME_FEATURES],
    ),
  ],
)
def test_the_narrow_scopes_pick_their_covariates_by_name(
  scope: str, variable: str | None, expected: list[str]
) -> None:
  """A scope read by position would pick another covariate in another file."""
  names = [
    "coe:air_temperature_ensavg",
    "coe:relative_humidity_ensavg",
    "time:sin_dayofyear",
    "coe:leadtime",
    "time:cos_dayofyear",
    "time:sin_hourofday",
  ]
  columns = feature_columns(names, scope, variable)

  assert [names[c] for c in columns] == expected


def test_the_all_scope_leaves_out_what_is_constant() -> None:
  """The lead and the hour of day are constant within a station's cases."""
  names = [
    "coe:a_ensavg",
    "coe:leadtime",
    "time:sin_hourofday",
    "other",
    "time:cos_dayofyear",
  ]

  assert [names[c] for c in feature_columns(names, "all")] == [
    "coe:a_ensavg",
    "time:cos_dayofyear",
  ]


@pytest.mark.parametrize(
  ("names", "scope", "variable", "match"),
  [
    (["time:sin_dayofyear"], "time", None, "missing"),
    (list(TIME_FEATURES), "emos", "obs:air_temperature", "missing"),
    (list(TIME_FEATURES), "emos", None, "variable"),
    (list(TIME_FEATURES), "hourly", None, "scope"),
  ],
)
def test_a_scope_that_cannot_be_met_is_refused(
  names: list[str], scope: str, variable: str | None, match: str
) -> None:
  """A model on fewer covariates than its scope names is another model."""
  with pytest.raises(ValueError, match=match):
    feature_columns(names, scope, variable)


def test_synthetic_data_is_reproducible(tmp_path: Path) -> None:
  """A dry run and the tests must see the same files for the same seed."""
  write_synthetic(tmp_path / "a", n_stations=2, step_days=30, seed=5)
  write_synthetic(tmp_path / "b", n_stations=2, step_days=30, seed=5)
  for name in ("x.nc", "y.nc"):
    with (
      xr.open_dataset(tmp_path / "a" / name, engine="h5netcdf") as a,
      xr.open_dataset(tmp_path / "b" / name, engine="h5netcdf") as b,
    ):
      xr.testing.assert_identical(a, b)


# ---------------------------------------------------------------------------
# The published files: fetching and verifying them
# ---------------------------------------------------------------------------

_PAYLOADS = {"x.nc": b"covariates " * 1000, "y.nc": b"responses " * 700}


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
  """Two small files as the published ones, served by a fake ``urlopen``.

  Returns each file's URL, which the fake server answers with its payload;
  nothing reaches the network.
  """
  files = {
    name: data_module.DataFile(
      url=f"https://example.invalid/{name}",
      sha256=hashlib.sha256(payload).hexdigest(),
    )
    for name, payload in _PAYLOADS.items()
  }
  monkeypatch.setattr(data_module, "DATA_FILES", files)
  served = {f.url: _PAYLOADS[name] for name, f in files.items()}

  def urlopen(url: str, timeout: float) -> io.BytesIO:
    del timeout
    return io.BytesIO(served[url])

  monkeypatch.setattr(data_module, "urlopen", urlopen)
  return {name: f.url for name, f in files.items()}


def _listing(directory: Path) -> list[str]:
  return sorted(p.name for p in directory.iterdir())


def test_the_published_digests_are_the_notebooks() -> None:
  """A mistyped digest would refuse the real files, or accept another."""
  assert DATA_SHA256 == {
    "x.nc": "c6acaf62051b81dfd3d5a4aa516d545615fd2597c8c38f4db4e571a621201878",
    "y.nc": "6265a5f0272e5427c823b95725b8aabbc48a9a97d7554fd5732e6c4b480f3ab3",
  }
  assert all(f.url.endswith("?download=1") for f in DATA_FILES.values())


def test_fetch_writes_each_file_only_once_verified(
  published: dict[str, str], tmp_path: Path
) -> None:
  """A file must reach its name whole and checked, or not at all."""
  target = tmp_path / "data"

  digests = fetch_data(target)

  assert _listing(target) == ["x.nc", "y.nc"]
  for name, payload in _PAYLOADS.items():
    assert (target / name).read_bytes() == payload
    assert digests[name] == hashlib.sha256(payload).hexdigest()
  assert fetch_data(target, verify_only=True) == digests


def test_a_download_with_another_digest_is_not_kept(
  published: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A truncated or substituted download must not sit where it is read."""
  monkeypatch.setattr(
    data_module, "urlopen", lambda url, timeout: io.BytesIO(b"an error page")
  )
  target = tmp_path / "data"

  with pytest.raises(ValueError, match="must have SHA-256"):
    fetch_data(target)
  assert _listing(target) == []


def test_a_refused_download_says_to_use_a_browser(
  published: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The server answers scripts with 403; the message must say what to do.

  It names the folder, both URLs and ``fetch --verify-only``, and tries no
  other way in.
  """
  calls: list[str] = []

  def forbidden(url: str, timeout: float) -> io.BytesIO:
    del timeout
    calls.append(url)
    raise HTTPError(url, 403, "Forbidden", Message(), None)

  monkeypatch.setattr(data_module, "urlopen", forbidden)
  target = tmp_path / "data"

  with pytest.raises(RuntimeError, match="HTTP Error 403") as caught:
    fetch_data(target)
  message = str(caught.value)
  assert "browser" in message
  assert str(target) in message
  assert "--verify-only" in message
  for url in published.values():
    assert url in message
  assert calls == [published["x.nc"]]
  assert _listing(target) == []


def test_a_file_in_place_is_kept_when_published_and_refused_otherwise(
  published: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Fetching again must neither download what is there nor overwrite it."""
  target = tmp_path / "data"
  fetch_data(target)

  def offline(url: str, timeout: float) -> io.BytesIO:
    raise AssertionError(f"downloaded {url} again")

  monkeypatch.setattr(data_module, "urlopen", offline)
  fetch_data(target)
  (target / "y.nc").write_bytes(b"something else")
  with pytest.raises(ValueError, match="y.nc must have SHA-256"):
    fetch_data(target)
  with pytest.raises(ValueError, match="y.nc must have SHA-256"):
    fetch_data(target, verify_only=True)
  (target / "y.nc").unlink()
  with pytest.raises(FileNotFoundError, match="y.nc must exist"):
    fetch_data(target, verify_only=True)


def test_a_station_is_not_read_from_files_that_are_not_the_published_ones(
  data_dir: Path,
) -> None:
  """Results computed on another file would be reported as the paper's data."""
  with pytest.raises(ValueError, match="published file's"):
    load_station(data_dir, 0, _SELECTION)
  digests = {k: v for k, v in data_sha256(data_dir).items() if v is not None}
  assert sorted(digests) == ["x.nc", "y.nc"]
  assert load_station(data_dir, 0, _SELECTION, sha256=digests).station == 0
  assert verify_data(data_dir, digests) == digests
