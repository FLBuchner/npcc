"""The raw-data physical-consistency EDA, on synthetic files and planted cases."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from npcc.experiments.application.data import write_synthetic
from npcc.experiments.application.derived import dew_point, mixing_ratio

_SCRIPT = (
  Path(__file__).resolve().parents[2]
  / "scripts"
  / "paper"
  / "application"
  / "eda_physical_consistency.py"
)


def _module() -> ModuleType:
  spec = importlib.util.spec_from_file_location("eda_physical", _SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _cases(t: list[float], rh: list[float], p: list[float]) -> pd.DataFrame:
  """Observed cases whose Td and w are derived from ``T``, ``RH``, ``p``."""
  ta, rha, pa = (np.asarray(v, dtype=np.float64) for v in (t, rh, p))
  return pd.DataFrame(
    {
      "source_row": np.arange(len(ta)),
      "station": np.arange(len(ta)) % 2,
      "initialization": np.datetime64("2020-01-01", "ns"),
      "source": "Observations",
      "air_temperature": ta,
      "dew_point_temperature": dew_point(ta, rha),
      "surface_air_pressure": pa,
      "relative_humidity": rha,
      "water_vapor_mixing_ratio": mixing_ratio(ta, rha, pa),
    }
  )


def test_the_eda_reads_only_the_cases_and_recovers_measured_responses(
  tmp_path: Path,
) -> None:
  """A wrong selection or formula would misreport the data's consistency.

  The synthetic files hold, beside the 24 h forecasts from midnight, a 12 h
  lead and a noon initialization per date, which must not be counted; their
  dew point and mixing ratio are computed from T, RH and p by the
  application's formulas, so the residuals are rounding alone.
  """
  eda = _module()
  data = tmp_path / "data"
  write_synthetic(data, n_stations=2, step_days=30, measured=True)
  out = tmp_path / "eda"
  assert eda.main(["--data", str(data), "--out", str(out), "--no-verify"]) == 0

  coverage = pd.read_csv(out / "coverage.csv").set_index("source")
  n_dates = len(
    np.arange(
      np.datetime64("2016-12-01"),
      np.datetime64("2022-02-01"),
      np.timedelta64(30, "D"),
    )
  )
  assert coverage.loc["Observations", "cases"] == 2 * n_dates
  assert coverage.loc["Observations", "stations"] == 2
  assert coverage.loc["Observations", "domain"] == 2 * n_dates

  summary = pd.read_csv(out / "residual_summary.csv")
  observed = summary[summary["source"] == "Observations"]
  assert (observed["max_abs"] < 1e-9).all()
  checks = pd.read_csv(out / "bounds_checks.csv").set_index(["source", "check"])
  assert checks.loc[("Observations", "Td_above_T_0.001C"), "violations"] == 0
  assert checks.loc[("Observations", "RH_outside_0_100"), "violations"] == 0

  residuals = pd.read_csv(out / "case_residuals.csv.gz")
  assert len(residuals) == 4 * n_dates
  metadata = json.loads((out / "metadata.json").read_text())
  assert metadata["input_files"]["x.nc"]["published"] is False
  assert (out / "physical_consistency.pdf").stat().st_size > 10_000


def test_each_planted_violation_is_counted_by_its_own_check() -> None:
  """A check that read the wrong column, or the wrong side, would miss one.

  One case breaks each bound and nothing else; a sixth carries a non-finite
  pressure, which leaves it out of the residuals and of the pressure check.
  """
  eda = _module()
  frame = _cases(
    [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
    [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
    [900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0],
  )
  frame.loc[1, "dew_point_temperature"] = 10.0 + 1e-6
  frame.loc[2, "dew_point_temperature"] = 10.5
  frame.loc[3, "relative_humidity"] = 100.5
  frame.loc[4, "water_vapor_mixing_ratio"] = -0.1
  frame.loc[5, "surface_air_pressure"] = 5.0
  frame.loc[6, "surface_air_pressure"] = np.nan
  analyzed = eda.analyze(frame)
  table = eda.summarize(analyzed)["bounds_checks"].set_index("check")
  broken = {
    "Td_above_T": 2,
    "Td_above_T_0.001C": 1,
    "RH_outside_0_100": 1,
    "w_negative": 1,
    "p_not_above_e": 1,
    "nonfinite": 1,
  }
  for check, count in broken.items():
    assert table.loc[check, "violations"] == count, check
  assert table.loc["p_not_above_e", "eligible"] == 6
  assert table.loc["Td_above_T", "max_Td_minus_T"] == pytest.approx(0.5)
  assert analyzed["domain"].tolist() == [True] * 5 + [False, False]
  assert np.isnan(analyzed.loc[5:, "RH_residual"]).all()


def test_the_own_sign_convention_departs_only_where_t_and_td_straddle_zero() -> (
  None
):
  """The sensitivity table must isolate the coefficient choice.

  With the constants of T's sign the derivation recovers RH; with Td's own,
  the residual is nonzero exactly where T is at or above 0 and Td below.
  """
  eda = _module()
  frame = _cases([5.0, 2.0, -3.0, 15.0], [40.0, 60.0, 70.0, 80.0], [950.0] * 4)
  assert (frame["dew_point_temperature"] < 0).tolist() == [
    True,
    True,
    True,
    False,
  ]
  analyzed = eda.analyze(frame)
  np.testing.assert_allclose(analyzed["RH_residual"], 0.0, atol=1e-10)
  alternative = analyzed["RH_alternative_residual"].abs().to_numpy()
  assert (alternative[:2] > 1e-3).all()
  np.testing.assert_allclose(alternative[2:], 0.0, atol=1e-10)
  sensitivity = eda.summarize(analyzed)["coefficient_sensitivity"]
  assert set(sensitivity["straddling_cases"]) == {2}


def test_files_that_are_not_the_published_ones_are_refused(
  tmp_path: Path,
) -> None:
  """An EDA of other files must not read as the published data's."""
  eda = _module()
  data = tmp_path / "data"
  write_synthetic(data, n_stations=1, step_days=60)
  with pytest.raises(ValueError, match="SHA-256"):
    eda.build(data, tmp_path / "eda")
