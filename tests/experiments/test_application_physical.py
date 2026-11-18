"""Physically impossible draws, counted per case and tabled by evaluate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from npcc.experiments.application.cli import main
from npcc.experiments.application.data import write_synthetic
from npcc.experiments.application.derived import (
  DEW_POINT,
  METEOSWISS,
  MIXING_RATIO,
  PRESSURE,
  RELATIVE_HUMIDITY,
  TEMPERATURE,
  physical_violations,
)
from npcc.experiments.application.store import copula_dir
from tests.experiments._application import config_text

_FIVE = (TEMPERATURE, DEW_POINT, RELATIVE_HUMIDITY, PRESSURE, MIXING_RATIO)
_TRIPLE = (
  'modeled = ["obs:air_temperature", "obs:relative_humidity", '
  '"obs:surface_air_pressure"]\n'
  'derived = ["obs:dew_point_temperature", "obs:water_vapor_mixing_ratio"]\n'
)


def test_each_check_counts_the_draws_that_break_it() -> None:
  """A miscounted check would certify impossible draws as physical.

  Case 0's draws are physical; case 1's break one bound each, the last
  only its vapor pressure: at Td = 30 C, ``e_s = 6.107 exp(521.04 /
  268.83) = 42.42`` hPa, above p = 40 hPa.
  """
  good = [10.0, 5.0, 70.9, 950.0, 5.8]
  draws = np.array(
    [
      [good, good, good, good],
      [
        [10.0, 12.0, 113.7, 950.0, 9.4],
        [10.0, 5.0, -1.0, 950.0, 5.8],
        [10.0, 5.0, 70.9, 950.0, -0.1],
        [35.0, 30.0, 75.0, 40.0, 5.8],
      ],
    ]
  )
  found = physical_violations(draws, _FIVE)

  assert METEOSWISS.saturation_vapor_pressure(
    np.array(30.0), np.array(35.0)
  ) == pytest.approx(42.42, abs=0.01)
  np.testing.assert_array_equal(found["td_above_t"], [0, 1])
  np.testing.assert_array_equal(found["rh_outside_0_100"], [0, 2])
  np.testing.assert_array_equal(found["w_negative"], [0, 1])
  np.testing.assert_array_equal(found["p_not_above_e"], [0, 1])
  np.testing.assert_array_equal(found["any"], [0, 4])


def test_a_check_without_its_responses_is_left_out() -> None:
  """A check run on missing responses would count nothing and read as zero."""
  draws = np.zeros((2, 3, 2))
  assert physical_violations(draws, (TEMPERATURE, PRESSURE)) == {}
  found = physical_violations(draws, (TEMPERATURE, MIXING_RATIO))
  assert set(found) == {"w_negative", "any"}


def _run(tmp_path: Path, *, triple: bool) -> Path:
  data = tmp_path / "data"
  write_synthetic(data, n_stations=1, step_days=7, seed=3, measured=triple)
  text = config_text(data, stations="[0]", arms=("independence",))
  if triple:
    text = text.replace("[responses]\n", "[responses]\n" + _TRIPLE)
  text = text.replace("[copulas]\n", "[copulas]\nsave_draws = true\n")
  config = tmp_path / "app.toml"
  config.write_text(text)
  out = tmp_path / "run"
  for stage in ("margins", "copulas", "evaluate"):
    assert main([stage, "--config", str(config), "--out", str(out)]) == 0
  return out


def test_the_triple_draws_nothing_impossible(
  tmp_path: Path, register_linear_gaussian_backend: None
) -> None:
  """(T, RH, p) with RH in (0, 100) gives Td <= T and w > 0 by construction.

  Independent draws of the three are the hardest case for it, and evaluate
  must still report every check at zero, with every draw counted.
  """
  del register_linear_gaussian_backend
  out = _run(tmp_path, triple=True)
  table = pd.read_csv(out / "tables" / "copulas_physical_validity.csv")
  assert len(table) == 3
  assert (table["n_draws"] == table["n_cases"] * 8).all()
  checks = [c for c in table if c.startswith("broken_") and "share" not in c]
  assert set(checks) == {
    "broken_td_above_t",
    "broken_rh_outside_0_100",
    "broken_w_negative",
    "broken_p_not_above_e",
    "broken_any",
  }
  assert (table[checks] == 0).all().all()


def test_five_responses_drawn_at_once_can_be_impossible(
  tmp_path: Path, register_linear_gaussian_backend: None
) -> None:
  """Independent draws of T and Td put Td above T; the table must say so.

  The counts in ``scores.parquet`` are those of the stored draws.
  """
  del register_linear_gaussian_backend
  out = _run(tmp_path, triple=False)
  table = pd.read_csv(out / "tables" / "copulas_physical_validity.csv")
  assert (table["broken_td_above_t"] > 0).all()
  assert (table["broken_any"] >= table["broken_td_above_t"]).all()
  unit = copula_dir(out, 0, "linear-gaussian-all", "independence")
  draws = pd.read_parquet(unit / "draws.parquet")
  scores = pd.read_parquet(unit / "scores.parquet")
  above = (draws[DEW_POINT] > draws[TEMPERATURE]).groupby(draws["row"]).sum()
  np.testing.assert_array_equal(
    above.loc[scores["row"]].to_numpy(), scores["broken_td_above_t"]
  )
