"""The stored draws: every scored copula's draws, readable without npcc."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from npcc.experiments.application.cli import main
from npcc.experiments.application.config import load_config
from npcc.experiments.application.data import write_synthetic
from npcc.experiments.application.derived import (
  DEW_POINT,
  MIXING_RATIO,
  PRESSURE,
  RELATIVE_HUMIDITY,
  TEMPERATURE,
  derive,
)
from npcc.experiments.application.joint_scores import energy_score
from npcc.experiments.application.store import (
  copula_dir,
  draws_frame,
  read_marginal_set,
)
from tests.experiments._application import FakeCopulaResult, config_text

_MODELED = (TEMPERATURE, RELATIVE_HUMIDITY, PRESSURE)
_DERIVED = (DEW_POINT, MIXING_RATIO)
_TRIPLE = (
  'modeled = ["obs:air_temperature", "obs:relative_humidity", '
  '"obs:surface_air_pressure"]\n'
  'derived = ["obs:dew_point_temperature", "obs:water_vapor_mixing_ratio"]\n'
)


def _triple(tmp_path: Path, *, save_draws: bool) -> Path:
  data = tmp_path / "data"
  if not data.exists():
    write_synthetic(data, n_stations=1, step_days=7, seed=3, measured=True)
  text = config_text(data, stations="[0]").replace(
    "[responses]\n", "[responses]\n" + _TRIPLE
  )
  if save_draws:
    text = text.replace("[copulas]\n", "[copulas]\nsave_draws = true\n")
  path = tmp_path / f"triple_{save_draws}.toml"
  path.write_text(text)
  return path


@pytest.fixture
def run(tmp_path: Path, register_linear_gaussian_backend: None) -> Path:
  """The triple, margins and the real independence and Gaussian arms."""
  del register_linear_gaussian_backend
  config = _triple(tmp_path, save_draws=True)
  out = tmp_path / "run"
  for stage in ("margins", "copulas", "evaluate"):
    assert main([stage, "--config", str(config), "--out", str(out)]) == 0
  return out


def test_the_stored_draws_are_the_ones_the_scores_were_computed_from(
  run: Path,
) -> None:
  """Draws other than the scored ones would let a reader score another run.

  Read with pandas alone: one row per test case and draw, ``float32``
  responses named as in ``y.nc``, the derived ones the derivation of the
  modeled ones; and each case's energy score, with the unit's stored
  scales, and every response's CRPS, recomputed from the stored draws,
  match the stored scores to float32 precision.
  """
  ms = read_marginal_set(run, 0, "linear-gaussian-all")
  for arm in ("independence", "gaussian"):
    unit = copula_dir(run, 0, "linear-gaussian-all", arm)
    schema = pq.read_schema(unit / "draws.parquet")
    for variable in (*_MODELED, *_DERIVED):
      assert str(schema.field(variable).type) == "float"
    draws = pd.read_parquet(unit / "draws.parquet")
    scores = pd.read_parquet(unit / "scores.parquet")
    n_test, n_draws = len(ms.y_test), 8
    assert len(draws) == n_test * n_draws
    assert list(draws.columns[:6]) == [
      "station",
      "model",
      "run_arm",
      "arm",
      "row",
      "draw",
    ]
    assert set(draws["arm"]) == {arm}
    np.testing.assert_array_equal(
      draws["row"].to_numpy().reshape(n_test, n_draws)[:, 0], scores["row"]
    )
    cube = (
      draws[[*_MODELED, *_DERIVED]]
      .to_numpy(np.float64)
      .reshape(n_test, n_draws, 5)
    )
    np.testing.assert_allclose(
      cube[..., 3:],
      derive(cube[..., :3], _MODELED, _DERIVED),
      rtol=1e-5,
      atol=1e-5,
    )
    observed = np.concatenate([ms.y_test, ms.y_derived_test], axis=1)
    info = json.loads((unit / "info.json").read_text())
    scale = np.array([info["scales"][v] for v in (*_MODELED, *_DERIVED)])
    np.testing.assert_allclose(
      scale,
      np.concatenate(
        [np.std(ms.y_train, axis=0), np.std(ms.y_derived_train, axis=0)]
      ),
    )
    np.testing.assert_allclose(
      energy_score(cube / scale, observed / scale),
      scores["energy"],
      rtol=1e-5,
    )
    for j, variable in enumerate((*_MODELED, *_DERIVED)):
      column = "crps_" + variable.removeprefix("obs:")
      # float32 keeps about 7 digits: a few units of pressure's last one.
      resolution = np.finfo(np.float32).eps * np.abs(observed[:, j]).max()
      np.testing.assert_allclose(
        energy_score(cube[:, :, j : j + 1], observed[:, j : j + 1]),
        scores[column],
        rtol=1e-5,
        atol=4 * resolution,
      )


def test_draws_are_stored_only_when_the_config_asks(
  tmp_path: Path, register_linear_gaussian_backend: None
) -> None:
  """Draws a run did not ask for would cost gigabytes it did not budget.

  The setting is part of the signature off its default, so a folder with
  draws for some units and none for others cannot arise by a resume.
  """
  del register_linear_gaussian_backend
  config = _triple(tmp_path, save_draws=False)
  out = tmp_path / "run"
  for stage in ("margins", "copulas"):
    assert main([stage, "--config", str(config), "--out", str(out)]) == 0
  unit = copula_dir(out, 0, "linear-gaussian-all", "gaussian")
  assert (unit / "scores.parquet").exists()
  assert not (unit / "draws.parquet").exists()
  plain = load_config(config)
  saving = load_config(_triple(tmp_path, save_draws=True))
  assert saving.save_draws and not plain.save_draws
  assert saving.signature() != plain.signature()
  assert saving.echo()["save_draws"] is True


def test_a_result_scored_on_draws_it_does_not_carry_is_refused(
  run: Path,
) -> None:
  """A sampled result without its draws would store an incomplete record.

  A chain mixture's order, which draws nothing of its own, is skipped.
  """
  ms = read_marginal_set(run, 0, "linear-gaussian-all")
  n = len(ms.y_test)
  zeros = np.zeros(n)
  order = FakeCopulaResult(
    "chain-m1", zeros, zeros, zeros, zeros, sampled=False
  )
  sampled = FakeCopulaResult(
    "chain", zeros, zeros, zeros, zeros, draws=np.zeros((n, 4, 5))
  )
  frame = draws_frame(ms, "chain", [sampled, order], rows=np.arange(n))
  assert set(frame["arm"]) == {"chain"}
  assert len(frame) == n * 4
  with pytest.raises(ValueError, match="draws must have shape"):
    draws_frame(
      ms,
      "chain",
      [FakeCopulaResult("chain", zeros, zeros, zeros, zeros)],
      rows=np.arange(n),
    )
  with pytest.raises(ValueError, match="must hold the 5 responses"):
    draws_frame(
      ms,
      "chain",
      [
        FakeCopulaResult(
          "chain", zeros, zeros, zeros, zeros, draws=np.zeros((n, 4, 3))
        )
      ],
      rows=np.arange(n),
    )
