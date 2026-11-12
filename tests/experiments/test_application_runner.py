"""``npcc-application``, stage by stage, on synthetic data.

Margins run on the hermetic ``linear-gaussian`` backend; the copula stage is
:class:`~tests.experiments._application.FakeCopulas` unless a test says
otherwise, so the runner's copula path is tested with or without the real
stage installed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from npcc.experiments._study import derive_seed
from npcc.experiments.application import runner
from npcc.experiments.application.cli import main
from npcc.experiments.application.config import (
  ApplicationConfig,
  MarginalModel,
  load_config,
)
from npcc.experiments.application.data import StationData, write_synthetic
from npcc.experiments.application.margins import StationMargins
from npcc.experiments.application.store import (
  copula_status,
  margin_dir,
  stored_stations,
)
from tests.experiments._application import (
  MODELS,
  FakeCopulas,
  config_text,
  write_config,
  write_data,
)

_COPULAS = "npcc.experiments.application.copulas"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
  return write_data(tmp_path_factory.mktemp("data"))


@pytest.fixture
def config_path(
  data_dir: Path, tmp_path: Path, register_linear_gaussian_backend: None
) -> Path:
  del register_linear_gaussian_backend
  return write_config(tmp_path / "app.toml", data_dir)


@pytest.fixture
def fake_copulas(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeCopulas]:
  copulas = FakeCopulas()
  monkeypatch.setitem(sys.modules, _COPULAS, copulas.module())
  yield copulas


def _margins(config: Path, out: Path, *extra: str) -> int:
  return main(["margins", "--config", str(config), "--out", str(out), *extra])


def test_margins_then_evaluate_from_the_command_line(
  config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """The console script runs the marginal stage end to end.

  Every station's every model is stored, and the tables score each model on
  the same cases, EMOS's skill against itself being zero.
  """
  out = tmp_path / "run"
  assert _margins(config_path, out, "--device", "cpu") == 0
  assert (
    main(["evaluate", "--config", str(config_path), "--out", str(out)]) == 0
  )

  assert stored_stations(out) == (0, 1, 2)
  for station in (0, 1, 2):
    for model in MODELS:
      assert margin_dir(out, station, model).is_dir()
  tables = out / "tables"
  printed = capsys.readouterr().out
  assert str(tables / "margins_pooled.csv") in printed
  by_station = pd.read_csv(tables / "margins_by_station.csv")
  assert len(by_station) == 3 * len(MODELS) * 5
  pooled = pd.read_csv(tables / "margins_pooled.csv")
  emos = pooled.loc[pooled["model"] == "emos-seasonal"]
  np.testing.assert_allclose(emos["crpss"], 0.0)
  assert (pooled["n_cases"] == pooled["n_cases"].iloc[0]).all()
  assert pooled["n_stations"].eq(3).all()
  mean = pd.read_csv(tables / "margins_station_mean.csv")
  assert {"crpss_low", "crpss_high", "log_score_gain"} <= set(mean.columns)
  timings = pd.read_csv(tables / "margins_timings.csv")
  assert set(timings["stage"]) >= {"fit", "quantiles_test"}
  assert not (tables / "copulas_pooled.csv").exists()


def test_shards_split_the_stations_and_a_resume_skips_what_is_stored(
  config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Two shards must cover every station once; a resume must refit nothing.

  Without ``--resume``, stored units are refused rather than refitted or
  overwritten.
  """
  out = tmp_path / "run"
  fitted: list[tuple[int, str]] = []
  fit_station = runner.fit_station

  def counted(
    data: StationData,
    model: MarginalModel,
    *,
    config: ApplicationConfig,
    device: str | None,
  ) -> StationMargins:
    fitted.append((data.station, model.name))
    return fit_station(data, model, config=config, device=device)

  monkeypatch.setattr(runner, "fit_station", counted)
  assert _margins(config_path, out, "--shard", "0/2") == 0
  assert sorted({s for s, _ in fitted}) == [0, 2]
  assert _margins(config_path, out, "--shard", "1/2") == 0
  assert sorted({s for s, _ in fitted}) == [0, 1, 2]
  assert len(fitted) == 3 * len(MODELS)

  with pytest.raises(FileExistsError, match="--resume"):
    _margins(config_path, out)
  assert _margins(config_path, out, "--resume") == 0
  assert len(fitted) == 3 * len(MODELS)

  assert (
    _margins(config_path, out, "--resume", "--models", "emos-seasonal") == 0
  )
  assert len(fitted) == 3 * len(MODELS)


def test_a_resume_on_changed_data_or_configuration_is_refused(
  data_dir: Path,
  config_path: Path,
  tmp_path: Path,
  register_linear_gaussian_backend: None,
) -> None:
  """Results from two datasets or two configurations must not mix."""
  del register_linear_gaussian_backend
  out = tmp_path / "run"
  assert (
    _margins(config_path, out, "--stations", "0", "--models", "emos-seasonal")
    == 0
  )

  other_seed = write_config(tmp_path / "seed.toml", data_dir, base_seed=12)
  with pytest.raises(ValueError, match="fresh --out"):
    _margins(other_seed, out, "--resume", "--stations", "0")

  changed = tmp_path / "changed"
  write_synthetic(changed, n_stations=3, step_days=7, seed=4)
  moved = write_config(tmp_path / "moved.toml", changed)
  with pytest.raises(ValueError, match="input data changed"):
    _margins(moved, out, "--resume", "--stations", "0")


def test_a_model_or_station_outside_the_configuration_is_refused(
  config_path: Path, tmp_path: Path
) -> None:
  """The command line narrows the configuration; it never widens it."""
  with pytest.raises(ValueError, match="models must be drawn"):
    _margins(config_path, tmp_path / "run", "--models", "lightgbm-all")
  with pytest.raises(SystemExit):
    _margins(config_path, tmp_path / "run", "--stations", "zero")


def test_the_copula_stage_runs_each_arm_and_records_failures(
  config_path: Path, tmp_path: Path, fake_copulas: FakeCopulas
) -> None:
  """Every arm on every marginal set, with the contract's arguments.

  A ``CopulaFailure`` is recorded and the run goes on; the evaluation leaves
  the failed arm out of every comparison on that margin, at every station,
  and lists it.
  """
  out = tmp_path / "run"
  assert _margins(config_path, out) == 0
  fake_copulas.fail = {(1, "emos-seasonal", "gaussian")}
  copulas = ["copulas", "--config", str(config_path), "--out", str(out)]
  assert main([*copulas, "--device", "cpu"]) == 0

  config = load_config(config_path)
  assert len(fake_copulas.calls) == 3 * len(MODELS) * 2
  seeds = {call["seed"] for call in fake_copulas.calls}
  assert seeds == {derive_seed(config.base_seed, "copulas")}
  call = fake_copulas.calls[0]
  assert call["estimators"] == dict(config.estimators_by_backend)
  assert (call["n_draws"], call["sample_batch_size"], call["mixture_k"]) == (
    config.n_draws,
    config.sample_batch_size,
    config.mixture_k,
  )
  assert {c["device"] for c in fake_copulas.calls} == {"cpu"}
  assert copula_status(out, 1, "emos-seasonal", "gaussian") == "failed"
  assert copula_status(out, 0, "emos-seasonal", "gaussian") == "done"
  assert fake_copulas.r_checks == 0

  with pytest.raises(FileExistsError, match="--resume"):
    main(copulas)
  assert main([*copulas, "--resume"]) == 0
  assert len(fake_copulas.calls) == 3 * len(MODELS) * 2

  assert (
    main(["evaluate", "--config", str(config_path), "--out", str(out)]) == 0
  )
  tables = out / "tables"
  failures = pd.read_csv(tables / "copula_failures.csv")
  assert failures[["station", "model", "arm"]].values.tolist() == [
    [1, "emos-seasonal", "gaussian"]
  ]
  pooled = pd.read_csv(tables / "copulas_pooled.csv")
  arms = pooled.groupby("model")["arm"].apply(set).to_dict()
  assert arms["emos-seasonal"] == {"independence"}
  assert arms["linear-gaussian-all"] == {"independence", "gaussian"}
  gaussian = pooled.loc[pooled["arm"] == "gaussian"]
  assert (gaussian["joint_nll_gain"] > 0).all()
  assert (gaussian["n_stations"] == 3).all()


def test_the_copula_stage_reads_only_its_own_marginal_sets(
  data_dir: Path,
  tmp_path: Path,
  fake_copulas: FakeCopulas,
  register_linear_gaussian_backend: None,
) -> None:
  """``copulas.margins`` chooses the sets; ``--models`` narrows within them."""
  del register_linear_gaussian_backend
  config = write_config(
    tmp_path / "app.toml",
    data_dir,
    copula_margins=["emos-seasonal", "linear-gaussian-all"],
    stations="[0]",
  )
  out = tmp_path / "run"
  assert _margins(config, out) == 0
  copulas = ["copulas", "--config", str(config), "--out", str(out)]
  with pytest.raises(ValueError, match="models must be drawn"):
    main([*copulas, "--models", "linear-gaussian-emos"])
  assert main([*copulas, "--models", "linear-gaussian-all"]) == 0
  assert {c["model"] for c in fake_copulas.calls} == {"linear-gaussian-all"}
  assert main([*copulas, "--resume", "--copulas", "independence"]) == 0
  assert {c["model"] for c in fake_copulas.calls} == {
    "emos-seasonal",
    "linear-gaussian-all",
  }
  assert len(fake_copulas.calls) == 3


def test_a_margin_runs_and_is_scored_on_its_own_arms_only(
  data_dir: Path,
  tmp_path: Path,
  fake_copulas: FakeCopulas,
  register_linear_gaussian_backend: None,
) -> None:
  """An arm left out of one margin's list must not be fitted there.

  Nor may evaluate then demand that unit: it reads each margin's own arms,
  and still scores the other margin's every arm.
  """
  del register_linear_gaussian_backend
  config = tmp_path / "app.toml"
  config.write_text(
    config_text(
      data_dir,
      stations="[0, 2]",
      copula_margins=["emos-seasonal", "linear-gaussian-all"],
    ).replace(
      'arms = ["independence", "gaussian"]',
      'arms = ["independence", "gaussian"]\n'
      'arms_by_margin = { "emos-seasonal" = ["independence"] }',
      1,
    )
  )
  out = tmp_path / "run"
  assert _margins(config, out) == 0
  assert main(["copulas", "--config", str(config), "--out", str(out)]) == 0

  ran = {(c["model"], c["arm"]) for c in fake_copulas.calls}
  assert ran == {
    ("emos-seasonal", "independence"),
    ("linear-gaussian-all", "independence"),
    ("linear-gaussian-all", "gaussian"),
  }
  assert len(fake_copulas.calls) == 2 * 3
  assert copula_status(out, 0, "emos-seasonal", "gaussian") is None
  assert main(["evaluate", "--config", str(config), "--out", str(out)]) == 0
  pooled = pd.read_csv(out / "tables" / "copulas_pooled.csv")
  arms = pooled.groupby("model")["arm"].apply(set).to_dict()
  assert arms == {
    "emos-seasonal": {"independence"},
    "linear-gaussian-all": {"independence", "gaussian"},
  }


def test_a_chain_order_scored_by_its_log_score_alone_is_evaluated(
  data_dir: Path,
  tmp_path: Path,
  fake_copulas: FakeCopulas,
  register_linear_gaussian_backend: None,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """An order with nan sample scores must reach the tables as such.

  A chain mixture's orders have a log score and nothing else. Evaluate must
  keep them, with their log score and nan sample scores, rather than drop
  them as failed or stop on the nan; the primary summaries then compare
  them by the log score alone.
  """
  del register_linear_gaussian_backend
  chain = "npcc-chain-linear-gaussian-all"
  fake_copulas.arms = ("independence", "gaussian", chain)
  monkeypatch.setitem(sys.modules, _COPULAS, fake_copulas.module())
  config = tmp_path / "app.toml"
  config.write_text(
    config_text(data_dir, stations="[0, 1]").replace(
      'arms = ["independence", "gaussian"]',
      f"arms = {json.dumps(list(fake_copulas.arms))}",
      1,
    )
  )
  out = tmp_path / "run"
  assert _margins(config, out) == 0
  assert main(["copulas", "--config", str(config), "--out", str(out)]) == 0
  assert main(["evaluate", "--config", str(config), "--out", str(out)]) == 0
  pooled = pd.read_csv(out / "tables" / "copulas_pooled.csv").set_index(
    ["model", "arm"]
  )
  order = pooled.loc[("linear-gaussian-all", f"{chain}-m1")]
  assert np.isnan(order["energy"]) and np.isnan(order["variogram"])
  assert np.isfinite(order["joint_nll"])
  assert np.isfinite(pooled.loc[("linear-gaussian-all", chain), "energy"])
  failures = pd.read_csv(out / "tables" / "copula_failures.csv")
  assert failures.empty

  spec = importlib.util.spec_from_file_location(
    "paired_summary",
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "paper"
    / "application"
    / "paired_summary.py",
  )
  assert spec is not None and spec.loader is not None
  summary = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(summary)
  summary.build(out, out / "analysis", without=[1])
  paired = pd.read_csv(out / "analysis" / "paired.csv")
  rows = paired[paired["arm"] == f"{chain}-m1"]
  assert set(rows["score"]) == {"joint_nll"}
  assert (rows.loc[rows["stations"] == "all", "n_stations"] == 2).all()


def test_an_unknown_arm_or_a_missing_r_stops_before_any_fit(
  config_path: Path,
  tmp_path: Path,
  fake_copulas: FakeCopulas,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Hours of npcc fits must not run before a check that could fail now."""
  out = tmp_path / "run"
  assert _margins(config_path, out, "--stations", "0") == 0
  fake_copulas.arms = ("independence",)
  monkeypatch.setitem(sys.modules, _COPULAS, fake_copulas.module())
  with pytest.raises(ValueError, match="copula arms must be drawn"):
    main(["copulas", "--config", str(config_path), "--out", str(out)])

  fake_copulas.arms = ("independence", "gaussian", "boostcopula-all")
  fake_copulas.r_error = "Rscript must be on PATH"
  monkeypatch.setitem(sys.modules, _COPULAS, fake_copulas.module())
  boost = write_config(
    tmp_path / "boost.toml",
    load_config(config_path).data_dir,
    arms=["independence", "boostcopula-all"],
  )
  with pytest.raises(RuntimeError, match="Rscript"):
    main(["copulas", "--config", str(boost), "--out", str(out)])
  assert fake_copulas.calls == []
  assert fake_copulas.r_checks == 1


def test_preflight_reads_every_station_and_checks_the_stages(
  config_path: Path,
  fake_copulas: FakeCopulas,
  capsys: pytest.CaptureFixture[str],
) -> None:
  """A run that preflight passes must not fail on its inputs later."""
  assert main(["preflight", "--config", str(config_path)]) == 0

  printed = capsys.readouterr().out
  for station in (0, 1, 2):
    assert f"station {station}:" in printed
  assert "linear-gaussian-all: linear-gaussian-probit" in printed
  assert "copula arms: independence, gaussian" in printed
  assert printed.strip().endswith("preflight passed")
  assert fake_copulas.calls == []


def test_fetch_verify_only_reports_missing_files(
  tmp_path: Path,
  register_linear_gaussian_backend: None,
  capsys: pytest.CaptureFixture[str],
) -> None:
  """``fetch`` fails with what to do, not a traceback, and fetches nothing."""
  del register_linear_gaussian_backend
  empty = tmp_path / "empty"
  config = write_config(tmp_path / "app.toml", empty)

  assert main(["fetch", "--config", str(config), "--verify-only"]) == 1
  assert "x.nc must exist" in capsys.readouterr().err
  assert not empty.exists()


def test_the_real_copula_stage_runs_on_the_stored_margins(
  config_path: Path, tmp_path: Path
) -> None:
  """The two halves meet: the copula stage reads what the margins stored.

  Only the arms that need neither R nor a foundation model run here.
  """
  copulas = pytest.importorskip(_COPULAS)
  arms = [a for a in ("independence", "gaussian") if a in copulas.COPULA_ARMS]
  if not arms:
    pytest.skip("the copula stage has no cheap arm to run")
  out = tmp_path / "run"
  assert _margins(config_path, out, "--stations", "0") == 0
  common = ["--config", str(config_path), "--out", str(out), "--stations", "0"]
  assert main(["copulas", *common, "--copulas", *arms]) == 0
  assert main(["evaluate", *common, "--copulas", *arms]) == 0

  pooled = pd.read_csv(out / "tables" / "copulas_pooled.csv")
  assert set(pooled["arm"]) == set(arms)
  assert np.isfinite(pooled[["joint_nll", "energy", "variogram"]]).all().all()
