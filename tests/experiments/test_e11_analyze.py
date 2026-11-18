"""E11's conditional analysis: the runs it pairs and what it compares."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

_SCRIPT = (
  Path(__file__).resolve().parents[2]
  / "scripts"
  / "paper"
  / "e11"
  / "analyze_chain.py"
)


def _load_script() -> ModuleType:
  spec = importlib.util.spec_from_file_location("analyze_chain", _SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


analyze = _load_script()

#: Each arm's KL at every x of a repetition, so a difference reads off them.
KL = {
  "chain": 0.30,
  "chain-m1": 0.40,
  "chain-m2": 0.36,
  "mixture": 0.25,
  "selected": 0.35,
  "oracle": 0.20,
  "random": 0.50,
}


def _write(
  folder: Path, arms: list[str], reps: range, *, loglik: float = -1.5
) -> Path:
  """A run's ``metrics_by_x.parquet`` at d = 5, n = 1000, two x values."""
  rows = [
    {
      "d": 5,
      "n": 1000,
      "rep": rep,
      "seed": 100 + rep,
      "arm": arm,
      "x": x,
      "loglik_true": loglik + rep,
      "KL": KL[arm] + 0.01 * rep,
      "KL_copula": KL[arm] / 2,
    }
    for arm in arms
    for rep in reps
    for x in (0.2, 0.8)
  ]
  folder.mkdir(parents=True)
  pd.DataFrame(rows).to_parquet(folder / "metrics_by_x.parquet")
  return folder


def _chain_run(tmp_path: Path) -> Path:
  run = _write(tmp_path / "chain", ["chain", "chain-m1", "chain-m2"], range(4))
  pd.DataFrame(
    [
      {
        "d": 5,
        "n": 1000,
        "rep": rep,
        "arm": "chain",
        "chain_orders": json.dumps([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]),
      }
      for rep in range(4)
    ]
  ).to_parquet(run / "runtime.parquet")
  return run


def test_the_chain_is_paired_with_every_reference_on_shared_cells(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Each comparison is paired by repetition, on the cells all runs hold.

  A repetition one reference lacks must leave every comparison, or the
  chain's mean would be over other cells than the reference's; the orders'
  mean and the best order must be the chain's own, cell by cell.
  """
  run = _chain_run(tmp_path)
  e2 = _write(tmp_path / "e2", ["oracle", "random"], range(4))
  selected = _write(tmp_path / "selected", ["selected"], range(3))
  e10 = _write(tmp_path / "e10", ["mixture"], range(4))
  out = tmp_path / "analysis"
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "analyze_chain.py",
      str(run),
      "--with",
      str(e2),
      str(selected),
      str(e10),
      "--out",
      str(out),
    ],
  )
  analyze.main()
  [row] = pd.read_csv(out / "chain.csv").to_dict("records")
  assert row["n_rep"] == 3
  assert row["references"] == "mixture;selected;oracle;random"
  for ref in ("mixture", "selected", "oracle", "random"):
    assert row[f"joint_chain_minus_{ref}_mean"] == pytest.approx(
      KL["chain"] - KL[ref]
    )
    assert row[f"joint_chain_wins_over_{ref}"] == (
      3 if KL["chain"] < KL[ref] else 0
    )
  assert row["joint_KL_orders_mean_mean"] == pytest.approx(0.38 + 0.01)
  assert row["joint_KL_best_order_mean"] == pytest.approx(0.36 + 0.01)
  assert row["copula_chain_minus_oracle_mean"] == pytest.approx(0.05)
  assert row["n_distinct_orders_mean"] == 2.0
  checks = json.loads((out / "chain_checks.json").read_text())["d5_n1000"]
  assert checks["reps_left_out"] == [3]
  assert checks["joint_jensen_max"] <= 0.0
  cells = pd.read_csv(out / "chain_cells.csv")
  assert cells["rep"].tolist() == [0, 1, 2]


def test_a_run_on_other_test_points_is_refused(tmp_path: Path) -> None:
  """A reference scored on other test points must not be paired.

  Its KL would then differ from the chain's by the points as well as by
  the model, which a paired difference cannot tell apart.
  """
  run = _chain_run(tmp_path)
  other = _write(tmp_path / "e10", ["mixture"], range(4), loglik=-1.4)
  with pytest.raises(SystemExit, match="analyzed run's cells and test points"):
    analyze.others([other], analyze.load(run), analyze.REFERENCES)
