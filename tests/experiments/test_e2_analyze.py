"""E2's analysis: the runs it pairs and what it says of a selected structure."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

_SCRIPT = (
  Path(__file__).resolve().parents[2]
  / "scripts"
  / "paper"
  / "e2"
  / "analyze.py"
)


def _load_script() -> ModuleType:
  spec = importlib.util.spec_from_file_location("e2_analyze", _SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


analyze = _load_script()

#: A C-vine on 4 variables rooted at 1, and a structure with the same first
#: tree that shares one of its two second-tree edges and not its third tree.
C_VINE = [[1, 1, 1, 1], [2, 2, 2, 0], [3, 3, 0, 0], [4, 0, 0, 0]]
SAME_FIRST_TREE = [[1, 1, 1, 1], [4, 4, 4, 0], [2, 2, 0, 0], [3, 0, 0, 0]]
#: One D-vine on 1 - 2 - 3, written from either end.
D_VINE = [[2, 3, 3], [3, 2, 0], [1, 0, 0]]
D_VINE_REVERSED = [[2, 1, 1], [1, 2, 0], [3, 0, 0]]


def _metrics(
  arms: list[str], *, seed: int = 7, loglik: float = -1.5
) -> pd.DataFrame:
  rows = [
    {
      "d": 3,
      "n": n,
      "rep": rep,
      "seed": seed,
      "arm": arm,
      "label": "tabpfn" if arm != "tll" else "kde-tll",
      "x": x,
      "loglik_true": loglik,
      "KL": 0.1,
      "KL_copula": 0.05,
      "ISE": 0.01,
    }
    for arm in arms
    for n in (500, 1000)
    for rep in range(2)
    for x in (0.1, 0.9)
  ]
  return pd.DataFrame(rows)


def _run(
  path: Path,
  *,
  base_seed: int = 317,
  eval_m: int = 1000,
  truths: list[dict[str, object]] | None = None,
) -> Path:
  path.mkdir(parents=True)
  config = {
    "run": {"base_seed": base_seed},
    "grid": {"eval_x_n": 2, "eval_m": eval_m, "projection_grid_size": 30},
  }
  (path / "config.json").write_text(json.dumps(config))
  truths = truths or [
    {"d": 3, "rep": rep, "matrix": D_VINE} for rep in range(2)
  ]
  (path / "truths.json").write_text(json.dumps(truths))
  return path


def test_runs_that_pair_are_accepted(tmp_path: Path) -> None:
  """The check must pass the runs it is there for, or --with is unusable."""
  analyze.check_pairing(
    _run(tmp_path / "main"),
    _metrics(["oracle", "random", "tll"]),
    _run(tmp_path / "selected"),
    _metrics(["selected"]),
  )


_ONE_TRUTH = [{"d": 3, "rep": 0, "matrix": D_VINE}]


@pytest.mark.parametrize(
  ("base_seed", "eval_m", "truths", "seed", "loglik", "match"),
  [
    (1, 1000, None, 7, -1.5, "base_seed"),
    (317, 500, None, 7, -1.5, "eval_m"),
    (317, 1000, _ONE_TRUTH, 7, -1.5, "truths.json"),
    (317, 1000, None, 8, -1.5, "same seed and true log-density"),
    (317, 1000, None, 7, -1.25, "same seed and true log-density"),
  ],
)
def test_runs_that_do_not_pair_are_refused(
  tmp_path: Path,
  base_seed: int,
  eval_m: int,
  truths: list[dict[str, object]] | None,
  seed: int,
  loglik: float,
  match: str,
) -> None:
  """An arm scored on other data or test points cannot be paired.

  Its difference from the oracle would then be partly a difference of data,
  and nothing in the tables would say so. The mean true log-density over a
  cell's test points changes if any of those points does.
  """
  main = _run(tmp_path / "main")
  other = _run(
    tmp_path / "other", base_seed=base_seed, eval_m=eval_m, truths=truths
  )
  with pytest.raises(SystemExit, match=match):
    analyze.check_pairing(
      main,
      _metrics(["oracle", "random", "tll"]),
      other,
      _metrics(["selected"], seed=seed, loglik=loglik),
    )


def test_an_arm_in_both_runs_is_refused(tmp_path: Path) -> None:
  """An arm held twice would be averaged over two fits of each repetition."""
  with pytest.raises(SystemExit, match="both hold"):
    analyze.check_pairing(
      _run(tmp_path / "main"),
      _metrics(["oracle", "random"]),
      _run(tmp_path / "other"),
      _metrics(["random", "selected"]),
    )


def _selected_run(
  path: Path, truth: list[list[int]], chosen: list[list[int]]
) -> Path:
  run = _run(path, truths=[{"d": len(truth), "rep": 0, "matrix": truth}])
  pd.DataFrame(
    [
      {
        "d": len(truth),
        "n": 500,
        "rep": 0,
        "arm": "selected",
        "structure": json.dumps(chosen),
        "selection_time": 0.25,
      }
    ]
  ).to_parquet(run / "runtime.parquet")
  return run


def test_one_vine_written_two_ways_is_the_truth(tmp_path: Path) -> None:
  """Two R-vine matrices can encode one vine; comparing matrices would miss it."""
  run = _selected_run(tmp_path / "run", D_VINE, D_VINE_REVERSED)
  [row] = analyze.selected_structures(run).to_dict("records")
  assert row["selected_is_true"]
  assert row["tree1_recovery"] == 1.0
  assert row["recovery"] == 1.0


def test_recovery_is_counted_tree_by_tree(tmp_path: Path) -> None:
  """The first tree is found whole while the vine is not the truth.

  At d = 5 the whole structure is rarely recovered, so the share of the
  first tree, which holds the strongest edges, is what says whether the
  selection found them.
  """
  run = _selected_run(tmp_path / "run", C_VINE, SAME_FIRST_TREE)
  [row] = analyze.selected_structures(run).to_dict("records")
  assert not row["selected_is_true"]
  assert row["tree1_recovery"] == 1.0
  assert row["recovery"] == pytest.approx(4 / 6)
  assert row["selection_seconds"] == 0.25


def test_the_selected_arm_is_paired_by_repetition(tmp_path: Path) -> None:
  """selected - oracle and selected - random pair each repetition's fits."""
  xavg = pd.DataFrame(
    [
      {"d": 3, "n": 500, "rep": rep, "arm": arm, "KL": kl, "KL_copula": kl / 2}
      for rep, values in enumerate(((0.1, 0.4, 0.2), (0.3, 0.6, 0.2)))
      for arm, kl in zip(("oracle", "random", "selected"), values, strict=True)
    ]
  )
  structures = pd.DataFrame(
    [
      {
        "d": 3,
        "n": 500,
        "rep": rep,
        "selected_is_true": rep == 0,
        "tree1_recovery": 1.0,
        "recovery": 1.0 if rep == 0 else 2 / 3,
        "selection_seconds": 0.5,
      }
      for rep in range(2)
    ]
  )
  [row] = analyze.selected_table(xavg, structures).to_dict("records")
  assert row["selected_minus_oracle_joint_KL_mean"] == pytest.approx(0.0)
  assert row["selected_minus_random_joint_KL_mean"] == pytest.approx(-0.3)
  assert row["oracle_wins"] == 1
  assert row["selected_wins"] == 2
  assert row["selected_is_true"] == 1
  assert row["recovery_mean"] == pytest.approx(5 / 6)
  assert row["n_rep"] == 2
