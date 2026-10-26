"""Vine CLI: an end-to-end run writes every table (hermetic)."""

from __future__ import annotations

import json
from pathlib import Path

from npcc.experiments import vine_cli

_TOML = """
[grid]
dims = [3]
n = [40]
n_rep = 1
arms = ["oracle", "tll"]
eval_x_n = 2
eval_m = 20

[[grid.estimators]]
label = "uni"
backend = "uniform-native"
transform = "logit"
"""


def test_vine_cli_writes_every_table(
  register_uniform_backends: None, tmp_path: Path
) -> None:
  """A missing table or truth record would leave a run unanalyzable."""
  config = tmp_path / "vine.toml"
  config.write_text(_TOML)
  out = tmp_path / "results"
  rc = vine_cli.main(
    [
      "--config",
      str(config),
      "--out",
      str(out),
      "--device",
      "cpu",
      "--format",
      "csv",
    ]
  )
  assert rc == 0
  for name in (
    "metrics_by_x",
    "summary_by_x",
    "summary_over_x",
    "runtime",
    "runtime_summary",
  ):
    path = out / f"{name}.csv"
    assert path.exists(), f"missing {name}.csv"
    assert path.stat().st_size > 0
  truths = json.loads((out / "truths.json").read_text())
  assert len(truths) == 1
  assert {"matrix", "random_matrix", "edges"} <= set(truths[0])
  assert (out / "config.json").exists()
