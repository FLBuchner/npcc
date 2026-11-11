"""E10 on E3d: the structures a mixture's members get, and how it is scored."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import pyvinecopulib as pv
import torch

from npcc.experiments import vine_runner
from npcc.experiments._study import derive_seed

_SCRIPT = (
  Path(__file__).resolve().parents[2]
  / "scripts"
  / "paper"
  / "e3"
  / "e3d_mixture.py"
)


def _load_script() -> ModuleType:
  spec = importlib.util.spec_from_file_location("e3d_mixture", _SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


mixture = _load_script()


def _logs() -> tuple[np.ndarray, np.ndarray]:
  """Three members' and the truth's log-densities at 400 points."""
  rng = np.random.default_rng(3)
  z = rng.standard_normal(400)
  logtrue = -0.5 * z**2
  members = np.stack([-0.5 * (z - mu) ** 2 for mu in (-0.4, 0.1, 0.7)])
  return members, logtrue


def test_identical_members_score_as_the_member() -> None:
  """A mixture of k copies of one member must score as that member.

  Dropping the ``- log k``, or flooring the members before mixing, would
  move the mixture away from a member it only repeats; NaN must stay where
  the member has it.
  """
  members, logtrue = _logs()
  member = members[0].copy()
  member[[5, 17]] = np.nan
  row = mixture.score_members(np.stack([member] * 4), logtrue)
  np.testing.assert_allclose(row["mixture"], member, rtol=0, atol=1e-12)
  assert row["excess_nll"] == pytest.approx(
    row["member1_excess_nll"], abs=1e-12
  )
  assert row["n_nonfinite"] == 2
  assert row["mixture_minus_member1"] == pytest.approx(0.0, abs=1e-12)


def test_the_mixture_and_its_members_are_compared_on_one_point_set() -> None:
  """A point one member cannot evaluate must leave every common-set score.

  The mixture is NaN wherever any member is, so a member scored on its own
  finite points is scored on more points than the mixture, and the gap
  between the two would mix a modeling difference with a point-set one.
  """
  members, logtrue = _logs()
  members[1, 9] = np.nan
  row = mixture.score_members(members, logtrue)
  keep = np.arange(logtrue.size) != 9
  common = [
    float(np.mean((logtrue - lp)[keep])) for lp in (members[0], members[2])
  ]
  scored = json.loads(row["member_excess_nll_common"])
  assert scored[0] == pytest.approx(common[0])
  assert scored[2] == pytest.approx(common[1])
  own = json.loads(row["member_excess_nll"])
  assert own[0] == pytest.approx(float(np.mean(logtrue - members[0])))
  assert json.loads(row["member_n_nonfinite"]) == [0, 1, 0]
  assert math.isnan(row["mixture"][9])


def test_the_mixture_scores_at_most_its_members_mean() -> None:
  """The log of a mean density is at least the mean of the log-densities.

  So on the points where all are finite, a mixture whose excess NLL exceeds
  its members' mean is wired wrong.
  """
  members, logtrue = _logs()
  row = mixture.score_members(members, logtrue)
  assert row["excess_nll"] <= row["members_mean_excess_nll_common"]
  assert row["mixture_minus_members_mean"] < -1e-3


def test_the_best_member_is_the_lowest_on_the_common_points() -> None:
  """The best member must be read on the points the mixture is scored on.

  A member that is best only because a NaN dropped its worst point would
  make the mixture look worse than any of its members.
  """
  members, logtrue = _logs()
  members[0, 0] = np.nan
  row = mixture.score_members(members, logtrue)
  common = json.loads(row["member_excess_nll_common"])
  assert row["best_member"] == int(np.argmin(common)) + 1
  assert row["best_member_excess_nll_common"] == min(common)
  assert row["mixture_minus_best_member"] == pytest.approx(
    row["excess_nll"] - min(common)
  )


def test_distinct_structures_are_counted_as_vines() -> None:
  """Two matrices of one vine must count as one structure.

  A D-vine and the D-vine on its reversed order are one vine written as two
  matrices; counting matrices would overstate how diverse the members are.
  """
  forward = pv.DVineStructure(order=[1, 2, 3, 4])
  backward = pv.DVineStructure(order=[4, 3, 2, 1])
  other = pv.DVineStructure(order=[2, 1, 3, 4])
  matrices = [np.asarray(s.matrix).tolist() for s in (forward, backward, other)]
  assert matrices[0] != matrices[1]
  assert mixture.distinct(matrices[:2]) == 1
  assert mixture.distinct(matrices) == 2


def test_every_arm_gets_the_same_structures() -> None:
  """Every arm and process must get the same structures for one cell.

  Member 1 is E3d's ``tll`` and ``tll-ns`` structure, which the mixture is
  read against; each other member is a function of the data and of its
  seed, so the competitors on the CPU and NPCC on the GPU mix the same
  vines.
  """
  gen = torch.Generator().manual_seed(2)
  z = torch.randn(300, 4, generator=gen, dtype=torch.float64)
  u = torch.special.ndtr(z + 0.9 * z[:, [0]] + 0.5 * z[:, [1]])
  structures, seeds, seconds = mixture.structures_for(u, 4, 101, 3)
  again, _, _ = mixture.structures_for(u, 4, 101, 3)
  matrices = [np.asarray(s.matrix).tolist() for s in structures]
  assert matrices == [np.asarray(s.matrix).tolist() for s in again]
  selected = vine_runner.select_structure(u)
  assert matrices[0] == np.asarray(selected.matrix).tolist()
  assert seeds == [
    None,
    derive_seed("E3d", "mixture", 4, 101, 2),
    derive_seed("E3d", "mixture", 4, 101, 3),
  ]
  drawn = vine_runner.weighted_random_structure(u, seeds[2])
  assert matrices[2] == np.asarray(drawn.matrix).tolist()
  assert len(seconds) == 3


def test_a_cell_refuses_another_arms_structures(tmp_path: Path) -> None:
  """Arms that run in separate processes must not mix other structures.

  A cell whose structures differ from those an earlier arm recorded for it
  would compare mixtures of different vines as if they were one design.
  """
  matrices = [
    np.asarray(pv.RVineStructure.from_order([1, 2, 3]).matrix).tolist(),
    np.asarray(pv.RVineStructure.from_order([2, 1, 3]).matrix).tolist(),
  ]
  row = tmp_path / "cells" / "tll" / "decay_d3_s101.json"
  row.parent.mkdir(parents=True)
  row.write_text(json.dumps({"structure_matrices": json.dumps(matrices)}))
  mixture.check_structures(tmp_path, "decay_d3_s101", matrices)
  with pytest.raises(SystemExit, match="structures must equal"):
    mixture.check_structures(tmp_path, "decay_d3_s101", matrices[::-1])


def _row(method: str, matrices: list[list[list[int]]]) -> dict[str, object]:
  """A finished cell's row, with every field the summary reads."""
  row: dict[str, object] = {
    src: 0.0 for src in mixture.SUMMARY.values() if src is not None
  }
  row.update(
    design="decay",
    d=3,
    seed=101,
    method=method,
    structure_matrices=json.dumps(matrices),
    member_recovery_overall=json.dumps([1.0, 0.5]),
  )
  return row


def test_the_summary_refuses_methods_that_mixed_other_structures(
  tmp_path: Path,
) -> None:
  """Cells copied in from another machine must have mixed the same vines.

  The GPU arm may run elsewhere, where its structures are recomputed; a
  summary pairing it with competitors that mixed other structures would
  compare two designs as one.
  """
  matrices = [
    np.asarray(pv.RVineStructure.from_order([1, 2, 3]).matrix).tolist(),
    np.asarray(pv.RVineStructure.from_order([2, 1, 3]).matrix).tolist(),
  ]
  for method, these in (("tll", matrices), ("npcc-x", matrices)):
    path = tmp_path / "cells" / method / "decay_d3_s101.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_row(method, these)))
  mixture.write_summary(tmp_path)
  assert (tmp_path / "summary.csv").read_text().count("decay,3,101") == 2
  path = tmp_path / "cells" / "vdc_shared" / "decay_d3_s101.json"
  path.parent.mkdir(parents=True)
  path.write_text(json.dumps(_row("vdc_shared", matrices[::-1])))
  with pytest.raises(SystemExit, match="must have mixed the structures"):
    mixture.write_summary(tmp_path)
