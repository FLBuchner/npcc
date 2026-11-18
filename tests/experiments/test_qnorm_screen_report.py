"""The qnorm screen's rule: what it adopts, what it vetoes, what it refuses."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (
  Path(__file__).resolve().parents[2]
  / "scripts"
  / "paper"
  / "qnorm_screen"
  / "report.py"
)


def _load_script() -> ModuleType:
  spec = importlib.util.spec_from_file_location("qnorm_report", _SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


report = _load_script()

E1_FAMILIES = ("clayton", "gaussian")
E3A_CASES = {
  "indep": "indep",
  "gaussian_rho=0.7": "gaussian",
  "gaussian_rho=-0.7": "gaussian",
  "student_rho=0.7_nu=5": "student",
  "clayton_theta=3": "clayton",
  "clayton_theta=3_r90": "clayton",
  "gumbel_theta=2.5": "gumbel",
  "joe_theta=3": "joe",
  "frank_theta=5": "frank",
}


def _design() -> pd.DataFrame:
  """Every pair of the screen's design, the baseline's KL and IAE filled in."""
  rows = []
  for family in E1_FAMILIES:
    for scenario in ("linear", "sin"):
      for n in (200, 500, 1000, 2000):
        for rep in range(5, 10):
          rows.append(
            {
              "study": "E1",
              "n": n,
              "family": family,
              "unit": f"{scenario}/rep{rep}",
              "base_KL": 2.0 / math.sqrt(n),
            }
          )
  for case, family in E3A_CASES.items():
    for n in (2000, 10000):
      for seed in (101, 202, 303, 404, 505):
        rows.append(
          {
            "study": "E3a",
            "n": n,
            "family": family,
            "unit": f"{case}/s{seed}",
            "base_KL": 0.3 / math.sqrt(n),
          }
        )
  pairs = pd.DataFrame(rows)
  pairs["base_IAE"] = 0.05
  pairs["arm_IAE"] = 0.05
  return pairs


def _with_arm(
  shift: pd.Series | float, *, sd: float, seed: int = 0
) -> pd.DataFrame:
  pairs = _design()
  noise = np.random.default_rng(seed).normal(0.0, sd, len(pairs))
  pairs["arm_KL"] = pairs["base_KL"] + shift + noise
  return pairs


def _verdict(pairs: pd.DataFrame) -> tuple[bool, str]:
  assert report.check_design(pairs) == []
  adopt, why = report.verdict(report.groups(pairs))
  return bool(adopt), str(why)


def test_a_clear_gain_everywhere_is_adopted() -> None:
  """A rule that could not adopt a clear, uniform gain would screen nothing."""
  adopt, why = _verdict(_with_arm(-2e-3, sd=1e-3))
  assert adopt, why


def test_no_difference_keeps_u() -> None:
  """Adoption must need evidence: two equal arms keep the baseline."""
  pairs = _design()
  # Exactly zero mean in every stratum, with spread, so the SE is not zero.
  sign = np.where(np.arange(len(pairs)) % 2 == 0, 1.0, -1.0)
  pairs["arm_KL"] = pairs["base_KL"] + 1e-3 * sign
  adopt, why = _verdict(pairs)
  assert not adopt
  assert "no gain" in why


def test_an_overall_loss_keeps_u() -> None:
  """A worse arm must never be adopted, whatever else the guard reads."""
  adopt, why = _verdict(_with_arm(+2e-3, sd=1e-3))
  assert not adopt
  assert "no gain" in why


def test_a_loss_in_one_family_vetoes_an_overall_gain() -> None:
  """A gain on average must not hide a family the change makes worse.

  The independence copula is E3a's alone, two strata of five fits: its loss
  is outweighed in the overall mean, which is why the guard reads it.
  """
  pairs = _design()
  shift = np.where(pairs["family"] == "indep", 4e-3, -2e-3)
  pairs["arm_KL"] = pairs["base_KL"] + shift
  pairs["arm_KL"] += np.random.default_rng(1).normal(0.0, 5e-4, len(pairs))
  overall = report.groups(pairs).iloc[0]
  assert overall["score"] < -report.TIE_SE * overall["se"]
  adopt, why = _verdict(pairs)
  assert not adopt
  assert "family indep" in why
  assert "n " not in why.split("SE in")[-1]


def test_a_loss_at_one_n_vetoes_an_overall_gain() -> None:
  """A gain at small n must not hide a loss at the largest n."""
  pairs = _design()
  shift = np.where(pairs["n"] == 10000, 3e-3, -3e-3)
  pairs["arm_KL"] = pairs["base_KL"] + shift
  pairs["arm_KL"] += np.random.default_rng(2).normal(0.0, 5e-4, len(pairs))
  adopt, why = _verdict(pairs)
  assert not adopt
  assert "n 10000" in why


def test_the_score_weighs_strata_equally() -> None:
  """The pooled score is the mean of stratum means, not of every pair.

  E3a's strata hold more pairs than E1's; weighting by pairs would let the
  study with more fits decide, which the rule as written does not.
  """
  pairs = pd.DataFrame(
    {
      "study": ["E1"] * 2 + ["E3a"] * 4,
      "n": [200] * 2 + [2000] * 4,
      "family": ["clayton"] * 6,
      "unit": [str(i) for i in range(6)],
      "base_KL": [1.0] * 6,
      "arm_KL": [0.0, 0.2, 1.0, 1.2, 1.4, 1.6],
    }
  )
  got, se = report.score(pairs, "KL")
  d1, d2 = np.array([-1.0, -0.8]), np.array([0.0, 0.2, 0.4, 0.6])
  se1 = d1.std(ddof=1) / math.sqrt(2)
  se2 = d2.std(ddof=1) / math.sqrt(4)
  assert got == pytest.approx((d1.mean() + d2.mean()) / 2)
  assert se == pytest.approx(math.sqrt(se1**2 + se2**2) / 2)


def test_a_group_without_an_se_is_refused() -> None:
  """A missing SE must stop the rule, not read as neither gain nor loss.

  A stratum of a single pair has no spread; skipping its NaN would give its
  group an SE of zero, and any score at all would then count as beyond it.
  """
  pairs = _with_arm(-2e-3, sd=1e-3)
  lone = pairs[(pairs["family"] == "joe") & (pairs["n"] == 2000)].index[1:]
  table = report.groups(pairs.drop(index=lone))
  joe = table[(table["group"] == "family") & (table["level"] == "joe")]
  assert joe["se"].isna().all()
  with pytest.raises(SystemExit, match="family joe"):
    report.verdict(table)


def test_the_design_check_names_what_a_run_lacks() -> None:
  """A verdict on a partial run would be read as the screen's."""
  pairs = _design()
  pairs["arm_KL"] = pairs["base_KL"]
  short = pairs.drop(index=[0, len(pairs) - 1])
  assert report.check_design(short) == [
    "E1 has 79 pairs of 80",
    "E3a at n = 10000 has 44 pairs of 45",
  ]


def _write_e3a_fit(
  folder: Path, label: str, n: int, k: int, case: str, kl: float
) -> None:
  cell = folder / "npcc_cells" / label / f"n{n}"
  cell.mkdir(parents=True, exist_ok=True)
  row = {
    "estimator": label,
    "case": case,
    "family": E3A_CASES[case],
    "seed": 101,
    "n_train": n,
    "kl": kl,
    "iae": 0.05,
    "npcc_commit": "abc1234",
  }
  (cell / f"k{k:02d}_{case}_s101.json").write_text(json.dumps(row))


def test_e3a_pairs_join_the_two_arms_fit_by_fit(tmp_path: Path) -> None:
  """Each E3a fit must meet its own counterpart, keyed by copula, seed and n."""
  for label, kl in ((report.BASE, 0.01), (report.ARM, 0.008)):
    _write_e3a_fit(tmp_path, label, 2000, 0, "indep", kl)
    _write_e3a_fit(tmp_path, label, 2000, 1, "joe_theta=3", 2 * kl)
  pairs = report.e3a_pairs(tmp_path).set_index("unit")
  assert pairs.loc["joe_theta=3/s101", "base_KL"] == pytest.approx(0.02)
  assert pairs.loc["joe_theta=3/s101", "arm_KL"] == pytest.approx(0.016)
  assert set(pairs["family"]) == {"indep", "joe"}


def test_e3a_refuses_a_fit_only_one_arm_has(tmp_path: Path) -> None:
  """A failed fit must stop the screen, not shrink it to the fits that ran."""
  _write_e3a_fit(tmp_path, report.BASE, 2000, 0, "indep", 0.01)
  _write_e3a_fit(tmp_path, report.BASE, 2000, 1, "joe_theta=3", 0.02)
  _write_e3a_fit(tmp_path, report.ARM, 2000, 0, "indep", 0.01)
  with pytest.raises(SystemExit, match="must have both arms"):
    report.e3a_pairs(tmp_path)


def test_e3a_refuses_a_non_finite_kl(tmp_path: Path) -> None:
  """A NaN KL would drop out of every mean and flatter the arm that had it."""
  _write_e3a_fit(tmp_path, report.BASE, 2000, 0, "indep", 0.01)
  _write_e3a_fit(tmp_path, report.ARM, 2000, 0, "indep", math.nan)
  with pytest.raises(SystemExit, match="KL must be finite"):
    report.e3a_pairs(tmp_path)


def _write_e1_run(run: Path, reps: range) -> None:
  for label, kl in ((report.BASE, 0.02), (report.ARM, 0.01)):
    for rep in reps:
      cell = run / "cells" / f"clayton__linear__n200__rep{rep}"
      cell.mkdir(parents=True, exist_ok=True)
      (cell / "DONE").touch()
      rows = pd.DataFrame(
        {
          "label": label,
          "quantity": "pdf",
          "family": "clayton",
          "tau_scenario": "linear",
          "n": 200,
          "rep": rep,
          "x": [0.0, 1.0],
          "KL": [9.0, 9.0],
          "KL_mc": [kl, 3 * kl],
          "KL_mc_n": 2000,
          "KL_mc_nonfinite": 0,
          "IAE": [0.1, 0.3],
        }
      )
      path = cell / "metrics.parquet"
      if path.exists():
        rows = pd.concat([pd.read_parquet(path), rows], ignore_index=True)
      rows.to_parquet(path)


def test_e1_pairs_average_the_monte_carlo_kl_over_x(tmp_path: Path) -> None:
  """E1's pair is the cell's Monte Carlo KL over its x values, not the grid's.

  The grid KL misses the boundary band and is not the paper's metric; a pair
  scored on it, or on one x, would rank the arms on something else.
  """
  _write_e1_run(tmp_path, range(5, 7))
  pairs = report.e1_pairs(tmp_path)
  assert len(pairs) == 2
  assert pairs["base_KL"].tolist() == pytest.approx([0.04, 0.04])
  assert pairs["arm_KL"].tolist() == pytest.approx([0.02, 0.02])
  assert pairs["base_IAE"].tolist() == pytest.approx([0.2, 0.2])


def test_e1_refuses_the_reps_e7a_chose_on(tmp_path: Path) -> None:
  """Reps 0-4 are not held out, so a screen that read them would not be one."""
  _write_e1_run(tmp_path, range(4, 6))
  with pytest.raises(SystemExit, match="reps 5-9"):
    report.e1_pairs(tmp_path)
