"""E12: the truth-only importance-sampling estimate of a vine's mass."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_SCRIPT = (
  Path(__file__).resolve().parents[2]
  / "scripts"
  / "paper"
  / "e12"
  / "mass_common.py"
)


def _load_script() -> ModuleType:
  spec = importlib.util.spec_from_file_location("mass_common", _SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


mass = _load_script()


def test_a_constant_ratio_gives_its_mass_exactly() -> None:
  """``f_hat = c f`` has mass ``c``, with no Monte Carlo error to hide in.

  A log-ratio taken the wrong way round (``log f - log f_hat``) would report
  ``1 / c``, and ``-log m`` with its sign lost would read as a mass above 1
  where it is below; every companion estimate must agree with ``c`` too.
  """
  c, n = 0.8, 5000
  s = mass.stats(np.full(n, np.log(c)))
  assert s["mass"] == pytest.approx(c, rel=1e-12)
  assert s["mass_se"] == pytest.approx(0.0, abs=1e-12)
  assert s["neglogmass"] == pytest.approx(-np.log(c), rel=1e-12)
  assert s["excess_nll"] == pytest.approx(-np.log(c), rel=1e-12)
  assert s["mass_mom20"] == pytest.approx(c, rel=1e-12)
  assert s["mass_trim_top0.1pct"] == pytest.approx(c, rel=1e-12)
  assert s["ess_share"] == pytest.approx(1.0, rel=1e-12)
  assert s["top1_share"] == pytest.approx(1 / n, rel=1e-12)


def test_the_estimate_recovers_a_closed_form_mass_and_ess() -> None:
  """On a closed form, the mass, its SE and the ESS share match theory.

  ``f`` is uniform on [0, 1] and ``f_hat(y) = 2 c y``, of mass ``c``. Under
  ``f`` the ratio ``r = 2 c y`` has mean ``c``, sd ``c / sqrt(3)`` and
  ``E r^2 = 4 c^2 / 3``, so the ESS share tends to 3/4. Averaging the
  log-ratios instead of the ratios would give ``2 c / e``, a quarter below
  ``c``, and an ESS share built from the wrong sums would miss 3/4.
  """
  c, n = 1.3, 200_000
  y = np.random.default_rng(12).random(n)
  s = mass.stats(np.log(2 * c * y))
  assert abs(s["mass"] - c) < 4 * s["mass_se"]
  assert s["mass_se"] == pytest.approx(c / np.sqrt(3 * n), rel=0.02)
  assert s["ess_share"] == pytest.approx(0.75, abs=0.005)
  assert s["excess_nll"] == pytest.approx(1 - np.log(2 * c), abs=0.01)
