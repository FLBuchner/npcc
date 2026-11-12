"""E12: ``refit_mass.py``'s JSON lines, one readable line each.

Per cell, the controls' line gives the mixture-proposal mass (``mix``) and
SE of the truth, of TLL and of the parametric vine (with its per-component
means), the parametric's and TLL's truth-only estimates, and the truth's
check against the stored log-density. Each refitted arm's line gives the
largest difference from E3d's stored log-density, the mixture-proposal mass
and SE with its per-component means, the truth-only estimates on E3d's test
points and on the new draws from the truth, the excess NLL, the non-finite
weights and the fit and evaluation seconds. Lines that are not JSON (a
log's warnings) are skipped.

**Usage**::

    python scripts/paper/e12/show_refit.py results/e12_<TAG>/refit_results.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def show(j: dict) -> str:
  """One JSON line of ``refit_mass.py`` as text."""
  if "arm" in j:
    return (
      f"{j['cell']} {j['arm'][21:]} repr {j['reproduce_max_abs']:.1e} "
      f"mix {j['mass_mis']:.4f}±{j['mass_mis_se']:.4f} "
      f"by {[round(x, 3) for x in j['mass_mis_by_component']]} "
      f"IS-test {j['mass_is_truth_test']:.4f}±"
      f"{j['mass_is_truth_test_se']:.4f} "
      f"IS-new {j['mass_is_truth_new']:.4f}±{j['mass_is_truth_new_se']:.4f} "
      f"exc {j['excess_nll_test']:.3f} nnf {j['n_nonfinite']} "
      f"fit {j['fit_s']:.0f} eval {j['eval_s']:.0f}"
    )
  truth, tll, par = j["truth"], j["tll"], j["parametric"]
  par_is, tll_is = j["parametric_is_truth"], j["tll_is_truth"]
  return (
    f"{j['cell']} CONTROLS truth {truth[0]:.4f}±{truth[1]:.4f} "
    f"tll {tll[0]:.4f}±{tll[1]:.4f} par {par[0]:.4f}±{par[1]:.4f} "
    f"par_by {[round(x, 3) for x in par[2]]} "
    f"par-IS {par_is[0]:.4f}±{par_is[1]:.4f} "
    f"tll-IS {tll_is[0]:.4f}±{tll_is[1]:.4f} chk {j['truth_check_max_abs']}"
  )


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("files", type=Path, nargs="+", help="JSON lines or logs")
  a = p.parse_args()
  for fn in a.files:
    for line in fn.read_text().splitlines():
      if line.startswith("{"):
        print(show(json.loads(line)))


if __name__ == "__main__":
  main()
