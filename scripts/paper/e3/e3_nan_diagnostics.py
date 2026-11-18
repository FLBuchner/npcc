#!/usr/bin/env python3
"""E3c / E3d: where the non-finite vine densities come from.

Run in the NPCC environment, from the repository root, after the fits:

    cd /home/thibault/npcc
    OMP_NUM_THREADS=1 taskset -c 14 nice -n 10 uv run --no-sync python \
        scripts/paper/e3/e3_nan_diagnostics.py

**Why.** pyvinecopulib's compiled vine returns a NaN density at a few test
points of some fits. By the paper lead's rule such points are not floored:
``e3_collect.py`` scores every arm on the test points where all arms and the
truth are finite. This script says which arms, trees and pair copulas
produce them, and how many points they cost.

**How.** Every row with ``n_nonfinite > 0`` is listed with its count. For an
arm fitted by pyvinecopulib's compiled ``Vinecop`` (E3c ``parametric`` and
``tll_cpp``; E3d ``parametric``, ``tll``, ``parametric_true`` and
``tll_true``), the fit is repeated (it is deterministic) and the NaN is
localized:

1. the vine truncated after tree ``t`` is evaluated at the bad points for
   ``t = 1, 2, ...``; the first ``t`` whose density is NaN is the tree;
2. in that tree, each edge's pair copula is replaced by the independence
   copula in turn; the edges whose replacement makes the density finite are
   the culprits, reported with their family, rotation and parameters.

**Output.** ``results/e3_vdc/e3_nan_diagnostics.csv``, one row per culprit
edge (or per arm and cell when the arm is not a compiled vine), with the
counts and the coordinates of the bad test points.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyvinecopulib as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_common as common  # noqa: E402

E3D = common.NPCC_REPO / "results" / "e3d_unknown_structure"
PARAMETRIC_FAMILIES = ("indep", "gaussian", "student", "clayton", "gumbel", "frank",
                       "joe", "bb1", "bb6", "bb7", "bb8", "tawn")


def controls(experiment: str, method: str) -> pv.FitControlsVinecop:
  """The compiled arms' controls, exactly as their fit scripts build them."""
  if experiment == "E3d":
    from e3d_npcc_fit import compiled_controls

    return compiled_controls(method)
  # e3c_npcc_fit.py: one control builder for both compiled arms.
  fams = PARAMETRIC_FAMILIES if method == "parametric" else ("tll",)
  return pv.FitControlsVinecop(family_set=[getattr(pv.families, f) for f in fams],
                               selection_criterion="bic", allow_rotations=True, num_threads=1)


def truncated(vc: pv.Vinecop, pcs: list, t: int) -> pv.Vinecop:
  """``vc``'s structure and pair copulas after tree ``t`` (1-based)."""
  structure = pv.RVineStructure.from_matrix(np.asarray(vc.matrix))
  structure.truncate(t)
  return pv.Vinecop.from_structure(structure=structure, pair_copulas=[list(row) for row in pcs[:t]])


def localize(vc: pv.Vinecop, bad_u: np.ndarray) -> list[dict]:
  """The culprit edges of a compiled vine's NaN at the points ``bad_u``."""
  d = vc.dim
  pcs = [[vc.get_pair_copula(t, e) for e in range(d - 1 - t)] for t in range(vc.trunc_lvl)]
  for t in range(1, vc.trunc_lvl + 1):
    if np.isfinite(np.asarray(truncated(vc, pcs, t).pdf(bad_u))).all():
      continue
    culprits = []
    for e in range(d - t):
      trial = [list(row) for row in pcs[:t]]
      trial[t - 1][e] = pv.Bicop()
      structure = pv.RVineStructure.from_matrix(np.asarray(vc.matrix))
      structure.truncate(t)
      if np.isfinite(np.asarray(pv.Vinecop.from_structure(structure=structure, pair_copulas=trial).pdf(bad_u))).all():
        bc = pcs[t - 1][e]
        culprits.append({"tree": t, "edge": e, "family": str(bc.family).split(".")[-1],
                         "rotation": int(bc.rotation),
                         "parameters": json.dumps(np.asarray(bc.parameters).round(6).ravel().tolist())})
    return culprits or [{"tree": t, "edge": None, "family": "no single edge", "rotation": None,
                         "parameters": None}]
  return [{"tree": None, "edge": None, "family": "not reproduced", "rotation": None, "parameters": None}]


def rows_with_nonfinite() -> list[tuple[str, str, dict, Path]]:
  found = []
  for experiment, design, folder, pattern in (
      ("E3c", "strong", common.RESULTS / "e3c", "npcc_cells/*/*.json"),
      ("E3c", "decay", common.RESULTS / "e3c_decay", "npcc_cells/*/*.json"),
      ("E3d", None, E3D, "cells/*/*.json")):
    for path in sorted(folder.glob(pattern)):
      row = json.loads(path.read_text())
      if row.get("n_nonfinite", 0):
        found.append((experiment, design or row["design"], row, folder))
    for side in (folder / "vdc_side.csv",):
      if side.exists():
        import csv
        for row in csv.DictReader(side.open()):
          if int(float(row.get("n_nonfinite") or 0)):
            found.append((experiment, design or row["design"], row, folder))
  return found


def main() -> None:
  prov = common.provenance(__file__, "cpu")
  out_rows = []
  for experiment, design, row, folder in rows_with_nonfinite():
    d, seed, method = int(row["d"]), int(row["seed"]), row["method"]
    key = f"d{d}_s{seed}" if experiment == "E3c" else f"{design}_d{d}_s{seed}"
    with np.load(folder / "samples.npz") as f:
      tr, te = f[f"{key}_train"], f[f"{key}_test"]
      matrix_key = f"{key}_matrix" if experiment == "E3c" else f"{key}_true_matrix"
      true_matrix = f[matrix_key]
    base = {"experiment": experiment, "design": design, "d": d, "seed": seed, "method": method,
            "n_nonfinite": int(float(row["n_nonfinite"])), "n_test": int(te.shape[0])}
    compiled = method in ("parametric", "tll_cpp") if experiment == "E3c" else \
      method in ("parametric", "tll", "parametric_true", "tll_true")
    if not compiled:
      out_rows.append({**base, "localized": False, **prov})
      continue
    given = experiment == "E3c" or method.endswith("_true")
    ctl = controls(experiment, method)
    vc = (pv.Vinecop.from_data(tr, structure=pv.RVineStructure.from_matrix(true_matrix), controls=ctl)
          if given else pv.Vinecop.from_data(tr, controls=ctl))
    pdf = np.asarray(vc.pdf(te))
    bad = np.where(~np.isfinite(pdf))[0]
    if bad.size != base["n_nonfinite"]:
      raise RuntimeError(f"{key} {method}: the refit must reproduce {base['n_nonfinite']} "
                         f"non-finite points; got {bad.size}")
    for culprit in localize(vc, te[bad]):
      out_rows.append({**base, "localized": True, **culprit,
                       "bad_points": json.dumps(np.round(te[bad], 7).tolist()), **prov})
    print(f"{experiment} {design} {key} {method}: {bad.size} NaN points -> "
          f"{[(c['tree'], c['family'], c['rotation']) for c in localize(vc, te[bad])]}", flush=True)
  if out_rows:
    common.write_csv(out_rows, common.RESULTS / "e3_nan_diagnostics.csv")
  else:
    print("no non-finite densities")


if __name__ == "__main__":
  main()
